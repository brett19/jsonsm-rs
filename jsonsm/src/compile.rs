//! Compilation: [`Expr`] → [`MatchDef`].
//!
//! A [`MatchDef`] pairs two structures the [`crate::matcher`] walks in lockstep:
//! - the **exec trie** — an arena of exec nodes keyed by field name, plus typed array
//!   indices for element references like `a[0]`, describing which document fields matter and
//!   what to do at each;
//! - the **logic tree** ([`crate::logic_tree`]) — the boolean structure whose leaves are
//!   the operations attached to exec nodes.
//!
//! Each operation carries the logic-tree bucket it reports into. Negation is lowered
//! structurally at compile time — `NotEquals` → `NOT (Equals)`, `NotExists` → `NOT (Exists)` —
//! which is sound because negation is applied to a three-valued result: `NOT Unknown` is
//! `Unknown`, so lowering `!=` this way cannot make an absent field match, while `NOT Exists`
//! still reports `true` because an `Exists` leaf answers `false` when its field is absent (see
//! precomputed per node).
//!
//! A comparison referencing **two or more fields** (in the same context) — including
//! `f(a, b) <op> const`, `f(a) <op> b`, and plain `a <op> b` — cannot use a single active
//! value, so each referenced field is recorded in a **slot** and the whole comparison is
//! deferred to the current scope's **after-node**: at the root it runs after the document
//! (order-independent), and in a loop body it runs after each element. Built-in functions
//! ([`crate::func`]) are supported as operands with any number of field/constant arguments.
//!
//! A loop whose body reads a field from an **enclosing** scope is handled with an
//! **after-loop**: the array is stored in a slot and the loop is deferred to that scope's
//! after-node, so it runs once the scope has been parsed and the fields it references are
//! available (order-independent). At the document scope that means "after the document"; in a
//! loop body it means "after each element". Nesting composes to any depth: a body reaching
//! past its immediately enclosing scope also causes *that* loop to be deferred, so by the
//! time the innermost loop runs every scope it reads has been parsed.
//!
//! [`compile`] takes *all* the expressions to evaluate at once: they are joined by
//! non-short-circuiting `Neor` nodes so every one is fully evaluated, and each expression's
//! result is reported individually (see
//! [`MatchOutcome::expression_matched`](crate::matcher::MatchOutcome::expression_matched)).
//!
//! It also takes a [`Projection`] — a list of document field paths to **capture** during the
//! same scan. Each projected path is marked to store its value's byte range into a slot
//! (sharing a slot with a comparison that already stores the same field), and the captured
//! values are read back through
//! [`MatchOutcome::projected`](crate::matcher::MatchOutcome::projected).
//!
//! `exists` and `matches` accept a field from an *enclosing* scope as well as the current one:
//! the outer field is stored in a slot and the op attached to the current scope's node, which is
//! visited unconditionally, so by then the slot is filled. `CompileError::CrossContext` is now
//! raised only for a **loop target** — the array a loop iterates must be a field of the current
//! scope.

use crate::collation::{Collation, CollationError, ValueMatcher};
use crate::logic_tree::{LogicTree, NodeIdx, NodeType, TreeError, Tri};
use crate::value::{FastStr, FastVal};
use jsonsm_ast::{CompareOp, Expr, Field, Literal, LoopType, PathComponent, VariableId};
use std::sync::Arc;

/// Index of an [`ExecNode`] within a [`MatchDef`]'s arena. `0` is the root.
pub(crate) type ExecId = usize;
/// A logic-tree bucket (node) index.
pub(crate) type BucketId = NodeIdx;
/// A storage slot index: a field's scanned byte range is recorded here for later
/// reference by a deferred (after-node) op.
pub(crate) type SlotId = usize;

/// Maximum nesting depth of an expression accepted by [`compile`].
///
/// Compilation (and the front-ends' name resolution) walks the expression recursively, so an
/// adversarially deep tree — say a filter string of a hundred thousand nested `NOT`s — could
/// otherwise exhaust the stack. Real expressions are orders of magnitude shallower.
///
/// Note this bounds what the *engine* will process; a caller that builds such a tree still
/// pays recursive `Drop` for it, so front-ends reject over-deep input before returning an AST.
pub const MAX_EXPR_DEPTH: usize = 256;

/// A list of document field paths to **capture** (project) during matching.
///
/// A path is a sequence of [`PathComponent`]s. Strings convert to object keys and `usize`
/// to array indices, so `["name", "first"]` refers to `$doc.name.first`; mix the two by
/// naming the components (`[PathComponent::Key("a".into()), PathComponent::Index(0)]` is
/// `$doc.a[0]`). The empty path refers to the whole document. Order is significant — the
/// index of a path here is the index used to read its captured value back from
/// [`MatchOutcome::projected`](crate::matcher::MatchOutcome::projected).
///
/// ```
/// use jsonsm::compile::Projection;
/// use jsonsm::ast::PathComponent;
/// let projection = Projection::new()
///     .field(["name", "first"])
///     .field([PathComponent::Key("tags".into()), PathComponent::Index(0)]);
/// assert_eq!(projection.len(), 2);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Projection {
    paths: Vec<Vec<PathComponent>>,
}

impl Projection {
    /// An empty projection (capture nothing).
    pub fn new() -> Self {
        Projection::default()
    }

    /// Add a field path, returning `self` for chaining.
    pub fn field<I, S>(mut self, path: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<PathComponent>,
    {
        self.push(path);
        self
    }

    /// Add a field path, returning the index it was assigned. A duplicate path is added
    /// again (and gets its own index); both indices report the same captured value.
    pub fn push<I, S>(&mut self, path: I) -> usize
    where
        I: IntoIterator<Item = S>,
        S: Into<PathComponent>,
    {
        self.paths.push(path.into_iter().map(Into::into).collect());
        self.paths.len() - 1
    }

    /// The projected paths, in index order.
    pub fn paths(&self) -> &[Vec<PathComponent>] {
        &self.paths
    }

    /// The index of `path`, if it is projected (the first match, if it was added twice).
    pub fn index_of(&self, path: &[PathComponent]) -> Option<usize> {
        self.paths.iter().position(|p| p.as_slice() == path)
    }

    /// Number of projected paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Whether nothing is projected.
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

impl<I, S> FromIterator<I> for Projection
where
    I: IntoIterator<Item = S>,
    S: Into<PathComponent>,
{
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut p = Projection::new();
        for path in iter {
            p.push(path);
        }
        p
    }
}

/// A projected field: the requested path and the slot its value's byte range lands in.
#[derive(Debug, Clone)]
pub(crate) struct ProjectedField {
    pub(crate) path: Vec<PathComponent>,
    pub(crate) slot: SlotId,
}

/// Compile a literal to the [`FastVal`] the matcher will compare against.
///
/// The value is built **once, here**, and stored in the [`MatchDef`] — comparison then
/// borrows it rather than rebuilding it per evaluation. A `FastVal<'static>` is what makes
/// that possible: every form a literal can take owns its contents, so the stored value
/// borrows nothing and outlives every document. (`FastStr::Owned` is what buys this. A
/// `FastStr::Unescaped` would have to point at a `String` stored beside it, which is
/// self-referential and is why an earlier `ConstVal` enum mirrored `FastVal` instead of being
/// one — paying a reconstruction on every comparison to avoid the lifetime.)
///
/// Each literal takes the **eagerly decoded** variant, never a lazy one. `FastVal` keeps
/// document numbers as `IntBytes`/`FloatBytes` and document strings as `Escaped` because a
/// scanned value may never be compared and parsing it would be wasted; a constant is going to
/// be compared or it would not be in the expression, and it is built once regardless. So the
/// laziness has no one to serve and would only put a decode on the hot path.
///
/// A string literal is already decoded — the parser resolved its escapes — so its bytes *are*
/// the logical string and `FastStr::Owned` states exactly that. This is the distinction
/// `FastStr` draws for document strings, made deliberately for constants rather than by
/// default: a constant is never `Escaped`, whatever characters it contains. A literal holding
/// a quote or a newline is a two- or one-character string here, not the four or two bytes JSON
/// would spell it with, and `Collation::compare` reaches `cmp_plain_vs_escaped` to compare it
/// against a document string that *is* escaped.
fn fastval_from_literal(lit: &Literal) -> FastVal<'static> {
    match lit {
        Literal::Null => FastVal::Null,
        Literal::Bool(b) => FastVal::Bool(*b),
        Literal::Int(i) => FastVal::Int(*i),
        Literal::Uint(u) => FastVal::Uint(*u),
        Literal::Float(f) => FastVal::Float(*f),
        Literal::String(s) => FastVal::Str(FastStr::Owned(s.clone())),
    }
}

/// A reference to an operand value at match time.
#[derive(Debug, Clone)]
pub(crate) enum DataRef {
    /// The value currently being scanned at the exec node the op is attached to.
    Active,
    /// A constant from the expression, built at compile time and compared by reference.
    Const(FastVal<'static>),
    /// A value stored earlier in a slot (used by deferred after-node ops). Resolves to
    /// [`FastVal::Missing`](crate::value::FastVal::Missing) if the slot was never filled
    /// (the field was absent).
    Slot(SlotId),
    /// A built-in function applied to resolved argument values.
    Func(FuncRef),
}

/// A compiled function application: a name plus the data refs for its arguments.
#[derive(Debug, Clone)]
pub(crate) struct FuncRef {
    pub(crate) name: String,
    pub(crate) params: Vec<DataRef>,
}

/// Engine comparison operators (the AST's `NotEquals` is lowered to `NOT (Equals)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CmpOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    fn from_ast(op: CompareOp) -> Option<CmpOp> {
        Some(match op {
            CompareOp::Equals => CmpOp::Eq,
            CompareOp::LessThan => CmpOp::Lt,
            CompareOp::LessEquals => CmpOp::Le,
            CompareOp::GreaterThan => CmpOp::Gt,
            CompareOp::GreaterEquals => CmpOp::Ge,
            CompareOp::NotEquals => return None, // lowered to NOT(Equals)
        })
    }
}

/// A single operation, reporting its boolean result into `bucket`.
#[derive(Debug, Clone)]
pub(crate) struct OpNode {
    pub(crate) bucket: BucketId,
    pub(crate) kind: OpKind,
}

#[derive(Debug, Clone)]
pub(crate) enum OpKind {
    /// A comparison between two operands.
    Compare {
        op: CmpOp,
        lhs: DataRef,
        rhs: DataRef,
    },
    /// True when `of` resolves to a value at all.
    ///
    /// `of` is [`DataRef::Active`] for a field in the current scope, where reaching the op *is*
    /// the answer. For a field from an enclosing scope it is a [`DataRef::Slot`] and the op is
    /// deferred to the scope's after-node, exactly as a cross-field comparison is: the slot is
    /// filled iff the field was present, so "did it resolve" answers the question either way.
    Exists { of: DataRef },
    /// `of` matches a compiled pattern. Same two operand shapes as [`OpKind::Exists`].
    Matches {
        matcher: Arc<dyn ValueMatcher>,
        of: DataRef,
    },
    /// A constant boolean (from `True`/`False` / empty `And`/`Or`).
    Always(bool),
}

/// A loop over the array at the exec node it is attached to.
#[derive(Debug, Clone)]
pub(crate) struct LoopNode {
    /// The loop *body* bucket (the logic-tree Loop node's child).
    pub(crate) bucket: BucketId,
    pub(crate) mode: LoopType,
    /// The exec node evaluated for each array element.
    pub(crate) node: ExecId,
    /// Slots stored by nodes inside the body, cleared before each iteration (see
    /// [`fill_loop_clear_slots`]).
    pub(crate) clear_slots: Vec<SlotId>,
}

/// A loop deferred until its enclosing scope is fully parsed, so its body can reference
/// outer fields (stored in slots) regardless of document field order. The array itself is
/// read back from `array_slot`.
#[derive(Debug, Clone)]
pub(crate) struct AfterLoopNode {
    pub(crate) bucket: BucketId,
    pub(crate) mode: LoopType,
    pub(crate) node: ExecId,
    pub(crate) array_slot: SlotId,
    /// Slots stored by nodes inside the body, cleared before each iteration (see
    /// [`fill_loop_clear_slots`]).
    pub(crate) clear_slots: Vec<SlotId>,
}

/// Operations and loops deferred until a scope is fully parsed (so any slots they
/// reference are filled regardless of document field order). Attached to a scope's root
/// exec node.
#[derive(Debug, Clone, Default)]
pub(crate) struct AfterNode {
    pub(crate) ops: Vec<OpNode>,
    pub(crate) loops: Vec<AfterLoopNode>,
}

/// The object-key children of an exec node.
///
/// A flat vector, not a `HashMap`. The asymmetry is the whole point: an exec node has one
/// child per *distinct field the expression names* at that level — nearly always one to
/// five — while the document offers a key for every field it happens to contain, all of
/// which must be looked up. Hashing 25 document keys to probe a 3-entry table costs more
/// than comparing them against those 3 entries, and the hash is not free: `SipHash` plus the
/// `str::from_utf8` a `&str` key requires, on every field of every document.
///
/// Keys are `[u8]`, so a lookup needs no UTF-8 validation at all. Each entry carries two
/// summaries that reject a mismatch before any memcmp — `tag` and `head`/`mask` — because
/// the two lookups arrive with the key in different forms; see [`KeyEntry`].
///
/// Each entry stores the key **quoted** — `"name"`, the closing quote included — because
/// that is the form the matcher compares against the document. See [`KeyMap::match_quoted`].
#[derive(Debug, Clone, Default)]
pub(crate) struct KeyMap {
    entries: Vec<KeyEntry>,
    /// Set when some key is not its own JSON encoding, which disables
    /// [`KeyMap::match_quoted`] for the whole map. See there for what it would break.
    escapable: bool,
}

/// One object-key child.
#[derive(Debug, Clone)]
struct KeyEntry {
    /// The first up-to-eight bytes of `quoted`, little-endian, with `mask` covering them.
    ///
    /// This is the prefilter that keeps [`KeyMap::match_quoted`] off the dependency chain.
    /// `tag` cannot serve there: it needs the key's *length*, which is what the raw-byte
    /// match exists to avoid discovering. Held inline in the vector, so testing an entry is
    /// three register operations against a word the caller already loaded — no dereference
    /// of `quoted`, which is the whole point. For a key of seven characters or fewer the
    /// masked compare is not a filter at all but the entire answer.
    head: u64,
    /// `0xff` in each byte position `head` covers, zero above.
    mask: u64,
    /// Length in the low 16 bits, first byte next, last byte on top, over the *unquoted*
    /// key. Keys longer than `u16::MAX` all collide into the same tag, which is merely a
    /// missed prefilter. Used by [`KeyMap::get`], which is given a key that is already
    /// delimited and decoded.
    tag: u32,
    id: ExecId,
    /// `"key"`, the quotes included.
    quoted: Box<[u8]>,
}

#[inline]
fn key_tag(key: &[u8]) -> u32 {
    let len = key.len().min(u16::MAX as usize) as u32;
    let first = *key.first().unwrap_or(&0) as u32;
    let last = *key.last().unwrap_or(&0) as u32;
    len | (first << 16) | (last << 24)
}

/// True when `key` is byte-for-byte what a JSON encoder would put between the quotes, so
/// that `"` + key + `"` is *the* encoding of it rather than merely one of several.
fn is_verbatim(key: &[u8]) -> bool {
    !key.iter().any(|&b| b == b'"' || b == b'\\' || b < 0x20)
}

/// The first `min(8, bytes.len())` bytes as a little-endian word, zero-padded above.
///
/// Reading short is not a special case to be avoided but the ordinary one: it is how a key
/// near the end of the document, and a key shorter than a word, both work. The padding is
/// zero, and no byte of a quoted verbatim key is zero — a NUL is a control character, which
/// [`is_verbatim`] rejects — so a padded word can never satisfy a mask that reaches into the
/// padding. Short input therefore fails to match rather than matching wrongly.
///
/// The two arms are not stylistic. Written as one copy of `min(8, len)` bytes the length is
/// a runtime value, and LLVM emits a `memcpy` — a call and a branch tree where the whole
/// point is a single load feeding a chain the matcher is latency-bound on. Spelling the full
/// word out separately makes it one `mov`, and leaves the short read as the cold branch it is.
#[inline(always)]
pub(crate) fn head_word(bytes: &[u8]) -> u64 {
    match bytes.first_chunk::<8>() {
        Some(w) => u64::from_le_bytes(*w),
        None => {
            let mut buf = [0u8; 8];
            buf[..bytes.len()].copy_from_slice(bytes);
            u64::from_le_bytes(buf)
        }
    }
}

impl KeyMap {
    #[inline]
    pub(crate) fn get(&self, key: &[u8]) -> Option<ExecId> {
        let tag = key_tag(key);
        self.entries
            .iter()
            .find(|e| e.tag == tag && e.key() == key)
            .map(|e| e.id)
    }

    /// Match the document bytes at an object key's opening quote against every child, without
    /// first finding where the key ends. On a hit the quoted key's length *is* the end, so a
    /// few register compares replace running the tokenizer over the key.
    ///
    /// `at` starts at the opening quote and runs to the end of the document; `word` is
    /// [`head_word`] of it, passed in because the caller needs the same word for the miss
    /// path. Returns the child and the length of the quoted key it matched.
    ///
    /// Only sound when [`Self::verbatim`] holds, and the caller must check it. The closing
    /// quote is what makes a prefix hit conclusive — `"tag"` cannot match `"tags":` — and a
    /// key ending in `\` would break exactly that: the quoted form of the key `a\` is
    /// `"a\"`, which is a prefix of the document key `"a\"x"`, whose content is `a"x`.
    #[inline(always)]
    pub(crate) fn match_quoted(&self, word: u64, at: &[u8]) -> Option<(ExecId, usize)> {
        debug_assert!(self.verbatim());
        debug_assert_eq!(word, head_word(at));
        self.match_quoted_inner(word, at)
    }

    /// [`Self::match_quoted`] without the guard, so a test can show what the guard prevents.
    ///
    /// `a_key_needing_escapes_disables_raw_comparison` demonstrates the wrong answer a
    /// non-verbatim map would produce here, which it cannot do through the checked entry
    /// point — the `debug_assert` is the thing being demonstrated. Splitting them lets the
    /// guard stay live in debug builds *and* the test keep its point, which matters because
    /// that assertion firing in the debug suite is why the suite went unrun.
    fn match_quoted_inner(&self, word: u64, at: &[u8]) -> Option<(ExecId, usize)> {
        for e in &self.entries {
            if (word ^ e.head) & e.mask != 0 {
                continue;
            }
            let len = e.quoted.len();
            // Up to eight bytes the word settled outright; beyond that the head has merely
            // narrowed it to one candidate, so read the rest.
            if len <= 8 || (at.len() >= len && at[8..len] == e.quoted[8..]) {
                return Some((e.id, len));
            }
        }
        None
    }

    /// Whether [`Self::match_quoted`] may be used against this map.
    #[inline(always)]
    pub(crate) fn verbatim(&self) -> bool {
        !self.escapable
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many distinct object keys this node names.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Which entry `id` is, for a caller tracking whether every named key has been seen.
    ///
    /// A linear scan, and deliberately not folded into the lookups: it runs only when a key
    /// *matched*, which is rare against the number of keys a document carries, and keeping it
    /// out of `match_quoted` leaves that function's dependency chain alone.
    #[inline]
    pub(crate) fn ordinal(&self, id: ExecId) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// Append a child. Unlike a map's `insert` this does not deduplicate, so callers must
    /// have established the key is absent — [`Transformer::navigate_key`] does, since it
    /// only inserts on a failed `get`.
    pub(crate) fn insert(&mut self, key: &str, id: ExecId) {
        debug_assert!(self.get(key.as_bytes()).is_none(), "duplicate key {key:?}");
        self.escapable |= !is_verbatim(key.as_bytes());
        let mut quoted = Vec::with_capacity(key.len() + 2);
        quoted.push(b'"');
        quoted.extend_from_slice(key.as_bytes());
        quoted.push(b'"');
        let covered = quoted.len().min(8);
        self.entries.push(KeyEntry {
            head: head_word(&quoted),
            mask: if covered == 8 {
                u64::MAX
            } else {
                (1u64 << (covered * 8)) - 1
            },
            tag: key_tag(key.as_bytes()),
            id,
            quoted: quoted.into(),
        });
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = ExecId> + '_ {
        self.entries.iter().map(|e| e.id)
    }
}

impl KeyEntry {
    /// The key inside the stored `"key"`.
    #[inline(always)]
    fn key(&self) -> &[u8] {
        &self.quoted[1..self.quoted.len() - 1]
    }
}

#[cfg(test)]
impl std::ops::Index<&str> for KeyMap {
    type Output = ExecId;

    fn index(&self, key: &str) -> &ExecId {
        let tag = key_tag(key.as_bytes());
        &self
            .entries
            .iter()
            .find(|e| e.tag == tag && e.key() == key.as_bytes())
            .expect("no such key")
            .id
    }
}

/// A node in the exec trie: what to do when a particular document field is reached.
#[derive(Debug, Clone, Default)]
pub(crate) struct ExecNode {
    /// Children reached by object key.
    pub(crate) elems: KeyMap,
    /// Children reached by array index, as `(index, node)`. Kept as a small sorted vector
    /// rather than map keys like `"[0]"`: an array scan then needs no per-element string
    /// formatting or hashing, and an object key that happens to be spelled `"[0]"` stays
    /// distinct from element 0 (which Go's string-keyed trie conflates).
    pub(crate) indexed: Vec<(usize, ExecId)>,
    pub(crate) ops: Vec<OpNode>,
    pub(crate) loops: Vec<LoopNode>,
    /// If set, record this field's scanned byte range into the given slot.
    pub(crate) store: Option<SlotId>,
    /// Whether `store`'s slot is (also) a projection target. The matcher tracks how many
    /// projection slots are still unfilled so it does not short-circuit the scan before
    /// every projected field has been captured.
    pub(crate) store_projected: bool,
    /// Deferred ops to run after this node's scope is fully parsed.
    pub(crate) after: Option<AfterNode>,
    /// Every logic-tree bucket written by this node's exec subtree — its own ops and loops, its
    /// deferred ops and loops, and recursively those of all its children — paired with the value
    /// that bucket takes if its field turns out to be absent.
    ///
    /// This is what a field's *absence* means in bucket terms. When a container closes, any
    /// bucket in here still unset belongs to a path the document did not contain, so the matcher
    /// seals it to the recorded value (see [`crate::matcher::FastMatcher`]). Precomputed because
    /// the close of every container consults it, and it is a property of the expression rather
    /// than of the document.
    ///
    /// Nearly every bucket seals to [`Tri::Unknown`]: a comparison against a value that is not
    /// there has no answer. The exception is `Exists`, which asks about presence rather than
    /// about a value and so is answerable precisely when the field is missing — it seals to
    /// `False`. That is what keeps `NOT EXISTS` true on an absent field, where an `Unknown`
    /// would have made it unmatched.
    pub(crate) seal_buckets: Vec<(BucketId, Tri)>,
}

/// A compiled expression (or set of expressions): everything the matcher needs to
/// evaluate it against a document.
#[derive(Debug, Clone)]
pub struct MatchDef {
    pub(crate) arena: Vec<ExecNode>,
    pub(crate) root: ExecId,
    pub(crate) tree: LogicTree,
    /// The logic-tree bucket holding the overall result (the OR of all expressions).
    pub(crate) root_bucket: BucketId,
    /// The bucket holding each individual expression's result (for multi-expression
    /// compilation); `expr_buckets[i]` is expression `i`.
    pub(crate) expr_buckets: Vec<BucketId>,
    pub(crate) num_slots: usize,
    /// The projected fields, in the order the caller requested them.
    pub(crate) projections: Vec<ProjectedField>,
    /// How many *distinct* slots the projections capture into (two projections of the same
    /// path share one slot). The matcher counts down from this while scanning.
    pub(crate) num_projection_slots: usize,
}

impl MatchDef {
    /// Number of logic-tree buckets (nodes).
    pub fn num_buckets(&self) -> usize {
        self.tree.len()
    }

    /// Number of storage slots the matcher must allocate (one per stored field).
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Number of expressions compiled into this definition.
    pub fn num_expressions(&self) -> usize {
        self.expr_buckets.len()
    }

    /// Number of projected fields (see [`compile`]).
    pub fn num_projections(&self) -> usize {
        self.projections.len()
    }

    /// The path of projected field `i`. Panics if `i` is out of range.
    pub fn projection_path(&self, i: usize) -> &[PathComponent] {
        &self.projections[i].path
    }

    /// The index of a projected path, if it was projected.
    pub fn projection_index(&self, path: &[PathComponent]) -> Option<usize> {
        self.projections.iter().position(|p| p.path == path)
    }
}

/// An error encountered while compiling an expression.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("expected a boolean expression, found an operand node")]
    NotABoolean,
    #[error("expected an operand expression, found a boolean node")]
    NotAnOperand,
    #[error("this operator requires a field from the innermost scope")]
    CrossContext,
    #[error("field references an unknown or out-of-scope variable ({0})")]
    UnknownVariable(VariableId),
    #[error("a function is not valid here; a plain field reference is required")]
    Func,
    #[error("a loop's `in` operand must be a field reference")]
    BadLoopTarget,
    #[error("a match pattern must be a constant string")]
    BadPattern,
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    #[error("expression is nested deeper than the {MAX_EXPR_DEPTH} level limit")]
    TooDeep,
    #[error(transparent)]
    Collation(#[from] CollationError),
    #[error("invalid logic tree: {0}")]
    Tree(#[from] TreeError),
}

/// Compile expressions and a [`Projection`] into one [`MatchDef`], evaluated in a single
/// pass. `collation` supplies pattern compilation and the missing-field default.
///
/// The overall match result
/// ([`MatchOutcome::matched`](crate::matcher::MatchOutcome::matched)) is the OR of all the
/// expressions, and each expression's individual result is available via
/// [`MatchOutcome::expression_matched`](crate::matcher::MatchOutcome::expression_matched).
/// Expressions are joined with non-short-circuiting `Neor` nodes so every one is fully
/// evaluated.
///
/// The `projection` names document field paths whose values are **captured** during that
/// same pass, and read back with
/// [`MatchOutcome::projected`](crate::matcher::MatchOutcome::projected). Capturing is
/// independent of the match result — a projected field present in the document is captured
/// whether or not the document matched, so the caller decides what to do with it. Pass
/// `&Projection::new()` to capture nothing.
///
/// An empty `exprs` compiles a projection-only definition: nothing ever matches, but every
/// projected field is still extracted.
///
/// ```
/// use jsonsm::collation::DefaultCollation;
/// use jsonsm::compile::{compile, Projection};
/// use jsonsm::matcher::FastMatcher;
///
/// let projection = Projection::new().field(["name"]).field(["age"]);
/// let def = compile(&[], &projection, &DefaultCollation).unwrap();
/// let mut m = FastMatcher::new(&def);
/// let out = m.matches(br#"{"name": "Brett", "age": 41}"#).unwrap();
/// let name = out.projected(0).unwrap();
/// assert_eq!(name.as_str().unwrap().to_decoded_bytes().as_ref(), b"Brett");
/// ```
pub fn compile<C: Collation>(
    exprs: &[Expr],
    projection: &Projection,
    collation: &C,
) -> Result<MatchDef, CompileError> {
    if exprs.iter().any(|e| e.exceeds_depth(MAX_EXPR_DEPTH)) {
        return Err(CompileError::TooDeep);
    }
    let mut t = Transformer::new(collation);
    let expr_buckets = match exprs {
        [] => {
            // No expressions: never matches.
            t.always(false);
            Vec::new()
        }
        [single] => {
            t.transform_one(single)?;
            vec![0]
        }
        many => {
            let mut buckets = Vec::with_capacity(many.len());
            t.transform_merge(many, 0, &mut buckets)?;
            buckets
        }
    };
    t.tree.validate()?;
    // Projections are registered after the expressions so a projected field that is already
    // stored for a cross-field comparison reuses that field's existing slot.
    let projections = t.add_projections(projection);
    // Once the arena is final, work out which slots each loop body owns (cleared per element).
    fill_loop_clear_slots(&mut t.arena);
    // Also once the arena is final: which buckets each node's absence would leave unanswerable.
    fill_seal_buckets(&mut t.arena);
    let mut slot_seen = vec![false; t.slot_idx];
    let mut num_projection_slots = 0;
    for p in &projections {
        if !std::mem::replace(&mut slot_seen[p.slot], true) {
            num_projection_slots += 1;
        }
    }
    Ok(MatchDef {
        arena: t.arena,
        root: 0,
        tree: t.tree,
        root_bucket: 0,
        expr_buckets,
        num_slots: t.slot_idx,
        projections,
        num_projection_slots,
    })
}

/// A loop-variable scope: its variable id and the exec node that roots field lookups.
struct Ctx {
    var: VariableId,
    exec: ExecId,
}

/// The classification of an operand during compilation.
enum Operand {
    /// A field-free operand (a constant, or a function over constants): its `DataRef`
    /// computes the value with no active field.
    Value(DataRef),
    /// An operand referencing exactly one current-context field at `exec`; `dref` computes
    /// the operand value given that field as the active value (either `Active` itself, or a
    /// function wrapping it).
    Field { exec: ExecId, dref: DataRef },
}

struct Transformer<'c, C: Collation> {
    collation: &'c C,
    arena: Vec<ExecNode>,
    tree: LogicTree,
    active: BucketId,
    ctx: Vec<Ctx>,
    slot_idx: usize,
    /// The shallowest scope index any field reference has resolved to since this was last
    /// reset — `Some(0)` means "the document root was read". `transform_loop` uses it to
    /// decide whether a loop must be deferred to an after-loop, and how far out.
    min_ref_scope: Option<usize>,
}

impl<'c, C: Collation> Transformer<'c, C> {
    fn new(collation: &'c C) -> Self {
        Transformer {
            collation,
            arena: vec![ExecNode::default()], // root exec node at id 0
            tree: LogicTree::new(),           // root bucket at 0
            active: 0,
            ctx: vec![Ctx {
                var: jsonsm_ast::ROOT_VAR,
                exec: 0,
            }],
            slot_idx: 0,
            min_ref_scope: None,
        }
    }

    fn cur(&self) -> &Ctx {
        self.ctx.last().expect("context stack is never empty")
    }

    fn push_exec(&mut self) -> ExecId {
        self.arena.push(ExecNode::default());
        self.arena.len() - 1
    }

    /// Navigate/create the exec chain for `path` starting at exec node `base`.
    fn navigate(&mut self, base: ExecId, path: &[PathComponent]) -> ExecId {
        let mut node = base;
        for comp in path {
            node = match comp {
                PathComponent::Key(k) => self.navigate_key(node, k.clone()),
                PathComponent::Index(i) => self.navigate_index(node, *i),
            };
        }
        node
    }

    /// Navigate/create the child of `node` for one field key.
    fn navigate_key(&mut self, node: ExecId, key: String) -> ExecId {
        match self.arena[node].elems.get(key.as_bytes()) {
            Some(child) => child,
            None => {
                let child = self.push_exec();
                self.arena[node].elems.insert(&key, child);
                child
            }
        }
    }

    /// Navigate/create the child of `node` for one array index.
    fn navigate_index(&mut self, node: ExecId, index: usize) -> ExecId {
        if let Some(&(_, child)) = self.arena[node].indexed.iter().find(|(i, _)| *i == index) {
            return child;
        }
        let child = self.push_exec();
        let slot = self.arena[node]
            .indexed
            .partition_point(|(i, _)| *i < index);
        self.arena[node].indexed.insert(slot, (index, child));
        child
    }

    /// Mark every projected path's exec node to store its value, returning the resulting
    /// path → slot mapping. Paths are rooted at the document, so they resolve in the root
    /// exec node regardless of any loop scopes the expressions introduced.
    fn add_projections(&mut self, projection: &Projection) -> Vec<ProjectedField> {
        projection
            .paths()
            .iter()
            .map(|path| {
                // Projected paths are rooted at the document.
                let exec = self.navigate(0, path);
                let slot = self.store_field(exec);
                self.arena[exec].store_projected = true;
                ProjectedField {
                    path: path.clone(),
                    slot,
                }
            })
            .collect()
    }

    /// Resolve a field to its exec node and the depth of the scope it resolved in (`0` is the
    /// document, deeper numbers are enclosing loop bodies). Any scope on the stack is
    /// accepted, at any nesting depth: the depth is recorded in `min_ref_scope` so
    /// [`Self::transform_loop`] can defer the enclosing loop(s) far enough out that the
    /// referenced values are available when the body runs.
    fn resolve_field(&mut self, field: &Field) -> Result<(ExecId, usize), CompileError> {
        let found = self
            .ctx
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.var == field.root)
            .map(|(depth, c)| (depth, c.exec));
        let Some((depth, base)) = found else {
            return Err(CompileError::UnknownVariable(field.root));
        };
        self.min_ref_scope = Some(match self.min_ref_scope {
            Some(m) => m.min(depth),
            None => depth,
        });
        Ok((self.navigate(base, &field.path), depth))
    }

    /// Whether a resolved scope depth is the current (innermost) one.
    fn is_local(&self, depth: usize) -> bool {
        depth + 1 == self.ctx.len()
    }

    /// Classify a comparison operand, resolving fields to exec nodes and building the
    /// `DataRef` that computes it. A local (current-context) field becomes the single
    /// `Active` value; an outer-context field becomes a stored `Slot`. A function may
    /// reference at most one *local* field.
    fn make_operand(&mut self, e: &Expr) -> Result<Operand, CompileError> {
        match e {
            Expr::Value(lit) => Ok(Operand::Value(DataRef::Const(fastval_from_literal(lit)))),
            Expr::Field(f) => {
                let (exec, depth) = self.resolve_field(f)?;
                if self.is_local(depth) {
                    Ok(Operand::Field {
                        exec,
                        dref: DataRef::Active,
                    })
                } else {
                    // An enclosing scope's field: read back from a slot by a deferred op.
                    Ok(Operand::Value(DataRef::Slot(self.store_field(exec))))
                }
            }
            Expr::Func(func) => {
                let mut params = Vec::with_capacity(func.args.len());
                let mut active: Option<ExecId> = None;
                for arg in &func.args {
                    match self.make_operand(arg)? {
                        Operand::Value(d) => params.push(d),
                        Operand::Field { exec, dref } => {
                            if active.is_some() {
                                return Err(CompileError::Unsupported(
                                    "function with multiple local field arguments",
                                ));
                            }
                            active = Some(exec);
                            params.push(dref);
                        }
                    }
                }
                let dref = DataRef::Func(FuncRef {
                    name: func.name.clone(),
                    params,
                });
                Ok(match active {
                    Some(exec) => Operand::Field { exec, dref },
                    None => Operand::Value(dref),
                })
            }
            _ => Err(CompileError::NotAnOperand),
        }
    }

    /// Resolve the field operand of an operator that inspects a value directly (`exists`,
    /// `matches`), returning the exec node to attach the op to and the [`DataRef`] that reads it.
    ///
    /// Mirrors what [`Transformer::make_operand`] does for comparisons. A field in the current
    /// scope is the actively scanned value, so the op belongs on that field's own node. A field
    /// from an **enclosing** scope is stored in a slot and the op goes on the current scope's node
    /// instead — which is visited unconditionally (once per element, in a loop body), and by then
    /// the slot is filled, because `resolve_field` has recorded the scope depth and that is what
    /// defers the enclosing loop far enough out. This is the same route `name = "a"` inside a loop
    /// body already takes; only `exists`/`matches` were missing it.
    fn value_operand(&mut self, e: &Expr) -> Result<(ExecId, DataRef), CompileError> {
        match e {
            Expr::Field(f) => {
                let (exec, depth) = self.resolve_field(f)?;
                if self.is_local(depth) {
                    Ok((exec, DataRef::Active))
                } else {
                    let slot = self.store_field(exec);
                    Ok((self.cur().exec, DataRef::Slot(slot)))
                }
            }
            Expr::Func(_) => Err(CompileError::Func),
            _ => Err(CompileError::NotAnOperand),
        }
    }

    /// Resolve an expression that must be a plain current-context field, returning its exec node.
    ///
    /// Only loop targets need this now: the array a loop iterates has to be scanned in the scope
    /// the loop lives in. Operators that merely *read* a value (`exists`, `matches`) go through
    /// [`Transformer::value_operand`], which accepts an enclosing scope's field via a slot.
    fn require_field(&mut self, e: &Expr) -> Result<ExecId, CompileError> {
        match e {
            Expr::Field(f) => match self.resolve_field(f)? {
                (exec, depth) if self.is_local(depth) => Ok(exec),
                _ => Err(CompileError::CrossContext),
            },
            Expr::Func(_) => Err(CompileError::Func),
            _ => Err(CompileError::NotAnOperand),
        }
    }

    fn add_op(&mut self, exec: ExecId, kind: OpKind) {
        let bucket = self.active;
        self.arena[exec].ops.push(OpNode { bucket, kind });
    }

    /// Merge multiple expressions under non-short-circuiting `Neor` nodes so every one is
    /// fully evaluated (enabling per-expression results). Records each expression's bucket
    /// into `buckets`.
    fn transform_merge(
        &mut self,
        exprs: &[Expr],
        i: usize,
        buckets: &mut Vec<BucketId>,
    ) -> Result<(), CompileError> {
        if i == exprs.len() - 1 {
            buckets.push(self.active);
            return self.transform_one(&exprs[i]);
        }
        let base = self.active;
        self.tree.set_type(base, NodeType::Neor);
        let left = self.tree.add_child(base);
        self.tree.set_left(base, left);
        self.active = left;
        buckets.push(left);
        self.transform_one(&exprs[i])?;
        let right = self.tree.add_child(base);
        self.tree.set_right(base, right);
        self.active = right;
        self.transform_merge(exprs, i + 1, buckets)
    }

    /// Ensure `exec`'s scanned value is stored in a slot, returning that slot.
    fn store_field(&mut self, exec: ExecId) -> SlotId {
        if let Some(slot) = self.arena[exec].store {
            return slot;
        }
        let slot = self.slot_idx;
        self.slot_idx += 1;
        self.arena[exec].store = Some(slot);
        slot
    }

    /// Attach a deferred op to the current scope's root exec node.
    fn add_after_op(&mut self, kind: OpKind) {
        let bucket = self.active;
        let exec = self.cur().exec;
        self.arena[exec]
            .after
            .get_or_insert_with(AfterNode::default)
            .ops
            .push(OpNode { bucket, kind });
    }

    fn transform_one(&mut self, expr: &Expr) -> Result<(), CompileError> {
        match expr {
            Expr::True => {
                self.always(true);
                Ok(())
            }
            Expr::False => {
                self.always(false);
                Ok(())
            }
            Expr::And(subs) if subs.is_empty() => {
                self.always(true);
                Ok(())
            }
            Expr::Or(subs) if subs.is_empty() => {
                self.always(false);
                Ok(())
            }
            Expr::And(subs) => self.transform_junction(NodeType::And, subs),
            Expr::Or(subs) => self.transform_junction(NodeType::Or, subs),
            Expr::Not(sub) => self.transform_not(sub),
            Expr::Exists(sub) => self.transform_exists(sub),
            Expr::NotExists(sub) => self.transform_not(&Expr::Exists(sub.clone())),
            Expr::Compare {
                op: CompareOp::NotEquals,
                lhs,
                rhs,
            } => {
                let inner = Expr::compare(CompareOp::Equals, (**lhs).clone(), (**rhs).clone());
                self.transform_not(&inner)
            }
            Expr::Compare { op, lhs, rhs } => self.transform_compare(*op, lhs, rhs),
            Expr::Matches { lhs, pattern } => self.transform_matches(lhs, pattern),
            Expr::Loop {
                loop_type,
                var,
                in_expr,
                sub_expr,
            } => self.transform_loop(*loop_type, *var, in_expr, sub_expr),
            Expr::Value(_) | Expr::Field(_) | Expr::Func(_) => Err(CompileError::NotABoolean),
        }
    }

    fn always(&mut self, value: bool) {
        let exec = self.cur().exec;
        self.add_op(exec, OpKind::Always(value));
    }

    fn transform_junction(&mut self, ty: NodeType, subs: &[Expr]) -> Result<(), CompileError> {
        if subs.len() == 1 {
            return self.transform_one(&subs[0]);
        }
        let base = self.active;
        self.tree.set_type(base, ty);
        let left = self.tree.add_child(base);
        self.tree.set_left(base, left);
        self.active = left;
        self.transform_one(&subs[0])?;
        let right = self.tree.add_child(base);
        self.tree.set_right(base, right);
        self.active = right;
        // Right-associate the remainder under the right child.
        if subs.len() == 2 {
            self.transform_one(&subs[1])
        } else {
            self.transform_junction(ty, &subs[1..])
        }
    }

    fn transform_not(&mut self, sub: &Expr) -> Result<(), CompileError> {
        let base = self.active;
        self.tree.set_type(base, NodeType::Not);
        let left = self.tree.add_child(base);
        self.tree.set_left(base, left);
        self.active = left;
        self.transform_one(sub)
    }

    fn transform_exists(&mut self, sub: &Expr) -> Result<(), CompileError> {
        let (exec, of) = self.value_operand(sub)?;
        self.add_op(exec, OpKind::Exists { of });
        Ok(())
    }

    fn transform_compare(
        &mut self,
        op: CompareOp,
        lhs: &Expr,
        rhs: &Expr,
    ) -> Result<(), CompileError> {
        let cmp = CmpOp::from_ast(op).expect("NotEquals lowered before here");
        let cur_var = self.cur().var;

        // When the comparison references at most one *local* (current-context) field, that
        // field can be the single Active value and the op runs inline as the field is
        // scanned (fast path); outer-context fields become slots on either path. With two
        // or more local fields, no single value suffices, so store every referenced field
        // in a slot and defer the whole comparison to the current scope's after-node —
        // which sees all of them once the scope is fully parsed (at the root: after the
        // document; in a loop body: after each element). This uniformly covers
        // field-vs-field, `f(a, b) <op> const`, and `f(a) <op> b`, in any context.
        if count_local_fields(lhs, cur_var) + count_local_fields(rhs, cur_var) <= 1 {
            let lhs_ref;
            let rhs_ref;
            let exec = match (self.make_operand(lhs)?, self.make_operand(rhs)?) {
                (Operand::Field { exec, dref: ld }, Operand::Value(rd)) => {
                    (lhs_ref, rhs_ref) = (ld, rd);
                    exec
                }
                (Operand::Value(ld), Operand::Field { exec, dref: rd }) => {
                    (lhs_ref, rhs_ref) = (ld, rd);
                    exec
                }
                // No field at all: evaluate at the current scope's node (always visited).
                (Operand::Value(ld), Operand::Value(rd)) => {
                    (lhs_ref, rhs_ref) = (ld, rd);
                    self.cur().exec
                }
                (Operand::Field { .. }, Operand::Field { .. }) => {
                    unreachable!("at most one field total on this path")
                }
            };
            self.add_op(
                exec,
                OpKind::Compare {
                    op: cmp,
                    lhs: lhs_ref,
                    rhs: rhs_ref,
                },
            );
        } else {
            let lhs_ref = self.operand_slotref(lhs)?;
            let rhs_ref = self.operand_slotref(rhs)?;
            self.add_after_op(OpKind::Compare {
                op: cmp,
                lhs: lhs_ref,
                rhs: rhs_ref,
            });
        }
        Ok(())
    }

    /// Build an operand's [`DataRef`] with every field reference stored in a slot (so a
    /// deferred after-node op can read it). Used for multi-field comparisons.
    fn operand_slotref(&mut self, e: &Expr) -> Result<DataRef, CompileError> {
        match e {
            Expr::Value(lit) => Ok(DataRef::Const(fastval_from_literal(lit))),
            Expr::Field(f) => {
                let (exec, _depth) = self.resolve_field(f)?;
                Ok(DataRef::Slot(self.store_field(exec)))
            }
            Expr::Func(func) => {
                let mut params = Vec::with_capacity(func.args.len());
                for arg in &func.args {
                    params.push(self.operand_slotref(arg)?);
                }
                Ok(DataRef::Func(FuncRef {
                    name: func.name.clone(),
                    params,
                }))
            }
            _ => Err(CompileError::NotAnOperand),
        }
    }

    fn transform_matches(&mut self, lhs: &Expr, pattern: &Expr) -> Result<(), CompileError> {
        let (exec, of) = self.value_operand(lhs)?;
        let pattern_str = match pattern {
            Expr::Value(Literal::String(s)) => s.as_str(),
            _ => return Err(CompileError::BadPattern),
        };
        let matcher = self.collation.compile_matcher(pattern_str)?;
        self.add_op(
            exec,
            OpKind::Matches {
                matcher: Arc::from(matcher),
                of,
            },
        );
        Ok(())
    }

    fn transform_loop(
        &mut self,
        mode: LoopType,
        var: VariableId,
        in_expr: &Expr,
        sub_expr: &Expr,
    ) -> Result<(), CompileError> {
        // The array being looped must be a field in the current context.
        let in_exec = self.require_field(in_expr).map_err(|e| match e {
            CompileError::Func | CompileError::NotAnOperand => CompileError::BadLoopTarget,
            other => other,
        })?;

        let base = self.active;
        self.tree.set_type(base, NodeType::Loop);
        let body_bucket = self.tree.add_child(base);
        self.tree.set_left(base, body_bucket);
        let body_exec = self.push_exec();
        // The scope this loop lives in; its body is one deeper.
        let host_scope = self.ctx.len() - 1;
        let body_scope = host_scope + 1;

        // Transform the body, isolating which scopes *it* reads.
        let saved_min = self.min_ref_scope.take();
        self.ctx.push(Ctx {
            var,
            exec: body_exec,
        });
        self.active = body_bucket;
        let result = self.transform_one(sub_expr);
        self.ctx.pop();
        let body_min = self.min_ref_scope.take();
        // Restore the enclosing scope's tracking, propagating only references that reach
        // *past* this body — those make the enclosing loop defer too, recursively.
        self.min_ref_scope = match (saved_min, body_min.filter(|&m| m < body_scope)) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        result?;

        if body_min.is_some_and(|m| m <= host_scope) {
            // The body reads fields from the scope containing this loop (or shallower), which
            // are only knowable once that scope has been fully parsed. Store the array and
            // defer the loop to that scope's after-node: at the root it runs after the whole
            // document, and inside an enclosing loop body it runs after each element. If the
            // body reached further out still, the enclosing loop was deferred as well (see
            // the propagation above), so by the time this loop runs every scope it reads has
            // been parsed — regardless of document field order.
            let array_slot = self.store_field(in_exec);
            let host_exec = self.ctx[host_scope].exec;
            self.arena[host_exec]
                .after
                .get_or_insert_with(AfterNode::default)
                .loops
                .push(AfterLoopNode {
                    bucket: body_bucket,
                    mode,
                    node: body_exec,
                    array_slot,
                    clear_slots: Vec::new(), // filled by `fill_loop_clear_slots`
                });
        } else {
            // Inline loop over the array as it is scanned.
            self.arena[in_exec].loops.push(LoopNode {
                bucket: body_bucket,
                mode,
                node: body_exec,
                clear_slots: Vec::new(), // filled by `fill_loop_clear_slots`
            });
        }
        Ok(())
    }
}

/// Record, for every loop, the slots stored by nodes **inside its body**, so the matcher can
/// clear them before each iteration.
///
/// Slots live for the whole document, but a loop body's slots describe *one element*: without
/// clearing, an element missing a field would read the previous element's value (e.g.
/// `ANY e IN a SATISFIES e.x == e.y` on `[{"x":1,"y":2},{"x":2}]` would see `y = 2` for the
/// second element and wrongly match). Only nodes within the body subtree are collected —
/// outer-scope slots the body reads (cross-scope references) must survive, and projection
/// slots are never cleared so the matcher's pending-capture count stays exact.
///
/// Run as a post-pass, once the arena is final.
fn fill_loop_clear_slots(arena: &mut [ExecNode]) {
    // Collect every loop's location first: `(owner node, in an after-node?, index, body)`.
    let mut loops: Vec<(ExecId, bool, usize, ExecId)> = Vec::new();
    for (id, node) in arena.iter().enumerate() {
        loops.extend(
            node.loops
                .iter()
                .enumerate()
                .map(|(i, l)| (id, false, i, l.node)),
        );
        if let Some(after) = &node.after {
            loops.extend(
                after
                    .loops
                    .iter()
                    .enumerate()
                    .map(|(i, l)| (id, true, i, l.node)),
            );
        }
    }

    for (owner, in_after, i, body) in loops {
        let slots = subtree_slots(arena, body);
        if in_after {
            arena[owner]
                .after
                .as_mut()
                .expect("after-node exists")
                .loops[i]
                .clear_slots = slots;
        } else {
            arena[owner].loops[i].clear_slots = slots;
        }
    }
}

/// The slots stored by `root` and everything beneath it in the exec trie (which is a tree:
/// each node has exactly one parent path). Projection slots are excluded — see
/// [`fill_loop_clear_slots`].
fn subtree_slots(arena: &[ExecNode], root: ExecId) -> Vec<SlotId> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let node = &arena[id];
        if let Some(slot) = node.store {
            if !node.store_projected {
                out.push(slot);
            }
        }
        stack.extend(node.elems.values());
        stack.extend(node.indexed.iter().map(|&(_, child)| child));
        stack.extend(node.loops.iter().map(|l| l.node));
        if let Some(after) = &node.after {
            stack.extend(after.loops.iter().map(|l| l.node));
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Fill every node's [`ExecNode::seal_buckets`] — the buckets its exec subtree writes.
fn fill_seal_buckets(arena: &mut [ExecNode]) {
    for id in 0..arena.len() {
        arena[id].seal_buckets = subtree_buckets(arena, id);
    }
}

/// Every logic-tree bucket written anywhere in `root`'s exec subtree, with the value it takes
/// if its field is absent.
///
/// Walks the same edges as [`subtree_slots`] and collects buckets instead of slots: a node's own
/// ops, its loops' body buckets, its deferred ops and loops, and all of that again for every
/// child. A loop contributes its body bucket *and* the buckets inside the body, so an array field
/// that never appears seals its whole body rather than only the loop's result.
fn subtree_buckets(arena: &[ExecNode], root: ExecId) -> Vec<(BucketId, Tri)> {
    /// The value a bucket takes when the op that would have written it never runs.
    fn absent_value(kind: &OpKind) -> Tri {
        match kind {
            // Presence is exactly what this asks, and the field is not present.
            OpKind::Exists { .. } => Tri::False,
            _ => Tri::Unknown,
        }
    }

    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let node = &arena[id];
        out.extend(node.ops.iter().map(|o| (o.bucket, absent_value(&o.kind))));
        // An absent array is not an empty one: neither `ANY` nor `EVERY` over it has an answer.
        out.extend(node.loops.iter().map(|l| (l.bucket, Tri::Unknown)));
        if let Some(after) = &node.after {
            out.extend(after.ops.iter().map(|o| (o.bucket, absent_value(&o.kind))));
            out.extend(after.loops.iter().map(|l| (l.bucket, Tri::Unknown)));
        }
        stack.extend(node.elems.values());
        stack.extend(node.indexed.iter().map(|&(_, child)| child));
        stack.extend(node.loops.iter().map(|l| l.node));
        if let Some(after) = &node.after {
            stack.extend(after.loops.iter().map(|l| l.node));
        }
    }
    out.sort_unstable_by_key(|&(b, _)| b);
    out.dedup_by_key(|&mut (b, _)| b);
    out
}

/// Count the *local* (current-context) field references within an operand expression
/// (recursing through function arguments). Outer-context fields become stored slots on
/// either path, so they do not count toward the single-Active fast-path decision.
fn count_local_fields(e: &Expr, cur_var: VariableId) -> usize {
    match e {
        Expr::Field(f) => usize::from(f.root == cur_var),
        Expr::Func(func) => func
            .args
            .iter()
            .map(|a| count_local_fields(a, cur_var))
            .sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collation::DefaultCollation;

    fn field(keys: &[&str]) -> Expr {
        Expr::Field(Field::root(
            keys.iter()
                .map(|k| PathComponent::Key((*k).to_owned()))
                .collect(),
        ))
    }

    /// A path of plain object keys.
    fn key_path(keys: &[&str]) -> Vec<PathComponent> {
        keys.iter()
            .map(|k| PathComponent::Key((*k).into()))
            .collect()
    }

    /// The `KeyMap` tag is a prefilter, never an answer: keys agreeing on length, first and
    /// last byte must still be separated by the full compare. `bat`/`bit` and the empty key
    /// are the cases that would survive a tag-only lookup.
    #[test]
    fn keymap_tag_is_only_a_prefilter() {
        let mut m = KeyMap::default();
        for (i, k) in ["bat", "bit", "", "b", "batt", "tab"].iter().enumerate() {
            m.insert(k, i);
        }
        for (i, k) in ["bat", "bit", "", "b", "batt", "tab"].iter().enumerate() {
            assert_eq!(m.get(k.as_bytes()), Some(i), "{k:?}");
        }
        assert_eq!(key_tag(b"bat"), key_tag(b"bit"), "tags collide as designed");
        for absent in ["bot", "ba", "batty", "x", "BAT"] {
            assert_eq!(m.get(absent.as_bytes()), None, "{absent:?}");
        }
        // A key is bytes, not text: invalid UTF-8 simply never equals an inserted key.
        assert_eq!(m.get(&[0xff, 0xfe, 0xff]), None);
        assert!(!m.is_empty());
        assert_eq!(m.values().count(), 6);
        assert!(KeyMap::default().is_empty());
    }

    fn compile_ok(expr: &Expr) -> MatchDef {
        compile(
            std::slice::from_ref(expr),
            &Projection::new(),
            &DefaultCollation,
        )
        .expect("compiles")
    }

    /// Compile one expression, expecting an error.
    fn compile_err(expr: &Expr) -> CompileError {
        compile(
            std::slice::from_ref(expr),
            &Projection::new(),
            &DefaultCollation,
        )
        .expect_err("should not compile")
    }

    #[test]
    fn compiles_field_vs_constant() {
        let d = compile_ok(&Expr::compare(
            CompareOp::Equals,
            field(&["name"]),
            Expr::Value(Literal::String("Brett".into())),
        ));
        // Single leaf bucket; op attached under the "name" child of the root exec node.
        assert_eq!(d.num_buckets(), 1);
        let name = d.arena[d.root].elems["name"];
        assert_eq!(d.arena[name].ops.len(), 1);
        assert!(matches!(
            d.arena[name].ops[0].kind,
            OpKind::Compare { op: CmpOp::Eq, .. }
        ));
        assert_eq!(d.arena[name].ops[0].bucket, 0);
    }

    #[test]
    fn compiles_and_or_into_tree() {
        let d = compile_ok(&Expr::And(vec![
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["active"]),
                Expr::Value(Literal::Bool(true)),
            ),
        ]));
        // root And + two leaves.
        assert_eq!(d.num_buckets(), 3);
    }

    #[test]
    fn lowers_notequals_to_not_equals() {
        let d = compile_ok(&Expr::compare(
            CompareOp::NotEquals,
            field(&["x"]),
            Expr::Value(Literal::Int(1)),
        ));
        // root Not + one leaf (the Equals).
        assert_eq!(d.num_buckets(), 2);
        let x = d.arena[d.root].elems["x"];
        assert!(matches!(
            d.arena[x].ops[0].kind,
            OpKind::Compare { op: CmpOp::Eq, .. }
        ));
        // The op reports into the Not's child bucket (1), not the root.
        assert_eq!(d.arena[x].ops[0].bucket, 1);
    }

    /// `exists` and `matches` reach an enclosing scope's field; a **loop target** still may not.
    ///
    /// Pins the boundary in both directions, so `CrossContext` narrowing to exactly one case is a
    /// stated property rather than something the next reader has to rediscover. The array a loop
    /// iterates has to be scanned in the scope the loop lives in; an operator that merely reads a
    /// value can take it from a slot, which is what the first two accept.
    #[test]
    fn cross_scope_is_allowed_for_value_operators_but_not_loop_targets() {
        // ANY t IN tags SATISFIES <op on the document-scope field `name`> END
        let body_over_outer = |body: Expr| Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(body),
        };
        let outer = || field(&["name"]);

        compile_ok(&body_over_outer(Expr::Exists(Box::new(outer()))));
        compile_ok(&body_over_outer(Expr::Matches {
            lhs: Box::new(outer()),
            pattern: Box::new(Expr::Value(Literal::String("b".into()))),
        }));
        // A comparison always could, and still can.
        compile_ok(&body_over_outer(Expr::compare(
            CompareOp::Equals,
            outer(),
            Expr::Value(Literal::Int(1)),
        )));

        // But a nested loop whose *array* comes from an enclosing scope is still rejected.
        let inner_over_outer = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::Loop {
                loop_type: LoopType::Any,
                var: 2,
                // `ys` is rooted at the document, not at the outer loop's element.
                in_expr: Box::new(field(&["ys"])),
                sub_expr: Box::new(Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field {
                        root: 2,
                        path: vec![],
                    }),
                    Expr::Value(Literal::Int(1)),
                )),
            }),
        };
        assert!(matches!(
            compile_err(&inner_over_outer),
            CompileError::CrossContext
        ));
    }

    #[test]
    fn compiles_exists_and_notexists() {
        let d = compile_ok(&Expr::Exists(Box::new(field(&["maybe"]))));
        let n = d.arena[d.root].elems["maybe"];
        assert!(matches!(d.arena[n].ops[0].kind, OpKind::Exists { .. }));

        let d = compile_ok(&Expr::NotExists(Box::new(field(&["maybe"]))));
        assert_eq!(d.num_buckets(), 2); // Not + Exists leaf
    }

    #[test]
    fn compiles_loop_with_body_scope() {
        let d = compile_ok(&Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("x".into())),
            )),
        });
        // root Loop + body leaf.
        assert_eq!(d.num_buckets(), 2);
        let tags = d.arena[d.root].elems["tags"];
        assert_eq!(d.arena[tags].loops.len(), 1);
        let body_exec = d.arena[tags].loops[0].node;
        // The body op compares the loop element itself (Active) to a constant.
        assert!(matches!(
            d.arena[body_exec].ops[0].kind,
            OpKind::Compare { .. }
        ));
    }

    #[test]
    fn compiles_regex_matches() {
        let d = compile_ok(&Expr::Matches {
            lhs: Box::new(field(&["email"])),
            pattern: Box::new(Expr::Value(Literal::String("@x\\.com$".into()))),
        });
        let n = d.arena[d.root].elems["email"];
        assert!(matches!(d.arena[n].ops[0].kind, OpKind::Matches { .. }));
    }

    #[test]
    fn compiles_root_cross_field_with_slots_and_after() {
        let d = compile_ok(&Expr::compare(
            CompareOp::Equals,
            field(&["a"]),
            field(&["b"]),
        ));
        assert_eq!(d.num_slots(), 2);
        let a = d.arena[d.root].elems["a"];
        let b = d.arena[d.root].elems["b"];
        assert!(d.arena[a].store.is_some());
        assert!(d.arena[b].store.is_some());
        let after = d.arena[d.root].after.as_ref().expect("after node");
        assert_eq!(after.ops.len(), 1);
        assert!(matches!(
            after.ops[0].kind,
            OpKind::Compare {
                lhs: DataRef::Slot(_),
                rhs: DataRef::Slot(_),
                ..
            }
        ));
    }

    #[test]
    fn multi_field_comparisons_use_slots_and_after() {
        // Two-field function argument: mathAdd(a, b) == 1 -> both fields stored, deferred.
        let d = compile_ok(&Expr::compare(
            CompareOp::Equals,
            Expr::Func(jsonsm_ast::Func {
                name: "mathAdd".into(),
                args: vec![field(&["a"]), field(&["b"])],
            }),
            Expr::Value(Literal::Int(1)),
        ));
        assert_eq!(d.num_slots(), 2);
        let after = d.arena[d.root].after.as_ref().expect("after node");
        assert!(matches!(
            after.ops[0].kind,
            OpKind::Compare {
                lhs: DataRef::Func(_),
                rhs: DataRef::Const(_),
                ..
            }
        ));

        // Cross-field inside a loop body now compiles (body's after runs per element).
        compile_ok(&Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["arr"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("a".into())],
                }),
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("b".into())],
                }),
            )),
        });
    }

    #[test]
    fn projection_marks_fields_for_storage() {
        let projection = Projection::new().field(["name", "first"]).field(["age"]);
        let d = compile(
            &[Expr::compare(
                CompareOp::Equals,
                field(&["age"]),
                Expr::Value(Literal::Int(1)),
            )],
            &projection,
            &DefaultCollation,
        )
        .expect("compiles");

        assert_eq!(d.num_projections(), 2);
        assert_eq!(d.projection_path(0), key_path(&["name", "first"]));
        assert_eq!(d.projection_index(&key_path(&["age"])), Some(1));
        assert_eq!(d.projection_index(&key_path(&["nope"])), None);
        assert_eq!(d.num_projection_slots, 2);
        assert_eq!(d.num_slots(), 2);

        // Both projected paths resolved to exec nodes under the document root, marked to
        // store their scanned range.
        let name = d.arena[d.root].elems["name"];
        let first = d.arena[name].elems["first"];
        let age = d.arena[d.root].elems["age"];
        assert!(d.arena[first].store.is_some() && d.arena[first].store_projected);
        assert!(d.arena[age].store.is_some() && d.arena[age].store_projected);
        // An intermediate path node is navigated but not itself stored.
        assert!(d.arena[name].store.is_none());
        // The `age` comparison is untouched by projecting the same field.
        assert_eq!(d.arena[age].ops.len(), 1);
    }

    #[test]
    fn projection_shares_slots_with_cross_field_comparisons() {
        // `a == b` already stores both fields; projecting them adds no new slots.
        let d = compile(
            &[Expr::compare(
                CompareOp::Equals,
                field(&["a"]),
                field(&["b"]),
            )],
            &Projection::new().field(["a"]).field(["b"]),
            &DefaultCollation,
        )
        .expect("compiles");
        assert_eq!(d.num_slots(), 2);
        assert_eq!(d.num_projection_slots, 2);
    }

    #[test]
    fn duplicate_projection_paths_share_one_slot() {
        let d = compile(
            &[],
            &Projection::new().field(["a"]).field(["a"]),
            &DefaultCollation,
        )
        .expect("compiles");
        assert_eq!(d.num_projections(), 2);
        assert_eq!(d.num_slots(), 1);
        assert_eq!(d.num_projection_slots, 1);
    }

    #[test]
    fn projection_only_definition_compiles() {
        // No expressions: never matches, but the projection is still compiled.
        let d = compile(
            &[],
            &Projection::from_iter([vec!["a"], vec![]]),
            &DefaultCollation,
        )
        .expect("compiles");
        assert_eq!(d.num_expressions(), 0);
        assert_eq!(d.num_projections(), 2);
        // The empty path is the document root itself.
        assert!(d.projection_path(1).is_empty());
        assert!(d.arena[d.root].store_projected);
    }

    #[test]
    fn rejects_unsupported_shapes() {
        // exists on a function is not a field
        assert!(matches!(
            compile_err(&Expr::Exists(Box::new(Expr::Func(jsonsm_ast::Func {
                name: "mathAbs".into(),
                args: vec![field(&["a"])]
            })))),
            CompileError::Func
        ));
        // bad pattern (non-string)
        assert!(matches!(
            compile_err(&Expr::Matches {
                lhs: Box::new(field(&["a"])),
                pattern: Box::new(Expr::Value(Literal::Int(1))),
            }),
            CompileError::BadPattern
        ));
    }
}

/// [`KeyMap::match_quoted`] is a hand-rolled byte comparison, and the differential sweep is
/// structurally weak at exactly that: its oracle navigates by name, so a comparison that is
/// wrong only for keys the generator never emits produces the same answer on both sides.
/// These check it against a reference that shares none of its machinery — find the closing
/// quote, then ask `slice::==` — over every pairing of a small set of adversarial keys.
///
/// Key equality is byte equality, with no collation and no Unicode normalisation: JSON
/// strings are UTF-8 by specification, so two keys name the same field exactly when their
/// decoded bytes agree. That is what makes comparing the document's raw bytes legitimate in
/// the first place.
#[cfg(test)]
mod key_match_tests {
    use super::*;

    /// Deliberately clustered around the eight-byte word `match_quoted` prefilters on, and
    /// around each other: prefixes, extensions, and pairs differing only past the word.
    const KEYS: &[&str] = &[
        "a",
        "ab",
        "abc",
        "abcdef",
        "abcdefg",
        "abcdefgh",
        "abcdefghi",
        "abcdefghij",
        "abcdefghi_",
        "b",
        "",
        "\u{e9}\u{e9}\u{e9}\u{e9}",
    ];

    fn map_of(keys: &[&str]) -> KeyMap {
        let mut m = KeyMap::default();
        for (i, k) in keys.iter().enumerate() {
            m.insert(k, i);
        }
        m
    }

    /// What the answer should be, computed without any of the code under test: take the
    /// document key to be the bytes up to the next quote, then compare whole strings.
    fn reference(keys: &[&str], doc: &[u8]) -> Option<(ExecId, usize)> {
        assert_eq!(doc.first(), Some(&b'"'));
        let end = doc[1..].iter().position(|&b| b == b'"')? + 1;
        let key = &doc[1..end];
        keys.iter()
            .position(|k| k.as_bytes() == key)
            .map(|i| (i, end + 1))
    }

    #[test]
    fn match_quoted_agrees_with_a_scan_and_compare() {
        let map = map_of(KEYS);
        for probe in KEYS {
            // Vary what follows the key: a hit must not depend on it, and a near-miss whose
            // trailing bytes happen to look like a key's must still be rejected.
            for tail in [":1", ":\"x\"", "\":1", "x\":1", "\u{e9}\":1"] {
                let doc = format!("\"{probe}{tail}").into_bytes();
                let word = head_word(&doc);
                assert_eq!(
                    map.match_quoted(word, &doc),
                    reference(KEYS, &doc),
                    "key {probe:?} tail {tail:?}"
                );
            }
        }
    }

    /// A key running up to the very end of the buffer, where the eight-byte load reads short
    /// and is zero-padded. The padding must never complete a match.
    #[test]
    fn match_quoted_never_matches_past_the_end() {
        let map = map_of(KEYS);
        for probe in KEYS {
            let full = format!("\"{probe}\":1").into_bytes();
            for cut in 0..full.len() {
                let doc = &full[..cut];
                if doc.first() != Some(&b'"') {
                    continue;
                }
                assert_eq!(
                    map.match_quoted(head_word(doc), doc),
                    reference(KEYS, doc),
                    "key {probe:?} truncated to {cut}"
                );
            }
        }
    }

    /// A map is only comparable raw when every key is its own JSON encoding, since the
    /// closing quote is what makes a match conclusive. A key ending in a backslash is the
    /// case that breaks it: quoted, `a\` is a prefix of the document key `a"x`.
    #[test]
    fn a_key_needing_escapes_disables_raw_comparison() {
        assert!(map_of(&["plain", "also_plain"]).verbatim());
        assert!(!map_of(&["plain", "back\\slash"]).verbatim());
        assert!(!map_of(&["plain", "quo\"te"]).verbatim());
        assert!(!map_of(&["plain", "nl\n"]).verbatim());
        // The one that would actually mismatch, spelled out. `reference` is no use here —
        // it finds the closing quote by scanning for `"`, which is the same thing being
        // fooled — so the authority is a real JSON decoder.
        let unsound = map_of(&["a\\"]);
        assert!(!unsound.verbatim());
        let doc = br#"{"a\"x":1}"#;
        let decoded: serde_json::Value = serde_json::from_slice(doc).unwrap();
        assert_eq!(
            decoded.as_object().unwrap().keys().next().unwrap(),
            "a\"x",
            "the document's only key"
        );
        // Yet the raw comparison would claim the key `a\` is here, which is why a map
        // holding it must never reach `match_quoted`.
        assert_eq!(
            unsound.match_quoted_inner(head_word(&doc[1..]), &doc[1..]),
            Some((0, 4))
        );
    }
}
