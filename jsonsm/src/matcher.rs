//! The single-pass [`FastMatcher`]: evaluates a compiled [`MatchDef`] against a raw JSON
//! document in one tokenizing scan, with zero allocation on the hot path.
//!
//! It walks the tokenizer and the exec trie in lockstep. At each field that matters it
//! runs the attached operations (recording their results into the logic tree) and recurses
//! into sub-objects / loops; irrelevant fields are structurally skipped. Short-circuiting
//! lets the scan stop as soon as the root bucket is decided.
//!
//! A comparison whose field is absent has no answer, so its leaf resolves to
//! [`Unknown`](crate::logic_tree) rather than to a boolean. Absence is discovered at the close of
//! the enclosing container — that is the moment "not seen yet" becomes "not there" — where
//! `seal_absent` settles every bucket beneath it that is still unset, letting the tree resolve as
//! soon as the logic allows rather than after the document ends.
//!
//! If the definition carries a [`Projection`](crate::compile::Projection), the same scan
//! also **captures** those fields' values, read back from the returned [`MatchOutcome`].
//! Capture is independent of the match result, so the scan short-circuits only once the
//! logic tree *and* every projected field are settled.

use crate::collation::{Collation, DefaultCollation};
use crate::compile::{
    AfterNode, BucketId, CmpOp, DataRef, ExecId, ExecNode, KeyMap, head_word, LoopNode, MatchDef,
    OpKind, OpNode, SlotId,
};
use crate::logic_tree::{LogicTreeState, Tri};
use crate::tokenizer::{
    GenericTokenizer, Scan, SkipError, Token, TokenType, Tokenizer, TokenizerError,
};
use crate::value::{FastStr, FastVal};
use jsonsm_ast::{LoopType, PathComponent};
use std::cmp::Ordering;

/// A stored value's location in the document: `(start, len)` in bytes.
type SlotRange = (usize, usize);

/// Maximum nesting depth of a document the matcher will scan.
///
/// Structural skipping is iterative and recursion only follows the *expression's* field paths,
/// so depth does not by itself threaten the stack — but an explicit ceiling makes the bound a
/// stated guarantee rather than an emergent property, and rejects absurd input early.
///
/// Defined once and shared with the bulk skip, which enforces the same ceiling on nesting it
/// crosses without tokenizing — otherwise the limit would depend on whether an expression
/// happened to name a field in the deep region or skip past it.
pub const MAX_DEPTH: usize = crate::tokenizer::MAX_SKIP_DEPTH;

/// An error encountered while matching a document.
#[derive(Debug, thiserror::Error)]
pub enum MatchError {
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("malformed JSON structure: {0}")]
    Structure(&'static str),
    #[error("document is nested deeper than the {MAX_DEPTH} level limit")]
    TooDeep,
}

/// A reusable matcher for one compiled [`MatchDef`].
///
/// Borrows the `MatchDef` (which is `Sync`), so a shared def can back one `FastMatcher`
/// per thread. Call [`FastMatcher::matches`] repeatedly; state is reset each call.
pub struct FastMatcher<'d, C: Collation = DefaultCollation> {
    def: &'d MatchDef,
    collation: C,
    state: LogicTreeState<'d>,
    /// Per-match storage: byte `(start, len)` of each stored field's value, filled as the
    /// document is scanned and read back by deferred after-node ops and by projection.
    slots: Vec<Option<SlotRange>>,
    /// How many projection slots are still unfilled. While non-zero the scan must not
    /// short-circuit, or a projected field appearing later in the document would be missed.
    pending_projections: usize,
    /// Which scan backend this matcher runs. Resolved **once**, here, by CPU feature
    /// detection; [`FastMatcher::scan`] branches on it a single time per document and
    /// everything below that point is monomorphised for the chosen backend.
    #[cfg(feature = "simd")]
    backend: crate::simd::Backend,
}

impl<'d> FastMatcher<'d, DefaultCollation> {
    /// Create a matcher using [`DefaultCollation`].
    pub fn new(def: &'d MatchDef) -> Self {
        Self::with_collation(def, DefaultCollation)
    }
}

impl<'d, C: Collation> FastMatcher<'d, C> {
    /// Create a matcher with an explicit collation. It should match the collation the
    /// `def` was compiled with.
    pub fn with_collation(def: &'d MatchDef, collation: C) -> Self {
        FastMatcher {
            def,
            collation,
            state: def.tree.new_state(),
            slots: vec![None; def.num_slots()],
            pending_projections: def.num_projection_slots,
            #[cfg(feature = "simd")]
            backend: crate::simd::Backend::detect(),
        }
    }

    /// Match against a raw JSON document, returning the [`MatchOutcome`]: whether the
    /// document matched, each expression's individual result, and the values captured for
    /// the definition's projected fields.
    ///
    /// Captured values borrow `doc`, and the outcome borrows the matcher, so they stay valid
    /// exactly until the next match — the borrow checker enforces this.
    ///
    /// ```
    /// use jsonsm::collation::DefaultCollation;
    /// use jsonsm::compile::{compile, Projection};
    /// use jsonsm::matcher::FastMatcher;
    /// use jsonsm_ast::{CompareOp, Expr, Field, Literal, PathComponent};
    ///
    /// let expr = Expr::compare(
    ///     CompareOp::GreaterThan,
    ///     Expr::Field(Field::root(vec![PathComponent::Key("age".into())])),
    ///     Expr::Value(Literal::Int(21)),
    /// );
    /// let projection = Projection::new().field(["name", "first"]);
    /// let def = compile(&[expr], &projection, &DefaultCollation).unwrap();
    /// let mut m = FastMatcher::new(&def);
    ///
    /// let out = m.matches(br#"{"name": {"first": "Brett"}, "age": 41}"#)?;
    /// assert!(out.matched());
    /// let first = out.projected(0).unwrap();
    /// assert_eq!(first.as_str().unwrap().to_decoded_bytes().as_ref(), b"Brett");
    /// # Ok::<(), jsonsm::matcher::MatchError>(())
    /// ```
    pub fn matches<'a>(&mut self, doc: &'a [u8]) -> Result<MatchOutcome<'_, 'a>, MatchError> {
        self.state.reset();
        self.slots.iter_mut().for_each(|s| *s = None);
        self.pending_projections = self.def.num_projection_slots;
        if !doc.is_empty() {
            self.scan(doc)?;
        }
        // The root's seal list spans every bucket in the tree, so this gives each one the value
        // its own absence implies — `False` for an `Exists`, `Unknown` for a comparison —
        // before the tree-wide backstop below settles whatever is left. Containers seal as they
        // close, so this normally finds little to do; it is here for the paths that never reach
        // a container's close, such as a document that is a bare scalar.
        self.seal_absent(self.def.root);
        self.state.resolve();
        Ok(MatchOutcome {
            def: self.def,
            state: &self.state,
            slots: &self.slots,
            doc,
        })
    }

    /// Choose the scan backend and run the document through a fully monomorphised matcher.
    ///
    /// This is the *only* backend dispatch: one predicted branch per document, against a
    /// value fixed when the matcher was constructed. Everything below it — the state
    /// machine, the bulk scan kernels, the structural skipping — is specialised for the
    /// chosen `Scan` and inlines freely. For a backend whose instructions are not in the
    /// target's baseline, [`Scan::enter`] additionally wraps the whole scan in the required
    /// `#[target_feature]` context, which is what lets its kernels inline at all.
    fn scan<'a>(&mut self, doc: &'a [u8]) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        #[cfg(feature = "simd")]
        {
            use crate::simd::Backend;
            match self.backend {
                #[cfg(target_arch = "x86_64")]
                Backend::Sse2 => return self.run::<crate::simd::Sse2Scan>(doc),
                // No `enter` here: `HybridScan` deliberately runs the state machine in the
                // baseline target and opens its AVX2 context inside `skip_container` alone.
                #[cfg(target_arch = "x86_64")]
                Backend::Hybrid => return self.run::<crate::simd::HybridScan>(doc),
                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => {
                    return <crate::simd::Avx2Scan as Scan>::enter(|| {
                        self.run::<crate::simd::Avx2Scan>(doc)
                    })
                }
                Backend::Scalar => {}
            }
        }
        self.run::<crate::tokenizer::ScalarScan>(doc)
    }

    fn run<'a, S: Scan>(&mut self, doc: &'a [u8]) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        let mut tokens = GenericTokenizer::<S>::new(doc);
        let tok = tokens.step()?;
        if tok.token_type != TokenType::End {
            self.match_exec(&mut tokens, tok, self.def.root, 0)?;
        }
        Ok(())
    }

    /// Override the scan backend chosen by CPU detection.
    ///
    /// Exists so tests and benchmarks can drive *every* backend this CPU supports rather
    /// than only the preferred one — the matcher is monomorphised per backend, so an
    /// unexercised backend is genuinely untested code, not a shared path with a different
    /// constant. Panics if this CPU cannot run `backend`.
    #[cfg(feature = "simd")]
    pub fn force_backend(&mut self, backend: crate::simd::Backend) {
        assert!(
            crate::simd::Backend::available().contains(&backend),
            "{backend:?} is not supported by this CPU"
        );
        self.backend = backend;
    }

    /// The scan backend this matcher runs.
    #[cfg(feature = "simd")]
    pub fn backend(&self) -> crate::simd::Backend {
        self.backend
    }

    /// Whether the scan can stop: the logic tree's verdict is settled *and* every projected
    /// field has been captured. Asked after every operation, so both halves are a plain load.
    #[inline]
    fn done(&self) -> bool {
        self.state.root_settled() && self.pending_projections == 0
    }

    /// Process the value denoted by `token` against exec node `exec`. `depth` is how many
    /// containers enclose this value.
    /// Everything a *scalar* value needs: run the ops attached to it and record its byte
    /// range. No depth check, no children, no after-node, no bracket balancing.
    ///
    /// Split out of [`Self::match_exec`] so a caller that already holds the token can handle a
    /// scalar **without entering the general walker at all**. `match_exec` is one function
    /// covering the object, array, loop and after-node cases, and its frame is the union of
    /// them; a matched scalar field would pay that entry and exit — the callee-saved pushes
    /// and pops and the call and return, not the frame's size — to run the four lines below.
    /// Most fields an expression names hold scalars, so this is the common case.
    ///
    /// Deliberately not a special case for arrays or for any one shape. The property it
    /// exploits — a scalar contains nothing, so none of the walker's machinery can apply —
    /// holds at the document root, inside an object and inside a loop body alike.
    /// `match_exec` still routes literals here, so there is exactly one implementation of what
    /// a scalar means.
    #[inline(always)]
    fn match_literal<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        token: Token<'a>,
        exec: ExecId,
    ) where
        'd: 'a,
    {
        let start = tokens.position() - token.value.len();
        let val = FastVal::from_scalar_token(token).expect("literal token");
        self.run_ops(tokens, exec, &val);
        self.store_slot(exec, start, tokens.position());
    }

    fn match_exec<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        token: Token<'a>,
        exec: ExecId,
        depth: usize,
    ) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        let def = self.def;
        // The value starts where this token started (works for literals and for the '{'/'['
        // opener, whose value is one byte).
        let start = tokens.position() - token.value.len();

        if token.token_type.is_literal() {
            self.match_literal(tokens, token, exec);
            return Ok(());
        }

        if depth >= MAX_DEPTH {
            return Err(MatchError::TooDeep);
        }

        // Read once. The arena holds a large struct, so `arena[exec]` is a bounds check and a
        // multiply, and this function reached for it four separate times per container — for
        // the children, for the ops, for the slot and for the after-node.
        let node: &'d ExecNode = &def.arena[exec];

        match token.token_type {
            TokenType::ObjectStart => {
                if node.elems.is_empty() {
                    leave_value(tokens, depth)?;
                } else {
                    self.match_object(tokens, exec, depth)?;
                }
                let end = tokens.position();
                self.store_range(node.store, node.store_projected, start, end);
                if self.done() {
                    return Ok(());
                }
                if !node.ops.is_empty() {
                    let val = FastVal::Object(&tokens.input()[start..end]);
                    self.run_op_list(tokens, &node.ops, &val);
                }
                if !self.done() {
                    if let Some(after) = node.after.as_ref() {
                        self.run_after_node(tokens, after, depth)?;
                    }
                }
                self.seal_absent(exec);
                Ok(())
            }
            TokenType::ArrayStart => {
                // An array may be visited by indexed element references (`a[0]`) and by any
                // number of loops. Each is a separate pass over the array, so rewind between
                // them; every pass consumes through the closing `]`.
                let indexed = !node.indexed.is_empty();
                let n_loops = node.loops.len();
                if !indexed && n_loops == 0 {
                    leave_value(tokens, depth)?;
                } else {
                    let save = tokens.position();
                    let mut pass = 0;
                    if indexed {
                        self.match_array(tokens, exec, depth)?;
                        if self.done() {
                            return Ok(());
                        }
                        pass += 1;
                    }
                    for i in 0..n_loops {
                        if pass > 0 {
                            tokens.seek(save);
                        }
                        pass += 1;
                        // The loop node is borrowed from `def` (lifetime 'd), independent
                        // of `self`, so the &mut self call below is fine.
                        let lp: &LoopNode = &node.loops[i];
                        self.match_loop(
                            tokens,
                            lp.bucket,
                            lp.mode,
                            lp.node,
                            &lp.clear_slots,
                            depth,
                        )?;
                        if self.done() {
                            return Ok(());
                        }
                    }
                }
                let end = tokens.position();
                self.store_range(node.store, node.store_projected, start, end);
                if self.done() {
                    return Ok(());
                }
                if !node.ops.is_empty() {
                    let val = FastVal::Array(&tokens.input()[start..end]);
                    self.run_op_list(tokens, &node.ops, &val);
                }
                if !self.done() {
                    if let Some(after) = node.after.as_ref() {
                        self.run_after_node(tokens, after, depth)?;
                    }
                }
                self.seal_absent(exec);
                Ok(())
            }
            _ => Err(MatchError::Structure(
                "unexpected token where a value was expected",
            )),
        }
    }

    /// A container has closed: every field beneath `exec` that the document did not contain is
    /// now known to be absent, so seal its buckets to `Unknown`.
    ///
    /// This is what makes absence actionable mid-scan. A bucket left unset is ambiguous while the
    /// scan is running — the field might simply not have been reached yet — and the only place
    /// that ambiguity resolves is the close of the enclosing container, because that is the moment
    /// "not seen" becomes "not there".
    ///
    /// Whether a seal then *ends* the scan depends on what sits above the bucket, and the useful
    /// cases are narrower than they first look. A bare comparison, a `NOT`, or an `EXISTS` is
    /// settled by the seal alone and the root falls out immediately. Under `AND`/`OR` it is not:
    /// `Unknown AND False` is `False` while `Unknown AND True` is `Unknown`, so the sibling is
    /// still required, and such a node finishes early only when its other operand was already
    /// resolved — which for a document-ordered scan means it appeared earlier. Either way the
    /// tree resolves as soon as the logic permits instead of waiting for the document to end,
    /// which is the point.
    ///
    /// Only still-unset buckets are touched, so this never disturbs a field that was present:
    /// its ops have already run, and a nested container settled its own subtree when *it*
    /// closed. Must run after [`Self::run_after`] — deferred ops and loops write buckets in
    /// this same subtree, and sealing first would call them absent before they ran.
    ///
    /// Called only for containers, not for scalars, though a scalar where the expression
    /// expected an object (`a.x == 1` meeting `"a": 5`) does also make `a.x` unreachable. That
    /// case is left to the enclosing container's seal, which is correct but one step later,
    /// because paying for a seal on every scalar in the document is not worth catching it
    /// sooner.
    ///
    /// The decline is inline and the sweep is not: every container that closes asks this, and
    /// inside a loop body — an array of objects, once per element — the answer is always no.
    #[inline(always)]
    fn seal_absent(&mut self, exec: ExecId) {
        // Inside a loop body, `match_loop` seals the body subtree after every element, which
        // subsumes this. Skipping here keeps an array of objects from re-walking the same bucket
        // list once per element for no added information.
        if self.state.in_loop_body() {
            return;
        }
        self.seal_absent_buckets(exec);
    }

    /// [`Self::seal_absent`] outside a loop body; see the note there.
    #[inline(never)]
    fn seal_absent_buckets(&mut self, exec: ExecId) {
        // `def` is borrowed from `'d`, independent of `self`, so the &mut self calls below are
        // fine while iterating it.
        let def = self.def;
        for &(bucket, absent) in &def.arena[exec].seal_buckets {
            if !self.state.is_resolved(bucket) {
                self.state.mark_tri(bucket, absent);
            }
        }
    }

    /// Record `exec`'s scanned byte range into its slot, if it has one.
    #[inline(always)]
    fn store_slot(&mut self, exec: ExecId, start: usize, end: usize) {
        let node = &self.def.arena[exec];
        self.store_range(node.store, node.store_projected, start, end);
    }

    /// [`Self::store_slot`] for a caller that already holds the node's slot fields — a loop
    /// reads them once instead of indexing the arena for every element.
    #[inline(always)]
    fn store_range(
        &mut self,
        store: Option<SlotId>,
        store_projected: bool,
        start: usize,
        end: usize,
    ) {
        if let Some(slot) = store {
            // A projection slot filled for the first time brings the scan closer to being
            // able to stop (duplicate keys refill the slot without double-counting).
            if store_projected && self.slots[slot].is_none() {
                self.pending_projections -= 1;
            }
            self.slots[slot] = Some((start, end - start));
        }
    }

    /// Run a node's deferred after-node ops and after-loops (their slots are now filled).
    ///
    /// Most nodes have no after-node at all — deferral is what a *cross-field* comparison
    /// needs, and most comparisons name one field — so the common answer is "nothing to do",
    /// and every container that closes asks. Callers hold the node, so they ask by matching on
    /// `after` and only come here when there is something in it.
    #[inline(never)]
    fn run_after_node<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        after: &'d AfterNode,
        depth: usize,
    ) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        for op in &after.ops {
            if self.state.is_resolved(op.bucket) {
                continue;
            }
            let result = self.eval_op(tokens, &op.kind, None);
            self.state.mark_tri(op.bucket, result);
            if self.done() {
                return Ok(());
            }
        }

        // Deferred loops: seek back to the stored array and iterate now that outer fields
        // referenced by the body are available. Copy the descriptors out first so the
        // borrow of `def` does not overlap the `&mut self` loop calls.
        let loops: Vec<(BucketId, LoopType, ExecId, usize, &[SlotId])> = after
            .loops
            .iter()
            .map(|l| {
                (
                    l.bucket,
                    l.mode,
                    l.node,
                    l.array_slot,
                    l.clear_slots.as_slice(),
                )
            })
            .collect();
        for (bucket, mode, node, array_slot, clear) in loops {
            if self.state.is_resolved(bucket) {
                continue;
            }
            // If the array field was absent or not an array, the loop does not apply; its
            // node stays unresolved and `resolve` defaults it to false.
            if let Some((start, _)) = self.slots[array_slot] {
                let save = tokens.position();
                tokens.seek(start);
                if tokens.step()?.token_type == TokenType::ArrayStart {
                    self.match_loop(tokens, bucket, mode, node, clear, depth)?;
                }
                tokens.seek(save);
            }
            if self.done() {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Scan an object, recursing into fields present in `exec.elems` and skipping the rest.
    fn match_object<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        exec: ExecId,
        depth: usize,
    ) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        // One reach into the arena for the whole object: the key map is consulted per field,
        // to look the key up and again to take the matched child's ordinal.
        let elems: &'d KeyMap = &self.def.arena[exec].elems;
        // Once every key this node names has been seen, nothing left in the object can change
        // any answer, so the remainder is skipped in bulk rather than walked key by key. See
        // the note on `seen` below for what this costs on duplicate keys.
        let wanted = elems.len();
        let mut seen: u64 = 0;
        // Set once every named key has been supplied; from then on the object's remaining
        // fields are unreachable to the expression. `spare` counts how many have been walked
        // since, because the bulk skip is only worth entering once the tail is known to be
        // long enough to pay for it — see the note at the exit below.
        let mut complete = false;
        let mut spare = 0u32;
        let mut first = true;
        loop {
            if !first {
                let more = match take_delim(tokens, b'}') {
                    Some(more) => more,
                    None => match tokens.step()?.token_type {
                        TokenType::ObjectEnd => false,
                        TokenType::ListDelim => true,
                        _ => return Err(MatchError::Structure("expected ',' or '}' in object")),
                    },
                };
                if !more {
                    return Ok(());
                }
            }
            first = false;

            let child = match take_key(tokens, elems) {
                KeyStep::End => return Ok(()),
                KeyStep::Resolved(child) => child,
                KeyStep::Slow => {
                    let key_tok = tokens.step()?;
                    let key_content = match key_tok.token_type {
                        TokenType::ObjectEnd => return Ok(()),
                        TokenType::String | TokenType::EscString => strip_quotes(key_tok.value),
                        _ => return Err(MatchError::Structure("expected an object key")),
                    };
                    // Decode the key just enough to look it up (borrow when no escapes).
                    let owned_key: Vec<u8>;
                    let decoded: &[u8] = if key_tok.token_type == TokenType::EscString {
                        owned_key = FastStr::Escaped(key_content)
                            .to_decoded_bytes()
                            .into_owned();
                        &owned_key
                    } else {
                        key_content
                    };
                    // Compared as bytes: the trie's keys came from UTF-8 `String`s, so a
                    // byte-equal document key is UTF-8 by construction and validating it
                    // would be redundant work on every field, matching or not.
                    elems.get(decoded)
                }
            };
            if !take_structural(tokens, b':')
                && tokens.step()?.token_type != TokenType::ObjectKeyDelim
            {
                return Err(MatchError::Structure("expected ':' after object key"));
            }

            // The lookup comes *before* the value is read, so a field the expression does
            // not name never costs a token: most fields of most documents are that case.
            match child {
                Some(child) => {
                    // A named field holding a plain string is what most compared fields are,
                    // and reading it here is what `match_literal` would do anyway — run this
                    // child's ops against it, record its range — without `step` running the
                    // string state machine to build a 24-byte token that is read once and
                    // dropped. The same trade [`take_key`] makes for the key, on the value.
                    // Everything else declines and is tokenized below exactly as before.
                    match take_str_value(tokens) {
                        Some(bytes) => {
                            let end = tokens.position();
                            let val = FastVal::Str(FastStr::Unescaped(bytes));
                            // One reach into the arena for the child, not one for its ops and
                            // another for its slot.
                            let child_node: &'d ExecNode = &self.def.arena[child];
                            self.run_op_list(tokens, &child_node.ops, &val);
                            // The field spans its two quotes as well as its content.
                            self.store_range(
                                child_node.store,
                                child_node.store_projected,
                                end - bytes.len() - 2,
                                end,
                            );
                        }
                        None => {
                            let val_tok = tokens.step()?;
                            if val_tok.token_type.is_literal() {
                                self.match_literal(tokens, val_tok, child);
                            } else {
                                self.match_exec(tokens, val_tok, child, depth + 1)?;
                            }
                        }
                    }
                    if self.done() {
                        return Ok(());
                    }
                    // Field-completeness exit. A node names a fixed set of keys; once the
                    // document has supplied every one of them, the rest of this object is
                    // unreachable to the expression and `leave_value` crosses it in bulk
                    // instead of matching and skipping field by field. The ops on this node,
                    // its after-node and its slot all still run, in `match_exec`, over a byte
                    // range that stays correct because the skip lands past the closing brace.
                    //
                    // This is what makes a comparison between two fields affordable. Such a
                    // comparison defers to the after-node, so the logic tree cannot resolve
                    // mid-scan and `done()` never fires however early the fields appear; a
                    // wide record would otherwise be walked to its end every time.
                    //
                    // It is not free on narrow objects: `leave_value` is a windowed scan whose
                    // setup does not pay for a few trailing bytes. The bookkeeping here is
                    // negligible; the bulk skip is what has to be earned, hence the threshold
                    // at the exit below.
                    //
                    // The `seen` bitmask counts *distinct* keys, so a repeated key cannot
                    // complete the set on its own. It does mean a later duplicate is never
                    // read: this engine takes the **first** occurrence where gojsonsm and
                    // `serde_json` take the last.
                    //
                    // An object carrying one field twice is not valid input here — RFC 8259
                    // says names should be unique and leaves the behaviour unpredictable
                    // otherwise — so this is the same licence `leave_value` and `take_key`
                    // already take: the engine may assume valid JSON, and owes an invalid
                    // document nothing beyond terminating. Pinned by
                    // `duplicate_keys_take_the_first_occurrence`.
                    if !complete && wanted <= 64 {
                        if let Some(i) = elems.ordinal(child) {
                            seen |= 1u64 << i;
                            complete = seen.count_ones() as usize == wanted;
                        }
                    }
                }
                None => {
                    skip_unnamed_value(tokens, depth)?;
                    if complete {
                        // Every key this node names has been seen, so nothing further can
                        // matter. `leave_value` crosses the rest in one windowed scan, which
                        // is a large win on a wide object but whose setup does not pay for a
                        // couple of trailing bytes. Entering it unconditionally costs far more
                        // in cycles than in work on a narrow one, because the scan is a serial
                        // dependency where walking two short fields is not.
                        //
                        // So walk a couple first. If the object ends there, the exit cost
                        // nothing; if it does not, the tail is long enough that one bulk scan
                        // beats matching and skipping the rest field by field. The threshold
                        // is a cost ratio, not a guess about document shape — which is what
                        // gating on depth or on being inside a loop would have been.
                        spare += 1;
                        if spare >= 2 {
                            leave_value(tokens, depth)?;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    /// Scan an array, recursing into the elements referenced by index (`a[0]`) and skipping
    /// the rest. The opening `[` has already been consumed.
    fn match_array<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        exec: ExecId,
        depth: usize,
    ) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        let def = self.def;
        // `indexed` is sorted, so once the cursor passes the last wanted index the remainder
        // of the array can be skipped wholesale.
        let last_wanted = def.arena[exec]
            .indexed
            .last()
            .map(|&(i, _)| i)
            .expect("only called with indexed children");

        let mut index = 0usize;
        let mut first = true;
        loop {
            if !first {
                let more = match take_delim(tokens, b']') {
                    Some(more) => more,
                    None => match tokens.step()?.token_type {
                        TokenType::ArrayEnd => false,
                        TokenType::ListDelim => true,
                        _ => return Err(MatchError::Structure("expected ',' or ']' in array")),
                    },
                };
                if !more {
                    return Ok(());
                }
                index += 1;
            }
            first = false;

            let elem = tokens.step()?;
            if elem.token_type == TokenType::ArrayEnd {
                return Ok(());
            }

            match def.arena[exec]
                .indexed
                .iter()
                .find(|&&(i, _)| i == index)
                .map(|&(_, child)| child)
            {
                Some(child) => {
                    self.match_exec(tokens, elem, child, depth + 1)?;
                    if self.done() {
                        return Ok(());
                    }
                }
                None => skip_value(tokens, elem, depth)?,
            }
            if index >= last_wanted {
                // Nothing further in this array is referenced.
                leave_value(tokens, depth)?;
                return Ok(());
            }
        }
    }

    /// Run one loop over the array whose opening `[` has just been consumed. Shared by
    /// inline loops and deferred after-loops. `body` is the loop's body bucket, `node` its
    /// per-element exec node, and `clear` the slots owned by the body (reset per element so
    /// one element never reads another's value).
    fn match_loop<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        body: BucketId,
        mode: LoopType,
        node: ExecId,
        clear: &[SlotId],
        depth: usize,
    ) -> Result<(), MatchError>
    where
        'd: 'a,
    {
        if self.state.is_resolved(body) {
            leave_value(tokens, depth)?;
            return Ok(());
        }

        // Borrowed from the `MatchDef` at `'d`, so it survives every `&mut self` call below.
        // The arena is a `Vec<ExecNode>` of a large struct: reaching one is a bounds check and
        // a multiply, and the per-element path would otherwise pay it for the ops and again
        // for the slot. Which node the loop is over cannot change between elements.
        let body_node: &'d ExecNode = &self.def.arena[node];

        // A quantifier over elements: `ANY` is an OR, `EVERY` an AND. `decided` is set only when
        // an element settles the loop outright by supplying that connective's absorbing value —
        // a `True` for `ANY`, a `False` for `EVERY` — which is also when the scan can stop.
        // `Unknown` is not absorbing, so it cannot end the loop; it is remembered instead,
        // because it denies the loop the *other* verdict: `EVERY` cannot conclude `True` over an
        // element it could not evaluate, nor `ANY` conclude `False`.
        let mut decided: Option<Tri> = None;
        let mut saw_unknown = false;
        let mut saw_true = false;
        // `AnyEvery` is an `EVERY` that additionally requires the array to be non-empty, so it
        // shares `EVERY`'s absorbing value. A property of the loop, so computed once.
        let absorbing = match mode {
            LoopType::Any => Tri::True,
            LoopType::Every | LoopType::AnyEvery => Tri::False,
        };
        let prev_stall = self.state.set_stall(body);

        let mut first = true;
        loop {
            if !first {
                let more = match take_delim(tokens, b']') {
                    Some(more) => more,
                    None => match tokens.step()?.token_type {
                        TokenType::ArrayEnd => false,
                        TokenType::ListDelim => true,
                        _ => return Err(MatchError::Structure("expected ',' or ']' in array")),
                    },
                };
                if !more {
                    break;
                }
            }
            first = false;

            // A string element is a scalar, so [`Self::match_exec`] would route it straight
            // to [`Self::match_literal`]: run the node's ops against it and record its byte
            // range. Reading it here does exactly that, without `step` building a 24-byte
            // token and without entering the walker whose frame is sized for the container
            // cases a scalar cannot take. Nothing about the *body* is assumed — the ops, the
            // slot and the seal below are the same ones the tokenized arm runs.
            //
            // The reset and the slot clear are written out in each arm rather than hoisted
            // above the `match`. Sharing them would put the fused read's branch on the
            // tokenized arm's path, which an element that can never settle here would pay
            // once per element for nothing.
            match take_str_value(tokens) {
                Some(bytes) => {
                    let end = tokens.position();
                    self.state.reset_node(body);
                    for &slot in clear {
                        self.slots[slot] = None;
                    }
                    let val = FastVal::Str(FastStr::Unescaped(bytes));
                    self.run_op_list(tokens, &body_node.ops, &val);
                    // The element spans its two quotes as well as its content, which is what
                    // `match_literal` derives from the token's length.
                    self.store_range(
                        body_node.store,
                        body_node.store_projected,
                        end - bytes.len() - 2,
                        end,
                    );
                }
                None => {
                    // Not a plain string, so it is a container, the end of the array, or
                    // something only the tokenizer can name. The first two are decided by the
                    // byte itself; `match_exec` wants the token `step` would have returned, so
                    // it is rebuilt from the cursor the probe just moved.
                    let elem = match take_structural_head(tokens) {
                        Some(TokenType::ArrayEnd) => break,
                        Some(token_type) => {
                            let at = tokens.position() - 1;
                            Token {
                                token_type,
                                value: &tokens.input()[at..at + 1],
                            }
                        }
                        None => {
                            let elem = tokens.step()?;
                            if elem.token_type == TokenType::ArrayEnd {
                                break;
                            }
                            elem
                        }
                    };
                    self.state.reset_node(body);
                    // This element starts with none of the body's stored fields known.
                    for &slot in clear {
                        self.slots[slot] = None;
                    }
                    // A scalar the fused read declined — a number, a boolean, `null`, an
                    // escaped string — contains nothing, so none of the walker's machinery can
                    // apply and `match_exec` would route it straight back out to
                    // `match_literal`. Going there directly skips the frame `match_literal`
                    // exists to avoid, which an array of numbers would otherwise pay once per
                    // element — only strings and containers have a probe that names them.
                    if elem.token_type.is_literal() {
                        self.match_literal(tokens, elem, node);
                    } else {
                        self.match_exec(tokens, elem, node, depth + 1)?;
                    }
                }
            }
            // Seal this element's body: anything still unset names a field this element did
            // not have, which is unanswerable for this element.
            let matched = self.state.seal_and_value(body);

            match matched {
                m if m == absorbing => {
                    decided = Some(m);
                    leave_value(tokens, depth)?;
                    break;
                }
                Tri::Unknown => saw_unknown = true,
                _ => saw_true = true,
            }
        }

        // No element was absorbing, so the verdict comes from the elements as a whole. An
        // element that could not be evaluated makes the quantifier unanswerable — `False OR
        // Unknown` and `True AND Unknown` are both `Unknown` — and otherwise the empty/all-
        // definite defaults apply: `ANY` over nothing is false, `EVERY` over nothing is
        // vacuously true, and `AnyEvery` needs at least one element to have satisfied it.
        let loop_state = decided.unwrap_or(if saw_unknown {
            Tri::Unknown
        } else {
            match mode {
                LoopType::Any => Tri::from_bool(false),
                LoopType::Every => Tri::from_bool(true),
                LoopType::AnyEvery => Tri::from_bool(saw_true),
            }
        });

        self.state.reset_node(body);
        self.state.set_stall(prev_stall);
        self.state.mark_tri(body, loop_state);
        Ok(())
    }

    /// Evaluate every op on `exec` against the active value, recording results.
    ///
    /// `#[inline(always)]` because the active value reaches this by *reference*: outlined, the
    /// 24-byte `FastVal` the caller just built has to be written somewhere the callee can
    /// address, and it ends up copied piecewise through misaligned stack slots on every
    /// matched field. Inlined, it stays where it was built.
    #[inline(always)]
    fn run_ops<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        exec: ExecId,
        active: &FastVal<'a>,
    ) where
        'd: 'a,
    {
        self.run_op_list(tokens, &self.def.arena[exec].ops, active);
    }

    /// [`Self::run_ops`] against a list already in hand.
    ///
    /// The arena is a `Vec<ExecNode>` and an `ExecNode` is a large struct, so `arena[exec]` is
    /// a bounds check and a multiply. Taking the slice pays that once rather than per op, and
    /// lets a caller that already holds the node — in a loop, one that holds it for every
    /// element — not pay it at all. The list is borrowed from the `MatchDef` at `'d`,
    /// independent of `self`, so it stays valid across the `&mut self` calls below.
    #[inline(always)]
    fn run_op_list<'a, S: Scan>(
        &mut self,
        tokens: &mut GenericTokenizer<'a, S>,
        ops: &'d [OpNode],
        active: &FastVal<'a>,
    ) where
        'd: 'a,
    {
        for op in ops {
            if self.state.is_resolved(op.bucket) {
                continue;
            }
            let result = self.eval_op(tokens, &op.kind, Some(active));
            self.state.mark_tri(op.bucket, result);
            if self.done() {
                return;
            }
        }
    }

    /// Evaluate one op. `active` is the value being scanned at the op's node (present for
    /// regular ops; `None` for deferred after-node ops, which reference only slots/consts).
    ///
    /// Returns [`Tri::Unknown`] when an operand is missing. A comparison against an absent
    /// value is not an ordering question with an unfortunate answer — it has no answer, and
    /// saying so is what stops an enclosing `NOT` from reading it as a match.
    ///
    /// Split the way [`Self::resolve_ref`] and [`Self::match_literal`] are, and for the same
    /// reason: one arm is what the hot path does, the rest is what makes the function too big
    /// to inline. A comparison between two operands that already exist in memory is the whole
    /// of the arm below — no operand is *produced*, so nothing is returned by value, and
    /// `Collation::compare` is handed the two addresses directly. Everything else needs a
    /// `FastVal` built (a slot re-parsed, a function applied) or is a rarer operator, and a
    /// call is nothing beside that.
    ///
    /// Note what the cost of *not* splitting it is: not the operands, which
    /// [`Self::operand_ref`] already borrows in place, but reaching an outlined `eval_op` at
    /// all, through a frame sized for the arms below.
    #[inline(always)]
    fn eval_op<'a, S: Scan>(
        &self,
        tokens: &mut GenericTokenizer<'a, S>,
        kind: &'a OpKind,
        active: Option<&FastVal<'a>>,
    ) -> Tri
    where
        'd: 'a,
    {
        if let OpKind::Compare { op, lhs, rhs } = kind {
            if let (Some(l), Some(r)) = (
                self.operand_ref(lhs, active),
                self.operand_ref(rhs, active),
            ) {
                if matches!(l, FastVal::Missing) || matches!(r, FastVal::Missing) {
                    return Tri::Unknown;
                }
                return Tri::from_bool(apply_cmp(*op, self.collation.compare(l, r).ordering));
            }
        }
        self.eval_op_slow(tokens, kind, active)
    }

    /// Every op that has to *build* an operand, and every operator but `Compare`. Outlined so
    /// [`Self::eval_op`] can inline; see the note there.
    #[inline(never)]
    fn eval_op_slow<'a, S: Scan>(
        &self,
        tokens: &mut GenericTokenizer<'a, S>,
        kind: &'a OpKind,
        active: Option<&FastVal<'a>>,
    ) -> Tri
    where
        'd: 'a,
    {
        match kind {
            OpKind::Always(b) => Tri::from_bool(*b),
            // `Exists` asks about presence, so it is always answerable and stays definite. For a
            // current-scope field `of` is the active value and reaching the op *is* the answer;
            // absence is then handled by the leaf never running and being sealed to `False`. For a
            // field from an enclosing scope `of` is a slot, and an unfilled slot resolves to
            // `Missing` — which answers the same question by the same test.
            OpKind::Exists { of } => {
                let v = self.resolve_ref(tokens, of, active);
                Tri::from_bool(!matches!(v, FastVal::Missing))
            }
            OpKind::Matches { matcher, of } => {
                let v = self.resolve_ref(tokens, of, active);
                // Unlike `Exists`, a pattern match against a value that is not there has no
                // answer, so it is `Unknown` like any other comparison.
                if matches!(v, FastVal::Missing) {
                    Tri::Unknown
                } else {
                    Tri::from_bool(matcher.matches(&v))
                }
            }
            OpKind::Compare { op, lhs, rhs } => {
                // Reached only when [`Self::eval_op`]'s arm declined, so at least one operand
                // is a slot or a function result and has to be built. Both are produced by
                // value, which is what `operand_ref` exists to avoid where it can.
                let l = self.resolve_ref(tokens, lhs, active);
                let r = self.resolve_ref(tokens, rhs, active);
                if matches!(l, FastVal::Missing) || matches!(r, FastVal::Missing) {
                    return Tri::Unknown;
                }
                let ord = self.collation.compare(&l, &r).ordering;
                Tri::from_bool(apply_cmp(*op, ord))
            }
        }
    }

    /// Borrow an operand that already exists, instead of producing one.
    ///
    /// `Active` is the value currently being scanned and `Const` was built by the compiler
    /// and lives in the [`MatchDef`], so both are already in memory and a comparison can take
    /// their addresses. `Slot` and `Func` have to be constructed, and decline here so the
    /// caller falls back to [`Self::resolve_ref`].
    ///
    /// The lifetimes work out because `FastVal` is covariant: a `FastVal<'static>` is usable
    /// wherever a `FastVal<'a>` is wanted, which is exactly what storing constants with no
    /// borrows buys.
    #[inline(always)]
    fn operand_ref<'s, 'a>(
        &'s self,
        r: &'s DataRef,
        active: Option<&'s FastVal<'a>>,
    ) -> Option<&'s FastVal<'a>> {
        match r {
            DataRef::Active => active,
            DataRef::Const(v) => Some(v),
            DataRef::Slot(_) | DataRef::Func(_) => None,
        }
    }

    /// Produce one operand of an op.
    ///
    /// `#[inline(always)]` here is load-bearing and only possible because [`DataRef::Func`]
    /// lives in [`Self::resolve_func`] rather than in this body: a self-recursive function
    /// cannot be inlined, so with the `Func` arm here the attribute would be silently a no-op.
    /// Outlined, the operand goes back to the caller through memory — a 24-byte `FastVal`
    /// returned by value, which the ABI hands over via a hidden out-pointer into a caller
    /// stack slot and the caller immediately reloads to compare. Two of those per comparison,
    /// on the chain that carries one array element into the next.
    #[inline(always)]
    fn resolve_ref<'a, S: Scan>(
        &self,
        tokens: &mut GenericTokenizer<'a, S>,
        r: &'a DataRef,
        active: Option<&FastVal<'a>>,
    ) -> FastVal<'a>
    where
        'd: 'a,
    {
        match r {
            DataRef::Active => active
                .expect("active operand without an active value")
                .clone(),
            DataRef::Const(c) => borrow_const(c),
            DataRef::Slot(slot) => self.literal_from_slot(tokens, *slot),
            DataRef::Func(func) => self.resolve_func(tokens, func, active),
        }
    }

    /// The recursive operand case, kept out of [`Self::resolve_ref`] so that one can inline.
    ///
    /// Outlined deliberately: it recurses, it allocates a `Vec` per call, and a function call
    /// is nothing beside those. Every other `DataRef` is a load.
    #[inline(never)]
    fn resolve_func<'a, S: Scan>(
        &self,
        tokens: &mut GenericTokenizer<'a, S>,
        func: &'a crate::compile::FuncRef,
        active: Option<&FastVal<'a>>,
    ) -> FastVal<'a>
    where
        'd: 'a,
    {
        let mut args = Vec::with_capacity(func.params.len());
        for p in &func.params {
            args.push(self.resolve_ref(tokens, p, active));
        }
        crate::func::apply(&func.name, &args)
    }

    /// Read the value stored in `slot` by seeking back to its recorded byte range and
    /// re-parsing it. Returns [`FastVal::Missing`] if the slot was never filled.
    fn literal_from_slot<'a, S: Scan>(
        &self,
        tokens: &mut GenericTokenizer<'a, S>,
        slot: usize,
    ) -> FastVal<'a> {
        let Some(range) = self.slots[slot] else {
            return FastVal::Missing;
        };
        let save = tokens.position();
        tokens.seek(range.0);
        let val = value_at(tokens, range);
        tokens.seek(save);
        val
    }
}

/// Re-parse the value in `range` from a tokenizer already positioned at its start. Scalars
/// come back in their lazy/borrowed form; containers as their raw document bytes.
fn value_at<'a, S: Scan>(
    tokens: &mut GenericTokenizer<'a, S>,
    (start, size): SlotRange,
) -> FastVal<'a> {
    match tokens.step() {
        Ok(tok) if tok.token_type.is_literal() => {
            FastVal::from_scalar_token(tok).unwrap_or(FastVal::Missing)
        }
        Ok(tok) if tok.token_type == TokenType::ObjectStart => {
            FastVal::Object(&tokens.input()[start..start + size])
        }
        Ok(tok) if tok.token_type == TokenType::ArrayStart => {
            FastVal::Array(&tokens.input()[start..start + size])
        }
        _ => FastVal::Missing,
    }
}

/// The result of a [`FastMatcher::matches`] call: whether the document matched, the
/// per-expression results, and the values captured for the compiled
/// [`Projection`](crate::compile::Projection).
///
/// Captured values borrow the document (`'a`) and are read out of the matcher's per-match
/// slots (`'m`), so this handle keeps the matcher borrowed until it is dropped — the values
/// are valid exactly until the next match.
///
/// Bounding it through the *matcher* rather than by giving [`FastMatcher`] a document lifetime
/// is deliberate. A matcher exists to be reused across documents; parameterising it by the
/// document's lifetime would tie one matcher to one document's lifetime and defeat that. This
/// way the borrow checker enforces "valid until the next match" for free, and the matcher's
/// public shape stays free of the document.
#[derive(Debug)]
pub struct MatchOutcome<'m, 'a> {
    def: &'m MatchDef,
    state: &'m LogicTreeState<'m>,
    slots: &'m [Option<SlotRange>],
    doc: &'a [u8],
}

impl<'m, 'a> MatchOutcome<'m, 'a> {
    /// Whether the document matched: the OR of all the compiled expressions (`false` if none
    /// were compiled).
    pub fn matched(&self) -> bool {
        self.state.is_true(self.def.root_bucket)
    }

    /// Whether expression `i` matched. Each expression compiled by
    /// [`compile`](crate::compile::compile) is tracked independently; with a single
    /// expression, `expression_matched(0)` equals [`Self::matched`]. Panics if `i` is out of
    /// range.
    pub fn expression_matched(&self, i: usize) -> bool {
        self.state.is_true(self.def.expr_buckets[i])
    }

    /// Number of projected fields.
    pub fn num_projections(&self) -> usize {
        self.def.projections.len()
    }

    /// The path of projected field `i`. Panics if `i` is out of range.
    pub fn projected_path(&self, i: usize) -> &'m [PathComponent] {
        self.def.projection_path(i)
    }

    /// The value captured for projected field `i`, or `None` if that field was absent from
    /// the document. Panics if `i` is out of range (i.e. it is not a projected field).
    ///
    /// Strings and containers borrow the document bytes; an escaped string is decoded only
    /// if and when the caller asks for its decoded form.
    pub fn projected(&self, i: usize) -> Option<FastVal<'a>> {
        let (start, size) = self.slots[self.def.projections[i].slot]?;
        let mut tokens = crate::tokenizer::JsonTokenizer::new(self.doc);
        tokens.seek(start);
        match value_at(&mut tokens, (start, size)) {
            FastVal::Missing => None,
            val => Some(val),
        }
    }

    /// The value captured for a projected path, or `None` if the path was not projected or
    /// the field was absent.
    pub fn projected_by_path(&self, path: &[PathComponent]) -> Option<FastVal<'a>> {
        self.projected(self.def.projection_index(path)?)
    }

    /// Whether projected field `i` was present in the document. Panics if `i` is out of
    /// range.
    pub fn is_projected_present(&self, i: usize) -> bool {
        self.slots[self.def.projections[i].slot].is_some()
    }

    /// Iterate over every projected field in request order, as `(path, value)` pairs; the
    /// value is `None` for a field absent from the document.
    pub fn projections(
        &self,
    ) -> impl Iterator<Item = (&'m [PathComponent], Option<FastVal<'a>>)> + '_ {
        (0..self.num_projections()).map(move |i| (self.projected_path(i), self.projected(i)))
    }
}

#[inline]
fn apply_cmp(op: CmpOp, ord: Ordering) -> bool {
    match op {
        CmpOp::Eq => ord == Ordering::Equal,
        CmpOp::Lt => ord == Ordering::Less,
        CmpOp::Le => ord != Ordering::Greater,
        CmpOp::Gt => ord == Ordering::Greater,
        CmpOp::Ge => ord != Ordering::Less,
    }
}

#[inline]
fn strip_quotes(bytes: &[u8]) -> &[u8] {
    if bytes.len() >= 2 {
        &bytes[1..bytes.len() - 1]
    } else {
        bytes
    }
}

/// Consume the structural byte `want` if it is literally the next byte of the document.
///
/// Object and array syntax is largely single structural bytes — `:` between a key and its
/// value, `,` between members — and recognising one through the tokenizer costs a call that
/// runs the state machine and returns a 24-byte `Result<Token, _>`, to identify a byte the
/// grammar has already told the caller to expect. Only whitespace can intervene, so peeking
/// at one byte settles the overwhelmingly common case.
///
/// Anything else — whitespace, a different byte, end of input — returns `false` with the
/// cursor untouched, and the caller falls back to a full [`Tokenizer::step`]. So this never
/// decides whether a document is well-formed; it only declines to be the one that looks.
#[inline(always)]
fn take_structural<S: Scan>(tokens: &mut GenericTokenizer<'_, S>, want: u8) -> bool {
    let pos = tokens.position();
    if tokens.input().get(pos) == Some(&want) {
        tokens.seek(pos + 1);
        true
    } else {
        false
    }
}

/// Consume the byte separating two container members: `,` (returns `Some(true)` — another
/// member follows) or `close` (returns `Some(false)` — the container ends here).
///
/// `None` means neither was the next byte, so the caller falls back to a full
/// [`Tokenizer::step`]; see [`take_structural`] for why peeking is worth it.
#[inline(always)]
fn take_delim<S: Scan>(tokens: &mut GenericTokenizer<'_, S>, close: u8) -> Option<bool> {
    let pos = tokens.position();
    match tokens.input().get(pos) {
        Some(&b',') => {
            tokens.seek(pos + 1);
            Some(true)
        }
        Some(&c) if c == close => {
            tokens.seek(pos + 1);
            Some(false)
        }
        _ => None,
    }
}

/// View a stored constant without cloning it.
///
/// A string constant is stored as [`FastStr::Owned`], the only form that borrows nothing and
/// so the only one a `FastVal<'static>` in a [`MatchDef`] can take. `clone` would allocate a
/// `String` per call; the bytes are already the decoded string, so viewing them as
/// `Unescaped` is the same value without the allocation. Every other constant form is a
/// scalar that clones for free.
///
/// The representation change matters only where it removes an allocation, which is worth
/// stating because the obvious reading is that it removes an indirection. It does not:
/// reading `Owned` is not a pointer chase — `String`'s pointer and length
/// sit inline in the enum just as a slice's do; the difference is one discriminant arm.
#[inline(always)]
fn borrow_const<'a>(c: &'a FastVal<'static>) -> FastVal<'a> {
    match c {
        FastVal::Str(FastStr::Owned(s)) => FastVal::Str(FastStr::Unescaped(s.as_bytes())),
        scalar => scalar.clone(),
    }
}

/// Read a plain unescaped string value straight from the document, **without running the
/// tokenizer's state machine over it**. The cursor must be at the value's first byte.
///
/// Returns the value's *content* bytes — quotes stripped, as [`FastVal::from_scalar_token`]
/// would leave them — and advances the cursor past the closing quote, exactly where
/// [`Tokenizer::step`] would. Declines with the cursor untouched for anything else, which the
/// caller then tokenizes normally.
///
/// This is [`take_key`]'s trade applied to values. `step` is a large outlined function that
/// returns a 24-byte [`Token`] through a stack slot the caller reloads on the following
/// instruction, and over an array of scalars it is the single largest per-element cost.
/// Finding the closing quote is one bulk scan.
///
/// Used for both array elements and the values of object fields the expression names, which
/// are the same question asked in two places: a string is a scalar, so all either caller does
/// with it is run the node's operations and record where it was.
///
/// # Validation
///
/// Unlike the skip paths, this one does **not** widen the engine's divergence from a
/// validating parse: it settles only elements whose closing quote the scan reaches with no
/// escape and no control character in between, which is precisely the tokenizer's `String`
/// token. An escaped string, a control character, an unterminated string, and every non-string
/// value all decline, and the tokenizer then produces exactly the token — or exactly the error
/// — it always did. A compared value is still fully validated.
#[inline(always)]
fn take_str_value<'a, S: Scan>(tokens: &mut GenericTokenizer<'a, S>) -> Option<&'a [u8]> {
    let data = tokens.input();
    let scan = tokens.scan();
    let mut pos = tokens.position();
    let &first = data.get(pos)?;
    if first != b'"' {
        if first > b' ' {
            return None;
        }
        pos = S::enter(|| scan.skip_ws(data, pos));
        if !matches!(data.get(pos), Some(b'"')) {
            return None;
        }
    }
    take_quoted(tokens, pos)
}

/// The logical content of the string whose opening quote is at `pos`, or `None` if it is not
/// one the tokenizer would call a plain `String` token.
///
/// The first `"`, `\`, or control character at or after the opening quote decides it. Landing
/// on a quote means the string held none of the other two, which is what makes the bytes
/// between them the logical string. On success the cursor is past the closing quote, exactly
/// where [`Tokenizer::step`] would leave it; on decline it is untouched.
///
/// # Validation
///
/// Unlike the skip paths, this one does **not** widen the engine's divergence from a
/// validating parse: it settles only strings whose closing quote the scan reaches with no
/// escape and no control character in between, which is precisely the tokenizer's `String`
/// token. An escaped string, a control character and an unterminated string all decline, and
/// the tokenizer then produces exactly the token — or exactly the error — it always did. A
/// compared value is still fully validated.
#[inline(always)]
fn take_quoted<'a, S: Scan>(
    tokens: &mut GenericTokenizer<'a, S>,
    pos: usize,
) -> Option<&'a [u8]> {
    let data = tokens.input();
    let scan = tokens.scan();
    let end = S::enter(|| scan.string_event(data, pos + 1));
    if !matches!(data.get(end), Some(b'"')) {
        return None;
    }
    tokens.seek(end + 1);
    Some(&data[pos + 1..end])
}

/// Recognize a structural byte standing where an array element would — `{`, `[` or the `]`
/// that ends the array — **without running the tokenizer's state machine over it**. Returns
/// the token type [`Tokenizer::step`] would have produced and leaves the cursor exactly where
/// it would have, or `None` with the cursor untouched.
///
/// `step` is a large outlined function returning a 24-byte [`Token`] through a stack slot, and
/// for a container that token says nothing the opening byte did not: `{` is an `ObjectStart`
/// whose value is that byte. The caller rebuilds it from the cursor, which is why this returns
/// only the *type* — a `Token` here would be a 24-byte return, reintroducing the memory
/// round-trip this exists to remove. That round-trip costs more than the call: the token comes
/// back through a stack slot and is copied again on its way to [`FastMatcher::match_exec`].
///
/// Kept separate from [`take_str_value`] rather than folded into one classifier. One function
/// answering "string, container, or end?" has to return the string's bytes *and* a token type,
/// which is 32 bytes returned through memory — and that lands on the string path, which is the
/// one already settling without ever asking the question.
#[inline(always)]
fn take_structural_head<S: Scan>(tokens: &mut GenericTokenizer<'_, S>) -> Option<TokenType> {
    let data = tokens.input();
    let scan = tokens.scan();
    let mut pos = tokens.position();
    let &first = data.get(pos)?;
    // The caller has already established this is not a quote and not whitespace it needed to
    // cross, so in the common case this is one load and one compare.
    let head = if first > b' ' {
        first
    } else {
        pos = S::enter(|| scan.skip_ws(data, pos));
        *data.get(pos)?
    };
    let token_type = match head {
        b'{' => TokenType::ObjectStart,
        b'[' => TokenType::ArrayStart,
        b']' => TokenType::ArrayEnd,
        _ => return None,
    };
    tokens.seek(pos + 1);
    Some(token_type)
}

/// What [`take_key`] managed to settle./// What [`take_key`] managed to settle.
enum KeyStep {
    /// The object ended: a `}` stood where a key would have. Consumed.
    End,
    /// The key was resolved to its child (or to `None`, meaning the expression does not name
    /// it). The cursor sits just past the key's closing quote.
    Resolved(Option<ExecId>),
    /// Declined. The cursor is untouched and the caller must tokenize the key.
    Slow,
}

/// Resolve an object key to its child **without tokenizing it**. The cursor must be at the
/// key's opening quote (its token not yet read).
///
/// Tokenizing a key runs the string state machine over every byte of it, validating escapes
/// and control characters, to build a token that — for the great majority of keys, which the
/// expression does not name — is compared once and dropped. But an exec node has only a
/// handful of children, and each of them knows the exact bytes a document key must have to
/// be it. So the question "which child is this?" can be asked of the raw bytes directly, and
/// asking it that way answers "where does the key end?" for free on a hit: the answer is the
/// length of the key that matched.
///
/// A miss still has to reach the `:`, which needs the key's end. That is the same scan
/// [`skip_unnamed_value`] runs over an unnamed string value — [`Scan::string_event`] to the
/// next `"`, `\` or control byte — and it also reports the one thing that can invalidate the
/// miss: an escape. A key spelled with a `\u` escape names the same field as one spelled
/// literally, so a byte comparison that rejected it is not evidence, and an escaped key is
/// handed back to the tokenizer. Keys without escapes — all of them, in practice — are
/// settled here.
///
/// # Validation
///
/// This extends the divergence [`leave_value`] and [`skip_unnamed_value`] document from
/// skipped values to skipped *keys*: an unnamed key's escapes are no longer checked, so
/// `{"a\q":1}` is accepted when no expression names a field at that position. The engine may
/// assume its input is valid JSON; it owes an invalid document nothing beyond terminating.
/// Keys the expression *does* name are still matched exactly, and a key that is not settled
/// here is still fully tokenized.
#[inline(always)]
fn take_key<S: Scan>(tokens: &mut GenericTokenizer<'_, S>, elems: &KeyMap) -> KeyStep {
    // A map holding a key that is not its own JSON encoding cannot be compared raw at all;
    // see `KeyMap::match_quoted`.
    if !elems.verbatim() {
        return KeyStep::Slow;
    }
    let data = tokens.input();
    let scan = tokens.scan();
    let mut pos = tokens.position();
    if !matches!(data.get(pos), Some(b'"' | b'}')) {
        // Only whitespace may legitimately precede a key. Anything else is malformed, and
        // the tokenizer is the one that says so.
        pos = S::enter(|| scan.skip_ws(data, pos));
    }
    match data.get(pos) {
        Some(b'"') => {}
        Some(b'}') => {
            tokens.seek(pos + 1);
            return KeyStep::End;
        }
        _ => return KeyStep::Slow,
    }
    let at = &data[pos..];
    let word = head_word(at);
    if let Some((child, len)) = elems.match_quoted(word, at) {
        tokens.seek(pos + len);
        return KeyStep::Resolved(Some(child));
    }
    // None of the children's bytes are here, so the key still has to be crossed to reach the
    // `:`. Ask the word already in hand first: most keys are short enough to end inside it,
    // and answering from a register keeps a second dependent load off the chain that carries
    // one key's end into the next key's start. That chain is what this loop is bound by —
    // the instructions this function removes were never the binding constraint.
    let end = match key_end_in_word(word) {
        WordEnd::Quote(i) => pos + i,
        WordEnd::Escape => return KeyStep::Slow,
        WordEnd::Beyond => S::enter(|| scan.string_event(data, pos + 8)),
    };
    match data.get(end) {
        Some(b'"') => {
            tokens.seek(end + 1);
            KeyStep::Resolved(None)
        }
        // An escape, a control character, or the end of input. Only the first can occur in a
        // valid document, and it means the comparison above was inconclusive.
        _ => KeyStep::Slow,
    }
}

/// Where a key ends within the first eight of its bytes.
enum WordEnd {
    /// The closing quote is at this offset from the opening one.
    Quote(usize),
    /// A `\` came first, so the key is escaped and nothing about it can be settled here.
    Escape,
    /// Neither appears in these eight bytes; the key runs on.
    Beyond,
}

/// Find a key's closing quote inside the eight bytes [`head_word`] loaded, by SWAR.
///
/// Byte 0 is the *opening* quote and is excluded. Zero padding past the end of the document
/// is neither `"` nor `\`, so a short read reports [`WordEnd::Beyond`] and the caller's
/// scan handles it.
#[inline(always)]
fn key_end_in_word(word: u64) -> WordEnd {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    // The classic has-a-zero-byte test: marks the high bit of every byte that is zero.
    #[inline(always)]
    fn zero_bytes(x: u64) -> u64 {
        x.wrapping_sub(LO) & !x & HI
    }
    let quotes = zero_bytes(word ^ (LO * b'"' as u64)) & !0x80;
    let escapes = zero_bytes(word ^ (LO * b'\\' as u64));
    if quotes == 0 {
        // A `\` with no quote after it inside the word still means "escaped": the escape is
        // part of this key, wherever it ends.
        return if escapes == 0 {
            WordEnd::Beyond
        } else {
            WordEnd::Escape
        };
    }
    if escapes != 0 && escapes.trailing_zeros() < quotes.trailing_zeros() {
        return WordEnd::Escape;
    }
    WordEnd::Quote((quotes.trailing_zeros() >> 3) as usize)
}

/// Advance past a value the expression does not name, **without tokenizing it**. The
/// cursor must be at the value's first byte (its opening token not yet read).
///
/// Only where such a value *ends* can matter, never what it contains — so this asks the
/// cheapest question that finds the end for each of the three shapes JSON offers. A
/// container ends where its brackets balance, which [`leave_value`] already finds in bulk. A
/// string ends at its first unescaped quote. Anything else — a number, `true`, `false`,
/// `null` — ends at the first byte that cannot belong to one. Tokenizing instead runs the
/// state machine over every digit and letter to build a value that is then dropped, and in a
/// skip-heavy document most fields are this case.
///
/// # Validation
///
/// This extends the divergence [`leave_value`] documents from skipped *containers* to
/// skipped *scalars*: `{"a":tru,"b":2}` is accepted when no expression names `a`, where
/// tokenizing it raised a syntax error. Strings are still checked for termination and for
/// control characters, and brackets still have to balance. Fields an expression does name
/// remain fully tokenized. Same contract as before — decide this expression against this
/// document, do not validate the document — applied consistently to everything skipped.
#[inline(always)]
fn skip_unnamed_value<S: Scan>(
    tokens: &mut GenericTokenizer<'_, S>,
    depth: usize,
) -> Result<(), MatchError> {
    let data = tokens.input();
    let scan = tokens.scan();
    // `enter` for the same reason `leave_value` opens one: these are `S`'s kernels, and a
    // backend supplies them as intrinsics that only vectorise inside its feature context.
    // `Ok(start, None)` means "a container opens at `start`", which needs `leave_value` and
    // so has to happen outside this borrow of the tokenizer.
    let (start, end) = S::enter(|| -> Result<(usize, Option<usize>), MatchError> {
        let start = scan.skip_ws(data, tokens.position());
        match data.get(start) {
            Some(b'{' | b'[') => Ok((start, None)),
            Some(b'"') => {
                let mut pos = start + 1;
                loop {
                    pos = scan.string_event(data, pos);
                    match data.get(pos) {
                        Some(b'"') => return Ok((start, Some(pos + 1))),
                        // An escape consumes the next byte, whatever it is; a `\uXXXX`'s
                        // digits are ordinary content to a scan that only needs the end.
                        Some(b'\\') => pos += 2,
                        // A control character is malformed inside a string, and running off
                        // the end means the string never closed.
                        _ => return Err(MatchError::Structure("unterminated string")),
                    }
                }
            }
            // A number, `true`, `false` or `null`. None of these can contain a delimiter,
            // whitespace, or a quote, so the first one of those ends the value. `"` is in
            // the set for damage control rather than for valid input: without it a
            // malformed scalar would run on through the following key and swallow whole
            // members, where stopping here confines the divergence below to one value.
            Some(_) => {
                let mut pos = start;
                while !matches!(
                    data.get(pos),
                    None | Some(b',' | b'}' | b']' | b'"' | b' ' | b'\t' | b'\r' | b'\n')
                ) {
                    pos += 1;
                }
                if pos == start {
                    // The value's first byte already ends it, so there is no value: `{"a":,}`.
                    return Err(MatchError::Structure(
                        "unexpected token where a value was expected",
                    ));
                }
                Ok((start, Some(pos)))
            }
            None => Err(MatchError::Structure("unexpected end of input")),
        }
    })?;
    match end {
        Some(end) => tokens.seek(end),
        None => {
            tokens.seek(start + 1);
            leave_value(tokens, depth + 1)?;
        }
    }
    Ok(())
}

/// Skip a value whose leading `token` has already been read.
fn skip_value<S: Scan>(
    tokens: &mut GenericTokenizer<'_, S>,
    token: Token<'_>,
    depth: usize,
) -> Result<(), MatchError> {
    match token.token_type {
        TokenType::ObjectStart | TokenType::ArrayStart => leave_value(tokens, depth + 1),
        t if t.is_literal() => Ok(()),
        _ => Err(MatchError::Structure(
            "unexpected token while skipping a value",
        )),
    }
}

/// Advance past the currently-open container (whose opening token was already read).
///
/// This does **not** tokenize. Once a value is known to be irrelevant, the only bytes that
/// can matter are the four brackets — which change nesting — and `"`, which begins a region
/// where a bracket is mere content. Numbers, literals, keys, colons and commas inside the
/// skipped value need not be recognised at all, so this walks the bytes with
/// [`Scan::structural_event`] instead of running the state machine once per token. On
/// `people.json` that is ~14 tokens' worth of FSM work replaced by one vector compare for
/// most skipped fields.
///
/// Iterative: nesting inside the skipped value costs no stack. `outer` is how many containers
/// enclose it, so the shared [`MAX_DEPTH`] ceiling still applies.
///
/// # Validation
///
/// A skipped region is checked for bracket balance and string termination, and for nothing
/// else. `{"a": [01, tru, ,]}` skipped wholesale is accepted, where tokenizing it would have
/// raised a syntax error. The engine's contract is "decide this expression against this
/// document", not "validate this document", and the fields an expression names *are* still
/// fully tokenized. Note gojsonsm does tokenize what it skips, so this is a divergence.
///
/// `#[inline]` is load-bearing, not a hint about size: this body calls `S`'s kernels, and a
/// backend like [`Avx2Scan`](crate::simd::Avx2Scan) supplies those as intrinsics that are
/// only legal — and only compile to vector instructions — inside the `#[target_feature]`
/// region [`Scan::enter`] wraps around the whole scan. Left outlined, LLVM compiles the
/// 256-bit intrinsics for the baseline target and scalarises them, which is far slower than
/// the tokenizing skip this replaces.
#[inline(always)]
fn leave_value<S: Scan>(
    tokens: &mut GenericTokenizer<'_, S>,
    outer: usize,
) -> Result<(), MatchError> {
    let scan = tokens.scan();
    let data = tokens.input();
    let from = tokens.position();
    // `enter` is load-bearing, not decoration. A backend like `Avx2Scan` supplies its
    // kernels as intrinsics that only compile to vector instructions inside the
    // `#[target_feature]` region this opens. Relying on the caller's region instead means
    // relying on this call being inlined into it, and it is not: LLVM outlines the skip and
    // the 256-bit intrinsics get built for the baseline target, which is far slower than the
    // tokenizing skip this replaces. Opening a region here is idempotent when one
    // is already open, and free for backends whose `enter` is the identity.
    let end = S::enter(|| scan.skip_container(data, from, outer));
    tokens.seek(end.map_err(|e| match e {
        SkipError::Unterminated => MatchError::Structure("unexpected end of input"),
        SkipError::TooDeep => MatchError::TooDeep,
    })?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{compile, Projection};
    use jsonsm_ast::{CompareOp, Expr, Field, Literal, PathComponent};

    fn field(keys: &[&str]) -> Expr {
        Expr::Field(Field::root(
            keys.iter()
                .map(|k| PathComponent::Key((*k).to_owned()))
                .collect(),
        ))
    }

    fn run(expr: &Expr, doc: &str) -> bool {
        let def = compile(
            std::slice::from_ref(expr),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        m.matches(doc.as_bytes()).unwrap().matched()
    }

    #[test]
    fn field_vs_constant() {
        let e = Expr::compare(
            CompareOp::Equals,
            field(&["name"]),
            Expr::Value(Literal::String("Brett".into())),
        );
        assert!(run(&e, r#"{"name": "Brett", "age": 30}"#));
        assert!(!run(&e, r#"{"name": "Alice"}"#));
        // Escaped key + escaped value still compare by decoded content.
        assert!(run(
            &Expr::compare(
                CompareOp::Equals,
                field(&["na\tme"]),
                Expr::Value(Literal::String("a\nb".into()))
            ),
            r#"{"na\tme": "a\nb"}"#
        ));
    }

    #[test]
    fn nested_and_numeric_and_bool() {
        let e = Expr::And(vec![
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["nested", "ok"]),
                Expr::Value(Literal::Bool(true)),
            ),
        ]);
        assert!(run(&e, r#"{"age": 30, "nested": {"ok": true}}"#));
        assert!(!run(&e, r#"{"age": 60, "nested": {"ok": true}}"#));
        assert!(!run(&e, r#"{"age": 30, "nested": {"ok": false}}"#));
    }

    #[test]
    fn missing_field_semantics() {
        // age < 50 with no age -> false.
        assert!(!run(
            &Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            r#"{"other": 1}"#
        ));
        // age != 50 with no age -> false: `==` is Unknown, and negating Unknown leaves it
        // Unknown, so nothing about an absent field can satisfy a comparison.
        assert!(!run(
            &Expr::compare(
                CompareOp::NotEquals,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            r#"{"other": 1}"#
        ));
        // exists / notexists.
        assert!(!run(
            &Expr::Exists(Box::new(field(&["age"]))),
            r#"{"other": 1}"#
        ));
        assert!(run(
            &Expr::NotExists(Box::new(field(&["age"]))),
            r#"{"other": 1}"#
        ));
        assert!(run(
            &Expr::Exists(Box::new(field(&["other"]))),
            r#"{"other": 1}"#
        ));
    }

    /// How a **missing** operand interacts with the structure above it.
    ///
    /// The rule, in one sentence: a comparison against an absent field is `Unknown`, and no
    /// amount of boolean structure can turn `Unknown` into a match. `NOT Unknown` is `Unknown`,
    /// `Unknown AND True` is `Unknown`, and only at the root — where `Unknown` and `False` are
    /// both "no match" — does the distinction stop mattering.
    ///
    /// **This is a deliberate divergence from gojsonsm**, which is why each case below carries
    /// the reference's answer too. gojsonsm settles an unevaluated leaf to `false` and lets that
    /// propagate, so a `NOT` above it yields `true`: it reports `age != 50` and `NOT (age == 50)`
    /// as matching a document with no `age` at all. The consequence is that `NOT (age < 50)` and
    /// `age >= 50` select different documents, which makes the most ordinary rewrite in query
    /// planning unsound. Three-valued logic keeps them equal, at the price of the law of excluded
    /// middle — `p OR NOT p` is `Unknown` when `p` is.
    ///
    /// `EXISTS` is the deliberate exception and is asserted here as well: it asks about presence
    /// rather than about a value, so absence *is* its answer and it stays definite. That is what
    /// keeps `NOT EXISTS` usable as the escape hatch for "match documents lacking this field",
    /// which is now the only way to write that.
    #[test]
    fn missing_operands_interact_with_negation_and_connectives() {
        let no_age = r#"{"other":1}"#;
        let eq_age_50 = || {
            Expr::compare(
                CompareOp::Equals,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            )
        };
        let lt_age_50 = || {
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            )
        };
        let eq_b_1 = || {
            Expr::compare(
                CompareOp::Equals,
                field(&["b"]),
                Expr::Value(Literal::Int(1)),
            )
        };

        // The leaf itself is falsey.
        assert!(!run(&eq_age_50(), no_age), "age == 50");
        assert!(!run(&lt_age_50(), no_age), "age < 50");

        // …and negation does not change that. gojsonsm answers `true` to both of these.
        assert!(
            !run(&Expr::Not(Box::new(eq_age_50())), no_age),
            "NOT (age == 50) — Unknown, not true"
        );
        assert!(
            !run(&Expr::Not(Box::new(lt_age_50())), no_age),
            "NOT (age < 50) — Unknown, so it agrees with `age >= 50`"
        );
        assert!(
            !run(
                &Expr::Not(Box::new(Expr::Not(Box::new(eq_age_50())))),
                no_age
            ),
            "NOT NOT (age == 50)"
        );

        // Through the connectives: with `b` present and equal to 1, the AND is
        // `Unknown AND True` = `Unknown` (gojsonsm: false, so its negation true), while the OR is
        // `Unknown OR True` = `True` — the one case where a definite sibling rescues the result,
        // and the one case both engines agree on.
        let b_1 = r#"{"b":1}"#;
        assert!(
            !run(
                &Expr::Not(Box::new(Expr::And(vec![eq_age_50(), eq_b_1()]))),
                b_1
            ),
            "NOT (age == 50 AND b == 1) — Unknown"
        );
        assert!(
            run(&Expr::Or(vec![eq_age_50(), eq_b_1()]), b_1),
            "age == 50 OR b == 1 — True, rescued by the present operand"
        );
        assert!(
            !run(
                &Expr::Not(Box::new(Expr::Or(vec![eq_age_50(), eq_b_1()]))),
                b_1
            ),
            "NOT (age == 50 OR b == 1)"
        );

        // The escape hatch: presence questions stay definite, so this remains the way to select
        // documents that lack a field.
        assert!(
            !run(&Expr::Exists(Box::new(field(&["age"]))), no_age),
            "EXISTS(age)"
        );
        assert!(
            run(&Expr::NotExists(Box::new(field(&["age"]))), no_age),
            "NOT EXISTS(age) must stay true — Unknown here would make absence unmatchable"
        );
        assert!(
            run(&Expr::Exists(Box::new(field(&["other"]))), no_age),
            "EXISTS(other)"
        );
        // And nested, where the absent field is under a present container.
        assert!(
            run(
                &Expr::NotExists(Box::new(field(&["a", "x"]))),
                r#"{"a":{"y":1}}"#
            ),
            "NOT EXISTS(a.x) with a present but x absent"
        );

        // Sanity: a present field is unaffected by any of the above.
        assert!(run(&eq_age_50(), r#"{"age":50}"#), "age == 50 [present]");
        assert!(
            !run(&Expr::Not(Box::new(eq_age_50())), r#"{"age":50}"#),
            "NOT (age == 50) [present]"
        );

        // A *cross-field* comparison reaches missingness by the other of the two routes: the
        // op does run, and an operand resolves to `FastVal::Missing` from an unfilled slot, so
        // `eval_op` returns `missing_result` directly instead of the tree defaulting an
        // unvisited leaf. Worth asserting separately — inverting `eval_op`'s missing branch
        // leaves every case above passing, because there the field is absent from the document
        // and the op is never evaluated at all. Values again taken from gojsonsm.
        let a_eq_b = || Expr::compare(CompareOp::Equals, field(&["a"]), field(&["b"]));
        let a_lt_b = || Expr::compare(CompareOp::LessThan, field(&["a"]), field(&["b"]));
        assert!(run(&a_eq_b(), r#"{"a":1,"b":1}"#), "a == b [both present]");
        assert!(!run(&a_eq_b(), r#"{"a":1}"#), "a == b [b missing]");
        assert!(!run(&a_eq_b(), r#"{"b":1}"#), "a == b [a missing]");
        assert!(!run(&a_eq_b(), r#"{"c":1}"#), "a == b [both missing]");
        assert!(!run(&a_lt_b(), r#"{"a":1}"#), "a < b [b missing]");
        // …and negation does not rescue it on this route either (gojsonsm: true for all three).
        assert!(
            !run(&Expr::Not(Box::new(a_eq_b())), r#"{"a":1}"#),
            "NOT (a == b) [b missing]"
        );
        assert!(
            !run(&Expr::Not(Box::new(a_eq_b())), r#"{"c":1}"#),
            "NOT (a == b) [both missing]"
        );
        assert!(
            !run(&Expr::Not(Box::new(a_lt_b())), r#"{"a":1}"#),
            "NOT (a < b) [b missing]"
        );
    }

    /// Sealing at a container's close is what lets absence decide the tree *during* the scan, and
    /// this pins that observably: the tail of each document below is nested past [`MAX_DEPTH`],
    /// so the match can only succeed if the scan stopped before reaching it.
    ///
    /// The tail is deep rather than malformed on purpose. A malformed tail is the more obvious
    /// tripwire, but the engine may assume its input is valid JSON — so "a malformed tail is
    /// reported" is not behaviour it promises, and pinning an optimisation to it would make any
    /// later change to the skip paths look like a regression. Depth is a *documented* limit, enforced identically whether the deep field is
    /// matched or skipped ([`MAX_SKIP_DEPTH`](crate::tokenizer::MAX_SKIP_DEPTH) says why), so it
    /// is an observable the contract actually supports.
    ///
    /// It also marks the boundary of the optimisation, which is narrower than "an absent field
    /// ends the scan". Sealing produces a definite `Unknown`, and whether that settles the root
    /// depends on what sits above it: a bare comparison, a `NOT`, or an `EXISTS` is decided at
    /// once, but under `AND`/`OR` the sibling is still needed, because `Unknown AND False` is
    /// `False` while `Unknown AND True` is `Unknown`. So an `AND` only finishes early if its
    /// other operand happens to have been resolved already — which, for a document-ordered scan,
    /// means it appeared earlier.
    #[test]
    fn sealing_lets_absence_decide_before_the_document_ends() {
        // Valid JSON, but `zzz` is nested past the ceiling: reaching the tail is a `TooDeep`
        // error, not a mismatch.
        let deep_value = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        let bad_tail_doc = format!(r#"{{"a":{{"y":1}},"zzz":{deep_value}}}"#);
        let bad_tail = bad_tail_doc.as_str();

        // EXISTS is settled by absence itself, so this decides at `a`'s close.
        assert!(
            run(&Expr::NotExists(Box::new(field(&["a", "x"]))), bad_tail),
            "NOT EXISTS(a.x) should resolve at a's close, before the malformed tail"
        );

        // A bare comparison is decided too — as Unknown, so it does not match, but it does stop:
        // the malformed tail must not turn a non-match into an error.
        let m = compile(
            std::slice::from_ref(&Expr::compare(
                CompareOp::Equals,
                field(&["a", "x"]),
                Expr::Value(Literal::Int(1)),
            )),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut fm = FastMatcher::new(&m);
        assert!(!fm
            .matches(bad_tail.as_bytes())
            .expect("scan should have stopped at a's close")
            .matched());

        // An AND stops too, even though its other operand has not been seen and the AND is
        // therefore *unresolved as a value*: `Unknown AND anything` is never `True`, so the
        // verdict is already settled. This is what `root_cannot_be_true` adds over waiting for the
        // root to resolve — without it, this case reads on and dies on the malformed tail.
        let stops = |e: &Expr| -> Result<bool, MatchError> {
            let d = compile(
                std::slice::from_ref(e),
                &Projection::new(),
                &DefaultCollation,
            )
            .unwrap();
            let mut fm = FastMatcher::new(&d);
            fm.matches(bad_tail.as_bytes()).map(|o| o.matched())
        };
        let x_eq_1 = || {
            Expr::compare(
                CompareOp::Equals,
                field(&["a", "x"]),
                Expr::Value(Literal::Int(1)),
            )
        };
        let zzz_eq_2 = || {
            Expr::compare(
                CompareOp::Equals,
                field(&["zzz"]),
                Expr::Value(Literal::Int(2)),
            )
        };
        assert!(!stops(&Expr::And(vec![x_eq_1(), zzz_eq_2()])).expect("AND settles as a verdict"));

        // Shapes an `And`-only rule misses but the bounds analysis catches. `NOT (absent OR
        // unseen)`: the `Or`'s *lower* bound is Unknown, and `Not` reverses the order, so that
        // becomes the upper bound — never `True`.
        assert!(
            !stops(&Expr::Not(Box::new(Expr::Or(vec![x_eq_1(), zzz_eq_2()]))))
                .expect("NOT over an OR containing an absent field settles")
        );
        // A disjunction every branch of which is separately capped.
        let a_z_eq_1 = Expr::compare(
            CompareOp::Equals,
            field(&["a", "z"]),
            Expr::Value(Literal::Int(1)),
        );
        assert!(!stops(&Expr::Or(vec![
            Expr::And(vec![x_eq_1(), zzz_eq_2()]),
            Expr::And(vec![a_z_eq_1, zzz_eq_2()]),
        ]))
        .expect("an OR whose every branch is capped settles"));

        // The boundary is `OR`, where the unseen operand genuinely can still change the answer:
        // `Unknown OR True` is `True`. So this one must read on — and hits the malformed tail.
        assert!(
            stops(&Expr::Or(vec![x_eq_1(), zzz_eq_2()])).is_err(),
            "an OR whose sibling could still be true must not stop early"
        );

        // …and once that sibling is known and not true, the OR is capped as well. Here `zzz` is
        // present and unequal, so by the time `a` closes the OR can no longer be true.
        let seen_then_deep = format!(r#"{{"zzz":9,"a":{{"y":1}},"q":{deep_value}}}"#);
        assert!(!stops_on(&Expr::Or(vec![x_eq_1(), zzz_eq_2()]), &seen_then_deep)
        .expect("OR with a resolved false sibling settles"));
    }

    /// The two ways the "cannot be `True`" walk could overreach. Both are unsound *optimisations*
    /// — they would stop a scan that still had a match to find — so each fixture is arranged so
    /// the correct answer is `true`, which an early exit turns into `false`.
    #[test]
    fn cannot_be_true_does_not_overreach() {
        // Through a NOT. `a` closes first, sealing `a.x` to Unknown, which caps the AND below
        // `True`. But the AND is under a NOT, and `NOT False` is `True` — and the AND really is
        // `False` here, because `b` is 9. So this must match, which means the walk must stop at
        // the NOT rather than concluding the root can never be true.
        let not_and = Expr::Not(Box::new(Expr::And(vec![
            Expr::compare(
                CompareOp::Equals,
                field(&["a", "x"]),
                Expr::Value(Literal::Int(1)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["b"]),
                Expr::Value(Literal::Int(2)),
            ),
        ])));
        assert!(
            stops_on(&not_and, r#"{"a":{"y":1},"b":9}"#).unwrap(),
            "NOT (a.x == 1 AND b == 2) must match: the AND is definitely false"
        );

        // Out of a loop body. The first element lacks `y`, so the body's AND is capped below
        // `True` for *that element* — which says nothing about the loop, let alone the root. The
        // second element satisfies the body, so ANY is true; a walk that crossed the loop's stall
        // boundary would have stopped the scan at the first element.
        let any_x_and_y = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::And(vec![
                Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field {
                        root: 1,
                        path: vec![PathComponent::Key("x".into())],
                    }),
                    Expr::Value(Literal::Int(1)),
                ),
                Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field {
                        root: 1,
                        path: vec![PathComponent::Key("y".into())],
                    }),
                    Expr::Value(Literal::Int(2)),
                ),
            ])),
        };
        assert!(
            stops_on(&any_x_and_y, r#"{"xs":[{"x":1},{"x":1,"y":2}]}"#).unwrap(),
            "ANY must keep scanning: an unevaluable element says nothing about the root"
        );
    }

    /// `matches` against an arbitrary document, for tests that need a specific field order.
    fn stops_on(e: &Expr, doc: &str) -> Result<bool, MatchError> {
        let d = compile(
            std::slice::from_ref(e),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut fm = FastMatcher::new(&d);
        fm.matches(doc.as_bytes()).map(|o| o.matched())
    }

    /// Quantifiers are connectives over elements, so `Unknown` flows through them the same way —
    /// and an absent array is not an empty one.
    #[test]
    fn loops_over_absent_and_unevaluable_elements() {
        let any_x_eq_1 = |mode: LoopType| Expr::Loop {
            loop_type: mode,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("x".into())],
                }),
                Expr::Value(Literal::Int(1)),
            )),
        };

        // An absent array has nothing to quantify over: neither ANY nor EVERY has an answer, so
        // neither matches — note EVERY is *not* vacuously true here, because there is no empty
        // array, there is no array.
        for mode in [LoopType::Any, LoopType::Every, LoopType::AnyEvery] {
            assert!(
                !run(&any_x_eq_1(mode), r#"{"other":1}"#),
                "{mode:?} over an absent array"
            );
        }
        // Under a NOT is where an absent array's `Unknown` is distinguishable from `False`: a
        // `False` here would flip to a match, which is the same mistake as for a comparison.
        for mode in [LoopType::Any, LoopType::Every, LoopType::AnyEvery] {
            assert!(
                !run(&Expr::Not(Box::new(any_x_eq_1(mode))), r#"{"other":1}"#),
                "NOT {mode:?} over an absent array must not match"
            );
        }
        // …whereas over an *empty* array the quantifier is definite, so NOT does invert it.
        assert!(
            run(
                &Expr::Not(Box::new(any_x_eq_1(LoopType::Any))),
                r#"{"xs":[]}"#
            ),
            "NOT ANY over [] is a real false, so its negation matches"
        );

        // An empty array *is* empty: ANY is false, EVERY vacuously true, ANYEVERY false.
        assert!(
            !run(&any_x_eq_1(LoopType::Any), r#"{"xs":[]}"#),
            "ANY over []"
        );
        assert!(
            run(&any_x_eq_1(LoopType::Every), r#"{"xs":[]}"#),
            "EVERY over [] is vacuously true"
        );
        assert!(
            !run(&any_x_eq_1(LoopType::AnyEvery), r#"{"xs":[]}"#),
            "ANYEVERY over [] needs an element"
        );

        // An element lacking the field the body names is Unknown for that element. ANY can still
        // be rescued by a later element that does match; EVERY cannot conclude true over one it
        // could not evaluate.
        let mixed = r#"{"xs":[{"y":9},{"x":1}]}"#;
        assert!(
            run(&any_x_eq_1(LoopType::Any), mixed),
            "ANY — a definite match outweighs an Unknown element"
        );
        assert!(
            !run(&any_x_eq_1(LoopType::Every), mixed),
            "EVERY — an Unknown element denies it"
        );
        // All elements definite: the ordinary answers.
        assert!(run(
            &any_x_eq_1(LoopType::Every),
            r#"{"xs":[{"x":1},{"x":1}]}"#
        ));
        assert!(!run(
            &any_x_eq_1(LoopType::Every),
            r#"{"xs":[{"x":1},{"x":2}]}"#
        ));
    }

    #[test]
    fn no_cross_type_coercion() {
        // n == "5" is false when n is the number 5.
        assert!(!run(
            &Expr::compare(
                CompareOp::Equals,
                field(&["n"]),
                Expr::Value(Literal::String("5".into()))
            ),
            r#"{"n": 5}"#
        ));
    }

    #[test]
    fn or_short_circuit_and_not() {
        let e = Expr::Or(vec![
            Expr::compare(
                CompareOp::Equals,
                field(&["a"]),
                Expr::Value(Literal::Int(1)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["b"]),
                Expr::Value(Literal::Int(2)),
            ),
        ]);
        assert!(run(&e, r#"{"a": 1}"#));
        assert!(run(&e, r#"{"b": 2}"#));
        assert!(!run(&e, r#"{"a": 9, "b": 9}"#));
        assert!(run(
            &Expr::Not(Box::new(Expr::compare(
                CompareOp::Equals,
                field(&["a"]),
                Expr::Value(Literal::Int(9))
            ))),
            r#"{"a": 1}"#
        ));
    }

    #[test]
    fn loops_over_arrays() {
        let anyin = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("cillum".into())),
            )),
        };
        assert!(run(&anyin, r#"{"tags": ["a", "cillum", "z"]}"#));
        assert!(!run(&anyin, r#"{"tags": ["a", "b"]}"#));
        assert!(!run(&anyin, r#"{"tags": []}"#));
        // missing array -> loop does not apply -> false.
        assert!(!run(&anyin, r#"{"other": 1}"#));

        // any-in over objects in an array, comparing a nested field of each element.
        let anyin_obj = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["items"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::GreaterThan,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("qty".into())],
                }),
                Expr::Value(Literal::Int(10)),
            )),
        };
        assert!(run(&anyin_obj, r#"{"items": [{"qty": 5}, {"qty": 20}]}"#));
        assert!(!run(&anyin_obj, r#"{"items": [{"qty": 5}, {"qty": 8}]}"#));
    }

    /// Re-render compact JSON with a space around every structural byte that is outside a
    /// string, so `{"a":[1,2]}` becomes `{ "a" : [ 1 , 2 ] }`.
    fn spaced(compact: &str) -> String {
        let mut out = String::with_capacity(compact.len() * 3);
        let (mut in_str, mut escaped) = (false, false);
        for c in compact.chars() {
            if in_str {
                out.push(c);
                match c {
                    _ if escaped => escaped = false,
                    '\\' => escaped = true,
                    '"' => in_str = false,
                    _ => {}
                }
                continue;
            }
            match c {
                '"' => {
                    in_str = true;
                    out.push(c);
                }
                ':' | ',' | '{' | '}' | '[' | ']' => {
                    out.push(' ');
                    out.push(c);
                    out.push(' ');
                }
                _ => out.push(c),
            }
        }
        out
    }

    /// Whitespace before a structural byte is what sends [`take_structural`] /
    /// [`take_delim`] to their tokenizer fallback, and `serde_json::to_vec` — which every
    /// generated document in the differential sweeps goes through — never emits any. So
    /// without this the fallback is unexecuted code across the whole test suite.
    ///
    /// Each case asserts the *same* answer with and without the whitespace. Scalars only:
    /// container values collate by raw bytes, so respacing one genuinely changes its value.
    #[test]
    fn structural_bytes_may_be_preceded_by_whitespace() {
        let cases: &[(Expr, &str, bool)] = &[
            (
                Expr::compare(
                    CompareOp::Equals,
                    field(&["b", "c"]),
                    Expr::Value(Literal::Int(2)),
                ),
                r#"{"a":1,"b":{"c":2},"d":3}"#,
                true,
            ),
            (
                Expr::compare(
                    CompareOp::Equals,
                    field(&["b", "c"]),
                    Expr::Value(Literal::Int(9)),
                ),
                r#"{"a":1,"b":{"c":2},"d":3}"#,
                false,
            ),
            // Indexed element access, and a loop: the two array walks.
            (
                Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field::root(vec![
                        PathComponent::Key("xs".into()),
                        PathComponent::Index(1),
                    ])),
                    Expr::Value(Literal::Int(7)),
                ),
                r#"{"xs":[6,7,8]}"#,
                true,
            ),
            (
                Expr::Loop {
                    loop_type: LoopType::Any,
                    var: 1,
                    in_expr: Box::new(field(&["xs"])),
                    sub_expr: Box::new(Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field {
                            root: 1,
                            path: vec![],
                        }),
                        Expr::Value(Literal::Int(7)),
                    )),
                },
                r#"{"xs":[6,7,8]}"#,
                true,
            ),
            // A loop over objects, which walks both container kinds per element.
            (
                Expr::Loop {
                    loop_type: LoopType::Every,
                    var: 1,
                    in_expr: Box::new(field(&["xs"])),
                    sub_expr: Box::new(Expr::compare(
                        CompareOp::GreaterThan,
                        Expr::Field(Field {
                            root: 1,
                            path: vec![PathComponent::Key("q".into())],
                        }),
                        Expr::Value(Literal::Int(0)),
                    )),
                },
                r#"{"xs":[{"q":1,"r":0},{"q":2}]}"#,
                true,
            ),
            // A `:` and a `,` inside string content must not be mistaken for structure.
            (
                Expr::compare(
                    CompareOp::Equals,
                    field(&["k"]),
                    Expr::Value(Literal::String("a:b,c}".into())),
                ),
                r#"{"j":"x,y","k":"a:b,c}"}"#,
                true,
            ),
        ];

        for (expr, doc, want) in cases {
            assert_eq!(run_all_backends(expr, doc), *want, "compact: {doc}");
            let spaced_doc = spaced(doc);
            assert_eq!(
                run_all_backends(expr, &spaced_doc),
                *want,
                "spaced: {spaced_doc}"
            );
        }
    }

    /// A value the expression does not name is crossed by scanning for its end rather than
    /// tokenizing it ([`skip_unnamed_value`]), so the fields *after* it are found only if
    /// that scan stops in exactly the right place.
    ///
    /// Strings are where that is easy to get wrong: a `\"` inside one is content, and an
    /// escape scan that advances by one instead of two mistakes it for the closing quote and
    /// desynchronises the rest of the object.
    #[test]
    fn skipped_values_end_where_they_should() {
        let expr = Expr::compare(
            CompareOp::Equals,
            field(&["z"]),
            Expr::Value(Literal::Int(1)),
        );
        // Each document puts an awkward value in `y`, which no expression names, ahead of
        // the `z` the expression does name.
        for doc in [
            r#"{"y":"a\"b","z":1}"#,         // escaped quote
            r#"{"y":"a\\","z":1}"#,          // trailing escaped backslash
            r#"{"y":"A\"x","z":1}"#,         // unicode escape then an escaped quote
            r#"{"y":"},\"z\":9,","z":1}"#,   // structure-looking bytes inside the string
            r#"{"y":"","z":1}"#,             // empty string
            r#"{"y":-1.5e+10,"z":1}"#,       // number with sign and exponent
            r#"{"y":[1,{"k":"]"}],"z":1}"#,  // nested containers, bracket inside a string
            r#"{"y":{"a":{"b":[]}},"z":1}"#, // nested objects
            r#"{"y":null,"z":1}"#,
            r#"{ "y" : true , "z" : 1 }"#, // whitespace everywhere
        ] {
            assert!(
                run_all_backends(&expr, doc),
                "should have found z after y: {doc}"
            );
        }

        // The same, with the awkward value last, so its end must coincide with the `}`.
        let expr_z_first = Expr::compare(
            CompareOp::Equals,
            field(&["z"]),
            Expr::Value(Literal::Int(1)),
        );
        for doc in [
            r#"{"z":1,"y":"a\"b"}"#,
            r#"{"z":1,"y":[1,2]}"#,
            r#"{"z":1,"y":7}"#,
        ] {
            assert!(run_all_backends(&expr_z_first, doc), "{doc}");
        }
    }

    /// The peeking fast paths must not become the thing that decides a document is
    /// well-formed: when the expected byte is not there, the tokenizer still has to run and
    /// still has to reject.
    ///
    /// Each expression is chosen so the damage is on a path the matcher actually *walks*.
    /// Two things would otherwise mask a failure to reject:
    ///
    /// * short-circuiting — with `b == 2`, even `{"a":1,"b":2` is accepted, because the root
    ///   is decided at `"b":2` and the scan stops there; so these name an absent field, or
    ///   one whose value is past the damage;
    /// * the bulk skip, which by design checks only bracket balance and string termination,
    ///   so `{"xs":[1;2]}` is accepted whenever nothing looks inside `xs` (see
    ///   [`leave_value`]) — hence the array cases use expressions that enter the array.
    ///
    /// What this pins is that the peeking paths *decline* rather than decide, not that any
    /// particular malformed document is rejected. The engine may assume valid JSON and its
    /// answer for an invalid one is undefined, so these cases record best-effort behaviour.
    /// If a future fast path stops detecting one of them, relax the case — do not add work to
    /// the fast path to keep it passing.
    #[test]
    fn malformed_structure_is_still_rejected() {
        let absent = Expr::compare(
            CompareOp::Equals,
            field(&["absent"]),
            Expr::Value(Literal::Int(1)),
        );
        // Walks the array element by element, so its delimiters are tokenized.
        let in_loop = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::Int(99)),
            )),
        };
        // Walks the array by index, which is the other array walk.
        let by_index = Expr::compare(
            CompareOp::Equals,
            Expr::Field(Field::root(vec![
                PathComponent::Key("xs".into()),
                PathComponent::Index(3),
            ])),
            Expr::Value(Literal::Int(99)),
        );

        let cases: &[(&Expr, &str)] = &[
            (&absent, r#"{"a" 1,"b":2}"#),      // no ':' after a key
            (&absent, r#"{"a":1;"b":2}"#),      // neither ',' nor '}' between members
            (&absent, r#"{"a":1,"b":2"#),       // object never closed
            (&absent, r#"{"a":,"b":2}"#),       // missing value
            (&in_loop, r#"{"xs":[1;2]}"#),      // neither ',' nor ']' between elements
            (&in_loop, r#"{"xs":[1,2}"#),       // array closed by the wrong bracket
            (&by_index, r#"{"xs":[1;2,3,4]}"#), // same, on the indexed walk
        ];

        for (expr, bad) in cases {
            let def = compile(
                std::slice::from_ref(*expr),
                &Projection::new(),
                &DefaultCollation,
            )
            .unwrap();
            let mut m = FastMatcher::new(&def);
            assert!(
                m.matches(bad.as_bytes()).is_err(),
                "should have been rejected: {bad}"
            );
        }
    }

    /// Object keys are resolved by comparing the document's raw bytes against each child's
    /// quoted key, which never tokenizes the key and so never decodes it. These are the
    /// document shapes where the bytes and the key disagree, or where one key's bytes lead
    /// into another's.
    ///
    /// Key equality is byte equality of the *decoded* key — JSON strings are UTF-8 by
    /// specification, so no collation or normalisation enters into it. That is what makes
    /// the raw comparison legitimate, and also what bounds it: a key spelled with escapes
    /// decodes to something its bytes do not show, so the comparison cannot settle it and
    /// must hand it back to the tokenizer rather than call it a miss.
    #[test]
    fn keys_resolve_through_their_encoding_not_their_bytes() {
        let a_eq = |n: i64| {
            Expr::compare(
                CompareOp::Equals,
                field(&["a"]),
                Expr::Value(Literal::Int(n)),
            )
        };

        // A key spelled with escapes is the same key. A raw comparison misses it, and that
        // miss must not be taken as final.
        assert!(
            run_all_backends(&a_eq(1), r#"{"\u0061":1}"#),
            "\\u0061 is the key a"
        );
        assert!(
            run_all_backends(&a_eq(1), r#"{"\u0062":2,"a":1}"#),
            "an escaped key before the wanted one"
        );
        // …and an escaped key that is *not* the wanted one must not be mistaken for it.
        assert!(
            !run_all_backends(&a_eq(2), r#"{"\u0062":2,"a":1}"#),
            "\\u0062 is b, not a"
        );

        // A named key that is a prefix of a document key, and vice versa. The closing quote
        // is the only thing separating these.
        assert!(!run_all_backends(&a_eq(1), r#"{"ab":1}"#), "a must not match ab");
        assert!(
            run_all_backends(&a_eq(2), r#"{"ab":1,"a":2}"#),
            "ab must not consume a"
        );
        // An escaped quote inside a key is content, not the key's end.
        assert!(
            run_all_backends(&a_eq(2), r#"{"a\"b":1,"a":2}"#),
            "a\\\"b must not be read as a"
        );

        // Keys longer than the eight-byte word the comparison prefilters on, including a
        // pair that agree across the whole word and differ only past it.
        let long_eq = |n: i64| {
            Expr::compare(
                CompareOp::Equals,
                field(&["abcdefgh"]),
                Expr::Value(Literal::Int(n)),
            )
        };
        assert!(run_all_backends(&long_eq(1), r#"{"abcdefgh":1}"#));
        assert!(
            !run_all_backends(&long_eq(1), r#"{"abcdefg":1,"abcdefghi":1}"#),
            "neighbours sharing the whole prefilter word must not match"
        );
        assert!(
            run_all_backends(&long_eq(2), r#"{"abcdefg":1,"abcdefghi":1,"abcdefgh":2}"#),
            "…and must not consume the real one either"
        );

        // A *named* key that is not its own JSON encoding disables the raw comparison
        // outright, because the closing quote stops being a reliable terminator: quoted, the
        // key `a\\` is `"a\\"`, which is a prefix of the document key `"a\\"x"` — whose
        // content is `a"x`, a different field. Compared raw, the first key here would be
        // taken for the second.
        assert!(
            run_all_backends(
                &Expr::compare(
                    CompareOp::Equals,
                    field(&["a\\"]),
                    Expr::Value(Literal::Int(2)),
                ),
                r#"{"a\"x":1,"a\\":2}"#
            ),
            "a backslash-terminated field name must not be matched by prefix"
        );

        // Whitespace may precede a key, and an object may simply be empty.
        assert!(run_all_backends(&a_eq(1), "{ \"zz\" : 2 ,\n\t\"a\" : 1 }"));
        assert!(!run_all_backends(&a_eq(1), "{}"));
    }

    /// Match `expr` against `doc` on **every** backend this CPU supports, asserting they
    /// agree, and return the shared result.
    ///
    /// Each backend is separately monomorphised code, so a path only one of them takes is
    /// untested code. That matters here because the direct loop-body path consumes container
    /// elements with [`leave_value`], which dispatches into the backend's scan kernels.
    fn run_all_backends(expr: &Expr, doc: &str) -> bool {
        let def = compile(
            std::slice::from_ref(expr),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut result: Option<bool> = None;
        #[cfg(feature = "simd")]
        let backends = crate::simd::Backend::available();
        #[cfg(not(feature = "simd"))]
        let backends = [()];
        for b in backends {
            let mut m = FastMatcher::new(&def);
            #[cfg(feature = "simd")]
            m.force_backend(b);
            #[cfg(not(feature = "simd"))]
            let _ = b;
            let got = m.matches(doc.as_bytes()).unwrap().matched();
            if let Some(prev) = result {
                assert_eq!(prev, got, "backends disagree on {doc}");
            }
            result = Some(got);
        }
        result.expect("at least one backend")
    }

    /// The loop-body shapes the per-element path has to get right: negation chains that must
    /// cancel rather than accumulate, the quantifiers' empty-array defaults, container elements
    /// typed and consumed whole, and an operand from an enclosing scope. Gathered in one test
    /// because each is a way for a body to be more than the single comparison the shape
    /// suggests.
    #[test]
    fn loop_body_shapes_the_element_path_must_get_right() {
        let loop_over = |lt, op, lit: Literal| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::compare(
                op,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(lit),
            )),
        };

        // `!=` lowers to Not(Equals), so the body bucket is a Not above the leaf the op
        // writes, and the element's result reaches the body only by propagating across it.
        let any_ne = loop_over(LoopType::Any, CompareOp::NotEquals, Literal::Int(7));
        assert!(run_all_backends(&any_ne, r#"{"xs": [7, 7, 8]}"#));
        assert!(!run_all_backends(&any_ne, r#"{"xs": [7, 7]}"#));

        let every_ne = loop_over(LoopType::Every, CompareOp::NotEquals, Literal::Int(7));
        assert!(run_all_backends(&every_ne, r#"{"xs": [1, 2, 3]}"#));
        assert!(!run_all_backends(&every_ne, r#"{"xs": [1, 7, 3]}"#));

        let anyevery_ne = loop_over(LoopType::AnyEvery, CompareOp::NotEquals, Literal::Int(7));
        assert!(run_all_backends(&anyevery_ne, r#"{"xs": [1, 2]}"#));
        assert!(!run_all_backends(&anyevery_ne, r#"{"xs": [1, 7]}"#));
        // ANY AND EVERY is false on an empty array, where EVERY is true.
        assert!(!run_all_backends(&anyevery_ne, r#"{"xs": []}"#));
        assert!(run_all_backends(&every_ne, r#"{"xs": []}"#));

        // A chain of two negations must cancel, not accumulate.
        let not_ne = Expr::Not(Box::new(loop_over(
            LoopType::Any,
            CompareOp::NotEquals,
            Literal::Int(7),
        )));
        assert!(!run_all_backends(&not_ne, r#"{"xs": [7, 8]}"#));
        assert!(run_all_backends(&not_ne, r#"{"xs": [7, 7]}"#));

        // Container elements: the element path has to consume the whole element and describe
        // it by the right type. Ordering is by type precedence (number < string < array <
        // object), so these discriminate an array element from an object one.
        let any_gt_str = loop_over(
            LoopType::Any,
            CompareOp::GreaterThan,
            Literal::String("s".into()),
        );
        assert!(run_all_backends(&any_gt_str, r#"{"xs": [1, [2, 3]]}"#));
        assert!(run_all_backends(
            &any_gt_str,
            r#"{"xs": [1, {"k": [4, 5]}]}"#
        ));
        assert!(!run_all_backends(&any_gt_str, r#"{"xs": [1, 2]}"#));
        // Nested containers, and a `]` inside a string, must not confuse the skip.
        let every_gt_str = loop_over(
            LoopType::Every,
            CompareOp::GreaterThan,
            Literal::String("s".into()),
        );
        assert!(run_all_backends(
            &every_gt_str,
            r#"{"xs": [[[1], {"a": "]]]"}], {"b": [{"c": 2}]}]}"#
        ));
        assert!(!run_all_backends(
            &every_gt_str,
            r#"{"xs": [[1], "a", {"b": 2}]}"#
        ));

        // An array element compared against a whole array held in an outer field: this
        // reaches the direct path through the *deferred* loop route, and only compares equal
        // if the element is described as an array rather than an object.
        let any_eq_ref = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                field(&["ref"]),
            )),
        };
        assert!(run_all_backends(
            &any_eq_ref,
            r#"{"ref": [1,2], "xs": [{"k":1}, [1,2]]}"#
        ));
        assert!(!run_all_backends(
            &any_eq_ref,
            r#"{"ref": [1,2], "xs": [{"k":1}, [1,3]]}"#
        ));
        // Same, with the array field appearing *after* the loop's array in the document.
        assert!(run_all_backends(
            &any_eq_ref,
            r#"{"xs": [{"k":1}, [1,2]], "ref": [1,2]}"#
        ));

        // Non-scalar, non-comparison bodies: `EXISTS` on the element is true for every
        // element that is there at all, including containers.
        let any_exists = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::Exists(Box::new(Expr::Field(Field {
                root: 1,
                path: vec![],
            })))),
        };
        assert!(run_all_backends(&any_exists, r#"{"xs": [{"k": 1}]}"#));
        assert!(!run_all_backends(&any_exists, r#"{"xs": []}"#));
    }

    #[test]
    fn every_and_anyevery() {
        let mk = |lt| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::GreaterThan,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::Int(0)),
            )),
        };
        assert!(run(&mk(LoopType::Every), r#"{"xs": [1, 2, 3]}"#));
        assert!(!run(&mk(LoopType::Every), r#"{"xs": [1, -1, 3]}"#));
        assert!(run(&mk(LoopType::Every), r#"{"xs": []}"#)); // vacuously true
        assert!(run(&mk(LoopType::AnyEvery), r#"{"xs": [1, 2]}"#));
        assert!(!run(&mk(LoopType::AnyEvery), r#"{"xs": []}"#)); // needs at least one
    }

    #[test]
    fn regex_matches() {
        let e = Expr::Matches {
            lhs: Box::new(field(&["email"])),
            pattern: Box::new(Expr::Value(Literal::String("@example\\.com$".into()))),
        };
        assert!(run(&e, r#"{"email": "a@example.com"}"#));
        assert!(!run(&e, r#"{"email": "a@other.org"}"#));
        assert!(!run(&e, r#"{"other": 1}"#)); // missing -> false
    }

    #[test]
    fn function_over_field() {
        use jsonsm_ast::Func;
        // mathRound(latitude) == 37
        let e = Expr::compare(
            CompareOp::Equals,
            Expr::Func(Func {
                name: "mathRound".into(),
                args: vec![field(&["latitude"])],
            }),
            Expr::Value(Literal::Int(37)),
        );
        assert!(run(&e, r#"{"latitude": 37.42}"#));
        assert!(run(&e, r#"{"latitude": 36.5}"#)); // rounds to 37
        assert!(!run(&e, r#"{"latitude": 38.1}"#));
        // missing field -> function of missing -> Missing -> comparison false.
        assert!(!run(&e, r#"{"other": 1}"#));
        // nested + two-const function: mathAdd(mathAbs(x), 1) == 4
        let e2 = Expr::compare(
            CompareOp::Equals,
            Expr::Func(Func {
                name: "mathAdd".into(),
                args: vec![
                    Expr::Func(Func {
                        name: "mathAbs".into(),
                        args: vec![field(&["x"])],
                    }),
                    Expr::Value(Literal::Int(1)),
                ],
            }),
            Expr::Value(Literal::Int(4)),
        );
        assert!(run(&e2, r#"{"x": -3}"#));
        assert!(!run(&e2, r#"{"x": 2}"#));
    }

    #[test]
    fn root_cross_field_comparison() {
        // $doc.a == $doc.b, deferred to an after-node; order-independent.
        let e = Expr::compare(CompareOp::Equals, field(&["a"]), field(&["b"]));
        assert!(run(&e, r#"{"a": 5, "b": 5}"#));
        assert!(run(&e, r#"{"b": 5, "a": 5}"#)); // field order reversed
        assert!(!run(&e, r#"{"a": 5, "b": 6}"#));
        // one side missing -> false; a < b with a missing -> false.
        assert!(!run(&e, r#"{"a": 5}"#));
        // a != b with b missing -> false: there is no value to differ from.
        assert!(!run(
            &Expr::compare(CompareOp::NotEquals, field(&["a"]), field(&["b"])),
            r#"{"a": 5}"#
        ));
    }

    /// `EXISTS` and `matches` applied to a field from an **enclosing** scope, inside a loop body.
    ///
    /// Both operators take their operand as a `DataRef`, so an outer field is stored in a slot
    /// exactly as a comparison's would be, and the op is attached to the current scope's node,
    /// which is visited once per element and by which time the slot is filled. A plain
    /// comparison on the same outer field has always compiled; these two must agree with it.
    ///
    /// Run on every backend, since each is separately compiled code.
    #[test]
    fn exists_and_matches_on_an_enclosing_scope_field() {
        // The loop variable is 1; `field(&[..])` is rooted at the document (variable 0), so
        // referring to it from inside the body is the cross-scope case.
        let any_over_tags = |body: Expr| Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(body),
        };
        let exists_name = || Expr::Exists(Box::new(field(&["name"])));

        assert!(run_all_backends(
            &any_over_tags(exists_name()),
            r#"{"name":"a","tags":[1]}"#
        ));
        // Order-independent: the outer field may appear *after* the array it is read from.
        assert!(run_all_backends(
            &any_over_tags(exists_name()),
            r#"{"tags":[1],"name":"a"}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(exists_name()),
            r#"{"tags":[1]}"#
        ));
        // An absent outer field makes EXISTS definitely false, so NOT EXISTS is true — the
        // carve-out that keeps absence selectable still holds across scopes.
        assert!(run_all_backends(
            &any_over_tags(Expr::Not(Box::new(exists_name()))),
            r#"{"tags":[1]}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(Expr::Not(Box::new(exists_name()))),
            r#"{"tags":[1],"name":"a"}"#
        ));
        // No elements: nothing to quantify over, so the body never runs.
        assert!(!run_all_backends(
            &any_over_tags(exists_name()),
            r#"{"name":"a","tags":[]}"#
        ));
        // Alongside a reference to the loop variable, which still resolves per element.
        let with_elem = Expr::And(vec![
            exists_name(),
            Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::Int(1)),
            ),
        ]);
        assert!(run_all_backends(
            &any_over_tags(with_elem.clone()),
            r#"{"tags":[2,1],"name":"a"}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(with_elem),
            r#"{"tags":[2,3],"name":"a"}"#
        ));

        // A nested path on the outer field.
        assert!(run_all_backends(
            &any_over_tags(Expr::Exists(Box::new(field(&["name", "first"])))),
            r#"{"name":{"first":"b"},"tags":[1]}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(Expr::Exists(Box::new(field(&["name", "first"])))),
            r#"{"name":{"last":"b"},"tags":[1]}"#
        ));

        // `matches` takes the same route. Absent is `Unknown` here rather than false, since a
        // pattern match against a value that is not there has no answer.
        let regex_on_name = || Expr::Matches {
            lhs: Box::new(field(&["name"])),
            pattern: Box::new(Expr::Value(Literal::String("br".into()))),
        };
        assert!(run_all_backends(
            &any_over_tags(regex_on_name()),
            r#"{"name":"brett","tags":[1]}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(regex_on_name()),
            r#"{"name":"zed","tags":[1]}"#
        ));
        assert!(!run_all_backends(
            &any_over_tags(regex_on_name()),
            r#"{"tags":[1]}"#
        ));
        // …and that `Unknown` is not a `False` in disguise: negating it must still not match. This
        // is the only place the difference is observable, because a slot is the only way `matches`
        // ever sees a missing operand — a local field that is absent never runs the op at all.
        assert!(!run_all_backends(
            &any_over_tags(Expr::Not(Box::new(regex_on_name()))),
            r#"{"tags":[1]}"#
        ));
        // With the field present the negation does invert, so the assertion above is not vacuous.
        assert!(run_all_backends(
            &any_over_tags(Expr::Not(Box::new(regex_on_name()))),
            r#"{"name":"zed","tags":[1]}"#
        ));
    }

    /// The same, two scopes deep: an inner loop body reading either the document or the *middle*
    /// scope. Exercises the compiler deferring both loops far enough out.
    #[test]
    fn exists_on_an_enclosing_scope_from_a_nested_loop() {
        // ANY a IN xs SATISFIES (ANY b IN a.ys SATISFIES EXISTS(<outer>) END) END
        let nested = |outer: Expr| Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["xs"])),
            sub_expr: Box::new(Expr::Loop {
                loop_type: LoopType::Any,
                var: 2,
                in_expr: Box::new(Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("ys".into())],
                })),
                sub_expr: Box::new(Expr::Exists(Box::new(outer))),
            }),
        };

        // Reading the document scope, two levels out.
        assert!(run_all_backends(
            &nested(field(&["name"])),
            r#"{"name":"a","xs":[{"ys":[1]}]}"#
        ));
        assert!(!run_all_backends(
            &nested(field(&["name"])),
            r#"{"xs":[{"ys":[1]}]}"#
        ));

        // Reading the *middle* scope (the outer loop's element), not the document.
        let middle = || {
            Expr::Field(Field {
                root: 1,
                path: vec![PathComponent::Key("k".into())],
            })
        };
        assert!(run_all_backends(
            &nested(middle()),
            r#"{"xs":[{"k":1,"ys":[1]}]}"#
        ));
        assert!(!run_all_backends(
            &nested(middle()),
            r#"{"xs":[{"ys":[1]}]}"#
        ));
        // Per-element: only the second outer element has `k`, and only it should satisfy.
        assert!(run_all_backends(
            &nested(middle()),
            r#"{"xs":[{"ys":[1]},{"k":1,"ys":[1]}]}"#
        ));
    }

    #[test]
    fn cross_scope_loop() {
        // ANY f IN friends SATISFIES f.id == $doc.index  (loop body reads an outer field)
        let e = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["friends"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("id".into())],
                }),
                field(&["index"]), // root = document
            )),
        };
        assert!(run(
            &e,
            r#"{"index": 2, "friends": [{"id": 1}, {"id": 2}]}"#
        ));
        assert!(!run(
            &e,
            r#"{"index": 5, "friends": [{"id": 1}, {"id": 2}]}"#
        ));
        // Order-independent: the outer field appears *after* the array.
        assert!(run(&e, r#"{"friends": [{"id": 2}], "index": 2}"#));
        // Missing outer field -> comparison has a missing operand -> no element matches.
        assert!(!run(&e, r#"{"friends": [{"id": 2}]}"#));
        // Missing array -> loop does not apply -> false.
        assert!(!run(&e, r#"{"index": 2}"#));
    }

    /// A field path with an array index, e.g. `$doc.a[1].b`.
    fn indexed_field(parts: &[(&str, Option<usize>)]) -> Expr {
        let mut path = Vec::new();
        for (key, index) in parts {
            if !key.is_empty() {
                path.push(PathComponent::Key((*key).to_owned()));
            }
            if let Some(i) = index {
                path.push(PathComponent::Index(*i));
            }
        }
        Expr::Field(Field::root(path))
    }

    #[test]
    fn indexed_array_element_paths() {
        // a[0] == 1
        let e = |i, v: i64| {
            Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(i))]),
                Expr::Value(Literal::Int(v)),
            )
        };
        assert!(run(&e(0, 1), r#"{"a": [1, 2]}"#));
        assert!(!run(&e(0, 2), r#"{"a": [1, 2]}"#));
        assert!(run(&e(1, 2), r#"{"a": [1, 2]}"#));
        // Out of range / not an array / absent -> the field is missing.
        assert!(!run(&e(5, 1), r#"{"a": [1, 2]}"#));
        assert!(!run(&e(0, 1), r#"{"a": {"0": 1}}"#));
        assert!(!run(&e(0, 1), r#"{"a": 7}"#));
        assert!(!run(&e(0, 1), r#"{"b": [1]}"#));
        // An object key spelled "[0]" is *not* element 0 (and vice versa).
        assert!(!run(&e(0, 1), r#"{"a": {"[0]": 1}}"#));
        assert!(run(
            &Expr::compare(
                CompareOp::Equals,
                field(&["a", "[0]"]),
                Expr::Value(Literal::Int(1))
            ),
            r#"{"a": {"[0]": 1}}"#
        ));

        // Paths through and past an index: a[1].b, a[0][1], nested objects in arrays.
        assert!(run(
            &Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(1)), ("b", None)]),
                Expr::Value(Literal::Int(9))
            ),
            r#"{"a": [{"b": 1}, {"b": 9}]}"#
        ));
        assert!(run(
            &Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(0)), ("", Some(1))]),
                Expr::Value(Literal::Int(4))
            ),
            r#"{"a": [[3, 4], [5]]}"#
        ));

        // Several indices of the same array in one expression, in either order.
        let both = Expr::And(vec![
            Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(2))]),
                Expr::Value(Literal::Int(30)),
            ),
            Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(0))]),
                Expr::Value(Literal::Int(10)),
            ),
        ]);
        assert!(run(&both, r#"{"a": [10, 20, 30, 40]}"#));
        assert!(!run(&both, r#"{"a": [10, 20, 31, 40]}"#));

        // Indexed access coexists with a loop over the same array, and with exists.
        let mixed = Expr::And(vec![
            Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(0))]),
                Expr::Value(Literal::Int(1)),
            ),
            Expr::Loop {
                loop_type: LoopType::Any,
                var: 1,
                in_expr: Box::new(field(&["a"])),
                sub_expr: Box::new(Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field {
                        root: 1,
                        path: vec![],
                    }),
                    Expr::Value(Literal::Int(3)),
                )),
            },
        ]);
        assert!(run(&mixed, r#"{"a": [1, 2, 3]}"#));
        assert!(!run(&mixed, r#"{"a": [1, 2, 4]}"#));
        assert!(run(
            &Expr::Exists(Box::new(indexed_field(&[("a", Some(1))]))),
            r#"{"a": [1, 2]}"#
        ));
        assert!(!run(
            &Expr::Exists(Box::new(indexed_field(&[("a", Some(2))]))),
            r#"{"a": [1, 2]}"#
        ));

        // Cross-field comparison between two elements (deferred via slots).
        assert!(run(
            &Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(0))]),
                indexed_field(&[("a", Some(1))])
            ),
            r#"{"a": [5, 5]}"#
        ));
        assert!(!run(
            &Expr::compare(
                CompareOp::Equals,
                indexed_field(&[("a", Some(0))]),
                indexed_field(&[("a", Some(1))])
            ),
            r#"{"a": [5, 6]}"#
        ));
    }

    #[test]
    fn projects_indexed_paths() {
        let doc: &[u8] = br#"{"a": [10, {"b": "x"}], "c": [[1, 2]]}"#;
        let projection = Projection::new()
            .field([PathComponent::Index(0)]) // not an array at the root -> absent
            .field([PathComponent::Key("a".into()), PathComponent::Index(0)])
            .field([
                PathComponent::Key("a".into()),
                PathComponent::Index(1),
                PathComponent::Key("b".into()),
            ])
            .field([PathComponent::Key("a".into()), PathComponent::Index(9)]);
        let def = compile(&[], &projection, &DefaultCollation).unwrap();
        let mut m = FastMatcher::new(&def);
        let p = m.matches(doc).unwrap();
        assert!(p.projected(0).is_none());
        assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"10"));
        assert_eq!(
            p.projected(2)
                .unwrap()
                .as_str()
                .unwrap()
                .to_decoded_bytes()
                .as_ref(),
            b"x"
        );
        assert!(p.projected(3).is_none());
        assert!(p
            .projected_by_path(&[PathComponent::Key("a".into()), PathComponent::Index(0)])
            .is_some());
    }

    #[test]
    fn loop_body_slots_do_not_leak_between_elements() {
        // A deferred (cross-field) comparison in a loop body stores each field in a slot.
        // Slots live for the whole document, so every element must start clean: otherwise an
        // element missing a field reads the previous element's value.
        let mk = |lt| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["arr"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("x".into())],
                }),
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("y".into())],
                }),
            )),
        };

        // ANY: element 0 does not match (1 != 2) but leaves y = 2 behind; element 1 has no
        // `y` at all, so its comparison has a missing operand and cannot match.
        assert!(!run(
            &mk(LoopType::Any),
            r#"{"arr": [{"x": 1, "y": 2}, {"x": 2}]}"#
        ));
        // Same shape, but element 1 genuinely matches.
        assert!(run(
            &mk(LoopType::Any),
            r#"{"arr": [{"x": 1, "y": 2}, {"x": 2, "y": 2}]}"#
        ));
        // EVERY: element 1 is missing `y`, so it fails and the loop is false — it must not
        // inherit element 0's y = 7.
        assert!(!run(
            &mk(LoopType::Every),
            r#"{"arr": [{"x": 7, "y": 7}, {"x": 7}]}"#
        ));
        assert!(run(
            &mk(LoopType::Every),
            r#"{"arr": [{"x": 7, "y": 7}, {"x": 8, "y": 8}]}"#
        ));
        // An absent field in the *first* element must not read anything either.
        assert!(!run(&mk(LoopType::Any), r#"{"arr": [{"x": 1}, {"y": 1}]}"#));

        // A cross-scope loop (deferred to the root after-node) clears its body slots too,
        // while the *outer* field it references must survive that clearing.
        let cross = |lt| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["arr"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("x".into())],
                }),
                field(&["want"]),
            )),
        };
        assert!(run(
            &cross(LoopType::Any),
            r#"{"arr": [{"x": 1}, {"x": 9}], "want": 9}"#
        ));
        // EVERY: element 1 has no `x`, so it fails — it must not reuse element 0's x = 5
        // (which would equal `want` and wrongly satisfy every element).
        assert!(!run(
            &cross(LoopType::Every),
            r#"{"arr": [{"x": 5}, {"z": 0}], "want": 5}"#
        ));
        assert!(run(
            &cross(LoopType::Every),
            r#"{"arr": [{"x": 5}, {"x": 5}], "want": 5}"#
        ));
    }

    #[test]
    fn nested_loops_reading_outer_scopes() {
        // ANY o IN outer SATISFIES (ANY i IN o.items SATISFIES i.v == <ref> END) END
        // where <ref> is a field of an enclosing scope — two loops deep.
        let nested = |reference: Expr| Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["outer"])),
            sub_expr: Box::new(Expr::Loop {
                loop_type: LoopType::Any,
                var: 2,
                in_expr: Box::new(Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("items".into())],
                })),
                sub_expr: Box::new(Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field {
                        root: 2,
                        path: vec![PathComponent::Key("v".into())],
                    }),
                    reference,
                )),
            }),
        };

        // (a) the inner body reads the *document* root: two scopes out.
        let to_doc = nested(field(&["want"]));
        assert!(run(
            &to_doc,
            r#"{"want": 7, "outer": [{"items": [{"v": 1}]}, {"items": [{"v": 7}]}]}"#
        ));
        assert!(!run(
            &to_doc,
            r#"{"want": 9, "outer": [{"items": [{"v": 1}]}, {"items": [{"v": 7}]}]}"#
        ));
        // Order-independent: the outer field appears after the arrays.
        assert!(run(
            &to_doc,
            r#"{"outer": [{"items": [{"v": 7}]}], "want": 7}"#
        ));
        // Missing outer field -> no element can match.
        assert!(!run(&to_doc, r#"{"outer": [{"items": [{"v": 7}]}]}"#));

        // (b) the inner body reads the *middle* scope (the outer loop's element).
        let to_middle = nested(Expr::Field(Field {
            root: 1,
            path: vec![PathComponent::Key("want".into())],
        }));
        assert!(run(
            &to_middle,
            r#"{"outer": [{"want": 1, "items": [{"v": 2}]}, {"want": 3, "items": [{"v": 3}]}]}"#
        ));
        assert!(!run(
            &to_middle,
            r#"{"outer": [{"want": 1, "items": [{"v": 2}]}, {"want": 3, "items": [{"v": 4}]}]}"#
        ));
        // Per-element and order-independent: the middle field may appear after the inner
        // array...
        assert!(run(
            &to_middle,
            r#"{"outer": [{"items": [{"v": 5}], "want": 5}]}"#
        ));
        // ...and must not leak into an element that lacks it (element 1 sets want = 5 but
        // does not match; element 2 has v = 5 and no `want`, so it must not match either).
        assert!(!run(
            &to_middle,
            r#"{"outer": [{"want": 5, "items": [{"v": 6}]}, {"items": [{"v": 5}]}]}"#
        ));

        // (c) three loops deep, innermost reading the document root.
        let three = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["l1"])),
            sub_expr: Box::new(Expr::Loop {
                loop_type: LoopType::Any,
                var: 2,
                in_expr: Box::new(Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("l2".into())],
                })),
                sub_expr: Box::new(Expr::Loop {
                    loop_type: LoopType::Any,
                    var: 3,
                    in_expr: Box::new(Expr::Field(Field {
                        root: 2,
                        path: vec![PathComponent::Key("l3".into())],
                    })),
                    sub_expr: Box::new(Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field {
                            root: 3,
                            path: vec![],
                        }),
                        field(&["want"]),
                    )),
                }),
            }),
        };
        assert!(run(
            &three,
            r#"{"l1": [{"l2": [{"l3": [1, 2]}]}], "want": 2}"#
        ));
        assert!(!run(
            &three,
            r#"{"l1": [{"l2": [{"l3": [1, 2]}]}], "want": 3}"#
        ));
        assert!(run(
            &three,
            r#"{"want": 1, "l1": [{"l2": [{"l3": [9]}, {"l3": [1]}]}]}"#
        ));

        // An unknown loop variable is still rejected at compile time.
        let bad = Expr::compare(
            CompareOp::Equals,
            Expr::Field(Field {
                root: 42,
                path: vec![],
            }),
            Expr::Value(Literal::Int(1)),
        );
        assert!(matches!(
            compile(&[bad], &Projection::new(), &DefaultCollation),
            Err(crate::compile::CompileError::UnknownVariable(42))
        ));
    }

    #[test]
    fn multi_expression() {
        // Three independent expressions matched in one pass.
        let exprs = vec![
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["eyeColor"]),
                Expr::Value(Literal::String("brown".into())),
            ),
            Expr::Exists(Box::new(field(&["missing"]))),
        ];
        let def = compile(&exprs, &Projection::new(), &DefaultCollation).unwrap();
        assert_eq!(def.num_expressions(), 3);
        let mut m = FastMatcher::new(&def);

        {
            let o = m.matches(br#"{"age": 30, "eyeColor": "blue"}"#).unwrap();
            assert!(o.matched()); // overall = OR
            assert!(o.expression_matched(0)); // age < 50
            assert!(!o.expression_matched(1)); // eyeColor != brown
            assert!(!o.expression_matched(2)); // missing absent
        }
        {
            let o = m.matches(br#"{"age": 80, "eyeColor": "blue"}"#).unwrap();
            assert!(!o.matched()); // none match
            assert!(!o.expression_matched(0));
            assert!(!o.expression_matched(1));
            assert!(!o.expression_matched(2));
        }
        {
            let o = m.matches(br#"{"age": 80, "eyeColor": "brown"}"#).unwrap();
            assert!(o.matched());
            assert!(!o.expression_matched(0));
            assert!(o.expression_matched(1));
        }
    }

    // ---- field projection ---------------------------------------------------------------

    /// A path of plain object keys.
    fn key_path(keys: &[&str]) -> Vec<PathComponent> {
        keys.iter()
            .map(|k| PathComponent::Key((*k).into()))
            .collect()
    }

    /// Compile `exprs` with `paths` projected and run against `doc`, handing the outcome to
    /// `check` (which borrows the matcher, so it cannot outlive the match).
    fn projected<R>(
        exprs: &[Expr],
        paths: &[&[&str]],
        doc: &[u8],
        check: impl FnOnce(MatchOutcome<'_, '_>) -> R,
    ) -> R {
        let mut projection = Projection::new();
        for p in paths {
            projection.push(p.iter().copied());
        }
        let def = compile(exprs, &projection, &DefaultCollation).unwrap();
        let mut m = FastMatcher::new(&def);
        check(m.matches(doc).unwrap())
    }

    /// The decoded bytes of a captured string value.
    fn as_string(v: &FastVal<'_>) -> Vec<u8> {
        v.as_str().expect("a string").to_decoded_bytes().to_vec()
    }

    #[test]
    fn projects_scalars_and_containers() {
        let doc: &[u8] = br#"{"s": "hi", "i": -7, "f": 1.5, "b": true, "n": null,
                              "o": {"k": [1, 2]}, "a": [1, {"x": 2}]}"#;
        projected(
            &[],
            &[&["s"], &["i"], &["f"], &["b"], &["n"], &["o"], &["a"]],
            doc,
            |p| {
                assert_eq!(p.num_projections(), 7);
                assert_eq!(as_string(&p.projected(0).unwrap()), b"hi");
                // Numbers stay lazy: the raw literal bytes, parsed on demand.
                assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"-7"));
                assert!(matches!(p.projected(2).unwrap(), FastVal::FloatBytes(b) if b == b"1.5"));
                assert_eq!(p.projected(3).unwrap().as_bool(), Some(true));
                assert!(matches!(p.projected(4).unwrap(), FastVal::Null));
                // Containers come back as their exact raw document bytes.
                assert_eq!(
                    p.projected(5).unwrap().container_bytes().unwrap(),
                    br#"{"k": [1, 2]}"#
                );
                assert_eq!(
                    p.projected(6).unwrap().container_bytes().unwrap(),
                    br#"[1, {"x": 2}]"#
                );
            },
        );
    }

    #[test]
    fn projects_nested_paths_and_reports_absence() {
        let doc: &[u8] = br#"{"name": {"first": "Brett", "last": "Lawson"}, "arr": [{"k": 1}]}"#;
        projected(
            &[],
            &[
                &["name", "first"],
                &["name", "middle"], // absent sub-field
                &["nope"],           // absent field
                &["arr", "k"],       // path traverses an array: never captured
            ],
            doc,
            |p| {
                assert_eq!(as_string(&p.projected(0).unwrap()), b"Brett");
                assert!(p.is_projected_present(0));
                for i in 1..4 {
                    assert!(p.projected(i).is_none(), "index {i} should be absent");
                    assert!(!p.is_projected_present(i));
                }
                // Paths are reported back as requested.
                assert_eq!(p.projected_path(0), key_path(&["name", "first"]));
                assert_eq!(
                    as_string(&p.projected_by_path(&key_path(&["name", "first"])).unwrap()),
                    b"Brett"
                );
                assert!(p
                    .projected_by_path(&key_path(&["not", "projected"]))
                    .is_none());
                let all: Vec<_> = p.projections().map(|(_, v)| v.is_some()).collect();
                assert_eq!(all, [true, false, false, false]);
            },
        );
    }

    #[test]
    fn projected_strings_borrow_the_document() {
        // An unescaped string must be a borrow *into* the document, not a copy...
        let doc: &[u8] = br#"{"s": "plain", "e": "a\nb"}"#;
        projected(&[], &[&["s"], &["e"]], doc, |p| {
            let v = p.projected(0).unwrap();
            let borrowed = match v.as_str().unwrap() {
                FastStr::Unescaped(b) => *b,
                other => panic!("expected an unescaped borrow, got {other:?}"),
            };
            assert_eq!(borrowed, b"plain");
            let (base, end) = (doc.as_ptr() as usize, doc.as_ptr() as usize + doc.len());
            let at = borrowed.as_ptr() as usize;
            assert!(at >= base && at < end, "value does not point into the doc");

            // ...and an escaped one stays escaped until asked to decode.
            let e = p.projected(1).unwrap();
            assert!(matches!(e.as_str().unwrap(), FastStr::Escaped(b) if b == br"a\nb"));
            assert_eq!(as_string(&e), b"a\nb");
        });
    }

    #[test]
    fn projection_is_independent_of_the_match_result() {
        let expr = Expr::compare(
            CompareOp::Equals,
            field(&["age"]),
            Expr::Value(Literal::Int(99)),
        );
        // The document does not match, yet the projected field is still captured.
        projected(
            std::slice::from_ref(&expr),
            &[&["name"]],
            br#"{"name": "x", "age": 1}"#,
            |p| {
                assert!(!p.matched());
                assert_eq!(as_string(&p.projected(0).unwrap()), b"x");
            },
        );
        projected(&[expr], &[&["name"]], br#"{"name": "x", "age": 99}"#, |p| {
            assert!(p.matched());
            assert_eq!(as_string(&p.projected(0).unwrap()), b"x");
        });
    }

    #[test]
    fn projection_survives_what_would_have_been_a_short_circuit() {
        // `a == 1` is decided at the very first field; the projected fields appear later in
        // the document (one after a big skipped subtree) and must still be captured.
        let expr = Expr::compare(
            CompareOp::Equals,
            field(&["a"]),
            Expr::Value(Literal::Int(1)),
        );
        let doc: &[u8] = br#"{"a": 1, "noise": {"deep": [1, 2, {"x": "y"}]}, "z": "last",
                              "w": {"inner": 5}}"#;
        projected(&[expr], &[&["z"], &["w", "inner"]], doc, |p| {
            assert!(p.matched());
            assert_eq!(as_string(&p.projected(0).unwrap()), b"last");
            assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"5"));
        });
    }

    #[test]
    fn projection_alongside_cross_field_and_loop_expressions() {
        // A cross-field comparison already stores both fields; projecting them shares those
        // slots and must not disturb the (deferred) comparison.
        let cross = Expr::compare(CompareOp::Equals, field(&["a"]), field(&["b"]));
        projected(
            std::slice::from_ref(&cross),
            &[&["a"], &["b"]],
            br#"{"b": 5, "a": 5}"#,
            |p| {
                assert!(p.matched());
                assert!(matches!(p.projected(0).unwrap(), FastVal::IntBytes(b) if b == b"5"));
                assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"5"));
            },
        );
        projected(&[cross], &[&["a"], &["b"]], br#"{"a": 5, "b": 6}"#, |p| {
            assert!(!p.matched());
            assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"6"));
        });

        // A cross-scope loop (deferred to the root's after-node) plus projection.
        let loop_expr = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["friends"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![PathComponent::Key("id".into())],
                }),
                field(&["index"]),
            )),
        };
        projected(
            &[loop_expr],
            &[&["index"], &["friends"]],
            br#"{"friends": [{"id": 2}], "index": 2}"#,
            |p| {
                assert!(p.matched());
                assert!(matches!(p.projected(0).unwrap(), FastVal::IntBytes(b) if b == b"2"));
                assert_eq!(
                    p.projected(1).unwrap().container_bytes().unwrap(),
                    br#"[{"id": 2}]"#
                );
            },
        );
    }

    #[test]
    fn projects_the_whole_document_and_escaped_keys() {
        let doc: &[u8] = br#"{"a\tb": 1}"#;
        projected(&[], &[&[], &["a\tb"]], doc, |p| {
            // The empty path is the document itself.
            assert_eq!(p.projected(0).unwrap().container_bytes().unwrap(), doc);
            // Keys are matched by decoded content, like everywhere else in the engine.
            assert!(matches!(p.projected(1).unwrap(), FastVal::IntBytes(b) if b == b"1"));
        });
    }

    #[test]
    fn projection_state_resets_between_documents() {
        let def = compile(
            &[Expr::Exists(Box::new(field(&["a"])))],
            &Projection::new().field(["a"]).field(["b"]),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);

        {
            let p = m.matches(br#"{"a": 1, "b": 2}"#).unwrap();
            assert!(p.matched());
            assert!(p.is_projected_present(0) && p.is_projected_present(1));
        }
        {
            // `b` is gone this time: its slot must not carry over from the previous document.
            let p = m.matches(br#"{"a": 3}"#).unwrap();
            assert!(p.matched());
            assert!(matches!(p.projected(0).unwrap(), FastVal::IntBytes(b) if b == b"3"));
            assert!(p.projected(1).is_none());
        }
        {
            let p = m.matches(br#"{"c": 4}"#).unwrap();
            assert!(!p.matched());
            assert!(p.projected(0).is_none() && p.projected(1).is_none());
        }
    }

    #[test]
    fn projection_with_multiple_expressions() {
        let exprs = vec![
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            ),
            Expr::Exists(Box::new(field(&["nope"]))),
        ];
        let def = compile(&exprs, &Projection::new().field(["age"]), &DefaultCollation).unwrap();
        let mut m = FastMatcher::new(&def);
        {
            let p = m.matches(br#"{"age": 30}"#).unwrap();
            assert!(p.matched());
            assert!(p.expression_matched(0) && !p.expression_matched(1));
            assert!(matches!(p.projected(0).unwrap(), FastVal::IntBytes(b) if b == b"30"));
        }

        // Projecting a field changes neither the overall nor the per-expression results.
        let plain = compile(&exprs, &Projection::new(), &DefaultCollation).unwrap();
        let mut pm = FastMatcher::new(&plain);
        let o = pm.matches(br#"{"age": 30}"#).unwrap();
        assert!(o.matched());
        assert!(o.expression_matched(0) && !o.expression_matched(1));
    }

    #[test]
    fn depth_limits_are_enforced() {
        // A document nested past MAX_DEPTH is rejected rather than scanned.
        let e = Expr::compare(
            CompareOp::Equals,
            field(&["a"]),
            Expr::Value(Literal::Int(1)),
        );
        let def = compile(&[e], &Projection::new(), &DefaultCollation).unwrap();
        let mut m = FastMatcher::new(&def);

        // The deep field comes *first*, so the scan reaches it before `a` decides the match
        // (otherwise short-circuiting would end the scan and the nesting would never be seen).
        let deep = |n: usize| {
            let mut s = String::from("{\"b\": ");
            s.push_str(&"[".repeat(n));
            s.push_str(&"]".repeat(n));
            s.push_str(", \"a\": 1}");
            s.into_bytes()
        };
        // Comfortably inside the limit: scanned normally (the `a` op still matches).
        assert!(m.matches(&deep(64)).unwrap().matched());
        // Past the limit: an explicit error, not a stack overflow or a wrong answer.
        assert!(matches!(
            m.matches(&deep(MAX_DEPTH + 8)),
            Err(MatchError::TooDeep)
        ));
        // Pinned to the byte, not just "somewhere past the limit": `b` is a field no
        // expression names, so this is the *skip* path's depth budget, and a budget that is
        // off by one is invisible to a test that overshoots by eight.
        assert!(m.matches(&deep(MAX_DEPTH - 2)).unwrap().matched());
        assert!(matches!(
            m.matches(&deep(MAX_DEPTH - 1)),
            Err(MatchError::TooDeep)
        ));

        // An expression nested past MAX_EXPR_DEPTH is rejected at compile time. Build it
        // iteratively; `depth()` is iterative too, so measuring it is safe.
        let mut nested = Expr::True;
        for _ in 0..crate::compile::MAX_EXPR_DEPTH + 4 {
            nested = Expr::Not(Box::new(nested));
        }
        assert!(nested.depth() > crate::compile::MAX_EXPR_DEPTH);
        assert!(matches!(
            compile(&[nested], &Projection::new(), &DefaultCollation),
            Err(crate::compile::CompileError::TooDeep)
        ));

        // A merely deep-ish expression still compiles.
        let mut ok = Expr::True;
        for _ in 0..32 {
            ok = Expr::Not(Box::new(ok));
        }
        assert!(compile(&[ok], &Projection::new(), &DefaultCollation).is_ok());
    }


    /// A repeated object key resolves to its **first** occurrence, where gojsonsm and
    /// `serde_json` resolve to the last.
    ///
    /// The cause is the field-completeness exit in [`FastMatcher::match_object`]: once every
    /// key an exec node names has been supplied, the rest of the object is skipped in bulk, so
    /// a second copy of an already-seen key is never read. A repeated key is **not valid input
    /// to this engine** — the same licence the skip paths already take — so this pins the
    /// behaviour rather than promising it. It has to be pinned here because **no sweep can
    /// reach it**:
    /// all three differential generators build documents as `serde_json::Value`, whose object
    /// map cannot represent a duplicate key at all. Here it is the *data model* that forbids
    /// the case, not the encoding.
    #[test]
    fn duplicate_keys_take_the_first_occurrence() {
        let eq = |v: i64| {
            Expr::compare(CompareOp::Equals, field(&["a"]), Expr::Value(Literal::Int(v)))
        };
        assert!(run(&eq(1), r#"{"a":1,"a":2}"#), "the first occurrence decides");
        assert!(!run(&eq(2), r#"{"a":1,"a":2}"#), "the last occurrence is never read");

        // Only *distinct* keys complete the set, so a repeat cannot end the scan early and
        // hide a key the expression still needs.
        let a_and_b = Expr::And(vec![
            Expr::compare(CompareOp::Equals, field(&["a"]), Expr::Value(Literal::Int(1))),
            Expr::compare(CompareOp::Equals, field(&["b"]), Expr::Value(Literal::Int(9))),
        ]);
        assert!(run(&a_and_b, r#"{"a":1,"a":1,"b":9}"#), "b is still reached past a repeat");
        assert!(!run(&a_and_b, r#"{"a":1,"a":1,"b":8}"#));

        // A field the expression does not name never counts toward completeness.
        assert!(run(&eq(1), r#"{"z":0,"a":1,"z":0,"a":2}"#));
    }
    #[test]
    fn irrelevant_fields_are_skipped() {
        // Deeply nested irrelevant structure must be skipped without affecting the match.
        let e = Expr::compare(
            CompareOp::Equals,
            field(&["want"]),
            Expr::Value(Literal::Int(7)),
        );
        assert!(run(
            &e,
            r#"{"noise": {"a": [1,2,{"b": "x"}], "c": null}, "want": 7, "more": [true,false]}"#
        ));
    }

    /// [`take_str_value`] reads a string element from raw bytes instead of tokenizing it,
    /// so every shape it can meet has to agree with the token it replaces.
    ///
    /// The whitespace cases are the reason this test is written by hand. All three
    /// differential sweeps build documents through `serde_json::to_vec`, which never emits
    /// whitespace before an array element, so no generated document reaches the `skip_ws`
    /// branch at all — the encoding bounds what is reachable, not just the data model.
    ///
    /// Note what that does and does not buy. Deleting the `skip_ws` branch still passes this
    /// test, and *should*: declining is always safe, so without it a spaced element merely
    /// falls back to the tokenizer and gets the same answer. The branch is a pure
    /// optimisation and no test can witness its removal. What these cases pin is the
    /// *behaviour* on spaced input, which a wrong `skip_ws` result would break.
    #[test]
    fn fused_string_elements_agree_with_the_tokenizer() {
        let any_tag_is_b = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("b".into())),
            )),
        };

        // Compact — the shape the fast path exists for.
        assert!(run(&any_tag_is_b, r#"{"tags":["a","b","c"]}"#));
        assert!(!run(&any_tag_is_b, r#"{"tags":["a","c"]}"#));

        // Whitespace before an element, before the closing bracket, and around the
        // delimiters: legal JSON that no serializer here produces.
        assert!(run(&any_tag_is_b, "{\"tags\":[ \"a\" , \"b\" ]}"));
        assert!(run(&any_tag_is_b, "{\"tags\":[\n\t\"b\"\r\n]}"));
        assert!(!run(&any_tag_is_b, "{\"tags\":[ \"a\" , \"c\" ]}"));
        // Whitespace only, then a non-string: must decline, not mis-read.
        assert!(!run(&any_tag_is_b, "{\"tags\":[ 1 , 2 ]}"));

        // An escape makes the raw bytes unequal to the logical string, so the fast path must
        // decline and let the tokenizer decode. `b` is `b`.
        assert!(run(&any_tag_is_b, r#"{"tags":["a","b"]}"#));
        assert!(run(&any_tag_is_b, r#"{"tags":["b"]}"#));
        // ...and an escaped string that is *not* the target still must not match.
        assert!(!run(&any_tag_is_b, r#"{"tags":["c"]}"#));
        // An escape in an earlier element must not disturb a later plain one.
        assert!(run(&any_tag_is_b, r#"{"tags":["\t","b"]}"#));

        // Mixed element types in one array: the fast path settles some and declines others.
        assert!(run(&any_tag_is_b, r#"{"tags":[1,"a",true,null,"b"]}"#));
        assert!(!run(&any_tag_is_b, r#"{"tags":[1,"a",true,null,{"x":"b"}]}"#));

        // Empty and single-element arrays, where the loop's first iteration meets `]`.
        assert!(!run(&any_tag_is_b, r#"{"tags":[]}"#));
        assert!(!run(&any_tag_is_b, "{\"tags\":[ ]}"));
        assert!(run(&any_tag_is_b, r#"{"tags":["b"]}"#));
        // The empty string is a string: a zero-length span must not be read as a decline.
        assert!(!run(&any_tag_is_b, r#"{"tags":[""]}"#));

        // EVERY takes the same path and cannot short-circuit on the first element.
        let every_tag_is_b = Expr::Loop {
            loop_type: LoopType::Every,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("b".into())),
            )),
        };
        assert!(run(&every_tag_is_b, r#"{"tags":["b","b"]}"#));
        assert!(run(&every_tag_is_b, "{\"tags\":[ \"b\" , \"b\" ]}"));
        assert!(!run(&every_tag_is_b, r#"{"tags":["b","c"]}"#));
    }

    /// A constant is stored decoded, so a literal holding a character JSON spells with an
    /// escape is *shorter* than the document bytes it must match, and the two are equal only
    /// if something decodes. `borrow_const` viewing `FastStr::Owned` as `FastStr::Unescaped`
    /// is what asserts the constant is decoded, and it would be wrong if the constant were
    /// ever the escaped form — so drive both directions across every escape shape.
    #[test]
    fn constants_holding_escapable_characters_compare_decoded() {
        // (logical string, how a document spells it)
        let cases: &[(&str, &str)] = &[
            ("a\"b", r#""a\"b""#),                 // quote
            ("a\\b", r#""a\\b""#),                 // backslash
            ("a\nb", r#""a\nb""#),                 // newline, as a short escape
            ("a\tb", r#""a\tb""#),                 // tab
            ("a\u{1}b", r#""a\u0001b""#),          // control char, only spellable as \u
            ("é", r#""\u00e9""#),                // non-ASCII written as an escape
            ("é", "\"é\""),                    // ...and spelled literally
            ("😀", r#""\ud83d\ude00""#),         // surrogate pair
            ("😀", "\"😀\""),                  // ...and spelled literally
            ("plain", r#""plain""#),               // the unescaped control case
        ];
        for (logical, encoded) in cases {
            let e = Expr::compare(
                CompareOp::Equals,
                field(&["v"]),
                Expr::Value(Literal::String((*logical).into())),
            );
            let doc = format!(r#"{{"v":{encoded}}}"#);
            assert!(run(&e, &doc), "expected {logical:?} == {encoded} in {doc}");

            // And the same constant must *not* match a different string.
            let other = r#"{"v":"definitely-not-it"}"#;
            assert!(!run(&e, other), "{logical:?} wrongly matched {other}");
        }

        // The above all reach `eval_op`'s `operand_ref` path, which hands the stored constant
        // over as-is. `borrow_const` — which re-views that `Owned` string as `Unescaped` — is
        // reached only from `resolve_ref`, so it needs its own cases or a mutation making it
        // view constants as *escaped* survives. A backslash is what separates the two
        // readings: as decoded bytes `a\b` is three characters, but read as escaped it is
        // `a` followed by a backspace.
        let any_tag_backslash = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("a\\b".into())),
            )),
        };
        assert!(run(&any_tag_backslash, r#"{"tags":["x","a\\b"]}"#));
        // `"a\b"` in a document is `a`+backspace, which is *not* the constant `a\b`.
        assert!(!run(&any_tag_backslash, r#"{"tags":["x","a\b"]}"#));

        // `resolve_ref` runs only when the *opposite* operand must be built — when it is a
        // slot or a function result. A field read from the enclosing scope inside a loop body
        // is a slot, so this is the shape that reaches it, and nothing above does:
        // `eval_op`'s `operand_ref` path hands the stored constant over untouched.
        let outer_eq_backslash = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                field(&["name"]),
                Expr::Value(Literal::String("a\\b".into())),
            )),
        };
        assert!(run(&outer_eq_backslash, r#"{"name":"a\\b","tags":[1]}"#));
        // `"a\b"` in a document is `a`+backspace, which is not the constant `a\b`.
        assert!(!run(&outer_eq_backslash, r#"{"name":"a\b","tags":[1]}"#));

        // Ordering, not just equality: a decoded constant must order against an escaped
        // document string by logical content. "a\nb" is "a\u{a}b"; "a b" is "a\u{20}b".
        let gt = Expr::compare(
            CompareOp::GreaterThan,
            field(&["v"]),
            Expr::Value(Literal::String("a\nb".into())),
        );
        assert!(run(&gt, r#"{"v":"a b"}"#), "0x20 > 0x0a");
        assert!(!run(&gt, r#"{"v":"a\nb"}"#), "equal is not greater");
    }

    /// An unterminated or control-character string must still be the tokenizer's error, not
    /// something the fused read swallows: it declines, and the tokenizer reports as before.
    #[test]
    fn fused_string_elements_leave_malformed_input_to_the_tokenizer() {
        let any_tag_is_b = Expr::Loop {
            loop_type: LoopType::Any,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("b".into())),
            )),
        };
        let def = compile(
            std::slice::from_ref(&any_tag_is_b),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();

        for doc in [
            r#"{"tags":["a","unterminated]}"#,
            "{\"tags\":[\"a\",\"has\ttab\"]}",
            r#"{"tags":["a""#,
        ] {
            let mut m = FastMatcher::new(&def);
            assert!(
                m.matches(doc.as_bytes()).is_err(),
                "expected a tokenizer error for {doc:?}"
            );
        }
    }
}
