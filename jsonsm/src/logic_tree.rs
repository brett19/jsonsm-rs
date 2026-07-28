//! The boolean **logic tree**: the compiled boolean structure of an expression, plus the
//! per-match execution state that resolves it as operations report their results.
//!
//! The tree is a flat array laid out in pre-order, so every node's subtree is a contiguous
//! range — which lets a loop cheaply reset the subtree beneath its loop node between array
//! iterations. Index `0` is the root and is never any node's child, so it doubles as the
//! "no child" sentinel.
//!
//! Each leaf corresponds to one operation (a comparison / existence / match). As ops run,
//! [`LogicTreeState::mark`] records a leaf's result and propagates it upward with
//! short-circuiting (an `Or` with one true child resolves immediately, etc.), so matching
//! can stop as soon as the root is decided. A loop node's subtree is evaluated per array
//! element behind a *stall index* that prevents per-iteration results from propagating
//! above the loop until the loop's overall result is known.
//!
//! Node values are **three-valued**: besides `True`/`False` a node can be `Unknown`,
//! which is what a comparison yields when the field it names is absent. `Unknown` is a value,
//! not a gap — it combines by Kleene's tables and, decisively, `NOT Unknown` is `Unknown`, so
//! writing `!=` or `NOT` around an absent field cannot turn it into a match. Only at the root
//! do `Unknown` and `False` mean the same thing (no match).
//!
//! That is why a node's state distinguishes `Unset` from `Unknown`: the first says the scan has not
//! reached this node, the second that its answer is unknowable. Collapsing them would force
//! absence to be settled by a sweep after the document ends; keeping them apart lets the matcher
//! record a field's absence the moment its container closes, so the result propagates — and the
//! scan can stop — as early as the logic permits. [`LogicTreeState::resolve`] is then only a
//! backstop for whatever the scan never reached at all.

/// Index of a node within a [`LogicTree`]. `0` is the root and the "no child" sentinel.
pub type NodeIdx = usize;

/// The boolean role of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// A terminal: its value is set directly by an operation.
    Leaf,
    /// Short-circuiting logical OR of both children.
    Or,
    /// Logical AND of both children.
    And,
    /// Logical negation of the (single, left) child.
    Not,
    /// Non-short-circuiting OR — waits for *both* children before resolving. Used to merge
    /// multiple top-level expressions while tracking each independently.
    Neor,
    /// A loop node: its value is the loop's overall result (its single left child is the
    /// per-iteration sub-tree root).
    Loop,
}

impl NodeType {
    #[inline]
    fn has_left(self) -> bool {
        self != NodeType::Leaf
    }
    #[inline]
    fn has_right(self) -> bool {
        matches!(self, NodeType::Or | NodeType::And | NodeType::Neor)
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    node_type: NodeType,
    parent: NodeIdx,
    left: NodeIdx,
    right: NodeIdx,
}

/// An error found while validating a [`LogicTree`]'s structure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TreeError {
    #[error("node {0} has a parent link that does not match its position")]
    BadParent(NodeIdx),
    #[error("node {0} has a child index outside the tree")]
    ChildOutOfRange(NodeIdx),
    #[error("node {0} is missing a required child")]
    MissingChild(NodeIdx),
    #[error("node {0} has a child that should not be set")]
    UnexpectedChild(NodeIdx),
    #[error("the tree does not form a single connected structure")]
    NotConnected,
}

/// The compiled boolean structure. Build it with [`LogicTree::new`] + the `add`/`set`
/// methods (the compiler does this), then create per-match state with [`Self::new_state`].
#[derive(Debug, Clone, Default)]
pub struct LogicTree {
    nodes: Vec<Node>,
    /// One past the last index of each node's subtree, precomputed by [`Self::validate`].
    ///
    /// Subtrees are contiguous (nodes are appended in pre-order), which
    /// [`LogicTreeState::seal_node`] already relied on — but it *recomputed* the bound by
    /// walking the subtree, once per call, per loop element. Both that walk and
    /// `reset_node`'s recursion are a whole traversal to learn something fixed at compile
    /// time. Held here, the extent is a load, `reset_node` becomes a `fill` over a slice, and
    /// contiguity is asserted rather than assumed — see `subtrees_are_contiguous`.
    ends: Vec<NodeIdx>,
}

impl LogicTree {
    /// A new tree containing just the root leaf (index 0).
    pub fn new() -> Self {
        LogicTree {
            nodes: vec![Node {
                node_type: NodeType::Leaf,
                parent: 0,
                left: 0,
                right: 0,
            }],
            ends: Vec::new(),
        }
    }

    /// Append a new leaf whose parent is `parent`, returning its index. Nodes are added in
    /// pre-order by the compiler, keeping subtrees contiguous.
    pub fn add_child(&mut self, parent: NodeIdx) -> NodeIdx {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            node_type: NodeType::Leaf,
            parent,
            left: 0,
            right: 0,
        });
        idx
    }

    /// Precompute every node's subtree extent.
    ///
    /// Called from [`Self::validate`] rather than exposed, so a tree that has been checked is
    /// also a tree that is ready: there is no second step to forget, and no way to hold a
    /// validated tree whose extents are missing. Walks back to front so a node's children are
    /// always already done, which makes this one linear pass rather than a traversal per node.
    fn fill_extents(&mut self) {
        let n = self.nodes.len();
        let mut ends = vec![0; n];
        for idx in (0..n).rev() {
            let node = self.nodes[idx];
            let mut end = idx + 1;
            if node.node_type.has_left() {
                end = end.max(ends[node.left]);
            }
            if node.node_type.has_right() {
                end = end.max(ends[node.right]);
            }
            ends[idx] = end;
        }
        self.ends = ends;
        debug_assert!(
            self.subtrees_are_contiguous(),
            "logic-tree subtrees must be contiguous; reset_node fills a slice and seal_node \
             scans one, and both are silently wrong otherwise"
        );
    }

    /// As [`Self::subtrees_are_contiguous`], but fills the extents first — for checking a
    /// tree that deliberately was not validated.
    #[cfg(test)]
    fn subtrees_are_contiguous_unchecked(&mut self) -> bool {
        let n = self.nodes.len();
        let mut ends = vec![0; n];
        for idx in (0..n).rev() {
            let node = self.nodes[idx];
            let mut end = idx + 1;
            if node.node_type.has_left() {
                end = end.max(ends[node.left]);
            }
            if node.node_type.has_right() {
                end = end.max(ends[node.right]);
            }
            ends[idx] = end;
        }
        self.ends = ends;
        self.subtrees_are_contiguous()
    }

    /// Whether every subtree occupies the contiguous range `idx..ends[idx]`.
    ///
    /// This is the invariant `reset_node` and `seal_node` are built on, and **nothing in the
    /// builder API enforces it** — it holds because the compiler appends nodes in pre-order.
    /// Add children breadth-first instead and both operations quietly touch a sibling's state
    /// or miss their own, with no error anywhere. Hence the `debug_assert` in
    /// [`Self::fill_extents`], which checks it on every tree the compiler actually produces.
    fn subtrees_are_contiguous(&self) -> bool {
        fn walk(nodes: &[Node], idx: NodeIdx, out: &mut Vec<NodeIdx>) {
            out.push(idx);
            let n = nodes[idx];
            if n.node_type.has_left() {
                walk(nodes, n.left, out);
            }
            if n.node_type.has_right() {
                walk(nodes, n.right, out);
            }
        }
        (0..self.nodes.len()).all(|idx| {
            let mut seen = Vec::new();
            walk(&self.nodes, idx, &mut seen);
            seen.sort_unstable();
            seen.dedup();
            seen == (idx..self.ends[idx]).collect::<Vec<_>>()
        })
    }

    /// One past the last index of the subtree rooted at `idx`.
    #[inline]
    pub fn subtree_end(&self, idx: NodeIdx) -> NodeIdx {
        self.ends[idx]
    }

    /// Set a node's boolean role.
    pub fn set_type(&mut self, idx: NodeIdx, node_type: NodeType) {
        self.nodes[idx].node_type = node_type;
    }

    /// Set a node's left child.
    pub fn set_left(&mut self, idx: NodeIdx, child: NodeIdx) {
        self.nodes[idx].left = child;
    }

    /// Set a node's right child.
    pub fn set_right(&mut self, idx: NodeIdx, child: NodeIdx) {
        self.nodes[idx].right = child;
    }

    /// Whether node `idx` is a leaf (its value comes straight from an operation).
    pub(crate) fn is_leaf(&self, idx: NodeIdx) -> bool {
        self.nodes[idx].node_type == NodeType::Leaf
    }

    /// Number of nodes (equivalently, the number of buckets).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty (no nodes at all).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Validate structural invariants: parent links, child presence per node type, child
    /// indices in range, and that the whole array is one connected pre-order tree.
    pub fn validate(&mut self) -> Result<(), TreeError> {
        if self.nodes.is_empty() {
            return Ok(());
        }
        let end = self.validate_node(0, 0)?;
        if end != self.nodes.len() {
            return Err(TreeError::NotConnected);
        }
        // The tree is known good and known connected, which is exactly the point at which
        // its subtree extents are meaningful.
        self.fill_extents();
        Ok(())
    }

    /// Validate the subtree rooted at `idx` (whose parent must be `parent`), returning the
    /// index one past the subtree (subtrees are contiguous).
    fn validate_node(&self, idx: NodeIdx, parent: NodeIdx) -> Result<NodeIdx, TreeError> {
        let node = self.nodes[idx];
        if node.parent != parent {
            return Err(TreeError::BadParent(idx));
        }
        let check = |present: bool, child: NodeIdx| -> Result<(), TreeError> {
            if present {
                if child == 0 || child >= self.nodes.len() {
                    return Err(TreeError::ChildOutOfRange(idx));
                }
            } else if child != 0 {
                return Err(TreeError::UnexpectedChild(idx));
            }
            Ok(())
        };
        check(node.node_type.has_left(), node.left)?;
        check(node.node_type.has_right(), node.right)?;

        let mut pos = idx + 1;
        if node.node_type.has_left() {
            pos = self.validate_node(pos, idx)?;
        }
        if node.node_type.has_right() {
            pos = self.validate_node(pos, idx)?;
        }
        Ok(pos)
    }

    /// Create fresh execution state for one match.
    ///
    /// Requires [`Self::validate`] to have run: `reset_node` and `seal_node` read precomputed
    /// subtree extents, and without them they would index an empty table. Asserted rather
    /// than assumed because the builder cannot tell a half-built tree from a finished one,
    /// and the failure would otherwise land inside the matcher rather than at the mistake.
    pub fn new_state(&self) -> LogicTreeState<'_> {
        assert_eq!(
            self.ends.len(),
            self.nodes.len(),
            "LogicTree::validate must run before matching"
        );
        LogicTreeState {
            tree: self,
            data: vec![State::Unset; self.nodes.len()],
            stall: 0,
            root_not_true: false,
            root_settled: false,
            bound_lo: vec![Tri::False; self.nodes.len()],
            bound_hi: vec![Tri::True; self.nodes.len()],
        }
    }
}

/// The value a node holds once it is no longer waiting: a three-valued (Kleene) logic result.
///
/// `Unknown` is a *terminal* value, not a gap. It is what a comparison yields when an operand
/// is absent, and — crucially — it is immune to negation: `NOT Unknown` is `Unknown`, so an
/// absent field can never be turned into a match by writing `!=` or `NOT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tri {
    True,
    False,
    Unknown,
}

impl Tri {
    #[inline]
    pub(crate) fn from_bool(b: bool) -> Self {
        if b {
            Tri::True
        } else {
            Tri::False
        }
    }

    /// Kleene negation: the two definite values swap, `Unknown` is a fixed point.
    #[inline]
    pub(crate) fn not(self) -> Self {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }

    #[inline]
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Tri::True, _) | (_, Tri::True) => Tri::True,
            (Tri::Unknown, _) | (_, Tri::Unknown) => Tri::Unknown,
            _ => Tri::False,
        }
    }

    #[inline]
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Tri::False, _) | (_, Tri::False) => Tri::False,
            (Tri::Unknown, _) | (_, Tri::Unknown) => Tri::Unknown,
            _ => Tri::True,
        }
    }
}

/// A node's state during one match.
///
/// Four states, and the distinction that matters is between the first and the rest: `Unset` is
/// a statement about the *scan* ("not reached yet, may still be determined"), while the other
/// three are statements about the *data* and are final. Conflating "not yet known" with
/// "unknowable" is what would force absence to be settled by an end-of-scan sweep; keeping
/// them apart lets a field's absence be recorded the moment its enclosing container closes,
/// so the result can propagate — and the scan can stop — as early as the logic allows.
///
/// `Unknown` does double duty. Besides a genuinely unanswerable comparison it is also what a
/// node becomes when its value can no longer affect anything (the pruned sibling of a
/// short-circuited branch). Those need not be distinguished: pruning only ever happens
/// *beneath a node that is already decided*, so a pruned value is never an input to an
/// undecided computation — and "this will never be known" is an honest description of both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Not yet reached by the scan; may still become any of the below.
    Unset,
    /// Terminally unknown: unanswerable, or no longer able to matter.
    Unknown,
    True,
    False,
}

impl State {
    /// This node's three-valued result, or `None` while it is still `Unset`.
    #[inline]
    fn tri(self) -> Option<Tri> {
        match self {
            State::Unset => None,
            State::Unknown => Some(Tri::Unknown),
            State::True => Some(Tri::True),
            State::False => Some(Tri::False),
        }
    }

    #[inline]
    fn from_tri(t: Tri) -> Self {
        match t {
            Tri::True => State::True,
            Tri::False => State::False,
            Tri::Unknown => State::Unknown,
        }
    }
}

/// Per-match execution state over a [`LogicTree`]: the current value of each node plus the
/// active loop stall boundary.
#[derive(Debug, Clone)]
pub struct LogicTreeState<'t> {
    tree: &'t LogicTree,
    data: Vec<State>,
    stall: NodeIdx,
    /// Whether the root's value is already known not to be `True`, even though the root itself is
    /// not resolved. See [`LogicTreeState::root_settled`].
    root_not_true: bool,
    /// Whether the root's *verdict* is settled, by either route: the root resolved to a value,
    /// or `True` became unreachable. Cached rather than derived because the matcher asks it
    /// after every operation, and deriving it means reaching through the `MatchDef` for the
    /// root bucket and then bounds-checking an index into `data` to read one byte that two
    /// booleans already know.
    root_settled: bool,
    /// Scratch for [`LogicTreeState::root_can_be_true`], kept here so the analysis allocates once
    /// per matcher rather than once per absent field.
    bound_lo: Vec<Tri>,
    bound_hi: Vec<Tri>,
}

impl LogicTreeState<'_> {
    /// Reset all state for reuse on a new document.
    pub fn reset(&mut self) {
        self.stall = 0;
        self.root_not_true = false;
        self.root_settled = false;
        self.data.iter_mut().for_each(|s| *s = State::Unset);
    }

    /// Whether the scan can stop as far as the logic is concerned: the root has a value, or it
    /// can no longer come out `True`.
    ///
    /// A matcher answers a boolean, so `Unknown` and `False` are the same verdict at the root and
    /// the scan is finished the moment `True` becomes unreachable — which happens strictly earlier
    /// than the root resolving. An `And` with one `Unknown` child is the case that matters:
    /// `Unknown AND True` is `Unknown` and `Unknown AND False` is `False`, so it can never be
    /// `True`, yet the tree must still wait for the sibling to tell those two apart. That
    /// distinction is only ever observed by a `NOT` above the node; nothing else can turn a
    /// `False` into a match. So the value has to wait, but the *verdict* does not.
    #[inline]
    pub fn root_settled(&self) -> bool {
        self.root_settled
    }

    /// Whether node `idx` has any resolved value yet — including `Unknown`, which is a
    /// decision (the operand is absent) and not a pending state.
    #[inline]
    pub fn is_resolved(&self, idx: NodeIdx) -> bool {
        self.data[idx] != State::Unset
    }

    /// Node `idx`'s three-valued value, or `None` if it is still `Unset`.
    ///
    /// Test-only: the matcher reads a node's value exactly once per array element, and does it
    /// through [`Self::seal_and_value`] so the seal and the read share one indexing.
    #[cfg(test)]
    fn value(&self, idx: NodeIdx) -> Option<Tri> {
        self.data[idx].tri()
    }

    /// Whether node `idx` is resolved to `true`.
    #[inline]
    pub fn is_true(&self, idx: NodeIdx) -> bool {
        self.data[idx] == State::True
    }

    /// Whether evaluation is currently inside a loop body.
    ///
    /// True exactly when a stall boundary is set, since node `0` is the root and can never be a
    /// loop body. The matcher uses this to skip sealing a container that closes inside a body:
    /// the loop seals the whole body after each element anyway, so doing it per nested container
    /// as well is redundant — and in an array of objects that redundancy is paid per element.
    #[inline]
    pub fn in_loop_body(&self) -> bool {
        self.stall != 0
    }

    /// Set the loop stall boundary (results do not propagate above this node), returning
    /// the previous boundary so callers can restore it (loops nest like a stack).
    pub fn set_stall(&mut self, idx: NodeIdx) -> NodeIdx {
        std::mem::replace(&mut self.stall, idx)
    }

    /// Record a leaf/loop node's boolean result and propagate it upward.
    pub fn mark(&mut self, idx: NodeIdx, value: bool) {
        self.mark_tri(idx, Tri::from_bool(value));
    }

    /// Record node `idx`'s three-valued result and propagate it upward, short-circuiting where
    /// possible and stopping at the stall boundary or the root. A node already resolved is
    /// left unchanged.
    ///
    /// A mark is three things — store the value, prune below, propagate above — and the case
    /// that runs once per array element is the one where the second and third are *empty*: a
    /// leaf has nothing beneath it, and a leaf that is itself the stall boundary has nothing
    /// above it that the loop will let it reach. That is the whole of an ordinary loop body,
    /// `ANY t IN tags SATISFIES t == "x"`, and it is a single store.
    ///
    /// Reaching that case must not cost a call. Propagation is recursive and a recursive
    /// function cannot be inlined, so the store moves out in front of the recursion — here,
    /// where it can inline — and [`Self::mark_tri_full`] is left as one outlined function.
    ///
    /// Deliberately *not* written as "inline the preamble, call the rest". Propagation reaches
    /// `mark_tri_full` several times per node as it walks up, and inlining a guard at each of
    /// those sites helps none of them while making every one bigger. The guard belongs only
    /// where the fast case actually occurs.
    #[inline(always)]
    pub(crate) fn mark_tri(&mut self, idx: NodeIdx, value: Tri) {
        // Inside a loop body, marking a leaf is *the* per-element operation, and what it
        // triggers is short by construction: the body is the stall boundary, and the leaf an
        // operation writes sits at it or just below it. `t == "x"` is the first, `t != "x"` —
        // which lowers to `Not(Equals)` — the second. Both are stores, and reaching them
        // through the general walk cost its frame every element: `mark_tri_full`'s prologue
        // and epilogue are a significant part of what a per-element mark costs.
        //
        // `stall` is zero when no loop is running and node zero is always the root, so testing
        // it also excludes the root, which must reach the `root_settled` bookkeeping in the
        // outlined body however leaf-like it is.
        if self.stall != 0 && self.tree.is_leaf(idx) {
            if self.data[idx] != State::Unset {
                // Already resolved is exactly what the outlined body would return on.
                return;
            }
            self.data[idx] = State::from_tri(value);
            // A leaf has no descendants to prune, and the stall boundary has no ancestor the
            // loop will let this reach.
            if idx == self.stall {
                return;
            }
            // One hop, in place. Beyond that — a body that is a real tree — hand over; and
            // hand over by value, so the walk resumes at the parent rather than re-deriving
            // what was just stored. A parent still waiting on its other operand stops here,
            // which is most of a wide disjunction's terms.
            // The hop below is a deliberate trade, not a free win. `mark_tri` is
            // `#[inline(always)]`, so this block is emitted at every call site — including the
            // ones that can never be inside a loop body and can only ever fail the guard
            // above. It is taken because a loop body is where the single per-element path is
            // most exposed. Anything further added here widens the same tax.
            let parent = self.tree.nodes[idx].parent;
            let Some(v) = self.combine(parent) else {
                // The parent still needs its other operand, which is most of a wide
                // disjunction's terms.
                return;
            };
            if parent == self.stall && self.data[parent] == State::Unset {
                self.data[parent] = State::from_tri(v);
                self.prune_children(parent);
                return;
            }
            self.mark_tri_full(parent, v);
            return;
        }
        self.mark_tri_full(idx, value);
    }

    /// A mark that has something to propagate: store the value, prune the subtree it settles,
    /// then walk up recomputing each ancestor until one is still undecided.
    ///
    /// The walk is a **loop**, not a recursion, and the level count is not the deep number
    /// that might suggest: `a != b` compiles to `Not(Equals)`, so an ordinary `EVERY t IN tags
    /// SATISFIES t != "x"` crosses exactly one level per array element. Propagation is a path
    /// from a node to its ancestor, which is the shape a loop describes; splitting the
    /// *decision* out into [`Self::combine`], which returns a value instead of marking one, is
    /// what lets it be one rather than a pair of mutually recursive calls per level.
    fn mark_tri_full(&mut self, mut idx: NodeIdx, mut value: Tri) {
        // `Unknown` is the only value that can cap an ancestor below `True` without resolving
        // it, so it is the only one worth running the reachability analysis for. Recorded
        // across the walk and acted on once at the end: the analysis reads the whole tree, so
        // the answer it gives after the walk is the one every level would have converged on,
        // and running it per level only re-asked the question with less information.
        let mut saw_unknown = false;
        loop {
            if self.data[idx] != State::Unset {
                break;
            }
            self.data[idx] = State::from_tri(value);
            self.prune_children(idx);
            saw_unknown |= value == Tri::Unknown;

            if idx == 0 {
                if value != Tri::True {
                    self.root_not_true = true;
                }
                self.root_settled = true;
                return;
            }
            // The loop's stall boundary: a per-element result stops here, and the loop marks
            // the boundary itself once the whole array has been read.
            if idx == self.stall {
                return;
            }
            let parent = self.tree.nodes[idx].parent;
            match self.combine(parent) {
                Some(v) => {
                    idx = parent;
                    value = v;
                }
                // The parent still needs its other operand.
                None => break,
            }
        }
        // Skipped inside a loop body: that subtree holds one element's state and will be reset,
        // so it says nothing about the root. The loop's own result re-triggers this when it
        // marks the body with the stall already restored.
        if saw_unknown && self.stall == 0 && !self.root_not_true {
            self.root_not_true = !self.root_can_be_true();
            self.root_settled |= self.root_not_true;
        }
    }

    /// Whether the root can still come out `True`, given everything known so far.
    ///
    /// Bounds each node by the interval of values it could still take, over the ordering
    /// `False < Unknown < True`, and reports whether `True` is still within the root's reach.
    /// This is an analysis *about* the logic rather than part of it, which is why it can end a
    /// scan early without weakening the tables: it never produces a value that feeds another
    /// operation, and the answer is still computed by the ordinary Kleene propagation afterwards.
    /// De Morgan is a property of the connectives and is untouched.
    ///
    /// The connectives are exactly the order operations, so bounds map straight through: `And` is
    /// a minimum and `Or` a maximum — both monotone, so each bound combines with its counterpart —
    /// while `Not` reverses the order and therefore *swaps* them, and a `Loop` takes its body's.
    /// An unresolved leaf could still become anything, so it spans `[False, True]`.
    ///
    /// Complete as well as sound: it stops exactly when no assignment to the unresolved leaves
    /// yields `True`. That is strictly more than the obvious `And`-only rule catches — it also
    /// covers `NOT (absent OR unseen)`, whose upper bound is `Unknown` because the `Not` swaps a
    /// lower bound of `Unknown` into the upper position, and a disjunction whose every branch is
    /// separately capped.
    fn root_can_be_true(&mut self) -> bool {
        // Pre-order layout: a node's children always have higher indices, so a single reverse
        // sweep settles every child before its parent.
        for i in (0..self.data.len()).rev() {
            if let Some(v) = self.data[i].tri() {
                self.bound_lo[i] = v;
                self.bound_hi[i] = v;
                continue;
            }
            let node = self.tree.nodes[i];
            let (lo, hi) = match node.node_type {
                NodeType::Leaf => (Tri::False, Tri::True),
                NodeType::And => (
                    self.bound_lo[node.left].and(self.bound_lo[node.right]),
                    self.bound_hi[node.left].and(self.bound_hi[node.right]),
                ),
                NodeType::Or | NodeType::Neor => (
                    self.bound_lo[node.left].or(self.bound_lo[node.right]),
                    self.bound_hi[node.left].or(self.bound_hi[node.right]),
                ),
                // Order-reversing, so the bounds cross over.
                NodeType::Not => (
                    self.bound_hi[node.left].not(),
                    self.bound_lo[node.left].not(),
                ),
                NodeType::Loop => (self.bound_lo[node.left], self.bound_hi[node.left]),
            };
            self.bound_lo[i] = lo;
            self.bound_hi[i] = hi;
        }
        self.bound_hi[0] == Tri::True
    }

    /// Mark a node's still-`Unset` children `Unknown` — this node is decided, so their values
    /// can no longer affect anything and will never be learned.
    ///
    /// Recursive, so it cannot inline, so every caller pays a call to discover that a leaf has
    /// no children — and a leaf is what almost every mark lands on. The extent table answers
    /// it in one load: a subtree of one node is a leaf. A body of many terms marks one leaf
    /// per term per array element, so that call is the common case and this removes it.
    #[inline(always)]
    fn prune_children(&mut self, idx: NodeIdx) {
        if self.subtree_end(idx) == idx + 1 {
            return;
        }
        // Having descendants is not the same as having any to *prune*. A node reached by
        // propagation was decided *from* its children, so for `Not` and `Loop` — the whole of
        // an ordinary `!=` body — every child is already set and the recursive walk finds
        // nothing. Asking here costs two loads; reaching the same answer through the call cost
        // a frame per element.
        let node = self.tree.nodes[idx];
        let left_open = self.data[node.left] == State::Unset;
        let right_open = node.node_type.has_right() && self.data[node.right] == State::Unset;
        if left_open || right_open {
            self.prune_children_below(idx);
        }
    }

    /// The recursive part of [`Self::prune_children`], reached only for a node that has
    /// descendants.
    fn prune_children_below(&mut self, idx: NodeIdx) {
        let node = self.tree.nodes[idx];
        if node.node_type.has_left() && self.data[node.left] == State::Unset {
            self.data[node.left] = State::Unknown;
            self.prune_children(node.left);
        }
        if node.node_type.has_right() && self.data[node.right] == State::Unset {
            self.data[node.right] = State::Unknown;
            self.prune_children(node.right);
        }
    }

    /// An internal node's value given what its children currently hold, or `None` while it is
    /// still undecided.
    ///
    /// `And`/`Or` short-circuit, but only on their *absorbing* value — `False` for `And`, `True`
    /// for `Or` — because those are the only inputs that settle the result without the sibling.
    /// `Unknown` is not absorbing (`Unknown OR True` is `True`), so a node with one `Unknown`
    /// child still has to wait for the other.
    ///
    /// Answers rather than acts, so [`Self::mark_tri_full`] can walk up in a loop instead of
    /// recursing back into itself.
    fn combine(&self, idx: NodeIdx) -> Option<Tri> {
        let node = self.tree.nodes[idx];
        // Both operands are read up front even though `Not` and `Loop` have only one (their
        // `right` is the zero sentinel, so the extra read is of the root's state). Reading
        // lazily inside the arms that need it puts a branch on the `Or` path, which is the
        // hot one for a body of many terms, and costs more there than the extra read.
        let (l, r) = (self.data[node.left].tri(), self.data[node.right].tri());
        match node.node_type {
            NodeType::Or => {
                if l == Some(Tri::True) || r == Some(Tri::True) {
                    Some(Tri::True)
                } else {
                    Some(l?.or(r?))
                }
            }
            // Non-short-circuiting by design: it merges independently-tracked top-level
            // expressions, so both sides must be given the chance to resolve on their own.
            NodeType::Neor => Some(l?.or(r?)),
            NodeType::And => {
                if l == Some(Tri::False) || r == Some(Tri::False) {
                    Some(Tri::False)
                } else {
                    Some(l?.and(r?))
                }
            }
            NodeType::Not => Some(l?.not()),
            NodeType::Loop => l,
            NodeType::Leaf => unreachable!("leaf nodes are not re-checked"),
        }
    }

    /// Index one past the (contiguous, pre-order) subtree rooted at `idx`.
    #[inline]
    fn subtree_end(&self, idx: NodeIdx) -> NodeIdx {
        self.tree.subtree_end(idx)
    }

    /// Seal the subtree rooted at `root`: every still-`Unset` node within it becomes
    /// `Unknown`, propagating up to `root`.
    ///
    /// Used to finalize a loop body once an element has been scanned — anything in the body
    /// still unset refers to a field that element did not have, which is unanswerable rather
    /// than false. The caller should have `root` set as the stall boundary so nothing escapes
    /// above it.
    ///
    /// Back-to-front so leaves settle before their ancestors and the result propagates through
    /// negation like any other value. Note there is no special case for nested loop nodes: an
    /// unrun loop's body seals to `Unknown`, `NOT Unknown` is `Unknown`, and a loop over it
    /// stays `Unknown` — where a boolean default would have let a `!=` in the body flip the
    /// loop true and had to be patched around.
    ///
    /// The early-out is inline and the sweep is not. A loop body that the element *did*
    /// answer is already resolved, so the seal is one load and a branch — and that is the
    /// common case, once per element, on a call whose outlined body exists for the elements
    /// that left something unset.
    #[inline(always)]
    pub fn seal_node(&mut self, root: NodeIdx) {
        if self.data[root] != State::Unset {
            return;
        }
        self.seal_unset_node(root);
    }

    /// Seal `root` and return the value it ends up with, treating a node the seal could not
    /// settle as `Unknown`.
    ///
    /// One operation because it is one question — "what did this element answer?" — and asking
    /// it as `seal_node` followed by `value` reads the same byte through two separate
    /// bounds-checked indexings, once per array element.
    #[inline(always)]
    pub(crate) fn seal_and_value(&mut self, root: NodeIdx) -> Tri {
        match self.data[root] {
            // The element answered the body outright, which is the ordinary case: one read.
            State::True => Tri::True,
            State::False => Tri::False,
            State::Unknown => Tri::Unknown,
            State::Unset => {
                self.seal_unset_node(root);
                self.data[root].tri().unwrap_or(Tri::Unknown)
            }
        }
    }

    /// [`Self::seal_node`] for a root that is still `Unset`; see the note there.
    #[inline(never)]
    fn seal_unset_node(&mut self, root: NodeIdx) {
        let end = self.subtree_end(root);
        // One pass, children before parents. Subtrees are contiguous and pre-order, so a
        // node's descendants all have higher indices than it does and a reverse sweep settles
        // every operand before the node that reads it. Each node is then its own combinator
        // applied to values already in hand — no upward propagation, no re-pruning.
        //
        // Marking each unset node individually instead would prune that node's descendants
        // and walk up to the subtree root for every one of them, costing O(unset x depth) with
        // the pruning repeated on the way — paid per element on a loop body naming fields the
        // elements lack, which is ordinary in heterogeneous documents.
        for i in (root + 1..end).rev() {
            if self.data[i] == State::Unset {
                let v = self.sealed_value(i);
                self.data[i] = State::from_tri(v);
            }
        }
        // The root still goes through `mark_tri`, so everything that happens *outside* this
        // subtree — propagation past a stall boundary, the `root_not_true` tracking — is
        // exactly what it was. Its `prune_children` now finds every descendant already set.
        let v = self.sealed_value(root);
        self.mark_tri(root, v);
    }

    /// The value a node takes when its subtree is sealed: an unreached leaf is `Unknown`, and
    /// anything above it is its connective over operands that are already settled.
    ///
    /// The Kleene combinators subsume `check`'s short-circuits rather than skipping them —
    /// `True.or(Unknown)` is `True` and `False.and(Unknown)` is `False` — so a node whose
    /// other operand had already resolved keeps the answer it always had.
    #[inline]
    fn sealed_value(&self, idx: NodeIdx) -> Tri {
        let node = self.tree.nodes[idx];
        let left = || self.data[node.left].tri().unwrap_or(Tri::Unknown);
        let right = || self.data[node.right].tri().unwrap_or(Tri::Unknown);
        match node.node_type {
            NodeType::Leaf => Tri::Unknown,
            NodeType::Not => left().not(),
            NodeType::Loop => left(),
            NodeType::Or | NodeType::Neor => left().or(right()),
            NodeType::And => left().and(right()),
        }
    }

    /// Reset the subtree rooted at `idx` back to `Unset` (used between loop iterations).
    #[inline(always)]
    pub fn reset_node(&mut self, idx: NodeIdx) {
        // The subtree is contiguous and its extent is known at compile time, so this is a
        // slice fill rather than a recursive walk with a call per node.
        let end = self.subtree_end(idx);
        // `fill` on a byte-sized type is `memset`, and a `memset` through the PLT to clear one
        // or two bytes costs more than the bytes. Those are the sizes loop bodies actually
        // have: a bare comparison is one leaf, and `!=` compiles to `Not(Equals)`, which is
        // two, and a library call for either is pure overhead. The single-node
        // case is written as an index rather than a slice pattern, because taking the
        // sub-slice costs its own range check before the length can be tested at all.
        if end == idx + 1 {
            self.data[idx] = State::Unset;
        } else if let [a, b] = &mut self.data[idx..end] {
            *a = State::Unset;
            *b = State::Unset;
        } else {
            self.data[idx..end].fill(State::Unset);
        }
    }

    /// Force the tree to a final result: seal the whole tree, so anything the scan never
    /// reached becomes `Unknown` and propagates.
    ///
    /// This is only a backstop. The matcher seals each container's absent fields as that
    /// container closes, so by the time the document ends most nodes are already decided; what
    /// remains here are fields whose enclosing scope was the document itself.
    ///
    /// A `Unknown` root means "no match" — the caller reads the result with
    /// [`Self::is_true`] — but it is deliberately *not* collapsed to `False` here, because
    /// `Unknown` and `False` are only interchangeable at the root. Everywhere below it the
    /// difference is what stops negation turning an absent field into a match.
    pub fn resolve(&mut self) {
        self.seal_node(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `root(left_type? ...)`: helper to construct a two-level tree
    /// `op(leaf, leaf)` and return (tree, left_idx, right_idx).
    fn binary(op: NodeType) -> (LogicTree, NodeIdx, NodeIdx) {
        let mut t = LogicTree::new();
        t.set_type(0, op);
        let l = t.add_child(0);
        let r = t.add_child(0);
        t.set_left(0, l);
        t.set_right(0, r);
        t.validate().expect("valid");
        (t, l, r)
    }

    #[test]
    fn and_resolves_true_and_short_circuits_false() {
        let (t, l, r) = binary(NodeType::And);
        let mut s = t.new_state();
        s.mark(l, true);
        assert!(!s.is_resolved(0));
        s.mark(r, true);
        assert!(s.is_true(0));

        // Short-circuit: a single false child decides the AND immediately.
        let mut s = t.new_state();
        s.mark(l, false);
        assert!(s.is_resolved(0) && !s.is_true(0));
    }

    #[test]
    fn or_short_circuits_true() {
        let (t, l, _r) = binary(NodeType::Or);
        let mut s = t.new_state();
        s.mark(l, true);
        assert!(s.is_true(0));
    }

    #[test]
    fn not_inverts() {
        let mut t = LogicTree::new();
        t.set_type(0, NodeType::Not);
        let l = t.add_child(0);
        t.set_left(0, l);
        t.validate().expect("valid");

        let mut s = t.new_state();
        s.mark(l, true);
        assert!(s.is_resolved(0) && !s.is_true(0));

        let mut s = t.new_state();
        s.mark(l, false);
        assert!(s.is_true(0));
    }

    #[test]
    fn resolve_seals_unset_leaves_to_unknown() {
        // OR of two leaves, neither op ran (both fields absent): Unknown OR Unknown, which is
        // resolved-but-not-true, hence no match.
        let (t, _l, _r) = binary(NodeType::Or);
        let mut s = t.new_state();
        s.resolve();
        assert_eq!(s.value(0), Some(Tri::Unknown));
        assert!(s.is_resolved(0) && !s.is_true(0));
    }

    /// The property the whole three-valued design exists for: an absent field cannot be turned
    /// into a match by negating it. Under a boolean default this NOT resolved to `true`.
    #[test]
    fn not_of_an_absent_leaf_stays_unknown() {
        let mut t = LogicTree::new();
        t.set_type(0, NodeType::Not);
        let l = t.add_child(0);
        t.set_left(0, l);
        t.validate().expect("valid");

        let mut s = t.new_state();
        s.resolve();
        assert_eq!(
            s.value(0),
            Some(Tri::Unknown),
            "NOT Unknown must be Unknown"
        );
        assert!(!s.is_true(0));

        // Any number of negations is still Unknown — it is a fixed point, not a parity flip.
        let mut t2 = LogicTree::new();
        t2.set_type(0, NodeType::Not);
        let a = t2.add_child(0);
        t2.set_type(a, NodeType::Not);
        let b = t2.add_child(a);
        t2.set_left(0, a);
        t2.set_left(a, b);
        t2.validate().expect("valid");
        let mut s2 = t2.new_state();
        s2.resolve();
        assert_eq!(s2.value(0), Some(Tri::Unknown));
    }

    /// *Why* the tables are what they are, as two properties rather than a list of entries.
    ///
    /// The tables are not a style choice among several workable three-valued logics — they are
    /// forced, by the rule that a connective may return a definite value exactly when every way
    /// of resolving the unknown operands agrees on it:
    ///
    /// * **sound** — never answer when the resolutions disagree (`Unknown AND True` must be
    ///   `Unknown`, since `True AND True` is true and `False AND True` is false);
    /// * **complete** — always answer when they agree (`Unknown AND False` must be `False`,
    ///   because both resolutions are false — refusing here would discard a known answer).
    ///
    /// Together those pin every cell, and they are what makes De Morgan's laws hold, which is
    /// the property the engine actually depends on: it is why `NOT (a < b)` and `a >= b` select
    /// the same documents. A tempting alternative — make `Unknown` poison `AND` outright and
    /// vanish in `OR` — fails both halves at once (it answers `False` for `Unknown OR False`,
    /// which is not determined, and refuses `Unknown AND False`, which is) and duality collapses
    /// with them.
    #[test]
    fn kleene_tables_are_sound_complete_and_dual() {
        use Tri::{False, True, Unknown};
        let all = [True, False, Unknown];
        // Evaluate `op` through a real two-leaf tree, so this tests `check`, not just `Tri`.
        let via_tree = |op: NodeType, a: Tri, b: Tri| -> Tri {
            let (t, l, r) = binary(op);
            let mut s = t.new_state();
            s.mark_tri(l, a);
            s.mark_tri(r, b);
            s.value(0)
                .expect("both children marked, so the node is decided")
        };
        let resolutions = |x: Tri| -> Vec<Tri> {
            if x == Unknown {
                vec![True, False]
            } else {
                vec![x]
            }
        };

        for &a in &all {
            for &b in &all {
                for op in [NodeType::And, NodeType::Or] {
                    // What the definite cases give, over every resolution of the unknowns.
                    let mut outs: Vec<Tri> = Vec::new();
                    for x in resolutions(a) {
                        for y in resolutions(b) {
                            let o = via_tree(op, x, y);
                            if !outs.contains(&o) {
                                outs.push(o);
                            }
                        }
                    }
                    let got = via_tree(op, a, b);
                    if let [only] = outs[..] {
                        assert_eq!(got, only, "{op:?}({a:?}, {b:?}) must answer {only:?}");
                    } else {
                        assert_eq!(got, Unknown, "{op:?}({a:?}, {b:?}) is not determined");
                    }
                }

                // De Morgan, both directions.
                assert_eq!(
                    via_tree(NodeType::And, a, b).not(),
                    via_tree(NodeType::Or, a.not(), b.not()),
                    "NOT({a:?} AND {b:?}) == NOT {a:?} OR NOT {b:?}"
                );
                assert_eq!(
                    via_tree(NodeType::Or, a, b).not(),
                    via_tree(NodeType::And, a.not(), b.not()),
                    "NOT({a:?} OR {b:?}) == NOT {a:?} AND NOT {b:?}"
                );
            }
            // Negation is an involution, so `!=` cannot drift from `==` under double negation.
            assert_eq!(a.not().not(), a, "NOT NOT {a:?}");
        }

        // The same two properties applied to `Tri::and`/`Tri::or` directly.
        //
        // Needed because `check` enforces the absorbing cases itself, short-circuiting on `False`
        // for `And` and `True` for `Or` before it consults the table at all — so those arms are
        // unreachable through a tree, and a table-only regression would otherwise pass every
        // assertion above while leaving the table wrong for any future caller.
        for &a in &all {
            for &b in &all {
                let expect = |definite: fn(Tri, Tri) -> Tri| -> Tri {
                    let mut outs: Vec<Tri> = Vec::new();
                    for x in resolutions(a) {
                        for y in resolutions(b) {
                            let o = definite(x, y);
                            if !outs.contains(&o) {
                                outs.push(o);
                            }
                        }
                    }
                    match outs[..] {
                        [only] => only,
                        _ => Unknown,
                    }
                };
                // The definite-input behaviour each table must extend.
                fn d_and(x: Tri, y: Tri) -> Tri {
                    Tri::from_bool(x == Tri::True && y == Tri::True)
                }
                fn d_or(x: Tri, y: Tri) -> Tri {
                    Tri::from_bool(x == Tri::True || y == Tri::True)
                }
                assert_eq!(a.and(b), expect(d_and), "Tri::and({a:?}, {b:?})");
                assert_eq!(a.or(b), expect(d_or), "Tri::or({a:?}, {b:?})");
                // Duality on the tables themselves.
                assert_eq!(a.and(b).not(), a.not().or(b.not()), "De Morgan on Tri::and");
                assert_eq!(a.or(b).not(), a.not().and(b.not()), "De Morgan on Tri::or");
            }
        }
    }

    /// Kleene's tables, exhaustively, through the tree rather than on `Tri` directly — so this
    /// also pins that `check` reaches the same answers by its short-circuiting route.
    #[test]
    fn connectives_follow_kleene_tables() {
        use Tri::{False, True, Unknown};
        let all = [True, False, Unknown];

        // Expected: AND is min, OR is max, under False < Unknown < True.
        let expect_and = |a: Tri, b: Tri| match (a, b) {
            (False, _) | (_, False) => False,
            (Unknown, _) | (_, Unknown) => Unknown,
            _ => True,
        };
        let expect_or = |a: Tri, b: Tri| match (a, b) {
            (True, _) | (_, True) => True,
            (Unknown, _) | (_, Unknown) => Unknown,
            _ => False,
        };

        for op in [NodeType::And, NodeType::Or, NodeType::Neor] {
            for &a in &all {
                for &b in &all {
                    let (t, l, r) = binary(op);
                    let mut s = t.new_state();
                    s.mark_tri(l, a);
                    s.mark_tri(r, b);
                    let want = match op {
                        NodeType::And => expect_and(a, b),
                        // Neor is an OR that declines to short-circuit; same truth table.
                        _ => expect_or(a, b),
                    };
                    assert_eq!(
                        s.value(0),
                        Some(want),
                        "{op:?}({a:?}, {b:?}) after both children marked"
                    );
                }
            }
        }

        // And the two absorbing cases must resolve from *one* child, without the sibling ever
        // being marked — that is what lets a scan stop early.
        let (t, l, _r) = binary(NodeType::And);
        let mut s = t.new_state();
        s.mark_tri(l, False);
        assert_eq!(s.value(0), Some(False), "False AND _ resolves immediately");

        let (t, l, _r) = binary(NodeType::Or);
        let mut s = t.new_state();
        s.mark_tri(l, True);
        assert_eq!(s.value(0), Some(True), "True OR _ resolves immediately");

        // Unknown is *not* absorbing: with one child Unknown the node must still wait, because
        // the sibling can decide it (Unknown OR True is True, Unknown AND False is False).
        for (op, sibling, want) in [
            (NodeType::Or, True, True),
            (NodeType::Or, False, Unknown),
            (NodeType::And, False, False),
            (NodeType::And, True, Unknown),
        ] {
            let (t, l, r) = binary(op);
            let mut s = t.new_state();
            s.mark_tri(l, Unknown);
            assert!(
                !s.is_resolved(0),
                "{op:?} must wait when only one child is Unknown"
            );
            s.mark_tri(r, sibling);
            assert_eq!(s.value(0), Some(want), "{op:?}(Unknown, {sibling:?})");
        }
    }

    #[test]
    fn loop_stall_prevents_propagation_then_marks() {
        // root Loop(0) -> body(1). The stall boundary is the loop *body* bucket: per-
        // iteration marks land on the body and must not propagate to the loop node until
        // the loop's overall result is applied.
        let mut t = LogicTree::new();
        t.set_type(0, NodeType::Loop);
        let body = t.add_child(0);
        t.set_left(0, body);
        t.validate().expect("valid");

        let mut s = t.new_state();
        let prev = s.set_stall(body);

        // Iteration 1: body false — stops at the stall, loop node stays undecided.
        s.reset_node(body);
        s.mark(body, false);
        assert!(!s.is_true(body));
        assert!(!s.is_resolved(0));

        // Iteration 2: body true — still stalled.
        s.reset_node(body);
        s.mark(body, true);
        assert!(s.is_true(body));
        assert!(!s.is_resolved(0));

        // Apply the overall loop result to the body bucket with the stall lifted; it now
        // propagates up to the loop node.
        s.reset_node(body);
        s.set_stall(prev);
        s.mark(body, true);
        assert!(s.is_true(0));
    }

    #[test]
    fn validate_rejects_broken_trees() {
        let mut t = LogicTree::new();
        t.set_type(0, NodeType::And);
        // And needs two children but has none set.
        assert!(t.validate().is_err());
    }

    /// A pre-order-built tree has contiguous subtrees; a breadth-first-built one does not.
    ///
    /// Both halves matter. The first is the invariant `reset_node` and `seal_node` depend on.
    /// The second is why `LogicTree::validate` carries a `debug_assert` rather than a comment:
    /// the builder API accepts either order, and the wrong one produces a tree that is quietly
    /// mis-reset instead of rejected.
    #[test]
    fn subtrees_are_contiguous_only_in_pre_order() {
        // Pre-order: each child's own subtree is finished before the next sibling is added.
        let mut pre = LogicTree::new();
        let a = pre.add_child(0);
        let c = pre.add_child(a);
        let d = pre.add_child(a);
        pre.set_type(a, NodeType::Or);
        pre.set_left(a, c);
        pre.set_right(a, d);
        let b = pre.add_child(0);
        let e = pre.add_child(b);
        pre.set_type(b, NodeType::Not);
        pre.set_left(b, e);
        pre.set_type(0, NodeType::And);
        pre.set_left(0, a);
        pre.set_right(0, b);
        pre.validate().expect("valid");
        assert!(pre.subtrees_are_contiguous());
        assert_eq!(pre.subtree_end(0), 6);
        assert_eq!(pre.subtree_end(a), 4, "a holds nodes 1, 2 and 3");

        // Breadth-first: both of the root's children first. Node 1's descendants are {1,3,4},
        // which is not a range, so a slice fill rooted at 1 would also clear node 2.
        let mut bfs = LogicTree::new();
        let a = bfs.add_child(0);
        let b = bfs.add_child(0);
        bfs.set_type(0, NodeType::And);
        bfs.set_left(0, a);
        bfs.set_right(0, b);
        let c = bfs.add_child(a);
        let d = bfs.add_child(a);
        bfs.set_type(a, NodeType::Or);
        bfs.set_left(a, c);
        bfs.set_right(a, d);
        assert!(!bfs.subtrees_are_contiguous_unchecked());
    }

}
