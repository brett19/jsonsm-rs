#[derive(Debug)]
pub struct LeafOp {
    pub parent_idx: usize,
}

#[derive(Debug)]
pub struct UnaryOp {
    pub parent_idx: usize,
    pub bound_idx: usize,
    pub child_idx: usize,
}

#[derive(Debug)]
pub struct BinOp {
    pub parent_idx: usize,
    pub bound_idx: usize,
    pub left_idx: usize,
    pub right_idx: usize,
}

#[derive(Debug)]
pub enum Node {
    // a leaf node is a node that will be directly set
    Leaf(LeafOp),

    // an or node is a node that will be set to true if any of its children are true
    Or(BinOp),

    // an and node is a node that will be set to true if all of its children are true
    And(BinOp),

    // a not node is a node that will be set to true if its child is false
    Not(UnaryOp),

    // a loop node is a placeholder node used to detect whether a loop is resolved
    Loop(UnaryOp),

    // a neor node is a node that will be set to true if any of its children are true,
    // but does not short-circuit.  this is used when distinct expressions we want to
    // check are merged, and each needs to be checked regardless of the resolution of
    // the others.
    Neor(BinOp),
}
