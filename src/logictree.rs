use crate::logictree_node::{BinOp, LeafOp, Node};

#[derive(Debug)]
pub struct LogicTree {
    pub nodes: Vec<Node>,
}

impl LogicTree {
    pub fn new() -> LogicTree {
        let mut nodes = Vec::new();
        nodes.push(Node::Leaf(LeafOp { parent_idx: 0 }));
        LogicTree { nodes: nodes }
    }

    pub fn add_node(&mut self) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node::Leaf(LeafOp { parent_idx: idx }));
        idx
    }

    pub fn set_node(&mut self, idx: usize, node: Node) {
        self.nodes[idx] = node;
    }
}

fn retreflocal() -> &Node {
    let x = Node::Leaf(LeafOp { parent_idx: 0 });
    return &x;
}

fn test2() {
    let x = retreflocal();
    println!("{:?}", x);
}

pub fn test() {
    let mut tree = LogicTree::new();

    let nr = 0;
    let n1 = tree.add_node();
    let n2 = tree.add_node();
    tree.set_node(n1, Node::Leaf(LeafOp { parent_idx: nr }));
    tree.set_node(n2, Node::Leaf(LeafOp { parent_idx: nr }));
    tree.set_node(
        0,
        Node::Or(BinOp {
            parent_idx: 0,
            bound_idx: n2,
            left_idx: n1,
            right_idx: n2,
        }),
    );
}
