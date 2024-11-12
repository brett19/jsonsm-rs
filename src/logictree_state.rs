use crate::{logictree::LogicTree, logictree_node::Node};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum LogicNodeState {
    Unset,
    True,
    False,
}

#[derive(Debug)]
pub struct LogicTreeState<'a> {
    pub tree: &'a LogicTree,
    pub state: Vec<LogicNodeState>,
}

impl LogicTreeState<'_> {
    fn check_node(&mut self, node_idx: usize) {
        let node = &self.tree.nodes[node_idx];
        match node {
            Node::Leaf(_) => panic!("leaf nodes are never parents"),
            Node::Loop(_) => {
                // loop nodes intentionally do not propagate upwards
            }
            Node::Not(unary) => {
                let child_state = self.state[unary.child_idx];
                if child_state != LogicNodeState::Unset {
                    let child_value = child_state == LogicNodeState::True;
                    self.mark_node(node_idx, !child_value);
                }
            }
            Node::Or(bin) => {
                let left_state = self.state[bin.left_idx];
                let right_state = self.state[bin.right_idx];
                if left_state == LogicNodeState::True || right_state == LogicNodeState::True {
                    self.mark_node(node_idx, true);
                } else if left_state == LogicNodeState::False
                    && right_state == LogicNodeState::False
                {
                    self.mark_node(node_idx, false);
                }
            }
            Node::And(bin) => {
                let left_state = self.state[bin.left_idx];
                let right_state = self.state[bin.right_idx];
                if left_state == LogicNodeState::False || right_state == LogicNodeState::False {
                    self.mark_node(node_idx, false);
                } else if left_state == LogicNodeState::True && right_state == LogicNodeState::True
                {
                    self.mark_node(node_idx, true);
                }
            }
            Node::Neor(bin) => {
                let left_state = self.state[bin.left_idx];
                let right_state = self.state[bin.right_idx];
                if left_state != LogicNodeState::Unset && right_state != LogicNodeState::Unset {
                    if left_state == LogicNodeState::True || right_state == LogicNodeState::True {
                        self.mark_node(node_idx, true);
                    } else {
                        self.mark_node(node_idx, false);
                    }
                }
            }
        }
    }

    pub fn mark_node(&mut self, node_idx: usize, value: bool) {
        if self.state[node_idx] != LogicNodeState::Unset {
            panic!("cannot mark the same node twice");
        }

        if value {
            self.state[node_idx] = LogicNodeState::True;
        } else {
            self.state[node_idx] = LogicNodeState::False;
        }

        if node_idx != 0 {
            match &self.tree.nodes[node_idx] {
                Node::Leaf(leaf) => {
                    self.check_node(leaf.parent_idx);
                }
                Node::Not(unary) | Node::Loop(unary) => {
                    self.check_node(unary.parent_idx);
                }
                Node::Or(bin) | Node::And(bin) | Node::Neor(bin) => {
                    self.check_node(bin.parent_idx);
                }
            }
        }
    }

    pub fn resolve(&mut self) -> bool {
        for node_idx in 0..self.tree.nodes.len() {
            if self.state[0] != LogicNodeState::Unset {
                break;
            }

            match self.tree.nodes[node_idx] {
                Node::Leaf(_) => {
                    // set all leaf nodes to false
                    self.mark_node(node_idx, false);
                }
                _ => {}
            }
        }

        if self.state[0] == LogicNodeState::True {
            return true;
        } else if self.state[0] == LogicNodeState::False {
            return false;
        }

        panic!("resolve should have resolved the tree")
    }

    pub fn reset(&mut self, node_idx: usize) {
        self.state[node_idx] = LogicNodeState::Unset;

        match &self.tree.nodes[node_idx] {
            Node::Leaf(_) => {}
            Node::Not(unary) | Node::Loop(unary) => {
                for child_idx in node_idx + 1..unary.bound_idx {
                    self.state[child_idx] = LogicNodeState::Unset;
                }
            }
            Node::Or(bin) | Node::And(bin) | Node::Neor(bin) => {
                for child_idx in node_idx + 1..bin.bound_idx {
                    self.state[child_idx] = LogicNodeState::Unset;
                }
            }
        }
    }
}
