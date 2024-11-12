use crate::{logictree::LogicTree, logictree_node::Node};

type ValidateResult<T> = std::result::Result<T, ValidateError>;

#[derive(Debug)]
pub enum ValidateError {
    ReferenceLoop(usize),
    InvalidParent(usize, usize),
    InvalidBound(usize, usize),
    LeftBeforeRight(usize),
    RightAfterLeft(usize),
}

impl LogicTree {
    pub fn validate(&self) -> ValidateResult<()> {
        let mut seen: Vec<usize> = Vec::with_capacity(self.nodes.len());
        match self.validate_node(&mut seen, 0, 0) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn validate_node(
        &self,
        seen: &mut Vec<usize>,
        node_idx: usize,
        parent_node_idx: usize,
    ) -> ValidateResult<usize> {
        // if we see the same node twice, we might have a loop
        if seen.contains(&node_idx) {
            return Err(ValidateError::ReferenceLoop(node_idx));
        }
        seen.push(node_idx);

        let node = &self.nodes[node_idx];
        match node {
            Node::Leaf(leaf) => {
                if leaf.parent_idx != parent_node_idx {
                    return Err(ValidateError::InvalidParent(node_idx, parent_node_idx));
                }

                return Ok(node_idx + 1);
            }

            Node::Not(unary) | Node::Loop(unary) => {
                if unary.parent_idx != parent_node_idx {
                    return Err(ValidateError::InvalidParent(node_idx, parent_node_idx));
                }

                let child_bound_idx = self.validate_node(seen, unary.child_idx, node_idx)?;

                if unary.bound_idx != child_bound_idx {
                    return Err(ValidateError::InvalidBound(node_idx, child_bound_idx));
                }

                return Ok(child_bound_idx);
            }
            Node::Or(bin) | Node::And(bin) | Node::Neor(bin) => {
                if bin.parent_idx != parent_node_idx {
                    return Err(ValidateError::InvalidParent(node_idx, parent_node_idx));
                }

                if bin.right_idx <= bin.left_idx {
                    return Err(ValidateError::LeftBeforeRight(node_idx));
                }

                let left_bound_idx = self.validate_node(seen, bin.left_idx, node_idx)?;

                if bin.right_idx != left_bound_idx {
                    return Err(ValidateError::RightAfterLeft(node_idx));
                }

                let right_bound_idx = self.validate_node(seen, bin.right_idx, node_idx)?;

                if bin.bound_idx != right_bound_idx {
                    return Err(ValidateError::InvalidBound(node_idx, right_bound_idx));
                }

                return Ok(right_bound_idx);
            }
        }
    }
}
