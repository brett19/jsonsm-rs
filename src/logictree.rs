use crate::{
    logictree_node::Node,
    logictree_state::{LogicNodeState, LogicTreeState},
};

#[derive(Debug)]
pub struct LogicTree {
    pub nodes: Vec<Node>,
}

impl LogicTree {
    pub fn new_state(&self) -> LogicTreeState {
        let mut state = Vec::with_capacity(self.nodes.len());
        for _ in 0..self.nodes.len() {
            state.push(LogicNodeState::Unset);
        }

        LogicTreeState {
            tree: self,
            state: state,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::logictree::LogicTree;
    use crate::logictree_node::{BinOp, LeafOp, Node, UnaryOp};
    use crate::logictree_state::LogicNodeState;

    #[test]
    fn logictree_basic() {
        let x = LogicTree {
            nodes: vec![
                Node::Or(BinOp {
                    parent_idx: 0,
                    bound_idx: 6,
                    left_idx: 1,
                    right_idx: 2,
                }),
                Node::Leaf(LeafOp { parent_idx: 0 }),
                Node::And(BinOp {
                    parent_idx: 0,
                    bound_idx: 6,
                    left_idx: 3,
                    right_idx: 4,
                }),
                Node::Leaf(LeafOp { parent_idx: 2 }),
                Node::Not(UnaryOp {
                    parent_idx: 2,
                    bound_idx: 6,
                    child_idx: 5,
                }),
                Node::Leaf(LeafOp { parent_idx: 4 }),
            ],
        };

        println!("LogicTree: {:?}", x);

        match x.validate() {
            Ok(_) => println!("Valid X"),
            Err(e) => println!("Invalid X: {:?}", e),
        }

        {
            let mut s1 = x.new_state();

            s1.mark_node(1, false);
            assert_eq!(s1.state[0], LogicNodeState::Unset);
            assert_eq!(s1.state[1], LogicNodeState::False);
            assert_eq!(s1.state[2], LogicNodeState::Unset);
            assert_eq!(s1.state[3], LogicNodeState::Unset);
            assert_eq!(s1.state[4], LogicNodeState::Unset);
            assert_eq!(s1.state[5], LogicNodeState::Unset);

            s1.mark_node(3, true);
            assert_eq!(s1.state[0], LogicNodeState::Unset);
            assert_eq!(s1.state[1], LogicNodeState::False);
            assert_eq!(s1.state[2], LogicNodeState::Unset);
            assert_eq!(s1.state[3], LogicNodeState::True);
            assert_eq!(s1.state[4], LogicNodeState::Unset);
            assert_eq!(s1.state[5], LogicNodeState::Unset);

            s1.mark_node(5, true);
            assert_eq!(s1.state[0], LogicNodeState::False);
            assert_eq!(s1.state[1], LogicNodeState::False);
            assert_eq!(s1.state[2], LogicNodeState::False);
            assert_eq!(s1.state[3], LogicNodeState::True);
            assert_eq!(s1.state[4], LogicNodeState::False);
            assert_eq!(s1.state[5], LogicNodeState::True);

            println!("State 1: {:?}", s1);

            s1.reset(2);
            assert_eq!(s1.state[0], LogicNodeState::False);
            assert_eq!(s1.state[1], LogicNodeState::False);
            assert_eq!(s1.state[2], LogicNodeState::Unset);
            assert_eq!(s1.state[3], LogicNodeState::Unset);
            assert_eq!(s1.state[4], LogicNodeState::Unset);
            assert_eq!(s1.state[5], LogicNodeState::Unset);

            println!("State 1: {:?}", s1);
        }

        {
            let mut s2 = x.new_state();

            println!("State 2: {:?}", s2);

            let res = s2.resolve();
            println!("State 2 Resolution: {:?}", res);

            println!("State 2: {:?}", s2);
        }

        {
            let y = LogicTree {
                nodes: vec![
                    Node::Not(UnaryOp {
                        parent_idx: 0,
                        bound_idx: 2,
                        child_idx: 1,
                    }),
                    Node::Leaf(LeafOp { parent_idx: 0 }),
                ],
            };

            match y.validate() {
                Ok(_) => println!("Valid Y"),
                Err(e) => println!("Invalid Y: {:?}", e),
            }

            let mut s3 = y.new_state();

            println!("State 3: {:?}", s3);

            let res = s3.resolve();
            println!("State 3 Resolution: {:?}", res);

            println!("State 3: {:?}", s3);
        }
    }
}
