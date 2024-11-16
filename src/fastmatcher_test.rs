#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        fastmatcher::{ExecNode, FastMatcher, FastMatcherDef},
        logictree::LogicTree,
    };

    #[test]
    fn fastmatcher_basic() {
        let d = FastMatcherDef {
            parse_node: ExecNode {
                elems: HashMap::from([
                    (
                        "x".to_string(),
                        ExecNode {
                            elems: HashMap::new(),
                        },
                    ),
                    (
                        "y".to_string(),
                        ExecNode {
                            elems: HashMap::new(),
                        },
                    ),
                ]),
            },
            logic_tree: LogicTree::new(),
            num_buckets: 0,
            num_slots: 0,
        };
        let mut m = FastMatcher::new(&d);
    }
}
