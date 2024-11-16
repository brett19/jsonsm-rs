use std::collections::HashMap;

use crate::{
    jsontokenizer::{JsonTokenizer, TokenizerError},
    jsontokenizer_token::JsonTokenType,
    logictree::LogicTree,
};

pub enum ScanError {
    TokenizerError(TokenizerError),
    UnexpectedToken,
}

pub enum OpType {
    Eq,
    Lt,
    LtEq,
}

pub struct Op {
    op_type: OpType,
    bucket_idx: usize,
}

pub struct ExecNode {
    pub elems: HashMap<String, ExecNode>,
}

pub struct FastMatcherDef {
    pub parse_node: ExecNode,
    pub logic_tree: LogicTree,
    pub num_buckets: usize,
    pub num_slots: usize,
}

pub struct FastMatcher<'a> {
    def: &'a FastMatcherDef,
}

type ScanResult = std::result::Result<bool, ScanError>;

impl<'a> FastMatcher<'a> {
    pub fn new(def: &'a FastMatcherDef) -> Self {
        FastMatcher { def }
    }

    fn exec_object(&mut self, tkns: &mut JsonTokenizer, node: &ExecNode) -> ScanResult {
        if node.elems.len() == 0 {
            match tkns.skip_value() {
                Ok(_) => Ok(true),
                Err(e) => Err(ScanError::TokenizerError(e)),
            }
        } else {
            loop {
                let key = match tkns.step() {
                    Ok(tkn) => match tkn.token_type {
                        JsonTokenType::String => String::from_utf8_lossy(tkn.value),
                        JsonTokenType::EscString => String::from_utf8_lossy(tkn.value),
                        _ => return Err(ScanError::UnexpectedToken),
                    },
                    Err(e) => return Err(ScanError::TokenizerError(e)),
                };

                match tkns.step() {
                    Ok(tkn) => match tkn.token_type {
                        JsonTokenType::ObjectKeyDelim => (),
                        _ => return Err(ScanError::UnexpectedToken),
                    },
                    Err(e) => return Err(ScanError::TokenizerError(e)),
                }

                match node.elems.get(key.as_ref()) {
                    Some(keynode) => match self.exec(tkns, keynode) {
                        Ok(_) => {}
                        Err(e) => return Err(e),
                    },
                    None => match tkns.skip_value() {
                        Ok(_) => (),
                        Err(e) => return Err(ScanError::TokenizerError(e)),
                    },
                }
            }
        }
    }

    fn exec(&mut self, tkns: &mut JsonTokenizer, node: &ExecNode) -> ScanResult {
        match tkns.step() {
            Ok(tkn) => match tkn.token_type {
                JsonTokenType::ObjectStart => self.exec_object(tkns, node),
                JsonTokenType::ArrayStart => todo!(),
                JsonTokenType::String => todo!(),
                JsonTokenType::EscString => todo!(),
                JsonTokenType::Integer => todo!(),
                JsonTokenType::Number => todo!(),
                JsonTokenType::Null => todo!(),
                JsonTokenType::True => todo!(),
                JsonTokenType::False => todo!(),
                _ => Err(ScanError::UnexpectedToken),
            },
            Err(e) => Err(ScanError::TokenizerError(e)),
        }
    }

    pub fn run(&mut self, input: &[u8]) -> ScanResult {
        let mut tkns = JsonTokenizer::new(input);
        self.exec(&mut tkns, &self.def.parse_node)
    }
}
