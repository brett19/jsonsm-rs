#![feature(portable_simd)]
#![feature(slice_pattern)]

pub mod bytesiterator;
pub mod expression;
pub mod fastmatcher;
mod fastmatcher_test;
pub mod jsontokenizer;
mod jsontokenizer_parse;
mod jsontokenizer_skip;
mod jsontokenizer_test;
pub mod jsontokenizer_token;
pub mod jsontokenizerx;
pub mod logictree;
pub mod logictree_node;
pub mod logictree_state;
pub mod logictree_test;
pub mod logictree_validate;
pub mod simdsearch;
pub mod simdsearch_ops;
mod simdsearch_test;
