//! Single-pass expression matching against raw JSON.
//!
//! `jsonsm` evaluates a compiled boolean expression against a JSON document in **one
//! tokenizing scan**, with no allocation on the hot path and no intermediate parse tree. The
//! scan walks the document and the compiled expression in lockstep: fields the expression
//! names are matched, everything else is crossed structurally, and the scan stops as soon as
//! the result is decided.
//!
//! ```
//! use jsonsm::compile::{compile, Projection};
//! use jsonsm::collation::DefaultCollation;
//! use jsonsm::matcher::FastMatcher;
//! use jsonsm::ast::{CompareOp, Expr, Field, Literal, PathComponent};
//!
//! let expr = Expr::compare(
//!     CompareOp::Equals,
//!     Expr::Field(Field::root(vec![PathComponent::Key("name".into())])),
//!     Expr::Value(Literal::String("Brett".into())),
//! );
//! let def = compile(&[expr], &Projection::default(), &DefaultCollation)?;
//! let mut matcher = FastMatcher::new(&def);
//!
//! assert!(matcher.matches(br#"{"name": "Brett", "age": 41}"#)?.matched());
//! assert!(!matcher.matches(br#"{"name": "Ada"}"#)?.matched());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Semantics in brief
//!
//! Comparison is **strict**: values of different logical types are never equal, and are
//! ordered by type precedence. Numbers compare exactly, with no epsilon. Strings compare by
//! decoded value, so escaped and literal spellings of the same text are equal.
//!
//! A comparison against a **field the document does not contain** is `UNKNOWN`, not `false`.
//! `UNKNOWN` combines by Kleene's tables and `NOT UNKNOWN` is `UNKNOWN`, so writing `!=` or
//! `NOT` around an absent field cannot turn it into a match. `EXISTS` is the deliberate
//! exception: absence is its answer, so it yields `false`.
//!
//! The engine requires **well-formed JSON**. Regions no expression names are crossed without
//! being parsed, so malformed content there may go undetected; values that are actually
//! compared are always fully tokenized.
//!
//! Full detail lives in `docs/semantics.md` and `docs/limits-and-caveats.md`.
//!
//! # Modules
//!
//! [`ast`] is the expression tree; [`compile`] turns one or more expressions into a
//! [`MatchDef`](compile::MatchDef); [`matcher`] evaluates it. [`collation`] is the extension
//! seam for comparison policy and pattern compilation, [`value`] the runtime value model,
//! [`tokenizer`] the scanner, and [`logic_tree`] the boolean structure that resolves as
//! operations report their results.

// All `unsafe` in this crate lives in `simd`, which carries a narrow
// `#![allow(unsafe_code)]` and documents its invariants. Every other module is still
// rejected outright, and a scalar-only build keeps the stronger, unoverridable `forbid`.
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![cfg_attr(feature = "simd", deny(unsafe_code))]

pub use jsonsm_ast as ast;

pub mod collation;
pub mod compile;
pub mod date;
pub mod func;
pub mod logic_tree;
pub mod matcher;
#[cfg(feature = "simd")]
pub mod simd;
pub mod tokenizer;
pub mod value;
