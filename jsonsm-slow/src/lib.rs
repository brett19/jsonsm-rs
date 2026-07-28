//! A simple, allocation-happy reference matcher used as a correctness *oracle* for the
//! fast [`jsonsm`] engine.
//!
//! It walks the [`Expr`] AST directly over a parsed [`serde_json::Value`] document, using
//! the exact same [`Collation`] the engine uses (reused from `jsonsm`, not duplicated) —
//! so its results define the behavior the engine's `FastMatcher` must reproduce. It is
//! written for obvious correctness, not speed.
//!
//! Semantics of note (shared with the engine):
//! - comparisons use [`Collation::compare`]; a **missing** operand yields
//!   [`Tri::Unknown`] — a third logical value, propagated by Kleene's tables and collapsed to
//!   "no match" only at the root;
//! - negation is structural: `NotEquals` is `NOT (Equals)` and `NotExists` is
//!   `NOT (Exists)`, so a missing operand under `!=` / `IS NOT MISSING` inverts the
//!   default to `true`;
//! - `null` is a present, orderable value (only *absent* fields are "missing").

#![forbid(unsafe_code)]

use jsonsm::collation::{Collation, CollationError, DefaultCollation, ValueMatcher};
use jsonsm::value::{FastStr, FastVal};
use jsonsm_ast::{CompareOp, Expr, Field, PathComponent, VariableId};
use serde_json::Value;

/// An error from the reference matcher.
#[derive(Debug, thiserror::Error)]
pub enum SlowError {
    #[error("invalid document JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expected a boolean expression, found an operand node")]
    NotABoolean,
    #[error("expected an operand expression, found a boolean node")]
    NotAnOperand,
    #[error("pattern operand must be a string")]
    NonStringPattern,
    #[error(transparent)]
    Collation(#[from] CollationError),
}

/// A reference matcher over a single expression.
/// A three-valued logical result: `Unknown` is what a comparison against an absent field
/// yields.
///
/// Defined here rather than borrowed from `jsonsm` on purpose. The tables below *are* the
/// semantics under test, so importing the engine's version would make the differential sweep
/// compare an implementation against itself — the same trap as sharing a comparison primitive.
/// Three variants and one negation is a small price for the oracle staying independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tri {
    True,
    False,
    Unknown,
}

impl From<bool> for Tri {
    fn from(b: bool) -> Self {
        if b {
            Tri::True
        } else {
            Tri::False
        }
    }
}

/// Kleene negation: the definite values swap; `Unknown` is unchanged.
impl std::ops::Not for Tri {
    type Output = Tri;
    fn not(self) -> Self {
        match self {
            Tri::True => Tri::False,
            Tri::False => Tri::True,
            Tri::Unknown => Tri::Unknown,
        }
    }
}

pub struct SlowMatcher<C = DefaultCollation> {
    expr: Expr,
    collation: C,
}

impl SlowMatcher<DefaultCollation> {
    /// Build a reference matcher using [`DefaultCollation`].
    pub fn new(expr: Expr) -> Self {
        SlowMatcher {
            expr,
            collation: DefaultCollation,
        }
    }
}

impl<C: Collation> SlowMatcher<C> {
    /// Build a reference matcher with an explicit collation.
    pub fn with_collation(expr: Expr, collation: C) -> Self {
        SlowMatcher { expr, collation }
    }

    /// Match against a parsed JSON document.
    ///
    /// The expression is evaluated three-valued and collapsed here, at the root: only `True`
    /// matches, so an `Unknown` result — some comparison naming a field the document lacks —
    /// reads as no match, exactly like `False`. The collapse happens *only* here; everywhere
    /// below, `Unknown` stays distinct so negation cannot turn it into a match.
    pub fn matches(&self, doc: &Value) -> Result<bool, SlowError> {
        let mut env: Env<'_> = Vec::new();
        Ok(self.eval(&self.expr, doc, &mut env)? == Tri::True)
    }

    /// Parse `doc` as JSON and match against it.
    pub fn matches_bytes(&self, doc: &[u8]) -> Result<bool, SlowError> {
        let value: Value = serde_json::from_slice(doc)?;
        self.matches(&value)
    }

    /// Evaluate an expression to a three-valued result.
    ///
    /// Deliberately structured as a plain recursive walk with Kleene's tables written out
    /// inline: the fast engine reaches these answers through a flat logic tree that resolves
    /// nodes as ops report and seals absent fields at container boundaries, and the differential
    /// sweep is only worth anything if the two arrive by genuinely different routes.
    fn eval<'v>(&self, e: &Expr, doc: &'v Value, env: &mut Env<'v>) -> Result<Tri, SlowError> {
        match e {
            Expr::True => Ok(Tri::True),
            Expr::False => Ok(Tri::False),
            // The one that matters: `Unknown` is a fixed point of negation.
            Expr::Not(sub) => Ok(!self.eval(sub, doc, env)?),
            // `And`/`Or` short-circuit only on their absorbing value, and an `Unknown` seen
            // along the way is remembered because it denies the *other* verdict: an `And` whose
            // remaining operands are all true is still only `Unknown`.
            Expr::And(subs) => {
                let mut unknown = false;
                for s in subs {
                    match self.eval(s, doc, env)? {
                        Tri::False => return Ok(Tri::False),
                        Tri::Unknown => unknown = true,
                        Tri::True => {}
                    }
                }
                Ok(if unknown { Tri::Unknown } else { Tri::True }) // empty AND is vacuously true
            }
            Expr::Or(subs) => {
                let mut unknown = false;
                for s in subs {
                    match self.eval(s, doc, env)? {
                        Tri::True => return Ok(Tri::True),
                        Tri::Unknown => unknown = true,
                        Tri::False => {}
                    }
                }
                Ok(if unknown { Tri::Unknown } else { Tri::False }) // empty OR is false
            }
            // Presence questions are always answerable — absence *is* their answer — so these
            // two are the only way an absent field yields a definite result.
            Expr::Exists(sub) => Ok(Tri::from(!self.resolve(sub, doc, env)?.is_missing())),
            Expr::NotExists(sub) => Ok(Tri::from(self.resolve(sub, doc, env)?.is_missing())),
            Expr::Compare { op, lhs, rhs } => self.eval_compare(*op, lhs, rhs, doc, env),
            Expr::Matches { lhs, pattern } => self.eval_matches(lhs, pattern, doc, env),
            Expr::Loop {
                loop_type,
                var,
                in_expr,
                sub_expr,
            } => self.eval_loop(*loop_type, *var, in_expr, sub_expr, doc, env),
            // Operand nodes are not booleans.
            Expr::Value(_) | Expr::Field(_) | Expr::Func(_) => Err(SlowError::NotABoolean),
        }
    }

    fn eval_compare<'v>(
        &self,
        op: CompareOp,
        lhs: &Expr,
        rhs: &Expr,
        doc: &'v Value,
        env: &mut Env<'v>,
    ) -> Result<Tri, SlowError> {
        // `!=` is the negation of `==`, and under three-valued logic that lowering is exactly
        // right: a missing operand makes both `Unknown`, so neither direction manufactures a
        // match out of an absent field.
        if op == CompareOp::NotEquals {
            return Ok(!self.eval_compare(CompareOp::Equals, lhs, rhs, doc, env)?);
        }

        let l = self.resolve(lhs, doc, env)?;
        let r = self.resolve(rhs, doc, env)?;
        // Not an ordering question with an awkward answer — there is no value to order.
        if l.is_missing() || r.is_missing() {
            return Ok(Tri::Unknown);
        }

        use std::cmp::Ordering::*;
        let ord = self
            .collation
            .compare(&l.as_fastval(), &r.as_fastval())
            .ordering;
        Ok(Tri::from(match op {
            CompareOp::Equals => ord == Equal,
            CompareOp::LessThan => ord == Less,
            CompareOp::LessEquals => ord != Greater,
            CompareOp::GreaterThan => ord == Greater,
            CompareOp::GreaterEquals => ord != Less,
            CompareOp::NotEquals => unreachable!("handled above"),
        }))
    }

    fn eval_matches<'v>(
        &self,
        lhs: &Expr,
        pattern: &Expr,
        doc: &'v Value,
        env: &mut Env<'v>,
    ) -> Result<Tri, SlowError> {
        let l = self.resolve(lhs, doc, env)?;
        if l.is_missing() {
            return Ok(Tri::Unknown);
        }
        let p = self.resolve(pattern, doc, env)?;
        let pattern_str = match &p {
            Owned::Str(s) => s.as_str(),
            _ => return Err(SlowError::NonStringPattern),
        };
        let matcher: Box<dyn ValueMatcher> = self.collation.compile_matcher(pattern_str)?;
        Ok(Tri::from(matcher.matches(&l.as_fastval())))
    }

    fn eval_loop<'v>(
        &self,
        loop_type: jsonsm_ast::LoopType,
        var: VariableId,
        in_expr: &Expr,
        sub_expr: &Expr,
        doc: &'v Value,
        env: &mut Env<'v>,
    ) -> Result<Tri, SlowError> {
        use jsonsm_ast::LoopType::*;

        // The `in` operand must resolve to an array. An absent field is not an empty array:
        // there is nothing to quantify over, so the loop is unanswerable rather than false.
        // A value that is present but not an array is a type error for the quantifier, which
        // has no answer either.
        let Some(Value::Array(items)) = self.resolve_field_value(in_expr, doc, env) else {
            return Ok(Tri::Unknown);
        };

        // A quantifier is a connective over elements, so it takes Kleene's tables too: `Any` is
        // an OR (absorbing `True`), `Every` an AND (absorbing `False`), and an element that
        // could not be evaluated leaves the quantifier `Unknown` unless some other element
        // settles it outright.
        let mut unknown = false;
        let mut saw_true = false;
        for item in items {
            env.push((var, item));
            let matched = self.eval(sub_expr, doc, env);
            env.pop();
            match matched? {
                Tri::True => {
                    if loop_type == Any {
                        return Ok(Tri::True);
                    }
                    saw_true = true;
                }
                Tri::False => {
                    if loop_type == Every || loop_type == AnyEvery {
                        return Ok(Tri::False);
                    }
                }
                Tri::Unknown => unknown = true,
            }
        }
        if unknown {
            return Ok(Tri::Unknown);
        }
        Ok(Tri::from(match loop_type {
            Any => saw_true,
            Every => true, // vacuously true for an empty array
            AnyEvery => !items.is_empty() && saw_true,
        }))
    }

    /// Resolve an operand expression to an owned value; absent fields become
    /// [`Owned::Missing`].
    fn resolve<'v>(&self, e: &Expr, doc: &'v Value, env: &Env<'v>) -> Result<Owned, SlowError> {
        match e {
            Expr::Value(lit) => Ok(Owned::from_literal(lit)),
            Expr::Field(f) => Ok(self
                .resolve_field(f, doc, env)
                .map_or(Owned::Missing, Owned::from_value)),
            Expr::Func(func) => {
                // Resolve args, then apply the shared function implementation so the oracle
                // and the fast engine evaluate functions identically.
                let mut owned_args = Vec::with_capacity(func.args.len());
                for arg in &func.args {
                    owned_args.push(self.resolve(arg, doc, env)?);
                }
                let fvals: Vec<FastVal<'_>> = owned_args.iter().map(Owned::as_fastval).collect();
                Ok(Owned::from_fastval(&jsonsm::func::apply(
                    &func.name, &fvals,
                )))
            }
            _ => Err(SlowError::NotAnOperand),
        }
    }

    /// If `e` is a field reference, return the borrowed document value it points at (used
    /// by loops, which need the live array).
    fn resolve_field_value<'v>(
        &self,
        e: &Expr,
        doc: &'v Value,
        env: &Env<'v>,
    ) -> Option<&'v Value> {
        match e {
            Expr::Field(f) => self.resolve_field(f, doc, env),
            _ => None,
        }
    }

    fn resolve_field<'v>(&self, f: &Field, doc: &'v Value, env: &Env<'v>) -> Option<&'v Value> {
        let mut cur = if f.root == jsonsm_ast::ROOT_VAR {
            doc
        } else {
            *env.iter()
                .rev()
                .find(|(id, _)| *id == f.root)
                .map(|(_, v)| v)?
        };
        for comp in &f.path {
            cur = match comp {
                PathComponent::Key(k) => cur.as_object()?.get(k)?,
                PathComponent::Index(i) => cur.as_array()?.get(*i)?,
            };
        }
        Some(cur)
    }
}

/// The loop-variable environment: (variable id, bound document value), innermost last.
type Env<'v> = Vec<(VariableId, &'v Value)>;

/// An owned resolved operand value. Owning it sidesteps borrow gymnastics; it lends a
/// borrowing [`FastVal`] for the duration of a comparison via [`Owned::as_fastval`].
enum Owned {
    Missing,
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    Str(String),
    Array(Vec<u8>),
    Object(Vec<u8>),
}

impl Owned {
    #[inline]
    fn is_missing(&self) -> bool {
        matches!(self, Owned::Missing)
    }

    fn from_literal(lit: &jsonsm_ast::Literal) -> Owned {
        use jsonsm_ast::Literal;
        match lit {
            Literal::Null => Owned::Null,
            Literal::Bool(b) => Owned::Bool(*b),
            Literal::Int(i) => Owned::Int(*i),
            Literal::Uint(u) => Owned::Uint(*u),
            Literal::Float(f) => Owned::Float(*f),
            Literal::String(s) => Owned::Str(s.clone()),
        }
    }

    /// Convert a (function-result) [`FastVal`] back into an owned value.
    fn from_fastval(v: &FastVal<'_>) -> Owned {
        match v {
            FastVal::Missing => Owned::Missing,
            FastVal::Null => Owned::Null,
            FastVal::Bool(b) => Owned::Bool(*b),
            FastVal::Int(i) => Owned::Int(*i),
            FastVal::Uint(u) => Owned::Uint(*u),
            FastVal::Float(f) => Owned::Float(*f),
            FastVal::IntBytes(_) | FastVal::FloatBytes(_) => match v.as_num() {
                Some(jsonsm::value::Num::I(i)) => Owned::Int(i),
                Some(jsonsm::value::Num::U(u)) => Owned::Uint(u),
                Some(jsonsm::value::Num::F(f)) => Owned::Float(f),
                None => Owned::Missing,
            },
            FastVal::Str(s) => {
                Owned::Str(String::from_utf8_lossy(&s.to_decoded_bytes()).into_owned())
            }
            FastVal::Array(b) => Owned::Array(b.to_vec()),
            FastVal::Object(b) => Owned::Object(b.to_vec()),
        }
    }

    fn from_value(v: &Value) -> Owned {
        match v {
            Value::Null => Owned::Null,
            Value::Bool(b) => Owned::Bool(*b),
            Value::String(s) => Owned::Str(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Owned::Int(i)
                } else if let Some(u) = n.as_u64() {
                    Owned::Uint(u)
                } else {
                    Owned::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            Value::Array(_) => Owned::Array(serde_json::to_vec(v).expect("serialize array")),
            Value::Object(_) => Owned::Object(serde_json::to_vec(v).expect("serialize object")),
        }
    }

    fn as_fastval(&self) -> FastVal<'_> {
        match self {
            Owned::Missing => FastVal::Missing,
            Owned::Null => FastVal::Null,
            Owned::Bool(b) => FastVal::Bool(*b),
            Owned::Int(i) => FastVal::Int(*i),
            Owned::Uint(u) => FastVal::Uint(*u),
            Owned::Float(f) => FastVal::Float(*f),
            Owned::Str(s) => FastVal::Str(FastStr::Unescaped(s.as_bytes())),
            Owned::Array(b) => FastVal::Array(b),
            Owned::Object(b) => FastVal::Object(b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonsm_ast::{Field, Literal, LoopType, PathComponent};

    fn doc(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }
    fn field(keys: &[&str]) -> Expr {
        Expr::Field(Field::root(
            keys.iter()
                .map(|k| PathComponent::Key((*k).to_owned()))
                .collect(),
        ))
    }
    fn m(expr: Expr, d: &Value) -> bool {
        SlowMatcher::new(expr).matches(d).unwrap()
    }

    #[test]
    fn scalar_comparisons() {
        let d = doc(r#"{"age": 30, "name": "Brett", "active": true}"#);
        assert!(m(
            Expr::compare(
                CompareOp::Equals,
                field(&["name"]),
                Expr::Value(Literal::String("Brett".into()))
            ),
            &d
        ));
        assert!(m(
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            &d
        ));
        assert!(!m(
            Expr::compare(
                CompareOp::GreaterThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            &d
        ));
        assert!(m(
            Expr::compare(
                CompareOp::Equals,
                field(&["active"]),
                Expr::Value(Literal::Bool(true))
            ),
            &d
        ));
    }

    #[test]
    fn no_cross_type_coercion() {
        let d = doc(r#"{"n": 5, "s": "5"}"#);
        // 5 == "5" is false under strict N1QL.
        assert!(!m(
            Expr::compare(
                CompareOp::Equals,
                field(&["n"]),
                Expr::Value(Literal::String("5".into()))
            ),
            &d
        ));
    }

    #[test]
    fn missing_field_semantics() {
        let d = doc(r#"{"other": 1}"#);
        // age < 50 on a doc without `age` -> false.
        assert!(!m(
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            &d
        ));
        // age != 50 on a missing age -> false: `==` is Unknown and `NOT Unknown` is Unknown.
        assert!(!m(
            Expr::compare(
                CompareOp::NotEquals,
                field(&["age"]),
                Expr::Value(Literal::Int(50))
            ),
            &d
        ));
        // …while the presence questions stay definite, which is what keeps absence selectable.
        // exists / notexists.
        assert!(!m(Expr::Exists(Box::new(field(&["age"]))), &d));
        assert!(m(Expr::NotExists(Box::new(field(&["age"]))), &d));
        assert!(m(Expr::Exists(Box::new(field(&["other"]))), &d));
    }

    #[test]
    fn null_is_present_and_orderable() {
        let d = doc(r#"{"x": null}"#);
        // null exists (it is a present value, not missing).
        assert!(m(Expr::Exists(Box::new(field(&["x"]))), &d));
        // null == null.
        assert!(m(
            Expr::compare(CompareOp::Equals, field(&["x"]), Expr::Value(Literal::Null)),
            &d
        ));
        // null != 5 (different types).
        assert!(m(
            Expr::compare(
                CompareOp::NotEquals,
                field(&["x"]),
                Expr::Value(Literal::Int(5))
            ),
            &d
        ));
    }

    #[test]
    fn logic_and_or_not() {
        let d = doc(r#"{"a": 1, "b": 2}"#);
        let a_is_1 = Expr::compare(
            CompareOp::Equals,
            field(&["a"]),
            Expr::Value(Literal::Int(1)),
        );
        let b_is_9 = Expr::compare(
            CompareOp::Equals,
            field(&["b"]),
            Expr::Value(Literal::Int(9)),
        );
        assert!(m(Expr::Or(vec![a_is_1.clone(), b_is_9.clone()]), &d));
        assert!(!m(Expr::And(vec![a_is_1.clone(), b_is_9.clone()]), &d));
        assert!(m(Expr::Not(Box::new(b_is_9)), &d));
    }

    #[test]
    fn loops_any_every_anyevery() {
        let d = doc(r#"{"tags": ["a", "cillum", "z"], "empty": []}"#);
        let elem = |lt| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["tags"])),
            sub_expr: Box::new(Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field {
                    root: 1,
                    path: vec![],
                }),
                Expr::Value(Literal::String("cillum".into())),
            )),
        };
        assert!(m(elem(LoopType::Any), &d)); // one element is "cillum"
        assert!(!m(elem(LoopType::Every), &d)); // not all are
        assert!(!m(elem(LoopType::AnyEvery), &d));

        // every over an empty array is vacuously true; any/anyevery are false.
        let empty = |lt| Expr::Loop {
            loop_type: lt,
            var: 1,
            in_expr: Box::new(field(&["empty"])),
            sub_expr: Box::new(Expr::True),
        };
        assert!(m(empty(LoopType::Every), &d));
        assert!(!m(empty(LoopType::Any), &d));
        assert!(!m(empty(LoopType::AnyEvery), &d));
    }

    #[test]
    fn matches_uses_default_regex() {
        let d = doc(r#"{"email": "a@example.com", "n": 5}"#);
        assert!(m(
            Expr::Matches {
                lhs: Box::new(field(&["email"])),
                pattern: Box::new(Expr::Value(Literal::String("@example\\.com$".into()))),
            },
            &d
        ));
        // Non-string / missing operands do not match.
        assert!(!m(
            Expr::Matches {
                lhs: Box::new(field(&["n"])),
                pattern: Box::new(Expr::Value(Literal::String("5".into()))),
            },
            &d
        ));
        assert!(!m(
            Expr::Matches {
                lhs: Box::new(field(&["missing"])),
                pattern: Box::new(Expr::Value(Literal::String("x".into()))),
            },
            &d
        ));
    }
}
