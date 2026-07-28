//! Collation: the compile-time-selectable strategy for comparing values and for
//! compiling pattern matchers.
//!
//! Comparison policy is a property of the [`Collation`] in force, not hard-wired into the
//! engine. [`DefaultCollation`] implements **strict N1QL** semantics: values of different
//! logical types are never equal and are ordered purely by type precedence
//! (`missing < null < boolean < number < string < array < object`); values of the same
//! type are compared by value.
//!
//! Two things are *not* a collation's choice. Cross-type equality is fixed: `5` never equals
//! `"5"`, whatever collation is in force, so a collation may decide how values of the same type
//! order but not that values of different types are interchangeable. And **absence** is not a
//! comparison outcome at all — a comparison against a missing field yields
//! [`Unknown`](crate::logic_tree), a third logical value the engine propagates, rather than a
//! boolean the collation declares. See [`crate::logic_tree`] for why that distinction is what
//! stops `NOT` turning an absent field into a match.
//!
//! Pattern matching (`matches` / `LIKE`) is also a collation concern: [`Collation::compile_matcher`]
//! turns a pattern string into a runtime [`ValueMatcher`]. [`DefaultCollation`] supports
//! this out of the box using the standard [`regex`] crate (unanchored "contains"
//! matching against the *decoded* string value). The trait method still defaults to an
//! error so a minimal custom collation may opt out.

use crate::value::{FastVal, ValueType};
use std::cmp::Ordering;

/// The outcome of comparing two values under a [`Collation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Comparison {
    /// The ordering of the two values.
    pub ordering: Ordering,
    /// Whether this was a meaningful within-type value comparison (`true`), or a
    /// cross-type result resolved by type precedence (`false`). A `false` here is what the
    /// matcher surfaces as "collation was used".
    pub within_type: bool,
}

impl Comparison {
    #[inline]
    fn within(ordering: Ordering) -> Self {
        Comparison {
            ordering,
            within_type: true,
        }
    }

    #[inline]
    fn cross(ordering: Ordering) -> Self {
        Comparison {
            ordering,
            within_type: false,
        }
    }
}

/// A compiled runtime matcher (e.g. a regex), produced by [`Collation::compile_matcher`]
/// and invoked by the `matches` operator. Kept behind a trait object so the engine stays
/// non-generic and the per-byte scan path never sees dynamic dispatch.
pub trait ValueMatcher: Send + Sync + std::fmt::Debug {
    /// Whether `value` matches the compiled pattern.
    fn matches(&self, value: &FastVal<'_>) -> bool;
}

/// An error produced while configuring a collation (e.g. compiling a pattern).
#[derive(Debug, thiserror::Error)]
pub enum CollationError {
    #[error("this collation does not support pattern matching")]
    MatcherUnsupported,
    #[error("invalid pattern: {0}")]
    InvalidPattern(String),
}

/// The comparison + pattern-compilation strategy supplied when an expression is compiled.
pub trait Collation {
    /// Compare two runtime values.
    fn compare(&self, a: &FastVal<'_>, b: &FastVal<'_>) -> Comparison;

    /// Whether two values are equal under this collation.
    #[inline]
    fn equals(&self, a: &FastVal<'_>, b: &FastVal<'_>) -> bool {
        self.compare(a, b).ordering == Ordering::Equal
    }

    /// Compile a pattern string into a runtime matcher for the `matches` operator.
    ///
    /// The default implementation reports [`CollationError::MatcherUnsupported`]; a
    /// regex-backed collation overrides it.
    fn compile_matcher(&self, pattern: &str) -> Result<Box<dyn ValueMatcher>, CollationError> {
        let _ = pattern;
        Err(CollationError::MatcherUnsupported)
    }
}

/// Strict-N1QL collation: no cross-type coercion.
///
/// - Different logical types → ordered by [`ValueType`] precedence, `within_type = false`.
/// - Same type → compared by value, `within_type = true`:
///   - `missing`/`null`: equal to themselves;
///   - booleans: `false < true`;
///   - numbers: exact numeric comparison (no epsilon);
///   - strings: logical (decoded) codepoint order.
///
/// Array/object comparison is a deterministic comparison of the raw JSON bytes, matching
/// gojsonsm (see the note in [`DefaultCollation::compare`]) — so container equality is
/// byte-exact and whitespace/key-order sensitive.
///
/// Pattern matching is backed by the standard [`regex`] crate.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultCollation;

/// A [`ValueMatcher`] backed by a compiled [`regex::Regex`]. Matches (unanchored) against
/// the decoded text of string values; non-string values never match.
#[derive(Debug)]
struct RegexMatcher {
    re: regex::Regex,
}

impl ValueMatcher for RegexMatcher {
    fn matches(&self, value: &FastVal<'_>) -> bool {
        match value.as_str() {
            Some(s) => {
                let bytes = s.to_decoded_bytes();
                // Decoded JSON strings are UTF-8; anything else cannot match.
                std::str::from_utf8(&bytes).is_ok_and(|text| self.re.is_match(text))
            }
            None => false,
        }
    }
}

impl Collation for DefaultCollation {
    fn compile_matcher(&self, pattern: &str) -> Result<Box<dyn ValueMatcher>, CollationError> {
        let re = regex::Regex::new(pattern)
            .map_err(|e| CollationError::InvalidPattern(e.to_string()))?;
        Ok(Box::new(RegexMatcher { re }))
    }

    #[inline(always)]
    fn compare(&self, a: &FastVal<'_>, b: &FastVal<'_>) -> Comparison {
        // Two strings, asked directly. The general route below reaches the same answer, but by
        // way of `value_type` on each operand, a comparison of the two types, a match that
        // arrives back at `String`, and `as_str` on each to recover what the discriminants
        // already said — five enum inspections to conclude what one pattern states. It is the
        // single most common comparison an expression makes, and this changes nothing about
        // what it means.
        if let (FastVal::Str(x), FastVal::Str(y)) = (a, b) {
            return Comparison::within(x.cmp_str(y));
        }
        // There is deliberately no numeric arm beside it. This function is
        // `#[inline(always)]`, so every arm it gains is emitted at every call site whether or
        // not that site can reach it — and a numeric arm, worth a little on a loop over
        // numbers, costs every expression that compares strings or booleans. Such a dispatcher
        // supports about two fast arms before the callers that need neither start paying for
        // both; a third case wants its own call site, not another arm here.
        let (ta, tb) = (a.value_type(), b.value_type());
        if ta != tb {
            return Comparison::cross(ta.cmp(&tb));
        }
        let ordering = match ta {
            // missing == missing, null == null.
            ValueType::Missing | ValueType::Null => Ordering::Equal,
            ValueType::Boolean => a.as_bool().cmp(&b.as_bool()), // false < true
            ValueType::Number => a
                .cmp_num(b)
                .expect("both operands are numeric by value_type"),
            ValueType::String => a
                .as_str()
                .zip(b.as_str())
                .map(|(x, y)| x.cmp_str(y))
                .expect("both operands are strings by value_type"),
            // Containers compare by their raw JSON bytes — a deliberate parity choice, not a
            // stub: gojsonsm does the same (`compareObjArrData` compares length then raw
            // bytes, with its own "need a better way" note), so element-wise N1QL ordering
            // would *diverge* from the reference implementation. The shared consequence is
            // that equality is byte-exact, so `[1,2]` and `[1, 2]` are unequal, and key order
            // matters for objects. We order purely lexicographically rather than Go's
            // length-first, which agrees with Go on equality and differs only in how unequal
            // containers sort. Revisit only as an explicit, tested semantic change.
            ValueType::Array | ValueType::Object => a.container_bytes().cmp(&b.container_bytes()),
        };
        Comparison::within(ordering)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::FastStr;

    fn s(v: &str) -> FastVal<'_> {
        FastVal::Str(FastStr::Unescaped(v.as_bytes()))
    }

    #[test]
    fn cross_type_ordering_follows_type_precedence() {
        let c = DefaultCollation;
        // missing < null < bool < number < string < array < object
        let missing = FastVal::Missing;
        let null = FastVal::Null;
        let boolean = FastVal::Bool(true);
        let number = FastVal::Int(1);
        let string = s("z");
        let array = FastVal::Array(b"[]");
        let object = FastVal::Object(b"{}");

        let ordered = [&missing, &null, &boolean, &number, &string, &array, &object];
        for i in 0..ordered.len() {
            for j in 0..ordered.len() {
                let r = c.compare(ordered[i], ordered[j]);
                assert_eq!(r.ordering, i.cmp(&j), "types {i} vs {j}");
                // Only equal-index pairs here are same-type (and each is a singleton type).
                assert_eq!(r.within_type, i == j, "within_type {i} vs {j}");
            }
        }
    }

    #[test]
    fn no_cross_type_coercion() {
        let c = DefaultCollation;
        // 5 vs "5": not equal; number < string.
        let five = FastVal::Int(5);
        let five_str = s("5");
        assert!(!c.equals(&five, &five_str));
        assert_eq!(c.compare(&five, &five_str).ordering, Ordering::Less);
        assert!(!c.compare(&five, &five_str).within_type);
        // true vs 1: not equal (bool < number).
        assert!(!c.equals(&FastVal::Bool(true), &FastVal::Int(1)));
    }

    #[test]
    fn missing_and_null_are_orderable() {
        let c = DefaultCollation;
        assert!(c.equals(&FastVal::Missing, &FastVal::Missing));
        assert!(c.equals(&FastVal::Null, &FastVal::Null));
        assert!(!c.equals(&FastVal::Missing, &FastVal::Null));
        assert_eq!(
            c.compare(&FastVal::Missing, &FastVal::Null).ordering,
            Ordering::Less
        );
        // Same-type missing/null comparisons are "within type".
        assert!(c.compare(&FastVal::Null, &FastVal::Null).within_type);
    }

    #[test]
    fn within_type_value_comparisons() {
        let c = DefaultCollation;
        // booleans: false < true
        assert_eq!(
            c.compare(&FastVal::Bool(false), &FastVal::Bool(true))
                .ordering,
            Ordering::Less
        );
        // numbers: exact, lazy vs parsed
        assert!(c.equals(&FastVal::IntBytes(b"42"), &FastVal::Int(42)));
        assert_eq!(
            c.compare(&FastVal::Float(1.5), &FastVal::Int(2)).ordering,
            Ordering::Less
        );
        // strings: codepoint order
        assert_eq!(
            c.compare(&s("apple"), &s("banana")).ordering,
            Ordering::Less
        );
        assert!(c.equals(&s("café"), &FastVal::Str(FastStr::Escaped(b"caf\\u00e9"))));
    }

    #[test]
    fn default_collation_matches_regex_by_default() {
        let c = DefaultCollation;
        let m = c.compile_matcher("^h.llo").expect("valid pattern compiles");

        // Unanchored "contains" semantics against the decoded string value.
        assert!(m.matches(&s("hello world")));
        assert!(m.matches(&s("hallo")));
        assert!(!m.matches(&s("goodbye")));

        // Matches against decoded content: an escaped form is decoded first.
        let contains_e = c.compile_matcher("café").expect("valid");
        assert!(contains_e.matches(&FastVal::Str(FastStr::Escaped(b"a caf\\u00e9 here"))));

        // Non-string values never match.
        assert!(!m.matches(&FastVal::Int(5)));
        assert!(!m.matches(&FastVal::Null));
    }

    #[test]
    fn invalid_regex_reports_error() {
        let c = DefaultCollation;
        assert!(matches!(
            c.compile_matcher("("),
            Err(CollationError::InvalidPattern(_))
        ));
    }

    #[test]
    fn comparison_is_total_and_antisymmetric() {
        let c = DefaultCollation;
        let vals = [
            FastVal::Missing,
            FastVal::Null,
            FastVal::Bool(false),
            FastVal::Bool(true),
            FastVal::Int(-3),
            FastVal::Int(0),
            FastVal::Uint(u64::MAX),
            FastVal::Float(2.5),
            FastVal::FloatBytes(b"2.5"),
            s("a"),
            s("ab"),
            FastVal::Array(b"[1]"),
            FastVal::Object(b"{}"),
        ];
        for a in &vals {
            for b in &vals {
                let ab = c.compare(a, b).ordering;
                let ba = c.compare(b, a).ordering;
                assert_eq!(ab, ba.reverse(), "antisymmetry {a:?} vs {b:?}");
            }
            // Reflexivity.
            assert_eq!(c.compare(a, a).ordering, Ordering::Equal, "reflexive {a:?}");
        }
    }
}
