//! Built-in functions applied to values during matching (`mathRound`, `mathAbs`, …).
//!
//! [`apply`] dispatches by name over already-resolved argument values and returns a new
//! value. It is shared by the fast engine and the reference oracle so both evaluate
//! functions identically.
//!
//! Numeric functions operate in `f64` and return a [`FastVal::Float`] (a deliberate
//! simplification over gojsonsm's per-type integer handling; the result still compares
//! exactly against integer constants via the numeric collation). A function given the
//! wrong arity or a non-numeric argument, or an unknown name, returns
//! [`FastVal::Missing`] — so a comparison against it takes the collation's missing result
//! rather than a spurious ordering. Division/modulo by zero likewise yields `Missing`.
//!
//! Function names are gojsonsm's internal identifiers (e.g. `mathSubract`,
//! `mathMultiply`); front-ends map their surface syntax (`-`, `*`, `ABS`, …) to these.

use crate::value::FastVal;

/// Apply the named function to `args`, returning the result value.
pub fn apply(name: &str, args: &[FastVal<'_>]) -> FastVal<'static> {
    // Numeric accessor for argument `i`.
    let num = |i: usize| -> Option<f64> { args.get(i)?.as_num().map(|n| n.as_f64()) };
    let one = |f: fn(f64) -> f64| -> FastVal<'static> {
        match num(0) {
            Some(x) if args.len() == 1 => finite(f(x)),
            _ => FastVal::Missing,
        }
    };
    let two = |f: fn(f64, f64) -> f64| -> FastVal<'static> {
        match (num(0), num(1)) {
            (Some(a), Some(b)) if args.len() == 2 => finite(f(a, b)),
            _ => FastVal::Missing,
        }
    };

    match name {
        // DATE(str): parse an ISO-8601 date to epoch seconds so date comparisons are
        // numeric. A non-string or unparseable argument yields Missing.
        "date" => match args {
            [arg] => arg
                .as_str()
                .and_then(|s| {
                    std::str::from_utf8(&s.to_decoded_bytes())
                        .ok()
                        .and_then(crate::date::parse_epoch)
                })
                .map_or(FastVal::Missing, FastVal::Float),
            _ => FastVal::Missing,
        },

        // Zero-argument constants.
        "mathPi" if args.is_empty() => FastVal::Float(std::f64::consts::PI),
        "mathE" if args.is_empty() => FastVal::Float(std::f64::consts::E),

        // One-argument.
        "mathAbs" => one(f64::abs),
        "mathAcos" => one(f64::acos),
        "mathAsin" => one(f64::asin),
        "mathAtan" => one(f64::atan),
        "mathCeil" => one(f64::ceil),
        "mathCos" => one(f64::cos),
        "mathDegrees" => one(f64::to_degrees),
        "mathExp" => one(f64::exp),
        "mathFloor" => one(f64::floor),
        "mathLn" => one(f64::ln),
        "mathLog" => one(f64::log10),
        "mathRadians" => one(f64::to_radians),
        "mathRound" => one(f64::round),
        "mathSin" => one(f64::sin),
        "mathSqrt" => one(f64::sqrt),
        "mathTan" => one(f64::tan),
        "mathNegate" => one(|x| -x),

        // Two-argument.
        "mathAtan2" => two(f64::atan2),
        "mathPow" => two(f64::powf),
        "mathAdd" => two(|a, b| a + b),
        "mathSubract" => two(|a, b| a - b),
        "mathMultiply" => two(|a, b| a * b),
        "mathDivide" => two(|a, b| if b == 0.0 { f64::NAN } else { a / b }),
        "mathModulo" => two(|a, b| if b == 0.0 { f64::NAN } else { a % b }),

        _ => FastVal::Missing,
    }
}

/// Wrap a computed float as a value, mapping non-finite results (NaN/∞ from domain errors
/// or division by zero) to `Missing`.
#[inline]
fn finite(x: f64) -> FastVal<'static> {
    if x.is_finite() {
        FastVal::Float(x)
    } else {
        FastVal::Missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    fn f(name: &str, args: &[FastVal<'_>]) -> FastVal<'static> {
        apply(name, args)
    }

    fn approx(v: &FastVal<'_>, expected: f64) -> bool {
        match v.as_num() {
            Some(n) => (n.as_f64() - expected).abs() < 1e-9,
            None => false,
        }
    }

    #[test]
    fn one_arg_math() {
        assert!(approx(&f("mathAbs", &[FastVal::Int(-3)]), 3.0));
        assert!(approx(&f("mathRound", &[FastVal::Float(37.42)]), 37.0));
        assert!(approx(&f("mathRound", &[FastVal::Float(2.5)]), 3.0));
        assert!(approx(&f("mathFloor", &[FastVal::Float(2.9)]), 2.0));
        assert!(approx(&f("mathCeil", &[FastVal::Float(2.1)]), 3.0));
        assert!(approx(&f("mathSqrt", &[FastVal::Int(16)]), 4.0));
        assert!(approx(&f("mathNegate", &[FastVal::Int(5)]), -5.0));
    }

    #[test]
    fn two_arg_math() {
        assert!(approx(
            &f("mathAdd", &[FastVal::Int(2), FastVal::Int(3)]),
            5.0
        ));
        assert!(approx(
            &f("mathSubract", &[FastVal::Int(2), FastVal::Int(3)]),
            -1.0
        ));
        assert!(approx(
            &f("mathMultiply", &[FastVal::Int(4), FastVal::Float(2.5)]),
            10.0
        ));
        assert!(approx(
            &f("mathDivide", &[FastVal::Int(9), FastVal::Int(2)]),
            4.5
        ));
        assert!(approx(
            &f("mathPow", &[FastVal::Int(2), FastVal::Int(10)]),
            1024.0
        ));
        assert!(approx(
            &f("mathModulo", &[FastVal::Int(7), FastVal::Int(3)]),
            1.0
        ));
    }

    #[test]
    fn zero_arg_constants() {
        assert!(approx(&f("mathPi", &[]), std::f64::consts::PI));
        assert!(approx(&f("mathE", &[]), std::f64::consts::E));
    }

    #[test]
    fn invalid_inputs_yield_missing() {
        // wrong arity
        assert!(matches!(f("mathAbs", &[]), FastVal::Missing));
        assert!(matches!(f("mathAdd", &[FastVal::Int(1)]), FastVal::Missing));
        // non-numeric argument
        assert!(matches!(
            f("mathAbs", &[FastVal::Bool(true)]),
            FastVal::Missing
        ));
        // division / modulo by zero
        assert!(matches!(
            f("mathDivide", &[FastVal::Int(1), FastVal::Int(0)]),
            FastVal::Missing
        ));
        assert!(matches!(
            f("mathModulo", &[FastVal::Int(1), FastVal::Int(0)]),
            FastVal::Missing
        ));
        // unknown function
        assert!(matches!(f("nope", &[FastVal::Int(1)]), FastVal::Missing));
        // domain error (sqrt of negative -> NaN -> Missing)
        assert!(matches!(
            f("mathSqrt", &[FastVal::Int(-1)]),
            FastVal::Missing
        ));
    }

    #[test]
    fn date_function() {
        use crate::value::FastStr;
        let d = f(
            "date",
            &[FastVal::Str(FastStr::Unescaped(b"2020-01-01T00:00:00Z"))],
        );
        assert!(approx(&d, 1_577_836_800.0));
        // escaped input is decoded first
        let d = f(
            "date",
            &[FastVal::Str(FastStr::Escaped(b"2020-01-01T00:00:00Z"))],
        );
        assert!(approx(&d, 1_577_836_800.0));
        // non-string or unparseable -> Missing
        assert!(matches!(f("date", &[FastVal::Int(5)]), FastVal::Missing));
        assert!(matches!(
            f("date", &[FastVal::Str(FastStr::Unescaped(b"nope"))]),
            FastVal::Missing
        ));
    }

    #[test]
    fn result_compares_exactly_against_int_constant() {
        // mathRound(37.42) == 37 (Float 37.0 vs Int 37).
        let r = f("mathRound", &[FastVal::Float(37.42)]);
        assert_eq!(r.cmp_num(&FastVal::Int(37)), Some(Ordering::Equal));
    }
}
