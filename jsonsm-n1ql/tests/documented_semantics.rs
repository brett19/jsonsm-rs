//! Executable form of the behaviour `docs/semantics.md` promises.
//!
//! That document is the engine's user-facing contract: the Kleene tables, what a quantifier
//! answers over an empty array versus an absent one, and that negation cannot rescue an absent
//! field. Prose has no way of noticing when the code moves out from under it, so every claim
//! it makes that can be reduced to "this expression, on this document, matches or does not" is
//! reduced to one here.
//!
//! Written against the N1QL front end because that is the syntax the document uses, and
//! deliberately as one table rather than many small tests: the value is in the *set* being
//! complete against the document, and a table makes a missing row visible.
//!
//! If a case here fails, one of the two is wrong — fix whichever, but do not fix only the test.

use jsonsm::collation::DefaultCollation;
use jsonsm::compile::Projection;
use jsonsm::matcher::FastMatcher;
use jsonsm_n1ql::compile_str;

/// `(expression, document, expected)`, grouped by the section of `docs/semantics.md` each
/// group pins.
const CASES: &[(&str, &str, bool)] = &[
    // -- Three-valued logic: a comparison against an absent field is UNKNOWN, and no amount
    //    of negation around it produces a match.
    ("age != 50", r#"{"name":"Ada"}"#, false),
    ("NOT (age = 50)", r#"{"name":"Ada"}"#, false),
    ("NOT (NOT (age = 50))", r#"{"name":"Ada"}"#, false),
    ("age IS NOT NULL", r#"{"name":"Ada"}"#, false),
    // -- ...but EXISTS is the deliberate exception: absence is its answer, so it stays
    //    definite and negates normally. This is the only way to select on absence.
    ("age IS MISSING", r#"{"name":"Ada"}"#, true),
    ("age IS NOT MISSING", r#"{"name":"Ada"}"#, false),
    ("age IS MISSING", r#"{"age":50}"#, false),
    // -- The Kleene tables, exercised through a present sibling so the row is unambiguous.
    ("age = 50 OR name = 'Ada'", r#"{"name":"Ada"}"#, true), // unknown OR true  = true
    ("age = 50 OR name = 'Bob'", r#"{"name":"Ada"}"#, false), // unknown OR false = unknown
    ("age = 50 AND name = 'Ada'", r#"{"name":"Ada"}"#, false), // unknown AND true = unknown
    ("age = 50 AND name = 'Bob'", r#"{"name":"Ada"}"#, false), // unknown AND false = false
    // -- Quantifiers over an empty array. EVERY is vacuously true; ANY AND EVERY is not,
    //    because it additionally requires the array to be non-empty.
    ("ANY t IN xs SATISFIES t = 1 END", r#"{"xs":[]}"#, false),
    ("EVERY t IN xs SATISFIES t = 1 END", r#"{"xs":[]}"#, true),
    ("ANY AND EVERY t IN xs SATISFIES t = 1 END", r#"{"xs":[]}"#, false),
    // -- An absent array is not an empty array. EVERY over `[]` is true and negates to false;
    //    EVERY over a field that is not there is UNKNOWN and negates to UNKNOWN.
    ("EVERY t IN xs SATISFIES t = 1 END", r#"{"other":1}"#, false),
    ("NOT (EVERY t IN xs SATISFIES t = 1 END)", r#"{"xs":[]}"#, false),
    ("NOT (EVERY t IN xs SATISFIES t = 1 END)", r#"{"other":1}"#, false),
    // -- Nor is a present-but-not-an-array target.
    ("EVERY t IN xs SATISFIES t = 1 END", r#"{"xs":5}"#, false),
    ("ANY t IN xs SATISFIES t = 1 END", r#"{"xs":5}"#, false),
    // -- An element the body cannot evaluate is UNKNOWN for that element: it does not end the
    //    loop, but it denies the loop the verdict it would otherwise reach.
    ("ANY t IN xs SATISFIES t.a = 1 END", r#"{"xs":[{"b":1},{"a":1}]}"#, true),
    ("ANY t IN xs SATISFIES t.a = 1 END", r#"{"xs":[{"b":1},{"a":2}]}"#, false),
    ("EVERY t IN xs SATISFIES t.a = 1 END", r#"{"xs":[{"a":1},{"b":2}]}"#, false),
    // -- Comparison is strict: different logical types are never equal, whatever their
    //    spelling. Numbers compare exactly and across representations.
    ("n = '5'", r#"{"n":5}"#, false),
    ("n = 5", r#"{"n":"5"}"#, false),
    ("n = 5", r#"{"n":5.0}"#, true),
    ("n = 1", r#"{"n":true}"#, false),
    ("n IS NULL", r#"{"n":null}"#, true),
    ("n IS NULL", r#"{"other":1}"#, false), // null is a value; absence is not it
    // -- Strings compare by decoded value, so escaped and literal spellings are equal.
    (r#"s = 'a/b'"#, r#"{"s":"a\/b"}"#, true),
    (r#"s = 'ab'"#, r#"{"s":"ab"}"#, true),
    // -- A path running through a scalar is absent, not an error.
    ("a.x = 1", r#"{"a":5}"#, false),
    ("a.x IS MISSING", r#"{"a":5}"#, true),
    // -- Array indices are zero-based, and are distinct from an object key of the same text.
    ("xs[0] = 7", r#"{"xs":[7,8]}"#, true),
    ("xs[1] = 7", r#"{"xs":[7,8]}"#, false),
    // -- `EXISTS` on a field from an *enclosing* scope. This reaches absence by a different
    //    route than the cases above: a current-scope field that is absent never runs its op
    //    at all and is settled by the seal, whereas an enclosing-scope field is read from a
    //    slot that was never filled. Both must yield the same definite `false`, and only the
    //    second exercises the operand path that decides it.
    ("ANY t IN xs SATISFIES name IS MISSING END", r#"{"xs":[1]}"#, true),
    ("ANY t IN xs SATISFIES name IS NOT MISSING END", r#"{"xs":[1]}"#, false),
    ("ANY t IN xs SATISFIES name IS MISSING END", r#"{"xs":[1],"name":"Ada"}"#, false),
    ("ANY t IN xs SATISFIES name IS NOT MISSING END", r#"{"xs":[1],"name":"Ada"}"#, true),
];

#[test]
fn documented_semantics_hold() {
    let mut failures = Vec::new();
    for (expr, doc, want) in CASES {
        let def = compile_str(expr, &Projection::default(), &DefaultCollation)
            .unwrap_or_else(|e| panic!("compiling {expr:?}: {e}"));
        let mut m = FastMatcher::new(&def);
        let got = m
            .matches(doc.as_bytes())
            .unwrap_or_else(|e| panic!("matching {expr:?} against {doc}: {e}"))
            .matched();
        if got != *want {
            failures.push(format!(
                "  {expr}\n    on {doc}\n    documented: {want}, engine: {got}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} case(s) disagree with docs/semantics.md:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A loop variable shadowed by a nested loop of the same name resolves to the innermost
/// binding — and a loop's `in` expression is resolved in the *enclosing* scope, before its own
/// binding exists.
///
/// Both halves are consequences of how names are bound: the scope stack is searched
/// innermost-first, and a loop pushes its binding only after its `in` expression is resolved.
/// Neither is obvious from reading the grammar, and nothing else exercises a repeated name.
#[test]
fn a_shadowed_loop_variable_resolves_to_the_innermost_binding() {
    let run = |expr: &str, doc: &str| {
        let def = compile_str(expr, &Projection::default(), &DefaultCollation).unwrap();
        FastMatcher::new(&def).matches(doc.as_bytes()).unwrap().matched()
    };
    // The inner `x` shadows the outer one in the body, while `x.ys` — the inner loop's target
    // — still means the *outer* `x`. If the body read the outer binding instead, it would be
    // comparing an object to 1 and could never match.
    let shadowed = "ANY x IN xs SATISFIES ANY x IN x.ys SATISFIES x = 1 END END";
    assert!(run(shadowed, r#"{"xs":[{"ys":[1,2]}]}"#));
    assert!(!run(shadowed, r#"{"xs":[{"ys":[3]}]}"#));
    assert!(run(shadowed, r#"{"xs":[{"ys":[3]},{"ys":[1]}]}"#));

    // Without shadowing, the outer binding stays reachable from the inner body.
    let distinct = "ANY x IN xs SATISFIES ANY y IN x.ys SATISFIES x.k = 1 END END";
    assert!(run(distinct, r#"{"xs":[{"k":1,"ys":[9]}]}"#));
    assert!(!run(distinct, r#"{"xs":[{"k":2,"ys":[9]}]}"#));
}

/// The quantifiers' empty-array defaults, stated as the table the document prints.
///
/// Kept separate from the list above because it is the one place three quantifiers have to be
/// compared against each other rather than each checked in isolation, and because the empty
/// row is the one most easily broken by a change to loop handling.
#[test]
fn quantifiers_over_an_empty_array_differ_as_documented() {
    let run = |expr: &str, doc: &str| {
        let def = compile_str(expr, &Projection::default(), &DefaultCollation).unwrap();
        FastMatcher::new(&def).matches(doc.as_bytes()).unwrap().matched()
    };
    let empty = r#"{"xs":[]}"#;
    let all_true = r#"{"xs":[1,1]}"#;
    let mixed = r#"{"xs":[1,2]}"#;

    for (quantifier, on_empty, on_all_true, on_mixed) in [
        ("ANY", false, true, true),
        ("EVERY", true, true, false),
        ("ANY AND EVERY", false, true, false),
    ] {
        let expr = format!("{quantifier} t IN xs SATISFIES t = 1 END");
        assert_eq!(run(&expr, empty), on_empty, "{quantifier} over []");
        assert_eq!(run(&expr, all_true), on_all_true, "{quantifier} over all-true");
        assert_eq!(run(&expr, mixed), on_mixed, "{quantifier} over mixed");
    }
}
