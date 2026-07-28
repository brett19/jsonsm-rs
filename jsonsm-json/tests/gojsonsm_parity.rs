//! Behavioral parity with gojsonsm: the same expressions and the same `people.json` as
//! gojsonsm's `fastMatcher_test.go`, asserting the same set of matched document `_id`s
//! that the Go authors verified. Both engines consume the identical JSON-array expression
//! format, so this is a direct cross-implementation check of the match engine.
//!
//! Cases here are ones where our (corrected-N1QL) semantics are expected to agree with
//! Go. Deliberate divergences (float epsilon, escaped-byte string comparison, implicit
//! cross-type coercion) are covered by other tests and intentionally excluded.

use jsonsm::collation::DefaultCollation;
use jsonsm::compile::Projection;
use jsonsm::matcher::FastMatcher;
use serde_json::Value;
use std::path::Path;

/// (compact record bytes, `_id`) for every person in the fixture.
fn people() -> Vec<(Vec<u8>, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../jsonsm/testdata/people.json");
    let bytes = std::fs::read(path).unwrap();
    let arr: Vec<Value> = serde_json::from_slice(&bytes).unwrap();
    arr.iter()
        .map(|v| {
            let id = v.get("_id").and_then(Value::as_str).unwrap().to_owned();
            (serde_json::to_vec(v).unwrap(), id)
        })
        .collect()
}

/// Compile `expr_json`, match every record, and return the sorted matched `_id`s.
fn matched_ids(expr_json: &str) -> Vec<String> {
    let def = jsonsm_json::compile(expr_json.as_bytes(), &Projection::new(), &DefaultCollation)
        .unwrap_or_else(|e| panic!("compile {expr_json}: {e}"));
    let mut m = FastMatcher::new(&def);
    let mut ids: Vec<String> = people()
        .into_iter()
        .filter(|(rec, _)| m.matches(rec).unwrap().matched())
        .map(|(_, id)| id)
        .collect();
    ids.sort();
    ids
}

fn expect(expr_json: &str, ids: &[&str]) {
    let mut want: Vec<String> = ids.iter().map(|s| (*s).to_owned()).collect();
    want.sort();
    assert_eq!(matched_ids(expr_json), want, "for expression: {expr_json}");
}

#[test]
fn string_equals() {
    expect(
        r#"["equals", ["field","name"], ["value","Daphne Sutton"]]"#,
        &["5b47eb0936ff92a567a0307e"],
    );
}

#[test]
fn numeric_equals() {
    expect(
        r#"["equals", ["field","age"], ["value",25]]"#,
        &["5b47eb091f57571d3c3b1aa1"],
    );
}

#[test]
fn float_equals() {
    expect(
        r#"["equals", ["field","latitude"], ["value",-40.262556]]"#,
        &["5b47eb096b1d911c0b9492fb"],
    );
}

#[test]
fn true_equals() {
    expect(
        r#"["equals", ["field","isActive"], ["value",true]]"#,
        &[
            "5b47eb0936ff92a567a0307e",
            "5b47eb0950e9076fc0aecd52",
            "5b47eb095c3ad73b9925f7f8",
            "5b47eb0962222a37d066e231",
            "5b47eb09996a4154c35b2f98",
            "5b47eb098eee4b4c4330ec64",
        ],
    );
}

#[test]
fn false_equals() {
    expect(
        r#"["equals", ["field","isActive"], ["value",false]]"#,
        &[
            "5b47eb096b1d911c0b9492fb",
            "5b47eb093771f06ced629663",
            "5b47eb09ffac5a6ce37042e7",
            "5b47eb091f57571d3c3b1aa1",
        ],
    );
}

#[test]
fn not_true_equals() {
    expect(
        r#"["not", ["equals", ["field","isActive"], ["value",true]]]"#,
        &[
            "5b47eb096b1d911c0b9492fb",
            "5b47eb093771f06ced629663",
            "5b47eb09ffac5a6ce37042e7",
            "5b47eb091f57571d3c3b1aa1",
        ],
    );
}

#[test]
fn exists() {
    expect(
        r#"["exists", ["field","sometimesValue"]]"#,
        &[
            "5b47eb0936ff92a567a0307e",
            "5b47eb096b1d911c0b9492fb",
            "5b47eb0950e9076fc0aecd52",
        ],
    );
}

#[test]
fn not_exists() {
    expect(
        r#"["notexists", ["field","sometimesValue"]]"#,
        &[
            "5b47eb093771f06ced629663",
            "5b47eb09ffac5a6ce37042e7",
            "5b47eb095c3ad73b9925f7f8",
            "5b47eb0962222a37d066e231",
            "5b47eb09996a4154c35b2f98",
            "5b47eb091f57571d3c3b1aa1",
            "5b47eb098eee4b4c4330ec64",
        ],
    );
}

#[test]
fn missing_string_equals_never_matches() {
    expect(
        r#"["equals", ["field","someValueWhichNeverExists"], ["value","hello"]]"#,
        &[],
    );
}

#[test]
fn any_in_equals() {
    expect(
        r#"["anyin", 1, ["field","tags"], ["equals", ["field",1], ["value","cillum"]]]"#,
        &[
            "5b47eb0936ff92a567a0307e",
            "5b47eb09ffac5a6ce37042e7",
            "5b47eb095c3ad73b9925f7f8",
        ],
    );
}

#[test]
fn nested_any_in() {
    expect(
        r#"["anyin", 1, ["field","nestedArray"],
            ["anyin", 2, ["field",1],
                ["equals", ["field",2], ["value","g"]]]]"#,
        &["5b47eb0936ff92a567a0307e"],
    );
}

#[test]
fn every_in_equals() {
    expect(
        r#"["everyin", 1, ["field","testArray"],
            ["equals", ["field",1], ["value","jewels"]]]"#,
        &["5b47eb0936ff92a567a0307e", "5b47eb09ffac5a6ce37042e7"],
    );
}

#[test]
fn any_every_in_equals() {
    expect(
        r#"["anyeveryin", 1, ["field","testArray"],
            ["equals", ["field",1], ["value","jewels"]]]"#,
        &["5b47eb0936ff92a567a0307e"],
    );
}

#[test]
fn cross_scope_loop() {
    expect(
        r#"["anyin", 1, ["field","friends"],
            ["equals", ["field",1,"id"], ["field","index"]]]"#,
        &[
            "5b47eb0936ff92a567a0307e",
            "5b47eb096b1d911c0b9492fb",
            "5b47eb0950e9076fc0aecd52",
        ],
    );
}

#[test]
fn equals_func() {
    expect(
        r#"["equals", ["func","mathRound",["field","latitude"]], ["value",37]]"#,
        &["5b47eb093771f06ced629663"],
    );
}
