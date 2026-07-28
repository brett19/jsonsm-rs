//! Differential test: the fast engine (`jsonsm::matcher::FastMatcher`) must agree with the
//! reference oracle (`jsonsm_slow::SlowMatcher`) on every (expression, document) pair.
//!
//! A deterministic generator produces random documents and random expressions drawn from
//! the subset the compiler currently supports (field-vs-constant comparisons, boolean
//! logic, exists, loops, regex matches). Each pair is compiled and run through the fast matcher
//! and run the oracle, and assert identical results. Any divergence is a bug in one side.
//!
//! A second sweep does the same for **field projection**: the values the fast matcher
//! captures during the scan are compared against navigating the same document with
//! `serde_json` — an independent oracle for "what value lives at this path". It also
//! re-checks that adding a projection does not change the match result.

use jsonsm::collation::DefaultCollation;
use jsonsm::compile::{compile, Projection};
use jsonsm::matcher::FastMatcher;
use jsonsm_ast::{CompareOp, Expr, Field, Literal, LoopType, PathComponent};
use jsonsm_slow::SlowMatcher;

/// A matcher per scan backend this CPU supports.
///
/// `FastMatcher` is monomorphised over its `Scan` backend, so each backend is separately
/// compiled code — running only the detected one would leave the others completely
/// unexercised by this sweep. Every case below is therefore checked on *all* of them, and
/// they must agree with each other as well as with the oracle.
fn matchers<'d>(def: &'d jsonsm::compile::MatchDef) -> Vec<(String, FastMatcher<'d>)> {
    build_matchers(def)
}

/// Rotate through the available backends, so a long sweep that builds one matcher per case
/// still covers every monomorphisation without paying for all of them on every case.
fn matcher_for<'d>(def: &'d jsonsm::compile::MatchDef, nth: usize) -> FastMatcher<'d> {
    let mut m = FastMatcher::new(def);
    #[cfg(feature = "simd")]
    {
        let avail = jsonsm::simd::Backend::available();
        m.force_backend(avail[nth % avail.len()]);
    }
    let _ = nth;
    m
}

fn build_matchers<'d>(def: &'d jsonsm::compile::MatchDef) -> Vec<(String, FastMatcher<'d>)> {
    #[cfg(feature = "simd")]
    {
        jsonsm::simd::Backend::available()
            .into_iter()
            .map(|b| {
                let mut m = FastMatcher::new(def);
                m.force_backend(b);
                (format!("{b:?}"), m)
            })
            .collect()
    }
    #[cfg(not(feature = "simd"))]
    {
        vec![("scalar".to_string(), FastMatcher::new(def))]
    }
}
use serde_json::{json, Value};

/// A tiny deterministic PRNG (SplitMix64-ish) so failures reproduce.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }
}

/// The field names expressions may reference, and so also the keys a compiled `KeyMap`
/// holds. `abcdefgh` is long on purpose: quoted it is ten bytes, so it outruns the eight-byte
/// word the matcher's key comparison prefilters on and is the only one of these whose match
/// needs the compare past that word to happen at all. With three one-character names, that
/// code was unreachable and a sweep of 20,000 cases could not see it deleted.
const FIELDS: &[&str] = &["a", "b", "c", "abcdefgh"];
/// The short-string pool, drawn for both document values and expression constants.
///
/// `a\b` earns its place: it is the only entry containing a **backslash**, and a backslash is
/// the only byte for which reading a string as escaped differs from reading it as decoded.
/// Without it, a constant wrongly read as escaped gives the same answer for every string this
/// generator can produce — `long_string`'s newline and quote both decode to themselves — so
/// the bug survives the sweep however many expression shapes reach it.
// Deliberately tiny. Small pools are what make *equal* operands common: with four strings
// and integers drawn from a handful of values, a generated comparison lands on equality often
// enough to exercise the paths that only differ there. Widening them weakens the sweep rather
// than strengthening it — drawing constants from a compared field's own value was tried and
// changed the match rate barely at all, and caught no mutant the plain generator missed.
const STRINGS: &[&str] = &["p", "q", "r", "a\\b"];
const PATTERNS: &[&str] = &["p", "^q", "[pr]", "q$"];

/// Strings long enough to reach the tokenizer's bulk scan path.
///
/// Every other string this generator produces is one byte long, so the SIMD kernels — which
/// only engage on runs of 32+ bytes — were never reached: a deliberately broken AVX2 kernel
/// passed this whole sweep while the `people.json` parity test failed instantly. The pool is
/// small on purpose so equality comparisons between two draws still hit sometimes, and each
/// entry puts a different construct astride the 32-byte boundary.
fn long_string(i: usize) -> String {
    match i % 5 {
        0 => "a".repeat(31),                             // one byte short of a vector
        1 => "a".repeat(32),                             // exactly a vector
        2 => format!("{}\n{}", "b".repeat(30), "b"),     // escape astride the boundary
        3 => format!("{}\u{e9}{}", "c".repeat(31), "c"), // multi-byte UTF-8 astride
        _ => format!("{}\"{}", "d".repeat(33), "d"),     // escaped quote past a block
    }
}

fn gen_scalar(rng: &mut Rng) -> Value {
    match rng.below(8) {
        0 => json!(rng.below(4) as i64 - 1), // -1..2
        1 => json!(rng.below(4) as f64 / 2.0),
        2 => json!(STRINGS[rng.below(STRINGS.len())]),
        3 => json!(rng.chance(2)),
        4 => Value::Null,
        5 => json!(long_string(rng.below(5))),
        6 => json!(rng.below(3) as i64),
        _ => json!(rng.below(3) as i64),
    }
}

/// A document: an object whose fields are sometimes present, as scalars / arrays / a
/// nested object.
/// Keys no expression ever names, whose *shapes* are what the matcher's raw-byte key
/// comparison can get wrong.
///
/// It resolves a key by comparing the document's bytes against each child's quoted key, so
/// what can break it is a document key that a compared key nearly matches. The last two are
/// the pointed ones: quoted, their first eight bytes are *identical* to the field name
/// `abcdefgh`'s, so the word-sized prefilter passes and only the compare beyond it can tell
/// them apart.
const DECOY_KEYS: &[&str] = &[
    "ab",        // `a` is a prefix of this: the closing quote must reject it
    "xy",        // likewise for a sub-object key
    "a0",        //
    "abcdef",    // quoted length exactly 8 — fills the word, nothing beyond it
    "abcdefg",   // shares the whole word with `abcdefgh`, one byte shorter
    "abcdefghi", // shares the whole word with `abcdefgh`, one byte longer
];

/// Sprinkle unreferenced keys into an object. They cost the oracle nothing — it navigates
/// by name — but every one of them is a document key the matcher must decline to match.
fn add_decoys(rng: &mut Rng, map: &mut serde_json::Map<String, Value>) {
    for _ in 0..rng.below(3) {
        map.insert(
            DECOY_KEYS[rng.below(DECOY_KEYS.len())].into(),
            gen_scalar(rng),
        );
    }
}

fn gen_doc(rng: &mut Rng) -> Value {
    let mut map = serde_json::Map::new();
    // A padding field no expression ever references, purely to slide everything that
    // follows it across the tokenizer's 32-byte vector boundaries. `_` sorts before the
    // real field names, so it really does come first in the serialized bytes.
    if !rng.chance(3) {
        map.insert("_pad".into(), json!("z".repeat(rng.below(70))));
    }
    add_decoys(rng, &mut map);
    for &f in FIELDS {
        if rng.chance(4) {
            continue; // sometimes absent
        }
        let v = match rng.below(5) {
            0 => {
                // array of scalars
                let len = rng.below(4);
                Value::Array((0..len).map(|_| gen_scalar(rng)).collect())
            }
            1 => {
                // Array of small objects. Each of `x`/`y` is sometimes *absent*, which is
                // what exercises per-element slot lifetime: a deferred comparison in a loop
                // body must not read a previous element's value for a field this element
                // lacks. (Always emitting both fields hid a real bug here.)
                let len = rng.below(3);
                Value::Array(
                    (0..len)
                        .map(|_| {
                            let mut obj = serde_json::Map::new();
                            if !rng.chance(3) {
                                obj.insert("x".into(), gen_scalar(rng));
                            }
                            if !rng.chance(3) {
                                obj.insert("y".into(), gen_scalar(rng));
                            }
                            add_decoys(rng, &mut obj);
                            Value::Object(obj)
                        })
                        .collect(),
                )
            }
            2 => json!({"x": gen_scalar(rng)}),
            3 => {
                // Array of objects each holding an inner array (`z`) plus a scalar `x`:
                // material for nested loops whose inner body reads an outer scope.
                let len = rng.below(3);
                Value::Array(
                    (0..len)
                        .map(|_| {
                            let mut obj = serde_json::Map::new();
                            if !rng.chance(4) {
                                obj.insert("x".into(), gen_scalar(rng));
                            }
                            let inner = rng.below(3);
                            obj.insert(
                                "z".into(),
                                Value::Array(
                                    (0..inner)
                                        .map(|_| {
                                            if rng.chance(2) {
                                                json!({"x": gen_scalar(rng)})
                                            } else {
                                                gen_scalar(rng)
                                            }
                                        })
                                        .collect(),
                                ),
                            );
                            add_decoys(rng, &mut obj);
                            Value::Object(obj)
                        })
                        .collect(),
                )
            }
            _ => gen_scalar(rng),
        };
        map.insert(f.to_string(), v);
    }
    Value::Object(map)
}

fn field(keys: &[&str]) -> Expr {
    Expr::Field(Field::root(
        keys.iter()
            .map(|k| PathComponent::Key((*k).to_owned()))
            .collect(),
    ))
}

/// `$doc.<key>[<index>]<.sub?>` — an indexed element reference, optionally into a sub-field.
fn indexed_field(key: &str, index: usize, sub: Option<&str>) -> Expr {
    let mut path = vec![
        PathComponent::Key(key.to_owned()),
        PathComponent::Index(index),
    ];
    if let Some(s) = sub {
        path.push(PathComponent::Key(s.to_owned()));
    }
    Expr::Field(Field::root(path))
}

/// A randomly chosen indexed reference into one of the document's array fields.
fn gen_indexed(rng: &mut Rng) -> Expr {
    let key = FIELDS[rng.below(FIELDS.len())];
    let index = rng.below(4); // 0..3, sometimes out of range
    let sub = match rng.below(3) {
        0 => Some("x"),
        1 => Some("y"),
        _ => None,
    };
    indexed_field(key, index, sub)
}

fn gen_const(rng: &mut Rng) -> Expr {
    Expr::Value(match gen_scalar(rng) {
        Value::Null => Literal::Null,
        Value::Bool(b) => Literal::Bool(b),
        Value::String(s) => Literal::String(s),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Literal::Int(i)
            } else {
                Literal::Float(n.as_f64().unwrap())
            }
        }
        _ => Literal::Null,
    })
}

const OPS: &[CompareOp] = &[
    CompareOp::Equals,
    CompareOp::NotEquals,
    CompareOp::LessThan,
    CompareOp::LessEquals,
    CompareOp::GreaterThan,
    CompareOp::GreaterEquals,
];

fn gen_leaf(rng: &mut Rng) -> Expr {
    match rng.below(8) {
        7 => {
            // cross-field comparison (compiles only at the root context; skipped elsewhere),
            // sometimes between two array elements.
            let mk = |rng: &mut Rng| {
                if rng.chance(3) {
                    gen_indexed(rng)
                } else {
                    field(&[FIELDS[rng.below(FIELDS.len())]])
                }
            };
            let lhs = mk(rng);
            let rhs = mk(rng);
            Expr::compare(OPS[rng.below(OPS.len())], lhs, rhs)
        }
        0 => Expr::Exists(Box::new(if rng.chance(3) {
            gen_indexed(rng)
        } else {
            field(&[FIELDS[rng.below(FIELDS.len())]])
        })),
        1 => Expr::NotExists(Box::new(field(&[FIELDS[rng.below(FIELDS.len())]]))),
        2 => Expr::True,
        3 => Expr::False,
        4 => Expr::Matches {
            lhs: Box::new(field(&[FIELDS[rng.below(FIELDS.len())]])),
            pattern: Box::new(Expr::Value(Literal::String(
                PATTERNS[rng.below(PATTERNS.len())].to_owned(),
            ))),
        },
        _ => {
            // field <op> const, occasionally wrapped in a single-arg math function or on
            // a nested path.
            let base = match rng.below(4) {
                0 => field(&["c", "x"]),
                1 => gen_indexed(rng), // array element reference
                _ => field(&[FIELDS[rng.below(FIELDS.len())]]),
            };
            let lhs = match rng.below(4) {
                0 => {
                    const FUNCS: &[&str] = &["mathAbs", "mathRound", "mathFloor", "mathNegate"];
                    Expr::Func(jsonsm_ast::Func {
                        name: FUNCS[rng.below(FUNCS.len())].to_owned(),
                        args: vec![base],
                    })
                }
                1 => {
                    // two-field function argument (deferred via slots + after)
                    let g = field(&[FIELDS[rng.below(FIELDS.len())]]);
                    const FUNCS: &[&str] = &["mathAdd", "mathSubract", "mathMultiply"];
                    Expr::Func(jsonsm_ast::Func {
                        name: FUNCS[rng.below(FUNCS.len())].to_owned(),
                        args: vec![base, g],
                    })
                }
                _ => base,
            };
            let op = OPS[rng.below(OPS.len())];
            let k = gen_const(rng);
            // The compiler preserves operand order (a `Value` on the left stays on the
            // left), and ordering comparisons are not symmetric, so constant-on-the-left is
            // a distinct shape rather than a rewrite of the same one. Nothing else here
            // produces it: a shape census over the whole sweep found `Compare(Const, _)`
            // never compiled once.
            if rng.chance(4) {
                Expr::compare(op, k, lhs)
            } else {
                Expr::compare(op, lhs, k)
            }
        }
    }
}

fn elem_field(path: &[&str]) -> Expr {
    Expr::Field(Field {
        root: 1,
        path: path
            .iter()
            .map(|k| PathComponent::Key((*k).to_owned()))
            .collect(),
    })
}

/// A field of loop variable `var`.
fn var_field(var: jsonsm_ast::VariableId, path: &[&str]) -> Expr {
    Expr::Field(Field {
        root: var,
        path: path
            .iter()
            .map(|k| PathComponent::Key((*k).to_owned()))
            .collect(),
    })
}

/// A loop nested inside a loop, whose inner body compares an inner-element field against
/// something from an enclosing scope — the outer loop element (`o.x`), the document root, or
/// a constant. This is what exercises deferring loops out through more than one scope.
fn gen_nested_loop(rng: &mut Rng) -> Expr {
    let modes = [LoopType::Any, LoopType::Every, LoopType::AnyEvery];
    let reference = match rng.below(3) {
        0 => var_field(1, &["x"]),                      // the middle scope
        1 => field(&[FIELDS[rng.below(FIELDS.len())]]), // the document root
        _ => gen_const(rng),
    };
    let inner_lhs = if rng.chance(2) {
        var_field(2, &["x"])
    } else {
        var_field(2, &[])
    };
    // Occasionally make the inner body a value operator on an enclosing scope instead of a
    // comparison, which is what defers *both* loops for an `exists`/`matches` operand.
    let inner_body = if rng.chance(3) {
        let outer = if rng.chance(2) {
            var_field(1, &["x"])
        } else {
            field(&[FIELDS[rng.below(FIELDS.len())]])
        };
        Expr::Exists(Box::new(outer))
    } else {
        Expr::compare(OPS[rng.below(OPS.len())], inner_lhs, reference)
    };
    let inner = Expr::Loop {
        loop_type: modes[rng.below(modes.len())],
        var: 2,
        in_expr: Box::new(var_field(1, &["z"])),
        sub_expr: Box::new(inner_body),
    };
    Expr::Loop {
        loop_type: modes[rng.below(modes.len())],
        var: 1,
        in_expr: Box::new(field(&[FIELDS[rng.below(FIELDS.len())]])),
        sub_expr: Box::new(inner),
    }
}

fn gen_loop(rng: &mut Rng) -> Expr {
    let modes = [LoopType::Any, LoopType::Every, LoopType::AnyEvery];
    let body = match rng.below(7) {
        // element (scalar) <op> const
        0 => Expr::compare(OPS[rng.below(OPS.len())], elem_field(&[]), gen_const(rng)),
        // element.x <op> const
        1 => Expr::compare(
            OPS[rng.below(OPS.len())],
            elem_field(&["x"]),
            gen_const(rng),
        ),
        // cross-field within the loop body: element.x <op> element.y
        2 => Expr::compare(
            OPS[rng.below(OPS.len())],
            elem_field(&["x"]),
            elem_field(&["y"]),
        ),
        // cross-scope: element.x <op> a document field (only compiles for a root-level
        // loop; nested occurrences are skipped as unsupported)
        3 => Expr::compare(
            OPS[rng.below(OPS.len())],
            elem_field(&["x"]),
            field(&[FIELDS[rng.below(FIELDS.len())]]),
        ),
        // cross-scope `exists` / `matches`: a value operator whose operand comes from the
        // *enclosing* scope, reached through a slot rather than the active value. These were
        // rejected outright until the operators learned to take a `DataRef`, so the body shapes
        // below are the only ones covering that route — and `exists` is the one operator for
        // which an absent operand is a definite `false` rather than unknown.
        4 => {
            let outer = field(&[FIELDS[rng.below(FIELDS.len())]]);
            if rng.chance(2) {
                Expr::Exists(Box::new(outer))
            } else {
                Expr::NotExists(Box::new(outer))
            }
        }
        5 => Expr::Matches {
            lhs: Box::new(field(&[FIELDS[rng.below(FIELDS.len())]])),
            pattern: Box::new(Expr::Value(Literal::String(
                PATTERNS[rng.below(PATTERNS.len())].into(),
            ))),
        },
        // An enclosing-scope field compared against a constant, with the element named
        // nowhere in the body. The outer field cannot be the active value, so it is reached
        // through a slot: `Compare(Slot, Const)`. That shape compiled *zero* times across the
        // whole sweep before this case existed, and it is the only one that catches a
        // constant being read as escaped rather than decoded.
        _ => {
            let op = OPS[rng.below(OPS.len())];
            let outer = field(&[FIELDS[rng.below(FIELDS.len())]]);
            let k = gen_const(rng);
            if rng.chance(2) {
                Expr::compare(op, k, outer)
            } else {
                Expr::compare(op, outer, k)
            }
        }
    };
    Expr::Loop {
        loop_type: modes[rng.below(modes.len())],
        var: 1,
        in_expr: Box::new(field(&[FIELDS[rng.below(FIELDS.len())]])),
        sub_expr: Box::new(body),
    }
}

fn gen_expr(rng: &mut Rng, depth: u32) -> Expr {
    if depth == 0 {
        return gen_leaf(rng);
    }
    match rng.below(6) {
        0 => Expr::Not(Box::new(gen_expr(rng, depth - 1))),
        1 => {
            let n = 1 + rng.below(3);
            Expr::And((0..n).map(|_| gen_expr(rng, depth - 1)).collect())
        }
        2 => {
            let n = 1 + rng.below(3);
            Expr::Or((0..n).map(|_| gen_expr(rng, depth - 1)).collect())
        }
        3 => gen_loop(rng),
        4 if depth >= 2 => gen_nested_loop(rng),
        _ => gen_leaf(rng),
    }
}

#[test]
fn fast_matcher_agrees_with_oracle() {
    let mut rng = Rng(0xDEAD_BEEF_1234_5678);
    let mut checked = 0usize;
    let mut skipped = 0usize;

    for _ in 0..20_000 {
        let expr = gen_expr(&mut rng, 3);
        let doc = gen_doc(&mut rng);
        let bytes = serde_json::to_vec(&doc).unwrap();

        // Only exercise expressions the compiler supports; skip the rest.
        let def = match compile(
            std::slice::from_ref(&expr),
            &Projection::new(),
            &DefaultCollation,
        ) {
            Ok(d) => d,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let mut backends = matchers(&def);
        let fast = {
            let mut agreed: Option<bool> = None;
            for (name, fm) in &mut backends {
                let got = fm.matches(&bytes).expect("fast match").matched();
                match agreed {
                    None => agreed = Some(got),
                    Some(prev) => assert_eq!(
                        prev,
                        got,
                        "backends disagree ({name}) on doc {}",
                        String::from_utf8_lossy(&bytes)
                    ),
                }
            }
            agreed.expect("at least one backend")
        };
        let slow = SlowMatcher::new(expr.clone())
            .matches(&doc)
            .expect("slow match");

        assert_eq!(
            fast, slow,
            "mismatch\n  expr: {expr:?}\n  doc:  {doc}\n  fast={fast} slow={slow}"
        );
        checked += 1;
    }

    // Sanity: the generator should mostly produce supported, exercised cases.
    assert!(
        checked > 10_000,
        "expected many checked cases, got {checked} (skipped {skipped})"
    );
}

/// Nested loops whose inner body reads an enclosing scope get their own sweep: the compiler
/// has to defer each loop out to the scope it reads (recursively), and a silent regression
/// here would otherwise hide behind the general sweep's random shape mix.
#[test]
fn nested_cross_scope_loops_agree_with_oracle() {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut checked = 0usize;
    let mut matched = 0usize;

    for _ in 0..5_000 {
        let expr = gen_nested_loop(&mut rng);
        let doc = gen_doc(&mut rng);
        let bytes = serde_json::to_vec(&doc).unwrap();

        let def = compile(
            std::slice::from_ref(&expr),
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap_or_else(|e| panic!("nested loops must compile: {e}\n  expr: {expr:?}"));

        let mut backends = matchers(&def);
        let fast = {
            let mut agreed: Option<bool> = None;
            for (name, fm) in &mut backends {
                let got = fm.matches(&bytes).expect("fast match").matched();
                match agreed {
                    None => agreed = Some(got),
                    Some(prev) => assert_eq!(
                        prev,
                        got,
                        "backends disagree ({name}) on doc {}",
                        String::from_utf8_lossy(&bytes)
                    ),
                }
            }
            agreed.expect("at least one backend")
        };
        let slow = SlowMatcher::new(expr.clone())
            .matches(&doc)
            .expect("slow match");
        assert_eq!(
            fast, slow,
            "mismatch\n  expr: {expr:?}\n  doc:  {doc}\n  fast={fast} slow={slow}"
        );
        checked += 1;
        matched += usize::from(fast);
    }

    // Non-vacuous: the sweep must produce a healthy mix, not all-false.
    assert_eq!(checked, 5_000);
    assert!(
        matched > 200,
        "expected a meaningful number of matches, got {matched}"
    );
}

// ---- field projection -------------------------------------------------------------------

/// Candidate projection paths: present/absent, nested, array elements (in and out of range),
/// and the whole document (the empty path).
fn project_paths() -> Vec<Vec<PathComponent>> {
    let key = |k: &str| PathComponent::Key(k.to_owned());
    vec![
        vec![key("a")],
        vec![key("b")],
        vec![key("c")],
        vec![key("c"), key("x")],
        vec![key("a"), key("x")], // `a` is often an array or scalar: usually absent
        vec![key("b"), key("zz")],
        vec![key("zz")],
        vec![key("a"), PathComponent::Index(0)],
        vec![key("b"), PathComponent::Index(1), key("x")],
        vec![key("c"), PathComponent::Index(7)], // usually out of range
        vec![],
    ]
}

/// The value at `path` in `doc`: object keys index objects, indices index arrays. Anything
/// else in the middle of a path means "absent", which is what the exec trie does too.
fn navigate<'v>(doc: &'v Value, path: &[PathComponent]) -> Option<&'v Value> {
    let mut cur = doc;
    for comp in path {
        cur = match comp {
            PathComponent::Key(k) => cur.as_object()?.get(k)?,
            PathComponent::Index(i) => cur.as_array()?.get(*i)?,
        };
    }
    Some(cur)
}

/// Whether a captured [`FastVal`] represents exactly `want`. Numbers and containers are
/// checked by re-parsing the captured raw document bytes, which also verifies the byte range
/// the matcher recorded is precisely the value's own extent.
fn same_value(got: &jsonsm::value::FastVal<'_>, want: &Value) -> bool {
    use jsonsm::value::FastVal;
    match (got, want) {
        (FastVal::Null, Value::Null) => true,
        (FastVal::Bool(b), Value::Bool(w)) => b == w,
        (FastVal::IntBytes(raw) | FastVal::FloatBytes(raw), Value::Number(_))
        | (FastVal::Array(raw), Value::Array(_))
        | (FastVal::Object(raw), Value::Object(_)) => {
            serde_json::from_slice::<Value>(raw).is_ok_and(|reparsed| &reparsed == want)
        }
        (FastVal::Str(s), Value::String(w)) => s.to_decoded_bytes().as_ref() == w.as_bytes(),
        _ => false,
    }
}

#[test]
fn projected_values_agree_with_serde_navigation() {
    let mut rng = Rng(0x0BAD_C0DE_F00D_5EED);
    let mut checked = 0usize;
    let mut captured = 0usize;

    for _ in 0..10_000 {
        let expr = gen_expr(&mut rng, 3);
        let doc = gen_doc(&mut rng);
        let bytes = serde_json::to_vec(&doc).unwrap();

        // One to three projected paths, chosen independently (so duplicates happen too).
        let candidates = project_paths();
        let paths: Vec<Vec<PathComponent>> = (0..1 + rng.below(3))
            .map(|_| candidates[rng.below(candidates.len())].clone())
            .collect();
        let mut projection = Projection::new();
        for p in &paths {
            projection.push(p.iter().cloned());
        }

        let exprs = std::slice::from_ref(&expr);
        let Ok(def) = compile(exprs, &projection, &DefaultCollation) else {
            continue; // unsupported expression shape
        };
        // The same expression without a projection: the match result must be identical.
        let plain = compile(exprs, &Projection::new(), &DefaultCollation).expect("compiles");

        let mut fm = matcher_for(&def, checked);
        let with = {
            let out = fm.matches(&bytes).expect("fast match");

            for (i, path) in paths.iter().enumerate() {
                let want = navigate(&doc, path);
                match (out.projected(i), want) {
                    (None, None) => {}
                    (Some(got), Some(want)) => {
                        assert!(
                            same_value(&got, want),
                            "projected value mismatch at {path:?}\n  doc:  {doc}\n  \
                             got:  {got:?}\n  want: {want}"
                        );
                        captured += 1;
                    }
                    (got, want) => panic!(
                        "presence mismatch at {path:?}: got {:?}, want {:?}\n  doc: {doc}",
                        got.is_some(),
                        want.is_some()
                    ),
                }
                assert_eq!(out.projected_path(i), path.as_slice());
            }
            out.matched()
        };

        let plain_result = matcher_for(&plain, checked)
            .matches(&bytes)
            .expect("fast match")
            .matched();
        assert_eq!(
            with, plain_result,
            "projection changed the match result\n  expr: {expr:?}\n  doc:  {doc}"
        );
        checked += 1;
    }

    // Sanity: the sweep must actually have captured a healthy number of live values.
    assert!(
        checked > 5_000 && captured > 5_000,
        "expected many captured values, got {captured} over {checked} documents"
    );
}
