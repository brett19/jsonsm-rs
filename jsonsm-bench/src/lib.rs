//! The measurement harness for `jsonsm`. Every performance figure quoted for this project
//! comes from here.
//!
//! Two front doors over one set of workloads, so they cannot drift apart:
//!
//! * `cargo bench -p jsonsm-bench` — the whole table, every engine, wall-clock MB/s.
//! * `target/release/jsonsm-bench <workload> <engine> <iters>` — one workload, one engine,
//!   no warm-up and no extra output, so `perf stat` measures the thing and not the harness.
//!
//! Every backend lives in this one build and is selected at runtime, so all arms share one
//! code layout. That is not a convenience: a cross-build A/B is untrustworthy here, and
//! putting both arms in one binary is the only way to settle a difference of a few percent.
//!
//! # Which number to believe
//!
//! **Instructions retired is the metric of record; wall-clock is not.** On the development
//! machine, *identical* code has measured 15.2 µs or 20.4 µs on the same workload depending
//! only on what else was linked into the binary — adding one monomorphisation moved untouched
//! code by 27%. Instruction counts reproduce across rebuilds to five significant figures. So
//! a wall-clock delta under ~5% between two builds means nothing, and `cargo bench` output is
//! for a quick look, not for a decision:
//!
//! ```text
//! perf stat -r 5 -e instructions,cycles target/release/jsonsm-bench match/late_field sse2 2000
//! ```
//!
//! Report cycles too, and read them together: instructions retired is not a proxy for speed.
//! Two changes to the same loop have been ranked in *opposite* orders by the two metrics, and
//! the one that retired six times more instructions was the one to discard — the loop was
//! latency-bound, so deleting off-critical-path work bought nothing. If instructions fall and
//! cycles do not, check IPC and look for a dependency chain.
//!
//! # Reading the workload names
//!
//! `tokenize/*` runs the tokenizer alone — the floor a matcher workload is measured against.
//! It is a good control for "is this the same build?" in instruction terms, but **not** a
//! control for matcher layout: it shares no code with the matcher, and has been stable to five
//! significant figures across matcher changes that moved matcher benchmarks 21% in cycles.
//!
//! The `*_20` / `*_220` pairs differ only in element count, so
//! `(insn(220) - insn(20)) / 200 / iters` is the cost of one loop element with the document
//! prefix, the compile and the harness all cancelled out.
//!
//! See `README.md` for the workload table and the fixture rationale.

use jsonsm::collation::DefaultCollation;
use jsonsm::compile::{compile, MatchDef, Projection};
use jsonsm::matcher::FastMatcher;
use jsonsm::simd::Backend;
use jsonsm::tokenizer::{GenericTokenizer, Scan, ScalarScan, TokenType, Tokenizer};
use jsonsm_ast::{CompareOp, Expr, Field, Literal, LoopType, PathComponent};
use jsonsm_slow::SlowMatcher;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

/// `people.json` and `bigvector.json` are the library's own test fixtures; the bench crate
/// reads them where they live rather than keeping a second copy that could drift.
fn testdata(name: &str) -> Vec<u8> {
    read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../jsonsm/testdata")
            .join(name),
    )
}

/// Fixtures that exist only for benchmarking, under this crate's `corpus/`.
fn corpus(name: &str) -> Vec<u8> {
    read(Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus").join(name))
}

fn read(path: impl AsRef<Path>) -> Vec<u8> {
    let path = path.as_ref();
    std::fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// The canonical corpus: `people.json` split into one compact buffer per record, **in the
/// document's original key order**. Kept on disk rather than derived at startup, because the
/// Go harness reads the same files — going through `serde_json::Value` would silently
/// alphabetise the keys (its default object map is a `BTreeMap`) and move `favoriteFruit`
/// relative to where gojsonsm sees it, so the two engines would not be handed the same bytes.
pub fn people_records() -> Vec<Vec<u8>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus/people");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "corpus is empty");
    paths.iter().map(read).collect()
}

fn field(keys: &[&str]) -> Expr {
    Expr::Field(Field::root(
        keys.iter()
            .map(|k| PathComponent::Key((*k).to_owned()))
            .collect(),
    ))
}

fn total_bytes(records: &[Vec<u8>]) -> u64 {
    records.iter().map(|r| r.len() as u64).sum()
}

// ---------------------------------------------------------------- workloads

fn drive_tokenize<S: Scan>(records: &[Vec<u8>]) -> usize {
    let mut n = 0usize;
    for rec in records {
        let mut t = GenericTokenizer::<S>::new(rec);
        loop {
            let tok = t.step().unwrap();
            if tok.token_type == TokenType::End {
                break;
            }
            black_box(tok.value);
            n += 1;
        }
    }
    n
}

fn tokenize(backend: Backend, records: &[Vec<u8>]) -> usize {
    match backend {
        Backend::Scalar => drive_tokenize::<ScalarScan>(records),
        Backend::Sse2 => drive_tokenize::<jsonsm::simd::Sse2Scan>(records),
        Backend::Avx2 => <jsonsm::simd::Avx2Scan as Scan>::enter(|| {
            drive_tokenize::<jsonsm::simd::Avx2Scan>(records)
        }),
        // `HybridScan`'s state-machine kernels are SSE2 and its `enter` is the identity, so
        // there is nothing to wrap: the tokenizer never calls `skip_container`, which is the
        // only method it overrides. Expect it to track `sse2` here to five figures.
        Backend::Hybrid => drive_tokenize::<jsonsm::simd::HybridScan>(records),
    }
}

fn run_match(m: &mut FastMatcher<'_>, records: &[Vec<u8>]) -> usize {
    let mut hits = 0usize;
    for rec in records {
        if m.matches(rec).unwrap().matched() {
            hits += 1;
        }
    }
    hits
}

// ---------------------------------------------------------------- expressions


/// Two fields of the same record compared against each other — the shape whose cost no other
/// workload here could see.
///
/// A comparison naming two local fields cannot run as either is scanned, so the compiler
/// stores both in slots and defers it to the scope's after-node, which runs only once the
/// whole record has been walked. The logic tree therefore stays undecided for the entire
/// document and `done()` never fires, however early the fields appear. `index` is key 1 of 25
/// and `age` is key 6, and this still costs **2.6x** `match/and_or`, whose fields sit at 3, 6
/// and 7 — and 79% of tokenizing the record outright.
///
/// Every other matcher workload here names fields against constants, where the tree resolves
/// mid-scan and the scan stops. This one is the counter-case, and it is not exotic: `a op b`,
/// `f(a, b) op c` and cross-scope loops all compile this way.
fn expr_cross_field_early() -> Expr {
    Expr::compare(CompareOp::LessThan, field(&["index"]), field(&["age"]))
}


/// A loop whose *body* is a wide disjunction, so the logic-tree subtree reset between
/// elements is a real tree rather than a single node.
fn expr_wide_body(n: usize) -> Expr {
    Expr::Loop {
        loop_type: LoopType::Any,
        var: 1,
        in_expr: Box::new(field(&["tags"])),
        sub_expr: Box::new(Expr::Or(
            (0..n)
                .map(|i| {
                    Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field { root: 1, path: vec![] }),
                        Expr::Value(Literal::String(format!("nomatch{i}"))),
                    )
                })
                .collect(),
        )),
    }
}


/// A loop whose body names a field the elements do not have, over a wide disjunction.
///
/// The body can never be evaluated, so its logic-tree node is still `Unset` when the element
/// ends and `seal_node` has to walk the subtree — the path that the usual workloads skip
/// entirely, because there the op runs and the body root is already set.
fn expr_absent_wide(n: usize) -> Expr {
    Expr::Loop {
        loop_type: LoopType::Any,
        var: 1,
        in_expr: Box::new(field(&["tags"])),
        sub_expr: Box::new(Expr::Or(
            (0..n)
                .map(|i| {
                    Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field {
                            root: 1,
                            path: vec![PathComponent::Key(format!("absent{i}"))],
                        }),
                        Expr::Value(Literal::String("v".into())),
                    )
                })
                .collect(),
        )),
    }
}

fn expr_and_or() -> Expr {
    // (age < 50 AND isActive == true) OR eyeColor == "brown"
    Expr::Or(vec![
        Expr::And(vec![
            Expr::compare(
                CompareOp::LessThan,
                field(&["age"]),
                Expr::Value(Literal::Int(50)),
            ),
            Expr::compare(
                CompareOp::Equals,
                field(&["isActive"]),
                Expr::Value(Literal::Bool(true)),
            ),
        ]),
        Expr::compare(
            CompareOp::Equals,
            field(&["eyeColor"]),
            Expr::Value(Literal::String("brown".into())),
        ),
    ])
}

fn expr_any_loop() -> Expr {
    // ANY tag IN tags SATISFIES tag == "cillum" END
    Expr::Loop {
        loop_type: LoopType::Any,
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
    }
}

/// `ANY v IN tags SATISFIES <v-path> <op> "nomatch" END` — never matches, so the loop
/// exhausts every element. Paired with fixtures differing only in element count, the
/// instruction *slope* between them is the per-element cost with nothing else in it.
fn expr_loop_over(mode: LoopType, path: &[&str], op: CompareOp) -> Expr {
    Expr::Loop {
        loop_type: mode,
        var: 1,
        in_expr: Box::new(field(&["tags"])),
        sub_expr: Box::new(Expr::compare(
            op,
            Expr::Field(Field {
                root: 1,
                path: path
                    .iter()
                    .map(|k| PathComponent::Key((*k).to_owned()))
                    .collect(),
            }),
            Expr::Value(Literal::String("nomatch".into())),
        )),
    }
}

/// `ANY v IN tags SATISFIES v == <number> END` — [`expr_loop_over`] with a numeric constant,
/// and the only per-element probe in this suite that does not compare strings.
///
/// Every 20-vs-220 pair here compared string equality, which meant no per-element figure the
/// project has ever recorded said anything about a *number*. That is not a small gap: a
/// document integer arrives as [`jsonsm::value::FastVal::IntBytes`] and is parsed on every
/// comparison, so a loop over numbers pays a parse per element where a loop over strings pays
/// a `memcmp`.
///
/// The constant is outside the fixtures' range, so the loop exhausts every element and the hit
/// count is zero, exactly as `any_str_*`.
fn expr_loop_over_num(lit: Literal) -> Expr {
    Expr::Loop {
        loop_type: LoopType::Any,
        var: 1,
        in_expr: Box::new(field(&["tags"])),
        sub_expr: Box::new(Expr::compare(
            CompareOp::Equals,
            Expr::Field(Field {
                root: 1,
                path: vec![],
            }),
            Expr::Value(lit),
        )),
    }
}

/// The control for the loop workloads: reaches `tags` over the same prefix but structurally
/// skips the array instead of iterating it. `any_str_N` minus this is the loop machinery.
fn expr_tags_scalar() -> Expr {
    Expr::compare(
        CompareOp::Equals,
        field(&["tags"]),
        Expr::Value(Literal::String("nomatch".into())),
    )
}

fn expr_late_field() -> Expr {
    // ~86% of the way into each record: dominated by structural skipping.
    Expr::compare(
        CompareOp::Equals,
        field(&["favoriteFruit"]),
        Expr::Value(Literal::String("strawberry".into())),
    )
}

fn def_for(expr: &Expr) -> MatchDef {
    compile(
        std::slice::from_ref(expr),
        &Projection::new(),
        &DefaultCollation,
    )
    .unwrap()
}

// ---------------------------------------------------------------- driver

pub struct Workload {
    pub name: &'static str,
    pub records: Vec<Vec<u8>>,
    /// `None` for the tokenize-only workloads. Both engines are built from this one value.
    pub expr: Option<Expr>,
}

pub fn workloads() -> Vec<Workload> {
    let people = people_records();
    let bigvector = vec![testdata("bigvector.json")];
    vec![
        Workload {
            name: "tokenize/people",
            records: people.clone(),
            expr: None,
        },
        Workload {
            name: "tokenize/bigvector",
            records: bigvector,
            expr: None,
        },
        Workload {
            name: "match/and_or",
            records: people.clone(),
            expr: Some(expr_and_or()),
        },
        Workload {
            name: "match/any_loop",
            records: people.clone(),
            expr: Some(expr_any_loop()),
        },
        Workload {
            // A `tag` field behind a 1600-element array and a 200-element array of objects:
            // ~99% of this document is skipped wholesale, which is the case a bulk
            // structural skip exists for. `people.json` skips mostly *scalars*, which
            // `skip_value` returns from without scanning at all.
            name: "match/skip_big",
            records: vec![corpus("skipbig.json")],
            expr: Some(Expr::compare(
                CompareOp::Equals,
                field(&["tag"]),
                Expr::Value(Literal::String("strawberry".into())),
            )),
        },
        Workload {
            name: "match/cross_field",
            records: people_records(),
            expr: Some(expr_cross_field_early()),
        },
        Workload {
            name: "match/late_field",
            records: people,
            expr: Some(expr_late_field()),
        },
        // The floor: what the tokenizer alone costs on the loop fixtures, same 20-vs-220
        // slope. Any per-element cost above this is the matcher's, not the scan's.
        Workload {
            name: "tokenize/loop_str_20",
            records: vec![corpus("loop_str_20.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_str_220",
            records: vec![corpus("loop_str_220.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_int_20",
            records: vec![corpus("loop_int_20.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_int_220",
            records: vec![corpus("loop_int_220.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_smallint_20",
            records: vec![corpus("loop_smallint_20.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_smallint_220",
            records: vec![corpus("loop_smallint_220.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_float_20",
            records: vec![corpus("loop_float_20.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_float_220",
            records: vec![corpus("loop_float_220.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_obj_20",
            records: vec![corpus("loop_obj_20.json")],
            expr: None,
        },
        Workload {
            name: "tokenize/loop_obj_220",
            records: vec![corpus("loop_obj_220.json")],
            expr: None,
        },
        // ---- loop-cost probes. Each pair differs only in element count (20 vs 220), so
        // (insn(220) - insn(20)) / 200 is the per-element cost of that loop shape, free of
        // the document prefix, the compile, and the harness.
        Workload {
            name: "match/any_str_20",
            records: vec![corpus("loop_str_20.json")],
            expr: Some(expr_loop_over(
                LoopType::Any,
                &[],
                CompareOp::Equals,
            )),
        },
        Workload {
            name: "match/any_str_220",
            records: vec![corpus("loop_str_220.json")],
            expr: Some(expr_loop_over(
                LoopType::Any,
                &[],
                CompareOp::Equals,
            )),
        },
        // The same loop over numbers instead of strings: `IntBytes` and `FloatBytes` are
        // parsed per comparison, which no string pair can show.
        Workload {
            name: "match/any_int_20",
            records: vec![corpus("loop_int_20.json")],
            expr: Some(expr_loop_over_num(Literal::Int(99_999_999))),
        },
        Workload {
            name: "match/any_int_220",
            records: vec![corpus("loop_int_220.json")],
            expr: Some(expr_loop_over_num(Literal::Int(99_999_999))),
        },
        // Two digits per element rather than eight. Varying the *width* of the number with
        // element count held fixed is what separates what a numeric comparison costs per
        // element from what it costs per digit — the same two-slope trick `any_obj{0,3,7}`
        // uses for object width.
        Workload {
            name: "match/any_smallint_20",
            records: vec![corpus("loop_smallint_20.json")],
            expr: Some(expr_loop_over_num(Literal::Int(9))),
        },
        Workload {
            name: "match/any_smallint_220",
            records: vec![corpus("loop_smallint_220.json")],
            expr: Some(expr_loop_over_num(Literal::Int(9))),
        },
        Workload {
            name: "match/any_float_20",
            records: vec![corpus("loop_float_20.json")],
            expr: Some(expr_loop_over_num(Literal::Float(9_999.999_9))),
        },
        Workload {
            name: "match/any_float_220",
            records: vec![corpus("loop_float_220.json")],
            expr: Some(expr_loop_over_num(Literal::Float(9_999.999_9))),
        },
        Workload {
            name: "match/skip_str_220",
            records: vec![corpus("loop_str_220.json")],
            expr: Some(expr_tags_scalar()),
        },
        Workload {
            name: "match/any_obj_20",
            records: vec![corpus("loop_obj_20.json")],
            expr: Some(expr_loop_over(
                LoopType::Any,
                &["t"],
                CompareOp::Equals,
            )),
        },
        Workload {
            name: "match/any_obj_220",
            records: vec![corpus("loop_obj_220.json")],
            expr: Some(expr_loop_over(
                LoopType::Any,
                &["t"],
                CompareOp::Equals,
            )),
        },
        // EVERY cannot stop at the first hit the way ANY does: every element satisfies
        // `!= "nomatch"`, so all 220 are visited and each one resolves through the body.
        Workload {
            name: "tokenize/loop_obj0_20",
            records: vec![corpus("loop_obj0_20.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj0_20",
            records: vec![corpus("loop_obj0_20.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "tokenize/loop_obj0_220",
            records: vec![corpus("loop_obj0_220.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj0_220",
            records: vec![corpus("loop_obj0_220.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "tokenize/loop_obj3_20",
            records: vec![corpus("loop_obj3_20.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj3_20",
            records: vec![corpus("loop_obj3_20.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "tokenize/loop_obj3_220",
            records: vec![corpus("loop_obj3_220.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj3_220",
            records: vec![corpus("loop_obj3_220.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "tokenize/loop_obj7_20",
            records: vec![corpus("loop_obj7_20.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj7_20",
            records: vec![corpus("loop_obj7_20.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "tokenize/loop_obj7_220",
            records: vec![corpus("loop_obj7_220.json")],
            expr: None,
        },
        Workload {
            name: "match/any_obj7_220",
            records: vec![corpus("loop_obj7_220.json")],
            expr: Some(expr_loop_over(LoopType::Any, &["t"], CompareOp::Equals)),
        },
        Workload {
            name: "match/wide_body_20",
            records: vec![corpus("loop_str_20.json")],
            expr: Some(expr_wide_body(12)),
        },
        Workload {
            name: "match/wide_body_220",
            records: vec![corpus("loop_str_220.json")],
            expr: Some(expr_wide_body(12)),
        },
        Workload {
            name: "match/absent_body_20",
            records: vec![corpus("loop_obj_20.json")],
            expr: Some(expr_absent_wide(12)),
        },
        Workload {
            name: "match/absent_body_220",
            records: vec![corpus("loop_obj_220.json")],
            expr: Some(expr_absent_wide(12)),
        },
        Workload {
            name: "match/every_str_220",
            records: vec![corpus("loop_str_220.json")],
            expr: Some(expr_loop_over(
                LoopType::Every,
                &[],
                CompareOp::NotEquals,
            )),
        },
    ]
}

/// Which implementation to time.
///
/// `Slow` is `jsonsm-slow`'s `SlowMatcher` — the reference interpreter that walks the AST over
/// a parsed `serde_json::Value`, used elsewhere as the differential oracle. Its throughput
/// necessarily includes the `serde_json` parse, because that is how it works: it is a
/// parse-then-interpret design, and the parse is not separable overhead but the approach. It
/// is here as the "obvious implementation" baseline the streaming engine is measured against.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Engine {
    Fast(Backend),
    Slow,
}

fn engine_name(e: Engine) -> &'static str {
    match e {
        Engine::Fast(b) => backend_name(b),
        Engine::Slow => "slow",
    }
}

/// Both engines are built from the workload's single `Expr`, so a throughput comparison
/// cannot silently be between two different questions.
fn run_slow(m: &SlowMatcher, records: &[Vec<u8>]) -> usize {
    records
        .iter()
        .filter(|r| m.matches_bytes(r).expect("slow match"))
        .count()
}

/// One timed pass: `iters` full sweeps of the corpus. Returns seconds. Compilation happens
/// outside the timed region for both engines.
fn once(w: &Workload, engine: Engine, iters: u64) -> f64 {
    match (&w.expr, engine) {
        (None, Engine::Fast(backend)) => {
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(tokenize(backend, &w.records));
            }
            t0.elapsed().as_secs_f64()
        }
        (Some(expr), Engine::Fast(backend)) => {
            let def = def_for(expr);
            let mut m = FastMatcher::new(&def);
            m.force_backend(backend);
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(run_match(&mut m, &w.records));
            }
            t0.elapsed().as_secs_f64()
        }
        (Some(expr), Engine::Slow) => {
            let m = SlowMatcher::new(expr.clone());
            let t0 = Instant::now();
            for _ in 0..iters {
                black_box(run_slow(&m, &w.records));
            }
            t0.elapsed().as_secs_f64()
        }
        // The reference matcher has no tokenize-only mode; it never sees tokens.
        (None, Engine::Slow) => f64::NAN,
    }
}

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Scalar => "scalar",
        Backend::Sse2 => "sse2",
        Backend::Avx2 => "avx2",
        Backend::Hybrid => "hybrid",
    }
}

fn parse_engine(s: &str) -> Engine {
    match s {
        "slow" => Engine::Slow,
        other => Engine::Fast(parse_backend(other)),
    }
}

fn parse_backend(s: &str) -> Backend {
    match s {
        "scalar" => Backend::Scalar,
        "sse2" => Backend::Sse2,
        "avx2" => Backend::Avx2,
        "hybrid" => Backend::Hybrid,
        _ => panic!("backend must be scalar|sse2|avx2|hybrid"),
    }
}

/// Print the whole table: every workload against every available engine, wall-clock MB/s.
///
/// Prefixed by a `CHECK` line per workload, which is not decoration: a throughput comparison
/// only means something if the engines answer the same question and get the same answer, so
/// the reference matcher is asserted to agree before anything is timed.
pub fn run_table(iters: u64, reps: u64) {
    let ws = workloads();
        // Agreement check: the throughput comparison only means something if both engines
        // answer the same question and get the same answer.
        for w in &ws {
            match &w.expr {
                None => eprintln!(
                    "CHECK {:<20} tokens={} bytes={}",
                    w.name,
                    tokenize(Backend::Scalar, &w.records),
                    total_bytes(&w.records)
                ),
                Some(expr) => {
                    let def = def_for(expr);
                    // Every backend, not just the detected one. Each is a separate
                    // monomorphisation, so checking one says nothing about the others — and
                    // every one of them gets a timed row below. Checking only the default
                    // let a backend be timed but never validated.
                    let slow = run_slow(&SlowMatcher::new(expr.clone()), &w.records);
                    let mut fast = None;
                    for backend in Backend::available() {
                        let mut m = FastMatcher::new(&def);
                        m.force_backend(backend);
                        let hits = run_match(&mut m, &w.records);
                        // The reference matcher must agree, or the throughput columns below
                        // are not answering the same question.
                        assert_eq!(
                            hits, slow,
                            "{:?} disagrees with slow on {}",
                            backend, w.name
                        );
                        fast = Some(hits);
                    }
                    eprintln!(
                        "CHECK {:<20} hits={} bytes={} (slow agrees, all {} backends)",
                        w.name,
                        fast.expect("at least one backend"),
                        total_bytes(&w.records),
                        Backend::available().len(),
                    )
                }
            }
        }
        println!("{:<20} {:>8} {:>10} {:>10}", "workload", "backend", "us/sweep", "MB/s");
        for w in &ws {
            let bytes = total_bytes(&w.records) as f64;
            let mut engines: Vec<Engine> = Backend::available().into_iter().map(Engine::Fast).collect();
            if w.expr.is_some() {
                engines.push(Engine::Slow);
            }
            for e in engines {
                // The reference matcher is ~100x slower; give it proportionally fewer sweeps
                // so a full run does not take minutes.
                let n = if e == Engine::Slow {
                    (iters / 50).max(1)
                } else {
                    iters
                };
                // warm-up, then best-of-reps (min time = least interference)
                once(w, e, n / 10 + 1);
                let best = (0..reps)
                    .map(|_| once(w, e, n))
                    .fold(f64::INFINITY, f64::min);
                let per = best / n as f64;
                println!(
                    "{:<20} {:>8} {:>10.3} {:>10.1}",
                    w.name,
                    engine_name(e),
                    per * 1e6,
                    bytes / per / 1e6
                );
            }
        }
}

/// Run one workload on one engine, with no warm-up and no extra output — the shape
/// `perf stat` wants. Prints a one-line summary to stderr.
pub fn run_single(name: &str, engine: &str, iters: u64) {
    let ws = workloads();
    let engine = parse_engine(engine);
    // Single-workload mode, for `perf stat` (no warm-up noise, no extra printing).
    let w = ws.iter().find(|w| w.name == name).expect("unknown workload");
    let secs = once(w, engine, iters);
    let bytes = total_bytes(&w.records) as f64;
    eprintln!(
        "{} {} iters={} {:.3} us/sweep {:.1} MB/s",
        name,
        engine_name(engine),
        iters,
        secs / iters as f64 * 1e6,
        bytes * iters as f64 / secs / 1e6
    );
}
