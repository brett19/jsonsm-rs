# jsonsm-rs

Single-pass, streaming expression matching against raw JSON.

`jsonsm-rs` evaluates a compiled boolean expression against a JSON document in **one
tokenizing scan** — no parse tree, no intermediate value map, no allocation on the hot path.
The scan walks the document and the compiled expression in lockstep: fields the expression
names are matched, everything else is crossed structurally, and the scan stops as soon as the
result is decided.

It is a port of Couchbase's [gojsonsm](https://github.com/couchbase/gojsonsm), with corrected
comparison semantics and several fixed bugs — see [differences from
gojsonsm](docs/gojsonsm-differences.md).

```rust
use jsonsm::compile::{compile, Projection};
use jsonsm::collation::DefaultCollation;
use jsonsm::matcher::FastMatcher;
use jsonsm_n1ql::compile_str;

let def = compile_str("age > 21 AND ANY t IN tags SATISFIES t = 'rust' END",
                      &Projection::default(), &DefaultCollation)?;
let mut matcher = FastMatcher::new(&def);

assert!(matcher.matches(br#"{"age": 41, "tags": ["go", "rust"]}"#)?.matched());
assert!(!matcher.matches(br#"{"age": 41, "tags": ["go"]}"#)?.matched());
```

Expressions can be written as [N1QL-like text](jsonsm-n1ql), as the [JSON array
format](jsonsm-json) gojsonsm uses, or built directly as an [AST](jsonsm-ast).

## Two things to know before using it

**A comparison against a field the document does not contain is `UNKNOWN`, not `false`.**
`UNKNOWN` combines by Kleene's tables and `NOT UNKNOWN` is `UNKNOWN`, so writing `!=` or `NOT`
around an absent field cannot turn it into a match. Use `EXISTS` / `IS MISSING` to select on
absence. Full tables in [semantics.md](docs/semantics.md).

**The engine requires well-formed JSON.** Regions no expression names are crossed without
being parsed, so malformed content there may go undetected. Values that are actually compared
are always fully tokenized. Details and the rest of the contract in
[limits-and-caveats.md](docs/limits-and-caveats.md).

## Documentation

| Document | What it covers |
| --- | --- |
| [semantics.md](docs/semantics.md) | The matching contract: types, comparison, three-valued logic, quantifiers, projection |
| [limits-and-caveats.md](docs/limits-and-caveats.md) | Input requirements, what is and is not validated, depth and numeric limits, unsupported cases |
| [gojsonsm-differences.md](docs/gojsonsm-differences.md) | How this differs from the Go original, including bugs fixed |
| [design.md](docs/design.md) | How the engine is built and why |
| [jsonsm-bench/README.md](jsonsm-bench/README.md) | The measurement harness, and which number to believe |

## Performance

Throughput on one core, matching each document end to end. `gojsonsm` is the Go original over
a byte-identical corpus, running the same expressions and asserting the same match counts.

| Workload | jsonsm-rs | gojsonsm | |
| --- | ---: | ---: | ---: |
| An everyday filter — three fields, short-circuits early | **9,379 MB/s** | 2,279 MB/s | **4.1x** |
| One field 86% into the record, the rest skipped | **3,468 MB/s** | 641 MB/s | **5.4x** |
| A quantifier over an array of strings | **3,486 MB/s** | 461 MB/s | **7.6x** |
| One field behind a 1,600-element array; ~99% skipped | **10,993 MB/s** | 772 MB/s | **14.2x** |
| Tokenizing alone, no matching | **3,409 MB/s** | 752 MB/s | **4.5x** |

The spread is the point rather than any single figure: the further a document is from the
fields an expression names, the more the single-pass design wins, because that distance is
crossed without being parsed. The everyday filter at 4x is the honest number to plan with; the
14x case is what structural skipping buys when a document is mostly irrelevant.

Two workloads have no Go counterpart. Comparing two fields of the same record — which defers
to the end of the record, so the scan cannot stop early — still runs at **4,497 MB/s**. A
quantifier over an array of objects, the most matcher-heavy shape here, runs at
**984 MB/s**.

For scale in the other direction, `jsonsm-slow` parses each document to a `serde_json::Value`
and interprets the expression over it — the conventional approach. It manages **487 MB/s** on
the everyday filter, which the streaming engine beats by **19x**.

Measured on an AMD Ryzen 9 9950X3D with `rustc` 1.95 and `go` 1.26, medians of nine rounds.
Reproduce with `cargo bench -p jsonsm-bench`. Wall-clock on one machine is a rough guide;
[jsonsm-bench/README.md](jsonsm-bench/README.md) explains which number to believe when
comparing two builds of this engine against each other.

## Workspace

| Crate | Purpose |
| --- | --- |
| [`jsonsm-ast`](jsonsm-ast) | The expression AST — pure data, no dependencies |
| [`jsonsm`](jsonsm) | The engine: tokenizer, value model, collation, compiler, matcher |
| [`jsonsm-json`](jsonsm-json) | The JSON-array expression format ↔ AST |
| [`jsonsm-n1ql`](jsonsm-n1ql) | The N1QL-like string grammar ↔ AST |
| [`jsonsm-slow`](jsonsm-slow) | A reference interpreter over `serde_json`, used as a test oracle |
| [`jsonsm-bench`](jsonsm-bench) | The measurement harness |

## Features

`jsonsm` has one feature, `simd`, on by default. It enables the vector scan backends, selected
at runtime by CPU detection. Building with `--no-default-features` gives a scalar-only engine
with identical behaviour and a crate-level `forbid(unsafe_code)`.

## Building and testing

```bash
cargo test --workspace              # includes the differential sweeps
cargo test --workspace --release
cargo bench -p jsonsm-bench         # throughput for every workload and backend
```

The test suite checks every available scan backend against the reference interpreter, not just
the one the CPU would select.

## Portability

The engine is portable Rust; the vector backends are x86-64 and are feature-gated, with a
scalar fallback everywhere else. `no_std` is not supported.

## License

Apache-2.0. Copyright Couchbase, Inc.
