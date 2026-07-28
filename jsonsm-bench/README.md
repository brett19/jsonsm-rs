# jsonsm-bench

The measurement harness for `jsonsm`. Every performance figure quoted for this project comes from here.

## Running it

```bash
# The whole table: every workload, every engine, wall-clock MB/s.
cargo bench -p jsonsm-bench

# One workload, one engine, no warm-up — the shape `perf stat` wants.
cargo build --release -p jsonsm-bench
perf stat -r 5 -e instructions,cycles \
    target/release/jsonsm-bench match/late_field sse2 2000
```

`cargo bench` and the binary are the same measurement reached two ways: the workloads, the
engine selection and the agreement checks all live in the library, and the two targets are a
few lines each. Sweeps and repetitions are tunable without recompiling:

```bash
JSONSM_BENCH_ITERS=5000 JSONSM_BENCH_REPS=7 cargo bench -p jsonsm-bench
```

Run the binary with no arguments for the workload list. Engines are `scalar`, `sse2`, `avx2`,
`hybrid` and `slow`. `hybrid` is what `Backend::detect` returns on a CPU with AVX2: SSE2
kernels for the state machine, AVX2 for `skip_container` alone. `slow` is `jsonsm-slow`'s
reference interpreter, which parses to a `serde_json::Value` first. That parse is not overhead
to be subtracted; it is how a parse-then-interpret design works, and it is the baseline the
streaming engine is measured against.

The `CHECK` lines assert the reference matcher's answer against **every** available backend,
not just the detected one — each is a separate monomorphisation, so checking one says nothing
about the others, and every one of them gets a timed row.

## Which number to believe

**Instructions retired is the metric of record. Wall-clock is not.**

On a typical development machine, *identical* code has measured 15.2 µs or 20.4 µs on the same
workload depending only on what else was linked into the binary — adding one monomorphisation
moved untouched code by 27%. Instruction counts reproduce across rebuilds to five significant
figures. So a wall-clock delta under ~5% between two builds means nothing, and `cargo bench`
output is for a quick look, not for a decision.

Three rules worth stating explicitly:

- **Report cycles too, and read them together.** Instructions retired is not a proxy for
  speed. Two changes to the same loop have been ranked in *opposite* orders by the two
  metrics, and the one that retired six times more instructions was the one to discard: the
  loop was latency-bound, so deleting off-critical-path work bought nothing. If instructions
  fall and cycles do not, check IPC and look for a dependency chain.

- **A cross-build cycle comparison cannot resolve a few percent.** To settle one, put both
  implementations in *one* binary behind a runtime flag and alternate the arms over several
  rounds, taking medians. `Backend` already works this way, which is why every scan backend is
  selectable at runtime rather than by feature.

- **`tokenize/*` is not a control for matcher work.** It shares no code with the matcher. It
  is a good control for "is this the same build?" in instruction terms, and it has been stable
  to five significant figures across matcher changes that moved matcher benchmarks 21% in
  cycles. Say which of the two you mean.

- **A same-binary comparison is still a comparison of two layouts.** `Backend` is runtime-
  selected so both arms share a build, which is necessary and not sufficient: the arms are
  different monomorphisations, and one can lose the layout lottery. Changing which variant
  `Backend::detect` *returns* — nothing the timed path calls, since the harness forces its
  backend — has moved a matcher workload by tens of percent in cycles at identical instruction
  counts. When cycles move and instructions do not, that is the null hypothesis, not a
  discovery.

- **Confirm the change is actually in the binary, and that the baseline is too.** Instruction
  counts here reproduce to about six significant figures — five runs of one binary have spanned
  800 out of 128 million — so a ~2% gap between two builds of supposedly identical source is
  not noise, it is evidence the sources differ. Two failures follow. An `#[inline]` attribute
  can be lost to an edit of the doc comment that describes it, shipping prose that claims an
  optimisation absent from the binary; `nm` for the symbol settles it in one command. And a
  baseline is a measurement too: a change measured against a build whose attribute had been
  dropped read as a regression, was reverted, and was in fact a win.

- **Read the disassembly; the sampled profile lies on these loops.** `perf record` attributes a
  sample some instructions past the one that actually cost, and on a per-element loop that skid
  is larger than the thing being measured. A `lea` that cannot cost anything has carried 97% of
  a function's samples, skid from a call three instructions earlier. Treat a hot-looking cheap
  instruction as evidence about what precedes it, and locate a per-element cost by reading the
  generated code or by a same-binary A/B, not by ranking symbols.

### Ablations

Pricing a mechanism by switching it off is only a measurement while **both arms still do the
same work**. Turning off a mechanism that has a decline path — one that falls back to code
which already exists — is safe, because the fallback does the same work by another route.
Deleting a step that other code depends on is not: suppressing the logic tree's per-element
reset leaves the body resolved after the first element, so the guard that skips already-decided
operations then skips the comparison itself, and the arm reports a large "saving" that is
really deleted work.

Two practical rules. Read any runtime switch **once, outside the measured loop** — reading one
with `env::var_os` per element has cost several hundred instructions per element, more than the
entire quantity under measurement. And state what the ablation cannot see: a fixture pair
isolates exactly one cost and is silent about every other, so identical numbers across two
configurations are evidence that the fixture cannot distinguish them, not that they are
equivalent.

`[profile.release] debug = true` in the workspace root is part of the measurement
configuration, not a convenience — see the comment there. So is `#[inline(never)]` on
`Backend::detect`, for the reason in the paragraph above; both are documented where they sit.

## Workloads

`tokenize/*` runs the tokenizer alone: the floor a matcher workload is measured against.

| workload | what it represents |
| --- | --- |
| `match/and_or` | the everyday filter — a few early fields, short-circuits |
| `match/late_field` | one field ~86% into the record; dominated by structural skipping |
| `match/cross_field` | two fields of one record compared to each other — deferred to the after-node, so the logic tree cannot resolve mid-scan and the whole record is walked however early the fields appear |
| `match/any_loop` | a quantifier over a small array of strings |
| `match/skip_big` | one field behind a 1600-element array and 200 objects; ~99% skipped |
| `match/any_str_*` | ANY over N strings, never matches, so the loop is exhausted |
| `match/every_str_*` | EVERY over N strings, all satisfy, so it cannot short-circuit |
| `match/any_obj_*` | ANY over N objects, with a path into each |
| `match/any_obj{0,3,7}_*` | the same, with 0/3/7 fields per element the expression never names |
| `match/skip_str_220` | the same document as `any_str_220`, but the array is skipped whole |
| `match/wide_body_*` | a loop body of twelve terms — every other body here is a logic tree of one to three nodes, too small to show anything about logic trees |
| `match/absent_body_*` | a loop body naming fields the elements lack, so it cannot be evaluated and must be *sealed* per element — the path every other workload skips |
| `match/any_int_*` | ANY over N eight-digit integers — the same loop over a number rather than a string |
| `match/any_smallint_*` | the same over two-digit integers, so the *width* of a number varies with element count held fixed |
| `match/any_float_*` | ANY over N floats, which reach the fractional parse rather than the integer one |

The `*_20` / `*_220` pairs differ only in element count, so
`(insn(220) − insn(20)) / 200 / iters` gives the cost of one loop element with the document
prefix, the compile and the harness all cancelled out. That is how every per-element figure
for this project is derived, and it is a far better question than "what percentage of this
benchmark is the loop?".

`any_int` / `any_smallint` are the same two-slope trick for a number's digit count: the
tokenizing floor is identical at both widths, so the difference between them is purely what the
matcher spends per digit. Without both, tuning the numeric path would be tuning it against
eight-digit integers, which real documents rarely contain.

The `any_obj{0,1,3,7}` family varies a *second* quantity — how many fields an element carries
that the expression does not name — with element count held at the same 20/220. Two slopes
separate what scales with the object's width from what is paid once per element, which one pair
alone cannot do. The floor is linear in width at **223 instructions per unnamed key** (measured
223/223/223 across the three gaps, which is the check that the fixtures differ in nothing else);
the matcher is **flat past three keys**, because it stops walking and crosses the remainder in
one scan. `any_obj_*` itself, at one unnamed key, is the narrowest case in the family and the
worst for that exit.

## Fixtures

`corpus/people/` is `people.json` split into one compact buffer per record, **in the
document's original key order**. It is checked in rather than derived at startup because
deriving it is exactly what goes wrong: re-serialising through `serde_json::Value`
alphabetises the keys — its default object map is a `BTreeMap` — moving `favoriteFruit` from
index 21 to about 9, which turns `match/late_field` into an early-field benchmark. The Go
harness keeps a byte-identical copy of these files for the same reason.

`people.json` and `bigvector.json` themselves are read from `jsonsm/testdata/`, where they
already live as library fixtures, rather than copied here to drift.

## Comparing against gojsonsm

The Go side lives in the `gojsonsm` repository under `bench/`. It keeps its **own copy** of
this corpus, so each repository builds and benchmarks standalone — but the copies must stay
byte-identical, which is the one thing the comparison rests on. That repository's
`bench/corpus/SHA256SUMS` pins its bytes; to check the two against each other:

```bash
diff -r jsonsm-bench/corpus/people ../gojsonsm/bench/corpus/people
cmp jsonsm-bench/corpus/skipbig.json ../gojsonsm/bench/corpus/skipbig.json
cmp jsonsm/testdata/bigvector.json  ../gojsonsm/bench/corpus/bigvector.json
```

Agreement is not assumed either: the `CHECK` lines both harnesses print report hit counts per
workload, and they must match before any throughput number means anything. Two engines
answering different questions will still produce two numbers, and one will look better.
