# Design

How the engine is put together, and why it is shaped this way. For what it *means* see
[semantics.md](semantics.md); for what it refuses see [limits-and-caveats.md](limits-and-caveats.md).

## The shape of a match

A match is one pass over the document bytes. There is no parse tree, no intermediate value
map, and no allocation on the hot path.

```
  expression  ──compile──>  MatchDef  ──┐
                                        ├──>  FastMatcher::matches(bytes)  ──>  MatchOutcome
  document bytes  ──────────────────────┘
```

`compile` turns one or more expressions into a `MatchDef`: an **exec trie** describing which
fields matter and what to do at each, plus a **logic tree** holding the boolean structure.
Compilation happens once; a `MatchDef` is immutable and shared, and a `FastMatcher` borrows
one per thread.

`matches` then walks the document and the exec trie in lockstep. At each field the expression
names it runs the attached operations and records their results into the logic tree; every
other field is crossed structurally without being interpreted. The scan stops as soon as the
logic tree's verdict is settled and every projected field has been captured.

## The four structures

**The tokenizer** (`tokenizer`) is a state machine over bytes, generic in a `Scan` trait that
supplies three bulk primitives: find the next string event, skip whitespace, skip digits. A
scalar implementation and SIMD implementations satisfy the same trait, so there is exactly one
state machine and backend parity is structural rather than tested into existence. Each
primitive is contracted to stop no later than the byte the state machine must inspect, and the
state machine re-inspects it — so a scanner that stops early is slow, never wrong.

**The exec trie** (`compile::ExecNode`) has one node per distinct field path prefix the
expression names. A node carries its object-key children, its array-index children, the
operations to run against its value, the loops rooted at it, and the slot its value is stored
into if something needs it later. Children by key live in a flat `KeyMap`, not a hash map: a
node names a handful of keys while the document offers one for every field it contains, so the
document's keys are what get looked up, and comparing against three entries beats hashing to
probe them.

**The logic tree** (`logic_tree`) is the expression's boolean structure as a flat array in
pre-order, so every subtree is a contiguous range. Leaves are set by operations as they run
and results propagate upward with short-circuiting, which is what lets a scan stop early.
Values are three-valued — see [semantics.md](semantics.md) — and a loop body evaluates behind a
*stall boundary* that keeps one element's result from escaping into the enclosing expression.

**The matcher** (`matcher`) is the walk itself, plus the skip paths.

## The ideas that make it fast

**Look the key up before reading the value.** Most fields of most documents are named by no
expression, and those never cost a token: the key decides, and the value is crossed.

**Cross what does not matter without interpreting it.** A skipped value is crossed by counting
brackets and tracking string boundaries, not by tokenizing. This is where the largest wins are
on real documents, and it is the source of the validation caveat.

**Resolve keys and short scalars from raw bytes.** A node's children each know the exact bytes
a document key must have, so "which child is this?" is asked of the raw bytes directly — and
on a hit that also answers "where does the key end?" for free. The same trade applies to a
plain unescaped string value and to a container's opening byte: the tokenizer is a large
outlined function returning a token through memory, and for these shapes the byte already says
everything the token would.

**Stop as soon as the answer cannot change.** Three separate mechanisms: the logic tree
resolving, a *verdict* settling before the value does (an `AND` over an absent field can no
longer be true whatever else is found), and an object whose every named key has been supplied
crossing its remainder in one scan.

**Decide once what cannot change.** Anything fixed at compile time is computed at compile time
— constants are stored as finished runtime values, subtree extents are precomputed, and the
set of buckets a field's absence settles is precomputed per node. Anything fixed for a loop is
read once for the loop rather than once per element.

**Monomorphise the backend, do not branch on it.** The scan backend is a type parameter
resolved once per document, not an enum tested per bulk scan. `Backend::detect` picks it; the
matcher's public shape is unchanged by it.

## Decisions that look like oversights

Each of these is a road already taken and rejected; the shape of the code is the result.

**One collation per `MatchDef`, not per expression.** A caller needing expressions under
different collations compiles one definition per collation and groups them above the engine.
Attaching a collation to each expression would put a dispatch on the comparison path, which is
today resolved entirely at compile time.

**`Collation` will not grow a "string equality is byte equality" capability.** The trait takes
its operands by reference, which looks like it forces both into memory and invites a fast-path
hook. It is not the trait's problem: both operands already exist — one is the scanned value,
the other belongs to the `MatchDef` — so a comparison is a pair of discriminant tests and a byte
compare. Before proposing that an interface permit a shortcut, check whether one of the two
values is being *rebuilt* on the hot path.

**Constants are owned, not arena-allocated.** Storing string constants as slices into an arena
inside the `MatchDef` is structurally impossible rather than merely unattractive: `MatchDef` is
public and derives `Clone`, so arena-pointing constants would be self-referential and the
derived `Clone` would dangle them. Away from a tight loop the representation is not worth
anything anyway.

**The scan backend parameterises the tokenizer, not `FastMatcher`.** A tokenizer borrows the
document, so its lifetime is per-call rather than per-matcher; making the matcher generic over a
tokenizer needs a GAT-based factory and a second public type parameter to express that. Keeping
the parameter on the state machine also makes scalar and vector grammar parity structural.

**The backend stays a runtime choice.** Which vector width wins is a property of the
microarchitecture and the document shape, not a fact — the ranking has already inverted once
under this project's own measurements. Compiling down to whatever the baseline ISA guarantees
would bake in one machine's answer.

**Vector kernels use `std::arch`, not `core::simd`.** Portable SIMD is nightly-only; this
project pins stable and its correctness gate is `cargo test --workspace`, so a nightly backend
would be untested by the thing that tests everything else. Portability comes from the `Scan`
trait and its scalar fallback instead.

**The reference interpreter is a separate crate.** `jsonsm-slow` is not a module or a feature of
`jsonsm`, so that `serde_json` never appears in the engine's dependency graph — not even
optionally.

## Vector backends

`Scan` has three implementations on x86-64. `Sse2Scan` needs no detection — SSE2 is in the
base ABI — and its kernels inline into the state machine directly. `Avx2Scan` reaches its
kernels through a `#[target_feature]` context opened once around the whole scan, because
opening one per kernel makes every bulk scan a non-inlinable call.

The two kernel groups want different widths. The state machine's primitives run per token over
runs a few bytes long, where a 32-byte load and movemask cannot amortise; `skip_container`
runs once per skipped value and may cross kilobytes, where it clearly does. `HybridScan` takes
each where it wins, and is what `Backend::detect` returns on a CPU with AVX2.

All `unsafe` in the workspace lives in the `simd` module, behind the default `simd` feature. A
build without it keeps a crate-level `forbid(unsafe_code)`.

## Testing

Three layers, deliberately independent.

**A differential oracle.** `jsonsm-slow` interprets the same AST over a `serde_json::Value`
tree. Sweeps of generated (expression, document) pairs assert the two agree. The oracle
reuses parts of `jsonsm` where duplicating them would add nothing — and that is exactly where
the sweep is blind, so anything the oracle *imports* needs a unit test against an independent
reference instead. Three-valued logic is implemented twice on purpose for this reason.

**Parity with gojsonsm.** `jsonsm-json` ports the reference implementation's own expression
suite over the same corpus and asserts the same matched-document sets.

**Every backend, not the detected one.** Each `Scan` implementation is separately compiled
code, so tests iterate `Backend::available()` and require the backends to agree with each
other as well as with the oracle.

**The documented semantics, executed.** `docs/semantics.md` is the user-facing contract, and
every claim in it that reduces to "this expression, on this document, matches or does not" is a
row in `jsonsm-n1ql/tests/documented_semantics.rs`. Prose cannot notice when the code moves out
from under it.

A generator's blind spots have repeatedly mattered more than its size. It is worth asking, of
any new behaviour, what the generator's *alphabet*, *shapes* and *encoding* make impossible —
a `serde_json::Value` cannot hold a duplicate key, and `serde_json::to_vec` never emits
whitespace before an array element, so both cases need hand-written tests.

### Three practices worth keeping

**A test is validated by breaking the code, not by passing.** Adequacy here is established by
deliberately mutating the implementation and confirming the test fails. This is not a formality:
a broken vector kernel once passed the entire differential sweep, and a table of semantics cases
that looked complete against the document missed `EXISTS` on an absent field, because a
current-scope absent field is settled by the seal and never reaches the branch that decides it.
Several test doc comments name the mutations they exist to catch; those lists are worth
maintaining when the tests change.

**The debug suite is where the assertions live.** `jsonsm` carries a handful of `debug_assert`s
guarding invariants nothing else checks — most importantly that logic-tree subtrees are
contiguous, which `reset_node` and `seal_node` are built on and which no error would otherwise
report. They run only under `cargo test --workspace` in a debug profile. One test that trips one
of them disables the whole set, so the debug run has to stay green; a test that must demonstrate
what an assertion forbids belongs on an unguarded inner function, not on the guarded entry
point.

**After a mechanical edit to a test module, diff the inventory.** A scripted rewrite once
computed a function's end by searching for the next closing brace, matched the end of the
enclosing module instead, and deleted nine tests. `cargo test` stayed green — what had been
removed was the tests. Clippy noticed, reporting a helper whose only callers were gone. Run
clippy, and compare the list of test names against the previous revision.

## Performance

`jsonsm-bench` is the measurement harness and its
[README](../jsonsm-bench/README.md) explains which number to believe. In short: instructions
retired is the metric of record, wall-clock between two builds is not, and cycles are reported
alongside because the two can rank changes in opposite orders.

The harness reports throughput for every workload, the reference interpreter for scale, and —
through fixture pairs differing only in element count — the cost of a single array element,
which is a far more useful number than a percentage of a whole-document benchmark.
