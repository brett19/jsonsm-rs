# Differences from gojsonsm

`jsonsm-rs` is a port of [`gojsonsm`](https://github.com/couchbase/gojsonsm). It is
feature-complete against the Go implementation, but it is not bug-for-bug compatible. This
document records every place the two disagree on an answer, and why.

Throughout, "gojsonsm" means its `FastMatcher` unless `SlowMatcher` is named explicitly.

## Semantics

### Absent fields are unknown, not false

A comparison whose operand is not present in the document yields `Unknown` in jsonsm-rs.
`Unknown` propagates by Kleene's strong three-valued tables and collapses to "no match" only
at the root. Decisively, `NOT Unknown` is `Unknown`.

gojsonsm settles a leaf that never ran to `false` and lets the tree above it invert that. So
on `{"other":1}`:

| expression | gojsonsm | jsonsm-rs |
|---|---|---|
| `age == 50` | false | false |
| `age != 50` | **true** | **false** |
| `NOT (age == 50)` | **true** | **false** |
| `NOT NOT (age == 50)` | false | false |
| `NOT (age == 50 AND b == 1)` on `{"b":1}` | **true** | **false** |
| `age == 50 OR b == 1` on `{"b":1}` | true | true |
| `NOT EXISTS(age)` | true | true |

Why: under the Go behaviour `NOT (age < 50)` and `age >= 50` select different document sets,
so the most ordinary rewrite a query planner performs silently changes results — and only on
documents that happen to lack the field. Three properties are mutually unsatisfiable: (A)
`a != b` is equivalent to `NOT (a == b)`; (B) `NOT p` matches exactly the documents `p` does
not; (C) a comparison never matches a document lacking the field. gojsonsm drops (C);
jsonsm-rs drops (B), because losing (B) costs only concision — `NOT EXISTS(x) OR x != v` is
the exact complement of `x == v` — while losing (A) cannot be recovered. (A) + (C) is also
what N1QL does.

`EXISTS` is the deliberate exception: absence *is* its answer, so an absent field makes an
`EXISTS` leaf definitely `false`, keeping `NOT EXISTS` true. `NOT EXISTS` is the way to select
on absence.

### Float equality is exact

jsonsm-rs compares numbers exactly, with no tolerance, and exactly across the `i64` / `u64` /
`f64` boundary at magnitudes where an intermediate `f64` would lose bits.

gojsonsm's `compareFloat` treats two numbers as equal when `math.Abs(a-b) < EPSILON` with
`EPSILON := 0.0000001`, and reaches every numeric comparison through `f64`.

Why: a fuzzy equality makes `==` non-transitive and makes `<` disagree with `!(>=)` near the
threshold. The Go source carries its own note that the epsilon "possibly even 0 if we want to
force exact matching".

### Strings compare by decoded value

jsonsm-rs compares strings by their decoded (logical) contents, so `"é"` and `"é"`
compare equal and order by codepoint.

gojsonsm's `compareStrings` re-encodes both operands with `toJsonStringInternal` and compares
the JSON-*escaped* bytes, so the two spellings of the same string are unequal there.

Why: escaping is a property of the encoding, not of the value. The decoded comparison is done
without allocating — `memcmp` when both sides are already plain, a block scan when one side
carries escapes, and a streaming decode when both do.

### Strict types, no implicit conversion

`DefaultCollation` performs no cross-type coercion: operands of different types are ordered by
type precedence (`missing` < `null` < boolean < number < string < array < object) and never
compare equal, so `5 != "5"`.

gojsonsm carries an `ImplicitConvTable` and a `userDefined` flag that coerce strings to
booleans, numbers, and times during comparison.

Why: comparison policy is a compile-time strategy in jsonsm-rs (the `Collation` trait), so a
coercing collation can be supplied without changing the engine. The default states one rule.

### Dates are an explicit function

jsonsm-rs has no `["time", …]` expression node; `jsonsm-json` rejects it with an explanatory
error. Dates are expressed as `DATE(str)`, which parses ISO-8601 to epoch seconds so the
comparison is numeric.

Why: an implicit string-to-time coercion inside a comparison is invisible at the call site and
inconsistent with strict typing.

### Container literals are rejected, not silently mis-compared

`jsonsm-json` rejects an array or object literal inside `["value", …]`, which gojsonsm accepts.

Why: a literal container could only be compared byte-exactly against document bytes (see
container comparison below), so it would be whitespace- and key-order sensitive and would
compare unreliably. An explicit error beats a wrong answer.

## Bugs fixed

### Loop bodies leaked slots between elements

A comparison in a loop body that names two fields of the element is deferred: each operand is
stored in a slot and the comparison runs at the end of the element. gojsonsm's `matchLoop`
resets only the logic tree per element (`m.buckets.ResetNode(loopBucketIdx)`); slots are
cleared only per *document*, in `FastMatcher.Reset()`. An element that lacks a field therefore
reads the previous element's leftover value for it.

jsonsm-rs records, per loop, the set of slots its body owns and clears them at the top of each
iteration.

Reproducers, on gojsonsm:

- `["anyin",1,["field","arr"],["equals",["field",1,"x"],["field",1,"y"]]]`
  - on `{"arr":[{"x":1,"y":2},{"x":2}]}` — `FastMatcher` returns **true**; correct is false.
  - on `{"arr":[{"x":1},{"y":1}]}` — `FastMatcher` returns **true**; correct is false.
- `["everyin",1,["field","arr"],["equals",["field",1,"x"],["field",1,"y"]]]`
  - on `{"arr":[{"x":7,"y":7},{"x":7}]}` — returns **true**; correct is false.

In the first two cases gojsonsm's own `SlowMatcher` returns false, i.e. the reference
disagrees with the implementation.

### gojsonsm's reference matcher never covered EVERY loops

gojsonsm's `SlowMatcher.matchOne` switches on `OrExpr`, `AndExpr`, `AnyInExpr`, and the six
comparison expressions, then falls through to `panic("unexpected expression")`. `EveryInExpr`
and `AnyEveryInExpr` — and `NotExpr` — are absent, so the oracle panics rather than answering
for them.

Consequence: gojsonsm's differential testing validated `ANY` loops only. `EVERY` and
`ANY AND EVERY` were never checked against a reference at all, which is how the `everyin` case
above survived.

jsonsm-rs's oracle (`jsonsm-slow`) is a complete recursive interpreter over the AST, with its
own three-valued tables rather than the engine's, so the differential sweep compares two
independent routes to an answer.

### Array index paths never matched

Array index path segments (`a[0]`) parsed and compiled in gojsonsm-derived code but the
matcher's array branch only ever ran loops, so an indexed field always read as absent.
jsonsm-rs matches indexed children; projection paths accept indices too. See the
representation note below for how they are keyed.

## Representation and ordering

### Array indices are typed, not string keys

jsonsm-rs stores a node's indexed children as a sorted `(index, node)` vector, separate from
its object-key children.

gojsonsm formats each array index into a string key — `keyString = fmt.Sprintf("[%d]",
arrayIndex)` — and looks it up in `node.Elems`, the same map object keys use. An object key
literally spelled `"[0]"` and array element 0 are therefore the same lookup in Go and are
conflated; in jsonsm-rs they stay distinct.

Why: correctness, and no per-element string formatting or hashing on the hot path.

### Duplicate object keys resolve to the first occurrence

Once a document has supplied every key an exec node names, jsonsm-rs skips the remainder of
that object in bulk, so a later copy of an already-seen key is never read. gojsonsm walks
every key, and `serde_json` builds a map; both take the **last** occurrence.

A repeated key is not valid input to this engine — RFC 8259 says names should be unique and
calls the behaviour unpredictable otherwise — so this is a licence taken alongside the others
below, not a semantics choice. It is recorded because the *answer* changed.

### Containers sort lexicographically, not length-first

Both implementations compare arrays and objects **by their raw JSON bytes**. jsonsm-rs orders
purely lexicographically; gojsonsm's `compareObjArrData` compares lengths first and only then
bytes. They agree on equality and differ only in how two unequal containers sort.

The shared consequence, which matters more than the ordering difference: container equality is
byte-exact. `[1,2]` and `[1, 2]` are **not** equal, and object key order is significant.

Why byte-based at all: element-wise N1QL container ordering would diverge from the reference
implementation, so byte comparison is a deliberate parity choice rather than a stub.

### Skipped regions are not validated

A region of the document that no expression names is checked for bracket balance and string
termination and nothing else — its values' syntax and its keys' escapes are never checked. So
`{"a":[01,tru,,]}` is accepted when nothing names those fields, and `{"a\q":1}` is accepted
when nothing names that key. Fields an expression *does* name are fully tokenized. gojsonsm
tokenizes what it skips.

Why: the engine's contract is "decide this expression against this document", not "validate
this document". Adversarial nesting is still defended (see below) — terminating is not the
same as validating.

## Engineering differences

### Thread safety

gojsonsm has package-level mutable state reached from the matching path: `var
toJsonStringBuffer []byte` in `fastval.go` is a shared scratch buffer that
`toJsonStringInternal` truncates and appends to, and it is reached by `compareStrings` on
every string comparison against a numeric operand. Two matchers running concurrently race on
it.

jsonsm-rs has no global mutable state. `MatchDef` is `Sync` and shareable; `FastMatcher` is
`Send` and cheap to clone per thread from a shared `MatchDef`; all scratch is per-matcher.

The reference implementation's literal parsers were checked for the same hazard and are
function-local, so they are not a shared-state problem.

### Errors instead of panics

gojsonsm panics on malformed structure and on unexpected tokens. jsonsm-rs returns `Result`
throughout, with no panics in the compile or match hot paths.

### Depth limits

Structural skipping is iterative, and both sides are bounded: documents at `matcher::MAX_DEPTH`
(1024) and expressions at `compile::MAX_EXPR_DEPTH` (256), the latter checked iteratively so
measuring a deep tree is itself safe. gojsonsm recurses without a limit. One residual case is
documented: a caller who builds an over-deep AST by hand still pays a recursive `Drop` for it.
The matcher also borrows the input document (`&'a [u8]`) rather than owning or copying it.

## Additions

### Field projection

The caller names document field paths to extract. They are compiled into the `MatchDef`, their
values are captured during the *same* single matching pass that evaluates the expression, and
they are read back from the returned `MatchOutcome` (`projected`, `projected_by_path`,
`projected_path`). An empty path projects the whole document. Paths may contain array indices.

gojsonsm has no equivalent; extracting a field means parsing the document a second time.

### Cross-scope loop references at any depth

A loop body may read a field of any enclosing scope, not just the document from a root-level
loop, including `EXISTS` and `MATCHES` on an enclosing scope's field. The compiler tracks the
shallowest scope a body reads and defers each loop to the after-node of the scope containing
it. The one remaining restriction is that a loop's target array must be a field of the scope
the loop lives in.

### Pluggable collation

Comparison policy and pattern compilation are supplied at `compile()` time through the
`Collation` trait, which produces a `ValueMatcher` per pattern. Dispatch happens once per
operation against a pre-compiled constant, so the per-byte scan never sees it. `DefaultCollation`
implements the semantics described above, backed by the `regex` crate; a coercing or
PCRE-backed collation can be substituted without touching the engine.

## Shared deliberate choices

These are places jsonsm-rs matches gojsonsm on purpose, and are listed so they are not mistaken
for oversights.

- **Container comparison is byte-based**, with the equality consequences noted above.
- **The missing-value default is applied per node, not at the root.** Absence settles at each
  leaf and propagates upward through the connectives like any other value; gojsonsm's
  `binTreeState.Resolve` is the same back-to-front loop with `false` hardcoded where jsonsm-rs
  writes `Unknown`.
- **Multiple expressions compile into one `MatchDef`** and are evaluated in a single pass,
  with each expression's individual result reported — the same shape as Go's
  `Transform([]Expression)` / `ExpressionMatched`.
- **`EVERY` over an empty array is vacuously true.** An *absent* array is distinct: it is
  `Unknown` for every quantifier.
- **Key equality is byte equality of the decoded key.** JSON strings are UTF-8 by
  specification, so no collation and no normalisation enter into field-name matching.
