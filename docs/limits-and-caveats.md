# Limits and caveats

What the engine requires of its input, what it guarantees, and where it stops.

## The input contract: well-formed JSON

**The engine requires well-formed JSON.** This is a firm precondition, not a best-effort
tolerance. The engine's job is to decide an expression against a document, not to validate the
document, and it takes that licence wherever validating would cost work on bytes no expression
cares about.

The consequence is concrete. Regions of the document that no expression names are crossed
**structurally** rather than parsed: the engine finds where such a region ends and moves on,
without recognising what is inside it. Malformed content there may therefore go undetected,
and the engine's answer for a malformed document is not defined.

What **is** still checked in a skipped region:

- **Bracket balance.** `{` `[` and their closers are counted, and an unbalanced region is an
  error. Nesting inside a skipped region also counts toward the depth ceiling below.
- **String termination.** A string must reach a closing quote before the end of input; a
  backslash consumes the byte after it. An unterminated string is an error.

What is **not** checked in a skipped region:

- **Literal spelling.** `{"a": tru, "b": 2}` is accepted when no expression names `a`.
- **Number syntax.** Leading zeros, doubled decimal points, a bare exponent — `01`, `1.2.3`,
  `1e` — are all crossed without complaint.
- **`\uXXXX` escape validity** inside a skipped string. Only the closing quote is sought.
- **Control characters** inside a string within a bulk-skipped container. They are explicitly
  ignored there, though a skipped scalar string at a level the engine is still walking does
  reject them.
- **Object structure** inside a bulk-skipped container: keys, colons and commas are not
  recognised at all.

Object keys are validated — escapes included — only while the engine is walking the object key
by key. It stops doing so in two cases: when the expression names no key at this level at all,
and once every key it *does* name has been supplied, after which the remainder of the object is
crossed in bulk. Keys crossed that way get the same treatment as anything else in a skipped
region.

Two shapes are still rejected while walking, because they would otherwise let one malformed
value swallow the members after it: a value position with no value in it (`{"a":,}`), and an
unterminated or control-character-bearing string.

**Fields an expression does name are fully tokenized**, always. A value that is compared,
matched against a pattern, or captured by a projection goes through the complete state machine
and produces a syntax error if it is malformed. The divergence above applies only to bytes no
expression reads.

Failures surface as `MatchError`: `Tokenizer` for a syntax error in a value that was read,
`Structure` for a document shape the walker could not make sense of, and `TooDeep` for the
limit below.

## UTF-8 is not validated on string values

String values are compared as bytes and are never validated as UTF-8. This follows from the
well-formed-JSON requirement rather than contradicting it: JSON is UTF-8 by specification, so a
document key or value that byte-compares equal to a UTF-8 constant is UTF-8 by construction —
and validating every string in every document would be work on every field, matched or not.

Practical consequences:

- A string containing invalid UTF-8 still compares, deterministically, by bytes. It cannot
  equal any constant from an expression, because constants are Rust `String`s and therefore
  valid UTF-8.
- Comparison remains total: no comparison can fail or panic on such a value.
- Where UTF-8 genuinely is required, it is checked at the point of use and a failure is a
  definite answer rather than an error. A pattern match against a value that does not decode as
  UTF-8 does not match; `DATE()` over one returns missing.
- Decoding `\uXXXX` escapes maps unpaired or invalid surrogates to `U+FFFD` rather than
  failing.

Values that **are** compared are still fully tokenized, so escapes, control characters and
string termination in them are checked exactly as a validating parse would check them.

## Depth limits

| Limit | Value | Applies to | Error |
| --- | --- | --- | --- |
| `matcher::MAX_DEPTH` | 1024 | Container nesting in a document | `MatchError::TooDeep` |
| `compile::MAX_EXPR_DEPTH` | 256 | Nesting of an expression tree | `CompileError::TooDeep` |

The document limit is enforced uniformly across matched and skipped regions, so whether a
document is rejected does not depend on whether an expression happened to name a field in the
deep part or cross over it.

The expression limit is checked iteratively, so measuring an adversarially deep tree is itself
safe. The N1QL front end applies it before returning an AST, ahead of its own recursive
name-resolution pass. One residual remains: a caller that *builds* an over-deep AST by hand
still pays a recursive `Drop` for it when it goes out of scope.

## Duplicate object keys

A repeated object key resolves to its **first** occurrence. Once the document has supplied
every key the expression names at a given level, the rest of that object is crossed in bulk, so
a second copy of an already-seen key is never read.

This is a licence, not a guarantee. RFC 8259 says object names *should* be unique and leaves
the behaviour unpredictable when they are not, so an object carrying the same field twice is
not valid input to this engine. The rule is documented because the answer is observable, not
because a valid document can reach it.

## Numeric domain

Numbers are 64-bit. An integer literal parses as `i64` where it fits, otherwise as `u64` where
it fits, and otherwise falls back to `f64` — which is **lossy** for integers beyond the `u64`
range or below `i64::MIN`. This is a documented rule, not an accident: values outside the
64-bit integer range are compared as the nearest `f64`.

Within the three representations, comparison is exact — including across them, and including at
magnitudes where an `f64` cannot represent a neighbouring integer. There is no epsilon
anywhere.

Built-in numeric functions compute in `f64` regardless of their arguments' representation, so a
function applied to a large integer is subject to `f64` precision. A non-finite result is
missing rather than a number.

## Unsupported compile cases

`compile` rejects the following. Each produces the stated `CompileError`.

| Case | Error |
| --- | --- |
| A loop whose target array comes from an enclosing scope | `CrossContext` |
| A loop whose `in` operand is not a field reference at all | `BadLoopTarget` |
| A function with more than one field argument from the innermost scope | `Unsupported("function with multiple local field arguments")` |
| A function where a plain field reference is required — the operand of `exists` or `matches` | `Func` |
| A `matches` pattern that is not a constant string | `BadPattern` |
| A field naming a variable that is not in scope | `UnknownVariable(id)` |
| An operand node (literal, field, function) used where a boolean is required | `NotABoolean` |
| A boolean node used where an operand is required | `NotAnOperand` |
| An expression nested deeper than `MAX_EXPR_DEPTH` | `TooDeep` |
| A pattern the collation rejects, or a collation with no pattern support | `Collation(…)` |
| An internally malformed logic tree | `Tree(…)` |

`exists` and `matches` **do** accept a field from an enclosing scope, and so does either side
of a comparison; only a loop's target array is restricted to the innermost scope.

The JSON-array front end additionally rejects array and object **literals** in expressions
(`UnsupportedValue`): container comparison is byte-exact, so a literal container could not be
compared reliably against a document's formatting.

## Other things worth knowing

- **Container comparison is byte-exact.** `[1,2] != [1, 2]`, and object key order is
  significant. Unequal containers order lexicographically by their raw bytes, which is
  deterministic but is not element-wise ordering.
- **An empty or whitespace-only document is not an error.** It simply matches nothing, and any
  projected field reports as absent.
- **A document that is a bare scalar is valid input.** Expressions naming fields inside it
  resolve as absent.
- **A path that runs through a scalar is absent, not an error.** With `a.x` against
  `{"a": 5}`, the comparison is UNKNOWN.
- **Threading.** A `MatchDef` is `Sync` and can be shared; a `FastMatcher` borrows one and
  should be created per thread. Matcher state is reset on every call.
- **Borrow lifetimes.** The outcome borrows both the matcher and the document, so captured
  values stay valid exactly until the next match on that matcher — enforced by the borrow
  checker.
- **Collation consistency.** A matcher should run with the same collation the definition was
  compiled with; nothing checks this.
- **Case sensitivity.** Field names, string comparison and pattern matching are all
  case-sensitive and apply no Unicode normalisation.
- **Document transformation is out of scope.** This library answers whether a document matches
  and captures the fields an expression asks for. gojsonsm's `jsonComposer`
  (`MatchAndRemoveItemsFromJsonObject`) has no counterpart here and is not planned.
- **`no_std` is not supported and is not planned.** The engine is portable Rust and the vector
  backends are feature-gated with a scalar fallback, but the crate depends on `std`.
