# Matching semantics

This document is the behavioural contract of the matcher: what a value is, how two values
compare, how absence propagates, and what each operator answers. It describes the engine as
configured with `DefaultCollation`; the parts a collation may change are called out in
[Collation as an extension seam](#collation-as-an-extension-seam).

## The value model

Every value the engine handles — whether scanned from a document, written as a literal in an
expression, or produced by a built-in function — has one of seven logical types. Types are
totally ordered by N1QL collation precedence, and that order is what decides a comparison
between two values of different type.

| Order | Logical type | Notes |
| --- | --- | --- |
| 1 | missing | A field the document does not contain, or a function that could not produce a value. Not writable as a literal. |
| 2 | null | The JSON `null`. Distinct from missing. |
| 3 | boolean | `false` sorts before `true`. |
| 4 | number | One logical type; `i64`, `u64` and `f64` are representations of it, not separate types. |
| 5 | string | Compared by decoded (logical) text. |
| 6 | array | Compared as raw JSON bytes. |
| 7 | object | Compared as raw JSON bytes. |

`missing` and `null` are separate values and never equal each other: `missing < null`. A field
present in the document holding `null` is a value; a field that is not there is not.

Values scanned from a document borrow the document's bytes. Numbers and escaped strings are
kept in their raw form and decoded only when a comparison actually needs them, which is not
observable — it changes no answer.

## Comparison

Comparison is strict N1QL: there is no cross-type coercion anywhere.

- **Different logical types are never equal**, and order purely by the type precedence above.
  `5` never equals `"5"`, and `true` never equals `1`. This holds under every collation.
- **Numbers compare exactly, with no epsilon**, and exactly across `i64`, `u64` and `f64`
  including at magnitudes where an `f64` cannot represent the integer. `1 == 1.0` is true;
  `1 == 1.0000001` is false; `i64::MAX < 9223372036854775808.0` is true.
- **Strings compare by decoded value.** Escaped and literal spellings of the same text are
  equal, so `"café"`, `"café"` and a `café` string constant are all the same string.
  Ordering is by decoded bytes, which for UTF-8 is codepoint order.
- **Object keys follow the same rule**: a key is matched by its decoded bytes, so
  `{"na\tme": 1}` is named by the path component `na<TAB>me`. There is no Unicode
  normalisation and no case folding.
- **Arrays and objects compare as raw JSON bytes.** Both operands are the document's bytes
  between (and including) their brackets.

That last rule has consequences worth stating outright: container equality is byte-exact.
`[1,2]` and `[1, 2]` are *not* equal, because their bytes differ. `{"a":1,"b":2}` and
`{"b":2,"a":1}` are *not* equal, because key order is part of the bytes. Two containers that
are unequal are ordered lexicographically by those bytes, which is deterministic but is not
element-wise N1QL ordering.

Same-type comparison otherwise does what one expects: `missing == missing`, `null == null`,
`false < true`.

## Three-valued (Kleene) logic

This is the part most likely to surprise. **A comparison against a field the document does
not contain is UNKNOWN, not false.**

UNKNOWN is a value, not a gap. It combines with `true` and `false` by Kleene's tables:

**AND**

| AND | true | false | unknown |
| --- | --- | --- | --- |
| **true** | true | false | unknown |
| **false** | false | false | false |
| **unknown** | unknown | false | unknown |

**OR**

| OR | true | false | unknown |
| --- | --- | --- | --- |
| **true** | true | true | true |
| **false** | true | false | unknown |
| **unknown** | true | unknown | unknown |

**NOT**

| operand | NOT |
| --- | --- |
| true | false |
| false | true |
| unknown | unknown |

The decisive line is the last one: **`NOT` of UNKNOWN is UNKNOWN.** Negation cannot rescue an
absent field. `age != 50` is compiled as `NOT (age == 50)`, so on a document with no `age`
field the inner comparison is UNKNOWN, the negation is UNKNOWN, and the document does not
match. The same is true of `NOT (age == 50)` written out, of `age IS NOT NULL`, and of any
depth of nesting around them.

Only **at the root** do UNKNOWN and false mean the same thing. A match is a boolean, so a root
that resolves to UNKNOWN reports "no match" exactly as `false` does. Everywhere below the root
the distinction is live, and it is what stops a negation turning absence into a match.

Sources of UNKNOWN:

- a comparison in which either operand is missing — an absent field, or a built-in function
  that could not produce a value;
- a `matches` (pattern) operation against an absent field;
- a quantifier whose target array is absent, or is present but is not an array;
- a quantifier over elements it could not evaluate (see below).

### The EXISTS exception

`EXISTS` is deliberately different. It asks about **presence**, so absence is not a failure to
answer — absence *is* the answer. An `EXISTS` against a missing field yields `false`, a
definite value.

That is precisely what keeps `NOT EXISTS` usable: `false` negates to `true`, so
`NOT EXISTS(a)` matches a document with no `a`. `EXISTS` and `NOT EXISTS` are therefore the
way to select on absence — no comparison operator can do it.

```
document: {"name": "Ada"}

age != 50           →  unknown  →  no match
NOT (age == 50)     →  unknown  →  no match
age IS NOT NULL     →  unknown  →  no match
NOT EXISTS(age)     →  true     →  match
age IS MISSING      →  true     →  match
```

## Quantifiers

A loop binds a variable to each element of an array and evaluates a sub-expression per
element. There are three quantifiers, and they differ on the cases that matter.

- `ANY` — at least one element satisfies the body. Behaves as an OR over elements.
- `EVERY` — every element satisfies the body. Behaves as an AND over elements, so it is
  vacuously true over an empty array.
- `ANY AND EVERY` — `EVERY`, plus the array must be non-empty.

| Elements | ANY | EVERY | ANY AND EVERY |
| --- | --- | --- | --- |
| All true | true | true | true |
| Mixed true and false | true | false | false |
| All false | false | false | false |
| Empty array `[]` | false | true | false |
| Some unknown, at least one true | true | unknown | unknown |
| Some unknown, at least one false | unknown | false | false |
| Some unknown, no definite element | unknown | unknown | unknown |
| Target field absent | unknown | unknown | unknown |
| Target present but not an array | unknown | unknown | unknown |

Note the two rows at the bottom against the empty-array row. **An absent array is not an
empty array.** `EVERY` over `[]` is vacuously true and negates to false; `EVERY` over a field
that is not there is UNKNOWN and negates to UNKNOWN, so `NOT EVERY(…)` matches in the first
case and not in the second.

An element for which the body cannot be evaluated — typically because that element lacks the
field the body names — is UNKNOWN for that element. It does not end the loop: `ANY` can still
be settled `true` by a later element that matches, and `EVERY` by a later element that fails.
What it denies is the *other* verdict, since neither `ANY` can conclude `false` nor `EVERY`
conclude `true` over an element it could not read.

## Field paths

A field reference is a root variable plus a path. The root is either the document (the
implicit `$doc` variable) or a variable bound by an enclosing loop. The path is a sequence of
steps, each either an **object key** or a **zero-based array index**.

- `name.first` — two object keys.
- `tags[0]` — an object key then an array index.
- An empty path denotes the rooted value itself: for the document root, the whole document;
  for a loop variable, the current element.

Object keys and array indices are distinct kinds of step. An object whose key happens to be
spelled `[0]` is reached only by a key step, never by an index step.

A loop body may reference fields from any **enclosing** scope as well as its own — an outer
loop's element, or the document root. Such a reference is order-independent: the engine defers
whatever must wait until the enclosing scope has been read, so a body comparing `t.id` against
a document-level `wanted` behaves the same whether `wanted` appears before or after the array
in the document.

The one restriction is on the array a loop **iterates**: it must be a field of the innermost
scope in force where the loop is written. A loop whose target array comes from an enclosing
scope is a compile error (`CompileError::CrossContext`).

## Built-in functions as operands

Functions may appear wherever an operand may, including as arguments to other functions.
Numeric functions compute in `f64` and return a number, which still compares exactly against
integer constants.

A function that cannot produce a value returns **missing**, which makes any comparison over it
UNKNOWN. That covers: an unknown function name, the wrong number of arguments, a non-numeric
argument, division or modulo by zero, and any result that is not finite (so `sqrt` of a
negative number is missing, not NaN). `DATE()` converts an ISO-8601 date string to epoch
seconds as a number — so date comparisons are ordinary numeric comparisons — and returns
missing for a non-string or unparseable argument.

## Field projection

Projection captures field values during the same single scan that evaluates the expressions,
without a second pass over the document.

A projection is a list of paths rooted at the document, in a caller-chosen order; the index of
a path in that list is the index used to read its value back. The empty path projects the whole
document. Adding the same path twice gives it two indices that report the same value.

```rust
let projection = Projection::new().field(["name", "first"]).field(["age"]);
let def = compile(&exprs, &projection, &DefaultCollation)?;
let out = matcher.matches(doc)?;
let first = out.projected(0);          // Option<FastVal>, borrowing `doc`
let present = out.is_projected_present(1);
for (path, value) in out.projections() { /* … */ }
```

`projected(i)` returns `None` when the field was absent from the document, and otherwise a
value borrowing the document's bytes — an escaped string is decoded only if the caller asks
for its decoded form.

**Capture is independent of whether the document matched.** A projected field present in the
document is captured either way, and the caller decides what to do with it. This is why the
scan does not stop the moment the boolean result is decided: it also waits until every
projected field has been seen. Compiling zero expressions with a non-empty projection gives a
projection-only definition — it never matches, but it still extracts every projected field.

Captured values borrow the document, and the outcome borrows the matcher, so both stay valid
exactly until the next match on that matcher.

## Collation as an extension seam

The `Collation` trait is where comparison policy and pattern compilation live. It supplies two
things:

- `compare(a, b)` — the ordering of two values, plus whether that ordering was a meaningful
  within-type comparison or a cross-type result resolved by type precedence;
- `compile_matcher(pattern)` — turns a pattern string into a runtime matcher for the `matches`
  operator.

`DefaultCollation` implements the strict-N1QL rules described above, and backs `matches` with
the standard `regex` crate: unanchored "contains" matching against the **decoded** string
value. A non-string value never matches a pattern (a definite `false`, not UNKNOWN). A pattern
must be a constant string in the expression; it is compiled once, when the expression is.

Two things are **not** a collation's choice:

- **Cross-type equality.** A collation may decide how two values *of the same type* order. It
  may not make values of different types interchangeable — `5` never equals `"5"` under any
  collation.
- **Absence.** A missing operand does not reach the collation as a comparison whose result it
  gets to declare. It yields UNKNOWN, which the engine propagates by the Kleene tables above.

A collation is supplied at compile time, and the matcher that runs a definition should use the
same collation the definition was compiled with.

## Matching several expressions at once

Any number of expressions compile into a single `MatchDef` and are evaluated in one scan. They
are joined so that every one is fully evaluated rather than short-circuited away, and each
reports its own result:

- `matched()` — the OR of all the expressions (`false` if none were compiled);
- `expression_matched(i)` — expression `i`'s own result, tracked independently.

With a single expression, `expression_matched(0)` is the same thing as `matched()`.
