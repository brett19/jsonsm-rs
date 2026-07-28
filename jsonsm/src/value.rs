//! Runtime values ([`FastVal`]) and their string representation ([`FastStr`]).
//!
//! The design goal is to compare values — especially strings — while decoding as little
//! as possible and never allocating on the comparison hot path.
//!
//! # String forms
//!
//! A JSON string arrives in one of three forms, and [`FastStr`] models each so that
//! the cheapest comparison strategy can be selected without re-scanning:
//!
//! - [`FastStr::Unescaped`] — content bytes that contain no escape sequences, so the
//!   bytes *are* the logical UTF-8 string. The tokenizer reports these as
//!   [`crate::tokenizer::TokenType::String`].
//! - [`FastStr::Escaped`] — content bytes that still contain JSON `\` escapes
//!   (reported as [`crate::tokenizer::TokenType::EscString`]); decoded on demand.
//! - [`FastStr::Owned`] — an already-decoded owned string, e.g. a constant taken from a
//!   compiled expression, or a materialized intermediate value.
//!
//! # Comparison
//!
//! Comparison is by *logical* (decoded) value, and — because UTF-8 preserves codepoint
//! order under byte-lexicographic comparison — reduces to comparing decoded byte streams:
//!
//! - decoded vs decoded → a word-at-a-time byte comparison, inline (see `cmp_bytes`), with
//!   `memcmp` kept for operands long enough to repay the call;
//! - decoded vs escaped → block-based: [`memchr`] locates each `\`, the literal run
//!   before it is bulk-compared, and only that one escape is decoded (≤4 bytes) and
//!   compared — early-exiting at the first differing byte, never allocating;
//! - escaped vs escaped → a lazy decode iterator on each side, compared streaming.
//!
//! So `"é"` and `"é"` compare equal, and ordering is correct, without either string
//! being fully materialized.

use crate::tokenizer::{Token, TokenType};
use std::borrow::Cow;
use std::cmp::Ordering;

/// The logical type of a value, ordered by N1QL collation precedence
/// (`missing < null < boolean < number < string < array < object`). Used as the
/// cross-type ordering fallback when two values cannot be meaningfully compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueType {
    Missing,
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
}

/// A JSON string value in one of three representations (see the [module docs](self)).
#[derive(Debug, Clone)]
pub enum FastStr<'a> {
    /// Content bytes with no escapes: the bytes are the logical UTF-8 string.
    Unescaped(&'a [u8]),
    /// JSON-escaped content bytes (contain `\` escapes); decoded on demand.
    Escaped(&'a [u8]),
    /// An already-decoded, owned string.
    Owned(String),
}

impl<'a> FastStr<'a> {
    /// Build a [`FastStr`] from a string token's *content* bytes (i.e. with the
    /// surrounding quotes already stripped). `escaped` should be `true` iff the token was
    /// [`crate::tokenizer::TokenType::EscString`].
    #[inline]
    pub fn from_content(content: &'a [u8], escaped: bool) -> Self {
        if escaped {
            FastStr::Escaped(content)
        } else {
            FastStr::Unescaped(content)
        }
    }

    /// Borrow an owned/decoded Rust string as a [`FastStr`].
    #[inline]
    pub fn borrowed_str(s: &'a str) -> Self {
        FastStr::Unescaped(s.as_bytes())
    }

    /// The logical bytes if this form is already decoded (`Unescaped`/`Owned`); `None`
    /// for `Escaped`, which requires decoding.
    #[inline]
    fn decoded_bytes(&self) -> Option<&[u8]> {
        match self {
            FastStr::Unescaped(b) => Some(b),
            FastStr::Owned(s) => Some(s.as_bytes()),
            FastStr::Escaped(_) => None,
        }
    }

    /// Materialize the full decoded bytes. Borrows for already-decoded forms; allocates
    /// once for `Escaped`. Intended for consumers that genuinely need the whole string
    /// (e.g. regex matching or date parsing) — *not* for comparison.
    pub fn to_decoded_bytes(&self) -> Cow<'_, [u8]> {
        match self {
            FastStr::Unescaped(b) => Cow::Borrowed(b),
            FastStr::Owned(s) => Cow::Borrowed(s.as_bytes()),
            FastStr::Escaped(e) => Cow::Owned(DecodeIter::new(e).collect()),
        }
    }

    /// Compare two strings by logical (decoded) value, decoding as little as possible.
    ///
    /// Both operands already decoded is the case the engine is in essentially always — a
    /// document string without escapes against a constant the compiler decoded once — and it
    /// is a byte comparison. The three arms that have to *decode* are what make this function
    /// too big to inline, so they are outlined and this one is a pair of discriminant tests
    /// over an inline byte comparison. Left outlined, a comparison that ends inside eight bytes pays a
    /// call and a return for it, once per value compared.
    #[inline]
    pub fn cmp_str(&self, other: &FastStr<'_>) -> Ordering {
        if let (Some(a), Some(b)) = (self.decoded_bytes(), other.decoded_bytes()) {
            return cmp_bytes(a, b);
        }
        self.cmp_str_escaped(other)
    }

    /// [`Self::cmp_str`] where at least one side still holds JSON escapes, so the comparison
    /// has to decode as it goes. Outlined; see the note there.
    #[inline(never)]
    fn cmp_str_escaped(&self, other: &FastStr<'_>) -> Ordering {
        match (self.decoded_bytes(), other.decoded_bytes()) {
            (Some(a), None) => cmp_plain_vs_escaped(a, as_escaped(other)),
            (None, Some(b)) => cmp_plain_vs_escaped(b, as_escaped(self)).reverse(),
            (None, None) => {
                DecodeIter::new(as_escaped(self)).cmp(DecodeIter::new(as_escaped(other)))
            }
            // `cmp_str` handles this without calling here.
            (Some(_), Some(_)) => unreachable!("both operands are already decoded"),
        }
    }

    /// Whether two strings are logically equal.
    #[inline]
    pub fn eq_str(&self, other: &FastStr<'_>) -> bool {
        self.cmp_str(other) == Ordering::Equal
    }
}

/// Byte-lexicographic comparison of two decoded strings — identical in result to
/// `a.cmp(b)`, but without the call.
///
/// `<[u8] as Ord>::cmp` lowers to glibc `memcmp`, which this crate reaches through the GOT.
/// That is the right trade for long strings and the wrong one here: the values being compared
/// are typically a short field and a shorter constant, so the call, its dispatch prologue and
/// its return dominate the handful of bytes actually examined — and because the result feeds
/// the branch that decides a loop's next step, the latency is exposed rather than overlapped.
/// Measured on a 220-element array of 8-byte strings, `memcmp` was reached once per element
/// for a comparison that a length check alone could settle.
///
/// Words are compared big-endian so that integer order *is* byte-lexicographic order; the
/// tail is finished a byte at a time, and equal prefixes fall through to the length rule.
/// Long operands keep the library call, which vectorizes far better than this loop can.
#[inline]
fn cmp_bytes(a: &[u8], b: &[u8]) -> Ordering {
    let n = a.len().min(b.len());
    if n > 32 {
        return a.cmp(b);
    }
    let mut i = 0;
    while i + 8 <= n {
        // Indexing is bounded by `i + 8 <= n <= len`, so both reads are in range.
        let x = u64::from_be_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_be_bytes(b[i..i + 8].try_into().unwrap());
        if x != y {
            return x.cmp(&y);
        }
        i += 8;
    }
    while i < n {
        if a[i] != b[i] {
            return a[i].cmp(&b[i]);
        }
        i += 1;
    }
    a.len().cmp(&b.len())
}

/// Extract the escaped bytes of an `Escaped` variant. Only called from paths that have
/// already established the variant is `Escaped` (via `decoded_bytes() == None`).
#[inline]
fn as_escaped<'b>(s: &'b FastStr<'_>) -> &'b [u8] {
    match s {
        FastStr::Escaped(e) => e,
        // Unreachable in practice; return empty rather than panic to keep hot paths total.
        _ => &[],
    }
}

/// Compare an already-decoded byte string `plain` against a still-`Escaped` byte string,
/// block by block. Returns the ordering of `plain` relative to the escaped string.
fn cmp_plain_vs_escaped(mut plain: &[u8], mut esc: &[u8]) -> Ordering {
    loop {
        match memchr::memchr(b'\\', esc) {
            None => {
                // No more escapes: the remainder of `esc` is literal.
                return plain.cmp(esc);
            }
            Some(i) => {
                // Compare the literal run that precedes the escape.
                let run = &esc[..i];
                let n = run.len().min(plain.len());
                match plain[..n].cmp(&run[..n]) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                if plain.len() < run.len() {
                    // `plain` ran out inside the run: it is a proper prefix, hence smaller.
                    return Ordering::Less;
                }
                plain = &plain[run.len()..];
                esc = &esc[i..];

                // Decode the single escape and compare its bytes.
                let (buf, out_len, consumed) = decode_escape_at(esc);
                let dec = &buf[..out_len];
                let m = out_len.min(plain.len());
                match plain[..m].cmp(&dec[..m]) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
                if plain.len() < out_len {
                    return Ordering::Less;
                }
                plain = &plain[out_len..];
                esc = &esc[consumed..];
            }
        }
    }
}

/// Hex digit value, or `None`.
#[inline]
fn hex_val(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

/// Read the 4 hex digits of a `\uXXXX` escape starting at `s[at]`.
#[inline]
fn read_u4(s: &[u8], at: usize) -> Option<u32> {
    if s.len() < at + 4 {
        return None;
    }
    let (a, b, c, d) = (
        hex_val(s[at])?,
        hex_val(s[at + 1])?,
        hex_val(s[at + 2])?,
        hex_val(s[at + 3])?,
    );
    Some((a << 12) | (b << 8) | (c << 4) | d)
}

/// Encode a scalar codepoint (or the replacement char for invalid ones) to UTF-8.
#[inline]
fn encode_cp(cp: u32, consumed: usize) -> ([u8; 4], usize, usize) {
    let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
    let mut buf = [0u8; 4];
    let out_len = ch.encode_utf8(&mut buf).len();
    (buf, out_len, consumed)
}

/// Decode the single escape sequence at the start of `s` (where `s[0] == b'\\'`).
///
/// Returns `(utf8_buf, out_len, input_consumed)`. Never panics: malformed or lone
/// surrogates decode to `U+FFFD`. Well-formedness of the basic 2-char and `\uXXXX`
/// shapes is guaranteed by the tokenizer; this stays defensive regardless.
fn decode_escape_at(s: &[u8]) -> ([u8; 4], usize, usize) {
    #[inline]
    fn one(b: u8, consumed: usize) -> ([u8; 4], usize, usize) {
        ([b, 0, 0, 0], 1, consumed)
    }

    if s.len() < 2 {
        return encode_cp(0xFFFD, s.len());
    }
    match s[1] {
        b'"' => one(b'"', 2),
        b'\\' => one(b'\\', 2),
        b'/' => one(b'/', 2),
        b'b' => one(0x08, 2),
        b'f' => one(0x0c, 2),
        b'n' => one(b'\n', 2),
        b'r' => one(b'\r', 2),
        b't' => one(b'\t', 2),
        b'u' => {
            let hi = read_u4(s, 2).unwrap_or(0xFFFD);
            if (0xD800..=0xDBFF).contains(&hi) {
                // High surrogate: needs a following low surrogate to form a scalar.
                if s.len() >= 12 && s[6] == b'\\' && s[7] == b'u' {
                    if let Some(lo) = read_u4(s, 8) {
                        if (0xDC00..=0xDFFF).contains(&lo) {
                            let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                            return encode_cp(cp, 12);
                        }
                    }
                }
                encode_cp(0xFFFD, 6) // lone/invalid high surrogate
            } else if (0xDC00..=0xDFFF).contains(&hi) {
                encode_cp(0xFFFD, 6) // lone low surrogate
            } else {
                encode_cp(hi, 6)
            }
        }
        // Tokenizer guarantees this is unreachable; be lenient rather than panic.
        other => one(other, 2),
    }
}

/// A lazy iterator over the decoded bytes of a (possibly escaped) JSON string body.
struct DecodeIter<'a> {
    s: &'a [u8],
    pos: usize,
    buf: [u8; 4],
    buf_len: u8,
    buf_pos: u8,
}

impl<'a> DecodeIter<'a> {
    #[inline]
    fn new(s: &'a [u8]) -> Self {
        DecodeIter {
            s,
            pos: 0,
            buf: [0; 4],
            buf_len: 0,
            buf_pos: 0,
        }
    }
}

impl Iterator for DecodeIter<'_> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.buf_pos < self.buf_len {
            let b = self.buf[self.buf_pos as usize];
            self.buf_pos += 1;
            return Some(b);
        }
        if self.pos >= self.s.len() {
            return None;
        }
        let c = self.s[self.pos];
        if c == b'\\' {
            let (buf, out_len, consumed) = decode_escape_at(&self.s[self.pos..]);
            self.pos += consumed;
            self.buf = buf;
            self.buf_len = out_len as u8;
            self.buf_pos = 1;
            Some(buf[0])
        } else {
            self.pos += 1;
            Some(c)
        }
    }
}

/// A runtime JSON value, borrowing from the document being scanned where possible.
///
/// Numeric values come in matched *parsed* and *lazy* forms so that parsing cost is only
/// paid when a comparison actually needs the value:
/// - parsed: [`FastVal::Int`] / [`FastVal::Uint`] / [`FastVal::Float`];
/// - lazy raw document bytes: [`FastVal::IntBytes`] (an integer literal, from the
///   tokenizer's `Integer`) and [`FastVal::FloatBytes`] (a fractional/exponent literal,
///   from `Number`). These parse on demand.
///
/// Because matching visits each field value once and reuses that single `FastVal` for all
/// of that field's ops, a lazy value is parsed at most once in practice — no caching is
/// needed at this layer.
///
/// Strings use [`FastStr`].
#[derive(Debug, Clone)]
pub enum FastVal<'a> {
    /// An absent value (a field that did not exist).
    Missing,
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    /// An integer literal kept as its raw document bytes (tokenizer `Integer`); parsed on
    /// demand into [`FastVal::Int`] or, on positive overflow, [`FastVal::Uint`].
    IntBytes(&'a [u8]),
    /// A fractional/exponent numeric literal kept as its raw document bytes (tokenizer
    /// `Number`); parsed on demand into [`FastVal::Float`].
    FloatBytes(&'a [u8]),
    Str(FastStr<'a>),
    /// An array kept as its raw document bytes (`[` … `]`).
    Array(&'a [u8]),
    /// An object kept as its raw document bytes (`{` … `}`).
    Object(&'a [u8]),
}

/// A numeric value normalized to one of the three concrete numeric kinds, produced by
/// parsing lazy byte forms. Used internally for numeric comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    I(i64),
    U(u64),
    F(f64),
}

impl Num {
    /// This number as an `f64` (lossy for very large magnitudes).
    #[inline]
    pub fn as_f64(self) -> f64 {
        match self {
            Num::I(i) => i as f64,
            Num::U(u) => u as f64,
            Num::F(f) => f,
        }
    }
}

impl<'a> FastVal<'a> {
    /// The value's logical type, used for cross-type collation ordering.
    pub fn value_type(&self) -> ValueType {
        match self {
            FastVal::Missing => ValueType::Missing,
            FastVal::Null => ValueType::Null,
            FastVal::Bool(_) => ValueType::Boolean,
            FastVal::Int(_)
            | FastVal::Uint(_)
            | FastVal::Float(_)
            | FastVal::IntBytes(_)
            | FastVal::FloatBytes(_) => ValueType::Number,
            FastVal::Str(_) => ValueType::String,
            FastVal::Array(_) => ValueType::Array,
            FastVal::Object(_) => ValueType::Object,
        }
    }

    /// Whether this value is numeric.
    #[inline]
    pub fn is_numeric(&self) -> bool {
        self.value_type() == ValueType::Number
    }

    /// Borrow the string representation, if this value is a string.
    pub fn as_str(&self) -> Option<&FastStr<'a>> {
        match self {
            FastVal::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The boolean value, if this is a boolean.
    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            FastVal::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The raw JSON bytes if this is an array or object container.
    #[inline]
    pub fn container_bytes(&self) -> Option<&'a [u8]> {
        match self {
            FastVal::Array(b) | FastVal::Object(b) => Some(b),
            _ => None,
        }
    }

    /// Build a scalar value from a tokenizer [`Token`], choosing the representation
    /// directly from the token classification: `Integer`/`Number` become the lazy
    /// [`FastVal::IntBytes`]/[`FastVal::FloatBytes`], and `String`/`EscString` become the
    /// unescaped/escaped [`FastStr`] forms (with surrounding quotes stripped). Returns
    /// `None` for non-scalar tokens (structural delimiters, `End`).
    ///
    /// `#[inline]` because this sits between the tokenizer and every comparison: it runs once
    /// per scalar the matcher looks at. Left to the default, a consumer crate (and, at the
    /// default `codegen-units = 16`, another codegen unit of this one) calls it out of line —
    /// which on x86-64 PIC means an indirect call through the GOT, plus returning a
    /// `FastVal` through memory — for a jump table that inlines to a few instructions.
    #[inline]
    pub fn from_scalar_token(token: Token<'a>) -> Option<Self> {
        Some(match token.token_type {
            TokenType::String => FastVal::Str(FastStr::Unescaped(strip_quotes(token.value))),
            TokenType::EscString => FastVal::Str(FastStr::Escaped(strip_quotes(token.value))),
            TokenType::Integer => FastVal::IntBytes(token.value),
            TokenType::Number => FastVal::FloatBytes(token.value),
            TokenType::True => FastVal::Bool(true),
            TokenType::False => FastVal::Bool(false),
            TokenType::Null => FastVal::Null,
            _ => return None,
        })
    }

    /// Normalize a numeric value to a concrete [`Num`], parsing lazy byte forms.
    ///
    /// Integer bytes that overflow `i64` positively parse as [`Num::U`]; integers outside
    /// both `i64` and `u64` (or otherwise unparseable) fall back to `f64` (documented,
    /// lossy). Returns `None` for non-numeric values.
    ///
    /// `#[inline]` for the reason [`Self::from_scalar_token`] gives, and one more: `Num` is 16
    /// bytes and `Option<Num>` is 24, which is over the two-register return limit — so
    /// outlined, every numeric comparison writes *two* of these to stack slots and reads them
    /// straight back to compare them. The float arm stays out of line so this is small enough
    /// for the attribute to take.
    #[inline]
    pub fn as_num(&self) -> Option<Num> {
        match self {
            FastVal::Int(v) => Some(Num::I(*v)),
            FastVal::Uint(v) => Some(Num::U(*v)),
            FastVal::Float(v) => Some(Num::F(*v)),
            FastVal::IntBytes(b) => Some(parse_int_bytes(b)),
            FastVal::FloatBytes(b) => Some(Num::F(parse_float_bytes(b))),
            _ => None,
        }
    }

    /// Compare two numeric values exactly (no epsilon). Returns `None` if either value is
    /// not numeric.
    ///
    /// `#[inline]` for the same reason as [`Self::as_num`]: outlined, the two 24-byte
    /// `Option<Num>`s it consumes are written to stack slots and read straight back, and the
    /// collation reaches it once per numeric comparison.
    #[inline]
    pub fn cmp_num(&self, other: &FastVal<'_>) -> Option<Ordering> {
        Some(cmp_num(self.as_num()?, other.as_num()?))
    }
}

/// Strip the surrounding quotes from a string token's raw bytes.
#[inline]
fn strip_quotes(b: &[u8]) -> &[u8] {
    if b.len() >= 2 {
        &b[1..b.len() - 1]
    } else {
        b
    }
}

/// Parse integer bytes, preferring `i64`, then positive `u64`, then lossy `f64`.
fn parse_int_bytes(b: &[u8]) -> Num {
    // The bytes come from the tokenizer's `Integer` classification — ASCII digits with an
    // optional leading `-` — so validating them as UTF-8 is a scan of known-ASCII bytes, and
    // `str::parse` re-derives a sign and a digit loop this already knows the shape of. Both
    // show up in a profile, `from_utf8` with a symbol of its own. This is
    // [`cmp_bytes`]' trade — hand-roll what the library would do generically — applied to
    // numbers.
    //
    // Accumulating in `u64` covers `i64` and `u64` in one pass, which is what the old
    // `parse::<i64>()`-then-`parse::<u64>()` ladder needed two for. Anything this cannot
    // represent — an overflow, or a byte that is not a digit — falls back to the library
    // through `parse_float_bytes`, exactly as before, so the documented lossy `f64` behaviour
    // for out-of-range integers is unchanged.
    let (neg, digits) = match b.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, b),
    };
    if digits.is_empty() {
        return Num::F(parse_float_bytes(b));
    }
    let Some(acc) = digits_to_u64(digits) else {
        return Num::F(parse_float_bytes(b));
    };
    if neg {
        // `i64::MIN` is `-(2^63)`, whose magnitude is one past `i64::MAX`. Negating it as an
        // `i64` is the one case where `wrapping_neg` is the right answer rather than a bug:
        // `2^63 as i64` is already `i64::MIN`, and negating that yields itself.
        if acc <= 1 << 63 {
            Num::I((acc as i64).wrapping_neg())
        } else {
            Num::F(parse_float_bytes(b))
        }
    } else if acc <= i64::MAX as u64 {
        Num::I(acc as i64)
    } else {
        Num::U(acc)
    }
}

/// A run of ASCII digits as a `u64`, or `None` if it is not one or does not fit.
///
/// `u64::MAX` has twenty digits, so **nineteen or fewer cannot overflow** — which means the
/// accumulation needs no per-digit overflow check. That check is not just two instructions: it
/// is a branch sitting on the dependency chain that carries each digit into the next, and that
/// chain is what the loop is bound by. Measured across the two integer widths, a digit costs
/// **6.4 cycles**, against an `imul` latency of 3.
///
/// Twenty digits — a `u64` at its very limit — go to the library, which gets the boundary
/// exactly right and is not worth hand-rolling for a case no real document contains.
#[inline]
fn digits_to_u64(digits: &[u8]) -> Option<u64> {
    if digits.len() > 19 {
        return std::str::from_utf8(digits).ok()?.parse::<u64>().ok();
    }
    let mut acc: u64 = 0;
    for &c in digits {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        acc = acc * 10 + d as u64;
    }
    Some(acc)
}

/// Parse fractional/exponent numeric bytes as `f64`.
///
/// Outlined: `str::parse::<f64>` is a large function (the correctly-rounded decimal-to-binary
/// path), and keeping it behind a call is what lets [`FastVal::as_num`] inline. It is also the
/// fallback [`parse_int_bytes`] uses for integers it cannot represent, so it must not be on
/// that function's hot path either.
#[inline(never)]
fn parse_float_bytes(b: &[u8]) -> f64 {
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::NAN)
}

/// Exact comparison of two normalized numbers across `i64`/`u64`/`f64`.
///
/// `#[inline]` so the whole numeric comparison — parse, normalize, compare — collapses into
/// its caller. Left out of line, this and `FastVal::cmp_num` are two calls and two 16-byte
/// values passed through memory to reach a nine-arm match.
#[inline]
fn cmp_num(a: Num, b: Num) -> Ordering {
    match (a, b) {
        (Num::I(x), Num::I(y)) => x.cmp(&y),
        (Num::U(x), Num::U(y)) => x.cmp(&y),
        (Num::I(x), Num::U(y)) => cmp_i64_u64(x, y),
        (Num::U(x), Num::I(y)) => cmp_i64_u64(y, x).reverse(),
        (Num::F(x), Num::F(y)) => cmp_f64(x, y),
        (Num::I(x), Num::F(y)) => cmp_i64_f64(x, y),
        (Num::F(x), Num::I(y)) => cmp_i64_f64(y, x).reverse(),
        (Num::U(x), Num::F(y)) => cmp_u64_f64(x, y),
        (Num::F(x), Num::U(y)) => cmp_u64_f64(y, x).reverse(),
    }
}

#[inline]
fn cmp_i64_u64(a: i64, b: u64) -> Ordering {
    if a < 0 {
        Ordering::Less
    } else {
        (a as u64).cmp(&b)
    }
}

/// Compare finite/`NaN` floats. JSON numbers are always finite; a stray `NaN` (only
/// reachable via an expression constant) is ordered greatest so the result stays total.
#[inline]
fn cmp_f64(x: f64, y: f64) -> Ordering {
    match x.partial_cmp(&y) {
        Some(o) => o,
        None => match (x.is_nan(), y.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => Ordering::Equal, // unreachable
        },
    }
}

/// Exact comparison of an `i64` against a finite `f64` (no precision loss for large
/// magnitudes). `NaN` is ordered greatest.
fn cmp_i64_f64(a: i64, b: f64) -> Ordering {
    if b.is_nan() {
        return Ordering::Less;
    }
    // 2^63 is exactly representable in f64; i64::MAX (2^63 - 1) is not, so compare against
    // the powers of two that bound the i64 range.
    if b >= 9_223_372_036_854_775_808.0 {
        return Ordering::Less; // a <= i64::MAX < 2^63 <= b
    }
    if b < -9_223_372_036_854_775_808.0 {
        return Ordering::Greater; // b < i64::MIN <= a
    }
    // b is within (-2^63, 2^63): its truncation fits i64.
    let bt = b.trunc() as i64;
    match a.cmp(&bt) {
        Ordering::Equal => {
            // Equal integer parts: the fractional part breaks the tie.
            if b > b.trunc() {
                Ordering::Less
            } else if b < b.trunc() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        ord => ord,
    }
}

/// Exact comparison of a `u64` against a finite `f64`. `NaN` is ordered greatest.
fn cmp_u64_f64(a: u64, b: f64) -> Ordering {
    if b.is_nan() {
        return Ordering::Less;
    }
    if b < 0.0 {
        return Ordering::Greater; // a >= 0 > b
    }
    if b >= 18_446_744_073_709_551_616.0 {
        return Ordering::Less; // a <= u64::MAX < 2^64 <= b
    }
    let bt = b.trunc() as u64;
    match a.cmp(&bt) {
        Ordering::Equal => {
            if b > b.trunc() {
                Ordering::Less
            } else if b < b.trunc() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        ord => ord,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: fully decode a (possibly escaped) body, then compare. The optimized
    /// paths in `cmp_str` must always agree with this.
    fn oracle_cmp(a: &[u8], b: &[u8]) -> Ordering {
        let da: Vec<u8> = DecodeIter::new(a).collect();
        let db: Vec<u8> = DecodeIter::new(b).collect();
        da.cmp(&db)
    }

    fn unescaped(s: &str) -> FastStr<'_> {
        FastStr::Unescaped(s.as_bytes())
    }
    fn escaped(s: &str) -> FastStr<'_> {
        FastStr::Escaped(s.as_bytes())
    }

    fn num_cmp(a: FastVal<'_>, b: FastVal<'_>) -> Ordering {
        a.cmp_num(&b).unwrap()
    }

    /// `cmp_bytes` replaced `<[u8] as Ord>::cmp` on the decoded-vs-decoded path, so the two
    /// must agree on every input — that slice comparison *is* the specification here.
    ///
    /// The cases pin each structural feature separately, because a bug in any one of them
    /// survives a test that only checks short unequal strings: the 8-byte word loop, the
    /// byte-at-a-time tail after it, the equal-prefix length rule that ends the function, and
    /// the `memcmp` fallback for long operands — with lengths on both sides of the threshold
    /// that selects it, so neither the fast path nor the fallback goes unexecuted.
    ///
    /// **This test is the only guard on comparison, and cannot be replaced by the differential
    /// sweep.** `jsonsm-slow`'s oracle reuses this crate's `Collation` rather than duplicating
    /// it (deliberately — see that crate's module docs), so a bug in `cmp_bytes` corrupts the
    /// oracle and the engine identically and the two still agree. Measured: with the word
    /// order reversed, the word stride skipping a byte, or the length rule dropped, the
    /// differential sweep passed at ten times its normal length in all three cases while these
    /// assertions failed immediately. That sweep proves the two *traversals* agree; comparison
    /// is common code beneath both, and only an independent oracle — here, `slice::cmp` —
    /// can check it.
    #[test]
    fn cmp_bytes_agrees_with_slice_cmp() {
        let mut cases: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

        for len in [1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 40, 64] {
            // Bytes stay in 'a'..='w', so b'z' is always greater and b'A' always less —
            // a difference planted at `at` has a known direction.
            let base: Vec<u8> = (0..len).map(|i| b'a' + (i % 23) as u8).collect();
            cases.push((base.clone(), base.clone()));
            for at in 0..len {
                let mut greater = base.clone();
                greater[at] = b'z';
                let mut less = base.clone();
                less[at] = b'A';
                cases.push((base.clone(), greater));
                cases.push((base.clone(), less));
            }
            // Equal common prefix, decided only by length.
            for cut in 0..len {
                cases.push((base[..cut].to_vec(), base.clone()));
            }
        }
        cases.push((Vec::new(), Vec::new()));
        cases.push((Vec::new(), b"a".to_vec()));

        // Two differences inside one word, pointing opposite ways. A single planted
        // difference cannot distinguish word byte order — whichever byte differs decides the
        // comparison under either endianness — so without these the word loop could read
        // little-endian and every case above would still pass. Here the earlier byte must
        // win, which is true only if the word compares big-endian.
        cases.push((b"az______".to_vec(), b"za______".to_vec()));
        cases.push((b"aXXXXXXz".to_vec(), b"zXXXXXXa".to_vec()));
        cases.push((b"abcdefgh_z".to_vec(), b"abcdefgh_a".to_vec()));
        // …and straddling the word boundary, where the second word must not overrule the first.
        cases.push((b"aaaaaaazaaaaaaaa".to_vec(), b"aaaaaaaaaaaaaaaz".to_vec()));

        // A deterministic sweep over a two-letter alphabet: short strings over {a,b} collide
        // on long prefixes and differ in several positions at once, which is the shape that
        // exercises word order, the tail, and the length rule together rather than one at a
        // time. Seeded so a failure is reproducible.
        let mut seed = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..4000 {
            let la = (next() % 40) as usize;
            let lb = (next() % 40) as usize;
            let a: Vec<u8> = (0..la).map(|_| b'a' + (next() & 1) as u8).collect();
            let b: Vec<u8> = (0..lb).map(|_| b'a' + (next() & 1) as u8).collect();
            cases.push((a, b));
        }

        for (a, b) in cases {
            let (x, y) = (a.as_slice(), b.as_slice());
            assert_eq!(cmp_bytes(x, y), x.cmp(y), "cmp_bytes({x:?}, {y:?})");
            // Both argument orders: the length rule and the word loop are asymmetric.
            assert_eq!(cmp_bytes(y, x), y.cmp(x), "cmp_bytes({y:?}, {x:?})");
        }
    }

    /// The same agreement, reached the way the matcher reaches it: through `cmp_str` on two
    /// already-decoded strings. Guards the wiring, not the algorithm.
    #[test]
    fn cmp_str_matches_slice_cmp_on_decoded_forms() {
        // Long enough to cross the `memcmp` threshold, and differing only in the tail.
        let long_a = "x".repeat(40) + "a";
        let long_b = "x".repeat(40) + "b";
        for (a, b) in [
            ("tag00000", "c"),
            ("c", "tag00000"),
            ("", "a"),
            ("abc", "abc"),
            ("abcdefgh", "abcdefgi"),
            (long_a.as_str(), long_b.as_str()),
        ] {
            assert_eq!(
                unescaped(a).cmp_str(&unescaped(b)),
                a.as_bytes().cmp(b.as_bytes()),
                "cmp_str({a:?}, {b:?})"
            );
        }
    }

    #[test]
    fn numeric_comparison_is_exact_across_kinds() {
        use FastVal::*;
        // Same kinds.
        assert_eq!(num_cmp(Int(1), Int(2)), Ordering::Less);
        assert_eq!(num_cmp(Uint(5), Uint(5)), Ordering::Equal);
        assert_eq!(num_cmp(Float(2.5), Float(2.4)), Ordering::Greater);
        // int vs float: exact, no epsilon (1.0 == 1, but 1.0000001 != 1).
        assert_eq!(num_cmp(Int(1), Float(1.0)), Ordering::Equal);
        assert_eq!(num_cmp(Int(1), Float(1.000_000_1)), Ordering::Less);
        assert_eq!(num_cmp(Float(1.000_000_1), Int(1)), Ordering::Greater);
        // int vs uint across the sign boundary.
        assert_eq!(num_cmp(Int(-1), Uint(0)), Ordering::Less);
        assert_eq!(num_cmp(Uint(u64::MAX), Int(i64::MAX)), Ordering::Greater);
    }

    #[test]
    fn numeric_comparison_is_exact_at_large_magnitudes() {
        use FastVal::*;
        // i64::MAX vs the f64 just above it: no precision collapse.
        assert_eq!(
            num_cmp(Int(i64::MAX), Float(9_223_372_036_854_775_808.0)),
            Ordering::Less
        );
        // u64::MAX vs 2^64 as f64.
        assert_eq!(
            num_cmp(Uint(u64::MAX), Float(18_446_744_073_709_551_616.0)),
            Ordering::Less
        );
        // Very negative float vs i64::MIN.
        assert_eq!(num_cmp(Int(i64::MIN), Float(-1e300)), Ordering::Greater);
        // Negative float vs any uint.
        assert_eq!(num_cmp(Uint(0), Float(-0.5)), Ordering::Greater);
    }

    #[test]
    fn lazy_number_bytes_parse_on_demand() {
        // Integer that fits i64.
        assert_eq!(FastVal::IntBytes(b"42").as_num(), Some(Num::I(42)));
        // Integer above i64::MAX parses as u64.
        assert_eq!(
            FastVal::IntBytes(b"18446744073709551615").as_num(),
            Some(Num::U(u64::MAX))
        );
        // Negative integer.
        assert_eq!(FastVal::IntBytes(b"-7").as_num(), Some(Num::I(-7)));
        // Float bytes.
        assert_eq!(FastVal::FloatBytes(b"3.5").as_num(), Some(Num::F(3.5)));
        // Lazy vs parsed compare equal.
        assert_eq!(
            FastVal::IntBytes(b"100").cmp_num(&FastVal::Int(100)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            FastVal::FloatBytes(b"2.0").cmp_num(&FastVal::Int(2)),
            Some(Ordering::Equal)
        );
    }

    /// `parse_int_bytes` hand-rolls what `str::parse` does generically, so it is checked
    /// against `str::parse` — an independent reference — rather than against the engine.
    ///
    /// `jsonsm-slow` reaches numeric comparison through this same function, so the
    /// differential sweep cannot see a bug in it: both sides would agree on the same wrong
    /// answer. That is the standing rule for anything the oracle imports.
    #[test]
    fn integer_bytes_parse_as_the_library_would() {
        // Every boundary the u64 accumulator and the i64/u64/f64 ladder can straddle.
        let cases: &[&str] = &[
            "0",
            "-0",
            "7",
            "42",
            "-7",
            "9223372036854775806",
            "9223372036854775807",           // i64::MAX
            "9223372036854775808",           // one past i64::MAX -> u64
            "-9223372036854775807",
            "-9223372036854775808",          // i64::MIN, whose magnitude is past i64::MAX
            "-9223372036854775809",          // one past i64::MIN -> f64
            "18446744073709551615",          // u64::MAX
            "18446744073709551616",          // one past u64::MAX -> f64
            "007",                           // leading zeros: not valid JSON, still defined
            "99999999999999999999999999999", // far past u64
        ];
        for c in cases {
            let got = parse_int_bytes(c.as_bytes());
            // The reference: exactly the ladder the old implementation used.
            let want = if let Ok(i) = c.parse::<i64>() {
                Num::I(i)
            } else if let Ok(u) = c.parse::<u64>() {
                Num::U(u)
            } else {
                Num::F(c.parse::<f64>().unwrap_or(f64::NAN))
            };
            assert_eq!(got, want, "parse_int_bytes({c:?})");
        }
        // Shapes the tokenizer cannot produce, checked only for "does not panic, still
        // numeric" — the fallback hands them to the library, which decides.
        for c in ["", "-", "+5", "1_2"] {
            let got = parse_int_bytes(c.as_bytes());
            let want_nan = c.parse::<f64>().is_err();
            assert_eq!(
                matches!(got, Num::F(f) if f.is_nan()),
                want_nan,
                "parse_int_bytes({c:?}) NaN-ness"
            );
        }
    }

    #[test]
    fn from_scalar_token_selects_representation() {
        use crate::tokenizer::{JsonTokenizer, Tokenizer};

        // A tiny helper: tokenize one value and build a FastVal from it.
        fn val(input: &str) -> FastVal<'_> {
            let mut t = JsonTokenizer::new(input.as_bytes());
            FastVal::from_scalar_token(t.step().unwrap()).unwrap()
        }

        assert!(matches!(val("42"), FastVal::IntBytes(b"42")));
        assert!(matches!(val("3.14"), FastVal::FloatBytes(b"3.14")));
        assert!(matches!(val("true"), FastVal::Bool(true)));
        assert!(matches!(val("null"), FastVal::Null));
        // Strings: quotes stripped, escaped vs unescaped chosen from the token type.
        match val(r#""plain""#) {
            FastVal::Str(FastStr::Unescaped(b)) => assert_eq!(b, b"plain"),
            other => panic!("expected unescaped string, got {other:?}"),
        }
        match val(r#""a\nb""#) {
            FastVal::Str(FastStr::Escaped(b)) => assert_eq!(b, br"a\nb"),
            other => panic!("expected escaped string, got {other:?}"),
        }
        // Numeric equality end-to-end through the tokenizer bridge.
        assert_eq!(val("42").cmp_num(&val("42")), Some(Ordering::Equal));
    }

    #[test]
    fn escaped_and_unescaped_forms_are_equal() {
        // é == é, tab == \t, quote == \"
        assert!(unescaped("é").eq_str(&escaped("\\u00e9")));
        assert!(escaped("\\u00e9").eq_str(&unescaped("é")));
        assert!(unescaped("a\tb").eq_str(&escaped("a\\tb")));
        assert!(unescaped("say \"hi\"").eq_str(&escaped("say \\\"hi\\\"")));
        assert!(escaped("\\ud83d\\ude00").eq_str(&unescaped("😀"))); // surrogate pair
    }

    #[test]
    fn owned_form_compares_by_value() {
        let owned = FastStr::Owned("café".to_string());
        assert!(owned.eq_str(&escaped("caf\\u00e9")));
        assert!(owned.eq_str(&unescaped("café")));
        assert_eq!(owned.cmp_str(&unescaped("cafd")), Ordering::Greater);
    }

    #[test]
    fn ordering_is_correct_across_forms() {
        assert_eq!(
            unescaped("apple").cmp_str(&unescaped("banana")),
            Ordering::Less
        );
        assert_eq!(
            escaped("a\\u0070ple").cmp_str(&unescaped("apple")),
            Ordering::Equal
        ); // p == 'p'
        assert_eq!(
            unescaped("apple").cmp_str(&escaped("a\\u0071ple")),
            Ordering::Less
        ); // 'p' < 'q'
           // Prefix is less than the longer string, in every form pairing.
        assert_eq!(unescaped("ab").cmp_str(&unescaped("abc")), Ordering::Less);
        assert_eq!(
            unescaped("ab").cmp_str(&escaped("a\\u0062c")),
            Ordering::Less
        );
        assert_eq!(
            escaped("a\\u0062").cmp_str(&unescaped("abc")),
            Ordering::Less
        );
    }

    #[test]
    fn to_decoded_bytes_matches_forms() {
        assert_eq!(&*unescaped("abc").to_decoded_bytes(), b"abc");
        assert_eq!(&*escaped("a\\nb").to_decoded_bytes(), b"a\nb");
        assert_eq!(&*escaped("\\u00e9").to_decoded_bytes(), "é".as_bytes());
    }

    #[test]
    fn value_type_ordering_follows_n1ql() {
        use ValueType::*;
        assert!(Missing < Null);
        assert!(Null < Boolean);
        assert!(Boolean < Number);
        assert!(Number < String);
        assert!(String < Array);
        assert!(Array < Object);
    }

    /// Deterministic pseudo-random exercise of all three comparison paths against the
    /// full-decode oracle, including escapes at varied positions.
    #[test]
    fn property_optimized_paths_agree_with_oracle() {
        // A small alphabet plus a set of escape sequences to splice in.
        let escapes: &[&str] = &[
            "\\n",
            "\\t",
            "\\\"",
            "\\\\",
            "\\u0062",
            "\\u00e9",
            "\\ud83d\\ude00",
        ];
        let plains: &[&str] = &["a", "b", "c", "é", "😀", "\t", "\"", "\\"];

        // Build a batch of (raw-body-bytes, is-escaped) fragments.
        let mut samples: Vec<String> = Vec::new();
        // Plain fragments (their raw body encodes specials as escapes so both sides decode equal).
        let mut lcg: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lcg >> 33) as usize
        };
        for _ in 0..400 {
            let parts = 1 + next() % 4;
            let mut s = String::new();
            for _ in 0..parts {
                if next() % 2 == 0 {
                    s.push_str(plains[next() % plains.len()]);
                } else {
                    s.push_str(escapes[next() % escapes.len()]);
                }
            }
            samples.push(s);
        }

        // Compare many pairs across all form combinations.
        for i in 0..samples.len() {
            for j in [i, (i + 1) % samples.len(), (i * 7 + 3) % samples.len()] {
                let ai = samples[i].as_bytes();
                let bj = samples[j].as_bytes();
                let want = oracle_cmp(ai, bj);

                // escaped vs escaped
                assert_eq!(
                    escaped(&samples[i]).cmp_str(&escaped(&samples[j])),
                    want,
                    "esc/esc {i},{j}"
                );

                // decoded(plain) vs escaped, and the reverse — build a plain form by fully
                // decoding one side into an owned String.
                let ai_dec = String::from_utf8(DecodeIter::new(ai).collect()).unwrap();
                let bj_dec = String::from_utf8(DecodeIter::new(bj).collect()).unwrap();
                assert_eq!(
                    FastStr::Owned(ai_dec.clone()).cmp_str(&escaped(&samples[j])),
                    want,
                    "plain/esc {i},{j}"
                );
                assert_eq!(
                    escaped(&samples[i]).cmp_str(&FastStr::Owned(bj_dec.clone())),
                    want,
                    "esc/plain {i},{j}"
                );

                // decoded vs decoded
                assert_eq!(
                    FastStr::Owned(ai_dec).cmp_str(&FastStr::Owned(bj_dec)),
                    want,
                    "plain/plain {i},{j}"
                );
            }
        }
    }
}
