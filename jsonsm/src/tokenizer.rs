//! A fast, zero-copy, single-token-at-a-time JSON scanner.
//!
//! This is a *scanner*, not a parser: each [`Tokenizer::step`] call returns the next
//! structural or value token (including delimiters like `,` and `:`), and the caller is
//! responsible for enforcing document grammar. Tokens borrow directly from the input
//! byte slice — no allocation, no copying.
//!
//! Two enhancements over a naive scanner (carried over from `gojsonsm`, used by later
//! stages to avoid re-scanning):
//! - strings are reported as [`TokenType::String`] (no escapes, usable verbatim) vs.
//!   [`TokenType::EscString`] (contains escapes, must be decoded);
//! - numbers are reported as [`TokenType::Integer`] (no fraction/exponent) vs.
//!   [`TokenType::Number`].
//!
//! The scanner is strict about JSON: literals must be exactly `true`/`false`/`null`
//! (no case-insensitive variants), and it returns [`TokenizerError`] rather than
//! panicking on malformed input.
//!
//! # Where SIMD plugs in
//!
//! There is exactly **one** state machine ([`GenericTokenizer`]), parameterised by a
//! [`Scan`] strategy that supplies four *bulk* primitives: cross a run of ordinary
//! string bytes, a run of whitespace, a run of digits, and a run of bytes that cannot
//! change container nesting. Each returns precisely the index the byte-at-a-time FSM would
//! have stopped at, so the scalar and SIMD tokenizers cannot drift in grammar, error kind,
//! or error offset — the only difference is how fast an uninteresting run is crossed.
//!
//! [`Scan`] additionally supplies `skip_container`, which is not a run-crossing primitive
//! but a whole loop: it advances past a value the caller has already decided is irrelevant.
//! The portable version walks bytes; vector backends replace it entirely with a
//! block-at-a-time counter. That one is *not* self-healing (see below) — it is checked
//! against the portable walk by exhaustive differential test instead.
//!
//! - [`JsonTokenizer`] = `GenericTokenizer<ScalarScan>` — always available, no `unsafe`.
//! - `simd::SimdTokenizer` = `GenericTokenizer<SimdScan>` — the
//!   `simd` feature; picks a CPU backend once per construction.
//! - [`DocTokenizer`] — the alias [`FastMatcher`](crate::matcher::FastMatcher) actually
//!   uses, selected by feature.
//!
//! Every bulk primitive is also *self-healing*: the FSM re-examines the byte the scanner
//! stopped on, so a scanner that stops too early is merely slow, never wrong. (Stopping
//! too late would be wrong, which is what the parity tests exist to catch.)

/// The kind of a scanned token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    ObjectStart,
    ObjectEnd,
    ObjectKeyDelim,
    ArrayStart,
    ArrayEnd,
    ListDelim,
    /// A string literal containing no escape sequences (`value` includes the quotes).
    String,
    /// A string literal containing at least one escape sequence (`value` includes the quotes).
    EscString,
    /// A number with no fraction or exponent.
    Integer,
    /// A number with a fraction and/or exponent.
    Number,
    Null,
    True,
    False,
    /// End of input.
    End,
}

impl TokenType {
    /// Whether this token is a scalar value literal (string/number/null/bool).
    #[inline]
    pub fn is_literal(self) -> bool {
        matches!(
            self,
            TokenType::String
                | TokenType::EscString
                | TokenType::Integer
                | TokenType::Number
                | TokenType::Null
                | TokenType::True
                | TokenType::False
        )
    }
}

/// A scanned token, borrowing its raw bytes from the input.
///
/// `value` is the verbatim slice of the source the token spans: for strings this
/// includes the surrounding quotes; for structural tokens it is the single delimiter
/// byte; for [`TokenType::End`] it is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub token_type: TokenType,
    pub value: &'a [u8],
}

/// The specific cause of a tokenizer failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TokenizerErrorKind {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unexpected byte {0:#04x} at start of value")]
    UnexpectedByte(u8),
    #[error("control character in string literal")]
    ControlCharInString,
    #[error("invalid string escape sequence")]
    InvalidEscape,
    #[error("invalid \\u unicode escape")]
    InvalidUnicodeEscape,
    #[error("invalid number literal")]
    InvalidNumber,
    #[error("invalid literal (expected `{expected}`)")]
    InvalidLiteral { expected: &'static str },
}

/// A tokenizer error, annotated with the byte offset at which it occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("JSON tokenizer error at byte {pos}: {kind}")]
pub struct TokenizerError {
    pub kind: TokenizerErrorKind,
    pub pos: usize,
}

/// The seam behind which alternative scanners (e.g. a SIMD implementation) can be
/// substituted. `'a` is the lifetime of the input buffer being scanned.
pub trait Tokenizer<'a> {
    /// Scan and return the next token, advancing the cursor past it.
    fn step(&mut self) -> Result<Token<'a>, TokenizerError>;
    /// The current cursor position (a byte offset into the input).
    fn position(&self) -> usize;
    /// Move the cursor to `pos` (a byte offset previously obtained from [`Self::position`]).
    fn seek(&mut self, pos: usize);
    /// The full input buffer being scanned (used to slice out container byte ranges).
    fn input(&self) -> &'a [u8];
}

#[inline]
fn is_ws(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\r' || c == b'\n'
}

#[inline]
fn is_hex(c: u8) -> bool {
    c.is_ascii_digit() || (b'a'..=b'f').contains(&c) || (b'A'..=b'F').contains(&c)
}

/// A byte that ends a run of ordinary string-literal content: the closing quote, the start
/// of an escape, or a control character (which is a hard error inside a string).
#[inline]
fn is_string_event(c: u8) -> bool {
    c == b'"' || c == b'\\' || c < 0x20
}

/// A byte that can change container nesting, or open a string in which such a byte would be
/// mere content. The complete set of bytes a *skip* has to look at.
#[inline]
pub(crate) fn is_structural_event(c: u8) -> bool {
    matches!(c, b'{' | b'}' | b'[' | b']' | b'"')
}

/// Bulk run-crossing primitives — the only thing that differs between the scalar and SIMD
/// tokenizers.
///
/// Every method takes the whole input and a start offset and returns the index of the first
/// byte at or after `from` that the state machine must look at, or `data.len()` if the run
/// reaches the end of input. `from <= data.len()` is guaranteed by the caller.
///
/// # Contract
///
/// An implementation **must not** return an index past the first byte satisfying its
/// predicate — that would let the FSM swallow a token boundary or an error. Returning an
/// index *before* it is permitted (the FSM simply re-examines the byte and calls again), so
/// a conservative implementation is always sound; this is what makes the scalar and SIMD
/// paths byte-identical by construction rather than by agreement.
pub trait Scan: Copy + std::fmt::Debug {
    /// Build a scanner. Any runtime CPU feature detection happens **here**, once, never
    /// per byte or per token.
    fn new() -> Self;

    /// First byte that is `"`, `\`, or a control character (`< 0x20`).
    fn string_event(&self, data: &[u8], from: usize) -> usize;

    /// First byte that is not JSON whitespace (space, tab, CR, LF).
    fn skip_ws(&self, data: &[u8], from: usize) -> usize;

    /// First byte that is not an ASCII digit.
    fn skip_digits(&self, data: &[u8], from: usize) -> usize;

    /// First byte that is `{`, `}`, `[`, `]`, or `"`.
    ///
    /// Unlike the others this serves the *skip* path rather than the state machine: when a
    /// value's contents cannot affect the result, nothing in it matters except what nests a
    /// container or opens a string — so the matcher's skip path crosses it with this instead
    /// of running the state machine once per token.
    fn structural_event(&self, data: &[u8], from: usize) -> usize;

    /// Advance past a container whose opening bracket has already been consumed, returning
    /// the offset just past its matching close.
    ///
    /// `outer` is how many containers enclose this one, so a shared depth ceiling still
    /// applies. The default walks bytes, stopping at each structural event; vector backends
    /// override it with a block-at-a-time counter (see `simd::skip_container_blocked`) that
    /// does not stop at all for the common case of a block wholly interior to the value.
    ///
    /// # Validation
    ///
    /// Bracket balance and string termination only — see the matcher's `leave_value`.
    fn skip_container(&self, data: &[u8], from: usize, outer: usize) -> Result<usize, SkipError> {
        skip_container_bytewise(self, data, from, outer)
    }

    /// Run `f` inside whatever CPU-feature context this scanner's kernels require.
    ///
    /// This is the multiversioning hook. A kernel marked
    /// `#[target_feature(enable = "avx2")]` cannot be inlined into a caller lacking the
    /// feature, so wrapping only the kernel leaves a call per bulk scan. Wrapping the
    /// *whole state machine* instead lets the kernels fold into it. Scanners whose
    /// instructions are in the target's baseline (scalar, SSE2 on x86-64) leave this as
    /// the identity.
    #[inline(always)]
    fn enter<R>(f: impl FnOnce() -> R) -> R {
        f()
    }
}

/// The depth ceiling a skip enforces.
///
/// The same number as [`MAX_DEPTH`](crate::matcher::MAX_DEPTH), which re-exports it: a
/// document must not pass or fail depending on whether the expression happened to name a
/// field inside the deep part or skipped over it.
pub const MAX_SKIP_DEPTH: usize = 1024;

/// Why a bulk skip could not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipError {
    /// Input ended inside the container (or inside a string within it).
    Unterminated,
    /// Nesting inside the skipped value exceeded [`MAX_SKIP_DEPTH`].
    TooDeep,
}

/// The portable [`Scan::skip_container`], and the behavioural reference every vector
/// implementation is tested against.
///
/// Stops at each structural byte in turn. That is the right shape for a scanner with no
/// bulk classification available, and the wrong one where there is: it makes each scan
/// depend on where the previous one stopped, which on dense JSON is a latency chain rather
/// than throughput: the scan is a serial dependency where the state machine is not.
#[inline(always)]
pub(crate) fn skip_container_bytewise<S: Scan>(
    scan: &S,
    data: &[u8],
    from: usize,
    outer: usize,
) -> Result<usize, SkipError> {
    let len = data.len();
    let mut pos = from;
    // Containers still open, counting the one being skipped. Done when it reaches zero.
    let mut depth = 1usize;

    loop {
        pos = scan.structural_event(data, pos);
        if pos >= len {
            return Err(SkipError::Unterminated);
        }
        match data[pos] {
            b'{' | b'[' => {
                depth += 1;
                if outer + depth >= MAX_SKIP_DEPTH {
                    return Err(SkipError::TooDeep);
                }
                pos += 1;
            }
            b'}' | b']' => {
                pos += 1;
                depth -= 1;
                if depth == 0 {
                    return Ok(pos);
                }
            }
            // A string: brackets inside it are content, so cross it with the escape-aware
            // kernel rather than the structural one.
            _ => {
                pos += 1;
                loop {
                    pos = scan.string_event(data, pos);
                    if pos >= len {
                        return Err(SkipError::Unterminated);
                    }
                    match data[pos] {
                        b'"' => {
                            pos += 1;
                            break;
                        }
                        // `\X` hides at most one byte, and `\uXXXX`'s four hex digits are
                        // not string events, so skipping two is enough for every escape.
                        b'\\' => pos += 2,
                        // A control character. Illegal in a JSON string, but this region is
                        // not being validated; it is not a quote, so keep going.
                        _ => pos += 1,
                    }
                }
                // A trailing `\` can push `pos` past the end; the next `string_event` would
                // then be called with `from > len`, breaking its contract.
                if pos > len {
                    return Err(SkipError::Unterminated);
                }
            }
        }
    }
}

/// The portable, `unsafe`-free scanner: plain byte loops. Always available, and the
/// behavioural reference every other [`Scan`] is checked against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScalarScan;

impl Scan for ScalarScan {
    #[inline]
    fn new() -> Self {
        ScalarScan
    }

    #[inline]
    fn string_event(&self, data: &[u8], from: usize) -> usize {
        let mut i = from;
        while i < data.len() && !is_string_event(data[i]) {
            i += 1;
        }
        i
    }

    #[inline]
    fn skip_ws(&self, data: &[u8], from: usize) -> usize {
        let mut i = from;
        while i < data.len() && is_ws(data[i]) {
            i += 1;
        }
        i
    }

    #[inline]
    fn skip_digits(&self, data: &[u8], from: usize) -> usize {
        let mut i = from;
        while i < data.len() && data[i].is_ascii_digit() {
            i += 1;
        }
        i
    }

    #[inline]
    fn structural_event(&self, data: &[u8], from: usize) -> usize {
        let mut i = from;
        while i < data.len() && !is_structural_event(data[i]) {
            i += 1;
        }
        i
    }
}

/// Internal scan state.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    BeginValue,
    InString,
    InStringEsc,
    InStringEscU,
    InStringEscU1,
    InStringEscU12,
    InStringEscU123,
    Neg,
    Zero,
    One,
    Dot,
    Dot0,
    E,
    ESign,
    E0,
    T,
    Tr,
    Tru,
    F,
    Fa,
    Fal,
    Fals,
    N,
    Nu,
    Nul,
}

/// The JSON tokenizer state machine, parameterised by its [`Scan`] strategy.
///
/// Use the [`JsonTokenizer`] / `simd::SimdTokenizer` aliases rather
/// than naming this directly.
#[derive(Debug, Clone, Copy)]
pub struct GenericTokenizer<'a, S: Scan> {
    data: &'a [u8],
    pos: usize,
    scan: S,
}

/// The portable JSON tokenizer. No `unsafe`, available on every target.
pub type JsonTokenizer<'a> = GenericTokenizer<'a, ScalarScan>;

/// The tokenizer [`FastMatcher`](crate::matcher::FastMatcher) scans documents with:
/// [`SimdTokenizer`](crate::simd::SimdTokenizer) when the `simd` feature is on (the
/// default), [`JsonTokenizer`] otherwise.
#[cfg(feature = "simd")]
pub type DocTokenizer<'a> = crate::simd::SimdTokenizer<'a>;
/// The tokenizer [`FastMatcher`](crate::matcher::FastMatcher) scans documents with.
#[cfg(not(feature = "simd"))]
pub type DocTokenizer<'a> = JsonTokenizer<'a>;

impl<'a, S: Scan> GenericTokenizer<'a, S> {
    /// Create a tokenizer positioned at the start of `data`. This is where a SIMD scanner
    /// resolves its CPU backend, so prefer reusing one tokenizer over many short documents
    /// (see [`Self::reset`]) to per-document construction where that is convenient.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            scan: S::new(),
        }
    }

    /// Create a tokenizer with an explicitly supplied scanner, skipping [`Scan::new`].
    /// Mainly for tests that need to drive a specific backend rather than whichever one
    /// this CPU happens to select.
    pub fn with_scan(data: &'a [u8], scan: S) -> Self {
        Self { data, pos: 0, scan }
    }

    /// This tokenizer's already-resolved scanner, for callers that scan the same buffer
    /// themselves rather than through [`Tokenizer::step`] — the matcher's skip path does.
    /// Cheaper than `S::new()`, which for a detected backend re-runs (and re-asserts) CPU
    /// detection.
    pub fn scan(&self) -> S {
        self.scan
    }

    /// Point the tokenizer at a fresh input buffer and rewind to the start, keeping the
    /// already-resolved scanner.
    pub fn reset(&mut self, data: &'a [u8]) {
        self.data = data;
        self.pos = 0;
    }

    /// The full input buffer being scanned (used to slice out container byte ranges).
    #[inline]
    pub fn input(&self) -> &'a [u8] {
        self.data
    }
}

#[inline]
fn err(kind: TokenizerErrorKind, pos: usize) -> TokenizerError {
    TokenizerError { kind, pos }
}

impl<'a, S: Scan> Tokenizer<'a> for GenericTokenizer<'a, S> {
    #[inline]
    fn position(&self) -> usize {
        self.pos
    }

    #[inline]
    fn seek(&mut self, pos: usize) {
        self.pos = pos;
    }

    #[inline]
    fn input(&self) -> &'a [u8] {
        self.data
    }

    #[inline]
    fn step(&mut self) -> Result<Token<'a>, TokenizerError> {
        S::enter(|| self.step_impl())
    }
}

impl<'a, S: Scan> GenericTokenizer<'a, S> {
    /// The state machine proper. `inline(always)` so that when [`Scan::enter`] wraps it in a
    /// `#[target_feature]` context, the whole thing — kernels included — lands inside.
    #[inline(always)]
    fn step_impl(&mut self) -> Result<Token<'a>, TokenizerError> {
        let data = self.data;
        let len = data.len();
        let mut pos = self.pos;

        if pos >= len {
            return Ok(Token {
                token_type: TokenType::End,
                value: &[],
            });
        }

        let mut start = pos;
        let mut state = State::BeginValue;
        let mut has_escapes = false;
        let mut non_integer = false;

        let token_type = loop {
            if pos >= len {
                // Numbers terminate by look-ahead, so a number resting in an accepting
                // state at EOF is complete; whitespace-only input yields End.
                match state {
                    State::Zero | State::One | State::Dot0 | State::E0 => break TokenType::Number,
                    State::BeginValue => break TokenType::End,
                    _ => return Err(err(TokenizerErrorKind::UnexpectedEof, pos)),
                }
            }

            let c = data[pos];
            pos += 1;

            match state {
                State::BeginValue => {
                    if is_ws(c) {
                        // Leading whitespace is not part of the token; cross the run and
                        // restart the token there.
                        pos = self.scan.skip_ws(data, pos);
                        start = pos;
                        continue;
                    }
                    match c {
                        b'{' => break TokenType::ObjectStart,
                        b'}' => break TokenType::ObjectEnd,
                        b':' => break TokenType::ObjectKeyDelim,
                        b'[' => break TokenType::ArrayStart,
                        b']' => break TokenType::ArrayEnd,
                        b',' => break TokenType::ListDelim,
                        b'"' => state = State::InString,
                        b'-' => state = State::Neg,
                        b'0' => state = State::Zero,
                        b't' => state = State::T,
                        b'f' => state = State::F,
                        b'n' => state = State::N,
                        b'1'..=b'9' => state = State::One,
                        _ => return Err(err(TokenizerErrorKind::UnexpectedByte(c), pos - 1)),
                    }
                }

                State::InString => match c {
                    b'"' => break TokenType::EscString,
                    b'\\' => state = State::InStringEsc,
                    0x00..=0x1f => {
                        return Err(err(TokenizerErrorKind::ControlCharInString, pos - 1))
                    }
                    // Ordinary content: cross the rest of the run in one call. The three
                    // arms above are exactly `is_string_event`, so the scanner lands on a
                    // byte one of them handles (or on EOF, caught at the top of the loop).
                    _ => pos = self.scan.string_event(data, pos),
                },

                State::InStringEsc => {
                    has_escapes = true;
                    match c {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            state = State::InString
                        }
                        b'u' => state = State::InStringEscU,
                        _ => return Err(err(TokenizerErrorKind::InvalidEscape, pos - 1)),
                    }
                }

                State::InStringEscU => {
                    if is_hex(c) {
                        state = State::InStringEscU1;
                    } else {
                        return Err(err(TokenizerErrorKind::InvalidUnicodeEscape, pos - 1));
                    }
                }
                State::InStringEscU1 => {
                    if is_hex(c) {
                        state = State::InStringEscU12;
                    } else {
                        return Err(err(TokenizerErrorKind::InvalidUnicodeEscape, pos - 1));
                    }
                }
                State::InStringEscU12 => {
                    if is_hex(c) {
                        state = State::InStringEscU123;
                    } else {
                        return Err(err(TokenizerErrorKind::InvalidUnicodeEscape, pos - 1));
                    }
                }
                State::InStringEscU123 => {
                    if is_hex(c) {
                        state = State::InString;
                    } else {
                        return Err(err(TokenizerErrorKind::InvalidUnicodeEscape, pos - 1));
                    }
                }

                State::Neg => match c {
                    b'0' => state = State::Zero,
                    b'1'..=b'9' => state = State::One,
                    _ => return Err(err(TokenizerErrorKind::InvalidNumber, pos - 1)),
                },

                State::One => match c {
                    b'0'..=b'9' => pos = self.scan.skip_digits(data, pos),
                    b'.' => state = State::Dot,
                    b'e' | b'E' => state = State::E,
                    _ => {
                        // Non-numeric byte terminates the number; rewind so the caller
                        // re-reads it as the next token.
                        pos -= 1;
                        break TokenType::Number;
                    }
                },

                State::Zero => match c {
                    b'.' => state = State::Dot,
                    b'e' | b'E' => state = State::E,
                    _ => {
                        pos -= 1;
                        break TokenType::Number;
                    }
                },

                State::Dot => {
                    non_integer = true;
                    match c {
                        b'0'..=b'9' => state = State::Dot0,
                        _ => return Err(err(TokenizerErrorKind::InvalidNumber, pos - 1)),
                    }
                }

                State::Dot0 => match c {
                    b'0'..=b'9' => pos = self.scan.skip_digits(data, pos),
                    b'e' | b'E' => state = State::E,
                    _ => {
                        pos -= 1;
                        break TokenType::Number;
                    }
                },

                State::E => {
                    non_integer = true;
                    match c {
                        b'+' | b'-' => state = State::ESign,
                        b'0'..=b'9' => state = State::E0,
                        _ => return Err(err(TokenizerErrorKind::InvalidNumber, pos - 1)),
                    }
                }

                State::ESign => match c {
                    b'0'..=b'9' => state = State::E0,
                    _ => return Err(err(TokenizerErrorKind::InvalidNumber, pos - 1)),
                },

                State::E0 => match c {
                    b'0'..=b'9' => pos = self.scan.skip_digits(data, pos),
                    _ => {
                        pos -= 1;
                        break TokenType::Number;
                    }
                },

                State::T => {
                    if c == b'r' {
                        state = State::Tr;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "true" },
                            pos - 1,
                        ));
                    }
                }
                State::Tr => {
                    if c == b'u' {
                        state = State::Tru;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "true" },
                            pos - 1,
                        ));
                    }
                }
                State::Tru => {
                    if c == b'e' {
                        break TokenType::True;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "true" },
                            pos - 1,
                        ));
                    }
                }

                State::F => {
                    if c == b'a' {
                        state = State::Fa;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "false" },
                            pos - 1,
                        ));
                    }
                }
                State::Fa => {
                    if c == b'l' {
                        state = State::Fal;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "false" },
                            pos - 1,
                        ));
                    }
                }
                State::Fal => {
                    if c == b's' {
                        state = State::Fals;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "false" },
                            pos - 1,
                        ));
                    }
                }
                State::Fals => {
                    if c == b'e' {
                        break TokenType::False;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "false" },
                            pos - 1,
                        ));
                    }
                }

                State::N => {
                    if c == b'u' {
                        state = State::Nu;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "null" },
                            pos - 1,
                        ));
                    }
                }
                State::Nu => {
                    if c == b'l' {
                        state = State::Nul;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "null" },
                            pos - 1,
                        ));
                    }
                }
                State::Nul => {
                    if c == b'l' {
                        break TokenType::Null;
                    } else {
                        return Err(err(
                            TokenizerErrorKind::InvalidLiteral { expected: "null" },
                            pos - 1,
                        ));
                    }
                }
            }
        };

        // Refine the coarse classification made during scanning.
        let token_type = match token_type {
            TokenType::Number if !non_integer => TokenType::Integer,
            TokenType::EscString if !has_escapes => TokenType::String,
            other => other,
        };

        let value = &data[start..pos];
        self.pos = pos;
        Ok(Token { token_type, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collect the full `(type, value-as-str)` token stream, stopping at `End`.
    fn scan(input: &str) -> Result<Vec<(TokenType, String)>, TokenizerError> {
        let mut t = JsonTokenizer::new(input.as_bytes());
        let mut out = Vec::new();
        loop {
            let tok = t.step()?;
            if tok.token_type == TokenType::End {
                break;
            }
            out.push((
                tok.token_type,
                String::from_utf8(tok.value.to_vec()).unwrap(),
            ));
        }
        Ok(out)
    }

    fn types(input: &str) -> Vec<TokenType> {
        scan(input).unwrap().into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn scans_object_structure() {
        use TokenType::*;
        assert_eq!(
            scan(r#"{ "a": 1 }"#).unwrap(),
            vec![
                (ObjectStart, "{".into()),
                (String, "\"a\"".into()),
                (ObjectKeyDelim, ":".into()),
                (Integer, "1".into()),
                (ObjectEnd, "}".into()),
            ]
        );
    }

    #[test]
    fn scans_array_structure() {
        use TokenType::*;
        assert_eq!(
            types("[1, 2, 3]"),
            vec![ArrayStart, Integer, ListDelim, Integer, ListDelim, Integer, ArrayEnd]
        );
    }

    #[test]
    fn token_values_include_quotes_and_verbatim_bytes() {
        let toks = scan(r#""hello""#).unwrap();
        assert_eq!(toks, vec![(TokenType::String, "\"hello\"".into())]);
    }

    #[test]
    fn distinguishes_plain_from_escaped_strings() {
        // EscString is specifically about backslash escapes, not non-ASCII content.
        assert_eq!(types(r#""plain""#), vec![TokenType::String]);
        assert_eq!(types(r#""unié""#), vec![TokenType::String]); // raw UTF-8, no escapes
        assert_eq!(types(r#""esc\n""#), vec![TokenType::EscString]);
        assert_eq!(types(r#""esc\té""#), vec![TokenType::EscString]); // escape + UTF-8
    }

    #[test]
    fn distinguishes_integer_from_number() {
        assert_eq!(types("42"), vec![TokenType::Integer]);
        assert_eq!(types("-42"), vec![TokenType::Integer]);
        assert_eq!(types("0"), vec![TokenType::Integer]);
        assert_eq!(types("3.14"), vec![TokenType::Number]);
        assert_eq!(types("1e10"), vec![TokenType::Number]);
        assert_eq!(types("-0.5e+3"), vec![TokenType::Number]);
        assert_eq!(types("1.0583162526594752e+308"), vec![TokenType::Number]);
    }

    #[test]
    fn scans_keywords() {
        use TokenType::*;
        assert_eq!(types("true"), vec![True]);
        assert_eq!(types("false"), vec![False]);
        assert_eq!(types("null"), vec![Null]);
        assert_eq!(
            types("[true, false, null]"),
            vec![ArrayStart, True, ListDelim, False, ListDelim, Null, ArrayEnd]
        );
    }

    #[test]
    fn bare_number_at_eof_terminates() {
        // Numbers end by look-ahead; a number that runs to EOF must still be emitted.
        assert_eq!(types("123"), vec![TokenType::Integer]);
        assert_eq!(types("1.5"), vec![TokenType::Number]);
        assert_eq!(types("10e5"), vec![TokenType::Number]);
    }

    #[test]
    fn empty_and_whitespace_only_input_is_end() {
        assert_eq!(types(""), Vec::<TokenType>::new());
        assert_eq!(types("   \t\r\n "), Vec::<TokenType>::new());
    }

    #[test]
    fn end_is_idempotent() {
        let mut t = JsonTokenizer::new(b"1");
        assert_eq!(t.step().unwrap().token_type, TokenType::Integer);
        assert_eq!(t.step().unwrap().token_type, TokenType::End);
        assert_eq!(t.step().unwrap().token_type, TokenType::End);
    }

    #[test]
    fn seek_and_position_round_trip() {
        let mut t = JsonTokenizer::new(b"[10, 20]");
        assert_eq!(t.step().unwrap().token_type, TokenType::ArrayStart);
        let mark = t.position();
        let first = t.step().unwrap();
        assert_eq!(
            (first.token_type, first.value),
            (TokenType::Integer, &b"10"[..])
        );
        // Rewind and re-read the same token.
        t.seek(mark);
        let again = t.step().unwrap();
        assert_eq!(
            (again.token_type, again.value),
            (TokenType::Integer, &b"10"[..])
        );
    }

    #[test]
    fn rejects_non_standard_case_insensitive_literals() {
        // Strict JSON: only lowercase keywords are accepted.
        assert!(scan("TRUE").is_err());
        assert!(scan("True").is_err());
        assert!(scan("NULL").is_err());
        assert!(scan("False").is_err());
    }

    #[test]
    fn reports_errors_with_kind_and_position() {
        use TokenizerErrorKind::*;
        let cases: &[(&str, TokenizerErrorKind)] = &[
            ("@", UnexpectedByte(b'@')),
            ("nope", InvalidLiteral { expected: "null" }),
            (r#""a\x""#, InvalidEscape),
            (r#""a\u12g4""#, InvalidUnicodeEscape),
        ];
        for (input, _kind) in cases {
            assert!(scan(input).is_err(), "expected error scanning {input:?}");
        }

        // Control character (raw newline) inside a string literal is rejected.
        let err = scan("\"a\nb\"").unwrap_err();
        assert_eq!(err.kind, ControlCharInString);

        // Bad escape carries a precise position.
        let err = scan(r#""a\x""#).unwrap_err();
        assert_eq!(err.kind, InvalidEscape);
        assert_eq!(err.pos, 3); // the 'x'
    }

    #[test]
    fn unterminated_string_is_eof_error() {
        let err = scan(r#""abc"#).unwrap_err();
        assert_eq!(err.kind, TokenizerErrorKind::UnexpectedEof);
    }
}
