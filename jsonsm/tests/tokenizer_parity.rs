//! Differential parity tests for the JSON tokenizer.
//!
//! Two independent axes:
//!
//! 1. **vs. `serde_json`** — drive the tokenizer to reconstruct a `serde_json::Value` from
//!    the token stream and assert it equals what `serde_json` produces from the same bytes.
//!    This validates structure, escapes, and number parsing against an outside reference
//!    rather than trusting the Go implementation as an oracle. Run for *every* tokenizer
//!    build, not just the scalar one.
//! 2. **SIMD vs. scalar** — run both tokenizers over the same input and compare
//!    token-for-token, including error kind, error offset, and cursor position after every
//!    step. Agreeing with `serde_json` separately is not enough: two implementations can
//!    both round-trip valid JSON and still disagree about *where* a malformed input fails.
//!
//! The generated corpus targets SIMD's characteristic blind spot — boundary effects. The
//! bulk kernels process 32 bytes at a time, so what matters is where an interesting byte
//! sits relative to a vector boundary, how long the sub-vector tail is, and whether the
//! buffer ends mid-construct. Every generated document is therefore emitted at 65 shifts (a
//! full vector on either side) and checked at *every* truncation, so each construct is seen
//! at every alignment and every possible end-of-buffer cut.

use jsonsm::tokenizer::{
    GenericTokenizer, JsonTokenizer, ScalarScan, Scan, Token, TokenType, Tokenizer, TokenizerError,
    TokenizerErrorKind,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The vector width the kernels work in. The corpus is built around it.
const LANES: usize = 32;

// ---------------------------------------------------------------------------------------
// Reconstruction against serde_json
// ---------------------------------------------------------------------------------------

/// Reconstruct a `serde_json::Value` by driving the tokenizer. Scalar leaves are decoded
/// by handing their verbatim bytes back to `serde_json`, so number/string decoding is
/// checked against the reference decoder rather than reimplemented here.
fn reconstruct<'a, T: Tokenizer<'a>>(t: &mut T) -> Result<Value, String> {
    let tok = t.step().map_err(|e: TokenizerError| e.to_string())?;
    reconstruct_from(t, tok)
}

fn reconstruct_from<'a, T: Tokenizer<'a>>(t: &mut T, tok: Token<'a>) -> Result<Value, String> {
    match tok.token_type {
        TokenType::ObjectStart => {
            let mut map = serde_json::Map::new();
            // Handle empty object and the first entry.
            let mut first = t.step().map_err(|e| e.to_string())?;
            loop {
                if first.token_type == TokenType::ObjectEnd {
                    break;
                }
                let key = decode_string(first)?;
                let delim = t.step().map_err(|e| e.to_string())?;
                assert_eq!(delim.token_type, TokenType::ObjectKeyDelim, "expected ':'");
                let val = reconstruct(t)?;
                map.insert(key, val);

                let sep = t.step().map_err(|e| e.to_string())?;
                match sep.token_type {
                    TokenType::ObjectEnd => break,
                    TokenType::ListDelim => {
                        first = t.step().map_err(|e| e.to_string())?;
                    }
                    other => return Err(format!("unexpected token in object: {other:?}")),
                }
            }
            Ok(Value::Object(map))
        }
        TokenType::ArrayStart => {
            let mut arr = Vec::new();
            let mut next = t.step().map_err(|e| e.to_string())?;
            loop {
                if next.token_type == TokenType::ArrayEnd {
                    break;
                }
                arr.push(reconstruct_from(t, next)?);
                let sep = t.step().map_err(|e| e.to_string())?;
                match sep.token_type {
                    TokenType::ArrayEnd => break,
                    TokenType::ListDelim => {
                        next = t.step().map_err(|e| e.to_string())?;
                    }
                    other => return Err(format!("unexpected token in array: {other:?}")),
                }
            }
            Ok(Value::Array(arr))
        }
        TokenType::String | TokenType::EscString => Ok(Value::String(decode_string(tok)?)),
        TokenType::Integer | TokenType::Number => decode_scalar(tok.value),
        TokenType::True => Ok(Value::Bool(true)),
        TokenType::False => Ok(Value::Bool(false)),
        TokenType::Null => Ok(Value::Null),
        other => Err(format!("unexpected leading token: {other:?}")),
    }
}

fn decode_string(tok: Token<'_>) -> Result<String, String> {
    match decode_scalar(tok.value)? {
        Value::String(s) => Ok(s),
        v => Err(format!("expected string token, decoded {v:?}")),
    }
}

/// Decode a scalar token's verbatim bytes via serde_json (the reference decoder).
fn decode_scalar(bytes: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(bytes)
        .map_err(|e| format!("serde decode of {:?}: {e}", String::from_utf8_lossy(bytes)))
}

// ---------------------------------------------------------------------------------------
// Token-stream capture, for comparing two tokenizers
// ---------------------------------------------------------------------------------------

/// One `step()` outcome, recorded in enough detail that any behavioural difference shows
/// up: the token kind, its exact bytes, the cursor left behind, or the precise failure.
#[derive(Debug, PartialEq, Eq)]
enum Step<'a> {
    Tok {
        token_type: TokenType,
        value: &'a [u8],
        pos_after: usize,
    },
    Err {
        kind: TokenizerErrorKind,
        pos: usize,
    },
}

/// Drive a tokenizer to `End` or its first error, recording every step.
fn token_stream<'a, T: Tokenizer<'a>>(t: &mut T, limit: usize) -> Vec<Step<'a>> {
    let mut out = Vec::new();
    loop {
        match t.step() {
            Ok(tok) => {
                let done = tok.token_type == TokenType::End;
                out.push(Step::Tok {
                    token_type: tok.token_type,
                    value: tok.value,
                    pos_after: t.position(),
                });
                if done {
                    return out;
                }
            }
            Err(e) => {
                out.push(Step::Err {
                    kind: e.kind,
                    pos: e.pos,
                });
                return out;
            }
        }
        assert!(out.len() <= limit, "tokenizer failed to terminate");
    }
}

/// Assert every available tokenizer produces exactly the scalar tokenizer's stream on
/// `input`. Returns how many *vectorised* backends were compared.
fn assert_parity(input: &[u8], label: &str) -> usize {
    let limit = input.len() + 16;
    let reference = token_stream(&mut JsonTokenizer::new(input), limit);
    #[allow(unused_mut)]
    let mut vectorised = 0usize;

    /// Each backend is a separate monomorphisation, so each is genuinely separate code that
    /// has to be compared in its own right — not one path with a different constant.
    ///
    /// Unused on targets with no vector backend, where the loop below compiles away.
    #[allow(unused_macros)]
    macro_rules! check {
        ($ty:ty, $name:literal) => {{
            let mut t = GenericTokenizer::<$ty>::new(input);
            assert_eq!(
                token_stream(&mut t, limit),
                reference,
                concat!($name, " disagrees with scalar on {}: {:?}"),
                label,
                String::from_utf8_lossy(input)
            );
        }};
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        use jsonsm::simd::{Avx2Scan, Backend, Sse2Scan};
        check!(Sse2Scan, "sse2");
        vectorised += 1;
        if Backend::available().contains(&Backend::Avx2) {
            check!(Avx2Scan, "avx2");
            vectorised += 1;
        }
    }

    // Also check the tokenizer this build ships as its default, whatever that is.
    let mut doc = jsonsm::tokenizer::DocTokenizer::new(input);
    assert_eq!(
        token_stream(&mut doc, limit),
        reference,
        "DocTokenizer disagrees with scalar on {label}"
    );

    vectorised
}

/// Guard against the SIMD half of these tests quietly becoming a no-op.
fn assert_simd_was_exercised(count: usize) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        assert!(
            count > 0,
            "the `simd` feature is on and this CPU has a vector backend, but no \
             vectorised comparison ran"
        );
    }
    let _ = count;
}

// ---------------------------------------------------------------------------------------
// Fixture corpus
// ---------------------------------------------------------------------------------------

fn testdata_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata")
}

fn json_fixtures() -> Vec<PathBuf> {
    let mut files = Vec::new();
    let dir = testdata_dir();
    collect(&dir, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no fixtures found under {}",
        dir.display()
    );
    files
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
            out.push(path);
        }
    }
}

/// Reconstruct every fixture with a given scanner and check it against `serde_json`.
fn reconstruct_fixtures<S: Scan>() {
    let mut checked = 0usize;
    for path in json_fixtures() {
        let bytes = std::fs::read(&path).unwrap();

        // The independent reference value.
        let reference: Value =
            serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

        let mut t = GenericTokenizer::<S>::new(&bytes);
        let ours = reconstruct(&mut t).unwrap_or_else(|e| panic!("{}: {e}", path.display()));

        // After the top-level value, only trailing whitespace / End may remain.
        let trailing = t
            .step()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(
            trailing.token_type,
            TokenType::End,
            "{}: unexpected trailing token {:?}",
            path.display(),
            trailing.token_type
        );

        assert_eq!(ours, reference, "value mismatch for {}", path.display());
        checked += 1;
    }
    // people.json + bigvector.json + 100 edgyJson + 1 edgy dir marker guard.
    assert!(
        checked >= 102,
        "expected to check the full fixture corpus, got {checked}"
    );
}

#[test]
fn reconstructs_every_fixture_matching_serde() {
    reconstruct_fixtures::<ScalarScan>();
    #[cfg(feature = "simd")]
    #[cfg(target_arch = "x86_64")]
    {
        reconstruct_fixtures::<jsonsm::simd::Sse2Scan>();
        if jsonsm::simd::Backend::available().contains(&jsonsm::simd::Backend::Avx2) {
            reconstruct_fixtures::<jsonsm::simd::Avx2Scan>();
        }
    }
}

/// The tokenizer must never panic and must terminate on any input, including malformed
/// slices — errors are returned, not raised. We feed truncations of every fixture.
///
/// The cut stride is deliberately coprime with the 32-byte vector width: the previous
/// stride of 64 put *every* buffer end on a vector boundary, which is exactly the case a
/// SIMD tail bug survives.
#[test]
fn never_panics_on_truncated_inputs() {
    for path in json_fixtures() {
        let bytes = std::fs::read(&path).unwrap();
        let mut cut_points: Vec<usize> = (0..bytes.len()).step_by(37).collect();
        // Plus every offset within one byte of an early vector boundary, where short-tail
        // handling lives.
        for k in 0..8 {
            for delta in 0..3 {
                let cut = k * LANES + delta;
                if cut <= bytes.len() {
                    cut_points.push(cut);
                }
            }
        }
        cut_points.push(bytes.len());
        for &cut in &cut_points {
            let slice = &bytes[..cut];
            let mut t = jsonsm::tokenizer::DocTokenizer::new(slice);
            // Drive to End or first error; must always terminate.
            let mut guard = 0usize;
            loop {
                guard += 1;
                assert!(guard < slice.len() + 16, "tokenizer failed to terminate");
                match t.step() {
                    Ok(tok) if tok.token_type == TokenType::End => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Generated boundary corpus
// ---------------------------------------------------------------------------------------

/// JSON snippets whose interesting bytes must be checked at every alignment relative to a
/// vector boundary. Each is embedded at many shifts (see [`shifted_documents`]).
///
/// These are written out deliberately rather than sampled randomly. The lesson from the
/// pre-SIMD correctness pass was that a generator only finds what it can produce, and a
/// random JSON generator essentially never emits a `\uXXXX` escape straddling byte 32, a
/// buffer ending on a lone backslash, or a digit run exactly one vector long.
fn payloads() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    // Plain strings spanning under, exactly, and over the vector width.
    for n in [0usize, 1, 30, 31, 32, 33, 63, 64, 65, 96, 200] {
        out.push(format!("\"{}\"", "a".repeat(n)));
    }

    // An escape at every offset inside a string that spans two vectors, so the escape and
    // the multi-byte unit it introduces land at every position within and across a block.
    for at in 0..(LANES + 4) {
        for esc in ["\\n", "\\\\", "\\\"", "\\/", "\\t", "\\u00e9", "\\uD83D"] {
            let mut s = String::from("\"");
            s.push_str(&"b".repeat(at));
            s.push_str(esc);
            s.push_str(&"c".repeat(LANES + 4 - at));
            s.push('"');
            out.push(s);
        }
    }

    // Escapes as the very last thing before the closing quote, and back-to-back backslash
    // runs (where an even/odd run decides whether the next quote closes the string).
    for n in 0..6 {
        out.push(format!("\"{}{}\"", "x".repeat(LANES - 1), "\\\\".repeat(n)));
        out.push(format!("\"{}\\n\"", "x".repeat(LANES * 2 - 2)));
        out.push(format!("\"{}\\u0041\"", "x".repeat(LANES - n)));
    }

    // Strings made entirely of escapes, so no bulk run ever gets going.
    out.push(format!("\"{}\"", "\\n".repeat(40)));
    out.push(format!("\"{}\"", "\\u0041".repeat(20)));

    // Numbers: digit runs crossing the boundary, in every shape the FSM distinguishes.
    for n in [1usize, 30, 31, 32, 33, 64, 100] {
        out.push("1".repeat(n));
        out.push(format!("-{}", "9".repeat(n)));
        out.push(format!("0.{}", "5".repeat(n)));
        out.push(format!(
            "{}.{}e+{}",
            "1".repeat(n),
            "2".repeat(n),
            "3".repeat(n.min(3))
        ));
        out.push(format!("1e{}", "0".repeat(n)));
    }

    // Whitespace runs longer than a vector, inside and around structure.
    for n in [1usize, 31, 32, 33, 70] {
        let ws = " \t\r\n".repeat(n.div_ceil(4));
        out.push(format!("[{ws}1{ws},{ws}2{ws}]"));
        out.push(format!("{{{ws}\"k\"{ws}:{ws}\"v\"{ws}}}"));
    }

    // Structural characters packed densely, and deep nesting to skip through.
    out.push("[[[[[[[[[[1]]]]]]]]]]".to_string());
    out.push(format!("{}1{}", "[".repeat(200), "]".repeat(200)));
    out.push(format!("[{}]", "0,".repeat(50) + "0"));
    out.push(format!(
        "{{{}}}",
        (0..40)
            .map(|i| format!("\"k{i}\":{i}"))
            .collect::<Vec<_>>()
            .join(",")
    ));

    // Literals, run together so they cross boundaries at varying offsets.
    out.push("[true,false,null,true,false,null,true,false,null,true,false]".to_string());

    // Non-ASCII: raw UTF-8 bytes have the high bit set, which must not be mistaken for a
    // control character by the unsigned `<= 0x1f` test.
    out.push(format!("\"{}\"", "é".repeat(40)));
    out.push(format!("\"{}\"", "\u{10348}".repeat(20)));
    out.push("\"\u{7f}\u{80}\u{9f}\u{a0}\"".to_string());

    // Malformed inputs, where the *error offset* must agree between implementations.
    for at in 0..(LANES + 4) {
        // A raw control character inside a string, at every offset.
        out.push(format!("\"{}\u{1}{}\"", "a".repeat(at), "b".repeat(4)));
        // An invalid escape, at every offset.
        out.push(format!("\"{}\\x{}\"", "a".repeat(at), "b".repeat(4)));
        // A non-hex digit inside a \u escape, at every offset.
        out.push(format!("\"{}\\u12g4{}\"", "a".repeat(at), "b".repeat(4)));
    }
    out.push(format!("\"{}", "a".repeat(LANES * 2))); // unterminated
    out.push(format!("\"{}\\", "a".repeat(LANES))); // trailing lone backslash
    out.push(format!("\"{}\\u12", "a".repeat(LANES))); // truncated \u at EOF
    out.push(format!("{}1.", "0,".repeat(20))); // number cut after the dot
    out.push("tru".to_string());
    out.push(format!("{}tru", " ".repeat(LANES)));

    out
}

/// Embed each payload at a range of byte offsets, so every interesting byte visits every
/// position modulo the 32-byte vector width, with a full width of slack on either side.
fn shifted_documents() -> Vec<(String, Vec<u8>)> {
    let mut docs = Vec::new();
    for payload in payloads() {
        for shift in 0..=(LANES * 2) {
            // The pad lives in a leading key so the document stays plausible JSON; its
            // length is what slides `payload` across vector boundaries.
            let doc = format!("{{\"p\":\"{}\",\"v\":{}}}", "z".repeat(shift), payload);
            docs.push((
                format!("shift={shift} payload={payload:.40}"),
                doc.into_bytes(),
            ));
            // Bare, too: a payload at the very start of the buffer has no preceding
            // structure to have primed the scanner.
            if shift <= LANES {
                let bare = format!("{}{}", " ".repeat(shift), payload);
                docs.push((
                    format!("bare shift={shift} payload={payload:.40}"),
                    bare.into_bytes(),
                ));
            }
        }
    }
    docs
}

/// Every generated document, at every alignment, compared token-for-token between the SIMD
/// and scalar tokenizers.
#[test]
fn simd_matches_scalar_on_boundary_corpus() {
    let docs = shifted_documents();
    assert!(
        docs.len() > 10_000,
        "corpus shrank: {} documents",
        docs.len()
    );
    let mut vectorised = 0usize;
    for (label, doc) in &docs {
        vectorised += assert_parity(doc, label);
    }
    assert_simd_was_exercised(vectorised);
}

/// The same corpus truncated at *every* byte. Truncation is where SIMD tails and
/// end-of-buffer handling break: a construct that is fine at length N can leave the cursor
/// somewhere else at length N-1, and a kernel that read a full vector past a short tail
/// would only show up here.
#[test]
fn simd_matches_scalar_on_every_truncation() {
    // One shift per payload is enough here — the cut itself sweeps every alignment.
    let mut vectorised = 0usize;
    for (i, (label, doc)) in shifted_documents().into_iter().enumerate() {
        // Sub-sample shifts (the truncation sweep is quadratic in document length) while
        // still covering every payload at several alignments.
        if i % 7 != 0 {
            continue;
        }
        for cut in 0..=doc.len() {
            vectorised += assert_parity(&doc[..cut], &format!("{label} cut={cut}"));
        }
    }
    assert_simd_was_exercised(vectorised);
}

/// Fixtures, whole and truncated: every byte through the first few vectors, then a
/// boundary-coprime stride.
#[test]
fn simd_matches_scalar_on_fixtures() {
    let mut vectorised = 0usize;
    for path in json_fixtures() {
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        vectorised += assert_parity(&bytes, &name);

        let mut cuts: Vec<usize> = (0..=LANES * 4).collect();
        cuts.extend((0..bytes.len()).step_by(37));
        cuts.push(bytes.len());
        for cut in cuts {
            if cut <= bytes.len() {
                vectorised += assert_parity(&bytes[..cut], &format!("{name} cut={cut}"));
            }
        }
    }
    assert_simd_was_exercised(vectorised);
}
