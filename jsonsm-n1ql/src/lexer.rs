//! The lexer for the N1QL-ish filter grammar, built with [`logos`], feeding the
//! LALRPOP-generated parser via its `extern` token interface.
//!
//! An external lexer rather than LALRPOP's own: its built-in regex lexer does not handle this
//! grammar's string literals, so collapsing the two back into one grammar file is a dead end.

use logos::Logos;

/// A lexical token. Keywords are case-insensitive; keyword `#[token]`s outrank the
/// identifier `#[regex]` on ties, and logos's longest-match rule keeps `android` an
/// identifier rather than the keyword `and`.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
pub enum Token {
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("=")]
    Eq,
    #[token("==")]
    EqEq,
    #[token("<>")]
    Ne1,
    #[token("!=")]
    Ne2,
    #[token("<")]
    Lt,
    #[token("<=")]
    Le,
    #[token(">")]
    Gt,
    #[token(">=")]
    Ge,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("!")]
    Bang,
    #[token("and", ignore(ascii_case))]
    And,
    #[token("or", ignore(ascii_case))]
    Or,
    #[token("not", ignore(ascii_case))]
    Not,
    #[token("is", ignore(ascii_case))]
    Is,
    #[token("null", ignore(ascii_case))]
    Null,
    #[token("missing", ignore(ascii_case))]
    Missing,
    #[token("true", ignore(ascii_case))]
    True,
    #[token("false", ignore(ascii_case))]
    False,
    #[token("exists", ignore(ascii_case))]
    Exists,
    #[token("regexp_contains", ignore(ascii_case))]
    Regexp,
    #[token("any", ignore(ascii_case))]
    Any,
    #[token("every", ignore(ascii_case))]
    Every,
    #[token("in", ignore(ascii_case))]
    In,
    #[token("satisfies", ignore(ascii_case))]
    Satisfies,
    #[token("end", ignore(ascii_case))]
    End,
    #[regex(r"[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?", |l| l.slice().to_owned())]
    Num(String),
    #[regex(r#""([^"\\]|\\.)*""#, |l| l.slice().to_owned())]
    DqStr(String),
    #[regex(r"'([^'\\]|\\.)*'", |l| l.slice().to_owned())]
    SqStr(String),
    #[regex(r"`[^`]*`", |l| l.slice().to_owned())]
    BqIdent(String),
    #[regex(r"[A-Za-z_$][A-Za-z0-9_$]*", |l| l.slice().to_owned())]
    Ident(String),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// A lexing failure, carrying the byte offset of the offending input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unexpected character at byte {0}")]
pub struct LexError(pub usize);

/// Produce the spanned `(start, token, end)` stream LALRPOP consumes.
pub fn lex(input: &str) -> impl Iterator<Item = Result<(usize, Token, usize), LexError>> + '_ {
    Token::lexer(input).spanned().map(|(res, span)| match res {
        Ok(tok) => Ok((span.start, tok, span.end)),
        Err(()) => Err(LexError(span.start)),
    })
}
