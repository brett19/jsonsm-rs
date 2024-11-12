use crate::jsontokenizer_token::{JsonToken, JsonTokenType};

type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

type TokenizerInternalResult<T> = std::result::Result<T, TokenizerErrorType>;

#[derive(Debug)]
pub enum TokenizerErrorType {
    UnexpectedEndOfInput,
    UnexpectedBeginChar,
    UnexpectedEscapeCode,
    UnexpectedCharInNumericLiteral,
    UnexpectedCharInExponentLiteral,
    UnexpectedCharInTrueLiteral,
    UnexpectedCharInFalseLiteral,
    UnexpectedCharInNullLiteral,
}

#[derive(Debug)]
pub struct TokenizerError {
    pub error_type: TokenizerErrorType,
    pub pos: usize,
}

pub struct JsonTokenizer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> JsonTokenizer<'a> {
    pub fn new(input: &'a [u8]) -> JsonTokenizer<'a> {
        JsonTokenizer { input, pos: 0 }
    }

    #[inline]
    fn peek_or_null(&mut self) -> u8 {
        if self.pos >= self.input.len() {
            return 0;
        }

        self.input[self.pos]
    }

    #[inline]
    fn read_or_null(&mut self) -> u8 {
        if self.pos >= self.input.len() {
            return 0;
        }

        let c = self.input[self.pos];
        self.pos += 1;
        c
    }

    #[inline]
    fn eat(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn skip_unicodeescape(&mut self) -> TokenizerInternalResult<()> {
        match self.read_or_null() {
            b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => match self.read_or_null() {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => match self.read_or_null() {
                    b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => match self.read_or_null() {
                        b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => Ok(()),
                        _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
                    },
                    _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
                },
                _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
            },
            _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
        }
    }

    #[inline]
    fn skip_stringescape(&mut self) -> TokenizerInternalResult<()> {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'b' | b'f' | b'n' | b'r' | b't' | b'"' | b'\\' | b'/' => Ok(()),
            b'u' => self.skip_unicodeescape(),
            _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
        }
    }

    // intentionally not inlined as its the minority case
    fn parse_string_escaped(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        self.skip_stringescape()?;

        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break Ok(JsonTokenType::EscString);
                }
                b'\\' => {
                    self.skip_stringescape()?;
                    continue;
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn parse_string(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break Ok(JsonTokenType::String);
                }
                b'\\' => {
                    break self.parse_string_escaped();
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn parse_number_neg(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0' => self.parse_number_zero(),
            b'1'..=b'9' => self.parse_number_one(),
            _ => Err(TokenizerErrorType::UnexpectedCharInNumericLiteral),
        }
    }

    #[inline]
    fn parse_number_zero(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Ok(JsonTokenType::Integer),
            b'.' => self.parse_number_dot(),
            b'e' | b'E' => self.parse_number_e(),
            _ => {
                self.pos -= 1;
                Ok(JsonTokenType::Integer)
            }
        }
    }

    #[inline]
    fn parse_number_one(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Integer);
                }
                b'0'..=b'9' => continue,
                b'.' => break self.parse_number_dot(),
                b'e' | b'E' => break self.parse_number_e(),
                _ => {
                    self.pos -= 1;
                    break Ok(JsonTokenType::Integer);
                }
            }
        }
    }

    #[inline]
    fn parse_number_dot(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0'..=b'9' => self.parse_number_dot0(),
            _ => Err(TokenizerErrorType::UnexpectedCharInNumericLiteral),
        }
    }

    #[inline]
    fn parse_number_dot0(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Number);
                }
                b'0'..=b'9' => continue,
                b'e' | b'E' => break self.parse_number_e(),
                _ => {
                    self.pos -= 1;
                    break Ok(JsonTokenType::Number);
                }
            }
        }
    }

    #[inline]
    fn parse_number_e(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0'..=b'9' => self.parse_number_e0(),
            b'+' | b'-' => match self.read_or_null() {
                b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
                b'0'..=b'9' => self.parse_number_e0(),
                _ => Err(TokenizerErrorType::UnexpectedCharInExponentLiteral),
            },
            _ => Err(TokenizerErrorType::UnexpectedCharInExponentLiteral),
        }
    }

    #[inline]
    fn parse_number_e0(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Number);
                }
                b'0'..=b'9' => continue,
                _ => {
                    self.pos -= 1;
                    break Ok(JsonTokenType::Number);
                }
            }
        }
    }

    #[inline]
    fn parse_true(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'r' | b'R' => match self.read_or_null() {
                b'u' | b'U' => match self.read_or_null() {
                    b'e' | b'E' => Ok(JsonTokenType::True),
                    _ => Err(TokenizerErrorType::UnexpectedCharInTrueLiteral),
                },
                _ => Err(TokenizerErrorType::UnexpectedCharInTrueLiteral),
            },
            _ => Err(TokenizerErrorType::UnexpectedCharInTrueLiteral),
        }
    }

    #[inline]
    fn parse_false(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'a' | b'A' => match self.read_or_null() {
                b'l' | b'L' => match self.read_or_null() {
                    b's' | b'S' => match self.read_or_null() {
                        b'e' | b'E' => Ok(JsonTokenType::False),
                        _ => Err(TokenizerErrorType::UnexpectedCharInFalseLiteral),
                    },
                    _ => Err(TokenizerErrorType::UnexpectedCharInFalseLiteral),
                },
                _ => Err(TokenizerErrorType::UnexpectedCharInFalseLiteral),
            },
            _ => Err(TokenizerErrorType::UnexpectedCharInFalseLiteral),
        }
    }

    #[inline]
    fn parse_null(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'u' | b'U' => match self.read_or_null() {
                b'l' | b'L' => match self.read_or_null() {
                    b'l' | b'L' => Ok(JsonTokenType::Null),
                    _ => Err(TokenizerErrorType::UnexpectedCharInNullLiteral),
                },
                _ => Err(TokenizerErrorType::UnexpectedCharInNullLiteral),
            },
            _ => Err(TokenizerErrorType::UnexpectedCharInNullLiteral),
        }
    }

    #[inline]
    fn skip_whitespace(&mut self) -> TokenizerResult<()> {
        loop {
            let c = self.peek_or_null();
            match c {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    self.eat();
                    continue;
                }
                _ => break Ok(()),
            }
        }
    }

    #[inline]
    fn parse_value(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Ok(JsonTokenType::End),
            b'{' => Ok(JsonTokenType::ObjectStart),
            b'}' => Ok(JsonTokenType::ObjectEnd),
            b':' => Ok(JsonTokenType::ObjectKeyDelim),
            b'[' => Ok(JsonTokenType::ArrayStart),
            b']' => Ok(JsonTokenType::ArrayEnd),
            b',' => Ok(JsonTokenType::ListDelim),
            b'"' => self.parse_string(),
            b'-' => self.parse_number_neg(),
            b'0' => self.parse_number_zero(),
            b'1'..=b'9' => self.parse_number_one(),
            b't' | b'T' => self.parse_true(),
            b'f' | b'F' => self.parse_false(),
            b'n' | b'N' => self.parse_null(),

            _ => Err(TokenizerErrorType::UnexpectedBeginChar),
        }
    }

    #[inline]
    pub fn step(&mut self) -> TokenizerResult<JsonToken> {
        self.skip_whitespace()?;
        let start_pos = self.pos;
        match self.parse_value() {
            Ok(token_type) => {
                let value = &self.input[start_pos..self.pos];
                Ok(JsonToken { token_type, value })
            }
            Err(e) => Err(TokenizerError {
                error_type: e,
                pos: self.pos,
            }),
        }
    }
}
