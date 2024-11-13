use crate::{
    jsontokenizer::{JsonTokenizer, TokenizerErrorType},
    jsontokenizer_token::JsonTokenType,
};

type TokenizerSkipResult = std::result::Result<(), TokenizerErrorType>;
type TokenizerParseResult = std::result::Result<JsonTokenType, TokenizerErrorType>;

impl<'a> JsonTokenizer<'a> {
    #[inline]
    pub(crate) fn parse_unicodeescape(&mut self) -> TokenizerSkipResult {
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
    pub(crate) fn parse_stringescape(&mut self) -> TokenizerSkipResult {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'b' | b'f' | b'n' | b'r' | b't' | b'"' | b'\\' | b'/' => Ok(()),
            b'u' => self.parse_unicodeescape(),
            _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
        }
    }

    // intentionally not inlined as its the minority case
    fn parse_string_escaped(&mut self) -> TokenizerParseResult {
        self.parse_stringescape()?;

        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break Ok(JsonTokenType::EscString);
                }
                b'\\' => {
                    self.parse_stringescape()?;
                    continue;
                }
                _ => {}
            }
        }
    }

    #[inline]
    fn parse_string(&mut self) -> TokenizerParseResult {
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
    fn parse_number_neg(&mut self) -> TokenizerParseResult {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0' => self.parse_number_zero(),
            b'1'..=b'9' => self.parse_number_one(),
            _ => Err(TokenizerErrorType::UnexpectedCharInNumericLiteral),
        }
    }

    #[inline]
    fn parse_number_zero(&mut self) -> TokenizerParseResult {
        match self.read_or_null() {
            b'\0' => Ok(JsonTokenType::Integer),
            b'.' => self.parse_number_dot(),
            b'e' | b'E' => self.parse_number_e(),
            _ => {
                self.rewind();
                Ok(JsonTokenType::Integer)
            }
        }
    }

    #[inline]
    fn parse_number_one(&mut self) -> TokenizerParseResult {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Integer);
                }
                b'0'..=b'9' => continue,
                b'.' => break self.parse_number_dot(),
                b'e' | b'E' => break self.parse_number_e(),
                _ => {
                    self.rewind();
                    break Ok(JsonTokenType::Integer);
                }
            }
        }
    }

    #[inline]
    fn parse_number_dot(&mut self) -> TokenizerParseResult {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0'..=b'9' => self.parse_number_dot0(),
            _ => Err(TokenizerErrorType::UnexpectedCharInNumericLiteral),
        }
    }

    #[inline]
    fn parse_number_dot0(&mut self) -> TokenizerParseResult {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Number);
                }
                b'0'..=b'9' => continue,
                b'e' | b'E' => break self.parse_number_e(),
                _ => {
                    self.rewind();
                    break Ok(JsonTokenType::Number);
                }
            }
        }
    }

    #[inline]
    fn parse_number_e(&mut self) -> TokenizerParseResult {
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
    fn parse_number_e0(&mut self) -> TokenizerParseResult {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(JsonTokenType::Number);
                }
                b'0'..=b'9' => continue,
                _ => {
                    self.rewind();
                    break Ok(JsonTokenType::Number);
                }
            }
        }
    }

    #[inline]
    pub(crate) fn parse_true(&mut self) -> TokenizerParseResult {
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
    pub(crate) fn parse_false(&mut self) -> TokenizerParseResult {
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
    pub(crate) fn parse_whitespace(&mut self) -> TokenizerSkipResult {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => {}
                _ => break,
            }

            self.pos += 1;
        }

        Ok(())
    }

    #[inline]
    pub(crate) fn parse_null(&mut self) -> TokenizerParseResult {
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
    pub(crate) fn parse_token(&mut self) -> TokenizerParseResult {
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
}
