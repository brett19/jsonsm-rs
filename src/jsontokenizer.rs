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
    fn skip_string_escape(&mut self) -> TokenizerInternalResult<()> {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'b' | b'f' | b'n' | b'r' | b't' | b'"' | b'\\' | b'/' => Ok(()),
            b'u' => match self.read_or_null() {
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
            },
            _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
        }
    }

    #[inline]
    fn parse_string_escaped(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        'stringloop: loop {
            match self.read_or_null() {
                b'\0' => {
                    return Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break 'stringloop;
                }
                b'\\' => {
                    self.skip_string_escape()?;
                    continue 'stringloop;
                }
                _ => {}
            }
        }

        Ok(JsonTokenType::EscString)
    }

    #[inline]
    fn parse_string(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        'stringloop: loop {
            match self.read_or_null() {
                b'\0' => {
                    return Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break 'stringloop;
                }
                b'\\' => {
                    self.skip_string_escape()?;
                    return self.parse_string_escaped();
                }
                _ => {}
            }
        }

        Ok(JsonTokenType::String)
    }

    #[inline]
    fn parse_number_neg(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.peek_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'0' => self.eat(); self.parse_number_zero(),
            b'1'..=b'9' => self.parse_number_one(),
            _ => Err(TokenizerErrorType::UnexpectedCharInNumericLiteral),
        }
    }

    #[inline]
    fn parse_number_zero(&mut self) -> TokenizerInternalResult<JsonTokenType> {
        match self.read_or_null() {
            b'\0' => Ok(JsonTokenType::Integer),
            b'.' => self.parse_number_dot(start_pos),
            b'e' | b'E' => self.parse_number_e(start_pos),
            _ => {
                self.pos -= 1;
                Ok(JsonTokenType::Integer)
            }
        }
    }

    #[inline]
    fn parse_number_one(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Integer,
                        value: &self.input[start_pos..self.pos],
                    });
                }
                b'0'..=b'9' => continue,
                b'.' => return self.parse_number_dot(start_pos),
                b'e' | b'E' => return self.parse_number_e(start_pos),
                _ => {
                    self.pos -= 1;
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Integer,
                        value: &self.input[start_pos..self.pos],
                    });
                }
            }
        }
    }

    fn parse_number_dot(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        match self.read_or_null() {
            b'\0' => return Err(TokenizerError::UnexpectedEndOfInput),
            b'0'..=b'9' => {}
            _ => return Err(TokenizerError::UnexpectedCharInNumericLiteral(start_pos)),
        }

        loop {
            match self.read_or_null() {
                b'\0' => {
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Number,
                        value: &self.input[start_pos..self.pos],
                    });
                }
                b'0'..=b'9' => continue,
                b'e' | b'E' => return self.parse_number_e(start_pos),
                _ => {
                    self.pos -= 1;
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Number,
                        value: &self.input[start_pos..self.pos],
                    });
                }
            }
        }
    }

    fn parse_number_e(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        match self.read_or_null() {
            b'\0' => return Err(TokenizerError::UnexpectedEndOfInput),
            b'0'..=b'9' => {}
            b'+' | b'-' => match self.read_or_null() {
                b'\0' => return Err(TokenizerError::UnexpectedEndOfInput),
                b'0'..=b'9' => {}
                _ => return Err(TokenizerError::UnexpectedCharInExponentLiteral(start_pos)),
            },
            _ => {
                return Err(TokenizerError::UnexpectedCharInExponentLiteral(start_pos));
            }
        }

        loop {
            match self.read_or_null() {
                b'\0' => {
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Number,
                        value: &self.input[start_pos..self.pos],
                    });
                }
                b'0'..=b'9' => continue,
                _ => {
                    self.pos -= 1;
                    return Ok(JsonToken {
                        token_type: JsonTokenType::Number,
                        value: &self.input[start_pos..self.pos],
                    });
                }
            }
        }
    }

    #[inline]
    fn parse_true(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        match self.read_or_null() {
            b'r' | b'R' => match self.read_or_null() {
                b'u' | b'U' => match self.read_or_null() {
                    b'e' | b'E' => Ok(JsonToken {
                        token_type: JsonTokenType::True,
                        value: &self.input[start_pos..self.pos],
                    }),
                    _ => Err(TokenizerError::UnexpectedCharInTrueLiteral(start_pos)),
                },
                _ => Err(TokenizerError::UnexpectedCharInTrueLiteral(start_pos)),
            },
            _ => Err(TokenizerError::UnexpectedCharInTrueLiteral(start_pos)),
        }
    }

    #[inline]
    fn parse_false(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        match self.read_or_null() {
            b'a' | b'A' => match self.read_or_null() {
                b'l' | b'L' => match self.read_or_null() {
                    b's' | b'S' => match self.read_or_null() {
                        b'e' | b'E' => Ok(JsonToken {
                            token_type: JsonTokenType::False,
                            value: &self.input[start_pos..self.pos],
                        }),
                        _ => Err(TokenizerError::UnexpectedCharInFalseLiteral(start_pos)),
                    },
                    _ => Err(TokenizerError::UnexpectedCharInFalseLiteral(start_pos)),
                },
                _ => Err(TokenizerError::UnexpectedCharInFalseLiteral(start_pos)),
            },
            _ => Err(TokenizerError::UnexpectedCharInFalseLiteral(start_pos)),
        }
    }

    #[inline]
    fn parse_null(&mut self, start_pos: usize) -> TokenizerResult<JsonToken> {
        match self.read_or_null() {
            b'u' | b'U' => match self.read_or_null() {
                b'l' | b'L' => match self.read_or_null() {
                    b'l' | b'L' => Ok(JsonToken {
                        token_type: JsonTokenType::Null,
                        value: &self.input[start_pos..self.pos],
                    }),
                    _ => Err(TokenizerError::UnexpectedCharInNullLiteral(start_pos)),
                },
                _ => Err(TokenizerError::UnexpectedCharInNullLiteral(start_pos)),
            },
            _ => Err(TokenizerError::UnexpectedCharInNullLiteral(start_pos)),
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
    fn parse_value(&mut self) -> TokenizerResult<JsonToken> {
        let start_pos = self.pos;


        match self.read_or_null() {
            b'\0' => Ok(JsonToken {
                token_type: JsonTokenType::End,
                value: &self.input[start_pos..self.pos],
            }),
            b'{' => Ok(JsonToken {
                token_type: JsonTokenType::ObjectStart,
                value: &self.input[start_pos..self.pos],
            }),
            b'}' => Ok(JsonToken {
                token_type: JsonTokenType::ObjectEnd,
                value: &self.input[start_pos..self.pos],
            }),
            b':' => Ok(JsonToken {
                token_type: JsonTokenType::ObjectKeyDelim,
                value: &self.input[start_pos..self.pos],
            }),
            b'[' => Ok(JsonToken {
                token_type: JsonTokenType::ArrayStart,
                value: &self.input[start_pos..self.pos],
            }),
            b']' => Ok(JsonToken {
                token_type: JsonTokenType::ArrayEnd,
                value: &self.input[start_pos..self.pos],
            }),
            b',' => Ok(JsonToken {
                token_type: JsonTokenType::ListDelim,
                value: &self.input[start_pos..self.pos],
            }),
            b'"' => self.parse_string(start_pos),
            b'-' => self.parse_number_neg(start_pos),
            b'0' => self.parse_number_zero(start_pos),
            b'1'..=b'9' => self.parse_number_one(start_pos),
            b't' | b'T' => self.parse_true(start_pos),
            b'f' | b'F' => self.parse_false(start_pos),
            b'n' | b'N' => self.parse_null(start_pos),

            _ => Err(TokenizerError::UnexpectedBeginChar(start_pos)),
        }
    }

    #[inline]
    pub fn step(&mut self) -> TokenizerResult<JsonToken> {
        self.skip_whitespace()?;
        let start_pos = self.pos;
        let res = self.parse_value()?;

    }
}
