use crate::jsontokenizer_token::JsonToken;

type TokenizerResult<T> = std::result::Result<T, TokenizerError>;

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
    pub(crate) input: &'a [u8],
    pub(crate) pos: usize,
}

impl<'a> JsonTokenizer<'a> {
    pub fn new(input: &'a [u8]) -> JsonTokenizer<'a> {
        JsonTokenizer { input, pos: 0 }
    }

    #[inline]
    pub(crate) fn read_or_null(&mut self) -> u8 {
        if self.pos >= self.input.len() {
            return 0;
        }

        let c = unsafe { *self.input.get_unchecked(self.pos) };
        self.pos += 1;
        c
    }

    #[inline]
    pub(crate) fn rewind(&mut self) {
        self.pos -= 1;
    }

    #[inline]
    pub fn step<'b>(&'b mut self) -> TokenizerResult<JsonToken<'a>> {
        match self.parse_whitespace() {
            Ok(_) => {}
            Err(e) => {
                return Err(TokenizerError {
                    error_type: e,
                    pos: self.pos,
                })
            }
        }

        let start_pos = self.pos;
        match self.parse_token() {
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

    #[inline]
    pub fn skip_value(&mut self) -> TokenizerResult<()> {
        match self.skip_over_value() {
            Ok(_) => Ok(()),
            Err(e) => Err(TokenizerError {
                error_type: e,
                pos: self.pos,
            }),
        }
    }

    #[inline]
    pub fn leave_value(&mut self) -> TokenizerResult<()> {
        match self.skip_out_of_object_or_array() {
            Ok(_) => Ok(()),
            Err(e) => Err(TokenizerError {
                error_type: e,
                pos: self.pos,
            }),
        }
    }
}
