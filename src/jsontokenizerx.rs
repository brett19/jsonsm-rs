use crate::{
    bytesiterator::BytesIterator, jsontokenizer::TokenizerErrorType,
    jsontokenizer_token::JsonTokenType,
};

type TokenizerFnResult = std::result::Result<(), TokenizerErrorType>;
type TokenizerParseResult = std::result::Result<JsonTokenType, TokenizerErrorType>;

pub struct JsonTokenizerX<'a> {
    iter: BytesIterator<'a>,
}

impl<'a> JsonTokenizerX<'a> {
    #[inline]
    pub fn new(input: &'a [u8]) -> JsonTokenizerX<'a> {
        JsonTokenizerX {
            iter: BytesIterator::new(input),
        }
    }

    #[inline]
    fn skip_true(&mut self) -> TokenizerFnResult {
        match self.iter.read_multi::<3>() {
            Ok(_) => Ok(()),
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    fn skip_false(&mut self) -> TokenizerFnResult {
        match self.iter.read_multi::<4>() {
            Ok(_) => Ok(()),
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    fn skip_null(&mut self) -> TokenizerFnResult {
        match self.iter.read_multi::<3>() {
            Ok(_) => Ok(()),
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    fn skip_string(&mut self) -> TokenizerFnResult {
        match self.iter.skip_until_and_get(|c| match c {
            b'"' | b'\\' => true,
            _ => false,
        }) {
            Ok(c) => match c {
                b'\\' => {}
                _ => return Ok(()),
            },
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }

        let mut is_escaped = true;
        match self.iter.skip_until_and_get(|c| {
            if is_escaped {
                is_escaped = false;
                false
            } else {
                match c {
                    b'"' => true,
                    b'\\' => {
                        is_escaped = true;
                        false
                    }
                    _ => false,
                }
            }
        }) {
            Ok(_) => Ok(()),
            Err(_) => Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    fn skip_number(&mut self) -> TokenizerFnResult {
        match self.iter.skip_until(|c| match c {
            b'0'..=b'9' => false,
            b'.' => false,
            b'e' | b'E' => false,
            b'+' | b'-' => false,
            _ => true,
        }) {
            Ok(_) => Ok(()),
            Err(_) => Ok(()),
        }
    }

    #[inline]
    fn skip_out_of_object_or_array(&mut self) -> TokenizerFnResult {
        let mut depth = 1;

        loop {
            let c = match self.iter.skip_until_and_get(|c| match c {
                b'{' | b'[' => true,
                b'}' | b']' => true,
                b'"' => true,
                _ => false,
            }) {
                Ok(c) => c,
                Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
            };

            match c {
                b'{' | b'[' => {
                    depth += 1;
                }
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                b'"' => match self.skip_string() {
                    Ok(_) => {}
                    Err(e) => return Err(e),
                },
                _ => {}
            }
        }
    }

    #[inline]
    pub fn skip_over_value(&mut self) -> TokenizerFnResult {
        match self.iter.skip_until_and_get(|c| !c.is_ascii_whitespace()) {
            Ok(c) => match c {
                b'{' | b'[' => return self.skip_out_of_object_or_array(),
                b'"' => return self.skip_string(),
                b'-' | b'0'..=b'9' => return self.skip_number(),
                b't' | b'T' => return self.skip_true(),
                b'f' | b'F' => return self.skip_false(),
                b'n' | b'N' => return self.skip_null(),
                _ => return Err(TokenizerErrorType::UnexpectedBeginChar),
            },
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    pub fn step(&mut self) -> TokenizerParseResult {
        match self.iter.skip_until(|c| !c.is_ascii_whitespace()) {
            Ok(_) => Ok(JsonTokenType::Null),
            Err(_) => Ok(JsonTokenType::End),
        }
    }
}
