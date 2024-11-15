use crate::{
    bytesiterator::BytesIterator,
    jsontokenizer::TokenizerErrorType,
    jsontokenizer_token::JsonTokenType,
    simdsearch_ops::{SimdSearch, SimdSearchDualExec, SimdSearchEq, SimdSearchNot},
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
        loop {
            match self
                .iter
                .skip_fast_until_and_get(&mut (), &mut SimdSearchEq::new(b'"').or_eq(b'\\'))
            {
                Ok(c) => match c {
                    b'\\' => self.iter.skip(1),
                    _ => return Ok(()),
                },
                Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
            }
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
    fn skip_out_of_object_or_array(
        &mut self,
        deep_byte: u8,
        shallow_byte: u8,
    ) -> TokenizerFnResult {
        struct DepthState {
            depth: u32,
        }
        let mut state = DepthState { depth: 1 };

        loop {
            match self.iter.skip_fast_until_and_get(
                &mut state,
                &mut SimdSearchDualExec::new(
                    SimdSearchEq::new(deep_byte),
                    SimdSearchEq::new(shallow_byte),
                    |s: &mut DepthState, dv, sv| {
                        s.depth += if dv { 1 } else { 0 };
                        s.depth -= if sv { 1 } else { 0 };
                        s.depth == 0
                    },
                )
                .or(SimdSearchEq::new(b'\\')),
            ) {
                Ok(c) => match c {
                    // if we hit a \\, we need to skip the next byte since we could
                    // be skipping over one of the depth control characters
                    b'\\' => self.iter.skip(1),
                    _ => return Ok(()),
                },
                Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
            };
        }
    }

    pub fn skip_out_of_object(&mut self) -> TokenizerFnResult {
        self.skip_out_of_object_or_array(b'{', b'}')
    }

    pub fn skip_out_of_array(&mut self) -> TokenizerFnResult {
        self.skip_out_of_object_or_array(b'[', b']')
    }

    #[inline]
    fn skip_whitespace_and_get(&mut self) -> Result<u8, TokenizerErrorType> {
        match self.iter.skip_fast_until_and_get(
            &mut (),
            &mut SimdSearchNot::new(
                SimdSearchEq::new(b'\t')
                    .or_eq(b'\n')
                    .or_eq(b'\x0C')
                    .or_eq(b'\r')
                    .or_eq(b' '),
            ),
        ) {
            Ok(c) => Ok(c),
            Err(_) => return Err(TokenizerErrorType::UnexpectedEndOfInput),
        }
    }

    #[inline]
    pub fn skip_over_value(&mut self) -> TokenizerFnResult {
        match self.skip_whitespace_and_get() {
            Ok(c) => match c {
                b'{' => return self.skip_out_of_object(),
                b'[' => return self.skip_out_of_array(),
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
        match self.skip_whitespace_and_get() {
            Ok(_) => Ok(JsonTokenType::Null),
            Err(_) => Ok(JsonTokenType::End),
        }
    }
}
