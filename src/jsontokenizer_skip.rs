use crate::jsontokenizer::{JsonTokenizer, TokenizerErrorType};

type TokenizerSkipResult = std::result::Result<(), TokenizerErrorType>;

impl<'a> JsonTokenizer<'a> {
    #[inline]
    pub(crate) fn skip_unicodeescape(&mut self) -> TokenizerSkipResult {
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
    pub(crate) fn skip_stringescape(&mut self) -> TokenizerSkipResult {
        match self.read_or_null() {
            b'\0' => Err(TokenizerErrorType::UnexpectedEndOfInput),
            b'b' | b'f' | b'n' | b'r' | b't' | b'"' | b'\\' | b'/' => Ok(()),
            b'u' => self.skip_unicodeescape(),
            _ => Err(TokenizerErrorType::UnexpectedEscapeCode),
        }
    }

    #[inline]
    pub(crate) fn skip_string(&mut self) -> TokenizerSkipResult {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b'"' => {
                    break Ok(());
                }
                b'\\' => match self.read_or_null() {
                    b'\0' => {
                        break Err(TokenizerErrorType::UnexpectedEndOfInput);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    #[inline]
    pub(crate) fn skip_number(&mut self) -> TokenizerSkipResult {
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Ok(());
                }
                b'0'..=b'9' => continue,
                b'.' => continue,
                b'e' | b'E' => continue,
                b'+' | b'-' => continue,
                _ => {
                    self.rewind();
                    break Ok(());
                }
            }
        }
    }

    #[inline]
    pub(crate) fn skip_true(&mut self) -> TokenizerSkipResult {
        self.parse_true()?;
        Ok(())
    }

    #[inline]
    pub(crate) fn skip_false(&mut self) -> TokenizerSkipResult {
        self.parse_false()?;
        Ok(())
    }

    #[inline]
    pub(crate) fn skip_null(&mut self) -> TokenizerSkipResult {
        self.parse_null()?;
        Ok(())
    }

    #[inline]
    pub(crate) fn skip_whitespace(&mut self) -> TokenizerSkipResult {
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
    pub(crate) fn skip_object_or_array(&mut self) -> TokenizerSkipResult {
        let mut depth = 1;
        loop {
            match self.read_or_null() {
                b'\0' => {
                    break Err(TokenizerErrorType::UnexpectedEndOfInput);
                }
                b' ' | b'\t' | b'\n' | b'\r' => {}
                b'{' | b'[' => {
                    depth += 1;
                }
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        break Ok(());
                    }
                }
                b':' | b',' => {}
                b'"' => self.skip_string()?,
                b'-' | b'0'..=b'9' => self.skip_number()?,
                b't' | b'T' => self.skip_true()?,
                b'f' | b'F' => self.skip_false()?,
                b'n' | b'N' => self.skip_true()?,
                _ => {}
            }
        }
    }

    #[inline]
    pub(crate) fn skip_any_value(&mut self) -> TokenizerSkipResult {
        self.skip_whitespace()?;

        match self.read_or_null() {
            b'\0' => Ok(()),
            b'{' => self.skip_object_or_array(),
            b'}' => Err(TokenizerErrorType::UnexpectedBeginChar),
            b':' => self.skip_object_or_array(),
            b'[' => self.skip_object_or_array(),
            b']' => Err(TokenizerErrorType::UnexpectedBeginChar),
            b',' => self.skip_object_or_array(),
            b'"' => self.skip_string(),
            b'-' => self.skip_number(),
            b'0' => self.skip_number(),
            b'1'..=b'9' => self.skip_number(),
            b't' | b'T' => self.skip_true(),
            b'f' | b'F' => self.skip_false(),
            b'n' | b'N' => self.skip_null(),

            _ => Err(TokenizerErrorType::UnexpectedBeginChar),
        }
    }
}
