use crate::jsontokenizer::{JsonTokenizer, TokenizerErrorType};

type TokenizerSkipResult = std::result::Result<(), TokenizerErrorType>;

impl<'a> JsonTokenizer<'a> {
    #[inline]
    pub(crate) fn skip_string(&mut self) -> TokenizerSkipResult {
        let mut is_escaped = false;
        for pos in self.pos..self.input.len() {
            match self.input[pos] {
                b'"' => {
                    if !is_escaped {
                        self.pos = pos + 1;
                        return Ok(());
                    }
                    is_escaped = false;
                }
                b'\\' => {
                    is_escaped = true;
                }
                _ => {
                    is_escaped = false;
                }
            }
        }

        return Err(TokenizerErrorType::UnexpectedEndOfInput);
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
            let c = self.read_or_null();
            match c {
                b'\0' => {
                    break Ok(());
                }
                b' ' | b'\t' | b'\n' | b'\r' => {
                    continue;
                }
                _ => {
                    self.rewind();
                    break Ok(());
                }
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
