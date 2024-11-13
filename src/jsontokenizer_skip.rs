use crate::jsontokenizer::{JsonTokenizer, TokenizerErrorType};

type TokenizerFnResult = std::result::Result<usize, (TokenizerErrorType, usize)>;

#[inline]
fn skip_string(data: &[u8], pos: usize) -> TokenizerFnResult {
    let mut is_escaped = false;
    for pos in pos + 1..data.len() {
        if is_escaped {
            is_escaped = false;
        } else {
            match data[pos] {
                b'"' => {
                    return Ok(pos + 1);
                }
                b'\\' => {
                    is_escaped = true;
                }
                _ => {}
            }
        }
    }

    return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
}

#[inline]
fn skip_number(data: &[u8], pos: usize) -> TokenizerFnResult {
    for pos in pos + 1..data.len() {
        match data[pos] {
            b'0'..=b'9' => continue,
            b'.' => continue,
            b'e' | b'E' => continue,
            b'+' | b'-' => continue,
            _ => {
                return Ok(pos);
            }
        }
    }

    Ok(data.len())
}

#[inline]
fn skip_true(data: &[u8], pos: usize) -> TokenizerFnResult {
    let end_pos = pos + 4;
    if end_pos > data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let v = &data[pos..end_pos];

    // v[0] is checked before entry to this function

    match v[1] {
        b'r' | b'R' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 1));
        }
    }

    match v[2] {
        b'u' | b'U' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 2));
        }
    }

    match v[3] {
        b'e' | b'E' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 3));
        }
    }

    Ok(pos + 4)
}

#[inline]
fn skip_false(data: &[u8], pos: usize) -> TokenizerFnResult {
    let end_pos = pos + 5;
    if end_pos > data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let v = &data[pos..end_pos];

    // v[0] is checked before entry to this function

    match v[1] {
        b'a' | b'A' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 1));
        }
    }

    match v[2] {
        b'l' | b'L' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 2));
        }
    }

    match v[3] {
        b's' | b'S' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 3));
        }
    }

    match v[4] {
        b'e' | b'E' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 4));
        }
    }

    Ok(pos + 5)
}

#[inline]
fn skip_null(data: &[u8], pos: usize) -> TokenizerFnResult {
    let end_pos = pos + 4;
    if end_pos > data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let v = &data[pos..end_pos];

    // v[0] is checked before entry to this function

    match v[1] {
        b'u' | b'U' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 1));
        }
    }

    match v[2] {
        b'l' | b'L' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 2));
        }
    }

    match v[3] {
        b'l' | b'L' => {}
        _ => {
            return Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos + 3));
        }
    }

    Ok(end_pos)
}

#[inline]
fn skip_whitespace(data: &[u8], pos: usize) -> TokenizerFnResult {
    for pos in pos..data.len() {
        match data[pos] {
            b' ' | b'\t' | b'\n' | b'\r' => continue,
            _ => {
                return Ok(pos);
            }
        }
    }

    Ok(data.len())
}

#[inline]
fn skip_object_or_array(data: &[u8], pos: usize, initial_depth: u32) -> TokenizerFnResult {
    let mut depth = initial_depth;
    let mut pos = pos;
    while pos < data.len() {
        match data[pos] {
            b' ' | b'\t' | b'\n' | b'\r' => {
                pos += 1;
                continue;
            }
            b':' | b',' => pos += 1,
            b'{' | b'[' => {
                pos += 1;
                depth += 1;
            }
            b'}' | b']' => {
                pos += 1;
                depth -= 1;
            }
            b'"' => pos = skip_string(data, pos)?,
            b'-' | b'0'..=b'9' => pos = skip_number(data, pos)?,
            b't' | b'T' => pos = skip_true(data, pos)?,
            b'f' | b'F' => pos = skip_false(data, pos)?,
            b'n' | b'N' => pos = skip_null(data, pos)?,
            _ => return Err((TokenizerErrorType::UnexpectedBeginChar, pos)),
        }

        if depth == 0 {
            return Ok(pos);
        }
    }

    Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()))
}

type TokenizerSkipResult = std::result::Result<(), TokenizerErrorType>;

impl<'a> JsonTokenizer<'a> {
    #[inline]
    pub(crate) fn skip_string(&mut self) -> TokenizerSkipResult {
        match skip_string(self.input, self.pos - 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_number(&mut self) -> TokenizerSkipResult {
        match skip_number(self.input, self.pos - 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_true(&mut self) -> TokenizerSkipResult {
        match skip_true(self.input, self.pos - 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_false(&mut self) -> TokenizerSkipResult {
        match skip_false(self.input, self.pos - 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_null(&mut self) -> TokenizerSkipResult {
        match skip_null(self.input, self.pos - 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_object_or_array(&mut self) -> TokenizerSkipResult {
        match skip_object_or_array(self.input, self.pos, 1) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_whitespace(&mut self) -> TokenizerSkipResult {
        match skip_whitespace(self.input, self.pos) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }

    #[inline]
    pub(crate) fn skip_any_value(&mut self) -> TokenizerSkipResult {
        match skip_object_or_array(self.input, self.pos, 0) {
            Ok(new_pos) => {
                self.pos = new_pos;
                Ok(())
            }
            Err((e, new_pos)) => {
                self.pos = new_pos;
                Err(e)
            }
        }
    }
}
