use crate::jsontokenizer::{JsonTokenizer, TokenizerErrorType};

type TokenizerFnResult = std::result::Result<usize, (TokenizerErrorType, usize)>;

#[inline]
fn skip_string(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let mut is_escaped = false;
    match data[pos..].iter().skip(1).position(|&c| {
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
        Some(end_pos) => Ok(pos + 1 + end_pos + 1),
        None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
    }
}

#[inline]
fn skip_number(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    match data[pos..].iter().skip(1).position(|&c| match c {
        b'0'..=b'9' => false,
        b'.' => false,
        b'e' | b'E' => false,
        b'+' | b'-' => false,
        _ => true,
    }) {
        Some(end_pos) => Ok(pos + 1 + end_pos),
        None => Ok(data.len()),
    }
}

#[inline]
fn skip_true(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let mut i = data[pos..].iter();

    i.next();
    match i.next() {
        Some(b'r') | Some(b'R') => match i.next() {
            Some(b'u') | Some(b'U') => match i.next() {
                Some(b'e') | Some(b'E') => Ok(pos + 4),
                Some(_) => Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos)),
                None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
            },
            Some(_) => Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos)),
            None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
        },
        Some(_) => Err((TokenizerErrorType::UnexpectedCharInTrueLiteral, pos)),
        None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
    }
}

#[inline]
fn skip_false(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let mut i = data[pos..].iter();

    i.next();
    match i.next() {
        Some(b'a') | Some(b'A') => match i.next() {
            Some(b'l') | Some(b'L') => match i.next() {
                Some(b's') | Some(b'S') => match i.next() {
                    Some(b'e') | Some(b'E') => Ok(pos + 5),
                    Some(_) => Err((TokenizerErrorType::UnexpectedCharInFalseLiteral, pos)),
                    None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
                },
                Some(_) => Err((TokenizerErrorType::UnexpectedCharInFalseLiteral, pos)),
                None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
            },
            Some(_) => Err((TokenizerErrorType::UnexpectedCharInFalseLiteral, pos)),
            None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
        },
        Some(_) => Err((TokenizerErrorType::UnexpectedCharInFalseLiteral, pos)),
        None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
    }
}

#[inline]
fn skip_null(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let mut i = data[pos..].iter();

    i.next();
    match i.next() {
        Some(b'u') | Some(b'U') => match i.next() {
            Some(b'l') | Some(b'L') => match i.next() {
                Some(b'l') | Some(b'L') => Ok(pos + 4),
                Some(_) => Err((TokenizerErrorType::UnexpectedCharInNullLiteral, pos)),
                None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
            },
            Some(_) => Err((TokenizerErrorType::UnexpectedCharInNullLiteral, pos)),
            None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
        },
        Some(_) => Err((TokenizerErrorType::UnexpectedCharInNullLiteral, pos)),
        None => Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
    }
}

#[inline]
fn skip_out_of_object_or_array(data: &[u8], pos: usize) -> TokenizerFnResult {
    let mut depth = 1;
    let mut pos = pos;
    while pos < data.len() {
        pos = match data[pos..].iter().position(|&c| match c {
            b' ' | b'\t' | b'\n' | b'\r' => false,
            b':' | b',' => false,
            _ => true,
        }) {
            Some(end_pos) => pos + end_pos,
            None => return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
        };

        match data[pos] {
            b'{' | b'[' => {
                depth += 1;
                pos += 1;
            }
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(pos + 1);
                }
                pos += 1;
            }

            b'"' => pos = skip_string(data, pos)?,
            b'-' | b'0'..=b'9' => pos = skip_number(data, pos)?,
            b't' | b'T' => pos = skip_true(data, pos)?,
            b'f' | b'F' => pos = skip_false(data, pos)?,
            b'n' | b'N' => pos = skip_null(data, pos)?,
            _ => return Err((TokenizerErrorType::UnexpectedBeginChar, pos)),
        }
    }

    Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()))
}

#[inline]
fn skip_over_value(data: &[u8], pos: usize) -> TokenizerFnResult {
    if pos >= data.len() {
        return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len()));
    }

    let pos = match data[pos..].iter().position(|&c| match c {
        b' ' | b'\t' | b'\n' | b'\r' => false,
        _ => true,
    }) {
        Some(pos) => pos,
        None => return Err((TokenizerErrorType::UnexpectedEndOfInput, data.len())),
    };

    match data[pos] {
        b'{' | b'[' => return skip_out_of_object_or_array(data, pos + 1),
        b'"' => return skip_string(data, pos),
        b'-' | b'0'..=b'9' => return skip_number(data, pos),
        b't' | b'T' => return skip_true(data, pos),
        b'f' | b'F' => return skip_false(data, pos),
        b'n' | b'N' => return skip_null(data, pos),
        _ => return Err((TokenizerErrorType::UnexpectedBeginChar, pos)),
    }
}

type TokenizerSkipResult = std::result::Result<(), TokenizerErrorType>;

impl<'a> JsonTokenizer<'a> {
    #[inline]
    pub(crate) fn skip_out_of_object_or_array(&mut self) -> TokenizerSkipResult {
        match skip_out_of_object_or_array(self.input, self.pos) {
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
    pub(crate) fn skip_over_value(&mut self) -> TokenizerSkipResult {
        match skip_over_value(self.input, self.pos) {
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
