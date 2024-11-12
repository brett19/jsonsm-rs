#[derive(Debug, PartialEq)]
pub enum JsonTokenType {
    ObjectStart,
    ObjectEnd,
    ObjectKeyDelim,
    ArrayStart,
    ArrayEnd,
    ListDelim,
    String,
    EscString,
    Integer,
    Number,
    Null,
    True,
    False,
    End,
}

impl JsonTokenType {
    pub fn is_literal(&self) -> bool {
        return match self {
            JsonTokenType::ObjectStart => false,
            JsonTokenType::ObjectEnd => false,
            JsonTokenType::ObjectKeyDelim => false,
            JsonTokenType::ArrayStart => false,
            JsonTokenType::ArrayEnd => false,
            JsonTokenType::ListDelim => false,
            JsonTokenType::String => true,
            JsonTokenType::EscString => true,
            JsonTokenType::Integer => true,
            JsonTokenType::Number => true,
            JsonTokenType::Null => true,
            JsonTokenType::True => true,
            JsonTokenType::False => true,
            JsonTokenType::End => false,
        };
    }
}

#[derive(Debug)]
pub struct JsonToken<'a> {
    pub token_type: JsonTokenType,
    pub value: &'a [u8],
}
