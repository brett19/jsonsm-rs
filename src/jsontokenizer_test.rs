#[cfg(test)]
mod tests {
    use std::{fs, str::from_utf8};

    use crate::{
        jsontokenizer::JsonTokenizer, jsontokenizer_token::JsonTokenType,
        jsontokenizerx::JsonTokenizerX,
    };

    fn assert_step(
        tokenizer: &mut JsonTokenizer,
        expected_type: JsonTokenType,
        expected_value: &str,
    ) {
        let token = tokenizer.step().unwrap();
        assert_eq!(token.token_type, expected_type);
        assert_eq!(from_utf8(token.value).unwrap(), expected_value);
    }

    fn assert_single_value(input: &str, expected_type: JsonTokenType) {
        let mut t = JsonTokenizer::new(input.as_bytes());
        assert_step(&mut t, expected_type, input);
        assert_step(&mut t, JsonTokenType::End, "");

        let mut s = JsonTokenizerX::new(input.as_bytes());
        s.skip_over_value().unwrap();
        let end_token = s.step().unwrap();
        assert_eq!(end_token, JsonTokenType::End);
    }

    #[test]
    fn jsontokenizer_object() {
        let mut t = JsonTokenizer::new(
            r#"{
                "a": "5b47eb0936ff92a567a0307e",
                "b": false
            }"#
            .as_bytes(),
        );

        assert_step(&mut t, JsonTokenType::ObjectStart, "{");
        assert_step(&mut t, JsonTokenType::String, "\"a\"");
        assert_step(&mut t, JsonTokenType::ObjectKeyDelim, ":");
        assert_step(
            &mut t,
            JsonTokenType::String,
            "\"5b47eb0936ff92a567a0307e\"",
        );
        assert_step(&mut t, JsonTokenType::ListDelim, ",");
        assert_step(&mut t, JsonTokenType::String, "\"b\"");
        assert_step(&mut t, JsonTokenType::ObjectKeyDelim, ":");
        assert_step(&mut t, JsonTokenType::False, "false");
        assert_step(&mut t, JsonTokenType::ObjectEnd, "}");
        assert_step(&mut t, JsonTokenType::End, "");
    }

    #[test]
    fn jsontokenizer_array() {
        let mut t = JsonTokenizer::new(
            r#"[
                1,
                2999.22,
                null,
                "hello\u2932world"
            ]"#
            .as_bytes(),
        );

        assert_step(&mut t, JsonTokenType::ArrayStart, "[");
        assert_step(&mut t, JsonTokenType::Integer, "1");
        assert_step(&mut t, JsonTokenType::ListDelim, ",");
        assert_step(&mut t, JsonTokenType::Number, "2999.22");
        assert_step(&mut t, JsonTokenType::ListDelim, ",");
        assert_step(&mut t, JsonTokenType::Null, "null");
        assert_step(&mut t, JsonTokenType::ListDelim, ",");
        assert_step(&mut t, JsonTokenType::EscString, "\"hello\\u2932world\"");
        assert_step(&mut t, JsonTokenType::ArrayEnd, "]");
        assert_step(&mut t, JsonTokenType::End, "");
    }

    #[test]
    fn jsontokenizer_string() {
        assert_single_value(r#""hello world""#, JsonTokenType::String);
    }

    #[test]
    fn jsontokenizer_escstring() {
        assert_single_value(r#""l\nol""#, JsonTokenType::EscString);
        assert_single_value(r#""l\u2321ol""#, JsonTokenType::EscString);
    }

    #[test]
    fn jsontokenizer_integer() {
        assert_single_value("0", JsonTokenType::Integer);
        assert_single_value("123", JsonTokenType::Integer);
        assert_single_value("4565464651846548", JsonTokenType::Integer);
        assert_single_value("-0", JsonTokenType::Integer);
        assert_single_value("-123", JsonTokenType::Integer);
        assert_single_value("-4565464651846548", JsonTokenType::Integer);
    }

    #[test]
    fn jsontokenizer_number() {
        assert_single_value("0.1", JsonTokenType::Number);
        assert_single_value("1999.1", JsonTokenType::Number);
        assert_single_value("14.29438383", JsonTokenType::Number);
        assert_single_value("1.0E+2", JsonTokenType::Number);
        assert_single_value("1.9e+22", JsonTokenType::Number);
        assert_single_value("-0.1", JsonTokenType::Number);
        assert_single_value("-1999.1", JsonTokenType::Number);
        assert_single_value("-14.29438383", JsonTokenType::Number);
        assert_single_value("-1.0E+2", JsonTokenType::Number);
        assert_single_value("-1.9e+22", JsonTokenType::Number);
    }

    #[test]
    fn jsontokenizer_bool() {
        assert_single_value("true", JsonTokenType::True);
        assert_single_value("TRUE", JsonTokenType::True);
        assert_single_value("tRuE", JsonTokenType::True);
        assert_single_value("TrUe", JsonTokenType::True);

        assert_single_value("false", JsonTokenType::False);
        assert_single_value("FALSE", JsonTokenType::False);
        assert_single_value("fAlSe", JsonTokenType::False);
        assert_single_value("FaLsE", JsonTokenType::False);
    }

    #[test]
    fn jsontokenizer_null() {
        assert_single_value("null", JsonTokenType::Null);
        assert_single_value("NULL", JsonTokenType::Null);
        assert_single_value("nUlL", JsonTokenType::Null);
        assert_single_value("NuLl", JsonTokenType::Null);
    }

    #[test]
    fn jsontokenizer_ends_forever() {
        let mut t = JsonTokenizer::new(r#""hello world""#.as_bytes());
        assert_step(&mut t, JsonTokenType::String, "\"hello world\"");
        assert_step(&mut t, JsonTokenType::End, "");
        assert_step(&mut t, JsonTokenType::End, "");
    }

    fn jsontokenizer_testdata(path: &str) {
        let testdata = fs::read_to_string(path).unwrap();
        let json_bytes = testdata.as_bytes();

        let mut t = JsonTokenizer::new(json_bytes);
        loop {
            let token = t.step().unwrap();
            if token.token_type == JsonTokenType::End {
                break;
            }
        }

        t = JsonTokenizer::new(json_bytes);
        t.skip_value().unwrap();
        let token = t.step().unwrap();
        if token.token_type != JsonTokenType::End {
            panic!("skip over did not lead to end of document");
        }
    }

    #[test]
    fn jsontokenizer_testdata_people() {
        jsontokenizer_testdata("testdata/people.json");
    }

    #[test]
    fn jsontokenizer_testdata_bigvector() {
        jsontokenizer_testdata("testdata/bigvector.json");
    }
}
