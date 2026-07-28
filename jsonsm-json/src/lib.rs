//! The JSON-array expression format for `jsonsm` ↔ [`jsonsm_ast`].
//!
//! Expressions are written as nested JSON arrays whose first element names the node type,
//! e.g.:
//!
//! ```json
//! ["and",
//!   ["equals", ["field", "name", "first"], ["value", "Brett"]],
//!   ["lessthan", ["field", "age"], ["value", 50]]
//! ]
//! ```
//!
//! This mirrors the format used by `gojsonsm`. Node types:
//!
//! - operands: `["value", X]`, `["field", <root?>, <key>…]`, `["func", name, arg…]`;
//! - logic: `["not", e]`, `["and", e…]`, `["or", e…]`;
//! - existence: `["exists", e]`, `["notexists", e]`;
//! - comparisons: `["equals"|"notequals"|"lessthan"|"lessequals"|"greaterthan"|"greaterequals", lhs, rhs]`;
//! - pattern match: `["like", lhs, pattern]` (pattern is `["value", "…"]` or `["regex", "…"]`);
//! - loops: `["anyin"|"everyin"|"anyeveryin", <var-id>, in, sub]`.
//!
//! In a `field`, an optional leading integer is the root variable id (a loop variable);
//! remaining elements are object keys. Constant roots are written `["value", true]` etc.

#![forbid(unsafe_code)]

use jsonsm_ast::{CompareOp, Expr, Field, Func, Literal, LoopType, PathComponent, VariableId};
use serde_json::Value;

/// An error encountered while parsing the JSON-array expression format.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("expected a JSON array expression")]
    NotAnArray,
    #[error("empty expression array")]
    Empty,
    #[error("expression type must be a string")]
    TypeNotString,
    #[error("unknown expression type: {0}")]
    UnknownType(String),
    #[error("`{0}` expression is malformed or has the wrong number of arguments")]
    Malformed(&'static str),
    #[error("invalid field path")]
    BadFieldPath,
    #[error(
        "unsupported value literal: array/object literals are not supported (container \
             comparison is byte-exact, so a literal container could not be compared \
             reliably); only scalars are"
    )]
    UnsupportedValue,
    #[error("unsupported expression type: {0}")]
    Unsupported(&'static str),
}

/// Parse an expression from JSON bytes.
pub fn parse(input: &[u8]) -> Result<Expr, ParseError> {
    let v: Value = serde_json::from_slice(input)?;
    parse_expr(&v)
}

/// Parse an expression from a JSON string.
pub fn parse_str(input: &str) -> Result<Expr, ParseError> {
    let v: Value = serde_json::from_str(input)?;
    parse_expr(&v)
}

/// An error from the one-call [`compile`]/[`compile_str`] convenience: either parsing the
/// JSON-array expression or compiling the resulting AST.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Compile(#[from] jsonsm::compile::CompileError),
}

/// Parse a JSON-array expression and compile it into a ready-to-run
/// [`MatchDef`](jsonsm::compile::MatchDef) in one call, using `collation` (e.g.
/// [`DefaultCollation`](jsonsm::collation::DefaultCollation)) and capturing the fields named
/// by `projection` (pass `&Projection::new()` to capture nothing).
///
/// ```
/// use jsonsm::collation::DefaultCollation;
/// use jsonsm::compile::Projection;
/// use jsonsm::matcher::FastMatcher;
///
/// let def = jsonsm_json::compile(
///     br#"["lessthan", ["field", "age"], ["value", 50]]"#,
///     &Projection::new().field(["name"]),
///     &DefaultCollation,
/// )
/// .unwrap();
/// let mut m = FastMatcher::new(&def);
/// let out = m.matches(br#"{"name": "Brett", "age": 30}"#).unwrap();
/// assert!(out.matched());
/// assert_eq!(
///     out.projected(0).unwrap().as_str().unwrap().to_decoded_bytes().as_ref(),
///     b"Brett",
/// );
/// assert!(!m.matches(br#"{"age": 80}"#).unwrap().matched());
/// ```
pub fn compile<C: jsonsm::collation::Collation>(
    input: &[u8],
    projection: &jsonsm::compile::Projection,
    collation: &C,
) -> Result<jsonsm::compile::MatchDef, BuildError> {
    let expr = parse(input)?;
    Ok(jsonsm::compile::compile(
        std::slice::from_ref(&expr),
        projection,
        collation,
    )?)
}

/// Like [`compile`], but from a `&str`.
pub fn compile_str<C: jsonsm::collation::Collation>(
    input: &str,
    projection: &jsonsm::compile::Projection,
    collation: &C,
) -> Result<jsonsm::compile::MatchDef, BuildError> {
    let expr = parse_str(input)?;
    Ok(jsonsm::compile::compile(
        std::slice::from_ref(&expr),
        projection,
        collation,
    )?)
}

fn parse_expr(v: &Value) -> Result<Expr, ParseError> {
    let arr = v.as_array().ok_or(ParseError::NotAnArray)?;
    let ty = arr
        .first()
        .ok_or(ParseError::Empty)?
        .as_str()
        .ok_or(ParseError::TypeNotString)?;

    match ty {
        "value" => Ok(Expr::Value(parse_literal(arg(arr, 1, "value")?)?)),
        "field" => parse_field(arr),
        "func" => parse_func(arr),
        "not" => Ok(Expr::Not(boxed(arg(arr, 1, "not")?)?)),
        "exists" => Ok(Expr::Exists(boxed(arg(arr, 1, "exists")?)?)),
        "notexists" => Ok(Expr::NotExists(boxed(arg(arr, 1, "notexists")?)?)),
        "or" => Ok(Expr::Or(parse_list(&arr[1..])?)),
        "and" => Ok(Expr::And(parse_list(&arr[1..])?)),
        "anyin" => parse_loop(arr, LoopType::Any),
        "everyin" => parse_loop(arr, LoopType::Every),
        "anyeveryin" => parse_loop(arr, LoopType::AnyEvery),
        "equals" => parse_cmp(arr, CompareOp::Equals),
        "notequals" => parse_cmp(arr, CompareOp::NotEquals),
        "lessthan" => parse_cmp(arr, CompareOp::LessThan),
        "lessequals" => parse_cmp(arr, CompareOp::LessEquals),
        "greaterthan" => parse_cmp(arr, CompareOp::GreaterThan),
        "greaterequals" => parse_cmp(arr, CompareOp::GreaterEquals),
        "like" => Ok(Expr::Matches {
            lhs: boxed(arg(arr, 1, "like")?)?,
            pattern: boxed(arg(arr, 2, "like")?)?,
        }),
        // A regex operand is just a string pattern in this AST; the enclosing `like`
        // gives it match semantics.
        "regex" => {
            let p = arg(arr, 1, "regex")?
                .as_str()
                .ok_or(ParseError::Malformed("regex"))?;
            Ok(Expr::Value(Literal::String(p.to_owned())))
        }
        "true" => Ok(Expr::True),
        "false" => Ok(Expr::False),
        // gojsonsm has a `["time", "..."]` node; date comparison here is the `DATE()`
        // function instead (an explicit string -> epoch conversion, no implicit coercion),
        // so this node is deliberately not supported.
        "time" => Err(ParseError::Unsupported(
            "time (use the DATE() function instead)",
        )),
        other => Err(ParseError::UnknownType(other.to_owned())),
    }
}

fn arg<'a>(arr: &'a [Value], i: usize, name: &'static str) -> Result<&'a Value, ParseError> {
    arr.get(i).ok_or(ParseError::Malformed(name))
}

fn boxed(v: &Value) -> Result<Box<Expr>, ParseError> {
    Ok(Box::new(parse_expr(v)?))
}

fn parse_list(items: &[Value]) -> Result<Vec<Expr>, ParseError> {
    items.iter().map(parse_expr).collect()
}

fn parse_field(arr: &[Value]) -> Result<Expr, ParseError> {
    let mut idx = 1;
    let mut root: VariableId = jsonsm_ast::ROOT_VAR;
    if let Some(Value::Number(n)) = arr.get(1) {
        root = n
            .as_u64()
            .and_then(|r| u32::try_from(r).ok())
            .ok_or(ParseError::BadFieldPath)?;
        idx = 2;
    }
    let mut path = Vec::with_capacity(arr.len().saturating_sub(idx));
    for v in &arr[idx..] {
        let key = v.as_str().ok_or(ParseError::BadFieldPath)?;
        path.push(parse_path_component(key));
    }
    Ok(Expr::Field(Field { root, path }))
}

/// One path segment. `"[N]"` addresses array element `N` (the spelling gojsonsm uses, whose
/// field paths are plain strings); anything else is an object key. A key that genuinely
/// contains brackets is still reachable — only a well-formed all-digit `[N]` is an index.
fn parse_path_component(seg: &str) -> PathComponent {
    seg.strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse().ok())
        .map_or_else(|| PathComponent::Key(seg.to_owned()), PathComponent::Index)
}

fn parse_func(arr: &[Value]) -> Result<Expr, ParseError> {
    let name = arg(arr, 1, "func")?
        .as_str()
        .ok_or(ParseError::Malformed("func"))?
        .to_owned();
    let args = parse_list(&arr[2.min(arr.len())..])?;
    Ok(Expr::Func(Func { name, args }))
}

fn parse_loop(arr: &[Value], loop_type: LoopType) -> Result<Expr, ParseError> {
    let var = arg(arr, 1, "loop")?
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or(ParseError::Malformed("loop"))?;
    Ok(Expr::Loop {
        loop_type,
        var,
        in_expr: boxed(arg(arr, 2, "loop")?)?,
        sub_expr: boxed(arg(arr, 3, "loop")?)?,
    })
}

fn parse_cmp(arr: &[Value], op: CompareOp) -> Result<Expr, ParseError> {
    Ok(Expr::compare(
        op,
        parse_expr(arg(arr, 1, "comparison")?)?,
        parse_expr(arg(arr, 2, "comparison")?)?,
    ))
}

fn parse_literal(v: &Value) -> Result<Literal, ParseError> {
    Ok(match v {
        Value::Null => Literal::Null,
        Value::Bool(b) => Literal::Bool(*b),
        Value::String(s) => Literal::String(s.clone()),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Literal::Int(i)
            } else if let Some(u) = n.as_u64() {
                Literal::Uint(u)
            } else if let Some(f) = n.as_f64() {
                Literal::Float(f)
            } else {
                return Err(ParseError::UnsupportedValue);
            }
        }
        Value::Array(_) | Value::Object(_) => return Err(ParseError::UnsupportedValue),
    })
}

/// Serialize an expression back to the JSON-array [`Value`] form.
///
/// The inverse of parsing for every node the format supports. Array indices round-trip as
/// `"[N]"` segments, matching gojsonsm's string-only field paths. Currently total (never
/// `None`), but kept fallible for formats nodes added later may not represent.
pub fn to_value(expr: &Expr) -> Option<Value> {
    Some(match expr {
        Expr::True => Value::Array(vec!["true".into()]),
        Expr::False => Value::Array(vec!["false".into()]),
        Expr::Value(lit) => arr2("value", literal_to_value(lit)),
        Expr::Field(f) => field_to_value(f)?,
        Expr::Func(f) => {
            let mut items = vec![Value::from("func"), Value::from(f.name.clone())];
            for a in &f.args {
                items.push(to_value(a)?);
            }
            Value::Array(items)
        }
        Expr::Not(e) => arr2("not", to_value(e)?),
        Expr::Exists(e) => arr2("exists", to_value(e)?),
        Expr::NotExists(e) => arr2("notexists", to_value(e)?),
        Expr::And(es) => list_to_value("and", es)?,
        Expr::Or(es) => list_to_value("or", es)?,
        Expr::Compare { op, lhs, rhs } => Value::Array(vec![
            Value::from(cmp_name(*op)),
            to_value(lhs)?,
            to_value(rhs)?,
        ]),
        Expr::Matches { lhs, pattern } => {
            Value::Array(vec!["like".into(), to_value(lhs)?, to_value(pattern)?])
        }
        Expr::Loop {
            loop_type,
            var,
            in_expr,
            sub_expr,
        } => Value::Array(vec![
            Value::from(loop_name(*loop_type)),
            Value::from(*var),
            to_value(in_expr)?,
            to_value(sub_expr)?,
        ]),
    })
}

fn arr2(tag: &str, v: Value) -> Value {
    Value::Array(vec![Value::from(tag), v])
}

fn list_to_value(tag: &str, es: &[Expr]) -> Option<Value> {
    let mut items = vec![Value::from(tag)];
    for e in es {
        items.push(to_value(e)?);
    }
    Some(Value::Array(items))
}

fn field_to_value(f: &Field) -> Option<Value> {
    let mut items = vec![Value::from("field")];
    if f.root != jsonsm_ast::ROOT_VAR {
        items.push(Value::from(f.root));
    }
    for c in &f.path {
        match c {
            PathComponent::Key(k) => items.push(Value::from(k.clone())),
            PathComponent::Index(i) => items.push(Value::from(format!("[{i}]"))),
        }
    }
    Some(Value::Array(items))
}

fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Null => Value::Null,
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Int(i) => Value::from(*i),
        Literal::Uint(u) => Value::from(*u),
        Literal::Float(f) => serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number),
        Literal::String(s) => Value::from(s.clone()),
    }
}

fn cmp_name(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Equals => "equals",
        CompareOp::NotEquals => "notequals",
        CompareOp::LessThan => "lessthan",
        CompareOp::LessEquals => "lessequals",
        CompareOp::GreaterThan => "greaterthan",
        CompareOp::GreaterEquals => "greaterequals",
    }
}

fn loop_name(lt: LoopType) -> &'static str {
    match lt {
        LoopType::Any => "anyin",
        LoopType::Every => "everyin",
        LoopType::AnyEvery => "anyeveryin",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(k: &str) -> PathComponent {
        PathComponent::Key(k.to_owned())
    }

    #[test]
    fn array_index_path_segments() {
        // gojsonsm spells an array index as the path string "[N]"; it is parsed here into a typed
        // index and round-trip it back to the same spelling.
        let e = parse_str(r#"["equals", ["field", "a", "[1]", "b"], ["value", 9]]"#).unwrap();
        assert_eq!(
            e,
            Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field::root(vec![
                    key("a"),
                    PathComponent::Index(1),
                    key("b")
                ])),
                Expr::Value(Literal::Int(9))
            )
        );
        assert_eq!(
            to_value(&e).unwrap(),
            serde_json::json!(["equals", ["field", "a", "[1]", "b"], ["value", 9]])
        );

        // Only a well-formed all-digit `[N]` is an index; other bracketed text stays a key.
        for spelling in ["[]", "[x]", "[1", "1]", "[-1]", "[ 1]", "[01x]"] {
            let e = parse_str(&format!(r#"["field", "{spelling}"]"#)).unwrap();
            assert_eq!(
                e,
                Expr::Field(Field::root(vec![key(spelling)])),
                "{spelling} should stay an object key"
            );
        }

        // And it matches the array element end to end.
        let def = compile_str(
            r#"["equals", ["field", "a", "[1]"], ["value", 20]]"#,
            &jsonsm::compile::Projection::new(),
            &jsonsm::collation::DefaultCollation,
        )
        .unwrap();
        let mut m = jsonsm::matcher::FastMatcher::new(&def);
        assert!(m.matches(br#"{"a": [10, 20]}"#).unwrap().matched());
        assert!(!m.matches(br#"{"a": [20, 10]}"#).unwrap().matched());
    }

    #[test]
    fn parses_string_equals() {
        let e = parse_str(r#"["equals", ["field", "name"], ["value", "Daphne"]]"#).unwrap();
        assert_eq!(
            e,
            Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field::root(vec![key("name")])),
                Expr::Value(Literal::String("Daphne".into())),
            )
        );
    }

    #[test]
    fn parses_numeric_literals_by_kind() {
        assert_eq!(
            parse_str(r#"["value", 25]"#).unwrap(),
            Expr::Value(Literal::Int(25))
        );
        assert_eq!(
            parse_str(r#"["value", -40.26]"#).unwrap(),
            Expr::Value(Literal::Float(-40.26))
        );
        assert_eq!(
            parse_str(r#"["value", 18446744073709551615]"#).unwrap(),
            Expr::Value(Literal::Uint(u64::MAX))
        );
        assert_eq!(
            parse_str(r#"["value", true]"#).unwrap(),
            Expr::Value(Literal::Bool(true))
        );
        assert_eq!(
            parse_str(r#"["value", null]"#).unwrap(),
            Expr::Value(Literal::Null)
        );
    }

    #[test]
    fn parses_field_with_root_variable_and_nested_path() {
        // ["field", 1, "id"] -> root var 1, path ["id"]
        let e = parse_str(r#"["field", 1, "id"]"#).unwrap();
        assert_eq!(
            e,
            Expr::Field(Field {
                root: 1,
                path: vec![key("id")]
            })
        );
        // Multi-key path from the document root.
        let e = parse_str(r#"["field", "name", "first"]"#).unwrap();
        assert_eq!(e, Expr::Field(Field::root(vec![key("name"), key("first")])));
    }

    #[test]
    fn parses_loop_and_nested_field_reference() {
        // anyin over tags, matching element == "cillum"
        let e = parse_str(
            r#"["anyin", 1, ["field", "tags"],
                ["equals", ["field", 1], ["value", "cillum"]]]"#,
        )
        .unwrap();
        match e {
            Expr::Loop {
                loop_type: LoopType::Any,
                var: 1,
                in_expr,
                sub_expr,
            } => {
                assert_eq!(*in_expr, Expr::Field(Field::root(vec![key("tags")])));
                assert_eq!(
                    *sub_expr,
                    Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field {
                            root: 1,
                            path: vec![]
                        }),
                        Expr::Value(Literal::String("cillum".into())),
                    )
                );
            }
            other => panic!("expected anyin loop, got {other:?}"),
        }
    }

    #[test]
    fn parses_func_and_like_and_exists() {
        assert_eq!(
            parse_str(r#"["func", "mathRound", ["field", "latitude"]]"#).unwrap(),
            Expr::Func(Func {
                name: "mathRound".into(),
                args: vec![Expr::Field(Field::root(vec![key("latitude")]))],
            })
        );
        assert_eq!(
            parse_str(r#"["like", ["field", "x"], ["regex", "^a.*z$"]]"#).unwrap(),
            Expr::Matches {
                lhs: Box::new(Expr::Field(Field::root(vec![key("x")]))),
                pattern: Box::new(Expr::Value(Literal::String("^a.*z$".into()))),
            }
        );
        assert_eq!(
            parse_str(r#"["exists", ["field", "sometimes"]]"#).unwrap(),
            Expr::Exists(Box::new(Expr::Field(Field::root(vec![key("sometimes")]))))
        );
    }

    #[test]
    fn reports_useful_errors() {
        assert!(matches!(parse_str("not json{"), Err(ParseError::Json(_))));
        assert!(matches!(parse_str("42"), Err(ParseError::NotAnArray)));
        assert!(matches!(parse_str("[]"), Err(ParseError::Empty)));
        assert!(matches!(
            parse_str(r#"["nope"]"#),
            Err(ParseError::UnknownType(_))
        ));
        assert!(matches!(
            parse_str(r#"["equals", ["field","a"]]"#),
            Err(ParseError::Malformed("comparison"))
        ));
        assert!(matches!(
            parse_str(r#"["value", [1,2]]"#),
            Err(ParseError::UnsupportedValue)
        ));
        assert!(matches!(
            parse_str(r#"["time", "2020-01-01"]"#),
            Err(ParseError::Unsupported(msg)) if msg.starts_with("time")
        ));
    }

    #[test]
    fn round_trips_through_value_form() {
        let exprs = vec![
            Expr::Or(vec![
                Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field::root(vec![key("name"), key("first")])),
                    Expr::Value(Literal::String("Brett".into())),
                ),
                Expr::And(vec![
                    Expr::compare(
                        CompareOp::LessThan,
                        Expr::Field(Field::root(vec![key("age")])),
                        Expr::Value(Literal::Int(50)),
                    ),
                    Expr::compare(
                        CompareOp::GreaterEquals,
                        Expr::Field(Field::root(vec![key("score")])),
                        Expr::Value(Literal::Float(1.5)),
                    ),
                ]),
            ]),
            Expr::Not(Box::new(Expr::Exists(Box::new(Expr::Field(Field::root(
                vec![key("x")],
            )))))),
            Expr::Loop {
                loop_type: LoopType::AnyEvery,
                var: 2,
                in_expr: Box::new(Expr::Field(Field::root(vec![key("items")]))),
                sub_expr: Box::new(Expr::compare(
                    CompareOp::NotEquals,
                    Expr::Field(Field {
                        root: 2,
                        path: vec![],
                    }),
                    // Only values above i64::MAX stay Uint; smaller non-negative integers
                    // canonicalize to Int (they are numerically identical in JSON).
                    Expr::Value(Literal::Uint(u64::MAX)),
                )),
            },
            Expr::Matches {
                lhs: Box::new(Expr::Field(Field::root(vec![key("email")]))),
                pattern: Box::new(Expr::Value(Literal::String("@example\\.com$".into()))),
            },
        ];
        for e in exprs {
            let v = to_value(&e).expect("serializable");
            let back = parse_expr(&v).expect("re-parses");
            assert_eq!(back, e, "round-trip mismatch for {e:?}");
        }
    }
}
