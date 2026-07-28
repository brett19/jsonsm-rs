//! The N1QL-ish string-grammar front-end for `jsonsm` ↔ [`jsonsm_ast`].
//!
//! Parses filter strings such as `age < 50 AND isActive = true` or
//! `REGEXP_CONTAINS(email, "@example\\.com$")` into a [`jsonsm_ast::Expr`], using an
//! LALRPOP-generated LR parser (`grammar.lalrpop`). The grammar mirrors gojsonsm's
//! `filterExprParser`: comparisons (`= == <> != < <= > >=`), `AND`/`OR`/`NOT` (also
//! `&& || !`), `IS [NOT] NULL`/`MISSING`, arithmetic (`+ - * / %`, unary `-`) lowered to
//! math functions, function calls, `EXISTS(field)`, and `REGEXP_CONTAINS(field, pat)`.
//! Field paths support `a.b`, `a[0]`, and backtick-quoted segments. Keywords are
//! case-insensitive. Array loops are written `ANY`/`EVERY`/`ANY AND EVERY <var> IN <array>
//! SATISFIES <predicate> END`; the loop variable is bound by name and resolved to the
//! AST's numeric variable id in a post-parse pass.

use jsonsm_ast::{Expr, Func, Literal, PathComponent, VariableId};

mod lexer;

lalrpop_util::lalrpop_mod!(grammar);

/// Parse-time context threaded through the grammar: allocates a fresh variable id per
/// loop and records its name so the post-parse resolution pass can bind field references.
pub(crate) struct ParseCtx {
    /// `names[id - 1]` is the source name of loop variable `id` (ids are 1-based).
    names: Vec<String>,
}

impl ParseCtx {
    fn new() -> Self {
        ParseCtx { names: Vec::new() }
    }

    /// Build a loop node, allocating a fresh variable id for `name`.
    pub(crate) fn loop_expr(
        &mut self,
        loop_type: jsonsm_ast::LoopType,
        name: String,
        in_expr: Expr,
        body: Expr,
    ) -> Expr {
        self.names.push(name);
        let var = self.names.len() as VariableId; // 1-based
        Expr::Loop {
            loop_type,
            var,
            in_expr: Box::new(in_expr),
            sub_expr: Box::new(body),
        }
    }
}

/// An error parsing a N1QL-ish filter string.
#[derive(Debug, thiserror::Error)]
#[error("N1QL parse error: {0}")]
pub struct ParseError(String);

/// Parse a filter string into an expression AST.
///
/// Rejects input nested deeper than [`MAX_EXPR_DEPTH`](jsonsm::compile::MAX_EXPR_DEPTH). The
/// LR parse itself is iterative, but the name-resolution pass below (and compilation
/// afterwards) recurses, so the depth is checked — iteratively — before either runs.
pub fn parse_str(input: &str) -> Result<Expr, ParseError> {
    let mut ctx = ParseCtx::new();
    let mut expr = grammar::FilterParser::new()
        .parse(&mut ctx, lexer::lex(input))
        .map_err(|e| ParseError(e.to_string()))?;
    if expr.exceeds_depth(jsonsm::compile::MAX_EXPR_DEPTH) {
        return Err(ParseError(format!(
            "expression is nested deeper than the {} level limit",
            jsonsm::compile::MAX_EXPR_DEPTH
        )));
    }
    resolve(&mut expr, &ctx.names, &mut Vec::new());
    Ok(expr)
}

/// Post-parse name resolution: rewrite each document-rooted field whose first path segment
/// names an in-scope loop variable into a reference rooted at that variable. Loop bodies
/// bind the loop variable; the `in` array is resolved in the enclosing scope.
///
/// This is a separate pass because the variable cannot be bound during the parse. LALRPOP is
/// bottom-up, so a loop's *body* is reduced before the rule that introduces the loop — at which
/// point the variable it binds does not exist yet. The grammar therefore allocates a fresh id
/// per loop and records `id -> name`, and the binding is applied here.
///
/// The scope stack is searched innermost-first, so a name bound by two nested loops resolves to
/// the inner one inside the inner body. Because a loop's `in` expression is resolved *before*
/// its own binding is pushed, `ANY x IN xs SATISFIES ANY x IN x.ys SATISFIES ... END END` reads
/// the outer `x` for the inner `in` and the inner `x` within the inner body.
fn resolve(e: &mut Expr, names: &[String], scope: &mut Vec<(String, VariableId)>) {
    match e {
        Expr::Field(f) => {
            if f.root == jsonsm_ast::ROOT_VAR {
                let bound = match f.path.first() {
                    Some(PathComponent::Key(seg0)) => scope
                        .iter()
                        .rev()
                        .find(|(n, _)| n == seg0)
                        .map(|&(_, id)| id),
                    _ => None,
                };
                if let Some(id) = bound {
                    f.root = id;
                    f.path.remove(0);
                }
            }
        }
        Expr::Func(func) => func.args.iter_mut().for_each(|a| resolve(a, names, scope)),
        Expr::Not(s) | Expr::Exists(s) | Expr::NotExists(s) => resolve(s, names, scope),
        Expr::And(v) | Expr::Or(v) => v.iter_mut().for_each(|x| resolve(x, names, scope)),
        Expr::Compare { lhs, rhs, .. } => {
            resolve(lhs, names, scope);
            resolve(rhs, names, scope);
        }
        Expr::Matches { lhs, pattern } => {
            resolve(lhs, names, scope);
            resolve(pattern, names, scope);
        }
        Expr::Loop {
            var,
            in_expr,
            sub_expr,
            ..
        } => {
            resolve(in_expr, names, scope); // enclosing scope
            let name = names[(*var - 1) as usize].clone();
            scope.push((name, *var));
            resolve(sub_expr, names, scope);
            scope.pop();
        }
        Expr::Value(_) | Expr::True | Expr::False => {}
    }
}

/// An error from the one-call [`compile_str`] convenience.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Compile(#[from] jsonsm::compile::CompileError),
}

/// Parse a filter string and compile it into a ready-to-run
/// [`MatchDef`](jsonsm::compile::MatchDef) in one call, capturing the fields named by
/// `projection` (pass `&Projection::new()` to capture nothing).
///
/// ```
/// use jsonsm::collation::DefaultCollation;
/// use jsonsm::compile::Projection;
/// use jsonsm::matcher::FastMatcher;
///
/// let projection = Projection::new().field(["name", "first"]);
/// let def =
///     jsonsm_n1ql::compile_str("age < 50 AND isActive = true", &projection, &DefaultCollation)
///         .unwrap();
/// let mut m = FastMatcher::new(&def);
/// let out = m.matches(br#"{"name": {"first": "Brett"}, "age": 30, "isActive": true}"#).unwrap();
/// assert!(out.matched());
/// assert_eq!(
///     out.projected(0).unwrap().as_str().unwrap().to_decoded_bytes().as_ref(),
///     b"Brett",
/// );
/// assert!(!m.matches(br#"{"age": 30, "isActive": false}"#).unwrap().matched());
/// ```
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

// ---- helpers invoked from the grammar actions -------------------------------------------

/// Flatten a left-associated `OR` chain into a single [`Expr::Or`].
pub(crate) fn or_join(l: Expr, r: Expr) -> Expr {
    match l {
        Expr::Or(mut v) => {
            v.push(r);
            Expr::Or(v)
        }
        other => Expr::Or(vec![other, r]),
    }
}

/// Flatten a left-associated `AND` chain into a single [`Expr::And`].
pub(crate) fn and_join(l: Expr, r: Expr) -> Expr {
    match l {
        Expr::And(mut v) => {
            v.push(r);
            Expr::And(v)
        }
        other => Expr::And(vec![other, r]),
    }
}

/// A bare operand used at boolean position: a boolean literal becomes the constant
/// `True`/`False`; anything else is left as-is (the compiler rejects non-boolean operands).
pub(crate) fn as_condition(e: Expr) -> Expr {
    match e {
        Expr::Value(Literal::Bool(true)) => Expr::True,
        Expr::Value(Literal::Bool(false)) => Expr::False,
        other => other,
    }
}

/// Negate an operand: fold numeric literals; otherwise wrap in `mathNegate`.
pub(crate) fn negate(e: Expr) -> Expr {
    match e {
        Expr::Value(Literal::Int(i)) => Expr::Value(Literal::Int(-i)),
        Expr::Value(Literal::Float(f)) => Expr::Value(Literal::Float(-f)),
        other => func("mathNegate", vec![other]),
    }
}

/// Build a function-call operand, mapping N1QL function names to the engine's internal
/// identifiers (e.g. `ABS` → `mathAbs`). Unknown names pass through unchanged.
pub(crate) fn func(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Func(Func {
        name: map_func_name(name),
        args,
    })
}

/// Parse a numeric literal token into an `Int`/`Uint`/`Float` value.
pub(crate) fn num_literal(s: &str) -> Expr {
    let lit = if s.bytes().any(|b| matches!(b, b'.' | b'e' | b'E')) {
        Literal::Float(s.parse().unwrap_or(f64::NAN))
    } else if let Ok(i) = s.parse::<i64>() {
        Literal::Int(i)
    } else if let Ok(u) = s.parse::<u64>() {
        Literal::Uint(u)
    } else {
        Literal::Float(s.parse().unwrap_or(f64::NAN))
    };
    Expr::Value(lit)
}

/// Decode a quoted string literal token (surrounding quotes stripped, escapes resolved).
pub(crate) fn string_literal(s: &str) -> String {
    let inner = &s[1..s.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                    out.push(ch);
                }
            }
            Some(other) => out.push(other), // \\  \"  \'  \/  and any other escaped char
            None => {}
        }
    }
    out
}

/// Strip the surrounding backticks from a quoted identifier token.
pub(crate) fn strip_backticks(s: &str) -> String {
    s[1..s.len() - 1].to_string()
}

/// Append an object-key segment to a field path.
pub(crate) fn append_key(mut path: Vec<PathComponent>, seg: PathComponent) -> Vec<PathComponent> {
    path.push(seg);
    path
}

/// Append an array-index segment (`[N]`) to a field path.
pub(crate) fn append_index(mut path: Vec<PathComponent>, n: &str) -> Vec<PathComponent> {
    path.push(PathComponent::Index(n.parse().unwrap_or(0)));
    path
}

fn map_func_name(name: &str) -> String {
    let mapped = match name.to_ascii_uppercase().as_str() {
        "ABS" => "mathAbs",
        "ACOS" => "mathAcos",
        "ASIN" => "mathAsin",
        "ATAN" => "mathAtan",
        "ATAN2" => "mathAtan2",
        "CEIL" => "mathCeil",
        "COS" => "mathCos",
        "DEGREES" => "mathDegrees",
        "EXP" => "mathExp",
        "FLOOR" => "mathFloor",
        "LN" => "mathLn",
        "LOG" => "mathLog",
        "POW" => "mathPow",
        "RADIANS" => "mathRadians",
        "ROUND" => "mathRound",
        "SIN" => "mathSin",
        "SQRT" => "mathSqrt",
        "TAN" => "mathTan",
        "PI" => "mathPi",
        "E" => "mathE",
        "DATE" => "date",
        _ => return name.to_string(), // already-internal (mathAdd, …) or unknown: pass through
    };
    mapped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonsm::compile::Projection;
    use jsonsm_ast::{CompareOp, Field};

    fn key(k: &str) -> PathComponent {
        PathComponent::Key(k.to_owned())
    }
    fn fld(keys: &[&str]) -> Expr {
        Expr::Field(Field::root(keys.iter().map(|k| key(k)).collect()))
    }
    fn p(s: &str) -> Expr {
        parse_str(s).unwrap()
    }

    #[test]
    fn simple_comparison() {
        assert_eq!(
            p("age < 50"),
            Expr::compare(
                CompareOp::LessThan,
                fld(&["age"]),
                Expr::Value(Literal::Int(50))
            )
        );
        assert_eq!(
            p(r#"name = "Brett""#),
            Expr::compare(
                CompareOp::Equals,
                fld(&["name"]),
                Expr::Value(Literal::String("Brett".into()))
            )
        );
        assert_eq!(p("a == 1"), p("a = 1"));
        assert_eq!(p("a <> 1"), p("a != 1"));
    }

    #[test]
    fn precedence_and_logic() {
        let e = p("a = 1 OR b = 2 AND c = 3");
        match e {
            Expr::Or(v) => {
                assert_eq!(v.len(), 2);
                assert!(matches!(v[1], Expr::And(_)));
            }
            _ => panic!("expected Or at top"),
        }
        assert!(matches!(p("NOT a = 1"), Expr::Not(_)));
        assert!(matches!(p("!(a = 1)"), Expr::Not(_)));
        assert!(matches!(p("(a = 1 OR b = 2) AND c = 3"), Expr::And(_)));
    }

    #[test]
    fn field_paths() {
        assert_eq!(
            p("a.b.c IS NOT MISSING"),
            Expr::Exists(Box::new(fld(&["a", "b", "c"])))
        );
        assert_eq!(
            p("`weird key`[2] = 1"),
            Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field::root(vec![key("weird key"), PathComponent::Index(2)])),
                Expr::Value(Literal::Int(1))
            )
        );
    }

    #[test]
    fn indexed_paths_match_end_to_end() {
        use jsonsm::collation::DefaultCollation;
        use jsonsm::matcher::FastMatcher;

        // `a[1]` addresses the array element, and reaches the engine as a typed index.
        let def = compile_str("a[1].b = 9", &Projection::new(), &DefaultCollation).unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"a": [{"b": 1}, {"b": 9}]}"#)
            .unwrap()
            .matched());
        assert!(!m
            .matches(br#"{"a": [{"b": 9}, {"b": 1}]}"#)
            .unwrap()
            .matched());
        // Out of range / not an array -> missing.
        assert!(!m.matches(br#"{"a": [{"b": 9}]}"#).unwrap().matched());
        assert!(!m.matches(br#"{"a": {"1": {"b": 9}}}"#).unwrap().matched());
    }

    #[test]
    fn nested_loops_reading_outer_scopes() {
        use jsonsm::collation::DefaultCollation;
        use jsonsm::matcher::FastMatcher;

        // Two loops deep, the inner body reading both the middle scope and the document.
        let def = compile_str(
            "ANY o IN outer SATISFIES ANY i IN o.items SATISFIES i.v = o.want AND i.v > lo END END",
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"lo": 0, "outer": [{"want": 3, "items": [{"v": 3}]}]}"#)
            .unwrap()
            .matched());
        // v == o.want but not > lo.
        assert!(!m
            .matches(br#"{"lo": 5, "outer": [{"want": 3, "items": [{"v": 3}]}]}"#)
            .unwrap()
            .matched());
        // v != o.want.
        assert!(!m
            .matches(br#"{"lo": 0, "outer": [{"want": 4, "items": [{"v": 3}]}]}"#)
            .unwrap()
            .matched());
        // Order-independent: both outer fields appear after the arrays.
        assert!(m
            .matches(br#"{"outer": [{"items": [{"v": 3}], "want": 3}], "lo": 0}"#)
            .unwrap()
            .matched());
    }

    #[test]
    fn is_null_and_missing() {
        assert_eq!(
            p("x IS NULL"),
            Expr::compare(CompareOp::Equals, fld(&["x"]), Expr::Value(Literal::Null))
        );
        assert_eq!(
            p("x IS NOT NULL"),
            Expr::compare(
                CompareOp::NotEquals,
                fld(&["x"]),
                Expr::Value(Literal::Null)
            )
        );
        assert_eq!(p("x IS MISSING"), Expr::NotExists(Box::new(fld(&["x"]))));
        assert_eq!(p("x IS NOT MISSING"), Expr::Exists(Box::new(fld(&["x"]))));
    }

    #[test]
    fn functions_and_arithmetic() {
        assert_eq!(
            p("ABS(x) = 5"),
            Expr::compare(
                CompareOp::Equals,
                Expr::Func(Func {
                    name: "mathAbs".into(),
                    args: vec![fld(&["x"])]
                }),
                Expr::Value(Literal::Int(5)),
            )
        );
        // a + b * 2  ->  mathAdd(a, mathMultiply(b, 2))  (multiplicative binds tighter)
        assert_eq!(
            p("a + b * 2 = 9"),
            Expr::compare(
                CompareOp::Equals,
                Expr::Func(Func {
                    name: "mathAdd".into(),
                    args: vec![
                        fld(&["a"]),
                        Expr::Func(Func {
                            name: "mathMultiply".into(),
                            args: vec![fld(&["b"]), Expr::Value(Literal::Int(2))],
                        }),
                    ],
                }),
                Expr::Value(Literal::Int(9)),
            )
        );
        assert_eq!(
            p("x = -3"),
            Expr::compare(
                CompareOp::Equals,
                fld(&["x"]),
                Expr::Value(Literal::Int(-3))
            )
        );
    }

    #[test]
    fn regexp_and_exists_and_bools() {
        assert_eq!(
            p(r#"REGEXP_CONTAINS(email, "@x\\.com$")"#),
            Expr::Matches {
                lhs: Box::new(fld(&["email"])),
                pattern: Box::new(Expr::Value(Literal::String("@x\\.com$".into()))),
            }
        );
        assert_eq!(p("EXISTS(x)"), Expr::Exists(Box::new(fld(&["x"]))));
        assert_eq!(p("TRUE"), Expr::True);
        assert_eq!(
            p("active = true"),
            Expr::compare(
                CompareOp::Equals,
                fld(&["active"]),
                Expr::Value(Literal::Bool(true))
            )
        );
    }

    #[test]
    fn end_to_end_matches() {
        use jsonsm::collation::DefaultCollation;
        use jsonsm::matcher::FastMatcher;

        let def = compile_str(
            r#"(age < 50 AND isActive = true) OR eyeColor = "brown""#,
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"age": 30, "isActive": true, "eyeColor": "blue"}"#)
            .unwrap()
            .matched());
        assert!(m
            .matches(br#"{"age": 80, "isActive": false, "eyeColor": "brown"}"#)
            .unwrap()
            .matched());
        assert!(!m
            .matches(br#"{"age": 80, "isActive": false, "eyeColor": "blue"}"#)
            .unwrap()
            .matched());
    }

    #[test]
    fn loop_any_binds_variable() {
        use jsonsm_ast::LoopType;
        // ANY tag IN tags SATISFIES tag = "cillum" END
        let e = p(r#"ANY tag IN tags SATISFIES tag = "cillum" END"#);
        assert_eq!(
            e,
            Expr::Loop {
                loop_type: LoopType::Any,
                var: 1,
                in_expr: Box::new(fld(&["tags"])),
                sub_expr: Box::new(Expr::compare(
                    CompareOp::Equals,
                    // `tag` resolved to the loop variable (root = 1, empty path).
                    Expr::Field(Field {
                        root: 1,
                        path: vec![]
                    }),
                    Expr::Value(Literal::String("cillum".into())),
                )),
            }
        );
    }

    #[test]
    fn loop_cross_scope_resolves_outer_field() {
        // ANY f IN friends SATISFIES f.id = index END
        // `f.id` -> loop var; `index` -> document field (not the loop variable).
        let e = p("ANY f IN friends SATISFIES f.id = index END");
        match e {
            Expr::Loop {
                var: 1,
                in_expr,
                sub_expr,
                ..
            } => {
                assert_eq!(*in_expr, fld(&["friends"]));
                assert_eq!(
                    *sub_expr,
                    Expr::compare(
                        CompareOp::Equals,
                        Expr::Field(Field {
                            root: 1,
                            path: vec![key("id")]
                        }),
                        fld(&["index"]), // root = document
                    )
                );
            }
            other => panic!("expected loop, got {other:?}"),
        }
    }

    #[test]
    fn loop_every_and_any_every() {
        use jsonsm_ast::LoopType;
        assert!(matches!(
            p("EVERY x IN xs SATISFIES x > 0 END"),
            Expr::Loop {
                loop_type: LoopType::Every,
                ..
            }
        ));
        assert!(matches!(
            p("ANY AND EVERY x IN xs SATISFIES x > 0 END"),
            Expr::Loop {
                loop_type: LoopType::AnyEvery,
                ..
            }
        ));
    }

    #[test]
    fn loops_end_to_end() {
        use jsonsm::collation::DefaultCollation;
        use jsonsm::matcher::FastMatcher;

        // any-in over a scalar array
        let def = compile_str(
            r#"ANY t IN tags SATISFIES t = "cillum" END"#,
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"tags": ["a", "cillum"]}"#)
            .unwrap()
            .matched());
        assert!(!m.matches(br#"{"tags": ["a", "b"]}"#).unwrap().matched());

        // cross-scope loop through the N1QL grammar
        let def = compile_str(
            "ANY f IN friends SATISFIES f.id = index END",
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"index": 2, "friends": [{"id": 1}, {"id": 2}]}"#)
            .unwrap()
            .matched());
        assert!(!m
            .matches(br#"{"index": 9, "friends": [{"id": 1}, {"id": 2}]}"#)
            .unwrap()
            .matched());

        // combined with outer boolean logic
        let def = compile_str(
            r#"active = true AND ANY t IN tags SATISFIES t = "x" END"#,
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"active": true, "tags": ["x"]}"#)
            .unwrap()
            .matched());
        assert!(!m
            .matches(br#"{"active": false, "tags": ["x"]}"#)
            .unwrap()
            .matched());
    }

    #[test]
    fn date_comparison_end_to_end() {
        use jsonsm::collation::DefaultCollation;
        use jsonsm::matcher::FastMatcher;

        // DATE(ts) is compared numerically (epoch seconds) against a DATE constant.
        let def = compile_str(
            r#"DATE(ts) >= DATE("2020-01-01T00:00:00Z")"#,
            &Projection::new(),
            &DefaultCollation,
        )
        .unwrap();
        let mut m = FastMatcher::new(&def);
        assert!(m
            .matches(br#"{"ts": "2020-06-15T12:00:00Z"}"#)
            .unwrap()
            .matched());
        assert!(!m
            .matches(br#"{"ts": "2019-12-31T23:59:59Z"}"#)
            .unwrap()
            .matched());
        // missing / non-date field -> DATE() is Missing -> comparison false
        assert!(!m.matches(br#"{"other": 1}"#).unwrap().matched());
        assert!(!m.matches(br#"{"ts": "not a date"}"#).unwrap().matched());
    }

    #[test]
    fn syntax_errors_are_reported() {
        assert!(parse_str("age <").is_err());
        assert!(parse_str("(a = 1").is_err());
        assert!(parse_str("").is_err());
    }
}
