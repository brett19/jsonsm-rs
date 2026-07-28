//! The Rust-native expression AST for `jsonsm`.
//!
//! This crate is **pure data**: it describes an expression to be matched against a
//! JSON document, with no engine, no compiled programs, and no I/O. Front-end crates
//! (`jsonsm-json`, `jsonsm-n1ql`) parse their respective surface syntaxes *into* this
//! AST; the `jsonsm` engine compiles this AST into an executable match definition.
//!
//! Regex/`LIKE` patterns are represented here as plain strings (see [`Expr::Matches`]).
//! How a pattern is compiled and executed — and how values collate — is decided at
//! compile time by the engine's collation strategy, not by this tree.

#![forbid(unsafe_code)]

/// Identifier for a variable bound in the expression.
///
/// [`ROOT_VAR`] (`0`) refers to the document root (`$doc`). Non-zero ids are bound by
/// enclosing [`Expr::Loop`] nodes.
pub type VariableId = u32;

/// The document-root variable (`$doc`).
pub const ROOT_VAR: VariableId = 0;

/// A constant value literal that can appear in an expression.
///
/// This is the *authoring-time* value type (owned, simple). It is distinct from the
/// engine's runtime value type, which additionally models values borrowed from the
/// document being scanned.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(i64),
    Uint(u64),
    Float(f64),
    String(String),
}

/// A binary comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Equals,
    NotEquals,
    LessThan,
    LessEquals,
    GreaterThan,
    GreaterEquals,
}

/// The three array-looping quantifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoopType {
    /// At least one element satisfies the sub-expression.
    Any,
    /// Every element satisfies the sub-expression (vacuously true for empty arrays).
    Every,
    /// Non-empty *and* every element satisfies the sub-expression.
    AnyEvery,
}

/// One step in a field path: an object key or an array index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathComponent {
    /// An object member, by key.
    Key(String),
    /// An array element, by zero-based index.
    Index(usize),
}

impl From<String> for PathComponent {
    fn from(key: String) -> Self {
        PathComponent::Key(key)
    }
}

impl From<&str> for PathComponent {
    fn from(key: &str) -> Self {
        PathComponent::Key(key.to_owned())
    }
}

impl From<usize> for PathComponent {
    fn from(index: usize) -> Self {
        PathComponent::Index(index)
    }
}

/// A reference to a value within the document (or a loop variable).
///
/// `root` names the variable the path is relative to ([`ROOT_VAR`] for the document),
/// and `path` walks object keys / array indices from there.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub root: VariableId,
    pub path: Vec<PathComponent>,
}

impl Field {
    /// A path into the document root.
    pub fn root(path: Vec<PathComponent>) -> Self {
        Self {
            root: ROOT_VAR,
            path,
        }
    }
}

/// A built-in function application (e.g. `mathRound`, `DATE`).
///
/// Function *names* are kept as strings here; the set of supported functions is defined
/// by the engine, not the AST.
#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub name: String,
    pub args: Vec<Expr>,
}

/// An expression node.
///
/// The tree mixes boolean-valued nodes (combinators, comparisons, existence, loops) with
/// value-valued nodes ([`Expr::Value`], [`Expr::Field`], [`Expr::Func`]) that appear as
/// operands. Validation that operands and boolean nodes are used in the right positions
/// is the compiler's responsibility.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Constant `true`.
    True,
    /// Constant `false`.
    False,

    /// A literal constant operand.
    Value(Literal),
    /// A field/variable reference operand.
    Field(Field),
    /// A function-call operand.
    Func(Func),

    /// Logical negation.
    Not(Box<Expr>),
    /// Logical conjunction of zero or more sub-expressions.
    And(Vec<Expr>),
    /// Logical disjunction of zero or more sub-expressions.
    Or(Vec<Expr>),

    /// True if the operand path exists in the document.
    Exists(Box<Expr>),
    /// True if the operand path does *not* exist in the document.
    NotExists(Box<Expr>),

    /// A binary comparison between two operands.
    Compare {
        op: CompareOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    /// Pattern match (regex / `LIKE`). `pattern` is expected to resolve to a string;
    /// it is compiled and executed by the collation strategy chosen at compile time.
    Matches { lhs: Box<Expr>, pattern: Box<Expr> },

    /// Array iteration with a quantifier. Binds `var` to each element of `in_expr`
    /// while evaluating `sub_expr`.
    Loop {
        loop_type: LoopType,
        var: VariableId,
        in_expr: Box<Expr>,
        sub_expr: Box<Expr>,
    },
}

impl Expr {
    /// Convenience constructor for a comparison node.
    pub fn compare(op: CompareOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr::Compare {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    /// Nesting depth of this expression: `1` for a leaf, one more per level of nesting.
    ///
    /// Computed **iteratively** (an explicit stack, not recursion) so that measuring an
    /// adversarially deep tree cannot itself overflow the stack — the point being to reject
    /// such a tree before any recursive pass (name resolution, compilation, matching) walks
    /// it. See `Expr::exceeds_depth` for the cheap form.
    pub fn depth(&self) -> usize {
        let mut max = 0;
        let mut stack = vec![(self, 1usize)];
        while let Some((e, d)) = stack.pop() {
            max = max.max(d);
            e.for_each_child(&mut |child| stack.push((child, d + 1)));
        }
        max
    }

    /// Whether this expression is nested deeper than `limit`. Stops as soon as the limit is
    /// exceeded, so it stays cheap on deep input.
    pub fn exceeds_depth(&self, limit: usize) -> bool {
        let mut stack = vec![(self, 1usize)];
        while let Some((e, d)) = stack.pop() {
            if d > limit {
                return true;
            }
            e.for_each_child(&mut |child| stack.push((child, d + 1)));
        }
        false
    }

    /// Apply `f` to each direct sub-expression.
    fn for_each_child<'e>(&'e self, f: &mut impl FnMut(&'e Expr)) {
        match self {
            Expr::Not(e) | Expr::Exists(e) | Expr::NotExists(e) => f(e),
            Expr::And(es) | Expr::Or(es) => es.iter().for_each(f),
            Expr::Func(func) => func.args.iter().for_each(f),
            Expr::Compare { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            Expr::Matches { lhs, pattern } => {
                f(lhs);
                f(pattern);
            }
            Expr::Loop {
                in_expr, sub_expr, ..
            } => {
                f(in_expr);
                f(sub_expr);
            }
            Expr::Value(_) | Expr::Field(_) | Expr::True | Expr::False => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_a_representative_tree() {
        // $doc.name.first == "Brett" OR ($doc.age < 50 AND $doc.isActive == true)
        let expr = Expr::Or(vec![
            Expr::compare(
                CompareOp::Equals,
                Expr::Field(Field::root(vec![
                    PathComponent::Key("name".into()),
                    PathComponent::Key("first".into()),
                ])),
                Expr::Value(Literal::String("Brett".into())),
            ),
            Expr::And(vec![
                Expr::compare(
                    CompareOp::LessThan,
                    Expr::Field(Field::root(vec![PathComponent::Key("age".into())])),
                    Expr::Value(Literal::Int(50)),
                ),
                Expr::compare(
                    CompareOp::Equals,
                    Expr::Field(Field::root(vec![PathComponent::Key("isActive".into())])),
                    Expr::Value(Literal::Bool(true)),
                ),
            ]),
        ]);

        // Round-trips through Clone/PartialEq and matches structurally.
        assert_eq!(expr.clone(), expr);
        match &expr {
            Expr::Or(branches) => assert_eq!(branches.len(), 2),
            _ => panic!("expected Or at the root"),
        }
    }
}
