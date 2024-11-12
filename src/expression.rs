#[derive(Debug)]
pub struct FieldExpression<'a> {
    pub path: Vec<&'a str>,
}

#[derive(Debug)]
pub struct FuncExpression<'a> {
    pub name: String,
    pub args: Vec<Expression<'a>>,
}

#[derive(Debug)]
pub struct LoopExpression<'a> {
    pub var_name: String,
    pub in_expr: Box<Expression<'a>>,
    pub sub_expr: Box<Expression<'a>>,
}

#[derive(Debug)]
pub enum Constant<'a> {
    Bool(bool),
    String(&'a str),
    Int(i64),
    Uint(u64),
    Float(f64),
}

#[derive(Debug)]
pub enum Expression<'a> {
    Constant(Constant<'a>),
    Exists(Box<Expression<'a>>),

    Not(Box<Expression<'a>>),
    And(Vec<Expression<'a>>),
    Or(Vec<Expression<'a>>),

    Field(FieldExpression<'a>),
    Func(FuncExpression<'a>),

    AnyIn(LoopExpression<'a>),
    EveryIn(LoopExpression<'a>),
    AnyEveryIn(LoopExpression<'a>),

    Equals(Box<Expression<'a>>, Box<Expression<'a>>),
    LessThan(Box<Expression<'a>>, Box<Expression<'a>>),
    LessThanEquals(Box<Expression<'a>>, Box<Expression<'a>>),
}

#[cfg(test)]
mod tests {
    use crate::expression::Constant;
    use crate::expression::Expression;
    use crate::expression::FieldExpression;

    #[test]
    fn expr_basic() {
        let expr: Expression = Expression::Equals(
            Box::new(Expression::Field(FieldExpression {
                path: vec!["$doc", "name"],
            })),
            Box::new(Expression::Constant(Constant::String("Daphne Sutton"))),
        );

        println!("Expr: {:?}", expr);
    }
}
