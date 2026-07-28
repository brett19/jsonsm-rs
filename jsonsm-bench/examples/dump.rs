use jsonsm::collation::DefaultCollation;
use jsonsm::compile::{compile, Projection};
use jsonsm_ast::{CompareOp, Expr, Field, Literal, LoopType, PathComponent};

fn field(keys: &[&str]) -> Expr {
    Expr::Field(Field::root(
        keys.iter().map(|k| PathComponent::Key((*k).to_owned())).collect(),
    ))
}

fn main() {
    let e = Expr::Loop {
        loop_type: LoopType::Any,
        var: 1,
        in_expr: Box::new(field(&["tags"])),
        sub_expr: Box::new(Expr::compare(
            CompareOp::Equals,
            Expr::Field(Field { root: 1, path: vec![] }),
            Expr::Value(Literal::String("cillum".into())),
        )),
    };
    let def = compile(std::slice::from_ref(&e), &Projection::new(), &DefaultCollation).unwrap();
    println!("{def:#?}");
}
