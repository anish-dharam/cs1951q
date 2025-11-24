use crate::egraph::lang::RiceLang;
use egg::{Rewrite, rewrite as rw};

pub fn all_rules() -> Vec<Rewrite<RiceLang, ()>> {
    vec![
        rw!("add-const"; "(+ ?a ?b)" => "(+ ?a ?b)"),
        rw!("sub-const"; "(- ?a ?b)" => "(- ?a ?b)"),
        rw!("mul-const"; "(* ?a ?b)" => "(* ?a ?b)"),
        rw!("div-const"; "(/ ?a ?b)" => "(/ ?a ?b)"),
        rw!("rem-const"; "(% ?a ?b)" => "(% ?a ?b)"),
        rw!("add-zero"; "(+ ?x 0)" => "?x"),
        rw!("add-zero-comm"; "(+ 0 ?x)" => "?x"),
        rw!("sub-zero"; "(- ?x 0)" => "?x"),
        rw!("mul-zero"; "(* ?x 0)" => "0"),
        rw!("mul-zero-comm"; "(* 0 ?x)" => "0"),
        rw!("mul-one"; "(* ?x 1)" => "?x"),
        rw!("mul-one-comm"; "(* 1 ?x)" => "?x"),
        rw!("div-one"; "(/ ?x 1)" => "?x"),
        rw!("eq-self"; "(= ?x ?x)" => "true"),
        rw!("lt-self"; "(< ?x ?x)" => "false"),
        rw!("gt-self"; "(> ?x ?x)" => "false"),
        rw!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("mul-assoc"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
    ]
}
