use egg::{EGraph, Runner};
use rice::egraph::{lang::RiceLang, rules::all_rules, cost::RiceCost};

fn run(expr: &str) -> String {
    let mut egraph = EGraph::<RiceLang, ()>::default();
    let id = egraph.add_expr(&expr.parse().unwrap());
    let runner = Runner::default().with_egraph(egraph).run(&all_rules());
    let extractor = egg::Extractor::new(&runner.egraph, RiceCost);
    let (_, best) = extractor.find_best(id);
    best.to_string()
}

#[test]
fn test_add_zero() {
    assert_eq!(run("(+ x 0)"), "x");
}

#[test]
fn test_add_zero_left() {
    assert_eq!(run("(+ 0 x)"), "x");
}

#[test]
fn test_sub_zero() {
    assert_eq!(run("(- x 0)"), "x");
}

#[test]
fn test_mul_zero() {
    assert_eq!(run("(* x 0)"), "0");
}

#[test]
fn test_mul_zero_left() {
    assert_eq!(run("(* 0 x)"), "0");
}

#[test]
fn test_mul_one() {
    assert_eq!(run("(* x 1)"), "x");
}

#[test]
fn test_mul_one_left() {
    assert_eq!(run("(* 1 x)"), "x");
}

#[test]
fn test_div_one() {
    assert_eq!(run("(/ x 1)"), "x");
}

#[test]
fn test_eq_self() {
    assert_eq!(run("(= x x)"), "true");
}

#[test]
fn test_lt_self() {
    assert_eq!(run("(< x x)"), "false");
}

#[test]
fn test_gt_self() {
    assert_eq!(run("(> x x)"), "false");
}

#[test]
fn test_add_assoc() {
    assert_eq!(run("(+ (+ a b) c)"), "(+ a (+ b c))");
}

#[test]
fn test_mul_assoc() {
    assert_eq!(run("(* (* a b) c)"), "(* a (* b c))");
}