use egg::RecExpr;
use crate::egraph::lang::RiceLang;

pub fn lower_num(n: i32) -> RecExpr<RiceLang> {
    let mut r = RecExpr::default();
    r.add(RiceLang::Num(n));
    r
}

pub fn lower_var(name: &str) -> RecExpr<RiceLang> {
    let mut r = RecExpr::default();
    r.add(RiceLang::Var(name.to_string()));
    r
}

