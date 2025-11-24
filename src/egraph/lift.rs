use egg::{RecExpr, Id};
use crate::egraph::lang::RiceLang;

pub fn extract_num(rec: &RecExpr<RiceLang>, id: Id) -> Option<i32> {
    match &rec[id] {
        RiceLang::Num(n) => Some(*n),
        _ => None,
    }
}
