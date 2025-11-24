use egg::{CostFunction, Id, Language};
use crate::egraph::lang::RiceLang;

pub struct RiceCost;

impl CostFunction<RiceLang> for RiceCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &RiceLang, mut child_cost: C) -> usize
    where
        C: FnMut(Id) -> usize,
    {
        1 + enode
            .children()
            .iter()
            .map(|&id| child_cost(id))
            .sum::<usize>()
    }
}
