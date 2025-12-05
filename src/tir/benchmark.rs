use std::time::Instant;
use csv::Writer;

use egg::{Runner, EGraph, Id, AstSize, RecExpr, CostFunction};

use crate::tir::rewrite::{
    TirLang, TirCost, TirSmartCost, AnyCostFn, make_rules
};


pub struct BenchResult {
    pub iter_limit: usize,
    pub time_ms: u128,
    pub egraph_nodes: usize,
    pub egraph_classes: usize,
    pub saturated: bool,
    pub ast_size_before: usize,
    pub ast_size_after: usize,
}


pub fn benchmark_once(
    original_expr: &RecExpr<TirLang>, 
    iter_limit: usize,
    cost_model: &str
) -> BenchResult {

    let ast_size_before = AstSize.cost_rec(original_expr);

    let mut egraph = EGraph::<TirLang, ()>::default();
    let root = egraph.add_expr(original_expr);

    let start = Instant::now();

    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(iter_limit)
        .run(&make_rules());

    let duration = start.elapsed().as_millis();


    let mut cost_fn = match cost_model {
        "ast" => AnyCostFn::Ast(AstSize),
        "tir" => AnyCostFn::Tir(TirCost),
        "smart" => AnyCostFn::Smart(TirSmartCost),
        _ => AnyCostFn::Ast(AstSize),
    };

    let extractor = egg::Extractor::new(&runner.egraph, cost_fn);
    let (_best_cost, best_expr) = extractor.find_best(root);

    let ast_size_after = AstSize.cost_rec(&best_expr);
    let saturated = matches!(runner.stop_reason, Some(egg::StopReason::Saturated));

    BenchResult {
        iter_limit,
        time_ms: duration,
        egraph_nodes: runner.egraph.total_size(),
        egraph_classes: runner.egraph.number_of_classes(),
        saturated,
        ast_size_before,
        ast_size_after,
    }
}



pub fn sweep(
    original_expr: RecExpr<TirLang>,
    limits: &[usize],
    cost_model: &str,
    csv_path: &str
) {
    let mut writer = Writer::from_path(csv_path).unwrap();

    writer.write_record(&[
        "iter_limit",
        "time_ms",
        "e_nodes",
        "e_classes",
        "saturated",
        "ast_before",
        "ast_after",
    ]).unwrap();

    for &limit in limits {
        let res = benchmark_once(&original_expr, limit, cost_model);
        writer.write_record(&[
            res.iter_limit.to_string(),
            res.time_ms.to_string(),
            res.egraph_nodes.to_string(),
            res.egraph_classes.to_string(),
            res.saturated.to_string(),
            res.ast_size_before.to_string(),
            res.ast_size_after.to_string(),
        ]).unwrap();
    }

    writer.flush().unwrap();
}
