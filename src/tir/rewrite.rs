//! Typed intermediate representation (TIR).
//!
//! The TIR is an expression-oriented IR with types attached to each expression node,
//! and with some AST features removed.

use super::types::{Expr, ExprKind, MethodRef, Program};
use crate::ast::types::{Binop, Const, ParamList, Type};
use crate::utils::Symbol;
use ordered_float::OrderedFloat;

use egg::{
    AstSize, EGraph, Extractor, Id, Pattern, RecExpr, Rewrite, Runner, define_language, rewrite,
};

pub use super::typeck::Tcx;

define_language! {
    enum TirLang {

        Bool(bool),
        Int(i32),
        Float(OrderedFloat<f32>),
        String(String),
        // Leaf nodes
        Var(Symbol),
        // Const(Const),

        "tuple" = Tuple(Vec<Id>),
        "struct" = Struct(Vec<Id>),
        Index(usize), //should never be used by itself
        "project" = Project([Id; 2]), // second should be index
        // Binary operators - these can be parsed directly by egg
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Rem([Id; 2]),
        "**" = Exp([Id; 2]),
        "==" = Eq([Id; 2]),
        "!=" = Neq([Id; 2]),
        "<" = Lt([Id; 2]),
        ">" = Gt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">=" = Ge([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "&" = BitAnd([Id; 2]),
        "<<" = Shl([Id; 2]),
        ">>" = Shr([Id; 2]),
        "|" = BitOr([Id; 2]),
        "^" = Concat([Id; 2]),

        Type(Type),
        "cast" = Cast([Id; 2]), // second should be type
        "arglist" = ArgList(Vec<Id>),
        "call" = Call([Id; 2]), // first should be function, second should be arglist
        "methodref" = MethodRef([Id; 2]), // first should be interface, second should be method
        "methodcall" = MethodCall([Id; 3]), // first should be receiver, second should be methodref, third should be arglist
        "paramlist" = ParamList(Vec<Id>), // list of param ids (loses type?)
        "lambda" = Lambda([Id; 4]), // first should be paramlist, second should be envlist, third should be returntype, fourth should be body
        Fname(Symbol),
        "envlist" = EnvList(Vec<Id>),
        "closure" = Closure([Id; 2]), // first should be function, second should be envlist
        "seq" = Seq([Id; 2]), //  first shouldn't be seq (i think)
        "let" = Let([Id; 4]), // first should be name, second should be type, third should be e1, fourth should be e2
        "return" = Return(Id),
        "loop" = Loop(Id),
        "while" = While([Id; 2]),
        "if" = If([Id; 3]),
        "assign" = Assign([Id; 2]),
        "break" = Break,
        "arrayliteral" = ArrayLiteral(Vec<Id>),
        "arrayindex" = ArrayIndex([Id; 2]), // first should be array, second should be index
        Count(usize), // should never be used by itself
        "arraycopy" = ArrayCopy([Id; 2]), // first should be value, second should be count

    }
}

fn make_rules() -> Vec<Rewrite<TirLang, ()>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        // rewrite!("add-0"; "(+ ?a Const(0))" => "?a"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
    ]
}

/// parse an expression, simplify it using egg, and pretty print it back out
fn simplify(s: &str) -> String {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<TirLang> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr).run(&make_rules());

    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(root);
    println!("Simplified {} to {} with cost {}", expr, best, best_cost);
    best.to_string()
}

/// Convert a Vec<Expr> to a list representation in the e-graph
// fn vec_to_list(
//     exprs: &[Expr],
//     egraph: &mut EGraph<TirLang, ()>,
//     convert_fn: &mut dyn FnMut(&Expr, &mut EGraph<TirLang, ()>) -> Id,
// ) -> Id {
//     let head_id = convert_fn(&exprs[0], egraph);
//     let tail_id = vec_to_list(&exprs[1..], egraph, convert_fn);
// }

fn symbol_id(egraph: &mut EGraph<TirLang, ()>, sym: Symbol) -> Id {
    egraph.add(TirLang::Var(sym))
}

fn type_id(egraph: &mut EGraph<TirLang, ()>, ty: &Type) -> Id {
    egraph.add(TirLang::Type(*ty))
}

fn expr_vec_ids(exprs: &[Expr], egraph: &mut EGraph<TirLang, ()>) -> Vec<Id> {
    exprs.iter().map(|expr| expr_to_egg(expr, egraph)).collect()
}

fn paramlist_id(params: &ParamList, egraph: &mut EGraph<TirLang, ()>) -> Id {
    let symbols = params
        .iter()
        .map(|(sym, _)| symbol_id(egraph, *sym))
        .collect();
    egraph.add(TirLang::ParamList(symbols))
}

fn env_paramlist_id(env: &ParamList, egraph: &mut EGraph<TirLang, ()>) -> Id {
    let symbols = env.iter().map(|(sym, _)| symbol_id(egraph, *sym)).collect();
    egraph.add(TirLang::EnvList(symbols))
}

fn env_exprs_id(exprs: &[Expr], egraph: &mut EGraph<TirLang, ()>) -> Id {
    let ids = expr_vec_ids(exprs, egraph);
    egraph.add(TirLang::EnvList(ids))
}

fn arglist_id(args: &[Expr], egraph: &mut EGraph<TirLang, ()>) -> Id {
    let ids = expr_vec_ids(args, egraph);
    egraph.add(TirLang::ArgList(ids))
}

fn methodref_id(method: &MethodRef, egraph: &mut EGraph<TirLang, ()>) -> Id {
    let interface = symbol_id(egraph, method.interface);
    let name = symbol_id(egraph, method.method);
    egraph.add(TirLang::MethodRef([interface, name]))
}

fn unit_id(egraph: &mut EGraph<TirLang, ()>) -> Id {
    egraph.add(TirLang::Tuple(Vec::new()))
}

fn binop_id(op: Binop, left: Id, right: Id, egraph: &mut EGraph<TirLang, ()>) -> Id {
    match op {
        Binop::Add => egraph.add(TirLang::Add([left, right])),
        Binop::Sub => egraph.add(TirLang::Sub([left, right])),
        Binop::Mul => egraph.add(TirLang::Mul([left, right])),
        Binop::Div => egraph.add(TirLang::Div([left, right])),
        Binop::Rem => egraph.add(TirLang::Rem([left, right])),
        Binop::Exp => egraph.add(TirLang::Exp([left, right])),
        Binop::Eq => egraph.add(TirLang::Eq([left, right])),
        Binop::Neq => egraph.add(TirLang::Neq([left, right])),
        Binop::Lt => egraph.add(TirLang::Lt([left, right])),
        Binop::Gt => egraph.add(TirLang::Gt([left, right])),
        Binop::Le => egraph.add(TirLang::Le([left, right])),
        Binop::Ge => egraph.add(TirLang::Ge([left, right])),
        Binop::And => egraph.add(TirLang::And([left, right])),
        Binop::Or => egraph.add(TirLang::Or([left, right])),
        Binop::Shl => egraph.add(TirLang::Shl([left, right])),
        Binop::Shr => egraph.add(TirLang::Shr([left, right])),
        Binop::BitAnd => egraph.add(TirLang::BitAnd([left, right])),
        Binop::BitOr => egraph.add(TirLang::BitOr([left, right])),
        Binop::Concat => egraph.add(TirLang::Concat([left, right])),
    }
}

/// Convert a TIR expression into an egg `TirLang` node and return its e-graph Id.
fn expr_to_egg(expr: &Expr, egraph: &mut EGraph<TirLang, ()>) -> Id {
    match &expr.kind {
        ExprKind::Var(sym) => symbol_id(egraph, *sym),
        ExprKind::Const(c) => match c {
            Const::Bool(b) => egraph.add(TirLang::Bool(*b)),
            Const::Int(i) => egraph.add(TirLang::Int(*i)),
            Const::Float(f) => egraph.add(TirLang::Float(*f)),
            Const::String(s) => egraph.add(TirLang::String(s.clone().to_string())),
        },
        ExprKind::Tuple(exprs) => {
            let ids = expr_vec_ids(exprs, egraph);
            egraph.add(TirLang::Tuple(ids))
        }
        ExprKind::Struct(exprs) => {
            let ids = expr_vec_ids(exprs, egraph);
            egraph.add(TirLang::Struct(ids))
        }
        ExprKind::Project { e, i } => {
            let expr_id = expr_to_egg(e, egraph);
            let index_id = egraph.add(TirLang::Index(*i));
            egraph.add(TirLang::Project([expr_id, index_id]))
        }
        ExprKind::BinOp { left, right, op } => {
            let left_id = expr_to_egg(left, egraph);
            let right_id = expr_to_egg(right, egraph);
            binop_id(*op, left_id, right_id, egraph)
        }
        ExprKind::Cast { e, ty } => {
            let expr_id = expr_to_egg(e, egraph);
            let ty_id = type_id(egraph, ty);
            egraph.add(TirLang::Cast([expr_id, ty_id]))
        }
        ExprKind::Call { f, args } => {
            let f_id = expr_to_egg(f, egraph);
            let args_id = arglist_id(args, egraph);
            egraph.add(TirLang::Call([f_id, args_id]))
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            let receiver_id = expr_to_egg(receiver, egraph);
            let method_id = methodref_id(method, egraph);
            let args_id = arglist_id(args, egraph);
            egraph.add(TirLang::MethodCall([receiver_id, method_id, args_id]))
        }
        ExprKind::Lambda {
            params,
            env,
            ret_ty,
            body,
        } => {
            let params_id = paramlist_id(params, egraph);
            let env_id = env_paramlist_id(env, egraph);
            let ret_ty_id = type_id(egraph, ret_ty);
            let body_id = expr_to_egg(body, egraph);
            egraph.add(TirLang::Lambda([params_id, env_id, ret_ty_id, body_id]))
        }
        ExprKind::Closure { f, env } => {
            let fname_id = egraph.add(TirLang::Fname(*f));
            let env_id = env_exprs_id(env, egraph);
            egraph.add(TirLang::Closure([fname_id, env_id]))
        }
        ExprKind::Seq(e1, e2) => {
            let e1_id = expr_to_egg(e1, egraph);
            let e2_id = expr_to_egg(e2, egraph);
            egraph.add(TirLang::Seq([e1_id, e2_id]))
        }
        ExprKind::Let { name, ty, e1, e2 } => {
            let name_id = symbol_id(egraph, *name);
            let ty_id = type_id(egraph, ty);
            let e1_id = expr_to_egg(e1, egraph);
            let e2_id = expr_to_egg(e2, egraph);
            egraph.add(TirLang::Let([name_id, ty_id, e1_id, e2_id]))
        }
        ExprKind::Return(e) => {
            let e_id = expr_to_egg(e, egraph);
            egraph.add(TirLang::Return(e_id))
        }
        ExprKind::Loop(body) => {
            let body_id = expr_to_egg(body, egraph);
            egraph.add(TirLang::Loop(body_id))
        }
        ExprKind::While { cond, body } => {
            let cond_id = expr_to_egg(cond, egraph);
            let body_id = expr_to_egg(body, egraph);
            egraph.add(TirLang::While([cond_id, body_id]))
        }
        ExprKind::If { cond, then_, else_ } => {
            let cond_id = expr_to_egg(cond, egraph);
            let then_id = expr_to_egg(then_, egraph);
            let else_id = else_
                .as_ref()
                .map(|expr| expr_to_egg(expr, egraph))
                .unwrap_or_else(|| unit_id(egraph));
            egraph.add(TirLang::If([cond_id, then_id, else_id]))
        }
        ExprKind::Assign { dst, src } => {
            let dst_id = expr_to_egg(dst, egraph);
            let src_id = expr_to_egg(src, egraph);
            egraph.add(TirLang::Assign([dst_id, src_id]))
        }
        ExprKind::Break => egraph.add(TirLang::Break),
        ExprKind::ArrayLiteral(exprs) => {
            let ids = expr_vec_ids(exprs, egraph);
            egraph.add(TirLang::ArrayLiteral(ids))
        }
        ExprKind::ArrayIndex { array, index } => {
            let array_id = expr_to_egg(array, egraph);
            let index_id = expr_to_egg(index, egraph);
            egraph.add(TirLang::ArrayIndex([array_id, index_id]))
        }
        ExprKind::ArrayCopy { value, count } => {
            let value_id = expr_to_egg(value, egraph);
            let count_id = expr_to_egg(count, egraph);
            egraph.add(TirLang::ArrayCopy([value_id, count_id]))
        }
    }
}

pub fn main(tcx: Tcx, tir: Program) -> (Tcx, Program) {
    println!("{}", simplify("(if (if true true false) 1 2)"));

    let mut egraph = EGraph::<TirLang, ()>::default();
    let mut main_id = None;
    let same_add: Pattern<TirLang> = "(+ ?a 0)".parse().unwrap();
    println!("\n{:?}\n", same_add);

    for func in tir.functions() {
        let res = expr_to_egg(&func.body, &mut egraph);
        if func.name == Symbol::main() {
            main_id = Some(res);
        }
        // egraph.rebuild();
    }

    if matches!(main_id, None) {
        panic!("No main function found in e-graph");
    }

    println!("{:?}", egraph);

    let runner = Runner::default().with_egraph(egraph).run(&make_rules());

    let extractor = Extractor::new(&runner.egraph, AstSize);

    let (best_cost, best) = extractor.find_best(main_id.unwrap());
    println!("Simplified to {} with cost {}", best, best_cost);
    best.to_string();
    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    // let runner = Runner::default().with_expr(&expr).run(&make_rules());

    // // the Runner knows which e-class the expression given with `with_expr` is in
    // let root = runner.roots[0];

    // // use an Extractor to pick the best element of the root eclass
    // let extractor = Extractor::new(&runner.egraph, AstSize);
    // let (best_cost, best) = extractor.find_best(root);
    // println!("Simplified {} to {} with cost {}", expr, best, best_cost);
    // best.to_string();

    (tcx, tir)
}
