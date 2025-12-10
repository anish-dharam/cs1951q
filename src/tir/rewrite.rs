//! Typed intermediate representation (TIR).
//!
//! The TIR is an expression-oriented IR with types attached to each expression node,
//! and with some AST features removed.

use super::types::{Expr, ExprKind, Function, MethodRef, Program};
use crate::ast::types::{Binop, Const, ParamList, Span, Type};
use crate::tir::benchmark::{sweep, sweep_with_rules};
use crate::utils::Symbol;
use ordered_float::OrderedFloat;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    pub static REWRITE_ITER_LIMIT: RefCell<usize> = RefCell::new(30);
    pub static REWRITE_TIME_LIMIT: RefCell<u64> = RefCell::new(1);
    pub static REWRITE_COST_MODEL: RefCell<String> = RefCell::new("ast".to_string());
}

use egg::{
    AstSize, EGraph, Extractor, Id, Language, Pattern, RecExpr, Rewrite, Runner, StopReason,
    define_language, rewrite,
};

pub use super::typeck::Tcx;

/// maps RecExpr node indices to their type and span information.
/// Used to preserve type/span information when converting between Expr and TirLang.
#[derive(Debug, Clone)]
pub struct TypeSpanMap {
    /// Map from RecExpr node index to (Type, Span)
    pub map: HashMap<usize, (Type, Span)>,
    /// Default span to use when not found
    pub default_span: Span,
}

#[derive(Clone, Copy)]
pub struct AblationConfig {
    pub commutation: bool,
    pub identity: bool,
    pub zero: bool,
    pub self_compare: bool,
    pub associativity: bool,
}

impl AblationConfig {
    pub fn full() -> Self {
        Self {
            commutation: true,
            identity: true,
            zero: true,
            self_compare: true,
            associativity: true,
        }
    }

    pub fn no_comm() -> Self {
        Self {
            commutation: false,
            ..Self::full()
        }
    }

    pub fn no_identity() -> Self {
        Self {
            identity: false,
            ..Self::full()
        }
    }

    pub fn no_zero() -> Self {
        Self {
            zero: false,
            ..Self::full()
        }
    }

    pub fn no_self_compare() -> Self {
        Self {
            self_compare: false,
            ..Self::full()
        }
    }

    pub fn no_assoc() -> Self {
        Self {
            associativity: false,
            ..Self::full()
        }
    }
}

impl TypeSpanMap {
    pub fn new(default_span: Span) -> Self {
        TypeSpanMap {
            map: HashMap::new(),
            default_span,
        }
    }

    pub fn insert(&mut self, idx: usize, ty: Type, span: Span) {
        self.map.insert(idx, (ty, span));
    }

    pub fn get(&self, idx: usize) -> (Type, Span) {
        self.map
            .get(&idx)
            .copied()
            .unwrap_or((Type::unit(), self.default_span))
    }
}

define_language! {
    pub enum TirLang {

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
        "<<" = Shl([Id; 2]),
        ">>" = Shr([Id; 2]),
        "|" = BitOr([Id; 2]),
        "&" = BitAnd([Id; 2]),
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
        "arraycopy" = ArrayCopy([Id; 2]), // first should be value, second should be count

    }
}

// pub fn make_rules() -> Vec<Rewrite<TirLang, ()>> {
//     vec![
//         rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
//         rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
//         // rewrite!("add-0"; "(+ ?a Const(0))" => "?a"),
//         rewrite!("add-0"; "(+ ?a 0)" => "?a"),
//         rewrite!("mul-0"; "(* ?a 0)" => "0"),
//         rewrite!("mul-1"; "(* ?a 1)" => "?a"),
//         rewrite!("sub-zero"; "(- ?x 0)" => "?x"),
//         rewrite!("div-one"; "(/ ?x 1)" => "?x"),
//         rewrite!("eq-self"; "(== ?x ?x)" => "true"),
//         rewrite!("lt-self"; "(< ?x ?x)" => "false"),
//         rewrite!("gt-self"; "(> ?x ?x)" => "false"),
//         rewrite!("lte-self"; "(<= ?x ?x)" => "true"),
//         rewrite!("gte-self"; "(>= ?x ?x)" => "true"),
//         rewrite!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
//         rewrite!("mul-assoc"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),

//         // // NEW STUFFS here
//         // rewrite!("let-id"; "(let ?x ?ty ?v ?x)" => "?v"),
//         // rewrite!("add-int"; "(+ (Int ?a) (Int ?b))" => { TirLang::Int(a + b) }),
//         // rewrite!("if-true"; "(if true ?t ?e)" => "?t"),
//         // rewrite!("if-false"; "(if false ?t ?e)" => "?e"),
//         // rewrite!("while-false"; "(while false ?body)" => "unit"),
//     ]
// }

pub fn make_rules(cfg: AblationConfig) -> Vec<Rewrite<TirLang, ()>> {
    let mut rules = vec![];
    //2 powers
    rules.push(rewrite!("shl-id"; "(* ?x 2)" => "(<< ?x 1)"));
    rules.push(rewrite!("shr-id"; "(/ ?x 2)" => "(>> ?x 1)"));
    rules.push(rewrite!("shl-id4"; "(* ?x 4)" => "(<< ?x 2)"));
    rules.push(rewrite!("shr-id4"; "(/ ?x 4)" => "(>> ?x 2)"));
    rules.push(rewrite!("shl-id8"; "(* ?x 8)" => "(<< ?x 3)"));
    rules.push(rewrite!("shr-id8"; "(/ ?x 8)" => "(>> ?x 3)"));
    rules.push(rewrite!("bitor-zero"; "(| ?x 0)" => "?x"));

    rules.push(rewrite!("bitand-all"; "(& ?x -1)" => "?x")); // two's complement all-ones
    rules.push(rewrite!("bitand-zero"; "(& ?x 0)" => "0"));
    rules.push(rewrite!("bitor-all";  "(| ?x -1)" => "-1"));

    // bools
    // Idempotence
    rules.push(rewrite!("and-idempotent"; "(&& ?x ?x)" => "?x"));
    rules.push(rewrite!("or-idempotent";  "(|| ?x ?x)" => "?x"));

    // Absorption
    rules.push(rewrite!("and-absorption"; "(&& ?x (|| ?x ?y))" => "?x"));
    rules.push(rewrite!("or-absorption";  "(|| ?x (&& ?x ?y))" => "?x"));

    // Distributivity
    rules.push(rewrite!("and-distrib-or";
    "(&& ?x (|| ?y ?z))" => "(|| (&& ?x ?y) (&& ?x ?z))"));
    rules.push(rewrite!("or-distrib-and";
    "(|| ?x (&& ?y ?z))" => "(&& (|| ?x ?y) (|| ?x ?z))"));

    // Constant conditionals
    rules.push(rewrite!("if-true";  "(if true  ?t ?e)" => "?t"));
    rules.push(rewrite!("if-false"; "(if false ?t ?e)" => "?e"));

    //distribute
    rules.push(rewrite!("distribute-add"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"));
    rules.push(rewrite!("distribute-sub"; "(* ?a (- ?b ?c))" => "(- (* ?a ?b) (* ?a ?c))"));
    rules.push(rewrite!("distribute-shl"; "(* ?a (<< ?b ?c))" => "(<< (* ?a ?b) ?c)"));
    rules.push(rewrite!("distribute-shr"; "(* ?a (>> ?b ?c))" => "(>> (* ?a ?b) ?c)"));
    rules.push(rewrite!("distribute-bitor"; "(* ?a (| ?b ?c))" => "(| (* ?a ?b) (* ?a ?c))"));

    rules.push(rewrite!("lte-from-lt-eq";
    "(<= ?a ?b)" => "(|| (< ?a ?b) (== ?a ?b))"));
    rules.push(rewrite!("gte-from-gt-eq";
    "(>= ?a ?b)" => "(|| (> ?a ?b) (== ?a ?b))"));

    // COMPLICATED ARITHMETIC
    rules.push(rewrite!("sub-to-add"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"));

    // Negation rules
    rules.push(rewrite!("double-neg"; "(* -1 (* -1 ?x))" => "?x"));
    rules.push(rewrite!("neg-zero"; "(* -1 0)" => "0"));

    // Factor common term out of a sum
    rules.push(rewrite!("factor-add";
        "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"));

    // Combine like terms (simple case)
    rules.push(rewrite!("add-same";
        "(+ ?x ?x)" => "(* 2 ?x)"));

    // Power/exponent simple identities
    rules.push(rewrite!("exp-1"; "(** ?x 1)" => "?x"));
    rules.push(rewrite!("exp-0"; "(** ?x 0)" => "1"));
    rules.push(rewrite!("mul-exp-same-base";
        "(* (** ?x ?a) (** ?x ?b))" => "(** ?x (+ ?a ?b))"));

    //tuples structs arrays
    // Project over tuple construction
    // assuming (project (tuple ?x ?y) 0) means first element, etc.
    rules.push(rewrite!("project-tuple-0";
    "(project (tuple ?x ?y) 0)" => "?x"));
    rules.push(rewrite!("project-tuple-1";
    "(project (tuple ?x ?y) 1)" => "?y"));

    // Same idea for structs if you encode them positionally
    rules.push(rewrite!("project-struct-0";
    "(project (struct ?x ?y) 0)" => "?x"));
    rules.push(rewrite!("project-struct-1";
    "(project (struct ?x ?y) 1)" => "?y"));

    // Array literal + index
    rules.push(rewrite!("arrayliteral-index-0";
    "(arrayindex (arrayliteral ?x ?y) 0)" => "?x"));
    rules.push(rewrite!("arrayliteral-index-1";
    "(arrayindex (arrayliteral ?x ?y) 1)" => "?y"));

    if cfg.commutation {
        rules.push(rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"));
        rules.push(rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"));
        rules.push(rewrite!("commute-bitor"; "(| ?a ?b)" => "(| ?b ?a)")); // not sure
        rules.push(rewrite!("commute-and"; "(&& ?a ?b)" => "(&& ?b ?a)"));
        rules.push(rewrite!("commute-or"; "(|| ?a ?b)" => "(|| ?b ?a)"));
    }

    if cfg.identity {
        rules.push(rewrite!("add-0"; "(+ ?a 0)" => "?a"));
        rules.push(rewrite!("mul-1"; "(* ?a 1)" => "?a"));
        rules.push(rewrite!("sub-zero"; "(- ?x 0)" => "?x"));
        rules.push(rewrite!("div-one"; "(/ ?x 1)" => "?x"));
    }

    if cfg.zero {
        rules.push(rewrite!("mul-0"; "(* ?a 0)" => "0"));
        rules.push(rewrite!("sub-id"; "(- ?x ?x)" => "0"));
    }

    if cfg.self_compare {
        rules.push(rewrite!("eq-self"; "(== ?x ?x)" => "true"));
        rules.push(rewrite!("lt-self"; "(< ?x ?x)" => "false"));
        rules.push(rewrite!("gt-self"; "(> ?x ?x)" => "false"));
        rules.push(rewrite!("lte-self"; "(<= ?x ?x)" => "true"));
        rules.push(rewrite!("gte-self"; "(>= ?x ?x)" => "true"));
    }

    if cfg.associativity {
        rules.push(rewrite!("add-assoc"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"));
        rules.push(rewrite!("mul-assoc"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"));
    }

    rules
}

fn symbol_id(egraph: &mut EGraph<TirLang, ()>, sym: Symbol) -> Id {
    egraph.add(TirLang::Var(sym))
}

fn type_id(egraph: &mut EGraph<TirLang, ()>, ty: &Type) -> Id {
    egraph.add(TirLang::Type(*ty))
}

fn expr_vec_ids(
    exprs: &[Expr],
    egraph: &mut EGraph<TirLang, ()>,
    type_span_map: &mut TypeSpanMap,
) -> Vec<Id> {
    exprs
        .iter()
        .map(|expr| expr_to_egg(expr, egraph, type_span_map))
        .collect()
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

fn env_exprs_id(
    exprs: &[Expr],
    egraph: &mut EGraph<TirLang, ()>,
    type_span_map: &mut TypeSpanMap,
) -> Id {
    let ids = expr_vec_ids(exprs, egraph, type_span_map);
    egraph.add(TirLang::EnvList(ids))
}

fn arglist_id(
    args: &[Expr],
    egraph: &mut EGraph<TirLang, ()>,
    type_span_map: &mut TypeSpanMap,
) -> Id {
    let ids = expr_vec_ids(args, egraph, type_span_map);
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

/// Convert a TIR expression into an egg `TirLang` node and return its e-graph Id. Also add type/span info to the map
fn expr_to_egg(
    expr: &Expr,
    egraph: &mut EGraph<TirLang, ()>,
    type_span_map: &mut TypeSpanMap,
) -> Id {
    let id = match &expr.kind {
        ExprKind::Var(sym) => symbol_id(egraph, *sym),
        ExprKind::Const(c) => match c {
            Const::Bool(b) => egraph.add(TirLang::Bool(*b)),
            Const::Int(i) => egraph.add(TirLang::Int(*i)),
            Const::Float(f) => egraph.add(TirLang::Float(*f)),
            Const::String(s) => egraph.add(TirLang::String(s.clone().to_string())),
        },
        ExprKind::Tuple(exprs) => {
            let ids = expr_vec_ids(exprs, egraph, type_span_map);
            egraph.add(TirLang::Tuple(ids))
        }
        ExprKind::Struct(exprs) => {
            let ids = expr_vec_ids(exprs, egraph, type_span_map);
            egraph.add(TirLang::Struct(ids))
        }
        ExprKind::Project { e, i } => {
            let expr_id = expr_to_egg(e, egraph, type_span_map);
            let index_id = egraph.add(TirLang::Index(*i));
            egraph.add(TirLang::Project([expr_id, index_id]))
        }
        ExprKind::BinOp { left, right, op } => {
            let left_id = expr_to_egg(left, egraph, type_span_map);
            let right_id = expr_to_egg(right, egraph, type_span_map);
            binop_id(*op, left_id, right_id, egraph)
        }
        ExprKind::Cast { e, ty } => {
            let expr_id = expr_to_egg(e, egraph, type_span_map);
            let ty_id = type_id(egraph, ty);
            egraph.add(TirLang::Cast([expr_id, ty_id]))
        }
        ExprKind::Call { f, args } => {
            let f_id = expr_to_egg(f, egraph, type_span_map);
            let args_id = arglist_id(args, egraph, type_span_map);
            egraph.add(TirLang::Call([f_id, args_id]))
        }
        ExprKind::MethodCall {
            receiver,
            method,
            args,
        } => {
            let receiver_id = expr_to_egg(receiver, egraph, type_span_map);
            let method_id = methodref_id(method, egraph);
            let args_id = arglist_id(args, egraph, type_span_map);
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
            let body_id = expr_to_egg(body, egraph, type_span_map);
            egraph.add(TirLang::Lambda([params_id, env_id, ret_ty_id, body_id]))
        }
        ExprKind::Closure { f, env } => {
            let fname_id = egraph.add(TirLang::Fname(*f));
            let env_id = env_exprs_id(env, egraph, type_span_map);
            egraph.add(TirLang::Closure([fname_id, env_id]))
        }
        ExprKind::Seq(e1, e2) => {
            let e1_id = expr_to_egg(e1, egraph, type_span_map);
            let e2_id = expr_to_egg(e2, egraph, type_span_map);
            egraph.add(TirLang::Seq([e1_id, e2_id]))
        }
        ExprKind::Let { name, ty, e1, e2 } => {
            let name_id = symbol_id(egraph, *name);
            let ty_id = type_id(egraph, ty);
            let e1_id = expr_to_egg(e1, egraph, type_span_map);
            let e2_id = expr_to_egg(e2, egraph, type_span_map);
            egraph.add(TirLang::Let([name_id, ty_id, e1_id, e2_id]))
        }
        ExprKind::Return(e) => {
            let e_id = expr_to_egg(e, egraph, type_span_map);
            egraph.add(TirLang::Return(e_id))
        }
        ExprKind::Loop(body) => {
            let body_id = expr_to_egg(body, egraph, type_span_map);
            egraph.add(TirLang::Loop(body_id))
        }
        ExprKind::While { cond, body } => {
            let cond_id = expr_to_egg(cond, egraph, type_span_map);
            let body_id = expr_to_egg(body, egraph, type_span_map);
            egraph.add(TirLang::While([cond_id, body_id]))
        }
        ExprKind::If { cond, then_, else_ } => {
            let cond_id = expr_to_egg(cond, egraph, type_span_map);
            let then_id = expr_to_egg(then_, egraph, type_span_map);
            let else_id = else_
                .as_ref()
                .map(|expr| expr_to_egg(expr, egraph, type_span_map))
                .unwrap_or_else(|| unit_id(egraph));
            egraph.add(TirLang::If([cond_id, then_id, else_id]))
        }
        ExprKind::Assign { dst, src } => {
            let dst_id = expr_to_egg(dst, egraph, type_span_map);
            let src_id = expr_to_egg(src, egraph, type_span_map);
            egraph.add(TirLang::Assign([dst_id, src_id]))
        }
        ExprKind::Break => egraph.add(TirLang::Break),
        ExprKind::ArrayLiteral(exprs) => {
            let ids = expr_vec_ids(exprs, egraph, type_span_map);
            egraph.add(TirLang::ArrayLiteral(ids))
        }
        ExprKind::ArrayIndex { array, index } => {
            let array_id = expr_to_egg(array, egraph, type_span_map);
            let index_id = expr_to_egg(index, egraph, type_span_map);
            egraph.add(TirLang::ArrayIndex([array_id, index_id]))
        }
        ExprKind::ArrayCopy { value, count } => {
            let value_id = expr_to_egg(value, egraph, type_span_map);
            let count_id = expr_to_egg(count, egraph, type_span_map);
            egraph.add(TirLang::ArrayCopy([value_id, count_id]))
        }
    };

    type_span_map.insert(id.into(), expr.ty, expr.span);
    id
}

/// Build a new type map from a RecExpr by looking up canonical IDs in the egraph.
fn rebuild_type_map_from_recexpr(
    rec_expr: &RecExpr<TirLang>,
    egraph: &EGraph<TirLang, ()>,
    original_type_map: &TypeSpanMap,
) -> TypeSpanMap {
    let mut new_map = TypeSpanMap::new(original_type_map.default_span);

    // Look up all nodes in the RecExpr to get their canonical IDs
    if let Some(canonical_ids) = egraph.lookup_expr_ids(rec_expr) {
        for (rec_expr_idx, canonical_id) in canonical_ids.iter().enumerate() {
            // Look up the type using the canonical ID from the original type map
            let (ty, span) = original_type_map.get((*canonical_id).into());
            new_map.insert(rec_expr_idx, ty, span);
        }
    }

    new_map
}

/// Convert a RecExpr node at the given index to an Expr.
fn rec_expr_to_expr(rec_expr: &RecExpr<TirLang>, idx: Id, type_span_map: &TypeSpanMap) -> Expr {
    let (ty, span) = type_span_map.get(idx.into());

    let kind = rec_expr_to_expr_kind(rec_expr, idx, type_span_map);
    Expr { kind, ty, span }
}

/// Convert a RecExpr node at the given index to an ExprKind.
fn rec_expr_to_expr_kind(
    rec_expr: &RecExpr<TirLang>,
    idx: Id,
    type_span_map: &TypeSpanMap,
) -> ExprKind {
    let node = &rec_expr[idx];
    match node {
        TirLang::Bool(b) => ExprKind::Const(Const::Bool(*b)),
        TirLang::Int(i) => ExprKind::Const(Const::Int(*i)),
        TirLang::Float(f) => ExprKind::Const(Const::Float(*f)),
        TirLang::String(s) => ExprKind::Const(Const::String(s.clone())),
        TirLang::Var(sym) => ExprKind::Var(*sym),
        TirLang::Tuple(ids) => {
            let exprs = ids_to_exprs(rec_expr, ids, type_span_map);
            ExprKind::Tuple(exprs)
        }
        TirLang::Struct(ids) => {
            let exprs = ids_to_exprs(rec_expr, ids, type_span_map);
            ExprKind::Struct(exprs)
        }
        TirLang::Project([e_idx, index_idx]) => {
            let e = rec_expr_to_expr(rec_expr, *e_idx, type_span_map);
            let index_node = &rec_expr[*index_idx];
            let i = match index_node {
                TirLang::Index(i) => *i,
                _ => panic!("Project second child must be Index"),
            };
            ExprKind::Project { e: Box::new(e), i }
        }
        TirLang::Add([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Add,
            }
        }
        TirLang::Sub([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Sub,
            }
        }
        TirLang::Mul([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Mul,
            }
        }
        TirLang::Div([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Div,
            }
        }
        TirLang::Rem([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Rem,
            }
        }
        TirLang::Exp([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Exp,
            }
        }
        TirLang::Eq([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Eq,
            }
        }
        TirLang::Neq([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Neq,
            }
        }
        TirLang::Lt([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Lt,
            }
        }
        TirLang::Gt([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Gt,
            }
        }
        TirLang::Le([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Le,
            }
        }
        TirLang::Ge([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Ge,
            }
        }
        TirLang::And([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::And,
            }
        }
        TirLang::Or([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Or,
            }
        }
        TirLang::Shl([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Shl,
            }
        }
        TirLang::Shr([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Shr,
            }
        }
        TirLang::BitAnd([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::BitAnd,
            }
        }
        TirLang::BitOr([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::BitOr,
            }
        }
        TirLang::Concat([left, right]) => {
            let left_expr = rec_expr_to_expr(rec_expr, *left, type_span_map);
            let right_expr = rec_expr_to_expr(rec_expr, *right, type_span_map);
            ExprKind::BinOp {
                left: Box::new(left_expr),
                right: Box::new(right_expr),
                op: Binop::Concat,
            }
        }
        TirLang::Cast([e_idx, ty_idx]) => {
            let e = rec_expr_to_expr(rec_expr, *e_idx, type_span_map);
            let ty_node = &rec_expr[*ty_idx];
            let ty = match ty_node {
                TirLang::Type(ty) => *ty,
                _ => panic!("Cast second child must be Type"),
            };
            ExprKind::Cast { e: Box::new(e), ty }
        }
        TirLang::Call([f_idx, args_idx]) => {
            let f = rec_expr_to_expr(rec_expr, *f_idx, type_span_map);
            let args_node = &rec_expr[*args_idx];
            let args = match args_node {
                TirLang::ArgList(ids) => ids_to_exprs(rec_expr, ids, type_span_map),
                _ => panic!("Call second child must be ArgList"),
            };
            ExprKind::Call {
                f: Box::new(f),
                args,
            }
        }
        TirLang::MethodCall([receiver_idx, method_idx, args_idx]) => {
            let receiver = rec_expr_to_expr(rec_expr, *receiver_idx, type_span_map);
            let method_node = &rec_expr[*method_idx];
            let method = match method_node {
                TirLang::MethodRef([interface_idx, method_name_idx]) => {
                    methodref_from_ids(rec_expr, *interface_idx, *method_name_idx)
                }
                _ => panic!("MethodCall second child must be MethodRef"),
            };
            let args_node = &rec_expr[*args_idx];
            let args = match args_node {
                TirLang::ArgList(ids) => ids_to_exprs(rec_expr, ids, type_span_map),
                _ => panic!("MethodCall third child must be ArgList"),
            };
            ExprKind::MethodCall {
                receiver: Box::new(receiver),
                method,
                args,
            }
        }
        TirLang::Lambda([params_idx, env_idx, ret_ty_idx, body_idx]) => {
            let params_node = &rec_expr[*params_idx];
            let params = match params_node {
                TirLang::ParamList(ids) => paramlist_from_ids(rec_expr, ids),
                _ => panic!("Lambda first child must be ParamList"),
            };
            let env_node = &rec_expr[*env_idx];
            let env = match env_node {
                TirLang::EnvList(ids) => paramlist_from_ids(rec_expr, ids),
                _ => panic!("Lambda second child must be EnvList"),
            };
            let ret_ty_node = &rec_expr[*ret_ty_idx];
            let ret_ty = match ret_ty_node {
                TirLang::Type(ty) => *ty,
                _ => panic!("Lambda third child must be Type"),
            };
            let body = rec_expr_to_expr(rec_expr, *body_idx, type_span_map);
            ExprKind::Lambda {
                params,
                env,
                ret_ty,
                body: Box::new(body),
            }
        }
        TirLang::Closure([fname_idx, env_idx]) => {
            let fname_node = &rec_expr[*fname_idx];
            let f = match fname_node {
                TirLang::Fname(sym) => *sym,
                _ => panic!("Closure first child must be Fname"),
            };
            let env_node = &rec_expr[*env_idx];
            let env = match env_node {
                TirLang::EnvList(ids) => ids_to_exprs(rec_expr, ids, type_span_map),
                _ => panic!("Closure second child must be EnvList"),
            };
            ExprKind::Closure { f, env }
        }
        TirLang::Seq([e1_idx, e2_idx]) => {
            let e1 = rec_expr_to_expr(rec_expr, *e1_idx, type_span_map);
            let e2 = rec_expr_to_expr(rec_expr, *e2_idx, type_span_map);
            ExprKind::Seq(Box::new(e1), Box::new(e2))
        }
        TirLang::Let([name_idx, ty_idx, e1_idx, e2_idx]) => {
            let name_node = &rec_expr[*name_idx];
            let name = match name_node {
                TirLang::Var(sym) => *sym,
                _ => panic!("Let first child must be Var"),
            };
            let ty_node = &rec_expr[*ty_idx];
            let ty = match ty_node {
                TirLang::Type(ty) => *ty,
                _ => panic!("Let second child must be Type"),
            };
            let e1 = rec_expr_to_expr(rec_expr, *e1_idx, type_span_map);
            let e2 = rec_expr_to_expr(rec_expr, *e2_idx, type_span_map);
            ExprKind::Let {
                name,
                ty,
                e1: Box::new(e1),
                e2: Box::new(e2),
            }
        }
        TirLang::Return(e_idx) => {
            let e = rec_expr_to_expr(rec_expr, *e_idx, type_span_map);
            ExprKind::Return(Box::new(e))
        }
        TirLang::Loop(body_idx) => {
            let body = rec_expr_to_expr(rec_expr, *body_idx, type_span_map);
            ExprKind::Loop(Box::new(body))
        }
        TirLang::While([cond_idx, body_idx]) => {
            let cond = rec_expr_to_expr(rec_expr, *cond_idx, type_span_map);
            let body = rec_expr_to_expr(rec_expr, *body_idx, type_span_map);
            ExprKind::While {
                cond: Box::new(cond),
                body: Box::new(body),
            }
        }
        TirLang::If([cond_idx, then_idx, else_idx]) => {
            let cond = rec_expr_to_expr(rec_expr, *cond_idx, type_span_map);
            let then_ = rec_expr_to_expr(rec_expr, *then_idx, type_span_map);
            let else_ = rec_expr_to_expr(rec_expr, *else_idx, type_span_map);
            // Check if else_ is a unit tuple (empty), which means no else branch
            let else_expr = match &else_.kind {
                ExprKind::Tuple(exprs) if exprs.is_empty() => None,
                _ => Some(Box::new(else_)),
            };
            ExprKind::If {
                cond: Box::new(cond),
                then_: Box::new(then_),
                else_: else_expr,
            }
        }
        TirLang::Assign([dst_idx, src_idx]) => {
            let dst = rec_expr_to_expr(rec_expr, *dst_idx, type_span_map);
            let src = rec_expr_to_expr(rec_expr, *src_idx, type_span_map);
            ExprKind::Assign {
                dst: Box::new(dst),
                src: Box::new(src),
            }
        }
        TirLang::Break => ExprKind::Break,
        TirLang::ArrayLiteral(ids) => {
            let exprs = ids_to_exprs(rec_expr, ids, type_span_map);
            ExprKind::ArrayLiteral(exprs)
        }
        TirLang::ArrayIndex([array_idx, index_idx]) => {
            let array = rec_expr_to_expr(rec_expr, *array_idx, type_span_map);
            let index = rec_expr_to_expr(rec_expr, *index_idx, type_span_map);
            ExprKind::ArrayIndex {
                array: Box::new(array),
                index: Box::new(index),
            }
        }
        TirLang::ArrayCopy([value_idx, count_idx]) => {
            let value = rec_expr_to_expr(rec_expr, *value_idx, type_span_map);
            let count = rec_expr_to_expr(rec_expr, *count_idx, type_span_map);
            ExprKind::ArrayCopy {
                value: Box::new(value),
                count: Box::new(count),
            }
        }
        TirLang::Index(_) => panic!("Index should not be used as root node"),
        TirLang::Type(_) => panic!("Type should not be used as root node"),
        TirLang::ArgList(_) => panic!("ArgList should not be used as root node"),
        TirLang::MethodRef(_) => panic!("MethodRef should not be used as root node"),
        TirLang::ParamList(_) => panic!("ParamList should not be used as root node"),
        TirLang::EnvList(_) => panic!("EnvList should not be used as root node"),
        TirLang::Fname(_) => panic!("Fname should not be used as root node"),
    }
}

/// Convert a Vec<Id> to a Vec<Expr>.
fn ids_to_exprs(rec_expr: &RecExpr<TirLang>, ids: &[Id], type_span_map: &TypeSpanMap) -> Vec<Expr> {
    ids.iter()
        .map(|&id| rec_expr_to_expr(rec_expr, id, type_span_map))
        .collect()
}

/// Reconstruct a ParamList from a Vec<Id> of Var nodes.
fn paramlist_from_ids(rec_expr: &RecExpr<TirLang>, ids: &[Id]) -> ParamList {
    ids.iter()
        .map(|&id| {
            let node = &rec_expr[id];
            match node {
                TirLang::Var(sym) => (*sym, Type::unit()), // Type info is lost, use unit
                _ => panic!("ParamList must contain Var nodes"),
            }
        })
        .collect()
}

/// Reconstruct a MethodRef from two Var node indices.
fn methodref_from_ids(
    rec_expr: &RecExpr<TirLang>,
    interface_idx: Id,
    method_name_idx: Id,
) -> MethodRef {
    let interface_node = &rec_expr[interface_idx];
    let interface = match interface_node {
        TirLang::Var(sym) => *sym,
        _ => panic!("MethodRef first child must be Var"),
    };
    let method_node = &rec_expr[method_name_idx];
    let method = match method_node {
        TirLang::Var(sym) => *sym,
        _ => panic!("MethodRef second child must be Var"),
    };
    MethodRef { interface, method }
}

pub struct TirCost;

impl CostFunction<TirLang> for TirCost {
    type Cost = usize;
    // cost = AST_size + op_complexity
    // cost(node) = operation_cost(node) + 1 + Σ cost(children)
    fn cost<C>(&mut self, enode: &TirLang, mut child: C) -> usize
    where
        C: FnMut(Id) -> usize,
    {
        let op_cost = match enode {
            TirLang::Int(_) | TirLang::Bool(_) | TirLang::Float(_) | TirLang::String(_) => 1,
            TirLang::Var(_) => 2,
            TirLang::Tuple(_) | TirLang::Struct(_) => 3,

            TirLang::Add(_)
            | TirLang::Sub(_)
            | TirLang::Mul(_)
            | TirLang::Div(_)
            | TirLang::Eq(_)
            | TirLang::Neq(_)
            | TirLang::Lt(_)
            | TirLang::Le(_)
            | TirLang::Gt(_)
            | TirLang::Ge(_) => 5,

            TirLang::Let(_) => 10,
            TirLang::Call(_) | TirLang::MethodCall(_) | TirLang::Cast(_) => 20,

            TirLang::Assign(_) | TirLang::Loop(_) | TirLang::While(_) | TirLang::If(_) => 50,

            _ => 100,
        };

        let mut total = op_cost + 1;
        enode.for_each(|id| {
            total += child(id);
        });
        total
    }
}

use egg::CostFunction;

pub struct TirSmartCost;

impl CostFunction<TirLang> for TirSmartCost {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &TirLang, mut child: C) -> usize
    where
        C: FnMut(Id) -> usize,
    {
        let mut total = 1;
        enode.for_each(|id| total += child(id));
        match enode {
            TirLang::Call(_) | TirLang::MethodCall(_) => {
                total += 5;
            }

            TirLang::ArrayLiteral(_) | TirLang::Struct(_) | TirLang::Tuple(_) => {
                total += 3;
            }
            _ => {}
        }
        total
    }
}

pub enum AnyCostFn {
    Ast(AstSize),
    Tir(TirCost),
    Smart(TirSmartCost),
}

impl egg::CostFunction<TirLang> for AnyCostFn {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &TirLang, mut child: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match self {
            AnyCostFn::Ast(inner) => inner.cost(enode, &mut child),
            AnyCostFn::Tir(inner) => inner.cost(enode, &mut child),
            AnyCostFn::Smart(inner) => inner.cost(enode, &mut child),
        }
    }
}

fn expr_to_recexpr(expr: &Expr, out: &mut RecExpr<TirLang>) -> Id {
    match &expr.kind {
        ExprKind::Const(Const::Int(i)) => out.add(TirLang::Int(*i)),
        ExprKind::Const(Const::Bool(b)) => out.add(TirLang::Bool(*b)),
        ExprKind::Const(Const::Float(f)) => out.add(TirLang::Float(*f)),
        ExprKind::Const(Const::String(s)) => out.add(TirLang::String(s.clone())),

        ExprKind::Var(sym) => out.add(TirLang::Var(*sym)),

        ExprKind::BinOp { left, right, op } => {
            let l = expr_to_recexpr(left, out);
            let r = expr_to_recexpr(right, out);

            match op {
                Binop::Add => out.add(TirLang::Add([l, r])),
                Binop::Sub => out.add(TirLang::Sub([l, r])),
                Binop::Mul => out.add(TirLang::Mul([l, r])),
                Binop::Div => out.add(TirLang::Div([l, r])),
                Binop::Eq => out.add(TirLang::Eq([l, r])),
                _ => out.add(TirLang::Tuple(vec![l, r])),
            }
        }

        ExprKind::Tuple(exprs) => {
            let ids = exprs.iter().map(|e| expr_to_recexpr(e, out)).collect();
            out.add(TirLang::Tuple(ids))
        }

        _ => out.add(TirLang::String(format!("unimplemented:{:?}", expr.kind))),
    }
}

pub fn main(tcx: Tcx, tir: Program) -> (Tcx, Program) {
    let mut egraph = EGraph::<TirLang, ()>::default();

    let mut function_data: Vec<(Function, Id)> = Vec::new();

    let mut type_span_map = TypeSpanMap::new(Span::DUMMY);

    for func in tir.functions() {
        let body_id = expr_to_egg(&func.body, &mut egraph, &mut type_span_map);

        // if std::env::var("RUN_BENCH").is_ok() {
        //     let mut recexpr = RecExpr::<TirLang>::default();
        //     let root = expr_to_recexpr(&func.body, &mut recexpr);

        //     let limits = vec![5, 10, 20, 40, 80];
        //     let csv_name = format!("bench_{}.csv", func.name);
        //     let extractor = Extractor::new(&egraph, AstSize);
        //     let (_c, recexpr) = extractor.find_best(body_id);

        //     sweep(recexpr.clone(), &limits, "smart", &csv_name);

        //     println!("[bench] wrote {}", csv_name);
        // }
        if std::env::var("RUN_BENCH").is_ok() {
            let configs = vec![
                ("full", AblationConfig::full()),
                ("no_comm", AblationConfig::no_comm()),
                ("no_identity", AblationConfig::no_identity()),
                ("no_zero", AblationConfig::no_zero()),
                ("no_self_compare", AblationConfig::no_self_compare()),
                ("no_assoc", AblationConfig::no_assoc()),
            ];
            let mut eg = EGraph::<TirLang, ()>::default();
            let mut map = TypeSpanMap::new(Span::DUMMY);
            let root = expr_to_egg(&func.body, &mut eg, &mut map);
            let full_rules = make_rules(AblationConfig::full());
            let runner = Runner::default()
                .with_egraph(eg)
                .with_iter_limit(30)
                .run(&full_rules);
            let extractor = egg::Extractor::new(&runner.egraph, AstSize);
            let (_c, recexpr) = extractor.find_best(root);
            let limits = vec![5, 10, 20, 40, 80];

            for (name, cfg) in configs {
                let rules = make_rules(cfg);
                let csv_path = format!("output_{}.csv", name);
                sweep_with_rules(recexpr.clone(), &limits, "smart", rules, &csv_path);
            }
            println!(
                "full: means full rule set; no_comm: no commutativity rules (like a + b → b + a); no_identity: no identity rules (like x + 0 → x); no_zero: no zero rules (x * 0 → 0); no_self_compare: no self-comparison rules ; no_assoc: no associativity rules (like (a + b) + c → a + (b + c));"
            );
        }

        function_data.push((func.clone(), body_id));
    }

    let iter_limit = REWRITE_ITER_LIMIT.with(|v| *v.borrow());
    let time_limit = REWRITE_TIME_LIMIT.with(|v| *v.borrow());

    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(iter_limit)
        .with_time_limit(std::time::Duration::from_secs(time_limit))
        .run(&make_rules(AblationConfig::full()));
    // .run(&make_rules());

    // let extractor = Extractor::new(&runner.egraph, AstSize);
    // let extractor = Extractor::new(&runner.egraph, TirSmartCost);
    let model = REWRITE_COST_MODEL.with(|v| v.borrow().clone());
    let mut cost_fn = match model.as_str() {
        "ast" => AnyCostFn::Ast(AstSize),
        "tir" => AnyCostFn::Tir(TirCost),
        "smart" => AnyCostFn::Smart(TirSmartCost),
        _ => AnyCostFn::Smart(TirSmartCost),
    };

    let extractor = Extractor::new(&runner.egraph, cost_fn);

    // Convert optimized expressions back to Expr and reconstruct functions
    let optimized_functions: Vec<Function> = function_data
        .into_iter()
        .map(|(mut func, body_id)| {
            let (_cost, optimized_rec_expr) = extractor.find_best(body_id);

            // Rebuild the type map for this RecExpr by looking up canonical IDs in the egraph
            let new_type_map =
                rebuild_type_map_from_recexpr(&optimized_rec_expr, &runner.egraph, &type_span_map);

            let root_idx = Id::from(optimized_rec_expr.as_ref().len() - 1);
            func.body = rec_expr_to_expr(&optimized_rec_expr, root_idx, &new_type_map);
            func
        })
        .collect();

    // println!(
    //     "Rewrote \n {:?} \n into \n {:?}",
    //     tir,
    //     Program::new(optimized_functions.clone())
    // );
    // println!(
    //     "[bench] nodes={} classes={} saturated={} time_limit_hit={}",
    //     runner.egraph.total_size(),
    //     runner.egraph.number_of_classes(),
    //     runner.stop_reason == Some(StopReason::Saturated),
    //     runner.stop_reason == Some(StopReason::TimeLimit),
    // );

    (tcx, Program::new(optimized_functions))
}
