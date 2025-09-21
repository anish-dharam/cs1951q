//! Lowers TIR into bytecode.
//!
//! Each function is lowered in parallel. For a given function, its typed syntax tree is lowered
//! into the bytecode, which is a [three-address code](https://en.wikipedia.org/wiki/Three-address_code)
//! (or really an N-address code) control-flow graph (CFG). For example, a let-binding like this:
//!
//! ```text
//! let x = 1 + 2 + 3 in
//! ...
//! ```
//!
//! Will get lowered into code like this:
//!
//! ```text
//! bb0:
//!   t0 = 1 + 2
//!   x = t0 + 3
//!   ...
//! ```
//!
//! The bytecode CFG is represented as as a directed graph of [basic blocks][bc::BasicBlockData] using the [petgraph] library.
//! Each basic block contains a sequence of [statements][bc::Statement] which assign an [rvalue][bc::Rvalue] to a [place][bc::Place],
//! followed by a [terminator][bc::Terminator] which jumps to another basic block or returns.

use std::{
    cell::RefCell,
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use indexical::IndexedDomain;
use petgraph::{
    adj::NodeIndex,
    graph::DiGraph,
    visit::{Dfs, EdgeRef, Walker},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::types::{
    self as bc, AllocArgs, AllocKind, AllocLoc, Local, LocalIdx, ProjectionElem, Type,
};
use crate::{
    tir::{Tcx, types as tir},
    utils::Symbol,
};

/// Lowers a [TIR program][tir::Program] into [bytecode][bc::Program].
pub fn lower(tcx: &Tcx, prog: tir::Program) -> bc::Program {
    let lowered_funcs = prog
        .into_functions()
        .into_par_iter()
        .map(|func| lower_func(tcx, func))
        .collect();
    bc::Program::new(lowered_funcs)
}

fn lower_func(tcx: &Tcx, func: tir::Function) -> bc::Function {
    // First, lower the TIR function into a CFG with potentially unused basic blocks.
    let mut lowerer = LowerBody::new(tcx, &func);
    lowerer.lower_body(&func.body, func.ret_ty);

    // Find all basic blocks reachable from the start.
    let reachable = Dfs::new(&lowerer.graph, lowerer.start_block.into())
        .iter(&lowerer.graph)
        .map(bc::BasicBlockIdx::from)
        .collect::<BTreeSet<_>>();

    // Copy all the reachable blocks into the final CFG.
    let mut cfg = bc::Cfg::new();
    let mut node_map = HashMap::new();
    for block in reachable {
        let block_data = lowerer
            .graph
            .node_weight_mut(NodeIndex::from(block))
            .as_mut()
            .unwrap()
            .take()
            .unwrap();
        let new_node = cfg.add_node(block_data);
        node_map.insert(block, bc::BasicBlockIdx::from(new_node));
    }

    for block_data in cfg.node_weights_mut() {
        block_data.terminator.remap(&node_map);
    }

    for edge in lowerer.graph.edge_references() {
        let src = bc::BasicBlockIdx::from(edge.source());
        let dst = bc::BasicBlockIdx::from(edge.target());
        if let (Some(new_src), Some(new_dst)) = (node_map.get(&src), node_map.get(&dst)) {
            cfg.add_edge((*new_src).into(), (*new_dst).into(), ());
        }
    }

    // Wrap up the CFG with its metadata.
    bc::Function {
        name: func.name,
        num_params: func.params.len(),
        ret_ty: func.ret_ty,
        body: bc::Body::new(cfg),
        locals: Arc::new(lowerer.locals),
        annots: func.annots.clone(),
    }
}

/// A CFG with nodes under-construction, i.e., potentially missing [`BasicBlockData`][bc::BasicBlockData].
type PartialCfg = DiGraph<Option<bc::BasicBlock>, ()>;

/// Workspace for a lowering-in-progress of a particular function body.
struct LowerBody<'a> {
    tcx: &'a Tcx,
    locals: IndexedDomain<Local>,
    name_map: HashMap<Symbol, LocalIdx>,
    graph: PartialCfg,
    cur_block: Vec<bc::Statement>,
    start_block: bc::BasicBlockIdx,
    cur_loc: bc::BasicBlockIdx,
    loop_stack: Vec<bc::BasicBlockIdx>,
}

/// A destination for writing the result of a computation.
///
/// Allows the lowerer to initially write to an [operand][bc::Operand], and
/// switch to write to a [place][bc::Place] if needed.
#[derive(Clone, Copy)]
enum WriteDst<'a> {
    Operand(&'a RefCell<bc::Operand>),
    Place(bc::Place),
}

impl WriteDst<'_> {
    fn unwrap_place(&self) -> bc::Place {
        match self {
            WriteDst::Place(place) => *place,
            WriteDst::Operand(op) => {
                let bc::Operand::Place(place) = *op.borrow() else {
                    unreachable!()
                };
                place
            }
        }
    }
}

impl<'a> LowerBody<'a> {
    fn new(tcx: &'a Tcx, func: &tir::Function) -> Self {
        let mut graph = PartialCfg::new();
        let start_block = bc::BasicBlockIdx::from(graph.add_node(None));

        let mut lower = LowerBody {
            graph,
            tcx,
            locals: IndexedDomain::new(),
            name_map: HashMap::new(),
            cur_block: Vec::new(),
            cur_loc: start_block,
            start_block,
            loop_stack: Vec::new(),
        };

        for (name, ty) in &func.params {
            lower.add_local(*ty, Some(*name));
        }

        lower
    }

    fn add_local(&mut self, ty: Type, name: Option<Symbol>) -> LocalIdx {
        let local = self.locals.insert(Local { ty, name });
        if let Some(name) = name {
            self.name_map.insert(name, local);
        }
        local
    }

    fn get_local(&mut self, name: Symbol, ty: Type) -> LocalIdx {
        match self.name_map.get(&name) {
            Some(local) => *local,
            None => self.add_local(ty, Some(name)),
        }
    }

    fn gensym(&mut self, ty: Type) -> bc::Place {
        let local = self.add_local(ty, None);
        bc::Place::new(local, vec![], ty)
    }

    fn new_block(&mut self) -> bc::BasicBlockIdx {
        self.graph.add_node(None).into()
    }

    fn lower_body(&mut self, body: &tir::Expr, ret_ty: Type) {
        let dst = RefCell::new(bc::Operand::Place(self.gensym(ret_ty)));
        self.lower_expr_into(body, WriteDst::Operand(&dst));
        let exit = self.new_block();
        self.finish_block(
            exit,
            bc::Terminator {
                kind: bc::TerminatorKind::Return(dst.into_inner()),
                span: body.span,
            },
        );
    }

    fn finish_block(&mut self, new_block: bc::BasicBlockIdx, terminator: bc::Terminator) {
        let statements = self.cur_block.drain(..).collect::<Vec<_>>();
        let cur_block = bc::BasicBlock {
            statements,
            terminator,
        };
        match cur_block.terminator.kind() {
            bc::TerminatorKind::Jump(dst) => {
                self.graph.add_edge(self.cur_loc.into(), (*dst).into(), ());
            }
            bc::TerminatorKind::CondJump { true_, false_, .. } => {
                self.graph
                    .add_edge(self.cur_loc.into(), (*true_).into(), ());
                self.graph
                    .add_edge(self.cur_loc.into(), (*false_).into(), ());
            }
            bc::TerminatorKind::Return(..) => {}
        }
        *self.graph.node_weight_mut(self.cur_loc.into()).unwrap() = Some(cur_block);
        self.cur_loc = new_block;
    }

    fn lower_expr_into_tmp(&mut self, expr: &tir::Expr) -> bc::Operand {
        let place = self.gensym(expr.ty);
        let op = RefCell::new(bc::Operand::Place(place));
        self.lower_expr_into(expr, WriteDst::Operand(&op));
        op.into_inner()
    }

    fn lower_expr_into(&mut self, expr: &tir::Expr, op: WriteDst<'_>) {
        macro_rules! add_operand {
            ($op:expr) => {{
                match op {
                    WriteDst::Operand(op) => *op.borrow_mut() = $op,
                    WriteDst::Place(_) => add_assign!(bc::Rvalue::Operand($op)),
                }
            }};
        }

        macro_rules! add_assign {
            ($rvalue:expr) => {{
                self.cur_block.push(bc::Statement {
                    place: op.unwrap_place(),
                    rvalue: $rvalue,
                    span: expr.span,
                })
            }};
        }

        match &expr.kind {
            tir::ExprKind::Const(c) => add_operand!(bc::Operand::Const(c.clone())),
            tir::ExprKind::BinOp { left, right, op } => {
                let left_place = self.lower_expr_into_tmp(left);
                let right_place = self.lower_expr_into_tmp(right);
                add_assign!(bc::Rvalue::Binop {
                    op: *op,
                    left: left_place,
                    right: right_place,
                });
            }
            tir::ExprKind::Cast { e, ty } => {
                let e_op = self.lower_expr_into_tmp(e);
                add_assign!(bc::Rvalue::Cast { op: e_op, ty: *ty })
            }
            tir::ExprKind::If { cond, then_, else_ } => {
                let then_block = self.new_block();
                let else_block = self.new_block();
                let join_block = self.new_block();

                let cond_place = self.lower_expr_into_tmp(cond);
                self.finish_block(
                    then_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::CondJump {
                            cond: cond_place,
                            true_: then_block,
                            false_: else_block,
                        },
                        span: expr.span,
                    },
                );

                let dst_place = op.unwrap_place();
                let dst_op = WriteDst::Place(dst_place);

                self.lower_expr_into(then_, dst_op);
                self.finish_block(
                    else_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Jump(join_block),
                        span: then_.span,
                    },
                );

                let else_ = else_
                    .as_ref()
                    .expect("else-block should exist after desugaring");
                self.lower_expr_into(else_, dst_op);
                self.finish_block(
                    join_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Jump(join_block),
                        span: else_.span,
                    },
                );
            }
            tir::ExprKind::Loop(body) => {
                let header_block = self.new_block();
                let footer_block = self.new_block();

                self.finish_block(
                    header_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Jump(header_block),
                        span: expr.span,
                    },
                );

                self.loop_stack.push(footer_block);
                self.lower_expr_into(body, op);
                let _ = self.loop_stack.pop();

                self.finish_block(
                    footer_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Jump(header_block),
                        span: expr.span,
                    },
                );

                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Tuple,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::Lit(Vec::new())
                });
            }
            tir::ExprKind::Let { name, ty, e1, e2 } => {
                let local = self.get_local(*name, *ty);
                let let_op = bc::Place::new(local, vec![], *ty);
                self.lower_expr_into(e1, WriteDst::Place(let_op));

                self.lower_expr_into(e2, op);
            }
            tir::ExprKind::Var(name) => {
                let local = self.get_local(*name, expr.ty);
                add_operand!(bc::Operand::Place(bc::Place::new(local, vec![], expr.ty)))
            }
            tir::ExprKind::Tuple(els) => {
                let els = els
                    .iter()
                    .map(|el| self.lower_expr_into_tmp(el))
                    .collect::<Vec<_>>();
                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Tuple,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::Lit(els)
                });
            }
            tir::ExprKind::Struct(els) => {
                let els = els
                    .iter()
                    .map(|el| self.lower_expr_into_tmp(el))
                    .collect::<Vec<_>>();
                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Struct,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::Lit(els)
                });
            }
            tir::ExprKind::Project { e, i } => {
                let place = self.gensym(e.ty);
                self.lower_expr_into(e, WriteDst::Place(place));

                let projected = place.extend_projection(
                    [ProjectionElem::Field {
                        index: *i,
                        ty: e.ty,
                    }],
                    expr.ty,
                );
                add_assign!(bc::Rvalue::Operand(bc::Operand::Place(projected)))
            }
            tir::ExprKind::Call { f, args } => {
                let f_place = self.lower_expr_into_tmp(f);
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr_into_tmp(arg))
                    .collect::<Vec<_>>();
                add_assign!(bc::Rvalue::Call { f: f_place, args });
            }
            tir::ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let receiver_place = self.lower_expr_into_tmp(receiver);
                let args = args
                    .iter()
                    .map(|arg| self.lower_expr_into_tmp(arg))
                    .collect::<Vec<_>>();
                let method_idx = self.tcx.globals().intfs[&method.interface]
                    .iter()
                    .position(|sig| sig.name.value == method.method)
                    .unwrap();
                add_assign!(bc::Rvalue::MethodCall {
                    receiver: receiver_place,
                    method: method_idx,
                    args
                })
            }
            tir::ExprKind::Closure { f, env } => {
                let env = env.iter().map(|e| self.lower_expr_into_tmp(e)).collect();
                add_assign!(bc::Rvalue::Closure { f: *f, env })
            }
            tir::ExprKind::Seq(e1, e2) => {
                self.lower_expr_into_tmp(e1);
                self.lower_expr_into(e2, op);
            }
            tir::ExprKind::Return(e) => {
                let ret_op = self.lower_expr_into_tmp(e);

                let exit = self.new_block();
                self.finish_block(
                    exit,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Return(ret_op),
                        span: expr.span,
                    },
                );
            }
            tir::ExprKind::Assign { dst, src } => {
                let place = self.lower_place(dst);
                self.lower_expr_into(src, WriteDst::Place(place));
                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Tuple,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::Lit(Vec::new())
                })
            }
            tir::ExprKind::While { .. } => {
                unreachable!("while loops should be desugared away before bytecode lowering")
            }
            tir::ExprKind::Lambda { .. } => {
                unreachable!("lambdas should be eliminated by closure conversion")
            }
            tir::ExprKind::Break => {
                let target = *self
                    .loop_stack
                    .last()
                    .expect("break should only appear inside a loop after typechecking");
                let exit_block = self.new_block();
                self.finish_block(
                    exit_block,
                    bc::Terminator {
                        kind: bc::TerminatorKind::Jump(target),
                        span: expr.span,
                    },
                );
            }
            tir::ExprKind::ArrayLiteral(exprs) => {
                let elems = exprs
                    .iter()
                    .map(|e| self.lower_expr_into_tmp(e))
                    .collect::<Vec<_>>();
                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Array,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::Lit(elems)
                });
            }
            tir::ExprKind::ArrayIndex { array, index } => {
                let arr_place = self.gensym(array.ty);
                self.lower_expr_into(array, WriteDst::Place(arr_place));
                assert!(index.ty.equiv(&tir::TypeKind::Int));
                let idx_place = self.gensym(index.ty);
                self.lower_expr_into(index, WriteDst::Place(idx_place));

                let projected = arr_place.extend_projection(
                    [ProjectionElem::ArrayIndex {
                        index: bc::Operand::Place(idx_place),
                        ty: array.ty,
                    }],
                    expr.ty,
                );

                add_assign!(bc::Rvalue::Operand(bc::Operand::Place(projected)))
            }
            tir::ExprKind::ArrayCopy { value, count } => {
                let val_place = self.gensym(value.ty);
                self.lower_expr_into(value, WriteDst::Place(val_place));

                assert!(count.ty.equiv(&tir::TypeKind::Int));
                let count_place = self.gensym(count.ty);
                self.lower_expr_into(count, WriteDst::Place(count_place));

                add_assign!(bc::Rvalue::Alloc {
                    kind: AllocKind::Array,
                    loc: AllocLoc::Heap,
                    args: AllocArgs::ArrayCopy {
                        value: bc::Operand::Place(val_place),
                        count: bc::Operand::Place(count_place)
                    }
                });
            }
        }
    }

    fn lower_place_impl(&mut self, e: &tir::Expr, proj: &mut Vec<bc::ProjectionElem>) -> LocalIdx {
        match &e.kind {
            tir::ExprKind::Var(name) => self.get_local(*name, e.ty),
            tir::ExprKind::Project { e, i } => {
                proj.push(bc::ProjectionElem::Field {
                    index: *i,
                    ty: e.ty,
                });
                self.lower_place_impl(e, proj)
            }
            tir::ExprKind::ArrayIndex { array, index } => {
                proj.push(bc::ProjectionElem::ArrayIndex {
                    index: self.lower_expr_into_tmp(index),
                    ty: array.ty,
                });
                self.lower_place_impl(array, proj)
            }
            _ => panic!("invalid place: {e:?}"),
        }
    }

    fn lower_place(&mut self, e: &tir::Expr) -> bc::Place {
        let mut proj = Vec::new();
        let local = self.lower_place_impl(e, &mut proj);
        proj.reverse();
        bc::Place::new(local, proj, e.ty)
    }
}
