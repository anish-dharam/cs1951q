//! Typed intermediate representation (TIR).
//!
//! The TIR is an expression-oriented IR with types attached to each expression node,
//! and with some AST features removed.

use self::types::Program;
use crate::ast;
use miette::Result;

use egg::{Extractor, Language, Rewrite, Runner, define_language, rewrite};

mod closure;
mod desugar;
mod print;
mod rewrite;
mod typeck;
pub use rewrite::{REWRITE_ITER_LIMIT, REWRITE_TIME_LIMIT, REWRITE_COST_MODEL};
pub mod types;
mod visit;

pub use typeck::{Globals, Tcx};


/// Type-checks a syntax tree into a [type context][Tcx] and a [typed program][Program].
pub fn typecheck(prog: ast::Ast) -> Result<(Tcx, Program)> {
    let mut tcx = Tcx::new(prog.num_holes);
    let mut tir = tcx.check(&prog.prog)?;
    desugar::desugar(&mut tir);
    closure::closure_conversion(&tcx, &mut tir);
    // println!("{:?}", tir.functions());
    Ok((tcx, tir))
}

pub fn rewrite_terms(tcx: Tcx, tir: Program) -> (Tcx, Program) {
    rewrite::main(tcx, tir)
}
