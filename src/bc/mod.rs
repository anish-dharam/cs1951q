//! Bytecode (BC) code representation.

use miette::Result;
use strum::{Display, EnumString};

use crate::bc::dataflow::{constant_propagation, dead_code};

use self::types::{Function, Program};

mod dataflow;
mod lower;
mod print;
pub mod types;
mod visit;

pub use self::lower::lower;

#[derive(Clone, Copy, Debug, Display, EnumString)]
pub enum OptLevel {
    #[strum(serialize = "0")]
    NoOpt,
    #[strum(serialize = "1")]
    AllOpt,
}

/// Run correctness analyses on the whole program.
pub fn analyze(_prog: &Program) -> Result<()> {
    Ok(())
}

pub struct OptimizeOptions {
    pub opt_level: OptLevel,
}

/// Run optimizations on the whole program.
///
/// Optimizations are disabled at [`OptLevel::NoOpt`].
pub fn optimize(prog: &mut Program, opts: OptimizeOptions) {
    if matches!(opts.opt_level, OptLevel::AllOpt) {
        for func in prog.functions_mut() {
            optimize_func(func);
        }
    }
}

type Pass = Box<dyn Fn(&mut Function) -> bool>;

/// Run optimization passes to a fixed point.
fn optimize_func(func: &mut Function) {
    // let passes: Vec<Pass> = vec![Box::new(dead_code)];
    let passes: Vec<Pass> = vec![Box::new(constant_propagation), Box::new(dead_code)];

    loop {
        let mut changed = false;
        for pass in &passes {
            changed |= pass(func);
        }
        if !changed {
            break;
        }
    }
}
