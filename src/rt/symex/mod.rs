use anyhow::Result;
use smallvec::SmallVec;
use std::{collections::HashMap, collections::VecDeque};
use z3::{Solver, ast::Dynamic};

pub use crate::tir::types::Type;

use crate::{bc::types as bc, tir::Tcx, utils::Symbol};

/// Options for symbolic execution.
#[derive(Debug, Clone, Default)]
pub struct SymexOptions {
    pub max_steps: Option<u64>,
    // Placeholder for future configuration options
}

/// Main entry point for symbolic execution.
///
/// Finds all functions annotated with `#[symex]` and symbolically executes them
/// to check for assertion failures.
pub fn run(tcx: Tcx, prog: &bc::Program, opts: SymexOptions) -> Result<()> {
    let mut engine = SymexEngine::new();
    engine.set_max_steps(opts.max_steps);
    // TODO: Find all functions with #[symex] annotation
    let mut symex_funcs = Vec::new();
    for f in prog.functions() {
        if f.symex() {
            symex_funcs.push(f.clone());
        }
    }
    // TODO: For each function, call execute_function
    for f in symex_funcs {
        execute_function(&engine, &f, prog)?;
    }
    Ok(())
}

/// Symbolically executes a single function.
///
/// Returns an error if an assertion failure is detected.
fn execute_function(engine: &SymexEngine, func: &bc::Function, prog: &bc::Program) -> Result<()> {
    // TODO: Initialize abstract configuration with symbolic parameters

    let config = AbstractConfig {
        prog: prog.clone(),
        stack: todo!(), //zip(func.locals, [z3::Dynamic::new() * k]
        heap: HashMap::new(),
        path: Solver::new(), //z3 solver
        engine: engine,
        steps: 0,
    };
    // TODO: Run symbolic execution until all paths are explored or assertion fails
    let mut queue = VecDeque::from([config]);
    while !queue.is_empty() {
        let curr_config = queue.pop_front().unwrap();
        if matches!(engine.max_steps, Some(n) if curr_config.steps >= n) {
            continue;
        };
        match curr_config.step() {
            Ok(res) => {
                queue.extend(res);
            }
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok(())
}

/// The symbolic execution engine.
///
/// Manages the Z3 solver and provides operations for symbolic execution.
#[derive(Debug)]
pub struct SymexEngine {
    pub max_steps: Option<u64>,
    // TODO: Add Z3 Solver for path conditions
    // TODO: Add any other necessary state
}

impl SymexEngine {
    /// Creates a new symbolic execution engine.
    pub fn new() -> Self {
        SymexEngine { max_steps: None }
    }

    pub fn set_max_steps(&mut self, max_steps: Option<u64>) {
        self.max_steps = max_steps;
    }
}

/// An abstract configuration for symbolic execution.
///
/// Represents a single execution path with its state.
#[derive(Debug)]
pub struct AbstractConfig<'a> {
    /// The program being executed.
    pub prog: bc::Program,
    /// The stack of frames.
    pub stack: Stack,
    /// The global heap mapping pointers to symbolic expressions.
    pub heap: Heap,
    /// The path condition (accumulated facts from conditional jumps).
    pub path: Solver,

    pub engine: &'a SymexEngine,
    pub steps: u64,
}

impl AbstractConfig<'_> {
    /// Steps the configuration forward by one instruction.
    ///
    /// Returns a SmallVec containing:
    /// - One element if execution continues along a single path
    /// - Two elements if execution forks (e.g., at a conditional branch)
    ///
    /// Errors if an assertion failure is detected.
    pub fn step(mut self) -> Result<SmallVec<[Self; 2]>> {
        // TODO: Get current instruction from top frame
        // TODO: Execute instruction symbolically
        // TODO: Update path condition for conditional branches
        // TODO: Check assertions
        // TODO: Return next configuration(s)
        todo!("Implement step")
    }
}

/// A stack of frames for symbolic execution.
pub type Stack = Vec<Frame>;

/// A single stack frame.
#[derive(Debug)]
pub struct Frame {
    /// The function being executed.
    pub func: Symbol,
    /// Local variables mapped to symbolic expressions.
    pub locals: Locals,
}

/// Local variables as a mapping from variable names to symbolic expressions.
pub type Locals = HashMap<Symbol, Dynamic>;

// The heap is represented as a mapping from pointers (natural numbers) to symbolic expressions.
pub type Heap = HashMap<u64, Dynamic>;

fn get_z3_type(ty: Type) -> z3::Sort {
    match ty.kind() {
        bc::TypeKind::Int => z3::Sort::int(),
        bc::TypeKind::Float => z3::Sort::float32(),
        bc::TypeKind::Bool => z3::Sort::bool(),
        bc::TypeKind::String => z3::Sort::string(),
        bc::TypeKind::Tuple(items) => {
            // Create a tuple sort from the element types
            // let element_sorts: Vec<z3::Sort> =
            //     items.iter().map(|item_ty| get_z3_type(*item_ty)).collect();
            // let dt = z3::DatatypeBuilder::new("new_tuple")
            //     .variant(
            //         "one_variant",
            //         element_sorts
            //             .iter()
            //             .enumerate()
            //             .map(|(i, ty)| {
            //                 (
            //                     format!("field_{i}").as_str(),
            //                     z3::DatatypeAccessor::sort(ty),
            //                 )
            //             })
            //             .collect(),
            //     )
            //     .finish();
            // dt.sort
            todo!("tuples")
        }
        bc::TypeKind::Func { inputs, output } => {
            // Functions are not supported as values in symbolic execution
            todo!("Not functions")
        }
        bc::TypeKind::Array(elem_ty) => {
            // Create an array sort: Array(IndexSort, ElemSort)
            // For arrays, we use Z3's array theory where index is Int and element is the element type
            let index_sort = z3::Sort::int();
            let elem_sort = get_z3_type(*elem_ty);
            z3::Sort::array(&index_sort, &elem_sort)
        }
        bc::TypeKind::Struct(_) | bc::TypeKind::Interface(_) => {
            // Structs and interfaces are not supported (as per user request)
            todo!("Structs and interfaces not supported")
        }
        bc::TypeKind::Self_ | bc::TypeKind::Hole(_) => {
            // These are type system internals, not runtime types
            todo!("Unsupported type: {:?}", ty)
        }
    }
}
