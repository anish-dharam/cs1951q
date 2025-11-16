use anyhow::{Context, Result};
use either::Either;
use smallvec::{SmallVec, smallvec};
use std::{collections::HashMap, collections::VecDeque, str::FromStr};
use z3::{
    SatResult, Solver, ast::Array, ast::Bool, ast::Dynamic, ast::Float as Z3Float, ast::Int,
    ast::String as Z3String,
};

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
pub fn run(_tcx: Tcx, prog: &bc::Program, opts: SymexOptions) -> Result<()> {
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
    println!("No assertion failures detected.");
    Ok(())
}

/// Symbolically executes a single function.
///
/// Returns an error if an assertion failure is detected.
fn execute_function(engine: &SymexEngine, func: &bc::Function, prog: &bc::Program) -> Result<()> {
    // Initialize all locals (parameters and temporaries)
    let mut locals = HashMap::new();
    for local_idx in func.locals.indices() {
        let local_data = func.locals.value(local_idx);
        // Skip function-typed locals (abstract functions not supported)
        if matches!(local_data.ty.kind(), bc::TypeKind::Func { .. }) {
            continue;
        }
        let local_symbol = AbstractConfig::get_local_symbol(func, local_idx);
        let sort = get_z3_type(local_data.ty);
        let sym_var = Dynamic::fresh_const(local_symbol.as_str(), &sort);
        locals.insert(local_symbol, sym_var);
    }

    // Create initial frame
    let initial_frame = Frame {
        func: func.name,
        locals,
        pc: bc::Location::START,
        return_dst: None,
    };

    let config = AbstractConfig {
        prog: prog.clone(),
        stack: vec![initial_frame],
        heap: HashMap::new(),
        path: Solver::new(), //z3 solver
        closures: HashMap::new(),
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
#[derive(Debug, Clone)]
pub struct AbstractConfig<'a> {
    /// The program being executed.
    pub prog: bc::Program,
    /// The stack of frames.
    pub stack: Stack,
    /// The global heap mapping pointers to symbolic expressions.
    pub heap: Heap,
    /// The path condition (accumulated facts from conditional jumps).
    pub path: Solver,
    /// Mapping from closure pointers to function names and environment pointers.
    pub closures: HashMap<u64, (Symbol, u64)>,

    pub engine: &'a SymexEngine,
    pub steps: u64,
}

impl AbstractConfig<'_> {
    /// Get symbol name for a local variable.
    fn get_local_symbol(func: &bc::Function, local_idx: bc::LocalIdx) -> Symbol {
        if let Some(name) = func.locals.value(local_idx).name {
            name
        } else {
            Symbol::new(format!("local_{}", local_idx.index()))
        }
    }

    /// Evaluate an operand to a symbolic expression.
    fn eval_operand(&self, op: &bc::Operand, locals: &Locals) -> Result<Dynamic> {
        match op {
            bc::Operand::Const(c) => match c {
                bc::Const::Bool(b) => Ok(Dynamic::from_ast(&Bool::from_bool(*b))),
                bc::Const::Int(i) => Ok(Dynamic::from_ast(&Int::from_i64(*i as i64))),
                bc::Const::Float(f) => Ok(Dynamic::from_ast(&Z3Float::from_f32(f.into_inner()))),
                bc::Const::String(s) => {
                    // For string constants, create a Z3 string constant from the literal value
                    // Use FromStr trait to parse the string literal into a Z3 string constant
                    let str_const = Z3String::from_str(s)
                        .map_err(|e| anyhow::anyhow!("Failed to create string constant: {}", e))?;
                    Ok(Dynamic::from_ast(&str_const))
                }
            },
            bc::Operand::Place(p) => self.eval_place(p, locals),
            bc::Operand::Func { .. } => {
                anyhow::bail!("Abstract functions not supported")
            }
        }
    }

    /// Evaluate a place to a symbolic expression.
    fn eval_place(&self, place: &bc::Place, locals: &Locals) -> Result<Dynamic> {
        // Get the function to look up local names
        let frame = self.stack.last().context("Empty stack")?;
        let func = self
            .prog
            .functions()
            .iter()
            .find(|f| f.name == frame.func)
            .context("Function not found")?;

        // Get base value from locals
        let local_symbol = Self::get_local_symbol(func, place.local);
        let mut value = locals
            .get(&local_symbol)
            .context(format!("Local {} not found", local_symbol))?
            .clone();

        // Follow projections
        for proj in &place.projection {
            match proj {
                bc::ProjectionElem::Field { index: _index, .. } => {
                    // Access tuple field using datatype accessor
                    // For now, simplified - we'll need to work with the AST directly
                    // This is a placeholder - proper implementation would use datatype accessors
                    anyhow::bail!(
                        "Field access not fully implemented - need proper datatype handling"
                    )
                }
                bc::ProjectionElem::ArrayIndex { index, .. } => {
                    // Array indexing: value[index]
                    let index_val = self.eval_operand(index, locals)?;
                    let index_int = index_val.as_int().context("Array index must be integer")?;
                    let array = value.as_array().context("Expected array for indexing")?;
                    value = Dynamic::from_ast(&array.select(&index_int));
                }
            }
        }

        Ok(value)
    }

    /// Evaluate an rvalue to a symbolic expression, returning updated heap.
    fn eval_rvalue(&self, rv: &bc::Rvalue, locals: &Locals) -> Result<(Dynamic, Heap)> {
        match rv {
            bc::Rvalue::Operand(op) => Ok((self.eval_operand(op, locals)?, self.heap.clone())),
            bc::Rvalue::Binop { op, left, right } => {
                let left_val = self.eval_operand(left, locals)?;
                let right_val = self.eval_operand(right, locals)?;

                let result = match op {
                    crate::tir::types::Binop::Add => {
                        let l = left_val.as_int().context("Add requires int")?;
                        let r = right_val.as_int().context("Add requires int")?;
                        Dynamic::from_ast(&Int::add(&[&l, &r]))
                    }
                    crate::tir::types::Binop::Sub => {
                        let l = left_val.as_int().context("Sub requires int")?;
                        let r = right_val.as_int().context("Sub requires int")?;
                        Dynamic::from_ast(&Int::sub(&[&l, &r]))
                    }
                    crate::tir::types::Binop::Mul => {
                        let l = left_val.as_int().context("Mul requires int")?;
                        let r = right_val.as_int().context("Mul requires int")?;
                        Dynamic::from_ast(&Int::mul(&[&l, &r]))
                    }
                    crate::tir::types::Binop::Div => {
                        let l = left_val.as_int().context("Div requires int")?;
                        let r = right_val.as_int().context("Div requires int")?;
                        Dynamic::from_ast(&l.div(&r))
                    }
                    crate::tir::types::Binop::Rem => {
                        let l = left_val.as_int().context("Rem requires int")?;
                        let r = right_val.as_int().context("Rem requires int")?;
                        Dynamic::from_ast(&l.rem(&r))
                    }
                    crate::tir::types::Binop::Lt => {
                        let l = left_val.as_int().context("Lt requires int")?;
                        let r = right_val.as_int().context("Lt requires int")?;
                        Dynamic::from_ast(&l.lt(&r))
                    }
                    crate::tir::types::Binop::Le => {
                        let l = left_val.as_int().context("Le requires int")?;
                        let r = right_val.as_int().context("Le requires int")?;
                        Dynamic::from_ast(&l.le(&r))
                    }
                    crate::tir::types::Binop::Gt => {
                        let l = left_val.as_int().context("Gt requires int")?;
                        let r = right_val.as_int().context("Gt requires int")?;
                        Dynamic::from_ast(&l.gt(&r))
                    }
                    crate::tir::types::Binop::Ge => {
                        let l = left_val.as_int().context("Ge requires int")?;
                        let r = right_val.as_int().context("Ge requires int")?;
                        Dynamic::from_ast(&l.ge(&r))
                    }
                    crate::tir::types::Binop::Eq => Dynamic::from_ast(&left_val.eq(&right_val)),
                    crate::tir::types::Binop::Neq => {
                        Dynamic::from_ast(&left_val.eq(&right_val).not())
                    }
                    crate::tir::types::Binop::And => {
                        let l = left_val.as_bool().context("And requires bool")?;
                        let r = right_val.as_bool().context("And requires bool")?;
                        Dynamic::from_ast(&Bool::and(&[&l, &r]))
                    }
                    crate::tir::types::Binop::Or => {
                        let l = left_val.as_bool().context("Or requires bool")?;
                        let r = right_val.as_bool().context("Or requires bool")?;
                        Dynamic::from_ast(&Bool::or(&[&l, &r]))
                    }
                    crate::tir::types::Binop::Exp => {
                        anyhow::bail!("Exponentiation not supported in symbolic execution")
                    }
                    crate::tir::types::Binop::Shl
                    | crate::tir::types::Binop::Shr
                    | crate::tir::types::Binop::BitAnd
                    | crate::tir::types::Binop::BitOr => {
                        anyhow::bail!("Bitwise operations not supported in symbolic execution")
                    }
                    crate::tir::types::Binop::Concat => {
                        let l = left_val.as_string().context("Concat requires string")?;
                        let r = right_val.as_string().context("Concat requires string")?;
                        Dynamic::from_ast(&Z3String::concat(&[&l, &r]))
                    }
                };

                Ok((result, self.heap.clone()))
            }
            bc::Rvalue::Alloc { kind, args, .. } => {
                let mut new_heap = self.heap.clone();
                let ptr = new_heap.len() as u64;

                match args {
                    bc::AllocArgs::Lit(elements) => {
                        let mut elem_vals = Vec::new();
                        for elem in elements {
                            elem_vals.push(self.eval_operand(elem, locals)?);
                        }

                        match kind {
                            bc::AllocKind::Tuple => {
                                if elem_vals.is_empty() {
                                    // Empty tuple - unit type
                                    let unit_sort = get_z3_type(Type::unit());
                                    let tuple_val =
                                        Dynamic::fresh_const(&format!("tuple_{}", ptr), &unit_sort);
                                    new_heap.insert(ptr, tuple_val.clone());
                                    Ok((Dynamic::from_ast(&Int::from_u64(ptr)), new_heap))
                                } else {
                                    // Create tuple datatype
                                    let elem_tys: Vec<Type> =
                                        elements.iter().map(|e| e.ty()).collect();
                                    let tuple_sort = get_z3_type(Type::tuple(elem_tys));

                                    // Create tuple value - simplified for now
                                    // We'll create a fresh symbolic variable for the tuple
                                    let tuple_val = Dynamic::fresh_const(
                                        &format!("tuple_{}", ptr),
                                        &tuple_sort,
                                    );
                                    new_heap.insert(ptr, tuple_val.clone());
                                    Ok((Dynamic::from_ast(&Int::from_u64(ptr)), new_heap))
                                }
                            }
                            bc::AllocKind::Array => {
                                // Create constant array
                                let index_sort = z3::Sort::int();
                                let const_val = if let Some(elem) = elem_vals.first() {
                                    elem.clone()
                                } else {
                                    // Empty array - use default value based on type
                                    let elem_ty =
                                        elements.first().map(|e| e.ty()).unwrap_or(Type::int());
                                    let elem_sort = get_z3_type(elem_ty);
                                    Dynamic::fresh_const(
                                        &format!("array_default_{}", ptr),
                                        &elem_sort,
                                    )
                                };

                                let array = Array::const_array(&index_sort, &const_val);

                                new_heap.insert(ptr, Dynamic::from_ast(&array));
                                Ok((Dynamic::from_ast(&Int::from_u64(ptr)), new_heap))
                            }
                            bc::AllocKind::Struct => {
                                anyhow::bail!("Structs not supported")
                            }
                        }
                    }
                    bc::AllocArgs::ArrayCopy {
                        value,
                        count: _count,
                    } => {
                        let value_val = self.eval_operand(value, locals)?;

                        // Create constant array with the value
                        let index_sort = z3::Sort::int();
                        let array = Array::const_array(&index_sort, &value_val);

                        new_heap.insert(ptr, Dynamic::from_ast(&array));
                        Ok((Dynamic::from_ast(&Int::from_u64(ptr)), new_heap))
                    }
                }
            }
            bc::Rvalue::Closure { f: _f, env } => {
                // Evaluate environment operands
                let env_vals: Vec<Dynamic> = env
                    .iter()
                    .map(|op| self.eval_operand(op, locals))
                    .collect::<Result<_>>()?;

                // Allocate closure on heap
                let mut new_heap = self.heap.clone();
                let closure_ptr = new_heap.len() as u64;

                // Store environment tuple on heap if not empty
                let _env_ptr = if !env_vals.is_empty() {
                    let env_ptr = closure_ptr + 1;
                    // Create tuple for environment - simplified
                    let env_tys: Vec<Type> = env.iter().map(|op| op.ty()).collect();
                    let env_tuple_sort = get_z3_type(Type::tuple(env_tys));
                    let env_tuple =
                        Dynamic::fresh_const(&format!("env_{}", closure_ptr), &env_tuple_sort);
                    new_heap.insert(env_ptr, env_tuple);
                    env_ptr
                } else {
                    // Empty environment - use unit tuple
                    let env_ptr = closure_ptr + 1;
                    let unit_sort = get_z3_type(Type::unit());
                    let env_tuple =
                        Dynamic::fresh_const(&format!("env_{}", closure_ptr), &unit_sort);
                    new_heap.insert(env_ptr, env_tuple);
                    env_ptr
                };

                // Store closure pointer value in heap
                let closure_val = Dynamic::from_ast(&Int::from_u64(closure_ptr));
                new_heap.insert(closure_ptr, closure_val.clone());

                // Note: closure metadata (function name, env_ptr) will be stored in closures map
                // This is handled when assigning the closure to a place
                Ok((Dynamic::from_ast(&Int::from_u64(closure_ptr)), new_heap))
            }
            bc::Rvalue::Call { .. } => {
                // Calls are handled in step()
                anyhow::bail!("Call should be handled in step()")
            }
            bc::Rvalue::Cast { op, .. } => {
                // Type casting - just return the value
                Ok((self.eval_operand(op, locals)?, self.heap.clone()))
            }
            bc::Rvalue::MethodCall { .. } => {
                anyhow::bail!("Method calls not supported")
            }
        }
    }

    /// Assign a value to a place, returning updated locals and heap.
    fn assign_place(
        &self,
        place: &bc::Place,
        value: Dynamic,
        locals: &Locals,
        heap: &Heap,
    ) -> Result<(Locals, Heap)> {
        // Get the function to look up local names
        let frame = self.stack.last().context("Empty stack")?;
        let func = self
            .prog
            .functions()
            .iter()
            .find(|f| f.name == frame.func)
            .context("Function not found")?;

        let mut new_locals = locals.clone();
        let mut new_heap = heap.clone();

        if place.projection.is_empty() {
            // Simple variable assignment
            let local_symbol = Self::get_local_symbol(func, place.local);
            new_locals.insert(local_symbol, value);
        } else {
            // Need to handle projections
            let local_symbol = Self::get_local_symbol(func, place.local);
            let base_val = locals
                .get(&local_symbol)
                .context(format!("Local {} not found", local_symbol))?
                .clone();

            // Handle projections - need to update heap for array/tuple fields
            if place.projection.len() == 1 {
                match &place.projection[0] {
                    bc::ProjectionElem::Field { index: _index, .. } => {
                        // Update field in tuple - simplified for now
                        if let Some(ptr_val) = base_val.as_int() {
                            if let Some(ptr_u64) = ptr_val.as_u64() {
                                // Simplified: just update the heap entry
                                // Proper implementation would reconstruct tuple with updated field
                                new_heap.insert(ptr_u64, value);
                            }
                        }
                    }
                    bc::ProjectionElem::ArrayIndex { index, .. } => {
                        // Update array element
                        let index_val = self.eval_operand(index, locals)?;
                        let index_int =
                            index_val.as_int().context("Array index must be integer")?;

                        // Check if base is a pointer (heap value)
                        if let Some(ptr_val) = base_val.as_int() {
                            if let Some(ptr_u64) = ptr_val.as_u64() {
                                if let Some(array_val) = new_heap.get(&ptr_u64) {
                                    let array = array_val.as_array().context("Expected array")?;
                                    let updated = array.store(&index_int, &value);
                                    new_heap.insert(ptr_u64, Dynamic::from_ast(&updated));
                                }
                            }
                        }
                    }
                }
            } else {
                // Nested projections - simplified handling
                new_locals.insert(local_symbol, value);
            }
        }

        Ok((new_locals, new_heap))
    }

    /// Steps the configuration forward by one instruction.
    ///
    /// Returns a SmallVec containing:
    /// - One element if execution continues along a single path
    /// - Two elements if execution forks (e.g., at a conditional branch)
    ///
    /// Errors if an assertion failure is detected.
    pub fn step(mut self) -> Result<SmallVec<[Self; 2]>> {
        // Get current frame info before borrowing
        let frame_func = {
            let frame = self.stack.last().context("Empty stack")?;
            frame.func
        };
        let frame_pc = {
            let frame = self.stack.last().context("Empty stack")?;
            frame.pc
        };

        // Get function from program
        let func = self
            .prog
            .functions()
            .iter()
            .find(|f| f.name == frame_func)
            .context(format!("Function {} not found", frame_func))?;

        // Get current instruction
        let instr = func.body.instr(frame_pc);

        // Get locals snapshot
        let locals_snapshot = {
            let frame = self.stack.last().context("Empty stack")?;
            frame.locals.clone()
        };

        match instr {
            Either::Left(stmt) => {
                // Statement: p = rv
                // Handle calls specially (for assertions and closures)
                if let bc::Rvalue::Call { f, args } = &stmt.rvalue {
                    // Need to handle call without borrowing self mutably
                    // Clone what we need and handle call
                    let mut call_config = self.clone();
                    let result = call_config.handle_call(&stmt.place, f, args, &locals_snapshot)?;
                    return Ok(result);
                }

                // Evaluate rvalue
                let (value, new_heap) = self.eval_rvalue(&stmt.rvalue, &locals_snapshot)?;

                // If this is a closure creation, track it
                if let bc::Rvalue::Closure { f, env: _ } = &stmt.rvalue {
                    if let Some(ptr_val) = value.as_int() {
                        if let Some(ptr_u64) = ptr_val.as_u64() {
                            // Store closure metadata: ptr -> (function_name, env_ptr)
                            // env_ptr is ptr + 1
                            let env_ptr = ptr_u64 + 1;
                            self.closures.insert(ptr_u64, (*f, env_ptr));
                        }
                    }
                }

                // Assign to place
                let (new_locals, final_heap) =
                    self.assign_place(&stmt.place, value, &locals_snapshot, &new_heap)?;

                // Update frame
                let frame = self.stack.last_mut().unwrap();
                frame.locals = new_locals;
                frame.pc.instr += 1;
                self.heap = final_heap;
                self.steps += 1;

                Ok(smallvec![self])
            }
            Either::Right(term) => {
                match &term.kind {
                    bc::TerminatorKind::Jump(target) => {
                        let frame = self.stack.last_mut().unwrap();
                        frame.pc = target.entry();
                        self.steps += 1;
                        Ok(smallvec![self])
                    }
                    bc::TerminatorKind::CondJump {
                        cond,
                        true_,
                        false_,
                    } => {
                        // Evaluate condition
                        let cond_val = self.eval_operand(cond, &locals_snapshot)?;
                        let cond_bool = cond_val.as_bool().context("Condition must be boolean")?;

                        let mut results = SmallVec::new();

                        // True branch
                        let mut true_config = self.clone();
                        true_config.path.assert(&cond_bool);
                        if true_config.path.check() == SatResult::Sat {
                            let true_frame = true_config.stack.last_mut().unwrap();
                            true_frame.pc = true_.entry();
                            true_config.steps += 1;
                            results.push(true_config);
                        }

                        // False branch - clone self before moving
                        let mut false_config = self.clone();
                        false_config.path.assert(&cond_bool.not());
                        if false_config.path.check() == SatResult::Sat {
                            let false_frame = false_config.stack.last_mut().unwrap();
                            false_frame.pc = false_.entry();
                            false_config.steps += 1;
                            results.push(false_config);
                        }

                        // If both branches unsatisfiable, return empty
                        Ok(results)
                    }
                    bc::TerminatorKind::Return(ret_op) => {
                        if self.stack.len() > 1 {
                            // Evaluate return value
                            let ret_val = self.eval_operand(ret_op, &locals_snapshot)?;

                            // Get the return destination from the current frame before popping
                            let return_dst = {
                                let current_frame = self.stack.last().context("Empty stack")?;
                                current_frame.return_dst.clone()
                            };

                            // Pop current frame to get caller
                            self.stack.pop();

                            // Get caller frame and locals
                            let caller_frame = self.stack.last().context("Empty stack")?;
                            let caller_locals = caller_frame.locals.clone();

                            // Assign return value to destination place if we have one
                            if let Some(dst) = return_dst {
                                let (new_locals, new_heap) =
                                    self.assign_place(&dst, ret_val, &caller_locals, &self.heap)?;
                                let caller_frame = self.stack.last_mut().unwrap();
                                caller_frame.locals = new_locals;
                                self.heap = new_heap;
                            }

                            // Continue execution in caller
                            let caller_frame = self.stack.last_mut().unwrap();
                            caller_frame.pc.instr += 1;
                            self.steps += 1;
                            Ok(smallvec![self])
                        } else {
                            // Top-level return - function complete
                            self.steps += 1;
                            Ok(SmallVec::new())
                        }
                    }
                }
            }
        }
    }

    /// Handle a function call, checking for assertions and handling closures.
    fn handle_call(
        &mut self,
        dst: &bc::Place,
        f: &bc::Operand,
        args: &[bc::Operand],
        locals: &Locals,
    ) -> Result<SmallVec<[Self; 2]>> {
        // Evaluate arguments
        let mut arg_vals = Vec::new();
        for arg in args {
            arg_vals.push(self.eval_operand(arg, locals)?);
        }

        // Check if direct function call (Operand::Func)
        if let bc::Operand::Func { f: func_name, .. } = f {
            // Regular function call - look up function
            let called_func = match self
                .prog
                .functions()
                .iter()
                .find(|func| func.name == *func_name)
            {
                Some(f) => f,
                None => {
                    // Function not found - check if it's a standard library function like assert
                    // Try multiple comparison methods to be sure
                    let func_name_str = format!("{}", func_name);
                    // Also try direct Symbol comparison
                    let assert_sym = Symbol::new("assert");
                    if func_name_str == "assert" || *func_name == assert_sym {
                        // Handle assert (standard library function)
                        if let Some(condition) = arg_vals.first() {
                            let cond_bool = condition
                                .as_bool()
                                .context("Assert condition must be boolean")?;
                            let mut test_solver = self.path.clone();
                            test_solver.assert(&cond_bool.not());
                            if test_solver.check() == SatResult::Sat {
                                println!("test_solver: {:?}", test_solver);
                                anyhow::bail!(
                                    "Assertion failure detected (condition can be false)"
                                );
                            }
                            // Assertion always holds - continue (caller will update pc)
                            let frame = self.stack.last_mut().unwrap();
                            frame.pc.instr += 1;
                            self.steps += 1;
                            return Ok(smallvec![self.clone()]);
                        }
                        // No condition provided - continue anyway
                        let frame = self.stack.last_mut().unwrap();
                        frame.pc.instr += 1;
                        self.steps += 1;
                        return Ok(smallvec![self.clone()]);
                    }
                    return Err(anyhow::anyhow!("Function {} not found", func_name));
                }
            };

            // Create new frame - initialize all locals
            let mut new_locals = HashMap::new();
            // Initialize all locals in the function (not just parameters)
            for local_idx in called_func.locals.indices() {
                let local_data = called_func.locals.value(local_idx);
                // Skip function-typed locals (abstract functions not supported)
                if matches!(local_data.ty.kind(), bc::TypeKind::Func { .. }) {
                    continue;
                }
                let local_symbol = Self::get_local_symbol(called_func, local_idx);
                // Check if this is a parameter
                if local_idx.index() < called_func.num_params {
                    // It's a parameter - use the argument value
                    let param_index = local_idx.index();
                    if param_index < arg_vals.len() {
                        new_locals.insert(local_symbol, arg_vals[param_index].clone());
                    } else {
                        // Missing argument - initialize to fresh symbolic variable
                        let sort = get_z3_type(local_data.ty);
                        let sym_var = Dynamic::fresh_const(local_symbol.as_str(), &sort);
                        new_locals.insert(local_symbol, sym_var);
                    }
                } else {
                    // It's a temporary/local variable - initialize to fresh symbolic variable
                    let sort = get_z3_type(local_data.ty);
                    let sym_var = Dynamic::fresh_const(local_symbol.as_str(), &sort);
                    new_locals.insert(local_symbol, sym_var);
                }
            }

            let new_frame = Frame {
                func: *func_name,
                locals: new_locals,
                pc: bc::Location::START,
                return_dst: Some(dst.clone()),
            };

            self.stack.push(new_frame);
            self.steps += 1;
            Ok(smallvec![self.clone()])
        } else if let bc::Operand::Place(closure_place) = f {
            // Closure call - evaluate closure place to get closure value (pointer)
            let closure_ptr_val = self.eval_place(closure_place, locals)?;
            let closure_ptr = closure_ptr_val
                .as_int()
                .context("Closure must be a pointer")?
                .as_u64()
                .context("Closure pointer must be u64")?;

            // Look up closure metadata
            let (func_name, env_ptr) = self
                .closures
                .get(&closure_ptr)
                .context(format!("Closure at ptr {} not found", closure_ptr))?
                .clone();

            // Get environment from heap
            let env_tuple_val = self
                .heap
                .get(&env_ptr)
                .context(format!("Environment at ptr {} not found", env_ptr))?
                .clone();

            // Look up the function
            let called_func = match self
                .prog
                .functions()
                .iter()
                .find(|func| func.name == func_name)
            {
                Some(f) => f,
                None => {
                    // Function not found - this shouldn't happen for closures, but handle assert just in case
                    let func_name_str = format!("{}", func_name);
                    let assert_sym = Symbol::new("assert");
                    if func_name_str == "assert" || func_name == assert_sym {
                        // Handle assert (standard library function)
                        if let Some(condition) = arg_vals.first() {
                            let cond_bool = condition
                                .as_bool()
                                .context("Assert condition must be boolean")?;
                            let mut test_solver = self.path.clone();
                            test_solver.assert(&cond_bool.not());
                            if test_solver.check() == SatResult::Sat {
                                anyhow::bail!(
                                    "Assertion failure detected (condition can be false)"
                                );
                            }
                            let frame = self.stack.last_mut().unwrap();
                            frame.pc.instr += 1;
                            self.steps += 1;
                            return Ok(smallvec![self.clone()]);
                        }
                        let frame = self.stack.last_mut().unwrap();
                        frame.pc.instr += 1;
                        self.steps += 1;
                        return Ok(smallvec![self.clone()]);
                    }
                    return Err(anyhow::anyhow!("Function {} not found", func_name));
                }
            };

            // Create new frame with function and environment
            // The function's first parameter is the environment (from closure conversion)
            let mut new_locals = HashMap::new();

            // Initialize all locals in the function
            for local_idx in called_func.locals.indices() {
                let local_data = called_func.locals.value(local_idx);
                // Skip function-typed locals (abstract functions not supported)
                if matches!(local_data.ty.kind(), bc::TypeKind::Func { .. }) {
                    continue;
                }
                let local_symbol = Self::get_local_symbol(called_func, local_idx);

                if local_idx.index() < called_func.num_params {
                    // It's a parameter
                    if local_idx.index() == 0 {
                        // First parameter is the environment
                        new_locals.insert(local_symbol, env_tuple_val.clone());
                    } else {
                        // Remaining parameters come from call arguments
                        let arg_index = local_idx.index() - 1;
                        if arg_index < arg_vals.len() {
                            new_locals.insert(local_symbol, arg_vals[arg_index].clone());
                        } else {
                            // Missing argument - initialize to fresh symbolic variable
                            let sort = get_z3_type(local_data.ty);
                            let sym_var = Dynamic::fresh_const(local_symbol.as_str(), &sort);
                            new_locals.insert(local_symbol, sym_var);
                        }
                    }
                } else {
                    // It's a temporary/local variable - initialize to fresh symbolic variable
                    let sort = get_z3_type(local_data.ty);
                    let sym_var = Dynamic::fresh_const(local_symbol.as_str(), &sort);
                    new_locals.insert(local_symbol, sym_var);
                }
            }

            let new_frame = Frame {
                func: func_name,
                locals: new_locals,
                pc: bc::Location::START,
                return_dst: Some(dst.clone()),
            };

            self.stack.push(new_frame);
            self.steps += 1;
            Ok(smallvec![self.clone()])
        } else {
            anyhow::bail!("Invalid call operand: {:?}", f)
        }
    }
}

/// A stack of frames for symbolic execution.
pub type Stack = Vec<Frame>;

/// A single stack frame.
#[derive(Debug, Clone)]
pub struct Frame {
    /// The function being executed.
    pub func: Symbol,
    /// Local variables mapped to symbolic expressions.
    pub locals: Locals,
    /// Program counter: current instruction location.
    pub pc: bc::Location,
    /// Destination place for return value (if this frame was called).
    pub return_dst: Option<bc::Place>,
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
            // Generate a unique name for this tuple type based on element types
            let type_name = format!("Tuple_{}", items.len());

            // Get Z3 sorts for each element type
            let element_sorts: Vec<z3::Sort> =
                items.iter().map(|item_ty| get_z3_type(*item_ty)).collect();

            // Create a datatype with a single variant containing all fields
            // We need to collect field names as owned strings to avoid lifetime issues
            let field_names: Vec<String> =
                (0..items.len()).map(|i| format!("field_{}", i)).collect();

            let fields: Vec<(&str, z3::DatatypeAccessor)> = element_sorts
                .iter()
                .zip(field_names.iter())
                .map(|(sort, name)| (name.as_str(), z3::DatatypeAccessor::sort(sort.clone())))
                .collect();

            let dt = z3::DatatypeBuilder::new(type_name.as_str())
                .variant("tuple", fields)
                .finish();

            dt.sort
        }
        bc::TypeKind::Func {
            inputs: _inputs,
            output: _output,
        } => {
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
