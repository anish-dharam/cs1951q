//! Intraprocedural dataflow analysis for the bytecode.
//!
//! API design is heavily inspired by the [rustc_mir_dataflow](https://doc.rust-lang.org/stable/nightly-rustc/rustc_mir_dataflow/index.html) crate.

#![allow(unused)]

use either::Either;
use indexical::{
    IndexedDomain, IndexedValue, ToIndex, bitset::bitvec::ArcIndexSet as IndexSet,
    vec::ArcIndexVec as IndexVec,
};
use itertools::{any, fold};
use rayon::range;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    iter::successors,
    sync::Arc,
};
use wasmparser::collections::Set;

use crate::bc::types::{
    AllocArgs, AllocKind, AllocLoc, Const, Local, LocalIdx, Operand, ProjectionElem, Rvalue,
    TerminatorKind,
};

use crate::utils::Symbol;

use super::types::{Function, Location, Statement, Terminator};

use super::visit::VisitMut;

/// A trait for types representing a [join-semilattice](https://en.wikipedia.org/wiki/Semilattice).
///
/// `join` must be associative, commutative, and idempotent.
pub trait JoinSemiLattice: Eq {
    /// Returns true if `self` is changed by `join`.
    fn join(&mut self, other: &Self) -> bool;
}

/// Direction for dataflow analysis.
#[derive(Clone, Copy)]
pub enum Direction {
    Forward,
    Backward,
}

/// Interface for dataflow analyses.
pub trait Analysis {
    /// The type of dataflow analysis state held at each program location.
    type Domain: JoinSemiLattice + Clone;

    /// The direction of the dataflow analysis, forward or backward.
    const DIRECTION: Direction;

    /// The bottom element of the bounded join-semilattice [`Self::Domain`].
    ///
    /// This is the initial value of the analysis state.
    fn bottom(&self, func: &Function) -> Self::Domain;

    /// Transfer function for statements.
    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location);

    /// Transfer function for terminators.
    fn handle_terminator(&self, state: &mut Self::Domain, terminator: &Terminator, loc: Location);
}

pub type AnalysisState<A> = IndexVec<Location, <A as Analysis>::Domain>;

/// Executes the dataflow analysis on the given function to a fixpoint, returning
/// the analysis state at each location.
pub fn analyze_to_fixpoint<A: Analysis>(analysis: &A, func: &Function) -> AnalysisState<A> {
    let mut worklist = VecDeque::new();
    for loc in func.body.locations().iter() {
        match A::DIRECTION {
            Direction::Forward => {
                worklist.push_back(loc);
            }
            Direction::Backward => {
                worklist.push_front(loc);
            }
        }
    }

    let mut in_states: AnalysisState<A> =
        IndexVec::from_elem(analysis.bottom(func), func.body.locations());

    let mut out_states: AnalysisState<A> =
        IndexVec::from_elem(analysis.bottom(func), func.body.locations());

    while let Some(loc) = worklist.pop_front() {
        match A::DIRECTION {
            Direction::Forward => {
                let mut new_in_state = analysis.bottom(func);
                for in_loc in func.body.predecessors(*loc) {
                    //union all predecessor out states
                    new_in_state.join(out_states.get(in_loc));
                }

                *in_states.get_mut(*loc) = new_in_state;
                let mut new_out_state = in_states.get_mut(*loc);

                match func.body.instr(*loc) {
                    // now it's actually out_state
                    Either::Left(statement) => {
                        analysis.handle_statement(&mut new_out_state, statement, *loc)
                    }
                    Either::Right(terminator) => {
                        analysis.handle_terminator(&mut new_out_state, terminator, *loc)
                    }
                };

                if (out_states.get_mut(*loc).join(&new_out_state)) {
                    for succ_loc in func.body.successors(*loc) {
                        worklist.push_back(
                            func.body
                                .locations()
                                .value(func.body.locations().index(&succ_loc)),
                        );
                    }
                }
            }
            Direction::Backward => {
                // Compute out-state from successor in-states
                let mut new_out_state = analysis.bottom(func);
                for succ_loc in func.body.successors(*loc) {
                    new_out_state.join(in_states.get(succ_loc));
                }
                *out_states.get_mut(*loc) = new_out_state;

                // Apply transfer function: in = F(out)
                let mut new_in_state = out_states.get_mut(*loc).clone();
                match func.body.instr(*loc) {
                    Either::Left(statement) => {
                        analysis.handle_statement(&mut new_in_state, statement, *loc)
                    }
                    Either::Right(terminator) => {
                        analysis.handle_terminator(&mut new_in_state, terminator, *loc)
                    }
                };

                // If in-state changed, enqueue predecessors
                if in_states.get_mut(*loc).join(&new_in_state) {
                    for pred_loc in func.body.predecessors(*loc) {
                        worklist.push_back(
                            func.body
                                .locations()
                                .value(func.body.locations().index(&pred_loc)),
                        );
                    }
                }
            }
        }
    }

    match A::DIRECTION {
        Direction::Forward => in_states,
        Direction::Backward => out_states,
    }
}

// =====================
// ANDERSEN POINTER ANALYSIS (flow-insensitive, field-sensitive for tuples)
// =====================

/// Allocation site represented by the `Location` of the `Rvalue::Alloc` statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Allocation(pub Location);

/// Field path representing a sequence of field accesses (e.g., [0, 1] for x.0.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldPath(smallvec::SmallVec<[usize; 4]>);

impl FieldPath {
    pub fn empty() -> Self {
        FieldPath(smallvec::SmallVec::new())
    }

    pub fn extend(&self, field: usize) -> Self {
        let mut path = self.0.clone();
        path.push(field);
        FieldPath(path)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Allocation site with field path
type AllocationSite = (Allocation, FieldPath);

/// Points-to set for a specific allocation site
type PointsToSet = HashSet<AllocationSite>;

/// Pointer domain for a function
type PointerDomain = HashMap<(LocalIdx, FieldPath), PointsToSet>;

impl JoinSemiLattice for PointerDomain {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for ((local, field_path), other_set) in other {
            let entry = self.entry((*local, field_path.clone())).or_default();
            let before = entry.len();
            entry.extend(other_set.iter().cloned());
            if entry.len() != before {
                changed = true;
            }
        }
        changed
    }
}

struct AndersenAnalysis;

impl AndersenAnalysis {
    fn base_local_of_operand(op: &Operand) -> Option<LocalIdx> {
        match op {
            Operand::Place(p) => Some(p.local),
            _ => None,
        }
    }

    fn field_path_of_operand(op: &Operand) -> FieldPath {
        match op {
            Operand::Place(place) => {
                let mut path = FieldPath::empty();
                for proj in &place.projection {
                    match proj {
                        ProjectionElem::Field { index, .. } => {
                            path = path.extend(*index);
                        }
                        ProjectionElem::ArrayIndex { .. } => {
                            // Arrays field-insensitive, so we don't extend the path
                            // This means arr[i] and arr[j] are treated the same
                        }
                    }
                }
                path
            }
            _ => FieldPath::empty(),
        }
    }

    fn union_into(dst: &mut PointsToSet, src: &PointsToSet) -> bool {
        let before = dst.len();
        dst.extend(src.iter().cloned());
        dst.len() != before
    }

    fn get_points_to(
        state: &PointerDomain,
        local: LocalIdx,
        field_path: &FieldPath,
    ) -> PointsToSet {
        state
            .get(&(local, field_path.clone()))
            .cloned()
            .unwrap_or_default()
    }

    fn set_points_to(
        state: &mut PointerDomain,
        local: LocalIdx,
        field_path: FieldPath,
        points_to: PointsToSet,
    ) {
        if !points_to.is_empty() {
            state.insert((local, field_path), points_to);
        }
    }
}

impl Analysis for AndersenAnalysis {
    type Domain = PointerDomain;

    const DIRECTION: Direction = Direction::Forward;

    fn bottom(&self, func: &Function) -> Self::Domain {
        // For field-sensitive analysis, we start with empty state
        // Field paths will be created on-demand as we encounter projections
        HashMap::new()
    }

    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location) {
        let dst_local = statement.place.local;
        let dst_field_path =
            AndersenAnalysis::field_path_of_operand(&Operand::Place(statement.place));

        match &statement.rvalue {
            Rvalue::Alloc { kind, args, .. } => {
                let allocation = Allocation(loc);

                // The destination points to the allocation at the root level
                let mut root_points_to = PointsToSet::new();
                root_points_to.insert((allocation, FieldPath::empty()));
                AndersenAnalysis::set_points_to(
                    state,
                    dst_local,
                    dst_field_path.clone(),
                    root_points_to,
                );

                // For tuples, also set up field-level points-to relationships
                match kind {
                    AllocKind::Tuple => {
                        if let AllocArgs::Lit(operands) = args {
                            for (field_idx, operand) in operands.iter().enumerate() {
                                if let Some(src_local) =
                                    AndersenAnalysis::base_local_of_operand(operand)
                                {
                                    // what the source tuple literal arg
                                    let src_points_to = AndersenAnalysis::get_points_to(
                                        state,
                                        src_local,
                                        &FieldPath::empty(),
                                    );

                                    // each field of the allocation points to what the corresponding operand points to
                                    // field path should be relative to the destination field path
                                    let field_path = dst_field_path.extend(field_idx);
                                    AndersenAnalysis::set_points_to(
                                        state,
                                        dst_local,
                                        field_path,
                                        src_points_to,
                                    );
                                }
                            }
                        }
                    }
                    AllocKind::Struct | AllocKind::Array => {
                        // For now, we only care about tuples for field-sensitivity
                        // Structs and arrays are treated as field-insensitive
                    }
                }
            }
            Rvalue::Operand(op) => {
                if let Some(src_local) = AndersenAnalysis::base_local_of_operand(op) {
                    let src_field_path = AndersenAnalysis::field_path_of_operand(op);
                    let src_points_to =
                        AndersenAnalysis::get_points_to(state, src_local, &src_field_path);
                    AndersenAnalysis::set_points_to(
                        state,
                        dst_local,
                        dst_field_path.clone(),
                        src_points_to,
                    );

                    // If both source and destination are root-level (no field paths),
                    // copy all field-level points-to sets from source to destination
                    if src_field_path.is_empty() && dst_field_path.is_empty() {
                        // Collect field-level points-to sets to copy
                        let field_sets: Vec<_> = state
                            .iter()
                            .filter(|((local, field_path), _)| {
                                *local == src_local && !field_path.is_empty()
                            })
                            .map(|((_, field_path), points_to)| {
                                (field_path.clone(), points_to.clone())
                            })
                            .collect();

                        // Copy all field-level points-to sets from src_local to dst_local
                        for (field_path, points_to) in field_sets {
                            AndersenAnalysis::set_points_to(
                                state, dst_local, field_path, points_to,
                            );
                        }
                    }
                }
            }
            Rvalue::Cast { op, .. } => {
                if let Some(src_local) = AndersenAnalysis::base_local_of_operand(op) {
                    let src_field_path = AndersenAnalysis::field_path_of_operand(op);
                    let src_points_to =
                        AndersenAnalysis::get_points_to(state, src_local, &src_field_path);
                    AndersenAnalysis::set_points_to(
                        state,
                        dst_local,
                        dst_field_path,
                        src_points_to,
                    );
                }
            }
            Rvalue::Call { f, args } => {
                // Conservative: all inputs flow to each other and to the output
                let mut participants: Vec<(LocalIdx, FieldPath)> = Vec::new();

                if let Some(l) = AndersenAnalysis::base_local_of_operand(f) {
                    participants.push((l, AndersenAnalysis::field_path_of_operand(f)));
                }
                for a in args {
                    if let Some(l) = AndersenAnalysis::base_local_of_operand(a) {
                        participants.push((l, AndersenAnalysis::field_path_of_operand(a)));
                    }
                }
                participants.push((dst_local, dst_field_path));

                // Build the union of all points-to sets
                let mut total: PointsToSet = PointsToSet::new();
                for (l, field_path) in &participants {
                    let points_to = AndersenAnalysis::get_points_to(state, *l, field_path);
                    total.extend(points_to.iter().cloned());
                }

                // Write back to all participants
                for (l, field_path) in participants {
                    AndersenAnalysis::set_points_to(state, l, field_path, total.clone());
                }
            }
            Rvalue::MethodCall { receiver, args, .. } => {
                // Same conservative treatment as Call
                let mut participants: Vec<(LocalIdx, FieldPath)> = Vec::new();

                if let Some(l) = AndersenAnalysis::base_local_of_operand(receiver) {
                    participants.push((l, AndersenAnalysis::field_path_of_operand(receiver)));
                }
                for a in args {
                    if let Some(l) = AndersenAnalysis::base_local_of_operand(a) {
                        participants.push((l, AndersenAnalysis::field_path_of_operand(a)));
                    }
                }
                participants.push((dst_local, dst_field_path));

                let mut total: PointsToSet = PointsToSet::new();
                for (l, field_path) in &participants {
                    let points_to = AndersenAnalysis::get_points_to(state, *l, field_path);
                    total.extend(points_to.iter().cloned());
                }

                for (l, field_path) in participants {
                    AndersenAnalysis::set_points_to(state, l, field_path, total.clone());
                }
            }
            Rvalue::Closure { env, .. } => {
                // Conservative: all env operands flow to each other and to the output
                let mut participants: Vec<(LocalIdx, FieldPath)> = Vec::new();
                for env_op in env {
                    if let Some(l) = AndersenAnalysis::base_local_of_operand(env_op) {
                        participants.push((l, AndersenAnalysis::field_path_of_operand(env_op)));
                    }
                }
                participants.push((dst_local, dst_field_path));

                // Build the union of all points-to sets
                let mut total: PointsToSet = PointsToSet::new();
                for (l, field_path) in &participants {
                    let points_to = AndersenAnalysis::get_points_to(state, *l, field_path);
                    total.extend(points_to.iter().cloned());
                }

                // Write back to all participants
                for (l, field_path) in participants {
                    AndersenAnalysis::set_points_to(state, l, field_path, total.clone());
                }
            }
            _ => {} // binop, etc. - no pointer flow
        }
    }

    fn handle_terminator(
        &self,
        _state: &mut Self::Domain,
        _terminator: &Terminator,
        _loc: Location,
    ) {
        // No pointer flow through terminators
    }
}

/// Run the Andersen-style pointer analysis and return the final flow-insensitive mapping
/// from locals to the set of allocation sites they may point to.
pub fn pointer_analysis(func: &Function) -> PointerDomain {
    let analysis_state = analyze_to_fixpoint(&AndersenAnalysis, func);

    // Aggregate all location states into a single flow-insensitive result
    let mut result: PointerDomain = HashMap::new();

    for state in analysis_state.iter() {
        let _ = result.join(state);
    }
    result
}

// =====================
// ESCAPE ANALYSIS
// =====================

/// Compute the set of allocation sites that escape the function.
/// An allocation site (Allocation, FieldPath) escapes if it:
/// 1. Is returned from the function
/// 2. Is passed as an argument to a function/method call
/// 3. Is assigned to a parameter variable
fn compute_escaping_allocations(
    func: &Function,
    pointer_domain: &PointerDomain,
) -> HashSet<Allocation> {
    let mut escaping = HashSet::new();

    // Helper to get base local and field path from operand
    let operand_info = |op: &Operand| -> Option<(LocalIdx, FieldPath)> {
        match op {
            Operand::Place(p) => {
                let field_path = AndersenAnalysis::field_path_of_operand(op);
                Some((p.local, field_path))
            }
            _ => None,
        }
    };

    // Check all locations in the function
    for loc in func.body.locations().iter() {
        match func.body.instr(*loc) {
            Either::Left(statement) => {
                // Check assignments to parameters
                let base_local = statement.place.local;
                if base_local.index() < func.num_params {
                    // This is assigning to a parameter - mark all allocation sites that could flow here as escaping
                    let field_path =
                        AndersenAnalysis::field_path_of_operand(&Operand::Place(statement.place));
                    if let Some(points_to) = pointer_domain.get(&(base_local, field_path)) {
                        escaping.extend(points_to.iter().cloned());
                    }
                }

                // Check function/method calls and allocations
                match &statement.rvalue {
                    Rvalue::Call { f, args } => {
                        // Check function operand
                        if let Some((local, field_path)) = operand_info(f) {
                            if let Some(points_to) = pointer_domain.get(&(local, field_path)) {
                                escaping.extend(points_to.iter().cloned());
                            }
                        }
                        // Check argument operands
                        for arg in args {
                            if let Some((local, field_path)) = operand_info(arg) {
                                if let Some(points_to) = pointer_domain.get(&(local, field_path)) {
                                    escaping.extend(points_to.iter().cloned());
                                }
                            }
                        }
                    }
                    Rvalue::MethodCall { receiver, args, .. } => {
                        // Check receiver operand
                        if let Some((local, field_path)) = operand_info(receiver) {
                            if let Some(points_to) = pointer_domain.get(&(local, field_path)) {
                                escaping.extend(points_to.iter().cloned());
                            }
                        }
                        // Check argument operands
                        for arg in args {
                            if let Some((local, field_path)) = operand_info(arg) {
                                if let Some(points_to) = pointer_domain.get(&(local, field_path)) {
                                    escaping.extend(points_to.iter().cloned());
                                }
                            }
                        }
                    }
                    Rvalue::Alloc { args, .. } => {
                        // Only mark operands as escaping if this allocation itself escapes
                        // We'll handle this in a second pass after we know which allocations escape
                    }
                    _ => {} // Other rvalues don't cause escapes
                }
            }
            Either::Right(terminator) => {
                // Check returns
                match terminator.kind() {
                    TerminatorKind::Return(operand) => {
                        if let Some((local, field_path)) = operand_info(operand) {
                            if let Some(points_to) = pointer_domain.get(&(local, field_path)) {
                                escaping.extend(points_to.iter().cloned());
                            }
                        }
                    }
                    _ => {} // Other terminators don't cause escapes
                }
            }
        }
    }

    // Second pass: propagate escape information through allocations
    // If an allocation escapes, then all allocations used in its operands also escape
    let mut changed = true;
    while changed {
        changed = false;
        for loc in func.body.locations().iter() {
            match func.body.instr(*loc) {
                Either::Left(statement) => {
                    if let Rvalue::Alloc { args, .. } = &statement.rvalue {
                        let allocation = Allocation(*loc);
                        if escaping.contains(&(allocation, FieldPath::empty())) {
                            // This allocation escapes, so mark all its operands as escaping
                            match args {
                                AllocArgs::Lit(ops) => {
                                    for op in ops {
                                        if let Some((local, field_path)) = operand_info(op) {
                                            if let Some(points_to) =
                                                pointer_domain.get(&(local, field_path))
                                            {
                                                let before_len = escaping.len();
                                                escaping.extend(points_to.iter().cloned());
                                                if escaping.len() > before_len {
                                                    changed = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                AllocArgs::ArrayCopy { value, count } => {
                                    if let Some((local, field_path)) = operand_info(value) {
                                        if let Some(points_to) =
                                            pointer_domain.get(&(local, field_path))
                                        {
                                            let before_len = escaping.len();
                                            escaping.extend(points_to.iter().cloned());
                                            if escaping.len() > before_len {
                                                changed = true;
                                            }
                                        }
                                    }
                                    if let Some((local, field_path)) = operand_info(count) {
                                        if let Some(points_to) =
                                            pointer_domain.get(&(local, field_path))
                                        {
                                            let before_len = escaping.len();
                                            escaping.extend(points_to.iter().cloned());
                                            if escaping.len() > before_len {
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Either::Right(_) => {}
            }
        }
    }

    // Mark entire allocations as escaping if any field of them escapes
    let mut escaping_allocations = HashSet::new();
    for (allocation, _) in &escaping {
        escaping_allocations.insert(*allocation);
    }
    escaping_allocations
}

/// Stack allocation optimization pass.
/// Changes AllocLoc from Heap to Stack for allocations that do not escape the function.
pub fn stack_allocate(func: &mut Function) -> bool {
    // Run pointer analysis to get the pointer domain (using immutable reference)
    let pointer_domain = {
        let func_ref = &*func;
        pointer_analysis(func_ref)
    };

    // Compute which allocation sites escape
    let escaping_allocations = {
        let func_ref = &*func;
        compute_escaping_allocations(func_ref, &pointer_domain)
    };

    let mut changed = false;

    // Visit all statements and change non-escaping allocations to stack
    let blocks: Vec<_> = func.body.blocks().collect();
    for block_idx in blocks {
        let block = func.body.data_mut(block_idx);

        for (instr_idx, statement) in block.statements.iter_mut().enumerate() {
            if let Rvalue::Alloc { loc, .. } = &mut statement.rvalue {
                let allocation = Allocation(Location {
                    block: block_idx,
                    instr: instr_idx,
                });

                // Check if this allocation escapes
                let allocation_escapes = escaping_allocations.contains(&allocation);

                // If this allocation doesn't escape, change it to stack
                if !allocation_escapes && *loc == AllocLoc::Heap {
                    *loc = AllocLoc::Stack;
                    changed = true;
                }
            }
        }
    }

    changed
}

//                   CONSTANT PROPAGATION ANALYSIS
struct ConstantAnalysis;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Constant {
    Bottom,
    Const(Const),
    Closure(Symbol),
    Top,
}

impl JoinSemiLattice for Constant {
    fn join(&mut self, other: &Self) -> bool {
        match (self.clone(), other.clone()) {
            (Constant::Bottom, Constant::Const(c)) => {
                *self = Constant::Const(c.clone());
                true
            }
            (Constant::Bottom, Constant::Closure(f)) => {
                *self = Constant::Closure(f.clone());
                true
            }
            (Constant::Const(_), Constant::Top)
            | (Constant::Bottom, Constant::Top)
            | (Constant::Closure(_), Constant::Top)
            | (Constant::Const(_), Constant::Closure(_))
            | (Constant::Closure(_), Constant::Const(_)) => {
                *self = Constant::Top;
                true
            }
            (Constant::Const(c1), Constant::Const(c2)) => {
                if (c1 == c2) {
                    false
                } else {
                    *self = Constant::Top;
                    true
                }
            }

            (Constant::Closure(f1), Constant::Closure(f2)) => {
                if (f1 == f2) {
                    false
                } else {
                    *self = Constant::Top;
                    true
                }
            }
            _ => false,
        }
    }
}

impl JoinSemiLattice for HashMap<LocalIdx, Constant> {
    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;

        // Process all variables in the other state
        for (var, other_val) in other {
            match self.get_mut(var) {
                Some(self_val) => {
                    // Variable exists in both states, join them
                    if self_val.join(other_val) {
                        changed = true;
                    }
                }
                None => {
                    // Variable only exists in other state, Everything should be everywhere
                    // println!("var: {:?} with constant: {:?}", var, other_val);
                    // panic!("Variable only exists in other state, every var should be bottom");
                }
            }
        }

        changed
    }
}

fn fold_constants(
    state: &HashMap<LocalIdx, Constant>,
    left: Constant,
    right: Constant,
    op: super::types::Binop,
) -> Constant {
    match (left, right) {
        //either top
        (Constant::Top, _) | (_, Constant::Top) => Constant::Top,
        // Both integers
        (Constant::Const(Const::Int(li)), Constant::Const(Const::Int(ri))) => {
            match op {
                super::types::Binop::Add => (Constant::Const(Const::Int(li + ri))),
                super::types::Binop::Sub => (Constant::Const(Const::Int(li - ri))),
                super::types::Binop::Mul => (Constant::Const(Const::Int(li * ri))),
                super::types::Binop::Div => {
                    if ri == 0 {
                        panic!("Division by zero");
                    } else {
                        (Constant::Const(Const::Int(li / ri)))
                    }
                }
                super::types::Binop::Rem => {
                    if ri == 0 {
                        panic!("Division by zero");
                    } else {
                        (Constant::Const(Const::Int(li % ri)))
                    }
                }
                super::types::Binop::Eq => (Constant::Const(Const::Bool(li == ri))),
                super::types::Binop::Neq => (Constant::Const(Const::Bool(li != ri))),
                super::types::Binop::Lt => (Constant::Const(Const::Bool(li < ri))),
                super::types::Binop::Gt => (Constant::Const(Const::Bool(li > ri))),
                super::types::Binop::Le => (Constant::Const(Const::Bool(li <= ri))),
                super::types::Binop::Ge => (Constant::Const(Const::Bool(li >= ri))),
                super::types::Binop::Shl => (Constant::Const(Const::Int(li << ri))),
                super::types::Binop::Shr => (Constant::Const(Const::Int(li >> ri))),
                super::types::Binop::BitAnd => (Constant::Const(Const::Int(li & ri))),
                super::types::Binop::BitOr => (Constant::Const(Const::Int(li | ri))),
                super::types::Binop::Exp => {
                    // Handle exponentiation (you might want to handle overflow)
                    (Constant::Const(Const::Int(li.pow(ri as u32))))
                }
                _ => panic!("Bad type checking in compiler!"),
            }
        }

        // Both floats
        (Constant::Const(Const::Float(lf)), Constant::Const(Const::Float(rf))) => match op {
            super::types::Binop::Add => (Constant::Const(Const::Float(lf + rf))),
            super::types::Binop::Sub => (Constant::Const(Const::Float(lf - rf))),
            super::types::Binop::Mul => (Constant::Const(Const::Float(lf * rf))),
            super::types::Binop::Div => {
                if *rf == 0.0 {
                    panic!("Division by zero");
                } else {
                    (Constant::Const(Const::Float(lf / rf)))
                }
            }
            super::types::Binop::Rem => {
                if *rf == 0.0 {
                    panic!("Division by zero");
                } else {
                    (Constant::Const(Const::Float(lf % rf)))
                }
            }
            super::types::Binop::Eq => (Constant::Const(Const::Bool(lf == rf))),
            super::types::Binop::Neq => (Constant::Const(Const::Bool(lf != rf))),
            super::types::Binop::Lt => (Constant::Const(Const::Bool(lf < rf))),
            super::types::Binop::Gt => (Constant::Const(Const::Bool(lf > rf))),
            super::types::Binop::Le => (Constant::Const(Const::Bool(lf <= rf))),
            super::types::Binop::Ge => (Constant::Const(Const::Bool(lf >= rf))),
            super::types::Binop::Exp => {
                (Constant::Const(Const::Float(ordered_float::OrderedFloat((lf.powf(*rf))))))
            }
            _ => panic!("Bad type checking in compiler!"),
        },

        // String concatenation
        (Constant::Const(Const::String(ls)), Constant::Const(Const::String(rs))) => match op {
            super::types::Binop::Concat => {
                (Constant::Const(Const::String(format!("{}{}", ls, rs))))
            }
            super::types::Binop::Eq => (Constant::Const(Const::Bool(ls == rs))),
            super::types::Binop::Neq => (Constant::Const(Const::Bool(ls != rs))),
            _ => panic!("Bad type checking in compiler!"),
        },

        // Boolean operations
        (Constant::Const(Const::Bool(lb)), Constant::Const(Const::Bool(rb))) => match op {
            super::types::Binop::And => (Constant::Const(Const::Bool(lb && rb))),
            super::types::Binop::Or => (Constant::Const(Const::Bool(lb || rb))),
            super::types::Binop::Eq => (Constant::Const(Const::Bool(lb == rb))),
            super::types::Binop::Neq => (Constant::Const(Const::Bool(lb != rb))),
            _ => panic!("Bad type checking in compiler!"),
        },

        // Other combinations - could be anything! (or bad type checking in compiler!)
        _ => Constant::Top,
    }
}

impl Analysis for ConstantAnalysis {
    type Domain = HashMap<LocalIdx, Constant>;

    const DIRECTION: Direction = Direction::Forward;

    fn bottom(&self, func: &Function) -> Self::Domain {
        HashMap::from_iter(
            func.locals
                .indices()
                .map(|local_idx| (local_idx, Constant::Bottom))
                .collect::<HashMap<_, _>>(),
        )
    }

    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location) {
        //helper
        match &statement.rvalue {
            super::types::Rvalue::Operand(operand) => match operand {
                super::types::Operand::Const(c) => {
                    if statement.place.projection.is_empty() {
                        state.insert(statement.place.local, Constant::Const(c.clone()));
                    } else {
                        state.insert(statement.place.local, Constant::Top);
                    }
                }
                super::types::Operand::Place(place) => match state.get(&place.local) {
                    Some(v) => {
                        state.insert(statement.place.local, v.clone());
                    }
                    None => {}
                },
                _ => (),
            },
            super::types::Rvalue::Cast { op, ty } => match op {
                crate::bc::types::Operand::Const(c) => {
                    state.insert(statement.place.local, Constant::Const(c.clone()));
                }
                crate::bc::types::Operand::Place(place) => {
                    state.insert(
                        statement.place.local,
                        state.get(&place.local).unwrap().clone(),
                    );
                }
                crate::bc::types::Operand::Func { f, ty } => (),
            },
            super::types::Rvalue::Closure { f, env } => match env.len() {
                0 => {
                    state.insert(statement.place.local, Constant::Closure(*f));
                }
                _ => (),
            },
            super::types::Rvalue::Binop { op, left, right } => {
                // Get constant values if they exist
                let left_const = match left {
                    super::types::Operand::Const(c) => Some(Constant::Const(c.clone())),
                    super::types::Operand::Place(p) => state.get(&p.local).cloned(),
                    _ => None,
                };

                let right_const = match right {
                    super::types::Operand::Const(c) => Some(Constant::Const(c.clone())),
                    super::types::Operand::Place(p) => state.get(&p.local).cloned(),
                    _ => None,
                };

                match (left_const.clone(), right_const.clone()) {
                    (Some(c1), Some(c2)) => {
                        state.insert(statement.place.local, fold_constants(state, c1, c2, *op));
                    }
                    _ => {
                        println!("statement : {}", statement);
                        println!(
                            "left_const: {:?}, right_const: {:?}",
                            left_const.clone(),
                            right_const.clone()
                        );
                        panic!("What BinOps are we using on Functions?")
                    }
                }
            }
            _ => (),
        }
    }

    fn handle_terminator(&self, state: &mut Self::Domain, terminator: &Terminator, loc: Location) {
        ()
    }
}

// pub fn constant_propagation(func: &mut Function) -> bool {
//     let analysis_state = analyze_to_fixpoint(&ConstantAnalysis, func);
//     let mut code_changed = false;

//     //iterate through basic blocks
//     for block_idx in func.body.clone().blocks() {
//         let block = func.body.data_mut(block_idx);

//         for i in (0..block.statements.len()) {
//             let statement = block.statements.get(i).unwrap();
//         }
//     }

//     code_changed
// }
pub fn constant_propagation(func: &mut Function) -> bool {
    let analysis_state = analyze_to_fixpoint(&ConstantAnalysis, func);
    let mut code_changed = false;

    // Create a visitor to propagate constants
    let mut propagator = ConstantPropagator {
        analysis_state: &analysis_state,
        code_changed: &mut code_changed,
    };

    propagator.visit_function(func);

    code_changed
}

struct ConstantPropagator<'a> {
    analysis_state: &'a AnalysisState<ConstantAnalysis>,
    code_changed: &'a mut bool,
}

impl<'a> VisitMut for ConstantPropagator<'a> {
    fn visit_operand(&mut self, operand: &mut Operand, loc: Location) {
        match operand {
            Operand::Place(place) => {
                // Check if this place holds a constant value
                if let Some(constant) = self.analysis_state.get(loc).get(&place.local) {
                    match constant {
                        Constant::Const(c) => {
                            *operand = Operand::Const(c.clone());
                            *self.code_changed = true;
                        }
                        Constant::Closure(f) => {
                            *operand = Operand::Func {
                                f: *f,
                                ty: place.ty,
                            };
                            *self.code_changed = true;
                        }
                        _ => {} // Don't propagate Top or Bottom
                    }
                }
            }
            _ => {} // Constants and functions are already constant
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue, loc: Location) {
        match rvalue {
            Rvalue::Binop { op, left, right } => {
                // First visit operands to potentially propagate constants
                self.visit_operand(left, loc);
                self.visit_operand(right, loc);

                // If both operands are now constants, fold the operation
                if let (Operand::Const(left_const), Operand::Const(right_const)) = (left, right) {
                    let result = fold_constants(
                        self.analysis_state.get(loc),
                        Constant::Const(left_const.clone()),
                        Constant::Const(right_const.clone()),
                        *op,
                    );

                    if let Constant::Const(folded_const) = result {
                        *rvalue = Rvalue::Operand(Operand::Const(folded_const));
                        *self.code_changed = true;
                    }
                }
            }
            Rvalue::Call { f, args } => {
                // Visit function and arguments first
                self.visit_operand(f, loc);
                for arg in args {
                    self.visit_operand(arg, loc);
                }

                // If function is a known closure and all args are constants,
                // we could potentially inline, but for now just propagate
            }
            _ => {
                // For other rvalues, just visit their operands
                self.super_visit_rvalue(rvalue, loc);
            }
        }
    }
}

//                   DEAD CODE ELIMINATION ANALYSIS
struct DeadCodeAnalysis;

impl JoinSemiLattice for HashSet<LocalIdx> {
    fn join(&mut self, other: &Self) -> bool {
        let original_size = self.len();
        self.extend(other.iter().cloned());
        self.len() > original_size
    }
}

impl Analysis for DeadCodeAnalysis {
    type Domain = HashSet<LocalIdx>;

    const DIRECTION: Direction = Direction::Backward;

    fn bottom(&self, func: &Function) -> Self::Domain {
        HashSet::new()
    }

    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location) {
        let def: HashSet<LocalIdx> = if statement.place.projection.is_empty() {
            HashSet::from([statement.place.local])
        } else {
            HashSet::new()
        };

        let used: HashSet<LocalIdx> = HashSet::from(
            statement
                .rvalue
                .places()
                .iter()
                .map(|p| p.local)
                .collect::<HashSet<LocalIdx>>(),
        );

        *state = state.difference(&def).cloned().collect();
        *state = state.union(&used).cloned().collect::<HashSet<LocalIdx>>();
    }

    fn handle_terminator(&self, state: &mut Self::Domain, terminator: &Terminator, loc: Location) {
        match terminator.kind() {
            super::types::TerminatorKind::Return(operand) => match operand {
                crate::bc::types::Operand::Const(_) => (),
                crate::bc::types::Operand::Place(place) => {
                    state.insert(place.local);
                }
                crate::bc::types::Operand::Func { f, ty } => (),
            },
            _ => (),
        }
    }
}

fn has_side_effect(statement: Statement) -> bool {
    any(statement.place.projection.clone(), |p| match p {
        crate::bc::types::ProjectionElem::Field { index, ty } => false,
        crate::bc::types::ProjectionElem::ArrayIndex { index, ty } => true,
    }) || match statement.rvalue {
        super::types::Rvalue::Operand(op) => match op {
            crate::bc::types::Operand::Place(place) => any(place.projection.clone(), |p| match p {
                crate::bc::types::ProjectionElem::Field { index, ty } => false,
                crate::bc::types::ProjectionElem::ArrayIndex { index, ty } => true,
            }),
            _ => false,
        },
        super::types::Rvalue::Binop { op, left, right } => match op {
            crate::bc::types::Binop::Div => true,
            _ => false,
        },
        super::types::Rvalue::Call { f, args } => true,
        super::types::Rvalue::MethodCall {
            receiver,
            method,
            args,
        } => true,
        _ => false,
    }
}

pub fn dead_code(func: &mut Function) -> bool {
    let analysis_state = analyze_to_fixpoint(&DeadCodeAnalysis, func);
    let mut code_eliminated = false;

    //iterate through basic blocks
    for block_idx in func.body.clone().blocks() {
        let block = func.body.data_mut(block_idx);
        let original_size = block.statements.len();

        for i in (0..block.statements.len()).rev() {
            let statement = block.statements.get(i).unwrap();
            if (!analysis_state // doesn't contain var
                .get(Location {
                    block: block_idx,
                    instr: i,
                })
                .contains(&statement.place.local)
                && !has_side_effect(statement.clone()))
            {
                code_eliminated = true;
                block.statements.remove(i);
            }
        }
    }

    if code_eliminated {
        func.body.regenerate_domain();
    }
    code_eliminated
    //run analysis (get all the state_outs)
    // then remove a statement v = ... if:
    //    v is not in state_out
    //    and there are no side effect

    // so leave a line if v is in state_out, or if it has a side effect
    // return true if code was eliminated
}
