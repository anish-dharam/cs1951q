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
use petgraph::{
    Direction as EdgeDirection,
    algo::dominators::{self, Dominators},
    visit::EdgeRef,
};
use rayon::range;
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{HashMap, HashSet, VecDeque},
    iter::successors,
    str::FromStr,
    sync::Arc,
};
use wasmparser::collections::Set;

use crate::bc::types::{
    AllocArgs, AllocKind, AllocLoc, Const, Local, LocalIdx, Operand, Place, ProjectionElem, Rvalue,
    TerminatorKind, Type,
};

use crate::utils::Symbol;
use miette::{Diagnostic, Result, bail};
use thiserror::Error;

use super::types::{Function, Location, Program, Statement, Terminator};

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

/// Represents either a concrete allocation site or an abstract parameter allocation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemLoc {
    /// Concrete allocation at a specific location in the function body
    Concrete(Location),
    /// Abstract allocation representing a function parameter
    /// Symbol is the function name, usize is the parameter index
    Abstract(Symbol, usize),
}

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
pub type AllocationSite = (MemLoc, FieldPath);

/// Points-to set for a specific allocation site
pub type PointsToSet = HashSet<AllocationSite>;

/// Pointer domain for a function
pub type PointerDomain = HashMap<(LocalIdx, FieldPath), PointsToSet>;

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

struct AndersenAnalysis {
    /// Optional map from parameter index to whether it should be treated as abstract
    param_context: Option<HashSet<usize>>,
}

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

    fn get_operand_type(op: &Operand) -> Type {
        match op {
            Operand::Place(p) => p.ty,
            Operand::Const(c) => c.ty(),
            Operand::Func { ty, .. } => *ty,
        }
    }

    /// Check if two types are compatible for pointer aliasing purposes.
    /// Types are compatible if they are the same type.
    fn types_compatible(ty1: &Type, ty2: &Type) -> bool {
        ty1 == ty2
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
        let mut state = HashMap::new();

        // If we have a param context, initialize parameters as abstract allocations
        if let Some(param_indices) = &self.param_context {
            for param_idx in param_indices {
                if *param_idx < func.num_params {
                    let local_idx = LocalIdx::from_usize(*param_idx);

                    // Create abstract allocation for this parameter
                    let abstract_alloc = MemLoc::Abstract(func.name, *param_idx);
                    let mut points_to = PointsToSet::new();
                    points_to.insert((abstract_alloc, FieldPath::empty()));

                    state.insert((local_idx, FieldPath::empty()), points_to);
                }
            }
        }

        state
    }

    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location) {
        let dst_local = statement.place.local;
        let dst_field_path =
            AndersenAnalysis::field_path_of_operand(&Operand::Place(statement.place));

        match &statement.rvalue {
            Rvalue::Alloc { kind, args, .. } => {
                let allocation = MemLoc::Concrete(loc);

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
                // only arguments of compatible types alias
                // direct allocations (arguments themselves) don't alias each other,// but anything that points to compatible-typed allocations should alias

                // track argument (local, field_path) pairs to exclude them from aliasing
                let mut arg_locations: HashSet<(LocalIdx, FieldPath)> = HashSet::new();
                let mut arg_info: Vec<(Type, PointsToSet)> = Vec::new();

                if let Some(l) = AndersenAnalysis::base_local_of_operand(f) {
                    let arg_ty = AndersenAnalysis::get_operand_type(f);
                    let field_path = AndersenAnalysis::field_path_of_operand(f);
                    let points_to = AndersenAnalysis::get_points_to(state, l, &field_path);
                    arg_locations.insert((l, field_path));
                    arg_info.push((arg_ty, points_to));
                }

                for a in args {
                    if let Some(l) = AndersenAnalysis::base_local_of_operand(a) {
                        let arg_ty = AndersenAnalysis::get_operand_type(a);
                        let field_path = AndersenAnalysis::field_path_of_operand(a);
                        let points_to = AndersenAnalysis::get_points_to(state, l, &field_path);
                        arg_locations.insert((l, field_path));
                        arg_info.push((arg_ty, points_to));
                    }
                }

                // Group arguments by compatible types
                let mut type_groups: HashMap<Type, PointsToSet> = HashMap::new();
                for (ty, points_to) in arg_info {
                    type_groups
                        .entry(ty)
                        .or_default()
                        .extend(points_to.iter().cloned());
                }

                // for each type group, update all locals that point to any allocation in that group
                // to point to the union of all allocations in that group
                // BUT: exclude the arguments themselves (they shouldn't alias each other)
                for (_, union_allocs) in &type_groups {
                    // Extract the set of MemLoc values from the union
                    let union_memlocs: HashSet<MemLoc> =
                        union_allocs.iter().map(|(memloc, _)| *memloc).collect();

                    // Find all (local, field_path) pairs in the state that point to any allocation in this group
                    let mut to_update: Vec<(LocalIdx, FieldPath, PointsToSet)> = Vec::new();

                    for ((local, field_path), current_points_to) in state.iter() {
                        // Skip argument locals themselves - they shouldn't alias each other
                        if arg_locations.contains(&(*local, field_path.clone())) {
                            continue;
                        }

                        // check if this local/field-path points to any MemLoc in the union
                        let points_to_group_allocation = current_points_to
                            .iter()
                            .any(|(memloc, _)| union_memlocs.contains(memloc));

                        if points_to_group_allocation {
                            let mut new_points_to = current_points_to.clone();
                            new_points_to.extend(union_allocs.iter().cloned());
                            to_update.push((*local, field_path.clone(), new_points_to));
                        }
                    }

                    for (local, field_path, new_points_to) in to_update {
                        AndersenAnalysis::set_points_to(state, local, field_path, new_points_to);
                    }
                }

                // update the return value to point to union of all compatible allocations
                let mut return_points_to = PointsToSet::new();
                for union_allocs in type_groups.values() {
                    return_points_to.extend(union_allocs.iter().cloned());
                }
                if !return_points_to.is_empty() {
                    AndersenAnalysis::set_points_to(
                        state,
                        dst_local,
                        dst_field_path.clone(),
                        return_points_to,
                    );
                }
            }
            Rvalue::MethodCall { receiver, args, .. } => {
                // Type-based aliasing: same logic as Call
                let mut arg_locations: HashSet<(LocalIdx, FieldPath)> = HashSet::new();
                let mut arg_info: Vec<(Type, PointsToSet)> = Vec::new();

                if let Some(l) = AndersenAnalysis::base_local_of_operand(receiver) {
                    let arg_ty = AndersenAnalysis::get_operand_type(receiver);
                    let field_path = AndersenAnalysis::field_path_of_operand(receiver);
                    let points_to = AndersenAnalysis::get_points_to(state, l, &field_path);
                    arg_locations.insert((l, field_path));
                    arg_info.push((arg_ty, points_to));
                }

                for a in args {
                    if let Some(l) = AndersenAnalysis::base_local_of_operand(a) {
                        let arg_ty = AndersenAnalysis::get_operand_type(a);
                        let field_path = AndersenAnalysis::field_path_of_operand(a);
                        let points_to = AndersenAnalysis::get_points_to(state, l, &field_path);
                        arg_locations.insert((l, field_path));
                        arg_info.push((arg_ty, points_to));
                    }
                }

                // group arguments by compatible types
                let mut type_groups: HashMap<Type, PointsToSet> = HashMap::new();
                for (ty, points_to) in arg_info {
                    type_groups
                        .entry(ty)
                        .or_default()
                        .extend(points_to.iter().cloned());
                }

                // For each type group, update all locals that point to any allocation in that group
                for (_, union_allocs) in &type_groups {
                    let union_memlocs: HashSet<MemLoc> =
                        union_allocs.iter().map(|(memloc, _)| *memloc).collect();
                    let mut to_update: Vec<(LocalIdx, FieldPath, PointsToSet)> = Vec::new();

                    for ((local, field_path), current_points_to) in state.iter() {
                        // Skip argument locals themselves
                        if arg_locations.contains(&(*local, field_path.clone())) {
                            continue;
                        }

                        let points_to_group_allocation = current_points_to
                            .iter()
                            .any(|(memloc, _)| union_memlocs.contains(memloc));

                        if points_to_group_allocation {
                            let mut new_points_to = current_points_to.clone();
                            new_points_to.extend(union_allocs.iter().cloned());
                            to_update.push((*local, field_path.clone(), new_points_to));
                        }
                    }

                    for (local, field_path, new_points_to) in to_update {
                        AndersenAnalysis::set_points_to(state, local, field_path, new_points_to);
                    }
                }

                // Update the return value
                let mut return_points_to = PointsToSet::new();
                for union_allocs in type_groups.values() {
                    return_points_to.extend(union_allocs.iter().cloned());
                }
                if !return_points_to.is_empty() {
                    AndersenAnalysis::set_points_to(
                        state,
                        dst_local,
                        dst_field_path.clone(),
                        return_points_to,
                    );
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
pub fn concrete_pointer_analysis(func: &Function) -> PointerDomain {
    pointer_analysis_with_context(func, None)
}

/// Run pointer analysis with an optional context specifying which parameters
/// should be treated as abstract allocations
pub fn pointer_analysis_with_context(
    func: &Function,
    abstract_params: Option<HashSet<usize>>,
) -> PointerDomain {
    let analysis = AndersenAnalysis {
        param_context: abstract_params,
    };
    let analysis_state = analyze_to_fixpoint(&analysis, func);

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
) -> HashSet<MemLoc> {
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
                        let allocation = MemLoc::Concrete(*loc);
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

    // allocations are escaping if any field of them escapes
    let mut escaping_allocations: HashSet<MemLoc> = HashSet::new();
    for (allocation, _) in &escaping {
        escaping_allocations.insert(*allocation);
    }
    escaping_allocations
}

/// Stack allocation optimization pass.
/// Changes AllocLoc from Heap to Stack for allocations that do not escape the function.
pub fn stack_allocate(func: &mut Function) -> bool {
    // pointer analysis to get the pointer domain
    let pointer_domain = {
        let func_ref = &*func;
        concrete_pointer_analysis(func_ref)
    };

    // compute which allocations escape
    let escaping_allocations = {
        let func_ref = &*func;
        compute_escaping_allocations(func_ref, &pointer_domain)
    };

    let mut changed = false;

    // visit statements and change non-escaping allocations to stack allocation
    let blocks: Vec<_> = func.body.blocks().collect();
    for block_idx in blocks {
        let block = func.body.data_mut(block_idx);

        for (instr_idx, statement) in block.statements.iter_mut().enumerate() {
            if let Rvalue::Alloc { loc, .. } = &mut statement.rvalue {
                let allocation = MemLoc::Concrete(Location {
                    block: block_idx,
                    instr: instr_idx,
                });

                //   // check if this allocation escapes
                let allocation_escapes = escaping_allocations.contains(&allocation);

                //  if this allocation doesn't escape, change it to stack allocation
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
pub struct ConstantAnalysis;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Constant {
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
                crate::bc::types::Operand::Func { .. } => (),
            },
            super::types::TerminatorKind::CondJump { cond, .. } => match cond {
                crate::bc::types::Operand::Const(_) => (),
                crate::bc::types::Operand::Place(place) => {
                    state.insert(place.local);
                }
                crate::bc::types::Operand::Func { .. } => (),
            },
            _ => (),
        }
    }
}

fn has_side_effect(statement: Statement) -> bool {
    any(statement.place.projection.clone(), |p| match p {
        crate::bc::types::ProjectionElem::Field { index, ty } => true,
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

// =====================
// CONTROL DEPENDENCES
// =====================

/// Compute control dependencies for a function.
/// Returns a map from each location to the set of locations that it is control-dependent on.

pub fn compute_control_dependencies(func: &Function) -> HashMap<Location, HashSet<Location>> {
    use petgraph::graph::{DiGraph, NodeIndex};

    // Build reverse CFG
    let cfg = func.body.cfg();
    let mut reverse_cfg = DiGraph::<(), ()>::with_capacity(cfg.node_count(), cfg.edge_count());

    for _ in cfg.node_indices() {
        reverse_cfg.add_node(());
    }

    // add reversed edges
    for edge in cfg.edge_references() {
        reverse_cfg.add_edge(edge.target(), edge.source(), ());
    }

    // find exit node
    let exit_node = cfg.node_indices().find(|&node| {
        cfg.neighbors_directed(node, EdgeDirection::Outgoing)
            .next()
            .is_none()
    });

    // if there's no single exit node, use the last block in RPO
    let exit_node = exit_node.unwrap_or_else(|| {
        func.body
            .blocks()
            .last()
            .map(|bb| bb.into())
            .unwrap_or_else(|| NodeIndex::new(0))
    });

    // compute dominators on reverse CFG
    let post_doms = dominators::simple_fast(&reverse_cfg, exit_node);

    // compute post-dominance frontiers
    let pdf = compute_post_dominance_frontiers(cfg, &post_doms, exit_node);

    // derive control dependencies from PDF
    let mut cd = HashMap::<Location, HashSet<Location>>::new();

    // for each basic block B and its post-dominance frontier nodes F:
    // all locations in B are control-dependent on the terminator of F
    for (bb_idx, frontier) in &pdf {
        // get the terminator location for this basic block
        let bb = func.body.data(*bb_idx);
        let terminator_loc = Location {
            block: *bb_idx,
            instr: bb.terminator_index(),
        };

        // all nodes in frontier need to have this as a control dependency
        for frontier_bb in frontier {
            // add CD for all locations in frontier_bb
            let frontier_bb_data = func.body.data(*frontier_bb);
            let num_instrs = frontier_bb_data.statements.len() + 1;
            for instr_idx in 0..num_instrs {
                let loc = Location {
                    block: *frontier_bb,
                    instr: instr_idx,
                };
                cd.entry(loc)
                    .or_insert_with(HashSet::new)
                    .insert(terminator_loc);
            }
        }
    }

    cd
}

fn compute_post_dominance_frontiers(
    cfg: &super::types::Cfg,
    post_doms: &Dominators<petgraph::graph::NodeIndex>,
    exit_node: petgraph::graph::NodeIndex,
) -> HashMap<super::types::BasicBlockIdx, HashSet<super::types::BasicBlockIdx>> {
    use petgraph::graph::NodeIndex;

    let mut rpo: Vec<NodeIndex> = cfg.node_indices().collect();
    // sort by reverse post-order
    rpo.sort_by_key(|&node| -(node.index() as i32));

    let mut pdf = HashMap::new();

    // initialize PDF for each node
    for node in cfg.node_indices() {
        pdf.insert(
            super::types::BasicBlockIdx::from(node),
            HashSet::<super::types::BasicBlockIdx>::new(),
        );
    }

    // bottom-up traversal of dominator tree
    for &x in &rpo {
        let x_bb = super::types::BasicBlockIdx::from(x);
        let mut pdf_x = HashSet::new();

        // local contribution: successors of X that X doesn't strictly post-dominate
        for y in cfg.neighbors_directed(x, EdgeDirection::Outgoing) {
            if post_doms.immediate_dominator(y) != Some(x) {
                pdf_x.insert(super::types::BasicBlockIdx::from(y));
            }
        }

        // upward contribution: from children in post-dominator tree
        for z in cfg.node_indices() {
            if post_doms.immediate_dominator(z) == Some(x) && z != x {
                if let Some(pdf_z) = pdf.get(&super::types::BasicBlockIdx::from(z)) {
                    for &y_bb in pdf_z {
                        let y_node: NodeIndex = y_bb.into();
                        if post_doms.immediate_dominator(y_node) != Some(x) {
                            pdf_x.insert(y_bb);
                        }
                    }
                }
            }
        }

        pdf.insert(x_bb, pdf_x);
    }

    pdf
}

// intraprocedural facts

pub struct Facts {
    /// (local, field_path) to set of allocation sites
    pub pointer_domain: PointerDomain,

    /// maps location to map of local -> constant value
    pub constant_domain: AnalysisState<ConstantAnalysis>,

    /// maps each (bb_idx, instr_idx) location to the set of locations it is control-dependent on
    pub control_dependencies: HashMap<Location, HashSet<Location>>,
}

impl Facts {
    //intraprocedural facts
    pub fn compute(func: &Function) -> Self {
        let pointer_domain = concrete_pointer_analysis(func);
        let constant_domain = analyze_to_fixpoint(&ConstantAnalysis, func);
        let control_dependencies = compute_control_dependencies(func);

        Facts {
            pointer_domain,
            constant_domain,
            control_dependencies,
        }
    }
}

// =====================
// TAINT ANALYSIS
// =====================

/// Type alias for tainted MemLoc set
type TaintedMemLocs = HashSet<MemLoc>;

/// Converts a Place to (LocalIdx, FieldPath) tuple
fn place_to_local_field_path(place: &Place) -> (LocalIdx, FieldPath) {
    let local = place.local;
    let mut field_path = FieldPath::empty();
    for proj in &place.projection {
        match proj {
            ProjectionElem::Field { index, .. } => {
                field_path = field_path.extend(*index);
            }
            ProjectionElem::ArrayIndex { .. } => {
                // Arrays are field-insensitive, so we don't extend the path
            }
        }
    }
    (local, field_path)
}

/// GlobalAnalysis holds taint analysis results for all functions
pub struct GlobalAnalysis<'a> {
    /// Taint analysis results: function name -> context -> tainted MemLocs
    results: RefCell<HashMap<Symbol, HashMap<Vec<Place>, TaintedMemLocs>>>,
    /// Intraprocedural facts per function (computed on-demand)
    facts: RefCell<HashMap<Symbol, Facts>>,
    /// All functions in the program
    pub(crate) functions: Vec<&'a Function>,
}

impl<'a> GlobalAnalysis<'a> {
    pub fn new(prog: &'a Program) -> Self {
        let functions: Vec<&'a Function> = prog.functions().iter().collect();

        GlobalAnalysis {
            results: RefCell::new(HashMap::new()),
            facts: RefCell::new(HashMap::new()),
            functions,
        }
    }

    /// Get or compute intraprocedural facts for a function
    fn get_or_compute_facts(&self, func: &Function) {
        // Check if facts already exist (using immutable borrow)
        let needs_compute = {
            let facts_map = self.facts.borrow();
            !facts_map.contains_key(&func.name)
        };

        // Only compute if needed (using mutable borrow)
        if needs_compute {
            let mut facts_map = self.facts.borrow_mut();
            facts_map
                .entry(func.name)
                .or_insert_with(|| Facts::compute(func));
        }
    }

    /// Precompute facts for all functions to avoid borrow conflicts during recursive analysis
    pub fn precompute_all_facts(&self) {
        for func in &self.functions {
            self.get_or_compute_facts(func);
        }
    }

    /// Analyze a function with a given context, returning tainted MemLocs
    pub fn analyze_function(&self, func: &Function, context: Vec<Place>) -> TaintedMemLocs {
        // Check if we already have results for this (function, context) pair
        {
            let results_map = self.results.borrow();
            if let Some(context_map) = results_map.get(&func.name) {
                if let Some(tainted) = context_map.get(&context) {
                    return tainted.clone();
                }
            }
        }

        // Build LocalAnalysis and run fixpoint analysis
        self.get_or_compute_facts(func);
        let facts_map = self.facts.borrow();
        let facts = facts_map.get(&func.name).unwrap();
        let local_analysis = LocalAnalysis {
            global: self,
            func,
            context: context.clone(),
            facts,
        };

        let analysis_state = analyze_to_fixpoint(&local_analysis, func);

        // Extract final tainted MemLocs from all locations in the function
        // For functions that return values, collect from return statements
        // For functions like main(), collect from all final locations
        let mut tainted_memlocs = HashSet::new();

        let mut has_return = false;
        // First, collect from return statements
        for block_idx in func.body.blocks() {
            let block = func.body.data(block_idx);
            let terminator_loc = Location {
                block: block_idx,
                instr: block.statements.len(),
            };

            if let Either::Right(terminator) = func.body.instr(terminator_loc) {
                if let TerminatorKind::Return(operand) = terminator.kind() {
                    has_return = true;
                    // Join all tainted MemLocs at this location
                    let state_at_return = analysis_state.get(&terminator_loc);
                    tainted_memlocs.extend(state_at_return.iter().cloned());

                    // Also check if return operand points to tainted MemLocs
                    if let Operand::Place(ret_place) = operand {
                        let (ret_local, ret_field_path) = place_to_local_field_path(ret_place);
                        if let Some(ret_points_to) =
                            facts.pointer_domain.get(&(ret_local, ret_field_path))
                        {
                            for (ret_memloc, _) in ret_points_to {
                                if state_at_return.contains(ret_memloc) {
                                    tainted_memlocs.insert(*ret_memloc);
                                }
                            }
                        }
                    }
                }
            }
        }

        // If no return statements, collect from all final locations (for main() etc.)
        if !has_return {
            for block_idx in func.body.blocks() {
                let block = func.body.data(block_idx);
                let terminator_loc = Location {
                    block: block_idx,
                    instr: block.statements.len(),
                };
                let state_at_end = analysis_state.get(&terminator_loc);
                tainted_memlocs.extend(state_at_end.iter().cloned());
            }
        }

        // Store results
        {
            let mut results_map = self.results.borrow_mut();
            results_map
                .entry(func.name)
                .or_insert_with(HashMap::new)
                .insert(context, tainted_memlocs.clone());
        }

        tainted_memlocs
    }

    pub fn find_function(&self, name: Symbol) -> Option<&'a Function> {
        self.functions.iter().find(|f| f.name == name).copied()
    }

    /// Check for taint violations at sensitive sinks (e.g., int_to_string)
    pub fn check_taint_violations(&self) -> Result<()> {
        use crate::utils::sym;
        let int_to_string_sym = sym("int_to_string");

        // Collect all violations
        let mut violations = Vec::new();

        // Collect all contexts first to avoid borrow conflicts
        let mut function_contexts: Vec<(Symbol, Vec<Vec<Place>>)> = Vec::new();
        {
            let results_map = self.results.borrow();
            for func in &self.functions {
                let contexts: Vec<Vec<Place>> = results_map
                    .get(&func.name)
                    .map(|ctx_map| ctx_map.keys().cloned().collect())
                    .unwrap_or_else(|| vec![Vec::new()]);
                function_contexts.push((func.name, contexts));
            }
        }

        // Check all functions for calls to sensitive sinks
        // We need to analyze each function, but avoid recursive calls during violation checking
        // So we'll use a flag or ensure all callees are analyzed first
        for (func_name, contexts) in function_contexts {
            let func = self.functions.iter().find(|f| f.name == func_name).unwrap();

            for context in contexts {
                // Re-run analysis to get state at each location
                // Note: We need to ensure all analysis is done upfront to avoid RefCell conflicts
                // First, ensure this function+context has been analyzed
                {
                    let results_map = self.results.borrow();
                    if results_map
                        .get(&func_name)
                        .and_then(|ctx_map| ctx_map.get(&context))
                        .is_none()
                    {
                        // Need to analyze this function+context first
                        drop(results_map);
                        self.analyze_function(func, context.clone());
                    }
                }

                // Now get facts and run analysis for violation checking
                // Since all functions are already analyzed, this should only use cached results
                let facts_map = self.facts.borrow();
                let facts = facts_map.get(&func_name).unwrap();
                let local_analysis = LocalAnalysis {
                    global: self,
                    func,
                    context: context.clone(),
                    facts,
                };

                let analysis_state = analyze_to_fixpoint(&local_analysis, func);

                // Check each statement for calls to sensitive sinks
                for block_idx in func.body.blocks() {
                    let block = func.body.data(block_idx);
                    for (stmt_idx, statement) in block.statements.iter().enumerate() {
                        let stmt_loc = Location {
                            block: block_idx,
                            instr: stmt_idx,
                        };

                        // Get taint state at this location
                        let state_at_stmt = analysis_state.get(&stmt_loc);

                        // Check if this is a call to a sensitive sink
                        if let Rvalue::Call { f, args } = &statement.rvalue {
                            // Check if this is a call to int_to_string
                            let mut is_int_to_string = false;
                            match f {
                                Operand::Func { f: func_name, .. } => {
                                    is_int_to_string = *func_name == int_to_string_sym;
                                }
                                Operand::Place(f_place) => {
                                    // Check constant domain
                                    if let Some(const_val) =
                                        facts.constant_domain.get(&stmt_loc).get(&f_place.local)
                                    {
                                        if let Constant::Closure(f_name) = const_val {
                                            is_int_to_string = *f_name == int_to_string_sym;
                                        }
                                    }
                                }
                                _ => {}
                            }

                            if is_int_to_string {
                                // Check if any argument is tainted
                                for arg in args {
                                    let mut is_tainted = false;

                                    match arg {
                                        Operand::Place(arg_place) => {
                                            let (arg_local, arg_field_path) =
                                                place_to_local_field_path(arg_place);

                                            // Check if this argument points to any tainted MemLoc
                                            if let Some(points_to) = facts
                                                .pointer_domain
                                                .get(&(arg_local, arg_field_path))
                                            {
                                                for (memloc, _) in points_to {
                                                    if state_at_stmt.contains(memloc) {
                                                        is_tainted = true;
                                                        break;
                                                    }
                                                }
                                            }

                                            // Also check if the local itself represents a tainted primitive
                                            // For primitives, we check if any abstract return MemLoc is in state
                                            // This means some tainted value was assigned to this local
                                            if !is_tainted {
                                                for memloc in state_at_stmt.iter() {
                                                    if let MemLoc::Abstract(_, usize::MAX) = memloc
                                                    {
                                                        // Check if this abstract return was propagated to this local
                                                        // We can check if the local's value could be the return value
                                                        // For now, conservatively assume it could be if state contains abstract returns
                                                        is_tainted = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }
                                        Operand::Const(_) => {
                                            // Constants are safe - no taint
                                        }
                                        _ => {
                                            // For other operands, conservatively assume safe
                                        }
                                    }

                                    if is_tainted {
                                        violations.push((func.name, stmt_loc));
                                        break; // One violation per call is enough
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !violations.is_empty() {
            // Report first violation for now
            let (func_name, loc) = &violations[0];
            bail!(TaintError {
                function: *func_name,
                location: *loc,
            });
        }

        Ok(())
    }
}

#[derive(Diagnostic, Error, Debug)]
#[error(
    "taint violation: tainted data flows into sensitive sink `int_to_string` in function `{function}`"
)]
pub struct TaintError {
    function: Symbol,
    location: Location,
}

/// LocalAnalysis performs taint propagation for a single function
struct LocalAnalysis<'a> {
    global: &'a GlobalAnalysis<'a>,
    func: &'a Function,
    context: Vec<Place>,
    facts: &'a Facts,
}

impl<'a> Analysis for LocalAnalysis<'a> {
    type Domain = TaintedMemLocs;

    const DIRECTION: Direction = Direction::Forward;

    fn bottom(&self, func: &Function) -> Self::Domain {
        let mut state = HashSet::new();

        // If this is a secure() function, mark its return value as tainted
        // Secure functions always return tainted data
        if func.secure() {
            let return_memloc = MemLoc::Abstract(func.name, usize::MAX);
            state.insert(return_memloc);
        }

        // If context has tainted parameters, use pointer analysis to find their MemLocs
        for place in &self.context {
            let (local, field_path) = place_to_local_field_path(place);

            // Look up in pointer_domain
            if let Some(points_to) = self.facts.pointer_domain.get(&(local, field_path)) {
                // Extract MemLocs from PointsToSet
                for (memloc, _) in points_to {
                    // Check if this MemLoc matches the field path
                    // For now, we'll add all MemLocs that could be at this location
                    // Field sensitivity will be handled during propagation
                    state.insert(*memloc);
                }
            }
        }

        state
    }

    fn handle_statement(&self, state: &mut Self::Domain, statement: &Statement, loc: Location) {
        let dst_local = statement.place.local;
        let (dst_local_idx, dst_field_path) = place_to_local_field_path(&statement.place);

        match &statement.rvalue {
            Rvalue::Operand(op) => {
                // Flow-sensitive assignment: check if RHS is constant
                let is_constant = match op {
                    Operand::Const(_) => true,
                    Operand::Place(p) => {
                        // Check constant domain
                        if let Some(const_val) = self.facts.constant_domain.get(&loc).get(&p.local)
                        {
                            matches!(const_val, Constant::Const(_))
                        } else {
                            false
                        }
                    }
                    Operand::Func { .. } => false,
                };

                if is_constant {
                    // Remove taint from LHS if RHS is constant (flow-sensitive)
                    // Clear all MemLocs that could be at this location
                    if let Some(points_to) = self
                        .facts
                        .pointer_domain
                        .get(&(dst_local_idx, dst_field_path.clone()))
                    {
                        for (memloc, _) in points_to {
                            state.remove(memloc);
                        }
                    }
                    // Also remove abstract return MemLocs for flow-sensitivity
                    // When assigning a constant, remove any abstract return MemLocs
                    let abstract_returns: Vec<MemLoc> = state
                        .iter()
                        .filter_map(|memloc| {
                            if let MemLoc::Abstract(_, usize::MAX) = memloc {
                                Some(*memloc)
                            } else {
                                None
                            }
                        })
                        .collect();
                    for return_memloc in &abstract_returns {
                        state.remove(return_memloc);
                    }
                } else if let Operand::Place(src_place) = op {
                    // Propagate taint from source to destination
                    let (src_local_idx, src_field_path) = place_to_local_field_path(src_place);

                    // Get all MemLocs that source points to
                    let mut src_memlocs = HashSet::new();
                    if let Some(points_to) = self
                        .facts
                        .pointer_domain
                        .get(&(src_local_idx, src_field_path.clone()))
                    {
                        for (memloc, field_path) in points_to {
                            // Field-sensitive: only propagate if field paths match or source is root
                            if src_field_path.is_empty() || field_path == &src_field_path {
                                src_memlocs.insert(*memloc);
                            }
                        }
                    }

                    // Check which of these MemLocs are tainted
                    let mut any_tainted = false;
                    for memloc in &src_memlocs {
                        if state.contains(memloc) {
                            any_tainted = true;
                            // Propagate to destination
                            if let Some(dst_points_to) = self
                                .facts
                                .pointer_domain
                                .get(&(dst_local_idx, dst_field_path.clone()))
                            {
                                for (dst_memloc, dst_field_path_in_alloc) in dst_points_to {
                                    // Field-sensitive propagation
                                    if dst_field_path.is_empty()
                                        || dst_field_path_in_alloc == &dst_field_path
                                    {
                                        state.insert(*dst_memloc);
                                    }
                                }
                            }
                        }
                    }

                    // Also check if source variable holds a tainted abstract return value
                    // For primitive types, we need to propagate abstract MemLocs that represent return values
                    // Collect abstract return MemLocs first to avoid borrow conflicts
                    let abstract_returns: Vec<MemLoc> = state
                        .iter()
                        .filter_map(|memloc| {
                            if let MemLoc::Abstract(_, usize::MAX) = memloc {
                                Some(*memloc)
                            } else {
                                None
                            }
                        })
                        .collect();

                    // If we have any abstract return MemLocs tainted, propagate to destination
                    // This handles the case where secure() returns a primitive value
                    // Always propagate abstract return MemLocs, even if destination has no points-to info
                    for return_memloc in &abstract_returns {
                        state.insert(*return_memloc);
                    }

                    // Also mark any MemLocs the destination points to as tainted
                    if any_tainted || !abstract_returns.is_empty() {
                        if let Some(dst_points_to) = self
                            .facts
                            .pointer_domain
                            .get(&(dst_local_idx, dst_field_path.clone()))
                        {
                            for (dst_memloc, _) in dst_points_to {
                                state.insert(*dst_memloc);
                            }
                        }
                    }
                }
            }
            Rvalue::Alloc { kind, args, .. } => {
                let allocation = MemLoc::Concrete(loc);

                // Check if any operands are tainted
                let mut any_tainted = false;

                if let AllocArgs::Lit(operands) = args {
                    match kind {
                        AllocKind::Tuple => {
                            // For tuple literals, check each operand
                            for (field_idx, operand) in operands.iter().enumerate() {
                                if let Operand::Place(op_place) = operand {
                                    let (op_local, op_field_path) =
                                        place_to_local_field_path(op_place);
                                    if let Some(points_to) =
                                        self.facts.pointer_domain.get(&(op_local, op_field_path))
                                    {
                                        for (memloc, _) in points_to {
                                            if state.contains(memloc) {
                                                any_tainted = true;
                                                // Mark corresponding field as tainted
                                                // The allocation's field path would be [field_idx]
                                                // For now, mark the whole allocation as tainted
                                                state.insert(allocation);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            // For other allocations, conservatively mark as tainted if any operand is tainted
                            for operand in operands {
                                if let Operand::Place(op_place) = operand {
                                    let (op_local, op_field_path) =
                                        place_to_local_field_path(op_place);
                                    if let Some(points_to) =
                                        self.facts.pointer_domain.get(&(op_local, op_field_path))
                                    {
                                        for (memloc, _) in points_to {
                                            if state.contains(memloc) {
                                                any_tainted = true;
                                                state.insert(allocation);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if any_tainted {
                    state.insert(allocation);
                }
            }
            Rvalue::Call { f, args } => {
                // Check if this is a call to secure()
                let mut is_secure = false;
                let mut func_name_opt = None;

                match f {
                    Operand::Func { f: func_name, .. } => {
                        func_name_opt = Some(*func_name);
                        if let Some(callee_func) = self.global.find_function(*func_name) {
                            is_secure = callee_func.secure();
                        }
                    }
                    Operand::Place(f_place) => {
                        // Function stored in a local - check if we can determine which function
                        // For now, check if any secure function could be called
                        // We'll need to track function values through constant propagation
                        let (f_local, f_field_path) = place_to_local_field_path(f_place);

                        // Check constant domain to see if this is a known function
                        if let Some(const_val) = self.facts.constant_domain.get(&loc).get(&f_local)
                        {
                            if let Constant::Closure(f_name) = const_val {
                                func_name_opt = Some(*f_name);
                                if let Some(callee_func) = self.global.find_function(*f_name) {
                                    is_secure = callee_func.secure();
                                }
                            }
                        }

                        // Also check all secure functions - if this local could point to any secure function
                        // This is conservative but necessary
                        // Collect function names first to avoid borrow conflicts
                        let secure_funcs: Vec<Symbol> = self
                            .global
                            .functions
                            .iter()
                            .filter(|f| f.secure())
                            .map(|f| f.name)
                            .collect();

                        if let Some(secure_func) = secure_funcs.first() {
                            // Check if this local could point to any secure function via pointer analysis
                            if let Some(points_to) =
                                self.facts.pointer_domain.get(&(f_local, f_field_path))
                            {
                                // If the local has any points-to information, it might point to a closure
                                // For now, conservatively assume it could be a secure function
                                is_secure = true;
                                func_name_opt = Some(*secure_func);
                            }
                        }
                    }
                    _ => {}
                }

                // Build call context: map caller operands to callee parameter Places
                let mut call_context = Vec::new();

                if let Some(func_name) = func_name_opt {
                    if let Some(callee_func) = self.global.find_function(func_name) {
                        // Map arguments to callee parameters
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if arg_idx < callee_func.num_params {
                                if let Operand::Place(arg_place) = arg {
                                    // Check if this argument is tainted in current state
                                    let (arg_local, arg_field_path) =
                                        place_to_local_field_path(arg_place);
                                    let mut arg_tainted = false;

                                    if let Some(points_to) =
                                        self.facts.pointer_domain.get(&(arg_local, arg_field_path))
                                    {
                                        for (memloc, _) in points_to {
                                            if state.contains(memloc) {
                                                arg_tainted = true;
                                            }
                                        }
                                    }

                                    // Also check for abstract return MemLocs (for primitives)
                                    for memloc in state.iter() {
                                        if let MemLoc::Abstract(_, usize::MAX) = memloc {
                                            arg_tainted = true;
                                            break;
                                        }
                                    }

                                    if arg_tainted {
                                        // Create Place for callee parameter
                                        let callee_param_local = LocalIdx::from_usize(arg_idx);
                                        let callee_param_ty =
                                            callee_func.locals.value(callee_param_local).ty;
                                        let callee_place = Place::new(
                                            callee_param_local,
                                            arg_place.projection.clone(),
                                            callee_param_ty,
                                        );
                                        call_context.push(callee_place);
                                    }
                                }
                            }
                        }

                        // Recursively analyze callee (for non-secure functions)
                        // Only analyze if it's not a recursive call and not secure (secure handled above)
                        if !is_secure && callee_func.name != self.func.name {
                            // Note: We have a borrow conflict here - self.facts is borrowed in LocalAnalysis
                            // but analyze_function needs to borrow self.global.facts. We can work around this
                            // by ensuring facts are computed before we get here, so analyze_function won't
                            // need to mutate the facts RefCell.
                            //
                            // However, analyze_function can still cause issues if it tries to borrow_mut
                            // for results. The safest approach is to avoid recursive calls here and instead
                            // do a separate pass. But that breaks the specification.
                            //
                            // Actually, analyze_function only needs to read facts (borrow), not write to them
                            // if they're already computed. And results uses a different RefCell, so it should work.
                            // Let's try it - if there's still a conflict, we'll need to restructure.
                            let callee_tainted =
                                self.global.analyze_function(callee_func, call_context);

                            // Propagate taint from callee's outputs to return value
                            state.extend(callee_tainted.iter().cloned());
                        }
                    }
                }

                // If secure(), mark return value as tainted
                if is_secure {
                    if let Some(func_name) = func_name_opt {
                        // Create an abstract MemLoc representing the return value of this secure() call
                        // Use a special index (usize::MAX) to represent return values
                        let return_memloc = MemLoc::Abstract(func_name, usize::MAX);
                        state.insert(return_memloc);

                        // Also mark any MemLocs that the destination points to (for pointer types)
                        // This handles both primitive and pointer return types
                        if let Some(dst_points_to) = self
                            .facts
                            .pointer_domain
                            .get(&(dst_local_idx, dst_field_path.clone()))
                        {
                            for (memloc, _) in dst_points_to {
                                state.insert(*memloc);
                            }
                        }
                    }
                }

                // For unknown functions or recursive calls, conservatively mark return as tainted
                if !is_secure && matches!(f, Operand::Place(_)) {
                    // Unknown function call
                    if let Some(dst_points_to) = self
                        .facts
                        .pointer_domain
                        .get(&(dst_local_idx, dst_field_path.clone()))
                    {
                        for (memloc, _) in dst_points_to {
                            state.insert(*memloc);
                        }
                    }
                }
            }
            Rvalue::Cast { op, .. } => {
                // Propagate taint through casts
                if let Operand::Place(src_place) = op {
                    let (src_local, src_field_path) = place_to_local_field_path(src_place);
                    if let Some(points_to) =
                        self.facts.pointer_domain.get(&(src_local, src_field_path))
                    {
                        for (memloc, _) in points_to {
                            if state.contains(memloc) {
                                // Propagate to destination
                                if let Some(dst_points_to) = self
                                    .facts
                                    .pointer_domain
                                    .get(&(dst_local_idx, dst_field_path.clone()))
                                {
                                    for (dst_memloc, _) in dst_points_to {
                                        state.insert(*dst_memloc);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // Other rvalues: propagate taint conservatively through pointer analysis
            }
        }
    }

    fn handle_terminator(&self, state: &mut Self::Domain, terminator: &Terminator, _loc: Location) {
        // For return statements, taint is already in state
        // This is a no-op since we're forward analysis
        match terminator.kind() {
            TerminatorKind::Return(_) => {
                // Taint is already propagated to return value in handle_statement
            }
            _ => {}
        }
    }
}

impl JoinSemiLattice for TaintedMemLocs {
    fn join(&mut self, other: &Self) -> bool {
        let before_len = self.len();
        self.extend(other.iter().cloned());
        self.len() > before_len
    }
}
