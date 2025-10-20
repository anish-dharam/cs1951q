//! Intraprocedural dataflow analysis for the bytecode.
//!
//! API design is heavily inspired by the [rustc_mir_dataflow](https://doc.rust-lang.org/stable/nightly-rustc/rustc_mir_dataflow/index.html) crate.

#![allow(unused)]

use either::Either;
use indexical::{
    IndexedValue, bitset::bitvec::ArcIndexSet as IndexSet, vec::ArcIndexVec as IndexVec,
};
use itertools::fold;
use std::{
    collections::{HashSet, VecDeque},
    iter::successors,
    sync::Arc,
};
use wasmparser::collections::Set;

use crate::bc::types::LocalIdx;

use super::types::{Function, Location, Statement, Terminator};

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
        let def: HashSet<LocalIdx> = HashSet::from([statement.place.local]);
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

pub fn dead_code(func: &mut Function) -> bool {
    true
    // return true if code was eliminated
}
