//! Type definitions for the bytecode.

use std::{collections::HashMap, sync::Arc};

use either::Either;
use indexical::IndexedDomain;
use internment::Intern;
use itertools::Itertools;
use petgraph::{
    Direction,
    graph::{DiGraph, NodeIndex},
};
use serde::Serialize;
use smallvec::{SmallVec, smallvec};
use strum::Display;

use crate::{
    ast::types::{Annotation, Span},
    interned,
    utils::{self, Symbol},
};

pub use crate::tir::types::{Binop, Const, Type, TypeKind};

#[derive(Debug, Clone, Serialize)]
pub struct Program(Vec<Function>);

impl Program {
    pub fn new(funcs: Vec<Function>) -> Self {
        Program(funcs)
    }

    pub fn functions(&self) -> &[Function] {
        &self.0
    }

    pub fn functions_mut(&mut self) -> &mut [Function] {
        &mut self.0
    }

    pub fn into_functions(self) -> Vec<Function> {
        self.0
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, Serialize)]
pub struct Local {
    /// The type of the local variable.
    pub ty: Type,

    /// A source-level name for the local variable, if it exists.
    pub name: Option<Symbol>,
}

indexical::define_index_type! {
    /// An index for locals, for use in indexical data structures.
    pub struct LocalIdx for Local = u32;
}

#[derive(Debug, Clone, Serialize)]
pub struct Function {
    /// The global name of the function. Must be unique.
    pub name: Symbol,

    /// The first `num_param` locals are parameters.
    pub num_params: usize,

    /// All the local variables in the function.
    pub locals: Arc<IndexedDomain<Local>>,

    /// The return type.
    pub ret_ty: Type,

    /// The control-flow graph of instructions.
    pub body: Body,

    /// Source-level annotations attached to this function.
    pub annots: Vec<Annotation>,
}

impl Function {
    /// Returns an iterator over the local index and type of each parameter.
    pub fn params(&self) -> impl Iterator<Item = (LocalIdx, Type)> {
        (0..self.num_params).map(|i| {
            let local = LocalIdx::from_usize(i);
            (local, self.locals.value(local).ty)
        })
    }

    /// Returns true if the function is annotated with `#[jit]`.
    pub fn jit(&self) -> bool {
        self.annots.iter().any(|annot| annot.name == "jit")
    }

    /// Returns true if the function is annotated with `#[secure]`.
    pub fn secure(&self) -> bool {
        self.annots.iter().any(|annot| annot.name == "secure")
    }
}

pub type Cfg = DiGraph<BasicBlock, ()>;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Body {
    /// The instructions in the body.    
    cfg: Cfg,

    /// A set of all locations in the body, useful for maps indexed by location.
    locations: Arc<IndexedDomain<Location>>,

    /// Reverse-postorder traversal of basic blocks in the CFG.    
    #[serde(skip)]
    rpo: Vec<BasicBlockIdx>,
}

impl Body {
    pub fn new(cfg: Cfg) -> Self {
        let rpo = utils::reverse_post_order(&cfg, Location::START.block.into())
            .into_iter()
            .map(BasicBlockIdx::from)
            .collect_vec();

        let locations = rpo
            .iter()
            .copied()
            .flat_map(|block| {
                let num_instrs = cfg.node_weight(block.into()).unwrap().statements.len() + 1;
                (0..num_instrs).map(move |instr| Location { block, instr })
            })
            .collect();
        let locations = Arc::new(locations);

        Body {
            cfg,
            rpo,
            locations,
        }
    }

    /// Returns the data corresponding to the given basic block.
    pub fn data(&self, node: BasicBlockIdx) -> &BasicBlock {
        self.cfg.node_weight(node.into()).unwrap()
    }

    /// Returns a mutable handle to the data in a given basic block.
    pub fn data_mut(&mut self, node: BasicBlockIdx) -> &mut BasicBlock {
        self.cfg.node_weight_mut(node.into()).unwrap()
    }

    /// Returns an iterator over basic block indices.
    ///
    /// Guaranteed to be a reverse-postorder.
    pub fn blocks(&self) -> impl DoubleEndedIterator<Item = BasicBlockIdx> {
        self.rpo.iter().copied()
    }

    /// Returns the underlying CFG.
    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    /// Returns the location domain.
    pub fn locations(&self) -> &Arc<IndexedDomain<Location>> {
        &self.locations
    }

    /// Returns the instruction at a given location.
    pub fn instr(&self, loc: Location) -> Either<&Statement, &Terminator> {
        self.data(loc.block).get(loc.instr)
    }

    /// Returns a mutable handle to the instruction at a given locatin.
    pub fn instr_mut(&mut self, loc: Location) -> Either<&mut Statement, &mut Terminator> {
        self.data_mut(loc.block).get_mut(loc.instr)
    }

    /// Returns all locations which can possibly come after the given location.
    pub fn successors(&self, loc: Location) -> SmallVec<[Location; 2]> {
        if loc.instr == self.data(loc.block).terminator_index() {
            self.cfg
                .neighbors_directed(loc.block.into(), Direction::Outgoing)
                .map(|block| Location {
                    block: BasicBlockIdx::from(block),
                    instr: 0,
                })
                .collect()
        } else {
            smallvec![Location {
                block: loc.block,
                instr: loc.instr + 1,
            }]
        }
    }

    /// Returns all locations which can possibly come before the given location.
    pub fn predecessors(&self, loc: Location) -> SmallVec<[Location; 2]> {
        if loc.instr == 0 {
            self.cfg
                .neighbors_directed(loc.block.into(), Direction::Incoming)
                .map(|block| {
                    let block = BasicBlockIdx::from(block);
                    let instr = self.data(block).terminator_index();
                    Location { block, instr }
                })
                .collect()
        } else {
            smallvec![Location {
                block: loc.block,
                instr: loc.instr - 1
            }]
        }
    }
}

index_vec::define_index_type! {
    pub struct BasicBlockIdx = u32;
}

impl BasicBlockIdx {
    /// Returns the location of the first instruction in the basic block.
    pub fn entry(self) -> Location {
        Location {
            block: self,
            instr: 0,
        }
    }
}

impl From<NodeIndex> for BasicBlockIdx {
    fn from(value: NodeIndex) -> Self {
        BasicBlockIdx::new(value.index())
    }
}

impl From<BasicBlockIdx> for NodeIndex {
    fn from(value: BasicBlockIdx) -> Self {
        NodeIndex::new(value.index())
    }
}

/// A coordinate for a particular instruction in a CFG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Location {
    /// The basic block containing the instruction.
    pub block: BasicBlockIdx,

    /// The index of the instruction within the basic block.
    pub instr: usize,
}

indexical::define_index_type! {
    pub struct LocationIdx for Location = u32;
}

impl Location {
    /// The location of the starting instruction in any CFG.
    pub const START: Location = Location {
        block: BasicBlockIdx { _raw: 0 },
        instr: 0,
    };
}

/// A basic block in the control flow graph, containing a sequence of statements followed by a terminator.
#[derive(Debug, Clone, Serialize)]
pub struct BasicBlock {
    /// Statements executed in sequence.
    pub statements: Vec<Statement>,

    /// The final instruction, which can branch to other basic blocks.
    pub terminator: Terminator,
}

impl BasicBlock {
    /// Get the ith instruction in the basic block.
    pub fn get(&self, i: usize) -> Either<&Statement, &Terminator> {
        assert!(i <= self.statements.len());
        if i == self.statements.len() {
            Either::Right(&self.terminator)
        } else {
            Either::Left(&self.statements[i])
        }
    }

    // Get a mutable handle to the ith instruction in the basic block.
    pub fn get_mut(&mut self, i: usize) -> Either<&mut Statement, &mut Terminator> {
        assert!(i <= self.statements.len());
        if i == self.statements.len() {
            Either::Right(&mut self.terminator)
        } else {
            Either::Left(&mut self.statements[i])
        }
    }

    pub fn terminator_index(&self) -> usize {
        self.statements.len()
    }
}

/// An instruction of the form `place = rvalue`.
#[derive(Debug, Clone, Serialize)]
pub struct Statement {
    /// The place in memory being assigned to.
    pub place: Place,

    /// An expression that generates a value to set to the place.
    pub rvalue: Rvalue,

    /// A best-effort source-level span for the statement.
    pub span: Span,
}

/// An argument to an [`Rvalue`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Operand {
    Const(Const),
    Place(Place),
    Func { f: Symbol, ty: Type },
}

impl Operand {
    pub fn ty(&self) -> Type {
        match self {
            Operand::Place(p) => p.ty,
            Operand::Const(c) => c.ty(),
            Operand::Func { ty, .. } => *ty,
        }
    }

    pub fn as_place(&self) -> Option<Place> {
        match self {
            Operand::Place(p) => Some(*p),
            _ => None,
        }
    }
}

/// The kind of object allocated by an [`Rvalue::Alloc`] instruction.
#[derive(Debug, Clone, Copy, Display, Serialize)]
pub enum AllocKind {
    Struct,
    Tuple,
}

/// The location in memory of an object allocated by an [`Rvalue::Alloc`] instruction.
#[derive(Debug, Clone, Copy, Display, Serialize)]
pub enum AllocLoc {
    Stack,
    Heap,
}

/// The inputs to an allocation by an [`Rvalue::Alloc`] instruction.
#[derive(Debug, Clone, Display, Serialize)]

pub enum AllocArgs {
    /// A fixed-size literal, e.g. `(x, y)` or `[x, y]`.
    Lit(Vec<Operand>),
}

/// An expression that is assigned to a [`Place`] in a [`Statement`].
#[derive(Debug, Clone, Serialize)]
pub enum Rvalue {
    /// The direct value of an operand, e.g. `1` or `x`.
    Operand(Operand),

    /// Cast another operand to a new type, e.g. `x as float`.
    Cast { op: Operand, ty: Type },

    /// A first-class function with a (potentially-empty) environment, e.g., `closure(lambda, [x, y])`.
    Closure { f: Symbol, env: Vec<Operand> },

    /// An allocation of a data structure larger than a register.
    Alloc {
        kind: AllocKind,
        loc: AllocLoc,
        args: AllocArgs,
    },

    /// A binary operator.
    Binop {
        op: Binop,
        left: Operand,
        right: Operand,
    },

    /// A call to a non-method function.
    Call { f: Operand, args: Vec<Operand> },

    /// A call to a method function w/ dynamic dispatch.
    MethodCall {
        /// The object containing a vtable.
        receiver: Operand,

        /// The index of the vtable to lookup.
        method: usize,

        /// The arguments to the method.
        args: Vec<Operand>,
    },
}

impl Rvalue {
    /// Returns a vector of all the places used in the rvalue.
    pub fn places(&self) -> SmallVec<[Place; 2]> {
        match self {
            Rvalue::Operand(op) | Rvalue::Cast { op, .. } => op.as_place().into_iter().collect(),
            Rvalue::Alloc { args, .. } => match args {
                AllocArgs::Lit(ops) => ops.iter().flat_map(|op| op.as_place()).collect(),
            },
            Rvalue::Call { args: ops, .. } | Rvalue::Closure { env: ops, .. } => {
                ops.iter().filter_map(|op| op.as_place()).collect()
            }
            Rvalue::MethodCall { receiver, args, .. } => args
                .iter()
                .chain([receiver])
                .filter_map(|op| op.as_place())
                .collect(),
            Rvalue::Binop { left, right, .. } => left
                .as_place()
                .into_iter()
                .chain(right.as_place())
                .collect(),
        }
    }
}

/// The final instruction in a basic block.
#[derive(Debug, Clone, Serialize)]
pub struct Terminator {
    pub kind: TerminatorKind,
    pub span: Span,
}

impl Terminator {
    pub fn kind(&self) -> &TerminatorKind {
        &self.kind
    }

    pub fn kind_mut(&mut self) -> &mut TerminatorKind {
        &mut self.kind
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum TerminatorKind {
    /// Return the value of the operand from the function.
    Return(Operand),

    /// Unconditionally jump to the given basic block.
    Jump(BasicBlockIdx),

    /// Conditionally jump to `true_` if `cond` is true, and jump to `false_` otherwise.
    CondJump {
        cond: Operand,
        true_: BasicBlockIdx,
        false_: BasicBlockIdx,
    },
}

impl Terminator {
    /// Remap the basic blocks inside the terminator, used during CFG construction.
    pub fn remap(&mut self, map: &HashMap<BasicBlockIdx, BasicBlockIdx>) {
        match &mut self.kind {
            TerminatorKind::Jump(block) => *block = map[block],
            TerminatorKind::CondJump { true_, false_, .. } => {
                *true_ = map[true_];
                *false_ = map[false_];
            }
            TerminatorKind::Return(..) => {}
        }
    }
}

/// A reference to a particular location in memory relative to a particular function.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct PlaceData {
    /// The root of the reference, a local on a stack frame.
    pub local: LocalIdx,

    /// A sequence of projections off of the local.
    pub projection: Vec<ProjectionElem>,

    /// The type of the data at the location.
    pub ty: Type,
}

interned!(Place, PlaceData);

impl Place {
    pub fn new(local: LocalIdx, projection: Vec<ProjectionElem>, ty: Type) -> Self {
        Place(Intern::new(PlaceData {
            local,
            projection,
            ty,
        }))
    }

    pub fn extend_projection(
        self,
        elems: impl IntoIterator<Item = ProjectionElem>,
        ty: Type,
    ) -> Self {
        let mut projection = self.projection.clone();
        projection.extend(elems);
        Place::new(self.local, projection, ty)
    }
}

/// A lookup into a field of a data structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum ProjectionElem {
    /// Get the field of a fixed-sized structure.
    Field { index: usize, ty: Type },
}
