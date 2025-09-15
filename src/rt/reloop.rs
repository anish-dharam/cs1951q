//! Converts a CFG into structured control flow, for use w/ Wasm.
//!
//! Implements the ["Beyond Relooper" algorithm](https://dl.acm.org/doi/pdf/10.1145/3547621)
//! with [modifications from GHC](https://github.com/ghc/ghc/blob/2d0ecdc6ef5d3372ad00038dc38a2b84229d9cf5/compiler/GHC/Wasm/ControlFlow/FromCmm.hs#L147).

use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

use itertools::Itertools;
use petgraph::{
    Direction,
    algo::dominators::{self, Dominators},
    graph::{DiGraph, NodeIndex},
    visit::EdgeRef,
};
use wasm_encoder::{BlockType, Function, ValType};

use crate::{
    bc::types::{BasicBlockIdx, Location},
    utils,
};

pub struct WasmBlock {
    pub instrs: Vec<u8>,
    pub terminator: WasmTerminator,
}

pub enum WasmTerminator {
    Br(BasicBlockIdx),
    BrIf(BasicBlockIdx, BasicBlockIdx),
    Return,
}

pub type WasmCfg = DiGraph<WasmBlock, ()>;

pub fn reloop(func: &mut Function, graph: WasmCfg, return_type: ValType) {
    let root = Location::START.block;
    let doms = dominators::simple_fast(&graph, root.into());
    let rpo = utils::reverse_post_order(&graph, root.into())
        .into_iter()
        .enumerate()
        .map(|(i, node)| (BasicBlockIdx::from(node), i))
        .collect::<HashMap<_, _>>();
    let merge_nodes = graph
        .node_indices()
        .filter(|loc| graph.edges_directed(*loc, Direction::Incoming).count() >= 2)
        .map(BasicBlockIdx::from)
        .collect::<HashSet<_>>();
    let loop_nodes = graph
        .node_indices()
        .filter(|loc| {
            graph
                .edges_directed(*loc, Direction::Incoming)
                .any(|edge| rpo[&edge.source().into()] > rpo[&(*loc).into()])
        })
        .map(BasicBlockIdx::from)
        .collect::<HashSet<_>>();
    Relooper {
        func: RefCell::new(func),
        graph,
        doms,
        merge_nodes,
        loop_nodes,
        rpo,
    }
    .do_tree(BlockType::Result(return_type), root, &[])
}

struct Relooper<'a> {
    func: RefCell<&'a mut Function>,
    graph: WasmCfg,
    doms: Dominators<NodeIndex>,
    merge_nodes: HashSet<BasicBlockIdx>,
    loop_nodes: HashSet<BasicBlockIdx>,
    rpo: HashMap<BasicBlockIdx, usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum ContainingSyntax {
    IfThenElse,
    #[allow(unused)]
    LoopHeadedBy(BasicBlockIdx),
    BlockFollowedBy(BasicBlockIdx),
}

type Context = [ContainingSyntax];

macro_rules! instr {
    ($self:expr) => {
        $self.func.borrow_mut().instructions()
    };
}

impl Relooper<'_> {
    fn do_tree(&self, return_type: BlockType, root: BasicBlockIdx, context: &Context) {
        let children = self
            .doms
            .immediately_dominated_by(root.into())
            .map(BasicBlockIdx::from)
            .filter(|child| self.merge_nodes.contains(child))
            .sorted_by_key(|child| self.rpo[child])
            .collect::<Vec<_>>();

        let is_loop_header = self.loop_nodes.contains(&root);
        if is_loop_header {
            instr!(self).loop_(return_type);

            let mut new_context = context.to_vec();
            new_context.push(ContainingSyntax::LoopHeadedBy(root));
            self.node_within(return_type, root, children, &new_context);

            instr!(self).end();
        } else {
            self.node_within(return_type, root, children, context)
        }
    }

    fn node_within(
        &self,
        return_type: BlockType,
        node: BasicBlockIdx,
        mut children: Vec<BasicBlockIdx>,
        context: &Context,
    ) {
        match children.pop() {
            Some(y_n) => {
                let mut new_context = context.to_vec();
                new_context.push(ContainingSyntax::BlockFollowedBy(y_n));

                instr!(self).block(BlockType::Empty);
                self.node_within(BlockType::Empty, node, children, &new_context);
                instr!(self).end();

                self.do_tree(return_type, y_n, context);
            }
            None => {
                let block = &self.graph.node_weight(node.into()).unwrap();
                self.func.borrow_mut().raw(block.instrs.iter().copied());

                match block.terminator {
                    WasmTerminator::Br(dst) => self.do_branch(return_type, node, dst, context),
                    WasmTerminator::BrIf(true_, false_) => {
                        let mut new_context = context.to_vec();
                        new_context.push(ContainingSyntax::IfThenElse);
                        instr!(self).if_(return_type);
                        self.do_branch(return_type, node, true_, &new_context);
                        instr!(self).else_();
                        self.do_branch(return_type, node, false_, &new_context);
                        instr!(self).end();
                    }
                    WasmTerminator::Return => {
                        instr!(self).return_();
                    }
                }
            }
        }
    }

    fn do_branch(
        &self,
        return_type: BlockType,
        src: BasicBlockIdx,
        dst: BasicBlockIdx,
        context: &Context,
    ) {
        macro_rules! index {
            () => {
                instr!(self).br(context
                    .iter()
                    .rev()
                    .position(|ctx| match ctx {
                        ContainingSyntax::BlockFollowedBy(l) => *l == dst,
                        ContainingSyntax::LoopHeadedBy(l) => *l == dst,
                        _ => false,
                    })
                    .unwrap() as u32)
            };
        }
        if self.rpo[&dst] <= self.rpo[&src] || self.merge_nodes.contains(&dst) {
            index!();
        } else {
            self.do_tree(return_type, dst, context)
        }
    }
}
