//! Visitor traits for the bytecode.

#![allow(unused)]

use either::Either;
use itertools::Itertools;

use crate::bc::types::*;

pub trait Visit {
    fn visit_program(&mut self, prog: &Program) {
        self.super_visit_program(prog);
    }

    fn super_visit_program(&mut self, prog: &Program) {
        for func in prog.functions() {
            self.visit_function(func);
        }
    }

    fn visit_function(&mut self, func: &Function) {
        self.super_visit_function(func);
    }

    fn super_visit_function(&mut self, func: &Function) {
        self.visit_body(&func.body);
    }

    fn visit_body(&mut self, body: &Body) {
        self.super_visit_body(body);
    }

    fn super_visit_body(&mut self, body: &Body) {
        let locations = body.locations().iter().copied();
        for loc in locations {
            match body.instr(loc) {
                Either::Left(stmt) => self.visit_statement(stmt, loc),
                Either::Right(term) => self.visit_terminator(term, loc),
            }
        }
    }

    fn visit_statement(&mut self, stmt: &Statement, loc: Location) {
        self.super_visit_statement(stmt, loc);
    }

    fn super_visit_statement(&mut self, stmt: &Statement, loc: Location) {
        self.visit_lvalue(&stmt.place, loc);
        self.visit_rvalue(&stmt.rvalue, loc);
    }

    fn visit_terminator(&mut self, term: &Terminator, loc: Location) {
        self.super_visit_terminator(term, loc);
    }

    fn super_visit_terminator(&mut self, term: &Terminator, loc: Location) {
        match term.kind() {
            TerminatorKind::Return(op) => {
                self.visit_operand(op, loc);
            }
            TerminatorKind::Jump(_) => {}
            TerminatorKind::CondJump { cond, .. } => {
                self.visit_operand(cond, loc);
            }
        }
    }

    fn visit_lvalue(&mut self, _place: &Place, _loc: Location) {}

    fn visit_rvalue_place(&mut self, _place: &Place, _loc: Location) {}

    fn visit_rvalue(&mut self, rvalue: &Rvalue, loc: Location) {
        self.super_visit_rvalue(rvalue, loc);
    }

    fn super_visit_rvalue(&mut self, rvalue: &Rvalue, loc: Location) {
        match rvalue {
            Rvalue::Operand(op) => {
                self.visit_operand(op, loc);
            }
            Rvalue::Cast { op, .. } => {
                self.visit_operand(op, loc);
            }
            Rvalue::Closure { env, .. } => {
                for op in env {
                    self.visit_operand(op, loc);
                }
            }
            Rvalue::Alloc { args, .. } => match args {
                AllocArgs::Lit(args) => {
                    for op in args {
                        self.visit_operand(op, loc);
                    }
                }
            },
            Rvalue::Binop { left, right, .. } => {
                self.visit_operand(left, loc);
                self.visit_operand(right, loc);
            }
            Rvalue::Call { f, args, .. } => {
                self.visit_operand(f, loc);
                for arg in args {
                    self.visit_operand(arg, loc);
                }
            }
            Rvalue::MethodCall { receiver, args, .. } => {
                self.visit_operand(receiver, loc);
                for arg in args {
                    self.visit_operand(arg, loc);
                }
            }
        }
    }

    fn visit_operand(&mut self, operand: &Operand, loc: Location) {
        self.super_visit_operand(operand, loc);
    }

    fn super_visit_operand(&mut self, operand: &Operand, loc: Location) {
        match operand {
            Operand::Place(place) => {
                self.visit_rvalue_place(place, loc);
            }
            Operand::Const(_) => {}
            Operand::Func { .. } => {}
        }
    }
}

pub trait VisitMut {
    fn visit_program(&mut self, prog: &mut Program) {
        self.super_visit_program(prog);
    }

    fn super_visit_program(&mut self, prog: &mut Program) {
        for func in prog.functions_mut() {
            self.visit_function(func);
        }
    }

    fn visit_function(&mut self, func: &mut Function) {
        self.super_visit_function(func);
    }

    fn super_visit_function(&mut self, func: &mut Function) {
        self.visit_body(&mut func.body);
    }

    fn visit_body(&mut self, body: &mut Body) {
        self.super_visit_body(body);
    }

    fn super_visit_body(&mut self, body: &mut Body) {
        let blocks = body.blocks().collect_vec();
        for block in blocks {
            self.visit_basic_block(body.data_mut(block), block);
        }
    }

    fn visit_basic_block(&mut self, data: &mut BasicBlock, block: BasicBlockIdx) {
        self.super_visit_basic_block(data, block);
    }

    fn super_visit_basic_block(&mut self, data: &mut BasicBlock, block: BasicBlockIdx) {
        for (instr, statement) in data.statements.iter_mut().enumerate() {
            self.visit_statement(statement, Location { block, instr })
        }

        let term_loc = Location {
            block,
            instr: data.terminator_index(),
        };
        self.visit_terminator(&mut data.terminator, term_loc);
    }

    fn visit_statement(&mut self, stmt: &mut Statement, loc: Location) {
        self.super_visit_statement(stmt, loc);
    }

    fn super_visit_statement(&mut self, stmt: &mut Statement, loc: Location) {
        self.visit_place(&mut stmt.place, loc);
        self.visit_rvalue(&mut stmt.rvalue, loc);
    }

    fn visit_terminator(&mut self, term: &mut Terminator, loc: Location) {
        self.super_visit_terminator(term, loc);
    }

    fn super_visit_terminator(&mut self, term: &mut Terminator, loc: Location) {
        match term.kind_mut() {
            TerminatorKind::Return(op) => {
                self.visit_operand(op, loc);
            }
            TerminatorKind::Jump(_) => {}
            TerminatorKind::CondJump { cond, .. } => {
                self.visit_operand(cond, loc);
            }
        }
    }

    fn visit_place(&mut self, place: &mut Place, loc: Location) {
        self.super_visit_place(place, loc);
    }

    fn super_visit_place(&mut self, place: &mut Place, loc: Location) {
        let new_projection = place
            .projection
            .iter()
            .map(|elem| match elem.clone() {
                ProjectionElem::Field { index, ty } => ProjectionElem::Field { index, ty },
            })
            .collect_vec();
        if place.projection != new_projection {
            *place = Place::new(place.local, new_projection, place.ty);
        }
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue, loc: Location) {
        self.super_visit_rvalue(rvalue, loc);
    }

    fn super_visit_rvalue(&mut self, rvalue: &mut Rvalue, loc: Location) {
        match rvalue {
            Rvalue::Operand(op) => {
                self.visit_operand(op, loc);
            }
            Rvalue::Cast { op, .. } => {
                self.visit_operand(op, loc);
            }
            Rvalue::Closure { env, .. } => {
                for op in env {
                    self.visit_operand(op, loc);
                }
            }
            Rvalue::Alloc { args, .. } => match args {
                AllocArgs::Lit(args) => {
                    for op in args {
                        self.visit_operand(op, loc);
                    }
                }
            },
            Rvalue::Binop { left, right, .. } => {
                self.visit_operand(left, loc);
                self.visit_operand(right, loc);
            }
            Rvalue::Call { f, args, .. } => {
                self.visit_operand(f, loc);
                for arg in args {
                    self.visit_operand(arg, loc);
                }
            }
            Rvalue::MethodCall { receiver, args, .. } => {
                self.visit_operand(receiver, loc);
                for arg in args {
                    self.visit_operand(arg, loc);
                }
            }
        }
    }

    fn visit_operand(&mut self, operand: &mut Operand, loc: Location) {
        self.super_visit_operand(operand, loc);
    }

    fn super_visit_operand(&mut self, operand: &mut Operand, loc: Location) {
        match operand {
            Operand::Place(place) => {
                self.visit_place(place, loc);
            }
            Operand::Const(_) => {}
            Operand::Func { .. } => {}
        }
    }
}
