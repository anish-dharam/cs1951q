//! Pretty-printing for the bytecode.

use std::fmt;

use crate::{
    bc::types::{AllocArgs, BasicBlockIdx, Location},
    utils::{indent, write_comma_separated, write_newline_separated},
};

use super::types::{
    BasicBlock, Body, Function, LocalIdx, Operand, Place, Program, ProjectionElem, Rvalue,
    Statement, Terminator, TerminatorKind,
};

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for func in self.functions() {
            write!(f, "{func}\n\n")?;
        }
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}(", self.name)?;
        write_comma_separated(f, self.params().map(|(local, ty)| format!("{local}: {ty}")))?;
        write!(f, ") -> {} {{\n", self.ret_ty,)?;
        for (local, data) in self.locals.iter_enumerated() {
            write!(f, "  local {local}: {}\n", data.ty)?;
        }
        write!(f, "{}\n}}", indent(format!("{}", self.body)))?;
        Ok(())
    }
}

impl fmt::Display for Body {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_newline_separated(
            f,
            self.blocks().map(|loc| {
                let block = self.data(loc);
                format!("{loc}:\n{}", indent(format!("{block}")))
            }),
        )
    }
}

impl fmt::Display for BasicBlockIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.index())
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for s in &self.statements {
            write!(f, "{s}\n")?;
        }
        write!(f, "{}", self.terminator)?;
        Ok(())
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.place, self.rvalue)
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Place(p) => write!(f, "{p}"),
            Operand::Const(c) => write!(f, "{c}"),
            Operand::Func { f: func, .. } => write!(f, "{func}"),
        }
    }
}

impl fmt::Display for Rvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rvalue::Operand(op) => write!(f, "{op}"),
            Rvalue::Cast { op, ty } => write!(f, "{op} as {ty}"),
            Rvalue::Alloc {
                args, kind, loc, ..
            } => {
                write!(f, "alloc[{kind} @ {loc}](")?;
                match args {
                    AllocArgs::Lit(args) => write_comma_separated(f, args)?,
                    AllocArgs::ArrayCopy { value, count } => write!(f, "[{value}; {count}]")?,
                };
                write!(f, ")")
            }
            Rvalue::Binop { op, left, right } => write!(f, "{left} {op} {right}"),
            Rvalue::Call { f: func, args, .. } => {
                write!(f, "{func}(")?;
                write_comma_separated(f, args)?;
                write!(f, ")")
            }
            Rvalue::MethodCall {
                receiver,
                method,
                args,
            } => {
                write!(f, "{receiver}.{method}(")?;
                write_comma_separated(f, args)?;
                write!(f, ")")
            }
            Rvalue::Closure { f: f_name, env } => {
                write!(f, "closure({f_name}, [")?;
                write_comma_separated(f, env)?;
                write!(f, "])")
            }
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl fmt::Display for TerminatorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminatorKind::Return(op) => write!(f, "return {op}"),
            TerminatorKind::Jump(loc) => write!(f, "jump {loc}"),
            TerminatorKind::CondJump {
                cond,
                true_,
                false_,
            } => {
                write!(f, "if {cond} {{ jump {true_} }} else {{ jump {false_} }}",)
            }
        }
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.local)?;
        for proj in &self.projection {
            match proj {
                ProjectionElem::Field { index, .. } => write!(f, ".{}", index)?,
                ProjectionElem::ArrayIndex { index, .. } => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

impl fmt::Display for LocalIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.index())
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}[{}]", self.block.index(), self.instr)
    }
}
