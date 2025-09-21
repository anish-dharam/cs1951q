//! Pretty-printer for the TIR.

use std::fmt;

use super::types::{Expr, ExprKind, Function, ParamList, Program};
use crate::{
    tir::types::MethodRef,
    utils::{indent, write_comma_separated},
};

fn write_params(f: &mut fmt::Formatter, params: &ParamList) -> fmt::Result {
    write_comma_separated(f, params.iter().map(|(name, ty)| format!("{name}: {ty}")))
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for func in self.functions() {
            write!(f, "{func}\n")?;
        }
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for annot in &self.annots {
            write!(f, "{annot}\n")?;
        }
        write!(f, "fn {}(", self.name)?;
        write_params(f, &self.params)?;
        write!(
            f,
            ") -> {} {{\n{}\n}}",
            self.ret_ty,
            indent(format!("{}", self.body))
        )
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if std::env::var("PRINT_TYPES").is_ok() {
            write!(f, "{} @{}", self.kind, self.ty)
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl fmt::Display for MethodRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{{}::{}}}", self.interface, self.method)
    }
}

impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ExprKind::Var(s) => write!(f, "{s}"),
            ExprKind::Const(c) => write!(f, "{c}"),
            ExprKind::Tuple(es) => {
                write!(f, "(")?;
                write_comma_separated(f, es)?;
                write!(f, ")")
            }
            ExprKind::Struct(es) => {
                write!(f, "(")?;
                write_comma_separated(f, es)?;
                write!(f, ")")
            }
            ExprKind::Project { e, i } => write!(f, "{}.{}", e, i),
            ExprKind::BinOp { left, right, op } => write!(f, "{} {} {}", left, op, right),
            ExprKind::Cast { e, ty } => write!(f, "{} as {}", e, ty),
            ExprKind::Call { f: func, args } => {
                write!(f, "{}(", func)?;
                write_comma_separated(f, args)?;
                write!(f, ")")
            }
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                write!(f, "{receiver}.{method}(")?;
                write_comma_separated(f, args)?;
                write!(f, ")")
            }
            ExprKind::Lambda {
                params,
                env,
                ret_ty,
                body,
            } => {
                write!(f, "\\(")?;
                write_params(f, params)?;
                write!(f, ")+[")?;
                write_params(f, env)?;
                write!(f, "] -> {}. {}", ret_ty, body)
            }
            ExprKind::Closure { f: func, env } => {
                write!(f, "closure({}, [", func)?;
                write_comma_separated(f, env)?;
                write!(f, "])")
            }
            ExprKind::Seq(e1, e2) => write!(f, "{};\n{}", e1, e2),
            ExprKind::Let { name, ty, e1, e2 } => {
                write!(f, "let {}: {} = {} in\n{}", name, ty, e1, e2)
            }
            ExprKind::Return(e) => write!(f, "return {}", e),
            ExprKind::If { cond, then_, else_ } => match else_ {
                Some(else_expr) => write!(
                    f,
                    "if {} {{\n{}\n}} else {{\n{}\n}}",
                    cond,
                    indent(format!("{then_}")),
                    indent(format!("{else_expr}"))
                ),
                None => write!(f, "if {} {{\n{}\n}}", cond, indent(format!("{then_}"))),
            },
            ExprKind::Loop(body) => write!(f, "loop {{\n{}\n}}", indent(format!("{body}"))),
            ExprKind::While { cond, body } => {
                write!(f, "while {} {{\n{}\n}}", cond, indent(format!("{body}")))
            }
            ExprKind::Assign { dst, src } => write!(f, "{} = {}", dst, src),
            ExprKind::Break => write!(f, "break"),
            ExprKind::ArrayLiteral(exprs) => {
                write!(f, "[")?;
                write_comma_separated(f, exprs)?;
                write!(f, "]")
            }
            ExprKind::ArrayIndex { array, index } => {
                write!(f, "{}[{}]", array, index)
            }
            ExprKind::ArrayCopy { value, count } => {
                write!(f, "[|{}; {}|]", value, count)
            }
        }
    }
}
