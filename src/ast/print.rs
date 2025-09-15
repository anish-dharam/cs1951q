//! Pretty-printing of AST types.

use std::fmt;

use crate::ast::types::{
    Annotation, Binop, Const, ExprKind, Function, Impl, Interface, Item, MethodSig, Program,
    Spanned, StructDef, TypeKind,
};
use crate::utils::{indent, write_comma_separated, write_newline_separated};

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for item in &self.0 {
            write!(f, "{}\n", item)?;
        }
        Ok(())
    }
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Item::Function(func) => write!(f, "{func}"),
            Item::StructDef(def) => write!(f, "{def}"),
            Item::Interface(intf) => write!(f, "{intf}"),
            Item::Impl(impl_) => write!(f, "{impl_}"),
        }
    }
}

impl fmt::Display for StructDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct {}(", self.name)?;
        write_comma_separated(f, &self.params)?;
        write!(f, ");")?;
        Ok(())
    }
}

impl fmt::Display for Interface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "interface {} {{\n", self.name)?;
        write_newline_separated(f, &self.methods)?;
        write!(f, "}}")
    }
}

impl fmt::Display for MethodSig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn {}{}", self.name, self.sig)
    }
}

impl fmt::Display for Impl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "impl {} for {} {{\n", self.intf, self.ty)?;
        write_newline_separated(f, &self.funcs)?;
        write!(f, "}}")
    }
}

impl fmt::Display for Annotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#[{}]", self.name)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for annot in &self.annots {
            write!(f, "{annot}\n")?;
        }
        write!(f, "fn {}(", self.name)?;
        let params: Vec<_> = self
            .params
            .iter()
            .map(|(name, ty)| format!("{}: {}", name, ty))
            .collect();
        write_comma_separated(f, &params)?;
        write!(f, ")")?;
        if let Some(ty) = &self.ret_ty {
            write!(f, " -> {}", ty)?;
        }
        write!(f, " {{\n{}\n}}", indent(format!("{}", self.body)))
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprKind::Var(name) => write!(f, "{}", name),
            ExprKind::Const(c) => write!(f, "{}", c),
            ExprKind::New { name, args } => {
                write!(f, "new {name}(")?;
                write_comma_separated(f, args)?;
                write!(f, ")")
            }
            ExprKind::Binop { left, op, right } => write!(f, "({} {} {})", left, op, right),
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
            ExprKind::If { cond, then_, else_ } => match else_ {
                Some(else_expr) => write!(
                    f,
                    "if {} {{\n{}\n}} else {{\n{}\n}}",
                    cond,
                    indent(format!("{}", then_)),
                    indent(format!("{}", else_expr))
                ),
                None => write!(f, "if {} {{\n{}\n}}", cond, indent(format!("{}", then_))),
            },
            ExprKind::Let { name, ty, e1, e2 } => match ty {
                Some(ty) => write!(f, "let {}: {} = {} in\n{}", name, ty, e1, e2),
                None => write!(f, "let {} = {} in\n{}", name, e1, e2),
            },
            ExprKind::Assign { dst, src } => write!(f, "{} := {}", dst, src),
            ExprKind::Lambda {
                params,
                ret_ty,
                body,
            } => {
                write!(f, "\\(")?;
                let params: Vec<_> = params
                    .iter()
                    .map(|(name, ty)| format!("{}: {}", name, ty))
                    .collect();
                write_comma_separated(f, &params)?;
                write!(f, ") -> {}. {}", ret_ty, body)
            }
            ExprKind::Seq(e1, e2) => write!(f, "{};\n{}", e1, e2),
            ExprKind::Tuple(exprs) => {
                write!(f, "(")?;
                write_comma_separated(f, exprs)?;
                write!(f, ")")
            }
            ExprKind::Project { e, i } => write!(f, "{}.{}", e, i),
            ExprKind::Cast { e, ty } => write!(f, "{} as {}", e, ty),
            ExprKind::Loop(body) => write!(f, "loop {{\n{}\n}}", indent(format!("{}", body))),
            ExprKind::While { cond, body } => {
                write!(f, "while {} {{\n{}\n}}", cond, indent(format!("{}", body)))
            }
            ExprKind::Return(e) => write!(f, "return {}", e),
        }
    }
}

impl fmt::Display for Binop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Binop::*;
        match self {
            Add => write!(f, "+"),
            Sub => write!(f, "-"),
            Mul => write!(f, "*"),
            Div => write!(f, "/"),
            Rem => write!(f, "%"),
            Exp => write!(f, "**"),
            Eq => write!(f, "=="),
            Neq => write!(f, "!="),
            Lt => write!(f, "<"),
            Gt => write!(f, ">"),
            Le => write!(f, "<="),
            Ge => write!(f, ">="),
            And => write!(f, "&&"),
            Or => write!(f, "||"),
            Shl => write!(f, "<<"),
            Shr => write!(f, ">>"),
            BitOr => write!(f, "|"),
            BitAnd => write!(f, "&"),
            Concat => write!(f, "^"),
        }
    }
}

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Const::Int(i) => write!(f, "{}", i),
            Const::Float(fl) => write!(f, "{}", fl.0),
            Const::Bool(b) => write!(f, "{}", b),
            Const::String(s) => write!(f, "\"{}\"", s),
        }
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeKind::Int => write!(f, "int"),
            TypeKind::Float => write!(f, "float"),
            TypeKind::Bool => write!(f, "bool"),
            TypeKind::String => write!(f, "string"),
            TypeKind::Self_ => write!(f, "Self"),
            TypeKind::Struct(name) => write!(f, "{name}"),
            TypeKind::Interface(name) => write!(f, "@{name}"),
            TypeKind::Tuple(types) => {
                write!(f, "(")?;
                write_comma_separated(f, types)?;
                write!(f, ")")
            }
            TypeKind::Func { inputs, output } => {
                write!(f, "fn(")?;
                write_comma_separated(f, inputs)?;
                write!(f, ") -> {}", output)
            }
            TypeKind::Hole(n) => write!(f, "?{n}"),
        }
    }
}
