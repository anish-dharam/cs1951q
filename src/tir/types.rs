//! TIR type definitions.

use crate::{
    ast::types::{Annotation, Span},
    utils::Symbol,
};

pub use crate::ast::types::{Binop, Const, ParamList, Type, TypeKind};

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Symbol,
    pub params: ParamList,
    pub ret_ty: Type,
    pub body: Expr,
    pub annots: Vec<Annotation>,
}

impl Function {
    pub fn ty(&self) -> Type {
        Type::func(self.params.iter().map(|(_, ty)| *ty).collect(), self.ret_ty)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MethodRef {
    pub interface: Symbol,
    pub method: Symbol,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImplRef {
    pub interface: Symbol,
    pub struct_: Symbol,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Var(Symbol),
    Const(Const),
    Tuple(Vec<Expr>),
    Struct(Vec<Expr>),
    Project {
        e: Box<Expr>,
        i: usize,
    },
    BinOp {
        left: Box<Expr>,
        right: Box<Expr>,
        op: Binop,
    },
    Cast {
        e: Box<Expr>,
        ty: Type,
    },
    Call {
        f: Box<Expr>,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: MethodRef,
        args: Vec<Expr>,
    },
    Lambda {
        params: ParamList,
        env: ParamList,
        ret_ty: Type,
        body: Box<Expr>,
    },
    Closure {
        f: Symbol,
        env: Vec<Expr>,
    },
    Seq(Box<Expr>, Box<Expr>),
    Let {
        name: Symbol,
        ty: Type,
        e1: Box<Expr>,
        e2: Box<Expr>,
    },
    Return(Box<Expr>),
    Loop(Box<Expr>),
    While {
        cond: Box<Expr>,
        body: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Option<Box<Expr>>,
    },
    Assign {
        dst: Box<Expr>,
        src: Box<Expr>,
    },
    Break,
    ArrayLiteral(Vec<Expr>),
    ArrayIndex {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    ArrayCopy {
        value: Box<Expr>,
        count: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Type,
    pub span: Span,
}

impl Expr {
    pub fn subst(self, f: &mut impl FnMut(usize) -> Type) -> Self {
        let ty = self.ty.subst(f);
        let kind = match self.kind {
            ExprKind::Var(sym) => ExprKind::Var(sym),
            ExprKind::Const(c) => ExprKind::Const(c),
            ExprKind::Tuple(exprs) => {
                ExprKind::Tuple(exprs.into_iter().map(|e| e.subst(f)).collect())
            }
            ExprKind::Struct(exprs) => {
                ExprKind::Struct(exprs.into_iter().map(|e| e.subst(f)).collect())
            }
            ExprKind::Project { e, i } => ExprKind::Project {
                e: Box::new(e.subst(f)),
                i,
            },
            ExprKind::BinOp { left, right, op } => ExprKind::BinOp {
                left: Box::new(left.subst(f)),
                right: Box::new(right.subst(f)),
                op,
            },
            ExprKind::Call { f: func, args } => ExprKind::Call {
                f: Box::new(func.subst(f)),
                args: args.into_iter().map(|e| e.subst(f)).collect(),
            },
            ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => ExprKind::MethodCall {
                receiver: Box::new(receiver.subst(f)),
                method,
                args: args.into_iter().map(|e| e.subst(f)).collect(),
            },
            ExprKind::Seq(e1, e2) => ExprKind::Seq(Box::new(e1.subst(f)), Box::new(e2.subst(f))),
            ExprKind::Let {
                name,
                ty: let_ty,
                e1,
                e2,
            } => ExprKind::Let {
                name,
                ty: let_ty.subst(f),
                e1: Box::new(e1.subst(f)),
                e2: Box::new(e2.subst(f)),
            },
            ExprKind::Return(e) => ExprKind::Return(Box::new(e.subst(f))),
            ExprKind::Loop(body) => ExprKind::Loop(Box::new(body.subst(f))),
            ExprKind::While { cond, body } => ExprKind::While {
                cond: Box::new(cond.subst(f)),
                body: Box::new(body.subst(f)),
            },
            ExprKind::If { cond, then_, else_ } => ExprKind::If {
                cond: Box::new(cond.subst(f)),
                then_: Box::new(then_.subst(f)),
                else_: else_.map(|e| Box::new(e.subst(f))),
            },
            ExprKind::Assign { dst, src } => ExprKind::Assign {
                dst: Box::new(dst.subst(f)),
                src: Box::new(src.subst(f)),
            },
            ExprKind::Break => ExprKind::Break,
            ExprKind::ArrayLiteral(exprs) => {
                ExprKind::ArrayLiteral(exprs.into_iter().map(|e| e.subst(f)).collect())
            }
            ExprKind::ArrayIndex { array, index } => ExprKind::ArrayIndex {
                array: Box::new(array.subst(f)),
                index: Box::new(index.subst(f)),
            },
            ExprKind::ArrayCopy { value, count } => ExprKind::ArrayCopy {
                value: Box::new(value.subst(f)),
                count: Box::new(count.subst(f)),
            },
            ExprKind::Lambda {
                params,
                env,
                ret_ty,
                body,
            } => ExprKind::Lambda {
                params: params
                    .into_iter()
                    .map(|(name, ty)| (name, ty.subst(f)))
                    .collect(),
                env: env
                    .into_iter()
                    .map(|(name, ty)| (name, ty.subst(f)))
                    .collect(),
                ret_ty: ret_ty.subst(f),
                body: Box::new(body.subst(f)),
            },
            ExprKind::Cast { e, ty } => ExprKind::Cast {
                e: Box::new(e.subst(f)),
                ty: ty.subst(f),
            },
            ExprKind::Closure { f: fname, env } => ExprKind::Closure {
                f: fname,
                env: env.into_iter().map(|e| e.subst(f)).collect(),
            },
        };
        Expr {
            kind,
            ty,
            span: self.span,
        }
    }
}

pub struct Program(Vec<Function>);

impl Program {
    pub fn new(funcs: Vec<Function>) -> Self {
        Program(funcs)
    }

    pub fn functions(&self) -> &[Function] {
        &self.0
    }

    pub fn functions_mut(&mut self) -> &mut Vec<Function> {
        &mut self.0
    }

    pub fn into_functions(self) -> Vec<Function> {
        self.0
    }
}
