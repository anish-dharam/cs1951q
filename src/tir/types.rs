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
