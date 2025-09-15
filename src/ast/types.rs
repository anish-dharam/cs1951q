//! Definitions of AST types.

use std::{fmt, sync::LazyLock};

use internment::Intern;
use ordered_float::OrderedFloat;
use serde::Serialize;

use crate::{interned, utils::Symbol};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum TypeKind {
    Int,
    Float,
    Bool,
    String,
    Self_,
    Tuple(Vec<Type>),
    Func { inputs: Vec<Type>, output: Type },
    Struct(Symbol),
    Interface(Symbol),
    Hole(usize),
}

interned!(Type, TypeKind);

impl Type {
    pub fn new(kind: TypeKind) -> Self {
        Type(Intern::new(kind))
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! type_constructor {
    ($name:ident, $kind:expr) => {
        pub fn $name() -> Self {
            static CELL: LazyLock<Type> = LazyLock::new(|| Type::new($kind));
            *CELL
        }
    };
}

impl Type {
    type_constructor!(int, TypeKind::Int);
    type_constructor!(float, TypeKind::Float);
    type_constructor!(bool, TypeKind::Bool);
    type_constructor!(string, TypeKind::String);
    type_constructor!(unit, TypeKind::Tuple(Vec::new()));
    type_constructor!(self_, TypeKind::Self_);

    pub fn struct_(name: Symbol) -> Self {
        Type(Intern::new(TypeKind::Struct(name)))
    }

    pub fn interface(name: Symbol) -> Self {
        Type(Intern::new(TypeKind::Interface(name)))
    }

    pub fn tuple(tys: Vec<Type>) -> Self {
        Type(Intern::new(TypeKind::Tuple(tys)))
    }

    pub fn func(inputs: Vec<Type>, output: Type) -> Self {
        Type(Intern::new(TypeKind::Func { inputs, output }))
    }

    pub fn hole(n: usize) -> Self {
        Type(Intern::new(TypeKind::Hole(n)))
    }

    pub fn kind(self) -> &'static TypeKind {
        self.0.as_ref()
    }

    /// Substitute holes for types by running a function `f` on each hole.
    pub fn subst(self, f: &mut impl FnMut(usize) -> Type) -> Type {
        match self.kind() {
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::String
            | TypeKind::Self_
            | TypeKind::Struct(..)
            | TypeKind::Interface(..) => self,
            TypeKind::Tuple(tys) => Type::tuple(tys.iter().map(|ty| ty.subst(f)).collect()),
            TypeKind::Func { inputs, output } => Type::func(
                inputs.iter().map(|ty| ty.subst(f)).collect(),
                output.subst(f),
            ),
            TypeKind::Hole(hole) => f(*hole),
        }
    }
}

impl TypeKind {
    pub fn is_unit(&self) -> bool {
        match self {
            TypeKind::Tuple(v) => v.is_empty(),
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, TypeKind::Int | TypeKind::Float)
    }

    /// Tests types for alpha-equivalence.
    pub fn equiv(&self, other: &TypeKind) -> bool {
        match (self, other) {
            (TypeKind::Int, TypeKind::Int)
            | (TypeKind::Float, TypeKind::Float)
            | (TypeKind::Bool, TypeKind::Bool)
            | (TypeKind::String, TypeKind::String) => true,
            (TypeKind::Struct(n1), TypeKind::Struct(n2))
            | (TypeKind::Interface(n1), TypeKind::Interface(n2)) => n1 == n2,
            (TypeKind::Tuple(t1), TypeKind::Tuple(t2)) => {
                t1.len() == t2.len() && t1.iter().zip(t2).all(|(t1, t2)| t1.equiv(t2))
            }
            (
                TypeKind::Func {
                    inputs: inputs1,
                    output: output1,
                },
                TypeKind::Func {
                    inputs: inputs2,
                    output: output2,
                },
            ) => {
                inputs1.len() == inputs2.len()
                    && inputs1.iter().zip(inputs2).all(|(t1, t2)| t1.equiv(t2))
                    && output1.equiv(output2)
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const DUMMY: Span = Span { start: 0, end: 0 };
}

impl From<Span> for miette::SourceSpan {
    fn from(value: Span) -> Self {
        miette::SourceSpan::from((value.start as usize, (value.end - value.start) as usize))
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub span: Span,
    pub value: T,
}

#[derive(Debug, Clone)]
pub enum Item {
    Function(Function),
    StructDef(StructDef),
    Interface(Interface),
    Impl(Impl),
}

#[derive(Debug, Clone)]
pub struct Interface {
    pub name: Spanned<Symbol>,
    pub methods: Vec<MethodSig>,
}

#[derive(Debug, Clone)]
pub struct MethodSig {
    pub name: Spanned<Symbol>,
    pub sig: Type,
}

impl MethodSig {
    pub fn inputs(&self) -> &[Type] {
        let TypeKind::Func { inputs, .. } = self.sig.kind() else {
            unreachable!()
        };
        inputs
    }

    pub fn output(&self) -> Type {
        let TypeKind::Func { output, .. } = self.sig.kind() else {
            unreachable!()
        };
        *output
    }
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: Symbol,
    pub params: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct Impl {
    pub intf: Spanned<Symbol>,
    pub ty: Spanned<Symbol>,
    pub funcs: Vec<Function>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Annotation {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Spanned<Symbol>,
    pub params: ParamList,
    pub ret_ty: Option<Type>,
    pub body: Expr,
    pub annots: Vec<Annotation>,
}

pub type ParamList = Vec<(Symbol, Type)>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Const {
    Bool(bool),
    Int(i32),
    Float(OrderedFloat<f32>),
    String(String),
}

impl Const {
    pub fn ty(&self) -> Type {
        match self {
            Const::Bool(..) => Type::bool(),
            Const::Int(..) => Type::int(),
            Const::Float(..) => Type::float(),
            Const::String(..) => Type::string(),
        }
    }

    pub fn as_int(&self) -> Option<i32> {
        match self {
            Const::Int(n) => Some(*n),
            _ => None,
        }
    }
}

pub type Expr = Spanned<ExprKind>;

#[derive(Debug, Clone)]
pub enum ExprKind {
    Var(Symbol),
    Const(Const),
    New {
        name: Symbol,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: Symbol,
        args: Vec<Expr>,
    },
    Tuple(Vec<Expr>),
    Project {
        e: Box<Expr>,
        i: usize,
    },
    Binop {
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
    Lambda {
        params: ParamList,
        ret_ty: Type,
        body: Box<Expr>,
    },
    Seq(Box<Expr>, Box<Expr>),
    Let {
        name: Symbol,
        ty: Option<Type>,
        e1: Box<Expr>,
        e2: Box<Expr>,
    },
    Return(Box<Expr>),
    If {
        cond: Box<Expr>,
        then_: Box<Expr>,
        else_: Option<Box<Expr>>,
    },
    Loop(Box<Expr>),
    While {
        cond: Box<Expr>,
        body: Box<Expr>,
    },
    Assign {
        dst: Box<Expr>,
        src: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub enum Binop {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Exp,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    Concat,
}

#[derive(Debug)]
pub struct Program(pub Vec<Item>);
