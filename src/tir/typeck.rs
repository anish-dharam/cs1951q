//! Type checking algorithm which transforms the AST into initial TIR.

use itertools::Itertools;
use miette::{Diagnostic, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use crate::{
    ast::types::{self as ast, Item, MethodSig, Span},
    tir::types::{self as tir, Binop, Expr, ImplRef, MethodRef, Type, TypeKind},
    utils::{Symbol, sym},
};

#[derive(Diagnostic, Error, Debug)]
pub enum TypeError {
    #[error("undefined variable `{name}`")]
    UndefinedVariable {
        name: Symbol,
        #[label]
        span: Span,
    },

    #[error("type mismatch")]
    TypeMismatch {
        expected: Type,
        actual: Type,
        #[label("expected `{expected}`, found `{actual}`")]
        span: Span,
    },

    #[error("type mismatch")]
    TypeMismatchCustom {
        expected: String,
        actual: Type,
        #[label("expected {expected}, found `{actual}`")]
        span: Span,
    },

    #[error("invalid cast")]
    InvalidCast {
        from: Type,
        to: Type,
        #[label("cannot cast from `{from}` to `{to}`")]
        span: Span,
    },

    #[error("cannot project from type {ty}")]
    InvalidProjectionType {
        ty: Type,
        #[label]
        span: Span,
    },

    #[error("invalid tuple projection index")]
    InvalidProjectionIndex {
        index: usize,
        #[label]
        span: Span,
    },

    #[error("type {ty} is not numeric")]
    NonNumericType {
        ty: Type,
        #[label("non-numeric type")]
        span: Span,
    },

    #[error("Struct {name} has not been defined")]
    UnknownStruct {
        name: Symbol,
        #[label]
        span: Span,
    },

    #[error("impl block refers to unknown interface `{intf}`")]
    UnknownInterface {
        intf: Symbol,
        #[label]
        span: Span,
    },

    #[error("could not find method {method} for type {ty}")]
    MethodNotFound {
        method: Symbol,
        ty: Type,
        #[label]
        span: Span,
    },

    #[error("method `{method}` is not part of the interface `{intf}`")]
    UnknownMethod {
        method: Symbol,
        intf: Symbol,
        #[label]
        span: Span,
    },

    #[error("impl is missing method `{method}`")]
    UnimplementedMethod {
        method: Symbol,
        #[label]
        span: Span,
    },

    #[error("interface {intf} is not implemented for type {ty}")]
    MissingImpl {
        intf: Symbol,
        ty: Type,
        #[label]
        span: Span,
    },

    #[error("methods can only be called on objects")]
    InvalidMethodCall {
        #[label]
        span: Span,
    },

    #[error("invalid method signature")]
    InvalidMethodSig {
        #[label]
        span: Span,
    },

    #[error("expected {expected} args, found {actual}")]
    WrongNumArgs {
        expected: usize,
        actual: usize,
        #[label]
        span: Span,
    },
}

pub struct TypeData {
    pub ty: Type,
    pub used: bool,
    pub global: bool,
    pub name: Symbol,
}

/// The global environment of the program.
#[derive(Default)]
pub struct Globals {
    /// Map of functions. TypeData is used to track free variables in closures.
    pub funcs: HashMap<Symbol, Vec<TypeData>>,

    /// Map of struct definitions to the list of field types.
    pub structs: HashMap<Symbol, Vec<Type>>,

    /// Map of interface definitions to the list of method signatures.
    pub intfs: HashMap<Symbol, Vec<MethodSig>>,

    /// Map of impl blocks to the list of function names, parallel to the method signature list.
    pub impls: HashMap<ImplRef, Vec<Symbol>>,
}

/// Type context contains info accumulated while type-checking, such as [`Globals`].
pub struct Tcx {
    globals: Globals,
}

/// A predicate which must hold on a concrete type, excluding equality.
#[derive(Clone, Copy, Debug)]
pub enum TypeConstraint {
    /// Type must be either int or float.
    Numeric,

    /// Type must be castable to the given type.
    CastableTo(Type),
}

impl TypeConstraint {
    /// Returns an error if `ty` does not satisfy the constraint `self`.
    pub fn satisfied_by(self, ty: Type, span: Span, globals: &Globals) -> Result<()> {
        match self {
            TypeConstraint::Numeric => {
                ensure!(ty.is_numeric(), TypeError::NonNumericType { ty, span })
            }
            TypeConstraint::CastableTo(ty2) => match (ty.kind(), ty2.kind()) {
                (TypeKind::Int, TypeKind::Float) => {}
                (TypeKind::Struct(struct_), TypeKind::Interface(intf)) => {
                    let impl_ref = ImplRef {
                        interface: *intf,
                        struct_: *struct_,
                    };
                    ensure!(
                        globals.impls.contains_key(&impl_ref),
                        TypeError::MissingImpl {
                            intf: *intf,
                            ty,
                            span
                        }
                    );
                }
                _ => bail!(TypeError::InvalidCast {
                    from: ty,
                    to: ty2,
                    span,
                }),
            },
        };
        Ok(())
    }
}

macro_rules! ensure_let {
    ($p:pat = $e:expr, $err:expr) => {
        let $p = $e else { bail!($err) };
    };
}

impl Tcx {
    pub fn new() -> Self {
        let mut tcx = Tcx {
            globals: Globals::default(),
        };

        // Load stdlib into the type context
        for (name, func) in crate::stdlib::stdlib() {
            tcx.push_var(*name, func.src_type(), true);
        }

        tcx
    }

    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    fn push_var(&mut self, name: Symbol, ty: Type, global: bool) {
        let tds = self.globals.funcs.entry(name).or_default();
        let name = if tds.is_empty() {
            name
        } else {
            sym(format!("{name}{}", tds.len()))
        };
        tds.push(TypeData {
            ty,
            name,
            global,
            used: false,
        });
    }

    fn pop_var(&mut self, name: Symbol) -> Symbol {
        self.globals
            .funcs
            .get_mut(&name)
            .unwrap()
            .pop()
            .unwrap()
            .name
    }

    fn get_var(&mut self, name: Symbol) -> Option<(Symbol, Type)> {
        let tds = self.globals.funcs.get_mut(&name)?;
        let td = tds.last_mut()?;
        td.used = true;
        Some((td.name, td.ty))
    }

    pub fn check(&mut self, prog: &ast::Program) -> Result<tir::Program> {
        let mut tir_prog = Vec::new();
        for item in &prog.0 {
            self.check_item(&mut tir_prog, item)?;
        }

        self.globals.funcs.retain(|_, tds| !tds.is_empty());

        let prog = tir::Program::new(tir_prog);

        Ok(prog)
    }

    fn check_item(&mut self, output: &mut Vec<tir::Function>, item: &ast::Item) -> Result<()> {
        match item {
            Item::Function(func) => {
                let tir_f = self.check_func(func)?;
                self.push_var(tir_f.name, tir_f.ty(), true);
                output.push(tir_f);
            }

            Item::StructDef(def) => {
                self.globals.structs.insert(def.name, def.params.clone());
            }

            Item::Interface(intf) => {
                self.check_intf(intf)?;
            }

            Item::Impl(impl_) => {
                output.extend(self.check_impl(impl_)?);
            }
        };

        Ok(())
    }

    fn check_intf(&mut self, intf: &ast::Interface) -> Result<()> {
        for method in &intf.methods {
            let inputs = method.inputs();
            ensure!(
                !inputs.is_empty() && matches!(inputs[0].kind(), TypeKind::Self_),
                TypeError::InvalidMethodSig {
                    span: method.name.span
                }
            );
        }

        self.globals
            .intfs
            .insert(intf.name.value, intf.methods.clone());
        Ok(())
    }

    fn check_impl(&mut self, impl_: &ast::Impl) -> Result<Vec<tir::Function>> {
        ensure_let!(
            Some(intf_methods) = self.globals.intfs.get(&impl_.intf.value),
            TypeError::UnknownInterface {
                intf: impl_.intf.value,
                span: impl_.intf.span
            }
        );

        ensure!(
            self.globals.structs.contains_key(&impl_.ty.value),
            TypeError::UnknownStruct {
                name: impl_.ty.value,
                span: impl_.ty.span
            }
        );
        let self_ty = Type::struct_(impl_.ty.value);

        let mut intf_methods = intf_methods.clone();
        let mut funcs = Vec::new();
        for func in &impl_.funcs {
            ensure_let!(
                Some(method_idx) = intf_methods
                    .iter()
                    .position(|sig| sig.name.value == func.name.value),
                TypeError::UnknownMethod {
                    method: func.name.value,
                    intf: impl_.intf.value,
                    span: func.name.span
                }
            );

            let method_sig = intf_methods.remove(method_idx);

            // Impl method signature should match interface method signature.
            let TypeKind::Func { inputs, output } = method_sig.sig.kind() else {
                unreachable!()
            };
            let mut func = func.clone();

            // Generate a unique name for the implemented function.
            // Note: there should ideally be some kind of uniqueness check or gensym.
            func.name.value = sym(format!("{}__{}__{}", impl_.ty, impl_.intf, func.name));

            // The first type must be `Self`, so we replace it with the actual self type.
            // Note: we would ideally substitute every instance of `Self` in all the parameters.
            func.params[0].1 = self_ty;

            for ((_, actual_param), expected_param) in func.params.iter().zip(inputs).skip(1) {
                self.ty_equiv(*expected_param, *actual_param, func.name.span)?;
            }
            self.ty_equiv(*output, func.ret_ty.unwrap_or(Type::unit()), func.name.span)?;

            funcs.push(self.check_func(&func)?);
        }

        ensure!(
            intf_methods.is_empty(),
            TypeError::UnimplementedMethod {
                method: intf_methods[0].name.value,
                span: impl_.ty.span
            }
        );

        let impl_ref = ImplRef {
            interface: impl_.intf.value,
            struct_: impl_.ty.value,
        };
        let method_names = funcs.iter().map(|func| func.name).collect();
        self.globals.impls.insert(impl_ref, method_names);

        Ok(funcs)
    }

    fn check_func(&mut self, func: &ast::Function) -> Result<tir::Function> {
        for (name, ty) in &func.params {
            self.push_var(*name, *ty, false);
        }

        let body = self.check_expr(&func.body)?;

        for (name, _) in &func.params {
            self.pop_var(*name);
        }

        let ret_ty = func.ret_ty.unwrap_or_else(Type::unit);
        self.ty_equiv(body.ty, ret_ty, body.span)?;

        Ok(tir::Function {
            name: func.name.value,
            params: func.params.clone(),
            ret_ty,
            body,
            annots: func.annots.clone(),
        })
    }

    fn ty_equiv(&mut self, expected: Type, actual: Type, span: Span) -> Result<()> {
        ensure!(
            expected.equiv(actual.kind()),
            TypeError::TypeMismatch {
                expected,
                actual,
                span,
            }
        );
        Ok(())
    }

    fn ty_constraint(&mut self, constraint: TypeConstraint, ty: Type, span: Span) -> Result<()> {
        constraint.satisfied_by(ty, span, &self.globals)
    }

    fn check_expr(&mut self, expr: &ast::Expr) -> Result<Expr> {
        let (expr_t, ty) = match &expr.value {
            ast::ExprKind::Var(name) => {
                ensure_let!(
                    Some((new_name, ty)) = self.get_var(*name),
                    TypeError::UndefinedVariable {
                        name: *name,
                        span: expr.span,
                    }
                );
                (tir::ExprKind::Var(new_name), ty)
            }

            ast::ExprKind::Const(c) => (tir::ExprKind::Const(c.clone()), c.ty()),

            ast::ExprKind::New { name, args } => {
                ensure_let!(
                    Some(params) = self.globals.structs.get(name),
                    TypeError::UnknownStruct {
                        name: *name,
                        span: expr.span
                    }
                );

                let params = params.clone(); // hack to avoid lifetime conflict
                let args = args
                    .iter()
                    .zip(params)
                    .map(|(arg, ty)| {
                        let arg = self.check_expr(arg)?;
                        self.ty_equiv(ty, arg.ty, arg.span)?;
                        Ok(arg)
                    })
                    .collect::<Result<Vec<_>>>()?;
                (tir::ExprKind::Struct(args), Type::struct_(*name))
            }

            ast::ExprKind::Binop { left, op, right } => {
                let left = self.check_expr(left)?;
                let right = self.check_expr(right)?;
                let out_ty = match op {
                    Binop::Shl | Binop::Shr | Binop::BitAnd | Binop::BitOr => {
                        self.ty_equiv(Type::int(), left.ty, left.span)?;
                        self.ty_equiv(Type::int(), right.ty, right.span)?;
                        Type::int()
                    }
                    Binop::Add | Binop::Sub | Binop::Mul | Binop::Div | Binop::Rem | Binop::Exp => {
                        self.ty_constraint(TypeConstraint::Numeric, left.ty, left.span)?;
                        self.ty_equiv(left.ty, right.ty, right.span)?;
                        left.ty
                    }
                    Binop::Ge | Binop::Gt | Binop::Le | Binop::Lt => {
                        self.ty_constraint(TypeConstraint::Numeric, left.ty, left.span)?;
                        self.ty_equiv(left.ty, right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Or | Binop::And => {
                        self.ty_equiv(Type::bool(), left.ty, left.span)?;
                        self.ty_equiv(Type::bool(), right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Eq | Binop::Neq => {
                        self.ty_equiv(left.ty, right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Concat => {
                        self.ty_equiv(Type::string(), left.ty, left.span)?;
                        self.ty_equiv(Type::string(), right.ty, left.span)?;
                        Type::string()
                    }
                };
                (
                    tir::ExprKind::BinOp {
                        left: Box::new(left),
                        right: Box::new(right),
                        op: *op,
                    },
                    out_ty,
                )
            }

            ast::ExprKind::Cast { e, ty } => {
                let e = self.check_expr(e)?;
                self.ty_constraint(TypeConstraint::CastableTo(*ty), e.ty, expr.span)?;
                (
                    tir::ExprKind::Cast {
                        e: Box::new(e),
                        ty: *ty,
                    },
                    *ty,
                )
            }

            ast::ExprKind::Tuple(es) => {
                let es = es
                    .iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<Vec<_>>>()?;
                let tys = es.iter().map(|e| e.ty).collect::<Vec<_>>();
                (tir::ExprKind::Tuple(es), Type::tuple(tys))
            }

            ast::ExprKind::Project { e, i } => {
                let e = self.check_expr(e)?;
                let tys = match e.ty.kind() {
                    TypeKind::Tuple(tys) => tys,
                    TypeKind::Struct(name) => {
                        ensure_let!(
                            Some(tys) = self.globals.structs.get(name),
                            TypeError::UnknownStruct {
                                name: *name,
                                span: e.span
                            }
                        );
                        tys
                    }
                    _ => bail!(TypeError::InvalidProjectionType {
                        ty: e.ty,
                        span: e.span
                    }),
                };

                ensure_let!(
                    Some(ith_ty) = tys.get(*i),
                    TypeError::InvalidProjectionIndex {
                        index: *i,
                        span: expr.span,
                    }
                );

                (
                    tir::ExprKind::Project {
                        e: Box::new(e),
                        i: *i,
                    },
                    *ith_ty,
                )
            }

            ast::ExprKind::Call { f, args } => {
                let f = self.check_expr(f)?;
                ensure_let!(
                    TypeKind::Func { inputs, output } = f.ty.kind(),
                    TypeError::TypeMismatchCustom {
                        expected: "function".into(),
                        actual: f.ty,
                        span: f.span
                    }
                );

                ensure!(
                    args.len() == inputs.len(),
                    TypeError::WrongNumArgs {
                        expected: inputs.len(),
                        actual: args.len(),
                        span: expr.span
                    }
                );

                let args = args
                    .iter()
                    .zip_eq(inputs)
                    .map(|(arg, param_ty)| {
                        let e = self.check_expr(arg)?;
                        self.ty_equiv(e.ty, *param_ty, e.span)?;
                        Ok(e)
                    })
                    .collect::<Result<Vec<_>>>()?;

                (
                    tir::ExprKind::Call {
                        f: Box::new(f),
                        args,
                    },
                    *output,
                )
            }

            ast::ExprKind::MethodCall {
                receiver,
                method,
                args,
            } => {
                let receiver = self.check_expr(receiver)?;

                let (receiver, intf, sig) = match receiver.ty.kind() {
                    TypeKind::Struct(struct_) => {
                        let sig_search = self
                            .globals
                            .intfs
                            .iter()
                            .filter_map(|(intf, methods)| {
                                let sig = methods.iter().find(|sig| sig.name.value == *method)?;
                                Some((intf, sig))
                            })
                            .next();
                        ensure_let!(
                            Some((intf, sig)) = sig_search,
                            TypeError::MethodNotFound {
                                method: *method,
                                ty: receiver.ty,
                                span: expr.span
                            }
                        );

                        let impl_ref = ImplRef {
                            interface: *intf,
                            struct_: *struct_,
                        };
                        ensure!(
                            self.globals.impls.contains_key(&impl_ref),
                            TypeError::MissingImpl {
                                intf: *intf,
                                ty: receiver.ty,
                                span: expr.span
                            }
                        );

                        let receiver_casted = tir::Expr {
                            span: receiver.span,
                            kind: tir::ExprKind::Cast {
                                e: Box::new(receiver),
                                ty: Type::interface(*intf),
                            },
                            ty: Type::interface(*intf),
                        };
                        (receiver_casted, *intf, sig.clone())
                    }

                    TypeKind::Interface(intf) => {
                        let methods = &self.globals.intfs[intf];
                        ensure_let!(
                            Some(sig) = methods.iter().find(|sig| sig.name.value == *method),
                            TypeError::MethodNotFound {
                                method: *method,
                                ty: receiver.ty,
                                span: expr.span
                            }
                        );
                        (receiver, *intf, sig.clone())
                    }

                    _ => bail!(TypeError::InvalidMethodCall { span: expr.span }),
                };

                let method = MethodRef {
                    interface: intf,
                    method: sig.name.value,
                };

                let args = args
                    .iter()
                    .zip(sig.inputs())
                    .map(|(arg, expected_ty)| {
                        let arg = self.check_expr(arg)?;
                        self.ty_equiv(*expected_ty, arg.ty, arg.span)?;
                        Ok(arg)
                    })
                    .collect::<Result<Vec<_>>>()?;

                (
                    tir::ExprKind::MethodCall {
                        receiver: Box::new(receiver),
                        method,
                        args,
                    },
                    sig.output(),
                )
            }

            ast::ExprKind::Seq(e1, e2) => {
                let e1 = self.check_expr(e1)?;
                let e2 = self.check_expr(e2)?;
                let e2_ty = e2.ty;
                (tir::ExprKind::Seq(Box::new(e1), Box::new(e2)), e2_ty)
            }

            ast::ExprKind::Let { name, ty, e1, e2 } => {
                let e1 = self.check_expr(e1)?;
                let inferred_ty = match ty {
                    Some(ty) => {
                        self.ty_equiv(e1.ty, *ty, e1.span)?;
                        *ty
                    }
                    None => e1.ty, // Use inferred type from e1
                };
                self.push_var(*name, inferred_ty, false);
                let e2 = self.check_expr(e2)?;
                let new_name = self.pop_var(*name);
                let e2_ty = e2.ty;
                (
                    tir::ExprKind::Let {
                        name: new_name,
                        ty: inferred_ty,
                        e1: Box::new(e1),
                        e2: Box::new(e2),
                    },
                    e2_ty,
                )
            }

            ast::ExprKind::Return(e) => {
                let e = self.check_expr(e)?;
                (tir::ExprKind::Return(Box::new(e)), Type::unit())
            }

            ast::ExprKind::If { cond, then_, else_ } => {
                let cond_span = cond.span;
                let cond = self.check_expr(cond)?;
                self.ty_equiv(Type::bool(), cond.ty, cond_span)?;
                let then_ = self.check_expr(then_)?;

                let (else_, ty) = match else_ {
                    Some(else_expr) => {
                        let else_ = self.check_expr(else_expr)?;
                        self.ty_equiv(then_.ty, else_.ty, else_.span)?;
                        (Some(Box::new(else_)), then_.ty)
                    }
                    None => {
                        // If without else must have unit type in then branch
                        self.ty_equiv(Type::unit(), then_.ty, then_.span)?;
                        (None, Type::unit())
                    }
                };

                (
                    tir::ExprKind::If {
                        cond: Box::new(cond),
                        then_: Box::new(then_),
                        else_,
                    },
                    ty,
                )
            }

            ast::ExprKind::Loop(body) => {
                let body = self.check_expr(body)?;
                (tir::ExprKind::Loop(Box::new(body)), Type::unit())
            }

            ast::ExprKind::While { cond, body } => {
                let cond = self.check_expr(cond)?;
                self.ty_equiv(Type::bool(), cond.ty, cond.span)?;
                let body = self.check_expr(body)?;
                (
                    tir::ExprKind::While {
                        cond: Box::new(cond),
                        body: Box::new(body),
                    },
                    Type::unit(),
                )
            }

            ast::ExprKind::Lambda {
                params,
                ret_ty,
                body,
            } => {
                let param_names: HashSet<_> = params.iter().map(|(name, _)| *name).collect();

                for (name, ty) in params {
                    self.push_var(*name, *ty, false);
                }

                let body = self.check_expr(body)?;

                let env = self
                    .globals
                    .funcs
                    .iter()
                    .filter_map(|(name, tds)| {
                        let td = tds.last()?;
                        (td.used && !td.global && !param_names.contains(name))
                            .then_some((*name, td.ty))
                    })
                    .collect_vec();

                let new_params = params
                    .iter()
                    .map(|(name, ty)| (self.pop_var(*name), *ty))
                    .collect_vec();

                self.ty_equiv(*ret_ty, body.ty, body.span)?;
                let func_ty = Type::func(new_params.iter().map(|(_, ty)| *ty).collect(), *ret_ty);
                (
                    tir::ExprKind::Lambda {
                        params: new_params,
                        env,
                        ret_ty: *ret_ty,
                        body: Box::new(body),
                    },
                    func_ty,
                )
            }

            ast::ExprKind::Assign { dst, src } => {
                let src = self.check_expr(src)?;
                let dst = self.check_expr(dst)?;
                self.ty_equiv(src.ty, dst.ty, dst.span)?;

                (
                    tir::ExprKind::Assign {
                        dst: Box::new(dst),
                        src: Box::new(src),
                    },
                    Type::unit(),
                )
            }
        };
        Ok(Expr {
            kind: expr_t,
            ty,
            span: expr.span,
        })
    }
}
