//! Performs closure conversion, transforming closures into top-level functions.

use std::{collections::HashMap, sync::LazyLock};

use super::{Tcx, types::*, visit::VisitMut};
use crate::utils::{Symbol, sym};

struct SubstEnv {
    env: HashMap<Symbol, usize>,
    env_ty: Type,
}

static ENV: LazyLock<Symbol> = LazyLock::new(|| sym("env"));

impl VisitMut for SubstEnv {
    fn visit_texpr(&mut self, expr: &mut Expr) {
        match &mut expr.kind {
            ExprKind::Var(name) => {
                if let Some(i) = self.env.get(name) {
                    expr.kind = ExprKind::Project {
                        e: Box::new(Expr {
                            ty: self.env_ty,
                            kind: ExprKind::Var(*ENV),
                            span: expr.span,
                        }),
                        i: *i,
                    }
                }
            }
            _ => self.super_visit_texpr(expr),
        }
    }
}

struct ClosureConversion<'a> {
    tcx: &'a Tcx,
    counter: usize,
    lambdas: Vec<Function>,
}

impl VisitMut for ClosureConversion<'_> {
    fn visit_program(&mut self, prog: &mut Program) {
        self.super_visit_program(prog);
        prog.functions_mut().splice(0..0, self.lambdas.drain(..));
    }

    fn visit_function(&mut self, func: &mut Function) {
        self.super_visit_function(func);

        func.params.insert(0, (*ENV, Type::unit()));
    }

    fn visit_texpr(&mut self, expr: &mut Expr) {
        match &mut expr.kind {
            ExprKind::Lambda {
                params,
                env,
                ret_ty,
                body,
            } => {
                let i = self.counter;
                self.counter += 1;
                let name = Symbol::new(format!("__closure{i}"));
                let mut new_params = params.clone();
                new_params.insert(
                    0,
                    (*ENV, Type::tuple(env.iter().map(|(_, ty)| *ty).collect())),
                );

                self.super_visit_texpr(body);

                let mut new_body = *body.clone();
                SubstEnv {
                    env: env
                        .iter()
                        .enumerate()
                        .map(|(i, (sym, _))| (*sym, i))
                        .collect(),
                    env_ty: Type::tuple(env.iter().map(|(_, ty)| *ty).collect()),
                }
                .visit_texpr(&mut new_body);

                let func = Function {
                    name,
                    params: new_params,
                    ret_ty: *ret_ty,
                    body: new_body,
                    annots: Vec::new(),
                };
                self.lambdas.push(func);

                let env = env
                    .iter()
                    .map(|(name, ty)| Expr {
                        kind: ExprKind::Var(*name),
                        ty: *ty,
                        span: expr.span,
                    })
                    .collect::<Vec<_>>();

                expr.kind = ExprKind::Closure { f: name, env };
            }

            ExprKind::Var(name) => {
                if self.tcx.globals().funcs.contains_key(name) {
                    expr.kind = ExprKind::Closure {
                        f: *name,
                        env: Vec::new(),
                    };
                }
            }

            _ => self.super_visit_texpr(expr),
        }
    }
}

pub fn closure_conversion(tcx: &Tcx, p: &mut Program) {
    ClosureConversion {
        tcx,
        counter: 0,
        lambdas: Vec::new(),
    }
    .visit_program(p);
}
