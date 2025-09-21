//! Visitor trait for the TIR.

use crate::tir::types::*;

pub trait VisitMut {
    fn visit_program(&mut self, prog: &mut Program) {
        self.super_visit_program(prog);
    }

    fn super_visit_program(&mut self, prog: &mut Program) {
        for f in prog.functions_mut() {
            self.visit_function(f);
        }
    }

    fn visit_function(&mut self, func: &mut Function) {
        self.super_visit_function(func);
    }

    fn super_visit_function(&mut self, func: &mut Function) {
        for (_, ty) in &mut func.params {
            self.visit_type(ty);
        }
        self.visit_texpr(&mut func.body);
    }

    fn visit_texpr(&mut self, expr: &mut Expr) {
        self.super_visit_texpr(expr);
    }

    fn super_visit_texpr(&mut self, expr: &mut Expr) {
        self.visit_type(&mut expr.ty);
        match &mut expr.kind {
            ExprKind::Var(_) | ExprKind::Const(_) => {}
            ExprKind::BinOp { left, right, .. } => {
                self.visit_texpr(&mut *left);
                self.visit_texpr(&mut *right);
            }
            ExprKind::Cast { e, ty } => {
                self.visit_texpr(&mut *e);
                self.visit_type(ty);
            }
            ExprKind::Tuple(es) => {
                for e in es {
                    self.visit_texpr(e);
                }
            }
            ExprKind::Struct(es) => {
                for e in es {
                    self.visit_texpr(e);
                }
            }
            ExprKind::Project { e, .. } => {
                self.visit_texpr(&mut *e);
            }
            ExprKind::Call { f, args } => {
                self.visit_texpr(&mut *f);
                for arg in args {
                    self.visit_texpr(arg);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                self.visit_texpr(&mut *receiver);
                for arg in args {
                    self.visit_texpr(arg);
                }
            }
            ExprKind::Lambda {
                params,
                env,
                ret_ty,
                body,
            } => {
                for (_, ty) in params {
                    self.visit_type(ty);
                }
                for (_, ty) in env {
                    self.visit_type(ty);
                }
                self.visit_type(ret_ty);
                self.visit_texpr(&mut *body);
            }
            ExprKind::Closure { env, .. } => {
                for e in env {
                    self.visit_texpr(e);
                }
            }
            ExprKind::Seq(e1, e2) => {
                self.visit_texpr(&mut *e1);
                self.visit_texpr(&mut *e2);
            }
            ExprKind::Let { e1, e2, ty, .. } => {
                self.visit_type(ty);
                self.visit_texpr(&mut *e1);
                self.visit_texpr(&mut *e2);
            }
            ExprKind::Return(e) => {
                self.visit_texpr(&mut *e);
            }
            ExprKind::If { cond, then_, else_ } => {
                self.visit_texpr(&mut *cond);
                self.visit_texpr(&mut *then_);
                if let Some(else_) = else_ {
                    self.visit_texpr(&mut *else_);
                }
            }
            ExprKind::Loop(body) => {
                self.visit_texpr(&mut *body);
            }
            ExprKind::While { cond, body } => {
                self.visit_texpr(&mut *cond);
                self.visit_texpr(&mut *body);
            }
            ExprKind::Assign { dst, src } => {
                self.visit_texpr(&mut *dst);
                self.visit_texpr(&mut *src);
            }
            ExprKind::Break => {}
            ExprKind::ArrayLiteral(exprs) => {
                for e in exprs {
                    self.visit_texpr(e);
                }
            }
            ExprKind::ArrayIndex { array, index } => {
                self.visit_texpr(&mut *array);
                self.visit_texpr(&mut *index);
            }
            ExprKind::ArrayCopy { value, count } => {
                self.visit_texpr(&mut *value);
                self.visit_texpr(&mut *count);
            }
        }
    }

    fn visit_type(&mut self, _ty: &mut Type) {}
}
