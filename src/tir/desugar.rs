//! Desugars while-loops and if-statements.

use super::{
    types::{Expr, ExprKind, Program, Type},
    visit::VisitMut,
};

pub struct Desugarer;

impl VisitMut for Desugarer {
    fn visit_texpr(&mut self, expr: &mut Expr) {
        // First recursively desugar children
        self.super_visit_texpr(expr);

        // Then desugar this expression
        match &mut expr.kind {
            // #[allow(unused)]
            ExprKind::While { cond, body } => {
                // Desugar while loop to: loop { if cond { body } else { break } }
                let cond_expr = Expr {
                    kind: cond.kind.clone(),
                    ty: cond.ty,
                    span: cond.span,
                };
                let then_expr = Expr {
                    kind: body.kind.clone(),
                    ty: body.ty,
                    span: body.span,
                };
                let else_expr = Expr {
                    kind: ExprKind::Break,
                    ty: Type::unit(),
                    span: expr.span,
                };
                let loop_expr = Expr {
                    ty: expr.ty,
                    span: expr.span,
                    kind: ExprKind::If {
                        cond: Box::new(cond_expr),
                        then_: Box::new(then_expr),
                        else_: Some(Box::new(else_expr)),
                    },
                };

                expr.kind = ExprKind::Loop(Box::new(loop_expr))
            }
            ExprKind::If { else_, .. } => {
                // Desugar if-without-else to if-else with unit else branch
                if else_.is_none() {
                    let unit_expr = Expr {
                        kind: ExprKind::Tuple(Vec::new()),
                        ty: Type::unit(),
                        span: expr.span,
                    };

                    *else_ = Some(Box::new(unit_expr));
                }
            }
            _ => {}
        }
    }
}

pub fn desugar(prog: &mut Program) {
    Desugarer.visit_program(prog);
}
