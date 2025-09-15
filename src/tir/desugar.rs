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
            #[allow(unused)]
            ExprKind::While { cond, body } => {
                // Desugar while loop to: loop { if cond { body } else { break } }
                todo!("finish once break statements are implemented");
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
