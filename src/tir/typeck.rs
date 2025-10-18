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

    #[error("break used outside of a loop")]
    InvalidBreak {
        #[label]
        span: Span,
    },

    #[error("can't construct empty array")]
    EmptyArray {
        #[label]
        span: Span,
    },

    #[error("uninferred type {ty}")]
    UninferredType {
        ty: Type,
        #[label]
        span: Span,
    },
    // #[error("all elements in array must have type {first_type}")]
    // NonHomogenousArray {
    //     #[label]
    //     span: Span,
    //     first_type: Type,
    // },
    #[error("tried to index into a {ty} as if it were an array")]
    IndexIntoNonArray {
        #[label]
        span: Span,
        ty: Type,
    },
}

struct TypeUnifier {
    parents: HashMap<Type, Type>, // hole type to type
    equational_constraints: Vec<(Type, Type, Span)>,
    projection_constraints: Vec<(Type, Type, usize, Span)>, // t1 = t2.usize
    castable_constraints: Vec<(Type, Type, Span)>,          // t1 as t2
}

impl TypeUnifier {
    fn new() -> Self {
        Self {
            parents: HashMap::new(),
            equational_constraints: Vec::new(),
            projection_constraints: Vec::new(),
            castable_constraints: Vec::new(),
        }
    }

    fn initialize_type(&mut self, ty: Type) {
        if self.parents.contains_key(&ty) {
            return;
        }
        self.parents.insert(ty, ty);
    }

    fn union(&mut self, ty1: Type, ty2: Type) -> bool {
        let root1 = self.find(ty1);
        let root2 = self.find(ty2);
        if root1 == root2 {
            return true;
        }

        match (root1.kind(), root2.kind()) {
            (TypeKind::Tuple(items1), TypeKind::Tuple(items2)) => {
                if items1.len() != items2.len() {
                    return false;
                }

                // Check if each pairwise element can be unified
                for (item1, item2) in items1.iter().zip(items2.iter()) {
                    if !self.union(*item1, *item2) {
                        return false;
                    }
                }

                // Choose the more constrained tuple as root (more concrete types)
                let root1_concrete_count = items1
                    .iter()
                    .filter(|t| !matches!(t.kind(), TypeKind::Hole(_)))
                    .count();
                let root2_concrete_count = items2
                    .iter()
                    .filter(|t| !matches!(t.kind(), TypeKind::Hole(_)))
                    .count();

                if root1_concrete_count >= root2_concrete_count {
                    self.parents.insert(root2, root1);
                } else {
                    self.parents.insert(root1, root2);
                }
                true
            }
            (TypeKind::Array(elem1), TypeKind::Array(elem2)) => {
                // Unify the element types
                // println!("Unifying arrays: {:?} and {:?}", elem1, elem2);
                if !self.union(*elem1, *elem2) {
                    return false;
                }

                let internal_root = self.find(*elem1);
                assert_eq!(internal_root, self.find(*elem2));

                // Create the unified array type with the resolved element type
                let unified_array = Type::array(internal_root);
                self.initialize_type(unified_array);

                self.parents.insert(root1, unified_array);
                self.parents.insert(root2, unified_array);

                true
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
                // Check that both functions have the same number of parameters
                if inputs1.len() != inputs2.len() {
                    return false;
                }

                for (input1, input2) in inputs1.iter().zip(inputs2.iter()) {
                    if !self.union(*input1, *input2) {
                        return false;
                    }
                }

                if !self.union(*output1, *output2) {
                    return false;
                }

                let rooted_inputs = inputs1.iter().map(|t| self.find(*t)).collect::<Vec<_>>();
                let rooted_output = self.find(*output1);

                let unified_func = Type::func(rooted_inputs, rooted_output);
                // println!(
                //     "func1: {:?}, func2: {:?}, unified_func: {:?}",
                //     root1, root2, unified_func
                // );
                self.initialize_type(unified_func);

                self.parents.insert(root1, unified_func);
                self.parents.insert(root2, unified_func);

                true
            }
            _ => {
                // If both are concrete types (not holes), they must be equal to unify
                if !matches!(root1.kind(), TypeKind::Hole(_))
                    && !matches!(root2.kind(), TypeKind::Hole(_))
                {
                    return root1.equiv(root2.kind());
                } else if !matches!(root1.kind(), TypeKind::Hole(_)) {
                    // root1 is concrete, root2 is hole - unify hole to concrete
                    self.parents.insert(root2, root1);
                } else {
                    // root1 is hole, root2 is concrete - unify hole to concrete (or both holes)
                    self.parents.insert(root1, root2);
                }
                return true;
            }
        }
    }

    fn find(&mut self, ty: Type) -> Type {
        self.initialize_type(ty);
        let res = match self.parents.get(&ty) {
            Some(parent_ty) => {
                if *parent_ty == ty {
                    ty
                } else {
                    self.find(*parent_ty)
                }
            }
            None => panic!("Shouldn't have a parent type that isn't a key in TypeUnifier.parents"),
        };
        self.parents.insert(ty, res);
        res
    }

    fn has_hole(&mut self, ty: Type) -> bool {
        match ty.kind() {
            TypeKind::Hole(_) => true,
            TypeKind::Tuple(items) => {
                let root_items = items.iter().map(|t| self.find(*t)).collect::<Vec<_>>();
                root_items.iter().any(|t| self.has_hole(*t))
            }
            TypeKind::Func { inputs, output } => {
                let rooted_inputs = inputs.iter().map(|t| self.find(*t)).collect::<Vec<_>>();
                let rooted_output = self.find(*output);
                rooted_inputs.iter().any(|t| self.has_hole(*t)) || self.has_hole(rooted_output)
            }
            TypeKind::Array(elem) => {
                let rooted_elem = self.find(*elem);
                self.has_hole(rooted_elem)
            }
            _ => false,
        }
    }

    fn solve_constraints(&mut self, globals: &Globals) -> Result<()> {
        self.solve_equational_constraints()?;
        self.solve_projection_constraints()?;
        self.solve_castable_constraints(globals)?;
        for ty in self.parents.keys().cloned().collect::<Vec<_>>() {
            let root_ty = self.find(ty);
            if self.has_hole(root_ty) {
                return Err(TypeError::UninferredType {
                    span: Span::DUMMY,
                    ty: root_ty,
                }
                .into());
            }
        }
        Ok(())
    }

    fn solve_equational_constraints(&mut self) -> Result<()> {
        let constraints = std::mem::take(&mut self.equational_constraints);
        for (ty1, ty2, span) in constraints {
            // println!("PREUNION: ty1: {:?}, root1: {:?}, ty2: {:?}, root2: {:?}",ty1,self.find(ty1),ty2,self.find(ty2));

            ensure!(self.union(ty1, ty2), {
                TypeError::TypeMismatch {
                    expected: ty1,
                    actual: ty2,
                    span: span,
                }
            });
            // println!("POSTUNION: ty1: {:?}, root1: {:?}, ty2: {:?}, root2: {:?}", ty1, self.find(ty1), ty2,self.find(ty2));
        }
        Ok(())
    }
    fn solve_projection_constraints(&mut self) -> Result<()> {
        loop {
            let constraints = std::mem::take(&mut self.projection_constraints);
            if constraints.is_empty() {
                break;
            }

            let mut progress_made = false;
            let mut remaining_constraints = Vec::new();

            for (ty1, ty2, i, span) in constraints {
                // Find the representative of ty2 after equational constraint solving
                let ty2_rep = self.find(ty2);

                match ty2_rep.kind() {
                    TypeKind::Tuple(items) => match items.get(i) {
                        Some(elem_type) => {
                            ensure!(
                                self.union(ty1, *elem_type),
                                TypeError::InvalidProjectionType {
                                    span: span,
                                    ty: ty1
                                }
                            );
                            progress_made = true;

                            // After unifying the element, try to unify the tuple with a more concrete version
                            self.try_unify_tuple_progressively(ty2_rep);
                        }
                        None => bail!(TypeError::InvalidProjectionIndex {
                            index: i,
                            span: span
                        }),
                    },
                    TypeKind::Hole(_) => {
                        // If ty2 is still a hole, we can't solve this constraint yet
                        // Put it back for the next iteration
                        remaining_constraints.push((ty1, ty2, i, span));
                    }
                    _ => ensure!(
                        false,
                        TypeError::InvalidProjectionType {
                            ty: ty2_rep,
                            span: span
                        }
                    ),
                }
            }

            self.projection_constraints = remaining_constraints;

            if !progress_made {
                break;
            }
        }

        // Check if there are any remaining unsolved projection constraints
        if !self.projection_constraints.is_empty() {
            // If there are unsolved constraints, it means we tried to project from
            // a hole type that was never resolved to a concrete tuple/struct type
            let (_, ty2, _, span) = self.projection_constraints[0];
            let ty2_rep = self.find(ty2);
            bail!(TypeError::InvalidProjectionType {
                ty: ty2_rep,
                span: span
            });
        }
        Ok(())
    }

    fn try_unify_tuple_progressively(&mut self, tuple_ty: Type) {
        if let TypeKind::Tuple(items) = tuple_ty.kind() {
            // Check if any elements have been resolved to concrete types
            let mut has_concrete = false;
            let mut resolved_items = Vec::new();

            for item in items {
                let resolved_item = self.find(*item);
                if !matches!(resolved_item.kind(), TypeKind::Hole(_)) {
                    has_concrete = true;
                }
                resolved_items.push(resolved_item);
            }

            // If we have any concrete types, create a more constrained tuple and unify
            if has_concrete {
                let more_concrete_tuple = Type::tuple(resolved_items);
                self.union(tuple_ty, more_concrete_tuple);
            }
        }
    }

    fn solve_castable_constraints(&mut self, globals: &Globals) -> Result<()> {
        let constraints = std::mem::take(&mut self.castable_constraints);
        for (ty1, ty2, span) in constraints {
            match (ty1.kind(), ty2.kind()) {
                (TypeKind::Int, TypeKind::Float) => {}
                (TypeKind::Struct(struct_), TypeKind::Interface(intf)) => {
                    // does struct implements interface
                    let impl_ref = ImplRef {
                        interface: *intf,
                        struct_: *struct_,
                    };
                    ensure!(
                        globals.impls.contains_key(&impl_ref),
                        TypeError::MissingImpl {
                            intf: *intf,
                            ty: ty1,
                            span
                        }
                    );
                }
                (TypeKind::Interface(_), TypeKind::Interface(_)) => {
                    bail!(TypeError::InvalidCast {
                        from: ty1,
                        to: ty2,
                        span,
                    });
                }
                (TypeKind::Struct(_), TypeKind::Struct(_)) => {
                    bail!(TypeError::InvalidCast {
                        from: ty1,
                        to: ty2,
                        span,
                    });
                }
                _ => {
                    bail!(TypeError::InvalidCast {
                        from: ty1,
                        to: ty2,
                        span,
                    });
                }
            }
        }
        Ok(())
    }

    fn subst_func(&mut self) -> impl FnMut(usize) -> Type {
        |hole_id| self.find(Type::hole(hole_id))
    }
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
    num_loops: u32,
    type_unifier: TypeUnifier,
    next_hole_id: usize,
}

/// A predicate which must hold on a concrete type, excluding equality.
#[derive(Clone, Copy, Debug)]
pub enum TypeConstraint {
    /// Type must be either int or float.
    #[allow(unused)]
    Numeric,

    /// Type must be castable to the given type.
    #[allow(unused)]
    CastableTo(Type),
}

impl TypeConstraint {
    /// Returns an error if `ty` does not satisfy the constraint `self`.
    #[allow(unused)]
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
    pub fn new(starting_hole_id: usize) -> Self {
        let mut tcx = Tcx {
            globals: Globals::default(),
            num_loops: 0,
            type_unifier: TypeUnifier::new(),
            next_hole_id: starting_hole_id,
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

    fn push_equational_constraint(&mut self, ty1: Type, ty2: Type, span: Span) {
        self.type_unifier
            .equational_constraints
            .push((ty1, ty2, span))
    }

    fn push_castable_constraint(&mut self, ty1: Type, ty2: Type, span: Span) {
        self.type_unifier
            .castable_constraints
            .push((ty1, ty2, span))
    }

    fn find_next_hole_id(&mut self) -> usize {
        let id = self.next_hole_id;
        self.next_hole_id += 1;
        id
    }

    pub fn check(&mut self, prog: &ast::Program) -> Result<tir::Program> {
        let mut tir_prog = Vec::new();
        for item in &prog.0 {
            self.check_item(&mut tir_prog, item)?;
        }

        // for (ty1, ty2, span) in &self.type_unifier.equational_constraints {
        //     println!("equational constraint: {:?}, {:?}, {:?}", ty1, ty2, span);
        // }

        self.type_unifier.solve_constraints(&self.globals)?;

        // for (k, v) in &self.type_unifier.parents {
        //     println!("parent key: {:?}, value: {:?}", k, v);
        // }

        self.globals.funcs.retain(|_, tds| !tds.is_empty());

        let mut s = self.type_unifier.subst_func();

        for (_, tds) in &mut self.globals.funcs {
            for td in tds {
                td.ty = td.ty.subst(&mut s);
            }
        }

        for f in &mut tir_prog {
            for (_, ty) in &mut f.params {
                *ty = ty.subst(&mut s);
                // println!("{}", ty);
            }
            f.ret_ty = f.ret_ty.subst(&mut s);
            // println!("{}", f.ret_ty);
            f.body = f.body.clone().subst(&mut s);
            // println!("{}", f.body.ty);
        }

        // println!("{:?}", tir_prog);

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
        self.push_equational_constraint(body.ty, ret_ty, body.span);
        // self.ty_equiv(body.ty, ret_ty, body.span)?;

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

    #[allow(unused)]
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
                        self.push_equational_constraint(ty, arg.ty, arg.span);
                        // self.ty_equiv(ty, arg.ty, arg.span)?;
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
                        self.push_equational_constraint(Type::int(), left.ty, left.span);
                        self.push_equational_constraint(Type::int(), right.ty, right.span);
                        // self.ty_equiv(Type::int(), left.ty, left.span)?;
                        // self.ty_equiv(Type::int(), right.ty, right.span)?;
                        Type::int()
                    }
                    Binop::Add | Binop::Sub | Binop::Mul | Binop::Div | Binop::Rem | Binop::Exp => {
                        self.push_equational_constraint(Type::int(), left.ty, left.span);
                        self.push_equational_constraint(Type::int(), right.ty, right.span);
                        // self.ty_constraint(TypeConstraint::Numeric, left.ty, left.span)?;
                        // self.ty_equiv(left.ty, right.ty, right.span)?;
                        left.ty
                    }
                    Binop::Ge | Binop::Gt | Binop::Le | Binop::Lt => {
                        self.push_equational_constraint(Type::int(), left.ty, left.span);
                        self.push_equational_constraint(Type::int(), right.ty, right.span);
                        // self.ty_constraint(TypeConstraint::Numeric, left.ty, left.span)?;
                        // self.ty_equiv(left.ty, right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Or | Binop::And => {
                        self.push_equational_constraint(Type::bool(), left.ty, left.span);
                        self.push_equational_constraint(Type::bool(), right.ty, right.span);
                        // self.ty_equiv(Type::bool(), left.ty, left.span)?;
                        // self.ty_equiv(Type::bool(), right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Eq | Binop::Neq => {
                        self.push_equational_constraint(left.ty, right.ty, right.span);
                        // self.ty_equiv(left.ty, right.ty, right.span)?;
                        Type::bool()
                    }
                    Binop::Concat => {
                        self.push_equational_constraint(Type::string(), left.ty, left.span);
                        self.push_equational_constraint(Type::string(), right.ty, right.span);
                        // self.ty_equiv(Type::string(), left.ty, left.span)?;
                        // self.ty_equiv(Type::string(), right.ty, left.span)?;
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
                self.push_castable_constraint(e.ty, *ty, expr.span);
                // self.ty_constraint(TypeConstraint::CastableTo(*ty), e.ty, expr.span)?;
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

                // we're projecting from a hole type
                if matches!(e.ty.kind(), TypeKind::Hole(_)) {
                    // we need to create a fresh hole for the projection result
                    // we'll create a new hole with a unique ID by finding the next available hole ID
                    let next_hole_id = self.find_next_hole_id();
                    let result_hole = Type::hole(next_hole_id);
                    self.type_unifier.initialize_type(result_hole);

                    self.type_unifier.projection_constraints.push((
                        result_hole,
                        e.ty,
                        *i,
                        expr.span,
                    ));

                    (
                        tir::ExprKind::Project {
                            e: Box::new(e),
                            i: *i,
                        },
                        result_hole,
                    )
                } else {
                    // handle concrete types as before
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
            }
            ast::ExprKind::Call { f, args } => {
                let f = self.check_expr(f)?;

                // Check arguments first
                let arg_exprs: Vec<_> = args
                    .iter()
                    .map(|arg| self.check_expr(arg))
                    .collect::<Result<_>>()?;

                let (_, output) = match f.ty.kind() {
                    TypeKind::Func { inputs, output } => {
                        ensure!(
                            args.len() == inputs.len(),
                            TypeError::WrongNumArgs {
                                expected: inputs.len(),
                                actual: args.len(),
                                span: expr.span
                            }
                        );
                        for (input, arg_expr) in inputs.iter().zip(arg_exprs.clone()) {
                            self.push_equational_constraint(*input, arg_expr.ty, arg_expr.span);
                        }
                        (inputs, output)
                    }
                    TypeKind::Hole(_) => {
                        // f is a hole type, infer function signature from arguments
                        // Create holes for each parameter type
                        let mut param_types = Vec::new();
                        for arg_expr in &arg_exprs {
                            let next_hole_id = self.find_next_hole_id();
                            let param_hole = Type::hole(next_hole_id);
                            self.type_unifier.initialize_type(param_hole);

                            // Constrain parameter to match argument type
                            self.push_equational_constraint(param_hole, arg_expr.ty, arg_expr.span);

                            param_types.push(param_hole);
                        }

                        let next_hole_id = self.find_next_hole_id();
                        let return_hole = Type::hole(next_hole_id);
                        self.type_unifier.initialize_type(return_hole);

                        let func_ty = Type::func(param_types.clone(), return_hole);
                        self.push_equational_constraint(f.ty, func_ty, f.span);

                        match func_ty.kind() {
                            TypeKind::Func {
                                inputs: func_inputs,
                                output: func_output,
                            } => (func_inputs, func_output),
                            _ => unreachable!(),
                        }
                    }
                    _ => bail!(TypeError::TypeMismatchCustom {
                        expected: "function".into(),
                        actual: f.ty,
                        span: f.span
                    }),
                };

                (
                    tir::ExprKind::Call {
                        f: Box::new(f),
                        args: arg_exprs,
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
                        self.push_equational_constraint(*expected_ty, arg.ty, arg.span);
                        // self.ty_equiv(*expected_ty, arg.ty, arg.span)?;
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
                        self.push_equational_constraint(e1.ty, *ty, e1.span);
                        // self.ty_equiv(e1.ty, *ty, e1.span)?;
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
                self.push_equational_constraint(Type::bool(), cond.ty, cond_span);
                // self.ty_equiv(Type::bool(), cond.ty, cond_span)?;
                let then_ = self.check_expr(then_)?;

                let (else_, ty) = match else_ {
                    Some(else_expr) => {
                        let else_ = self.check_expr(else_expr)?;
                        self.push_equational_constraint(then_.ty, else_.ty, else_.span);
                        // self.ty_equiv(then_.ty, else_.ty, else_.span)?;
                        (Some(Box::new(else_)), then_.ty)
                    }
                    None => {
                        // If without else must have unit type in then branch
                        self.push_equational_constraint(Type::unit(), then_.ty, then_.span);
                        // self.ty_equiv(Type::unit(), then_.ty, then_.span)?;
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
                self.num_loops += 1;
                let body = self.check_expr(body)?;
                self.num_loops -= 1;
                (tir::ExprKind::Loop(Box::new(body)), Type::unit())
            }
            ast::ExprKind::While { cond, body } => {
                let cond = self.check_expr(cond)?;
                self.push_equational_constraint(Type::bool(), cond.ty, cond.span);
                // self.ty_equiv(Type::bool(), cond.ty, cond.span)?;
                self.num_loops += 1;
                let body = self.check_expr(body)?;
                self.num_loops -= 1;
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

                self.push_equational_constraint(*ret_ty, body.ty, body.span);
                // self.ty_equiv(*ret_ty, body.ty, body.span)?;
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
                self.push_equational_constraint(src.ty, dst.ty, dst.span);
                // self.ty_equiv(src.ty, dst.ty, dst.span)?;

                (
                    tir::ExprKind::Assign {
                        dst: Box::new(dst),
                        src: Box::new(src),
                    },
                    Type::unit(),
                )
            }
            ast::ExprKind::Break => {
                ensure!(
                    self.num_loops > 0,
                    TypeError::InvalidBreak { span: expr.span }
                );
                (tir::ExprKind::Break, Type::unit())
            }
            ast::ExprKind::ArrayLiteral(exprs) => {
                ensure!(!exprs.is_empty(), TypeError::EmptyArray { span: expr.span });

                let exprs = exprs
                    .iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<Vec<_>>>()?;

                let tys = exprs.iter().map(|e| e.ty).collect::<Vec<_>>();

                // let all_same = tys
                //     .first()
                //     .map(|first| tys.iter().all(|x| x.equiv(first)))
                //     .unwrap_or(true);

                let internal_type = *tys.first().unwrap();
                for ty in &tys[..] {
                    self.push_equational_constraint(internal_type, *ty, expr.span);
                }

                // let all_same = tys
                //     .first()
                //     .map(|first| tys.iter().all(|x| x.equiv(first)))
                //     .unwrap_or(true);

                // ensure!(
                //     all_same,
                //     TypeError::NonHomogenousArray {
                //         span: exprs.first().unwrap().span,
                //         first_type: internal_type
                //     }
                // );

                (
                    tir::ExprKind::ArrayLiteral(exprs),
                    Type::array(internal_type),
                )
            }
            ast::ExprKind::ArrayIndex { array, index } => {
                let array = self.check_expr(array)?;

                let internal_type = match array.ty.kind() {
                    TypeKind::Array(ty) => *ty,
                    TypeKind::Hole(_) => {
                        // Create a new hole for the element type
                        let next_hole_id = self.find_next_hole_id();
                        let element_hole = Type::hole(next_hole_id);
                        self.type_unifier.initialize_type(element_hole);

                        // Add constraint that the array hole must be an array of the element hole type
                        // ?x = [?y] where ?x is the array hole and ?y is the element hole
                        let array_hole = Type::array(element_hole);
                        self.push_equational_constraint(array.ty, array_hole, array.span);

                        element_hole
                    }
                    _ => bail!(TypeError::IndexIntoNonArray {
                        span: array.span,
                        ty: array.ty
                    }),
                };
                let index = self.check_expr(index)?;
                self.push_equational_constraint(Type::int(), index.ty, index.span);
                (
                    tir::ExprKind::ArrayIndex {
                        array: Box::new(array),
                        index: Box::new(index),
                    },
                    internal_type,
                )
            }
            ast::ExprKind::ArrayCopy { value, count } => {
                let value = self.check_expr(value)?;
                let internal_type = value.ty;
                let count = self.check_expr(count)?;
                self.push_equational_constraint(Type::int(), count.ty, count.span);
                // self.ty_equiv(Type::int(), count.ty, count.span)?;
                (
                    tir::ExprKind::ArrayCopy {
                        value: Box::new(value),
                        count: Box::new(count),
                    },
                    Type::array(internal_type),
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
