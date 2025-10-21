//! The code generator that compiles bytecode to Wasm.
//!
//! The bytecode is designed to be relatively similar to Wasm, and the runtime uses Wasm GC objects,
//! so the compilation process is mostly straightforward. A few tricky bits:
//!
//! 1. Wasm expects structured control-flow rather than a CFG. The algorithm which converts into the Wasm control structures
//!    is described in the [`reloop`] module.
//!
//! 2. To make Wasm bytecode compact, everything is described in terms of indexes into tables: tables of types, tables of functions, so on.
//!    Most of the bookkeeping in this module is tracking indexes for code objects, contained on the [`CodegenModule`] struct.
//!
//! 3. This module uses the [`wasm_encoder`] library to generate Wasm bytecode. Unfortunately this library does not share types with [`wasmtime`],
//!    so we have to do a fair amount of conversions between the two, eg as in [`CodegenModule::convert_wasmtime_ty`].

use crate::{tir::Tcx, utils::Symbol};
use itertools::Itertools;
use std::collections::HashMap;
use wasm_encoder::{
    AbstractHeapType, CodeSection, ConstExpr, DataSection, ElementSection, Elements, EntityType,
    ExportKind, ExportSection, FieldType, FuncType, Function, FunctionSection, HeapType,
    ImportSection, InstructionSink, MemorySection, MemoryType, Module, RefType, StorageType,
    StructType, TypeSection, ValType,
};

use crate::bc::types as bc;

use super::reloop::{self, WasmTerminator};

// Note: the reconstructed CFG via the relooping algo doesn't satisfy Wasm's initialization checks
// even for types that are definitely initialized. This isn't an issue for primitive types, but Wasmtime
// will complain for possibly-uninitialized reference types. As a workaround, we say that reference types
// are nullable (and therefore have a default), even though they should never be null in practice.

const REFEXTERN: ValType = ValType::Ref(RefType {
    nullable: true,
    heap_type: HeapType::EXTERN,
});

const REFSTRUCT: ValType = ValType::Ref(RefType {
    nullable: true,
    heap_type: HeapType::Abstract {
        shared: false,
        ty: AbstractHeapType::Struct,
    },
});

const REFFUNC: ValType = ValType::Ref(RefType {
    nullable: true,
    heap_type: HeapType::FUNC,
});

pub struct Import {
    pub module: Symbol,
    pub function: Symbol,
    pub ty: wasmtime::FuncType,
}

pub fn codegen<'a>(
    tcx: &'a Tcx,
    functions: impl IntoIterator<Item = &'a bc::Function>,
    imports: Vec<Import>,
) -> Vec<u8> {
    let mut cg = CodegenModule {
        tcx,
        funcs: FunctionSection::new(),
        func_ty_to_ty_idx: HashMap::new(),
        func_name_to_code_idx: HashMap::new(),
        types: TypeSection::new(),
        struct_ty_idx: HashMap::new(),
        array_ty_idx: HashMap::new(),
        data: DataSection::new(),
        data_offset: 0,
        func_refs: Vec::new(),
        imports: ImportSection::new(),
        exports: ExportSection::new(),
        code: CodeSection::new(),
    };

    cg.add_imports(imports);
    for f in functions {
        cg.gen_func(f);
    }

    let mut elems = ElementSection::new();
    elems.declared(Elements::Expressions(
        RefType::FUNCREF,
        cg.func_refs.into_iter().map(ConstExpr::ref_func).collect(),
    ));

    cg.exports.export("memory", ExportKind::Memory, 0);

    let mut memory = MemorySection::new();
    memory.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    // Sections must be added in this order.
    // See: https://webassembly.github.io/spec/core/binary/modules.html#binary-module
    let mut module = Module::new();
    module.section(&cg.types);
    module.section(&cg.imports);
    module.section(&cg.funcs);
    module.section(&memory);
    module.section(&cg.exports);
    module.section(&elems);
    module.section(&cg.code);
    module.section(&cg.data);

    module.finish()
}

struct CodegenModule<'a> {
    tcx: &'a Tcx,
    funcs: FunctionSection,
    func_ty_to_ty_idx: HashMap<FuncType, u32>,
    func_name_to_code_idx: HashMap<Symbol, u32>,
    types: TypeSection,
    struct_ty_idx: HashMap<StructType, u32>,
    array_ty_idx: HashMap<ValType, u32>,
    data: DataSection,
    data_offset: u32,
    func_refs: Vec<u32>,
    imports: ImportSection,
    exports: ExportSection,
    code: CodeSection,
}

impl CodegenModule<'_> {
    fn func_ty_idx(&mut self, ty: &FuncType) -> u32 {
        *self.func_ty_to_ty_idx.entry(ty.clone()).or_insert_with(|| {
            self.types.ty().func_type(ty);
            self.types.len() - 1
        })
    }

    fn convert_wasmtime_field_ty(&mut self, field: wasmtime::FieldType) -> FieldType {
        FieldType {
            element_type: match field.element_type() {
                wasmtime::StorageType::ValType(ty) => {
                    StorageType::Val(self.convert_wasmtime_ty(ty.clone()))
                }
                wasmtime::StorageType::I16 => StorageType::I16,
                wasmtime::StorageType::I8 => StorageType::I8,
            },
            mutable: field.mutability().is_var(),
        }
    }

    fn convert_wasmtime_ty(&mut self, ty: wasmtime::ValType) -> ValType {
        match ty {
            wasmtime::ValType::I32 => ValType::I32,
            wasmtime::ValType::F32 => ValType::F32,
            wasmtime::ValType::I64 => ValType::I64,
            wasmtime::ValType::F64 => ValType::F64,
            wasmtime::ValType::V128 => ValType::V128,
            wasmtime::ValType::Ref(ref_ty) => ValType::Ref(RefType {
                nullable: true,
                heap_type: match ref_ty.heap_type() {
                    wasmtime::HeapType::Func => HeapType::FUNC,
                    wasmtime::HeapType::Extern => HeapType::EXTERN,
                    wasmtime::HeapType::Struct => HeapType::Abstract {
                        shared: false,
                        ty: AbstractHeapType::Struct,
                    },
                    wasmtime::HeapType::ConcreteStruct(ty) => {
                        let fields = ty
                            .fields()
                            .map(|field| self.convert_wasmtime_field_ty(field))
                            .collect::<Box<[_]>>();
                        let idx = self.wasm_struct_ty_idx(StructType { fields });
                        HeapType::Concrete(idx)
                    }
                    wasmtime::HeapType::ConcreteArray(ty) => {
                        let element_ty = self.convert_wasmtime_ty(
                            ty.field_type().element_type().unwrap_val_type().clone(),
                        );
                        let idx = self.array_ty_idx(element_ty);
                        HeapType::Concrete(idx)
                    }
                    _ => unimplemented!("{ref_ty:#?}"),
                },
            }),
        }
    }

    fn add_imports(&mut self, imports: Vec<Import>) {
        for import in imports {
            let params = import
                .ty
                .params()
                .map(|ty| self.convert_wasmtime_ty(ty))
                .collect::<Vec<_>>();
            let ty = FuncType::new(
                params,
                [self.convert_wasmtime_ty(import.ty.results().next().unwrap())],
            );
            let ty_idx = self.func_ty_idx(&ty);
            self.imports.import(
                import.module.as_str(),
                &import.function,
                EntityType::Function(ty_idx),
            );
            let code_idx = self.imports.len() - 1;
            self.func_name_to_code_idx.insert(import.function, code_idx);
        }
    }

    fn gen_func_ty(&mut self, params: Vec<ValType>, ret_ty: bc::Type) -> u32 {
        let ty = FuncType::new(params, [self.gen_ty(ret_ty)]);
        self.func_ty_idx(&ty)
    }

    fn gen_func(&mut self, f: &bc::Function) {
        let params = f
            .params()
            .map(|(_, ty)| self.gen_ty(ty))
            .collect::<Vec<_>>();

        let ty_idx = self.gen_func_ty(params.clone(), f.ret_ty);

        let locals = f
            .locals
            .iter()
            .skip(params.len())
            .map(|data| self.gen_ty(data.ty))
            .collect::<Vec<_>>();
        let output_func = Function::new_with_locals_types(locals.iter().copied());

        let output_func = {
            let mut cg = CodegenFunc {
                module: self,
                input_func: f,
                output_func,
            };
            cg.gen_func();
            cg.output_func
        };

        self.funcs.function(ty_idx);
        self.code.function(&output_func);
        let func_idx = self.imports.len() + self.code.len() - 1;
        self.func_name_to_code_idx.insert(f.name, func_idx);
        self.exports
            .export(f.name.as_str(), ExportKind::Func, func_idx);
    }

    fn wasm_struct_ty_idx(&mut self, ty: StructType) -> u32 {
        match self.struct_ty_idx.get(&ty) {
            Some(idx) => *idx,
            None => {
                self.types.ty().struct_(ty.fields.clone());
                let idx = self.types.len() - 1;
                self.struct_ty_idx.insert(ty, idx);
                idx
            }
        }
    }

    fn array_ty_idx(&mut self, element_ty: ValType) -> u32 {
        match self.array_ty_idx.get(&element_ty) {
            Some(idx) => *idx,
            None => {
                let storage_ty = StorageType::Val(element_ty);
                self.types.ty().array(&storage_ty, false);
                let idx = self.types.len() - 1;
                self.array_ty_idx.insert(element_ty, idx);
                idx
            }
        }
    }

    fn interface_ty_idx(&mut self, intf: Symbol) -> u32 {
        let methods = &self.tcx.globals().intfs[&intf];
        let fields = methods
            .iter()
            .map(|ty| {
                let params = ty.inputs().iter().map(|ty| self.gen_ty(*ty)).collect_vec();
                let func_idx = self.gen_func_ty(params, ty.output());
                let val_type = ValType::Ref(RefType {
                    nullable: true,
                    heap_type: HeapType::Concrete(func_idx),
                });
                FieldType {
                    element_type: StorageType::Val(val_type),
                    mutable: false,
                }
            })
            .collect::<Box<[_]>>();
        let struct_type = StructType { fields };
        self.wasm_struct_ty_idx(struct_type)
    }

    fn struct_ty_idx(&mut self, ty: bc::Type) -> u32 {
        let bc::TypeKind::Struct(struct_) = ty.kind() else {
            panic!("{ty:?} is not a struct")
        };
        let fields = &self.tcx.globals().structs[struct_];
        let fields = fields
            .iter()
            .map(|ty| FieldType {
                element_type: StorageType::Val(self.gen_ty(*ty)),
                mutable: true,
            })
            .collect::<Box<[_]>>();
        let struct_type = StructType { fields };
        self.wasm_struct_ty_idx(struct_type)
    }

    fn tuple_ty_idx(&mut self, ty: bc::Type) -> u32 {
        let bc::TypeKind::Tuple(el_tys) = ty.kind() else {
            panic!("{ty:?} is not a tuple")
        };
        let fields = el_tys
            .iter()
            .map(|ty| FieldType {
                element_type: StorageType::Val(self.gen_ty(*ty)),
                mutable: true,
            })
            .collect::<Box<[_]>>();
        let struct_type = StructType { fields };
        self.wasm_struct_ty_idx(struct_type)
    }

    fn closure_ty_idx(&mut self) -> u32 {
        let fields = Box::new([
            FieldType {
                element_type: StorageType::Val(REFFUNC),
                mutable: true,
            },
            FieldType {
                element_type: StorageType::Val(REFSTRUCT),
                mutable: true,
            },
        ]);
        let struct_type = StructType { fields };
        self.wasm_struct_ty_idx(struct_type)
    }

    fn gen_ty(&mut self, ty: bc::Type) -> ValType {
        match ty.kind() {
            bc::TypeKind::Bool | bc::TypeKind::Int => ValType::I32,
            bc::TypeKind::Float => ValType::F32,
            bc::TypeKind::String => REFEXTERN,
            bc::TypeKind::Tuple(tys) => {
                if tys.is_empty() {
                    REFSTRUCT
                } else {
                    let idx = self.tuple_ty_idx(ty);
                    ValType::Ref(RefType {
                        heap_type: HeapType::Concrete(idx),
                        nullable: true,
                    })
                }
            }
            bc::TypeKind::Func { .. } => ValType::Ref(RefType {
                heap_type: HeapType::Concrete(self.closure_ty_idx()),
                nullable: true,
            }),
            bc::TypeKind::Struct(_) => {
                let idx = self.struct_ty_idx(ty);
                ValType::Ref(RefType {
                    heap_type: HeapType::Concrete(idx),
                    nullable: true,
                })
            }
            bc::TypeKind::Interface(intf) => {
                let idx = self.interface_ty_idx(*intf);
                ValType::Ref(RefType {
                    heap_type: HeapType::Concrete(idx),
                    nullable: true,
                })
            }
            bc::TypeKind::Array(element_ty) => {
                let element_val_ty = self.gen_ty(*element_ty);
                let idx = self.array_ty_idx(element_val_ty);
                ValType::Ref(RefType {
                    heap_type: HeapType::Concrete(idx),
                    nullable: true,
                })
            }
            bc::TypeKind::Self_ => REFSTRUCT,
            bc::TypeKind::Hole(_) => unreachable!(),
        }
    }
}

struct CodegenFunc<'a, 'b> {
    module: &'a mut CodegenModule<'b>,
    input_func: &'a bc::Function,
    output_func: Function,
}

impl CodegenFunc<'_, '_> {
    fn gen_func(&mut self) {
        let wasm_cfg = self
            .input_func
            .body
            .cfg()
            .map(|_, block| self.gen_block(block), |_, _| ());
        let return_type = self.module.gen_ty(self.input_func.ret_ty);
        reloop::reloop(&mut self.output_func, wasm_cfg, return_type);
        self.output_func.instructions().end();
    }

    fn gen_block(&mut self, block: &bc::BasicBlock) -> reloop::WasmBlock {
        let mut buffer = Vec::new();
        let mut instrs = InstructionSink::new(&mut buffer);
        for stmt in &block.statements {
            self.gen_stmt(stmt, &mut instrs);
        }
        let terminator = match block.terminator.kind() {
            bc::TerminatorKind::Jump(dst) => WasmTerminator::Br(*dst),
            bc::TerminatorKind::CondJump {
                cond,
                true_,
                false_,
            } => {
                self.gen_operand(cond, &mut instrs);
                WasmTerminator::BrIf(*true_, *false_)
            }
            bc::TerminatorKind::Return(op) => {
                self.gen_operand(op, &mut instrs);
                WasmTerminator::Return
            }
        };
        reloop::WasmBlock {
            instrs: buffer,
            terminator,
        }
    }

    fn gen_stmt(&mut self, stmt: &bc::Statement, instrs: &mut InstructionSink) {
        let p = stmt.place;
        if p.projection.is_empty() {
            self.gen_rvalue(&stmt.rvalue, stmt.place.ty, instrs);
            instrs.local_set(p.local.raw());
        } else {
            instrs.local_get(p.local.raw());
            for elem in &p.projection[..p.projection.len() - 1] {
                match elem {
                    bc::ProjectionElem::Field { index, ty } => {
                        instrs.struct_get(self.module.tuple_ty_idx(*ty), *index as u32);
                    }
                    bc::ProjectionElem::ArrayIndex { index, ty } => {
                        self.gen_operand(index, instrs);
                        let element_ty = self.module.gen_ty(*ty);
                        let array_ty_idx = self.module.array_ty_idx(element_ty);
                        instrs.array_get(array_ty_idx);
                    }
                }
            }

            match p.projection.last().unwrap() {
                bc::ProjectionElem::Field { index, ty } => {
                    self.gen_rvalue(&stmt.rvalue, stmt.place.ty, instrs);
                    instrs.struct_set(self.module.tuple_ty_idx(*ty), *index as u32);
                }
                bc::ProjectionElem::ArrayIndex { index, ty } => {
                    self.gen_operand(index, instrs);
                    self.gen_rvalue(&stmt.rvalue, stmt.place.ty, instrs);
                    let element_ty = self.module.gen_ty(*ty);
                    let array_ty_idx = self.module.array_ty_idx(element_ty);
                    instrs.array_set(array_ty_idx);
                }
            }
        }
    }

    fn gen_operand(&mut self, op: &bc::Operand, instrs: &mut InstructionSink) {
        match op {
            bc::Operand::Const(c) => match c {
                bc::Const::Bool(b) => {
                    instrs.i32_const(if *b { 1 } else { 0 });
                }
                bc::Const::Int(n) => {
                    instrs.i32_const(*n);
                }
                bc::Const::Float(n) => {
                    instrs.f32_const(n.0.into());
                }
                bc::Const::String(s) => {
                    let offset = self.module.data_offset;
                    self.module.data.active(
                        0,
                        &ConstExpr::i32_const(offset as i32),
                        s.as_bytes().iter().copied(),
                    );
                    self.module.data_offset += s.len() as u32;

                    instrs.struct_new(self.module.tuple_ty_idx(bc::Type::unit()));
                    instrs.i32_const(offset as i32);
                    instrs.i32_const(s.len() as i32);
                    let alloc_idx = self.module.func_name_to_code_idx[&Symbol::new("alloc_string")];
                    instrs.call(alloc_idx);
                }
            },
            bc::Operand::Place(p) => self.gen_load(*p, instrs),
            bc::Operand::Func { f, ty } => {
                self.gen_rvalue(&bc::Rvalue::Closure { f: *f, env: vec![] }, *ty, instrs)
            }
        }
    }

    fn gen_rvalue(&mut self, rvalue: &bc::Rvalue, ty: bc::Type, instrs: &mut InstructionSink) {
        match rvalue {
            bc::Rvalue::Operand(op) => self.gen_operand(op, instrs),

            bc::Rvalue::Binop { op, left, right } => {
                use bc::{Binop::*, TypeKind::*};

                // Special case this because we need to add the empty environment when calling a function.
                if matches!(op, Concat) {
                    instrs.struct_new(self.module.tuple_ty_idx(bc::Type::unit()));
                }

                let lty = left.ty();
                self.gen_operand(left, instrs);
                self.gen_operand(right, instrs);
                match (op, lty.kind()) {
                    (Add, Int) => instrs.i32_add(),
                    (Add, Float) => instrs.f32_add(),
                    (Sub, Int) => instrs.i32_sub(),
                    (Sub, Float) => instrs.f32_sub(),
                    (Mul, Int) => instrs.i32_mul(),
                    (Mul, Float) => instrs.f32_mul(),
                    (Div, Int) => instrs.i32_div_s(),
                    (Div, Float) => instrs.f32_div(),
                    (Rem, Int) => instrs.i32_rem_s(),
                    (Ge, Int) => instrs.i32_ge_s(),
                    (Ge, Float) => instrs.f32_ge(),
                    (Gt, Int) => instrs.i32_gt_s(),
                    (Gt, Float) => instrs.f32_gt(),
                    (Le, Int) => instrs.i32_le_s(),
                    (Le, Float) => instrs.f32_le(),
                    (Lt, Int) => instrs.i32_lt_s(),
                    (Lt, Float) => instrs.f32_lt(),
                    (BitAnd | And, _) => instrs.i32_and(),
                    (BitOr | Or, _) => instrs.i32_or(),
                    (Shl, _) => instrs.i32_shl(),
                    (Shr, _) => instrs.i32_shr_s(),
                    (Eq, Int) => instrs.i32_eq(),
                    (Eq, Float) => instrs.f32_eq(),
                    (Neq, Int) => instrs.i32_ne(),
                    (Neq, Float) => instrs.f32_ne(),
                    (Concat, String) => {
                        let concat_idx =
                            self.module.func_name_to_code_idx[&Symbol::new("concat_string")];
                        instrs.call(concat_idx)
                    }
                    (op, lty) => unimplemented!("{op:?} {lty:?}"),
                };
            }

            bc::Rvalue::Cast { op, ty } => {
                self.gen_operand(op, instrs);
                match (op.ty().kind(), ty.kind()) {
                    (bc::TypeKind::Int, bc::TypeKind::Float) => {
                        instrs.f32_convert_i32_s();
                    }
                    (bc::TypeKind::Struct(_struct_), bc::TypeKind::Interface(_intf)) => {
                        todo!("cast struct to interface")
                    }
                    (ty1, ty2) => unimplemented!("cast {ty1} as {ty2}"),
                }
            }

            bc::Rvalue::Alloc { kind, args, loc } => match args {
                bc::AllocArgs::Lit(ops) => {
                    for el in ops.iter() {
                        self.gen_operand(el, instrs);
                    }

                    match loc {
                        bc::AllocLoc::Stack => {
                            // For now, treat stack allocations the same as heap allocations in JIT
                            // TODO: Implement proper stack allocation in JIT runtime
                            match kind {
                                bc::AllocKind::Tuple => {
                                    let ty_idx = self.module.tuple_ty_idx(ty);
                                    instrs.struct_new(ty_idx);
                                }
                                bc::AllocKind::Struct => {
                                    let ty_idx = self.module.struct_ty_idx(ty);
                                    instrs.struct_new(ty_idx);
                                }
                                bc::AllocKind::Array => {
                                    let bc::TypeKind::Array(element_ty) = ty.kind() else {
                                        panic!("{ty:?} is not an array")
                                    };
                                    let element_val_ty = self.module.gen_ty(*element_ty);
                                    let ty_idx = self.module.array_ty_idx(element_val_ty);
                                    instrs.array_new_fixed(ty_idx, ops.len() as u32);
                                }
                            }
                        }
                        bc::AllocLoc::Heap => match kind {
                            bc::AllocKind::Tuple => {
                                let ty_idx = self.module.tuple_ty_idx(ty);
                                instrs.struct_new(ty_idx);
                            }
                            bc::AllocKind::Struct => {
                                let ty_idx = self.module.struct_ty_idx(ty);
                                instrs.struct_new(ty_idx);
                            }
                            bc::AllocKind::Array => {
                                let bc::TypeKind::Array(element_ty) = ty.kind() else {
                                    panic!("{ty:?} is not an array")
                                };
                                let element_val_ty = self.module.gen_ty(*element_ty);
                                let ty_idx = self.module.array_ty_idx(element_val_ty);
                                instrs.array_new_fixed(ty_idx, ops.len() as u32);
                            }
                        },
                    }
                }
                bc::AllocArgs::ArrayCopy { value, count } => {
                    self.gen_operand(value, instrs);
                    self.gen_operand(count, instrs);
                    let bc::TypeKind::Array(element_ty) = ty.kind() else {
                        panic!("{ty:?} is not an array")
                    };
                    let element_val_ty = self.module.gen_ty(*element_ty);
                    let ty_idx = self.module.array_ty_idx(element_val_ty);
                    instrs.array_new(ty_idx);
                }
            },

            bc::Rvalue::Closure { f, env } => {
                let func_idx = *self
                    .module
                    .func_name_to_code_idx
                    .get(f)
                    .unwrap_or_else(|| panic!("Missing definition for function: {f}"));
                if !self.module.func_refs.contains(&func_idx) {
                    self.module.func_refs.push(func_idx);
                }
                instrs.ref_func(func_idx);

                self.gen_rvalue(
                    &bc::Rvalue::Alloc {
                        args: bc::AllocArgs::Lit(env.clone()),
                        kind: bc::AllocKind::Tuple,
                        loc: bc::AllocLoc::Heap,
                    },
                    bc::Type::tuple(env.iter().map(|op| op.ty()).collect()),
                    instrs,
                );

                let ty_idx = self.module.closure_ty_idx();
                instrs.struct_new(ty_idx);
            }

            bc::Rvalue::Call { f, args } => {
                let closure_ty_idx = self.module.closure_ty_idx();

                self.gen_operand(f, instrs);
                instrs.struct_get(closure_ty_idx, 1);

                for arg in args.iter().rev() {
                    self.gen_operand(arg, instrs);
                }

                self.gen_operand(f, instrs);
                instrs.struct_get(closure_ty_idx, 0);

                let bc::TypeKind::Func { inputs, output } = f.ty().kind() else {
                    panic!("{ty:?} is not a function")
                };
                let params = [REFSTRUCT]
                    .into_iter()
                    .chain(inputs.iter().map(|ty| self.module.gen_ty(*ty)))
                    .collect();
                let func_ty_idx = self.module.gen_func_ty(params, *output);
                instrs.ref_cast_non_null(HeapType::Concrete(func_ty_idx));

                instrs.call_ref(func_ty_idx);
            }

            bc::Rvalue::MethodCall { .. } => todo!("method call"),
        }
    }

    fn gen_load(&mut self, place: bc::Place, instrs: &mut InstructionSink) {
        instrs.local_get(place.local.raw());
        for elem in &place.projection {
            match elem {
                bc::ProjectionElem::Field { index, ty } => {
                    instrs.struct_get(self.module.tuple_ty_idx(*ty), *index as u32);
                }
                bc::ProjectionElem::ArrayIndex { index, ty } => {
                    self.gen_operand(index, instrs);
                    let element_ty = self.module.gen_ty(*ty);
                    let array_ty_idx = self.module.array_ty_idx(element_ty);
                    instrs.array_get(array_ty_idx);
                }
            }
        }
    }
}
