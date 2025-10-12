//! The runtime which executes bytecode.
//!
//! The runtime is implemented as a JIT compiler.
//! This module contains the Rice interpreter, which can call to the Wasm code generator in [`codegen`].
//! The runtime uses the [Wasmtime][wasmtime] engine to allow the interpreter to interoperate with the compiler.
//!
//! All Rice values are encoded as WebAssembly values using the Wasmtime [`Val`] type.
//! Primitive types like booleans, ints, and floats are encoded as such without allocation.
//! Anything larger requires a heap allocation (tuples, objects, closures), which uses the GC extension to Wasm/Wasmtime.
//! Structs are encoded as [`StructRef`], and functions as [`Func`].
//! The Wasm spec with the GC extension is documented here: <https://webassembly.github.io/gc/core/>
//!
//! Evaluation starts by registering functions with the runtime via [`Runtime::register`].
//! Functions annotated with `#[jit]` are eagerly compiled.
//! Then the registered functions can be called via [`Runtime::call_toplevel`].
//!
//! A function is either interpreted in a [`Frame`] or executed as Wasm by Wasmtime.
//! A frame contains primarily a [program counter][Frame::pc] and values for [locals][Frame::locals].
//! [`Frame::step`] executes the current instruction in the frame, continuing until the function returns.
//!
//! To simplify code generation with respect to the calling convention, all functions have a first parameter
//! which is their environment. This parameter is used for closures and unused for top-level
//!
//! Note that Wasmtime uses the anyhow crate to represent errors, so in this module we use anyhow
//! instead of miette which the rest of the codebase uses.

use std::{
    cell::OnceCell,
    collections::{HashMap, hash_map::Entry},
    ops::{Add, Deref, Div, Mul, Sub},
    string::String as StdString,
    sync::{Arc, LazyLock, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use anyhow::{Context, Result, bail};
use either::Either;
use indexical::{IndexicalIteratorExt, map::DenseArcIndexMap as IndexMap};
use itertools::Itertools;
use log::debug;
use wasmtime::{
    ArrayRefPre, AsContext, AsContextMut, Caller, Config, Engine, FieldType, Func, FuncType,
    HeapType, Linker, Module, Mutability, RefType, Rooted, StorageType, Store, StoreContext,
    StoreContextMut, StructRef, StructRefPre, StructType, Val, ValType,
};

use self::{codegen::Import, conversions::Wasmable};
use crate::{
    bc::types as bc,
    stdlib::stdlib,
    tir::{Tcx, types::ImplRef},
    utils::{Symbol, sym},
};

mod codegen;
pub mod conversions;
mod reloop;

/// The central data structure for the runtime.
///
/// This type is wrapped in an [`Arc`] by [`Runtime`] to allow sharing handles to it
/// without tricky lifetimes. Its mutable fields are wrapped by an [`RwLock`] to permit
/// interior mutability. Note that this structure is not expected to be used in a concurrent way!
/// The reader-writer locks are not used as locks (i.e., no blocking calls are made) but rather as
/// ref-cells which conveniently implement `Send` and `Sync`.
pub struct RuntimeInner {
    // Type info
    tcx: Tcx,
    opts: RuntimeOptions,

    // Wasmtime structures
    engine: Engine,
    linker: RwLock<Linker<()>>,
    store: RwLock<Store<()>>,
    struct_allocators: RwLock<HashMap<Vec<ValTypeEq>, StructRefPre>>,
    array_allocators: RwLock<HashMap<Vec<ValTypeEq>, ArrayRefPre>>,

    // Rice runtime data
    functions: RwLock<HashMap<Symbol, FuncDesc>>,
    vtables: RwLock<HashMap<ImplRef, Val>>,
    panic: RwLock<Option<String>>,
    unit: OnceLock<Val>,
}

/// A wrapper for the runtime. See [`RuntimeInner`].
#[derive(Clone)]
pub struct Runtime(Arc<RuntimeInner>);

impl Deref for Runtime {
    type Target = RuntimeInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Workaround so we can use a ValType as a key in a hash map.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Hash)]
struct ValTypeEq(ValType);

impl PartialEq for ValTypeEq {
    fn eq(&self, other: &Self) -> bool {
        ValType::eq(&self.0, &other.0)
    }
}

impl Eq for ValTypeEq {}

/// Describes a function linked into the Wasmtime runtime.
struct FuncDesc {
    /// The name of the module that contains the function,
    /// necessary for importing this function into other Wasm functions.
    module: Symbol,

    /// The callable handle to the function.
    handle: Func,
}

macro_rules! store {
    ($rt:expr) => {
        &mut $rt.store
    };
}

/// Converts a Rust type to a Wasm [`Val`].
macro_rules! to_val {
    ($self:expr, $e:expr) => {
        $e.to_val(&$self.rt, store!($self))?
    };
}

/// Converts a Wasm [`Val`] to a Rust type.
macro_rules! from_val {
    ($ty:ty, $self:expr, $e:expr) => {{
        let reduced = <$ty>::reduce_val($e, &mut $self.store.as_context_mut())?;
        let result = <$ty>::from_reduced(reduced, &$self.store.as_context())?;
        result
    }};
}

static REFSTRUCT: LazyLock<ValType> =
    LazyLock::new(|| ValType::Ref(RefType::new(true, HeapType::Struct)));
static REFFUNC: LazyLock<ValType> =
    LazyLock::new(|| ValType::Ref(RefType::new(true, HeapType::Func)));
static REFEXTERN: LazyLock<ValType> =
    LazyLock::new(|| ValType::Ref(RefType::new(true, HeapType::Extern)));

const HOST: &str = "host";

/// Interface for types which can be used as functions in the Wasm runtime.
///
/// Allows for calling either bytecode functions or Rust-level stdlib functions.
pub trait WasmFunc: Send + Sync + 'static {
    /// The source-level type of the function used for type-checking.
    ///
    /// Should exclude the dummy environment parameter.
    fn src_type(&self) -> bc::Type;

    /// The runtime-level type of the function used for execution.
    ///
    /// Should include the dummy environment parameter.
    fn rt_type(&self) -> bc::Type;

    fn call(&self, rt: &Runtime, caller: Caller<'_, ()>, args: &[Val]) -> Result<Val>;
}

impl WasmFunc for &'static dyn WasmFunc {
    fn call(&self, rt: &Runtime, caller: Caller<'_, ()>, args: &[Val]) -> Result<Val> {
        (*self).call(rt, caller, args)
    }

    fn src_type(&self) -> bc::Type {
        (*self).src_type()
    }

    fn rt_type(&self) -> bc::Type {
        (*self).rt_type()
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub disable_jit: bool,
}

#[allow(dead_code)]
impl Runtime {
    pub fn new(tcx: Tcx, opts: RuntimeOptions) -> Result<Runtime> {
        let mut config = Config::new();
        config.wasm_function_references(true).wasm_gc(true);

        let engine = Engine::new(&config)?;
        let linker = Linker::new(&engine);
        let store = Store::new(&engine, ());

        let rt = Runtime(Arc::new(RuntimeInner {
            engine,
            tcx,
            opts,
            linker: RwLock::new(linker),
            store: RwLock::new(store),
            struct_allocators: RwLock::new(HashMap::new()),
            array_allocators: RwLock::new(HashMap::new()),
            functions: RwLock::new(HashMap::new()),
            vtables: RwLock::new(HashMap::new()),
            panic: RwLock::new(None),
            unit: OnceLock::new(),
        }));

        let unit = rt.alloc_tuple(&mut *rt.store.try_write().unwrap(), vec![])?;
        rt.unit.set(unit).unwrap();

        for (name, f) in stdlib() {
            rt.register_function(*name, &**f)?;
        }

        Ok(rt)
    }

    // Array of helper functions for getting read or write access to runtime data structures.
    // See the doc comment on `RuntimeInner` for why this is needed.

    fn linker(&self) -> RwLockReadGuard<'_, Linker<()>> {
        self.linker.try_read().unwrap()
    }

    fn linker_mut(&self) -> RwLockWriteGuard<'_, Linker<()>> {
        self.linker.try_write().unwrap()
    }

    fn store(&self) -> RwLockReadGuard<'_, Store<()>> {
        self.store.try_read().unwrap()
    }

    fn store_mut(&self) -> RwLockWriteGuard<'_, Store<()>> {
        self.store.try_write().unwrap()
    }

    fn functions(&self) -> RwLockReadGuard<'_, HashMap<Symbol, FuncDesc>> {
        self.functions.try_read().unwrap()
    }

    fn functions_mut(&self) -> RwLockWriteGuard<'_, HashMap<Symbol, FuncDesc>> {
        self.functions.try_write().unwrap()
    }

    fn panic(&self) -> RwLockReadGuard<'_, Option<String>> {
        self.panic.try_read().unwrap()
    }

    fn panic_mut(&self) -> RwLockWriteGuard<'_, Option<String>> {
        self.panic.try_write().unwrap()
    }

    fn vtables(&self) -> RwLockReadGuard<'_, HashMap<ImplRef, Val>> {
        self.vtables.try_read().unwrap()
    }

    fn vtables_mut(&self) -> RwLockWriteGuard<'_, HashMap<ImplRef, Val>> {
        self.vtables.try_write().unwrap()
    }

    fn struct_allocators_mut(&self) -> RwLockWriteGuard<'_, HashMap<Vec<ValTypeEq>, StructRefPre>> {
        self.struct_allocators.try_write().unwrap()
    }

    fn array_allocators_mut(&self) -> RwLockWriteGuard<'_, HashMap<Vec<ValTypeEq>, ArrayRefPre>> {
        self.array_allocators.try_write().unwrap()
    }

    // End accessor functions

    pub fn function(&self, name: Symbol) -> Result<Func> {
        let functions = self.functions();
        let func_desc = functions
            .get(&name)
            .with_context(|| format!("attempted to access unknown function: {name}"))?;
        Ok(func_desc.handle)
    }

    fn call(&self, store: StoreContextMut<'_, ()>, func: &Func, args: &[Val]) -> Result<Val> {
        let mut results = Some(Val::null_any_ref());
        func.call(store, args, results.as_mut_slice())?;

        if let Some(panic) = &*self.panic() {
            bail!("function panicked with error: {}", panic)
        }

        Ok(results.take().unwrap())
    }

    pub fn call_toplevel(&self, func: &Func, mut args: Vec<Val>) -> Result<Val> {
        args.insert(0, *self.unit.get().unwrap());
        self.call(self.store_mut().as_context_mut(), func, &args)
    }

    pub fn register(&self, prog: bc::Program) -> Result<()> {
        for f in prog.into_functions() {
            if !self.opts.disable_jit && f.jit() {
                self.jit(&f, self.store_mut().as_context_mut())?;
            } else {
                self.register_function(f.name, f)?;
            }
        }
        Ok(())
    }

    fn register_function<F: WasmFunc>(&self, name: Symbol, func: F) -> Result<()> {
        let mut linker = self.linker_mut();
        let rt_handle = self.clone();

        fn translate_ty(ty: bc::Type) -> ValType {
            match ty.kind() {
                bc::TypeKind::Bool | bc::TypeKind::Int => ValType::I32,
                bc::TypeKind::Float => ValType::F32,
                bc::TypeKind::String => REFEXTERN.clone(),
                bc::TypeKind::Tuple(..)
                | bc::TypeKind::Func { .. }
                | bc::TypeKind::Struct(..)
                | bc::TypeKind::Interface(..) => REFSTRUCT.clone(),
                bc::TypeKind::Array(_) => ValType::Ref(RefType::new(true, HeapType::Array)),
                bc::TypeKind::Hole(_) | bc::TypeKind::Self_ => unreachable!(),
            }
        }

        let bc::TypeKind::Func { inputs, output } = func.rt_type().kind() else {
            unreachable!()
        };

        let func_type = FuncType::new(
            &self.engine,
            inputs.iter().copied().map(translate_ty),
            [translate_ty(*output)],
        );

        linker.func_new(
            HOST, // The host module contains all non-Wasm function.
            name.as_str(),
            func_type,
            move |caller, args, ret| {
                let ret_val = func.call(&rt_handle, caller, args)?;
                ret[0] = ret_val;
                Ok(())
            },
        )?;

        // Get the function we just linked and store it in our own dictionary for later use.
        let mut store = self.store_mut();
        let func_ref = linker
            .get(&mut *store, HOST, name.as_str())
            .expect("linker missing function that was just linked")
            .into_func()
            .expect("function that was just linked is somehow not a function");
        let func_desc = FuncDesc {
            module: sym(HOST),
            handle: func_ref,
        };
        self.functions_mut().insert(name, func_desc);

        Ok(())
    }

    fn jit(&self, f: &bc::Function, mut store: StoreContextMut<'_, ()>) -> Result<()> {
        let imports = self
            .functions()
            .iter()
            .map(|(name, desc)| Import {
                function: *name,
                module: desc.module,
                ty: desc.handle.ty(&mut store),
            })
            .collect_vec();

        let wasm = codegen::codegen(&self.tcx, [f], imports);
        debug!("Wasm:\n{}", wasmprinter::print_bytes(&wasm).unwrap());

        let module = Module::new(&self.engine, wasm)?;
        let module_name = f.name.as_str();

        let mut linker = self.linker_mut();
        linker.module(&mut store, module_name, &module)?;

        let handle = linker
            .get(&mut store, module_name, f.name.as_str())
            .expect("module missing exported function")
            .into_func()
            .expect("exported item is not function");
        let func_desc = FuncDesc {
            module: sym(module_name),
            handle,
        };
        self.functions_mut().insert(f.name, func_desc);

        Ok(())
    }

    fn get_abstract_ty(&self, store: StoreContext<'_, ()>, value: Val) -> ValType {
        match value {
            Val::I32(_) => ValType::I32,
            Val::F32(_) => ValType::F32,
            Val::FuncRef(_) => REFFUNC.clone(),
            Val::AnyRef(any_ref) => {
                let any_ref = any_ref.expect("any_ref is null");
                if any_ref.is_struct(&store).expect("reference unrooted") {
                    REFSTRUCT.clone()
                } else if any_ref.is_array(&store).expect("reference unrooted") {
                    let array_ref = any_ref
                        .as_array(&store)
                        .expect("reference is array")
                        .expect("array ref");
                    let array_type = array_ref.ty(&store).expect("array type");
                    ValType::Ref(RefType::new(true, HeapType::ConcreteArray(array_type)))
                } else {
                    unreachable!()
                }
            }
            Val::ExternRef(_) => REFEXTERN.clone(),
            _ => unimplemented!("{value:#?}"),
        }
    }

    fn alloc_tuple(
        &self,
        mut store: impl AsContextMut<Data = ()>,
        fields: Vec<Val>,
    ) -> Result<Val> {
        if fields.is_empty()
            && let Some(unit) = self.unit.get()
        {
            return Ok(*unit);
        }

        let field_tys = fields
            .iter()
            .copied()
            .map(|val| ValTypeEq(self.get_abstract_ty(store.as_context(), val)))
            .collect::<Vec<_>>();

        let make_struct_alloc = |field_tys: &Vec<ValTypeEq>| {
            let struct_ty = StructType::new(
                &self.engine,
                field_tys
                    .iter()
                    .map(|ty| FieldType::new(Mutability::Var, StorageType::ValType(ty.0.clone()))),
            )
            .expect("Tuple has too many fields");
            StructRefPre::new(&mut store, struct_ty)
        };

        let mut allocators = self.struct_allocators_mut();
        let alloc = allocators
            .entry(field_tys)
            .or_insert_with_key(make_struct_alloc);

        let struct_ref = StructRef::new(&mut store, alloc, &fields)?;
        Ok(Val::from(struct_ref))
    }

    fn alloc_array(
        &self,
        mut store: impl AsContextMut<Data = ()>,
        elements: Vec<Val>,
    ) -> Result<Val> {
        if elements.is_empty() {
            return Err(anyhow::anyhow!(
                "should've been caught in typecheck: array must have at least one element"
            ));
        }

        let element_ty = self.get_abstract_ty(store.as_context(), elements[0]);
        let element_tys = vec![ValTypeEq(element_ty)];

        let make_array_alloc = |element_tys: &Vec<ValTypeEq>| {
            let array_ty = wasmtime::ArrayType::new(
                &self.engine,
                FieldType::new(
                    Mutability::Var,
                    StorageType::ValType(element_tys[0].0.clone()),
                ),
            );
            ArrayRefPre::new(&mut store, array_ty)
        };

        let mut allocators = self.array_allocators_mut();
        let alloc = allocators
            .entry(element_tys)
            .or_insert_with_key(make_array_alloc);

        let array_ref = wasmtime::ArrayRef::new_fixed(&mut store, alloc, &elements)?;
        Ok(Val::from(array_ref))
    }

    fn alloc_array_copy(
        &self,
        mut store: impl AsContextMut<Data = ()>,
        value: Val,
        count: u32,
    ) -> Result<Val> {
        let element_ty = self.get_abstract_ty(store.as_context(), value);
        let element_tys = vec![ValTypeEq(element_ty)];

        let make_array_alloc = |element_tys: &Vec<ValTypeEq>| {
            let array_ty = wasmtime::ArrayType::new(
                &self.engine,
                FieldType::new(
                    Mutability::Var,
                    StorageType::ValType(element_tys[0].0.clone()),
                ),
            );
            ArrayRefPre::new(&mut store, array_ty)
        };

        let mut allocators = self.array_allocators_mut();
        let alloc = allocators
            .entry(element_tys)
            .or_insert_with_key(make_array_alloc);

        let array_ref = wasmtime::ArrayRef::new(&mut store, alloc, &value, count)?;
        Ok(Val::from(array_ref))
    }

    fn alloc_vtable(&self, store: impl AsContextMut<Data = ()>, impl_: ImplRef) -> Result<Val> {
        match self.vtables_mut().entry(impl_) {
            Entry::Occupied(entry) => Ok(*entry.get()),
            Entry::Vacant(entry) => {
                let methods = &self.tcx.globals().impls[&impl_];
                let funcs = methods
                    .iter()
                    .map(|name| Val::FuncRef(Some(self.functions()[name].handle)))
                    .collect_vec();
                let vtable = self.alloc_tuple(store, funcs)?;
                entry.insert(vtable);
                Ok(vtable)
            }
        }
    }
}

impl WasmFunc for bc::Function {
    fn call(&self, rt: &Runtime, mut caller: Caller<'_, ()>, args: &[Val]) -> Result<Val> {
        let locals = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let arg_local = bc::LocalIdx::new(i);
                (arg_local, *arg)
            })
            .collect_indexical(&self.locals);

        let mut frame = Frame {
            function: self,
            rt,
            locals,
            pc: bc::Location::START,
            store: caller.as_context_mut(),
            ret_val: OnceCell::new(),
        };
        frame.eval()
    }

    fn rt_type(&self) -> bc::Type {
        bc::Type::func(self.params().map(|(_, ty)| ty).collect(), self.ret_ty)
    }

    fn src_type(&self) -> bc::Type {
        unreachable!()
    }
}

/// A stack frame for an executing function.
struct Frame<'a> {
    function: &'a bc::Function,
    rt: &'a Runtime,
    locals: IndexMap<bc::Local, Val>,
    pc: bc::Location,
    store: StoreContextMut<'a, ()>,
    ret_val: OnceCell<Val>,
}

/// A location in memory that can be described by a [`bc::Place`].
#[derive(Clone, Copy)]
enum MemPlace {
    /// A local in the current stack frame.
    Local(bc::LocalIdx),

    /// A field of a heap-allocated struct.
    StructField(Rooted<StructRef>, usize),

    /// An element of a heap-allocated array.
    ArrayElement(Rooted<wasmtime::ArrayRef>, u32),
}

impl Frame<'_> {
    /// Executes a single instruction, returning true if there is more work to do in this frame.
    fn step(&mut self) -> Result<bool> {
        // Locate the instruction at the program counter
        let block = self.function.body.data(self.pc.block);
        let instr = block.get(self.pc.instr);

        // Execute the instruction
        match instr {
            Either::Left(stmt) => {
                let value = self.eval_rvalue(&stmt.rvalue)?;
                self.store(stmt.place, value)?;
                self.pc.instr += 1;
                Ok(true)
            }

            Either::Right(term) => match term.kind() {
                bc::TerminatorKind::Jump(block) => {
                    self.pc = block.entry();
                    Ok(true)
                }

                bc::TerminatorKind::CondJump {
                    cond: cond_op,
                    true_,
                    false_,
                } => {
                    let cond_val = self.eval_operand(cond_op)?;
                    let cond = from_val!(bool, self, cond_val);
                    let block = if cond { true_ } else { false_ };
                    self.pc = block.entry();
                    Ok(true)
                }

                bc::TerminatorKind::Return(op) => {
                    let value = self.eval_operand(op)?;
                    self.ret_val.set(value).expect("ret_val is set once");
                    Ok(false)
                }
            },
        }
    }

    fn eval_mem_place(&mut self, mem_place: MemPlace) -> Result<Val> {
        match mem_place {
            MemPlace::Local(local) => Ok(*self
                .locals
                .get(local)
                .with_context(|| format!("ICE: missing local: {local}"))?),
            MemPlace::StructField(struct_ref, i) => struct_ref.field(store!(self), i),
            MemPlace::ArrayElement(array_ref, i) => {
                // oob
                let len = array_ref.len(store!(self))?;
                if i >= len {
                    //changed
                    bail!("array index {} out of bounds (length {})", i, len);
                }
                array_ref.get(store!(self), i)
            }
        }
    }

    fn struct_ref(&mut self, val: Val) -> Result<Rooted<StructRef>> {
        val.any_ref()
            .context("ICE: not an anyref")?
            .context("ICE: null anyref")?
            .as_struct(store!(self))?
            .context("ICE: not a structref")
    }

    fn array_ref(&mut self, val: Val) -> Result<Rooted<wasmtime::ArrayRef>> {
        val.any_ref()
            .context("ICE: not an anyref")?
            .context("ICE: null anyref")?
            .as_array(store!(self))?
            .context("ICE: not an arrayref")
    }

    fn func_ref(&mut self, val: Val) -> Result<Func> {
        Ok(*val
            .func_ref()
            .context("ICE: not a funcref")?
            .context("ICE: funcref null")?)
    }

    /// Walks a place until reaching a final memory location described by [`MemPlace`].
    fn resolve_place(&mut self, place: bc::Place) -> Result<MemPlace> {
        let mut mem_place = MemPlace::Local(place.local);
        for elem in &place.projection {
            let val = self.eval_mem_place(mem_place)?;
            mem_place = match elem {
                bc::ProjectionElem::Field { index, .. } => {
                    let struct_ref = self.struct_ref(val)?;
                    MemPlace::StructField(struct_ref, *index)
                }
                bc::ProjectionElem::ArrayIndex { index, .. } => {
                    let array_ref = self.array_ref(val)?;
                    let index_val = self.eval_operand(index)?;
                    let index_int = from_val!(i32, self, index_val) as u32;
                    MemPlace::ArrayElement(array_ref, index_int)
                }
            }
        }
        Ok(mem_place)
    }

    fn store(&mut self, place: bc::Place, value: Val) -> Result<()> {
        let mem_place = self.resolve_place(place)?;
        match mem_place {
            MemPlace::Local(local) => {
                self.locals.insert(local, value);
            }
            MemPlace::StructField(struct_ref, i) => {
                struct_ref.set_field(store!(self), i, value)?;
            }
            MemPlace::ArrayElement(array_ref, i) => {
                // Bounds checking for array assignment
                let len = array_ref.len(store!(self))?;
                if i >= len {
                    //changed
                    bail!("array index {} out of bounds (length {})", i, len);
                }
                array_ref.set(store!(self), i, value)?;
            }
        }
        Ok(())
    }

    fn eval_operand(&mut self, op: &bc::Operand) -> Result<Val> {
        Ok(match op {
            bc::Operand::Const(c) => match c {
                bc::Const::Bool(b) => to_val!(self, b),
                bc::Const::Int(n) => to_val!(self, n),
                bc::Const::String(s) => to_val!(self, s),
                bc::Const::Float(f) => to_val!(self, f.0),
            },
            bc::Operand::Place(p) => {
                let mem_place = self.resolve_place(*p)?;
                self.eval_mem_place(mem_place)?
            }
            bc::Operand::Func { f, .. } => {
                self.eval_rvalue(&bc::Rvalue::Closure { f: *f, env: vec![] })?
            }
        })
    }

    fn eval_rvalue(&mut self, rvalue: &bc::Rvalue) -> Result<Val> {
        Ok(match rvalue {
            bc::Rvalue::Operand(op) => self.eval_operand(op)?,

            bc::Rvalue::Binop { op, left, right } => {
                let lty = left.ty();
                let left_val = self.eval_operand(left)?;
                let right_val = self.eval_operand(right)?;

                macro_rules! op {
                    ($lty:ty, $rty:ty, $op:expr) => {{
                        let context_mut = &mut self.store.as_context_mut();
                        let left_reduced = <$lty>::reduce_val(left_val, context_mut)?;
                        let right_reduced = <$rty>::reduce_val(right_val, context_mut)?;
                        let context = &self.store.as_context();
                        let left = <$lty>::from_reduced(left_reduced, context)?;
                        let right = <$rty>::from_reduced(right_reduced, context)?;
                        to_val!(self, $op(left, right))
                    }};
                }

                use bc::{Binop::*, TypeKind::*};
                match (op, lty.kind()) {
                    (Add, Int) => op!(i32, i32, i32::wrapping_add),
                    (Add, Float) => op!(f32, f32, f32::add),
                    (Sub, Int) => op!(i32, i32, i32::wrapping_sub),
                    (Sub, Float) => op!(f32, f32, f32::sub),
                    (Mul, Int) => op!(i32, i32, i32::wrapping_mul),
                    (Mul, Float) => op!(f32, f32, f32::mul),
                    (Div, Int) => op!(i32, i32, i32::wrapping_div),
                    (Div, Float) => op!(f32, f32, f32::div),
                    (Rem, Int) => op!(i32, i32, i32::wrapping_rem),
                    (Rem, Float) => op!(f32, f32, |n1, n2| n1 % n2),
                    (Exp, Int) => op!(i32, i32, |n1, n2| i32::pow(n1, n2 as u32)),
                    (Exp, Float) => op!(f32, f32, f32::powf),
                    (Shl, Int) => op!(i32, i32, |n1, n2| i32::wrapping_shl(n1, n2 as u32)),
                    (Shr, Int) => op!(i32, i32, |n1, n2| i32::wrapping_shr(n1, n2 as u32)),
                    (BitOr, Int) => op!(i32, i32, |n1, n2| n1 | n2),
                    (BitAnd, Int) => op!(i32, i32, |n1, n2| n1 & n2),
                    (Ge, Int) => op!(i32, i32, |n1, n2| i32::ge(&n1, &n2)),
                    (Ge, Float) => op!(f32, f32, |n1, n2| f32::ge(&n1, &n2)),
                    (Lt, Int) => op!(i32, i32, |n1, n2| i32::lt(&n1, &n2)),
                    (Lt, Float) => op!(f32, f32, |n1, n2| f32::lt(&n1, &n2)),
                    (Le, Int) => op!(i32, i32, |n1, n2| i32::le(&n1, &n2)),
                    (Le, Float) => op!(f32, f32, |n1, n2| f32::le(&n1, &n2)),
                    (Gt, Int) => op!(i32, i32, |n1, n2| i32::gt(&n1, &n2)),
                    (Gt, Float) => op!(f32, f32, |n1, n2| f32::gt(&n1, &n2)),
                    (Eq, Int) => op!(i32, i32, |n1, n2| i32::eq(&n1, &n2)),
                    (Eq, Float) => op!(f32, f32, |n1, n2| f32::eq(&n1, &n2)),
                    (Eq, Bool) => op!(bool, bool, |b1, b2| b1 == b2),
                    (Eq, String) => op!(StdString, StdString, |s1, s2| s1 == s2),
                    (Neq, Int) => op!(i32, i32, |n1, n2| i32::ne(&n1, &n2)),
                    (Neq, Float) => op!(f32, f32, |n1, n2| f32::ne(&n1, &n2)),
                    (Neq, Bool) => op!(bool, bool, |b1, b2| b1 != b2),
                    (And, Bool) => op!(bool, bool, |b1, b2| b1 && b2),
                    (Or, Bool) => op!(bool, bool, |b1, b2| b1 || b2),
                    (Concat, String) => op!(StdString, StdString, |s1: &str, s2: &str| {
                        s1.to_string() + s2
                    }),
                    _ => unimplemented!("{op:?} {lty:?}"),
                }
            }
            bc::Rvalue::Cast { op, ty } => {
                let val = self.eval_operand(op)?;
                match (op.ty().kind(), ty.kind()) {
                    (bc::TypeKind::Int, bc::TypeKind::Float) => {
                        let n = from_val!(i32, self, val);
                        to_val!(self, n as f32)
                    }
                    (bc::TypeKind::Struct(struct_), bc::TypeKind::Interface(intf)) => {
                        let impl_ = ImplRef {
                            interface: *intf,
                            struct_: *struct_,
                        };
                        let vtable = self.rt.alloc_vtable(store!(self), impl_)?;
                        self.rt.alloc_tuple(store!(self), vec![val, vtable])?
                    }
                    (ty1, ty2) => unimplemented!("cast {ty1} as {ty2}"),
                }
            }

            bc::Rvalue::Closure { f, env } => {
                let env = self.eval_rvalue(&bc::Rvalue::Alloc {
                    kind: bc::AllocKind::Tuple,
                    loc: bc::AllocLoc::Heap,
                    args: bc::AllocArgs::Lit(env.clone()),
                })?;
                let func = Val::FuncRef(Some(self.rt.functions()[f].handle));
                self.rt.alloc_tuple(store!(self), vec![func, env])?
            }

            bc::Rvalue::Alloc {
                kind,
                args,
                loc: _loc,
            } => match args {
                bc::AllocArgs::Lit(ops) => {
                    let fields = ops
                        .iter()
                        .map(|el| self.eval_operand(el))
                        .collect::<Result<Vec<_>>>()?;

                    match kind {
                        bc::AllocKind::Tuple | bc::AllocKind::Struct => {
                            self.rt.alloc_tuple(store!(self), fields)?
                        }
                        bc::AllocKind::Array => self.rt.alloc_array(store!(self), fields)?,
                    }
                }
                bc::AllocArgs::ArrayCopy { value, count } => {
                    let value_val = self.eval_operand(value)?;
                    let count_val = self.eval_operand(count)?;
                    let count_int = from_val!(i32, self, count_val) as u32;
                    self.rt
                        .alloc_array_copy(store!(self), value_val, count_int)?
                }
            },

            bc::Rvalue::Call { f, args, .. } => {
                let f_val = self.eval_operand(f)?;

                let mut arg_vals = args
                    .iter()
                    .map(|arg| self.eval_operand(arg))
                    .collect::<Result<Vec<_>>>()?;

                let f_struct = self.struct_ref(f_val)?;

                let f_ref = f_struct.field(store!(self), 0)?;
                let f_ref = self.func_ref(f_ref)?;

                let f_env = f_struct.field(store!(self), 1)?;
                arg_vals.insert(0, f_env);

                self.rt
                    .call(store!(self).as_context_mut(), &f_ref, &arg_vals)?
            }

            bc::Rvalue::MethodCall {
                receiver,
                method,
                args,
            } => {
                let receiver_val = self.eval_operand(receiver)?;
                let receiver_struct = self.struct_ref(receiver_val)?;

                let self_ = receiver_struct.field(store!(self), 0)?;
                let vtable = receiver_struct.field(store!(self), 1)?;

                let mut arg_vals = args
                    .iter()
                    .map(|arg| self.eval_operand(arg))
                    .collect::<Result<Vec<_>>>()?;
                arg_vals.insert(0, self_);
                let dummy_env = self.rt.alloc_tuple(store!(self), vec![])?;
                arg_vals.insert(0, dummy_env);

                let vtable_struct = self.struct_ref(vtable)?;
                let method_val = vtable_struct.field(store!(self), *method)?;
                let method_ref = self.func_ref(method_val)?;

                self.rt
                    .call(store!(self).as_context_mut(), &method_ref, &arg_vals)?
            }
        })
    }

    fn eval(&mut self) -> Result<Val> {
        while self.step()? {}
        Ok(*self.ret_val.get().expect("ret_val missing"))
    }
}
