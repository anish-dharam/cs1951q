//! Traits and implementations to convert between Wasm values and Rust values.

use std::marker::PhantomData;

use anyhow::{Context, Result, bail};
use wasmtime::{
    AsContext, AsContextMut, Caller, ExternRef, Memory, StoreContext, StoreContextMut, Val,
};

use super::{Runtime, WasmFunc};
use crate::bc::types as bc;

/// A type convertable to/from the wasmtime runtime.
///
/// Converting a Wasm type to a Rust type is separated into two stages,
/// [`Wasmable::reduce_val`] which has mutable access to the store, and
/// [`Wasmable::from_reduced`] which only has immutable access. This is necessary
/// right now because reading struct fields requires mutable access for some reason.
pub trait Wasmable: Send + Sync + 'static {
    /// The intermediate result of Wasm->Rust conversion.
    type Reduced;

    /// The final result of Wasm->Result conversion.
    ///
    /// Not necessarily `Self` for cases like `String` -> `&str`.
    type Target<'a>;

    /// The Rice language type encoding for `Self`.
    fn ty() -> bc::Type;

    fn reduce_val(val: Val, store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced>;

    fn from_reduced<'a>(
        reduced: Self::Reduced,
        store: &'a StoreContext<'_, ()>,
    ) -> Result<Self::Target<'a>>;

    fn to_val(&self, rt: &Runtime, store: &mut StoreContextMut<'_, ()>) -> Result<Val>;
}

impl Wasmable for i32 {
    type Target<'a> = i32;
    type Reduced = Val;

    fn ty() -> bc::Type {
        bc::Type::int()
    }

    fn reduce_val(val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        Ok(val)
    }

    fn from_reduced(val: Val, _store: &StoreContext<'_, ()>) -> Result<i32> {
        val.i32().context("value should be i32")
    }

    fn to_val(&self, _rt: &Runtime, _store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        Ok(Val::I32(*self))
    }
}

impl Wasmable for f32 {
    type Target<'a> = f32;
    type Reduced = Val;

    fn ty() -> bc::Type {
        bc::Type::float()
    }

    fn reduce_val(val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        Ok(val)
    }

    fn from_reduced(val: Val, _store: &StoreContext<'_, ()>) -> Result<f32> {
        val.f32().context("value should be f32")
    }

    fn to_val(&self, _rt: &Runtime, _store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        Ok(Val::F32(self.to_bits()))
    }
}

impl Wasmable for bool {
    type Target<'a> = bool;
    type Reduced = Val;

    fn ty() -> bc::Type {
        bc::Type::bool()
    }

    fn reduce_val(val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        Ok(val)
    }

    fn from_reduced(val: Val, _store: &StoreContext<'_, ()>) -> Result<bool> {
        let n = val.i32().context("value should be i32")?;
        match n {
            0 => Ok(false),
            1 => Ok(true),
            _ => bail!("bool should be 0 or 1"),
        }
    }

    fn to_val(&self, _rt: &Runtime, _store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        let n = if *self { 1 } else { 0 };
        Ok(Val::I32(n))
    }
}

impl Wasmable for String {
    type Target<'a> = &'a str;
    type Reduced = Val;

    fn ty() -> bc::Type {
        bc::Type::string()
    }

    fn reduce_val(val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        Ok(val)
    }

    fn from_reduced<'a>(val: Val, store: &'a StoreContext<'_, ()>) -> Result<Self::Target<'a>> {
        let extern_ref = val
            .extern_ref()
            .context("string is not extern ref")?
            .context("string is null")?;
        let any = extern_ref
            .data(store)?
            .context("extern ref is anyref wrapper")?;
        let string = any.downcast_ref::<String>().context("any is not string")?;
        Ok(string.as_str())
    }

    fn to_val(&self, _rt: &Runtime, store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        let extern_ref = ExternRef::new(store, self.clone())?;
        Ok(Val::from(extern_ref))
    }
}

impl Wasmable for () {
    type Target<'a> = ();
    type Reduced = ();

    fn ty() -> bc::Type {
        bc::Type::unit()
    }

    fn from_reduced<'a>(
        _reduced: Self::Reduced,
        _store: &'a StoreContext<'_, ()>,
    ) -> Result<Self::Target<'a>> {
        Ok(())
    }

    fn reduce_val(_val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        Ok(())
    }

    fn to_val(&self, rt: &Runtime, store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        rt.alloc_tuple(store, vec![])
    }
}

impl<T: Wasmable> Wasmable for Result<T, String> {
    type Target<'a> = ();
    type Reduced = Val;

    fn ty() -> bc::Type {
        T::ty()
    }

    fn from_reduced<'a>(
        _reduced: Self::Reduced,
        _store: &'a StoreContext<'_, ()>,
    ) -> Result<Self::Target<'a>> {
        unreachable!()
    }

    fn reduce_val(_val: Val, _store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        unreachable!()
    }

    fn to_val(&self, rt: &Runtime, store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        match self {
            Ok(t) => t.to_val(rt, store),
            Err(e) => {
                *rt.panic_mut() = Some(e.clone());
                Ok(*rt.unit.get().unwrap())
            }
        }
    }
}

macro_rules! gen_wasmable_tuple_impl {
  ($($t:ident $n:tt),*) => {
    impl<$($t),*> Wasmable for ($($t),*,) where $($t: Wasmable),* {
      type Target<'a> = ($($t::Target<'a>),*,);
      type Reduced = ($($t::Reduced),*,);

      fn ty() -> bc::Type {
        bc::Type::tuple(vec![$($t::ty()),*])
      }

      fn reduce_val(val: Val, store: &mut StoreContextMut<'_, ()>) -> Result<Self::Reduced> {
        let any_ref = val
          .any_ref()
          .context("tuple is not an anyref")?
          .context("tuple is null")?;
        let struct_ref = any_ref
          .as_struct(&store)?
          .context("tuple is not a struct")?;
        Ok((
          $($t::reduce_val(struct_ref.field(&mut *store, $n)?, &mut *store)?),*,
        ))
      }

      fn from_reduced<'a>(
        tup: Self::Reduced,
        store: &'a StoreContext<'_, ()>,
      ) -> Result<Self::Target<'a>> {
        Ok((
          $($t::from_reduced(tup.$n, store)?),*,
        ))
      }

      fn to_val(&self, rt: &Runtime, store: &mut StoreContextMut<'_, ()>) -> Result<Val> {
        let fields = vec![
          $(self.$n.to_val(rt, &mut *store)?),*
        ];
        rt.alloc_tuple(store, fields)
      }
    }
  };
}

gen_wasmable_tuple_impl!(T 0);
gen_wasmable_tuple_impl!(T 0, S 1);

pub struct TypedFunc<F, P> {
    f: F,
    _p: PhantomData<P>,
}

impl<F, P> TypedFunc<F, P> {
    pub fn new(f: F) -> Self {
        TypedFunc { f, _p: PhantomData }
    }
}

pub struct WithContext;
pub struct NoContext;

macro_rules! gen_wasmfunc_impls {
    ($($t:ident $n:tt),*) => {
        #[allow(unused_parens)]
        impl<$($t,)* R, F> WasmFunc for TypedFunc<F, (NoContext, R, $($t),*)>
        where
            $($t: Wasmable,)*
            R: Wasmable,
            F: (Fn($($t::Target<'_>),*) -> R) + Send + Sync + 'static,
        {
            fn src_type(&self) -> bc::Type {
                bc::Type::func(vec![$($t::ty()),*], R::ty())
            }

            fn rt_type(&self) -> bc::Type {
                bc::Type::func(vec![bc::Type::unit(), $($t::ty()),*], R::ty())
            }

            #[allow(unused)]
            fn call(
                &self,
                rt: &Runtime,
                mut caller: Caller<'_, ()>,
                args: &[Val],
            ) -> Result<Val> {
                let reduced = ($($t::reduce_val(args[$n+1], &mut caller.as_context_mut())?,)*);
                let result = (self.f)($($t::from_reduced(reduced.$n, &caller.as_context())?),*);
                result.to_val(rt, &mut caller.as_context_mut())
            }
        }

        #[allow(unused_parens)]
        impl<$($t,)* R, F> WasmFunc for TypedFunc<F, (WithContext, R, $($t),*)>
        where
            $($t: Wasmable,)*
            R: Wasmable,
            F: (Fn(&StoreContext<'_, ()>, Option<&Memory>, $($t::Target<'_>),*) -> R) + Send + Sync + 'static,
        {
            fn src_type(&self) -> bc::Type {
                bc::Type::func(vec![$($t::ty()),*], R::ty())
            }

            fn rt_type(&self) -> bc::Type {
                bc::Type::func(vec![bc::Type::unit(), $($t::ty()),*], R::ty())
            }

            #[allow(unused)]
            fn call(
                &self,
                rt: &Runtime,
                mut caller: Caller<'_, ()>,
                args: &[Val],
            ) -> Result<Val> {
                let reduced = ($($t::reduce_val(args[$n+1], &mut caller.as_context_mut())?,)*);
                let memory = caller.get_export("memory").and_then(|ext| ext.into_memory());
                let context = caller.as_context();
                let result = (self.f)(&context, memory.as_ref(), $($t::from_reduced(reduced.$n, &context)?),*);
                result.to_val(rt, &mut caller.as_context_mut())
            }
        }
    }
}

gen_wasmfunc_impls!();
gen_wasmfunc_impls!(T 0);
gen_wasmfunc_impls!(T 0, S 1);
gen_wasmfunc_impls!(T 0, S 1, U 2);
