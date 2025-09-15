//! The Rice standard library of functions.

use maplit::hashmap;
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    sync::LazyLock,
};
use wasmtime::{Memory, StoreContext};

use crate::{
    rt::{
        self,
        conversions::{NoContext, WithContext},
    },
    utils::{Symbol, sym},
};

fn alloc_string(s: &StoreContext<'_, ()>, mem: Option<&Memory>, addr: i32, len: i32) -> String {
    let mem = mem.expect("alloc_string not called from wasm");
    let mut buffer = vec![0_u8; len as usize];
    mem.read(s, addr as usize, &mut buffer)
        .expect("failed to read memory");
    String::from_utf8(buffer).expect("string is not utf-8")
}

fn concat_string(s1: &str, s2: &str) -> String {
    s1.to_string() + s2
}

fn print(s: &str) {
    print!("{s}");
}

fn println(s: &str) {
    println!("{s}");
}

fn putb(n: i32) {
    io::stdout().write_all(&[u8::try_from(n).unwrap()]).unwrap();
}

fn int_to_string(n: i32) -> String {
    n.to_string()
}

fn float_to_string(f: f32) -> String {
    f.to_string()
}

// assert is special because line_info is generated in the parser, not provided by the user
fn assert(p: bool, pred_text: &str, line_num: i32) -> Result<(), String> {
    if p {
        Ok(())
    } else {
        Err(format!("assertion `{pred_text}` failed on line {line_num}"))
    }
}

fn read_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| e.to_string())
}

fn nth_char(s: &str, i: i32) -> String {
    let i = i as usize;
    s[i..i + 1].to_string()
}

fn log(x: f32) -> f32 {
    x.ln()
}

fn exp(x: f32) -> f32 {
    x.exp()
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub fn stdlib() -> &'static HashMap<Symbol, Box<dyn rt::WasmFunc>> {
    macro_rules! stdlib {
        ($(($func:ident, $tys:ty)),*) => {
            hashmap! {
                $(sym(stringify!($func)) => Box::new(rt::conversions::TypedFunc::<_, $tys>::new($func)) as Box<dyn rt::WasmFunc>),*
            }
        }
    }

    static STDLIB: LazyLock<HashMap<Symbol, Box<dyn rt::WasmFunc>>> = LazyLock::new(|| {
        // To register a new stdlib function, put an entry in this list.
        // The second argument is a set of types describing the function. The order is a bit confusing.
        // Within the second argument:
        // - The first type should be either `WithContext` or `NoContext`.
        //   WithContext means that the first two arguments must be &StoreContext<'_, ()> and Option<&Memory>.
        // - The second type is the *return* type of the function.
        // - The remaining types are the *parameter* types, in order from left to right.
        //   This should be the type for which `Wasmable` is implemented. For example, if a function takes `&str`,
        //   then the parameter type is `String`.
        stdlib!(
            (alloc_string, (WithContext, String, i32, i32)),
            (concat_string, (NoContext, String, String, String)),
            (print, (NoContext, (), String)),
            (println, (NoContext, (), String)),
            (putb, (NoContext, (), i32)),
            (int_to_string, (NoContext, String, i32)),
            (float_to_string, (NoContext, String, f32)),
            (assert, (NoContext, Result<(), String>, bool, String, i32)),
            (read_file, (NoContext, Result<String, String>, String)),
            (nth_char, (NoContext, String, String, i32)),
            (log, (NoContext, f32, f32)),
            (exp, (NoContext, f32, f32)),
            (sigmoid, (NoContext, f32, f32))
        )
    });

    &STDLIB
}
