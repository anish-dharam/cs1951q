//! Miscellaneous utilities used across the codebase.

use std::{fmt, str::FromStr, sync::LazyLock};

use petgraph::visit::{DfsPostOrder, GraphRef, IntoNeighbors, Visitable};

pub fn write_comma_separated<T: fmt::Display>(
    f: &mut fmt::Formatter,
    items: impl IntoIterator<Item = T>,
) -> fmt::Result {
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", item)?;
    }
    Ok(())
}

pub fn write_newline_separated<T: fmt::Display>(
    f: &mut fmt::Formatter,
    items: impl IntoIterator<Item = T>,
) -> fmt::Result {
    for (i, item) in items.into_iter().enumerate() {
        if i > 0 {
            write!(f, "\n")?;
        }
        write!(f, "{}", item)?;
    }
    Ok(())
}

pub fn indent(s: impl AsRef<str>) -> String {
    textwrap::indent(s.as_ref(), "  ")
}

#[macro_export]
/// Generates a wrapper type for an [`Intern`][internment::Intern].
macro_rules! interned {
    ($name:ident, $data:ty) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize)]
        pub struct $name(internment::Intern<$data>);

        impl std::ops::Deref for $name {
            type Target = $data;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

interned!(Symbol, String);

impl Symbol {
    pub fn new(s: impl AsRef<str>) -> Self {
        Symbol(internment::Intern::from_ref(s.as_ref()))
    }

    pub fn main() -> Self {
        static MAIN: LazyLock<Symbol> = LazyLock::new(|| sym("main"));
        *MAIN
    }
}

pub fn sym(s: impl AsRef<str>) -> Symbol {
    Symbol::new(s)
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for Symbol {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Symbol::new(s))
    }
}

pub fn reverse_post_order<N, G>(g: G, entry: G::NodeId) -> Vec<N>
where
    N: Copy,
    G: GraphRef + Visitable<NodeId = N> + IntoNeighbors<NodeId = N>,
    G::NodeId: PartialEq,
{
    let mut postorder = Vec::new();
    let mut dfs = DfsPostOrder::new(g, entry);
    while let Some(node) = dfs.next(g) {
        postorder.push(node);
    }
    postorder.reverse();
    postorder
}
