use egg::{define_language, Id};

define_language! {
    pub enum RiceLang {
        Num(i32),
        Var(String),
        Bool(bool),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),       
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Rem([Id; 2]),

        "=" = Eq([Id; 2]),
        "<" = Lt([Id; 2]),
        ">" = Gt([Id; 2]),

        "and" = And([Id; 2]),
        "or"  = Or([Id; 2]),
        "not" = Not([Id; 1]),
    }
}
