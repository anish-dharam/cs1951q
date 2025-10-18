//! Abstract syntax tree and parser.
//!
//! The parser is implemented using the [LALRPOP](https://lalrpop.github.io/lalrpop/) parser generator.

use std::{
    cell::Cell,
    fs,
    path::{Path, PathBuf},
};

use crate::grammar::ProgramParser;
use itertools::Itertools;
use lalrpop_util::{ParseError as LalrpopError, lexer::Token};
use miette::{Diagnostic, NamedSource, Result, SourceCode, SourceSpan, miette};
use thiserror::Error;

mod print;
pub mod types;

#[derive(Diagnostic, Error, Debug)]
#[error("parse error")]
struct ParseError {
    explanation: String,

    #[label("{explanation}")]
    span: SourceSpan,
}

/// Converts [LALRPOP errors][LalrpopError] into the [`ParseError`] format.
fn gen_parse_error(err: LalrpopError<usize, Token<'_>, &'static str>) -> ParseError {
    let (span, explanation) = match err {
        LalrpopError::UnrecognizedToken {
            token: (start, _, end),
            expected,
        } => (
            SourceSpan::from((start, end - start)),
            format!(
                "unrecognized token, expected: {}",
                expected.iter().join(" or ")
            ),
        ),
        LalrpopError::InvalidToken { location } => {
            (SourceSpan::from((location, 0)), "invalid token".to_string())
        }
        LalrpopError::ExtraToken {
            token: (start, _, end),
        } => (
            SourceSpan::from((start, end - start)),
            "extra token".to_string(),
        ),
        LalrpopError::UnrecognizedEof { location, expected } => (
            SourceSpan::from((location, 0)),
            format!(
                "unrecognized EOF, expected: {}",
                expected.iter().join(" or ")
            ),
        ),
        LalrpopError::User { .. } => unreachable!(),
    };
    ParseError { explanation, span }
}

/// The abstract syntax tree parsed from a source file.
pub struct Ast {
    pub prog: self::types::Program,
    pub num_holes: usize,
}

/// Parse a source file into an AST.
pub fn parse(input: &Input) -> Result<Ast> {
    let holes = Cell::new(0);
    let prog = ProgramParser::new()
        .parse(&holes, input, &input.contents)
        .map_err(gen_parse_error)?;
    Ok(Ast { prog, num_holes: holes.get() })
}

/// A source file provided by the user.
#[derive(Clone)]
pub struct Input {
    contents: String,
    path: PathBuf,
}

impl Input {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// View the source file as a Miette [`SourceCode`] for getting eg line info.
    pub fn as_source(&self) -> impl SourceCode {
        self.contents.as_str()
    }

    /// Transform into a Miette [`NamedSource`] for diagnostic reporting.
    pub fn into_named_source(self) -> NamedSource<String> {
        let file_name = self.path.file_name().unwrap().to_string_lossy().to_string();
        NamedSource::new(file_name, self.contents)
    }
}

/// Read a source file from disk.
pub fn read(path: &Path) -> Result<Input> {
    let contents = fs::read_to_string(path).map_err(|e| miette!("{}", e.to_string()))?;
    Ok(Input {
        contents,
        path: path.to_path_buf(),
    })
}
