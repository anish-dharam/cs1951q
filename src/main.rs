use clap::{Parser, Subcommand};
use miette::{Result, miette};
use rice::{
    ast::Input,
    bc::{OptLevel, OptimizeOptions, types as bc},
    rt::{Runtime, RuntimeOptions},
    tir::Tcx,
    utils::Symbol,
};

use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufWriter},
    path::PathBuf,
};

#[derive(Subcommand)]
enum Command {
    /// Execute the source file.
    Exec,
    // E-graph optimizations
    Rewrite {
        #[arg(long, default_value_t = 30)]
        iterations: usize,

        #[arg(long, default_value_t = 1)]
        time_limit: u64,

        #[arg(long, default_value = "ast")]
        model: String,
    },
    /// Run symbolic execution on functions annotated with #[symex].
    Symex {
        /// Maximum number of steps to execute per path. Paths exceeding this limit are discarded.
        #[arg(long)]
        max_steps: Option<u64>,
    },
}

#[derive(Parser)]
struct Args {
    /// Path to Rice source file to execute.
    file: PathBuf,

    /// Set the level of compiler optimizations. -O0 is disabled, -O1 is enabled.
    #[arg(short = 'O', long, default_value_t = OptLevel::NoOpt)]
    opt_level: OptLevel,

    /// Disable the JIT from running.
    #[arg(long)]
    disable_jit: bool,

    /// Dump the intermediate representations to JSON files.
    #[arg(long)]
    dump_ir: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let input = rice::read(&args.file)?;

    let result = run(&args, &input);
    result.map_err(move |e| e.with_source_code(input.into_named_source()))
}

fn run(args: &Args, input: &Input) -> Result<()> {
    let ast = rice::parse(input)?;
    log::debug!("AST:\n{}", ast.prog);

    let (original_tcx, original_tir) = rice::typecheck(ast)?;
    log::debug!("TIR:\n{original_tir}");

    let (tcx, tir) = match &args.command {
        Some(Command::Rewrite { iterations, time_limit, model }) => {
            println!("E-graph optimization");
            rice::tir::REWRITE_ITER_LIMIT.with(|v| *v.borrow_mut() = *iterations);
            rice::tir::REWRITE_TIME_LIMIT.with(|v| *v.borrow_mut() = *time_limit);
            rice::tir::REWRITE_COST_MODEL.with(|v| *v.borrow_mut() = model.clone());
            let (new_tcx, new_tir) = rice::rewrite_terms(original_tcx, original_tir);
            log::debug!("TIR with e-graph rewrites:\n{new_tir}");
            (new_tcx, new_tir)
        }
        _ => (original_tcx, original_tir),
    };

    let mut bc = rice::lower(&tcx, tir);
    log::debug!("Initial BC:\n{bc}");
    if args.dump_ir {
        dump_ir(&bc, input, "bc-unopt")?;
    }

    rice::analyze(&bc)?;

    // Skip optimizations in symex mode
    let opt_level = match &args.command {
        Some(Command::Symex { .. }) => OptLevel::NoOpt,
        _ => args.opt_level,
    };
    let opts = OptimizeOptions { opt_level };
    rice::optimize(&mut bc, opts);
    log::debug!("Optimized BC:\n{bc}");
    if args.dump_ir {
        dump_ir(&bc, input, "bc-opt")?;
    }

    match &args.command {
        None | Some(Command::Exec) | Some(Command::Rewrite { .. }) => exec(args, tcx, bc),
        Some(Command::Symex { max_steps }) => symex(tcx, bc, *max_steps),
    }
}

fn exec(args: &Args, tcx: Tcx, bc: bc::Program) -> Result<()> {
    fn exec(args: &Args, tcx: Tcx, bc: bc::Program) -> anyhow::Result<()> {
        let opts = RuntimeOptions {
            disable_jit: args.disable_jit,
        };
        let rt = Runtime::new(tcx, opts)?;
        rt.register(bc)?;
        let main_func = rt.function(Symbol::main())?;
        rt.call_toplevel(&main_func, vec![])?;
        Ok(())
    }
    exec(args, tcx, bc).map_err(|e| miette!("{e:?}"))
}

fn symex(tcx: Tcx, bc: bc::Program, max_steps: Option<u64>) -> Result<()> {
    fn symex(tcx: Tcx, bc: bc::Program, max_steps: Option<u64>) -> anyhow::Result<()> {
        use rice::rt::symex;
        let opts = symex::SymexOptions { max_steps };
        symex::run(tcx, &bc, opts)?;
        Ok(())
    }
    symex(tcx, bc, max_steps).map_err(|e| miette!("{e:?}"))
}

fn dump_ir<T: Serialize>(t: &T, input: &Input, ext: &str) -> Result<()> {
    fn dump_ir<T: Serialize>(t: &T, input: &Input, ext: &str) -> io::Result<()> {
        let ir_path = format!(
            "{}.{ext}.json",
            input.path().file_stem().unwrap().to_string_lossy()
        );
        let writer = BufWriter::new(File::create(ir_path)?);
        serde_json::to_writer_pretty(writer, t)?;
        Ok(())
    }
    dump_ir(t, input, ext).map_err(|e| miette!("{e:?}"))
}
