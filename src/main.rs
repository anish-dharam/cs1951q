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

    let (tcx, tir) = rice::typecheck(ast)?;
    log::debug!("TIR:\n{tir}");

    let mut bc = rice::lower(&tcx, tir);
    log::debug!("Initial BC:\n{bc}");
    if args.dump_ir {
        dump_ir(&bc, input, "bc-unopt")?;
    }

    rice::analyze(&bc)?;

    let opts = OptimizeOptions {
        opt_level: args.opt_level,
    };
    rice::optimize(&mut bc, opts);
    log::debug!("Optimized BC:\n{bc}");
    if args.dump_ir {
        dump_ir(&bc, input, "bc-opt")?;
    }

    match &args.command {
        None | Some(Command::Exec) => exec(args, tcx, bc),
        // add additional commands here
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
