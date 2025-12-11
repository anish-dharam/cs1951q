# The Rice programming language

## Project Structure

This repository contains the Rice compiler implementation, including an e-graph-based term rewriting optimization pass.

### Source Files for E-Graph Rewriting

The following files contain the implementation of the e-graph rewriting optimization system:

#### Core Implementation

- **`src/tir/rewrite.rs`** - Main implementation file containing:

  - The `TirLang` language definition for e-graph representation (using egg's `define_language!` macro)
  - All rewriting rules (arithmetic, boolean, bitwise, structural, etc.)
  - **Language conversion system**:
    - `expr_to_egg()` - Converts TIR `Expr` (with `ExprKind` variants) to egg's `TirLang` representation
    - `rec_expr_to_expr()` - Converts optimized `RecExpr<TirLang>` back to TIR `Expr` after rewriting
    - Handles all TIR constructs: constants, variables, binary operations, control flow, function calls, closures, arrays, tuples, structs, etc.
  - **Type and span preservation**:
    - `TypeSpanMap` - Maps e-graph node indices to their original type and source span information
    - Preserves type information through the rewriting process (types are not part of the e-graph language)
    - `rebuild_type_map_from_recexpr()` - Reconstructs type mapping after optimization using canonical e-graph IDs
  - Cost models (`TirCost`, `TirSmartCost`, `AnyCostFn`)
  - Ablation configuration system for testing different rule subsets
  - The main rewriting pass that applies e-graph optimization

- **`src/tir/benchmark.rs`** - Benchmarking infrastructure for evaluating rewriting rules:
  - Functions to benchmark rewriting performance
  - Sweep functions for testing different iteration limits
  - Ablation study support for comparing rule subsets

#### Integration Points

- **`src/tir/mod.rs`** - Exports the `rewrite_terms` function and thread-local configuration variables
- **`src/main.rs`** - CLI integration with the `rewrite` subcommand (lines 23-32, 81-92)
- **`src/lib.rs`** - Public API export of `rewrite_terms` function

#### Related Files

- **`src/tir/types.rs`** - Type definitions used by the rewriting system
- **`src/egraph/`** - Additional e-graph infrastructure (if used)

### Language Conversion

The rewriting system performs bidirectional conversion between two representations:

1. **TIR `Expr`** - The compiler's typed intermediate representation with:

   - `ExprKind` enum variants for all language constructs
   - Type information attached to each expression node
   - Source span information for error reporting

2. **Egg `TirLang`** - The e-graph language representation (defined via `define_language!` macro) with:
   - Flat node structure suitable for e-graph rewriting
   - No type information (types are tracked separately in `TypeSpanMap`)
   - Support for pattern matching in rewrite rules

The conversion process:

- **Forward (TIR → E-graph)**: `expr_to_egg()` recursively converts `Expr` nodes to `TirLang` nodes, building the e-graph and populating `TypeSpanMap`
- **Rewriting**: Egg applies rewrite rules to find equivalent expressions
- **Extraction**: Cost-based extractor selects the best equivalent expression
- **Backward (E-graph → TIR)**: `rec_expr_to_expr()` converts the optimized `RecExpr<TirLang>` back to `Expr`, using the rebuilt type map to restore type information

### Key Features

The e-graph rewriting system includes:

1. **Comprehensive Rewriting Rules** (defined in `src/tir/rewrite.rs::make_rules`):

   - Arithmetic optimizations (commutativity, associativity, identity, zero rules)
   - Bitwise operations (shift optimizations, bitwise identities)
   - Boolean algebra (idempotence, absorption, distributivity)
   - Structural optimizations (tuple/struct projection, array indexing)
   - Constant folding and conditional simplification

2. **Configurable Cost Models**:

   - `ast` - AST size-based cost model
   - `tir` - TIR-specific cost model with operation complexity weights
   - `smart` - Simplified cost model focusing on expensive operations

3. **Ablation Study Support**:

   - Configurable rule subsets for evaluating individual rule contributions
   - Benchmarking infrastructure for performance analysis

4. **Type and Span Preservation**:

   - `TypeSpanMap` structure preserves type and source location information through rewriting
   - Since `TirLang` (the e-graph language) doesn't include type information, types must be tracked separately
   - During conversion from `Expr` to `TirLang`, each e-graph node ID is mapped to its original type and span
   - After optimization, the type map is rebuilt by looking up canonical e-graph IDs to recover type information
   - This ensures the optimized code maintains correct types for downstream compilation phases

### Usage

To use the e-graph rewriting optimization, run the compiler with the `rewrite` subcommand:

```bash
cargo run -- sample.rice rewrite --iterations 30 --time-limit 1 --model smart
```

Options:

- `--iterations`: Maximum number of rewriting iterations (default: 30)
- `--time-limit`: Time limit in seconds (default: 1)
- `--model`: Cost model to use - `ast`, `tir`, or `smart` (default: `ast`)

For benchmarking, set the `RUN_BENCH` environment variable:

```bash
RUN_BENCH=1 cargo run -- sample.rice rewrite
```

This will generate CSV files with benchmark results for different rule configurations.

### External Documents

<!-- If you have pull requests, papers, or other external documents related to this project, list them here:
- [Pull Request #X](link-to-pr) - Description of changes
- [Project Report](link-to-report) - Description
-->

## Build the compiler

For testing:

```
cargo build
```

For profiling:

```
cargo build --profile profile
```

For deployment:

```
cargo build --release
```

## Run the compiler

After building, run:

```
./target/debug/rice sample.rice
```

Replace `debug` with either `profile` or `release` depending on the build. Add `-h` to see all options. To build and run in one command, run:

```
cargo run -- sample.rice
```

## Test the compiler

To run the normal tests in the compiler, run:

```
cargo test
```

To run the snapshot tests, install [cargo-insta]. Then run:

```
cargo insta test
```

## Debug the compiler

The compiler uses [env_logger] to print logs. To get all logs at the debug level and higher, run with `RUST_LOG="rice=debug"`. For example:

```
RUST_LOG="rice=debug" cargo run -- sample.rice
```

You can reduce the noisiness by either increasing the log level like `RUST_LOG="rice=info"` or by narrowing the module scope like `RUST_LOG="rice::rt=debug"`.

## Profile the compiler

Install [samply], then run

```
cargo build --profile profile
samply record ./target/profile/rice perf/mandelbrot.rice
```

[env_logger]: https://docs.rs/env_logger/latest/env_logger/
[cargo-insta]: https://crates.io/crates/cargo-insta
[samply]: https://github.com/mstange/samply

## Reading the codebase

You can generate documentation for each module by running:

```
cargo doc --document-private-items --open
```
