# The Rice programming language

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