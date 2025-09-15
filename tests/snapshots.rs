use anyhow::Result;
use serde::Serialize;
use std::{fs, path::Path, process::Command};
use walkdir::WalkDir;

#[derive(Serialize)]
struct CompilerOutput {
    success: bool,
    stdout: String,
    stderr: String,
}

fn compile(path: &Path, args: &str) -> Result<CompilerOutput> {
    let mut cmd = Command::new("./target/debug/rice");
    cmd.arg(path);
    cmd.args(shlex::split(args).unwrap());
    let output = cmd.output()?;
    Ok(CompilerOutput {
        success: output.status.success(),
        stdout: String::from_utf8(output.stdout)?,
        stderr: String::from_utf8(output.stderr)?,
    })
}

#[test]
fn snapshots() -> Result<()> {
    let root = Path::new("tests/programs").canonicalize()?;
    for entry in WalkDir::new(&root) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let ext = path.extension().unwrap().to_str().unwrap();
        if ext != "rice" {
            continue;
        }

        let contents = fs::read_to_string(path)?;
        let first_line = contents.lines().next().unwrap();
        let args = match first_line.strip_prefix("//") {
            Some(args) => args,
            None => "",
        };

        let output = compile(path, args)?;
        let name = path.file_name().unwrap().to_str().unwrap();
        let snapshot_path = path.parent().unwrap();
        insta::with_settings!({
          snapshot_path => snapshot_path,
          prepend_module_to_snapshot => false,
        }, {
          insta::assert_toml_snapshot!(name, output);
        });
    }
    Ok(())
}
