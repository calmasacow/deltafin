//! Executable-level smoke tests for the complete public command help matrix.
//!
//! Parsing tests prove the enum mapping, while this test proves the installed
//! binary returns successful help before runtime/model preflight for every
//! supported subcommand.

use std::process::Command;

const SUBCOMMANDS: &[&str] = &[
    "run",
    "serve",
    "benchmark",
    "doctor",
    "upgrade",
    "setup",
    "setup-dspark",
    "setup-k3",
    "setup-qwen",
    "fetch-weights",
    "warm-expert-cache",
    "convert-spine-int8",
    "convert-experts-scale4",
    "pack-spine",
];

#[test]
fn every_public_subcommand_prints_successful_help_from_the_compiled_binary() {
    let executable = env!("CARGO_BIN_EXE_deltafin");
    for subcommand in SUBCOMMANDS {
        for help in ["-h", "--help"] {
            let output = Command::new(executable)
                .args([subcommand, help])
                .output()
                .unwrap_or_else(|error| panic!("execute deltafin {subcommand} {help}: {error}"));
            assert!(
                output.status.success(),
                "deltafin {subcommand} {help} failed: {}",
                String::from_utf8_lossy(&output.stderr),
            );
            let stdout = String::from_utf8(output.stdout)
                .unwrap_or_else(|error| panic!("help output was not UTF-8: {error}"));
            assert!(stdout.starts_with("Deltafin native runtime\n"));
            assert!(stdout.contains("Usage:\n"));
            let usage = if *subcommand == "run" {
                "deltafin [run]".to_owned()
            } else {
                format!("deltafin {subcommand}")
            };
            assert!(stdout.contains(&usage));
            assert!(output.stderr.is_empty());
        }
    }
}
