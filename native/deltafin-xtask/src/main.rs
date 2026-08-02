use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("deltafin-xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let command = arguments.next().unwrap_or_else(|| "native-test".to_owned());
    if command == "list" {
        for spec in deltafin_native_build::NATIVE_TEST_SPECS {
            println!("{}\t{:?}\t{:?}", spec.name, spec.platform, spec.provider);
        }
        return Ok(());
    }
    if command != "native-test" {
        return Err(format!(
            "unknown command {command:?}; use `native-test [NAME]` or `list`"
        ));
    }
    let requested = arguments.next().unwrap_or_else(|| "all".to_owned());
    if let Some(extra) = arguments.next() {
        return Err(format!("unexpected argument {extra:?}"));
    }
    let repository = find_repository(&std::env::current_dir().map_err(|error| error.to_string())?)?;
    let reports = deltafin_native_build::run_native_tests(
        &repository,
        Path::new("target/deltafin-xtask/native-tests"),
        &requested,
    )
    .map_err(|error| error.to_string())?;
    for report in reports {
        for case in report.cases {
            print!("{}", case.stdout);
        }
        eprintln!(
            "[xtask] {}: PASS ({})",
            report.name,
            report.executable.display()
        );
    }
    Ok(())
}

fn find_repository(start: &Path) -> Result<PathBuf, String> {
    for candidate in start.ancestors() {
        if candidate.join("Cargo.toml").is_file() && candidate.join("native/provider_gate").is_dir()
        {
            return candidate
                .canonicalize()
                .map_err(|error| format!("resolve repository {}: {error}", candidate.display()));
        }
    }
    Err(format!(
        "could not find a Deltafin repository above {}",
        start.display()
    ))
}
