use std::fs;
use std::path::PathBuf;

use ta_benchmarks::catalogue_matrix::{
    read_optimization_evidence, read_raw_rows, render_report_with_comparison,
};

#[derive(Debug)]
struct Args {
    raw: PathBuf,
    report: PathBuf,
    baseline: Option<PathBuf>,
    optimization_evidence: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Indicator Catalogue report generation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let rows = read_raw_rows(&args.raw)?;
    let baseline_rows = args
        .baseline
        .as_deref()
        .map(read_raw_rows)
        .transpose()?
        .unwrap_or_default();
    let optimization_evidence = args
        .optimization_evidence
        .as_deref()
        .map(read_optimization_evidence)
        .transpose()?
        .unwrap_or_default();
    let report = render_report_with_comparison(&rows, &baseline_rows, &optimization_evidence)?;
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(&args.report, report)
        .map_err(|error| format!("write {}: {error}", args.report.display()))?;
    println!("human report: {}", args.report.display());
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut raw = None;
    let mut report = None;
    let mut baseline = None;
    let mut optimization_evidence = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--raw") => raw = arguments.next().map(PathBuf::from),
            Some("--report") => report = arguments.next().map(PathBuf::from),
            Some("--baseline") => baseline = arguments.next().map(PathBuf::from),
            Some("--optimization-evidence") => {
                optimization_evidence = arguments.next().map(PathBuf::from)
            }
            Some(other) => return Err(format!("unknown argument {other:?}")),
            None => return Err("arguments must be valid UTF-8".to_owned()),
        }
    }
    Ok(Args {
        raw: raw.ok_or_else(|| "--raw PATH is required".to_owned())?,
        report: report.ok_or_else(|| "--report PATH is required".to_owned())?,
        baseline,
        optimization_evidence,
    })
}
