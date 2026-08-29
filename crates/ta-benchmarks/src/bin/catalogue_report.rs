use std::fs;
use std::path::PathBuf;

use ta_benchmarks::catalogue_evidence::read_report_evidence;
use ta_benchmarks::catalogue_report::{
    read_criterion_diagnostics, read_cycle_regression, read_diagnostic_evidence,
    read_platform_qualification, render_validated_report_with_comparison,
};

#[derive(Debug)]
struct Args {
    raw: PathBuf,
    report: PathBuf,
    baseline: PathBuf,
    diagnostic_evidence: PathBuf,
    criterion_diagnostics: PathBuf,
    cycle_regression: PathBuf,
    platform_qualifications: Vec<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Indicator Catalogue report generation failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let evidence = read_report_evidence(&args.raw).map_err(|error| error.to_string())?;
    let baseline = read_report_evidence(&args.baseline).map_err(|error| error.to_string())?;
    let diagnostic_evidence = read_diagnostic_evidence(&args.diagnostic_evidence)?;
    let criterion_diagnostics = read_criterion_diagnostics(&args.criterion_diagnostics)?;
    let cycle_regression = read_cycle_regression(&args.cycle_regression)?;
    let platform_qualifications = args
        .platform_qualifications
        .iter()
        .map(|path| read_platform_qualification(path))
        .collect::<Result<Vec<_>, _>>()?;
    let report = render_validated_report_with_comparison(
        &evidence,
        &baseline,
        &diagnostic_evidence,
        &criterion_diagnostics,
        &cycle_regression,
        &platform_qualifications,
    )?;
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
    let mut diagnostic_evidence = None;
    let mut criterion_diagnostics = None;
    let mut cycle_regression = None;
    let mut platform_qualifications = Vec::new();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--raw") => raw = arguments.next().map(PathBuf::from),
            Some("--report") => report = arguments.next().map(PathBuf::from),
            Some("--baseline") => baseline = arguments.next().map(PathBuf::from),
            Some("--diagnostic-evidence") => {
                diagnostic_evidence = arguments.next().map(PathBuf::from)
            }
            Some("--criterion-diagnostics") => {
                criterion_diagnostics = arguments.next().map(PathBuf::from)
            }
            Some("--cycle-regression") => cycle_regression = arguments.next().map(PathBuf::from),
            Some("--platform-qualification") => {
                platform_qualifications.push(
                    arguments
                        .next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--platform-qualification requires PATH".to_owned())?,
                );
            }
            Some(other) => return Err(format!("unknown argument {other:?}")),
            None => return Err("arguments must be valid UTF-8".to_owned()),
        }
    }
    Ok(Args {
        raw: raw.ok_or_else(|| "--raw PATH is required".to_owned())?,
        report: report.ok_or_else(|| "--report PATH is required".to_owned())?,
        baseline: baseline.ok_or_else(|| "--baseline PATH is required".to_owned())?,
        diagnostic_evidence: diagnostic_evidence
            .ok_or_else(|| "--diagnostic-evidence PATH is required".to_owned())?,
        criterion_diagnostics: criterion_diagnostics
            .ok_or_else(|| "--criterion-diagnostics PATH is required".to_owned())?,
        cycle_regression: cycle_regression
            .ok_or_else(|| "--cycle-regression PATH is required".to_owned())?,
        platform_qualifications,
    })
}
