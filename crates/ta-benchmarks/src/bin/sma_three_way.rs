use std::ffi::CStr;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::os::raw::{c_char, c_double, c_int};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use ta_benchmarks::sma_three_way::{
    input_checksum, read_raw_rows, render_report, series_fixture, timing_stats, validate_outputs,
    write_raw_rows, BenchmarkRow, INPUT_LENGTHS, PERIOD,
};
use ta_core::overlap::SMAConfig;
use ta_core::{IndicatorConfig, OutputRange};

const TA_SUCCESS: c_int = 0;
const TALIB_VERSION: &str = "0.6.4";
const TALIB_REVISION: &str = "43f9d5042ecc4bd367941846494ad907bf20ea50";
const NUMPY_VERSION: &str = "2.2.3";
const DEFAULT_SAMPLES: usize = 50;
const DEFAULT_WARMUP_MS: u64 = 250;
const DEFAULT_SAMPLE_MS: u64 = 10;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn TA_Initialize() -> c_int;
    fn TA_Shutdown() -> c_int;
    fn TA_GetVersionString() -> *const c_char;
    fn TA_SMA_Lookback(opt_in_time_period: c_int) -> c_int;
    fn TA_SMA(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        opt_in_time_period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
}

#[derive(Debug)]
struct Args {
    python: PathBuf,
    output_dir: PathBuf,
    samples: usize,
    warmup_ms: u64,
    sample_ms: u64,
}

#[derive(Debug)]
struct PythonSemantic {
    python_version: String,
    binding_version: String,
    ta_lib_version: String,
    numpy_version: String,
    range: (usize, usize),
    values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PythonProvenance {
    python_version: String,
    binding_version: String,
    ta_lib_version: String,
    numpy_version: String,
}

#[derive(Debug)]
struct PythonTiming {
    warmup_iterations: u64,
    iterations_per_sample: u64,
    samples_ns: Vec<f64>,
}

#[derive(Debug)]
struct VerifiedCase {
    input: Vec<f64>,
    input_path: PathBuf,
    checksum: String,
    range: OutputRange,
}

#[derive(Debug)]
struct Measurement {
    warmup_iterations: u64,
    iterations_per_sample: u64,
    samples_ns: Vec<f64>,
}

struct CBackend;

impl CBackend {
    fn initialize() -> Result<Self, String> {
        let result = unsafe { TA_Initialize() };
        check_c(result, "TA_Initialize")?;
        Ok(Self)
    }

    fn version(&self) -> Result<String, String> {
        let pointer = unsafe { TA_GetVersionString() };
        if pointer.is_null() {
            return Err("TA_GetVersionString returned null".to_owned());
        }
        let full = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .map_err(|error| format!("TA_GetVersionString returned invalid UTF-8: {error}"))?;
        full.split_whitespace()
            .next()
            .map(str::to_owned)
            .ok_or_else(|| "TA_GetVersionString returned an empty version".to_owned())
    }

    fn lookback(&self) -> Result<usize, String> {
        let lookback = unsafe { TA_SMA_Lookback(PERIOD as c_int) };
        usize::try_from(lookback).map_err(|_| format!("TA_SMA_Lookback returned {lookback}"))
    }

    fn compute(&self, input: &[f64], output: &mut [f64]) -> Result<OutputRange, String> {
        let required = input.len().saturating_sub(PERIOD - 1);
        if output.len() < required {
            return Err(format!(
                "TA_SMA output requires at least {required} values, got {}",
                output.len()
            ));
        }
        let end = c_int::try_from(input.len())
            .map_err(|_| "SMA input is too large for TA-Lib's i32 index".to_owned())?
            - 1;
        let mut output_begin: c_int = 0;
        let mut output_count: c_int = 0;
        let result = unsafe {
            TA_SMA(
                0,
                end,
                input.as_ptr(),
                PERIOD as c_int,
                &mut output_begin,
                &mut output_count,
                output.as_mut_ptr(),
            )
        };
        check_c(result, "TA_SMA")?;
        Ok(OutputRange::new(
            usize::try_from(output_begin)
                .map_err(|_| format!("TA_SMA returned negative output begin {output_begin}"))?,
            usize::try_from(output_count)
                .map_err(|_| format!("TA_SMA returned negative output count {output_count}"))?,
        ))
    }
}

impl Drop for CBackend {
    fn drop(&mut self) {
        let _ = unsafe { TA_Shutdown() };
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SMA three-way benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if std::mem::size_of::<ta_core::Float>() != std::mem::size_of::<f64>() {
        return Err(
            "the pinned C/Python comparison requires ta-core's default f64 precision".to_owned(),
        );
    }
    let args = parse_args()?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|error| format!("create {}: {error}", args.output_dir.display()))?;
    let raw_path = args.output_dir.join("sma-three-way-raw.tsv");
    let report_path = args.output_dir.join("sma-three-way-report.txt");
    remove_stale(&raw_path)?;
    remove_stale(&report_path)?;

    let c_backend = CBackend::initialize()?;
    let c_version = c_backend.version()?;
    if c_version != TALIB_VERSION {
        return Err(format!(
            "pinned C version mismatch: expected {TALIB_VERSION}, got {c_version}"
        ));
    }
    if c_backend.lookback()? != PERIOD - 1 {
        return Err(format!(
            "TA_SMA_Lookback mismatch: expected {}, got {}",
            PERIOD - 1,
            c_backend.lookback()?
        ));
    }

    // Every implementation and every size passes the semantic gate before any
    // performance API is called. A failure leaves neither raw timings nor a report.
    let mut verified_cases = Vec::with_capacity(INPUT_LENGTHS.len());
    let mut python_provenance = None;
    for input_length in INPUT_LENGTHS {
        let input = series_fixture(input_length, 0);
        if input.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "fixture for {input_length} contains a non-finite value"
            ));
        }
        let input_path = args
            .output_dir
            .join(format!("sma-input-{input_length}-f64le.bin"));
        write_f64le(&input_path, &input)?;

        let config = SMAConfig::new(PERIOD).map_err(|error| error.to_string())?;
        let count = input_length - config.lookback();
        let mut rust_output = vec![0.0; count];
        let rust_range = config
            .compute_into(&input, &mut rust_output)
            .map_err(|error| format!("fast-ta semantic run for {input_length}: {error}"))?;

        let mut c_output = vec![0.0; count];
        let c_range = c_backend.compute(&input, &mut c_output)?;
        validate_outputs(
            (rust_range.beg_idx, rust_range.nb_element),
            &rust_output,
            "TA-Lib C",
            (c_range.beg_idx, c_range.nb_element),
            &c_output[..c_range.nb_element],
        )?;

        let python_output_path = args
            .output_dir
            .join(format!("sma-python-{input_length}-f64le.bin"));
        let python = python_semantic(&args.python, &input_path, &python_output_path)?;
        if python.binding_version != TALIB_VERSION {
            return Err(format!(
                "pinned Python binding version mismatch: expected {TALIB_VERSION}, got {}",
                python.binding_version
            ));
        }
        if python.numpy_version != NUMPY_VERSION {
            return Err(format!(
                "pinned NumPy version mismatch: expected {NUMPY_VERSION}, got {}",
                python.numpy_version
            ));
        }
        if python.ta_lib_version != c_version {
            return Err(format!(
                "Python/C TA-Lib version mismatch: Python reports {}, C reports {}",
                python.ta_lib_version, c_version
            ));
        }
        let reported_provenance = PythonProvenance {
            python_version: python.python_version.clone(),
            binding_version: python.binding_version.clone(),
            ta_lib_version: python.ta_lib_version.clone(),
            numpy_version: python.numpy_version.clone(),
        };
        if let Some(expected) = &python_provenance {
            if expected != &reported_provenance {
                return Err("Python provenance changed between semantic cases".to_owned());
            }
        }
        python_provenance = Some(reported_provenance);
        validate_outputs(
            (rust_range.beg_idx, rust_range.nb_element),
            &rust_output,
            "TA-Lib Python",
            python.range,
            &python.values,
        )?;

        verified_cases.push(VerifiedCase {
            checksum: input_checksum(&input),
            input,
            input_path,
            range: rust_range,
        });
    }

    let python_provenance =
        python_provenance.ok_or_else(|| "Python binding did not report provenance".to_owned())?;
    let provenance = provenance();
    let mut rows = Vec::with_capacity(INPUT_LENGTHS.len() * 3);

    for case in verified_cases {
        let count = case.range.nb_element;
        let config = SMAConfig::new(PERIOD).map_err(|error| error.to_string())?;
        let mut rust_output = vec![0.0; count];
        let rust_measurement = measure(
            || {
                let range = config
                    .compute_into(
                        black_box(case.input.as_slice()),
                        black_box(rust_output.as_mut_slice()),
                    )
                    .map_err(|error| error.to_string())?;
                black_box((range, rust_output.as_slice()));
                Ok(())
            },
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        )?;
        rows.push(make_row(
            "fast-ta",
            "caller-owned Batch Computation",
            "IndicatorConfig::compute_into; caller output allocation excluded",
            &case,
            &rust_measurement,
            &python_provenance,
            &provenance,
        )?);

        let mut c_output = vec![0.0; count];
        let c_measurement = measure(
            || {
                let range = c_backend.compute(
                    black_box(case.input.as_slice()),
                    black_box(c_output.as_mut_slice()),
                )?;
                black_box((range, c_output.as_slice()));
                Ok(())
            },
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        )?;
        rows.push(make_row(
            "TA-Lib C",
            "direct C caller-owned",
            "TA_SMA direct call; caller output allocation excluded",
            &case,
            &c_measurement,
            &python_provenance,
            &provenance,
        )?);

        let python = python_timing(
            &args.python,
            &case.input_path,
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        )?;
        let python_measurement = Measurement {
            warmup_iterations: python.warmup_iterations,
            iterations_per_sample: python.iterations_per_sample,
            samples_ns: python.samples_ns,
        };
        rows.push(make_row(
            "TA-Lib Python",
            "official NumPy API",
            "talib.SMA call including API-owned output; NumPy construction and file loading excluded",
            &case,
            &python_measurement,
            &python_provenance,
            &provenance,
        )?);
    }

    write_raw_rows(&raw_path, &rows)?;
    let raw_rows = read_raw_rows(&raw_path)?;
    let report = render_report(&raw_rows)?;
    fs::write(&report_path, report)
        .map_err(|error| format!("write {}: {error}", report_path.display()))?;
    println!("raw rows: {}", raw_path.display());
    println!("human report: {}", report_path.display());
    Ok(())
}

fn make_row(
    implementation: &str,
    mode: &str,
    timed_boundary: &str,
    case: &VerifiedCase,
    measurement: &Measurement,
    python: &PythonProvenance,
    provenance: &Provenance,
) -> Result<BenchmarkRow, String> {
    Ok(BenchmarkRow {
        implementation: implementation.to_owned(),
        indicator_definition: "SMA: arithmetic mean of timeperiod consecutive real values"
            .to_owned(),
        mode: mode.to_owned(),
        parameters: format!("timeperiod={PERIOD}"),
        input_length: case.input.len(),
        stats: timing_stats(&measurement.samples_ns, case.input.len())?,
        output_begin: case.range.beg_idx,
        output_count: case.range.nb_element,
        semantic_verified: true,
        warmup_iterations: measurement.warmup_iterations,
        iterations_per_sample: measurement.iterations_per_sample,
        timed_boundary: timed_boundary.to_owned(),
        fixture: "benches::support::series_fixture(size, seed=0); shared f64 little-endian bytes"
            .to_owned(),
        input_checksum: case.checksum.clone(),
        ta_lib_version: TALIB_VERSION.to_owned(),
        ta_lib_revision: TALIB_REVISION.to_owned(),
        python_version: python.python_version.clone(),
        python_binding_version: python.binding_version.clone(),
        python_ta_lib_version: python.ta_lib_version.clone(),
        numpy_version: python.numpy_version.clone(),
        rustc: provenance.rustc.clone(),
        cpu: provenance.cpu.clone(),
        os: provenance.os.clone(),
        arch: std::env::consts::ARCH.to_owned(),
        float_width: 64,
        features: "ta-core=default(f64,std); ta-benchmarks=sma-three-way".to_owned(),
        commit: provenance.commit.clone(),
        dirty: provenance.dirty,
    })
}

fn measure<F>(
    mut operation: F,
    sample_count: usize,
    warmup_ms: u64,
    sample_ms: u64,
) -> Result<Measurement, String>
where
    F: FnMut() -> Result<(), String>,
{
    if sample_count < 2 || warmup_ms == 0 || sample_ms == 0 {
        return Err("samples must be at least 2 and timing durations must be positive".to_owned());
    }
    let warmup_deadline = Instant::now() + Duration::from_millis(warmup_ms);
    let mut warmup_iterations = 0_u64;
    while Instant::now() < warmup_deadline {
        operation()?;
        warmup_iterations += 1;
    }

    let calibration_start = Instant::now();
    operation()?;
    let calibration_ns = calibration_start.elapsed().as_nanos().max(1) as u64;
    let target_ns = sample_ms.saturating_mul(1_000_000);
    let iterations_per_sample = target_ns.div_ceil(calibration_ns).clamp(1, 1_000_000);
    let mut samples_ns = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let started = Instant::now();
        for _ in 0..iterations_per_sample {
            operation()?;
        }
        samples_ns.push(started.elapsed().as_nanos() as f64 / iterations_per_sample as f64);
    }
    Ok(Measurement {
        warmup_iterations,
        iterations_per_sample,
        samples_ns,
    })
}

fn python_semantic(python: &Path, input: &Path, output: &Path) -> Result<PythonSemantic, String> {
    let worker = python_worker();
    let result = run_command(
        python,
        &[
            worker.as_os_str(),
            "semantic".as_ref(),
            input.as_os_str(),
            output.as_os_str(),
            PERIOD.to_string().as_ref(),
        ],
    )?;
    let metadata = parse_metadata(&result)?;
    let values = read_f64le(output)?;
    Ok(PythonSemantic {
        python_version: required(&metadata, "python_version")?.to_owned(),
        binding_version: required(&metadata, "python_binding_version")?.to_owned(),
        ta_lib_version: required(&metadata, "python_ta_lib_version")?.to_owned(),
        numpy_version: required(&metadata, "numpy_version")?.to_owned(),
        range: (
            parse_metadata_value(&metadata, "output_begin")?,
            parse_metadata_value(&metadata, "output_count")?,
        ),
        values,
    })
}

fn python_timing(
    python: &Path,
    input: &Path,
    samples: usize,
    warmup_ms: u64,
    sample_ms: u64,
) -> Result<PythonTiming, String> {
    let worker = python_worker();
    let result = run_command(
        python,
        &[
            worker.as_os_str(),
            "timing".as_ref(),
            input.as_os_str(),
            PERIOD.to_string().as_ref(),
            samples.to_string().as_ref(),
            warmup_ms.to_string().as_ref(),
            sample_ms.to_string().as_ref(),
        ],
    )?;
    let metadata = parse_metadata(&result)?;
    let samples_ns = required(&metadata, "samples_ns")?
        .split(',')
        .map(|value| value.parse::<f64>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if samples_ns.len() != samples {
        return Err(format!(
            "Python returned {} timing samples, expected {samples}",
            samples_ns.len()
        ));
    }
    Ok(PythonTiming {
        warmup_iterations: parse_metadata_value(&metadata, "warmup_iterations")?,
        iterations_per_sample: parse_metadata_value(&metadata, "iterations_per_sample")?,
        samples_ns,
    })
}

fn python_worker() -> PathBuf {
    std::env::var_os("SMA_PYTHON_WORKER").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/sma_python_worker.py"),
        PathBuf::from,
    )
}

fn run_command(program: &Path, arguments: &[&std::ffi::OsStr]) -> Result<String, String> {
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| format!("run {}: {error}", program.display()))?;
    if !status.success() {
        return Err(format!(
            "{} exited with {status}: {}",
            program.display(),
            String::from_utf8_lossy(&stderr).trim()
        ));
    }
    String::from_utf8(stdout)
        .map_err(|error| format!("{} emitted invalid UTF-8: {error}", program.display()))
}

fn parse_metadata(output: &str) -> Result<std::collections::BTreeMap<String, String>, String> {
    output
        .lines()
        .map(|line| {
            line.split_once('=')
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .ok_or_else(|| format!("invalid Python worker metadata line {line:?}"))
        })
        .collect()
}

fn required<'a>(
    metadata: &'a std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, String> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("Python worker did not return {key}"))
}

fn parse_metadata_value<T>(
    metadata: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    required(metadata, key)?
        .parse::<T>()
        .map_err(|error| format!("invalid Python {key}: {error}"))
}

fn write_f64le(path: &Path, values: &[f64]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for value in values {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush {}: {error}", path.display()))
}

fn read_f64le(path: &Path) -> Result<Vec<f64>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{} is not a whole f64 array", path.display()));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect())
}

fn check_c(result: c_int, operation: &str) -> Result<(), String> {
    if result == TA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{operation} returned TA_RetCode {result}"))
    }
}

fn remove_stale(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove stale {}: {error}", path.display())),
    }
}

#[derive(Debug)]
struct Provenance {
    commit: String,
    dirty: bool,
    rustc: String,
    cpu: String,
    os: String,
}

fn provenance() -> Provenance {
    Provenance {
        commit: command_text("git", &["rev-parse", "HEAD"]),
        dirty: !command_text("git", &["status", "--porcelain"]).is_empty(),
        rustc: command_text("rustc", &["--version", "--verbose"]),
        cpu: cpu_model(),
        os: command_text("uname", &["-a"]),
    }
}

fn command_text(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(target_os = "macos")]
fn cpu_model() -> String {
    command_text("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                matches!(key.trim(), "model name" | "Hardware").then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn cpu_model() -> String {
    "unavailable".to_owned()
}

fn parse_args() -> Result<Args, String> {
    let mut python = None;
    let mut output_dir = None;
    let mut samples = DEFAULT_SAMPLES;
    let mut warmup_ms = DEFAULT_WARMUP_MS;
    let mut sample_ms = DEFAULT_SAMPLE_MS;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--python" => python = Some(PathBuf::from(value)),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--samples" => {
                samples = value
                    .parse()
                    .map_err(|error| format!("invalid --samples: {error}"))?
            }
            "--warmup-ms" => {
                warmup_ms = value
                    .parse()
                    .map_err(|error| format!("invalid --warmup-ms: {error}"))?
            }
            "--sample-ms" => {
                sample_ms = value
                    .parse()
                    .map_err(|error| format!("invalid --sample-ms: {error}"))?
            }
            _ => return Err(format!("unknown argument {argument}")),
        }
    }
    Ok(Args {
        python: python.ok_or_else(|| "--python is required".to_owned())?,
        output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
        samples,
        warmup_ms,
        sample_ms,
    })
}
