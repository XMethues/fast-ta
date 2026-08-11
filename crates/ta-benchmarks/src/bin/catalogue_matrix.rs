use std::ffi::CStr;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Read, Write};
use std::os::raw::{c_char, c_double, c_int};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use ta_benchmarks::catalogue_matrix::{
    catalogue_fixture, fixture_checksum, read_raw_rows, render_report, timing_stats,
    validate_outputs, write_raw_rows, BenchmarkRow, CaseKind, CaseSpec, Fixture, OutputValues,
    VerifiedOutput, C_DIRECT_MODE, FIXTURE_ID, INPUT_LENGTHS, MATRIX, PRIMARY_COMPARISON,
    PYTHON_MODE, RUST_CALLER_MODE, RUST_OWNED_MODE, RUST_PREPARED_MODE, RUST_STREAMING_MODE,
};
use ta_core::cycle::HT_DCPHASEConfig;
use ta_core::math_operators::{ADDConfig, BinaryInput, BinaryTick};
use ta_core::math_transform::SINConfig;
use ta_core::momentum::{
    ADXConfig, DirectionalInput, DirectionalTick, MACDConfig, MACDValuesMut, RSIConfig,
};
use ta_core::overlap::{BBANDSConfig, BBANDSValuesMut, PeriodMAType, SMAConfig};
use ta_core::pattern_recognition::{
    CDL3BLACKCROWSConfig, CDLDOJIConfig, CDLENGULFINGConfig, Candle, CandleInput, PatternSignal,
};
use ta_core::price_transform::{TYPPRICEConfig, TYPPRICEInput, TYPPRICETick};
use ta_core::statistic::LINEARREGConfig;
use ta_core::volatility::{ATRConfig, ATRInput, ATRTick};
use ta_core::volume::{OBVConfig, OBVInput, OBVTick};
use ta_core::{Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation};

const TA_SUCCESS: c_int = 0;
const TALIB_VERSION: &str = "0.6.4";
const TALIB_REVISION: &str = "43f9d5042ecc4bd367941846494ad907bf20ea50";
const PYTHON_BINDING_VERSION: &str = "0.6.4";
const NUMPY_VERSION: &str = "2.2.3";
const DEFAULT_SAMPLES: usize = 50;
const DEFAULT_WARMUP_MS: u64 = 250;
const DEFAULT_SAMPLE_MS: u64 = 10;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn TA_Initialize() -> c_int;
    fn TA_Shutdown() -> c_int;
    fn TA_GetVersionString() -> *const c_char;
    fn TA_SMA(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_BBANDS(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        period: c_int,
        nbdev_up: c_double,
        nbdev_down: c_double,
        ma_type: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        upper: *mut c_double,
        middle: *mut c_double,
        lower: *mut c_double,
    ) -> c_int;
    fn TA_RSI(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_MACD(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        fast_period: c_int,
        slow_period: c_int,
        signal_period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        macd: *mut c_double,
        signal: *mut c_double,
        histogram: *mut c_double,
    ) -> c_int;
    fn TA_ATR(
        start_idx: c_int,
        end_idx: c_int,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_ADX(
        start_idx: c_int,
        end_idx: c_int,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_HT_DCPHASE(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_CDLDOJI(
        start_idx: c_int,
        end_idx: c_int,
        open: *const c_double,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_int,
    ) -> c_int;
    fn TA_CDLENGULFING(
        start_idx: c_int,
        end_idx: c_int,
        open: *const c_double,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_int,
    ) -> c_int;
    fn TA_CDL3BLACKCROWS(
        start_idx: c_int,
        end_idx: c_int,
        open: *const c_double,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_int,
    ) -> c_int;
    fn TA_LINEARREG(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        period: c_int,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_TYPPRICE(
        start_idx: c_int,
        end_idx: c_int,
        high: *const c_double,
        low: *const c_double,
        close: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_OBV(
        start_idx: c_int,
        end_idx: c_int,
        close: *const c_double,
        volume: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_SIN(
        start_idx: c_int,
        end_idx: c_int,
        input: *const c_double,
        output_begin: *mut c_int,
        output_count: *mut c_int,
        output: *mut c_double,
    ) -> c_int;
    fn TA_ADD(
        start_idx: c_int,
        end_idx: c_int,
        left: *const c_double,
        right: *const c_double,
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
    case: Option<String>,
    input_length: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustMode {
    Owned,
    CallerOwned,
    Prepared,
    Streaming,
}

impl RustMode {
    const ALL: [Self; 4] = [
        Self::Owned,
        Self::CallerOwned,
        Self::Prepared,
        Self::Streaming,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Owned => RUST_OWNED_MODE,
            Self::CallerOwned => RUST_CALLER_MODE,
            Self::Prepared => RUST_PREPARED_MODE,
            Self::Streaming => RUST_STREAMING_MODE,
        }
    }

    fn timed_boundary(self) -> &'static str {
        match self {
            Self::Owned => "IndicatorConfig::compute; API-owned Compact Output allocation included",
            Self::CallerOwned => "IndicatorConfig::compute_into; caller output allocation excluded",
            Self::Prepared => "PreparedBatchRunner::compute_into; preparation and caller output allocation excluded",
            Self::Streaming => "StreamingComputation::reset plus one next call per observation; stream and output allocation excluded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Rust(RustMode),
    C,
    Python,
}

impl Variant {
    fn implementation(self) -> &'static str {
        match self {
            Self::Rust(_) => "fast-ta",
            Self::C => "TA-Lib C",
            Self::Python => "TA-Lib Python",
        }
    }

    fn mode(self) -> &'static str {
        match self {
            Self::Rust(mode) => mode.name(),
            Self::C => C_DIRECT_MODE,
            Self::Python => PYTHON_MODE,
        }
    }

    fn timed_boundary(self, case_id: &str) -> String {
        match self {
            Self::Rust(mode) => mode.timed_boundary().to_owned(),
            Self::C => format!("direct TA_{case_id} call; caller output allocation excluded"),
            Self::Python => format!("direct talib.{case_id} NumPy API expression including API-owned output; fixture loading excluded"),
        }
    }

    fn comparison(self) -> (&'static str, &'static str) {
        match self {
            Self::Rust(RustMode::CallerOwned) | Self::C => ("comparable", PRIMARY_COMPARISON),
            Self::Rust(_) => (
                "unavailable",
                "separate Rust execution cost; not aggregated with the direct C kernel",
            ),
            Self::Python => (
                "unavailable",
                "user-facing Python API cost; not aggregated with caller-owned kernels",
            ),
        }
    }
}

#[derive(Debug)]
struct Measurement {
    warmup_iterations: u64,
    iterations_per_sample: u64,
    samples_ns: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PythonProvenance {
    python_version: String,
    binding_version: String,
    ta_lib_version: String,
    numpy_version: String,
}

#[derive(Debug)]
struct PythonSemantic {
    provenance: PythonProvenance,
    output: VerifiedOutput,
}

#[derive(Debug)]
struct SemanticVariant {
    variant: Variant,
    output: Option<VerifiedOutput>,
    status: String,
    reason: String,
}

#[derive(Debug)]
struct MatrixCase {
    spec: CaseSpec,
    fixture: Fixture,
    fixture_dir: PathBuf,
    checksum: String,
    variants: Vec<SemanticVariant>,
}

#[derive(Debug)]
struct OutputBuffers {
    floats: Vec<Vec<f64>>,
    integers: Vec<Vec<c_int>>,
}

impl OutputBuffers {
    fn new(spec: CaseSpec, input_length: usize) -> Self {
        if spec.output_kind == "integer" {
            Self {
                floats: Vec::new(),
                integers: (0..spec.output_arity)
                    .map(|_| vec![0; input_length])
                    .collect(),
            }
        } else {
            Self {
                floats: (0..spec.output_arity)
                    .map(|_| vec![0.0; input_length])
                    .collect(),
                integers: Vec::new(),
            }
        }
    }

    fn verified(&self, begin: usize, count: usize) -> VerifiedOutput {
        let values = if self.integers.is_empty() {
            OutputValues::Float(
                self.floats
                    .iter()
                    .map(|column| column[..count].to_vec())
                    .collect(),
            )
        } else {
            OutputValues::Integer(
                self.integers
                    .iter()
                    .map(|column| column[..count].to_vec())
                    .collect(),
            )
        };
        VerifiedOutput {
            begin,
            count,
            values,
        }
    }
}

struct CBackend;

impl CBackend {
    fn initialize() -> Result<Self, String> {
        check_c(unsafe { TA_Initialize() }, "TA_Initialize")?;
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

    fn compute(&self, spec: CaseSpec, fixture: &Fixture) -> Result<VerifiedOutput, String> {
        let mut buffers = OutputBuffers::new(spec, fixture.len());
        let range = self.compute_into(spec, fixture, &mut buffers)?;
        Ok(buffers.verified(range.beg_idx, range.nb_element))
    }

    fn compute_into(
        &self,
        spec: CaseSpec,
        fixture: &Fixture,
        buffers: &mut OutputBuffers,
    ) -> Result<OutputRange, String> {
        let end_idx = c_int::try_from(fixture.len())
            .map_err(|_| format!("{} input is too large for TA-Lib's i32 index", spec.id))?
            - 1;
        let mut output_begin: c_int = 0;
        let mut output_count: c_int = 0;
        let result = match spec.kind {
            CaseKind::Sma => unsafe {
                TA_SMA(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    14,
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Bbands => {
                let upper = buffers.floats[0].as_mut_ptr();
                let middle = buffers.floats[1].as_mut_ptr();
                let lower = buffers.floats[2].as_mut_ptr();
                unsafe {
                    TA_BBANDS(
                        0,
                        end_idx,
                        fixture.close.as_ptr(),
                        20,
                        2.0,
                        2.0,
                        0,
                        &mut output_begin,
                        &mut output_count,
                        upper,
                        middle,
                        lower,
                    )
                }
            }
            CaseKind::Rsi => unsafe {
                TA_RSI(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    14,
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Macd => {
                let macd = buffers.floats[0].as_mut_ptr();
                let signal = buffers.floats[1].as_mut_ptr();
                let histogram = buffers.floats[2].as_mut_ptr();
                unsafe {
                    TA_MACD(
                        0,
                        end_idx,
                        fixture.close.as_ptr(),
                        12,
                        26,
                        9,
                        &mut output_begin,
                        &mut output_count,
                        macd,
                        signal,
                        histogram,
                    )
                }
            }
            CaseKind::Atr => unsafe {
                TA_ATR(
                    0,
                    end_idx,
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    14,
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Adx => unsafe {
                TA_ADX(
                    0,
                    end_idx,
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    14,
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::HtDcPhase => unsafe {
                TA_HT_DCPHASE(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::CdlDoji => unsafe {
                TA_CDLDOJI(
                    0,
                    end_idx,
                    fixture.open.as_ptr(),
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.integers[0].as_mut_ptr(),
                )
            },
            CaseKind::CdlEngulfing => unsafe {
                TA_CDLENGULFING(
                    0,
                    end_idx,
                    fixture.open.as_ptr(),
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.integers[0].as_mut_ptr(),
                )
            },
            CaseKind::Cdl3BlackCrows => unsafe {
                TA_CDL3BLACKCROWS(
                    0,
                    end_idx,
                    fixture.open.as_ptr(),
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.integers[0].as_mut_ptr(),
                )
            },
            CaseKind::LinearReg => unsafe {
                TA_LINEARREG(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    14,
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::TypPrice => unsafe {
                TA_TYPPRICE(
                    0,
                    end_idx,
                    fixture.high.as_ptr(),
                    fixture.low.as_ptr(),
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Obv => unsafe {
                TA_OBV(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    fixture.volume.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Sin => unsafe {
                TA_SIN(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
            CaseKind::Add => unsafe {
                TA_ADD(
                    0,
                    end_idx,
                    fixture.close.as_ptr(),
                    fixture.auxiliary.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    buffers.floats[0].as_mut_ptr(),
                )
            },
        };
        check_c(result, &format!("TA_{}", spec.id))?;
        let begin = usize::try_from(output_begin).map_err(|_| {
            format!(
                "TA_{} returned negative output begin {output_begin}",
                spec.id
            )
        })?;
        let count = usize::try_from(output_count).map_err(|_| {
            format!(
                "TA_{} returned negative output count {output_count}",
                spec.id
            )
        })?;
        if count > fixture.len() {
            return Err(format!(
                "TA_{} returned output count {count} for input length {}",
                spec.id,
                fixture.len()
            ));
        }
        Ok(if count == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(begin, count)
        })
    }

    fn operation<'a>(
        &'a self,
        spec: CaseSpec,
        fixture: &'a Fixture,
    ) -> Box<dyn FnMut() -> Result<(), String> + 'a> {
        let mut buffers = OutputBuffers::new(spec, fixture.len());
        Box::new(move || {
            let range = self.compute_into(spec, black_box(fixture), &mut buffers)?;
            black_box((range, &buffers));
            Ok(())
        })
    }
}

impl Drop for CBackend {
    fn drop(&mut self) {
        let _ = unsafe { TA_Shutdown() };
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Indicator Catalogue matrix benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if std::mem::size_of::<Float>() != std::mem::size_of::<f64>() {
        return Err(
            "the pinned C/Python comparison requires ta-core's default f64 precision".to_owned(),
        );
    }
    let args = parse_args()?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|error| format!("create {}: {error}", args.output_dir.display()))?;
    let raw_path = args.output_dir.join("catalogue-matrix-raw.tsv");
    let report_path = args.output_dir.join("catalogue-matrix-report.txt");
    remove_stale(&raw_path)?;
    remove_stale(&report_path)?;

    let c_backend = CBackend::initialize()?;
    let c_version = c_backend.version()?;
    if c_version != TALIB_VERSION {
        return Err(format!(
            "pinned C version mismatch: expected {TALIB_VERSION}, got {c_version}"
        ));
    }

    let mut matrix_cases = Vec::with_capacity(INPUT_LENGTHS.len() * MATRIX.len());
    let mut python_provenance = None;
    for input_length in INPUT_LENGTHS {
        if args
            .input_length
            .is_some_and(|selected| selected != input_length)
        {
            continue;
        }
        let fixture = catalogue_fixture(input_length);
        fixture.validate()?;
        let fixture_dir = args.output_dir.join(format!("fixture-{input_length}"));
        write_fixture(&fixture_dir, &fixture)?;
        let checksum = fixture_checksum(&fixture);
        for spec in MATRIX {
            if args
                .case
                .as_deref()
                .is_some_and(|selected| selected != spec.id)
            {
                continue;
            }
            let c_result = c_backend.compute(spec, &fixture);
            let mut variants = Vec::with_capacity(6);
            for mode in RustMode::ALL {
                let result = rust_compute(spec, mode, &fixture);
                variants.push(classify(Variant::Rust(mode), c_result.as_ref(), result));
            }
            variants.push(match &c_result {
                Ok(output) => SemanticVariant {
                    variant: Variant::C,
                    output: Some(output.clone()),
                    status: "verified".to_owned(),
                    reason: String::new(),
                },
                Err(error) => SemanticVariant {
                    variant: Variant::C,
                    output: None,
                    status: "unavailable".to_owned(),
                    reason: error.clone(),
                },
            });
            let python_dir = args.output_dir.join(format!(
                "python-{}-{input_length}",
                spec.id.to_ascii_lowercase()
            ));
            remove_dir_if_present(&python_dir)?;
            let python_result = python_semantic(&args.python, &fixture_dir, &python_dir, spec);
            let python_output = match python_result {
                Ok(python) => {
                    validate_python_provenance(&python.provenance, &c_version)?;
                    if let Some(expected) = &python_provenance {
                        if expected != &python.provenance {
                            return Err(
                                "Python provenance changed between semantic cases".to_owned()
                            );
                        }
                    }
                    python_provenance = Some(python.provenance);
                    Ok(python.output)
                }
                Err(error) => Err(error),
            };
            variants.push(classify(Variant::Python, c_result.as_ref(), python_output));
            matrix_cases.push(MatrixCase {
                spec,
                fixture: fixture.clone(),
                fixture_dir: fixture_dir.clone(),
                checksum: checksum.clone(),
                variants,
            });
        }
    }

    let python_provenance = python_provenance.unwrap_or_else(|| PythonProvenance {
        python_version: "unavailable".to_owned(),
        binding_version: "unavailable".to_owned(),
        ta_lib_version: "unavailable".to_owned(),
        numpy_version: "unavailable".to_owned(),
    });
    let provenance = provenance();
    let mut rows = Vec::with_capacity(matrix_cases.len() * 6);
    let mut failures = 0usize;
    for matrix_case in matrix_cases {
        let gate_passed = matrix_case
            .variants
            .iter()
            .all(|variant| variant.status == "verified");
        for semantic in &matrix_case.variants {
            let measurement = if gate_passed {
                match measure_variant(&c_backend, &args, &matrix_case, semantic.variant) {
                    Ok(measurement) => Some(measurement),
                    Err(error) => {
                        failures += 1;
                        rows.push(make_row(
                            &matrix_case,
                            semantic,
                            None,
                            "unavailable",
                            &error,
                            &python_provenance,
                            &provenance,
                        )?);
                        continue;
                    }
                }
            } else {
                failures += usize::from(semantic.status != "verified");
                None
            };
            let (timing_status, timing_reason) = if measurement.is_some() {
                ("measured", "")
            } else {
                (
                    "suppressed",
                    "another implementation or execution mode failed this case's semantic gate",
                )
            };
            rows.push(make_row(
                &matrix_case,
                semantic,
                measurement.as_ref(),
                timing_status,
                timing_reason,
                &python_provenance,
                &provenance,
            )?);
        }
    }

    write_raw_rows(&raw_path, &rows)?;
    let raw_rows = read_raw_rows(&raw_path)?;
    let report = render_report(&raw_rows)?;
    fs::write(&report_path, report)
        .map_err(|error| format!("write {}: {error}", report_path.display()))?;
    println!("raw rows: {}", raw_path.display());
    println!("human report: {}", report_path.display());
    print_focused_verdict(&rows, &args);
    if failures != 0 {
        return Err(format!("{failures} matrix rows were unavailable or failed semantic verification; timings were suppressed and reasons are recorded in the generated artifacts"));
    }
    Ok(())
}

fn print_focused_verdict(rows: &[BenchmarkRow], args: &Args) {
    let Some(case_id) = args.case.as_deref() else {
        return;
    };
    for input_length in INPUT_LENGTHS {
        if args
            .input_length
            .is_some_and(|selected| selected != input_length)
        {
            continue;
        }
        let case_rows = rows
            .iter()
            .filter(|row| row.case_id == case_id && row.input_length == input_length);
        let semantic_count = case_rows.clone().count();
        let semantics_verified = semantic_count == 6
            && case_rows
                .clone()
                .all(|row| row.semantic_status == "verified");
        let rust = case_rows.clone().find(|row| {
            row.implementation == "fast-ta"
                && row.mode == RUST_CALLER_MODE
                && row.timing_status == "measured"
        });
        let c = case_rows.clone().find(|row| {
            row.implementation == "TA-Lib C"
                && row.mode == C_DIRECT_MODE
                && row.timing_status == "measured"
        });
        let (Some(rust_stats), Some(c_stats)) = (
            rust.and_then(|row| row.stats.as_ref()),
            c.and_then(|row| row.stats.as_ref()),
        ) else {
            println!(
                "FOCUSED VERDICT: FAIL; case={case_id}; input_length={input_length}; \
                 semantic_verified={semantics_verified}; comparable timings unavailable"
            );
            continue;
        };
        let ratio = rust_stats.median_ns / c_stats.median_ns;
        let verdict = if semantics_verified && ratio <= 1.05 {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "FOCUSED VERDICT: {verdict}; case={case_id}; input_length={input_length}; \
             semantic_verified={semantics_verified}; fast-ta caller-owned median={:.3} us \
             ci95=[{:.3}, {:.3}] us throughput={:.3} Mobs/s; TA-Lib C median={:.3} us \
             ci95=[{:.3}, {:.3}] us throughput={:.3} Mobs/s; rust/c={ratio:.3}x; \
             pass_threshold<=1.050x",
            rust_stats.median_ns / 1_000.0,
            rust_stats.ci95_lower_ns / 1_000.0,
            rust_stats.ci95_upper_ns / 1_000.0,
            rust_stats.throughput_observations_per_second / 1.0e6,
            c_stats.median_ns / 1_000.0,
            c_stats.ci95_lower_ns / 1_000.0,
            c_stats.ci95_upper_ns / 1_000.0,
            c_stats.throughput_observations_per_second / 1.0e6,
        );
    }
}

fn classify(
    variant: Variant,
    reference: Result<&VerifiedOutput, &String>,
    actual: Result<VerifiedOutput, String>,
) -> SemanticVariant {
    match (reference, actual) {
        (Ok(expected), Ok(output)) => {
            match validate_outputs(expected, variant.implementation(), &output) {
                Ok(()) => SemanticVariant {
                    variant,
                    output: Some(output),
                    status: "verified".to_owned(),
                    reason: String::new(),
                },
                Err(error) => SemanticVariant {
                    variant,
                    output: Some(output),
                    status: "mismatch".to_owned(),
                    reason: error,
                },
            }
        }
        (Ok(_), Err(error)) => SemanticVariant {
            variant,
            output: None,
            status: "unavailable".to_owned(),
            reason: error,
        },
        (Err(reference_error), Ok(output)) => SemanticVariant {
            variant,
            output: Some(output),
            status: "unavailable".to_owned(),
            reason: format!("direct C reference unavailable: {reference_error}"),
        },
        (Err(reference_error), Err(error)) => SemanticVariant {
            variant,
            output: None,
            status: "unavailable".to_owned(),
            reason: format!(
                "direct C reference unavailable: {reference_error}; implementation error: {error}"
            ),
        },
    }
}

fn measure_variant(
    c_backend: &CBackend,
    args: &Args,
    matrix_case: &MatrixCase,
    variant: Variant,
) -> Result<Measurement, String> {
    match variant {
        Variant::Rust(mode) => measure(
            rust_operation(matrix_case.spec, mode, &matrix_case.fixture)?,
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        ),
        Variant::C => measure(
            c_backend.operation(matrix_case.spec, &matrix_case.fixture),
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        ),
        Variant::Python => python_timing(
            &args.python,
            &matrix_case.fixture_dir,
            matrix_case.spec,
            args.samples,
            args.warmup_ms,
            args.sample_ms,
        ),
    }
}

fn make_row(
    matrix_case: &MatrixCase,
    semantic: &SemanticVariant,
    measurement: Option<&Measurement>,
    timing_status: &str,
    timing_reason: &str,
    python: &PythonProvenance,
    provenance: &Provenance,
) -> Result<BenchmarkRow, String> {
    let stats = measurement
        .map(|value| timing_stats(&value.samples_ns, matrix_case.fixture.len()))
        .transpose()?;
    let (comparison_status, comparison_reason) = semantic.variant.comparison();
    Ok(BenchmarkRow {
        implementation: semantic.variant.implementation().to_owned(),
        indicator_family: matrix_case.spec.family.to_owned(),
        indicator_definition: matrix_case.spec.definition.to_owned(),
        case_id: matrix_case.spec.id.to_owned(),
        mode: semantic.variant.mode().to_owned(),
        parameters: matrix_case.spec.parameters.to_owned(),
        input_length: matrix_case.fixture.len(),
        output_kind: matrix_case.spec.output_kind.to_owned(),
        output_arity: semantic.output.as_ref().map(|output| output.values.arity()),
        stats,
        output_begin: semantic.output.as_ref().map(|output| output.begin),
        output_count: semantic.output.as_ref().map(|output| output.count),
        output_checksum: semantic
            .output
            .as_ref()
            .map_or_else(|| "unavailable".to_owned(), VerifiedOutput::checksum),
        semantic_status: semantic.status.clone(),
        semantic_reason: semantic.reason.clone(),
        timing_status: timing_status.to_owned(),
        timing_reason: timing_reason.to_owned(),
        comparison_status: comparison_status.to_owned(),
        comparison_reason: comparison_reason.to_owned(),
        warmup_iterations: measurement.map(|value| value.warmup_iterations),
        iterations_per_sample: measurement.map(|value| value.iterations_per_sample),
        timed_boundary: semantic.variant.timed_boundary(matrix_case.spec.id),
        fixture: FIXTURE_ID.to_owned(),
        input_checksum: matrix_case.checksum.clone(),
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
        features: "ta-core=default(f64,std); ta-benchmarks=catalogue-matrix".to_owned(),
        commit: provenance.commit.clone(),
        dirty: provenance.dirty,
    })
}

fn measure<'a>(
    mut operation: Box<dyn FnMut() -> Result<(), String> + 'a>,
    sample_count: usize,
    warmup_ms: u64,
    sample_ms: u64,
) -> Result<Measurement, String> {
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
    let calibration_ns = calibration_start.elapsed().as_nanos().max(1);
    let target_ns = u128::from(sample_ms) * 1_000_000;
    let iterations_per_sample =
        u64::try_from(target_ns.div_ceil(calibration_ns).clamp(1, 1_000_000))
            .map_err(|_| "calibrated iteration count does not fit u64".to_owned())?;
    let mut samples_ns = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let started = Instant::now();
        for _ in 0..iterations_per_sample {
            operation()?;
        }
        samples_ns.push(started.elapsed().as_secs_f64() * 1.0e9 / iterations_per_sample as f64);
    }
    Ok(Measurement {
        warmup_iterations,
        iterations_per_sample,
        samples_ns,
    })
}

fn rust_compute(
    spec: CaseSpec,
    mode: RustMode,
    fixture: &Fixture,
) -> Result<VerifiedOutput, String> {
    match spec.kind {
        CaseKind::Sma => rust_single(
            SMAConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::Bbands => rust_bbands(fixture, mode),
        CaseKind::Rsi => rust_single(
            RSIConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::Macd => rust_macd(fixture, mode),
        CaseKind::Atr => rust_atr(fixture, mode),
        CaseKind::Adx => rust_adx(fixture, mode),
        CaseKind::HtDcPhase => rust_single(HT_DCPHASEConfig::new(), &fixture.close, mode),
        CaseKind::CdlDoji => rust_pattern(CDLDOJIConfig::default(), fixture, mode),
        CaseKind::CdlEngulfing => rust_pattern(CDLENGULFINGConfig::default(), fixture, mode),
        CaseKind::Cdl3BlackCrows => rust_pattern(CDL3BLACKCROWSConfig::default(), fixture, mode),
        CaseKind::LinearReg => rust_single(
            LINEARREGConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::TypPrice => rust_typprice(fixture, mode),
        CaseKind::Obv => rust_obv(fixture, mode),
        CaseKind::Sin => rust_single(SINConfig::new(), &fixture.close, mode),
        CaseKind::Add => rust_add(fixture, mode),
    }
}

fn rust_operation<'a>(
    spec: CaseSpec,
    mode: RustMode,
    fixture: &'a Fixture,
) -> Result<Box<dyn FnMut() -> Result<(), String> + 'a>, String> {
    match spec.kind {
        CaseKind::Sma => rust_single_operation(
            SMAConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::Bbands => rust_bbands_operation(fixture, mode),
        CaseKind::Rsi => rust_single_operation(
            RSIConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::Macd => rust_macd_operation(fixture, mode),
        CaseKind::Atr => rust_atr_operation(fixture, mode),
        CaseKind::Adx => rust_adx_operation(fixture, mode),
        CaseKind::HtDcPhase => rust_single_operation(HT_DCPHASEConfig::new(), &fixture.close, mode),
        CaseKind::CdlDoji => rust_pattern_operation(CDLDOJIConfig::default(), fixture, mode),
        CaseKind::CdlEngulfing => {
            rust_pattern_operation(CDLENGULFINGConfig::default(), fixture, mode)
        }
        CaseKind::Cdl3BlackCrows => {
            rust_pattern_operation(CDL3BLACKCROWSConfig::default(), fixture, mode)
        }
        CaseKind::LinearReg => rust_single_operation(
            LINEARREGConfig::new(14).map_err(display_error)?,
            &fixture.close,
            mode,
        ),
        CaseKind::TypPrice => rust_typprice_operation(fixture, mode),
        CaseKind::Obv => rust_obv_operation(fixture, mode),
        CaseKind::Sin => rust_single_operation(SINConfig::new(), &fixture.close, mode),
        CaseKind::Add => rust_add_operation(fixture, mode),
    }
}

fn rust_single<C>(config: C, input: &[Float], mode: RustMode) -> Result<VerifiedOutput, String>
where
    C: IndicatorConfig<Output = Vec<Float>> + Copy,
    for<'input> C:
        IndicatorConfig<Input<'input> = &'input [Float], OutputMut<'input> = &'input mut [Float]>,
    C::Stream: StreamingComputation<C, Tick = Float, TickOutput = Float>,
{
    let count = input.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => {
            let output = config.compute(input).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input, &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(input.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input, &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for value in input {
                if let Some(output) = stream.next(*value).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_single_operation<'a, C>(
    config: C,
    input: &'a [Float],
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + 'a>, String>
where
    C: IndicatorConfig<Output = Vec<Float>> + Copy + 'static,
    for<'input> C:
        IndicatorConfig<Input<'input> = &'input [Float], OutputMut<'input> = &'input mut [Float]>,
    C::BatchRunner: 'a,
    C::Stream: StreamingComputation<C, Tick = Float, TickOutput = Float> + 'a,
{
    let count = input.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            black_box(config.compute(black_box(input)).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let range = config
                    .compute_into(black_box(input), black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(input.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let range = runner
                    .compute_into(black_box(input), black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for input_value in input {
                    if let Some(output) = stream
                        .next(black_box(*input_value))
                        .map_err(display_error)?
                    {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn float_verified(range: OutputRange, columns: Vec<Vec<Float>>) -> VerifiedOutput {
    VerifiedOutput {
        begin: range.beg_idx,
        count: range.nb_element,
        values: OutputValues::Float(columns),
    }
}

fn integer_verified(range: OutputRange, columns: Vec<Vec<PatternSignal>>) -> VerifiedOutput {
    VerifiedOutput {
        begin: range.beg_idx,
        count: range.nb_element,
        values: OutputValues::Integer(
            columns
                .into_iter()
                .map(|column| {
                    column
                        .into_iter()
                        .map(PatternSignal::to_talib_code)
                        .collect()
                })
                .collect(),
        ),
    }
}

fn stream_range(lookback: usize, count: usize) -> OutputRange {
    if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(lookback, count)
    }
}

fn display_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn rust_bbands(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = BBANDSConfig::new(20, 2.0, 2.0, PeriodMAType::SMA).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => {
            let output = config.compute(&fixture.close).map_err(display_error)?;
            let range = output.range();
            let values = output.into_values();
            Ok(float_verified(
                range,
                vec![values.upper, values.middle, values.lower],
            ))
        }
        RustMode::CallerOwned => {
            let (mut upper, mut middle, mut lower) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            let range = config
                .compute_into(
                    &fixture.close,
                    BBANDSValuesMut {
                        upper: &mut upper,
                        middle: &mut middle,
                        lower: &mut lower,
                    },
                )
                .map_err(display_error)?;
            Ok(float_verified(range, vec![upper, middle, lower]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let (mut upper, mut middle, mut lower) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            let range = runner
                .compute_into(
                    &fixture.close,
                    BBANDSValuesMut {
                        upper: &mut upper,
                        middle: &mut middle,
                        lower: &mut lower,
                    },
                )
                .map_err(display_error)?;
            Ok(float_verified(range, vec![upper, middle, lower]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let (mut upper, mut middle, mut lower) = (
                Vec::with_capacity(count),
                Vec::with_capacity(count),
                Vec::with_capacity(count),
            );
            for input in &fixture.close {
                if let Some(output) = stream.next(*input).map_err(display_error)? {
                    upper.push(output.upper);
                    middle.push(output.middle);
                    lower.push(output.lower);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), upper.len()),
                vec![upper, middle, lower],
            ))
        }
    }
}

fn rust_bbands_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = BBANDSConfig::new(20, 2.0, 2.0, PeriodMAType::SMA).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            black_box(
                config
                    .compute(black_box(fixture.close.as_slice()))
                    .map_err(display_error)?,
            );
            Ok(())
        })),
        RustMode::CallerOwned => {
            let (mut upper, mut middle, mut lower) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                let range = config
                    .compute_into(
                        black_box(fixture.close.as_slice()),
                        BBANDSValuesMut {
                            upper: black_box(&mut upper),
                            middle: black_box(&mut middle),
                            lower: black_box(&mut lower),
                        },
                    )
                    .map_err(display_error)?;
                black_box((range, &upper, &middle, &lower));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let (mut upper, mut middle, mut lower) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                let range = runner
                    .compute_into(
                        black_box(fixture.close.as_slice()),
                        BBANDSValuesMut {
                            upper: black_box(&mut upper),
                            middle: black_box(&mut middle),
                            lower: black_box(&mut lower),
                        },
                    )
                    .map_err(display_error)?;
                black_box((range, &upper, &middle, &lower));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let (mut upper, mut middle, mut lower) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for input in &fixture.close {
                    if let Some(output) = stream.next(black_box(*input)).map_err(display_error)? {
                        upper[output_index] = output.upper;
                        middle[output_index] = output.middle;
                        lower[output_index] = output.lower;
                        output_index += 1;
                    }
                }
                black_box((
                    &upper[..output_index],
                    &middle[..output_index],
                    &lower[..output_index],
                ));
                Ok(())
            }))
        }
    }
}

fn rust_macd(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = MACDConfig::new(12, 26, 9).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => {
            let output = config.compute(&fixture.close).map_err(display_error)?;
            let range = output.range();
            let values = output.into_values();
            Ok(float_verified(
                range,
                vec![values.macd, values.signal, values.histogram],
            ))
        }
        RustMode::CallerOwned => {
            let (mut macd, mut signal, mut histogram) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            let range = config
                .compute_into(
                    &fixture.close,
                    MACDValuesMut {
                        macd: &mut macd,
                        signal: &mut signal,
                        histogram: &mut histogram,
                    },
                )
                .map_err(display_error)?;
            Ok(float_verified(range, vec![macd, signal, histogram]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let (mut macd, mut signal, mut histogram) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            let range = runner
                .compute_into(
                    &fixture.close,
                    MACDValuesMut {
                        macd: &mut macd,
                        signal: &mut signal,
                        histogram: &mut histogram,
                    },
                )
                .map_err(display_error)?;
            Ok(float_verified(range, vec![macd, signal, histogram]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let (mut macd, mut signal, mut histogram) = (
                Vec::with_capacity(count),
                Vec::with_capacity(count),
                Vec::with_capacity(count),
            );
            for input in &fixture.close {
                if let Some(output) = stream.next(*input).map_err(display_error)? {
                    macd.push(output.macd);
                    signal.push(output.signal);
                    histogram.push(output.histogram);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), macd.len()),
                vec![macd, signal, histogram],
            ))
        }
    }
}

fn rust_macd_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = MACDConfig::new(12, 26, 9).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            black_box(
                config
                    .compute(black_box(fixture.close.as_slice()))
                    .map_err(display_error)?,
            );
            Ok(())
        })),
        RustMode::CallerOwned => {
            let (mut macd, mut signal, mut histogram) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                let range = config
                    .compute_into(
                        black_box(fixture.close.as_slice()),
                        MACDValuesMut {
                            macd: black_box(&mut macd),
                            signal: black_box(&mut signal),
                            histogram: black_box(&mut histogram),
                        },
                    )
                    .map_err(display_error)?;
                black_box((range, &macd, &signal, &histogram));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let (mut macd, mut signal, mut histogram) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                let range = runner
                    .compute_into(
                        black_box(fixture.close.as_slice()),
                        MACDValuesMut {
                            macd: black_box(&mut macd),
                            signal: black_box(&mut signal),
                            histogram: black_box(&mut histogram),
                        },
                    )
                    .map_err(display_error)?;
                black_box((range, &macd, &signal, &histogram));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let (mut macd, mut signal, mut histogram) =
                (vec![0.0; count], vec![0.0; count], vec![0.0; count]);
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for input in &fixture.close {
                    if let Some(output) = stream.next(black_box(*input)).map_err(display_error)? {
                        macd[output_index] = output.macd;
                        signal[output_index] = output.signal;
                        histogram[output_index] = output.histogram;
                        output_index += 1;
                    }
                }
                black_box((
                    &macd[..output_index],
                    &signal[..output_index],
                    &histogram[..output_index],
                ));
                Ok(())
            }))
        }
    }
}

fn rust_atr(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = ATRConfig::new(14).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    let input = || ATRInput {
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    };
    match mode {
        RustMode::Owned => {
            let output = config.compute(input()).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                let tick = ATRTick {
                    high: fixture.high[index],
                    low: fixture.low[index],
                    close: fixture.close[index],
                };
                if let Some(output) = stream.next(tick).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_atr_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = ATRConfig::new(14).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            let input = ATRInput {
                high: black_box(&fixture.high),
                low: black_box(&fixture.low),
                close: black_box(&fixture.close),
            };
            black_box(config.compute(input).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = ATRInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = config
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = ATRInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = runner
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    let tick = ATRTick {
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    };
                    if let Some(output) = stream.next(black_box(tick)).map_err(display_error)? {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn rust_adx(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = ADXConfig::new(14).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    let input = || DirectionalInput {
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    };
    match mode {
        RustMode::Owned => {
            let output = config.compute(input()).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                let tick = DirectionalTick {
                    high: fixture.high[index],
                    low: fixture.low[index],
                    close: fixture.close[index],
                };
                if let Some(output) = stream.next(tick).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_adx_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = ADXConfig::new(14).map_err(display_error)?;
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            let input = DirectionalInput {
                high: black_box(&fixture.high),
                low: black_box(&fixture.low),
                close: black_box(&fixture.close),
            };
            black_box(config.compute(input).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = DirectionalInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = config
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = DirectionalInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = runner
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    let tick = DirectionalTick {
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    };
                    if let Some(output) = stream.next(black_box(tick)).map_err(display_error)? {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn candle_input(fixture: &Fixture) -> CandleInput<'_> {
    CandleInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

fn candle(fixture: &Fixture, index: usize) -> Candle {
    Candle {
        open: fixture.open[index],
        high: fixture.high[index],
        low: fixture.low[index],
        close: fixture.close[index],
    }
}

fn rust_pattern<C>(config: C, fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String>
where
    C: IndicatorConfig<Output = Vec<PatternSignal>> + Copy,
    for<'input> C: IndicatorConfig<
        Input<'input> = CandleInput<'input>,
        OutputMut<'input> = &'input mut [PatternSignal],
    >,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal>,
{
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => {
            let output = config
                .compute(candle_input(fixture))
                .map_err(display_error)?;
            let range = output.range();
            Ok(integer_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![PatternSignal::NoMatch; count];
            let range = config
                .compute_into(candle_input(fixture), &mut values)
                .map_err(display_error)?;
            Ok(integer_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![PatternSignal::NoMatch; count];
            let range = runner
                .compute_into(candle_input(fixture), &mut values)
                .map_err(display_error)?;
            Ok(integer_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                if let Some(output) = stream.next(candle(fixture, index)).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(integer_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_pattern_operation<'a, C>(
    config: C,
    fixture: &'a Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + 'a>, String>
where
    C: IndicatorConfig<Output = Vec<PatternSignal>> + Copy + 'static,
    for<'input> C: IndicatorConfig<
        Input<'input> = CandleInput<'input>,
        OutputMut<'input> = &'input mut [PatternSignal],
    >,
    C::BatchRunner: 'a,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal> + 'a,
{
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            black_box(
                config
                    .compute(black_box(candle_input(fixture)))
                    .map_err(display_error)?,
            );
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![PatternSignal::NoMatch; count];
            Ok(Box::new(move || {
                let range = config
                    .compute_into(
                        black_box(candle_input(fixture)),
                        black_box(values.as_mut_slice()),
                    )
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![PatternSignal::NoMatch; count];
            Ok(Box::new(move || {
                let range = runner
                    .compute_into(
                        black_box(candle_input(fixture)),
                        black_box(values.as_mut_slice()),
                    )
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![PatternSignal::NoMatch; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    if let Some(output) = stream
                        .next(black_box(candle(fixture, index)))
                        .map_err(display_error)?
                    {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn rust_typprice(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = TYPPRICEConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    let input = || TYPPRICEInput {
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    };
    match mode {
        RustMode::Owned => {
            let output = config.compute(input()).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                let tick = TYPPRICETick {
                    high: fixture.high[index],
                    low: fixture.low[index],
                    close: fixture.close[index],
                };
                if let Some(output) = stream.next(tick).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_typprice_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = TYPPRICEConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            let input = TYPPRICEInput {
                high: black_box(&fixture.high),
                low: black_box(&fixture.low),
                close: black_box(&fixture.close),
            };
            black_box(config.compute(input).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = TYPPRICEInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = config
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = TYPPRICEInput {
                    high: black_box(&fixture.high),
                    low: black_box(&fixture.low),
                    close: black_box(&fixture.close),
                };
                let range = runner
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    let tick = TYPPRICETick {
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    };
                    if let Some(output) = stream.next(black_box(tick)).map_err(display_error)? {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn rust_obv(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = OBVConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    let input = || OBVInput {
        close: &fixture.close,
        volume: &fixture.volume,
    };
    match mode {
        RustMode::Owned => {
            let output = config.compute(input()).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                let tick = OBVTick {
                    close: fixture.close[index],
                    volume: fixture.volume[index],
                };
                if let Some(output) = stream.next(tick).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_obv_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = OBVConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            let input = OBVInput {
                close: black_box(&fixture.close),
                volume: black_box(&fixture.volume),
            };
            black_box(config.compute(input).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = OBVInput {
                    close: black_box(&fixture.close),
                    volume: black_box(&fixture.volume),
                };
                let range = config
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = OBVInput {
                    close: black_box(&fixture.close),
                    volume: black_box(&fixture.volume),
                };
                let range = runner
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    let tick = OBVTick {
                        close: fixture.close[index],
                        volume: fixture.volume[index],
                    };
                    if let Some(output) = stream.next(black_box(tick)).map_err(display_error)? {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn rust_add(fixture: &Fixture, mode: RustMode) -> Result<VerifiedOutput, String> {
    let config = ADDConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    let input = || BinaryInput {
        real0: &fixture.close,
        real1: &fixture.auxiliary,
    };
    match mode {
        RustMode::Owned => {
            let output = config.compute(input()).map_err(display_error)?;
            let range = output.range();
            Ok(float_verified(range, vec![output.into_values()]))
        }
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            let range = config
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            let range = runner
                .compute_into(input(), &mut values)
                .map_err(display_error)?;
            Ok(float_verified(range, vec![values]))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = Vec::with_capacity(count);
            for index in 0..fixture.len() {
                let tick = BinaryTick {
                    real0: fixture.close[index],
                    real1: fixture.auxiliary[index],
                };
                if let Some(output) = stream.next(tick).map_err(display_error)? {
                    values.push(output);
                }
            }
            Ok(float_verified(
                stream_range(config.lookback(), values.len()),
                vec![values],
            ))
        }
    }
}

fn rust_add_operation(
    fixture: &Fixture,
    mode: RustMode,
) -> Result<Box<dyn FnMut() -> Result<(), String> + '_>, String> {
    let config = ADDConfig::new();
    let count = fixture.len().saturating_sub(config.lookback());
    match mode {
        RustMode::Owned => Ok(Box::new(move || {
            let input = BinaryInput {
                real0: black_box(&fixture.close),
                real1: black_box(&fixture.auxiliary),
            };
            black_box(config.compute(input).map_err(display_error)?);
            Ok(())
        })),
        RustMode::CallerOwned => {
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = BinaryInput {
                    real0: black_box(&fixture.close),
                    real1: black_box(&fixture.auxiliary),
                };
                let range = config
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Prepared => {
            let mut runner = config.prepare_batch(fixture.len()).map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                let input = BinaryInput {
                    real0: black_box(&fixture.close),
                    real1: black_box(&fixture.auxiliary),
                };
                let range = runner
                    .compute_into(input, black_box(values.as_mut_slice()))
                    .map_err(display_error)?;
                black_box((range, values.as_slice()));
                Ok(())
            }))
        }
        RustMode::Streaming => {
            let mut stream = config.stream().map_err(display_error)?;
            let mut values = vec![0.0; count];
            Ok(Box::new(move || {
                stream.reset();
                let mut output_index = 0;
                for index in 0..fixture.len() {
                    let tick = BinaryTick {
                        real0: fixture.close[index],
                        real1: fixture.auxiliary[index],
                    };
                    if let Some(output) = stream.next(black_box(tick)).map_err(display_error)? {
                        values[output_index] = output;
                        output_index += 1;
                    }
                }
                black_box((&values[..output_index], output_index));
                Ok(())
            }))
        }
    }
}

fn python_semantic(
    python: &Path,
    fixture_dir: &Path,
    output_dir: &Path,
    spec: CaseSpec,
) -> Result<PythonSemantic, String> {
    fs::create_dir_all(output_dir)
        .map_err(|error| format!("create {}: {error}", output_dir.display()))?;
    let worker = python_worker();
    let output = run_command(
        python,
        &[
            worker.as_os_str(),
            std::ffi::OsStr::new("semantic"),
            fixture_dir.as_os_str(),
            output_dir.as_os_str(),
            std::ffi::OsStr::new(spec.id),
        ],
    )?;
    let metadata = parse_metadata(&output)?;
    let output_kind = required(&metadata, "output_kind")?;
    let arity = parse_metadata_value::<usize>(&metadata, "output_arity")?;
    let begin = parse_metadata_value::<usize>(&metadata, "output_begin")?;
    let count = parse_metadata_value::<usize>(&metadata, "output_count")?;
    if output_kind != spec.output_kind || arity != spec.output_arity {
        return Err(format!(
            "Python {} output identity mismatch: expected {} x{}, got {} x{}",
            spec.id, spec.output_kind, spec.output_arity, output_kind, arity
        ));
    }
    let values = if output_kind == "integer" {
        let mut columns = Vec::with_capacity(arity);
        for index in 0..arity {
            let path = output_dir.join(format!("column-{index}.i32le.bin"));
            let values = read_i32le(&path)?;
            if values.len() != count {
                return Err(format!(
                    "Python {} output column {index} contains {} values, expected {count}",
                    spec.id,
                    values.len()
                ));
            }
            columns.push(values);
        }
        OutputValues::Integer(columns)
    } else {
        let mut columns = Vec::with_capacity(arity);
        for index in 0..arity {
            let path = output_dir.join(format!("column-{index}.f64le.bin"));
            let values = read_f64le(&path)?;
            if values.len() != count {
                return Err(format!(
                    "Python {} output column {index} contains {} values, expected {count}",
                    spec.id,
                    values.len()
                ));
            }
            columns.push(values);
        }
        OutputValues::Float(columns)
    };
    Ok(PythonSemantic {
        provenance: PythonProvenance {
            python_version: required(&metadata, "python_version")?.to_owned(),
            binding_version: required(&metadata, "python_binding_version")?.to_owned(),
            ta_lib_version: required(&metadata, "python_ta_lib_version")?.to_owned(),
            numpy_version: required(&metadata, "numpy_version")?.to_owned(),
        },
        output: VerifiedOutput {
            begin,
            count,
            values,
        },
    })
}

fn python_timing(
    python: &Path,
    fixture_dir: &Path,
    spec: CaseSpec,
    samples: usize,
    warmup_ms: u64,
    sample_ms: u64,
) -> Result<Measurement, String> {
    let worker = python_worker();
    let sample_text = samples.to_string();
    let warmup_text = warmup_ms.to_string();
    let sample_ms_text = sample_ms.to_string();
    let output = run_command(
        python,
        &[
            worker.as_os_str(),
            std::ffi::OsStr::new("timing"),
            fixture_dir.as_os_str(),
            std::ffi::OsStr::new(spec.id),
            std::ffi::OsStr::new(&sample_text),
            std::ffi::OsStr::new(&warmup_text),
            std::ffi::OsStr::new(&sample_ms_text),
        ],
    )?;
    let metadata = parse_metadata(&output)?;
    let samples_ns = required(&metadata, "samples_ns")?
        .split(',')
        .map(|sample| {
            sample
                .parse::<f64>()
                .map_err(|error| format!("invalid Python timing sample {sample:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if samples_ns.len() != samples {
        return Err(format!(
            "Python {} timing returned {} samples, expected {samples}",
            spec.id,
            samples_ns.len()
        ));
    }
    Ok(Measurement {
        warmup_iterations: parse_metadata_value(&metadata, "warmup_iterations")?,
        iterations_per_sample: parse_metadata_value(&metadata, "iterations_per_sample")?,
        samples_ns,
    })
}

fn validate_python_provenance(python: &PythonProvenance, c_version: &str) -> Result<(), String> {
    if python.binding_version != PYTHON_BINDING_VERSION {
        return Err(format!(
            "pinned Python binding version mismatch: expected {PYTHON_BINDING_VERSION}, got {}",
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
            "Python/C TA-Lib version mismatch: Python reports {}, C reports {c_version}",
            python.ta_lib_version
        ));
    }
    Ok(())
}

fn python_worker() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/catalogue_python_worker.py")
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
            "{} failed with {status}: {}",
            program.display(),
            String::from_utf8_lossy(&stderr).trim()
        ));
    }
    String::from_utf8(stdout)
        .map_err(|error| format!("{} returned non-UTF-8 stdout: {error}", program.display()))
}

fn parse_metadata(output: &str) -> Result<std::collections::BTreeMap<String, String>, String> {
    let mut metadata = std::collections::BTreeMap::new();
    for line in output.lines().filter(|line| !line.is_empty()) {
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid worker metadata line {line:?}"))?;
        if metadata.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate worker metadata key {key:?}"));
        }
    }
    Ok(metadata)
}

fn required<'a>(
    metadata: &'a std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, String> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| format!("worker omitted {key}"))
}

fn parse_metadata_value<T>(
    metadata: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = required(metadata, key)?;
    value
        .parse()
        .map_err(|error| format!("invalid worker {key} value {value:?}: {error}"))
}

fn write_fixture(directory: &Path, fixture: &Fixture) -> Result<(), String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("create {}: {error}", directory.display()))?;
    for (name, values) in [
        ("open", fixture.open.as_slice()),
        ("high", fixture.high.as_slice()),
        ("low", fixture.low.as_slice()),
        ("close", fixture.close.as_slice()),
        ("volume", fixture.volume.as_slice()),
        ("auxiliary", fixture.auxiliary.as_slice()),
    ] {
        write_f64le(&directory.join(format!("{name}.f64le.bin")), values)?;
    }
    Ok(())
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
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() % 8 != 0 {
        return Err(format!(
            "{} does not contain whole f64 values",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect())
}

fn read_i32le(path: &Path) -> Result<Vec<i32>, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} does not contain whole i32 values",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
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

fn remove_dir_if_present(path: &Path) -> Result<(), String> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("remove stale {}: {error}", path.display())),
    }
}

#[derive(Debug)]
struct Provenance {
    rustc: String,
    cpu: String,
    os: String,
    commit: String,
    dirty: bool,
}

fn provenance() -> Provenance {
    Provenance {
        rustc: command_text("rustc", &["--version"]),
        cpu: cpu_model(),
        os: command_text("uname", &["-srv"]),
        commit: command_text("git", &["rev-parse", "HEAD"]),
        dirty: git_dirty(),
    }
}

fn command_text(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn git_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map_or(true, |output| {
            !output.status.success() || !output.stdout.is_empty()
        })
}

#[cfg(target_os = "macos")]
fn cpu_model() -> String {
    command_text("sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "linux")]
fn cpu_model() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|input| {
            input.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == "model name").then(|| value.trim().to_owned())
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
    let mut case = None;
    let mut input_length = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--python") => python = arguments.next().map(PathBuf::from),
            Some("--output-dir") => output_dir = arguments.next().map(PathBuf::from),
            Some("--samples") => samples = parse_argument(arguments.next(), "--samples")?,
            Some("--warmup-ms") => warmup_ms = parse_argument(arguments.next(), "--warmup-ms")?,
            Some("--sample-ms") => sample_ms = parse_argument(arguments.next(), "--sample-ms")?,
            Some("--case") => {
                let requested: String = parse_argument(arguments.next(), "--case")?;
                case = Some(requested.to_ascii_uppercase());
            }
            Some("--input-length") => {
                input_length = Some(parse_argument(arguments.next(), "--input-length")?)
            }
            Some(other) => return Err(format!("unknown argument {other:?}")),
            None => return Err("arguments must be valid UTF-8".to_owned()),
        }
    }
    if let Some(requested) = case.as_deref() {
        if !MATRIX.iter().any(|spec| spec.id == requested) {
            return Err(format!(
                "unknown --case {requested:?}; expected one of {}",
                MATRIX
                    .iter()
                    .map(|spec| spec.id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if let Some(requested) = input_length {
        if !INPUT_LENGTHS.contains(&requested) {
            return Err(format!(
                "unsupported --input-length {requested}; expected one of {}",
                INPUT_LENGTHS
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    Ok(Args {
        python: python.ok_or_else(|| "--python PATH is required".to_owned())?,
        output_dir: output_dir.ok_or_else(|| "--output-dir PATH is required".to_owned())?,
        samples,
        warmup_ms,
        sample_ms,
        case,
        input_length,
    })
}

fn parse_argument<T>(value: Option<std::ffi::OsString>, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = value.ok_or_else(|| format!("{name} requires a value"))?;
    let text = value
        .to_str()
        .ok_or_else(|| format!("{name} value must be valid UTF-8"))?;
    text.parse()
        .map_err(|error| format!("invalid {name} value {text:?}: {error}"))
}
