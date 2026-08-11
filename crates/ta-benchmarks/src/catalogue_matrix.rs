//! Deterministic fixtures, semantic gates, raw rows, statistics, and reporting
//! for the opt-in representative Indicator Catalogue comparison.

use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const INPUT_LENGTHS: [usize; 3] = [256, 4_096, 65_536];
pub const ABS_TOLERANCE: f64 = 1.0e-9;
pub const REL_TOLERANCE: f64 = 1.0e-12;
pub const FIXTURE_ID: &str = "catalogue_fixture_v1:f64le";
pub const RUST_OWNED_MODE: &str = "Owned Compact Output";
pub const RUST_CALLER_MODE: &str = "caller-owned Batch Computation";
pub const RUST_PREPARED_MODE: &str = "Prepared Batch Runner";
pub const RUST_STREAMING_MODE: &str = "Streaming Computation";
pub const C_DIRECT_MODE: &str = "direct C caller-owned";
pub const PYTHON_MODE: &str = "official Python NumPy API";
pub const PRIMARY_COMPARISON: &str = "primary caller-owned Rust/C kernel";
pub const RAW_HEADER: &str = "implementation\tindicator_family\tindicator_definition\tcase_id\tmode\tparameters\tinput_length\toutput_kind\toutput_arity\tmedian_ns\tci95_lower_ns\tci95_upper_ns\tthroughput_observations_per_second\tsample_count\toutlier_count\toutlier_low_count\toutlier_high_count\toutput_begin\toutput_count\toutput_checksum\tsemantic_status\tsemantic_reason\ttiming_status\ttiming_reason\tcomparison_status\tcomparison_reason\twarmup_iterations\titerations_per_sample\ttimed_boundary\tfixture\tinput_checksum\tta_lib_version\tta_lib_revision\tpython_version\tpython_binding_version\tpython_ta_lib_version\tnumpy_version\trustc\tcpu\tos\tarch\tfloat_width\tfeatures\tcommit\tdirty";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseKind {
    Sma,
    Bbands,
    Rsi,
    Macd,
    Atr,
    Adx,
    HtDcPhase,
    CdlDoji,
    CdlEngulfing,
    Cdl3WhiteSoldiers,
    LinearReg,
    TypPrice,
    Obv,
    Sin,
    Add,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseSpec {
    pub kind: CaseKind,
    pub id: &'static str,
    pub family: &'static str,
    pub definition: &'static str,
    pub parameters: &'static str,
    pub output_kind: &'static str,
    pub output_arity: usize,
}

pub use crate::pattern_shapes::{PatternShapeSpec, PATTERN_SHAPES};

pub const MATRIX: [CaseSpec; 15] = [
    CaseSpec {
        kind: CaseKind::Sma,
        id: "SMA",
        family: "Overlap Studies",
        definition: "SMA: Simple Moving Average",
        parameters: "timeperiod=14",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Bbands,
        id: "BBANDS",
        family: "Overlap Studies",
        definition: "BBANDS: Bollinger Bands",
        parameters: "timeperiod=20;nbdevup=2;nbdevdn=2;matype=SMA",
        output_kind: "float",
        output_arity: 3,
    },
    CaseSpec {
        kind: CaseKind::Rsi,
        id: "RSI",
        family: "Momentum Indicators",
        definition: "RSI: Relative Strength Index",
        parameters: "timeperiod=14",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Macd,
        id: "MACD",
        family: "Momentum Indicators",
        definition: "MACD: Moving Average Convergence/Divergence",
        parameters: "fastperiod=12;slowperiod=26;signalperiod=9",
        output_kind: "float",
        output_arity: 3,
    },
    CaseSpec {
        kind: CaseKind::Atr,
        id: "ATR",
        family: "Volatility Indicators",
        definition: "ATR: Average True Range",
        parameters: "timeperiod=14",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Adx,
        id: "ADX",
        family: "Momentum Indicators",
        definition: "ADX: Average Directional Movement Index",
        parameters: "timeperiod=14",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::HtDcPhase,
        id: "HT_DCPHASE",
        family: "Cycle Indicators",
        definition: "HT_DCPHASE: Hilbert Transform Dominant Cycle Phase",
        parameters: "none",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::CdlDoji,
        id: "CDLDOJI",
        family: "Pattern Recognition",
        definition: "CDLDOJI: Doji",
        parameters: "candle_settings=TA-Lib defaults",
        output_kind: "integer",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::CdlEngulfing,
        id: "CDLENGULFING",
        family: "Pattern Recognition",
        definition: "CDLENGULFING: Engulfing Pattern",
        parameters: "candle_settings=TA-Lib defaults",
        output_kind: "integer",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Cdl3WhiteSoldiers,
        id: "CDL3WHITESOLDIERS",
        family: "Pattern Recognition",
        definition: "CDL3WHITESOLDIERS: Three Advancing White Soldiers",
        parameters: "candle_settings=TA-Lib defaults",
        output_kind: "integer",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::LinearReg,
        id: "LINEARREG",
        family: "Statistic Functions",
        definition: "LINEARREG: Linear Regression endpoint",
        parameters: "timeperiod=14",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::TypPrice,
        id: "TYPPRICE",
        family: "Price Transform",
        definition: "TYPPRICE: Typical Price",
        parameters: "none",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Obv,
        id: "OBV",
        family: "Volume Indicators",
        definition: "OBV: On Balance Volume",
        parameters: "none",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Sin,
        id: "SIN",
        family: "Math Transform",
        definition: "SIN: Vector Trigonometric Sine",
        parameters: "none",
        output_kind: "float",
        output_arity: 1,
    },
    CaseSpec {
        kind: CaseKind::Add,
        id: "ADD",
        family: "Math Operators",
        definition: "ADD: Vector Arithmetic Add",
        parameters: "none",
        output_kind: "float",
        output_arity: 1,
    },
];

#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub auxiliary: Vec<f64>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        let length = self.close.len();
        for (name, values) in [
            ("open", &self.open),
            ("high", &self.high),
            ("low", &self.low),
            ("volume", &self.volume),
            ("auxiliary", &self.auxiliary),
        ] {
            if values.len() != length {
                return Err(format!(
                    "fixture {name} length {} differs from close length {length}",
                    values.len()
                ));
            }
        }
        for index in 0..length {
            let values = [
                self.open[index],
                self.high[index],
                self.low[index],
                self.close[index],
                self.volume[index],
                self.auxiliary[index],
            ];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "fixture contains a non-finite value at index {index}"
                ));
            }
            if self.high[index] < self.open[index].max(self.close[index]) {
                return Err(format!(
                    "fixture high violates OHLC invariants at index {index}"
                ));
            }
            if self.low[index] > self.open[index].min(self.close[index]) {
                return Err(format!(
                    "fixture low violates OHLC invariants at index {index}"
                ));
            }
            if self.volume[index] < 0.0 {
                return Err(format!("fixture volume is negative at index {index}"));
            }
        }
        Ok(())
    }
}

pub use crate::fixture::series_fixture;

pub fn catalogue_fixture(size: usize) -> Fixture {
    let close = series_fixture(size, 0);
    let auxiliary = series_fixture(size, 2)
        .into_iter()
        .enumerate()
        .map(|(index, value)| value * 0.75 + (index % 13) as f64 * 0.02)
        .collect::<Vec<_>>();
    let open = close
        .iter()
        .enumerate()
        .map(|(index, value)| value + ((index % 9) as f64 - 4.0) * 0.035)
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| open.max(*close) + 0.5 + (index % 11) as f64 * 0.03)
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| open.min(*close) - 0.5 - (index % 7) as f64 * 0.025)
        .collect::<Vec<_>>();
    let volume = series_fixture(size, 1)
        .into_iter()
        .map(|value| 10_000.0 + value * 100.0)
        .collect::<Vec<_>>();
    Fixture {
        open,
        high,
        low,
        close,
        volume,
        auxiliary,
    }
}

pub fn input_checksum(values: &[f64]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    hash_f64s(&mut hash, values);
    format!("fnv1a64:{hash:016x}")
}

pub fn fixture_checksum(fixture: &Fixture) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for (name, values) in [
        ("open", fixture.open.as_slice()),
        ("high", fixture.high.as_slice()),
        ("low", fixture.low.as_slice()),
        ("close", fixture.close.as_slice()),
        ("volume", fixture.volume.as_slice()),
        ("auxiliary", fixture.auxiliary.as_slice()),
    ] {
        hash_bytes(&mut hash, name.as_bytes());
        hash_bytes(&mut hash, &(values.len() as u64).to_le_bytes());
        hash_f64s(&mut hash, values);
    }
    format!("fnv1a64:{hash:016x}")
}

fn hash_f64s(hash: &mut u64, values: &[f64]) {
    for value in values {
        hash_bytes(hash, &value.to_le_bytes());
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputValues {
    Float(Vec<Vec<f64>>),
    Integer(Vec<Vec<i32>>),
}

impl OutputValues {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Float(_) => "float",
            Self::Integer(_) => "integer",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Self::Float(columns) => columns.len(),
            Self::Integer(columns) => columns.len(),
        }
    }

    pub fn column_len(&self) -> Result<usize, String> {
        let lengths = match self {
            Self::Float(columns) => columns.iter().map(Vec::len).collect::<Vec<_>>(),
            Self::Integer(columns) => columns.iter().map(Vec::len).collect::<Vec<_>>(),
        };
        let Some(&first) = lengths.first() else {
            return Err("output has no columns".to_owned());
        };
        if lengths.iter().any(|length| *length != first) {
            return Err(format!("output columns have unequal lengths: {lengths:?}"));
        }
        Ok(first)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedOutput {
    pub begin: usize,
    pub count: usize,
    pub values: OutputValues,
}

impl VerifiedOutput {
    pub fn validate_shape(&self) -> Result<(), String> {
        let actual = self.values.column_len()?;
        if actual != self.count {
            return Err(format!(
                "declared output count {} differs from column length {actual}",
                self.count
            ));
        }
        Ok(())
    }

    pub fn checksum(&self) -> String {
        let mut hash = 0xcbf29ce484222325_u64;
        hash_bytes(&mut hash, self.values.kind().as_bytes());
        hash_bytes(&mut hash, &(self.begin as u64).to_le_bytes());
        hash_bytes(&mut hash, &(self.count as u64).to_le_bytes());
        match &self.values {
            OutputValues::Float(columns) => {
                for column in columns {
                    hash_f64s(&mut hash, column);
                }
            }
            OutputValues::Integer(columns) => {
                for column in columns {
                    for value in column {
                        hash_bytes(&mut hash, &value.to_le_bytes());
                    }
                }
            }
        }
        format!("fnv1a64:{hash:016x}")
    }
}

pub fn validate_outputs(
    expected: &VerifiedOutput,
    implementation: &str,
    actual: &VerifiedOutput,
) -> Result<(), String> {
    expected
        .validate_shape()
        .map_err(|error| format!("reference output shape: {error}"))?;
    actual
        .validate_shape()
        .map_err(|error| format!("{implementation} output shape: {error}"))?;
    if (actual.begin, actual.count) != (expected.begin, expected.count) {
        return Err(format!(
            "{implementation} OutputRange mismatch: expected begin {} count {}, got begin {} count {}",
            expected.begin, expected.count, actual.begin, actual.count
        ));
    }
    if actual.values.kind() != expected.values.kind() {
        return Err(format!(
            "{implementation} output kind mismatch: expected {}, got {}",
            expected.values.kind(),
            actual.values.kind()
        ));
    }
    if actual.values.arity() != expected.values.arity() {
        return Err(format!(
            "{implementation} output arity mismatch: expected {}, got {}",
            expected.values.arity(),
            actual.values.arity()
        ));
    }
    match (&expected.values, &actual.values) {
        (OutputValues::Float(expected_columns), OutputValues::Float(actual_columns)) => {
            for (column, (expected_values, actual_values)) in
                expected_columns.iter().zip(actual_columns).enumerate()
            {
                for (index, (&expected_value, &actual_value)) in
                    expected_values.iter().zip(actual_values).enumerate()
                {
                    if !expected_value.is_finite() || !actual_value.is_finite() {
                        if expected_value.to_bits() != actual_value.to_bits() {
                            return Err(format!("{implementation} non-finite placement mismatch at column {column} compact index {index}: expected {expected_value:?}, got {actual_value:?}"));
                        }
                        continue;
                    }
                    let difference = (actual_value - expected_value).abs();
                    let tolerance = ABS_TOLERANCE.max(REL_TOLERANCE * expected_value.abs());
                    if difference > tolerance {
                        return Err(format!("{implementation} value mismatch at column {column} compact index {index}: expected {expected_value:.17e}, got {actual_value:.17e}, difference {difference:.3e}, tolerance {tolerance:.3e}"));
                    }
                }
            }
        }
        (OutputValues::Integer(expected_columns), OutputValues::Integer(actual_columns)) => {
            for (column, (expected_values, actual_values)) in
                expected_columns.iter().zip(actual_columns).enumerate()
            {
                for (index, (&expected_value, &actual_value)) in
                    expected_values.iter().zip(actual_values).enumerate()
                {
                    if actual_value != expected_value {
                        return Err(format!("{implementation} exact integer mismatch at column {column} compact index {index}: expected {expected_value}, got {actual_value}"));
                    }
                }
            }
        }
        _ => unreachable!("output kinds were checked above"),
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimingStats {
    pub median_ns: f64,
    pub ci95_lower_ns: f64,
    pub ci95_upper_ns: f64,
    pub throughput_observations_per_second: f64,
    pub sample_count: usize,
    pub outlier_count: usize,
    pub outlier_low_count: usize,
    pub outlier_high_count: usize,
}

pub fn timing_stats(samples_ns: &[f64], input_length: usize) -> Result<TimingStats, String> {
    if samples_ns.len() < 2 {
        return Err("at least two timing samples are required".to_owned());
    }
    if samples_ns
        .iter()
        .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err("timing samples must be positive and finite".to_owned());
    }
    let mut sorted = samples_ns.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let median_ns = quantile(&sorted, 0.5);
    let q1 = quantile(&sorted, 0.25);
    let q3 = quantile(&sorted, 0.75);
    let iqr = q3 - q1;
    let low_fence = q1 - 1.5 * iqr;
    let high_fence = q3 + 1.5 * iqr;
    let outlier_low_count = sorted
        .iter()
        .take_while(|sample| **sample < low_fence)
        .count();
    let outlier_high_count = sorted
        .iter()
        .rev()
        .take_while(|sample| **sample > high_fence)
        .count();
    let (ci95_lower_ns, ci95_upper_ns) = bootstrap_median_ci(samples_ns);
    Ok(TimingStats {
        median_ns,
        ci95_lower_ns,
        ci95_upper_ns,
        throughput_observations_per_second: input_length as f64 * 1.0e9 / median_ns,
        sample_count: samples_ns.len(),
        outlier_count: outlier_low_count + outlier_high_count,
        outlier_low_count,
        outlier_high_count,
    })
}

fn bootstrap_median_ci(samples: &[f64]) -> (f64, f64) {
    const REPLICATES: usize = 10_000;
    let mut state = 0x636174616c6f6755_u64 ^ samples.len() as u64;
    let mut resample = vec![0.0; samples.len()];
    let mut medians = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        for value in &mut resample {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = samples[(state as usize) % samples.len()];
        }
        resample.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        medians.push(quantile(&resample, 0.5));
    }
    medians.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    (quantile(&medians, 0.025), quantile(&medians, 0.975))
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRow {
    pub implementation: String,
    pub indicator_family: String,
    pub indicator_definition: String,
    pub case_id: String,
    pub mode: String,
    pub parameters: String,
    pub input_length: usize,
    pub output_kind: String,
    pub output_arity: Option<usize>,
    pub stats: Option<TimingStats>,
    pub output_begin: Option<usize>,
    pub output_count: Option<usize>,
    pub output_checksum: String,
    pub semantic_status: String,
    pub semantic_reason: String,
    pub timing_status: String,
    pub timing_reason: String,
    pub comparison_status: String,
    pub comparison_reason: String,
    pub warmup_iterations: Option<u64>,
    pub iterations_per_sample: Option<u64>,
    pub timed_boundary: String,
    pub fixture: String,
    pub input_checksum: String,
    pub ta_lib_version: String,
    pub ta_lib_revision: String,
    pub python_version: String,
    pub python_binding_version: String,
    pub python_ta_lib_version: String,
    pub numpy_version: String,
    pub rustc: String,
    pub cpu: String,
    pub os: String,
    pub arch: String,
    pub float_width: usize,
    pub features: String,
    pub commit: String,
    pub dirty: bool,
}

pub fn write_raw_rows(path: &Path, rows: &[BenchmarkRow]) -> Result<(), String> {
    let mut output = String::from(RAW_HEADER);
    output.push('\n');
    for (index, row) in rows.iter().enumerate() {
        validate_benchmark_row_evidence(row)
            .map_err(|error| format!("raw row {}: {error}", index + 2))?;
        output.push_str(&format_row(row));
        output.push('\n');
    }
    fs::write(path, output).map_err(|error| format!("write {}: {error}", path.display()))
}

pub fn read_raw_rows(path: &Path) -> Result<Vec<BenchmarkRow>, String> {
    let input =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_raw_rows(&input)
}

pub fn parse_raw_rows(input: &str) -> Result<Vec<BenchmarkRow>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(RAW_HEADER) {
        return Err("unexpected Indicator Catalogue matrix raw-row header".to_owned());
    }
    lines
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
        .map(|(index, line)| {
            parse_row(line).map_err(|error| format!("raw row {}: {error}", index + 2))
        })
        .collect()
}

pub const OPTIMIZATION_EVIDENCE_HEADER: &str = "ticket\tcase_ids\tfocused_commands\thypotheses\tconfirmed_evidence_kind\tconfirmed_evidence\tneighboring_workloads\tneighboring_disposition";

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationEvidenceRow {
    pub ticket: String,
    pub case_ids: Vec<String>,
    pub focused_commands: Vec<String>,
    pub hypotheses: Vec<String>,
    pub confirmed_evidence_kind: String,
    pub confirmed_evidence: String,
    pub neighboring_workloads: String,
    pub neighboring_disposition: String,
}

pub fn read_optimization_evidence(path: &Path) -> Result<Vec<OptimizationEvidenceRow>, String> {
    let input =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_optimization_evidence(&input)
}

pub fn parse_optimization_evidence(input: &str) -> Result<Vec<OptimizationEvidenceRow>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(OPTIMIZATION_EVIDENCE_HEADER) {
        return Err("unexpected optimization evidence header".to_owned());
    }
    lines
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
        .map(|(index, line)| {
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() != 8 {
                return Err(format!(
                    "optimization evidence row {} has {} fields, expected 8",
                    index + 2,
                    fields.len()
                ));
            }
            let case_ids = split_evidence_list(fields[1]);
            let focused_commands = split_evidence_list(fields[2]);
            let hypotheses = split_evidence_list(fields[3]);
            if case_ids.is_empty() || case_ids.len() != focused_commands.len() {
                return Err(format!(
                    "optimization evidence row {} must have one focused command per case",
                    index + 2
                ));
            }
            if !(3..=5).contains(&hypotheses.len()) {
                return Err(format!(
                    "optimization evidence row {} must contain 3 to 5 ranked hypotheses",
                    index + 2
                ));
            }
            if fields[4] != "source" && fields[4] != "profile" && fields[4] != "disassembly" {
                return Err(format!(
                    "optimization evidence row {} has unsupported confirmed evidence kind {:?}",
                    index + 2,
                    fields[4]
                ));
            }
            Ok(OptimizationEvidenceRow {
                ticket: fields[0].to_owned(),
                case_ids,
                focused_commands,
                hypotheses,
                confirmed_evidence_kind: fields[4].to_owned(),
                confirmed_evidence: fields[5].to_owned(),
                neighboring_workloads: fields[6].to_owned(),
                neighboring_disposition: fields[7].to_owned(),
            })
        })
        .collect()
}

fn split_evidence_list(value: &str) -> Vec<String> {
    value
        .split(" || ")
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::to_owned)
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlatformQualification {
    pub artifact: String,
    pub platform: String,
    pub precision: String,
    pub runtime: String,
    pub profile: String,
    pub cpu: String,
    pub os: String,
    pub commit: String,
    pub workflow_run_id: String,
    pub workflow_run_url: String,
    pub workflow_job: String,
    pub active_backend: String,
    pub feature_flags: String,
    pub measurements: Vec<QualificationMeasurement>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualificationMeasurement {
    pub mode: String,
    pub backend: String,
    pub input_length: usize,
    pub equivalent_to_scalar: bool,
    pub semantic_status: String,
    pub timing_status: String,
    pub median_ns: f64,
    pub ci95_lower_ns: f64,
    pub ci95_upper_ns: f64,
    pub throughput_observations_per_second: f64,
    pub sample_count: usize,
    pub timed_boundary: String,
}

pub fn read_platform_qualification(path: &Path) -> Result<PlatformQualification, String> {
    let input =
        fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    parse_platform_qualification(&input, &path.display().to_string())
}

pub fn parse_platform_qualification(
    input: &str,
    artifact: &str,
) -> Result<PlatformQualification, String> {
    let mut metadata = None;
    let mut has_aggregate_validation = false;
    let mut validated_backends = BTreeSet::new();
    let mut measurements = Vec::new();
    for (index, line) in input
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
    {
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("{artifact} JSONL row {}: {error}", index + 1))?;
        match json_string(&value, "record")?.as_str() {
            "metadata" => {
                if metadata.is_some() {
                    return Err(format!("{artifact} has more than one metadata record"));
                }
                let active_backend = value
                    .get("active_backend")
                    .and_then(Value::as_str)
                    .or_else(|| value.get("simd_backend").and_then(Value::as_str))
                    .ok_or_else(|| format!("{artifact} metadata is missing a runtime backend"))?
                    .to_owned();
                let feature_flags = value
                    .get("features")
                    .and_then(Value::as_str)
                    .filter(|features| !features.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        let cargo = value
                            .get("cargo_features")?
                            .as_str()
                            .filter(|features| !features.is_empty())?;
                        let target = value
                            .get("target_features")?
                            .as_str()
                            .filter(|features| !features.is_empty())?;
                        Some(format!("cargo_features={cargo}; target_features={target}"))
                    })
                    .or_else(|| {
                        let flags = ["scalar_feature_flags", "simd_feature_flags"]
                            .into_iter()
                            .map(|key| {
                                value
                                    .get(key)
                                    .and_then(Value::as_str)
                                    .filter(|flag| !flag.is_empty())
                                    .map(|flag| format!("{key}={flag}"))
                            })
                            .collect::<Option<Vec<_>>>()?;
                        Some(flags.join("; "))
                    })
                    .ok_or_else(|| {
                        format!("{artifact} metadata is missing string field \"features\"")
                    })?;
                let profile = value
                    .get("profile")
                    .and_then(Value::as_str)
                    .or_else(|| value.get("rust_profile").and_then(Value::as_str))
                    .filter(|profile| !profile.is_empty())
                    .ok_or_else(|| {
                        format!("{artifact} metadata is missing string field \"profile\"")
                    })?
                    .to_owned();
                metadata = Some((
                    json_string(&value, "platform")?,
                    json_nonempty_string(&value, "precision")?,
                    json_string(&value, "runtime")?,
                    profile,
                    json_string(&value, "cpu")?,
                    json_nonempty_string(&value, "os")?,
                    json_string(&value, "commit")?,
                    json_identifier(&value, "workflow_run_id")?,
                    json_string(&value, "workflow_run_url")?,
                    json_identifier_alias(&value, "workflow_job", "workflow_job_id")?,
                    active_backend,
                    feature_flags,
                ));
            }
            "measurement" => {
                let measurement = QualificationMeasurement {
                    mode: json_string(&value, "mode")?,
                    backend: json_string(&value, "backend")?,
                    input_length: json_u64(&value, "input_length")? as usize,
                    equivalent_to_scalar: json_bool(&value, "equivalent_to_scalar")?,
                    semantic_status: json_string(&value, "semantic_status")?,
                    timing_status: json_string(&value, "timing_status")?,
                    median_ns: json_f64(&value, "median_ns")?,
                    ci95_lower_ns: json_f64(&value, "ci95_lower_ns")?,
                    ci95_upper_ns: json_f64(&value, "ci95_upper_ns")?,
                    throughput_observations_per_second: json_f64(
                        &value,
                        "throughput_observations_per_second",
                    )?,
                    sample_count: json_u64(&value, "sample_count")? as usize,
                    timed_boundary: json_string(&value, "timed_boundary")?,
                };
                if !INPUT_LENGTHS.contains(&measurement.input_length) {
                    return Err(format!(
                        "{artifact} has unsupported input length {}",
                        measurement.input_length
                    ));
                }
                validate_positive_timing_evidence(
                    measurement.median_ns,
                    measurement.ci95_lower_ns,
                    measurement.ci95_upper_ns,
                    measurement.throughput_observations_per_second,
                    measurement.sample_count,
                    measurement.input_length,
                )
                .map_err(|error| format!("{artifact} measurement row {}: {error}", index + 1))?;
                measurements.push(measurement);
            }
            "validation" => match validate_qualification_validation(&value, artifact, index + 1)? {
                Some(backend) if !validated_backends.insert(backend.clone()) => {
                    return Err(format!(
                        "{artifact} has duplicate validation records for backend {backend:?}"
                    ));
                }
                Some(_) => {}
                None if has_aggregate_validation => {
                    return Err(format!(
                        "{artifact} has more than one aggregate validation record"
                    ));
                }
                None => has_aggregate_validation = true,
            },
            other => return Err(format!("{artifact} has unsupported record type {other:?}")),
        }
    }
    if !has_aggregate_validation && validated_backends.is_empty() {
        return Err(format!("{artifact} has no validation records"));
    }
    let (
        platform,
        precision,
        runtime,
        profile,
        cpu,
        os,
        commit,
        workflow_run_id,
        workflow_run_url,
        workflow_job,
        active_backend,
        feature_flags,
    ) = metadata.ok_or_else(|| format!("{artifact} has no metadata record"))?;
    if measurements.is_empty() {
        return Err(format!("{artifact} has no measurement records"));
    }
    for measurement in &measurements {
        if measurement.equivalent_to_scalar
            && !(measurement.backend == "ta-lib-c" && measurement.mode == "direct C caller-owned")
            && !has_aggregate_validation
            && !validated_backends.contains(&measurement.backend)
        {
            return Err(format!(
                "{artifact} reports scalar equivalence for backend {:?} without a matching validation record",
                measurement.backend
            ));
        }
    }
    Ok(PlatformQualification {
        artifact: artifact.to_owned(),
        platform,
        precision,
        runtime,
        cpu,
        os,
        profile,
        commit,
        workflow_run_id,
        workflow_run_url,
        workflow_job,
        active_backend,
        feature_flags,
        measurements,
    })
}

fn json_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("JSON record is missing string field {key:?}"))
}

fn json_nonempty_string(value: &Value, key: &str) -> Result<String, String> {
    let parsed = json_string(value, key)?;
    if parsed.is_empty() {
        return Err(format!("JSON record has empty string field {key:?}"));
    }
    Ok(parsed)
}

fn json_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("JSON record is missing integer field {key:?}"))
}

fn json_identifier(value: &Value, key: &str) -> Result<String, String> {
    match value.get(key) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        Some(Value::Number(value)) if value.is_u64() => Ok(value.to_string()),
        _ => Err(format!(
            "JSON record is missing string or integer field {key:?}"
        )),
    }
}

fn json_identifier_alias(value: &Value, key: &str, alias: &str) -> Result<String, String> {
    json_identifier(value, key).or_else(|_| json_identifier(value, alias))
}

fn json_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("JSON record is missing numeric field {key:?}"))
}

fn validate_qualification_validation(
    value: &Value,
    artifact: &str,
    row_number: usize,
) -> Result<Option<String>, String> {
    let invalid = |reason: &str| format!("{artifact} validation row {row_number} {reason}");
    if value.get("exact_scalar_equivalence").is_some() {
        for key in [
            "public_boundary",
            "exact_scalar_equivalence",
            "error_semantics_verified",
            "mismatched_length_error_equal_to_scalar",
            "non_finite_error_equal_to_scalar",
        ] {
            if !json_bool(value, key)? {
                return Err(invalid(&format!("has {key}=false")));
            }
        }
        let backend = json_nonempty_string(value, "backend")?;
        let observed_backend = json_string(value, "observed_backend")?;
        if backend != observed_backend {
            return Err(invalid("did not observe its requested backend"));
        }
        for key in ["precision", "mode"] {
            if json_string(value, key)?.is_empty() {
                return Err(invalid(&format!("has empty {key}")));
            }
        }
        return Ok(Some(backend));
    } else if value.get("errors_match_scalar").is_some() {
        for key in [
            "public_boundary",
            "unequal_lengths_verified",
            "non_finite_verified",
            "short_output_verified",
            "errors_match_scalar",
        ] {
            if !json_bool(value, key)? {
                return Err(invalid(&format!("has {key}=false")));
            }
        }
        let backend = json_nonempty_string(value, "backend")?;
        for key in [
            "unequal_lengths_error",
            "non_finite_error",
            "short_output_error",
        ] {
            if json_string(value, key)?.is_empty() {
                return Err(invalid(&format!("has empty {key}")));
            }
        }
        return Ok(Some(backend));
    } else {
        for key in ["unequal_lengths_verified", "non_finite_verified"] {
            if !json_bool(value, key)? {
                return Err(invalid(&format!("has {key}=false")));
            }
        }
        for key in [
            "scalar_unequal_lengths_error",
            "scalar_non_finite_error",
            "simd_unequal_lengths_error",
            "simd_non_finite_error",
        ] {
            if json_string(value, key)?.is_empty() {
                return Err(invalid(&format!("has empty {key}")));
            }
        }
    }
    Ok(None)
}

fn json_bool(value: &Value, key: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("JSON record is missing boolean field {key:?}"))
}

pub fn render_report(rows: &[BenchmarkRow]) -> Result<String, String> {
    if rows.is_empty() {
        return Err(
            "cannot render an Indicator Catalogue matrix report without raw rows".to_owned(),
        );
    }
    let first = &rows[0];
    let mut report =
        String::from("Pinned representative Indicator Catalogue performance matrix\n\n");
    report.push_str(&format!(
        "TA-Lib {} ({}) | Python {} binding {} / core {} | NumPy {} | float {}-bit\n",
        first.ta_lib_version,
        first.ta_lib_revision,
        first.python_version,
        first.python_binding_version,
        first.python_ta_lib_version,
        first.numpy_version,
        first.float_width
    ));
    report.push_str(&format!(
        "Commit {} (dirty: {}) | {} | {} {} | {}\n",
        first.commit, first.dirty, first.cpu, first.os, first.arch, first.rustc
    ));
    report.push_str(if first.dirty {
        "Run classification: diagnostic only; a dirty run cannot replace the canonical baseline.\n"
    } else {
        "Run classification: clean reference run; completeness and publication status follow.\n"
    });
    report.push_str("95% intervals are deterministic bootstrap confidence intervals for the median (10,000 resamples). Outliers use Tukey's 1.5 IQR fences. Rust/C ratios below use only same-run caller-owned rows with identical case, parameters, fixture, and input length.\n\n");

    report.push_str("Representative matrix\n\n| Family | Definition | Parameters | Output |\n|---|---|---|---|\n");
    let mut matrix = BTreeSet::new();
    for row in rows {
        matrix.insert((
            row.indicator_family.clone(),
            row.case_id.clone(),
            row.parameters.clone(),
            format!(
                "{} x{}",
                row.output_kind,
                row.output_arity
                    .map_or_else(|| "NA".to_owned(), |value| value.to_string())
            ),
        ));
    }
    for (family, case_id, parameters, output) in matrix {
        report.push_str(&format!(
            "| {family} | {case_id} | {parameters} | {output} |\n"
        ));
    }

    let case_ids = rows
        .iter()
        .map(|row| row.case_id.as_str())
        .collect::<BTreeSet<_>>();
    report.push_str("\nPattern Recognition execution-shape coverage\n\n| Definition | Execution shape | Rationale |\n|---|---|---|\n");
    for shape in PATTERN_SHAPES {
        if case_ids.contains(shape.case_id) {
            report.push_str(&format!(
                "| {} | {} | {} |\n",
                shape.case_id, shape.execution_shape, shape.rationale
            ));
        }
    }

    let pairs = primary_pairs(rows);
    report.push_str("\nSame-run geometric Rust/C caller-owned summary\n\n| Input | Mode | Comparable cases | Geometric Rust/C latency ratio | Semantics |\n|---:|---|---:|---:|---|\n");
    for input_length in INPUT_LENGTHS {
        let ratios = pairs
            .iter()
            .filter(|pair| pair.input_length == input_length)
            .map(|pair| pair.ratio)
            .collect::<Vec<_>>();
        if ratios.is_empty() {
            report.push_str(&format!("| {input_length} | {RUST_CALLER_MODE} vs {C_DIRECT_MODE} | 0 | unavailable | no comparable measured pairs |\n"));
        } else {
            let geometric =
                (ratios.iter().map(|ratio| ratio.ln()).sum::<f64>() / ratios.len() as f64).exp();
            report.push_str(&format!("| {input_length} | {RUST_CALLER_MODE} vs {C_DIRECT_MODE} | {} | {geometric:.3}x | comparable only |\n", ratios.len()));
        }
    }

    report.push_str("\nLarge-throughput optimization ordering\n\n| Rank | Definition | Rust/C latency ratio | Disposition |\n|---:|---|---:|---|\n");
    let mut large = pairs
        .into_iter()
        .filter(|pair| pair.input_length == 65_536)
        .collect::<Vec<_>>();
    large.sort_by(|left, right| {
        right
            .ratio
            .partial_cmp(&left.ratio)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.case_id.cmp(&right.case_id))
    });
    if large.is_empty() {
        report.push_str("| - | - | unavailable | no comparable measured pairs |\n");
    } else {
        for (index, pair) in large.iter().enumerate() {
            let disposition = if pair.ratio > 1.05 {
                "remaining comparative gap above 5%"
            } else if pair.ratio < 0.95 {
                "fast-ta faster"
            } else {
                "parity band"
            };
            report.push_str(&format!(
                "| {} | {} | {:.3}x | {disposition} |\n",
                index + 1,
                pair.case_id,
                pair.ratio
            ));
        }
    }

    report.push_str("\nDetailed raw-row projection\n\n| Definition | Input | Implementation | Mode | Semantic | Comparison | Median | 95% CI | Throughput | Output Range |\n|---|---:|---|---|---|---|---:|---:|---:|---:|\n");
    let mut ordered = rows.to_vec();
    ordered.sort_by(|left, right| {
        left.case_id
            .cmp(&right.case_id)
            .then_with(|| left.input_length.cmp(&right.input_length))
            .then_with(|| left.implementation.cmp(&right.implementation))
            .then_with(|| left.mode.cmp(&right.mode))
    });
    for row in &ordered {
        let (median, ci, throughput) = if let Some(stats) = &row.stats {
            (
                format!("{:.3} us", stats.median_ns / 1_000.0),
                format!(
                    "[{:.3}, {:.3}] us",
                    stats.ci95_lower_ns / 1_000.0,
                    stats.ci95_upper_ns / 1_000.0
                ),
                format!(
                    "{:.3} Mobs/s",
                    stats.throughput_observations_per_second / 1.0e6
                ),
            )
        } else {
            (
                "unavailable".to_owned(),
                clean(&row.timing_reason),
                "unavailable".to_owned(),
            )
        };
        let range = match (row.output_begin, row.output_count) {
            (Some(begin), Some(count)) => format!("{begin}..{}", begin + count),
            _ => "unavailable".to_owned(),
        };
        let semantic = if row.semantic_reason.is_empty() {
            row.semantic_status.clone()
        } else {
            format!("{}: {}", row.semantic_status, clean(&row.semantic_reason))
        };
        let comparison = if row.comparison_reason.is_empty() {
            row.comparison_status.clone()
        } else {
            format!(
                "{}: {}",
                row.comparison_status,
                clean(&row.comparison_reason)
            )
        };
        report.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.case_id,
            row.input_length,
            row.implementation,
            row.mode,
            semantic,
            comparison,
            median,
            ci,
            throughput,
            range
        ));
    }

    report.push_str("\nSuppressed or unavailable results\n\n");
    let failures = ordered
        .iter()
        .filter(|row| row.semantic_status != "verified" || row.timing_status != "measured")
        .collect::<Vec<_>>();
    if failures.is_empty() {
        report.push_str("None. Every matrix row passed semantic verification before timing.\n");
    } else {
        for row in failures {
            report.push_str(&format!(
                "- {}/{}/{}: semantic={} ({}) timing={} ({})\n",
                row.case_id,
                row.input_length,
                row.mode,
                row.semantic_status,
                clean(&row.semantic_reason),
                row.timing_status,
                clean(&row.timing_reason)
            ));
        }
    }
    Ok(report)
}

pub fn render_report_with_comparison(
    rows: &[BenchmarkRow],
    baseline_rows: &[BenchmarkRow],
    optimization_evidence: &[OptimizationEvidenceRow],
    platform_qualifications: &[PlatformQualification],
) -> Result<String, String> {
    if baseline_rows.is_empty() {
        return Err("clean pre-optimization baseline rows are required".to_owned());
    }
    if baseline_rows.iter().any(|row| row.dirty) {
        return Err("the canonical pre-optimization baseline must be clean".to_owned());
    }
    let report = render_report(rows)?;
    let marker = "\nDetailed raw-row projection\n";
    let (summary, details) = report
        .split_once(marker)
        .ok_or_else(|| "generated report is missing the detailed-row marker".to_owned())?;
    let first = rows
        .first()
        .ok_or_else(|| "cannot render a comparison report without raw rows".to_owned())?;
    let baseline_first = baseline_rows
        .first()
        .ok_or_else(|| "clean pre-optimization baseline rows are required".to_owned())?;
    let mut comparison = String::new();

    let verified = rows
        .iter()
        .filter(|row| row.semantic_status == "verified")
        .count();
    let measured = rows
        .iter()
        .filter(|row| row.timing_status == "measured")
        .count();
    let sample_counts = rows
        .iter()
        .filter_map(|row| row.stats.as_ref().map(|stats| stats.sample_count))
        .collect::<BTreeSet<_>>();
    let clean_run = rows.iter().all(|row| !row.dirty);
    comparison.push_str(
        "\nRun completeness and semantic gate\n\n| Raw rows | Semantic verified | Measured | Sample counts | Provenance |\n|---:|---:|---:|---|---|\n",
    );
    comparison.push_str(&format!(
        "| {} | {verified} | {measured} | {} | {} |\n",
        rows.len(),
        sample_counts
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if clean_run {
            "clean canonical final run"
        } else {
            "dirty diagnostic run"
        }
    ));
    comparison.push_str(&format!(
        "\nCanonical comparison baseline\n\nThe committed clean pre-optimization matrix is the sole comparison baseline: commit {} (dirty: {}), {} rows, {} on {} {}. The historical post-scalar dirty diagnostic is not an input to this report.\n",
        baseline_first.commit,
        baseline_first.dirty,
        baseline_rows.len(),
        clean(&baseline_first.cpu),
        clean(&baseline_first.os),
        clean(&baseline_first.arch)
    ));

    let current_pairs = primary_pairs(rows);
    let baseline_pairs = primary_pairs(baseline_rows);
    let current_by_case = current_pairs
        .iter()
        .cloned()
        .map(|pair| ((pair.case_id.clone(), pair.input_length), pair))
        .collect::<BTreeMap<_, _>>();
    let baseline_by_case = baseline_pairs
        .iter()
        .cloned()
        .map(|pair| ((pair.case_id.clone(), pair.input_length), pair))
        .collect::<BTreeMap<_, _>>();

    comparison.push_str(
        "\nPer-case caller-owned Rust/C latency ratios\n\n| Definition | 256 | 4,096 | 65,536 | 65,536 disposition |\n|---|---:|---:|---:|---|\n",
    );
    let case_ids = current_pairs
        .iter()
        .map(|pair| pair.case_id.clone())
        .collect::<BTreeSet<_>>();
    for case_id in case_ids {
        let ratios = INPUT_LENGTHS.map(|input_length| {
            current_by_case
                .get(&(case_id.clone(), input_length))
                .map(|pair| pair.ratio)
        });
        let large_disposition = match ratios[2] {
            Some(ratio) if ratio > 1.05 => "remaining gap above the 5% band",
            Some(ratio) if ratio < 0.95 => "fast-ta faster",
            Some(_) => "parity band",
            None => "unavailable",
        };
        comparison.push_str(&format!(
            "| {case_id} | {} | {} | {} | {large_disposition} |\n",
            format_ratio(ratios[0]),
            format_ratio(ratios[1]),
            format_ratio(ratios[2])
        ));
    }

    comparison.push_str(
        "\nExecution-path cost summaries\n\nRatios are geometric path/C latency indices over matching cases. Only the caller-owned Rust/C row is a kernel comparison; the other rows expose distinct public or user-facing costs.\n\n| Input | Path | Cases | Geometric path/C ratio | Interpretation |\n|---:|---|---:|---:|---|\n",
    );
    let paths = [
        (
            "fast-ta",
            RUST_OWNED_MODE,
            "Owned Compact Output",
            "API-owned compact allocation included",
        ),
        (
            "fast-ta",
            RUST_CALLER_MODE,
            "caller-owned Rust/C kernel",
            "primary comparable kernel seam",
        ),
        (
            "fast-ta",
            RUST_PREPARED_MODE,
            "Prepared reuse",
            "preparation and caller output allocation excluded",
        ),
        (
            "fast-ta",
            RUST_STREAMING_MODE,
            "Streaming reset plus ticks",
            "separate stateful execution cost",
        ),
        (
            "TA-Lib Python",
            PYTHON_MODE,
            "official Python NumPy API",
            "user-facing API-owned output cost",
        ),
    ];
    for input_length in INPUT_LENGTHS {
        for (implementation, mode, label, interpretation) in paths {
            let ratios = path_ratios(rows, implementation, mode, input_length);
            if ratios.is_empty() {
                comparison.push_str(&format!(
                    "| {input_length} | {label} | 0 | unavailable | {interpretation} |\n"
                ));
            } else {
                comparison.push_str(&format!(
                    "| {input_length} | {label} | {} | {:.3}x | {interpretation} |\n",
                    ratios.len(),
                    geometric_mean(&ratios)
                ));
            }
        }
    }

    comparison.push_str(
        "\nDurable optimization diagnosis and reproduction evidence\n\nEvidence kinds below are literal. `source` means source inspection and clean matrix behavior, not a sampled profiler or invented profiler percentage.\n\n| Ticket | Cases | Exact focused command(s) | Generated verdict from final clean rows |\n|---|---|---|---|\n",
    );
    for evidence in optimization_evidence {
        let verdicts = evidence
            .case_ids
            .iter()
            .map(|case_id| focused_verdict(case_id, &current_by_case))
            .collect::<Vec<_>>()
            .join("; ");
        comparison.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            clean(&evidence.ticket),
            evidence.case_ids.join(", "),
            evidence
                .focused_commands
                .iter()
                .map(|command| format!("`{}`", clean(command)))
                .collect::<Vec<_>>()
                .join("<br>"),
            verdicts
        ));
    }
    comparison.push_str(
        "\n| Ticket | Ranked falsifiable hypotheses | Confirmed evidence | Neighboring-workload disposition |\n|---|---|---|---|\n",
    );
    for evidence in optimization_evidence {
        comparison.push_str(&format!(
            "| {} | {} | {}: {} | {} — {} |\n",
            clean(&evidence.ticket),
            evidence
                .hypotheses
                .iter()
                .enumerate()
                .map(|(index, hypothesis)| format!("{}. {}", index + 1, clean(hypothesis)))
                .collect::<Vec<_>>()
                .join("<br>"),
            evidence.confirmed_evidence_kind,
            clean(&evidence.confirmed_evidence),
            clean(&evidence.neighboring_workloads),
            clean(&evidence.neighboring_disposition)
        ));
    }

    comparison.push_str(
        "\nClean before/after optimization effects\n\nEvery number in this table is read from matching caller-owned fast-ta rows in the committed clean pre-optimization and final matrices. Throughput is observations per second from those raw rows; the verdict is regenerated from the final same-run Rust/C pair.\n\n| Ticket | Definition | Input | Clean pre median [95% CI] | Clean pre throughput | Final median [95% CI] | Final throughput | Rust latency change | Focused verdict |\n|---|---|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for evidence in optimization_evidence {
        for case_id in &evidence.case_ids {
            for input_length in INPUT_LENGTHS {
                let before = find_measured_row(
                    baseline_rows,
                    "fast-ta",
                    case_id,
                    RUST_CALLER_MODE,
                    input_length,
                );
                let after =
                    find_measured_row(rows, "fast-ta", case_id, RUST_CALLER_MODE, input_length);
                match (before, after) {
                    (Some(before), Some(after)) => {
                        let before_stats = before.stats.as_ref().expect("measured row has stats");
                        let after_stats = after.stats.as_ref().expect("measured row has stats");
                        let change = (after_stats.median_ns / before_stats.median_ns - 1.0) * 100.0;
                        let verdict = current_by_case
                            .get(&(case_id.clone(), input_length))
                            .map_or_else(
                                || "unavailable".to_owned(),
                                |pair| focused_verdict_label(pair.ratio),
                            );
                        comparison.push_str(&format!(
                            "| {} | {case_id} | {input_length} | {} | {:.3} Mobs/s | {} | {:.3} Mobs/s | {change:+.1}% | {verdict} |\n",
                            clean(&evidence.ticket),
                            format_median_ci(before_stats),
                            before_stats.throughput_observations_per_second / 1.0e6,
                            format_median_ci(after_stats),
                            after_stats.throughput_observations_per_second / 1.0e6
                        ));
                    }
                    _ => comparison.push_str(&format!(
                        "| {} | {case_id} | {input_length} | unavailable | unavailable | unavailable | unavailable | unavailable | matching clean rows absent |\n",
                        clean(&evidence.ticket)
                    )),
                }
            }
        }
    }

    comparison.push_str(
        "\nCDLDOJI clean-data investigation\n\nThe clean comparison, not the historical dirty diagnostic, shows whether the representative regressed. Source inspection identifies the only CDLDOJI batch change as migration to the shared `compute_single_setting_batch_into` rolling helper; Streaming did not take that helper. The table keeps raw and Rust/C-normalized changes separate so reference-run movement cannot be mistaken for a source effect.\n\n| Input | Pre Rust | Final Rust | Rust change | Pre Rust/C | Final Rust/C | Normalized ratio change | Disposition |\n|---:|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for input_length in INPUT_LENGTHS {
        let before = find_measured_row(
            baseline_rows,
            "fast-ta",
            "CDLDOJI",
            RUST_CALLER_MODE,
            input_length,
        );
        let after = find_measured_row(rows, "fast-ta", "CDLDOJI", RUST_CALLER_MODE, input_length);
        let before_pair = baseline_by_case.get(&("CDLDOJI".to_owned(), input_length));
        let after_pair = current_by_case.get(&("CDLDOJI".to_owned(), input_length));
        if let (Some(before), Some(after), Some(before_pair), Some(after_pair)) =
            (before, after, before_pair, after_pair)
        {
            let before_stats = before.stats.as_ref().expect("measured row has stats");
            let after_stats = after.stats.as_ref().expect("measured row has stats");
            let rust_change = (after_stats.median_ns / before_stats.median_ns - 1.0) * 100.0;
            let ratio_change = (after_pair.ratio / before_pair.ratio - 1.0) * 100.0;
            let disposition = if ratio_change > 5.0 {
                "confirmed clean batch regression; source-correlated with shared-helper migration; unresolved"
            } else if ratio_change < -5.0 {
                "clean normalized improvement"
            } else {
                "within the 5% normalized band"
            };
            comparison.push_str(&format!(
                "| {input_length} | {:.3} us | {:.3} us | {rust_change:+.1}% | {:.3}x | {:.3}x | {ratio_change:+.1}% | {disposition} |\n",
                before_stats.median_ns / 1_000.0,
                after_stats.median_ns / 1_000.0,
                before_pair.ratio,
                after_pair.ratio
            ));
        }
    }

    let baseline_by_row = baseline_rows
        .iter()
        .map(|row| (comparison_row_key(row), row))
        .collect::<BTreeMap<_, _>>();
    let mut changes = rows
        .iter()
        .filter_map(|current| {
            let baseline = baseline_by_row.get(&comparison_row_key(current))?;
            let before = baseline.stats.as_ref()?;
            let after = current.stats.as_ref()?;
            let change = (after.median_ns / before.median_ns - 1.0) * 100.0;
            (change.abs() > 5.0).then_some((*baseline, current, change))
        })
        .collect::<Vec<_>>();
    changes.sort_by(|left, right| {
        left.1
            .case_id
            .cmp(&right.1.case_id)
            .then_with(|| left.1.input_length.cmp(&right.1.input_length))
            .then_with(|| left.1.implementation.cmp(&right.1.implementation))
            .then_with(|| left.1.mode.cmp(&right.1.mode))
    });
    comparison.push_str(&format!(
        "\nComplete >5% clean pre/final classification\n\n{} matching raw rows changed by more than 5%. This list is exhaustive for measured rows shared by the two committed clean artifacts. An investigation classification is not a same-run causal claim; the matrices were separate clean runs, and external C/Python movement is retained as a control.\n\n| Definition | Input | Implementation | Mode | Pre median | Final median | Change | Classification |\n|---|---:|---|---|---:|---:|---:|---|\n",
        changes.len()
    ));
    for (baseline, current, change) in changes {
        let before = baseline.stats.as_ref().expect("change row has stats");
        let after = current.stats.as_ref().expect("change row has stats");
        comparison.push_str(&format!(
            "| {} | {} | {} | {} | {:.3} us | {:.3} us | {change:+.1}% | {} |\n",
            current.case_id,
            current.input_length,
            current.implementation,
            current.mode,
            before.median_ns / 1_000.0,
            after.median_ns / 1_000.0,
            clean_change_classification(current, change)
        ));
    }

    comparison.push_str(
        "\nRuntime platform qualification from committed JSONL\n\nThese rows are parsed from the named JSONL artifacts only after their validation records and numeric timing evidence pass schema checks. Speedup is scalar median divided by the matching backend median only when mode, precision, input size, and timed boundary match. A value below 1x means the accelerated backend was slower on this runner. Rows without a scalar measurement at the same boundary are not compared.\n\n| Artifact | Platform | Precision | Profile / features | Runtime / CPU / OS | Active backend | Equivalence | Workflow provenance | Commit |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    for qualification in platform_qualifications {
        let equivalent = qualification.measurements.iter().all(|measurement| {
            measurement.equivalent_to_scalar
                && measurement.semantic_status == "verified"
                && measurement.timing_status == "measured"
        });
        comparison.push_str(&format!(
            "| {} | {} | {} | {} / {} | {} / {} / {} | {} | {} | run [{}]({}), job {} | {} |\n",
            clean(&qualification.artifact),
            clean(&qualification.platform),
            clean(&qualification.precision),
            clean(&qualification.profile),
            clean(&qualification.feature_flags),
            clean(&qualification.runtime),
            clean(&qualification.cpu),
            clean(&qualification.os),
            clean(&qualification.active_backend),
            if equivalent {
                "all measurement and validation rows verified"
            } else {
                "qualification contains an unverified row"
            },
            qualification.workflow_run_id,
            clean(&qualification.workflow_run_url),
            qualification.workflow_job,
            qualification.commit
        ));
    }
    comparison.push_str(
        "\n| Platform | Precision | Input | Mode | Backend | Median [95% CI] | Throughput | Speedup vs matching scalar | Disposition |\n|---|---|---:|---|---|---:|---:|---:|---|\n",
    );
    for qualification in platform_qualifications {
        let mut measurements = qualification.measurements.iter().collect::<Vec<_>>();
        measurements.sort_by(|left, right| {
            left.input_length
                .cmp(&right.input_length)
                .then_with(|| left.mode.cmp(&right.mode))
                .then_with(|| left.backend.cmp(&right.backend))
        });
        for measurement in measurements {
            let scalar = qualification.measurements.iter().find(|candidate| {
                candidate.mode == measurement.mode
                    && candidate.input_length == measurement.input_length
                    && candidate.backend == "scalar"
                    && candidate.timed_boundary == measurement.timed_boundary
            });
            let speedup = scalar.map(|scalar| scalar.median_ns / measurement.median_ns);
            let disposition = qualification_disposition(measurement, speedup);
            comparison.push_str(&format!(
                "| {} | {} | {} | {} | {} | {:.3} us [{:.3}, {:.3}] | {:.3} Mobs/s | {} | {disposition} |\n",
                clean(&qualification.platform),
                clean(&qualification.precision),
                measurement.input_length,
                clean(&measurement.mode),
                clean(&measurement.backend),
                measurement.median_ns / 1_000.0,
                measurement.ci95_lower_ns / 1_000.0,
                measurement.ci95_upper_ns / 1_000.0,
                measurement.throughput_observations_per_second / 1.0e6,
                format_ratio(speedup)
            ));
        }
    }

    comparison.push_str(
        "\nValidation and allocation boundaries\n\nThe Rust matrix timings retain public finite-input validation, capacity, Output Range, and validation-before-mutation contracts. Validation and computation were not timed separately in the catalogue matrix; the direct C row is a comparative kernel reference, not evidence that Rust validation should be removed.\n\n| Implementation | Mode | Timed allocation/boundary evidence from raw rows |\n|---|---|---|\n",
    );
    let boundaries = rows
        .iter()
        .map(|row| {
            (
                row.implementation.clone(),
                row.mode.clone(),
                row.timed_boundary.clone(),
            )
        })
        .collect::<BTreeSet<_>>();
    for (implementation, mode, boundary) in boundaries {
        comparison.push_str(&format!(
            "| {implementation} | {mode} | {} |\n",
            clean(&boundary)
        ));
    }

    comparison.push_str(&format!(
        "\nFinal AArch64 qualification\n\nThe clean final matrix exercised the public TYPPRICE caller-owned path on {} / {} with architecture `{}` and commit {}. This is the only AArch64 timing claim; scalar fallback remains available. x86_64 and WASM claims above come only from their committed runtime JSONL artifacts.\n",
        clean(&first.cpu),
        clean(&first.os),
        clean(&first.arch),
        first.commit
    ));

    Ok(format!("{summary}{comparison}{marker}{details}"))
}

fn find_measured_row<'a>(
    rows: &'a [BenchmarkRow],
    implementation: &str,
    case_id: &str,
    mode: &str,
    input_length: usize,
) -> Option<&'a BenchmarkRow> {
    rows.iter().find(|row| {
        row.implementation == implementation
            && row.case_id == case_id
            && row.mode == mode
            && row.input_length == input_length
            && row.semantic_status == "verified"
            && row.timing_status == "measured"
            && row.stats.is_some()
    })
}

fn format_median_ci(stats: &TimingStats) -> String {
    format!(
        "{:.3} us [{:.3}, {:.3}]",
        stats.median_ns / 1_000.0,
        stats.ci95_lower_ns / 1_000.0,
        stats.ci95_upper_ns / 1_000.0
    )
}

fn focused_verdict(
    case_id: &str,
    current_by_case: &BTreeMap<(String, usize), PrimaryPair>,
) -> String {
    let verdicts = INPUT_LENGTHS.map(|input_length| {
        current_by_case
            .get(&(case_id.to_owned(), input_length))
            .map_or_else(
                || format!("{input_length} unavailable"),
                |pair| format!("{input_length} {}", focused_verdict_label(pair.ratio)),
            )
    });
    format!("{case_id}: {}", verdicts.join(", "))
}

fn focused_verdict_label(ratio: f64) -> String {
    if ratio <= 1.05 {
        format!("PASS ({ratio:.3}x)")
    } else {
        format!("FAIL ({ratio:.3}x)")
    }
}

fn comparison_row_key(
    row: &BenchmarkRow,
) -> (String, String, String, String, usize, String, usize) {
    (
        row.implementation.clone(),
        row.case_id.clone(),
        row.mode.clone(),
        row.parameters.clone(),
        row.input_length,
        row.fixture.clone(),
        row.float_width,
    )
}

fn clean_change_classification(row: &BenchmarkRow, change: f64) -> &'static str {
    if row.implementation != "fast-ta" {
        return "external C/Python reference-path movement between clean runs; retained as host/run control, not attributed to fast-ta source";
    }
    if row.mode == RUST_STREAMING_MODE {
        return "untouched Streaming neighbor; batch optimization does not explain this clean cross-run movement; no causal claim";
    }
    if row.case_id == "CDLDOJI" && change > 5.0 {
        return "investigated clean regression: batch-only, source-correlated with migration to the shared single-setting helper; unresolved";
    }
    if row.case_id == "HT_DCPHASE" && change > 5.0 {
        return "investigated clean large-input regression despite ring-wrap source change; unresolved";
    }
    if change < -5.0 {
        return "clean batch improvement; targeted kernel work or shared SIMD finite-slice validation applies";
    }
    "clean batch regression above 5%; classified as unresolved"
}

fn qualification_disposition(
    measurement: &QualificationMeasurement,
    speedup: Option<f64>,
) -> &'static str {
    if !measurement.equivalent_to_scalar || measurement.semantic_status != "verified" {
        "invalid for performance comparison: equivalence not verified"
    } else if measurement.backend == "scalar" {
        "scalar reference"
    } else if let Some(speedup) = speedup {
        if speedup > 1.05 {
            "practical benefit on this runner"
        } else if speedup < 0.95 {
            "slower than scalar on this runner; no practical benefit"
        } else {
            "within the 5% band on this runner"
        }
    } else {
        "no matching scalar row with the same timed boundary"
    }
}

fn format_ratio(ratio: Option<f64>) -> String {
    ratio.map_or_else(|| "unavailable".to_owned(), |ratio| format!("{ratio:.3}x"))
}

fn geometric_mean(ratios: &[f64]) -> f64 {
    (ratios.iter().map(|ratio| ratio.ln()).sum::<f64>() / ratios.len() as f64).exp()
}

fn path_ratios(
    rows: &[BenchmarkRow],
    implementation: &str,
    mode: &str,
    input_length: usize,
) -> Vec<f64> {
    type Key = (String, String, usize, String, String);
    let mut c = BTreeMap::<Key, f64>::new();
    for row in rows {
        if row.implementation != "TA-Lib C"
            || row.mode != C_DIRECT_MODE
            || row.input_length != input_length
            || row.semantic_status != "verified"
            || row.timing_status != "measured"
        {
            continue;
        }
        if let Some(stats) = &row.stats {
            c.insert(
                (
                    row.case_id.clone(),
                    row.parameters.clone(),
                    row.input_length,
                    row.fixture.clone(),
                    row.input_checksum.clone(),
                ),
                stats.median_ns,
            );
        }
    }
    rows.iter()
        .filter(|row| {
            row.implementation == implementation
                && row.mode == mode
                && row.input_length == input_length
                && row.semantic_status == "verified"
                && row.timing_status == "measured"
        })
        .filter_map(|row| {
            let stats = row.stats.as_ref()?;
            let key = (
                row.case_id.clone(),
                row.parameters.clone(),
                row.input_length,
                row.fixture.clone(),
                row.input_checksum.clone(),
            );
            c.get(&key).map(|c_ns| stats.median_ns / c_ns)
        })
        .collect()
}

#[derive(Clone, Debug)]
struct PrimaryPair {
    case_id: String,
    input_length: usize,
    ratio: f64,
}

fn primary_pairs(rows: &[BenchmarkRow]) -> Vec<PrimaryPair> {
    type Key = (String, String, usize, String, String);
    let mut rust = BTreeMap::<Key, f64>::new();
    let mut c = BTreeMap::<Key, f64>::new();
    for row in rows {
        if row.semantic_status != "verified"
            || row.timing_status != "measured"
            || row.comparison_status != "comparable"
        {
            continue;
        }
        let Some(stats) = &row.stats else { continue };
        let key = (
            row.case_id.clone(),
            row.parameters.clone(),
            row.input_length,
            row.fixture.clone(),
            row.input_checksum.clone(),
        );
        if row.implementation == "fast-ta" && row.mode == RUST_CALLER_MODE {
            rust.insert(key, stats.median_ns);
        } else if row.implementation == "TA-Lib C" && row.mode == C_DIRECT_MODE {
            c.insert(key, stats.median_ns);
        }
    }
    rust.into_iter()
        .filter_map(|(key, rust_ns)| {
            c.get(&key).map(|c_ns| PrimaryPair {
                case_id: key.0,
                input_length: key.2,
                ratio: rust_ns / c_ns,
            })
        })
        .collect()
}

fn format_row(row: &BenchmarkRow) -> String {
    let stats = row.stats.as_ref();
    [
        clean(&row.implementation),
        clean(&row.indicator_family),
        clean(&row.indicator_definition),
        clean(&row.case_id),
        clean(&row.mode),
        clean(&row.parameters),
        row.input_length.to_string(),
        clean(&row.output_kind),
        opt(row.output_arity),
        opt_float(stats.map(|value| value.median_ns)),
        opt_float(stats.map(|value| value.ci95_lower_ns)),
        opt_float(stats.map(|value| value.ci95_upper_ns)),
        opt_float(stats.map(|value| value.throughput_observations_per_second)),
        opt(stats.map(|value| value.sample_count)),
        opt(stats.map(|value| value.outlier_count)),
        opt(stats.map(|value| value.outlier_low_count)),
        opt(stats.map(|value| value.outlier_high_count)),
        opt(row.output_begin),
        opt(row.output_count),
        clean(&row.output_checksum),
        clean(&row.semantic_status),
        clean(&row.semantic_reason),
        clean(&row.timing_status),
        clean(&row.timing_reason),
        clean(&row.comparison_status),
        clean(&row.comparison_reason),
        opt(row.warmup_iterations),
        opt(row.iterations_per_sample),
        clean(&row.timed_boundary),
        clean(&row.fixture),
        clean(&row.input_checksum),
        clean(&row.ta_lib_version),
        clean(&row.ta_lib_revision),
        clean(&row.python_version),
        clean(&row.python_binding_version),
        clean(&row.python_ta_lib_version),
        clean(&row.numpy_version),
        clean(&row.rustc),
        clean(&row.cpu),
        clean(&row.os),
        clean(&row.arch),
        row.float_width.to_string(),
        clean(&row.features),
        clean(&row.commit),
        row.dirty.to_string(),
    ]
    .join("\t")
}

fn parse_row(line: &str) -> Result<BenchmarkRow, String> {
    let fields = line.split('\t').collect::<Vec<_>>();
    if fields.len() != 45 {
        return Err(format!("expected 45 columns, got {}", fields.len()));
    }
    let parsed_stats = match fields[9] {
        "NA" => None,
        _ => Some(TimingStats {
            median_ns: parse(fields[9], "median_ns")?,
            ci95_lower_ns: parse(fields[10], "ci95_lower_ns")?,
            ci95_upper_ns: parse(fields[11], "ci95_upper_ns")?,
            throughput_observations_per_second: parse(fields[12], "throughput")?,
            sample_count: parse(fields[13], "sample_count")?,
            outlier_count: parse(fields[14], "outlier_count")?,
            outlier_low_count: parse(fields[15], "outlier_low_count")?,
            outlier_high_count: parse(fields[16], "outlier_high_count")?,
        }),
    };
    let row = BenchmarkRow {
        implementation: fields[0].to_owned(),
        indicator_family: fields[1].to_owned(),
        indicator_definition: fields[2].to_owned(),
        case_id: fields[3].to_owned(),
        mode: fields[4].to_owned(),
        parameters: fields[5].to_owned(),
        input_length: parse(fields[6], "input_length")?,
        output_kind: fields[7].to_owned(),
        output_arity: parse_opt(fields[8], "output_arity")?,
        stats: parsed_stats,
        output_begin: parse_opt(fields[17], "output_begin")?,
        output_count: parse_opt(fields[18], "output_count")?,
        output_checksum: fields[19].to_owned(),
        semantic_status: fields[20].to_owned(),
        semantic_reason: fields[21].to_owned(),
        timing_status: fields[22].to_owned(),
        timing_reason: fields[23].to_owned(),
        comparison_status: fields[24].to_owned(),
        comparison_reason: fields[25].to_owned(),
        warmup_iterations: parse_opt(fields[26], "warmup_iterations")?,
        iterations_per_sample: parse_opt(fields[27], "iterations_per_sample")?,
        timed_boundary: fields[28].to_owned(),
        fixture: fields[29].to_owned(),
        input_checksum: fields[30].to_owned(),
        ta_lib_version: fields[31].to_owned(),
        ta_lib_revision: fields[32].to_owned(),
        python_version: fields[33].to_owned(),
        python_binding_version: fields[34].to_owned(),
        python_ta_lib_version: fields[35].to_owned(),
        numpy_version: fields[36].to_owned(),
        rustc: fields[37].to_owned(),
        cpu: fields[38].to_owned(),
        os: fields[39].to_owned(),
        arch: fields[40].to_owned(),
        float_width: parse(fields[41], "float_width")?,
        features: fields[42].to_owned(),
        commit: fields[43].to_owned(),
        dirty: parse(fields[44], "dirty")?,
    };
    if row.stats.is_none() && fields[10..17].iter().any(|value| *value != "NA") {
        return Err("partial unavailable timing statistics".to_owned());
    }
    validate_benchmark_row_evidence(&row)?;
    Ok(row)
}

fn validate_benchmark_row_evidence(row: &BenchmarkRow) -> Result<(), String> {
    if row.input_length == 0 {
        return Err("input_length must be positive".to_owned());
    }
    for (name, iterations) in [
        ("warmup_iterations", row.warmup_iterations),
        ("iterations_per_sample", row.iterations_per_sample),
    ] {
        if iterations == Some(0) {
            return Err(format!("{name} must be positive when present"));
        }
    }
    let Some(stats) = &row.stats else {
        return Ok(());
    };
    if row.warmup_iterations.is_none() || row.iterations_per_sample.is_none() {
        return Err(
            "measured timing evidence requires warmup_iterations and iterations_per_sample"
                .to_owned(),
        );
    }
    validate_positive_timing_evidence(
        stats.median_ns,
        stats.ci95_lower_ns,
        stats.ci95_upper_ns,
        stats.throughput_observations_per_second,
        stats.sample_count,
        row.input_length,
    )?;
    let classified_outliers = stats
        .outlier_low_count
        .checked_add(stats.outlier_high_count)
        .ok_or_else(|| "outlier counts overflow".to_owned())?;
    if stats.outlier_count != classified_outliers || stats.outlier_count > stats.sample_count {
        return Err("outlier counts are incoherent with sample_count".to_owned());
    }
    Ok(())
}

fn validate_positive_timing_evidence(
    median_ns: f64,
    ci95_lower_ns: f64,
    ci95_upper_ns: f64,
    throughput_observations_per_second: f64,
    sample_count: usize,
    input_length: usize,
) -> Result<(), String> {
    if [
        median_ns,
        ci95_lower_ns,
        ci95_upper_ns,
        throughput_observations_per_second,
    ]
    .into_iter()
    .any(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("timing evidence must be positive and finite".to_owned());
    }
    if ci95_lower_ns > median_ns || median_ns > ci95_upper_ns {
        return Err("95% confidence interval must contain the median".to_owned());
    }
    if sample_count == 0 {
        return Err("sample_count must be positive".to_owned());
    }
    if input_length == 0 {
        return Err("input_length must be positive".to_owned());
    }
    let expected_throughput = input_length as f64 * 1.0e9 / median_ns;
    let relative_error =
        (throughput_observations_per_second - expected_throughput).abs() / expected_throughput;
    if relative_error > 1.0e-4 {
        return Err(format!(
            "throughput is incoherent with input_length and median_ns (relative error {relative_error:.6})"
        ));
    }
    Ok(())
}

fn parse<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_opt<T>(value: &str, name: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if value == "NA" {
        Ok(None)
    } else {
        parse(value, name).map(Some)
    }
}

fn opt<T: ToString>(value: Option<T>) -> String {
    value.map_or_else(|| "NA".to_owned(), |value| value.to_string())
}

fn opt_float(value: Option<f64>) -> String {
    value.map_or_else(|| "NA".to_owned(), |value| value.to_string())
}

fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n', '|'], " ")
}
