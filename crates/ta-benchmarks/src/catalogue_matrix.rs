//! Deterministic fixtures, semantic gates, raw rows, statistics, and reporting
//! for the opt-in representative Indicator Catalogue comparison.

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
    for row in rows {
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

pub const OPTIMIZATION_EVIDENCE_HEADER: &str =
    "stage\tcase_id\tinput_length\trust_median_ns\tc_median_ns\tsource";

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationEvidenceRow {
    pub stage: String,
    pub case_id: String,
    pub input_length: usize,
    pub rust_median_ns: f64,
    pub c_median_ns: f64,
    pub source: String,
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
            if fields.len() != 6 {
                return Err(format!(
                    "optimization evidence row {} has {} fields, expected 6",
                    index + 2,
                    fields.len()
                ));
            }
            let rust_median_ns = parse::<f64>(fields[3], "rust_median_ns")?;
            let c_median_ns = parse::<f64>(fields[4], "c_median_ns")?;
            if rust_median_ns <= 0.0 || c_median_ns <= 0.0 {
                return Err(format!(
                    "optimization evidence row {} has a non-positive median",
                    index + 2
                ));
            }
            Ok(OptimizationEvidenceRow {
                stage: fields[0].to_owned(),
                case_id: fields[1].to_owned(),
                input_length: parse(fields[2], "input_length")?,
                rust_median_ns,
                c_median_ns,
                source: fields[5].to_owned(),
            })
        })
        .collect()
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
) -> Result<String, String> {
    let report = render_report(rows)?;
    let marker = "\nDetailed raw-row projection\n";
    let (summary, details) = report
        .split_once(marker)
        .ok_or_else(|| "generated report is missing the detailed-row marker".to_owned())?;
    let first = rows
        .first()
        .ok_or_else(|| "cannot render a comparison report without raw rows".to_owned())?;
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
            "clean canonical run"
        } else {
            "dirty diagnostic run"
        }
    ));

    let current_pairs = primary_pairs(rows);
    let current_by_case = current_pairs
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
            Some(ratio) if ratio > 1.05 => "unresolved gap above the 5% band",
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
        "\nExecution-path cost summaries\n\nRatios are geometric path/C latency indices over matching cases. Only the caller-owned Rust/C row is a kernel comparison; the other rows deliberately expose distinct public or user-facing costs.\n\n| Input | Path | Cases | Geometric path/C ratio | Interpretation |\n|---:|---|---:|---:|---|\n",
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
        "\nTargeted optimization effects\n\nThe before values below are parsed from the committed evidence TSV. Issue 56 did not retain a clean complete first-delivery raw artifact, so each row names its actual evidence stage rather than promoting a dirty diagnostic run to a canonical baseline.\n\n| Definition | Input | Evidence stage | Source | Before Rust/C | Final Rust/C | Ratio change | Disposition |\n|---|---:|---|---|---:|---:|---:|---|\n",
    );
    let mut evidence = optimization_evidence.to_vec();
    evidence.sort_by(|left, right| {
        left.case_id
            .cmp(&right.case_id)
            .then_with(|| left.input_length.cmp(&right.input_length))
    });
    for historical in evidence {
        let before_ratio = historical.rust_median_ns / historical.c_median_ns;
        let Some(current) =
            current_by_case.get(&(historical.case_id.clone(), historical.input_length))
        else {
            comparison.push_str(&format!(
                "| {} | {} | {} | {} | {before_ratio:.3}x | unavailable | unavailable | final comparable row absent |\n",
                historical.case_id,
                historical.input_length,
                clean(&historical.stage),
                clean(&historical.source)
            ));
            continue;
        };
        let change = (current.ratio / before_ratio - 1.0) * 100.0;
        let disposition = if historical.stage.contains("baseline unavailable") {
            "post-fusion stability only; no retained pre-fusion measurement"
        } else if change < -5.0 {
            "practical improvement"
        } else if change > 5.0 {
            "documented cross-run worsening above 5%"
        } else {
            "within the 5% cross-run band"
        };
        comparison.push_str(&format!(
            "| {} | {} | {} | {} | {before_ratio:.3}x | {:.3}x | {change:+.1}% | {disposition} |\n",
            historical.case_id,
            historical.input_length,
            clean(&historical.stage),
            clean(&historical.source),
            current.ratio
        ));
    }

    comparison
        .push_str("\nPreserved diagnostic baseline comparison and regression dispositions\n\n");
    if baseline_rows.is_empty() {
        comparison.push_str(
            "Unavailable: issue 56 did not retain a clean complete first-delivery raw artifact, and no diagnostic comparison rows were supplied.\n",
        );
    } else {
        let baseline_first = &baseline_rows[0];
        comparison.push_str(&format!(
            "Comparison source: commit {} (dirty: {}), {} rows. Changes use Rust/C ratio normalization to reduce host-load drift. Because this predecessor is dirty and cross-run, changes above 5% are documented but are not presented as source-level regression verdicts.\n\n",
            baseline_first.commit,
            baseline_first.dirty,
            baseline_rows.len()
        ));
        comparison.push_str(
            "| Definition | Input | Diagnostic Rust/C | Final Rust/C | Change | Explicit disposition |\n|---|---:|---:|---:|---:|---|\n",
        );
        let baseline_by_case = primary_pairs(baseline_rows)
            .into_iter()
            .map(|pair| ((pair.case_id.clone(), pair.input_length), pair))
            .collect::<BTreeMap<_, _>>();
        for (key, current) in &current_by_case {
            let Some(baseline) = baseline_by_case.get(key) else {
                comparison.push_str(&format!(
                    "| {} | {} | unavailable | {:.3}x | unavailable | no matching diagnostic row |\n",
                    key.0, key.1, current.ratio
                ));
                continue;
            };
            let change = (current.ratio / baseline.ratio - 1.0) * 100.0;
            let disposition = if change < -5.0 {
                "directional improvement; dirty cross-run predecessor"
            } else if change > 5.0 {
                "documented >5% worsening; dirty cross-run predecessor cannot establish a source regression"
            } else {
                "within 5% normalized cross-run band"
            };
            comparison.push_str(&format!(
                "| {} | {} | {:.3}x | {:.3}x | {change:+.1}% | {disposition} |\n",
                key.0, key.1, baseline.ratio, current.ratio
            ));
        }
    }

    comparison.push_str(
        "\nValidation and allocation boundaries\n\nThe Rust timings retain the public finite-input validation, capacity, Output Range, and validation-before-mutation contracts. Validation and computation were not timed separately in this matrix; the direct C row is therefore a comparative kernel reference, not evidence that Rust validation should be removed.\n\n| Implementation | Mode | Timed allocation/boundary evidence from raw rows |\n|---|---|---|\n",
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

    comparison.push_str(
        "\nPlatform SIMD qualification\n\n| Platform | Status | Speed claim |\n|---|---|---|\n",
    );
    if first.arch == "aarch64" && current_by_case.contains_key(&("TYPPRICE".to_owned(), 65_536)) {
        comparison.push_str(&format!(
            "| AArch64 | public TYPPRICE batch path exercised on {} with std/f64; scalar fallback remains available | measured only on this Apple M2 run |\n",
            clean(&first.cpu)
        ));
    } else {
        comparison.push_str(
            "| AArch64 | unavailable in these raw rows; scalar fallback remains available | no measurement |\n",
        );
    }
    comparison.push_str(
        "| x86_64 | AVX-512F/AVX2 dispatch and scalar fallback compile-verified at 079a39c; not executed on this AArch64 host | no x86 speed measurement |\n",
    );
    comparison.push_str(
        "| wasm32 | SIMD128 and scalar fallback compile-verified at 079a39c; not executed as a speed benchmark on this host | no WASM speed measurement |\n",
    );

    Ok(format!("{summary}{comparison}{marker}{details}"))
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
    Ok(row)
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
