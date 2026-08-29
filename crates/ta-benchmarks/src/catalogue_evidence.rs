//! Raw evidence and publication policy for the Indicator Catalogue matrix.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::catalogue_cases::MATRIX;
use crate::catalogue_statistics::validate_positive_timing_evidence;
pub use crate::catalogue_statistics::{timing_stats, TimingStats};

pub const INPUT_LENGTHS: [usize; 3] = [256, 4_096, 65_536];
pub const FIXTURE_ID: &str = "catalogue_fixture_v1:f64le";
pub const RUST_OWNED_MODE: &str = "Owned Compact Output";
pub const RUST_CALLER_MODE: &str = "caller-owned Batch Computation";
pub const RUST_PREPARED_MODE: &str = "Prepared Batch Runner";
pub const RUST_STREAMING_MODE: &str = "Streaming Computation";
pub const C_DIRECT_MODE: &str = "direct C caller-owned";
pub const PRIMARY_COMPARISON: &str = "primary caller-owned Rust/C kernel";
pub const PYTHON_MODE: &str = "official Python NumPy API";
pub const RAW_HEADER: &str = "implementation\tindicator_family\tindicator_definition\tcase_id\tmode\tparameters\tinput_length\toutput_kind\toutput_arity\tmedian_ns\tci95_lower_ns\tci95_upper_ns\tthroughput_observations_per_second\tsample_count\toutlier_count\toutlier_low_count\toutlier_high_count\toutput_begin\toutput_count\toutput_checksum\tsemantic_status\tsemantic_reason\ttiming_status\ttiming_reason\tcomparison_status\tcomparison_reason\twarmup_iterations\titerations_per_sample\ttimed_boundary\tfixture\tinput_checksum\tta_lib_version\tta_lib_revision\tpython_version\tpython_binding_version\tpython_ta_lib_version\tnumpy_version\trustc\tcpu\tos\tarch\tfloat_width\tfeatures\tcommit\tdirty";

pub const PUBLICATION_VARIANTS: [(&str, &str); 6] = [
    ("fast-ta", RUST_OWNED_MODE),
    ("fast-ta", RUST_CALLER_MODE),
    ("fast-ta", RUST_PREPARED_MODE),
    ("fast-ta", RUST_STREAMING_MODE),
    ("TA-Lib C", C_DIRECT_MODE),
    ("TA-Lib Python", PYTHON_MODE),
];
pub const CANONICAL_INPUT_CHECKSUMS: [(usize, &str); 3] = [
    (256, "fnv1a64:73fedfe0ae0a803f"),
    (4_096, "fnv1a64:06be03be64d63c6c"),
    (65_536, "fnv1a64:a4171d0a7611733a"),
];

/// Canonical requirements for evidence that may replace the published matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CataloguePublicationPolicy {
    pub input_lengths: &'static [usize],
    pub variants: &'static [(&'static str, &'static str)],
    pub input_checksums: &'static [(usize, &'static str)],
    pub semantic_status: &'static str,
    pub timing_status: &'static str,
    pub sample_count: usize,
    pub fixture: &'static str,
    pub ta_lib_version: &'static str,
    pub ta_lib_revision: &'static str,
    pub python_binding_version: &'static str,
    pub python_ta_lib_version: &'static str,
    pub numpy_version: &'static str,
    pub float_width: usize,
    pub features: &'static str,
}

pub const PUBLICATION_POLICY: CataloguePublicationPolicy = CataloguePublicationPolicy {
    input_lengths: &INPUT_LENGTHS,
    variants: &PUBLICATION_VARIANTS,
    input_checksums: &CANONICAL_INPUT_CHECKSUMS,
    semantic_status: "verified",
    timing_status: "measured",
    sample_count: 50,
    fixture: FIXTURE_ID,
    ta_lib_version: "0.6.4",
    ta_lib_revision: "43f9d5042ecc4bd367941846494ad907bf20ea50",
    python_binding_version: "0.6.4",
    python_ta_lib_version: "0.6.4",
    numpy_version: "2.2.3",
    float_width: 64,
    features: "fast-ta=default(f64,std); ta-benchmarks=catalogue-matrix",
};
const LEGACY_REPORT_FEATURES: &str = "ta-core=default(f64,std); ta-benchmarks=catalogue-matrix";

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

/// Uniform provenance returned only after the complete publication policy passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicationProvenance {
    pub python_version: String,
    pub rustc: String,
    pub cpu: String,
    pub os: String,
    pub arch: String,
    pub commit: String,
}

/// Parsed rows whose completeness, timings, provenance, and publication policy are validated.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedCatalogueEvidence {
    rows: Vec<BenchmarkRow>,
    provenance: PublicationProvenance,
}

impl ValidatedCatalogueEvidence {
    pub fn rows(&self) -> &[BenchmarkRow] {
        &self.rows
    }

    pub fn provenance(&self) -> &PublicationProvenance {
        &self.provenance
    }

    pub fn into_rows(self) -> Vec<BenchmarkRow> {
        self.rows
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogueEvidenceError {
    message: String,
}

impl CatalogueEvidenceError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CatalogueEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CatalogueEvidenceError {}

pub fn read_publishable_evidence(
    path: &Path,
) -> Result<ValidatedCatalogueEvidence, CatalogueEvidenceError> {
    let rows = read_raw_rows(path).map_err(CatalogueEvidenceError::new)?;
    validate_publishable_rows(rows)
}

/// Reads evidence for report regeneration, including immutable legacy baselines.
///
/// Legacy crate provenance is accepted only through this report-specific seam;
/// publication validation requires the current `fast-ta` provenance.
pub fn read_report_evidence(
    path: &Path,
) -> Result<ValidatedCatalogueEvidence, CatalogueEvidenceError> {
    let rows = read_raw_rows(path).map_err(CatalogueEvidenceError::new)?;
    let expected_features = if rows
        .iter()
        .all(|row| row.features == LEGACY_REPORT_FEATURES)
    {
        LEGACY_REPORT_FEATURES
    } else {
        PUBLICATION_POLICY.features
    };
    validate_rows_with_features(rows, expected_features)
}

pub fn validate_publishable_rows(
    rows: Vec<BenchmarkRow>,
) -> Result<ValidatedCatalogueEvidence, CatalogueEvidenceError> {
    validate_rows_with_features(rows, PUBLICATION_POLICY.features)
}

fn validate_rows_with_features(
    rows: Vec<BenchmarkRow>,
    expected_features: &str,
) -> Result<ValidatedCatalogueEvidence, CatalogueEvidenceError> {
    validate_complete_matrix(&rows)?;
    for (index, row) in rows.iter().enumerate() {
        validate_benchmark_row_evidence(row).map_err(|error| {
            CatalogueEvidenceError::new(format!("cannot publish raw row {}: {error}", index + 2))
        })?;
    }
    validate_fixed_publication_fields(&rows, expected_features)?;
    let provenance = validate_uniform_provenance(&rows)?;
    validate_input_checksums(&rows)?;
    validate_case_provenance(&rows)?;
    Ok(ValidatedCatalogueEvidence { rows, provenance })
}

fn validate_complete_matrix(rows: &[BenchmarkRow]) -> Result<(), CatalogueEvidenceError> {
    type Cell = (String, usize, String, String);
    let mut expected = BTreeMap::<Cell, usize>::new();
    for case in MATRIX {
        for &input_length in PUBLICATION_POLICY.input_lengths {
            for &(implementation, mode) in PUBLICATION_POLICY.variants {
                *expected
                    .entry((
                        case.id.to_owned(),
                        input_length,
                        implementation.to_owned(),
                        mode.to_owned(),
                    ))
                    .or_default() += 1;
            }
        }
    }
    let mut actual = BTreeMap::<Cell, usize>::new();
    for row in rows {
        *actual
            .entry((
                row.case_id.clone(),
                row.input_length,
                row.implementation.clone(),
                row.mode.clone(),
            ))
            .or_default() += 1;
    }
    if actual == expected {
        return Ok(());
    }

    let mut missing = Vec::new();
    let mut unexpected = Vec::new();
    for (cell, count) in &expected {
        let difference = count.saturating_sub(actual.get(cell).copied().unwrap_or(0));
        missing.extend(std::iter::repeat_n(cell, difference));
    }
    for (cell, count) in &actual {
        let difference = count.saturating_sub(expected.get(cell).copied().unwrap_or(0));
        unexpected.extend(std::iter::repeat_n(cell, difference));
    }
    Err(CatalogueEvidenceError::new(format!(
        "cannot publish incomplete case/input/mode matrix: missing={:?}; unexpected_or_duplicate={:?}",
        &missing[..missing.len().min(5)],
        &unexpected[..unexpected.len().min(5)]
    )))
}

fn validate_fixed_publication_fields(
    rows: &[BenchmarkRow],
    expected_features: &str,
) -> Result<(), CatalogueEvidenceError> {
    require_fixed(
        rows,
        "semantic_status",
        PUBLICATION_POLICY.semantic_status,
        |row| row.semantic_status == PUBLICATION_POLICY.semantic_status,
    )?;
    require_fixed(
        rows,
        "timing_status",
        PUBLICATION_POLICY.timing_status,
        |row| row.timing_status == PUBLICATION_POLICY.timing_status,
    )?;
    require_fixed(rows, "sample_count", "50", |row| {
        row.stats
            .as_ref()
            .is_some_and(|stats| stats.sample_count == PUBLICATION_POLICY.sample_count)
    })?;
    require_fixed(rows, "dirty", "false", |row| !row.dirty)?;
    require_fixed(rows, "fixture", PUBLICATION_POLICY.fixture, |row| {
        row.fixture == PUBLICATION_POLICY.fixture
    })?;
    require_fixed(
        rows,
        "ta_lib_version",
        PUBLICATION_POLICY.ta_lib_version,
        |row| row.ta_lib_version == PUBLICATION_POLICY.ta_lib_version,
    )?;
    require_fixed(
        rows,
        "ta_lib_revision",
        PUBLICATION_POLICY.ta_lib_revision,
        |row| row.ta_lib_revision == PUBLICATION_POLICY.ta_lib_revision,
    )?;
    require_fixed(
        rows,
        "python_binding_version",
        PUBLICATION_POLICY.python_binding_version,
        |row| row.python_binding_version == PUBLICATION_POLICY.python_binding_version,
    )?;
    require_fixed(
        rows,
        "python_ta_lib_version",
        PUBLICATION_POLICY.python_ta_lib_version,
        |row| row.python_ta_lib_version == PUBLICATION_POLICY.python_ta_lib_version,
    )?;
    require_fixed(
        rows,
        "numpy_version",
        PUBLICATION_POLICY.numpy_version,
        |row| row.numpy_version == PUBLICATION_POLICY.numpy_version,
    )?;
    require_fixed(rows, "float_width", "64", |row| {
        row.float_width == PUBLICATION_POLICY.float_width
    })?;
    require_fixed(rows, "features", expected_features, |row| {
        row.features == expected_features
    })
}

fn require_fixed(
    rows: &[BenchmarkRow],
    field: &str,
    expected: &str,
    predicate: impl Fn(&BenchmarkRow) -> bool,
) -> Result<(), CatalogueEvidenceError> {
    let invalid = rows.iter().filter(|row| !predicate(row)).count();
    if invalid != 0 {
        return Err(CatalogueEvidenceError::new(format!(
            "cannot publish: {invalid} rows have {field} other than {expected:?}"
        )));
    }
    Ok(())
}

fn validate_uniform_provenance(
    rows: &[BenchmarkRow],
) -> Result<PublicationProvenance, CatalogueEvidenceError> {
    fn uniform(
        rows: &[BenchmarkRow],
        field: &str,
        value: impl Fn(&BenchmarkRow) -> &str,
    ) -> Result<String, CatalogueEvidenceError> {
        let values = rows.iter().map(value).collect::<BTreeSet<_>>();
        if values.len() != 1
            || values.first().is_none_or(|value| value.is_empty())
            || values.contains("unavailable")
        {
            return Err(CatalogueEvidenceError::new(format!(
                "cannot publish inconsistent {field} provenance: {values:?}"
            )));
        }
        Ok(values
            .into_iter()
            .next()
            .expect("one uniform value")
            .to_owned())
    }

    Ok(PublicationProvenance {
        python_version: uniform(rows, "python_version", |row| &row.python_version)?,
        rustc: uniform(rows, "rustc", |row| &row.rustc)?,
        cpu: uniform(rows, "cpu", |row| &row.cpu)?,
        os: uniform(rows, "os", |row| &row.os)?,
        arch: uniform(rows, "arch", |row| &row.arch)?,
        commit: uniform(rows, "commit", |row| &row.commit)?,
    })
}

fn validate_input_checksums(rows: &[BenchmarkRow]) -> Result<(), CatalogueEvidenceError> {
    for &(input_length, expected_checksum) in PUBLICATION_POLICY.input_checksums {
        let invalid = rows
            .iter()
            .filter(|row| {
                row.input_length == input_length && row.input_checksum != expected_checksum
            })
            .count();
        if invalid != 0 {
            return Err(CatalogueEvidenceError::new(format!(
                "cannot publish: {invalid} rows have noncanonical input checksum provenance for {input_length}"
            )));
        }
    }
    Ok(())
}

fn validate_case_provenance(rows: &[BenchmarkRow]) -> Result<(), CatalogueEvidenceError> {
    for case in MATRIX {
        let definitions = rows
            .iter()
            .filter(|row| row.case_id == case.id)
            .map(|row| {
                (
                    row.indicator_family.as_str(),
                    row.indicator_definition.as_str(),
                    row.parameters.as_str(),
                    row.output_kind.as_str(),
                    row.output_arity,
                )
            })
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([(
            case.family,
            case.definition,
            case.parameters,
            case.output_kind,
            Some(case.output_arity),
        )]);
        if definitions != expected {
            return Err(CatalogueEvidenceError::new(format!(
                "cannot publish case provenance that differs from catalogue-cases.tsv for {}: expected={expected:?}; actual={definitions:?}",
                case.id
            )));
        }
    }
    Ok(())
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

fn parse<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_opt<T>(value: &str, name: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
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

pub(crate) fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n', '|'], " ")
}
