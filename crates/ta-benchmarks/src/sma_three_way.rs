//! Deterministic inputs, semantic gates, statistics, and reporting for the
//! opt-in SMA Rust/C/Python comparison.

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

pub const INPUT_LENGTHS: [usize; 3] = [256, 4_096, 65_536];
pub const PERIOD: usize = 14;
pub const ABS_TOLERANCE: f64 = 1.0e-9;
pub const REL_TOLERANCE: f64 = 1.0e-12;
pub const RAW_HEADER: &str = "implementation\tindicator_definition\tmode\tparameters\tinput_length\tmedian_ns\tci95_lower_ns\tci95_upper_ns\tthroughput_observations_per_second\tsample_count\toutlier_count\toutlier_low_count\toutlier_high_count\toutput_begin\toutput_count\tsemantic_verified\twarmup_iterations\titerations_per_sample\ttimed_boundary\tfixture\tinput_checksum\tta_lib_version\tta_lib_revision\tpython_version\tpython_binding_version\tpython_ta_lib_version\tnumpy_version\trustc\tcpu\tos\tarch\tfloat_width\tfeatures\tcommit\tdirty";

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

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkRow {
    pub implementation: String,
    pub indicator_definition: String,
    pub mode: String,
    pub parameters: String,
    pub input_length: usize,
    pub stats: TimingStats,
    pub output_begin: usize,
    pub output_count: usize,
    pub semantic_verified: bool,
    pub warmup_iterations: u64,
    pub iterations_per_sample: u64,
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

pub fn series_fixture(size: usize, seed: usize) -> Vec<f64> {
    (0..size)
        .map(|index| {
            let trend = index as f64 * 0.001;
            let cycle = ((index * 37 + seed * 17) % 101) as f64;
            trend + cycle + 1.0
        })
        .collect()
}

pub fn input_checksum(values: &[f64]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in values.iter().flat_map(|value| value.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

pub fn validate_outputs(
    expected_range: (usize, usize),
    expected: &[f64],
    implementation: &str,
    actual_range: (usize, usize),
    actual: &[f64],
) -> Result<(), String> {
    if actual_range != expected_range {
        return Err(format!(
            "{implementation} OutputRange mismatch: expected begin {} count {}, got begin {} count {}",
            expected_range.0, expected_range.1, actual_range.0, actual_range.1
        ));
    }
    if actual.len() != expected.len() || actual.len() != actual_range.1 {
        return Err(format!(
            "{implementation} output count mismatch: expected {}, got {} values for declared count {}",
            expected.len(),
            actual.len(),
            actual_range.1
        ));
    }
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            if expected.to_bits() != actual.to_bits() {
                return Err(format!(
                    "{implementation} non-finite placement mismatch at compact index {index}: expected {expected:?}, got {actual:?}"
                ));
            }
            continue;
        }
        let difference = (actual - expected).abs();
        let tolerance = ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs());
        if difference > tolerance {
            return Err(format!(
                "{implementation} value mismatch at compact index {index}: expected {expected:.17e}, got {actual:.17e}, difference {difference:.3e}, tolerance {tolerance:.3e}"
            ));
        }
    }
    Ok(())
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
    let mut state = 0x736d615f63695f35_u64 ^ samples.len() as u64;
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
        return Err("unexpected SMA raw-row header".to_owned());
    }
    lines
        .enumerate()
        .filter(|(_, line)| !line.is_empty())
        .map(|(index, line)| {
            parse_row(line).map_err(|error| format!("raw row {}: {error}", index + 2))
        })
        .collect()
}

pub fn render_report(rows: &[BenchmarkRow]) -> Result<String, String> {
    if rows.is_empty() {
        return Err("cannot render an SMA report without raw rows".to_owned());
    }
    if rows.iter().any(|row| !row.semantic_verified) {
        return Err("cannot publish timings with an unverified semantic row".to_owned());
    }

    let first = &rows[0];
    let mut report = String::from("Pinned SMA three-way performance benchmark\n\n");
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
        "Commit {} (dirty: {}) | {} | {} | {}\n",
        first.commit, first.dirty, first.cpu, first.os, first.rustc
    ));
    report.push_str("95% intervals are deterministic bootstrap confidence intervals for the median (10,000 resamples). Outliers are observations outside Tukey's 1.5 IQR fences.\n\n");
    report.push_str("| Input | Implementation | Mode | Median | 95% CI | Throughput | Samples | Outliers | Output Range |\n");
    report.push_str("|---:|---|---|---:|---:|---:|---:|---:|---:|\n");

    let mut ordered = rows.to_vec();
    ordered.sort_by(|left, right| {
        left.input_length
            .cmp(&right.input_length)
            .then_with(|| left.implementation.cmp(&right.implementation))
    });
    for row in ordered {
        report.push_str(&format!(
            "| {} | {} | {} | {:.3} us | [{:.3}, {:.3}] us | {:.3} Mobs/s | {} | {} ({} low, {} high) | {}..{} |\n",
            row.input_length,
            row.implementation,
            row.mode,
            row.stats.median_ns / 1_000.0,
            row.stats.ci95_lower_ns / 1_000.0,
            row.stats.ci95_upper_ns / 1_000.0,
            row.stats.throughput_observations_per_second / 1.0e6,
            row.stats.sample_count,
            row.stats.outlier_count,
            row.stats.outlier_low_count,
            row.stats.outlier_high_count,
            row.output_begin,
            row.output_begin + row.output_count
        ));
    }
    Ok(report)
}

fn format_row(row: &BenchmarkRow) -> String {
    [
        clean(&row.implementation),
        clean(&row.indicator_definition),
        clean(&row.mode),
        clean(&row.parameters),
        row.input_length.to_string(),
        format!("{:.6}", row.stats.median_ns),
        format!("{:.6}", row.stats.ci95_lower_ns),
        format!("{:.6}", row.stats.ci95_upper_ns),
        format!("{:.6}", row.stats.throughput_observations_per_second),
        row.stats.sample_count.to_string(),
        row.stats.outlier_count.to_string(),
        row.stats.outlier_low_count.to_string(),
        row.stats.outlier_high_count.to_string(),
        row.output_begin.to_string(),
        row.output_count.to_string(),
        row.semantic_verified.to_string(),
        row.warmup_iterations.to_string(),
        row.iterations_per_sample.to_string(),
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
    let columns = line.split('\t').collect::<Vec<_>>();
    if columns.len() != 35 {
        return Err(format!("expected 35 columns, got {}", columns.len()));
    }
    let usize_at = |index: usize| {
        columns[index]
            .parse::<usize>()
            .map_err(|error| error.to_string())
    };
    let u64_at = |index: usize| {
        columns[index]
            .parse::<u64>()
            .map_err(|error| error.to_string())
    };
    let f64_at = |index: usize| {
        columns[index]
            .parse::<f64>()
            .map_err(|error| error.to_string())
    };
    let bool_at = |index: usize| {
        columns[index]
            .parse::<bool>()
            .map_err(|error| error.to_string())
    };
    Ok(BenchmarkRow {
        implementation: columns[0].to_owned(),
        indicator_definition: columns[1].to_owned(),
        mode: columns[2].to_owned(),
        parameters: columns[3].to_owned(),
        input_length: usize_at(4)?,
        stats: TimingStats {
            median_ns: f64_at(5)?,
            ci95_lower_ns: f64_at(6)?,
            ci95_upper_ns: f64_at(7)?,
            throughput_observations_per_second: f64_at(8)?,
            sample_count: usize_at(9)?,
            outlier_count: usize_at(10)?,
            outlier_low_count: usize_at(11)?,
            outlier_high_count: usize_at(12)?,
        },
        output_begin: usize_at(13)?,
        output_count: usize_at(14)?,
        semantic_verified: bool_at(15)?,
        warmup_iterations: u64_at(16)?,
        iterations_per_sample: u64_at(17)?,
        timed_boundary: columns[18].to_owned(),
        fixture: columns[19].to_owned(),
        input_checksum: columns[20].to_owned(),
        ta_lib_version: columns[21].to_owned(),
        ta_lib_revision: columns[22].to_owned(),
        python_version: columns[23].to_owned(),
        python_binding_version: columns[24].to_owned(),
        python_ta_lib_version: columns[25].to_owned(),
        numpy_version: columns[26].to_owned(),
        rustc: columns[27].to_owned(),
        cpu: columns[28].to_owned(),
        os: columns[29].to_owned(),
        arch: columns[30].to_owned(),
        float_width: usize_at(31)?,
        features: columns[32].to_owned(),
        commit: columns[33].to_owned(),
        dirty: bool_at(34)?,
    })
}

fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}
