//! Deterministic timing statistics for Catalogue measurements.

use std::cmp::Ordering;

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
pub(crate) fn validate_positive_timing_evidence(
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
