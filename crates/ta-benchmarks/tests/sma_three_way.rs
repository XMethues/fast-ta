use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use ta_benchmarks::sma_three_way::{
    input_checksum, read_raw_rows, render_report, series_fixture, timing_stats, validate_outputs,
    write_raw_rows, BenchmarkRow, TimingStats, INPUT_LENGTHS,
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

fn row(implementation: &str) -> BenchmarkRow {
    BenchmarkRow {
        implementation: implementation.to_owned(),
        indicator_definition: "SMA: arithmetic mean".to_owned(),
        mode: "caller-owned".to_owned(),
        parameters: "timeperiod=14".to_owned(),
        input_length: 256,
        stats: TimingStats {
            median_ns: 1_000.0,
            ci95_lower_ns: 900.0,
            ci95_upper_ns: 1_100.0,
            throughput_observations_per_second: 256_000_000.0,
            sample_count: 50,
            outlier_count: 2,
            outlier_low_count: 1,
            outlier_high_count: 1,
        },
        output_begin: 13,
        output_count: 243,
        semantic_verified: true,
        warmup_iterations: 100,
        iterations_per_sample: 10,
        timed_boundary: "API call only".to_owned(),
        fixture: "series_fixture_v1".to_owned(),
        input_checksum: "fnv1a64:1234".to_owned(),
        ta_lib_version: "0.6.4".to_owned(),
        ta_lib_revision: "43f9d5042ecc4bd367941846494ad907bf20ea50".to_owned(),
        python_version: "3.13.2".to_owned(),
        python_binding_version: "0.6.4".to_owned(),
        python_ta_lib_version: "0.6.4".to_owned(),
        numpy_version: "2.2.3".to_owned(),
        rustc: "rustc test".to_owned(),
        cpu: "test cpu".to_owned(),
        os: "test os".to_owned(),
        arch: "test arch".to_owned(),
        float_width: 64,
        features: "default(f64,std),sma-three-way".to_owned(),
        commit: "deadbeef".to_owned(),
        dirty: false,
    }
}

#[test]
fn shared_fixture_is_finite_deterministic_and_prefix_stable() {
    let fixtures = INPUT_LENGTHS.map(|length| series_fixture(length, 0));
    for (fixture, expected_length) in fixtures.iter().zip(INPUT_LENGTHS) {
        assert_eq!(fixture.len(), expected_length);
        assert!(fixture.iter().all(|value| value.is_finite()));
        assert_eq!(fixture, &series_fixture(expected_length, 0));
        assert_eq!(
            input_checksum(fixture),
            input_checksum(&series_fixture(expected_length, 0))
        );
    }
    assert_eq!(&fixtures[1][..INPUT_LENGTHS[0]], fixtures[0].as_slice());
    assert_eq!(&fixtures[2][..INPUT_LENGTHS[1]], fixtures[1].as_slice());
}

#[test]
fn semantic_gate_rejects_range_count_and_value_mismatches() {
    let expected = [1.0, 2.0, 3.0];
    let range_error = validate_outputs((13, 3), &expected, "candidate", (14, 3), &expected)
        .expect_err("a shifted OutputRange must fail");
    assert!(range_error.contains("OutputRange mismatch"));

    let count_error = validate_outputs((13, 3), &expected, "candidate", (13, 2), &expected[..2])
        .expect_err("a short output must fail");
    assert!(
        count_error.contains("OutputRange mismatch")
            || count_error.contains("output count mismatch")
    );

    let value_error = validate_outputs((13, 3), &expected, "candidate", (13, 3), &[1.0, 2.01, 3.0])
        .expect_err("an out-of-tolerance value must fail");
    assert!(value_error.contains("value mismatch at compact index 1"));
}

#[test]
fn statistics_report_median_ci_throughput_samples_and_outliers() {
    let mut samples = (1..=49).map(|value| value as f64).collect::<Vec<_>>();
    samples.push(10_000.0);
    let stats = timing_stats(&samples, 256).expect("valid timing samples");
    assert_eq!(stats.sample_count, 50);
    assert_eq!(stats.median_ns, 25.5);
    assert!(stats.ci95_lower_ns <= stats.median_ns);
    assert!(stats.ci95_upper_ns >= stats.median_ns);
    assert_eq!(stats.outlier_count, 1);
    assert_eq!(stats.outlier_high_count, 1);
    assert!(stats.throughput_observations_per_second > 0.0);
}

#[test]
fn report_is_generated_by_rereading_raw_rows_and_refuses_unverified_rows() {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fast-ta-sma-three-way-{}-{id}.tsv",
        std::process::id()
    ));
    let rows = vec![row("fast-ta"), row("TA-Lib C"), row("TA-Lib Python")];
    write_raw_rows(&path, &rows).expect("write raw benchmark rows");
    let reread = read_raw_rows(&path).expect("read raw benchmark rows");
    let report = render_report(&reread).expect("render report from raw rows");
    fs::remove_file(&path).expect("remove test raw rows");

    assert_eq!(reread, rows);
    assert!(report.contains("Pinned SMA three-way performance benchmark"));
    assert!(report.contains("TA-Lib Python"));
    assert!(report.contains("[0.900, 1.100] us"));
    assert!(report.contains("2 (1 low, 1 high)"));

    let mut unverified = rows;
    unverified[1].semantic_verified = false;
    assert!(render_report(&unverified).is_err());
}
