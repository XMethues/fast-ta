use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use ta_benchmarks::catalogue_matrix::{
    catalogue_fixture, fixture_checksum, parse_optimization_evidence, parse_platform_qualification,
    read_platform_qualification, read_raw_rows, render_report, render_report_with_comparison,
    timing_stats, validate_outputs, write_raw_rows, BenchmarkRow, CaseKind,
    OptimizationEvidenceRow, OutputValues, TimingStats, VerifiedOutput, C_DIRECT_MODE, FIXTURE_ID,
    INPUT_LENGTHS, MATRIX, PATTERN_SHAPES, RUST_CALLER_MODE,
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

fn row(implementation: &str, mode: &str, input_length: usize, median_ns: f64) -> BenchmarkRow {
    BenchmarkRow {
        implementation: implementation.to_owned(),
        indicator_family: "Overlap Studies".to_owned(),
        indicator_definition: "SMA: Simple Moving Average".to_owned(),
        case_id: "SMA".to_owned(),
        mode: mode.to_owned(),
        parameters: "timeperiod=14".to_owned(),
        input_length,
        output_kind: "float".to_owned(),
        output_arity: Some(1),
        stats: Some(TimingStats {
            median_ns,
            ci95_lower_ns: median_ns * 0.9,
            ci95_upper_ns: median_ns * 1.1,
            throughput_observations_per_second: input_length as f64 * 1.0e9 / median_ns,
            sample_count: 50,
            outlier_count: 2,
            outlier_low_count: 1,
            outlier_high_count: 1,
        }),
        output_begin: Some(13),
        output_count: Some(input_length - 13),
        output_checksum: "fnv1a64:5678".to_owned(),
        semantic_status: "verified".to_owned(),
        semantic_reason: String::new(),
        timing_status: "measured".to_owned(),
        timing_reason: String::new(),
        comparison_status: "comparable".to_owned(),
        comparison_reason: "primary caller-owned Rust/C kernel".to_owned(),
        warmup_iterations: Some(100),
        iterations_per_sample: Some(10),
        timed_boundary: "API call only".to_owned(),
        fixture: FIXTURE_ID.to_owned(),
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
        features: "default(f64,std),catalogue-matrix".to_owned(),
        commit: "deadbeef".to_owned(),
        dirty: false,
    }
}

#[test]
fn matrix_is_the_representative_ticket_catalogue() {
    let ids = MATRIX.iter().map(|case| case.id).collect::<BTreeSet<_>>();
    let required = [
        "SMA",
        "BBANDS",
        "RSI",
        "MACD",
        "ATR",
        "ADX",
        "HT_DCPHASE",
        "CDLDOJI",
    ];
    assert!(required.iter().all(|id| ids.contains(id)));
    assert_eq!(MATRIX.len(), 15);
    assert_eq!(
        MATRIX
            .iter()
            .map(|case| case.family)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "Overlap Studies",
            "Momentum Indicators",
            "Volatility Indicators",
            "Cycle Indicators",
            "Pattern Recognition",
            "Statistic Functions",
            "Price Transform",
            "Volume Indicators",
            "Math Transform",
            "Math Operators",
        ])
    );
    assert_eq!(INPUT_LENGTHS, [256, 4_096, 65_536]);
    assert!(MATRIX.iter().all(|case| !case.parameters.is_empty()));
}

#[test]
fn pattern_cases_cover_single_multi_and_setting_free_stateful_shapes() {
    let expected = [
        (
            "CDLDOJI",
            CaseKind::CdlDoji,
            "single-setting stateful rolling average",
            "BodyDoji",
        ),
        (
            "CDL3WHITESOLDIERS",
            CaseKind::Cdl3WhiteSoldiers,
            "multi-setting stateful rolling averages",
            "ShadowVeryShort, BodyShort, Far, and Near",
        ),
        (
            "CDLENGULFING",
            CaseKind::CdlEngulfing,
            "setting-free cross-candle predicate",
            "Streaming retains cross-candle state",
        ),
    ];
    let pattern_cases = MATRIX
        .iter()
        .filter(|case| case.family == "Pattern Recognition")
        .collect::<Vec<_>>();
    assert_eq!(pattern_cases.len(), expected.len());

    let mut report_rows = Vec::new();
    for (case_id, kind, execution_shape, rationale_fragment) in expected {
        let case = pattern_cases
            .iter()
            .find(|case| case.id == case_id)
            .expect("representative Pattern case");
        assert_eq!(case.kind, kind);
        assert_eq!(case.parameters, "candle_settings=TA-Lib defaults");

        let shape = PATTERN_SHAPES
            .iter()
            .find(|shape| shape.case_id == case_id)
            .expect("Pattern execution-shape metadata");
        assert_eq!(shape.execution_shape, execution_shape);
        assert!(shape.rationale.contains(rationale_fragment));

        let mut report_row = row("fast-ta", RUST_CALLER_MODE, 256, 1_000.0);
        report_row.indicator_family = case.family.to_owned();
        report_row.indicator_definition = case.definition.to_owned();
        report_row.case_id = case.id.to_owned();
        report_row.parameters = case.parameters.to_owned();
        report_row.output_kind = case.output_kind.to_owned();
        report_rows.push(report_row);
    }

    let report = render_report(&report_rows).expect("render Pattern shape metadata");
    for shape in PATTERN_SHAPES {
        assert!(report.contains(shape.case_id));
        assert!(report.contains(shape.execution_shape));
        assert!(report.contains(shape.rationale));
    }
}

#[test]
fn shared_fixture_is_finite_prefix_stable_and_ohlcv_valid() {
    let fixtures = INPUT_LENGTHS.map(catalogue_fixture);
    for (fixture, expected_length) in fixtures.iter().zip(INPUT_LENGTHS) {
        assert_eq!(fixture.len(), expected_length);
        fixture
            .validate()
            .expect("valid deterministic OHLCV fixture");
        assert_eq!(fixture, &catalogue_fixture(expected_length));
        assert_eq!(
            fixture_checksum(fixture),
            fixture_checksum(&catalogue_fixture(expected_length))
        );
    }
    let fields: [for<'a> fn(&'a ta_benchmarks::catalogue_matrix::Fixture) -> &'a [f64]; 6] = [
        |fixture| fixture.open.as_slice(),
        |fixture| fixture.high.as_slice(),
        |fixture| fixture.low.as_slice(),
        |fixture| fixture.close.as_slice(),
        |fixture| fixture.volume.as_slice(),
        |fixture| fixture.auxiliary.as_slice(),
    ];
    for field in fields {
        assert_eq!(
            &field(&fixtures[1])[..INPUT_LENGTHS[0]],
            field(&fixtures[0])
        );
        assert_eq!(
            &field(&fixtures[2])[..INPUT_LENGTHS[1]],
            field(&fixtures[1])
        );
    }
}

#[test]
fn semantic_gate_checks_range_arity_float_values_and_exact_integer_signals() {
    let floats = VerifiedOutput {
        begin: 13,
        count: 3,
        values: OutputValues::Float(vec![vec![1.0, 2.0, 3.0]]),
    };
    let shifted = VerifiedOutput {
        begin: 14,
        ..floats.clone()
    };
    assert!(validate_outputs(&floats, "candidate", &shifted)
        .expect_err("shifted range must fail")
        .contains("OutputRange mismatch"));

    let wrong_arity = VerifiedOutput {
        begin: 13,
        count: 3,
        values: OutputValues::Float(vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]]),
    };
    assert!(validate_outputs(&floats, "candidate", &wrong_arity)
        .expect_err("wrong output arity must fail")
        .contains("output arity mismatch"));

    let wrong_float = VerifiedOutput {
        begin: 13,
        count: 3,
        values: OutputValues::Float(vec![vec![1.0, 2.01, 3.0]]),
    };
    assert!(validate_outputs(&floats, "candidate", &wrong_float)
        .expect_err("out-of-tolerance float must fail")
        .contains("column 0 compact index 1"));

    let integers = VerifiedOutput {
        begin: 10,
        count: 3,
        values: OutputValues::Integer(vec![vec![0, 100, -100]]),
    };
    let wrong_integer = VerifiedOutput {
        begin: 10,
        count: 3,
        values: OutputValues::Integer(vec![vec![0, 80, -100]]),
    };
    assert!(validate_outputs(&integers, "candidate", &wrong_integer)
        .expect_err("Pattern Signal codes must compare exactly")
        .contains("exact integer mismatch"));
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
fn report_is_generated_from_reread_rows_and_separates_comparable_from_unavailable() {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fast-ta-catalogue-matrix-{}-{id}.tsv",
        std::process::id()
    ));
    let mut rows = Vec::new();
    for input_length in INPUT_LENGTHS {
        rows.push(row("fast-ta", RUST_CALLER_MODE, input_length, 2_000.0));
        rows.push(row("TA-Lib C", C_DIRECT_MODE, input_length, 1_000.0));
    }
    let mut python = row("TA-Lib Python", "official Python NumPy API", 256, 3_000.0);
    python.comparison_status = "unavailable".to_owned();
    python.comparison_reason = "user-facing API is not a caller-owned kernel".to_owned();
    rows.push(python);
    let mut mismatch = row("fast-ta", "Streaming Computation", 256, 2_500.0);
    mismatch.stats = None;
    mismatch.semantic_status = "mismatch".to_owned();
    mismatch.semantic_reason = "exact integer mismatch at compact index 7".to_owned();
    mismatch.timing_status = "suppressed".to_owned();
    mismatch.timing_reason = "semantic gate failed".to_owned();
    mismatch.comparison_status = "unavailable".to_owned();
    mismatch.comparison_reason = "separate Rust execution cost".to_owned();
    mismatch.warmup_iterations = None;
    mismatch.iterations_per_sample = None;
    rows.push(mismatch);

    write_raw_rows(&path, &rows).expect("write raw benchmark rows");
    let reread = read_raw_rows(&path).expect("read raw benchmark rows");
    let baseline = INPUT_LENGTHS
        .into_iter()
        .flat_map(|input_length| {
            [
                row("fast-ta", RUST_CALLER_MODE, input_length, 4_000.0),
                row("TA-Lib C", C_DIRECT_MODE, input_length, 1_000.0),
            ]
        })
        .collect::<Vec<_>>();
    let evidence = [OptimizationEvidenceRow {
        ticket: "test-ticket".to_owned(),
        case_ids: vec!["SMA".to_owned()],
        focused_commands: vec![
            "python3 crates/ta-benchmarks/scripts/run_catalogue_matrix.py --case SMA".to_owned(),
        ],
        hypotheses: vec![
            "validation dominates".to_owned(),
            "batch state dominates".to_owned(),
            "output writes dominate".to_owned(),
        ],
        confirmed_evidence_kind: "source".to_owned(),
        confirmed_evidence: "test source seam".to_owned(),
        neighboring_workloads: "EMA".to_owned(),
        neighboring_disposition: "not measured in test fixture".to_owned(),
    }];
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    let qualifications = [
        "typprice_x86_f64_qualification.jsonl",
        "typprice_x86_f32_qualification.jsonl",
        "typprice_wasm_qualification.jsonl",
    ]
    .map(|name| {
        read_platform_qualification(&baseline_dir.join(name))
            .expect("read committed qualification for renderer")
    });
    let report = render_report_with_comparison(&reread, &baseline, &evidence, &qualifications)
        .expect("render report from reread raw rows");
    fs::remove_file(&path).expect("remove test raw rows");

    assert_eq!(reread, rows);
    assert!(report.contains("Pinned representative Indicator Catalogue performance matrix"));
    assert!(report.contains("Geometric Rust/C latency ratio"));
    assert!(report.contains("2.000x"));
    assert!(report.contains("unavailable: user-facing API is not a caller-owned kernel"));
    assert!(report.contains("exact integer mismatch at compact index 7"));
    assert!(report.contains("256 | caller-owned Batch Computation vs direct C caller-owned"));
    assert!(report.contains("4096 | caller-owned Batch Computation vs direct C caller-owned"));
    assert!(report.contains("65536 | caller-owned Batch Computation vs direct C caller-owned"));
    assert!(report.contains("Run completeness and semantic gate"));
    assert!(report.contains("Per-case caller-owned Rust/C latency ratios"));
    assert!(report.contains("Clean before/after optimization effects"));
    assert!(report.contains("4.000 us [3.600, 4.400]"));
    assert!(report.contains("Runtime platform qualification from committed JSONL"));
    assert!(report.contains("run [31515400920]"));
    assert!(report.contains("practical benefit on this runner"));
    assert!(report.contains("Canonical comparison baseline"));
    assert!(!report.contains("clean baseline missing"));
}

#[test]
fn durable_evidence_requires_ranked_hypotheses_and_source_kind() {
    let input = concat!(
        "ticket\tcase_ids\tfocused_commands\thypotheses\tconfirmed_evidence_kind\tconfirmed_evidence\tneighboring_workloads\tneighboring_disposition\n",
        "issue\tADX\tpython3 runner.py --case ADX\tvalidation || state || writes\tsource\tAdxBatchState\tADXR\tnot represented\n",
    );
    let rows = parse_optimization_evidence(input).expect("parse durable evidence");
    assert_eq!(rows[0].case_ids, ["ADX"]);
    assert_eq!(rows[0].hypotheses.len(), 3);

    let too_few = input.replace("validation || state || writes", "validation || state");
    assert!(parse_optimization_evidence(&too_few)
        .expect_err("two hypotheses must fail")
        .contains("3 to 5"));
}

#[test]
fn platform_qualification_parser_requires_provenance_and_preserves_timings() {
    let input = concat!(
        r#"{"record":"metadata","platform":"x86_64","precision":"f64","runtime":"rustc test","cpu":"test cpu","commit":"abc","workflow_run_id":42,"workflow_run_url":"https://example.test/runs/42","workflow_job_id":7,"active_backend":"avx2"}"#,
        "\n",
        r#"{"record":"measurement","mode":"explicit kernel","backend":"scalar","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":100.0,"ci95_lower_ns":90.0,"ci95_upper_ns":110.0,"throughput_observations_per_second":2560000000.0,"sample_count":31,"timed_boundary":"kernel"}"#,
        "\n",
    );
    let qualification =
        parse_platform_qualification(input, "test.jsonl").expect("parse qualification JSONL");
    assert_eq!(qualification.workflow_run_id, "42");
    assert_eq!(qualification.active_backend, "avx2");
    assert_eq!(qualification.measurements[0].median_ns, 100.0);

    let missing_run = input.replace("\"workflow_run_id\":42,", "");
    assert!(parse_platform_qualification(&missing_run, "test.jsonl")
        .expect_err("workflow run is required")
        .contains("workflow_run_id"));
}

#[test]
fn committed_platform_qualification_artifacts_are_runtime_evidence() {
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    let expected = [
        ("typprice_x86_f64_qualification.jsonl", "x86_64", "f64", 12),
        ("typprice_x86_f32_qualification.jsonl", "x86_64", "f32", 12),
        (
            "typprice_wasm_qualification.jsonl",
            "wasm32-unknown-unknown",
            "f64",
            6,
        ),
    ];
    for (name, platform, precision, measurement_count) in expected {
        let qualification = read_platform_qualification(&baseline_dir.join(name))
            .expect("read committed qualification");
        assert_eq!(qualification.platform, platform);
        assert_eq!(qualification.precision, precision);
        assert_eq!(qualification.workflow_run_id, "31515400920");
        assert_eq!(qualification.measurements.len(), measurement_count);
        assert!(qualification
            .measurements
            .iter()
            .all(|measurement| measurement.equivalent_to_scalar));
    }
}
