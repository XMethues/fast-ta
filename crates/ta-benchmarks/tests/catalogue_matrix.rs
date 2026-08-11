use std::collections::BTreeSet;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use ta_benchmarks::catalogue_matrix::{
    catalogue_fixture, fixture_checksum, parse_criterion_diagnostics, parse_cycle_regression,
    parse_diagnostic_evidence, parse_platform_qualification, parse_raw_rows,
    read_criterion_diagnostics, read_cycle_regression, read_diagnostic_evidence,
    read_platform_qualification, read_raw_rows, render_report, render_report_with_comparison,
    timing_stats, validate_outputs, write_raw_rows, BenchmarkRow, CaseKind, OutputValues,
    TimingStats, VerifiedOutput, FIXTURE_ID, INPUT_LENGTHS, MATRIX, PATTERN_SHAPES, RAW_HEADER,
    RUST_CALLER_MODE,
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
fn raw_parser_rejects_malformed_numeric_evidence() {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "fast-ta-catalogue-numeric-evidence-{}-{id}.tsv",
        std::process::id()
    ));
    write_raw_rows(&path, &[row("fast-ta", RUST_CALLER_MODE, 256, 1_000.0)])
        .expect("write valid raw row");
    let raw = fs::read_to_string(&path).expect("read valid raw row");
    fs::remove_file(path).expect("remove valid raw row");
    let line = raw.lines().nth(1).expect("raw data row");
    let fields = line.split('\t').collect::<Vec<_>>();
    let mutations = [
        (9, "NaN", "positive and finite"),
        (10, "-1", "positive and finite"),
        (11, "inf", "positive and finite"),
        (10, "1001", "confidence interval"),
        (12, "1", "throughput is incoherent"),
        (13, "0", "sample_count must be positive"),
        (26, "0", "warmup_iterations must be positive"),
        (27, "0", "iterations_per_sample must be positive"),
        (27, "NA", "requires warmup_iterations"),
    ];
    for (column, value, message) in mutations {
        let mut malformed = fields.clone();
        malformed[column] = value;
        let input = format!("{RAW_HEADER}\n{}\n", malformed.join("\t"));
        let error = parse_raw_rows(&input).expect_err("malformed numeric evidence must fail");
        assert!(error.contains(message), "{error}");
    }
}

#[test]
fn committed_raw_matrices_pass_report_parse_validation() {
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    for name in [
        "catalogue_matrix_pre_optimization.tsv",
        "catalogue_matrix_optimized.tsv",
    ] {
        let rows = read_raw_rows(&baseline_dir.join(name)).expect("parse committed raw matrix");
        assert_eq!(rows.len(), 270, "{name}");
    }
}

#[test]
fn report_is_generated_from_all_committed_durable_evidence() {
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    let rows = read_raw_rows(&baseline_dir.join("catalogue_matrix_optimized.tsv"))
        .expect("parse final matrix");
    let baseline = read_raw_rows(&baseline_dir.join("catalogue_matrix_pre_optimization.tsv"))
        .expect("parse pre matrix");
    let diagnostic =
        read_diagnostic_evidence(&baseline_dir.join("issue_57_62_diagnostic_evidence.json"))
            .expect("parse diagnostic evidence");
    let criterion =
        read_criterion_diagnostics(&baseline_dir.join("issue_57_62_criterion_diagnostics.json"))
            .expect("parse Criterion diagnostics");
    let cycle = read_cycle_regression(&baseline_dir.join("issue61_cycle_regression.jsonl"))
        .expect("parse cycle regression");
    let qualifications = [
        "typprice_x86_f64_qualification.jsonl",
        "typprice_x86_f32_qualification.jsonl",
        "typprice_aarch64_f64_qualification.jsonl",
        "typprice_aarch64_f32_qualification.jsonl",
        "typprice_wasm_qualification.jsonl",
    ]
    .into_iter()
    .map(|name| read_platform_qualification(&baseline_dir.join(name)))
    .collect::<Result<Vec<_>, _>>()
    .expect("parse five platform qualifications");

    let report = render_report_with_comparison(
        &rows,
        &baseline,
        &diagnostic,
        &criterion,
        &cycle,
        &qualifications,
    )
    .expect("render canonical report");

    assert!(report.contains("Durable issue 57–62 diagnostic evidence"));
    assert!(report.contains("Issue 59 diagnostic record"));
    assert!(report.contains("Criterion same-session before/after diagnostics"));
    assert!(report.contains("Same-run semantic Rust/C pairs"));
    assert!(report.contains("Issue 61 Hilbert cycle regression control"));
    assert!(report.contains("fixed_by_clean_revert_to_b156ac1_implementation"));
    assert!(report.contains("typprice_aarch64_f32_qualification.jsonl"));
    assert!(report.contains("Complete >5% clean pre/final classification"));
    assert!(!report.contains("confirmed clean batch regression; source-correlated"));
}

#[test]
fn durable_diagnostic_parsers_reject_wrong_schemas() {
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    let diagnostic = fs::read_to_string(baseline_dir.join("issue_57_62_diagnostic_evidence.json"))
        .expect("read diagnostic fixture");
    let criterion = fs::read_to_string(baseline_dir.join("issue_57_62_criterion_diagnostics.json"))
        .expect("read Criterion fixture");
    let cycle = fs::read_to_string(baseline_dir.join("issue61_cycle_regression.jsonl"))
        .expect("read cycle fixture");

    assert!(parse_diagnostic_evidence(
        &diagnostic.replace("fast-ta.issue-57-62.diagnostic-evidence.v1", "wrong",)
    )
    .expect_err("wrong diagnostic schema must fail")
    .contains("schema"));
    assert!(parse_criterion_diagnostics(
        &criterion.replace("fast-ta.issue-57-62.criterion-diagnostics.v1", "wrong",)
    )
    .expect_err("wrong Criterion schema must fail")
    .contains("schema"));
    assert!(
        parse_cycle_regression(&cycle.replace("fast-ta.issue61.cycle-regression.v1", "wrong"))
            .expect_err("wrong cycle schema must fail")
            .contains("schema")
    );
}

#[test]
fn platform_qualification_parser_requires_provenance_validation_and_valid_timings() {
    let input = concat!(
        r#"{"record":"metadata","platform":"x86_64","precision":"f64","runtime":"rustc test","profile":"release","features":"simd-qualification","cpu":"test cpu","os":"test os","commit":"abc","workflow_run_id":42,"workflow_run_url":"https://example.test/runs/42","workflow_job_id":7,"active_backend":"avx2"}"#,
        "\n",
        r#"{"record":"validation","indicator":"TYPPRICE","unequal_lengths_verified":true,"non_finite_verified":true,"scalar_unequal_lengths_error":"length","scalar_non_finite_error":"finite","simd_unequal_lengths_error":"length","simd_non_finite_error":"finite"}"#,
        "\n",
        r#"{"record":"measurement","mode":"explicit kernel","backend":"scalar","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":100.0,"ci95_lower_ns":90.0,"ci95_upper_ns":110.0,"throughput_observations_per_second":2560000000.0,"sample_count":31,"timed_boundary":"kernel"}"#,
        "\n",
    );
    let qualification =
        parse_platform_qualification(input, "test.jsonl").expect("parse qualification JSONL");
    assert_eq!(qualification.workflow_run_id, "42");
    assert_eq!(qualification.workflow_job, "7");
    assert_eq!(qualification.profile, "release");
    assert_eq!(qualification.feature_flags, "simd-qualification");
    assert_eq!(qualification.os, "test os");
    assert_eq!(qualification.active_backend, "avx2");
    assert_eq!(qualification.measurements[0].median_ns, 100.0);

    for (needle, expected) in [
        ("\"workflow_run_id\":42,", "workflow_run_id"),
        ("\"precision\":\"f64\",", "precision"),
        ("\"profile\":\"release\",", "profile"),
        ("\"features\":\"simd-qualification\",", "features"),
        ("\"os\":\"test os\",", "os"),
    ] {
        let missing = input.replace(needle, "");
        let error = parse_platform_qualification(&missing, "test.jsonl")
            .expect_err("required metadata must fail");
        assert!(error.contains(expected), "{error}");
    }

    let missing_validation = input
        .lines()
        .filter(|line| !line.contains(r#""record":"validation""#))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        parse_platform_qualification(&missing_validation, "test.jsonl")
            .expect_err("validation evidence is required")
            .contains("no validation records")
    );

    let failed_validation = input.replace(
        r#""non_finite_verified":true"#,
        r#""non_finite_verified":false"#,
    );
    assert!(
        parse_platform_qualification(&failed_validation, "test.jsonl")
            .expect_err("failed validation evidence must fail")
            .contains("non_finite_verified=false")
    );

    for (needle, replacement, expected) in [
        (
            r#""median_ns":100.0"#,
            r#""median_ns":0.0"#,
            "positive and finite",
        ),
        (
            r#""ci95_lower_ns":90.0"#,
            r#""ci95_lower_ns":101.0"#,
            "confidence interval",
        ),
        (
            r#""throughput_observations_per_second":2560000000.0"#,
            r#""throughput_observations_per_second":1.0"#,
            "throughput is incoherent",
        ),
        (
            r#""sample_count":31"#,
            r#""sample_count":0"#,
            "sample_count",
        ),
    ] {
        let malformed = input.replace(needle, replacement);
        let error = parse_platform_qualification(&malformed, "test.jsonl")
            .expect_err("malformed timing evidence must fail");
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn platform_parser_accepts_complete_x86_metadata_and_backend_validation() {
    let input = concat!(
        r#"{"record":"metadata","platform":"x86_64","precision":"f32","runtime":"rustc test","rust_profile":"release","cargo_features":"f32,std,simd-qualification","target_features":"+avx2","cpu":"test cpu","os":"linux","commit":"abc","workflow_run_id":"42","workflow_run_url":"https://example.test/runs/42","workflow_job":7,"active_backend":"avx2"}"#,
        "\n",
        r#"{"record":"validation","backend":"avx2","public_boundary":true,"unequal_lengths_verified":true,"non_finite_verified":true,"short_output_verified":true,"errors_match_scalar":true,"unequal_lengths_error":"length","non_finite_error":"finite","short_output_error":"output"}"#,
        "\n",
        r#"{"record":"measurement","mode":"public TYPPRICE","backend":"avx2","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":100.0,"ci95_lower_ns":90.0,"ci95_upper_ns":110.0,"throughput_observations_per_second":2560000000.0,"sample_count":31,"timed_boundary":"public"}"#,
        "\n",
    );
    let qualification =
        parse_platform_qualification(input, "x86.jsonl").expect("parse complete x86 evidence");
    assert_eq!(qualification.workflow_run_id, "42");
    assert_eq!(qualification.workflow_job, "7");
    assert_eq!(qualification.profile, "release");
    assert!(qualification.feature_flags.contains("cargo_features=f32"));
    assert!(qualification
        .feature_flags
        .contains("target_features=+avx2"));
    let unmatched = input.replacen(r#""backend":"avx2""#, r#""backend":"scalar""#, 1);
    assert!(parse_platform_qualification(&unmatched, "x86.jsonl")
        .expect_err("equivalence requires matching backend validation")
        .contains("without a matching validation record"));
}

#[test]
fn platform_parser_accepts_aarch64_backend_validations_and_c_control() {
    let input = concat!(
        r#"{"record":"metadata","platform":"aarch64","precision":"f64","runtime":"rustc test","profile":"release","features":"simd-qualification","cpu":"test cpu","os":"linux","commit":"abc","workflow_run_id":42,"workflow_run_url":"https://example.test/runs/42","workflow_job":"aarch64-typprice","active_backend":"neon"}"#,
        "\n",
        r#"{"record":"validation","precision":"f64","backend":"scalar","mode":"public TYPPRICE","public_boundary":true,"exact_scalar_equivalence":true,"error_semantics_verified":true,"mismatched_length_error_equal_to_scalar":true,"non_finite_error_equal_to_scalar":true,"observed_backend":"scalar"}"#,
        "\n",
        r#"{"record":"validation","precision":"f64","backend":"neon","mode":"public TYPPRICE","public_boundary":true,"exact_scalar_equivalence":true,"error_semantics_verified":true,"mismatched_length_error_equal_to_scalar":true,"non_finite_error_equal_to_scalar":true,"observed_backend":"neon"}"#,
        "\n",
        r#"{"record":"measurement","mode":"public TYPPRICE","backend":"scalar","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":200.0,"ci95_lower_ns":190.0,"ci95_upper_ns":210.0,"throughput_observations_per_second":1280000000.0,"sample_count":31,"timed_boundary":"public"}"#,
        "\n",
        r#"{"record":"measurement","mode":"public TYPPRICE","backend":"neon","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":100.0,"ci95_lower_ns":90.0,"ci95_upper_ns":110.0,"throughput_observations_per_second":2560000000.0,"sample_count":31,"timed_boundary":"public"}"#,
        "\n",
        r#"{"record":"measurement","mode":"direct C caller-owned","backend":"ta-lib-c","input_length":256,"equivalent_to_scalar":true,"semantic_status":"verified","timing_status":"measured","median_ns":400.0,"ci95_lower_ns":390.0,"ci95_upper_ns":410.0,"throughput_observations_per_second":640000000.0,"sample_count":31,"timed_boundary":"C"}"#,
        "\n",
    );
    let qualification =
        parse_platform_qualification(input, "aarch64.jsonl").expect("parse aarch64 evidence");
    assert_eq!(qualification.platform, "aarch64");
    assert_eq!(qualification.active_backend, "neon");
    assert_eq!(qualification.measurements.len(), 3);
}

#[test]
fn committed_platform_qualifications_parse_or_name_missing_harness_fields() {
    let baseline_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines");
    let expected = [
        ("typprice_x86_f64_qualification.jsonl", "x86_64", "f64", 6),
        ("typprice_x86_f32_qualification.jsonl", "x86_64", "f32", 6),
        (
            "typprice_aarch64_f64_qualification.jsonl",
            "aarch64",
            "f64",
            9,
        ),
        (
            "typprice_aarch64_f32_qualification.jsonl",
            "aarch64",
            "f32",
            6,
        ),
        (
            "typprice_wasm_qualification.jsonl",
            "wasm32-unknown-unknown",
            "f64",
            6,
        ),
    ];
    for (name, platform, precision, measurement_count) in expected {
        match read_platform_qualification(&baseline_dir.join(name)) {
            Ok(qualification) => {
                assert_eq!(qualification.platform, platform);
                assert_eq!(qualification.precision, precision);
                assert!(!qualification.workflow_run_id.is_empty());
                assert_eq!(qualification.measurements.len(), measurement_count);
                assert!(qualification
                    .measurements
                    .iter()
                    .all(|measurement| measurement.equivalent_to_scalar));
            }
            Err(error) => assert!(
                ["profile", "features", "os", "validation"]
                    .iter()
                    .any(|field| error.contains(field)),
                "{name} failed without naming schema evidence needed from its qualification harness: {error}"
            ),
        }
    }
}
