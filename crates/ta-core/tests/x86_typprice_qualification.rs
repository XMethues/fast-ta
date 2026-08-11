#![cfg(all(target_arch = "x86_64", feature = "simd-qualification"))]

use std::{
    fmt::Write as _,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    time::Instant,
};

use ta_core::{
    price_transform::TYPPRICE,
    simd::dispatch::{
        active_indicator_backend,
        qualification::{backend_available, with_indicator_backend},
        IndicatorBackend,
    },
    Float, TalibError,
};

const LENGTHS: [usize; 3] = [256, 4_096, 65_536];
const SAMPLE_COUNT: usize = 31;
const BOOTSTRAP_RESAMPLES: usize = 2_000;

fn fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    let close = (0..size)
        .map(|index| {
            let trend = index as Float * 0.001 as Float;
            let cycle = ((index * 37) % 101) as Float;
            trend + cycle + 1.0 as Float
        })
        .collect::<Vec<_>>();
    let open = close
        .iter()
        .enumerate()
        .map(|(index, value)| *value + ((index % 9) as Float - 4.0 as Float) * 0.035 as Float)
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| {
            open.max(*close) + 0.5 as Float + (index % 11) as Float * 0.03 as Float
        })
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| {
            open.min(*close) - 0.5 as Float - (index % 7) as Float * 0.025 as Float
        })
        .collect::<Vec<_>>();
    (high, low, close)
}

fn iterations(size: usize) -> usize {
    match size {
        256 => 2_000,
        4_096 => 300,
        65_536 => 20,
        _ => unreachable!(),
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn confidence_interval(samples: &[f64]) -> (f64, f64) {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut resample = vec![0.0; samples.len()];
    for _ in 0..BOOTSTRAP_RESAMPLES {
        for value in &mut resample {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *value = samples[(state as usize) % samples.len()];
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    (
        medians[BOOTSTRAP_RESAMPLES * 25 / 1_000],
        medians[BOOTSTRAP_RESAMPLES * 975 / 1_000],
    )
}

fn checksum(values: &[Float]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut hash_bytes = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    };
    hash_bytes(b"float");
    hash_bytes(&0_u64.to_le_bytes());
    hash_bytes(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash_bytes(&value.to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            value if value.is_control() => {
                write!(escaped, "\\u{:04x}", value as u32).unwrap();
            }
            value => escaped.push(value),
        }
    }
    escaped.push('"');
    escaped
}

#[derive(Clone, Copy)]
struct Measurement {
    median_ns: f64,
    lower_ns: f64,
    upper_ns: f64,
}

fn measure(
    backend: IndicatorBackend,
    high: &[Float],
    low: &[Float],
    close: &[Float],
    output: &mut [Float],
) -> Measurement {
    with_indicator_backend(backend, || {
        let mut run = || {
            let range = TYPPRICE(
                black_box(high),
                black_box(low),
                black_box(close),
                black_box(&mut *output),
            )
            .unwrap();
            assert_eq!((range.beg_idx, range.nb_element), (0, high.len()));
            black_box(output[output.len() - 1]);
        };

        for _ in 0..10 {
            run();
        }
        let iterations = iterations(high.len());
        let mut samples = Vec::with_capacity(SAMPLE_COUNT);
        for _ in 0..SAMPLE_COUNT {
            let started = Instant::now();
            for _ in 0..iterations {
                run();
            }
            samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
        }
        let (lower_ns, upper_ns) = confidence_interval(&samples);
        Measurement {
            median_ns: median(&mut samples),
            lower_ns,
            upper_ns,
        }
    })
}

#[derive(Debug, PartialEq, Eq)]
struct ValidationEvidence {
    unequal_lengths_error: String,
    non_finite_error: String,
    short_output_error: String,
}

fn validate_public_boundary(backend: IndicatorBackend) -> ValidationEvidence {
    with_indicator_backend(backend, || {
        let mut mismatch_output = [91.0 as Float; 2];
        let unequal_lengths = TYPPRICE(
            &[2.0 as Float, 3.0 as Float],
            &[1.0 as Float],
            &[1.5 as Float, 2.5 as Float],
            &mut mismatch_output,
        )
        .unwrap_err();
        assert!(matches!(&unequal_lengths, TalibError::InvalidInput { .. }));
        assert_eq!(mismatch_output, [91.0 as Float; 2]);

        let mut non_finite_output = [92.0 as Float; 1];
        let non_finite = TYPPRICE(
            &[2.0 as Float],
            &[Float::NAN],
            &[1.5 as Float],
            &mut non_finite_output,
        )
        .unwrap_err();
        assert!(matches!(&non_finite, TalibError::InvalidInput { .. }));
        assert_eq!(non_finite_output, [92.0 as Float; 1]);

        let mut short_output = [93.0 as Float; 1];
        let short = TYPPRICE(
            &[2.0 as Float, 3.0 as Float],
            &[1.0 as Float, 2.0 as Float],
            &[1.5 as Float, 2.5 as Float],
            &mut short_output,
        )
        .unwrap_err();
        assert!(matches!(&short, TalibError::InvalidInput { .. }));
        assert_eq!(short_output, [93.0 as Float; 1]);

        ValidationEvidence {
            unequal_lengths_error: unequal_lengths.to_string(),
            non_finite_error: non_finite.to_string(),
            short_output_error: short.to_string(),
        }
    })
}

fn available_backends() -> Vec<IndicatorBackend> {
    [
        IndicatorBackend::Scalar,
        IndicatorBackend::Avx2,
        IndicatorBackend::Avx512,
    ]
    .into_iter()
    .filter(|backend| backend_available(*backend))
    .collect()
}

fn metadata(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set by the qualification script"))
}

#[test]
fn forced_backends_share_public_equivalence_and_error_boundaries() {
    let scalar_validation = validate_public_boundary(IndicatorBackend::Scalar);
    let (high, low, close) = fixture(257);
    let mut scalar = vec![0.0 as Float; high.len()];
    with_indicator_backend(IndicatorBackend::Scalar, || {
        TYPPRICE(&high, &low, &close, &mut scalar).unwrap();
    });

    for backend in available_backends() {
        let mut output = vec![0.0 as Float; high.len()];
        with_indicator_backend(backend, || {
            assert_eq!(active_indicator_backend(), backend);
            TYPPRICE(&high, &low, &close, &mut output).unwrap();
        });
        assert_eq!(output, scalar, "{} output", backend.as_str());
        assert_eq!(
            validate_public_boundary(backend),
            scalar_validation,
            "{} errors",
            backend.as_str()
        );
    }
}

#[test]
#[ignore = "platform qualification is run explicitly by the qualification workflow"]
fn qualify_public_typprice_on_x86_simd() {
    assert!(
        backend_available(IndicatorBackend::Avx2),
        "x86 qualification requires an AVX2-capable runner"
    );
    let active = active_indicator_backend();
    let expected_active = if cfg!(feature = "f32") {
        if backend_available(IndicatorBackend::Avx512) {
            IndicatorBackend::Avx512
        } else {
            IndicatorBackend::Avx2
        }
    } else {
        IndicatorBackend::Scalar
    };
    assert_eq!(
        active, expected_active,
        "runtime dispatch selected a backend without matching-boundary benefit"
    );

    let output_path = metadata("QUALIFICATION_OUTPUT");
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut evidence = BufWriter::new(File::create(&output_path).unwrap());
    let precision = if cfg!(feature = "f32") { "f32" } else { "f64" };
    let fixture_id = if cfg!(feature = "f32") {
        "catalogue_fixture_v1:f32le"
    } else {
        "catalogue_fixture_v1:f64le"
    };
    let backends = available_backends();
    writeln!(
        evidence,
        "{{\"record\":\"metadata\",\"indicator\":\"TYPPRICE\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"parameters\":\"none\",\"fixture\":{},\"platform\":\"x86_64\",\"os\":{},\"architecture\":{},\"precision\":\"{}\",\"runtime\":{},\"rust_profile\":{},\"profile\":{},\"cargo_features\":{},\"features\":{},\"target_features\":{},\"cpu\":{},\"commit\":{},\"qualification_command\":{},\"workflow_run_id\":{},\"workflow_run_url\":{},\"workflow_job\":{},\"active_backend\":\"{}\",\"avx2_available\":true,\"avx512_available\":{}}}",
        json_string(fixture_id),
        json_string(&metadata("QUALIFICATION_OS")),
        json_string(&metadata("QUALIFICATION_ARCHITECTURE")),
        precision,
        json_string(&metadata("QUALIFICATION_RUNTIME")),
        json_string(&metadata("QUALIFICATION_RUST_PROFILE")),
        json_string(&metadata("QUALIFICATION_RUST_PROFILE")),
        json_string(&metadata("QUALIFICATION_CARGO_FEATURES")),
        json_string(&metadata("QUALIFICATION_CARGO_FEATURES")),
        json_string(&metadata("QUALIFICATION_TARGET_FEATURES")),
        json_string(&metadata("QUALIFICATION_CPU")),
        json_string(&metadata("QUALIFICATION_COMMIT")),
        json_string(&metadata("QUALIFICATION_COMMAND")),
        json_string(&metadata("QUALIFICATION_WORKFLOW_RUN_ID")),
        json_string(&metadata("QUALIFICATION_WORKFLOW_RUN_URL")),
        json_string(&metadata("QUALIFICATION_WORKFLOW_JOB")),
        active.as_str(),
        backend_available(IndicatorBackend::Avx512),
    )
    .unwrap();

    let scalar_validation = validate_public_boundary(IndicatorBackend::Scalar);
    for backend in &backends {
        let validation = validate_public_boundary(*backend);
        assert_eq!(
            validation,
            scalar_validation,
            "{} public errors differ from scalar",
            backend.as_str()
        );
        writeln!(
            evidence,
            "{{\"record\":\"validation\",\"indicator\":\"TYPPRICE\",\"backend\":\"{}\",\"public_boundary\":true,\"unequal_lengths_verified\":true,\"non_finite_verified\":true,\"short_output_verified\":true,\"errors_match_scalar\":true,\"unequal_lengths_error\":{},\"non_finite_error\":{},\"short_output_error\":{}}}",
            backend.as_str(),
            json_string(&validation.unequal_lengths_error),
            json_string(&validation.non_finite_error),
            json_string(&validation.short_output_error),
        )
        .unwrap();
    }

    let mut selected_regressions = Vec::new();
    for size in LENGTHS {
        let (high, low, close) = fixture(size);
        let mut scalar = vec![0.0 as Float; size];
        with_indicator_backend(IndicatorBackend::Scalar, || {
            let range = TYPPRICE(&high, &low, &close, &mut scalar).unwrap();
            assert_eq!((range.beg_idx, range.nb_element), (0, size));
        });
        let expected_checksum = checksum(&scalar);

        let mut measurements = Vec::with_capacity(backends.len());
        for backend in &backends {
            let mut output = vec![0.0 as Float; size];
            with_indicator_backend(*backend, || {
                let range = TYPPRICE(&high, &low, &close, &mut output).unwrap();
                assert_eq!((range.beg_idx, range.nb_element), (0, size));
            });
            assert_eq!(
                output,
                scalar,
                "{} public boundary differs at length {size}",
                backend.as_str()
            );
            let timing = measure(*backend, &high, &low, &close, &mut output);
            measurements.push((*backend, timing));
        }

        let scalar_median_ns = measurements
            .iter()
            .find(|(backend, _)| *backend == IndicatorBackend::Scalar)
            .unwrap()
            .1
            .median_ns;
        for (backend, timing) in measurements {
            let scalar_ratio = timing.median_ns / scalar_median_ns;
            let slower_than_scalar_pct = (scalar_ratio - 1.0) * 100.0;
            let exceeds_5_percent = scalar_ratio > 1.05;
            let selected = backend == active;
            let disposition = if backend == IndicatorBackend::Scalar {
                "scalar control"
            } else if exceeds_5_percent && selected {
                selected_regressions.push((backend, size, slower_than_scalar_pct));
                "invalid selection: accelerated backend exceeds the 5% scalar gate"
            } else if exceeds_5_percent {
                "not selected: accelerated backend exceeds the 5% scalar gate"
            } else if selected {
                "selected: accelerated backend is within the 5% scalar gate"
            } else {
                "qualified but not selected: a higher-priority backend is active"
            };
            writeln!(
                evidence,
                "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"public TYPPRICE\",\"backend\":\"{}\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"error_semantics_verified\":true,\"same_public_boundary\":true,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"scalar_ratio\":{:.6},\"slower_than_scalar_pct\":{:.3},\"exceeds_5_percent\":{},\"selected_for_public_dispatch\":{},\"disposition\":{},\"fixture\":{},\"timed_boundary\":\"public TYPPRICE; validation included; caller-owned output; qualification override outside timed region\"}}",
                backend.as_str(),
                size,
                size,
                expected_checksum,
                SAMPLE_COUNT,
                timing.median_ns,
                timing.lower_ns,
                timing.upper_ns,
                size as f64 * 1_000_000_000.0 / timing.median_ns,
                scalar_ratio,
                slower_than_scalar_pct,
                exceeds_5_percent,
                selected,
                json_string(disposition),
                json_string(fixture_id),
            )
            .unwrap();
        }
    }
    evidence.flush().unwrap();
    assert!(
        selected_regressions.is_empty(),
        "public dispatch selected backends more than 5% slower than scalar: {selected_regressions:?}"
    );
    println!("x86 TYPPRICE qualification evidence: {output_path}");
}
