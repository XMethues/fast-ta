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
        qualification::{backend_available, typical_price},
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

fn measure(
    backend: IndicatorBackend,
    public_path: bool,
    high: &[Float],
    low: &[Float],
    close: &[Float],
    output: &mut [Float],
) -> (f64, f64, f64) {
    let run = |output: &mut [Float]| {
        if public_path {
            let range = TYPPRICE(
                black_box(high),
                black_box(low),
                black_box(close),
                black_box(output),
            )
            .unwrap();
            assert_eq!((range.beg_idx, range.nb_element), (0, high.len()));
        } else {
            typical_price(
                backend,
                black_box(high),
                black_box(low),
                black_box(close),
                black_box(output),
            );
        }
        black_box(output[output.len() - 1]);
    };

    for _ in 0..10 {
        run(output);
    }
    let iterations = iterations(high.len());
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let started = Instant::now();
        for _ in 0..iterations {
            run(output);
        }
        samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
    }
    let (lower, upper) = confidence_interval(&samples);
    (median(&mut samples), lower, upper)
}

#[test]
#[ignore = "platform qualification is run explicitly by the qualification workflow"]
fn qualify_public_typprice_on_x86_simd() {
    assert!(
        backend_available(IndicatorBackend::Avx2),
        "x86 qualification requires an AVX2-capable runner"
    );
    let active = active_indicator_backend();
    let expected_active = if backend_available(IndicatorBackend::Avx512) {
        IndicatorBackend::Avx512
    } else {
        IndicatorBackend::Avx2
    };
    assert_eq!(
        active, expected_active,
        "runtime dispatch selected the wrong backend"
    );

    let mismatch = TYPPRICE(&[1.0 as Float], &[], &[1.0 as Float], &mut [0.0 as Float]);
    assert!(matches!(mismatch, Err(TalibError::InvalidInput { .. })));
    let non_finite = TYPPRICE(
        &[1.0 as Float],
        &[Float::NAN],
        &[1.0 as Float],
        &mut [0.0 as Float],
    );
    assert!(matches!(non_finite, Err(TalibError::InvalidInput { .. })));

    let output_path = std::env::var("QUALIFICATION_OUTPUT")
        .unwrap_or_else(|_| "target/qualification/x86_typprice.jsonl".to_owned());
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
    writeln!(
        evidence,
        "{{\"record\":\"metadata\",\"indicator\":\"TYPPRICE\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"parameters\":\"none\",\"fixture\":{},\"platform\":\"x86_64\",\"precision\":\"{}\",\"runtime\":{},\"cpu\":{},\"commit\":{},\"active_backend\":\"{}\",\"avx2_available\":true,\"avx512_available\":{}}}",
        json_string(fixture_id),
        precision,
        json_string(&std::env::var("QUALIFICATION_RUNTIME").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_CPU").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_COMMIT").unwrap_or_else(|_| "unknown".to_owned())),
        active.as_str(),
        backend_available(IndicatorBackend::Avx512),
    )
    .unwrap();

    for size in LENGTHS {
        let (high, low, close) = fixture(size);
        let mut scalar = vec![0.0 as Float; size];
        typical_price(IndicatorBackend::Scalar, &high, &low, &close, &mut scalar);
        let expected_checksum = checksum(&scalar);

        let mut backends = vec![IndicatorBackend::Scalar, IndicatorBackend::Avx2];
        if backend_available(IndicatorBackend::Avx512) {
            backends.push(IndicatorBackend::Avx512);
        }
        for backend in backends {
            let mut output = vec![0.0 as Float; size];
            typical_price(backend, &high, &low, &close, &mut output);
            assert_eq!(
                output,
                scalar,
                "{} differs at length {size}",
                backend.as_str()
            );
            let (median_ns, lower_ns, upper_ns) =
                measure(backend, false, &high, &low, &close, &mut output);
            writeln!(
                evidence,
                "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"explicit kernel\",\"backend\":\"{}\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"fixture\":{},\"timed_boundary\":\"caller-owned kernel; validation excluded\"}}",
                backend.as_str(),
                size,
                size,
                expected_checksum,
                SAMPLE_COUNT,
                median_ns,
                lower_ns,
                upper_ns,
                size as f64 * 1_000_000_000.0 / median_ns,
                json_string(fixture_id),
            )
            .unwrap();
        }

        let mut public_output = vec![0.0 as Float; size];
        let range = TYPPRICE(&high, &low, &close, &mut public_output).unwrap();
        assert_eq!((range.beg_idx, range.nb_element), (0, size));
        assert_eq!(
            public_output, scalar,
            "public dispatch differs at length {size}"
        );
        let (median_ns, lower_ns, upper_ns) =
            measure(active, true, &high, &low, &close, &mut public_output);
        writeln!(
            evidence,
            "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"public TYPPRICE\",\"backend\":\"{}\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"error_semantics_verified\":true,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"fixture\":{},\"timed_boundary\":\"public TYPPRICE; validation included; caller-owned output\"}}",
            active.as_str(),
            size,
            size,
            expected_checksum,
            SAMPLE_COUNT,
            median_ns,
            lower_ns,
            upper_ns,
            size as f64 * 1_000_000_000.0 / median_ns,
            json_string(fixture_id),
        )
        .unwrap();
    }
    evidence.flush().unwrap();
    println!("x86 TYPPRICE qualification evidence: {output_path}");
}
