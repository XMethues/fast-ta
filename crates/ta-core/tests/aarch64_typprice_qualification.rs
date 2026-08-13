#![cfg(all(target_arch = "aarch64", feature = "simd-qualification"))]

use std::{
    fmt::Write as _,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    time::Instant,
};

use fast_ta::{
    price_transform::TYPPRICE,
    simd::dispatch::{
        active_indicator_backend,
        qualification::{backend_available, with_indicator_backend},
        IndicatorBackend,
    },
    Float,
};

const LENGTHS: [usize; 3] = [256, 4_096, 65_536];
const SAMPLE_COUNT: usize = 31;
const BOOTSTRAP_RESAMPLES: usize = 2_000;
const PERFORMANCE_THRESHOLD_PERCENT: f64 = 5.0;

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

fn run_public(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    let range = TYPPRICE(
        black_box(high),
        black_box(low),
        black_box(close),
        black_box(output),
    )
    .unwrap();
    assert_eq!((range.beg_idx, range.nb_element), (0, high.len()));
    black_box(output[output.len() - 1]);
}

fn measure_public(
    backend: IndicatorBackend,
    high: &[Float],
    low: &[Float],
    close: &[Float],
    output: &mut [Float],
) -> (f64, f64, f64) {
    with_indicator_backend(backend, || {
        assert_eq!(
            active_indicator_backend(),
            backend,
            "qualification override did not reach the public dispatch boundary"
        );
        for _ in 0..10 {
            run_public(high, low, close, output);
        }
        let iterations = iterations(high.len());
        let mut samples = Vec::with_capacity(SAMPLE_COUNT);
        for _ in 0..SAMPLE_COUNT {
            let started = Instant::now();
            for _ in 0..iterations {
                run_public(high, low, close, output);
            }
            samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
        }
        let (lower, upper) = confidence_interval(&samples);
        (median(&mut samples), lower, upper)
    })
}

fn performance_disposition(delta_percent: f64) -> &'static str {
    if delta_percent > PERFORMANCE_THRESHOLD_PERCENT {
        "benefit_over_5_percent"
    } else if delta_percent < -PERFORMANCE_THRESHOLD_PERCENT {
        "regression_over_5_percent"
    } else {
        "within_5_percent"
    }
}

fn public_errors(backend: IndicatorBackend) -> (fast_ta::TalibError, fast_ta::TalibError) {
    with_indicator_backend(backend, || {
        assert_eq!(active_indicator_backend(), backend);
        let mismatched =
            TYPPRICE(&[1.0 as Float], &[], &[1.0 as Float], &mut [0.0 as Float]).unwrap_err();
        let non_finite = TYPPRICE(
            &[1.0 as Float],
            &[Float::NAN],
            &[1.0 as Float],
            &mut [0.0 as Float],
        )
        .unwrap_err();
        (mismatched, non_finite)
    })
}

#[cfg(not(feature = "f32"))]
mod c_reference {
    use super::{confidence_interval, iterations, median, Float, SAMPLE_COUNT};
    use std::{
        ffi::{c_char, c_int, c_void, CStr, CString},
        hint::black_box,
        time::Instant,
    };

    const RTLD_NOW: c_int = 0x2;
    const TA_SUCCESS: c_int = 0;

    type InitializeFn = unsafe extern "C" fn() -> c_int;
    type ShutdownFn = unsafe extern "C" fn() -> c_int;
    type VersionFn = unsafe extern "C" fn() -> *const c_char;
    type TyppriceFn = unsafe extern "C" fn(
        c_int,
        c_int,
        *const f64,
        *const f64,
        *const f64,
        *mut c_int,
        *mut c_int,
        *mut f64,
    ) -> c_int;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    pub struct Library {
        handle: *mut c_void,
        typprice: TyppriceFn,
        shutdown: ShutdownFn,
    }

    impl Library {
        pub fn from_environment() -> Option<Self> {
            let path = std::env::var("QUALIFICATION_TALIB_LIBRARY").ok()?;
            let path = CString::new(path).expect("TA-Lib path contains a NUL byte");
            let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
            assert!(
                !handle.is_null(),
                "{}",
                dynamic_error("unable to load pinned TA-Lib")
            );
            let initialize = unsafe { symbol::<InitializeFn>(handle, b"TA_Initialize\0") };
            let shutdown = unsafe { symbol::<ShutdownFn>(handle, b"TA_Shutdown\0") };
            let typprice = unsafe { symbol::<TyppriceFn>(handle, b"TA_TYPPRICE\0") };
            let version = unsafe { symbol::<VersionFn>(handle, b"TA_GetVersionString\0") };
            let version = unsafe { CStr::from_ptr(version()) }
                .to_str()
                .expect("TA_GetVersionString returned invalid UTF-8");
            assert_eq!(
                version.split_whitespace().next(),
                Some("0.6.4"),
                "dynamically loaded TA-Lib does not match the pinned version"
            );
            assert_eq!(unsafe { initialize() }, TA_SUCCESS, "TA_Initialize failed");
            Some(Self {
                handle,
                typprice,
                shutdown,
            })
        }

        pub fn execute(
            &self,
            high: &[Float],
            low: &[Float],
            close: &[Float],
            output: &mut [Float],
        ) {
            let mut output_begin = -1;
            let mut output_count = -1;
            let result = unsafe {
                (self.typprice)(
                    0,
                    i32::try_from(high.len()).unwrap() - 1,
                    high.as_ptr(),
                    low.as_ptr(),
                    close.as_ptr(),
                    &mut output_begin,
                    &mut output_count,
                    output.as_mut_ptr(),
                )
            };
            assert_eq!(result, TA_SUCCESS, "TA_TYPPRICE failed");
            assert_eq!((output_begin, output_count), (0, high.len() as i32));
            black_box(output[output.len() - 1]);
        }

        pub fn measure(
            &self,
            high: &[Float],
            low: &[Float],
            close: &[Float],
            output: &mut [Float],
        ) -> (f64, f64, f64) {
            for _ in 0..10 {
                self.execute(high, low, close, output);
            }
            let iterations = iterations(high.len());
            let mut samples = Vec::with_capacity(SAMPLE_COUNT);
            for _ in 0..SAMPLE_COUNT {
                let started = Instant::now();
                for _ in 0..iterations {
                    self.execute(high, low, close, output);
                }
                samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
            }
            let (lower, upper) = confidence_interval(&samples);
            (median(&mut samples), lower, upper)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            assert_eq!(
                unsafe { (self.shutdown)() },
                TA_SUCCESS,
                "TA_Shutdown failed"
            );
            assert_eq!(unsafe { dlclose(self.handle) }, 0, "dlclose failed");
        }
    }

    unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &'static [u8]) -> T {
        let symbol = unsafe { dlsym(handle, name.as_ptr().cast()) };
        assert!(
            !symbol.is_null(),
            "{}",
            dynamic_error("missing TA-Lib symbol")
        );
        unsafe { std::mem::transmute_copy(&symbol) }
    }

    fn dynamic_error(prefix: &str) -> String {
        let error = unsafe { dlerror() };
        if error.is_null() {
            prefix.to_owned()
        } else {
            format!(
                "{prefix}: {}",
                unsafe { CStr::from_ptr(error) }.to_string_lossy()
            )
        }
    }
}

#[test]
#[ignore = "platform qualification is run explicitly by the qualification workflow"]
fn qualify_public_typprice_on_aarch64() {
    assert!(
        backend_available(IndicatorBackend::Neon),
        "AArch64 runner did not report an executable NEON qualification backend"
    );
    let active = active_indicator_backend();
    assert_eq!(
        active,
        IndicatorBackend::Neon,
        "production public dispatch did not select NEON; refusing to claim NEON evidence"
    );

    let scalar_errors = public_errors(IndicatorBackend::Scalar);
    let neon_errors = public_errors(IndicatorBackend::Neon);
    assert_eq!(
        neon_errors, scalar_errors,
        "NEON and scalar public boundaries returned different errors"
    );

    // Establish every exact public-boundary equivalence claim before any
    // validation record is emitted. A failed comparison therefore cannot
    // leave behind affirmative semantic evidence.
    for size in LENGTHS {
        let (high, low, close) = fixture(size);
        let mut scalar = vec![0.0 as Float; size];
        with_indicator_backend(IndicatorBackend::Scalar, || {
            assert_eq!(active_indicator_backend(), IndicatorBackend::Scalar);
            run_public(&high, &low, &close, &mut scalar);
        });
        let mut neon = vec![0.0 as Float; size];
        with_indicator_backend(IndicatorBackend::Neon, || {
            assert_eq!(active_indicator_backend(), IndicatorBackend::Neon);
            run_public(&high, &low, &close, &mut neon);
        });
        assert_eq!(
            neon, scalar,
            "public NEON differs from public scalar at length {size}"
        );
    }

    let output_path = std::env::var("QUALIFICATION_OUTPUT")
        .unwrap_or_else(|_| "target/qualification/aarch64_typprice.jsonl".to_owned());
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
    let workflow_run_id = std::env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "unknown".to_owned());
    let workflow_run_url = match (
        std::env::var("GITHUB_SERVER_URL"),
        std::env::var("GITHUB_REPOSITORY"),
        workflow_run_id.as_str(),
    ) {
        (Ok(server), Ok(repository), run_id) if run_id != "unknown" => {
            format!("{server}/{repository}/actions/runs/{run_id}")
        }
        _ => "unknown".to_owned(),
    };
    #[cfg(not(feature = "f32"))]
    let c_library = c_reference::Library::from_environment();
    #[cfg(feature = "f32")]
    let c_reference_available = false;
    #[cfg(not(feature = "f32"))]
    let c_reference_available = c_library.is_some();

    writeln!(
        evidence,
        "{{\"record\":\"metadata\",\"indicator\":\"TYPPRICE\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"parameters\":\"none\",\"fixture\":{},\"platform\":\"aarch64\",\"precision\":\"{}\",\"runtime\":{},\"os\":{},\"cpu\":{},\"cpu_features\":{},\"profile\":\"release\",\"features\":{},\"command\":{},\"commit\":{},\"source_repository\":{},\"source_revision\":{},\"workflow_name\":{},\"workflow_ref\":{},\"workflow_run_id\":{},\"workflow_run_url\":{},\"workflow_job\":{},\"workflow_attempt\":{},\"active_backend\":\"{}\",\"neon_available\":true,\"c_reference_available\":{},\"ta_lib_version\":\"0.6.4\",\"ta_lib_revision\":\"43f9d5042ecc4bd367941846494ad907bf20ea50\",\"ta_lib_archive_sha256\":\"aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86\"}}",
        json_string(fixture_id),
        precision,
        json_string(&std::env::var("QUALIFICATION_RUNTIME").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_OS").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_CPU").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_CPU_FEATURES").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_FEATURES").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_COMMAND").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_COMMIT").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("GITHUB_REPOSITORY").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("QUALIFICATION_COMMIT").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("GITHUB_WORKFLOW").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("GITHUB_WORKFLOW_REF").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&workflow_run_id),
        json_string(&workflow_run_url),
        json_string(&std::env::var("GITHUB_JOB").unwrap_or_else(|_| "unknown".to_owned())),
        json_string(&std::env::var("GITHUB_RUN_ATTEMPT").unwrap_or_else(|_| "unknown".to_owned())),
        active.as_str(),
        c_reference_available,
    )
    .unwrap();

    for backend in [IndicatorBackend::Scalar, IndicatorBackend::Neon] {
        writeln!(
            evidence,
            "{{\"record\":\"validation\",\"indicator\":\"TYPPRICE\",\"precision\":\"{}\",\"backend\":\"{}\",\"mode\":\"public TYPPRICE\",\"public_boundary\":true,\"exact_scalar_equivalence\":true,\"error_semantics_verified\":true,\"mismatched_length_error_equal_to_scalar\":true,\"non_finite_error_equal_to_scalar\":true,\"observed_backend\":\"{}\"}}",
            precision,
            backend.as_str(),
            backend.as_str(),
        )
        .unwrap();
    }

    let mut large_input_neon_deltas = Vec::with_capacity(2);
    for size in LENGTHS {
        let (high, low, close) = fixture(size);
        let mut scalar = vec![0.0 as Float; size];
        let scalar_timing =
            measure_public(IndicatorBackend::Scalar, &high, &low, &close, &mut scalar);
        let expected_checksum = checksum(&scalar);
        writeln!(
            evidence,
            "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"public TYPPRICE\",\"backend\":\"scalar\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"exact_scalar_equivalence\":true,\"error_semantics_verified\":true,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"performance_delta_percent_vs_scalar\":0.000,\"performance_threshold_percent\":5.000,\"performance_disposition\":\"scalar_control\",\"fixture\":{},\"timed_boundary\":\"public TYPPRICE; validation included; caller-owned output\"}}",
            size,
            size,
            expected_checksum,
            SAMPLE_COUNT,
            scalar_timing.0,
            scalar_timing.1,
            scalar_timing.2,
            size as f64 * 1_000_000_000.0 / scalar_timing.0,
            json_string(fixture_id),
        )
        .unwrap();

        let mut neon = vec![0.0 as Float; size];
        let neon_timing = measure_public(IndicatorBackend::Neon, &high, &low, &close, &mut neon);
        assert_eq!(
            neon, scalar,
            "public NEON differs from public scalar at length {size}"
        );
        let neon_delta = (scalar_timing.0 - neon_timing.0) * 100.0 / scalar_timing.0;
        if size >= 4_096 {
            large_input_neon_deltas.push((size, neon_delta));
        }
        writeln!(
            evidence,
            "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"public TYPPRICE\",\"backend\":\"neon\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"exact_scalar_equivalence\":true,\"error_semantics_verified\":true,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"performance_delta_percent_vs_scalar\":{:.3},\"performance_threshold_percent\":5.000,\"performance_disposition\":\"{}\",\"fixture\":{},\"timed_boundary\":\"public TYPPRICE; validation included; caller-owned output\"}}",
            size,
            size,
            expected_checksum,
            SAMPLE_COUNT,
            neon_timing.0,
            neon_timing.1,
            neon_timing.2,
            size as f64 * 1_000_000_000.0 / neon_timing.0,
            neon_delta,
            performance_disposition(neon_delta),
            json_string(fixture_id),
        )
        .unwrap();

        #[cfg(not(feature = "f32"))]
        if let Some(library) = &c_library {
            let mut c_output = vec![0.0 as Float; size];
            library.execute(&high, &low, &close, &mut c_output);
            assert_eq!(
                c_output, scalar,
                "pinned TA-Lib C differs from scalar at length {size}"
            );
            let c_timing = library.measure(&high, &low, &close, &mut c_output);
            let c_delta = (scalar_timing.0 - c_timing.0) * 100.0 / scalar_timing.0;
            writeln!(
                evidence,
                "{{\"record\":\"measurement\",\"indicator\":\"TYPPRICE\",\"indicator_family\":\"Price Transform\",\"indicator_definition\":\"TYPPRICE: Typical Price\",\"case_id\":\"TYPPRICE\",\"mode\":\"direct C caller-owned\",\"backend\":\"ta-lib-c\",\"parameters\":\"none\",\"input_length\":{},\"output_kind\":\"float\",\"output_arity\":1,\"output_begin\":0,\"output_count\":{},\"output_checksum\":\"{}\",\"equivalent_to_scalar\":true,\"exact_scalar_equivalence\":true,\"error_semantics_verified\":false,\"semantic_status\":\"verified\",\"timing_status\":\"measured\",\"sample_count\":{},\"median_ns\":{:.3},\"ci95_lower_ns\":{:.3},\"ci95_upper_ns\":{:.3},\"throughput_observations_per_second\":{:.3},\"performance_delta_percent_vs_scalar\":{:.3},\"performance_threshold_percent\":5.000,\"performance_disposition\":\"{}\",\"fixture\":{},\"timed_boundary\":\"direct TA_TYPPRICE call; caller output allocation excluded\",\"ta_lib_version\":\"0.6.4\",\"ta_lib_revision\":\"43f9d5042ecc4bd367941846494ad907bf20ea50\"}}",
                size,
                size,
                expected_checksum,
                SAMPLE_COUNT,
                c_timing.0,
                c_timing.1,
                c_timing.2,
                size as f64 * 1_000_000_000.0 / c_timing.0,
                c_delta,
                performance_disposition(c_delta),
                json_string(fixture_id),
            )
            .unwrap();
        }
    }
    evidence.flush().unwrap();

    for (size, delta) in large_input_neon_deltas {
        assert!(
            delta > PERFORMANCE_THRESHOLD_PERCENT,
            "public NEON must beat its same-boundary scalar control by more than 5% at length {size}, observed {delta:.3}%"
        );
    }
    println!("AArch64 TYPPRICE qualification evidence: {output_path}");
}
