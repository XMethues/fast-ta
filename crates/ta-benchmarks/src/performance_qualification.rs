//! Native SIMD Performance Qualification for the public TYPPRICE Batch Computation.
//!
//! This module owns the fixed measurement policy, semantic parity checks,
//! statistics, checksums, evidence schema, and performance gates. Platform
//! adapters supply observed provenance and already-prepared external references;
//! filesystem I/O and artifact publication remain outside this seam.

use fast_ta::price_transform::TYPPRICE;
use fast_ta::simd::dispatch::{active_indicator_backend, IndicatorBackend};
use fast_ta::Float;
use serde_json::{json, Value};
use std::fmt;
use std::hint::black_box;
use std::time::Instant;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use fast_ta::simd::dispatch::qualification::{backend_available, with_indicator_backend};

const INPUT_LENGTHS: [usize; 3] = [256, 4_096, 65_536];
const SAMPLE_COUNT: usize = 31;
const BOOTSTRAP_RESAMPLES: usize = 2_000;
const PERFORMANCE_THRESHOLD_PERCENT: f64 = 5.0;
const TALIB_VERSION: &str = "0.6.4";
const TALIB_REVISION: &str = "43f9d5042ecc4bd367941846494ad907bf20ea50";
const TALIB_ARCHIVE_SHA256: &str =
    "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86";

/// Explicit process and workflow facts supplied by the native adapter.
#[derive(Clone, Debug)]
pub struct Provenance {
    pub runtime: String,
    pub os: String,
    pub architecture: String,
    pub cpu: String,
    pub cpu_features: String,
    pub rust_profile: String,
    pub cargo_features: String,
    pub target_features: String,
    pub commit: String,
    pub qualification_command: String,
    pub source_repository: String,
    pub workflow_name: String,
    pub workflow_ref: String,
    pub workflow_run_id: String,
    pub workflow_run_url: String,
    pub workflow_job: String,
    pub workflow_attempt: String,
}

/// The real native platform adapters supported by this qualification.
#[derive(Clone, Debug)]
pub enum NativePlatform {
    #[cfg(target_arch = "x86_64")]
    X86,
    #[cfg(target_arch = "aarch64")]
    Aarch64 {
        /// Required for canonical f64 qualification and absent for f32.
        talib_library: Option<std::path::PathBuf>,
    },
}

/// One complete native TYPPRICE qualification request.
#[derive(Clone, Debug)]
pub struct QualificationRequest {
    pub platform: NativePlatform,
    pub provenance: Provenance,
}

/// A selected production backend that failed its platform performance gate.
#[derive(Clone, Debug, PartialEq)]
pub struct Regression {
    pub backend: &'static str,
    pub input_length: usize,
    pub delta_percent: f64,
    pub requirement: &'static str,
}

/// Complete validated evidence plus a separately enforceable performance verdict.
#[derive(Clone, Debug)]
pub struct QualificationOutcome {
    jsonl: String,
    regressions: Vec<Regression>,
}

impl QualificationOutcome {
    /// Returns the validated JSONL payload. Persist it before enforcing the gate.
    pub fn jsonl(&self) -> &str {
        &self.jsonl
    }

    /// Returns observed production-selection regressions.
    pub fn regressions(&self) -> &[Regression] {
        &self.regressions
    }

    /// Enforces the performance gate after evidence has been persisted.
    pub fn require_pass(&self) -> Result<(), RegressionGateError> {
        if self.regressions.is_empty() {
            Ok(())
        } else {
            Err(RegressionGateError(self.regressions.clone()))
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegressionGateError(Vec<Regression>);

impl fmt::Display for RegressionGateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "native TYPPRICE performance gate failed: {:?}",
            self.0
        )
    }
}

impl std::error::Error for RegressionGateError {}

#[derive(Debug)]
pub struct QualificationError(String);

impl QualificationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for QualificationError {}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    median_ns: f64,
    lower_ns: f64,
    upper_ns: f64,
}

#[derive(Debug, PartialEq, Eq)]
struct ValidationEvidence {
    unequal_lengths_error: String,
    non_finite_error: String,
    short_output_error: String,
}

/// Runs the fixed native TYPPRICE Performance Qualification.
///
/// Semantic or evidence failures return [`QualificationError`] and no affirmative
/// evidence. A performance failure returns a complete [`QualificationOutcome`]
/// whose verdict is enforced separately with [`QualificationOutcome::require_pass`].
pub fn qualify_typprice(
    request: QualificationRequest,
) -> Result<QualificationOutcome, QualificationError> {
    validate_provenance(&request.provenance)?;
    match request.platform {
        #[cfg(target_arch = "x86_64")]
        NativePlatform::X86 => qualify_x86(request.provenance),
        #[cfg(target_arch = "aarch64")]
        NativePlatform::Aarch64 { talib_library } => {
            qualify_aarch64(request.provenance, talib_library)
        }
    }
}

fn validate_provenance(provenance: &Provenance) -> Result<(), QualificationError> {
    for (name, value) in [
        ("runtime", provenance.runtime.as_str()),
        ("os", provenance.os.as_str()),
        ("architecture", provenance.architecture.as_str()),
        ("cpu", provenance.cpu.as_str()),
        ("rust_profile", provenance.rust_profile.as_str()),
        ("cargo_features", provenance.cargo_features.as_str()),
        ("commit", provenance.commit.as_str()),
        (
            "qualification_command",
            provenance.qualification_command.as_str(),
        ),
        ("workflow_run_id", provenance.workflow_run_id.as_str()),
        ("workflow_run_url", provenance.workflow_run_url.as_str()),
        ("workflow_job", provenance.workflow_job.as_str()),
    ] {
        if value.is_empty() {
            return Err(QualificationError::new(format!(
                "qualification provenance field {name:?} is empty"
            )));
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
fn qualify_x86(provenance: Provenance) -> Result<QualificationOutcome, QualificationError> {
    if provenance.architecture != "x86_64" {
        return Err(QualificationError::new(format!(
            "x86 qualification received architecture {:?}",
            provenance.architecture
        )));
    }
    if !backend_available(IndicatorBackend::Avx2) {
        return Err(QualificationError::new(
            "x86 qualification requires an AVX2-capable host",
        ));
    }
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
    if active != expected_active {
        return Err(QualificationError::new(format!(
            "x86 production dispatch selected {}, expected {}",
            active.as_str(),
            expected_active.as_str()
        )));
    }

    let backends: Vec<_> = [
        IndicatorBackend::Scalar,
        IndicatorBackend::Avx2,
        IndicatorBackend::Avx512,
    ]
    .into_iter()
    .filter(|backend| backend_available(*backend))
    .collect();
    let scalar_validation = validate_public_boundary(IndicatorBackend::Scalar)?;
    let mut records = vec![x86_metadata(&provenance, active)];
    for backend in &backends {
        let validation = validate_public_boundary(*backend)?;
        if validation != scalar_validation {
            return Err(QualificationError::new(format!(
                "{} public error semantics differ from scalar",
                backend.as_str()
            )));
        }
        records.push(json!({
            "record": "validation",
            "indicator": "TYPPRICE",
            "backend": backend.as_str(),
            "public_boundary": true,
            "unequal_lengths_verified": true,
            "non_finite_verified": true,
            "short_output_verified": true,
            "errors_match_scalar": true,
            "unequal_lengths_error": validation.unequal_lengths_error,
            "non_finite_error": validation.non_finite_error,
            "short_output_error": validation.short_output_error,
        }));
    }

    let mut regressions = Vec::new();
    for size in INPUT_LENGTHS {
        let (high, low, close) = fixture(size);
        let scalar = compute(IndicatorBackend::Scalar, &high, &low, &close)?;
        let checksum = checksum(&scalar);
        let mut measurements = Vec::with_capacity(backends.len());
        for backend in &backends {
            let output = compute(*backend, &high, &low, &close)?;
            if output != scalar {
                return Err(QualificationError::new(format!(
                    "{} differs from scalar at input length {size}",
                    backend.as_str()
                )));
            }
            let timing = measure_fast_ta(*backend, &high, &low, &close)?;
            measurements.push((*backend, timing));
        }
        let scalar_median = measurements
            .iter()
            .find(|(backend, _)| *backend == IndicatorBackend::Scalar)
            .expect("scalar backend is always present")
            .1
            .median_ns;
        for (backend, timing) in measurements {
            let ratio = timing.median_ns / scalar_median;
            let slower_percent = (ratio - 1.0) * 100.0;
            let exceeds = ratio > 1.05;
            let selected = backend == active;
            let disposition = if backend == IndicatorBackend::Scalar {
                "scalar control"
            } else if exceeds && selected {
                regressions.push(Regression {
                    backend: backend.as_str(),
                    input_length: size,
                    delta_percent: slower_percent,
                    requirement: "selected backend must be no more than 5% slower than scalar",
                });
                "invalid selection: accelerated backend exceeds the 5% scalar gate"
            } else if exceeds {
                "not selected: accelerated backend exceeds the 5% scalar gate"
            } else if selected {
                "selected: accelerated backend is within the 5% scalar gate"
            } else {
                "qualified but not selected: a higher-priority backend is active"
            };
            records.push(measurement_record(
                backend.as_str(),
                "public TYPPRICE",
                size,
                &checksum,
                timing,
                json!({
                    "scalar_ratio": ratio,
                    "slower_than_scalar_pct": slower_percent,
                    "exceeds_5_percent": exceeds,
                    "selected": selected,
                    "disposition": disposition,
                }),
            ));
        }
    }
    finish(records, regressions, "x86 native TYPPRICE qualification")
}

#[cfg(target_arch = "x86_64")]
fn x86_metadata(provenance: &Provenance, active: IndicatorBackend) -> Value {
    json!({
        "record": "metadata",
        "indicator": "TYPPRICE",
        "indicator_definition": "TYPPRICE: Typical Price",
        "parameters": "none",
        "fixture": fixture_id(),
        "platform": "x86_64",
        "os": provenance.os,
        "architecture": provenance.architecture,
        "precision": precision(),
        "runtime": provenance.runtime,
        "rust_profile": provenance.rust_profile,
        "profile": provenance.rust_profile,
        "cargo_features": provenance.cargo_features,
        "features": provenance.cargo_features,
        "target_features": provenance.target_features,
        "cpu": provenance.cpu,
        "commit": provenance.commit,
        "qualification_command": provenance.qualification_command,
        "workflow_run_id": provenance.workflow_run_id,
        "workflow_run_url": provenance.workflow_run_url,
        "workflow_job": provenance.workflow_job,
        "active_backend": active.as_str(),
        "avx2_available": true,
        "avx512_available": backend_available(IndicatorBackend::Avx512),
    })
}

#[cfg(target_arch = "aarch64")]
fn qualify_aarch64(
    provenance: Provenance,
    talib_library: Option<std::path::PathBuf>,
) -> Result<QualificationOutcome, QualificationError> {
    if provenance.architecture != "arm64" && provenance.architecture != "aarch64" {
        return Err(QualificationError::new(format!(
            "AArch64 qualification received architecture {:?}",
            provenance.architecture
        )));
    }
    if !backend_available(IndicatorBackend::Neon) {
        return Err(QualificationError::new(
            "AArch64 qualification requires an executable NEON backend",
        ));
    }
    let active = active_indicator_backend();
    if active != IndicatorBackend::Neon {
        return Err(QualificationError::new(format!(
            "AArch64 production dispatch selected {}, expected neon",
            active.as_str()
        )));
    }

    #[cfg(not(feature = "f32"))]
    let c_library = TalibLibrary::load(talib_library.ok_or_else(|| {
        QualificationError::new(
            "canonical AArch64 f64 qualification requires QUALIFICATION_TALIB_LIBRARY",
        )
    })?)?;
    #[cfg(feature = "f32")]
    if talib_library.is_some() {
        return Err(QualificationError::new(
            "AArch64 f32 qualification must not receive an f64 TA-Lib control",
        ));
    }

    let scalar_validation = validate_public_boundary(IndicatorBackend::Scalar)?;
    let neon_validation = validate_public_boundary(IndicatorBackend::Neon)?;
    if neon_validation != scalar_validation {
        return Err(QualificationError::new(
            "NEON public error semantics differ from scalar",
        ));
    }

    let mut records = vec![aarch64_metadata(&provenance, active)];
    for backend in [IndicatorBackend::Scalar, IndicatorBackend::Neon] {
        records.push(json!({
            "record": "validation",
            "indicator": "TYPPRICE",
            "precision": precision(),
            "backend": backend.as_str(),
            "mode": "public TYPPRICE",
            "public_boundary": true,
            "exact_scalar_equivalence": true,
            "error_semantics_verified": true,
            "mismatched_length_error_equal_to_scalar": true,
            "non_finite_error_equal_to_scalar": true,
            "observed_backend": backend.as_str(),
        }));
    }

    let mut regressions = Vec::new();
    for size in INPUT_LENGTHS {
        let (high, low, close) = fixture(size);
        let scalar = compute(IndicatorBackend::Scalar, &high, &low, &close)?;
        let neon = compute(IndicatorBackend::Neon, &high, &low, &close)?;
        if neon != scalar {
            return Err(QualificationError::new(format!(
                "NEON differs from scalar at input length {size}"
            )));
        }
        let checksum = checksum(&scalar);
        let scalar_timing = measure_fast_ta(IndicatorBackend::Scalar, &high, &low, &close)?;
        let neon_timing = measure_fast_ta(IndicatorBackend::Neon, &high, &low, &close)?;
        records.push(measurement_record(
            "scalar",
            "public TYPPRICE",
            size,
            &checksum,
            scalar_timing,
            json!({
                "performance_delta_percent_vs_scalar": 0.0,
                "performance_threshold_percent": PERFORMANCE_THRESHOLD_PERCENT,
                "performance_disposition": "scalar_control",
            }),
        ));
        let neon_delta =
            (scalar_timing.median_ns - neon_timing.median_ns) * 100.0 / scalar_timing.median_ns;
        if size >= 4_096 && neon_delta <= PERFORMANCE_THRESHOLD_PERCENT {
            regressions.push(Regression {
                backend: "neon",
                input_length: size,
                delta_percent: neon_delta,
                requirement: "selected NEON backend must beat scalar by more than 5%",
            });
        }
        records.push(measurement_record(
            "neon",
            "public TYPPRICE",
            size,
            &checksum,
            neon_timing,
            json!({
                "performance_delta_percent_vs_scalar": neon_delta,
                "performance_threshold_percent": PERFORMANCE_THRESHOLD_PERCENT,
                "performance_disposition": performance_disposition(neon_delta),
            }),
        ));

        #[cfg(not(feature = "f32"))]
        {
            let c_output = c_library.compute(&high, &low, &close)?;
            if c_output != scalar {
                return Err(QualificationError::new(format!(
                    "pinned TA-Lib C differs from scalar at input length {size}"
                )));
            }
            let c_timing = c_library.measure(&high, &low, &close)?;
            let c_delta =
                (scalar_timing.median_ns - c_timing.median_ns) * 100.0 / scalar_timing.median_ns;
            records.push(measurement_record(
                "ta-lib-c",
                "direct C caller-owned",
                size,
                &checksum,
                c_timing,
                json!({
                    "error_semantics_verified": false,
                    "performance_delta_percent_vs_scalar": c_delta,
                    "performance_threshold_percent": PERFORMANCE_THRESHOLD_PERCENT,
                    "performance_disposition": performance_disposition(c_delta),
                }),
            ));
        }
    }
    finish(
        records,
        regressions,
        "AArch64 native TYPPRICE qualification",
    )
}

#[cfg(target_arch = "aarch64")]
fn aarch64_metadata(provenance: &Provenance, active: IndicatorBackend) -> Value {
    json!({
        "record": "metadata",
        "indicator": "TYPPRICE",
        "indicator_definition": "TYPPRICE: Typical Price",
        "parameters": "none",
        "fixture": fixture_id(),
        "platform": "aarch64",
        "precision": precision(),
        "runtime": provenance.runtime,
        "os": provenance.os,
        "cpu": provenance.cpu,
        "cpu_features": provenance.cpu_features,
        "profile": provenance.rust_profile,
        "features": provenance.cargo_features,
        "command": provenance.qualification_command,
        "commit": provenance.commit,
        "source_repository": provenance.source_repository,
        "source_revision": provenance.commit,
        "workflow_name": provenance.workflow_name,
        "workflow_ref": provenance.workflow_ref,
        "workflow_run_id": provenance.workflow_run_id,
        "workflow_run_url": provenance.workflow_run_url,
        "workflow_job": provenance.workflow_job,
        "workflow_attempt": provenance.workflow_attempt,
        "active_backend": active.as_str(),
        "neon_available": true,
        "c_reference_available": !cfg!(feature = "f32"),
        "ta_lib_version": TALIB_VERSION,
        "ta_lib_revision": TALIB_REVISION,
        "ta_lib_archive_sha256": TALIB_ARCHIVE_SHA256,
    })
}

fn finish(
    records: Vec<Value>,
    regressions: Vec<Regression>,
    artifact: &str,
) -> Result<QualificationOutcome, QualificationError> {
    validate_records(&records, artifact)?;
    let mut jsonl = String::new();
    for record in records {
        jsonl.push_str(
            &serde_json::to_string(&record)
                .map_err(|error| QualificationError::new(format!("encode evidence: {error}")))?,
        );
        jsonl.push('\n');
    }
    Ok(QualificationOutcome { jsonl, regressions })
}

fn validate_records(records: &[Value], artifact: &str) -> Result<(), QualificationError> {
    let metadata_count = records
        .iter()
        .filter(|record| record.get("record").and_then(Value::as_str) == Some("metadata"))
        .count();
    let validation_count = records
        .iter()
        .filter(|record| record.get("record").and_then(Value::as_str) == Some("validation"))
        .count();
    let measurements: Vec<_> = records
        .iter()
        .filter(|record| record.get("record").and_then(Value::as_str) == Some("measurement"))
        .collect();
    if metadata_count != 1 || validation_count == 0 || measurements.is_empty() {
        return Err(QualificationError::new(format!(
            "{artifact} evidence must contain one metadata record, validation, and measurements"
        )));
    }
    for measurement in measurements {
        let input_length = measurement
            .get("input_length")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| INPUT_LENGTHS.contains(value))
            .ok_or_else(|| QualificationError::new("measurement has invalid input_length"))?;
        let sample_count = measurement
            .get("sample_count")
            .and_then(Value::as_u64)
            .ok_or_else(|| QualificationError::new("measurement has invalid sample_count"))?;
        if sample_count != SAMPLE_COUNT as u64 {
            return Err(QualificationError::new(format!(
                "measurement at {input_length} has sample_count {sample_count}"
            )));
        }
        let median = measurement
            .get("median_ns")
            .and_then(Value::as_f64)
            .ok_or_else(|| QualificationError::new("measurement has invalid median_ns"))?;
        let lower = measurement
            .get("ci95_lower_ns")
            .and_then(Value::as_f64)
            .ok_or_else(|| QualificationError::new("measurement has invalid ci95_lower_ns"))?;
        let upper = measurement
            .get("ci95_upper_ns")
            .and_then(Value::as_f64)
            .ok_or_else(|| QualificationError::new("measurement has invalid ci95_upper_ns"))?;
        if !median.is_finite() || median <= 0.0 || lower > median || median > upper {
            return Err(QualificationError::new(format!(
                "measurement at {input_length} has incoherent timing evidence"
            )));
        }
    }
    Ok(())
}

fn measurement_record(
    backend: &str,
    mode: &str,
    input_length: usize,
    output_checksum: &str,
    timing: Measurement,
    extra: Value,
) -> Value {
    let mut record = json!({
        "record": "measurement",
        "indicator": "TYPPRICE",
        "indicator_family": "Price Transform",
        "indicator_definition": "TYPPRICE: Typical Price",
        "case_id": "TYPPRICE",
        "mode": mode,
        "backend": backend,
        "parameters": "none",
        "input_length": input_length,
        "output_kind": "float",
        "output_arity": 1,
        "output_begin": 0,
        "output_count": input_length,
        "output_checksum": output_checksum,
        "equivalent_to_scalar": true,
        "exact_scalar_equivalence": true,
        "error_semantics_verified": true,
        "same_public_boundary": true,
        "semantic_status": "verified",
        "timing_status": "measured",
        "sample_count": SAMPLE_COUNT,
        "median_ns": timing.median_ns,
        "ci95_lower_ns": timing.lower_ns,
        "ci95_upper_ns": timing.upper_ns,
        "throughput_observations_per_second": input_length as f64 * 1_000_000_000.0 / timing.median_ns,
        "fixture": fixture_id(),
        "timed_boundary": if backend == "ta-lib-c" { "direct TA-Lib C caller-owned" } else { "public TYPPRICE validation, dispatch, and caller-owned output" },
    });
    if let (Some(target), Some(source)) = (record.as_object_mut(), extra.as_object()) {
        target.extend(source.clone());
    }
    record
}

fn validate_public_boundary(
    backend: IndicatorBackend,
) -> Result<ValidationEvidence, QualificationError> {
    with_indicator_backend(backend, || {
        let mut mismatch_output = [91.0 as Float; 2];
        let unequal_lengths = TYPPRICE(
            &[2.0 as Float, 3.0 as Float],
            &[1.0 as Float],
            &[1.5 as Float, 2.5 as Float],
            &mut mismatch_output,
        )
        .map_err(|error| error.to_string())
        .expect_err("unequal input lengths must fail");
        if mismatch_output != [91.0 as Float; 2] {
            return Err(QualificationError::new(
                "unequal-length failure mutated output",
            ));
        }

        let mut non_finite_output = [92.0 as Float; 1];
        let non_finite = TYPPRICE(
            &[2.0 as Float],
            &[Float::NAN],
            &[1.5 as Float],
            &mut non_finite_output,
        )
        .map_err(|error| error.to_string())
        .expect_err("non-finite input must fail");
        if non_finite_output != [92.0 as Float; 1] {
            return Err(QualificationError::new("non-finite failure mutated output"));
        }

        let mut short_output = [93.0 as Float; 1];
        let short = TYPPRICE(
            &[2.0 as Float, 3.0 as Float],
            &[1.0 as Float, 2.0 as Float],
            &[1.5 as Float, 2.5 as Float],
            &mut short_output,
        )
        .map_err(|error| error.to_string())
        .expect_err("short output must fail");
        if short_output != [93.0 as Float; 1] {
            return Err(QualificationError::new(
                "short-output failure mutated output",
            ));
        }

        Ok(ValidationEvidence {
            unequal_lengths_error: unequal_lengths,
            non_finite_error: non_finite,
            short_output_error: short,
        })
    })
}

fn compute(
    backend: IndicatorBackend,
    high: &[Float],
    low: &[Float],
    close: &[Float],
) -> Result<Vec<Float>, QualificationError> {
    with_indicator_backend(backend, || {
        if active_indicator_backend() != backend {
            return Err(QualificationError::new(format!(
                "requested {}, observed {}",
                backend.as_str(),
                active_indicator_backend().as_str()
            )));
        }
        let mut output = vec![0.0 as Float; high.len()];
        let range = TYPPRICE(high, low, close, &mut output)
            .map_err(|error| QualificationError::new(error.to_string()))?;
        if (range.beg_idx, range.nb_element) != (0, high.len()) {
            return Err(QualificationError::new(format!(
                "{} returned range {:?} for input length {}",
                backend.as_str(),
                range,
                high.len()
            )));
        }
        Ok(output)
    })
}

fn measure_fast_ta(
    backend: IndicatorBackend,
    high: &[Float],
    low: &[Float],
    close: &[Float],
) -> Result<Measurement, QualificationError> {
    with_indicator_backend(backend, || {
        let mut output = vec![0.0 as Float; high.len()];
        let mut run = || -> Result<(), QualificationError> {
            let range = TYPPRICE(
                black_box(high),
                black_box(low),
                black_box(close),
                black_box(&mut output),
            )
            .map_err(|error| QualificationError::new(error.to_string()))?;
            if (range.beg_idx, range.nb_element) != (0, high.len()) {
                return Err(QualificationError::new(
                    "timed TYPPRICE returned wrong range",
                ));
            }
            black_box(output[output.len() - 1]);
            Ok(())
        };
        measure(&mut run, high.len())
    })
}

fn measure(
    operation: &mut impl FnMut() -> Result<(), QualificationError>,
    input_length: usize,
) -> Result<Measurement, QualificationError> {
    for _ in 0..10 {
        operation()?;
    }
    let iterations = iterations(input_length);
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let started = Instant::now();
        for _ in 0..iterations {
            operation()?;
        }
        samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || *sample <= 0.0)
    {
        return Err(QualificationError::new(
            "qualification produced non-positive or non-finite timing",
        ));
    }
    let (lower_ns, upper_ns) = confidence_interval(&samples);
    let median_ns = median(&mut samples);
    if lower_ns > median_ns || median_ns > upper_ns {
        return Err(QualificationError::new(
            "qualification confidence interval does not contain median",
        ));
    }
    Ok(Measurement {
        median_ns,
        lower_ns,
        upper_ns,
    })
}

fn fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    let mut high = Vec::with_capacity(size);
    let mut low = Vec::with_capacity(size);
    let mut close = Vec::with_capacity(size);
    for index in 0..size {
        let base = 100.0 as Float + index as Float * 0.015625 as Float;
        let wave = ((index % 29) as Float - 14.0 as Float) * 0.03125 as Float;
        let lo = base + wave;
        low.push(lo);
        high.push(lo + 1.0 as Float + (index % 5) as Float * 0.0625 as Float);
        close.push(lo + 0.375 as Float + (index % 7) as Float * 0.03125 as Float);
    }
    (high, low, close)
}

fn iterations(size: usize) -> usize {
    match size {
        0..=256 => 2_000,
        257..=4_096 => 300,
        _ => 20,
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn confidence_interval(samples: &[f64]) -> (f64, f64) {
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let mut medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut resample = Vec::with_capacity(samples.len());
    for _ in 0..BOOTSTRAP_RESAMPLES {
        resample.clear();
        for _ in 0..samples.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            resample.push(samples[state as usize % samples.len()]);
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    let lower = ((BOOTSTRAP_RESAMPLES as f64 * 0.025) as usize).min(BOOTSTRAP_RESAMPLES - 1);
    let upper = ((BOOTSTRAP_RESAMPLES as f64 * 0.975) as usize).min(BOOTSTRAP_RESAMPLES - 1);
    (medians[lower], medians[upper])
}

fn checksum(values: &[Float]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("fnv1a64:{hash:016x}")
}

fn precision() -> &'static str {
    if cfg!(feature = "f32") {
        "f32"
    } else {
        "f64"
    }
}

fn fixture_id() -> &'static str {
    if cfg!(feature = "f32") {
        "catalogue_fixture_v1:f32le"
    } else {
        "catalogue_fixture_v1:f64le"
    }
}

#[cfg(target_arch = "aarch64")]
fn performance_disposition(delta_percent: f64) -> &'static str {
    if delta_percent > PERFORMANCE_THRESHOLD_PERCENT {
        "benefit_over_5_percent"
    } else if delta_percent < -PERFORMANCE_THRESHOLD_PERCENT {
        "regression_over_5_percent"
    } else {
        "within_5_percent"
    }
}

#[cfg(all(target_arch = "aarch64", not(feature = "f32")))]
struct TalibLibrary {
    handle: *mut std::ffi::c_void,
    typprice: unsafe extern "C" fn(
        std::ffi::c_int,
        std::ffi::c_int,
        *const f64,
        *const f64,
        *const f64,
        *mut std::ffi::c_int,
        *mut std::ffi::c_int,
        *mut f64,
    ) -> std::ffi::c_int,
    shutdown: unsafe extern "C" fn() -> std::ffi::c_int,
}

#[cfg(all(target_arch = "aarch64", not(feature = "f32")))]
impl TalibLibrary {
    fn load(path: std::path::PathBuf) -> Result<Self, QualificationError> {
        use std::ffi::{c_char, c_int, c_void, CStr, CString};
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
        }
        unsafe fn symbol<T: Copy>(handle: *mut c_void, name: &'static [u8]) -> T {
            let pointer = unsafe { dlsym(handle, name.as_ptr().cast()) };
            assert!(!pointer.is_null(), "missing TA-Lib symbol");
            unsafe { std::mem::transmute_copy(&pointer) }
        }

        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| QualificationError::new("TA-Lib path contains NUL"))?;
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(QualificationError::new("unable to load pinned TA-Lib"));
        }
        let initialize = unsafe { symbol::<InitializeFn>(handle, b"TA_Initialize\0") };
        let shutdown = unsafe { symbol::<ShutdownFn>(handle, b"TA_Shutdown\0") };
        let typprice = unsafe { symbol::<TyppriceFn>(handle, b"TA_TYPPRICE\0") };
        let version = unsafe { symbol::<VersionFn>(handle, b"TA_GetVersionString\0") };
        let version = unsafe { CStr::from_ptr(version()) }
            .to_str()
            .map_err(|_| QualificationError::new("TA-Lib version is not UTF-8"))?;
        if version.split_whitespace().next() != Some(TALIB_VERSION) {
            return Err(QualificationError::new(format!(
                "TA-Lib version mismatch: {version}"
            )));
        }
        if unsafe { initialize() } != TA_SUCCESS {
            return Err(QualificationError::new("TA_Initialize failed"));
        }
        Ok(Self {
            handle,
            typprice,
            shutdown,
        })
    }

    fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>, QualificationError> {
        let mut output = vec![0.0; high.len()];
        self.execute(high, low, close, &mut output)?;
        Ok(output)
    }

    fn execute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        output: &mut [Float],
    ) -> Result<(), QualificationError> {
        let mut begin = -1;
        let mut count = -1;
        let result = unsafe {
            (self.typprice)(
                0,
                i32::try_from(high.len())
                    .map_err(|_| QualificationError::new("TA-Lib input too long"))?
                    - 1,
                high.as_ptr(),
                low.as_ptr(),
                close.as_ptr(),
                &mut begin,
                &mut count,
                output.as_mut_ptr(),
            )
        };
        if result != 0 || (begin, count) != (0, high.len() as i32) {
            return Err(QualificationError::new("TA_TYPPRICE failed"));
        }
        black_box(output[output.len() - 1]);
        Ok(())
    }

    fn measure(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Measurement, QualificationError> {
        let mut output = vec![0.0; high.len()];
        let mut run = || self.execute(high, low, close, &mut output);
        measure(&mut run, high.len())
    }
}

#[cfg(all(target_arch = "aarch64", not(feature = "f32")))]
impl Drop for TalibLibrary {
    fn drop(&mut self) {
        unsafe extern "C" {
            fn dlclose(handle: *mut std::ffi::c_void) -> std::ffi::c_int;
        }
        let _ = unsafe { (self.shutdown)() };
        let _ = unsafe { dlclose(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_statistics_and_checksum_are_deterministic() {
        let mut values = [4.0, 1.0, 3.0, 2.0, 5.0];
        assert_eq!(median(&mut values), 3.0);
        assert_eq!(confidence_interval(&[1.0, 2.0, 3.0]), (1.0, 3.0));
        let values = [1.0 as Float, 2.0 as Float];
        assert_eq!(checksum(&values), checksum(&values));
    }

    #[test]
    fn performance_gate_is_separate_from_evidence() {
        let outcome = QualificationOutcome {
            jsonl: "evidence\n".to_owned(),
            regressions: vec![Regression {
                backend: "neon",
                input_length: 4_096,
                delta_percent: 4.9,
                requirement: "must beat scalar",
            }],
        };
        assert_eq!(outcome.jsonl(), "evidence\n");
        assert!(outcome.require_pass().is_err());
    }
}
