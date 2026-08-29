//! Support code for opt-in benchmark executables.

#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_cases;
#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_evidence;
#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_execution;
#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_matrix;
#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_report;
#[cfg(feature = "catalogue-matrix")]
pub mod catalogue_statistics;
pub mod fixture;
pub mod pattern_shapes;
#[cfg(all(
    feature = "simd-qualification",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub mod performance_qualification;
