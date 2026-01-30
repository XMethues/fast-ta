//! ARM64 SIMD implementation

#[cfg(all(target_arch = "aarch64", feature = "std"))]
pub mod neon;
