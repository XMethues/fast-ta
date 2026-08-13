//! x86_64 SIMD implementation

#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod avx512;
