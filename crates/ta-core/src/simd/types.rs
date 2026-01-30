//! SIMD types and constants.
//!
//! This module defines common types and platform-specific constants used across
//! all SIMD implementations.

use core::fmt;

// Platform-specific arch types (using core::arch for no_std compatibility)
// These will be used in future platform-specific implementations
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use core::arch::aarch64::*;

/// SIMD instruction set level.
///
/// Represents different levels of SIMD support available on different platforms.
/// Each level indicates the capabilities and vector width for that instruction set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SimdLevel {
    /// No SIMD acceleration (scalar operations)
    Scalar,
    /// AVX2 (Advanced Vector Extensions 2) - x86-64, 256-bit, 4 lanes of f64
    Avx2,
    /// AVX-512 (Advanced Vector Extensions 512) - x86-64, 512-bit, 8 lanes of f64
    Avx512,
    /// NEON - ARM/AArch64, 128-bit, 2 lanes of f64
    Neon,
    /// SIMD128 - WebAssembly, 128-bit, 2 lanes of f64
    Simd128,
}

impl SimdLevel {
    /// Detect the best available SIMD level at runtime.
    ///
    /// This function checks the CPU features and returns the highest supported
    /// SIMD level for the current platform.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ta_core::simd::SimdLevel;
    ///
    /// let level = SimdLevel::detect();
    /// println!("Best SIMD level: {:?}", level);
    /// ```
    #[inline]
    pub fn detect() -> Self {
        // Detect AVX-512
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_feature = "avx512f")]
            {
                if is_x86_feature_detected!("avx512f") {
                    return SimdLevel::Avx512;
                }
            }
            #[cfg(target_feature = "avx2")]
            {
                if is_x86_feature_detected!("avx2") {
                    return SimdLevel::Avx2;
                }
            }
        }

        // Detect NEON on ARM
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on AArch64
            return SimdLevel::Neon;
        }

        // Detect WASM SIMD
        #[cfg(target_arch = "wasm32")]
        {
            // SIMD128 is detected at compile-time via cfg
            return SimdLevel::Simd128;
        }

        // Fall back to scalar
        SimdLevel::Scalar
    }

    /// Get the number of lanes for this SIMD level.
    ///
    /// # f64 (default):
    /// - SCALAR: 1
    /// - AVX2: 4
    /// - AVX-512: 8
    /// - NEON: 2
    /// - SIMD128: 2
    ///
    /// # f32 (when "f32" feature is enabled):
    /// - SCALAR: 1
    /// - AVX2: 8
    /// - AVX-512: 16
    /// - NEON: 4
    /// - SIMD128: 4
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ta_core::simd::SimdLevel;
    ///
    /// // f64 (default)
    /// assert_eq!(SimdLevel::Avx2.lanes(), 4);
    /// assert_eq!(SimdLevel::Scalar.lanes(), 1);
    /// ```
    #[inline]
    pub fn lanes(&self) -> usize {
        match self {
            SimdLevel::Scalar => 1,
            #[cfg(all(feature = "f64", not(feature = "f32")))]
            SimdLevel::Avx2 => 4,
            #[cfg(feature = "f32")]
            SimdLevel::Avx2 => 8,
            #[cfg(all(feature = "f64", not(feature = "f32")))]
            SimdLevel::Avx512 => 8,
            #[cfg(feature = "f32")]
            SimdLevel::Avx512 => 16,
            #[cfg(all(feature = "f64", not(feature = "f32")))]
            SimdLevel::Neon => 2,
            #[cfg(feature = "f32")]
            SimdLevel::Neon => 4,
            #[cfg(all(feature = "f64", not(feature = "f32")))]
            SimdLevel::Simd128 => 2,
            #[cfg(feature = "f32")]
            SimdLevel::Simd128 => 4,
        }
    }

    /// Get the vector width in bits for this SIMD level.
    ///
    /// # f64 (default):
    /// - SCALAR: 64 bits
    /// - AVX2: 256 bits
    /// - AVX-512: 512 bits
    /// - NEON: 128 bits
    /// - SIMD128: 128 bits
    ///
    /// # f32 (when "f32" feature is enabled):
    /// - SCALAR: 32 bits
    /// - AVX2: 256 bits
    /// - AVX-512: 512 bits
    /// - NEON: 128 bits
    /// - SIMD128: 128 bits
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ta_core::simd::SimdLevel;
    ///
    /// // f64 (default)
    /// assert_eq!(SimdLevel::Avx2.width_bits(), 256);
    /// assert_eq!(SimdLevel::Scalar.width_bits(), 64);
    /// ```
    #[inline]
    pub fn width_bits(&self) -> usize {
        #[cfg(all(feature = "f64", not(feature = "f32")))]
        {
            self.lanes() * 64
        }
        #[cfg(feature = "f32")]
        {
            self.lanes() * 32
        }
        #[cfg(not(any(feature = "f32", feature = "f64")))]
        {
            // Fallback if neither feature is explicitly set
            self.lanes() * 64
        }
    }
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdLevel::Scalar => write!(f, "Scalar"),
            SimdLevel::Avx2 => write!(f, "AVX2"),
            SimdLevel::Avx512 => write!(f, "AVX-512"),
            SimdLevel::Neon => write!(f, "NEON"),
            SimdLevel::Simd128 => write!(f, "SIMD128"),
        }
    }
}

/// Lane count for each SIMD level.
///
/// This struct provides compile-time constants for the number of lanes
/// supported by each SIMD instruction set.
///
/// # f64 Lanes (default):
/// - SCALAR: 1 lane
/// - AVX2: 4 lanes (256-bit / 64-bit)
/// - AVX-512: 8 lanes (512-bit / 64-bit)
/// - NEON: 2 lanes (128-bit / 64-bit)
/// - SIMD128: 2 lanes (128-bit / 64-bit)
///
/// # f32 Lanes (when "f32" feature is enabled):
/// - SCALAR: 1 lane
/// - AVX2: 8 lanes (256-bit / 32-bit)
/// - AVX-512: 16 lanes (512-bit / 32-bit)
/// - NEON: 4 lanes (128-bit / 32-bit)
/// - SIMD128: 4 lanes (128-bit / 32-bit)
#[derive(Debug, Clone, Copy)]
pub struct Lanes;

impl Lanes {
    /// Number of lanes for scalar operations
    pub const SCALAR: usize = 1;

    /// Number of lanes for AVX2
    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub const AVX2: usize = 4;

    /// Number of lanes for AVX2 (f32)
    #[cfg(feature = "f32")]
    pub const AVX2: usize = 8;

    /// Number of lanes for AVX-512
    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub const AVX512: usize = 8;

    /// Number of lanes for AVX-512 (f32)
    #[cfg(feature = "f32")]
    pub const AVX512: usize = 16;

    /// Number of lanes for NEON
    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub const NEON: usize = 2;

    /// Number of lanes for NEON (f32)
    #[cfg(feature = "f32")]
    pub const NEON: usize = 4;

    /// Number of lanes for SIMD128
    #[cfg(all(feature = "f64", not(feature = "f32")))]
    pub const SIMD128: usize = 2;

    /// Number of lanes for SIMD128 (f32)
    #[cfg(feature = "f32")]
    pub const SIMD128: usize = 4;
}

/// Base trait for SIMD floating-point operations.
///
/// This trait defines common operations that all SIMD implementations must support.
/// Each SIMD level provides its own implementation of this trait.
pub trait SimdFloat: Sized {
    /// The underlying SIMD type for this implementation.
    type V;

    /// Create a SIMD vector from a scalar value (broadcast).
    fn splat(value: f64) -> Self::V;

    /// Load values from a slice into a SIMD vector.
    ///
    /// # Safety
    ///
    /// The slice must have at least the required number of elements for this SIMD level.
    unsafe fn load_from_slice(data: &[f64]) -> Self::V;

    /// Store SIMD vector values to a slice.
    ///
    /// # Safety
    ///
    /// The slice must have at least the required number of elements for this SIMD level.
    unsafe fn store_to_slice(value: Self::V, data: &mut [f64]);

    /// Add two SIMD vectors element-wise.
    fn add(a: Self::V, b: Self::V) -> Self::V;

    /// Subtract two SIMD vectors element-wise.
    fn sub(a: Self::V, b: Self::V) -> Self::V;

    /// Multiply two SIMD vectors element-wise.
    fn mul(a: Self::V, b: Self::V) -> Self::V;

    /// Divide two SIMD vectors element-wise.
    fn div(a: Self::V, b: Self::V) -> Self::V;

    /// Calculate the horizontal sum of all lanes in the SIMD vector.
    fn horizontal_sum(value: Self::V) -> f64;

    /// Calculate the horizontal dot product of two SIMD vectors.
    fn dot_product(a: Self::V, b: Self::V) -> f64 {
        let mul = Self::mul(a, b);
        Self::horizontal_sum(mul)
    }
}

/// Trait for SIMD mask/comparison operations.
///
/// This trait defines operations for comparing SIMD vectors and working with masks.
pub trait SimdMask: SimdFloat {
    /// Compare two SIMD vectors for equality.
    fn eq(a: Self::V, b: Self::V) -> Self::V;

    /// Compare two SIMD vectors for greater than.
    fn gt(a: Self::V, b: Self::V) -> Self::V;

    /// Compare two SIMD vectors for less than.
    fn lt(a: Self::V, b: Self::V) -> Self::V;

    /// Blend values from two vectors based on a mask.
    ///
    /// For each lane, select the value from `then` if the mask is true,
    /// otherwise select from `else_`.
    fn blend(mask: Self::V, then: Self::V, else_: Self::V) -> Self::V;
}

/// Common SIMD operations trait.
///
/// This trait combines floating-point and mask operations for convenience.
pub trait SimdOps: SimdFloat + SimdMask {}

/// Empty marker trait for scalar operations.
///
/// Scalar implementation doesn't have a true SIMD type, but we use this trait
/// for API consistency.
#[allow(dead_code)]
pub trait ScalarOps: SimdFloat + SimdMask {}

// ============================================================================
// SIMD type aliases using wide crate
// ============================================================================

/// SIMD vector type for AVX2 with f64.
#[cfg(all(feature = "f64", not(feature = "f32")))]
pub type SimdVecAvx2 = wide::f64x4;

/// SIMD vector type for AVX2 with f32.
#[cfg(feature = "f32")]
pub type SimdVecAvx2 = wide::f32x8;

/// SIMD vector type for AVX-512 with f64.
#[cfg(all(feature = "f64", not(feature = "f32")))]
pub type SimdVecAvx512 = wide::f64x8;

/// SIMD vector type for AVX-512 with f32.
#[cfg(feature = "f32")]
pub type SimdVecAvx512 = wide::f32x16;

/// Default SIMD lanes (AVX2).
///
/// This constant uses the existing Lanes struct to avoid duplication.
pub const SIMD_LANES: usize = Lanes::AVX2;

pub trait SimdVecExt {
    const ZERO: Self;

    unsafe fn from_slice_unaligned(data: &[crate::types::Float]) -> Self;

    fn horizontal_sum(self) -> crate::types::Float;
}

#[cfg(all(feature = "f64", not(feature = "f32")))]
impl SimdVecExt for wide::f64x4 {
    const ZERO: Self = wide::f64x4::splat(0.0);

    #[inline]
    unsafe fn from_slice_unaligned(data: &[crate::types::Float]) -> Self {
        wide::f64x4::from_slice_unaligned(data)
    }

    #[inline]
    fn horizontal_sum(self) -> crate::types::Float {
        self.reduce_add()
    }
}

#[cfg(feature = "f32")]
impl SimdVecExt for wide::f32x8 {
    const ZERO: Self = wide::f32x8::splat(0.0);

    #[inline]
    unsafe fn from_slice_unaligned(data: &[crate::types::Float]) -> Self {
        wide::f32x8::from_slice_unaligned(data)
    }

    #[inline]
    fn horizontal_sum(self) -> crate::types::Float {
        self.reduce_add()
    }
}

#[cfg(all(feature = "f64", not(feature = "f32")))]
impl SimdVecExt for wide::f64x8 {
    const ZERO: Self = wide::f64x8::splat(0.0);

    #[inline]
    unsafe fn from_slice_unaligned(data: &[crate::types::Float]) -> Self {
        wide::f64x8::from_slice_unaligned(data)
    }

    #[inline]
    fn horizontal_sum(self) -> crate::types::Float {
        self.reduce_add()
    }
}

#[cfg(feature = "f32")]
impl SimdVecExt for wide::f32x16 {
    const ZERO: Self = wide::f32x16::splat(0.0);

    #[inline]
    unsafe fn from_slice_unaligned(data: &[crate::types::Float]) -> Self {
        wide::f32x16::from_slice_unaligned(data)
    }

    #[inline]
    fn horizontal_sum(self) -> crate::types::Float {
        self.reduce_add()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_lanes_constants() {
        assert_eq!(Lanes::SCALAR, 1);
        #[cfg(all(feature = "f64", not(feature = "f32")))]
        {
            assert_eq!(Lanes::AVX2, 4);
            assert_eq!(Lanes::AVX512, 8);
            assert_eq!(Lanes::NEON, 2);
            assert_eq!(Lanes::SIMD128, 2);
        }
        #[cfg(feature = "f32")]
        {
            assert_eq!(Lanes::AVX2, 8);
            assert_eq!(Lanes::AVX512, 16);
            assert_eq!(Lanes::NEON, 4);
            assert_eq!(Lanes::SIMD128, 4);
        }
    }

    #[test]
    fn test_simd_level_display() {
        assert_eq!(format!("{}", SimdLevel::Scalar), "Scalar");
        assert_eq!(format!("{}", SimdLevel::Avx2), "AVX2");
        assert_eq!(format!("{}", SimdLevel::Avx512), "AVX-512");
        assert_eq!(format!("{}", SimdLevel::Neon), "NEON");
        assert_eq!(format!("{}", SimdLevel::Simd128), "SIMD128");
    }

    #[test]
    fn test_simd_level_lanes() {
        assert_eq!(SimdLevel::Scalar.lanes(), 1);
        #[cfg(all(feature = "f64", not(feature = "f32")))]
        {
            assert_eq!(SimdLevel::Avx2.lanes(), 4);
            assert_eq!(SimdLevel::Avx512.lanes(), 8);
            assert_eq!(SimdLevel::Neon.lanes(), 2);
            assert_eq!(SimdLevel::Simd128.lanes(), 2);
        }
        #[cfg(feature = "f32")]
        {
            assert_eq!(SimdLevel::Avx2.lanes(), 8);
            assert_eq!(SimdLevel::Avx512.lanes(), 16);
            assert_eq!(SimdLevel::Neon.lanes(), 4);
            assert_eq!(SimdLevel::Simd128.lanes(), 4);
        }
    }

    #[test]
    fn test_simd_level_width_bits() {
        #[cfg(all(feature = "f64", not(feature = "f32")))]
        {
            assert_eq!(SimdLevel::Scalar.width_bits(), 64);
            assert_eq!(SimdLevel::Avx2.width_bits(), 256);
            assert_eq!(SimdLevel::Avx512.width_bits(), 512);
            assert_eq!(SimdLevel::Neon.width_bits(), 128);
            assert_eq!(SimdLevel::Simd128.width_bits(), 128);
        }
        #[cfg(feature = "f32")]
        {
            assert_eq!(SimdLevel::Scalar.width_bits(), 32);
            assert_eq!(SimdLevel::Avx2.width_bits(), 256);
            assert_eq!(SimdLevel::Avx512.width_bits(), 512);
            assert_eq!(SimdLevel::Neon.width_bits(), 128);
            assert_eq!(SimdLevel::Simd128.width_bits(), 128);
        }
    }
}
