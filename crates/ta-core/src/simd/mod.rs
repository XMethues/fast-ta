//! # SIMD Accelerated Technical Analysis
//!
//! This module provides SIMD-accelerated implementations of technical analysis operations.
//! It automatically selects the best available SIMD instruction set based on the target platform
//! and provides a scalar fallback for unsupported platforms.
//!
//! ## SIMD Levels
//!
//! The library supports multiple SIMD levels:
//! - **Scalar**: No SIMD, pure scalar operations (always available)
//! - **Avx2**: Advanced Vector Extensions 2 (x86-64, 256-bit)
//! - **Avx512**: Advanced Vector Extensions 512 (x86-64, 512-bit)
//! - **Neon**: ARM Advanced SIMD (AArch64, 128-bit)
//! - **Simd128**: WebAssembly SIMD (128-bit)
//!
//! ## Usage
//!
//! The module provides a unified API for all SIMD levels. Functions are automatically
//! dispatched to the best available implementation at runtime or compile-time.
//!
//! ```rust
//! use ta_core::simd::{SimdLevel, sum};
//!
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0];
//! let result = sum(&data, SimdLevel::detect());
//! ```
//!
//! ## Performance Considerations
//!
//! - SIMD implementations require aligned data for optimal performance
//! - For very small arrays, scalar operations may be faster due to SIMD overhead
//! - Consider using [`rolling_sum`] for sliding window calculations

mod scalar;
mod types;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
mod arch;

use alloc::vec::Vec;

pub use types::{Lanes, SimdFloat, SimdLevel, SimdMask, SimdOps};

pub mod dispatch;

/// Calculate the sum of all elements in a slice.
///
/// # Arguments
///
/// * `data` - A slice of floating-point values
/// * `_level` - The SIMD level to use for computation (currently unused, reserved for future)
///
/// # Returns
///
/// The sum of all elements in the slice.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::{sum, SimdLevel};
///
/// let data = vec![1.0_f64, 2.0, 3.0];
/// let result = sum(&data, SimdLevel::detect());
/// assert_eq!(result, 6.0);
/// ```
#[inline]
pub fn sum(data: &[f64], _level: SimdLevel) -> f64 {
    scalar::sum(data)
}

/// Calculate the dot product of two vectors.
///
/// # Arguments
///
/// * `a` - First vector (slice of floating-point values)
/// * `b` - Second vector (slice of floating-point values)
/// * `_level` - The SIMD level to use for computation (currently unused, reserved for future)
///
/// # Returns
///
/// The dot product (element-wise multiplication sum) of the two vectors.
///
/// # Panics
///
/// Panics if the input vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::{dot_product, SimdLevel};
///
/// let a = vec![1.0_f64, 2.0, 3.0];
/// let b = vec![4.0_f64, 5.0, 6.0];
/// let result = dot_product(&a, &b, SimdLevel::detect());
/// // (1*4) + (2*5) + (3*6) = 32
/// assert_eq!(result, 32.0);
/// ```
#[inline]
pub fn dot_product(a: &[f64], b: &[f64], _level: SimdLevel) -> f64 {
    scalar::dot_product(a, b)
}

/// Calculate rolling sums with a specified window size.
///
/// This function computes the sum of each consecutive window of size `window_size`
/// in the input data. The result has length `data.len() - window_size + 1`.
///
/// # Arguments
///
/// * `data` - Input slice of floating-point values
/// * `window_size` - Size of the rolling window (must be >= 1)
/// * `_level` - The SIMD level to use for computation (currently unused, reserved for future)
///
/// # Returns
///
/// A vector containing the rolling sums.
///
/// # Panics
///
/// Panics if `window_size` is 0 or greater than the input data length.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::{rolling_sum, SimdLevel};
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let result = rolling_sum(&data, 3, SimdLevel::detect());
/// // Windows: [1,2,3]=6, [2,3,4]=9, [3,4,5]=12
/// assert_eq!(result, vec![6.0, 9.0, 12.0]);
/// ```
#[inline]
pub fn rolling_sum(data: &[f64], window_size: usize, _level: SimdLevel) -> Vec<f64> {
    scalar::rolling_sum(data, window_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sum(&data, SimdLevel::Scalar), 15.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b, SimdLevel::Scalar), 32.0);
    }

    #[test]
    fn test_rolling_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_sum(&data, 3, SimdLevel::Scalar);
        assert_eq!(result, vec![6.0, 9.0, 12.0]);
    }
}
