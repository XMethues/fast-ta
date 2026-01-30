//! Scalar fallback implementations of SIMD operations.
//!
//! This module provides pure scalar implementations of all SIMD operations.
//! These serve as a portable fallback when no SIMD acceleration is available.

use super::types::{SimdFloat, SimdMask, SimdOps};
use alloc::vec::Vec;

/// Scalar wrapper type that implements SimdFloat trait.
///
/// This type allows us to use the same API for scalar operations as we do
/// for vectorized operations, providing a consistent interface.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub struct Scalar(pub f64);

impl SimdFloat for Scalar {
    type V = f64;

    #[inline]
    fn splat(value: f64) -> Self::V {
        value
    }

    #[inline]
    unsafe fn load_from_slice(data: &[f64]) -> Self::V {
        *data.get_unchecked(0)
    }

    #[inline]
    unsafe fn store_to_slice(value: Self::V, data: &mut [f64]) {
        *data.get_unchecked_mut(0) = value;
    }

    #[inline]
    fn add(a: Self::V, b: Self::V) -> Self::V {
        a + b
    }

    #[inline]
    fn sub(a: Self::V, b: Self::V) -> Self::V {
        a - b
    }

    #[inline]
    fn mul(a: Self::V, b: Self::V) -> Self::V {
        a * b
    }

    #[inline]
    fn div(a: Self::V, b: Self::V) -> Self::V {
        a / b
    }

    #[inline]
    fn horizontal_sum(value: Self::V) -> f64 {
        value
    }
}

impl SimdMask for Scalar {
    #[inline]
    fn eq(a: Self::V, b: Self::V) -> Self::V {
        if a == b {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn gt(a: Self::V, b: Self::V) -> Self::V {
        if a > b {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn lt(a: Self::V, b: Self::V) -> Self::V {
        if a < b {
            1.0
        } else {
            0.0
        }
    }

    #[inline]
    fn blend(mask: Self::V, then: Self::V, else_: Self::V) -> Self::V {
        if mask != 0.0 {
            then
        } else {
            else_
        }
    }
}

impl SimdOps for Scalar {}

/// Calculate the sum of all elements in a slice using scalar operations.
///
/// This is the fallback implementation when no SIMD acceleration is available.
/// It provides a portable, safe implementation that works on all platforms.
///
/// # Arguments
///
/// * `data` - A slice of floating-point values
///
/// # Returns
///
/// The sum of all elements in the slice.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::scalar::sum;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(sum(&data), 15.0);
/// ```
#[inline]
pub fn sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

/// Calculate the dot product of two vectors using scalar operations.
///
/// This is the fallback implementation when no SIMD acceleration is available.
/// It computes the element-wise product and sum of two vectors.
///
/// # Arguments
///
/// * `a` - First vector (slice of floating-point values)
/// * `b` - Second vector (slice of floating-point values)
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
/// use ta_core::simd::scalar::dot_product;
///
/// let a = vec![1.0_f64, 2.0, 3.0];
/// let b = vec![4.0_f64, 5.0, 6.0];
/// // (1*4) + (2*5) + (3*6) = 32
/// assert_eq!(dot_product(&a, &b), 32.0);
/// ```
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Dot product requires vectors of equal length"
    );

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate rolling sums with a specified window size using scalar operations.
///
/// This is the fallback implementation when no SIMD acceleration is available.
/// It computes the sum of each consecutive window in the input data.
///
/// For improved performance with large windows, this implementation uses a sliding
/// window approach that subtracts the element leaving the window and adds the
/// new element entering, reducing the computational complexity from O(n*w) to O(n).
///
/// # Arguments
///
/// * `data` - Input slice of floating-point values
/// * `window_size` - Size of the rolling window (must be >= 1)
///
/// # Returns
///
/// A vector containing the rolling sums with length `data.len() - window_size + 1`.
///
/// # Panics
///
/// Panics if `window_size` is 0 or greater than the input data length.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::scalar::rolling_sum;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let result = rolling_sum(&data, 3);
/// // Windows: [1,2,3]=6, [2,3,4]=9, [3,4,5]=12
/// assert_eq!(result, vec![6.0, 9.0, 12.0]);
/// ```
#[inline]
pub fn rolling_sum(data: &[f64], window_size: usize) -> Vec<f64> {
    assert!(window_size >= 1, "Window size must be at least 1");
    assert!(
        data.len() >= window_size,
        "Data length must be at least window size"
    );

    let n = data.len();
    let result_len = n - window_size + 1;
    let mut result = Vec::with_capacity(result_len);

    // Calculate first window sum
    let mut current_sum: f64 = data[..window_size].iter().sum();
    result.push(current_sum);

    // Slide the window: subtract leaving element, add entering element
    for i in window_size..n {
        current_sum -= data[i - window_size];
        current_sum += data[i];
        result.push(current_sum);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_sum_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(sum(&data), 0.0);
    }

    #[test]
    fn test_sum_single() {
        let data = vec![5.0];
        assert_eq!(sum(&data), 5.0);
    }

    #[test]
    fn test_sum_multiple() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(sum(&data), 15.0);
    }

    #[test]
    fn test_sum_with_negatives() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        assert_eq!(sum(&data), 3.0);
    }

    #[test]
    fn test_sum_with_zeros() {
        let data = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        assert_eq!(sum(&data), 3.0);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(dot_product(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        let a = vec![3.0];
        let b = vec![4.0];
        assert_eq!(dot_product(&a, &b), 12.0);
    }

    #[test]
    fn test_dot_product_multiple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        assert_eq!(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_dot_product_with_negatives() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![4.0, 5.0, -6.0];
        // (1*4) + (-2*5) + (3*-6) = 4 - 10 - 18 = -24
        assert_eq!(dot_product(&a, &b), -24.0);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_dot_product_unequal_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        dot_product(&a, &b);
    }

    #[test]
    fn test_rolling_sum_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rolling_sum(&data, 3);
        assert_eq!(result, vec![6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_rolling_sum_window_size_1() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = rolling_sum(&data, 1);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_rolling_sum_full_window() {
        let data = vec![1.0, 2.0, 3.0];
        let result = rolling_sum(&data, 3);
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_rolling_sum_with_negatives() {
        let data = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let result = rolling_sum(&data, 3);
        // Windows: [1,-1,1]=1, [-1,1,-1]=-1, [1,-1,1]=1
        assert_eq!(result, vec![1.0, -1.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn test_rolling_sum_zero_window() {
        let data = vec![1.0, 2.0, 3.0];
        rolling_sum(&data, 0);
    }

    #[test]
    #[should_panic(expected = "at least window size")]
    fn test_rolling_sum_window_too_large() {
        let data = vec![1.0, 2.0, 3.0];
        rolling_sum(&data, 5);
    }

    #[test]
    fn test_rolling_sum_large_window() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let result = rolling_sum(&data, 10);

        assert_eq!(result.len(), 91);
        // First window: 1+2+...+10 = 55, Last window: 91+92+...+100 = 955
        assert_eq!(result[0], 55.0);
        assert_eq!(result[90], 955.0);
    }

    #[test]
    fn test_rolling_sum_consistency_with_sum() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // rolling_sum with window size = data.len() should give a single result equal to sum
        let result = rolling_sum(&data, data.len());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], sum(&data));
    }

    #[test]
    fn test_scalar_simd_float_trait() {
        use super::super::types::SimdFloat;

        let a = 3.0;
        let b = 2.0;

        assert_eq!(Scalar::splat(5.0), 5.0);
        assert_eq!(Scalar::add(a, b), 5.0);
        assert_eq!(Scalar::sub(a, b), 1.0);
        assert_eq!(Scalar::mul(a, b), 6.0);
        assert_eq!(Scalar::div(a, b), 1.5);
        assert_eq!(Scalar::horizontal_sum(7.0), 7.0);
        assert_eq!(Scalar::dot_product(a, b), 6.0);
    }

    #[test]
    fn test_scalar_simd_mask_trait() {
        use super::super::types::SimdMask;

        assert_eq!(<Scalar as SimdMask>::eq(3.0, 3.0), 1.0);
        assert_eq!(<Scalar as SimdMask>::eq(3.0, 4.0), 0.0);

        assert_eq!(<Scalar as SimdMask>::gt(5.0, 3.0), 1.0);
        assert_eq!(<Scalar as SimdMask>::gt(3.0, 5.0), 0.0);

        assert_eq!(<Scalar as SimdMask>::lt(3.0, 5.0), 1.0);
        assert_eq!(<Scalar as SimdMask>::lt(5.0, 3.0), 0.0);

        assert_eq!(<Scalar as SimdMask>::blend(1.0, 10.0, 20.0), 10.0);
        assert_eq!(<Scalar as SimdMask>::blend(0.0, 10.0, 20.0), 20.0);
    }
}
