//!
//! This module provides pure scalar implementations of all SIMD operations.
//!
//! These serve as a portable fallback when no SIMD acceleration is available.
//!
use crate::types::Float;

/// Calculate sum of all elements in a slice using scalar operations.
///
/// This is fallback implementation when no SIMD acceleration is available.
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
/// let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(sum(&data), 15.0);
/// ```
#[inline]
pub fn sum(data: &[Float]) -> Float {
    data.iter().sum()
}

/// Calculate dot product of two vectors using scalar operations.
///
/// This is fallback implementation when no SIMD acceleration is available.
/// It computes the element-wise product and sum of two vectors.
///
/// # Arguments
///
/// * `a` - First vector (slice of floating-point values)
/// * `b` - Second vector (slice of floating-point values)
///
/// # Returns
///
/// The dot product (element-wise multiplication sum) of two vectors.
///
/// # Panics
///
/// Panics if input vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use ta_core::simd::scalar::dot_product;
///
/// let a = vec![1.0_f32, 2.0, 3.0];
/// let b = vec![4.0_f32, 5.0, 6.0];
/// // (1*4) + (2*5) + (3*6) = 32
/// assert_eq!(dot_product(&a, &b), 32.0);
/// ```
#[inline]
pub fn dot_product(a: &[Float], b: &[Float]) -> Float {
    assert_eq!(
        a.len(),
        b.len(),
        "Dot product requires vectors of equal length"
    );

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate rolling sums with a specified window size using scalar operations.
///
/// This is fallback implementation when no SIMD acceleration is available.
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
/// let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
/// let result = rolling_sum(&data, 3);
/// // Windows: [1,2,3]=6, [2,3,4]=9, [3,4,5]=12
/// assert_eq!(result, vec![6.0, 9.0, 12.0]);
/// ```
#[inline]
pub fn rolling_sum(data: &[Float], window_size: usize) -> Vec<Float> {
    assert!(window_size >= 1, "Window size must be at least 1");
    assert!(
        data.len() >= window_size,
        "Data length must be at least window size"
    );

    let n = data.len();
    let result_len = n - window_size + 1;
    let mut result = Vec::with_capacity(result_len);

    // Calculate first window sum
    let mut current_sum: Float = data[..window_size].iter().sum();
    result.push(current_sum);

    // Slide: window: subtract leaving element, add entering element
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
    #[test]
    fn test_sum_empty() {
        let data: Vec<Float> = vec![];
        assert_eq!(sum(&data), Float::from(0.0));
    }

    #[test]
    fn test_sum_single() {
        let data = vec![Float::from(5.0)];
        assert_eq!(sum(&data), Float::from(5.0));
    }

    #[test]
    fn test_sum_multiple() {
        let data = vec![
            Float::from(1.0),
            Float::from(2.0),
            Float::from(3.0),
            Float::from(4.0),
            Float::from(5.0),
        ];
        assert_eq!(sum(&data), Float::from(15.0));
    }

    #[test]
    fn test_sum_with_negatives() {
        let data = vec![
            Float::from(1.0),
            Float::from(-2.0),
            Float::from(3.0),
            Float::from(-4.0),
            Float::from(5.0),
        ];
        assert_eq!(sum(&data), Float::from(3.0));
    }

    #[test]
    fn test_sum_with_zeros() {
        let data = vec![
            Float::from(0.0),
            Float::from(1.0),
            Float::from(0.0),
            Float::from(2.0),
            Float::from(0.0),
        ];
        assert_eq!(sum(&data), Float::from(3.0));
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<Float> = vec![];
        let b: Vec<Float> = vec![];
        assert_eq!(dot_product(&a, &b), Float::from(0.0));
    }

    #[test]
    fn test_dot_product_single() {
        let a = vec![Float::from(3.0)];
        let b = vec![Float::from(4.0)];
        assert_eq!(dot_product(&a, &b), Float::from(12.0));
    }

    #[test]
    fn test_dot_product_multiple() {
        let a = vec![Float::from(1.0), Float::from(2.0), Float::from(3.0)];
        let b = vec![Float::from(4.0), Float::from(5.0), Float::from(6.0)];
        // (1*4) + (2*5) + (3*6) = 32
        assert_eq!(dot_product(&a, &b), Float::from(32.0));
    }

    #[test]
    fn test_dot_product_with_negatives() {
        let a = vec![Float::from(1.0), Float::from(-2.0), Float::from(3.0)];
        let b = vec![Float::from(4.0), Float::from(5.0), Float::from(-6.0)];
        // (1*4) + (-2*5) + (3*-6) = 4 - 10 - 18 = -24
        assert_eq!(dot_product(&a, &b), Float::from(-24.0));
    }

    #[test]
    fn test_rolling_sum_basic() {
        let data = vec![
            Float::from(1.0),
            Float::from(2.0),
            Float::from(3.0),
            Float::from(4.0),
            Float::from(5.0),
        ];
        let result = rolling_sum(&data, 3);
        assert_eq!(
            result,
            vec![Float::from(6.0), Float::from(9.0), Float::from(12.0)]
        );
    }

    #[test]
    fn test_rolling_sum_window_size_1() {
        let data = vec![
            Float::from(1.0),
            Float::from(2.0),
            Float::from(3.0),
            Float::from(4.0),
        ];
        let result = rolling_sum(&data, 1);
        assert_eq!(
            result,
            vec![
                Float::from(1.0),
                Float::from(2.0),
                Float::from(3.0),
                Float::from(4.0)
            ]
        );
    }

    #[test]
    fn test_rolling_sum_full_window() {
        let data = vec![Float::from(1.0), Float::from(2.0), Float::from(3.0)];
        let result = rolling_sum(&data, 3);
        assert_eq!(result, vec![Float::from(6.0)]);
    }

    #[test]
    fn test_rolling_sum_with_negatives() {
        let data = vec![
            Float::from(1.0),
            Float::from(-1.0),
            Float::from(1.0),
            Float::from(-1.0),
            Float::from(1.0),
            Float::from(1.0),
        ];
        let result = rolling_sum(&data, 3);
        // Windows: [1,-1,1]=1, [-1,1,1]=1, [1,1,-1]=1
        assert_eq!(
            result,
            vec![
                Float::from(1.0),
                Float::from(-1.0),
                Float::from(1.0),
                Float::from(1.0)
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Window size must be at least 1")]
    fn test_rolling_sum_zero_window() {
        let data = vec![Float::from(1.0), Float::from(2.0), Float::from(3.0)];
        let _ = rolling_sum(&data, 0);
    }

    #[test]
    #[should_panic(expected = "Data length must be at least window size")]
    fn test_rolling_sum_window_too_large() {
        let data = vec![Float::from(1.0), Float::from(2.0), Float::from(3.0)];
        let _ = rolling_sum(&data, 5);
    }

    #[test]
    fn test_rolling_sum_large_window() {
        let data: Vec<Float> = (1..=100).map(|i| Float::from(i as f64)).collect();
        let result = rolling_sum(&data, 10);
        assert_eq!(result.len(), 91);
        // First window: 1+2+...+10 = 55, Last window: 90+91+92+...+100 = 955
        assert_eq!(result[0], Float::from(55.0));
        assert_eq!(result[90], Float::from(955.0));
    }

    #[test]
    fn test_rolling_sum_consistency_with_sum() {
        let data: Vec<Float> = vec![
            Float::from(1.0),
            Float::from(2.0),
            Float::from(3.0),
            Float::from(4.0),
            Float::from(5.0),
            Float::from(6.0),
        ];
        let result = rolling_sum(&data, data.len());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], sum(&data));
    }
}
