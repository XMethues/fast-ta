use crate::simd::types::{Lanes, SimdVecExt};
use crate::types::Float;
use crate::Result;

#[cfg(all(feature = "f64", not(feature = "f32")))]
type SimdVec = wide::f64x2;

#[cfg(feature = "f32")]
type SimdVec = wide::f32x4;

/// SIMD128 SIMD array sum
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(Lanes::SIMD128);
    let remainder = chunks.remainder();

    let mut sum_vec = SimdVec::ZERO;

    for chunk in chunks {
        let vec = SimdVec::from_slice_unaligned(chunk);
        sum_vec += vec;
    }

    let mut sum = sum_vec.horizontal_sum();

    for &x in remainder {
        sum += x;
    }

    sum
}

/// SIMD128 SIMD dot product calculation
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a
        .chunks_exact(Lanes::SIMD128)
        .zip(b.chunks_exact(Lanes::SIMD128));
    let remainder_a = a.chunks_exact(Lanes::SIMD128).remainder();
    let remainder_b = b.chunks_exact(Lanes::SIMD128).remainder();

    for (chunk_a, chunk_b) in chunks {
        let vec_a = SimdVec::from_slice_unaligned(chunk_a);
        let vec_b = SimdVec::from_slice_unaligned(chunk_b);
        sum += (vec_a * vec_b).horizontal_sum();
    }

    for (&x, &y) in remainder_a.iter().zip(remainder_b.iter()) {
        sum += x * y;
    }

    Ok(sum)
}

#[cfg(test)]
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let data: Vec<Float> = (0..1000).map(|i| i as Float).collect();
        unsafe {
            let result = sum(&data);
            let expected: Float = data.iter().sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sum_empty() {
        let data: Vec<Float> = vec![];
        unsafe {
            let result = sum(&data);
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn test_sum_single() {
        let data: Vec<Float> = vec![42.0];
        unsafe {
            let result = sum(&data);
            assert_eq!(result, 42.0);
        }
    }

    #[test]
    fn test_dot_product() {
        let a: Vec<Float> = (0..1000).map(|i| i as Float).collect();
        let b: Vec<Float> = (0..1000).map(|i| (i * 2) as Float).collect();
        unsafe {
            let result = dot_product(&a, &b).unwrap();
            let expected: Float = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a: Vec<Float> = vec![1.0, 2.0, 3.0];
        let b: Vec<Float> = vec![1.0, 2.0];
        unsafe {
            assert!(dot_product(&a, &b).is_err());
        }
    }
}
