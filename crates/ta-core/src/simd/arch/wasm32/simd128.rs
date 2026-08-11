//! WebAssembly SIMD128 kernels.
//!
//! These functions may only be called when the final module is compiled with
//! `target_feature=+simd128`. Dispatch keeps them unreachable in scalar WASM
//! builds and preserves the portable path for `no_std`.

use crate::simd::scalar;
use crate::types::Float;
use crate::Result;

/// SIMD128 SIMD array sum
#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn sum(data: &[Float]) -> Float {
    use core::arch::wasm32::{f64x2_add, f64x2_extract_lane, f64x2_splat, v128, v128_load};

    const LANES: usize = 2;
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();
    let mut sum_vec = f64x2_splat(0.0);

    for chunk in chunks {
        sum_vec = f64x2_add(sum_vec, v128_load(chunk.as_ptr().cast::<v128>()));
    }

    let mut sum = f64x2_extract_lane::<0>(sum_vec) + f64x2_extract_lane::<1>(sum_vec);
    for &value in remainder {
        sum += value;
    }
    sum
}

/// SIMD128 SIMD array sum
#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn sum(data: &[Float]) -> Float {
    use core::arch::wasm32::{f32x4_add, f32x4_extract_lane, f32x4_splat, v128, v128_load};

    const LANES: usize = 4;
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();
    let mut sum_vec = f32x4_splat(0.0);

    for chunk in chunks {
        sum_vec = f32x4_add(sum_vec, v128_load(chunk.as_ptr().cast::<v128>()));
    }

    let mut sum = f32x4_extract_lane::<0>(sum_vec) + f32x4_extract_lane::<1>(sum_vec);
    sum += f32x4_extract_lane::<2>(sum_vec) + f32x4_extract_lane::<3>(sum_vec);
    for &value in remainder {
        sum += value;
    }
    sum
}

/// SIMD128 SIMD dot product calculation
#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    use core::arch::wasm32::{
        f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat, v128, v128_load,
    };

    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Dot product requires vectors of equal length".into(),
        });
    }

    const LANES: usize = 2;
    let chunks = a.chunks_exact(LANES).zip(b.chunks_exact(LANES));
    let remainder_a = a.chunks_exact(LANES).remainder();
    let remainder_b = b.chunks_exact(LANES).remainder();
    let mut sum_vec = f64x2_splat(0.0);

    for (chunk_a, chunk_b) in chunks {
        let a_values = v128_load(chunk_a.as_ptr().cast::<v128>());
        let b_values = v128_load(chunk_b.as_ptr().cast::<v128>());
        sum_vec = f64x2_add(sum_vec, f64x2_mul(a_values, b_values));
    }

    let mut sum = f64x2_extract_lane::<0>(sum_vec) + f64x2_extract_lane::<1>(sum_vec);
    for (&x, &y) in remainder_a.iter().zip(remainder_b) {
        sum += x * y;
    }
    Ok(sum)
}

/// SIMD128 SIMD dot product calculation
#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    use core::arch::wasm32::{
        f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128, v128_load,
    };

    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Dot product requires vectors of equal length".into(),
        });
    }

    const LANES: usize = 4;
    let chunks = a.chunks_exact(LANES).zip(b.chunks_exact(LANES));
    let remainder_a = a.chunks_exact(LANES).remainder();
    let remainder_b = b.chunks_exact(LANES).remainder();
    let mut sum_vec = f32x4_splat(0.0);

    for (chunk_a, chunk_b) in chunks {
        let a_values = v128_load(chunk_a.as_ptr().cast::<v128>());
        let b_values = v128_load(chunk_b.as_ptr().cast::<v128>());
        sum_vec = f32x4_add(sum_vec, f32x4_mul(a_values, b_values));
    }

    let mut sum = f32x4_extract_lane::<0>(sum_vec) + f32x4_extract_lane::<1>(sum_vec);
    sum += f32x4_extract_lane::<2>(sum_vec) + f32x4_extract_lane::<3>(sum_vec);
    for (&x, &y) in remainder_a.iter().zip(remainder_b) {
        sum += x * y;
    }
    Ok(sum)
}

/// Returns the first non-finite lane using SIMD128, with exact scalar recovery.
#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::wasm32::{
        i64x2_eq, i64x2_splat, i8x16_bitmask, v128, v128_and, v128_load, v128_or,
    };

    const LANES: usize = 2;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: i64 = 0x7ff0_0000_0000_0000;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn invalid_lanes(values: *const f64, exponent_mask: v128) -> v128 {
        let bits = v128_load(values.cast::<v128>());
        i64x2_eq(v128_and(bits, exponent_mask), exponent_mask)
    }

    let exponent_mask = i64x2_splat(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let invalid = v128_or(
            v128_or(
                invalid_lanes(ptr, exponent_mask),
                invalid_lanes(ptr.add(LANES), exponent_mask),
            ),
            v128_or(
                invalid_lanes(ptr.add(LANES * 2), exponent_mask),
                invalid_lanes(ptr.add(LANES * 3), exponent_mask),
            ),
        );
        if i8x16_bitmask(invalid) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        if i8x16_bitmask(invalid_lanes(values.as_ptr().add(index), exponent_mask)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

/// Returns the first non-finite lane using SIMD128, with exact scalar recovery.
#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::wasm32::{
        i32x4_eq, i32x4_splat, i8x16_bitmask, v128, v128_and, v128_load, v128_or,
    };

    const LANES: usize = 4;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: i32 = 0x7f80_0000;

    #[inline]
    #[target_feature(enable = "simd128")]
    unsafe fn invalid_lanes(values: *const f32, exponent_mask: v128) -> v128 {
        let bits = v128_load(values.cast::<v128>());
        i32x4_eq(v128_and(bits, exponent_mask), exponent_mask)
    }

    let exponent_mask = i32x4_splat(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let invalid = v128_or(
            v128_or(
                invalid_lanes(ptr, exponent_mask),
                invalid_lanes(ptr.add(LANES), exponent_mask),
            ),
            v128_or(
                invalid_lanes(ptr.add(LANES * 2), exponent_mask),
                invalid_lanes(ptr.add(LANES * 3), exponent_mask),
            ),
        );
        if i8x16_bitmask(invalid) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        if i8x16_bitmask(invalid_lanes(values.as_ptr().add(index), exponent_mask)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

/// Computes Typical Price using two `f64` lanes and an exact scalar tail.
#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::wasm32::{f64x2_add, f64x2_div, f64x2_splat, v128, v128_load, v128_store};

    const LANES: usize = 2;
    let denominator = f64x2_splat(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = v128_load(high.as_ptr().add(index).cast::<v128>());
        let low_values = v128_load(low.as_ptr().add(index).cast::<v128>());
        let close_values = v128_load(close.as_ptr().add(index).cast::<v128>());
        let values = f64x2_div(
            f64x2_add(f64x2_add(high_values, low_values), close_values),
            denominator,
        );
        v128_store(output.as_mut_ptr().add(index).cast::<v128>(), values);
        index += LANES;
    }

    scalar::typical_price(
        &high[index..],
        &low[index..],
        &close[index..],
        &mut output[index..],
    );
}

/// Computes Typical Price using four `f32` lanes and an exact scalar tail.
#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "simd128")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::wasm32::{f32x4_add, f32x4_div, f32x4_splat, v128, v128_load, v128_store};

    const LANES: usize = 4;
    let denominator = f32x4_splat(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = v128_load(high.as_ptr().add(index).cast::<v128>());
        let low_values = v128_load(low.as_ptr().add(index).cast::<v128>());
        let close_values = v128_load(close.as_ptr().add(index).cast::<v128>());
        let values = f32x4_div(
            f32x4_add(f32x4_add(high_values, low_values), close_values),
            denominator,
        );
        v128_store(output.as_mut_ptr().add(index).cast::<v128>(), values);
        index += LANES;
    }

    scalar::typical_price(
        &high[index..],
        &low[index..],
        &close[index..],
        &mut output[index..],
    );
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

    #[test]
    fn first_non_finite_matches_scalar_for_lanes_tails_and_exact_recovery() {
        let lane_width = if cfg!(feature = "f32") { 4 } else { 2 };
        let lengths = [
            0,
            1,
            lane_width - 1,
            lane_width,
            lane_width + 1,
            lane_width * 4 - 1,
            lane_width * 4,
            lane_width * 4 + 1,
            65_537,
        ];

        for len in lengths {
            let values = vec![1.0 as Float; len];
            assert_eq!(
                unsafe { first_non_finite(&values) },
                scalar::first_non_finite(&values),
                "all-finite length {len}"
            );
        }

        let len = lane_width * 8 + 3;
        for (index, value) in [
            (0, Float::NAN),
            (lane_width - 1, Float::INFINITY),
            (lane_width, Float::NEG_INFINITY),
            (lane_width * 4 - 1, Float::NAN),
            (lane_width * 4, Float::INFINITY),
            (len - 1, Float::NEG_INFINITY),
        ] {
            let mut values = vec![1.0 as Float; len];
            values[index] = value;
            assert_eq!(unsafe { first_non_finite(&values) }, Some(index));
            assert_eq!(
                unsafe { first_non_finite(&values) },
                scalar::first_non_finite(&values)
            );
        }

        let mut values = vec![1.0 as Float; len];
        values[lane_width + 1] = Float::NAN;
        values[0] = Float::INFINITY;
        assert_eq!(unsafe { first_non_finite(&values) }, Some(0));
    }

    #[test]
    fn typical_price_matches_scalar_for_lanes_tails_and_large_input() {
        let lane_width = if cfg!(feature = "f32") { 4 } else { 2 };
        for len in [
            0,
            1,
            lane_width - 1,
            lane_width,
            lane_width + 1,
            lane_width * 2 + 1,
            65_537,
        ] {
            let high: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.5 as Float + 3.25 as Float)
                .collect();
            let low: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.25 as Float - 1.5 as Float)
                .collect();
            let close: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.125 as Float + 0.75 as Float)
                .collect();
            let mut expected = vec![-12_345.0 as Float; len + 1];
            let mut actual = vec![-12_345.0 as Float; len + 1];

            scalar::typical_price(&high, &low, &close, &mut expected);
            unsafe { typical_price(&high, &low, &close, &mut actual) };

            assert_eq!(actual, expected, "length {len}");
        }
    }
}
