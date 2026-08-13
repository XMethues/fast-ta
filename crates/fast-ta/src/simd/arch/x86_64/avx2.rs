//! AVX2 SIMD implementation for x86_64

use crate::simd::scalar;
use crate::types::Float;
use crate::Result;

#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn sum(data: &[Float]) -> Float {
    data.iter().copied().sum()
}

#[inline(never)]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Dot product requires vectors of equal length".into(),
        });
    }
    let mut sum = 0.0 as Float;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    Ok(sum)
}

#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmpeq_epi64,
        _mm256_loadu_pd, _mm256_movemask_pd, _mm256_or_si256, _mm256_set1_epi64x,
    };

    const LANES: usize = 4;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: i64 = 0x7ff0_0000_0000_0000;

    let exponent_mask = _mm256_set1_epi64x(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let first_bits = _mm256_castpd_si256(_mm256_loadu_pd(ptr));
        let second_bits = _mm256_castpd_si256(_mm256_loadu_pd(ptr.add(LANES)));
        let third_bits = _mm256_castpd_si256(_mm256_loadu_pd(ptr.add(LANES * 2)));
        let fourth_bits = _mm256_castpd_si256(_mm256_loadu_pd(ptr.add(LANES * 3)));
        let first = _mm256_cmpeq_epi64(_mm256_and_si256(first_bits, exponent_mask), exponent_mask);
        let second =
            _mm256_cmpeq_epi64(_mm256_and_si256(second_bits, exponent_mask), exponent_mask);
        let third = _mm256_cmpeq_epi64(_mm256_and_si256(third_bits, exponent_mask), exponent_mask);
        let fourth =
            _mm256_cmpeq_epi64(_mm256_and_si256(fourth_bits, exponent_mask), exponent_mask);
        let invalid = _mm256_or_si256(
            _mm256_or_si256(first, second),
            _mm256_or_si256(third, fourth),
        );
        if _mm256_movemask_pd(_mm256_castsi256_pd(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        let bits = _mm256_castpd_si256(_mm256_loadu_pd(values.as_ptr().add(index)));
        let invalid = _mm256_cmpeq_epi64(_mm256_and_si256(bits, exponent_mask), exponent_mask);
        if _mm256_movemask_pd(_mm256_castsi256_pd(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi32,
        _mm256_loadu_ps, _mm256_movemask_ps, _mm256_or_si256, _mm256_set1_epi32,
    };

    const LANES: usize = 8;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: i32 = 0x7f80_0000;

    let exponent_mask = _mm256_set1_epi32(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let first_bits = _mm256_castps_si256(_mm256_loadu_ps(ptr));
        let second_bits = _mm256_castps_si256(_mm256_loadu_ps(ptr.add(LANES)));
        let third_bits = _mm256_castps_si256(_mm256_loadu_ps(ptr.add(LANES * 2)));
        let fourth_bits = _mm256_castps_si256(_mm256_loadu_ps(ptr.add(LANES * 3)));
        let first = _mm256_cmpeq_epi32(_mm256_and_si256(first_bits, exponent_mask), exponent_mask);
        let second =
            _mm256_cmpeq_epi32(_mm256_and_si256(second_bits, exponent_mask), exponent_mask);
        let third = _mm256_cmpeq_epi32(_mm256_and_si256(third_bits, exponent_mask), exponent_mask);
        let fourth =
            _mm256_cmpeq_epi32(_mm256_and_si256(fourth_bits, exponent_mask), exponent_mask);
        let invalid = _mm256_or_si256(
            _mm256_or_si256(first, second),
            _mm256_or_si256(third, fourth),
        );
        if _mm256_movemask_ps(_mm256_castsi256_ps(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        let bits = _mm256_castps_si256(_mm256_loadu_ps(values.as_ptr().add(index)));
        let invalid = _mm256_cmpeq_epi32(_mm256_and_si256(bits, exponent_mask), exponent_mask);
        if _mm256_movemask_ps(_mm256_castsi256_ps(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::x86_64::{
        _mm256_add_pd, _mm256_div_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    const LANES: usize = 4;
    let denominator = _mm256_set1_pd(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = _mm256_loadu_pd(high.as_ptr().add(index));
        let low_values = _mm256_loadu_pd(low.as_ptr().add(index));
        let close_values = _mm256_loadu_pd(close.as_ptr().add(index));
        let values = _mm256_div_pd(
            _mm256_add_pd(_mm256_add_pd(high_values, low_values), close_values),
            denominator,
        );
        _mm256_storeu_pd(output.as_mut_ptr().add(index), values);
        index += LANES;
    }

    scalar::typical_price(
        &high[index..],
        &low[index..],
        &close[index..],
        &mut output[index..],
    );
}

#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "avx2")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };

    const LANES: usize = 8;
    let denominator = _mm256_set1_ps(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = _mm256_loadu_ps(high.as_ptr().add(index));
        let low_values = _mm256_loadu_ps(low.as_ptr().add(index));
        let close_values = _mm256_loadu_ps(close.as_ptr().add(index));
        let values = _mm256_div_ps(
            _mm256_add_ps(_mm256_add_ps(high_values, low_values), close_values),
            denominator,
        );
        _mm256_storeu_ps(output.as_mut_ptr().add(index), values);
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
mod tests {
    use super::*;

    #[test]
    fn avx2_typical_price_matches_scalar_at_lane_boundaries_and_tail() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let lanes = if cfg!(feature = "f32") { 8 } else { 4 };
        for len in [0, 1, lanes - 1, lanes, lanes + 1, lanes * 4 + 3, 65_537] {
            let high: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.5 as Float + 3.25 as Float)
                .collect();
            let low: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.25 as Float - 1.5 as Float)
                .collect();
            let close: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.125 as Float + 0.75 as Float)
                .collect();
            let mut expected = vec![0.0 as Float; len];
            let mut actual = vec![0.0 as Float; len];

            scalar::typical_price(&high, &low, &close, &mut expected);
            unsafe { typical_price(&high, &low, &close, &mut actual) };

            assert_eq!(actual, expected, "length {len}");
        }
    }

    #[test]
    fn avx2_first_non_finite_matches_scalar_and_preserves_first_index() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let finite_values = [
            0.0 as Float,
            -0.0 as Float,
            Float::MIN,
            Float::MAX,
            Float::MIN_POSITIVE,
            -Float::MIN_POSITIVE,
        ];
        assert_eq!(unsafe { first_non_finite(&finite_values) }, None);

        let lanes = if cfg!(feature = "f32") { 8 } else { 4 };
        for len in [0, 1, lanes - 1, lanes, lanes + 1, lanes * 4 + 3, 65_537] {
            let values = vec![1.0 as Float; len];
            assert_eq!(
                unsafe { first_non_finite(&values) },
                scalar::first_non_finite(&values),
                "all-finite length {len}"
            );
        }

        let len = lanes * 8 + 3;
        for invalid_index in 0..len {
            let mut values = vec![1.0 as Float; len];
            values[invalid_index] = match invalid_index % 3 {
                0 => Float::NAN,
                1 => Float::INFINITY,
                _ => Float::NEG_INFINITY,
            };
            assert_eq!(
                unsafe { first_non_finite(&values) },
                scalar::first_non_finite(&values),
                "invalid index {invalid_index}"
            );
        }

        let mut values = vec![1.0 as Float; len];
        values[lanes + 1] = Float::NAN;
        values[lanes * 4] = Float::INFINITY;
        assert_eq!(unsafe { first_non_finite(&values) }, Some(lanes + 1));
    }
}
