//! ARM NEON SIMD implementation for AArch64

use crate::simd::scalar;
use crate::types::Float;
use crate::Result;

#[inline(never)]
#[target_feature(enable = "neon")]
#[allow(dead_code)]
pub unsafe fn sum(data: &[Float]) -> Float {
    data.iter().copied().sum()
}

#[inline(never)]
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::aarch64::{
        uint64x2_t, vandq_u64, vceqq_u64, vdupq_n_u64, vgetq_lane_u64, vld1q_f64, vorrq_u64,
        vreinterpretq_u64_f64,
    };

    const LANES: usize = 2;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: u64 = 0x7ff0_0000_0000_0000;

    #[inline(always)]
    unsafe fn invalid_lanes(values: *const f64, exponent_mask: uint64x2_t) -> uint64x2_t {
        let bits = vreinterpretq_u64_f64(vld1q_f64(values));
        vceqq_u64(vandq_u64(bits, exponent_mask), exponent_mask)
    }

    let exponent_mask = vdupq_n_u64(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let first = invalid_lanes(ptr, exponent_mask);
        let second = invalid_lanes(ptr.add(LANES), exponent_mask);
        let third = invalid_lanes(ptr.add(LANES * 2), exponent_mask);
        let fourth = invalid_lanes(ptr.add(LANES * 3), exponent_mask);
        let invalid = vorrq_u64(vorrq_u64(first, second), vorrq_u64(third, fourth));
        if (vgetq_lane_u64::<0>(invalid) | vgetq_lane_u64::<1>(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        let invalid = invalid_lanes(values.as_ptr().add(index), exponent_mask);
        if (vgetq_lane_u64::<0>(invalid) | vgetq_lane_u64::<1>(invalid)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn first_non_finite(values: &[Float]) -> Option<usize> {
    use core::arch::aarch64::{
        uint32x4_t, vandq_u32, vceqq_u32, vdupq_n_u32, vld1q_f32, vmaxvq_u32, vorrq_u32,
        vreinterpretq_u32_f32,
    };

    const LANES: usize = 4;
    const UNROLLED_LANES: usize = LANES * 4;
    const EXPONENT_MASK: u32 = 0x7f80_0000;

    #[inline(always)]
    unsafe fn invalid_lanes(values: *const f32, exponent_mask: uint32x4_t) -> uint32x4_t {
        let bits = vreinterpretq_u32_f32(vld1q_f32(values));
        vceqq_u32(vandq_u32(bits, exponent_mask), exponent_mask)
    }

    let exponent_mask = vdupq_n_u32(EXPONENT_MASK);
    let mut index = 0;
    while index + UNROLLED_LANES <= values.len() {
        let ptr = values.as_ptr().add(index);
        let first = invalid_lanes(ptr, exponent_mask);
        let second = invalid_lanes(ptr.add(LANES), exponent_mask);
        let third = invalid_lanes(ptr.add(LANES * 2), exponent_mask);
        let fourth = invalid_lanes(ptr.add(LANES * 3), exponent_mask);
        let invalid = vorrq_u32(vorrq_u32(first, second), vorrq_u32(third, fourth));
        if vmaxvq_u32(invalid) != 0 {
            return scalar::first_non_finite(&values[index..index + UNROLLED_LANES])
                .map(|offset| index + offset);
        }
        index += UNROLLED_LANES;
    }

    while index + LANES <= values.len() {
        if vmaxvq_u32(invalid_lanes(values.as_ptr().add(index), exponent_mask)) != 0 {
            return scalar::first_non_finite(&values[index..index + LANES])
                .map(|offset| index + offset);
        }
        index += LANES;
    }

    scalar::first_non_finite(&values[index..]).map(|offset| index + offset)
}

#[cfg(not(feature = "f32"))]
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::aarch64::{vaddq_f64, vdivq_f64, vdupq_n_f64, vld1q_f64, vst1q_f64};

    const LANES: usize = 2;
    let denominator = vdupq_n_f64(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = vld1q_f64(high.as_ptr().add(index));
        let low_values = vld1q_f64(low.as_ptr().add(index));
        let close_values = vld1q_f64(close.as_ptr().add(index));
        let values = vdivq_f64(
            vaddq_f64(vaddq_f64(high_values, low_values), close_values),
            denominator,
        );
        vst1q_f64(output.as_mut_ptr().add(index), values);
        index += LANES;
    }

    while index < high.len() {
        output[index] = (high[index] + low[index] + close[index]) / 3.0;
        index += 1;
    }
}

#[cfg(feature = "f32")]
#[inline(never)]
#[target_feature(enable = "neon")]
pub unsafe fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    use core::arch::aarch64::{vaddq_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vst1q_f32};

    const LANES: usize = 4;
    let denominator = vdupq_n_f32(3.0);
    let mut index = 0;
    while index + LANES <= high.len() {
        let high_values = vld1q_f32(high.as_ptr().add(index));
        let low_values = vld1q_f32(low.as_ptr().add(index));
        let close_values = vld1q_f32(close.as_ptr().add(index));
        let values = vdivq_f32(
            vaddq_f32(vaddq_f32(high_values, low_values), close_values),
            denominator,
        );
        vst1q_f32(output.as_mut_ptr().add(index), values);
        index += LANES;
    }

    while index < high.len() {
        output[index] = (high[index] + low[index] + close[index]) / 3.0;
        index += 1;
    }
}
