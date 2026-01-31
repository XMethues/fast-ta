use crate::simd::types::{Lanes, SimdVecExt};
use crate::types::Float;
use crate::Result;

#[cfg(all(feature = "f64", not(feature = "f32")))]
#[allow(dead_code)]
type SimdVec = wide::f64x8;

#[cfg(feature = "f32")]
#[allow(dead_code)]
type SimdVec = wide::f32x16;

#[allow(dead_code)]
const AVX512_LANES: usize = Lanes::AVX512;

#[inline(never)]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
pub unsafe fn sum(data: &[Float]) -> Float {
    let chunks = data.chunks_exact(AVX512_LANES);
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

#[inline(never)]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
pub unsafe fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(crate::TalibError::InvalidInput {
            message: "Arrays must have same length".into(),
        });
    }

    let mut sum = Float::from(0.0);
    let chunks = a
        .chunks_exact(AVX512_LANES)
        .zip(b.chunks_exact(AVX512_LANES));
    let remainder_a = a.chunks_exact(AVX512_LANES).remainder();
    let remainder_b = b.chunks_exact(AVX512_LANES).remainder();

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
