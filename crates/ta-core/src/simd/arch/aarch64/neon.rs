//! ARM NEON SIMD implementation for AArch64

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
    let mut sum = Float::from(0.0);
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    Ok(sum)
}
