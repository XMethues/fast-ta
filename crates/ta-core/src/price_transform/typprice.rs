//! Typical Price (TYPPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Typical Price batch function: `(high + low + close) / 3`.
#[allow(non_snake_case)]
pub fn TYPPRICE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    validate_output_len("TYPPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx] + close[idx]) / 3.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Typical Price into a full-length vector.
#[allow(non_snake_case)]
pub fn TYPPRICE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = TYPPRICE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Typical Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct TYPPRICE {
    _private: (),
}

impl TYPPRICE {
    /// Creates a Typical Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        TYPPRICE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        TYPPRICE_vec(high, low, close)
    }
}
