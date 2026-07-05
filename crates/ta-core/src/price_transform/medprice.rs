//! Median Price (MEDPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Median Price batch function: `(high + low) / 2`.
#[allow(non_snake_case)]
pub fn MEDPRICE(high: &[Float], low: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let len = validate_all_same_len(&[("high", high.len()), ("low", low.len())])?;
    validate_finite_slices(&[("high", high), ("low", low)])?;
    validate_output_len("MEDPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx]) / 2.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Median Price into a full-length vector.
#[allow(non_snake_case)]
pub fn MEDPRICE_vec(high: &[Float], low: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = MEDPRICE(high, low, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Median Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct MEDPRICE {
    _private: (),
}

impl MEDPRICE {
    /// Creates a Median Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        MEDPRICE(high, low, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, high: &[Float], low: &[Float]) -> Result<Vec<Float>> {
        MEDPRICE_vec(high, low)
    }
}
