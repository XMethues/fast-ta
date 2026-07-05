//! Weighted Close Price (WCLPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Weighted Close Price batch function: `(high + low + 2 * close) / 4`.
#[allow(non_snake_case)]
pub fn WCLPRICE(
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
    validate_output_len("WCLPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx] + 2.0 as Float * close[idx]) / 4.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Weighted Close Price into a full-length vector.
#[allow(non_snake_case)]
pub fn WCLPRICE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = WCLPRICE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Weighted Close Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct WCLPRICE {
    _private: (),
}

impl WCLPRICE {
    /// Creates a Weighted Close Price calculator.
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
        WCLPRICE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        WCLPRICE_vec(high, low, close)
    }
}
