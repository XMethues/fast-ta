//! Average Price (AVGPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Average Price batch function: `(open + high + low + close) / 4`.
#[allow(non_snake_case)]
pub fn AVGPRICE(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_all_same_len(&[
        ("open", open.len()),
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[
        ("open", open),
        ("high", high),
        ("low", low),
        ("close", close),
    ])?;
    validate_output_len("AVGPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (open[idx] + high[idx] + low[idx] + close[idx]) / 4.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Average Price into a full-length vector.
#[allow(non_snake_case)]
pub fn AVGPRICE_vec(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(open.len());
    let range = AVGPRICE(open, high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        open.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Average Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct AVGPRICE {
    _private: (),
}

impl AVGPRICE {
    /// Creates an Average Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        AVGPRICE(open, high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        AVGPRICE_vec(open, high, low, close)
    }
}
