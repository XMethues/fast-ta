//! Median Price (MEDPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`MEDPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct MEDPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
}

/// One high/low tick for [`MEDPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MEDPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
}

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

impl Indicator for MEDPRICE {
    type Input<'a> = MEDPRICEInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MEDPRICE(input.high, input.low, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MEDPRICE_vec(input.high, input.low)
    }
}

impl StreamingIndicator for MEDPRICE {
    type Tick = MEDPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[("high", &[input.high]), ("low", &[input.low])])?;
        Ok(Some((input.high + input.low) / 2.0 as Float))
    }
}
