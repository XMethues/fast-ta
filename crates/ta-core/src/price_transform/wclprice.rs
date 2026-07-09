//! Weighted Close Price (WCLPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`WCLPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct WCLPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`WCLPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WCLPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

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

impl Indicator for WCLPRICE {
    type Input<'a> = WCLPRICEInput<'a>;
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
        WCLPRICE(input.high, input.low, input.close, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        WCLPRICE_vec(input.high, input.low, input.close)
    }
}

impl StreamingIndicator for WCLPRICE {
    type Tick = WCLPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;
        Ok(Some(
            (input.high + input.low + 2.0 as Float * input.close) / 4.0 as Float,
        ))
    }
}
