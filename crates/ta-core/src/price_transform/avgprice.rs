//! Average Price (AVGPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`AVGPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct AVGPRICEInput<'a> {
    /// Open price series.
    pub open: &'a [Float],
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One OHLC tick for [`AVGPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AVGPRICETick {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

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

impl Indicator for AVGPRICE {
    type Input<'a> = AVGPRICEInput<'a>;
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
        AVGPRICE(input.open, input.high, input.low, input.close, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        AVGPRICE_vec(input.open, input.high, input.low, input.close)
    }
}

impl StreamingIndicator for AVGPRICE {
    type Tick = AVGPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("open", &[input.open]),
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;
        Ok(Some(
            (input.open + input.high + input.low + input.close) / 4.0 as Float,
        ))
    }
}
