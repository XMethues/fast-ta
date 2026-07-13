//! True Range (TRANGE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`TRANGE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct TRANGEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`TRANGE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRANGETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Validates high/low/close slices and returns their shared length.
pub(super) fn validate_hlc(high: &[Float], low: &[Float], close: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    Ok(len)
}

/// Computes one TA-Lib true range value from current high/low and previous close.
#[inline]
pub(super) fn true_range(high: Float, low: Float, previous_close: Float) -> Float {
    let high_low = high - low;
    let high_close = (high - previous_close).abs();
    let low_close = (low - previous_close).abs();
    Float::max(high_low, Float::max(high_close, low_close))
}

/// TA-Lib-style True Range batch function.
#[allow(non_snake_case)]
pub fn TRANGE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_hlc(high, low, close)?;
    let lookback = 1;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("TRANGE", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        let input_idx = output_idx + lookback;
        out_real[output_idx] = true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn TRANGE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = TRANGE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// True Range indicator.
#[derive(Debug, Clone, Default)]
pub struct TRANGE {
    previous_close: Option<Float>,
}

impl TRANGE {
    /// Creates a True Range calculator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            previous_close: None,
        })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        TRANGE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        TRANGE_vec(high, low, close)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: TRANGETick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TRANGE {
    type Input<'a> = TRANGEInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRANGE(input.high, input.low, input.close, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TRANGE_vec(input.high, input.low, input.close)
    }
}

impl StreamingIndicator for TRANGE {
    type Tick = TRANGETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let value = true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);
        Ok(Some(value))
    }
}

impl Resettable for TRANGE {
    fn reset(&mut self) {
        self.previous_close = None;
    }
}
