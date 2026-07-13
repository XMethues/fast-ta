//! Chaikin Accumulation/Distribution Line (AD).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed high/low/close/volume inputs for [`AD`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct ADInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One high/low/close/volume tick for [`AD`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ADTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

/// Validates high/low/close/volume slices and returns their shared length.
pub(super) fn validate_hlcv(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
        ("volume", volume.len()),
    ])?;
    validate_finite_slices(&[
        ("high", high),
        ("low", low),
        ("close", close),
        ("volume", volume),
    ])?;
    Ok(len)
}

/// Computes one money-flow-volume contribution.
#[inline]
pub(super) fn money_flow_volume(high: Float, low: Float, close: Float, volume: Float) -> Float {
    let range = high - low;
    if range <= 0.0 as Float {
        0.0 as Float
    } else {
        (((close - low) - (high - close)) / range) * volume
    }
}

/// TA-Lib-style Chaikin Accumulation/Distribution Line batch function.
#[allow(non_snake_case)]
pub fn AD(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_hlcv(high, low, close, volume)?;
    validate_output_len("AD", out_real.len(), len)?;

    let mut cumulative = 0.0 as Float;
    for idx in 0..len {
        cumulative += money_flow_volume(high[idx], low[idx], close[idx], volume[idx]);
        out_real[idx] = cumulative;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes AD into a full-length vector.
#[allow(non_snake_case)]
pub fn AD_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = AD(high, low, close, volume, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Chaikin Accumulation/Distribution Line indicator.
#[derive(Debug, Clone, Default)]
pub struct AD {
    cumulative: Float,
}

impl AD {
    /// Creates a new AD indicator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            cumulative: 0.0 as Float,
        })
    }

    /// Computes compact AD outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        AD(high, low, close, volume, out_real)
    }

    /// Computes full-length AD outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
    ) -> Result<Vec<Float>> {
        AD_vec(high, low, close, volume)
    }

    /// Checked streaming update.
    pub fn next_checked(&mut self, input: ADTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for AD {
    type Input<'a> = ADInput<'a>;
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
        AD(input.high, input.low, input.close, input.volume, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        AD_vec(input.high, input.low, input.close, input.volume)
    }
}

impl StreamingIndicator for AD {
    type Tick = ADTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
            ("volume", &[input.volume]),
        ])?;
        self.cumulative += money_flow_volume(input.high, input.low, input.close, input.volume);
        Ok(Some(self.cumulative))
    }
}

impl Resettable for AD {
    fn reset(&mut self) {
        self.cumulative = 0.0 as Float;
    }
}
