//! On-Balance Volume (OBV).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed close/volume inputs for [`OBV`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct OBVInput<'a> {
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One close/volume tick for [`OBV`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OBVTick {
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

fn validate_close_volume(close: &[Float], volume: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[("close", close.len()), ("volume", volume.len())])?;
    validate_finite_slices(&[("close", close), ("volume", volume)])?;
    Ok(len)
}

#[inline]
fn update_obv(value: Float, current_close: Float, previous_close: Float, volume: Float) -> Float {
    if current_close > previous_close {
        value + volume
    } else if current_close < previous_close {
        value - volume
    } else {
        value
    }
}

/// On-Balance Volume batch function using first-observation warm-up.
#[allow(non_snake_case)]
pub fn OBV(close: &[Float], volume: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let len = validate_close_volume(close, volume)?;
    let lookback = 1;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("OBV", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut value = 0.0 as Float;
    for output_idx in 0..count {
        let input_idx = lookback + output_idx;
        value = update_obv(
            value,
            close[input_idx],
            close[input_idx - 1],
            volume[input_idx],
        );
        out_real[output_idx] = value;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes OBV into a full-length vector padded at the warm-up index.
#[allow(non_snake_case)]
pub fn OBV_vec(close: &[Float], volume: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(close.len());
    let range = OBV(close, volume, &mut compact)?;
    Ok(padded_from_compact(
        close.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// On-Balance Volume indicator using first-observation warm-up.
#[derive(Debug, Clone, Default)]
pub struct OBV {
    previous_close: Option<Float>,
    value: Float,
}

impl OBV {
    /// Creates a new OBV indicator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            previous_close: None,
            value: 0.0 as Float,
        })
    }

    /// Computes compact OBV outputs.
    pub fn compute(
        &self,
        close: &[Float],
        volume: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        OBV(close, volume, out_real)
    }

    /// Computes full-length padded OBV outputs.
    pub fn compute_to_vec(&self, close: &[Float], volume: &[Float]) -> Result<Vec<Float>> {
        OBV_vec(close, volume)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: OBVTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for OBV {
    type Input<'a> = OBVInput<'a>;
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
        OBV(input.close, input.volume, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        OBV_vec(input.close, input.volume)
    }
}

impl StreamingIndicator for OBV {
    type Tick = OBVTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[("close", &[input.close]), ("volume", &[input.volume])])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        self.value = update_obv(self.value, input.close, previous_close, input.volume);
        self.previous_close = Some(input.close);
        Ok(Some(self.value))
    }
}

impl Resettable for OBV {
    fn reset(&mut self) {
        self.previous_close = None;
        self.value = 0.0 as Float;
    }
}
