//! Weighted Moving Average (WMA).
//!
//! Valid batch outputs are written compactly. Padded wrappers preserve input
//! length and fill warm-up positions with `Float::NAN`.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn wma_denominator(timeperiod: usize) -> Float {
    (timeperiod * (timeperiod + 1) / 2) as Float
}

/// TA-Lib-style Weighted Moving Average batch function.
#[allow(non_snake_case)]
pub fn WMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("WMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let denominator = wma_denominator(timeperiod);
    let mut window_sum = real[..timeperiod].iter().copied().sum::<Float>();
    let mut weighted_sum = real[..timeperiod]
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx + 1) as Float * value)
        .sum::<Float>();
    out_real[0] = weighted_sum / denominator;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        weighted_sum = weighted_sum - window_sum + timeperiod as Float * real[new_idx];
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = weighted_sum / denominator;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes WMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn WMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = WMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Weighted Moving Average indicator.
#[derive(Debug, Clone)]
pub struct WMA {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl WMA {
    /// Creates a new WMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);
        Ok(Self {
            period: timeperiod,
            buffer,
            index: 0,
            count: 0,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact WMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        WMA(real, self.period, out_real)
    }

    /// Computes full-length padded WMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        WMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for WMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        WMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        WMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for WMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Ok(None);
        }

        let weighted_sum = (0..self.period)
            .map(|offset| {
                let ordered_idx = (self.index + offset) % self.period;
                (offset + 1) as Float * self.buffer[ordered_idx]
            })
            .sum::<Float>();
        Ok(Some(weighted_sum / wma_denominator(self.period)))
    }
}

impl Resettable for WMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}
