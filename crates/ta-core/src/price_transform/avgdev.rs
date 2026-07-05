//! Average Deviation (AVGDEV).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Average Deviation batch function.
#[allow(non_snake_case)]
pub fn AVGDEV(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("AVGDEV", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let period = timeperiod as Float;
    for output_idx in 0..count {
        let window = &real[output_idx..output_idx + timeperiod];
        let mean = window.iter().copied().sum::<Float>() / period;
        let deviation = window
            .iter()
            .map(|value| (*value - mean).abs())
            .sum::<Float>()
            / period;
        out_real[output_idx] = deviation;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Average Deviation into a full-length vector.
#[allow(non_snake_case)]
pub fn AVGDEV_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = AVGDEV(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Average Deviation indicator.
#[derive(Debug, Clone)]
pub struct AVGDEV {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl AVGDEV {
    /// Creates a new Average Deviation indicator.
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
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs.
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        AVGDEV(real, self.period, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        AVGDEV_vec(real, self.period)
    }
}

impl Indicator for AVGDEV {
    type Input = Float;
    type Output = Float;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<OutputRange> {
        AVGDEV(inputs, self.period, outputs)
    }

    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        AVGDEV_vec(inputs, self.period)
    }

    fn next(&mut self, input: Float) -> Float {
        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Float::NAN;
        }

        let mean = self.buffer.iter().copied().sum::<Float>() / self.period as Float;
        self.buffer
            .iter()
            .map(|value| (*value - mean).abs())
            .sum::<Float>()
            / self.period as Float
    }
}

impl Resettable for AVGDEV {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}
