//! Simple Moving Average (SMA).
//!
//! This module exposes both the TA-Lib-style zero-copy function [`SMA`] and the
//! stateful [`SMA`] struct. The free function writes compact valid outputs and
//! returns an [`OutputRange`](crate::OutputRange); [`SMA_vec`] returns a
//! full-length padded vector for convenience.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Simple Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn SMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("SMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let inv_period = 1.0 as Float / timeperiod as Float;
    let mut window_sum: Float = real[..timeperiod].iter().copied().sum();
    out_real[0] = window_sum * inv_period;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = window_sum * inv_period;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes SMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn SMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = SMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Simple Moving Average indicator.
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
    inv_period: Float,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
}

impl SMA {
    /// Creates a new SMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);

        Ok(Self {
            period: timeperiod,
            inv_period: 1.0 as Float / timeperiod as Float,
            buffer,
            index: 0,
            count: 0,
            sum: 0.0 as Float,
        })
    }

    /// Creates a new SMA indicator seeded from the most recent `timeperiod` values.
    pub fn from_data(timeperiod: usize, real: &[Float]) -> Result<Self> {
        validate_finite_slice("real", real)?;
        let mut sma = Self::new(timeperiod)?;
        let start = real.len().saturating_sub(timeperiod);
        for &value in &real[start..] {
            sma.next(value);
        }
        Ok(sma)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact SMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        SMA(real, self.period, out_real)
    }

    /// Computes full-length padded SMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        SMA_vec(real, self.period)
    }

    /// Checked streaming update that rejects non-finite inputs.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        if !input.is_finite() {
            return Err(TalibError::invalid_input("SMA input must be finite"));
        }
        Ok(self.next(input))
    }
}

impl Indicator for SMA {
    type Input = Float;
    type Output = Float;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<OutputRange> {
        SMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        SMA_vec(inputs, self.period)
    }

    #[inline]
    fn next(&mut self, input: Float) -> Float {
        if self.count < self.period {
            self.buffer[self.index] = input;
            self.sum += input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;

            if self.count < self.period {
                return Float::NAN;
            }

            return self.sum * self.inv_period;
        }

        let old = self.buffer[self.index];
        self.buffer[self.index] = input;
        self.sum += input - old;
        self.index = (self.index + 1) % self.period;
        self.sum * self.inv_period
    }
}

impl Resettable for SMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
    }
}
