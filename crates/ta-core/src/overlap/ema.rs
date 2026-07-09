//! Exponential Moving Average (EMA).
//!
//! This module exposes both the TA-Lib-style zero-copy function [`EMA`] and the
//! stateful [`EMA`] struct. The free function writes compact valid outputs and
//! returns an [`OutputRange`](crate::OutputRange); [`EMA_vec`] returns a
//! full-length padded vector for convenience.

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
pub(super) fn ema_multiplier(timeperiod: usize) -> Float {
    2.0 as Float / (timeperiod as Float + 1.0 as Float)
}

#[inline]
pub(super) fn ema_seed(real: &[Float], timeperiod: usize) -> Float {
    real[..timeperiod].iter().copied().sum::<Float>() / timeperiod as Float
}

#[inline]
pub(super) fn ema_step(previous: Float, input: Float, multiplier: Float) -> Float {
    (input - previous) * multiplier + previous
}

/// TA-Lib-style Exponential Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn EMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("EMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let multiplier = ema_multiplier(timeperiod);
    let mut ema = ema_seed(real, timeperiod);
    out_real[0] = ema;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        ema = ema_step(ema, real[input_idx], multiplier);
        out_real[output_idx] = ema;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes EMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn EMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = EMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    multiplier: Float,
    count: usize,
    sum: Float,
    value: Float,
}

impl EMA {
    /// Creates a new EMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self {
            period: timeperiod,
            multiplier: ema_multiplier(timeperiod),
            count: 0,
            sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    /// Creates a new EMA indicator seeded by processing `real` in order.
    pub fn from_data(timeperiod: usize, real: &[Float]) -> Result<Self> {
        validate_finite_slice("real", real)?;
        let mut ema = Self::new(timeperiod)?;
        for &value in real {
            let _ = ema.next(value)?;
        }
        Ok(ema)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact EMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        EMA(real, self.period, out_real)
    }

    /// Computes full-length padded EMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        EMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for EMA {
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
        EMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        EMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for EMA {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        if self.count < self.period {
            self.sum += input;
            self.count += 1;

            if self.count < self.period {
                return Ok(None);
            }

            self.value = self.sum / self.period as Float;
            return Ok(Some(self.value));
        }

        self.value = ema_step(self.value, input, self.multiplier);
        Ok(Some(self.value))
    }
}

impl Resettable for EMA {
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
