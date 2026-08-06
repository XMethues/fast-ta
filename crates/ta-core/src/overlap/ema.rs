//! Exponential Moving Average (EMA).
//!
//! The uppercase [`EMA`] function is the compact batch kernel behind
//! `compute_into`; [`EMAConfig`] is the Rust-first immutable Indicator
//! Configuration with owned, caller-owned, prepared, and streaming execution.

use crate::{
    period_lookback, validate_finite_slice, validate_input_len, validate_output_len, CompactOutput,
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation,
    TalibError,
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

fn validate_ema_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[inline]
pub(super) fn ema_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let multiplier = ema_multiplier(timeperiod);
    let mut ema = ema_seed(real, timeperiod);
    out_real[0] = ema;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        ema = ema_step(ema, real[input_idx], multiplier);
        out_real[output_idx] = ema;
    }

    OutputRange::new(lookback, count)
}

/// TA-Lib-style Exponential Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn EMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_ema_input(real, timeperiod)?;
    validate_output_len("EMA", out_real.len(), count)?;
    Ok(ema_kernel(real, timeperiod, lookback, count, out_real))
}

/// Immutable Exponential Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EMAConfig {
    period: usize,
}

impl EMAConfig {
    /// Creates a configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for EMAConfig {}

impl IndicatorConfig for EMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = EMABatchRunner;
    type Stream = EMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_ema_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = ema_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        EMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(EMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        EMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Exponential Moving Average.
///
/// EMA needs no heap scratch, so preparation stores only the configuration and
/// declared source capacity.
#[derive(Debug, Clone)]
pub struct EMABatchRunner {
    config: EMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for EMABatchRunner {}

impl PreparedBatchRunner<EMAConfig> for EMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <EMAConfig as IndicatorConfig>::Input<'a>,
        output: <EMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        EMAConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for Exponential Moving Average.
#[derive(Debug, Clone)]
pub struct EMAStream {
    period: usize,
    multiplier: Float,
    count: usize,
    sum: Float,
    value: Float,
}

impl EMAStream {
    pub(super) fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        Ok(Self {
            period,
            multiplier: ema_multiplier(period),
            count: 0,
            sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    #[inline]
    pub(super) fn next_unchecked(&mut self, input: Float) -> Option<Float> {
        if self.count < self.period {
            self.sum += input;
            self.count += 1;

            if self.count < self.period {
                return None;
            }

            self.value = self.sum / self.period as Float;
            return Some(self.value);
        }

        self.value = ema_step(self.value, input, self.multiplier);
        Some(self.value)
    }

    pub(super) fn reset_state(&mut self) {
        self.count = 0;
        self.sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}

impl crate::traits::sealed::Sealed for EMAStream {}

impl StreamingComputation<EMAConfig> for EMAStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;
        Ok(self.next_unchecked(input))
    }

    #[inline]
    fn reset(&mut self) {
        self.reset_state();
    }
}
