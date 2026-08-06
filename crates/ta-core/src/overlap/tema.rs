//! Triple Exponential Moving Average (TEMA).

use crate::common::validate_finite_value;
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
fn tema_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(3)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "TEMA lookback would overflow"))
}

fn validate_tema_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = tema_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

pub(super) fn tema_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut stream = TEMAStream::new(timeperiod)?;
    let mut output_idx = 0usize;

    for &value in real {
        let Some(output) = stream.next_validated(value)? else {
            continue;
        };
        out_real[output_idx] = output;
        output_idx += 1;
    }

    Ok(OutputRange::new(lookback, count))
}

/// TA-Lib-style Triple Exponential Moving Average batch function.
#[allow(non_snake_case)]
pub fn TEMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_tema_input(real, timeperiod)?;
    validate_output_len("TEMA", out_real.len(), count)?;
    tema_kernel(real, timeperiod, lookback, count, out_real)
}

/// Immutable Triple Exponential Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TEMAConfig {
    period: usize,
}

impl TEMAConfig {
    /// Creates a configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        tema_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for TEMAConfig {}

impl IndicatorConfig for TEMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = TEMABatchRunner;
    type Stream = TEMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        (self.period - 1) * 3
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_tema_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = tema_kernel(input, self.period, lookback, count, &mut values)?;
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TEMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(TEMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        TEMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Triple Exponential Moving Average.
#[derive(Debug, Clone)]
pub struct TEMABatchRunner {
    config: TEMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for TEMABatchRunner {}

impl PreparedBatchRunner<TEMAConfig> for TEMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <TEMAConfig as IndicatorConfig>::Input<'a>,
        output: <TEMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        TEMAConfig: 'a,
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

/// Independent Streaming Computation state for Triple Exponential Moving Average.
#[derive(Debug, Clone)]
pub struct TEMAStream {
    ema1: super::ema::EMAStream,
    ema2: super::ema::EMAStream,
    ema3: super::ema::EMAStream,
}

impl TEMAStream {
    fn new(period: usize) -> Result<Self> {
        tema_lookback(period)?;
        Ok(Self {
            ema1: super::ema::EMAStream::new(period)?,
            ema2: super::ema::EMAStream::new(period)?,
            ema3: super::ema::EMAStream::new(period)?,
        })
    }

    fn next_validated(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next_unchecked(input) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema1)?;
        let Some(ema2) = self.ema2.next_unchecked(ema1) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema2)?;
        let Some(ema3) = self.ema3.next_unchecked(ema2) else {
            return Ok(None);
        };
        Ok(Some(3.0 as Float * ema1 - 3.0 as Float * ema2 + ema3))
    }

    fn reset_state(&mut self) {
        self.ema1.reset_state();
        self.ema2.reset_state();
        self.ema3.reset_state();
    }
}

impl crate::traits::sealed::Sealed for TEMAStream {}

impl StreamingComputation<TEMAConfig> for TEMAStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;
        self.next_validated(input)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset_state();
    }
}
