//! Weighted Moving Average (WMA).
//!
//! Valid batch outputs are written compactly.

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
fn wma_denominator(timeperiod: usize) -> Float {
    (timeperiod * (timeperiod + 1) / 2) as Float
}

fn validate_wma_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

pub(super) fn wma_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
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

    for (output_idx, output_value) in out_real.iter_mut().enumerate().take(count).skip(1) {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        weighted_sum = weighted_sum - window_sum + timeperiod as Float * real[new_idx];
        window_sum += real[new_idx] - real[old_idx];
        *output_value = weighted_sum / denominator;
    }

    OutputRange::new(lookback, count)
}

/// TA-Lib-style Weighted Moving Average batch function.
#[allow(non_snake_case)]
pub fn WMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_wma_input(real, timeperiod)?;
    validate_output_len("WMA", out_real.len(), count)?;
    Ok(wma_kernel(real, timeperiod, lookback, count, out_real))
}

/// Immutable Weighted Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WMAConfig {
    period: usize,
}

impl WMAConfig {
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

impl crate::traits::sealed::Sealed for WMAConfig {}

impl IndicatorConfig for WMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = WMABatchRunner;
    type Stream = WMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_wma_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = wma_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        WMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(WMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        WMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Weighted Moving Average.
///
/// WMA needs no heap scratch, so preparation stores only the configuration and
/// declared source capacity.
#[derive(Debug, Clone)]
pub struct WMABatchRunner {
    config: WMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for WMABatchRunner {}

impl PreparedBatchRunner<WMAConfig> for WMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <WMAConfig as IndicatorConfig>::Input<'a>,
        output: <WMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        WMAConfig: 'a,
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

/// Independent Streaming Computation state for Weighted Moving Average.
#[derive(Debug, Clone)]
pub struct WMAStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl WMAStream {
    fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Ok(Self {
            period,
            buffer,
            index: 0,
            count: 0,
        })
    }
}

impl crate::traits::sealed::Sealed for WMAStream {}

impl StreamingComputation<WMAConfig> for WMAStream {
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

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}
