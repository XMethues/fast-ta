//! Simple Moving Average (SMA).
//!
//! [`SMAConfig`] is the Rust-first immutable Indicator Configuration. It can
//! produce owned Compact Output, write caller-owned output, create a reusable
//! [`SMABatchRunner`], or create an independent [`SMAStream`]. The uppercase
//! [`SMA`] function is the compact batch kernel behind `compute_into`.

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
fn validate_sma_batch(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[inline]
pub(super) fn sma_kernel(
    real: &[Float],
    timeperiod: usize,
    inv_period: Float,
    count: usize,
    out_real: &mut [Float],
) {
    if count == 0 {
        return;
    }

    let mut window_sum: Float = real[..timeperiod].iter().copied().sum();
    out_real[0] = window_sum * inv_period;

    // Preserve the historical indexed loop: this shape is measurably faster on
    // the representative Apple Silicon baseline than the iterator rewrite.
    #[allow(clippy::needless_range_loop)]
    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = window_sum * inv_period;
    }
}

/// TA-Lib-style Simple Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn SMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_sma_batch(real, timeperiod)?;
    validate_output_len("SMA", out_real.len(), count)?;

    sma_kernel(
        real,
        timeperiod,
        1.0 as Float / timeperiod as Float,
        count,
        out_real,
    );
    if count == 0 {
        Ok(OutputRange::empty())
    } else {
        Ok(OutputRange::new(lookback, count))
    }
}

/// Immutable Simple Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SMAConfig {
    period: usize,
}

impl SMAConfig {
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

    #[inline]
    fn inv_period(&self) -> Float {
        1.0 as Float / self.period as Float
    }
}

impl crate::traits::sealed::Sealed for SMAConfig {}

impl IndicatorConfig for SMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = SMABatchRunner;
    type Stream = SMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_sma_batch(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        sma_kernel(input, self.period, self.inv_period(), count, &mut values);
        let range = if count == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(lookback, count)
        };
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_sma_batch(input, self.period)?;
        validate_output_len("SMA", output.len(), count)?;
        sma_kernel(input, self.period, self.inv_period(), count, output);
        if count == 0 {
            Ok(OutputRange::empty())
        } else {
            Ok(OutputRange::new(lookback, count))
        }
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(SMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        SMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Simple Moving Average.
///
/// SMA needs no heap scratch, so preparation stores only the configuration and
/// declared source capacity.
#[derive(Debug, Clone)]
pub struct SMABatchRunner {
    config: SMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for SMABatchRunner {}

impl PreparedBatchRunner<SMAConfig> for SMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <SMAConfig as IndicatorConfig>::Input<'a>,
        output: <SMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        SMAConfig: 'a,
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

/// Independent Streaming Computation state for Simple Moving Average.
#[derive(Debug, Clone)]
pub struct SMAStream {
    period: usize,
    inv_period: Float,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
}

impl SMAStream {
    fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Ok(Self {
            period,
            inv_period: 1.0 as Float / period as Float,
            buffer,
            index: 0,
            count: 0,
            sum: 0.0 as Float,
        })
    }
}

impl crate::traits::sealed::Sealed for SMAStream {}

impl StreamingComputation<SMAConfig> for SMAStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        if self.count < self.period {
            self.buffer[self.index] = input;
            self.sum += input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;

            if self.count < self.period {
                return Ok(None);
            }

            return Ok(Some(self.sum * self.inv_period));
        }

        let old = self.buffer[self.index];
        self.buffer[self.index] = input;
        self.sum += input - old;
        self.index = (self.index + 1) % self.period;
        Ok(Some(self.sum * self.inv_period))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
    }
}
