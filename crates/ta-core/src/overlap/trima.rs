//! Triangular Moving Average (TRIMA).

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
fn trima_weight(index: usize, timeperiod: usize) -> usize {
    if timeperiod % 2 == 1 {
        let center = timeperiod / 2;
        if index <= center {
            index + 1
        } else {
            timeperiod - index
        }
    } else {
        let half = timeperiod / 2;
        if index < half {
            index + 1
        } else {
            timeperiod - index
        }
    }
}

#[inline]
fn trima_denominator(timeperiod: usize) -> Float {
    if timeperiod % 2 == 1 {
        let value = timeperiod / 2 + 1;
        (value * value) as Float
    } else {
        let half = timeperiod / 2;
        (half * (half + 1)) as Float
    }
}

fn trima_window(window: &[Float]) -> Float {
    let weighted_sum = window
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| trima_weight(idx, window.len()) as Float * value)
        .sum::<Float>();
    weighted_sum / trima_denominator(window.len())
}

fn validate_trima_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

pub(super) fn trima_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    for output_idx in 0..count {
        out_real[output_idx] = trima_window(&real[output_idx..output_idx + timeperiod]);
    }
    if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(lookback, count)
    }
}

/// TA-Lib-style Triangular Moving Average batch function.
#[allow(non_snake_case)]
pub fn TRIMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_trima_input(real, timeperiod)?;
    validate_output_len("TRIMA", out_real.len(), count)?;
    Ok(trima_kernel(real, timeperiod, lookback, count, out_real))
}

/// Immutable Triangular Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TRIMAConfig {
    period: usize,
}

impl TRIMAConfig {
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

impl crate::traits::sealed::Sealed for TRIMAConfig {}

impl IndicatorConfig for TRIMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = TRIMABatchRunner;
    type Stream = TRIMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_trima_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = trima_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRIMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(TRIMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        TRIMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Triangular Moving Average.
///
/// TRIMA needs no heap scratch, so preparation stores only the configuration
/// and declared source capacity.
#[derive(Debug, Clone)]
pub struct TRIMABatchRunner {
    config: TRIMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for TRIMABatchRunner {}

impl PreparedBatchRunner<TRIMAConfig> for TRIMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <TRIMAConfig as IndicatorConfig>::Input<'a>,
        output: <TRIMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        TRIMAConfig: 'a,
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

/// Independent Streaming Computation state for Triangular Moving Average.
#[derive(Debug, Clone)]
pub struct TRIMAStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl TRIMAStream {
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

impl crate::traits::sealed::Sealed for TRIMAStream {}

impl StreamingComputation<TRIMAConfig> for TRIMAStream {
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
                trima_weight(offset, self.period) as Float * self.buffer[ordered_idx]
            })
            .sum::<Float>();
        Ok(Some(weighted_sum / trima_denominator(self.period)))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}
