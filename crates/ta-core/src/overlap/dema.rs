//! Double Exponential Moving Average (DEMA).

use crate::common::validate_finite_value;
use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, CompactOutput, Float, Indicator, IndicatorConfig,
    OutputRange, PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn dema_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(2)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "DEMA lookback would overflow"))
}

fn validate_dema_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = dema_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

pub(super) fn dema_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut stream = DEMAStream::new(timeperiod)?;
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

/// TA-Lib-style Double Exponential Moving Average batch function.
#[allow(non_snake_case)]
pub fn DEMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_dema_input(real, timeperiod)?;
    validate_output_len("DEMA", out_real.len(), count)?;
    dema_kernel(real, timeperiod, lookback, count, out_real)
}

/// Computes DEMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn DEMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = DEMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Immutable Double Exponential Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DEMAConfig {
    period: usize,
}

impl DEMAConfig {
    /// Creates a configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        dema_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for DEMAConfig {}

impl IndicatorConfig for DEMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = DEMABatchRunner;
    type Stream = DEMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        (self.period - 1) * 2
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_dema_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = dema_kernel(input, self.period, lookback, count, &mut values)?;
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        DEMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(DEMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        DEMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for Double Exponential Moving Average.
#[derive(Debug, Clone)]
pub struct DEMABatchRunner {
    config: DEMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for DEMABatchRunner {}

impl PreparedBatchRunner<DEMAConfig> for DEMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <DEMAConfig as IndicatorConfig>::Input<'a>,
        output: <DEMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        DEMAConfig: 'a,
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

/// Independent Streaming Computation state for Double Exponential Moving Average.
#[derive(Debug, Clone)]
pub struct DEMAStream {
    period: usize,
    ema1: super::ema::EMAStream,
    ema2: super::ema::EMAStream,
}

impl DEMAStream {
    fn new(period: usize) -> Result<Self> {
        dema_lookback(period)?;
        Ok(Self {
            period,
            ema1: super::ema::EMAStream::new(period)?,
            ema2: super::ema::EMAStream::new(period)?,
        })
    }

    #[inline]
    const fn period(&self) -> usize {
        self.period
    }

    fn next_validated(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next_unchecked(input) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema1)?;
        let Some(ema2) = self.ema2.next_unchecked(ema1) else {
            return Ok(None);
        };
        Ok(Some(2.0 as Float * ema1 - ema2))
    }

    fn reset_state(&mut self) {
        self.ema1.reset_state();
        self.ema2.reset_state();
    }
}

impl crate::traits::sealed::Sealed for DEMAStream {}

impl StreamingComputation<DEMAConfig> for DEMAStream {
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

/// Legacy Double Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct DEMA {
    stream: DEMAStream,
}

impl DEMA {
    /// Creates a new legacy DEMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = DEMAConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.stream.period()
    }

    /// Computes compact DEMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        DEMA(real, self.period(), out_real)
    }

    /// Computes full-length padded DEMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        DEMA_vec(real, self.period())
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(StreamingIndicator::next(self, input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for DEMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        (self.period() - 1) * 2
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        DEMA(inputs, self.period(), outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        DEMA_vec(inputs, self.period())
    }
}

impl StreamingIndicator for DEMA {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        StreamingComputation::<DEMAConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for DEMA {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<DEMAConfig>::reset(&mut self.stream);
    }
}
