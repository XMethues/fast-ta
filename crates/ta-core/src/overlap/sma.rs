//! Simple Moving Average (SMA).
//!
//! [`SMAConfig`] is the Rust-first immutable Indicator Configuration. It can
//! produce owned Compact Output, write caller-owned output, create a reusable
//! [`SMABatchRunner`], or create an independent [`SMAStream`]. The uppercase
//! [`SMA`] type and functions remain as compatibility interfaces.

use crate::{
    period_lookback, validate_finite_slice, validate_input_len, validate_output_len, CompactOutput,
    Float, Indicator, IndicatorConfig, OutputRange, PreparedBatchRunner, Resettable, Result,
    StreamingComputation, StreamingIndicator, TalibError,
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

/// Computes SMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn SMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;

    // Retain this direct legacy loop (and temporary arithmetic duplication) for the
    // ADR-0001 performance gate until issue #15 removes the legacy path.
    let mut output = Vec::new();
    output.resize(real.len(), Float::NAN);
    if count == 0 {
        return Ok(output);
    }

    let inv_period = 1.0 as Float / timeperiod as Float;
    let mut window_sum: Float = real[..timeperiod].iter().copied().sum();
    output[lookback] = window_sum * inv_period;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        output[lookback + output_idx] = window_sum * inv_period;
    }

    Ok(output)
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

    #[inline]
    const fn period(&self) -> usize {
        self.period
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

/// Legacy Simple Moving Average indicator.
///
/// This compatibility adapter keeps the historical combined batch/streaming
/// signatures while storing only its [`SMAStream`] execution state.
#[derive(Debug, Clone)]
pub struct SMA {
    stream: SMAStream,
}

impl SMA {
    /// Creates a new legacy SMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = SMAConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { stream })
    }

    /// Creates a legacy SMA indicator seeded from the most recent `timeperiod` values.
    pub fn from_data(timeperiod: usize, real: &[Float]) -> Result<Self> {
        validate_finite_slice("real", real)?;
        let mut sma = Self::new(timeperiod)?;
        let start = real.len().saturating_sub(timeperiod);
        for &value in &real[start..] {
            let _ = StreamingIndicator::next(&mut sma, value)?;
        }
        Ok(sma)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.stream.period()
    }

    /// Computes compact SMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        SMA(real, self.period(), out_real)
    }

    /// Computes full-length padded SMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        SMA_vec(real, self.period())
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(StreamingIndicator::next(self, input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for SMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period() - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        SMA(inputs, self.period(), outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        SMA_vec(inputs, self.period())
    }
}

impl StreamingIndicator for SMA {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        StreamingComputation::<SMAConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for SMA {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<SMAConfig>::reset(&mut self.stream);
    }
}
