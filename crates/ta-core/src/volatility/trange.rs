//! True Range (TRANGE).

use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`TRANGE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct TRANGEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`TRANGE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRANGETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Validates high/low/close slices and returns their shared length.
pub(super) fn validate_hlc(high: &[Float], low: &[Float], close: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    Ok(len)
}
fn validate_trange_input(input: TRANGEInput<'_>) -> Result<(usize, usize)> {
    let len = validate_hlc(input.high, input.low, input.close)?;
    let count = validate_input_len(len, 1)?;
    Ok((len, count))
}

fn trange_kernel(input: TRANGEInput<'_>, count: usize, output: &mut [Float]) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    for output_idx in 0..count {
        let input_idx = output_idx + 1;
        output[output_idx] = true_range(
            input.high[input_idx],
            input.low[input_idx],
            input.close[input_idx - 1],
        );
    }
    OutputRange::new(1, count)
}

/// Computes one TA-Lib true range value from current high/low and previous close.
#[inline]
pub(super) fn true_range(high: Float, low: Float, previous_close: Float) -> Float {
    let high_low = high - low;
    let high_close = (high - previous_close).abs();
    let low_close = (low - previous_close).abs();
    Float::max(high_low, Float::max(high_close, low_close))
}

/// TA-Lib-style True Range batch function.
#[allow(non_snake_case)]
pub fn TRANGE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = TRANGEInput { high, low, close };
    let (_, count) = validate_trange_input(input)?;
    validate_output_len("TRANGE", out_real.len(), count)?;
    Ok(trange_kernel(input, count, out_real))
}

/// Immutable True Range Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TRANGEConfig;

impl TRANGEConfig {
    /// Creates a TRANGE configuration.
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for TRANGEConfig {}

impl IndicatorConfig for TRANGEConfig {
    type Input<'a> = TRANGEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = TRANGEBatchRunner;
    type Stream = TRANGEStream;

    #[inline]
    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, count) = validate_trange_input(input)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = trange_kernel(input, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRANGE(input.high, input.low, input.close, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(TRANGEBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(TRANGEStream::default())
    }
}

/// Reusable Prepared Batch Runner for TRANGE.
#[derive(Debug, Clone)]
pub struct TRANGEBatchRunner {
    config: TRANGEConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for TRANGEBatchRunner {}

impl PreparedBatchRunner<TRANGEConfig> for TRANGEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <TRANGEConfig as IndicatorConfig>::Input<'a>,
        output: <TRANGEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        TRANGEConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for TRANGE.
#[derive(Debug, Clone, Default)]
pub struct TRANGEStream {
    previous_close: Option<Float>,
}

impl crate::traits::sealed::Sealed for TRANGEStream {}

impl StreamingComputation<TRANGEConfig> for TRANGEStream {
    type Tick = TRANGETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let value = true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);
        Ok(Some(value))
    }

    #[inline]
    fn reset(&mut self) {
        self.previous_close = None;
    }
}
