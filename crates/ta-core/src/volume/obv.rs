//! On-Balance Volume (OBV).

use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed close/volume inputs for [`OBV`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct OBVInput<'a> {
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One close/volume tick for [`OBV`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OBVTick {
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

fn validate_close_volume(close: &[Float], volume: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[("close", close.len()), ("volume", volume.len())])?;
    validate_finite_slices(&[("close", close), ("volume", volume)])?;
    Ok(len)
}
fn validate_obv_input(input: OBVInput<'_>) -> Result<(usize, usize)> {
    let len = validate_close_volume(input.close, input.volume)?;
    let count = validate_input_len(len, 1)?;
    Ok((len, count))
}

fn obv_kernel(input: OBVInput<'_>, len: usize, count: usize, output: &mut [Float]) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut value = 0.0 as Float;
    let mut previous_close = input.close[0];
    for idx in 1..len {
        value = update_obv(value, input.close[idx], previous_close, input.volume[idx]);
        output[idx - 1] = value;
        previous_close = input.close[idx];
    }
    OutputRange::new(1, count)
}

#[inline]
fn update_obv(value: Float, current_close: Float, previous_close: Float, volume: Float) -> Float {
    if current_close > previous_close {
        value + volume
    } else if current_close < previous_close {
        value - volume
    } else {
        value
    }
}

/// On-Balance Volume batch function using first-observation warm-up.
#[allow(non_snake_case)]
pub fn OBV(close: &[Float], volume: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let input = OBVInput { close, volume };
    let (len, count) = validate_obv_input(input)?;
    validate_output_len("OBV", out_real.len(), count)?;
    Ok(obv_kernel(input, len, count, out_real))
}

/// Immutable On-Balance Volume Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OBVConfig;

impl OBVConfig {
    /// Creates an OBV configuration.
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for OBVConfig {}

impl IndicatorConfig for OBVConfig {
    type Input<'a> = OBVInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = OBVBatchRunner;
    type Stream = OBVStream;

    #[inline]
    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, count) = validate_obv_input(input)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = obv_kernel(input, len, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        OBV(input.close, input.volume, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(OBVBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(OBVStream {
            previous_close: None,
            value: 0.0 as Float,
        })
    }
}

/// Reusable Prepared Batch Runner for OBV.
#[derive(Debug, Clone)]
pub struct OBVBatchRunner {
    config: OBVConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for OBVBatchRunner {}

impl PreparedBatchRunner<OBVConfig> for OBVBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <OBVConfig as IndicatorConfig>::Input<'a>,
        output: <OBVConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        OBVConfig: 'a,
    {
        let actual_input_len = input.close.len().max(input.volume.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for OBV.
#[derive(Debug, Clone)]
pub struct OBVStream {
    previous_close: Option<Float>,
    value: Float,
}

impl crate::traits::sealed::Sealed for OBVStream {}

impl StreamingComputation<OBVConfig> for OBVStream {
    type Tick = OBVTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[("close", &[input.close]), ("volume", &[input.volume])])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        self.value = update_obv(self.value, input.close, previous_close, input.volume);
        self.previous_close = Some(input.close);
        Ok(Some(self.value))
    }

    #[inline]
    fn reset(&mut self) {
        self.previous_close = None;
        self.value = 0.0 as Float;
    }
}
