//! Average Price (AVGPRICE).

use crate::common::validate_finite_value;
use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, CompactOutput, Float, Indicator, IndicatorConfig, OutputRange,
    PreparedBatchRunner, Result, StreamingComputation, StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`AVGPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct AVGPRICEInput<'a> {
    /// Open price series.
    pub open: &'a [Float],
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One OHLC tick for [`AVGPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AVGPRICETick {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}
fn validate_avgprice_input(input: AVGPRICEInput<'_>) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("open", input.open.len()),
        ("high", input.high.len()),
        ("low", input.low.len()),
        ("close", input.close.len()),
    ])?;
    validate_finite_slices(&[
        ("open", input.open),
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    Ok(len)
}
#[inline]
fn avgprice_tick(input: AVGPRICETick) -> Result<Float> {
    validate_finite_value("open", 0, input.open)?;
    validate_finite_value("high", 0, input.high)?;
    validate_finite_value("low", 0, input.low)?;
    validate_finite_value("close", 0, input.close)?;
    Ok((input.open + input.high + input.low + input.close) / 4.0 as Float)
}

fn avgprice_kernel(input: AVGPRICEInput<'_>, len: usize, output: &mut [Float]) -> OutputRange {
    for idx in 0..len {
        output[idx] =
            (input.open[idx] + input.high[idx] + input.low[idx] + input.close[idx]) / 4.0 as Float;
    }
    OutputRange::new(0, len)
}

/// TA-Lib-style Average Price batch function: `(open + high + low + close) / 4`.
#[allow(non_snake_case)]
pub fn AVGPRICE(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = AVGPRICEInput {
        open,
        high,
        low,
        close,
    };
    let len = validate_avgprice_input(input)?;
    validate_output_len("AVGPRICE", out_real.len(), len)?;
    Ok(avgprice_kernel(input, len, out_real))
}

/// Computes Average Price into a full-length vector.
#[allow(non_snake_case)]
pub fn AVGPRICE_vec(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(open.len());
    let range = AVGPRICE(open, high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        open.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Immutable Average Price Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AVGPRICEConfig;

impl AVGPRICEConfig {
    /// Creates the parameter-free Average Price configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for AVGPRICEConfig {}

impl IndicatorConfig for AVGPRICEConfig {
    type Input<'a> = AVGPRICEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = AVGPRICEBatchRunner;
    type Stream = AVGPRICEStream;

    #[inline]
    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let len = validate_avgprice_input(input)?;
        let mut values = Vec::with_capacity(len);
        values.resize(len, 0.0 as Float);
        let range = avgprice_kernel(input, len, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let len = validate_avgprice_input(input)?;
        validate_output_len("AVGPRICE", output.len(), len)?;
        Ok(avgprice_kernel(input, len, output))
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(AVGPRICEBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(AVGPRICEStream)
    }
}

/// Prepared Batch Runner for Average Price.
#[derive(Debug, Clone)]
pub struct AVGPRICEBatchRunner {
    config: AVGPRICEConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for AVGPRICEBatchRunner {}

impl PreparedBatchRunner<AVGPRICEConfig> for AVGPRICEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <AVGPRICEConfig as IndicatorConfig>::Input<'a>,
        output: <AVGPRICEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        AVGPRICEConfig: 'a,
    {
        if input.open.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.open.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for Average Price.
#[derive(Debug, Clone, Copy, Default)]
pub struct AVGPRICEStream;

impl crate::traits::sealed::Sealed for AVGPRICEStream {}

impl StreamingComputation<AVGPRICEConfig> for AVGPRICEStream {
    type Tick = AVGPRICETick;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        avgprice_tick(input).map(Some)
    }

    #[inline]
    fn reset(&mut self) {}
}

/// Average Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct AVGPRICE {
    _private: (),
}

impl AVGPRICE {
    /// Creates an Average Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        AVGPRICE(open, high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        AVGPRICE_vec(open, high, low, close)
    }
}

impl Indicator for AVGPRICE {
    type Input<'a> = AVGPRICEInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        AVGPRICE(input.open, input.high, input.low, input.close, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        AVGPRICE_vec(input.open, input.high, input.low, input.close)
    }
}

impl StreamingIndicator for AVGPRICE {
    type Tick = AVGPRICETick;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        avgprice_tick(input).map(Some)
    }
}
