//! Weighted Close Price (WCLPRICE).

use crate::{
    validate_all_same_len, validate_finite_slices, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`WCLPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct WCLPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`WCLPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WCLPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}
fn validate_wclprice_input(input: WCLPRICEInput<'_>) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", input.high.len()),
        ("low", input.low.len()),
        ("close", input.close.len()),
    ])?;
    validate_finite_slices(&[
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    Ok(len)
}

fn wclprice_kernel(input: WCLPRICEInput<'_>, len: usize, output: &mut [Float]) -> OutputRange {
    for idx in 0..len {
        output[idx] =
            (input.high[idx] + input.low[idx] + 2.0 as Float * input.close[idx]) / 4.0 as Float;
    }
    OutputRange::new(0, len)
}

/// TA-Lib-style Weighted Close Price batch function: `(high + low + 2 * close) / 4`.
#[allow(non_snake_case)]
pub fn WCLPRICE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = WCLPRICEInput { high, low, close };
    let len = validate_wclprice_input(input)?;
    validate_output_len("WCLPRICE", out_real.len(), len)?;
    Ok(wclprice_kernel(input, len, out_real))
}

/// Immutable Weighted Close Price Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct WCLPRICEConfig;

impl WCLPRICEConfig {
    /// Creates the parameter-free Weighted Close Price configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for WCLPRICEConfig {}

impl IndicatorConfig for WCLPRICEConfig {
    type Input<'a> = WCLPRICEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = WCLPRICEBatchRunner;
    type Stream = WCLPRICEStream;

    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let len = validate_wclprice_input(input)?;
        let mut values = Vec::with_capacity(len);
        values.resize(len, 0.0 as Float);
        let range = wclprice_kernel(input, len, &mut values);
        CompactOutput::new(len, range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let len = validate_wclprice_input(input)?;
        validate_output_len("WCLPRICE", output.len(), len)?;
        Ok(wclprice_kernel(input, len, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(WCLPRICEBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(WCLPRICEStream)
    }
}

/// Prepared Batch Runner for Weighted Close Price.
#[derive(Debug, Clone)]
pub struct WCLPRICEBatchRunner {
    config: WCLPRICEConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for WCLPRICEBatchRunner {}

impl PreparedBatchRunner<WCLPRICEConfig> for WCLPRICEBatchRunner {
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <WCLPRICEConfig as IndicatorConfig>::Input<'a>,
        output: <WCLPRICEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        WCLPRICEConfig: 'a,
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

/// Independent Streaming Computation for Weighted Close Price.
#[derive(Debug, Clone, Copy, Default)]
pub struct WCLPRICEStream;

impl crate::traits::sealed::Sealed for WCLPRICEStream {}

impl StreamingComputation<WCLPRICEConfig> for WCLPRICEStream {
    type Tick = WCLPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;
        Ok(Some(
            (input.high + input.low + 2.0 as Float * input.close) / 4.0 as Float,
        ))
    }

    fn reset(&mut self) {}
}
