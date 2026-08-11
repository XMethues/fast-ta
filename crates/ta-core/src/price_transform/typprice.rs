//! Typical Price (TYPPRICE).

use crate::{
    validate_all_same_len, validate_finite_slices, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`TYPPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct TYPPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`TYPPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TYPPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}
fn validate_typprice_input(input: TYPPRICEInput<'_>) -> Result<usize> {
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

fn typprice_kernel(input: TYPPRICEInput<'_>, len: usize, output: &mut [Float]) -> OutputRange {
    for (idx, output_value) in output.iter_mut().enumerate().take(len) {
        *output_value = (input.high[idx] + input.low[idx] + input.close[idx]) / 3.0 as Float;
    }
    OutputRange::new(0, len)
}

/// TA-Lib-style Typical Price batch function: `(high + low + close) / 3`.
#[allow(non_snake_case)]
pub fn TYPPRICE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = TYPPRICEInput { high, low, close };
    let len = validate_typprice_input(input)?;
    validate_output_len("TYPPRICE", out_real.len(), len)?;
    Ok(typprice_kernel(input, len, out_real))
}

/// Immutable Typical Price Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct TYPPRICEConfig;

impl TYPPRICEConfig {
    /// Creates the parameter-free Typical Price configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for TYPPRICEConfig {}

impl IndicatorConfig for TYPPRICEConfig {
    type Input<'a> = TYPPRICEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = TYPPRICEBatchRunner;
    type Stream = TYPPRICEStream;

    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let len = validate_typprice_input(input)?;
        let mut values = Vec::with_capacity(len);
        values.resize(len, 0.0 as Float);
        let range = typprice_kernel(input, len, &mut values);
        CompactOutput::new(len, range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let len = validate_typprice_input(input)?;
        validate_output_len("TYPPRICE", output.len(), len)?;
        Ok(typprice_kernel(input, len, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(TYPPRICEBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(TYPPRICEStream)
    }
}

/// Prepared Batch Runner for Typical Price.
#[derive(Debug, Clone)]
pub struct TYPPRICEBatchRunner {
    config: TYPPRICEConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for TYPPRICEBatchRunner {}

impl PreparedBatchRunner<TYPPRICEConfig> for TYPPRICEBatchRunner {
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <TYPPRICEConfig as IndicatorConfig>::Input<'a>,
        output: <TYPPRICEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        TYPPRICEConfig: 'a,
    {
        let actual_input_len = input.high.len().max(input.low.len()).max(input.close.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for Typical Price.
#[derive(Debug, Clone, Copy, Default)]
pub struct TYPPRICEStream;

impl crate::traits::sealed::Sealed for TYPPRICEStream {}

impl StreamingComputation<TYPPRICEConfig> for TYPPRICEStream {
    type Tick = TYPPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;
        Ok(Some((input.high + input.low + input.close) / 3.0 as Float))
    }

    fn reset(&mut self) {}
}
