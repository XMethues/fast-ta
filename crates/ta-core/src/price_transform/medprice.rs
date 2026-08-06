//! Median Price (MEDPRICE).

use crate::{
    validate_all_same_len, validate_finite_slices, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`MEDPRICE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct MEDPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
}

/// One high/low tick for [`MEDPRICE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MEDPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
}
fn validate_medprice_input(input: MEDPRICEInput<'_>) -> Result<usize> {
    let len = validate_all_same_len(&[("high", input.high.len()), ("low", input.low.len())])?;
    validate_finite_slices(&[("high", input.high), ("low", input.low)])?;
    Ok(len)
}

fn medprice_kernel(input: MEDPRICEInput<'_>, len: usize, output: &mut [Float]) -> OutputRange {
    for idx in 0..len {
        output[idx] = (input.high[idx] + input.low[idx]) / 2.0 as Float;
    }
    OutputRange::new(0, len)
}

/// TA-Lib-style Median Price batch function: `(high + low) / 2`.
#[allow(non_snake_case)]
pub fn MEDPRICE(high: &[Float], low: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let input = MEDPRICEInput { high, low };
    let len = validate_medprice_input(input)?;
    validate_output_len("MEDPRICE", out_real.len(), len)?;
    Ok(medprice_kernel(input, len, out_real))
}

/// Immutable Median Price Indicator Configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MEDPRICEConfig;

impl MEDPRICEConfig {
    /// Creates the parameter-free Median Price configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for MEDPRICEConfig {}

impl IndicatorConfig for MEDPRICEConfig {
    type Input<'a> = MEDPRICEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MEDPRICEBatchRunner;
    type Stream = MEDPRICEStream;

    #[inline]
    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let len = validate_medprice_input(input)?;
        let mut values = Vec::with_capacity(len);
        values.resize(len, 0.0 as Float);
        let range = medprice_kernel(input, len, &mut values);
        CompactOutput::new(len, range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let len = validate_medprice_input(input)?;
        validate_output_len("MEDPRICE", output.len(), len)?;
        Ok(medprice_kernel(input, len, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(MEDPRICEBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(MEDPRICEStream)
    }
}

/// Prepared Batch Runner for Median Price.
#[derive(Debug, Clone)]
pub struct MEDPRICEBatchRunner {
    config: MEDPRICEConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for MEDPRICEBatchRunner {}

impl PreparedBatchRunner<MEDPRICEConfig> for MEDPRICEBatchRunner {
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MEDPRICEConfig as IndicatorConfig>::Input<'a>,
        output: <MEDPRICEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MEDPRICEConfig: 'a,
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

/// Independent Streaming Computation for Median Price.
#[derive(Debug, Clone, Copy, Default)]
pub struct MEDPRICEStream;

impl crate::traits::sealed::Sealed for MEDPRICEStream {}

impl StreamingComputation<MEDPRICEConfig> for MEDPRICEStream {
    type Tick = MEDPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[("high", &[input.high]), ("low", &[input.low])])?;
        Ok(Some((input.high + input.low) / 2.0 as Float))
    }

    fn reset(&mut self) {}
}
