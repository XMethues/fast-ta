//! Pearson's Correlation Coefficient (CORREL).

use super::{
    moments::{statistic_lookback, RollingPairedMoments},
    PairInput, PairTick,
};
use crate::common::validate_finite_value;
use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

fn validate_correl_input(input: PairInput<'_>, timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = statistic_lookback(timeperiod, 1, 0)?;
    let len = validate_all_same_len(&[("real0", input.real0.len()), ("real1", input.real1.len())])?;
    validate_finite_slices(&[("real0", input.real0), ("real1", input.real1)])?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count))
}

fn correl_kernel(
    input: PairInput<'_>,
    timeperiod: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut sum_x = 0.0 as Float;
    let mut sum_y = 0.0 as Float;
    let mut sum_x_sq = 0.0 as Float;
    let mut sum_y_sq = 0.0 as Float;
    let mut sum_xy = 0.0 as Float;
    for (&x, &y) in input.real0[..timeperiod]
        .iter()
        .zip(&input.real1[..timeperiod])
    {
        sum_x += x;
        sum_y += y;
        sum_x_sq += x * x;
        sum_y_sq += y * y;
        sum_xy += x * y;
    }

    for (output_idx, value) in output.iter_mut().take(count).enumerate() {
        *value = super::moments::PairedSnapshot::new(
            timeperiod, sum_x, sum_y, sum_x_sq, sum_y_sq, sum_xy,
        )
        .correlation();
        if output_idx + 1 < count {
            let old_x = input.real0[output_idx];
            let old_y = input.real1[output_idx];
            sum_x -= old_x;
            sum_y -= old_y;
            sum_x_sq -= old_x * old_x;
            sum_y_sq -= old_y * old_y;
            sum_xy -= old_x * old_y;

            let new_x = input.real0[output_idx + timeperiod];
            let new_y = input.real1[output_idx + timeperiod];
            sum_x += new_x;
            sum_y += new_y;
            sum_x_sq += new_x * new_x;
            sum_y_sq += new_y * new_y;
            sum_xy += new_x * new_y;
        }
    }
    OutputRange::new(lookback, count)
}

/// TA-Lib-style Pearson's Correlation Coefficient batch function.
#[allow(non_snake_case)]
pub fn CORREL(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = PairInput { real0, real1 };
    let (lookback, count) = validate_correl_input(input, timeperiod)?;
    validate_output_len("CORREL", out_real.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut moments = RollingPairedMoments::new(timeperiod);
    let mut output_idx = 0usize;
    for (&real0, &real1) in real0.iter().zip(real1) {
        if let Some(snapshot) = moments.push(real0, real1) {
            out_real[output_idx] = snapshot.correlation();
            output_idx += 1;
        }
    }
    Ok(OutputRange::new(lookback, count))
}

/// Immutable Pearson's Correlation Coefficient Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CORRELConfig {
    period: usize,
}

impl CORRELConfig {
    /// Creates a configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 0)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for CORRELConfig {}

impl IndicatorConfig for CORRELConfig {
    type Input<'a> = PairInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = CORRELBatchRunner;
    type Stream = CORRELStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_correl_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = correl_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.real0.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_correl_input(input, self.period)?;
        validate_output_len("CORREL", output.len(), count)?;
        Ok(correl_kernel(input, self.period, lookback, count, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(CORRELBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(CORRELStream {
            moments: RollingPairedMoments::new(self.period),
        })
    }
}

/// Reusable Prepared Batch Runner for CORREL.
#[derive(Debug, Clone)]
pub struct CORRELBatchRunner {
    config: CORRELConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for CORRELBatchRunner {}

impl PreparedBatchRunner<CORRELConfig> for CORRELBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <CORRELConfig as IndicatorConfig>::Input<'a>,
        output: <CORRELConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        CORRELConfig: 'a,
    {
        if input.real0.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.real0.len(),
            ));
        }
        let (lookback, count) = validate_correl_input(input, self.config.period)?;
        validate_output_len("CORREL", output.len(), count)?;
        Ok(correl_kernel(
            input,
            self.config.period,
            lookback,
            count,
            output,
        ))
    }
}

/// Independent Streaming Computation for CORREL.
#[derive(Debug, Clone)]
pub struct CORRELStream {
    moments: RollingPairedMoments,
}

impl crate::traits::sealed::Sealed for CORRELStream {}

impl StreamingComputation<CORRELConfig> for CORRELStream {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_value("real0", 0, input.real0)?;
        validate_finite_value("real1", 0, input.real1)?;
        Ok(self
            .moments
            .push(input.real0, input.real1)
            .map(|snapshot| snapshot.correlation()))
    }

    fn reset(&mut self) {
        self.moments.reset();
    }
}
