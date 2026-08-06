//! Beta Coefficient (BETA).

use super::{
    moments::{is_ta_zero, statistic_lookback, RollingPairedMoments},
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

#[inline]
fn price_return(current: Float, previous: Float) -> Float {
    if is_ta_zero(previous) {
        0.0 as Float
    } else {
        (current - previous) / previous
    }
}

#[derive(Debug, Clone)]
struct BetaState {
    previous: Option<PairTick>,
    moments: RollingPairedMoments,
}

impl BetaState {
    fn new(period: usize) -> Self {
        Self {
            previous: None,
            moments: RollingPairedMoments::new(period),
        }
    }

    fn push(&mut self, input: PairTick) -> Option<Float> {
        let previous = self.previous.replace(input)?;
        let real0_return = price_return(input.real0, previous.real0);
        let real1_return = price_return(input.real1, previous.real1);
        self.moments
            .push(real0_return, real1_return)
            .map(|snapshot| snapshot.beta())
    }

    fn reset(&mut self) {
        self.previous = None;
        self.moments.reset();
    }
}

fn validate_beta_input(input: PairInput<'_>, timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = statistic_lookback(timeperiod, 1, 1)?;
    let len = validate_all_same_len(&[("real0", input.real0.len()), ("real1", input.real1.len())])?;
    validate_finite_slices(&[("real0", input.real0), ("real1", input.real1)])?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count))
}

fn beta_kernel(
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
    for source_idx in 1..=timeperiod {
        let x = price_return(input.real0[source_idx], input.real0[source_idx - 1]);
        let y = price_return(input.real1[source_idx], input.real1[source_idx - 1]);
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
        .beta();
        if output_idx + 1 < count {
            let old_idx = output_idx + 1;
            let old_x = price_return(input.real0[old_idx], input.real0[old_idx - 1]);
            let old_y = price_return(input.real1[old_idx], input.real1[old_idx - 1]);
            sum_x -= old_x;
            sum_y -= old_y;
            sum_x_sq -= old_x * old_x;
            sum_y_sq -= old_y * old_y;
            sum_xy -= old_x * old_y;

            let new_idx = output_idx + timeperiod + 1;
            let new_x = price_return(input.real0[new_idx], input.real0[new_idx - 1]);
            let new_y = price_return(input.real1[new_idx], input.real1[new_idx - 1]);
            sum_x += new_x;
            sum_y += new_y;
            sum_x_sq += new_x * new_x;
            sum_y_sq += new_y * new_y;
            sum_xy += new_x * new_y;
        }
    }
    OutputRange::new(lookback, count)
}

/// TA-Lib-style Beta Coefficient batch function.
#[allow(non_snake_case)]
pub fn BETA(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = PairInput { real0, real1 };
    let (lookback, count) = validate_beta_input(input, timeperiod)?;
    validate_output_len("BETA", out_real.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut state = BetaState::new(timeperiod);
    let mut output_idx = 0usize;
    for (&real0, &real1) in real0.iter().zip(real1) {
        if let Some(value) = state.push(PairTick { real0, real1 }) {
            out_real[output_idx] = value;
            output_idx += 1;
        }
    }
    Ok(OutputRange::new(lookback, count))
}

/// Immutable Beta Coefficient Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BETAConfig {
    period: usize,
}

impl BETAConfig {
    /// Creates a configuration for `timeperiod` return observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 1)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for BETAConfig {}

impl IndicatorConfig for BETAConfig {
    type Input<'a> = PairInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = BETABatchRunner;
    type Stream = BETAStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_beta_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = beta_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.real0.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_beta_input(input, self.period)?;
        validate_output_len("BETA", output.len(), count)?;
        Ok(beta_kernel(input, self.period, lookback, count, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(BETABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(BETAStream {
            state: BetaState::new(self.period),
        })
    }
}

/// Reusable Prepared Batch Runner for BETA.
#[derive(Debug, Clone)]
pub struct BETABatchRunner {
    config: BETAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for BETABatchRunner {}

impl PreparedBatchRunner<BETAConfig> for BETABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <BETAConfig as IndicatorConfig>::Input<'a>,
        output: <BETAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        BETAConfig: 'a,
    {
        if input.real0.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.real0.len(),
            ));
        }
        let (lookback, count) = validate_beta_input(input, self.config.period)?;
        validate_output_len("BETA", output.len(), count)?;
        Ok(beta_kernel(
            input,
            self.config.period,
            lookback,
            count,
            output,
        ))
    }
}

/// Independent Streaming Computation for BETA.
#[derive(Debug, Clone)]
pub struct BETAStream {
    state: BetaState,
}

impl crate::traits::sealed::Sealed for BETAStream {}

impl StreamingComputation<BETAConfig> for BETAStream {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_value("real0", 0, input.real0)?;
        validate_finite_value("real1", 0, input.real1)?;
        Ok(self.state.push(input))
    }

    fn reset(&mut self) {
        self.state.reset();
    }
}
