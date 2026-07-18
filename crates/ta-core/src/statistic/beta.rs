//! Beta Coefficient (BETA).

use super::{
    moments::{is_ta_zero, statistic_lookback, RollingPairedMoments},
    PairInput, PairTick,
};
use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
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
        let Some(previous) = self.previous.replace(input) else {
            return None;
        };
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

/// TA-Lib-style Beta Coefficient batch function.
#[allow(non_snake_case)]
pub fn BETA(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 1, 1)?;
    let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
    validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
    let count = validate_input_len(len, lookback)?;
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

/// Computes Beta Coefficient into a full-length padded vector.
#[allow(non_snake_case)]
pub fn BETA_vec(real0: &[Float], real1: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real0.len());
    let range = BETA(real0, real1, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real0.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Beta Coefficient indicator.
#[derive(Debug, Clone)]
pub struct BETA {
    period: usize,
    state: BetaState,
}

impl BETA {
    /// Creates a new Beta Coefficient indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 1)?;
        Ok(Self {
            period: timeperiod,
            state: BetaState::new(timeperiod),
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs using this indicator's period.
    pub fn compute(
        &self,
        real0: &[Float],
        real1: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        BETA(real0, real1, self.period, out_real)
    }

    /// Computes full-length padded outputs using this indicator's period.
    pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
        BETA_vec(real0, real1, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: PairTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for BETA {
    type Input<'a> = PairInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        BETA(input.real0, input.real1, self.period, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        BETA_vec(input.real0, input.real1, self.period)
    }
}

impl StreamingIndicator for BETA {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_slices(&[("real0", &[input.real0]), ("real1", &[input.real1])])?;
        Ok(self.state.push(input))
    }
}

impl Resettable for BETA {
    fn reset(&mut self) {
        self.state.reset();
    }
}
