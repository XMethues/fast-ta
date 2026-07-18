//! Pearson's Correlation Coefficient (CORREL).

use super::{
    moments::{statistic_lookback, RollingPairedMoments},
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

/// TA-Lib-style Pearson's Correlation Coefficient batch function.
#[allow(non_snake_case)]
pub fn CORREL(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 1, 0)?;
    let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
    validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
    let count = validate_input_len(len, lookback)?;
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

/// Computes Pearson's Correlation Coefficient into a full-length padded vector.
#[allow(non_snake_case)]
pub fn CORREL_vec(real0: &[Float], real1: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real0.len());
    let range = CORREL(real0, real1, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real0.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Pearson's Correlation Coefficient indicator.
#[derive(Debug, Clone)]
pub struct CORREL {
    period: usize,
    moments: RollingPairedMoments,
}

impl CORREL {
    /// Creates a new Pearson's Correlation Coefficient indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 0)?;
        Ok(Self {
            period: timeperiod,
            moments: RollingPairedMoments::new(timeperiod),
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
        CORREL(real0, real1, self.period, out_real)
    }

    /// Computes full-length padded outputs using this indicator's period.
    pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
        CORREL_vec(real0, real1, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: PairTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for CORREL {
    type Input<'a> = PairInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        CORREL(input.real0, input.real1, self.period, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        CORREL_vec(input.real0, input.real1, self.period)
    }
}

impl StreamingIndicator for CORREL {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_slices(&[("real0", &[input.real0]), ("real1", &[input.real1])])?;
        Ok(self
            .moments
            .push(input.real0, input.real1)
            .map(|snapshot| snapshot.correlation()))
    }
}

impl Resettable for CORREL {
    fn reset(&mut self) {
        self.moments.reset();
    }
}
