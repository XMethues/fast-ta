//! Typed Hilbert Transform Trend Mode Cycle Indicator Definition.

use super::hilbert::HilbertPhaseState;
use super::to_internal;
use crate::{
    common::validate_finite_value, validate_finite_slice, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::{vec, vec::Vec};

const HISTORY_SIZE: usize = 50;
const DEGREE_TO_RADIAN: f64 = core::f64::consts::PI / 180.0;

/// Fixed Stabilization and Lookback for [`HT_TRENDMODE`], in observations.
///
/// Streaming Computation therefore has a 63-tick Warm-up and produces its
/// first [`TrendMode`] for source position 63.
pub const HT_TRENDMODE_LOOKBACK: usize = 63;

/// A mode classification produced by [`HT_TRENDMODE`].
///
/// This classifies a valid source position as either cycle-dominated or
/// trend-dominated. It is not trend direction, strength, or probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrendMode {
    /// The source position is classified as cycle-dominated.
    Cycle,
    /// The source position is classified as trend-dominated.
    Trend,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HilbertTrendModeState {
    phase: HilbertPhaseState,
    raw_price: [f64; HISTORY_SIZE],
    raw_price_index: usize,
    trend_average_1: f64,
    trend_average_2: f64,
    trend_average_3: f64,
    days_in_trend: usize,
    previous_phase: f64,
    sine: f64,
    lead_sine: f64,
}

impl Default for HilbertTrendModeState {
    fn default() -> Self {
        Self {
            phase: HilbertPhaseState::default(),
            raw_price: [0.0; HISTORY_SIZE],
            raw_price_index: 0,
            trend_average_1: 0.0,
            trend_average_2: 0.0,
            trend_average_3: 0.0,
            days_in_trend: 0,
            previous_phase: 0.0,
            sine: 0.0,
            lead_sine: 0.0,
        }
    }
}

impl HilbertTrendModeState {
    #[inline]
    const fn observations(&self) -> usize {
        self.phase.observations()
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    fn next(&mut self, input: f64) -> Option<TrendMode> {
        self.raw_price[self.raw_price_index] = input;
        self.raw_price_index = (self.raw_price_index + 1) % HISTORY_SIZE;

        let transition = self.phase.next_phase_transition(input)?;
        let previous_phase = self.previous_phase;
        self.previous_phase = transition.phase;

        let previous_sine = self.sine;
        let previous_lead_sine = self.lead_sine;
        self.sine = (transition.phase * DEGREE_TO_RADIAN).sin();
        self.lead_sine = ((transition.phase + 45.0) * DEGREE_TO_RADIAN).sin();

        let period = (transition.smooth_period + 0.5) as usize;
        let mut average = 0.0;
        let mut index = (self.raw_price_index + HISTORY_SIZE - 1) % HISTORY_SIZE;
        for _ in 0..period {
            average += self.raw_price[index];
            index = (index + HISTORY_SIZE - 1) % HISTORY_SIZE;
        }
        if period != 0 {
            average /= period as f64;
        }

        let trendline = (4.0 * average
            + 3.0 * self.trend_average_1
            + 2.0 * self.trend_average_2
            + self.trend_average_3)
            / 10.0;
        self.trend_average_3 = self.trend_average_2;
        self.trend_average_2 = self.trend_average_1;
        self.trend_average_1 = average;

        let crossed = (self.sine > self.lead_sine && previous_sine <= previous_lead_sine)
            || (self.sine < self.lead_sine && previous_sine >= previous_lead_sine);
        let mut mode = TrendMode::Trend;
        if crossed {
            self.days_in_trend = 0;
            mode = TrendMode::Cycle;
        }
        self.days_in_trend += 1;
        if (self.days_in_trend as f64) < 0.5 * transition.smooth_period {
            mode = TrendMode::Cycle;
        }

        let phase_change = transition.phase - previous_phase;
        let expected_change = 360.0 / transition.smooth_period;
        if phase_change > 0.67 * expected_change && phase_change < 1.5 * expected_change {
            mode = TrendMode::Cycle;
        }
        if trendline != 0.0 && ((transition.smoothed_value - trendline) / trendline).abs() >= 0.015
        {
            mode = TrendMode::Trend;
        }

        (transition.today >= HT_TRENDMODE_LOOKBACK).then_some(mode)
    }
}

#[inline]
fn validate_input(real: &[Float]) -> Result<usize> {
    validate_finite_slice("real", real)?;
    validate_input_len(real.len(), HT_TRENDMODE_LOOKBACK)
}

#[inline]
fn compute_validated(real: &[Float], output: &mut [TrendMode]) -> OutputRange {
    let mut state = HilbertTrendModeState::default();
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(mode) = state.next(to_internal(input)) {
            output[output_index] = mode;
            output_index += 1;
        }
    }
    if output_index == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(HT_TRENDMODE_LOOKBACK, output_index)
    }
}

/// Computes typed Cycle/Trend mode classifications into caller-owned Compact Output storage.
#[allow(non_snake_case)]
pub fn HT_TRENDMODE(real: &[Float], output: &mut [TrendMode]) -> Result<OutputRange> {
    let count = validate_input(real)?;
    validate_output_len("HT_TRENDMODE", output.len(), count)?;
    Ok(compute_validated(real, output))
}

/// Immutable, parameter-free `HT_TRENDMODE` Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HT_TRENDMODEConfig;

impl HT_TRENDMODEConfig {
    /// Creates the parameter-free `HT_TRENDMODE` configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for HT_TRENDMODEConfig {}

impl IndicatorConfig for HT_TRENDMODEConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<TrendMode>;
    type OutputMut<'a> = &'a mut [TrendMode];
    type BatchRunner = HT_TRENDMODEBatchRunner;
    type Stream = HT_TRENDMODEStream;

    #[inline]
    fn lookback(&self) -> usize {
        HT_TRENDMODE_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_input(input)?;
        let mut values = vec![TrendMode::Cycle; count];
        let range = compute_validated(input, values.as_mut_slice());
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        HT_TRENDMODE(input, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(HT_TRENDMODEBatchRunner { max_input_len })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(HT_TRENDMODEStream::default())
    }
}

/// Reusable Prepared Batch Runner for [`HT_TRENDMODEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct HT_TRENDMODEBatchRunner {
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for HT_TRENDMODEBatchRunner {}

impl PreparedBatchRunner<HT_TRENDMODEConfig> for HT_TRENDMODEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <HT_TRENDMODEConfig as IndicatorConfig>::Input<'a>,
        output: <HT_TRENDMODEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        HT_TRENDMODEConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        HT_TRENDMODE(input, output)
    }
}

/// Independent Streaming Computation for [`HT_TRENDMODEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HT_TRENDMODEStream {
    state: HilbertTrendModeState,
}

impl crate::traits::sealed::Sealed for HT_TRENDMODEStream {}

impl StreamingComputation<HT_TRENDMODEConfig> for HT_TRENDMODEStream {
    type Tick = Float;
    type TickOutput = TrendMode;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        Ok(self.state.next(to_internal(input)))
    }

    #[inline]
    fn reset(&mut self) {
        self.state.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_count_matches_fixed_lookback() {
        assert_eq!(
            crate::output_count(HT_TRENDMODE_LOOKBACK, HT_TRENDMODE_LOOKBACK),
            0
        );
        assert_eq!(
            crate::output_count(HT_TRENDMODE_LOOKBACK + 1, HT_TRENDMODE_LOOKBACK),
            1
        );
    }
}
