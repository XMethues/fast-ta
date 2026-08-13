//! Cycle Indicator Definitions.
//!
//! Cycle definitions and Hilbert-based Overlap Studies share one crate-private
//! Hilbert transition without exposing its recurrence state publicly.

pub(crate) mod hilbert;
mod outputs;
mod trend_mode;
pub use outputs::{
    HT_DCPHASEBatchRunner, HT_DCPHASEConfig, HT_DCPHASEStream, HT_PHASORBatchRunner,
    HT_PHASORConfig, HT_PHASORStream, HT_PHASORValue, HT_PHASORValues, HT_PHASORValuesMut,
    HT_SINEBatchRunner, HT_SINEConfig, HT_SINEStream, HT_SINEValue, HT_SINEValues,
    HT_SINEValuesMut, HT_DCPHASE, HT_DCPHASE_LOOKBACK, HT_PHASOR, HT_PHASOR_LOOKBACK, HT_SINE,
    HT_SINE_LOOKBACK,
};
pub use trend_mode::{
    HT_TRENDMODEBatchRunner, HT_TRENDMODEConfig, HT_TRENDMODEStream, TrendMode, HT_TRENDMODE,
    HT_TRENDMODE_LOOKBACK,
};

use crate::{
    common::validate_finite_value, validate_finite_slice, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};
use hilbert::HilbertState;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(feature = "f32")]
#[inline(always)]
fn to_internal(value: Float) -> f64 {
    f64::from(value)
}

#[cfg(not(feature = "f32"))]
#[inline(always)]
fn to_internal(value: Float) -> f64 {
    value
}

#[cfg(feature = "f32")]
#[inline(always)]
fn from_internal(value: f64) -> Float {
    value as Float
}

#[cfg(not(feature = "f32"))]
#[inline(always)]
fn from_internal(value: f64) -> Float {
    value
}

/// Fixed Stabilization for [`HT_DCPERIOD`].
pub const HT_DCPERIOD_LOOKBACK: usize = 32;

/// Immutable `HT_DCPERIOD` Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HT_DCPERIODConfig;

impl HT_DCPERIODConfig {
    /// Creates the parameter-free `HT_DCPERIOD` configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    fn validate_input(real: &[Float]) -> Result<usize> {
        validate_finite_slice("real", real)?;
        validate_input_len(real.len(), HT_DCPERIOD_LOOKBACK)
    }

    #[inline]
    fn compute_validated(real: &[Float], out_real: &mut [Float]) -> OutputRange {
        let mut state = HilbertState::default();
        let mut output_index = 0;
        for input in real.iter().copied() {
            if let Some(period) = state.next_dc_period(to_internal(input)) {
                out_real[output_index] = from_internal(period);
                output_index += 1;
            }
        }

        if output_index == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(HT_DCPERIOD_LOOKBACK, output_index)
        }
    }
}

/// Computes the dominant cycle period into caller-owned compact output storage.
#[allow(non_snake_case)]
pub fn HT_DCPERIOD(real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let count = HT_DCPERIODConfig::validate_input(real)?;
    validate_output_len("HT_DCPERIOD", out_real.len(), count)?;
    Ok(HT_DCPERIODConfig::compute_validated(real, out_real))
}

impl crate::traits::sealed::Sealed for HT_DCPERIODConfig {}

impl IndicatorConfig for HT_DCPERIODConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = HT_DCPERIODBatchRunner;
    type Stream = HT_DCPERIODStream;

    #[inline]
    fn lookback(&self) -> usize {
        HT_DCPERIOD_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = Self::validate_input(input)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = Self::compute_validated(input, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        HT_DCPERIOD(input, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(HT_DCPERIODBatchRunner { max_input_len })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(HT_DCPERIODStream::default())
    }
}

/// Reusable Prepared Batch Runner for [`HT_DCPERIODConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct HT_DCPERIODBatchRunner {
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for HT_DCPERIODBatchRunner {}

impl PreparedBatchRunner<HT_DCPERIODConfig> for HT_DCPERIODBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <HT_DCPERIODConfig as IndicatorConfig>::Input<'a>,
        output: <HT_DCPERIODConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        HT_DCPERIODConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        HT_DCPERIOD(input, output)
    }
}

/// Independent Streaming Computation for [`HT_DCPERIODConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HT_DCPERIODStream {
    state: HilbertState,
}

impl crate::traits::sealed::Sealed for HT_DCPERIODStream {}

impl StreamingComputation<HT_DCPERIODConfig> for HT_DCPERIODStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        Ok(self
            .state
            .next_dc_period(to_internal(input))
            .map(from_internal))
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
        assert_eq!(crate::output_count(0, HT_DCPERIOD_LOOKBACK), 0);
        assert_eq!(crate::output_count(33, HT_DCPERIOD_LOOKBACK), 1);
    }
}
