//! Dominant-cycle phase, phasor, and sine Cycle Indicator Definitions.

use super::hilbert::{HilbertPhaseState, HilbertState};
use super::{from_internal, to_internal};
use crate::common::CompactPayloadLen;
use crate::{
    common::validate_finite_value, validate_finite_slice, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::{vec, vec::Vec};

/// Fixed Stabilization and Lookback for [`HT_DCPHASE`], in observations.
pub const HT_DCPHASE_LOOKBACK: usize = 63;
/// Fixed Stabilization and Lookback for [`HT_PHASOR`], in observations.
pub const HT_PHASOR_LOOKBACK: usize = 32;
/// Fixed Stabilization and Lookback for [`HT_SINE`], in observations.
pub const HT_SINE_LOOKBACK: usize = 63;

/// One valid streaming `HT_PHASOR` output, in input-value units.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HT_PHASORValue {
    /// InPhase component, in input-value units.
    pub in_phase: Float,
    /// Quadrature component, in input-value units.
    pub quadrature: Float,
}

/// Named compact `HT_PHASOR` columns, both in input-value units.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct HT_PHASORValues {
    /// InPhase component column.
    pub in_phase: Vec<Float>,
    /// Quadrature component column.
    pub quadrature: Vec<Float>,
}

impl CompactPayloadLen for HT_PHASORValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.in_phase.len() != self.quadrature.len() {
            return Err(TalibError::invalid_input(
                "HT_PHASOR Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.in_phase.len())
    }
}

/// Caller-owned compact `HT_PHASOR` columns.
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct HT_PHASORValuesMut<'a> {
    /// InPhase component output buffer.
    pub in_phase: &'a mut [Float],
    /// Quadrature component output buffer.
    pub quadrature: &'a mut [Float],
}

/// One valid streaming `HT_SINE` output. Both fields are unitless.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HT_SINEValue {
    /// Sine of the Dominant Cycle Phase.
    pub sine: Float,
    /// Sine of the phase advanced by 45 degrees.
    pub lead_sine: Float,
}

/// Named compact `HT_SINE` columns. Both columns are unitless.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct HT_SINEValues {
    /// Sine column.
    pub sine: Vec<Float>,
    /// 45-degree LeadSine column.
    pub lead_sine: Vec<Float>,
}

impl CompactPayloadLen for HT_SINEValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.sine.len() != self.lead_sine.len() {
            return Err(TalibError::invalid_input(
                "HT_SINE Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.sine.len())
    }
}

/// Caller-owned compact `HT_SINE` columns.
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct HT_SINEValuesMut<'a> {
    /// Sine output buffer.
    pub sine: &'a mut [Float],
    /// 45-degree LeadSine output buffer.
    pub lead_sine: &'a mut [Float],
}

#[inline]
fn validate_input(real: &[Float], lookback: usize) -> Result<usize> {
    validate_finite_slice("real", real)?;
    validate_input_len(real.len(), lookback)
}

#[inline]
fn validate_two_outputs(
    real: &[Float],
    lookback: usize,
    first_name: &str,
    first_len: usize,
    second_name: &str,
    second_len: usize,
) -> Result<usize> {
    let count = validate_input(real, lookback)?;
    validate_output_len(first_name, first_len, count)?;
    validate_output_len(second_name, second_len, count)?;
    Ok(count)
}

#[inline]
fn compute_phase_validated(real: &[Float], out_phase: &mut [Float]) -> OutputRange {
    let mut state = HilbertPhaseState::default();
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(phase) = state.next_dc_phase(to_internal(input)) {
            out_phase[output_index] = from_internal(phase);
            output_index += 1;
        }
    }
    range(HT_DCPHASE_LOOKBACK, output_index)
}

#[inline]
fn compute_phasor_validated(
    real: &[Float],
    out_in_phase: &mut [Float],
    out_quadrature: &mut [Float],
) -> OutputRange {
    let mut state = HilbertState::default();
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(transition) = state.next_phasor(to_internal(input)) {
            out_in_phase[output_index] = from_internal(transition.in_phase);
            out_quadrature[output_index] = from_internal(transition.quadrature);
            output_index += 1;
        }
    }
    range(HT_PHASOR_LOOKBACK, output_index)
}

#[inline]
fn compute_sine_validated(
    real: &[Float],
    out_sine: &mut [Float],
    out_lead_sine: &mut [Float],
) -> OutputRange {
    let mut state = HilbertPhaseState::default();
    let mut output_index = 0;
    let degree_to_radian = core::f64::consts::PI / 180.0;
    for input in real.iter().copied() {
        if let Some(phase) = state.next_dc_phase(to_internal(input)) {
            out_sine[output_index] = from_internal((phase * degree_to_radian).sin());
            out_lead_sine[output_index] = from_internal(((phase + 45.0) * degree_to_radian).sin());
            output_index += 1;
        }
    }
    range(HT_SINE_LOOKBACK, output_index)
}

#[inline]
fn range(lookback: usize, count: usize) -> OutputRange {
    if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(lookback, count)
    }
}

/// Computes Dominant Cycle Phase in degrees into caller-owned compact storage.
#[allow(non_snake_case)]
pub fn HT_DCPHASE(real: &[Float], out_phase: &mut [Float]) -> Result<OutputRange> {
    let count = validate_input(real, HT_DCPHASE_LOOKBACK)?;
    validate_output_len("HT_DCPHASE", out_phase.len(), count)?;
    Ok(compute_phase_validated(real, out_phase))
}

/// Computes named InPhase and Quadrature columns into caller-owned compact storage.
#[allow(non_snake_case)]
pub fn HT_PHASOR(
    real: &[Float],
    out_in_phase: &mut [Float],
    out_quadrature: &mut [Float],
) -> Result<OutputRange> {
    validate_two_outputs(
        real,
        HT_PHASOR_LOOKBACK,
        "HT_PHASOR InPhase",
        out_in_phase.len(),
        "HT_PHASOR Quadrature",
        out_quadrature.len(),
    )?;
    Ok(compute_phasor_validated(real, out_in_phase, out_quadrature))
}

/// Computes named Sine and LeadSine columns into caller-owned compact storage.
#[allow(non_snake_case)]
pub fn HT_SINE(
    real: &[Float],
    out_sine: &mut [Float],
    out_lead_sine: &mut [Float],
) -> Result<OutputRange> {
    validate_two_outputs(
        real,
        HT_SINE_LOOKBACK,
        "HT_SINE Sine",
        out_sine.len(),
        "HT_SINE LeadSine",
        out_lead_sine.len(),
    )?;
    Ok(compute_sine_validated(real, out_sine, out_lead_sine))
}

macro_rules! define_single_output_execution {
    ($config:ident, $runner:ident, $stream:ident, $lookback:ident, $function:ident, $state:ty, $next:ident, $unit:literal) => {
        #[doc = concat!("Immutable `", stringify!($function), "` Indicator Configuration. Output unit: ", $unit, ".")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $config;

        impl $config {
            #[doc = concat!("Creates the parameter-free `", stringify!($function), "` configuration.")]
            #[inline]
            pub const fn new() -> Self {
                Self
            }
        }

        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = &'a [Float];
            type Output = Vec<Float>;
            type OutputMut<'a> = &'a mut [Float];
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                $lookback
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let count = validate_input(input, $lookback)?;
                let mut values = vec![0.0 as Float; count];
                let range = $function(input, values.as_mut_slice())?;
                CompactOutput::new(input.len(), range, values)
            }

            #[inline]
            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $function(input, output)
            }

            #[inline]
            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner { max_input_len })
            }

            #[inline]
            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream::default())
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for [`", stringify!($config), "`].")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $runner {
            max_input_len: usize,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            #[inline]
            fn compute_into<'a>(
                &mut self,
                input: <$config as IndicatorConfig>::Input<'a>,
                output: <$config as IndicatorConfig>::OutputMut<'a>,
            ) -> Result<OutputRange>
            where
                $config: 'a,
            {
                if input.len() > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input.len(),
                    ));
                }
                $function(input, output)
            }
        }

        #[doc = concat!("Independent Streaming Computation for [`", stringify!($config), "`].")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, Default, PartialEq)]
        pub struct $stream {
            state: $state,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            #[inline]
            fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
                validate_finite_value("input", self.state.observations(), input)?;
                Ok(self.state.$next(to_internal(input)).map(from_internal))
            }

            #[inline]
            fn reset(&mut self) {
                self.state.reset();
            }
        }
    };
}

define_single_output_execution!(
    HT_DCPHASEConfig,
    HT_DCPHASEBatchRunner,
    HT_DCPHASEStream,
    HT_DCPHASE_LOOKBACK,
    HT_DCPHASE,
    HilbertPhaseState,
    next_dc_phase,
    "degrees"
);

/// Immutable `HT_PHASOR` Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HT_PHASORConfig;

impl HT_PHASORConfig {
    /// Creates the parameter-free `HT_PHASOR` configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for HT_PHASORConfig {}

impl IndicatorConfig for HT_PHASORConfig {
    type Input<'a> = &'a [Float];
    type Output = HT_PHASORValues;
    type OutputMut<'a> = HT_PHASORValuesMut<'a>;
    type BatchRunner = HT_PHASORBatchRunner;
    type Stream = HT_PHASORStream;

    #[inline]
    fn lookback(&self) -> usize {
        HT_PHASOR_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_input(input, HT_PHASOR_LOOKBACK)?;
        let mut values = HT_PHASORValues {
            in_phase: vec![0.0 as Float; count],
            quadrature: vec![0.0 as Float; count],
        };
        let range = HT_PHASOR(
            input,
            values.in_phase.as_mut_slice(),
            values.quadrature.as_mut_slice(),
        )?;
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        HT_PHASOR(input, output.in_phase, output.quadrature)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(HT_PHASORBatchRunner { max_input_len })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(HT_PHASORStream::default())
    }
}

/// Reusable Prepared Batch Runner for [`HT_PHASORConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct HT_PHASORBatchRunner {
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for HT_PHASORBatchRunner {}

impl PreparedBatchRunner<HT_PHASORConfig> for HT_PHASORBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <HT_PHASORConfig as IndicatorConfig>::Input<'a>,
        output: <HT_PHASORConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        HT_PHASORConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        HT_PHASOR(input, output.in_phase, output.quadrature)
    }
}

/// Independent Streaming Computation for [`HT_PHASORConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HT_PHASORStream {
    state: HilbertState,
}

impl crate::traits::sealed::Sealed for HT_PHASORStream {}

impl StreamingComputation<HT_PHASORConfig> for HT_PHASORStream {
    type Tick = Float;
    type TickOutput = HT_PHASORValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        Ok(self
            .state
            .next_phasor(to_internal(input))
            .map(|transition| HT_PHASORValue {
                in_phase: from_internal(transition.in_phase),
                quadrature: from_internal(transition.quadrature),
            }))
    }

    #[inline]
    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Immutable `HT_SINE` Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HT_SINEConfig;

impl HT_SINEConfig {
    /// Creates the parameter-free `HT_SINE` configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for HT_SINEConfig {}

impl IndicatorConfig for HT_SINEConfig {
    type Input<'a> = &'a [Float];
    type Output = HT_SINEValues;
    type OutputMut<'a> = HT_SINEValuesMut<'a>;
    type BatchRunner = HT_SINEBatchRunner;
    type Stream = HT_SINEStream;

    #[inline]
    fn lookback(&self) -> usize {
        HT_SINE_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_input(input, HT_SINE_LOOKBACK)?;
        let mut values = HT_SINEValues {
            sine: vec![0.0 as Float; count],
            lead_sine: vec![0.0 as Float; count],
        };
        let range = HT_SINE(
            input,
            values.sine.as_mut_slice(),
            values.lead_sine.as_mut_slice(),
        )?;
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        HT_SINE(input, output.sine, output.lead_sine)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(HT_SINEBatchRunner { max_input_len })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(HT_SINEStream::default())
    }
}

/// Reusable Prepared Batch Runner for [`HT_SINEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct HT_SINEBatchRunner {
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for HT_SINEBatchRunner {}

impl PreparedBatchRunner<HT_SINEConfig> for HT_SINEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <HT_SINEConfig as IndicatorConfig>::Input<'a>,
        output: <HT_SINEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        HT_SINEConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        HT_SINE(input, output.sine, output.lead_sine)
    }
}

/// Independent Streaming Computation for [`HT_SINEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HT_SINEStream {
    state: HilbertPhaseState,
}

impl crate::traits::sealed::Sealed for HT_SINEStream {}

impl StreamingComputation<HT_SINEConfig> for HT_SINEStream {
    type Tick = Float;
    type TickOutput = HT_SINEValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        let degree_to_radian = core::f64::consts::PI / 180.0;
        Ok(self
            .state
            .next_dc_phase(to_internal(input))
            .map(|phase| HT_SINEValue {
                sine: from_internal((phase * degree_to_radian).sin()),
                lead_sine: from_internal(((phase + 45.0) * degree_to_radian).sin()),
            }))
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
    fn compact_counts_match_fixed_lookbacks() {
        assert_eq!(crate::output_count(64, HT_DCPHASE_LOOKBACK), 1);
        assert_eq!(crate::output_count(33, HT_PHASOR_LOOKBACK), 1);
        assert_eq!(crate::output_count(64, HT_SINE_LOOKBACK), 1);
    }
}
