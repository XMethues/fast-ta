//! Hilbert-based Overlap Study Indicator Definitions.

use crate::common::{validate_finite_value, CompactPayloadLen};
use crate::cycle::hilbert::HilbertState;
use crate::{
    validate_finite_slice, validate_input_len, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec, vec::Vec};

/// Fixed Stabilization and Lookback for [`MAMA`], in observations.
pub const MAMA_LOOKBACK: usize = 32;
/// Default upper adaptation limit for [`MAMAConfig`].
pub const MAMA_DEFAULT_FAST_LIMIT: Float = 0.5 as Float;
/// Default lower adaptation limit for [`MAMAConfig`].
pub const MAMA_DEFAULT_SLOW_LIMIT: Float = 0.05 as Float;
/// Fixed Stabilization and Lookback for [`HT_TRENDLINE`], in observations.
pub const HT_TRENDLINE_LOOKBACK: usize = 63;

const MINIMUM_MAMA_LIMIT: Float = 0.01 as Float;
const MAXIMUM_MAMA_LIMIT: Float = 0.99 as Float;
const TREND_HISTORY_SIZE: usize = 50;
const RAD_TO_DEG: f64 = 45.0 / core::f64::consts::FRAC_PI_4;

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

#[inline]
fn invalid_limit(name: &str, value: Float) -> TalibError {
    TalibError::invalid_parameter(
        name.to_string(),
        format!("{value}"),
        "finite value in the inclusive range [0.01, 0.99]".to_string(),
    )
}

#[inline]
fn validate_limit(name: &str, value: Float) -> Result<()> {
    if !value.is_finite() || !(MINIMUM_MAMA_LIMIT..=MAXIMUM_MAMA_LIMIT).contains(&value) {
        return Err(invalid_limit(name, value));
    }
    Ok(())
}

#[inline]
fn validate_mama_limits(fast_limit: Float, slow_limit: Float) -> Result<()> {
    validate_limit("fast_limit", fast_limit)?;
    validate_limit("slow_limit", slow_limit)
}

#[inline]
fn validate_input(real: &[Float], lookback: usize) -> Result<usize> {
    validate_finite_slice("real", real)?;
    validate_input_len(real.len(), lookback)
}

#[inline]
fn output_range(lookback: usize, count: usize) -> OutputRange {
    if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(lookback, count)
    }
}

/// One valid streaming MESA Adaptive Moving Average result.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MAMAValue {
    /// MESA Adaptive Moving Average.
    pub mama: Float,
    /// Following Adaptive Moving Average.
    pub fama: Float,
}

/// Named, source-aligned compact MAMA and FAMA columns.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, PartialEq)]
pub struct MAMAValues {
    /// MESA Adaptive Moving Average column.
    pub mama: Vec<Float>,
    /// Following Adaptive Moving Average column.
    pub fama: Vec<Float>,
}

impl CompactPayloadLen for MAMAValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.mama.len() != self.fama.len() {
            return Err(TalibError::invalid_input(
                "MAMA Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.mama.len())
    }
}

/// Caller-owned compact MAMA and FAMA columns.
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct MAMAValuesMut<'a> {
    /// MESA Adaptive Moving Average output buffer.
    pub mama: &'a mut [Float],
    /// Following Adaptive Moving Average output buffer.
    pub fama: &'a mut [Float],
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct MAMAState {
    hilbert: HilbertState,
    previous_phase: f64,
    mama: f64,
    fama: f64,
}

impl MAMAState {
    #[inline]
    fn observations(&self) -> usize {
        self.hilbert.observations()
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    fn next(&mut self, input: f64, fast_limit: f64, slow_limit: f64) -> Option<MAMAValue> {
        let today = self.hilbert.observations();
        let transition = self.hilbert.next_mama_transition(input)?;
        let phase = if transition.in_phase != 0.0 {
            (transition.quadrature / transition.in_phase).atan() * RAD_TO_DEG
        } else {
            0.0
        };
        let delta_phase = (self.previous_phase - phase).max(1.0);
        self.previous_phase = phase;
        let alpha = if delta_phase > 1.0 {
            (fast_limit / delta_phase).max(slow_limit)
        } else {
            fast_limit
        };

        self.mama = alpha * input + (1.0 - alpha) * self.mama;
        let fama_alpha = 0.5 * alpha;
        self.fama = fama_alpha * self.mama + (1.0 - fama_alpha) * self.fama;

        (today >= MAMA_LOOKBACK).then_some(MAMAValue {
            mama: from_internal(self.mama),
            fama: from_internal(self.fama),
        })
    }
}

#[inline]
fn compute_mama_validated(
    real: &[Float],
    fast_limit: Float,
    slow_limit: Float,
    out_mama: &mut [Float],
    out_fama: &mut [Float],
) -> OutputRange {
    let mut state = MAMAState::default();
    let mut output_index = 0;
    let fast_limit = to_internal(fast_limit);
    let slow_limit = to_internal(slow_limit);
    for input in real.iter().copied() {
        if let Some(value) = state.next(to_internal(input), fast_limit, slow_limit) {
            out_mama[output_index] = value.mama;
            out_fama[output_index] = value.fama;
            output_index += 1;
        }
    }
    output_range(MAMA_LOOKBACK, output_index)
}

/// Computes named MAMA and FAMA columns into caller-owned compact storage.
///
/// Both adaptation limits must be finite and in the inclusive range
/// `[0.01, 0.99]`. Output buffers hold `real.len() - MAMA_LOOKBACK` values.
#[allow(non_snake_case)]
pub fn MAMA(
    real: &[Float],
    fast_limit: Float,
    slow_limit: Float,
    out_mama: &mut [Float],
    out_fama: &mut [Float],
) -> Result<OutputRange> {
    validate_mama_limits(fast_limit, slow_limit)?;
    let count = validate_input(real, MAMA_LOOKBACK)?;
    validate_output_len("MAMA", out_mama.len(), count)?;
    validate_output_len("FAMA", out_fama.len(), count)?;
    Ok(compute_mama_validated(
        real, fast_limit, slow_limit, out_mama, out_fama,
    ))
}

/// Immutable MESA Adaptive Moving Average Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MAMAConfig {
    fast_limit: Float,
    slow_limit: Float,
}

impl MAMAConfig {
    /// Creates a MAMA configuration with explicit fast and slow adaptation limits.
    ///
    /// Each limit must be finite and in the inclusive range `[0.01, 0.99]`.
    pub fn new(fast_limit: Float, slow_limit: Float) -> Result<Self> {
        validate_mama_limits(fast_limit, slow_limit)?;
        Ok(Self {
            fast_limit,
            slow_limit,
        })
    }

    /// Returns the configured upper adaptation limit.
    #[inline]
    pub const fn fast_limit(&self) -> Float {
        self.fast_limit
    }

    /// Returns the configured lower adaptation limit.
    #[inline]
    pub const fn slow_limit(&self) -> Float {
        self.slow_limit
    }
}

impl Default for MAMAConfig {
    fn default() -> Self {
        Self {
            fast_limit: MAMA_DEFAULT_FAST_LIMIT,
            slow_limit: MAMA_DEFAULT_SLOW_LIMIT,
        }
    }
}

impl crate::traits::sealed::Sealed for MAMAConfig {}

impl IndicatorConfig for MAMAConfig {
    type Input<'a> = &'a [Float];
    type Output = MAMAValues;
    type OutputMut<'a> = MAMAValuesMut<'a>;
    type BatchRunner = MAMABatchRunner;
    type Stream = MAMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        MAMA_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_input(input, MAMA_LOOKBACK)?;
        let mut values = MAMAValues {
            mama: vec![0.0 as Float; count],
            fama: vec![0.0 as Float; count],
        };
        let range = compute_mama_validated(
            input,
            self.fast_limit,
            self.slow_limit,
            values.mama.as_mut_slice(),
            values.fama.as_mut_slice(),
        );
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let count = validate_input(input, MAMA_LOOKBACK)?;
        validate_output_len("MAMA", output.mama.len(), count)?;
        validate_output_len("FAMA", output.fama.len(), count)?;
        Ok(compute_mama_validated(
            input,
            self.fast_limit,
            self.slow_limit,
            output.mama,
            output.fama,
        ))
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(MAMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(MAMAStream {
            fast_limit: to_internal(self.fast_limit),
            slow_limit: to_internal(self.slow_limit),
            state: MAMAState::default(),
        })
    }
}

/// Reusable Prepared Batch Runner for [`MAMAConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MAMABatchRunner {
    config: MAMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for MAMABatchRunner {}

impl PreparedBatchRunner<MAMAConfig> for MAMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MAMAConfig as IndicatorConfig>::Input<'a>,
        output: <MAMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MAMAConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for [`MAMAConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MAMAStream {
    fast_limit: f64,
    slow_limit: f64,
    state: MAMAState,
}

impl crate::traits::sealed::Sealed for MAMAStream {}

impl StreamingComputation<MAMAConfig> for MAMAStream {
    type Tick = Float;
    type TickOutput = MAMAValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        Ok(self
            .state
            .next(to_internal(input), self.fast_limit, self.slow_limit))
    }

    #[inline]
    fn reset(&mut self) {
        self.state.reset();
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HTTrendlineState {
    hilbert: HilbertState,
    input_history: [f64; TREND_HISTORY_SIZE],
    trend_1: f64,
    trend_2: f64,
    trend_3: f64,
}

impl Default for HTTrendlineState {
    fn default() -> Self {
        Self {
            hilbert: HilbertState::default(),
            input_history: [0.0; TREND_HISTORY_SIZE],
            trend_1: 0.0,
            trend_2: 0.0,
            trend_3: 0.0,
        }
    }
}

impl HTTrendlineState {
    #[inline]
    fn observations(&self) -> usize {
        self.hilbert.observations()
    }

    #[inline]
    fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    fn next(&mut self, input: f64) -> Option<Float> {
        let today = self.hilbert.observations();
        let history_index = today % TREND_HISTORY_SIZE;
        self.input_history[history_index] = input;
        let transition = self.hilbert.next_trendline_transition(input)?;
        let period = (transition.smooth_period + 0.5) as usize;
        let mut average = 0.0;
        for offset in 0..period {
            let index = (history_index + TREND_HISTORY_SIZE - offset) % TREND_HISTORY_SIZE;
            average += self.input_history[index];
        }
        if period > 0 {
            average /= period as f64;
        }

        let trendline =
            (4.0 * average + 3.0 * self.trend_1 + 2.0 * self.trend_2 + self.trend_3) / 10.0;
        self.trend_3 = self.trend_2;
        self.trend_2 = self.trend_1;
        self.trend_1 = average;

        (today >= HT_TRENDLINE_LOOKBACK).then_some(from_internal(trendline))
    }
}

#[inline]
fn compute_trendline_validated(real: &[Float], out_real: &mut [Float]) -> OutputRange {
    let mut state = HTTrendlineState::default();
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(value) = state.next(to_internal(input)) {
            out_real[output_index] = value;
            output_index += 1;
        }
    }
    output_range(HT_TRENDLINE_LOOKBACK, output_index)
}

/// Computes the Hilbert Transform Instantaneous Trendline into compact storage.
///
/// The output buffer holds `real.len() - HT_TRENDLINE_LOOKBACK` valid values.
#[allow(non_snake_case)]
pub fn HT_TRENDLINE(real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let count = validate_input(real, HT_TRENDLINE_LOOKBACK)?;
    validate_output_len("HT_TRENDLINE", out_real.len(), count)?;
    Ok(compute_trendline_validated(real, out_real))
}

/// Immutable Hilbert Transform Instantaneous Trendline Indicator Configuration.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct HT_TRENDLINEConfig;

impl HT_TRENDLINEConfig {
    /// Creates the parameter-free HT_TRENDLINE configuration.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for HT_TRENDLINEConfig {}

impl IndicatorConfig for HT_TRENDLINEConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = HT_TRENDLINEBatchRunner;
    type Stream = HT_TRENDLINEStream;

    #[inline]
    fn lookback(&self) -> usize {
        HT_TRENDLINE_LOOKBACK
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_input(input, HT_TRENDLINE_LOOKBACK)?;
        let mut values = vec![0.0 as Float; count];
        let range = compute_trendline_validated(input, values.as_mut_slice());
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        HT_TRENDLINE(input, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(HT_TRENDLINEBatchRunner { max_input_len })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(HT_TRENDLINEStream::default())
    }
}

/// Reusable Prepared Batch Runner for [`HT_TRENDLINEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HT_TRENDLINEBatchRunner {
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for HT_TRENDLINEBatchRunner {}

impl PreparedBatchRunner<HT_TRENDLINEConfig> for HT_TRENDLINEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <HT_TRENDLINEConfig as IndicatorConfig>::Input<'a>,
        output: <HT_TRENDLINEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        HT_TRENDLINEConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        HT_TRENDLINE(input, output)
    }
}

/// Independent Streaming Computation for [`HT_TRENDLINEConfig`].
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HT_TRENDLINEStream {
    state: HTTrendlineState,
}

impl crate::traits::sealed::Sealed for HT_TRENDLINEStream {}

impl StreamingComputation<HT_TRENDLINEConfig> for HT_TRENDLINEStream {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", self.state.observations(), input)?;
        Ok(self.state.next(to_internal(input)))
    }

    #[inline]
    fn reset(&mut self) {
        self.state.reset();
    }
}
