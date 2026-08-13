//! Parabolic Stop and Reverse (`SAR`) definitions.

use crate::common::validate_finite_value;
use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

/// Default acceleration increment used by [`SARConfig`].
pub const SAR_DEFAULT_ACCELERATION: Float = 0.02 as Float;
/// Default acceleration limit used by [`SARConfig`].
pub const SAR_DEFAULT_MAXIMUM: Float = 0.2 as Float;
/// Default reversal offset used by [`SAREXTConfig`].
pub const SAREXT_DEFAULT_OFFSET_ON_REVERSE: Float = 0.0 as Float;
/// Default explicit start value used by [`SAREXTConfig`]; zero selects automatic direction.
pub const SAREXT_DEFAULT_START_VALUE: Float = 0.0 as Float;

/// Borrowed high/low inputs for SAR batch computation.
#[derive(Debug, Clone, Copy)]
pub struct SARInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
}

/// One high/low tick for SAR streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SARTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
}

fn invalid_parameter(name: &str, value: Float, expected: &str) -> TalibError {
    TalibError::invalid_parameter(name.to_string(), format!("{}", value), expected.to_string())
}

fn validate_non_negative(name: &str, value: Float) -> Result<()> {
    if !value.is_finite() || value < 0.0 as Float {
        return Err(invalid_parameter(
            name,
            value,
            "finite value greater than or equal to zero",
        ));
    }
    Ok(())
}

fn validate_sar_parameters(acceleration: Float, maximum: Float) -> Result<()> {
    validate_non_negative("acceleration", acceleration)?;
    validate_non_negative("maximum", maximum)?;
    if acceleration > maximum {
        return Err(invalid_parameter(
            "acceleration",
            acceleration,
            "value less than or equal to maximum",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_sarext_parameters(
    start_value: Float,
    offset_on_reverse: Float,
    acceleration_init_long: Float,
    acceleration_long: Float,
    acceleration_max_long: Float,
    acceleration_init_short: Float,
    acceleration_short: Float,
    acceleration_max_short: Float,
) -> Result<()> {
    if !start_value.is_finite() {
        return Err(invalid_parameter(
            "start_value",
            start_value,
            "finite value",
        ));
    }
    validate_non_negative("offset_on_reverse", offset_on_reverse)?;
    validate_non_negative("acceleration_init_long", acceleration_init_long)?;
    validate_non_negative("acceleration_long", acceleration_long)?;
    validate_non_negative("acceleration_max_long", acceleration_max_long)?;
    validate_non_negative("acceleration_init_short", acceleration_init_short)?;
    validate_non_negative("acceleration_short", acceleration_short)?;
    validate_non_negative("acceleration_max_short", acceleration_max_short)?;
    if acceleration_init_long > acceleration_max_long {
        return Err(invalid_parameter(
            "acceleration_init_long",
            acceleration_init_long,
            "value less than or equal to acceleration_max_long",
        ));
    }
    if acceleration_long > acceleration_max_long {
        return Err(invalid_parameter(
            "acceleration_long",
            acceleration_long,
            "value less than or equal to acceleration_max_long",
        ));
    }
    if acceleration_init_short > acceleration_max_short {
        return Err(invalid_parameter(
            "acceleration_init_short",
            acceleration_init_short,
            "value less than or equal to acceleration_max_short",
        ));
    }
    if acceleration_short > acceleration_max_short {
        return Err(invalid_parameter(
            "acceleration_short",
            acceleration_short,
            "value less than or equal to acceleration_max_short",
        ));
    }
    Ok(())
}

fn validate_input(input: SARInput<'_>) -> Result<(usize, usize)> {
    let len = validate_all_same_len(&[("high", input.high.len()), ("low", input.low.len())])?;
    validate_finite_slices(&[("high", input.high), ("low", input.low)])?;
    let count = validate_input_len(len, 1)?;
    Ok((len, count))
}

#[inline]
fn automatic_long(first: SARTick, second: SARTick) -> bool {
    let down_move = first.low - second.low;
    let up_move = second.high - first.high;
    !(down_move > 0.0 as Float && down_move > up_move)
}

#[derive(Debug, Clone, Copy)]
struct SARRecurrence {
    first: Option<SARTick>,
    previous: SARTick,
    initialized: bool,
    is_long: bool,
    acceleration: Float,
    maximum: Float,
    factor: Float,
    extreme: Float,
    stop: Float,
}

impl SARRecurrence {
    fn new(acceleration: Float, maximum: Float) -> Self {
        Self {
            first: None,
            previous: SARTick {
                high: 0.0 as Float,
                low: 0.0 as Float,
            },
            initialized: false,
            is_long: true,
            acceleration,
            maximum,
            factor: acceleration,
            extreme: 0.0 as Float,
            stop: 0.0 as Float,
        }
    }

    fn push(&mut self, tick: SARTick) -> Option<Float> {
        if !self.initialized {
            let Some(first) = self.first else {
                self.first = Some(tick);
                return None;
            };
            self.is_long = automatic_long(first, tick);
            self.factor = self.acceleration;
            if self.is_long {
                self.extreme = tick.high;
                self.stop = first.low;
            } else {
                self.extreme = tick.low;
                self.stop = first.high;
            }
            self.previous = tick;
            self.initialized = true;
        }

        let previous = self.previous;
        self.previous = tick;
        if self.is_long {
            if tick.low <= self.stop {
                self.is_long = false;
                self.stop = self.extreme.max(previous.high).max(tick.high);
                let output = self.stop;
                self.factor = self.acceleration;
                self.extreme = tick.low;
                self.stop = self
                    .factor
                    .mul_add(self.extreme - self.stop, self.stop)
                    .max(previous.high)
                    .max(tick.high);
                Some(output)
            } else {
                let output = self.stop;
                if tick.high > self.extreme {
                    self.extreme = tick.high;
                    self.factor = (self.factor + self.acceleration).min(self.maximum);
                }
                self.stop = self
                    .factor
                    .mul_add(self.extreme - self.stop, self.stop)
                    .min(previous.low)
                    .min(tick.low);
                Some(output)
            }
        } else if tick.high >= self.stop {
            self.is_long = true;
            self.stop = self.extreme.min(previous.low).min(tick.low);
            let output = self.stop;
            self.factor = self.acceleration;
            self.extreme = tick.high;
            self.stop = self
                .factor
                .mul_add(self.extreme - self.stop, self.stop)
                .min(previous.low)
                .min(tick.low);
            Some(output)
        } else {
            let output = self.stop;
            if tick.low < self.extreme {
                self.extreme = tick.low;
                self.factor = (self.factor + self.acceleration).min(self.maximum);
            }
            self.stop = self
                .factor
                .mul_add(self.extreme - self.stop, self.stop)
                .max(previous.high)
                .max(tick.high);
            Some(output)
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.acceleration, self.maximum);
    }
}

#[derive(Debug, Clone, Copy)]
struct SAREXTRecurrence {
    first: Option<SARTick>,
    previous: SARTick,
    initialized: bool,
    is_long: bool,
    config: SAREXTConfig,
    factor_long: Float,
    factor_short: Float,
    extreme: Float,
    stop: Float,
}

impl SAREXTRecurrence {
    fn new(config: SAREXTConfig) -> Self {
        Self {
            first: None,
            previous: SARTick {
                high: 0.0 as Float,
                low: 0.0 as Float,
            },
            initialized: false,
            is_long: true,
            factor_long: config.acceleration_init_long,
            factor_short: config.acceleration_init_short,
            extreme: 0.0 as Float,
            stop: 0.0 as Float,
            config,
        }
    }

    fn push(&mut self, tick: SARTick) -> Option<Float> {
        if !self.initialized {
            let Some(first) = self.first else {
                self.first = Some(tick);
                return None;
            };
            self.is_long = if self.config.start_value == 0.0 as Float {
                automatic_long(first, tick)
            } else {
                self.config.start_value > 0.0 as Float
            };
            if self.config.start_value == 0.0 as Float {
                if self.is_long {
                    self.extreme = tick.high;
                    self.stop = first.low;
                } else {
                    self.extreme = tick.low;
                    self.stop = first.high;
                }
            } else if self.is_long {
                self.extreme = tick.high;
                self.stop = self.config.start_value;
            } else {
                self.extreme = tick.low;
                self.stop = self.config.start_value.abs();
            }
            self.previous = tick;
            self.initialized = true;
        }

        let previous = self.previous;
        self.previous = tick;
        if self.is_long {
            if tick.low <= self.stop {
                self.is_long = false;
                self.stop = self.extreme.max(previous.high).max(tick.high);
                self.stop += self.stop * self.config.offset_on_reverse;
                let output = -self.stop;
                self.factor_short = self.config.acceleration_init_short;
                self.extreme = tick.low;
                self.stop = self
                    .factor_short
                    .mul_add(self.extreme - self.stop, self.stop)
                    .max(previous.high)
                    .max(tick.high);
                Some(output)
            } else {
                let output = self.stop;
                if tick.high > self.extreme {
                    self.extreme = tick.high;
                    self.factor_long = (self.factor_long + self.config.acceleration_long)
                        .min(self.config.acceleration_max_long);
                }
                self.stop = self
                    .factor_long
                    .mul_add(self.extreme - self.stop, self.stop)
                    .min(previous.low)
                    .min(tick.low);
                Some(output)
            }
        } else if tick.high >= self.stop {
            self.is_long = true;
            self.stop = self.extreme.min(previous.low).min(tick.low);
            self.stop -= self.stop * self.config.offset_on_reverse;
            let output = self.stop;
            self.factor_long = self.config.acceleration_init_long;
            self.extreme = tick.high;
            self.stop = self
                .factor_long
                .mul_add(self.extreme - self.stop, self.stop)
                .min(previous.low)
                .min(tick.low);
            Some(output)
        } else {
            let output = -self.stop;
            if tick.low < self.extreme {
                self.extreme = tick.low;
                self.factor_short = (self.factor_short + self.config.acceleration_short)
                    .min(self.config.acceleration_max_short);
            }
            self.stop = self
                .factor_short
                .mul_add(self.extreme - self.stop, self.stop)
                .max(previous.high)
                .max(tick.high);
            Some(output)
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.config);
    }
}

fn sar_kernel(
    input: SARInput<'_>,
    acceleration: Float,
    maximum: Float,
    output: &mut [Float],
) -> OutputRange {
    let mut state = SARRecurrence::new(acceleration, maximum);
    let mut output_idx = 0usize;
    for (&high, &low) in input.high.iter().zip(input.low) {
        if let Some(value) = state.push(SARTick { high, low }) {
            output[output_idx] = value;
            output_idx += 1;
        }
    }
    if output_idx == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(1, output_idx)
    }
}

fn sarext_kernel(input: SARInput<'_>, config: SAREXTConfig, output: &mut [Float]) -> OutputRange {
    let mut state = SAREXTRecurrence::new(config);
    let mut output_idx = 0usize;
    for (&high, &low) in input.high.iter().zip(input.low) {
        if let Some(value) = state.push(SARTick { high, low }) {
            output[output_idx] = value;
            output_idx += 1;
        }
    }
    if output_idx == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(1, output_idx)
    }
}

/// Computes Parabolic SAR into a caller-owned compact output buffer.
#[allow(non_snake_case)]
pub fn SAR(
    high: &[Float],
    low: &[Float],
    acceleration: Float,
    maximum: Float,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    validate_sar_parameters(acceleration, maximum)?;
    let input = SARInput { high, low };
    let (_, count) = validate_input(input)?;
    validate_output_len("SAR", out_real.len(), count)?;
    Ok(sar_kernel(input, acceleration, maximum, out_real))
}

/// Computes extended Parabolic SAR into a caller-owned compact output buffer.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn SAREXT(
    high: &[Float],
    low: &[Float],
    start_value: Float,
    offset_on_reverse: Float,
    acceleration_init_long: Float,
    acceleration_long: Float,
    acceleration_max_long: Float,
    acceleration_init_short: Float,
    acceleration_short: Float,
    acceleration_max_short: Float,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let config = SAREXTConfig::new(
        start_value,
        offset_on_reverse,
        acceleration_init_long,
        acceleration_long,
        acceleration_max_long,
        acceleration_init_short,
        acceleration_short,
        acceleration_max_short,
    )?;
    let input = SARInput { high, low };
    let (_, count) = validate_input(input)?;
    validate_output_len("SAREXT", out_real.len(), count)?;
    Ok(sarext_kernel(input, config, out_real))
}

/// Immutable Parabolic SAR Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SARConfig {
    acceleration: Float,
    maximum: Float,
}

impl SARConfig {
    /// Creates a SAR configuration with explicit acceleration increment and limit.
    pub fn new(acceleration: Float, maximum: Float) -> Result<Self> {
        validate_sar_parameters(acceleration, maximum)?;
        Ok(Self {
            acceleration,
            maximum,
        })
    }

    /// Creates the canonical SAR configuration (`0.02`, `0.2`).
    pub const fn with_defaults() -> Self {
        Self {
            acceleration: SAR_DEFAULT_ACCELERATION,
            maximum: SAR_DEFAULT_MAXIMUM,
        }
    }

    /// Returns the acceleration increment.
    pub const fn acceleration(&self) -> Float {
        self.acceleration
    }
    /// Returns the acceleration limit.
    pub const fn maximum(&self) -> Float {
        self.maximum
    }
}

impl Default for SARConfig {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl crate::traits::sealed::Sealed for SARConfig {}

impl IndicatorConfig for SARConfig {
    type Input<'a> = SARInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = SARBatchRunner;
    type Stream = SARStream;

    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, count) = validate_input(input)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = sar_kernel(input, self.acceleration, self.maximum, &mut values);
        CompactOutput::new(len, range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_input(input)?;
        validate_output_len("SAR", output.len(), count)?;
        Ok(sar_kernel(input, self.acceleration, self.maximum, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(SARBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(SARStream {
            state: SARRecurrence::new(self.acceleration, self.maximum),
        })
    }
}

/// Capacity-checked Prepared Batch Runner for SAR.
#[derive(Debug, Clone)]
pub struct SARBatchRunner {
    config: SARConfig,
    max_input_len: usize,
}
impl crate::traits::sealed::Sealed for SARBatchRunner {}
impl PreparedBatchRunner<SARConfig> for SARBatchRunner {
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }
    fn compute_into<'a>(
        &mut self,
        input: SARInput<'a>,
        output: &'a mut [Float],
    ) -> Result<OutputRange>
    where
        SARConfig: 'a,
    {
        let actual = input.high.len().max(input.low.len());
        if actual > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for SAR.
#[derive(Debug, Clone)]
pub struct SARStream {
    state: SARRecurrence,
}
impl crate::traits::sealed::Sealed for SARStream {}
impl StreamingComputation<SARConfig> for SARStream {
    type Tick = SARTick;
    type TickOutput = Float;
    fn next(&mut self, input: SARTick) -> Result<Option<Float>> {
        validate_finite_value("high", 0, input.high)?;
        validate_finite_value("low", 0, input.low)?;
        Ok(self.state.push(input))
    }
    fn reset(&mut self) {
        self.state.reset();
    }
}

/// Immutable extended Parabolic SAR Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SAREXTConfig {
    start_value: Float,
    offset_on_reverse: Float,
    acceleration_init_long: Float,
    acceleration_long: Float,
    acceleration_max_long: Float,
    acceleration_init_short: Float,
    acceleration_short: Float,
    acceleration_max_short: Float,
}

impl SAREXTConfig {
    /// Creates a SAREXT configuration with independent long and short dynamics.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        start_value: Float,
        offset_on_reverse: Float,
        acceleration_init_long: Float,
        acceleration_long: Float,
        acceleration_max_long: Float,
        acceleration_init_short: Float,
        acceleration_short: Float,
        acceleration_max_short: Float,
    ) -> Result<Self> {
        validate_sarext_parameters(
            start_value,
            offset_on_reverse,
            acceleration_init_long,
            acceleration_long,
            acceleration_max_long,
            acceleration_init_short,
            acceleration_short,
            acceleration_max_short,
        )?;
        Ok(Self {
            start_value,
            offset_on_reverse,
            acceleration_init_long,
            acceleration_long,
            acceleration_max_long,
            acceleration_init_short,
            acceleration_short,
            acceleration_max_short,
        })
    }

    /// Creates the canonical SAREXT configuration with automatic direction.
    pub const fn with_defaults() -> Self {
        Self {
            start_value: SAREXT_DEFAULT_START_VALUE,
            offset_on_reverse: SAREXT_DEFAULT_OFFSET_ON_REVERSE,
            acceleration_init_long: SAR_DEFAULT_ACCELERATION,
            acceleration_long: SAR_DEFAULT_ACCELERATION,
            acceleration_max_long: SAR_DEFAULT_MAXIMUM,
            acceleration_init_short: SAR_DEFAULT_ACCELERATION,
            acceleration_short: SAR_DEFAULT_ACCELERATION,
            acceleration_max_short: SAR_DEFAULT_MAXIMUM,
        }
    }

    /// Returns the signed explicit start value; zero selects automatic direction.
    pub const fn start_value(&self) -> Float {
        self.start_value
    }
    /// Returns the proportional offset applied only when reversing.
    pub const fn offset_on_reverse(&self) -> Float {
        self.offset_on_reverse
    }
    /// Returns the initial long acceleration factor.
    pub const fn acceleration_init_long(&self) -> Float {
        self.acceleration_init_long
    }
    /// Returns the long acceleration increment.
    pub const fn acceleration_long(&self) -> Float {
        self.acceleration_long
    }
    /// Returns the long acceleration limit.
    pub const fn acceleration_max_long(&self) -> Float {
        self.acceleration_max_long
    }
    /// Returns the initial short acceleration factor.
    pub const fn acceleration_init_short(&self) -> Float {
        self.acceleration_init_short
    }
    /// Returns the short acceleration increment.
    pub const fn acceleration_short(&self) -> Float {
        self.acceleration_short
    }
    /// Returns the short acceleration limit.
    pub const fn acceleration_max_short(&self) -> Float {
        self.acceleration_max_short
    }
}

impl Default for SAREXTConfig {
    fn default() -> Self {
        Self::with_defaults()
    }
}
impl crate::traits::sealed::Sealed for SAREXTConfig {}

impl IndicatorConfig for SAREXTConfig {
    type Input<'a> = SARInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = SAREXTBatchRunner;
    type Stream = SAREXTStream;

    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, count) = validate_input(input)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = sarext_kernel(input, *self, &mut values);
        CompactOutput::new(len, range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_input(input)?;
        validate_output_len("SAREXT", output.len(), count)?;
        Ok(sarext_kernel(input, *self, output))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(SAREXTBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(SAREXTStream {
            state: SAREXTRecurrence::new(*self),
        })
    }
}

/// Capacity-checked Prepared Batch Runner for SAREXT.
#[derive(Debug, Clone)]
pub struct SAREXTBatchRunner {
    config: SAREXTConfig,
    max_input_len: usize,
}
impl crate::traits::sealed::Sealed for SAREXTBatchRunner {}
impl PreparedBatchRunner<SAREXTConfig> for SAREXTBatchRunner {
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }
    fn compute_into<'a>(
        &mut self,
        input: SARInput<'a>,
        output: &'a mut [Float],
    ) -> Result<OutputRange>
    where
        SAREXTConfig: 'a,
    {
        let actual = input.high.len().max(input.low.len());
        if actual > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for SAREXT.
#[derive(Debug, Clone)]
pub struct SAREXTStream {
    state: SAREXTRecurrence,
}
impl crate::traits::sealed::Sealed for SAREXTStream {}
impl StreamingComputation<SAREXTConfig> for SAREXTStream {
    type Tick = SARTick;
    type TickOutput = Float;
    fn next(&mut self, input: SARTick) -> Result<Option<Float>> {
        validate_finite_value("high", 0, input.high)?;
        validate_finite_value("low", 0, input.low)?;
        Ok(self.state.push(input))
    }
    fn reset(&mut self) {
        self.state.reset();
    }
}
