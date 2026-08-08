//! Range-position Momentum Indicators: AROON, stochastic oscillators, and WILLR.
//!
//! The family shares one rolling-extrema transition. Stochastic definitions
//! compose the qualified Period-based Moving Average dispatcher, and STOCHRSI
//! composes the qualified RSI transition before applying the same stochastic
//! range projection.

use super::relative_strength::{RSIBatchRunner, RSIConfig, RSIStream};
use crate::common::{validate_finite_value, CompactPayloadLen};
use crate::math_operators::{rolling_range_extrema, RangeExtremaScratch};
use crate::overlap::{MABatchRunner, MAConfig, MAStream, PeriodMAType};
use crate::{
    validate_all_same_len, validate_finite_slice, validate_finite_slices, validate_input_len,
    validate_output_len, CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner,
    Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec, vec::Vec};

const MAX_PERIOD: usize = 100_000;
const TA_EPSILON: Float = 1e-14 as Float;

#[inline]
fn validate_bounded_period(name: &str, period: usize, minimum: usize) -> Result<()> {
    if !(minimum..=MAX_PERIOD).contains(&period) {
        return Err(TalibError::invalid_period(
            period,
            format!("{name} must be in {minimum}..={MAX_PERIOD}"),
        ));
    }
    Ok(())
}

#[inline]
fn ta_is_zero(value: Float) -> bool {
    value > -TA_EPSILON && value < TA_EPSILON
}

fn validate_two_outputs(
    name: &str,
    first_name: &str,
    first_len: usize,
    second_name: &str,
    second_len: usize,
    count: usize,
) -> Result<()> {
    validate_all_same_len(&[(first_name, first_len), (second_name, second_len)])?;
    validate_output_len(name, first_len, count)
}

/// Borrowed aligned high/low observations used by AROON and AROONOSC.
#[derive(Debug, Clone, Copy)]
pub struct AroonInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
}

/// One aligned high/low tick used by AROON and AROONOSC streams.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AroonTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
}

/// Borrowed aligned high/low/close observations used by stochastic and WILLR definitions.
#[derive(Debug, Clone, Copy)]
pub struct StochasticInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One aligned high/low/close tick used by stochastic and WILLR streams.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

fn validate_aroon_input(input: AroonInput<'_>, period: usize) -> Result<(usize, usize)> {
    validate_bounded_period("timeperiod", period, 2)?;
    let len = validate_all_same_len(&[("high", input.high.len()), ("low", input.low.len())])?;
    validate_finite_slices(&[("high", input.high), ("low", input.low)])?;
    let count = validate_input_len(len, period)?;
    Ok((period, count))
}

fn validate_stochastic_input(input: StochasticInput<'_>, lookback: usize) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", input.high.len()),
        ("low", input.low.len()),
        ("close", input.close.len()),
    ])?;
    validate_finite_slices(&[
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    validate_input_len(len, lookback)
}

#[inline]
fn validate_aroon_tick(input: AroonTick) -> Result<()> {
    validate_finite_value("high", 0, input.high)?;
    validate_finite_value("low", 0, input.low)
}

#[inline]
fn validate_stochastic_tick(input: StochasticTick) -> Result<()> {
    validate_finite_value("high", 0, input.high)?;
    validate_finite_value("low", 0, input.low)?;
    validate_finite_value("close", 0, input.close)
}

#[derive(Debug, Clone, Copy, Default)]
struct RangeObservation {
    low: Float,
    high: Float,
    index: usize,
}

#[derive(Debug, Clone, Copy)]
struct RangeValue {
    min: Float,
    max: Float,
    min_idx: usize,
    max_idx: usize,
}

/// Period-bounded paired extrema state. Equal extrema select the newest source
/// position, matching the range-position definitions without changing the
/// oldest-tie public Math Operators contract.
#[derive(Debug, Clone)]
struct RangeStream {
    period: usize,
    observations: Vec<RangeObservation>,
    slot: usize,
    count: usize,
    seen: usize,
}

impl RangeStream {
    fn new(period: usize) -> Self {
        Self {
            period,
            observations: vec![RangeObservation::default(); period],
            slot: 0,
            count: 0,
            seen: 0,
        }
    }

    fn next(&mut self, low: Float, high: Float) -> Result<Option<RangeValue>> {
        let next_seen = self
            .seen
            .checked_add(1)
            .ok_or_else(|| TalibError::computation_error("range stream position overflow"))?;
        let next_count = (self.count + 1).min(self.period);
        let incoming = RangeObservation {
            low,
            high,
            index: self.seen,
        };

        let output = if next_count < self.period {
            None
        } else {
            let mut min = low;
            let mut max = high;
            let mut min_idx = self.seen;
            let mut max_idx = self.seen;
            for (idx, observation) in self.observations[..self.count].iter().enumerate() {
                if self.count == self.period && idx == self.slot {
                    continue;
                }
                if observation.low < min || (observation.low == min && observation.index > min_idx)
                {
                    min = observation.low;
                    min_idx = observation.index;
                }
                if observation.high > max
                    || (observation.high == max && observation.index > max_idx)
                {
                    max = observation.high;
                    max_idx = observation.index;
                }
            }
            Some(RangeValue {
                min,
                max,
                min_idx,
                max_idx,
            })
        };

        self.observations[self.slot] = incoming;
        self.slot = (self.slot + 1) % self.period;
        self.count = next_count;
        self.seen = next_seen;
        Ok(output)
    }

    fn reset(&mut self) {
        self.observations.fill(RangeObservation::default());
        self.slot = 0;
        self.count = 0;
        self.seen = 0;
    }
}

/// Named compact AROON output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct AROONValues {
    /// Aroon Down values.
    pub down: Vec<Float>,
    /// Aroon Up values.
    pub up: Vec<Float>,
}

impl CompactPayloadLen for AROONValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.down.len() != self.up.len() {
            return Err(TalibError::invalid_input(
                "AROON Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.down.len())
    }
}

/// Caller-owned compact AROON output columns.
#[derive(Debug)]
pub struct AROONValuesMut<'a> {
    /// Aroon Down output buffer.
    pub down: &'a mut [Float],
    /// Aroon Up output buffer.
    pub up: &'a mut [Float],
}

/// One valid streaming AROON output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AROONValue {
    /// Aroon Down value.
    pub down: Float,
    /// Aroon Up value.
    pub up: Float,
}

fn aroon_kernel<const RESERVED_PUSH: bool>(
    input: AroonInput<'_>,
    period: usize,
    count: usize,
    down: &mut [Float],
    up: &mut [Float],
    scratch: &mut RangeExtremaScratch,
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }
    let factor = 100.0 as Float / period as Float;
    rolling_range_extrema::<true, RESERVED_PUSH, _>(
        input.low,
        input.high,
        period + 1,
        scratch,
        |output_idx, extrema| {
            let source_idx = output_idx + period;
            down[output_idx] = factor * (period - (source_idx - extrema.min_idx)) as Float;
            up[output_idx] = factor * (period - (source_idx - extrema.max_idx)) as Float;
        },
    );
    OutputRange::new(period, count)
}

fn aroonosc_kernel<const RESERVED_PUSH: bool>(
    input: AroonInput<'_>,
    period: usize,
    count: usize,
    output: &mut [Float],
    scratch: &mut RangeExtremaScratch,
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }
    let factor = 100.0 as Float / period as Float;
    rolling_range_extrema::<true, RESERVED_PUSH, _>(
        input.low,
        input.high,
        period + 1,
        scratch,
        |output_idx, extrema| {
            let source_idx = output_idx + period;
            let up = period - (source_idx - extrema.max_idx);
            let down = period - (source_idx - extrema.min_idx);
            output[output_idx] = factor * (up as Float - down as Float);
        },
    );
    OutputRange::new(period, count)
}

/// Computes AROON Down and Up into equal-length caller-owned columns.
#[allow(non_snake_case)]
pub fn AROON(
    high: &[Float],
    low: &[Float],
    timeperiod: usize,
    out_down: &mut [Float],
    out_up: &mut [Float],
) -> Result<OutputRange> {
    let config = AROONConfig::new(timeperiod)?;
    config.compute_into(
        AroonInput { high, low },
        AROONValuesMut {
            down: out_down,
            up: out_up,
        },
    )
}

/// Immutable AROON Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AROONConfig {
    period: usize,
}

impl AROONConfig {
    /// Creates an AROON configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        validate_bounded_period("timeperiod", timeperiod, 2)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for AROONConfig {}

impl IndicatorConfig for AROONConfig {
    type Input<'a> = AroonInput<'a>;
    type Output = AROONValues;
    type OutputMut<'a> = AROONValuesMut<'a>;
    type BatchRunner = AROONBatchRunner;
    type Stream = AROONStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_aroon_input(input, self.period)?;
        let mut values = AROONValues {
            down: vec![0.0 as Float; count],
            up: vec![0.0 as Float; count],
        };
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        let range = aroon_kernel::<false>(
            input,
            self.period,
            count,
            &mut values.down,
            &mut values.up,
            &mut scratch,
        );
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_aroon_input(input, self.period)?;
        validate_two_outputs(
            "AROON",
            "down",
            output.down.len(),
            "up",
            output.up.len(),
            count,
        )?;
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        Ok(aroon_kernel::<false>(
            input,
            self.period,
            count,
            output.down,
            output.up,
            &mut scratch,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(AROONBatchRunner {
            config: *self,
            max_input_len,
            scratch: RangeExtremaScratch::with_capacity(max_input_len),
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(AROONStream {
            period: self.period,
            range: RangeStream::new(self.period + 1),
        })
    }
}

/// Reusable Prepared Batch Runner for AROON.
#[derive(Debug)]
pub struct AROONBatchRunner {
    config: AROONConfig,
    max_input_len: usize,
    scratch: RangeExtremaScratch,
}

impl crate::traits::sealed::Sealed for AROONBatchRunner {}

impl PreparedBatchRunner<AROONConfig> for AROONBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <AROONConfig as IndicatorConfig>::Input<'a>,
        output: <AROONConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        AROONConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (_, count) = validate_aroon_input(input, self.config.period)?;
        validate_two_outputs(
            "AROON",
            "down",
            output.down.len(),
            "up",
            output.up.len(),
            count,
        )?;
        Ok(aroon_kernel::<true>(
            input,
            self.config.period,
            count,
            output.down,
            output.up,
            &mut self.scratch,
        ))
    }
}

/// Independent Streaming Computation state for AROON.
#[derive(Debug, Clone)]
pub struct AROONStream {
    period: usize,
    range: RangeStream,
}

impl crate::traits::sealed::Sealed for AROONStream {}

impl StreamingComputation<AROONConfig> for AROONStream {
    type Tick = AroonTick;
    type TickOutput = AROONValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_aroon_tick(input)?;
        let Some(extrema) = self.range.next(input.low, input.high)? else {
            return Ok(None);
        };
        let source_idx = self.range.seen - 1;
        let factor = 100.0 as Float / self.period as Float;
        Ok(Some(AROONValue {
            down: factor * (self.period - (source_idx - extrema.min_idx)) as Float,
            up: factor * (self.period - (source_idx - extrema.max_idx)) as Float,
        }))
    }

    fn reset(&mut self) {
        self.range.reset();
    }
}

/// Computes the AROON Up-minus-Down oscillator.
#[allow(non_snake_case)]
pub fn AROONOSC(
    high: &[Float],
    low: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let config = AROONOSCConfig::new(timeperiod)?;
    config.compute_into(AroonInput { high, low }, out_real)
}

/// Immutable AROONOSC Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AROONOSCConfig {
    period: usize,
}

impl AROONOSCConfig {
    /// Creates an AROONOSC configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        validate_bounded_period("timeperiod", timeperiod, 2)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for AROONOSCConfig {}

impl IndicatorConfig for AROONOSCConfig {
    type Input<'a> = AroonInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = AROONOSCBatchRunner;
    type Stream = AROONOSCStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_aroon_input(input, self.period)?;
        let mut values = vec![0.0 as Float; count];
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        let range = aroonosc_kernel::<false>(input, self.period, count, &mut values, &mut scratch);
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_aroon_input(input, self.period)?;
        validate_output_len("AROONOSC", output.len(), count)?;
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        Ok(aroonosc_kernel::<false>(
            input,
            self.period,
            count,
            output,
            &mut scratch,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(AROONOSCBatchRunner {
            config: *self,
            max_input_len,
            scratch: RangeExtremaScratch::with_capacity(max_input_len),
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(AROONOSCStream {
            period: self.period,
            range: RangeStream::new(self.period + 1),
        })
    }
}

/// Reusable Prepared Batch Runner for AROONOSC.
#[derive(Debug)]
pub struct AROONOSCBatchRunner {
    config: AROONOSCConfig,
    max_input_len: usize,
    scratch: RangeExtremaScratch,
}

impl crate::traits::sealed::Sealed for AROONOSCBatchRunner {}

impl PreparedBatchRunner<AROONOSCConfig> for AROONOSCBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <AROONOSCConfig as IndicatorConfig>::Input<'a>,
        output: <AROONOSCConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        AROONOSCConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (_, count) = validate_aroon_input(input, self.config.period)?;
        validate_output_len("AROONOSC", output.len(), count)?;
        Ok(aroonosc_kernel::<true>(
            input,
            self.config.period,
            count,
            output,
            &mut self.scratch,
        ))
    }
}

/// Independent Streaming Computation state for AROONOSC.
#[derive(Debug, Clone)]
pub struct AROONOSCStream {
    period: usize,
    range: RangeStream,
}

impl crate::traits::sealed::Sealed for AROONOSCStream {}

impl StreamingComputation<AROONOSCConfig> for AROONOSCStream {
    type Tick = AroonTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_aroon_tick(input)?;
        let Some(extrema) = self.range.next(input.low, input.high)? else {
            return Ok(None);
        };
        let factor = 100.0 as Float / self.period as Float;
        let source_idx = self.range.seen - 1;
        let up = self.period - (source_idx - extrema.max_idx);
        let down = self.period - (source_idx - extrema.min_idx);
        Ok(Some(factor * (up as Float - down as Float)))
    }

    fn reset(&mut self) {
        self.range.reset();
    }
}

#[inline]
fn stochastic_value(close: Float, extrema: RangeValue) -> Float {
    let diff = (extrema.max - extrema.min) / 100.0 as Float;
    if ta_is_zero(diff) {
        0.0 as Float
    } else {
        (close - extrema.min) / diff
    }
}

fn fast_k_kernel<const RESERVED_PUSH: bool>(
    input: StochasticInput<'_>,
    fast_k_period: usize,
    output: &mut [Float],
    scratch: &mut RangeExtremaScratch,
) {
    rolling_range_extrema::<true, RESERVED_PUSH, _>(
        input.low,
        input.high,
        fast_k_period,
        scratch,
        |output_idx, extrema| {
            output[output_idx] = stochastic_value(
                input.close[output_idx + fast_k_period - 1],
                RangeValue {
                    min: extrema.min,
                    max: extrema.max,
                    min_idx: extrema.min_idx,
                    max_idx: extrema.max_idx,
                },
            );
        },
    );
}

#[derive(Debug)]
struct StochasticScratch {
    extrema: RangeExtremaScratch,
    raw_k: Vec<Float>,
    smoothed_k: Vec<Float>,
}

impl StochasticScratch {
    fn with_capacity(max_input_len: usize, needs_smoothed_k: bool) -> Self {
        Self {
            extrema: RangeExtremaScratch::with_capacity(max_input_len),
            raw_k: Vec::with_capacity(max_input_len),
            smoothed_k: if needs_smoothed_k {
                Vec::with_capacity(max_input_len)
            } else {
                Vec::new()
            },
        }
    }
}

/// Named compact STOCHF output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct STOCHFValues {
    /// Fast %K values.
    pub fast_k: Vec<Float>,
    /// Fast %D values.
    pub fast_d: Vec<Float>,
}

impl CompactPayloadLen for STOCHFValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.fast_k.len() != self.fast_d.len() {
            return Err(TalibError::invalid_input(
                "STOCHF Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.fast_k.len())
    }
}

/// Caller-owned compact STOCHF output columns.
#[derive(Debug)]
pub struct STOCHFValuesMut<'a> {
    /// Fast %K output buffer.
    pub fast_k: &'a mut [Float],
    /// Fast %D output buffer.
    pub fast_d: &'a mut [Float],
}

/// One valid streaming STOCHF output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STOCHFValue {
    /// Fast %K value.
    pub fast_k: Float,
    /// Fast %D value.
    pub fast_d: Float,
}

/// Immutable STOCHF Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct STOCHFConfig {
    fast_k_period: usize,
    fast_d: MAConfig,
}

impl STOCHFConfig {
    /// Creates a fast stochastic configuration.
    pub fn new(
        fast_k_period: usize,
        fast_d_period: usize,
        fast_d_ma_type: PeriodMAType,
    ) -> Result<Self> {
        validate_bounded_period("fast_k_period", fast_k_period, 1)?;
        validate_bounded_period("fast_d_period", fast_d_period, 1)?;
        let fast_d = MAConfig::new(fast_d_period, fast_d_ma_type)?;
        Ok(Self {
            fast_k_period,
            fast_d,
        })
    }

    /// Returns the Fast %K Period.
    #[inline]
    pub const fn fast_k_period(&self) -> usize {
        self.fast_k_period
    }

    /// Returns the Fast %D Period.
    #[inline]
    pub const fn fast_d_period(&self) -> usize {
        self.fast_d.period()
    }

    /// Returns the Fast %D Period-based Moving Average kind.
    #[inline]
    pub const fn fast_d_ma_type(&self) -> PeriodMAType {
        self.fast_d.ma_type()
    }
}

impl crate::traits::sealed::Sealed for STOCHFConfig {}

fn stochf_lookback(config: &STOCHFConfig) -> usize {
    config.fast_k_period - 1 + config.fast_d.lookback()
}

fn validate_stochf(input: StochasticInput<'_>, config: &STOCHFConfig) -> Result<(usize, usize)> {
    let lookback = stochf_lookback(config);
    Ok((lookback, validate_stochastic_input(input, lookback)?))
}

fn stochf_kernel<const RESERVED_PUSH: bool>(
    input: StochasticInput<'_>,
    config: &STOCHFConfig,
    count: usize,
    fast_k: &mut [Float],
    fast_d: &mut [Float],
    scratch: &mut StochasticScratch,
    fast_d_runner: Option<&mut MABatchRunner>,
) -> Result<OutputRange> {
    if count == 0 {
        scratch.raw_k.clear();
        return Ok(OutputRange::empty());
    }
    let raw_count = input.high.len() - (config.fast_k_period - 1);
    scratch.raw_k.resize(raw_count, 0.0 as Float);
    fast_k_kernel::<RESERVED_PUSH>(
        input,
        config.fast_k_period,
        &mut scratch.raw_k,
        &mut scratch.extrema,
    );
    if let Some(runner) = fast_d_runner {
        runner.compute_into_bounded(&scratch.raw_k, fast_d)?;
    } else {
        config.fast_d.compute_into(&scratch.raw_k, fast_d)?;
    }
    let offset = config.fast_d.lookback();
    fast_k.copy_from_slice(&scratch.raw_k[offset..offset + count]);
    Ok(OutputRange::new(stochf_lookback(config), count))
}

/// Computes Fast Stochastic %K and %D into equal-length caller-owned columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn STOCHF(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    fast_k_period: usize,
    fast_d_period: usize,
    fast_d_ma_type: PeriodMAType,
    out_fast_k: &mut [Float],
    out_fast_d: &mut [Float],
) -> Result<OutputRange> {
    let config = STOCHFConfig::new(fast_k_period, fast_d_period, fast_d_ma_type)?;
    config.compute_into(
        StochasticInput { high, low, close },
        STOCHFValuesMut {
            fast_k: out_fast_k,
            fast_d: out_fast_d,
        },
    )
}

impl IndicatorConfig for STOCHFConfig {
    type Input<'a> = StochasticInput<'a>;
    type Output = STOCHFValues;
    type OutputMut<'a> = STOCHFValuesMut<'a>;
    type BatchRunner = STOCHFBatchRunner;
    type Stream = STOCHFStream;

    fn lookback(&self) -> usize {
        stochf_lookback(self)
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_stochf(input, self)?;
        let mut values = STOCHFValues {
            fast_k: vec![0.0 as Float; count],
            fast_d: vec![0.0 as Float; count],
        };
        let mut scratch = StochasticScratch::with_capacity(input.high.len(), false);
        let range = stochf_kernel::<false>(
            input,
            self,
            count,
            &mut values.fast_k,
            &mut values.fast_d,
            &mut scratch,
            None,
        )?;
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_stochf(input, self)?;
        validate_two_outputs(
            "STOCHF",
            "fast_k",
            output.fast_k.len(),
            "fast_d",
            output.fast_d.len(),
            count,
        )?;
        let mut scratch = StochasticScratch::with_capacity(input.high.len(), false);
        stochf_kernel::<false>(
            input,
            self,
            count,
            output.fast_k,
            output.fast_d,
            &mut scratch,
            None,
        )
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(STOCHFBatchRunner {
            config: *self,
            max_input_len,
            scratch: StochasticScratch::with_capacity(max_input_len, false),
            fast_d_runner: self.fast_d.prepare_batch(max_input_len)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(STOCHFStream {
            range: RangeStream::new(self.fast_k_period),
            fast_d: self.fast_d.stream()?,
        })
    }
}

/// Reusable Prepared Batch Runner for STOCHF.
#[derive(Debug)]
pub struct STOCHFBatchRunner {
    config: STOCHFConfig,
    max_input_len: usize,
    scratch: StochasticScratch,
    fast_d_runner: MABatchRunner,
}

impl crate::traits::sealed::Sealed for STOCHFBatchRunner {}

impl PreparedBatchRunner<STOCHFConfig> for STOCHFBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <STOCHFConfig as IndicatorConfig>::Input<'a>,
        output: <STOCHFConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        STOCHFConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (_, count) = validate_stochf(input, &self.config)?;
        validate_two_outputs(
            "STOCHF",
            "fast_k",
            output.fast_k.len(),
            "fast_d",
            output.fast_d.len(),
            count,
        )?;
        stochf_kernel::<true>(
            input,
            &self.config,
            count,
            output.fast_k,
            output.fast_d,
            &mut self.scratch,
            Some(&mut self.fast_d_runner),
        )
    }
}

/// Independent Streaming Computation state for STOCHF.
#[derive(Debug, Clone)]
pub struct STOCHFStream {
    range: RangeStream,
    fast_d: MAStream,
}

impl crate::traits::sealed::Sealed for STOCHFStream {}

impl StreamingComputation<STOCHFConfig> for STOCHFStream {
    type Tick = StochasticTick;
    type TickOutput = STOCHFValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_stochastic_tick(input)?;
        let Some(extrema) = self.range.next(input.low, input.high)? else {
            return Ok(None);
        };
        let fast_k = stochastic_value(input.close, extrema);
        let Some(fast_d) = StreamingComputation::<MAConfig>::next(&mut self.fast_d, fast_k)? else {
            return Ok(None);
        };
        Ok(Some(STOCHFValue { fast_k, fast_d }))
    }

    fn reset(&mut self) {
        self.range.reset();
        StreamingComputation::<MAConfig>::reset(&mut self.fast_d);
    }
}

/// Named compact STOCH output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct STOCHValues {
    /// Slow %K values.
    pub slow_k: Vec<Float>,
    /// Slow %D values.
    pub slow_d: Vec<Float>,
}

impl CompactPayloadLen for STOCHValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.slow_k.len() != self.slow_d.len() {
            return Err(TalibError::invalid_input(
                "STOCH Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.slow_k.len())
    }
}

/// Caller-owned compact STOCH output columns.
#[derive(Debug)]
pub struct STOCHValuesMut<'a> {
    /// Slow %K output buffer.
    pub slow_k: &'a mut [Float],
    /// Slow %D output buffer.
    pub slow_d: &'a mut [Float],
}

/// One valid streaming STOCH output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STOCHValue {
    /// Slow %K value.
    pub slow_k: Float,
    /// Slow %D value.
    pub slow_d: Float,
}

/// Immutable STOCH Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct STOCHConfig {
    fast_k_period: usize,
    slow_k: MAConfig,
    slow_d: MAConfig,
}

impl STOCHConfig {
    /// Creates a slow stochastic configuration.
    pub fn new(
        fast_k_period: usize,
        slow_k_period: usize,
        slow_k_ma_type: PeriodMAType,
        slow_d_period: usize,
        slow_d_ma_type: PeriodMAType,
    ) -> Result<Self> {
        validate_bounded_period("fast_k_period", fast_k_period, 1)?;
        validate_bounded_period("slow_k_period", slow_k_period, 1)?;
        let slow_k = MAConfig::new(slow_k_period, slow_k_ma_type)?;
        validate_bounded_period("slow_d_period", slow_d_period, 1)?;
        let slow_d = MAConfig::new(slow_d_period, slow_d_ma_type)?;
        Ok(Self {
            fast_k_period,
            slow_k,
            slow_d,
        })
    }

    /// Returns the Fast %K Period.
    #[inline]
    pub const fn fast_k_period(&self) -> usize {
        self.fast_k_period
    }

    /// Returns the Slow %K Period.
    #[inline]
    pub const fn slow_k_period(&self) -> usize {
        self.slow_k.period()
    }

    /// Returns the Slow %K Period-based Moving Average kind.
    #[inline]
    pub const fn slow_k_ma_type(&self) -> PeriodMAType {
        self.slow_k.ma_type()
    }

    /// Returns the Slow %D Period.
    #[inline]
    pub const fn slow_d_period(&self) -> usize {
        self.slow_d.period()
    }

    /// Returns the Slow %D Period-based Moving Average kind.
    #[inline]
    pub const fn slow_d_ma_type(&self) -> PeriodMAType {
        self.slow_d.ma_type()
    }
}

impl crate::traits::sealed::Sealed for STOCHConfig {}

fn stoch_lookback(config: &STOCHConfig) -> usize {
    config.fast_k_period - 1 + config.slow_k.lookback() + config.slow_d.lookback()
}

fn validate_stoch(input: StochasticInput<'_>, config: &STOCHConfig) -> Result<(usize, usize)> {
    let lookback = stoch_lookback(config);
    Ok((lookback, validate_stochastic_input(input, lookback)?))
}

#[allow(clippy::too_many_arguments)]
fn stoch_kernel<const RESERVED_PUSH: bool>(
    input: StochasticInput<'_>,
    config: &STOCHConfig,
    count: usize,
    slow_k: &mut [Float],
    slow_d: &mut [Float],
    scratch: &mut StochasticScratch,
    slow_k_runner: Option<&mut MABatchRunner>,
    slow_d_runner: Option<&mut MABatchRunner>,
) -> Result<OutputRange> {
    if count == 0 {
        scratch.raw_k.clear();
        scratch.smoothed_k.clear();
        return Ok(OutputRange::empty());
    }
    let raw_count = input.high.len() - (config.fast_k_period - 1);
    scratch.raw_k.resize(raw_count, 0.0 as Float);
    fast_k_kernel::<RESERVED_PUSH>(
        input,
        config.fast_k_period,
        &mut scratch.raw_k,
        &mut scratch.extrema,
    );
    let smoothed_count = raw_count - config.slow_k.lookback();
    scratch.smoothed_k.resize(smoothed_count, 0.0 as Float);
    if let Some(runner) = slow_k_runner {
        runner.compute_into_bounded(&scratch.raw_k, &mut scratch.smoothed_k)?;
    } else {
        config
            .slow_k
            .compute_into(&scratch.raw_k, &mut scratch.smoothed_k)?;
    }
    if let Some(runner) = slow_d_runner {
        runner.compute_into_bounded(&scratch.smoothed_k, slow_d)?;
    } else {
        config.slow_d.compute_into(&scratch.smoothed_k, slow_d)?;
    }
    let offset = config.slow_d.lookback();
    slow_k.copy_from_slice(&scratch.smoothed_k[offset..offset + count]);
    Ok(OutputRange::new(stoch_lookback(config), count))
}

/// Computes Slow Stochastic %K and %D into equal-length caller-owned columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn STOCH(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    fast_k_period: usize,
    slow_k_period: usize,
    slow_k_ma_type: PeriodMAType,
    slow_d_period: usize,
    slow_d_ma_type: PeriodMAType,
    out_slow_k: &mut [Float],
    out_slow_d: &mut [Float],
) -> Result<OutputRange> {
    let config = STOCHConfig::new(
        fast_k_period,
        slow_k_period,
        slow_k_ma_type,
        slow_d_period,
        slow_d_ma_type,
    )?;
    config.compute_into(
        StochasticInput { high, low, close },
        STOCHValuesMut {
            slow_k: out_slow_k,
            slow_d: out_slow_d,
        },
    )
}

impl IndicatorConfig for STOCHConfig {
    type Input<'a> = StochasticInput<'a>;
    type Output = STOCHValues;
    type OutputMut<'a> = STOCHValuesMut<'a>;
    type BatchRunner = STOCHBatchRunner;
    type Stream = STOCHStream;

    fn lookback(&self) -> usize {
        stoch_lookback(self)
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_stoch(input, self)?;
        let mut values = STOCHValues {
            slow_k: vec![0.0 as Float; count],
            slow_d: vec![0.0 as Float; count],
        };
        let mut scratch = StochasticScratch::with_capacity(input.high.len(), true);
        let range = stoch_kernel::<false>(
            input,
            self,
            count,
            &mut values.slow_k,
            &mut values.slow_d,
            &mut scratch,
            None,
            None,
        )?;
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_stoch(input, self)?;
        validate_two_outputs(
            "STOCH",
            "slow_k",
            output.slow_k.len(),
            "slow_d",
            output.slow_d.len(),
            count,
        )?;
        let mut scratch = StochasticScratch::with_capacity(input.high.len(), true);
        stoch_kernel::<false>(
            input,
            self,
            count,
            output.slow_k,
            output.slow_d,
            &mut scratch,
            None,
            None,
        )
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(STOCHBatchRunner {
            config: *self,
            max_input_len,
            scratch: StochasticScratch::with_capacity(max_input_len, true),
            slow_k_runner: self.slow_k.prepare_batch(max_input_len)?,
            slow_d_runner: self.slow_d.prepare_batch(max_input_len)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(STOCHStream {
            range: RangeStream::new(self.fast_k_period),
            slow_k: self.slow_k.stream()?,
            slow_d: self.slow_d.stream()?,
        })
    }
}

/// Reusable Prepared Batch Runner for STOCH.
#[derive(Debug)]
pub struct STOCHBatchRunner {
    config: STOCHConfig,
    max_input_len: usize,
    scratch: StochasticScratch,
    slow_k_runner: MABatchRunner,
    slow_d_runner: MABatchRunner,
}

impl crate::traits::sealed::Sealed for STOCHBatchRunner {}

impl PreparedBatchRunner<STOCHConfig> for STOCHBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <STOCHConfig as IndicatorConfig>::Input<'a>,
        output: <STOCHConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        STOCHConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (_, count) = validate_stoch(input, &self.config)?;
        validate_two_outputs(
            "STOCH",
            "slow_k",
            output.slow_k.len(),
            "slow_d",
            output.slow_d.len(),
            count,
        )?;
        stoch_kernel::<true>(
            input,
            &self.config,
            count,
            output.slow_k,
            output.slow_d,
            &mut self.scratch,
            Some(&mut self.slow_k_runner),
            Some(&mut self.slow_d_runner),
        )
    }
}

/// Independent Streaming Computation state for STOCH.
#[derive(Debug, Clone)]
pub struct STOCHStream {
    range: RangeStream,
    slow_k: MAStream,
    slow_d: MAStream,
}

impl crate::traits::sealed::Sealed for STOCHStream {}

impl StreamingComputation<STOCHConfig> for STOCHStream {
    type Tick = StochasticTick;
    type TickOutput = STOCHValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_stochastic_tick(input)?;
        let Some(extrema) = self.range.next(input.low, input.high)? else {
            return Ok(None);
        };
        let fast_k = stochastic_value(input.close, extrema);
        let Some(slow_k) = StreamingComputation::<MAConfig>::next(&mut self.slow_k, fast_k)? else {
            return Ok(None);
        };
        let Some(slow_d) = StreamingComputation::<MAConfig>::next(&mut self.slow_d, slow_k)? else {
            return Ok(None);
        };
        Ok(Some(STOCHValue { slow_k, slow_d }))
    }

    fn reset(&mut self) {
        self.range.reset();
        StreamingComputation::<MAConfig>::reset(&mut self.slow_k);
        StreamingComputation::<MAConfig>::reset(&mut self.slow_d);
    }
}

/// Named compact STOCHRSI output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct STOCHRSIValues {
    /// Fast stochastic RSI %K values.
    pub fast_k: Vec<Float>,
    /// Smoothed stochastic RSI %D values.
    pub fast_d: Vec<Float>,
}

impl CompactPayloadLen for STOCHRSIValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.fast_k.len() != self.fast_d.len() {
            return Err(TalibError::invalid_input(
                "STOCHRSI Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.fast_k.len())
    }
}

/// Caller-owned compact STOCHRSI output columns.
#[derive(Debug)]
pub struct STOCHRSIValuesMut<'a> {
    /// Fast stochastic RSI %K output buffer.
    pub fast_k: &'a mut [Float],
    /// Smoothed stochastic RSI %D output buffer.
    pub fast_d: &'a mut [Float],
}

/// One valid streaming STOCHRSI output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct STOCHRSIValue {
    /// Fast stochastic RSI %K value.
    pub fast_k: Float,
    /// Smoothed stochastic RSI %D value.
    pub fast_d: Float,
}

/// Immutable STOCHRSI Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct STOCHRSIConfig {
    rsi: RSIConfig,
    fast_k_period: usize,
    fast_d: MAConfig,
}

impl STOCHRSIConfig {
    /// Creates a stochastic RSI configuration.
    pub fn new(
        rsi_period: usize,
        fast_k_period: usize,
        fast_d_period: usize,
        fast_d_ma_type: PeriodMAType,
    ) -> Result<Self> {
        let rsi = RSIConfig::new(rsi_period)?;
        validate_bounded_period("fast_k_period", fast_k_period, 1)?;
        validate_bounded_period("fast_d_period", fast_d_period, 1)?;
        let fast_d = MAConfig::new(fast_d_period, fast_d_ma_type)?;
        Ok(Self {
            rsi,
            fast_k_period,
            fast_d,
        })
    }

    /// Returns the RSI Period.
    #[inline]
    pub const fn rsi_period(&self) -> usize {
        self.rsi.period()
    }

    /// Returns the Fast %K Period.
    #[inline]
    pub const fn fast_k_period(&self) -> usize {
        self.fast_k_period
    }

    /// Returns the Fast %D Period.
    #[inline]
    pub const fn fast_d_period(&self) -> usize {
        self.fast_d.period()
    }

    /// Returns the Fast %D Period-based Moving Average kind.
    #[inline]
    pub const fn fast_d_ma_type(&self) -> PeriodMAType {
        self.fast_d.ma_type()
    }
}

impl crate::traits::sealed::Sealed for STOCHRSIConfig {}

fn stochrsi_lookback(config: &STOCHRSIConfig) -> usize {
    config.rsi.lookback() + config.fast_k_period - 1 + config.fast_d.lookback()
}

fn validate_stochrsi(real: &[Float], config: &STOCHRSIConfig) -> Result<(usize, usize)> {
    validate_finite_slice("real", real)?;
    let lookback = stochrsi_lookback(config);
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[derive(Debug)]
struct STOCHRSIScratch {
    rsi: Vec<Float>,
    stochastic: StochasticScratch,
}

impl STOCHRSIScratch {
    fn with_capacity(max_input_len: usize) -> Self {
        Self {
            rsi: Vec::with_capacity(max_input_len),
            stochastic: StochasticScratch::with_capacity(max_input_len, false),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn stochrsi_kernel<const RESERVED_PUSH: bool>(
    real: &[Float],
    config: &STOCHRSIConfig,
    count: usize,
    fast_k: &mut [Float],
    fast_d: &mut [Float],
    scratch: &mut STOCHRSIScratch,
    rsi_runner: Option<&mut RSIBatchRunner>,
    fast_d_runner: Option<&mut MABatchRunner>,
) -> Result<OutputRange> {
    if count == 0 {
        scratch.rsi.clear();
        scratch.stochastic.raw_k.clear();
        return Ok(OutputRange::empty());
    }
    let rsi_count = real.len() - config.rsi.lookback();
    scratch.rsi.resize(rsi_count, 0.0 as Float);
    if let Some(runner) = rsi_runner {
        runner.compute_into_bounded(real, &mut scratch.rsi)?;
    } else {
        config.rsi.compute_into(real, &mut scratch.rsi)?;
    }

    let raw_count = rsi_count - (config.fast_k_period - 1);
    scratch.stochastic.raw_k.resize(raw_count, 0.0 as Float);
    let rsi_input = StochasticInput {
        high: &scratch.rsi,
        low: &scratch.rsi,
        close: &scratch.rsi,
    };
    fast_k_kernel::<RESERVED_PUSH>(
        rsi_input,
        config.fast_k_period,
        &mut scratch.stochastic.raw_k,
        &mut scratch.stochastic.extrema,
    );
    if let Some(runner) = fast_d_runner {
        runner.compute_into_bounded(&scratch.stochastic.raw_k, fast_d)?;
    } else {
        config
            .fast_d
            .compute_into(&scratch.stochastic.raw_k, fast_d)?;
    }
    let offset = config.fast_d.lookback();
    fast_k.copy_from_slice(&scratch.stochastic.raw_k[offset..offset + count]);
    Ok(OutputRange::new(stochrsi_lookback(config), count))
}

/// Computes Stochastic RSI %K and %D into equal-length caller-owned columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn STOCHRSI(
    real: &[Float],
    rsi_period: usize,
    fast_k_period: usize,
    fast_d_period: usize,
    fast_d_ma_type: PeriodMAType,
    out_fast_k: &mut [Float],
    out_fast_d: &mut [Float],
) -> Result<OutputRange> {
    let config = STOCHRSIConfig::new(rsi_period, fast_k_period, fast_d_period, fast_d_ma_type)?;
    config.compute_into(
        real,
        STOCHRSIValuesMut {
            fast_k: out_fast_k,
            fast_d: out_fast_d,
        },
    )
}

impl IndicatorConfig for STOCHRSIConfig {
    type Input<'a> = &'a [Float];
    type Output = STOCHRSIValues;
    type OutputMut<'a> = STOCHRSIValuesMut<'a>;
    type BatchRunner = STOCHRSIBatchRunner;
    type Stream = STOCHRSIStream;

    fn lookback(&self) -> usize {
        stochrsi_lookback(self)
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_stochrsi(input, self)?;
        let mut values = STOCHRSIValues {
            fast_k: vec![0.0 as Float; count],
            fast_d: vec![0.0 as Float; count],
        };
        let mut scratch = STOCHRSIScratch::with_capacity(input.len());
        let range = stochrsi_kernel::<false>(
            input,
            self,
            count,
            &mut values.fast_k,
            &mut values.fast_d,
            &mut scratch,
            None,
            None,
        )?;
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_stochrsi(input, self)?;
        validate_two_outputs(
            "STOCHRSI",
            "fast_k",
            output.fast_k.len(),
            "fast_d",
            output.fast_d.len(),
            count,
        )?;
        let mut scratch = STOCHRSIScratch::with_capacity(input.len());
        stochrsi_kernel::<false>(
            input,
            self,
            count,
            output.fast_k,
            output.fast_d,
            &mut scratch,
            None,
            None,
        )
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(STOCHRSIBatchRunner {
            config: *self,
            max_input_len,
            scratch: STOCHRSIScratch::with_capacity(max_input_len),
            rsi_runner: self.rsi.prepare_batch(max_input_len)?,
            fast_d_runner: self.fast_d.prepare_batch(max_input_len)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(STOCHRSIStream {
            rsi: self.rsi.stream()?,
            range: RangeStream::new(self.fast_k_period),
            fast_d: self.fast_d.stream()?,
        })
    }
}

/// Reusable Prepared Batch Runner for STOCHRSI.
#[derive(Debug)]
pub struct STOCHRSIBatchRunner {
    config: STOCHRSIConfig,
    max_input_len: usize,
    scratch: STOCHRSIScratch,
    rsi_runner: RSIBatchRunner,
    fast_d_runner: MABatchRunner,
}

impl crate::traits::sealed::Sealed for STOCHRSIBatchRunner {}

impl PreparedBatchRunner<STOCHRSIConfig> for STOCHRSIBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <STOCHRSIConfig as IndicatorConfig>::Input<'a>,
        output: <STOCHRSIConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        STOCHRSIConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let (_, count) = validate_stochrsi(input, &self.config)?;
        validate_two_outputs(
            "STOCHRSI",
            "fast_k",
            output.fast_k.len(),
            "fast_d",
            output.fast_d.len(),
            count,
        )?;
        stochrsi_kernel::<true>(
            input,
            &self.config,
            count,
            output.fast_k,
            output.fast_d,
            &mut self.scratch,
            Some(&mut self.rsi_runner),
            Some(&mut self.fast_d_runner),
        )
    }
}

/// Independent Streaming Computation state for STOCHRSI.
#[derive(Debug, Clone)]
pub struct STOCHRSIStream {
    rsi: RSIStream,
    range: RangeStream,
    fast_d: MAStream,
}

impl crate::traits::sealed::Sealed for STOCHRSIStream {}

impl StreamingComputation<STOCHRSIConfig> for STOCHRSIStream {
    type Tick = Float;
    type TickOutput = STOCHRSIValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", 0, input)?;
        let Some(rsi) = StreamingComputation::<RSIConfig>::next(&mut self.rsi, input)? else {
            return Ok(None);
        };
        let Some(extrema) = self.range.next(rsi, rsi)? else {
            return Ok(None);
        };
        let fast_k = stochastic_value(rsi, extrema);
        let Some(fast_d) = StreamingComputation::<MAConfig>::next(&mut self.fast_d, fast_k)? else {
            return Ok(None);
        };
        Ok(Some(STOCHRSIValue { fast_k, fast_d }))
    }

    fn reset(&mut self) {
        StreamingComputation::<RSIConfig>::reset(&mut self.rsi);
        self.range.reset();
        StreamingComputation::<MAConfig>::reset(&mut self.fast_d);
    }
}

#[inline]
fn willr_value(close: Float, extrema: RangeValue) -> Float {
    let denominator = extrema.max - extrema.min;
    if ta_is_zero(denominator) {
        0.0 as Float
    } else {
        -100.0 as Float * (extrema.max - close) / denominator
    }
}

fn validate_willr(input: StochasticInput<'_>, period: usize) -> Result<(usize, usize)> {
    validate_bounded_period("timeperiod", period, 2)?;
    let lookback = period - 1;
    Ok((lookback, validate_stochastic_input(input, lookback)?))
}

fn willr_kernel<const RESERVED_PUSH: bool>(
    input: StochasticInput<'_>,
    period: usize,
    count: usize,
    output: &mut [Float],
    scratch: &mut RangeExtremaScratch,
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }
    rolling_range_extrema::<true, RESERVED_PUSH, _>(
        input.low,
        input.high,
        period,
        scratch,
        |output_idx, extrema| {
            output[output_idx] = willr_value(
                input.close[output_idx + period - 1],
                RangeValue {
                    min: extrema.min,
                    max: extrema.max,
                    min_idx: extrema.min_idx,
                    max_idx: extrema.max_idx,
                },
            );
        },
    );
    OutputRange::new(period - 1, count)
}

/// Computes Williams' %R into caller-owned Compact Output.
#[allow(non_snake_case)]
pub fn WILLR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let config = WILLRConfig::new(timeperiod)?;
    config.compute_into(StochasticInput { high, low, close }, out_real)
}

/// Immutable WILLR Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WILLRConfig {
    period: usize,
}

impl WILLRConfig {
    /// Creates a Williams' %R configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        validate_bounded_period("timeperiod", timeperiod, 2)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for WILLRConfig {}

impl IndicatorConfig for WILLRConfig {
    type Input<'a> = StochasticInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = WILLRBatchRunner;
    type Stream = WILLRStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (_, count) = validate_willr(input, self.period)?;
        let mut values = vec![0.0 as Float; count];
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        let range = willr_kernel::<false>(input, self.period, count, &mut values, &mut scratch);
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (_, count) = validate_willr(input, self.period)?;
        validate_output_len("WILLR", output.len(), count)?;
        let mut scratch = RangeExtremaScratch::with_capacity(input.high.len());
        Ok(willr_kernel::<false>(
            input,
            self.period,
            count,
            output,
            &mut scratch,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(WILLRBatchRunner {
            config: *self,
            max_input_len,
            scratch: RangeExtremaScratch::with_capacity(max_input_len),
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(WILLRStream {
            range: RangeStream::new(self.period),
        })
    }
}

/// Reusable Prepared Batch Runner for WILLR.
#[derive(Debug)]
pub struct WILLRBatchRunner {
    config: WILLRConfig,
    max_input_len: usize,
    scratch: RangeExtremaScratch,
}

impl crate::traits::sealed::Sealed for WILLRBatchRunner {}

impl PreparedBatchRunner<WILLRConfig> for WILLRBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <WILLRConfig as IndicatorConfig>::Input<'a>,
        output: <WILLRConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        WILLRConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (_, count) = validate_willr(input, self.config.period)?;
        validate_output_len("WILLR", output.len(), count)?;
        Ok(willr_kernel::<true>(
            input,
            self.config.period,
            count,
            output,
            &mut self.scratch,
        ))
    }
}

/// Independent Streaming Computation state for WILLR.
#[derive(Debug, Clone)]
pub struct WILLRStream {
    range: RangeStream,
}

impl crate::traits::sealed::Sealed for WILLRStream {}

impl StreamingComputation<WILLRConfig> for WILLRStream {
    type Tick = StochasticTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_stochastic_tick(input)?;
        Ok(self
            .range
            .next(input.low, input.high)?
            .map(|extrema| willr_value(input.close, extrema)))
    }

    fn reset(&mut self) {
        self.range.reset();
    }
}
