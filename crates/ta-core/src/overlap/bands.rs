//! Acceleration Bands (ACCBANDS) and Bollinger Bands (BBANDS).

use super::{MABatchRunner, MAConfig, PeriodMAType, MA};
use crate::common::{validate_finite_value, CompactPayloadLen};
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

/// Default number of standard deviations above the BBANDS middle band.
pub const BBANDS_DEFAULT_NBDEV_UP: Float = 2.0 as Float;
/// Default number of standard deviations below the BBANDS middle band.
pub const BBANDS_DEFAULT_NBDEV_DOWN: Float = 2.0 as Float;

#[inline]
fn bands_period_lookback(timeperiod: usize) -> Result<usize> {
    if !(2..=MAX_PERIOD).contains(&timeperiod) {
        return Err(TalibError::invalid_period(
            timeperiod,
            format!("timeperiod must be in 2..={MAX_PERIOD}"),
        ));
    }
    Ok(timeperiod - 1)
}

fn validate_band_outputs(
    name: &str,
    upper_len: usize,
    middle_len: usize,
    lower_len: usize,
    count: usize,
) -> Result<()> {
    validate_all_same_len(&[
        ("upper", upper_len),
        ("middle", middle_len),
        ("lower", lower_len),
    ])?;
    validate_output_len(name, upper_len, count)
}

#[inline]
fn acceleration_values(high: Float, low: Float) -> (Float, Float) {
    let denominator = high + low;
    if denominator > -TA_EPSILON && denominator < TA_EPSILON {
        (high, low)
    } else {
        let width = 4.0 as Float * (high - low) / denominator;
        (high * (1.0 as Float + width), low * (1.0 as Float - width))
    }
}

/// Named ACCBANDS Compact Output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct ACCBANDSValues {
    /// Upper Acceleration Band.
    pub upper: Vec<Float>,
    /// Simple moving average of close.
    pub middle: Vec<Float>,
    /// Lower Acceleration Band.
    pub lower: Vec<Float>,
}

impl CompactPayloadLen for ACCBANDSValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.upper.len() != self.middle.len() || self.upper.len() != self.lower.len() {
            return Err(TalibError::invalid_input(
                "ACCBANDS Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.upper.len())
    }
}

/// Caller-owned ACCBANDS compact columns.
#[derive(Debug)]
pub struct ACCBANDSValuesMut<'a> {
    /// Upper-band output buffer.
    pub upper: &'a mut [Float],
    /// Middle-band output buffer.
    pub middle: &'a mut [Float],
    /// Lower-band output buffer.
    pub lower: &'a mut [Float],
}

/// Borrowed aligned high/low/close Observation Series for ACCBANDS.
#[derive(Debug, Clone, Copy)]
pub struct ACCBANDSInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One aligned high/low/close ACCBANDS tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ACCBANDSTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// One valid streaming ACCBANDS output.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ACCBANDSValue {
    /// Upper Acceleration Band.
    pub upper: Float,
    /// Simple moving average of close.
    pub middle: Float,
    /// Lower Acceleration Band.
    pub lower: Float,
}

fn validate_accbands(input: ACCBANDSInput<'_>, timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = bands_period_lookback(timeperiod)?;
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
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count))
}

#[allow(clippy::too_many_arguments)]
fn accbands_kernel(
    input: ACCBANDSInput<'_>,
    timeperiod: usize,
    lookback: usize,
    count: usize,
    upper: &mut [Float],
    middle: &mut [Float],
    lower: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut upper_sum = 0.0 as Float;
    let mut middle_sum = 0.0 as Float;
    let mut lower_sum = 0.0 as Float;
    for idx in 0..lookback {
        let (upper_value, lower_value) = acceleration_values(input.high[idx], input.low[idx]);
        upper_sum += upper_value;
        middle_sum += input.close[idx];
        lower_sum += lower_value;
    }

    let divisor = timeperiod as Float;
    for source_idx in lookback..input.high.len() {
        let (upper_value, lower_value) =
            acceleration_values(input.high[source_idx], input.low[source_idx]);
        upper_sum += upper_value;
        middle_sum += input.close[source_idx];
        lower_sum += lower_value;

        let output_idx = source_idx - lookback;
        upper[output_idx] = upper_sum / divisor;
        middle[output_idx] = middle_sum / divisor;
        lower[output_idx] = lower_sum / divisor;

        let trailing_idx = output_idx;
        let (trailing_upper, trailing_lower) =
            acceleration_values(input.high[trailing_idx], input.low[trailing_idx]);
        upper_sum -= trailing_upper;
        middle_sum -= input.close[trailing_idx];
        lower_sum -= trailing_lower;
    }

    OutputRange::new(lookback, count)
}

/// Computes Acceleration Bands into three equal-length caller-owned columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn ACCBANDS(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_upper: &mut [Float],
    out_middle: &mut [Float],
    out_lower: &mut [Float],
) -> Result<OutputRange> {
    let input = ACCBANDSInput { high, low, close };
    let (lookback, count) = validate_accbands(input, timeperiod)?;
    validate_band_outputs(
        "ACCBANDS",
        out_upper.len(),
        out_middle.len(),
        out_lower.len(),
        count,
    )?;
    Ok(accbands_kernel(
        input, timeperiod, lookback, count, out_upper, out_middle, out_lower,
    ))
}

/// Immutable Acceleration Bands Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ACCBANDSConfig {
    period: usize,
}

impl ACCBANDSConfig {
    /// Creates an ACCBANDS configuration for a Period in `2..=100_000`.
    pub fn new(timeperiod: usize) -> Result<Self> {
        bands_period_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for ACCBANDSConfig {}

impl IndicatorConfig for ACCBANDSConfig {
    type Input<'a> = ACCBANDSInput<'a>;
    type Output = ACCBANDSValues;
    type OutputMut<'a> = ACCBANDSValuesMut<'a>;
    type BatchRunner = ACCBANDSBatchRunner;
    type Stream = ACCBANDSStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_accbands(input, self.period)?;
        let mut values = ACCBANDSValues {
            upper: vec![0.0 as Float; count],
            middle: vec![0.0 as Float; count],
            lower: vec![0.0 as Float; count],
        };
        let range = accbands_kernel(
            input,
            self.period,
            lookback,
            count,
            &mut values.upper,
            &mut values.middle,
            &mut values.lower,
        );
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_accbands(input, self.period)?;
        validate_band_outputs(
            "ACCBANDS",
            output.upper.len(),
            output.middle.len(),
            output.lower.len(),
            count,
        )?;
        Ok(accbands_kernel(
            input,
            self.period,
            lookback,
            count,
            output.upper,
            output.middle,
            output.lower,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(ACCBANDSBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        ACCBANDSStream::new(self.period)
    }
}

/// Reusable Prepared Batch Runner for ACCBANDS.
#[derive(Debug, Clone)]
pub struct ACCBANDSBatchRunner {
    config: ACCBANDSConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for ACCBANDSBatchRunner {}

impl PreparedBatchRunner<ACCBANDSConfig> for ACCBANDSBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <ACCBANDSConfig as IndicatorConfig>::Input<'a>,
        output: <ACCBANDSConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        ACCBANDSConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for ACCBANDS.
#[derive(Debug, Clone)]
pub struct ACCBANDSStream {
    period: usize,
    trailing: Vec<ACCBANDSValue>,
    index: usize,
    count: usize,
    upper_sum: Float,
    middle_sum: Float,
    lower_sum: Float,
}

impl ACCBANDSStream {
    fn new(period: usize) -> Result<Self> {
        bands_period_lookback(period)?;
        Ok(Self {
            period,
            trailing: vec![ACCBANDSValue::default(); period - 1],
            index: 0,
            count: 0,
            upper_sum: 0.0 as Float,
            middle_sum: 0.0 as Float,
            lower_sum: 0.0 as Float,
        })
    }
}

impl crate::traits::sealed::Sealed for ACCBANDSStream {}

impl StreamingComputation<ACCBANDSConfig> for ACCBANDSStream {
    type Tick = ACCBANDSTick;
    type TickOutput = ACCBANDSValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("high", 0, input.high)?;
        validate_finite_value("low", 0, input.low)?;
        validate_finite_value("close", 0, input.close)?;
        let (upper, lower) = acceleration_values(input.high, input.low);
        let incoming = ACCBANDSValue {
            upper,
            middle: input.close,
            lower,
        };

        self.upper_sum += incoming.upper;
        self.middle_sum += incoming.middle;
        self.lower_sum += incoming.lower;
        if self.count < self.period {
            self.count += 1;
        }
        if self.count < self.period {
            self.trailing[self.index] = incoming;
            self.index = (self.index + 1) % self.trailing.len();
            return Ok(None);
        }

        let divisor = self.period as Float;
        let output = ACCBANDSValue {
            upper: self.upper_sum / divisor,
            middle: self.middle_sum / divisor,
            lower: self.lower_sum / divisor,
        };
        let outgoing = self.trailing[self.index];
        self.upper_sum -= outgoing.upper;
        self.middle_sum -= outgoing.middle;
        self.lower_sum -= outgoing.lower;
        self.trailing[self.index] = incoming;
        self.index = (self.index + 1) % self.trailing.len();
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.trailing.fill(ACCBANDSValue::default());
        self.index = 0;
        self.count = 0;
        self.upper_sum = 0.0 as Float;
        self.middle_sum = 0.0 as Float;
        self.lower_sum = 0.0 as Float;
    }
}

/// Named BBANDS Compact Output columns.
#[derive(Debug, Clone, PartialEq)]
pub struct BBANDSValues {
    /// Upper Bollinger Band.
    pub upper: Vec<Float>,
    /// Selected Period-based moving average.
    pub middle: Vec<Float>,
    /// Lower Bollinger Band.
    pub lower: Vec<Float>,
}

impl CompactPayloadLen for BBANDSValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.upper.len() != self.middle.len() || self.upper.len() != self.lower.len() {
            return Err(TalibError::invalid_input(
                "BBANDS Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.upper.len())
    }
}

/// Caller-owned BBANDS compact columns.
#[derive(Debug)]
pub struct BBANDSValuesMut<'a> {
    /// Upper-band output buffer.
    pub upper: &'a mut [Float],
    /// Middle-band output buffer.
    pub middle: &'a mut [Float],
    /// Lower-band output buffer.
    pub lower: &'a mut [Float],
}

/// One valid streaming BBANDS output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBANDSValue {
    /// Upper Bollinger Band.
    pub upper: Float,
    /// Selected Period-based moving average.
    pub middle: Float,
    /// Lower Bollinger Band.
    pub lower: Float,
}

fn validate_deviation(name: &str, value: Float) -> Result<()> {
    if !value.is_finite() {
        return Err(TalibError::invalid_parameter(
            format!("{name}"),
            format!("{value}"),
            format!("finite number"),
        ));
    }
    Ok(())
}

fn validate_bbands(real: &[Float], config: &BBANDSConfig) -> Result<(usize, usize)> {
    validate_finite_slice("real", real)?;
    let lookback = config.ma_config.lookback().max(config.period - 1);
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[inline]
fn standard_deviation(variance: Float) -> Float {
    if variance <= TA_EPSILON {
        0.0 as Float
    } else {
        variance.sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_bbands_from_middle(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    nbdev_up: Float,
    nbdev_down: Float,
    upper: &mut [Float],
    middle: &[Float],
    lower: &mut [Float],
) {
    if count == 0 {
        return;
    }

    let first_window = lookback + 1 - period;
    let mut sum = 0.0 as Float;
    let mut sum_sq = 0.0 as Float;
    for &value in &real[first_window..=lookback] {
        sum += value;
        sum_sq += value * value;
    }
    let divisor = period as Float;

    for output_idx in 0..count {
        let mean = sum / divisor;
        let deviation = standard_deviation(sum_sq / divisor - mean * mean);
        upper[output_idx] = middle[output_idx] + nbdev_up * deviation;
        lower[output_idx] = middle[output_idx] - nbdev_down * deviation;

        if output_idx + 1 < count {
            let old_idx = first_window + output_idx;
            let new_idx = lookback + output_idx + 1;
            let old = real[old_idx];
            let new = real[new_idx];
            sum -= old;
            sum_sq -= old * old;
            sum += new;
            sum_sq += new * new;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bbands_kernel(
    real: &[Float],
    config: &BBANDSConfig,
    lookback: usize,
    count: usize,
    upper: &mut [Float],
    middle: &mut [Float],
    lower: &mut [Float],
) -> Result<OutputRange> {
    let range = MA(real, config.period, config.ma_type(), middle)?;
    debug_assert_eq!(
        range,
        if count == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(lookback, count)
        }
    );
    write_bbands_from_middle(
        real,
        config.period,
        lookback,
        count,
        config.nbdev_up,
        config.nbdev_down,
        upper,
        middle,
        lower,
    );
    Ok(range)
}

/// Computes Bollinger Bands around the selected Period-based moving average.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn BBANDS(
    real: &[Float],
    timeperiod: usize,
    nbdev_up: Float,
    nbdev_down: Float,
    ma_type: PeriodMAType,
    out_upper: &mut [Float],
    out_middle: &mut [Float],
    out_lower: &mut [Float],
) -> Result<OutputRange> {
    let config = BBANDSConfig::new(timeperiod, nbdev_up, nbdev_down, ma_type)?;
    let (lookback, count) = validate_bbands(real, &config)?;
    validate_band_outputs(
        "BBANDS",
        out_upper.len(),
        out_middle.len(),
        out_lower.len(),
        count,
    )?;
    bbands_kernel(
        real, &config, lookback, count, out_upper, out_middle, out_lower,
    )
}

/// Immutable Bollinger Bands Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBANDSConfig {
    period: usize,
    nbdev_up: Float,
    nbdev_down: Float,
    ma_config: MAConfig,
}

impl BBANDSConfig {
    /// Creates a Bollinger Bands configuration.
    pub fn new(
        timeperiod: usize,
        nbdev_up: Float,
        nbdev_down: Float,
        ma_type: PeriodMAType,
    ) -> Result<Self> {
        bands_period_lookback(timeperiod)?;
        validate_deviation("nbdev_up", nbdev_up)?;
        validate_deviation("nbdev_down", nbdev_down)?;
        let ma_config = MAConfig::new(timeperiod, ma_type)?;
        Ok(Self {
            period: timeperiod,
            nbdev_up,
            nbdev_down,
            ma_config,
        })
    }

    /// Creates a configuration with two deviations on each side.
    pub fn with_default_deviations(timeperiod: usize, ma_type: PeriodMAType) -> Result<Self> {
        Self::new(
            timeperiod,
            BBANDS_DEFAULT_NBDEV_UP,
            BBANDS_DEFAULT_NBDEV_DOWN,
            ma_type,
        )
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the upper-band deviation multiplier.
    #[inline]
    pub const fn nbdev_up(&self) -> Float {
        self.nbdev_up
    }

    /// Returns the lower-band deviation multiplier.
    #[inline]
    pub const fn nbdev_down(&self) -> Float {
        self.nbdev_down
    }

    /// Returns the selected Period-based moving-average definition.
    #[inline]
    pub const fn ma_type(&self) -> PeriodMAType {
        self.ma_config.ma_type()
    }
}

impl crate::traits::sealed::Sealed for BBANDSConfig {}

impl IndicatorConfig for BBANDSConfig {
    type Input<'a> = &'a [Float];
    type Output = BBANDSValues;
    type OutputMut<'a> = BBANDSValuesMut<'a>;
    type BatchRunner = BBANDSBatchRunner;
    type Stream = BBANDSStream;

    fn lookback(&self) -> usize {
        self.ma_config.lookback().max(self.period - 1)
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_bbands(input, self)?;
        let mut values = BBANDSValues {
            upper: vec![0.0 as Float; count],
            middle: vec![0.0 as Float; count],
            lower: vec![0.0 as Float; count],
        };
        let range = bbands_kernel(
            input,
            self,
            lookback,
            count,
            &mut values.upper,
            &mut values.middle,
            &mut values.lower,
        )?;
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_bbands(input, self)?;
        validate_band_outputs(
            "BBANDS",
            output.upper.len(),
            output.middle.len(),
            output.lower.len(),
            count,
        )?;
        bbands_kernel(
            input,
            self,
            lookback,
            count,
            output.upper,
            output.middle,
            output.lower,
        )
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(BBANDSBatchRunner {
            config: *self,
            max_input_len,
            ma_runner: self.ma_config.prepare_batch(max_input_len)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        BBANDSStream::new(*self)
    }
}

/// Reusable Prepared Batch Runner for BBANDS.
#[derive(Debug, Clone)]
pub struct BBANDSBatchRunner {
    config: BBANDSConfig,
    max_input_len: usize,
    ma_runner: MABatchRunner,
}

impl crate::traits::sealed::Sealed for BBANDSBatchRunner {}

impl PreparedBatchRunner<BBANDSConfig> for BBANDSBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <BBANDSConfig as IndicatorConfig>::Input<'a>,
        output: <BBANDSConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        BBANDSConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let (lookback, count) = validate_bbands(input, &self.config)?;
        validate_band_outputs(
            "BBANDS",
            output.upper.len(),
            output.middle.len(),
            output.lower.len(),
            count,
        )?;

        let range = PreparedBatchRunner::<MAConfig>::compute_into(
            &mut self.ma_runner,
            input,
            &mut *output.middle,
        )?;
        debug_assert_eq!(
            range,
            if count == 0 {
                OutputRange::empty()
            } else {
                OutputRange::new(lookback, count)
            }
        );
        write_bbands_from_middle(
            input,
            self.config.period,
            lookback,
            count,
            self.config.nbdev_up,
            self.config.nbdev_down,
            output.upper,
            &*output.middle,
            output.lower,
        );
        Ok(range)
    }
}

#[derive(Debug, Clone)]
struct RollingDeviation {
    period: usize,
    trailing: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
    sum_sq: Float,
}

impl RollingDeviation {
    fn new(period: usize) -> Self {
        Self {
            period,
            trailing: vec![0.0 as Float; period - 1],
            index: 0,
            count: 0,
            sum: 0.0 as Float,
            sum_sq: 0.0 as Float,
        }
    }

    fn push(&mut self, input: Float) -> Option<Float> {
        self.sum += input;
        self.sum_sq += input * input;
        if self.count < self.period {
            self.count += 1;
        }
        if self.count < self.period {
            self.trailing[self.index] = input;
            self.index = (self.index + 1) % self.trailing.len();
            return None;
        }

        let mean = self.sum / self.period as Float;
        let deviation = standard_deviation(self.sum_sq / self.period as Float - mean * mean);
        let old = self.trailing[self.index];
        self.sum -= old;
        self.sum_sq -= old * old;
        self.trailing[self.index] = input;
        self.index = (self.index + 1) % self.trailing.len();
        Some(deviation)
    }

    fn reset(&mut self) {
        self.trailing.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
        self.sum_sq = 0.0 as Float;
    }
}

/// Independent Streaming Computation state for BBANDS.
#[derive(Debug, Clone)]
pub struct BBANDSStream {
    config: BBANDSConfig,
    ma_stream: super::MAStream,
    deviation: RollingDeviation,
}

impl BBANDSStream {
    fn new(config: BBANDSConfig) -> Result<Self> {
        Ok(Self {
            config,
            ma_stream: config.ma_config.stream()?,
            deviation: RollingDeviation::new(config.period),
        })
    }
}

impl crate::traits::sealed::Sealed for BBANDSStream {}

impl StreamingComputation<BBANDSConfig> for BBANDSStream {
    type Tick = Float;
    type TickOutput = BBANDSValue;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", 0, input)?;
        let middle = StreamingComputation::<MAConfig>::next(&mut self.ma_stream, input)?;
        let deviation = self.deviation.push(input);
        match middle {
            Some(middle) => {
                let deviation = deviation
                    .expect("Period-based MA output cannot precede the Period standard deviation");
                Ok(Some(BBANDSValue {
                    upper: middle + self.config.nbdev_up * deviation,
                    middle,
                    lower: middle - self.config.nbdev_down * deviation,
                }))
            }
            None => Ok(None),
        }
    }

    fn reset(&mut self) {
        StreamingComputation::<MAConfig>::reset(&mut self.ma_stream);
        self.deviation.reset();
    }
}
