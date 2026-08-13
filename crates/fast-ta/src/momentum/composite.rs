//! Composite-input Momentum Indicator Definitions.
//!
//! `BOP` and `CCI` consume OHLC observations, `MFI` consumes OHLCV
//! observations, and `ULTOSC` consumes HLC observations. Every batch input is a
//! typed Structure-of-Arrays view whose columns are validated for source
//! alignment and finite values before output is mutated. Streaming ticks model
//! the same observation shapes and validate before state transitions.

use crate::{
    validate_finite_slices, validate_input_len, validate_output_len, validate_same_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec::Vec};

const TA_EPSILON: Float = 1e-14 as Float;
const MAX_PERIOD: usize = 100_000;
const CCI_SCALE: Float = 0.015 as Float;

#[inline]
fn is_ta_zero(value: Float) -> bool {
    value > -TA_EPSILON && value < TA_EPSILON
}

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
fn typical_price(high: Float, low: Float, close: Float) -> Float {
    (high + low + close) / 3.0 as Float
}

fn validate_observation_columns(columns: &[(&str, &[Float])]) -> Result<usize> {
    let Some(&(first_name, first_values)) = columns.first() else {
        return Ok(0);
    };
    for &(name, values) in &columns[1..] {
        validate_same_len(first_name, first_values.len(), name, values.len())?;
    }
    validate_finite_slices(columns)?;
    Ok(first_values.len())
}

fn validate_tick(columns: &[(&str, Float)]) -> Result<()> {
    for &(name, value) in columns {
        validate_finite_slices(&[(name, &[value])])?;
    }
    Ok(())
}

// Balance of Power ---------------------------------------------------------

/// Borrowed aligned OHLC input for [`BOP`] Batch Computation.
#[derive(Debug, Clone, Copy)]
pub struct BOPInput<'a> {
    /// Open price series.
    pub open: &'a [Float],
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One OHLC tick for [`BOP`] Streaming Computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BOPTick {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

fn validate_bop_input(input: BOPInput<'_>) -> Result<usize> {
    validate_observation_columns(&[
        ("open", input.open),
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])
}

#[inline]
fn bop_value(open: Float, high: Float, low: Float, close: Float) -> Float {
    let range = high - low;
    if range < TA_EPSILON {
        0.0 as Float
    } else {
        (close - open) / range
    }
}

fn bop_kernel(input: BOPInput<'_>, output: &mut [Float]) -> OutputRange {
    for (index, value) in output.iter_mut().enumerate() {
        *value = bop_value(
            input.open[index],
            input.high[index],
            input.low[index],
            input.close[index],
        );
    }
    if output.is_empty() {
        OutputRange::empty()
    } else {
        OutputRange::new(0, output.len())
    }
}

/// Computes Balance of Power into caller-owned Compact Output storage.
#[allow(non_snake_case)]
pub fn BOP(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = BOPInput {
        open,
        high,
        low,
        close,
    };
    let len = validate_bop_input(input)?;
    validate_output_len("BOP", out_real.len(), len)?;
    Ok(bop_kernel(input, &mut out_real[..len]))
}

/// Immutable Balance of Power Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct BOPConfig;

impl BOPConfig {
    /// Creates a Balance of Power configuration.
    pub const fn new() -> Self {
        Self
    }
}

impl crate::traits::sealed::Sealed for BOPConfig {}

impl IndicatorConfig for BOPConfig {
    type Input<'a> = BOPInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = BOPBatchRunner;
    type Stream = BOPStream;

    #[inline]
    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let len = validate_bop_input(input)?;
        let mut values = Vec::with_capacity(len);
        values.resize(len, 0.0 as Float);
        let range = bop_kernel(input, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        BOP(input.open, input.high, input.low, input.close, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(BOPBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(BOPStream)
    }
}

/// Reusable Prepared Batch Runner for Balance of Power.
#[derive(Debug, Clone)]
pub struct BOPBatchRunner {
    config: BOPConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for BOPBatchRunner {}

impl PreparedBatchRunner<BOPConfig> for BOPBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <BOPConfig as IndicatorConfig>::Input<'a>,
        output: <BOPConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        BOPConfig: 'a,
    {
        let actual_input_len = input
            .open
            .len()
            .max(input.high.len())
            .max(input.low.len())
            .max(input.close.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent, stateless Streaming Computation for Balance of Power.
#[derive(Debug, Clone, Copy, Default)]
pub struct BOPStream;

impl crate::traits::sealed::Sealed for BOPStream {}

impl StreamingComputation<BOPConfig> for BOPStream {
    type Tick = BOPTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_tick(&[
            ("open", input.open),
            ("high", input.high),
            ("low", input.low),
            ("close", input.close),
        ])?;
        Ok(Some(bop_value(
            input.open,
            input.high,
            input.low,
            input.close,
        )))
    }

    #[inline]
    fn reset(&mut self) {}
}

// Commodity Channel Index -------------------------------------------------

/// Borrowed aligned open/high/low/close input for [`CCI`] Batch Computation.
#[derive(Debug, Clone, Copy)]
pub struct CCIInput<'a> {
    /// Open price series.
    pub open: &'a [Float],
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One open/high/low/close tick for [`CCI`] Streaming Computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CCITick {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

fn cci_lookback(period: usize) -> Result<usize> {
    validate_bounded_period("timeperiod", period, 2)?;
    Ok(period - 1)
}

fn validate_cci_input(input: CCIInput<'_>, period: usize) -> Result<(usize, usize, usize)> {
    let lookback = cci_lookback(period)?;
    let len = validate_observation_columns(&[
        ("open", input.open),
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    let count = validate_input_len(len, lookback)?;
    Ok((len, lookback, count))
}

#[inline]
fn cci_projection(last: Float, average: Float, deviation_sum: Float, period: usize) -> Float {
    let delta = last - average;
    if delta == 0.0 as Float || deviation_sum == 0.0 as Float {
        0.0 as Float
    } else {
        delta / (CCI_SCALE * (deviation_sum / period as Float))
    }
}

fn cci_kernel(
    input: CCIInput<'_>,
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    for (output_index, value) in output[..count].iter_mut().enumerate() {
        let source_index = lookback + output_index;
        let mut average = 0.0 as Float;
        for physical_slot in 0..period {
            let distance = (source_index % period + period - physical_slot) % period;
            let index = source_index - distance;
            average += typical_price(input.high[index], input.low[index], input.close[index]);
        }
        average /= period as Float;

        let mut deviation_sum = 0.0 as Float;
        for physical_slot in 0..period {
            let distance = (source_index % period + period - physical_slot) % period;
            let index = source_index - distance;
            deviation_sum +=
                (typical_price(input.high[index], input.low[index], input.close[index]) - average)
                    .abs();
        }
        let last = typical_price(
            input.high[source_index],
            input.low[source_index],
            input.close[source_index],
        );
        *value = cci_projection(last, average, deviation_sum, period);
    }

    OutputRange::new(lookback, count)
}

/// Computes Commodity Channel Index into caller-owned Compact Output storage.
#[allow(non_snake_case)]
pub fn CCI(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = CCIInput {
        open,
        high,
        low,
        close,
    };
    let (_, lookback, count) = validate_cci_input(input, timeperiod)?;
    validate_output_len("CCI", out_real.len(), count)?;
    Ok(cci_kernel(input, timeperiod, lookback, count, out_real))
}

/// Immutable Commodity Channel Index Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CCIConfig {
    period: usize,
}

impl CCIConfig {
    /// Creates a Commodity Channel Index configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        cci_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for CCIConfig {}

impl IndicatorConfig for CCIConfig {
    type Input<'a> = CCIInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = CCIBatchRunner;
    type Stream = CCIStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, lookback, count) = validate_cci_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = cci_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        CCI(
            input.open,
            input.high,
            input.low,
            input.close,
            self.period,
            output,
        )
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(CCIBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        let mut buffer = Vec::new();
        buffer.resize(self.period, 0.0 as Float);
        Ok(CCIStream {
            period: self.period,
            buffer,
            index: 0,
            count: 0,
        })
    }
}

/// Reusable Prepared Batch Runner for Commodity Channel Index.
#[derive(Debug, Clone)]
pub struct CCIBatchRunner {
    config: CCIConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for CCIBatchRunner {}

impl PreparedBatchRunner<CCIConfig> for CCIBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <CCIConfig as IndicatorConfig>::Input<'a>,
        output: <CCIConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        CCIConfig: 'a,
    {
        let actual_input_len = input
            .open
            .len()
            .max(input.high.len())
            .max(input.low.len())
            .max(input.close.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for Commodity Channel Index.
#[derive(Debug, Clone)]
pub struct CCIStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl crate::traits::sealed::Sealed for CCIStream {}

impl StreamingComputation<CCIConfig> for CCIStream {
    type Tick = CCITick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_tick(&[
            ("open", input.open),
            ("high", input.high),
            ("low", input.low),
            ("close", input.close),
        ])?;
        let current = typical_price(input.high, input.low, input.close);

        self.buffer[self.index] = current;
        self.index = (self.index + 1) % self.period;
        if self.count < self.period {
            self.count += 1;
            if self.count < self.period {
                return Ok(None);
            }
        }

        let average = self.buffer.iter().copied().sum::<Float>() / self.period as Float;
        let mut deviation_sum = 0.0 as Float;
        for &value in &self.buffer {
            deviation_sum += (value - average).abs();
        }
        Ok(Some(cci_projection(
            current,
            average,
            deviation_sum,
            self.period,
        )))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}

// Money Flow Index --------------------------------------------------------

/// Borrowed aligned open/high/low/close/volume input for [`MFI`] Batch Computation.
#[derive(Debug, Clone, Copy)]
pub struct MFIInput<'a> {
    /// Open price series.
    pub open: &'a [Float],
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One open/high/low/close/volume tick for [`MFI`] Streaming Computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MFITick {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

#[derive(Debug, Clone, Copy, Default)]
struct MoneyFlow {
    positive: Float,
    negative: Float,
}

fn mfi_lookback(period: usize) -> Result<usize> {
    validate_bounded_period("timeperiod", period, 2)?;
    Ok(period)
}

fn validate_mfi_input(input: MFIInput<'_>, period: usize) -> Result<(usize, usize, usize)> {
    let lookback = mfi_lookback(period)?;
    let len = validate_observation_columns(&[
        ("open", input.open),
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
        ("volume", input.volume),
    ])?;
    let count = validate_input_len(len, lookback)?;
    Ok((len, lookback, count))
}

#[inline]
fn directional_money_flow(current: Float, previous: Float, volume: Float) -> MoneyFlow {
    let difference = current - previous;
    if difference == 0.0 as Float {
        MoneyFlow::default()
    } else if difference < 0.0 as Float {
        MoneyFlow {
            positive: 0.0 as Float,
            negative: current * volume,
        }
    } else {
        MoneyFlow {
            positive: current * volume,
            negative: 0.0 as Float,
        }
    }
}

#[inline]
fn mfi_projection(positive: Float, negative: Float) -> Float {
    let total = positive + negative;
    if total < 1.0 as Float {
        0.0 as Float
    } else {
        100.0 as Float * (positive / total)
    }
}

#[inline]
fn money_flow_at(input: MFIInput<'_>, index: usize) -> MoneyFlow {
    let previous = typical_price(
        input.high[index - 1],
        input.low[index - 1],
        input.close[index - 1],
    );
    let current = typical_price(input.high[index], input.low[index], input.close[index]);
    directional_money_flow(current, previous, input.volume[index])
}

fn mfi_kernel(
    input: MFIInput<'_>,
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut positive = 0.0 as Float;
    let mut negative = 0.0 as Float;
    for source_index in 1..input.high.len() {
        if source_index > period {
            let trailing = money_flow_at(input, source_index - period);
            positive -= trailing.positive;
            negative -= trailing.negative;
        }

        let current = money_flow_at(input, source_index);
        positive += current.positive;
        negative += current.negative;

        if source_index >= lookback {
            output[source_index - lookback] = mfi_projection(positive, negative);
        }
    }

    OutputRange::new(lookback, count)
}

/// Computes Money Flow Index into caller-owned Compact Output storage.
#[allow(non_snake_case)]
pub fn MFI(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = MFIInput {
        open,
        high,
        low,
        close,
        volume,
    };
    let (_, lookback, count) = validate_mfi_input(input, timeperiod)?;
    validate_output_len("MFI", out_real.len(), count)?;
    Ok(mfi_kernel(input, timeperiod, lookback, count, out_real))
}

/// Immutable Money Flow Index Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MFIConfig {
    period: usize,
}

impl MFIConfig {
    /// Creates a Money Flow Index configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        mfi_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for MFIConfig {}

impl IndicatorConfig for MFIConfig {
    type Input<'a> = MFIInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MFIBatchRunner;
    type Stream = MFIStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, lookback, count) = validate_mfi_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = mfi_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MFI(
            input.open,
            input.high,
            input.low,
            input.close,
            input.volume,
            self.period,
            output,
        )
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(MFIBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        let mut flows = Vec::new();
        flows.resize(self.period, MoneyFlow::default());
        Ok(MFIStream {
            period: self.period,
            flows,
            index: 0,
            count: 0,
            previous_typical: None,
            positive: 0.0 as Float,
            negative: 0.0 as Float,
        })
    }
}

/// Reusable Prepared Batch Runner for Money Flow Index.
#[derive(Debug, Clone)]
pub struct MFIBatchRunner {
    config: MFIConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for MFIBatchRunner {}

impl PreparedBatchRunner<MFIConfig> for MFIBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MFIConfig as IndicatorConfig>::Input<'a>,
        output: <MFIConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MFIConfig: 'a,
    {
        let actual_input_len = input
            .open
            .len()
            .max(input.high.len())
            .max(input.low.len())
            .max(input.close.len())
            .max(input.volume.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for Money Flow Index.
#[derive(Debug, Clone)]
pub struct MFIStream {
    period: usize,
    flows: Vec<MoneyFlow>,
    index: usize,
    count: usize,
    previous_typical: Option<Float>,
    positive: Float,
    negative: Float,
}

impl crate::traits::sealed::Sealed for MFIStream {}

impl StreamingComputation<MFIConfig> for MFIStream {
    type Tick = MFITick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_tick(&[
            ("open", input.open),
            ("high", input.high),
            ("low", input.low),
            ("close", input.close),
            ("volume", input.volume),
        ])?;
        let current_typical = typical_price(input.high, input.low, input.close);
        let Some(previous_typical) = self.previous_typical else {
            self.previous_typical = Some(current_typical);
            return Ok(None);
        };
        let current = directional_money_flow(current_typical, previous_typical, input.volume);

        if self.count == self.period {
            let trailing = self.flows[self.index];
            self.positive -= trailing.positive;
            self.negative -= trailing.negative;
        } else {
            self.count += 1;
        }

        self.flows[self.index] = current;
        self.index = (self.index + 1) % self.period;
        self.positive += current.positive;
        self.negative += current.negative;
        self.previous_typical = Some(current_typical);

        if self.count < self.period {
            Ok(None)
        } else {
            Ok(Some(mfi_projection(self.positive, self.negative)))
        }
    }

    fn reset(&mut self) {
        self.flows.fill(MoneyFlow::default());
        self.index = 0;
        self.count = 0;
        self.previous_typical = None;
        self.positive = 0.0 as Float;
        self.negative = 0.0 as Float;
    }
}

// Ultimate Oscillator -----------------------------------------------------

/// Borrowed aligned high/low/close input for [`ULTOSC`] Batch Computation.
#[derive(Debug, Clone, Copy)]
pub struct ULTOSCInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`ULTOSC`] Streaming Computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ULTOSCTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

#[derive(Debug, Clone, Copy, Default)]
struct UltimateTerm {
    buying_pressure: Float,
    true_range: Float,
}

fn sorted_ultosc_periods(period1: usize, period2: usize, period3: usize) -> Result<[usize; 3]> {
    validate_bounded_period("timeperiod1", period1, 1)?;
    validate_bounded_period("timeperiod2", period2, 1)?;
    validate_bounded_period("timeperiod3", period3, 1)?;
    let mut periods = [period1, period2, period3];
    periods.sort_unstable();
    Ok(periods)
}

fn validate_ultosc_input(
    input: ULTOSCInput<'_>,
    periods: [usize; 3],
) -> Result<(usize, usize, usize)> {
    for (index, period) in periods.iter().copied().enumerate() {
        validate_bounded_period(
            match index {
                0 => "short_period",
                1 => "medium_period",
                _ => "long_period",
            },
            period,
            1,
        )?;
    }
    let len = validate_observation_columns(&[
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    let lookback = periods[2];
    let count = validate_input_len(len, lookback)?;
    Ok((len, lookback, count))
}

#[inline]
fn ultimate_term(high: Float, low: Float, close: Float, previous_close: Float) -> UltimateTerm {
    let true_low = Float::min(low, previous_close);
    let high_low = high - low;
    let high_close = (high - previous_close).abs();
    let low_close = (low - previous_close).abs();
    UltimateTerm {
        buying_pressure: close - true_low,
        true_range: Float::max(high_low, Float::max(high_close, low_close)),
    }
}

#[inline]
fn ultimate_value(buying_pressure: [Float; 3], true_range: [Float; 3]) -> Float {
    let weights = [4.0 as Float, 2.0 as Float, 1.0 as Float];
    let mut weighted = 0.0 as Float;
    for index in 0..3 {
        if !is_ta_zero(true_range[index]) {
            weighted += weights[index] * (buying_pressure[index] / true_range[index]);
        }
    }
    100.0 as Float * (weighted / 7.0 as Float)
}

#[inline]
fn ultimate_term_at(input: ULTOSCInput<'_>, index: usize) -> UltimateTerm {
    ultimate_term(
        input.high[index],
        input.low[index],
        input.close[index],
        input.close[index - 1],
    )
}

fn ultosc_kernel(
    input: ULTOSCInput<'_>,
    periods: [usize; 3],
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut buying_pressure = [0.0 as Float; 3];
    let mut true_range = [0.0 as Float; 3];
    for period_index in 0..3 {
        let first = lookback + 1 - periods[period_index];
        for source_index in first..=lookback {
            let term = ultimate_term_at(input, source_index);
            buying_pressure[period_index] += term.buying_pressure;
            true_range[period_index] += term.true_range;
        }
    }
    output[0] = ultimate_value(buying_pressure, true_range);

    for source_index in lookback + 1..input.high.len() {
        let current = ultimate_term_at(input, source_index);
        for period_index in 0..3 {
            let trailing = ultimate_term_at(input, source_index - periods[period_index]);
            buying_pressure[period_index] -= trailing.buying_pressure;
            true_range[period_index] -= trailing.true_range;
            buying_pressure[period_index] += current.buying_pressure;
            true_range[period_index] += current.true_range;
        }
        output[source_index - lookback] = ultimate_value(buying_pressure, true_range);
    }

    OutputRange::new(lookback, count)
}

/// Computes Ultimate Oscillator into caller-owned Compact Output storage.
///
/// The three Periods are order-insensitive. They are sorted from shortest to
/// longest before the 4:2:1 weighting is applied, matching the Indicator
/// Definition rather than assigning weights to argument positions.
#[allow(non_snake_case)]
pub fn ULTOSC(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod1: usize,
    timeperiod2: usize,
    timeperiod3: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let periods = sorted_ultosc_periods(timeperiod1, timeperiod2, timeperiod3)?;
    let input = ULTOSCInput { high, low, close };
    let (_, lookback, count) = validate_ultosc_input(input, periods)?;
    validate_output_len("ULTOSC", out_real.len(), count)?;
    Ok(ultosc_kernel(input, periods, lookback, count, out_real))
}

/// Immutable Ultimate Oscillator Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ULTOSCConfig {
    periods: [usize; 3],
}

impl ULTOSCConfig {
    /// Creates an Ultimate Oscillator configuration.
    ///
    /// The supplied Periods are canonicalized from shortest to longest.
    pub fn new(period1: usize, period2: usize, period3: usize) -> Result<Self> {
        Ok(Self {
            periods: sorted_ultosc_periods(period1, period2, period3)?,
        })
    }

    /// Returns the canonical shortest, middle, and longest Periods.
    pub const fn periods(&self) -> [usize; 3] {
        self.periods
    }

    /// Returns the shortest configured Period.
    pub const fn short_period(&self) -> usize {
        self.periods[0]
    }

    /// Returns the middle configured Period.
    pub const fn medium_period(&self) -> usize {
        self.periods[1]
    }

    /// Returns the longest configured Period.
    pub const fn long_period(&self) -> usize {
        self.periods[2]
    }
}

impl crate::traits::sealed::Sealed for ULTOSCConfig {}

impl IndicatorConfig for ULTOSCConfig {
    type Input<'a> = ULTOSCInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = ULTOSCBatchRunner;
    type Stream = ULTOSCStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.periods[2]
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (len, lookback, count) = validate_ultosc_input(input, self.periods)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = ultosc_kernel(input, self.periods, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ULTOSC(
            input.high,
            input.low,
            input.close,
            self.periods[0],
            self.periods[1],
            self.periods[2],
            output,
        )
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(ULTOSCBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        let mut terms = Vec::new();
        terms.resize(self.periods[2], UltimateTerm::default());
        Ok(ULTOSCStream {
            periods: self.periods,
            terms,
            index: 0,
            count: 0,
            previous_close: None,
            buying_pressure: [0.0 as Float; 3],
            true_range: [0.0 as Float; 3],
        })
    }
}

/// Reusable Prepared Batch Runner for Ultimate Oscillator.
#[derive(Debug, Clone)]
pub struct ULTOSCBatchRunner {
    config: ULTOSCConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for ULTOSCBatchRunner {}

impl PreparedBatchRunner<ULTOSCConfig> for ULTOSCBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <ULTOSCConfig as IndicatorConfig>::Input<'a>,
        output: <ULTOSCConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        ULTOSCConfig: 'a,
    {
        let actual_input_len = input.high.len().max(input.low.len()).max(input.close.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation for Ultimate Oscillator.
#[derive(Debug, Clone)]
pub struct ULTOSCStream {
    periods: [usize; 3],
    terms: Vec<UltimateTerm>,
    index: usize,
    count: usize,
    previous_close: Option<Float>,
    buying_pressure: [Float; 3],
    true_range: [Float; 3],
}

impl crate::traits::sealed::Sealed for ULTOSCStream {}

impl StreamingComputation<ULTOSCConfig> for ULTOSCStream {
    type Tick = ULTOSCTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_tick(&[
            ("high", input.high),
            ("low", input.low),
            ("close", input.close),
        ])?;
        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };
        let current = ultimate_term(input.high, input.low, input.close, previous_close);

        let was_full = self.count == self.terms.len();
        if was_full {
            for period_index in 0..3 {
                let period = self.periods[period_index];
                let trailing_index = (self.index + self.terms.len() - period) % self.terms.len();
                let trailing = self.terms[trailing_index];
                self.buying_pressure[period_index] -= trailing.buying_pressure;
                self.true_range[period_index] -= trailing.true_range;
            }
        }

        self.terms[self.index] = current;
        self.index = (self.index + 1) % self.terms.len();
        if !was_full {
            self.count += 1;
        }

        if was_full {
            for period_index in 0..3 {
                self.buying_pressure[period_index] += current.buying_pressure;
                self.true_range[period_index] += current.true_range;
            }
        } else if self.count == self.terms.len() {
            for period_index in 0..3 {
                let first = self.terms.len() - self.periods[period_index];
                for term in &self.terms[first..] {
                    self.buying_pressure[period_index] += term.buying_pressure;
                    self.true_range[period_index] += term.true_range;
                }
            }
        }
        self.previous_close = Some(input.close);

        if self.count < self.periods[2] {
            Ok(None)
        } else {
            Ok(Some(ultimate_value(self.buying_pressure, self.true_range)))
        }
    }

    fn reset(&mut self) {
        self.terms.fill(UltimateTerm::default());
        self.index = 0;
        self.count = 0;
        self.previous_close = None;
        self.buying_pressure = [0.0 as Float; 3];
        self.true_range = [0.0 as Float; 3];
    }
}
