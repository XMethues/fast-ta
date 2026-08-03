//! Average True Range (ATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, CompactOutput, Float, Indicator, IndicatorConfig,
    OutputRange, PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`ATR`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct ATRInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`ATR`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ATRTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Returns the default TA-Lib ATR lookback for a period.
pub(super) fn atr_lookback(timeperiod: usize) -> Result<usize> {
    validate_period("timeperiod", timeperiod)?;
    timeperiod.checked_add(1).ok_or_else(|| {
        crate::TalibError::invalid_period(timeperiod, "ATR lookback would overflow")
    })?;
    Ok(timeperiod)
}

/// Applies one Wilder smoothing step.
#[inline]
pub(super) fn wilder_smooth(previous: Float, true_range: Float, timeperiod: usize) -> Float {
    ((previous * (timeperiod - 1) as Float) + true_range) / timeperiod as Float
}
fn validate_atr_input(input: ATRInput<'_>, timeperiod: usize) -> Result<(usize, usize, usize)> {
    let lookback = atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(input.high, input.low, input.close)?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count, len))
}

fn atr_kernel(
    input: ATRInput<'_>,
    timeperiod: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut atr = 0.0 as Float;
    for input_idx in 1..=timeperiod {
        atr += super::trange::true_range(
            input.high[input_idx],
            input.low[input_idx],
            input.close[input_idx - 1],
        );
    }
    atr /= timeperiod as Float;
    output[0] = atr;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range = super::trange::true_range(
            input.high[input_idx],
            input.low[input_idx],
            input.close[input_idx - 1],
        );
        atr = wilder_smooth(atr, range, timeperiod);
        output[output_idx] = atr;
    }

    OutputRange::new(lookback, count)
}

/// TA-Lib-style Average True Range batch function.
#[allow(non_snake_case)]
pub fn ATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = ATRInput { high, low, close };
    let (lookback, count, _) = validate_atr_input(input, timeperiod)?;
    validate_output_len(
        if timeperiod == 1 { "TRANGE" } else { "ATR" },
        out_real.len(),
        count,
    )?;
    Ok(atr_kernel(input, timeperiod, lookback, count, out_real))
}

/// Computes Average True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn ATR_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = ATR(high, low, close, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}
/// Immutable Average True Range Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ATRConfig {
    period: usize,
}

impl ATRConfig {
    /// Creates an ATR configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        atr_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for ATRConfig {}

impl IndicatorConfig for ATRConfig {
    type Input<'a> = ATRInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = ATRBatchRunner;
    type Stream = ATRStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count, len) = validate_atr_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = atr_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(ATRBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(ATRStream {
            period: self.period,
            previous_close: None,
            count: 0,
            true_range_sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }
}

/// Reusable Prepared Batch Runner for ATR.
#[derive(Debug, Clone)]
pub struct ATRBatchRunner {
    config: ATRConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for ATRBatchRunner {}

impl PreparedBatchRunner<ATRConfig> for ATRBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <ATRConfig as IndicatorConfig>::Input<'a>,
        output: <ATRConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        ATRConfig: 'a,
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

/// Independent Streaming Computation state for ATR.
#[derive(Debug, Clone)]
pub struct ATRStream {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl crate::traits::sealed::Sealed for ATRStream {}

impl StreamingComputation<ATRConfig> for ATRStream {
    type Tick = ATRTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let range = super::trange::true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);

        if self.period == 1 {
            self.value = range;
            return Ok(Some(range));
        }

        if self.count < self.period {
            self.true_range_sum += range;
            self.count += 1;

            if self.count < self.period {
                return Ok(None);
            }

            self.value = self.true_range_sum / self.period as Float;
            return Ok(Some(self.value));
        }

        self.value = wilder_smooth(self.value, range, self.period);
        Ok(Some(self.value))
    }

    #[inline]
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}

/// Average True Range compatibility adapter.
#[derive(Debug, Clone)]
pub struct ATR {
    config: ATRConfig,
    stream: ATRStream,
}

impl ATR {
    /// Creates a new Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = ATRConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { config, stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.config.period()
    }

    /// Computes compact ATR outputs using this indicator's period.
    #[inline]
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        ATR(high, low, close, self.config.period, out_real)
    }

    /// Computes full-length padded ATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        ATR_vec(high, low, close, self.config.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: ATRTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for ATR {
    type Input<'a> = ATRInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.config.period
    }

    #[inline]
    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ATR(
            input.high,
            input.low,
            input.close,
            self.config.period,
            output,
        )
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        ATR_vec(input.high, input.low, input.close, self.config.period)
    }
}

impl StreamingIndicator for ATR {
    type Tick = ATRTick;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        StreamingComputation::<ATRConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for ATR {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<ATRConfig>::reset(&mut self.stream);
    }
}
