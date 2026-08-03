//! Normalized Average True Range (NATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, CompactOutput, Float, Indicator, IndicatorConfig, OutputRange,
    PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

const ZERO_TOLERANCE: Float = 1e-8 as Float;

/// Borrowed SoA inputs for [`NATR`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct NATRInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`NATR`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NATRTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

#[inline]
fn normalize(atr: Float, close: Float) -> Float {
    if close.abs() <= ZERO_TOLERANCE {
        0.0 as Float
    } else {
        (atr / close) * 100.0 as Float
    }
}
fn validate_natr_input(input: NATRInput<'_>, timeperiod: usize) -> Result<(usize, usize, usize)> {
    let lookback = super::atr::atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(input.high, input.low, input.close)?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count, len))
}

fn natr_kernel(
    input: NATRInput<'_>,
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
    output[0] = if timeperiod == 1 {
        atr
    } else {
        normalize(atr, input.close[lookback])
    };

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range = super::trange::true_range(
            input.high[input_idx],
            input.low[input_idx],
            input.close[input_idx - 1],
        );
        atr = super::atr::wilder_smooth(atr, range, timeperiod);
        output[output_idx] = if timeperiod == 1 {
            atr
        } else {
            normalize(atr, input.close[input_idx])
        };
    }

    OutputRange::new(lookback, count)
}

/// TA-Lib-style Normalized Average True Range batch function.
#[allow(non_snake_case)]
pub fn NATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = NATRInput { high, low, close };
    let (lookback, count, _) = validate_natr_input(input, timeperiod)?;
    validate_output_len(
        if timeperiod == 1 { "TRANGE" } else { "NATR" },
        out_real.len(),
        count,
    )?;
    Ok(natr_kernel(input, timeperiod, lookback, count, out_real))
}

/// Computes Normalized Average True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn NATR_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = NATR(high, low, close, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}
/// Immutable Normalized Average True Range Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NATRConfig {
    period: usize,
}

impl NATRConfig {
    /// Creates a NATR configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        super::atr::atr_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for NATRConfig {}

impl IndicatorConfig for NATRConfig {
    type Input<'a> = NATRInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = NATRBatchRunner;
    type Stream = NATRStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count, len) = validate_natr_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = natr_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        NATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(NATRBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(NATRStream {
            period: self.period,
            previous_close: None,
            count: 0,
            true_range_sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }
}

/// Reusable Prepared Batch Runner for NATR.
#[derive(Debug, Clone)]
pub struct NATRBatchRunner {
    config: NATRConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for NATRBatchRunner {}

impl PreparedBatchRunner<NATRConfig> for NATRBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <NATRConfig as IndicatorConfig>::Input<'a>,
        output: <NATRConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        NATRConfig: 'a,
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

/// Independent Streaming Computation state for NATR.
#[derive(Debug, Clone)]
pub struct NATRStream {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl crate::traits::sealed::Sealed for NATRStream {}

impl StreamingComputation<NATRConfig> for NATRStream {
    type Tick = NATRTick;
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
            return Ok(Some(normalize(self.value, input.close)));
        }

        self.value = super::atr::wilder_smooth(self.value, range, self.period);
        Ok(Some(normalize(self.value, input.close)))
    }

    #[inline]
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}

/// Normalized Average True Range compatibility adapter.
#[derive(Debug, Clone)]
pub struct NATR {
    config: NATRConfig,
    stream: NATRStream,
}

impl NATR {
    /// Creates a new Normalized Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = NATRConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { config, stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.config.period()
    }

    /// Computes compact NATR outputs using this indicator's period.
    #[inline]
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        NATR(high, low, close, self.config.period, out_real)
    }

    /// Computes full-length padded NATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        NATR_vec(high, low, close, self.config.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: NATRTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for NATR {
    type Input<'a> = NATRInput<'a>;
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
        NATR(
            input.high,
            input.low,
            input.close,
            self.config.period,
            output,
        )
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        NATR_vec(input.high, input.low, input.close, self.config.period)
    }
}

impl StreamingIndicator for NATR {
    type Tick = NATRTick;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        StreamingComputation::<NATRConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for NATR {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<NATRConfig>::reset(&mut self.stream);
    }
}
