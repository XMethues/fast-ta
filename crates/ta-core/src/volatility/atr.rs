//! Average True Range (ATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
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

/// TA-Lib-style Average True Range batch function.
#[allow(non_snake_case)]
pub fn ATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if timeperiod == 1 {
        validate_period("timeperiod", timeperiod)?;
        return super::trange::TRANGE(high, low, close, out_real);
    }

    let lookback = atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(high, low, close)?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("ATR", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut atr = 0.0 as Float;
    for input_idx in 1..=timeperiod {
        atr += super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }
    atr /= timeperiod as Float;
    out_real[0] = atr;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range =
            super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
        atr = wilder_smooth(atr, range, timeperiod);
        out_real[output_idx] = atr;
    }

    Ok(OutputRange::new(lookback, count))
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

/// Average True Range indicator.
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl ATR {
    /// Creates a new Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        atr_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            previous_close: None,
            count: 0,
            true_range_sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
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
        ATR(high, low, close, self.period, out_real)
    }

    /// Computes full-length padded ATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        ATR_vec(high, low, close, self.period)
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
        self.period
    }

    #[inline]
    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        ATR_vec(input.high, input.low, input.close, self.period)
    }
}

impl StreamingIndicator for ATR {
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
}

impl Resettable for ATR {
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
