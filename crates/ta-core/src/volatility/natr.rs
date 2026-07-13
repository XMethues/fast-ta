//! Normalized Average True Range (NATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
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

/// TA-Lib-style Normalized Average True Range batch function.
#[allow(non_snake_case)]
pub fn NATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if timeperiod == 1 {
        crate::validate_period("timeperiod", timeperiod)?;
        return super::trange::TRANGE(high, low, close, out_real);
    }

    let lookback = super::atr::atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(high, low, close)?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("NATR", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut atr = 0.0 as Float;
    for input_idx in 1..=timeperiod {
        atr += super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }
    atr /= timeperiod as Float;
    out_real[0] = normalize(atr, close[lookback]);

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range =
            super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
        atr = super::atr::wilder_smooth(atr, range, timeperiod);
        out_real[output_idx] = normalize(atr, close[input_idx]);
    }

    Ok(OutputRange::new(lookback, count))
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

/// Normalized Average True Range indicator.
#[derive(Debug, Clone)]
pub struct NATR {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl NATR {
    /// Creates a new Normalized Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        super::atr::atr_lookback(timeperiod)?;
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

    /// Computes compact NATR outputs using this indicator's period.
    #[inline]
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        NATR(high, low, close, self.period, out_real)
    }

    /// Computes full-length padded NATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        NATR_vec(high, low, close, self.period)
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
        self.period
    }

    #[inline]
    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        NATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        NATR_vec(input.high, input.low, input.close, self.period)
    }
}

impl StreamingIndicator for NATR {
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
}

impl Resettable for NATR {
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
