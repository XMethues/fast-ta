//! Triple Exponential Moving Average (TEMA).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn tema_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(3)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "TEMA lookback would overflow"))
}

/// TA-Lib-style Triple Exponential Moving Average batch function.
#[allow(non_snake_case)]
pub fn TEMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = tema_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("TEMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut ema1 = super::ema::EMA::new(timeperiod)?;
    let mut ema2 = super::ema::EMA::new(timeperiod)?;
    let mut ema3 = super::ema::EMA::new(timeperiod)?;
    let mut output_idx = 0usize;

    for &value in real {
        let Some(ema1_value) = ema1.next(value)? else {
            continue;
        };
        let Some(ema2_value) = ema2.next(ema1_value)? else {
            continue;
        };
        let Some(ema3_value) = ema3.next(ema2_value)? else {
            continue;
        };
        out_real[output_idx] = 3.0 as Float * ema1_value - 3.0 as Float * ema2_value + ema3_value;
        output_idx += 1;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes TEMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn TEMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = TEMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Triple Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct TEMA {
    period: usize,
    lookback: usize,
    ema1: super::ema::EMA,
    ema2: super::ema::EMA,
    ema3: super::ema::EMA,
}

impl TEMA {
    /// Creates a new TEMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let lookback = tema_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            lookback,
            ema1: super::ema::EMA::new(timeperiod)?,
            ema2: super::ema::EMA::new(timeperiod)?,
            ema3: super::ema::EMA::new(timeperiod)?,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact TEMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        TEMA(real, self.period, out_real)
    }

    /// Computes full-length padded TEMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        TEMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TEMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TEMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TEMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for TEMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next(input)? else {
            return Ok(None);
        };
        let Some(ema2) = self.ema2.next(ema1)? else {
            return Ok(None);
        };
        let Some(ema3) = self.ema3.next(ema2)? else {
            return Ok(None);
        };
        Ok(Some(3.0 as Float * ema1 - 3.0 as Float * ema2 + ema3))
    }
}

impl Resettable for TEMA {
    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
    }
}
