//! T3 Moving Average (T3).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

/// TA-Lib default T3 volume factor.
pub const T3_DEFAULT_VFACTOR: Float = 0.7 as Float;

#[inline]
fn t3_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(6)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "T3 lookback would overflow"))
}

fn validate_vfactor(vfactor: Float) -> Result<()> {
    if !vfactor.is_finite() || !(0.0 as Float..=1.0 as Float).contains(&vfactor) {
        return Err(TalibError::invalid_parameter(
            "vfactor".to_string(),
            format!("{}", vfactor),
            "value in [0.0, 1.0]".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn t3_coefficients(vfactor: Float) -> (Float, Float, Float, Float) {
    let v2 = vfactor * vfactor;
    let v3 = v2 * vfactor;
    let c1 = -v3;
    let c2 = 3.0 as Float * (v2 - c1);
    let c3 = -6.0 as Float * v2 - 3.0 as Float * (vfactor - c1);
    let c4 = 1.0 as Float + 3.0 as Float * vfactor - c1 + 3.0 as Float * v2;
    (c1, c2, c3, c4)
}

#[inline]
fn t3_value(
    ema3: Float,
    ema4: Float,
    ema5: Float,
    ema6: Float,
    coefficients: (Float, Float, Float, Float),
) -> Float {
    let (c1, c2, c3, c4) = coefficients;
    c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
}

/// TA-Lib-style T3 Moving Average batch function.
#[allow(non_snake_case)]
pub fn T3(
    real: &[Float],
    timeperiod: usize,
    vfactor: Float,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = t3_lookback(timeperiod)?;
    validate_vfactor(vfactor)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("T3", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut t3 = T3::new(timeperiod, vfactor)?;
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(output) = t3.next(value)? {
            out_real[output_idx] = output;
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// TA-Lib-style T3 batch function using `T3_DEFAULT_VFACTOR`.
#[allow(non_snake_case)]
pub fn T3_with_default_vfactor(
    real: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    T3(real, timeperiod, T3_DEFAULT_VFACTOR, out_real)
}

/// Computes T3 into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn T3_vec(real: &[Float], timeperiod: usize, vfactor: Float) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = T3(real, timeperiod, vfactor, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Computes T3 with the default vfactor into a full-length vector.
#[allow(non_snake_case)]
pub fn T3_vec_with_default_vfactor(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    T3_vec(real, timeperiod, T3_DEFAULT_VFACTOR)
}

/// T3 Moving Average indicator.
#[derive(Debug, Clone)]
pub struct T3 {
    period: usize,
    lookback: usize,
    vfactor: Float,
    coefficients: (Float, Float, Float, Float),
    ema1: super::ema::EMA,
    ema2: super::ema::EMA,
    ema3: super::ema::EMA,
    ema4: super::ema::EMA,
    ema5: super::ema::EMA,
    ema6: super::ema::EMA,
}

impl T3 {
    /// Creates a new T3 indicator with an explicit vfactor.
    pub fn new(timeperiod: usize, vfactor: Float) -> Result<Self> {
        let lookback = t3_lookback(timeperiod)?;
        validate_vfactor(vfactor)?;
        Ok(Self {
            period: timeperiod,
            lookback,
            vfactor,
            coefficients: t3_coefficients(vfactor),
            ema1: super::ema::EMA::new(timeperiod)?,
            ema2: super::ema::EMA::new(timeperiod)?,
            ema3: super::ema::EMA::new(timeperiod)?,
            ema4: super::ema::EMA::new(timeperiod)?,
            ema5: super::ema::EMA::new(timeperiod)?,
            ema6: super::ema::EMA::new(timeperiod)?,
        })
    }

    /// Creates a new T3 indicator with TA-Lib's default vfactor.
    pub fn with_default_vfactor(timeperiod: usize) -> Result<Self> {
        Self::new(timeperiod, T3_DEFAULT_VFACTOR)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured vfactor.
    #[inline]
    pub const fn vfactor(&self) -> Float {
        self.vfactor
    }

    /// Computes compact T3 outputs using this indicator's period and vfactor.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        T3(real, self.period, self.vfactor, out_real)
    }

    /// Computes full-length padded T3 outputs using this indicator's period and vfactor.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        T3_vec(real, self.period, self.vfactor)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for T3 {
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
        T3(inputs, self.period, self.vfactor, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        T3_vec(inputs, self.period, self.vfactor)
    }
}

impl StreamingIndicator for T3 {
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
        let Some(ema4) = self.ema4.next(ema3)? else {
            return Ok(None);
        };
        let Some(ema5) = self.ema5.next(ema4)? else {
            return Ok(None);
        };
        let Some(ema6) = self.ema6.next(ema5)? else {
            return Ok(None);
        };
        Ok(Some(t3_value(ema3, ema4, ema5, ema6, self.coefficients)))
    }
}

impl Resettable for T3 {
    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
        self.ema4.reset();
        self.ema5.reset();
        self.ema6.reset();
    }
}
