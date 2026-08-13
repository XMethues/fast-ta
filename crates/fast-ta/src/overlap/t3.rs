//! T3 Moving Average (T3).

use crate::common::validate_finite_value;
use crate::{
    period_lookback, validate_finite_slice, validate_input_len, validate_output_len, CompactOutput,
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation,
    TalibError,
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

fn validate_t3_input(real: &[Float], timeperiod: usize, vfactor: Float) -> Result<(usize, usize)> {
    let lookback = t3_lookback(timeperiod)?;
    validate_vfactor(vfactor)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

pub(super) fn t3_kernel(
    real: &[Float],
    timeperiod: usize,
    vfactor: Float,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut t3 = T3Stream::new(timeperiod, vfactor)?;
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(output) = t3.next_validated(value)? {
            out_real[output_idx] = output;
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// TA-Lib-style T3 Moving Average batch function.
#[allow(non_snake_case)]
pub fn T3(
    real: &[Float],
    timeperiod: usize,
    vfactor: Float,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let (lookback, count) = validate_t3_input(real, timeperiod, vfactor)?;
    validate_output_len("T3", out_real.len(), count)?;
    t3_kernel(real, timeperiod, vfactor, lookback, count, out_real)
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

/// Immutable T3 Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct T3Config {
    period: usize,
    vfactor: Float,
}

impl T3Config {
    /// Creates a configuration with an explicit volume factor.
    pub fn new(timeperiod: usize, vfactor: Float) -> Result<Self> {
        t3_lookback(timeperiod)?;
        validate_vfactor(vfactor)?;
        Ok(Self {
            period: timeperiod,
            vfactor,
        })
    }

    /// Creates a configuration with TA-Lib's default volume factor.
    pub fn with_default_vfactor(timeperiod: usize) -> Result<Self> {
        Self::new(timeperiod, T3_DEFAULT_VFACTOR)
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured volume factor.
    #[inline]
    pub const fn vfactor(&self) -> Float {
        self.vfactor
    }
}

impl crate::traits::sealed::Sealed for T3Config {}

impl IndicatorConfig for T3Config {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = T3BatchRunner;
    type Stream = T3Stream;

    #[inline]
    fn lookback(&self) -> usize {
        (self.period - 1) * 6
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_t3_input(input, self.period, self.vfactor)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = t3_kernel(
            input,
            self.period,
            self.vfactor,
            lookback,
            count,
            &mut values,
        )?;
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        T3(input, self.period, self.vfactor, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(T3BatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        T3Stream::new(self.period, self.vfactor)
    }
}

/// Prepared Batch Runner for T3 Moving Average.
#[derive(Debug, Clone)]
pub struct T3BatchRunner {
    config: T3Config,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for T3BatchRunner {}

impl PreparedBatchRunner<T3Config> for T3BatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <T3Config as IndicatorConfig>::Input<'a>,
        output: <T3Config as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        T3Config: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for T3 Moving Average.
#[derive(Debug, Clone)]
pub struct T3Stream {
    coefficients: (Float, Float, Float, Float),
    ema1: super::ema::EMAStream,
    ema2: super::ema::EMAStream,
    ema3: super::ema::EMAStream,
    ema4: super::ema::EMAStream,
    ema5: super::ema::EMAStream,
    ema6: super::ema::EMAStream,
}

impl T3Stream {
    fn new(period: usize, vfactor: Float) -> Result<Self> {
        t3_lookback(period)?;
        validate_vfactor(vfactor)?;
        Ok(Self {
            coefficients: t3_coefficients(vfactor),
            ema1: super::ema::EMAStream::new(period)?,
            ema2: super::ema::EMAStream::new(period)?,
            ema3: super::ema::EMAStream::new(period)?,
            ema4: super::ema::EMAStream::new(period)?,
            ema5: super::ema::EMAStream::new(period)?,
            ema6: super::ema::EMAStream::new(period)?,
        })
    }

    fn next_validated(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next_unchecked(input) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema1)?;
        let Some(ema2) = self.ema2.next_unchecked(ema1) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema2)?;
        let Some(ema3) = self.ema3.next_unchecked(ema2) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema3)?;
        let Some(ema4) = self.ema4.next_unchecked(ema3) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema4)?;
        let Some(ema5) = self.ema5.next_unchecked(ema4) else {
            return Ok(None);
        };
        validate_finite_value("input", 0, ema5)?;
        let Some(ema6) = self.ema6.next_unchecked(ema5) else {
            return Ok(None);
        };
        Ok(Some(t3_value(ema3, ema4, ema5, ema6, self.coefficients)))
    }

    fn reset_state(&mut self) {
        self.ema1.reset_state();
        self.ema2.reset_state();
        self.ema3.reset_state();
        self.ema4.reset_state();
        self.ema5.reset_state();
        self.ema6.reset_state();
    }
}

impl crate::traits::sealed::Sealed for T3Stream {}

impl StreamingComputation<T3Config> for T3Stream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;
        self.next_validated(input)
    }

    #[inline]
    fn reset(&mut self) {
        self.reset_state();
    }
}
