//! Generic Moving Average (MA) dispatcher.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, Float, Indicator, OutputRange,
    Resettable, Result, StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec::Vec};

/// Official TA-Lib moving-average type selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MAType {
    /// Simple Moving Average.
    SMA,
    /// Exponential Moving Average.
    EMA,
    /// Weighted Moving Average.
    WMA,
    /// Double Exponential Moving Average.
    DEMA,
    /// Triple Exponential Moving Average.
    TEMA,
    /// Triangular Moving Average.
    TRIMA,
    /// Kaufman Adaptive Moving Average (not implemented in this tranche).
    KAMA,
    /// MESA Adaptive Moving Average (not implemented in this tranche).
    MAMA,
    /// T3 Moving Average.
    T3,
}

impl MAType {
    /// Official TA-Lib integer id for this moving-average type.
    pub const fn talib_id(self) -> usize {
        match self {
            Self::SMA => 0,
            Self::EMA => 1,
            Self::WMA => 2,
            Self::DEMA => 3,
            Self::TEMA => 4,
            Self::TRIMA => 5,
            Self::KAMA => 6,
            Self::MAMA => 7,
            Self::T3 => 8,
        }
    }

    /// Stable display label used in error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SMA => "SMA",
            Self::EMA => "EMA",
            Self::WMA => "WMA",
            Self::DEMA => "DEMA",
            Self::TEMA => "TEMA",
            Self::TRIMA => "TRIMA",
            Self::KAMA => "KAMA",
            Self::MAMA => "MAMA",
            Self::T3 => "T3",
        }
    }
}

fn unsupported_ma_type(matype: MAType) -> TalibError {
    TalibError::not_implemented(format!("MAType::{}", matype.as_str()))
}

/// TA-Lib-style generic Moving Average dispatcher.
#[allow(non_snake_case)]
pub fn MA(
    real: &[Float],
    timeperiod: usize,
    matype: MAType,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    match matype {
        MAType::SMA => super::sma::SMA(real, timeperiod, out_real),
        MAType::EMA => super::ema::EMA(real, timeperiod, out_real),
        MAType::WMA => super::wma::WMA(real, timeperiod, out_real),
        MAType::DEMA => super::dema::DEMA(real, timeperiod, out_real),
        MAType::TEMA => super::tema::TEMA(real, timeperiod, out_real),
        MAType::TRIMA => super::trima::TRIMA(real, timeperiod, out_real),
        MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
        MAType::T3 => super::t3::T3_with_default_vfactor(real, timeperiod, out_real),
    }
}

/// Computes the selected moving average into a full-length padded vector.
#[allow(non_snake_case)]
pub fn MA_vec(real: &[Float], timeperiod: usize, matype: MAType) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MA(real, timeperiod, matype, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

#[derive(Debug, Clone)]
enum MAInner {
    SMA(super::sma::SMA),
    EMA(super::ema::EMA),
    WMA(super::wma::WMA),
    DEMA(super::dema::DEMA),
    TEMA(super::tema::TEMA),
    TRIMA(super::trima::TRIMA),
    T3(super::t3::T3),
}

impl MAInner {
    fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        match matype {
            MAType::SMA => Ok(Self::SMA(super::sma::SMA::new(timeperiod)?)),
            MAType::EMA => Ok(Self::EMA(super::ema::EMA::new(timeperiod)?)),
            MAType::WMA => Ok(Self::WMA(super::wma::WMA::new(timeperiod)?)),
            MAType::DEMA => Ok(Self::DEMA(super::dema::DEMA::new(timeperiod)?)),
            MAType::TEMA => Ok(Self::TEMA(super::tema::TEMA::new(timeperiod)?)),
            MAType::TRIMA => Ok(Self::TRIMA(super::trima::TRIMA::new(timeperiod)?)),
            MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
            MAType::T3 => Ok(Self::T3(super::t3::T3::with_default_vfactor(timeperiod)?)),
        }
    }

    fn lookback(&self) -> usize {
        match self {
            Self::SMA(inner) => inner.lookback(),
            Self::EMA(inner) => inner.lookback(),
            Self::WMA(inner) => inner.lookback(),
            Self::DEMA(inner) => inner.lookback(),
            Self::TEMA(inner) => inner.lookback(),
            Self::TRIMA(inner) => inner.lookback(),
            Self::T3(inner) => inner.lookback(),
        }
    }

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        match self {
            Self::SMA(inner) => inner.next(input),
            Self::EMA(inner) => inner.next(input),
            Self::WMA(inner) => inner.next(input),
            Self::DEMA(inner) => inner.next(input),
            Self::TEMA(inner) => inner.next(input),
            Self::TRIMA(inner) => inner.next(input),
            Self::T3(inner) => inner.next(input),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::SMA(inner) => inner.reset(),
            Self::EMA(inner) => inner.reset(),
            Self::WMA(inner) => inner.reset(),
            Self::DEMA(inner) => inner.reset(),
            Self::TEMA(inner) => inner.reset(),
            Self::TRIMA(inner) => inner.reset(),
            Self::T3(inner) => inner.reset(),
        }
    }
}

/// Generic Moving Average indicator dispatcher.
#[derive(Debug, Clone)]
pub struct MA {
    period: usize,
    matype: MAType,
    inner: MAInner,
}

impl MA {
    /// Creates a new MA dispatcher for the selected moving-average type.
    pub fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self {
            period: timeperiod,
            matype,
            inner: MAInner::new(timeperiod, matype)?,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured moving-average type.
    #[inline]
    pub const fn ma_type(&self) -> MAType {
        self.matype
    }

    /// Computes compact MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        MA(real, self.period, self.matype, out_real)
    }

    /// Computes full-length padded MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        MA_vec(real, self.period, self.matype)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for MA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.inner.lookback()
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MA(inputs, self.period, self.matype, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MA_vec(inputs, self.period, self.matype)
    }
}

impl StreamingIndicator for MA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        self.inner.next(input)
    }
}

impl Resettable for MA {
    fn reset(&mut self) {
        self.inner.reset();
    }
}
