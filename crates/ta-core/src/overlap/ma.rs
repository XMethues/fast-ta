//! Generic Moving Average (MA) dispatcher.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, CompactOutput, Float, Indicator, IndicatorConfig, OutputRange,
    PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator, TalibError,
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

fn ma_lookback(timeperiod: usize, matype: MAType) -> Result<usize> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    match matype {
        MAType::SMA | MAType::EMA | MAType::WMA | MAType::TRIMA => Ok(lookback),
        MAType::DEMA => lookback
            .checked_mul(2)
            .ok_or_else(|| TalibError::invalid_period(timeperiod, "DEMA lookback would overflow")),
        MAType::TEMA => lookback
            .checked_mul(3)
            .ok_or_else(|| TalibError::invalid_period(timeperiod, "TEMA lookback would overflow")),
        MAType::T3 => lookback
            .checked_mul(6)
            .ok_or_else(|| TalibError::invalid_period(timeperiod, "T3 lookback would overflow")),
        MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
    }
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

#[inline]
fn ma_kernel(
    real: &[Float],
    timeperiod: usize,
    matype: MAType,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    match matype {
        MAType::SMA => {
            super::sma::sma_kernel(
                real,
                timeperiod,
                1.0 as Float / timeperiod as Float,
                count,
                out_real,
            );
            if count == 0 {
                Ok(OutputRange::empty())
            } else {
                Ok(OutputRange::new(lookback, count))
            }
        }
        MAType::EMA => Ok(super::ema::ema_kernel(
            real, timeperiod, lookback, count, out_real,
        )),
        MAType::WMA => Ok(super::wma::wma_kernel(
            real, timeperiod, lookback, count, out_real,
        )),
        MAType::DEMA => super::dema::dema_kernel(real, timeperiod, lookback, count, out_real),
        MAType::TEMA => super::tema::tema_kernel(real, timeperiod, lookback, count, out_real),
        MAType::TRIMA => Ok(super::trima::trima_kernel(
            real, timeperiod, lookback, count, out_real,
        )),
        MAType::T3 => super::t3::t3_kernel(
            real,
            timeperiod,
            super::t3::T3_DEFAULT_VFACTOR,
            lookback,
            count,
            out_real,
        ),
        MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
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

/// Immutable generic Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MAConfig {
    period: usize,
    matype: MAType,
}

impl MAConfig {
    /// Creates a configuration for the selected moving-average definition.
    pub fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        ma_lookback(timeperiod, matype)?;
        Ok(Self {
            period: timeperiod,
            matype,
        })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured moving-average type.
    #[inline]
    pub const fn ma_type(&self) -> MAType {
        self.matype
    }
}

impl crate::traits::sealed::Sealed for MAConfig {}

impl IndicatorConfig for MAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MABatchRunner;
    type Stream = MAStream;

    #[inline]
    fn lookback(&self) -> usize {
        ma_lookback(self.period, self.matype)
            .expect("validated MA configuration must retain a valid lookback")
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let lookback = ma_lookback(self.period, self.matype)?;
        validate_finite_slice("real", input)?;
        let count = validate_input_len(input.len(), lookback)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = ma_kernel(
            input,
            self.period,
            self.matype,
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
        MA(input, self.period, self.matype, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(MABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        MAStream::new(*self)
    }
}

/// Prepared Batch Runner for the generic Moving Average dispatcher.
#[derive(Debug, Clone)]
pub struct MABatchRunner {
    config: MAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for MABatchRunner {}

impl PreparedBatchRunner<MAConfig> for MABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <MAConfig as IndicatorConfig>::Input<'a>,
        output: <MAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MAConfig: 'a,
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

#[derive(Debug, Clone)]
enum MAStreamInner {
    SMA(super::sma::SMAStream),
    EMA(super::ema::EMAStream),
    WMA(super::wma::WMAStream),
    DEMA(super::dema::DEMAStream),
    TEMA(super::tema::TEMAStream),
    TRIMA(super::trima::TRIMAStream),
    T3(super::t3::T3Stream),
}

impl MAStreamInner {
    fn new(config: MAConfig) -> Result<Self> {
        match config.matype {
            MAType::SMA => {
                let inner = super::sma::SMAConfig::new(config.period)?;
                Ok(Self::SMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::EMA => {
                let inner = super::ema::EMAConfig::new(config.period)?;
                Ok(Self::EMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::WMA => {
                let inner = super::wma::WMAConfig::new(config.period)?;
                Ok(Self::WMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::DEMA => {
                let inner = super::dema::DEMAConfig::new(config.period)?;
                Ok(Self::DEMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::TEMA => {
                let inner = super::tema::TEMAConfig::new(config.period)?;
                Ok(Self::TEMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::TRIMA => {
                let inner = super::trima::TRIMAConfig::new(config.period)?;
                Ok(Self::TRIMA(IndicatorConfig::stream(&inner)?))
            }
            MAType::T3 => {
                let inner = super::t3::T3Config::with_default_vfactor(config.period)?;
                Ok(Self::T3(IndicatorConfig::stream(&inner)?))
            }
            MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(config.matype)),
        }
    }

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        match self {
            Self::SMA(inner) => StreamingComputation::<super::sma::SMAConfig>::next(inner, input),
            Self::EMA(inner) => StreamingComputation::<super::ema::EMAConfig>::next(inner, input),
            Self::WMA(inner) => StreamingComputation::<super::wma::WMAConfig>::next(inner, input),
            Self::DEMA(inner) => {
                StreamingComputation::<super::dema::DEMAConfig>::next(inner, input)
            }
            Self::TEMA(inner) => {
                StreamingComputation::<super::tema::TEMAConfig>::next(inner, input)
            }
            Self::TRIMA(inner) => {
                StreamingComputation::<super::trima::TRIMAConfig>::next(inner, input)
            }
            Self::T3(inner) => StreamingComputation::<super::t3::T3Config>::next(inner, input),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::SMA(inner) => StreamingComputation::<super::sma::SMAConfig>::reset(inner),
            Self::EMA(inner) => StreamingComputation::<super::ema::EMAConfig>::reset(inner),
            Self::WMA(inner) => StreamingComputation::<super::wma::WMAConfig>::reset(inner),
            Self::DEMA(inner) => StreamingComputation::<super::dema::DEMAConfig>::reset(inner),
            Self::TEMA(inner) => StreamingComputation::<super::tema::TEMAConfig>::reset(inner),
            Self::TRIMA(inner) => StreamingComputation::<super::trima::TRIMAConfig>::reset(inner),
            Self::T3(inner) => StreamingComputation::<super::t3::T3Config>::reset(inner),
        }
    }
}

/// Independent Streaming Computation state for the generic Moving Average dispatcher.
#[derive(Debug, Clone)]
pub struct MAStream {
    config: MAConfig,
    inner: MAStreamInner,
}

impl MAStream {
    fn new(config: MAConfig) -> Result<Self> {
        let inner = MAStreamInner::new(config)?;
        Ok(Self { config, inner })
    }

    #[inline]
    const fn period(&self) -> usize {
        self.config.period
    }

    #[inline]
    const fn ma_type(&self) -> MAType {
        self.config.matype
    }
}

impl crate::traits::sealed::Sealed for MAStream {}

impl StreamingComputation<MAConfig> for MAStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        self.inner.next(input)
    }

    #[inline]
    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Legacy generic Moving Average indicator dispatcher.
#[derive(Debug, Clone)]
pub struct MA {
    stream: MAStream,
}

impl MA {
    /// Creates a new legacy MA dispatcher for the selected moving-average type.
    pub fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        let config = MAConfig::new(timeperiod, matype)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.stream.period()
    }

    /// Returns the configured moving-average type.
    #[inline]
    pub const fn ma_type(&self) -> MAType {
        self.stream.ma_type()
    }

    /// Computes compact MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        MA(real, self.period(), self.ma_type(), out_real)
    }

    /// Computes full-length padded MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        MA_vec(real, self.period(), self.ma_type())
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(StreamingIndicator::next(self, input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for MA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        ma_lookback(self.period(), self.ma_type())
            .expect("validated MA indicator must retain a valid lookback")
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MA(inputs, self.period(), self.ma_type(), outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MA_vec(inputs, self.period(), self.ma_type())
    }
}

impl StreamingIndicator for MA {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        StreamingComputation::<MAConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for MA {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<MAConfig>::reset(&mut self.stream);
    }
}
