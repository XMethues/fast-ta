//! Shared public values for Pattern Recognition indicators.

use crate::{Float, Result, TalibError};

/// Borrowed, source-aligned Open/High/Low/Close Candle observations.
///
/// The four columns must have equal lengths and contain finite values. Ordering,
/// missing-value repair, OHLC consistency, timestamps, and instrument identity
/// remain responsibilities of the caller's Observation Series.
#[derive(Debug, Clone, Copy)]
pub struct CandleInput<'a> {
    /// Open price column.
    pub open: &'a [Float],
    /// High price column.
    pub high: &'a [Float],
    /// Low price column.
    pub low: &'a [Float],
    /// Close price column.
    pub close: &'a [Float],
}

impl CandleInput<'_> {
    /// Returns the aligned source length using the Open column.
    #[inline]
    pub const fn len(&self) -> usize {
        self.open.len()
    }

    /// Returns whether the Open column is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.open.is_empty()
    }

    #[inline]
    pub(crate) fn max_len(&self) -> usize {
        self.open
            .len()
            .max(self.high.len())
            .max(self.low.len())
            .max(self.close.len())
    }

    #[inline]
    pub(crate) fn candle(&self, index: usize) -> Candle {
        Candle {
            open: self.open[index],
            high: self.high[index],
            low: self.low[index],
            close: self.close[index],
        }
    }
}

/// One finite Open/High/Low/Close Candle streaming tick.
///
/// No consistency relation between the four prices is imposed. Pattern
/// Recognition preserves TA-Lib's literal arithmetic for caller-supplied OHLC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candle {
    /// Open price.
    pub open: Float,
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Direction carried by a matched Pattern Signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternDirection {
    /// Bullish or positive TA-Lib signal direction.
    Bullish,
    /// Bearish or negative TA-Lib signal direction.
    Bearish,
}

/// Exact categorical magnitude carried by a matched Pattern Signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternStrength {
    /// TA-Lib magnitude 80 boundary match.
    Partial,
    /// TA-Lib magnitude 100 ordinary match.
    Standard,
    /// TA-Lib magnitude 200 Hikkake confirmation.
    Confirmed,
}

/// Categorical result for one valid Pattern Recognition source position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternSignal {
    /// The source position was valid but the pattern did not match.
    NoMatch,
    /// The pattern matched with an exact direction and strength.
    Match {
        /// Bullish or bearish match direction.
        direction: PatternDirection,
        /// Partial, Standard, or Confirmed match strength.
        strength: PatternStrength,
    },
}

impl PatternSignal {
    #[inline]
    pub(crate) const fn standard(direction: PatternDirection) -> Self {
        Self::Match {
            direction,
            strength: PatternStrength::Standard,
        }
    }

    /// Projects this signal to its exact TA-Lib integer code.
    ///
    /// The only possible codes are `0`, `±80`, `±100`, and `±200`.
    #[inline]
    pub const fn to_talib_code(self) -> i32 {
        let (direction, strength) = match self {
            Self::NoMatch => return 0,
            Self::Match {
                direction,
                strength,
            } => (direction, strength),
        };
        let magnitude = match strength {
            PatternStrength::Partial => 80,
            PatternStrength::Standard => 100,
            PatternStrength::Confirmed => 200,
        };
        match direction {
            PatternDirection::Bullish => magnitude,
            PatternDirection::Bearish => -magnitude,
        }
    }

    /// Converts one exact TA-Lib integer code into a Pattern Signal.
    ///
    /// Every code other than `0`, `±80`, `±100`, and `±200` is rejected.
    pub fn from_talib_code(code: i32) -> Result<Self> {
        let (direction, strength) = match code {
            0 => return Ok(Self::NoMatch),
            80 => (PatternDirection::Bullish, PatternStrength::Partial),
            -80 => (PatternDirection::Bearish, PatternStrength::Partial),
            100 => (PatternDirection::Bullish, PatternStrength::Standard),
            -100 => (PatternDirection::Bearish, PatternStrength::Standard),
            200 => (PatternDirection::Bullish, PatternStrength::Confirmed),
            -200 => (PatternDirection::Bearish, PatternStrength::Confirmed),
            _ => {
                return Err(TalibError::invalid_input(
                    "Pattern Signal code must be exactly 0, ±80, ±100, or ±200",
                ))
            }
        };
        Ok(Self::Match {
            direction,
            strength,
        })
    }
}

impl TryFrom<i32> for PatternSignal {
    type Error = TalibError;

    fn try_from(value: i32) -> core::result::Result<Self, Self::Error> {
        Self::from_talib_code(value)
    }
}

/// Range selected by a Candle Setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CandleRangeKind {
    /// Absolute Close minus Open.
    RealBody,
    /// Literal High minus Low.
    HighLow,
    /// Upper Shadow plus Lower Shadow.
    Shadows,
}

/// One validated immutable Candle Setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandleSetting {
    range_kind: CandleRangeKind,
    average_period: usize,
    factor: Float,
}

impl CandleSetting {
    /// Creates a validated Candle Setting.
    ///
    /// `average_period` may be `0..=100_000`. A zero period selects the
    /// classified Candle's current range. `factor` must be finite and
    /// nonnegative.
    pub fn new(range_kind: CandleRangeKind, average_period: usize, factor: Float) -> Result<Self> {
        if average_period > 100_000 {
            return Err(TalibError::invalid_period(
                average_period,
                "Candle Setting Average Period must not exceed 100000",
            ));
        }
        if !factor.is_finite() || factor < 0.0 as Float {
            return Err(TalibError::invalid_input(
                "Candle Setting Factor must be finite and nonnegative",
            ));
        }
        Ok(Self {
            range_kind,
            average_period,
            factor,
        })
    }

    const fn default_value(
        range_kind: CandleRangeKind,
        average_period: usize,
        factor: Float,
    ) -> Self {
        Self {
            range_kind,
            average_period,
            factor,
        }
    }

    /// Returns the selected Candle Range Kind.
    #[inline]
    pub const fn range_kind(self) -> CandleRangeKind {
        self.range_kind
    }

    /// Returns the Average Period.
    #[inline]
    pub const fn average_period(self) -> usize {
        self.average_period
    }

    /// Returns the nonnegative threshold Factor.
    #[inline]
    pub const fn factor(self) -> Float {
        self.factor
    }
}

/// Name of one of TA-Lib's eleven Candle Settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum CandleSettingType {
    /// Long real body threshold.
    BodyLong = 0,
    /// Very long real body threshold.
    BodyVeryLong = 1,
    /// Short real body threshold.
    BodyShort = 2,
    /// Doji real body threshold.
    BodyDoji = 3,
    /// Long shadow threshold.
    ShadowLong = 4,
    /// Very long shadow threshold.
    ShadowVeryLong = 5,
    /// Short shadow threshold.
    ShadowShort = 6,
    /// Very short shadow threshold.
    ShadowVeryShort = 7,
    /// Near-distance threshold.
    Near = 8,
    /// Far-distance threshold.
    Far = 9,
    /// Equal-distance threshold.
    Equal = 10,
}

impl CandleSettingType {
    /// All eleven Candle Setting names in pinned TA-Lib order.
    pub const ALL: [Self; 11] = [
        Self::BodyLong,
        Self::BodyVeryLong,
        Self::BodyShort,
        Self::BodyDoji,
        Self::ShadowLong,
        Self::ShadowVeryLong,
        Self::ShadowShort,
        Self::ShadowVeryShort,
        Self::Near,
        Self::Far,
        Self::Equal,
    ];

    #[inline]
    pub(crate) const fn index(self) -> usize {
        self as usize
    }
}

/// Immutable collection of all eleven Candle Settings.
///
/// The default triples exactly match TA-Lib v0.7.1. Configurations own this
/// value; replacing a setting returns a new collection and never changes
/// process-global state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandleSettings {
    settings: [CandleSetting; 11],
}

impl CandleSettings {
    /// Returns one setting by its official name.
    #[inline]
    pub const fn setting(self, setting_type: CandleSettingType) -> CandleSetting {
        self.settings[setting_type.index()]
    }

    /// Returns a new collection with one validated setting replaced.
    #[inline]
    pub fn with_setting(mut self, setting_type: CandleSettingType, setting: CandleSetting) -> Self {
        self.settings[setting_type.index()] = setting;
        self
    }
}

impl Default for CandleSettings {
    fn default() -> Self {
        use CandleRangeKind::{HighLow, RealBody, Shadows};
        Self {
            settings: [
                CandleSetting::default_value(RealBody, 10, 1.0 as Float),
                CandleSetting::default_value(RealBody, 10, 3.0 as Float),
                CandleSetting::default_value(RealBody, 10, 1.0 as Float),
                CandleSetting::default_value(HighLow, 10, 0.1 as Float),
                CandleSetting::default_value(RealBody, 0, 1.0 as Float),
                CandleSetting::default_value(RealBody, 0, 2.0 as Float),
                CandleSetting::default_value(Shadows, 10, 1.0 as Float),
                CandleSetting::default_value(HighLow, 10, 0.1 as Float),
                CandleSetting::default_value(HighLow, 5, 0.2 as Float),
                CandleSetting::default_value(HighLow, 5, 0.6 as Float),
                CandleSetting::default_value(HighLow, 5, 0.05 as Float),
            ],
        }
    }
}

/// Validated nonnegative ratio used only by definitions that expose Penetration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Penetration(Float);

impl Penetration {
    /// Creates a finite nonnegative Penetration value.
    ///
    /// Values above one are valid and retain TA-Lib's supported semantics.
    pub fn new(value: Float) -> Result<Self> {
        if !value.is_finite() || value < 0.0 as Float {
            return Err(TalibError::invalid_input(
                "Penetration must be finite and nonnegative",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the configured ratio.
    #[inline]
    pub const fn value(self) -> Float {
        self.0
    }

    #[inline]
    pub(crate) fn wide_value(self) -> f64 {
        #[cfg(feature = "f32")]
        {
            self.0 as f64
        }
        #[cfg(not(feature = "f32"))]
        {
            self.0
        }
    }
}
