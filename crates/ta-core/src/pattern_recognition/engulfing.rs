//! CDLENGULFING local definition and concrete execution types.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::Result;

/// Immutable CDLENGULFING Indicator Configuration.
///
/// The definition is setting-free, so Candle Settings are owned consistently
/// with other Pattern Recognition configurations but do not affect its fixed
/// Lookback or results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLENGULFINGConfig {
    candle_settings: CandleSettings,
}

impl CDLENGULFINGConfig {
    /// Creates CDLENGULFING with an immutable Candle Settings collection.
    pub fn new(candle_settings: CandleSettings) -> Result<Self> {
        Ok(Self { candle_settings })
    }

    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings {
        self.candle_settings
    }

    /// Returns the fixed Warm-up tick count, identical to Lookback.
    #[inline]
    pub const fn warm_up(&self) -> usize {
        2
    }
}

impl Default for CDLENGULFINGConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
        }
    }
}

impl PatternDefinition for CDLENGULFINGConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLENGULFING"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[]
    }

    fn lookback(&self) -> usize {
        2
    }

    fn transition_start(&self) -> usize {
        2
    }

    fn initial_state(&self) -> Self::State {}

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        let current = context.candle(0);
        let previous = context.candle(1);
        let current_color = context.color(0);
        let previous_color = context.color(1);

        let bullish = current_color == CandleColor::White
            && previous_color == CandleColor::Black
            && ((current.close >= previous.open && current.open < previous.close)
                || (current.close > previous.open && current.open <= previous.close));
        let bearish = current_color == CandleColor::Black
            && previous_color == CandleColor::White
            && ((current.open >= previous.close && current.close < previous.open)
                || (current.open > previous.close && current.close <= previous.open));

        if !bullish && !bearish {
            return PatternSignal::NoMatch;
        }

        let direction = if current_color == CandleColor::White {
            PatternDirection::Bullish
        } else {
            PatternDirection::Bearish
        };
        let strength = if current.open != previous.close && current.close != previous.open {
            PatternStrength::Standard
        } else {
            PatternStrength::Partial
        };
        PatternSignal::Match {
            direction,
            strength,
        }
    }
}

impl_pattern_execution!(
    CDLENGULFINGConfig,
    CDLENGULFINGBatchRunner,
    CDLENGULFINGStream
);
