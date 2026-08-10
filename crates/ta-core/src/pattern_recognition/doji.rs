//! CDLDOJI local definition and concrete execution types.

use super::engine::{PatternDefinition, RecognitionContext};
use super::{
    CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength,
};
use crate::Result;

/// Immutable CDLDOJI Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLDOJIConfig {
    candle_settings: CandleSettings,
}

impl CDLDOJIConfig {
    /// Creates CDLDOJI with an immutable Candle Settings collection.
    pub fn new(candle_settings: CandleSettings) -> Result<Self> {
        Ok(Self { candle_settings })
    }

    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings {
        self.candle_settings
    }

    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        self.candle_settings
            .setting(CandleSettingType::BodyDoji)
            .average_period()
    }
}

impl Default for CDLDOJIConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
        }
    }
}

impl PatternDefinition for CDLDOJIConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLDOJI"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyDoji]
    }

    fn lookback(&self) -> usize {
        self.candle_settings
            .setting(CandleSettingType::BodyDoji)
            .average_period()
    }

    fn transition_start(&self) -> usize {
        self.lookback()
    }

    fn initial_state(&self) -> Self::State {}

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0) {
            PatternSignal::Match {
                direction: PatternDirection::Bullish,
                strength: PatternStrength::Standard,
            }
        } else {
            PatternSignal::NoMatch
        }
    }
}

impl_pattern_execution!(CDLDOJIConfig, CDLDOJIBatchRunner, CDLDOJIStream);
