//! CDLDOJI local definition and concrete execution types.

use super::engine::{PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};

define_pattern_config!(
    CDLDOJIConfig,
    CDLDOJIBatchRunner,
    CDLDOJIStream,
    0,
    [CandleSettingType::BodyDoji]
);

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
