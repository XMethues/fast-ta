//! CDLDOJI local definition and concrete execution types.

use super::engine::{compute_single_setting_batch_into, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};

define_pattern_config!(CDLDOJIConfig, CDLDOJIBatchRunner, CDLDOJIStream);

impl CDLDOJIConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(self.candle_settings, &[CandleSettingType::BodyDoji])
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

    fn compute_batch_into(
        &self,
        input: super::CandleInput<'_>,
        output: &mut [PatternSignal],
    ) -> bool {
        let setting = self.candle_settings.setting(CandleSettingType::BodyDoji);
        compute_single_setting_batch_into::<1>(
            input,
            output,
            self.lookback(),
            setting,
            |_source_index, candle, averages| {
                if candle.real_body() <= averages[0] {
                    PatternSignal::Match {
                        direction: PatternDirection::Bullish,
                        strength: PatternStrength::Standard,
                    }
                } else {
                    PatternSignal::NoMatch
                }
            },
        );
        true
    }
}
