//! CDLDOJI local definition and concrete execution types.

use super::engine::{widen, PatternDefinition, RecognitionContext};
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
        let period = setting.average_period();
        let factor = widen(setting.factor());
        let divisor = if setting.range_kind() == super::CandleRangeKind::Shadows {
            2.0
        } else {
            1.0
        };
        let signal = |candle: super::Candle, average: f64| {
            if candle.real_body() <= average {
                PatternSignal::Match {
                    direction: PatternDirection::Bullish,
                    strength: PatternStrength::Standard,
                }
            } else {
                PatternSignal::NoMatch
            }
        };

        if period == 0 {
            for (output_value, source_index) in output.iter_mut().zip(0..input.len()) {
                let candle = input.candle(source_index);
                let average = factor * candle.range(setting.range_kind()) / divisor;
                *output_value = signal(candle, average);
            }
            return true;
        }

        if input.len() < period {
            return true;
        }

        let mut total = (0..period)
            .map(|source_index| input.candle(source_index).range(setting.range_kind()))
            .sum::<f64>();
        for (output_value, source_index) in output.iter_mut().zip(period..input.len()) {
            let candle = input.candle(source_index);
            let average = factor * (total / period as f64) / divisor;
            *output_value = signal(candle, average);
            total += candle.range(setting.range_kind())
                - input
                    .candle(source_index - period)
                    .range(setting.range_kind());
        }
        true
    }
}
