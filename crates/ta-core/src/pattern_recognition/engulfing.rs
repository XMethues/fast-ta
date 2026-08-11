//! CDLENGULFING local definition and concrete execution types.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext, WideCandle};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};

// Setting-free: owned Candle Settings do not affect its fixed Lookback or results.
define_pattern_config!(
    CDLENGULFINGConfig,
    CDLENGULFINGBatchRunner,
    CDLENGULFINGStream
);

#[inline]
fn engulfing_signal(
    current: WideCandle,
    previous: WideCandle,
    current_color: CandleColor,
    previous_color: CandleColor,
) -> PatternSignal {
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

impl CDLENGULFINGConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(self.candle_settings, &[]) + 2
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
        engulfing_signal(
            context.candle(0),
            context.candle(1),
            context.color(0),
            context.color(1),
        )
    }

    fn compute_batch_into(
        &self,
        input: super::CandleInput<'_>,
        output: &mut [PatternSignal],
    ) -> bool {
        for (output_value, source_index) in output.iter_mut().zip(self.lookback()..input.len()) {
            let current = input.candle(source_index);
            let previous = input.candle(source_index - 1);
            *output_value = engulfing_signal(
                current.into(),
                previous.into(),
                current.color(),
                previous.color(),
            );
        }
        true
    }
}
