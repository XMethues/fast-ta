//! Local definitions for five-Candle long formations.

use super::engine::{maximum_average_period, CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, Penetration};
use crate::{Float, Result};

macro_rules! definition {
    ($config:ident, $name:literal, [$($setting:expr),+ $(,)?], $body:expr) => {
        impl PatternDefinition for $config {
            type State = ();

            fn name(&self) -> &'static str {
                $name
            }

            fn settings(&self) -> CandleSettings {
                self.candle_settings
            }

            fn referenced_settings(&self) -> &'static [CandleSettingType] {
                &[$($setting),+]
            }

            fn lookback(&self) -> usize {
                self.warm_up()
            }

            fn transition_start(&self) -> usize {
                self.lookback()
            }

            fn initial_state(&self) -> Self::State {}

            fn transition(
                &self,
                context: &RecognitionContext<'_>,
                _: &mut Self::State,
            ) -> PatternSignal {
                $body(context)
            }
        }
    };
}

define_pattern_config!(
    CDLBREAKAWAYConfig,
    CDLBREAKAWAYBatchRunner,
    CDLBREAKAWAYStream,
    4,
    [CandleSettingType::BodyLong]
);

definition!(
    CDLBREAKAWAYConfig,
    "CDLBREAKAWAY",
    [CandleSettingType::BodyLong],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(4);
        let second = context.candle(3);
        let third = context.candle(2);
        let fourth = context.candle(1);
        let fifth = context.candle(0);
        let first_color = context.color(4);
        let fifth_color = context.color(0);

        let breaks_away = match first_color {
            CandleColor::Black => {
                context.real_body_gap_down(3, 4)
                    && third.high < second.high
                    && third.low < second.low
                    && fourth.high < third.high
                    && fourth.low < third.low
                    && fifth.close > second.open
                    && fifth.close < first.close
            }
            CandleColor::White => {
                context.real_body_gap_up(3, 4)
                    && third.high > second.high
                    && third.low > second.low
                    && fourth.high > third.high
                    && fourth.low > third.low
                    && fifth.close < second.open
                    && fifth.close > first.close
            }
        };

        if first_color == context.color(3)
            && context.color(3) == context.color(1)
            && context.color(1) != fifth_color
            && context.real_body(4) > context.average(CandleSettingType::BodyLong, 4)
            && breaks_away
        {
            PatternSignal::standard(fifth_color.direction())
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_pattern_config!(
    CDLLADDERBOTTOMConfig,
    CDLLADDERBOTTOMBatchRunner,
    CDLLADDERBOTTOMStream,
    4,
    [CandleSettingType::ShadowVeryShort]
);

definition!(
    CDLLADDERBOTTOMConfig,
    "CDLLADDERBOTTOM",
    [CandleSettingType::ShadowVeryShort],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(4);
        let second = context.candle(3);
        let third = context.candle(2);
        let fourth = context.candle(1);
        let fifth = context.candle(0);

        if context.color(4) == CandleColor::Black
            && context.color(3) == CandleColor::Black
            && context.color(2) == CandleColor::Black
            && first.open > second.open
            && second.open > third.open
            && first.close > second.close
            && second.close > third.close
            && context.color(1) == CandleColor::Black
            && context.upper_shadow(1) > context.average(CandleSettingType::ShadowVeryShort, 1)
            && context.color(0) == CandleColor::White
            && fifth.open > fourth.open
            && fifth.close > fourth.high
        {
            PatternSignal::standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

/// Immutable CDLMATHOLD Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLMATHOLDConfig {
    candle_settings: CandleSettings,
    penetration: Penetration,
}

impl CDLMATHOLDConfig {
    /// Creates the configuration with immutable Candle Settings and Penetration.
    pub fn new(candle_settings: CandleSettings, penetration: Penetration) -> Result<Self> {
        Ok(Self {
            candle_settings,
            penetration,
        })
    }

    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings {
        self.candle_settings
    }

    /// Returns the configured Penetration ratio.
    #[inline]
    pub const fn penetration(&self) -> Penetration {
        self.penetration
    }

    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        maximum_average_period(
            self.candle_settings,
            &[CandleSettingType::BodyLong, CandleSettingType::BodyShort],
        ) + 4
    }
}

impl Default for CDLMATHOLDConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
            penetration: Penetration::new(0.5 as Float).expect("valid pinned default"),
        }
    }
}

impl_pattern_execution!(CDLMATHOLDConfig, CDLMATHOLDBatchRunner, CDLMATHOLDStream);

impl PatternDefinition for CDLMATHOLDConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLMATHOLD"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::BodyShort]
    }

    fn lookback(&self) -> usize {
        self.warm_up()
    }

    fn transition_start(&self) -> usize {
        self.lookback()
    }

    fn initial_state(&self) -> Self::State {}

    fn transition(&self, context: &RecognitionContext<'_>, _: &mut Self::State) -> PatternSignal {
        let first = context.candle(4);
        let second = context.candle(3);
        let third = context.candle(2);
        let fourth = context.candle(1);
        let fifth = context.candle(0);
        let first_body = context.real_body(4);
        let reaction_floor = first.close - first_body * self.penetration.wide_value();
        let third_body_low = context.body_low(2);
        let third_body_high = context.body_high(2);
        let fourth_body_low = context.body_low(1);
        let fourth_body_high = context.body_high(1);

        if context.color(4) == CandleColor::White
            && context.color(3) == CandleColor::Black
            && context.color(0) == CandleColor::White
            && context.real_body_gap_up(3, 4)
            && third_body_low < first.close
            && fourth_body_low < first.close
            && third_body_low > reaction_floor
            && fourth_body_low > reaction_floor
            && third_body_high < second.open
            && fourth_body_high < third_body_high
            && fifth.open > fourth.close
            && fifth.close > second.high.max(third.high).max(fourth.high)
            && first_body > context.average(CandleSettingType::BodyLong, 4)
            && context.real_body(3) < context.average(CandleSettingType::BodyShort, 3)
            && context.real_body(2) < context.average(CandleSettingType::BodyShort, 2)
            && context.real_body(1) < context.average(CandleSettingType::BodyShort, 1)
        {
            PatternSignal::standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLRISEFALL3METHODSConfig,
    CDLRISEFALL3METHODSBatchRunner,
    CDLRISEFALL3METHODSStream,
    4,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLRISEFALL3METHODSConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLRISEFALL3METHODS"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::BodyShort]
    }

    fn lookback(&self) -> usize {
        self.warm_up()
    }

    fn transition_start(&self) -> usize {
        self.lookback()
    }

    fn initial_state(&self) -> Self::State {}

    fn transition(&self, context: &RecognitionContext<'_>, _: &mut Self::State) -> PatternSignal {
        let first = context.candle(4);
        let second = context.candle(3);
        let third = context.candle(2);
        let fourth = context.candle(1);
        let fifth = context.candle(0);
        let first_color = context.color(4);
        let reactions_move_against_first = match first_color {
            CandleColor::White => {
                third.close < second.close
                    && fourth.close < third.close
                    && fifth.open > fourth.close
                    && fifth.close > first.close
            }
            CandleColor::Black => {
                third.close > second.close
                    && fourth.close > third.close
                    && fifth.open < fourth.close
                    && fifth.close < first.close
            }
        };
        let reaction_within_first_range = |candle: super::engine::WideCandle| {
            let (body_low, body_high) = if candle.close >= candle.open {
                (candle.open, candle.close)
            } else {
                (candle.close, candle.open)
            };
            body_low < first.high && body_high > first.low
        };

        if first_color != context.color(3)
            && context.color(3) == context.color(2)
            && context.color(2) == context.color(1)
            && context.color(1) != context.color(0)
            && reaction_within_first_range(second)
            && reaction_within_first_range(third)
            && reaction_within_first_range(fourth)
            && reactions_move_against_first
            && context.real_body(4) > context.average(CandleSettingType::BodyLong, 4)
            && context.real_body(3) < context.average(CandleSettingType::BodyShort, 3)
            && context.real_body(2) < context.average(CandleSettingType::BodyShort, 2)
            && context.real_body(1) < context.average(CandleSettingType::BodyShort, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
        {
            PatternSignal::standard(first_color.direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}
