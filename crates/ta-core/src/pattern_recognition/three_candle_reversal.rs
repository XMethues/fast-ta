//! Local definitions for the three-Candle reversal and star family.

use super::engine::{maximum_average_period, CandleColor, PatternDefinition, RecognitionContext};
use super::{
    CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength,
    Penetration,
};
use crate::{Float, Result};

#[inline]
const fn standard(direction: PatternDirection) -> PatternSignal {
    PatternSignal::Match {
        direction,
        strength: PatternStrength::Standard,
    }
}

#[inline]
fn body_high(context: &RecognitionContext<'_>, offset: usize) -> f64 {
    context.body_high(offset)
}

#[inline]
fn body_low(context: &RecognitionContext<'_>, offset: usize) -> f64 {
    context.body_low(offset)
}

macro_rules! define_star_config {
    ($config:ident, $runner:ident, $stream:ident, [$($setting:expr),+ $(,)?]) => {
        #[doc = concat!("Immutable ", stringify!($config), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config {
            candle_settings: CandleSettings,
            penetration: Penetration,
        }

        impl $config {
            /// Creates the definition with immutable Candle Settings and Penetration.
            pub fn new(candle_settings: CandleSettings, penetration: Penetration) -> Result<Self> {
                Ok(Self { candle_settings, penetration })
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
                maximum_average_period(self.candle_settings, &[$($setting),+]) + 2
            }
        }

        impl Default for $config {
            fn default() -> Self {
                Self {
                    candle_settings: CandleSettings::default(),
                    penetration: Penetration::new(0.3 as Float).expect("valid pinned default"),
                }
            }
        }

        impl_pattern_execution!($config, $runner, $stream);
    };
}

define_pattern_config!(
    CDL3INSIDEConfig,
    CDL3INSIDEBatchRunner,
    CDL3INSIDEStream,
    2,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDL3INSIDEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDL3INSIDE"
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

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        let first_color = context.color(2);
        let reversal = match first_color {
            CandleColor::White => {
                context.color(0) == CandleColor::Black && third.close < first.open
            }
            CandleColor::Black => {
                context.color(0) == CandleColor::White && third.close > first.open
            }
        };
        if body_high(context, 1) < body_high(context, 2)
            && body_low(context, 1) > body_low(context, 2)
            && reversal
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyShort, 1)
        {
            standard(context.color(0).direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDL3OUTSIDEConfig,
    CDL3OUTSIDEBatchRunner,
    CDL3OUTSIDEStream,
    3,
    []
);

impl PatternDefinition for CDL3OUTSIDEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDL3OUTSIDE"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[]
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
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let second = context.candle(1);
        let third = context.candle(0);
        let second_color = context.color(1);
        let matches = match second_color {
            CandleColor::White => {
                context.color(2) == CandleColor::Black
                    && second.close > first.open
                    && second.open < first.close
                    && third.close > second.close
            }
            CandleColor::Black => {
                context.color(2) == CandleColor::White
                    && second.open > first.close
                    && second.close < first.open
                    && third.close < second.close
            }
        };
        if matches {
            standard(second_color.direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_star_config!(
    CDLABANDONEDBABYConfig,
    CDLABANDONEDBABYBatchRunner,
    CDLABANDONEDBABYStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::BodyLong,
        CandleSettingType::BodyShort,
    ]
);

impl PatternDefinition for CDLABANDONEDBABYConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLABANDONEDBABY"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
            CandleSettingType::BodyLong,
            CandleSettingType::BodyShort,
        ]
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
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        let third_color = context.color(0);
        let reversal = match third_color {
            CandleColor::Black => {
                context.color(2) == CandleColor::White
                    && third.close
                        < first.close - context.real_body(2) * self.penetration.wide_value()
                    && context.candle_gap_up(1, 2)
                    && context.candle_gap_down(0, 1)
            }
            CandleColor::White => {
                context.color(2) == CandleColor::Black
                    && third.close
                        > first.close + context.real_body(2) * self.penetration.wide_value()
                    && context.candle_gap_down(1, 2)
                    && context.candle_gap_up(0, 1)
            }
        };
        if context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyDoji, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
            && reversal
        {
            standard(third_color.direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_star_config!(
    CDLEVENINGDOJISTARConfig,
    CDLEVENINGDOJISTARBatchRunner,
    CDLEVENINGDOJISTARStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::BodyLong,
        CandleSettingType::BodyShort,
    ]
);

impl PatternDefinition for CDLEVENINGDOJISTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLEVENINGDOJISTAR"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
            CandleSettingType::BodyLong,
            CandleSettingType::BodyShort,
        ]
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
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        if context.color(2) == CandleColor::White
            && context.color(0) == CandleColor::Black
            && context.real_body_gap_up(1, 2)
            && third.close < first.close - context.real_body(2) * self.penetration.wide_value()
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyDoji, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_star_config!(
    CDLEVENINGSTARConfig,
    CDLEVENINGSTARBatchRunner,
    CDLEVENINGSTARStream,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLEVENINGSTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLEVENINGSTAR"
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

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        if context.color(2) == CandleColor::White
            && context.color(0) == CandleColor::Black
            && context.real_body_gap_up(1, 2)
            && third.close < first.close - context.real_body(2) * self.penetration.wide_value()
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyShort, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_star_config!(
    CDLMORNINGDOJISTARConfig,
    CDLMORNINGDOJISTARBatchRunner,
    CDLMORNINGDOJISTARStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::BodyLong,
        CandleSettingType::BodyShort,
    ]
);

impl PatternDefinition for CDLMORNINGDOJISTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLMORNINGDOJISTAR"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
            CandleSettingType::BodyLong,
            CandleSettingType::BodyShort,
        ]
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
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        if context.color(2) == CandleColor::Black
            && context.color(0) == CandleColor::White
            && context.real_body_gap_down(1, 2)
            && third.close > first.close + context.real_body(2) * self.penetration.wide_value()
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyDoji, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_star_config!(
    CDLMORNINGSTARConfig,
    CDLMORNINGSTARBatchRunner,
    CDLMORNINGSTARStream,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLMORNINGSTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLMORNINGSTAR"
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

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let third = context.candle(0);
        if context.color(2) == CandleColor::Black
            && context.color(0) == CandleColor::White
            && context.real_body_gap_down(1, 2)
            && third.close > first.close + context.real_body(2) * self.penetration.wide_value()
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) <= context.average(CandleSettingType::BodyShort, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLUNIQUE3RIVERConfig,
    CDLUNIQUE3RIVERBatchRunner,
    CDLUNIQUE3RIVERStream,
    2,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLUNIQUE3RIVERConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLUNIQUE3RIVER"
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

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        _state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let second = context.candle(1);
        let third = context.candle(0);
        if context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::White
            && second.close > first.close
            && second.open <= first.open
            && second.low < first.low
            && third.open > second.low
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}
