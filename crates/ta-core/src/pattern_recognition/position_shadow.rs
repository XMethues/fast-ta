//! Local definitions for the two-Candle position and shadow family.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::{Float, Result};

fn maximum_average_period(settings: CandleSettings, referenced: &[CandleSettingType]) -> usize {
    referenced
        .iter()
        .map(|&kind| settings.setting(kind).average_period())
        .max()
        .unwrap_or(0)
}

#[inline]
const fn standard(direction: PatternDirection) -> PatternSignal {
    PatternSignal::Match {
        direction,
        strength: PatternStrength::Standard,
    }
}

macro_rules! define_position_shadow_config {
    ($config:ident, $runner:ident, $stream:ident, [$($setting:expr),+ $(,)?]) => {
        #[doc = concat!("Immutable ", stringify!($config), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config {
            candle_settings: CandleSettings,
        }

        impl $config {
            /// Creates the definition with an immutable Candle Settings collection.
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
                maximum_average_period(self.candle_settings, &[$($setting),+]) + 1
            }
        }

        impl Default for $config {
            fn default() -> Self {
                Self { candle_settings: CandleSettings::default() }
            }
        }

        impl_pattern_execution!($config, $runner, $stream);
    };
}

define_position_shadow_config!(
    CDLHAMMERConfig,
    CDLHAMMERBatchRunner,
    CDLHAMMERStream,
    [
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::Near,
    ]
);

impl PatternDefinition for CDLHAMMERConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLHAMMER"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyShort,
            CandleSettingType::ShadowLong,
            CandleSettingType::ShadowVeryShort,
            CandleSettingType::Near,
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
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.upper_shadow(0)
                < context.average(CandleSettingType::ShadowVeryShort, 0)
            && current.open.min(current.close)
                <= previous.low + context.average(CandleSettingType::Near, 1)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLHANGINGMANConfig,
    CDLHANGINGMANBatchRunner,
    CDLHANGINGMANStream,
    [
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::Near,
    ]
);

impl PatternDefinition for CDLHANGINGMANConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLHANGINGMAN"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyShort,
            CandleSettingType::ShadowLong,
            CandleSettingType::ShadowVeryShort,
            CandleSettingType::Near,
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
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.upper_shadow(0)
                < context.average(CandleSettingType::ShadowVeryShort, 0)
            && current.open.min(current.close)
                >= previous.high - context.average(CandleSettingType::Near, 1)
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLINNECKConfig,
    CDLINNECKBatchRunner,
    CDLINNECKStream,
    [CandleSettingType::BodyLong, CandleSettingType::Equal]
);

impl PatternDefinition for CDLINNECKConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLINNECK"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::Equal]
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
        let current = context.candle(0);
        let previous = context.candle(1);
        let equal = context.average(CandleSettingType::Equal, 1);
        if context.color(1) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::White
            && current.open < previous.low
            && current.close <= previous.close + equal
            && current.close >= previous.close
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLINVERTEDHAMMERConfig,
    CDLINVERTEDHAMMERBatchRunner,
    CDLINVERTEDHAMMERStream,
    [
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
    ]
);

impl PatternDefinition for CDLINVERTEDHAMMERConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLINVERTEDHAMMER"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyShort,
            CandleSettingType::ShadowLong,
            CandleSettingType::ShadowVeryShort,
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
        if context.real_body_gap_down(0, 1)
            && context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.upper_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.lower_shadow(0)
                < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLONNECKConfig,
    CDLONNECKBatchRunner,
    CDLONNECKStream,
    [CandleSettingType::BodyLong, CandleSettingType::Equal]
);

impl PatternDefinition for CDLONNECKConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLONNECK"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::Equal]
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
        let current = context.candle(0);
        let previous = context.candle(1);
        let equal = context.average(CandleSettingType::Equal, 1);
        if context.color(1) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::White
            && current.open < previous.low
            && current.close <= previous.low + equal
            && current.close >= previous.low - equal
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLPIERCINGConfig,
    CDLPIERCINGBatchRunner,
    CDLPIERCINGStream,
    [CandleSettingType::BodyLong]
);

impl PatternDefinition for CDLPIERCINGConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLPIERCING"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong]
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
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.color(1) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::White
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && current.open < previous.low
            && current.close < previous.open
            && current.close > previous.close + context.real_body(1) * (0.5 as Float)
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLSEPARATINGLINESConfig,
    CDLSEPARATINGLINESBatchRunner,
    CDLSEPARATINGLINESStream,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::Equal,
        CandleSettingType::ShadowVeryShort,
    ]
);

impl PatternDefinition for CDLSEPARATINGLINESConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLSEPARATINGLINES"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyLong,
            CandleSettingType::Equal,
            CandleSettingType::ShadowVeryShort,
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
        let current = context.candle(0);
        let previous = context.candle(1);
        let equal = context.average(CandleSettingType::Equal, 1);
        let color = context.color(0);
        let short_leading_shadow = match color {
            CandleColor::White => {
                context.lower_shadow(0)
                    < context.average(CandleSettingType::ShadowVeryShort, 0)
            }
            CandleColor::Black => {
                context.upper_shadow(0)
                    < context.average(CandleSettingType::ShadowVeryShort, 0)
            }
        };
        if context.color(1) != color
            && current.open <= previous.open + equal
            && current.open >= previous.open - equal
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && short_leading_shadow
        {
            standard(match color {
                CandleColor::White => PatternDirection::Bullish,
                CandleColor::Black => PatternDirection::Bearish,
            })
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLSHOOTINGSTARConfig,
    CDLSHOOTINGSTARBatchRunner,
    CDLSHOOTINGSTARStream,
    [
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
    ]
);

impl PatternDefinition for CDLSHOOTINGSTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLSHOOTINGSTAR"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyShort,
            CandleSettingType::ShadowLong,
            CandleSettingType::ShadowVeryShort,
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
        if context.real_body_gap_up(0, 1)
            && context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.upper_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.lower_shadow(0)
                < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_position_shadow_config!(
    CDLTHRUSTINGConfig,
    CDLTHRUSTINGBatchRunner,
    CDLTHRUSTINGStream,
    [CandleSettingType::BodyLong, CandleSettingType::Equal]
);

impl PatternDefinition for CDLTHRUSTINGConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLTHRUSTING"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::Equal]
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
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.color(1) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::White
            && current.open < previous.low
            && current.close
                > previous.close + context.average(CandleSettingType::Equal, 1)
            && current.close <= previous.close + context.real_body(1) * (0.5 as Float)
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}
