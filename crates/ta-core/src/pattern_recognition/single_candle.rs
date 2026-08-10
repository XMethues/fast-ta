//! Local definitions for the single-Candle Pattern Recognition family.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::Result;

fn maximum_average_period(settings: CandleSettings, referenced: &[CandleSettingType]) -> usize {
    referenced
        .iter()
        .map(|&setting_type| settings.setting(setting_type).average_period())
        .max()
        .unwrap_or(0)
}

#[inline]
fn color_signal(context: &RecognitionContext<'_>) -> PatternSignal {
    let direction = match context.color(0) {
        CandleColor::White => PatternDirection::Bullish,
        CandleColor::Black => PatternDirection::Bearish,
    };
    PatternSignal::Match {
        direction,
        strength: PatternStrength::Standard,
    }
}

#[inline]
const fn positive_signal() -> PatternSignal {
    PatternSignal::Match {
        direction: PatternDirection::Bullish,
        strength: PatternStrength::Standard,
    }
}

macro_rules! define_single_candle_config {
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
                maximum_average_period(self.candle_settings, &[$($setting),+])
            }
        }

        impl Default for $config {
            fn default() -> Self {
                Self {
                    candle_settings: CandleSettings::default(),
                }
            }
        }

        impl_pattern_execution!($config, $runner, $stream);
    };
}

define_single_candle_config!(
    CDLBELTHOLDConfig,
    CDLBELTHOLDBatchRunner,
    CDLBELTHOLDStream,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowVeryShort,
    ]
);

impl PatternDefinition for CDLBELTHOLDConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLBELTHOLD"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyLong,
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
        let shadow = match context.color(0) {
            CandleColor::White => context.lower_shadow(0),
            CandleColor::Black => context.upper_shadow(0),
        };
        if context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && shadow < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLCLOSINGMARUBOZUConfig,
    CDLCLOSINGMARUBOZUBatchRunner,
    CDLCLOSINGMARUBOZUStream,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLCLOSINGMARUBOZUConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLCLOSINGMARUBOZU"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyLong,
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
        let shadow = match context.color(0) {
            CandleColor::White => context.upper_shadow(0),
            CandleColor::Black => context.lower_shadow(0),
        };
        if context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && shadow < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLDRAGONFLYDOJIConfig,
    CDLDRAGONFLYDOJIBatchRunner,
    CDLDRAGONFLYDOJIStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLDRAGONFLYDOJIConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLDRAGONFLYDOJI"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
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
        let shadow = context.average(CandleSettingType::ShadowVeryShort, 0);
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && context.upper_shadow(0) < shadow
            && context.lower_shadow(0) > shadow
        {
            positive_signal()
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLGRAVESTONEDOJIConfig,
    CDLGRAVESTONEDOJIBatchRunner,
    CDLGRAVESTONEDOJIStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLGRAVESTONEDOJIConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLGRAVESTONEDOJI"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
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
        let shadow = context.average(CandleSettingType::ShadowVeryShort, 0);
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && context.lower_shadow(0) < shadow
            && context.upper_shadow(0) > shadow
        {
            positive_signal()
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLHIGHWAVEConfig,
    CDLHIGHWAVEBatchRunner,
    CDLHIGHWAVEStream,
    [
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowVeryLong
    ]
);

impl PatternDefinition for CDLHIGHWAVEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLHIGHWAVE"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyShort,
            CandleSettingType::ShadowVeryLong,
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
        let shadow = context.average(CandleSettingType::ShadowVeryLong, 0);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.upper_shadow(0) > shadow
            && context.lower_shadow(0) > shadow
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLLONGLEGGEDDOJIConfig,
    CDLLONGLEGGEDDOJIBatchRunner,
    CDLLONGLEGGEDDOJIStream,
    [CandleSettingType::BodyDoji, CandleSettingType::ShadowLong]
);

impl PatternDefinition for CDLLONGLEGGEDDOJIConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLLONGLEGGEDDOJI"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyDoji, CandleSettingType::ShadowLong]
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
        let shadow = context.average(CandleSettingType::ShadowLong, 0);
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && (context.lower_shadow(0) > shadow || context.upper_shadow(0) > shadow)
        {
            positive_signal()
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLLONGLINEConfig,
    CDLLONGLINEBatchRunner,
    CDLLONGLINEStream,
    [CandleSettingType::BodyLong, CandleSettingType::ShadowShort]
);

impl PatternDefinition for CDLLONGLINEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLLONGLINE"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyLong, CandleSettingType::ShadowShort]
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
        let shadow = context.average(CandleSettingType::ShadowShort, 0);
        if context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && context.upper_shadow(0) < shadow
            && context.lower_shadow(0) < shadow
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLMARUBOZUConfig,
    CDLMARUBOZUBatchRunner,
    CDLMARUBOZUStream,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLMARUBOZUConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLMARUBOZU"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyLong,
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
        let shadow = context.average(CandleSettingType::ShadowVeryShort, 0);
        if context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && context.upper_shadow(0) < shadow
            && context.lower_shadow(0) < shadow
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLRICKSHAWMANConfig,
    CDLRICKSHAWMANBatchRunner,
    CDLRICKSHAWMANStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::ShadowLong,
        CandleSettingType::Near
    ]
);

impl PatternDefinition for CDLRICKSHAWMANConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLRICKSHAWMAN"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
            CandleSettingType::ShadowLong,
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
        let midpoint = context.low(0) + context.high_low_range(0) / 2.0;
        let near = context.average(CandleSettingType::Near, 0);
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.upper_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.body_low(0) <= midpoint + near
            && context.body_high(0) >= midpoint - near
        {
            positive_signal()
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLSHORTLINEConfig,
    CDLSHORTLINEBatchRunner,
    CDLSHORTLINEStream,
    [CandleSettingType::BodyShort, CandleSettingType::ShadowShort]
);

impl PatternDefinition for CDLSHORTLINEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLSHORTLINE"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyShort, CandleSettingType::ShadowShort]
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
        let shadow = context.average(CandleSettingType::ShadowShort, 0);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.upper_shadow(0) < shadow
            && context.lower_shadow(0) < shadow
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLSPINNINGTOPConfig,
    CDLSPINNINGTOPBatchRunner,
    CDLSPINNINGTOPStream,
    [CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLSPINNINGTOPConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLSPINNINGTOP"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyShort]
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
        let body = context.real_body(0);
        if context.upper_shadow(0) > body
            && context.lower_shadow(0) > body
            && body < context.average(CandleSettingType::BodyShort, 0)
        {
            color_signal(context)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_single_candle_config!(
    CDLTAKURIConfig,
    CDLTAKURIBatchRunner,
    CDLTAKURIStream,
    [
        CandleSettingType::BodyDoji,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::ShadowVeryLong
    ]
);

impl PatternDefinition for CDLTAKURIConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLTAKURI"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[
            CandleSettingType::BodyDoji,
            CandleSettingType::ShadowVeryShort,
            CandleSettingType::ShadowVeryLong,
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
        if context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowVeryLong, 0)
        {
            positive_signal()
        } else {
            PatternSignal::NoMatch
        }
    }
}
