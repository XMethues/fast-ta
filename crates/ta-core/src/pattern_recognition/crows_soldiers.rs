//! Local definitions for crow, soldier, and advance patterns.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::Result;

fn maximum_average_period(settings: CandleSettings, referenced: &[CandleSettingType]) -> usize {
    referenced
        .iter()
        .map(|&kind| settings.setting(kind).average_period())
        .max()
        .unwrap_or(0)
}

#[inline]
const fn signal(direction: PatternDirection) -> PatternSignal {
    PatternSignal::Match {
        direction,
        strength: PatternStrength::Standard,
    }
}

macro_rules! define_config {
    ($config:ident, $runner:ident, $stream:ident, $span:expr, [$($setting:expr),+ $(,)?]) => {
        #[doc = concat!("Immutable ", stringify!($config), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config { candle_settings: CandleSettings }
        impl $config {
            pub fn new(candle_settings: CandleSettings) -> Result<Self> { Ok(Self { candle_settings }) }
            #[inline] pub const fn candle_settings(&self) -> CandleSettings { self.candle_settings }
            #[inline] pub fn warm_up(&self) -> usize { maximum_average_period(self.candle_settings, &[$($setting),+]) + $span }
        }
        impl Default for $config { fn default() -> Self { Self { candle_settings: CandleSettings::default() } } }
        impl_pattern_execution!($config, $runner, $stream);
    };
}

macro_rules! definition {
    ($config:ident, $name:literal, [$($setting:expr),+ $(,)?], $body:expr) => {
        impl PatternDefinition for $config {
            type State = ();
            fn name(&self) -> &'static str { $name }
            fn settings(&self) -> CandleSettings { self.candle_settings }
            fn referenced_settings(&self) -> &'static [CandleSettingType] { &[$($setting),+] }
            fn lookback(&self) -> usize { self.warm_up() }
            fn transition_start(&self) -> usize { self.lookback() }
            fn initial_state(&self) -> Self::State {}
            fn transition(&self, context: &RecognitionContext<'_>, _: &mut Self::State) -> PatternSignal { $body(context) }
        }
    };
}

define_config!(
    CDL3BLACKCROWSConfig,
    CDL3BLACKCROWSBatchRunner,
    CDL3BLACKCROWSStream,
    3,
    [CandleSettingType::ShadowVeryShort]
);
definition!(
    CDL3BLACKCROWSConfig,
    "CDL3BLACKCROWS",
    [CandleSettingType::ShadowVeryShort],
    |context: &RecognitionContext<'_>| {
        let prior = context.candle(3);
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        if context.color(3) == CandleColor::White
            && context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::Black
            && second.open < first.open
            && second.open > first.close
            && current.open < second.open
            && current.open > second.close
            && prior.high > first.close
            && first.close > second.close
            && second.close > current.close
            && context.lower_shadow(2) < context.average(CandleSettingType::ShadowVeryShort, 2)
            && context.lower_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
            && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            signal(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDL3STARSINSOUTHConfig,
    CDL3STARSINSOUTHBatchRunner,
    CDL3STARSINSOUTHStream,
    2,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::BodyShort
    ]
);
definition!(
    CDL3STARSINSOUTHConfig,
    "CDL3STARSINSOUTH",
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::BodyShort
    ],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        if context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::Black
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.lower_shadow(2) > context.average(CandleSettingType::ShadowLong, 2)
            && context.real_body(1) < context.real_body(2)
            && second.open > first.close
            && second.open <= first.high
            && second.low < first.close
            && second.low >= first.low
            && context.lower_shadow(1) > context.average(CandleSettingType::ShadowVeryShort, 1)
            && context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && current.low > second.low
            && current.high < second.high
        {
            signal(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDL3WHITESOLDIERSConfig,
    CDL3WHITESOLDIERSBatchRunner,
    CDL3WHITESOLDIERSStream,
    2,
    [
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::BodyShort,
        CandleSettingType::Far,
        CandleSettingType::Near
    ]
);
definition!(
    CDL3WHITESOLDIERSConfig,
    "CDL3WHITESOLDIERS",
    [
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::BodyShort,
        CandleSettingType::Far,
        CandleSettingType::Near
    ],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        if context.color(2) == CandleColor::White
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::White
            && context.upper_shadow(2) < context.average(CandleSettingType::ShadowVeryShort, 2)
            && context.upper_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
            && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && current.close > second.close
            && second.close > first.close
            && second.open > first.open
            && second.open <= first.close + context.average(CandleSettingType::Near, 2)
            && current.open > second.open
            && current.open <= second.close + context.average(CandleSettingType::Near, 1)
            && context.real_body(1)
                > context.real_body(2) - context.average(CandleSettingType::Far, 2)
            && context.real_body(0)
                > context.real_body(1) - context.average(CandleSettingType::Far, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyShort, 0)
        {
            signal(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDLADVANCEBLOCKConfig,
    CDLADVANCEBLOCKBatchRunner,
    CDLADVANCEBLOCKStream,
    2,
    [
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowShort,
        CandleSettingType::Far,
        CandleSettingType::Near,
        CandleSettingType::BodyLong
    ]
);
definition!(
    CDLADVANCEBLOCKConfig,
    "CDLADVANCEBLOCK",
    [
        CandleSettingType::ShadowLong,
        CandleSettingType::ShadowShort,
        CandleSettingType::Far,
        CandleSettingType::Near,
        CandleSettingType::BodyLong
    ],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        let first_body = context.real_body(2);
        let second_body = context.real_body(1);
        let current_body = context.real_body(0);
        let weakening = (second_body < first_body - context.average(CandleSettingType::Far, 2)
            && current_body < second_body + context.average(CandleSettingType::Near, 1))
            || current_body < second_body - context.average(CandleSettingType::Far, 1)
            || (current_body < second_body
                && second_body < first_body
                && (context.upper_shadow(0) > context.average(CandleSettingType::ShadowShort, 0)
                    || context.upper_shadow(1)
                        > context.average(CandleSettingType::ShadowShort, 1)))
            || (current_body < second_body
                && context.upper_shadow(0) > context.average(CandleSettingType::ShadowLong, 0));
        if context.color(2) == CandleColor::White
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::White
            && current.close > second.close
            && second.close > first.close
            && second.open > first.open
            && second.open <= first.close + context.average(CandleSettingType::Near, 2)
            && current.open > second.open
            && current.open <= second.close + context.average(CandleSettingType::Near, 1)
            && first_body > context.average(CandleSettingType::BodyLong, 2)
            && context.upper_shadow(2) < context.average(CandleSettingType::ShadowShort, 2)
            && weakening
        {
            signal(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDLCONCEALBABYSWALLConfig,
    CDLCONCEALBABYSWALLBatchRunner,
    CDLCONCEALBABYSWALLStream,
    3,
    [CandleSettingType::ShadowVeryShort]
);
definition!(
    CDLCONCEALBABYSWALLConfig,
    "CDLCONCEALBABYSWALL",
    [CandleSettingType::ShadowVeryShort],
    |context: &RecognitionContext<'_>| {
        let second = context.candle(2);
        let third = context.candle(1);
        let current = context.candle(0);
        if context.color(3) == CandleColor::Black
            && context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::Black
            && context.lower_shadow(3) < context.average(CandleSettingType::ShadowVeryShort, 3)
            && context.upper_shadow(3) < context.average(CandleSettingType::ShadowVeryShort, 3)
            && context.lower_shadow(2) < context.average(CandleSettingType::ShadowVeryShort, 2)
            && context.upper_shadow(2) < context.average(CandleSettingType::ShadowVeryShort, 2)
            && context.real_body_gap_down(1, 2)
            && context.upper_shadow(1) > context.average(CandleSettingType::ShadowVeryShort, 1)
            && third.high > second.close
            && current.high > third.high
            && current.low < third.low
        {
            signal(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDLIDENTICAL3CROWSConfig,
    CDLIDENTICAL3CROWSBatchRunner,
    CDLIDENTICAL3CROWSStream,
    2,
    [CandleSettingType::ShadowVeryShort, CandleSettingType::Equal]
);
definition!(
    CDLIDENTICAL3CROWSConfig,
    "CDLIDENTICAL3CROWS",
    [CandleSettingType::ShadowVeryShort, CandleSettingType::Equal],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        let first_equal = context.average(CandleSettingType::Equal, 2);
        let second_equal = context.average(CandleSettingType::Equal, 1);
        if context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::Black
            && context.lower_shadow(2) < context.average(CandleSettingType::ShadowVeryShort, 2)
            && context.lower_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
            && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && first.close > second.close
            && second.close > current.close
            && second.open <= first.close + first_equal
            && second.open >= first.close - first_equal
            && current.open <= second.close + second_equal
            && current.open >= second.close - second_equal
        {
            signal(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
);

define_config!(
    CDLSTALLEDPATTERNConfig,
    CDLSTALLEDPATTERNBatchRunner,
    CDLSTALLEDPATTERNStream,
    2,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::Near
    ]
);
definition!(
    CDLSTALLEDPATTERNConfig,
    "CDLSTALLEDPATTERN",
    [
        CandleSettingType::BodyLong,
        CandleSettingType::BodyShort,
        CandleSettingType::ShadowVeryShort,
        CandleSettingType::Near
    ],
    |context: &RecognitionContext<'_>| {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        if context.color(2) == CandleColor::White
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::White
            && current.close > second.close
            && second.close > first.close
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.upper_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
            && second.open > first.open
            && second.open <= first.close + context.average(CandleSettingType::Near, 2)
            && context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && current.open
                >= second.close - context.real_body(0) - context.average(CandleSettingType::Near, 1)
        {
            signal(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
);
