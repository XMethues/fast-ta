//! Local definitions for the two-Candle body and containment family.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength, Penetration};
use crate::{Float, Result};

fn maximum_average_period(settings: CandleSettings, referenced: &[CandleSettingType]) -> usize {
    referenced.iter().map(|&kind| settings.setting(kind).average_period()).max().unwrap_or(0)
}

#[inline]
const fn standard(direction: PatternDirection) -> PatternSignal {
    PatternSignal::Match { direction, strength: PatternStrength::Standard }
}

#[inline]
const fn direction(color: CandleColor) -> PatternDirection {
    match color { CandleColor::White => PatternDirection::Bullish, CandleColor::Black => PatternDirection::Bearish }
}

#[inline]
const fn opposite_direction(color: CandleColor) -> PatternDirection {
    match color { CandleColor::White => PatternDirection::Bearish, CandleColor::Black => PatternDirection::Bullish }
}

#[inline]
fn body_high(context: &RecognitionContext<'_>, offset: usize) -> Float {
    let candle = context.candle(offset);
    candle.open.max(candle.close)
}

#[inline]
fn body_low(context: &RecognitionContext<'_>, offset: usize) -> Float {
    let candle = context.candle(offset);
    candle.open.min(candle.close)
}

macro_rules! define_two_candle_config {
    ($config:ident, $runner:ident, $stream:ident, [$($setting:expr),+ $(,)?]) => {
        #[doc = concat!("Immutable ", stringify!($config), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config { candle_settings: CandleSettings }

        impl $config {
            /// Creates the definition with an immutable Candle Settings collection.
            pub fn new(candle_settings: CandleSettings) -> Result<Self> { Ok(Self { candle_settings }) }
            /// Returns the owned immutable Candle Settings value.
            #[inline]
            pub const fn candle_settings(&self) -> CandleSettings { self.candle_settings }
            /// Returns the Warm-up tick count, identical to Lookback.
            #[inline]
            pub fn warm_up(&self) -> usize {
                maximum_average_period(self.candle_settings, &[$($setting),+]) + 1
            }
        }

        impl Default for $config {
            fn default() -> Self { Self { candle_settings: CandleSettings::default() } }
        }

        impl_pattern_execution!($config, $runner, $stream);
    };
}

define_two_candle_config!(CDLCOUNTERATTACKConfig, CDLCOUNTERATTACKBatchRunner, CDLCOUNTERATTACKStream,
    [CandleSettingType::BodyLong, CandleSettingType::Equal]);

impl PatternDefinition for CDLCOUNTERATTACKConfig {
    type State = ();
    fn name(&self) -> &'static str { "CDLCOUNTERATTACK" }
    fn settings(&self) -> CandleSettings { self.candle_settings }
    fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyLong, CandleSettingType::Equal] }
    fn lookback(&self) -> usize { self.warm_up() }
    fn transition_start(&self) -> usize { self.lookback() }
    fn initial_state(&self) -> Self::State {}
    fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
        let current = context.candle(0);
        let previous = context.candle(1);
        let equal = context.average(CandleSettingType::Equal, 1);
        if context.color(1) != context.color(0)
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && current.close <= previous.close + equal
            && current.close >= previous.close - equal
        { standard(direction(context.color(0))) } else { PatternSignal::NoMatch }
    }
}

/// Immutable CDLDARKCLOUDCOVER Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLDARKCLOUDCOVERConfig {
    candle_settings: CandleSettings,
    penetration: Penetration,
}

impl CDLDARKCLOUDCOVERConfig {
    /// Creates the definition with immutable Candle Settings and Penetration.
    pub fn new(candle_settings: CandleSettings, penetration: Penetration) -> Result<Self> {
        Ok(Self { candle_settings, penetration })
    }
    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings { self.candle_settings }
    /// Returns the configured Penetration ratio.
    #[inline]
    pub const fn penetration(&self) -> Penetration { self.penetration }
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        self.candle_settings.setting(CandleSettingType::BodyLong).average_period() + 1
    }
}

impl Default for CDLDARKCLOUDCOVERConfig {
    fn default() -> Self {
        Self { candle_settings: CandleSettings::default(), penetration: Penetration::new(0.5 as Float).expect("valid pinned default") }
    }
}

impl PatternDefinition for CDLDARKCLOUDCOVERConfig {
    type State = ();
    fn name(&self) -> &'static str { "CDLDARKCLOUDCOVER" }
    fn settings(&self) -> CandleSettings { self.candle_settings }
    fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyLong] }
    fn lookback(&self) -> usize { self.warm_up() }
    fn transition_start(&self) -> usize { self.lookback() }
    fn initial_state(&self) -> Self::State {}
    fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.color(1) == CandleColor::White
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::Black
            && current.open > previous.high
            && current.close > previous.open
            && current.close < previous.close - context.real_body(1) * self.penetration.value()
        { standard(PatternDirection::Bearish) } else { PatternSignal::NoMatch }
    }
}

impl_pattern_execution!(CDLDARKCLOUDCOVERConfig, CDLDARKCLOUDCOVERBatchRunner, CDLDARKCLOUDCOVERStream);

define_two_candle_config!(CDLDOJISTARConfig, CDLDOJISTARBatchRunner, CDLDOJISTARStream,
    [CandleSettingType::BodyDoji, CandleSettingType::BodyLong]);

impl PatternDefinition for CDLDOJISTARConfig {
    type State = ();
    fn name(&self) -> &'static str { "CDLDOJISTAR" }
    fn settings(&self) -> CandleSettings { self.candle_settings }
    fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyDoji, CandleSettingType::BodyLong] }
    fn lookback(&self) -> usize { self.warm_up() }
    fn transition_start(&self) -> usize { self.lookback() }
    fn initial_state(&self) -> Self::State {}
    fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
        let previous_color = context.color(1);
        let gap = match previous_color {
            CandleColor::White => context.real_body_gap_up(0, 1),
            CandleColor::Black => context.real_body_gap_down(0, 1),
        };
        if context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && gap
        { standard(opposite_direction(previous_color)) } else { PatternSignal::NoMatch }
    }
}

macro_rules! impl_harami_definition {
    ($config:ident, $short_setting:expr, $name:literal) => {
        impl PatternDefinition for $config {
            type State = ();
            fn name(&self) -> &'static str { $name }
            fn settings(&self) -> CandleSettings { self.candle_settings }
            fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyLong, $short_setting] }
            fn lookback(&self) -> usize { self.warm_up() }
            fn transition_start(&self) -> usize { self.lookback() }
            fn initial_state(&self) -> Self::State {}
            fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
                if context.real_body(1) <= context.average(CandleSettingType::BodyLong, 1)
                    || context.real_body(0) > context.average($short_setting, 0)
                { return PatternSignal::NoMatch; }
                let direction = opposite_direction(context.color(1));
                if body_high(context, 0) < body_high(context, 1) && body_low(context, 0) > body_low(context, 1) {
                    standard(direction)
                } else if body_high(context, 0) <= body_high(context, 1) && body_low(context, 0) >= body_low(context, 1) {
                    PatternSignal::Match { direction, strength: PatternStrength::Partial }
                } else { PatternSignal::NoMatch }
            }
        }
    };
}

define_two_candle_config!(CDLHARAMIConfig, CDLHARAMIBatchRunner, CDLHARAMIStream,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]);
impl_harami_definition!(CDLHARAMIConfig, CandleSettingType::BodyShort, "CDLHARAMI");

define_two_candle_config!(CDLHARAMICROSSConfig, CDLHARAMICROSSBatchRunner, CDLHARAMICROSSStream,
    [CandleSettingType::BodyLong, CandleSettingType::BodyDoji]);
impl_harami_definition!(CDLHARAMICROSSConfig, CandleSettingType::BodyDoji, "CDLHARAMICROSS");

define_two_candle_config!(CDLHOMINGPIGEONConfig, CDLHOMINGPIGEONBatchRunner, CDLHOMINGPIGEONStream,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]);

impl PatternDefinition for CDLHOMINGPIGEONConfig {
    type State = ();
    fn name(&self) -> &'static str { "CDLHOMINGPIGEON" }
    fn settings(&self) -> CandleSettings { self.candle_settings }
    fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyLong, CandleSettingType::BodyShort] }
    fn lookback(&self) -> usize { self.warm_up() }
    fn transition_start(&self) -> usize { self.lookback() }
    fn initial_state(&self) -> Self::State {}
    fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.color(1) == CandleColor::Black && context.color(0) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) <= context.average(CandleSettingType::BodyShort, 0)
            && current.open < previous.open && current.close > previous.close
        { standard(PatternDirection::Bullish) } else { PatternSignal::NoMatch }
    }
}

#[inline]
fn is_kicking(context: &RecognitionContext<'_>) -> bool {
    let previous_color = context.color(1);
    let gap = match previous_color {
        CandleColor::Black => context.candle_gap_up(0, 1),
        CandleColor::White => context.candle_gap_down(0, 1),
    };
    previous_color != context.color(0)
        && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
        && context.upper_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
        && context.lower_shadow(1) < context.average(CandleSettingType::ShadowVeryShort, 1)
        && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
        && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
        && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
        && gap
}

macro_rules! define_kicking_config {
    ($config:ident, $runner:ident, $stream:ident, $name:literal, $direction:expr) => {
        define_two_candle_config!($config, $runner, $stream,
            [CandleSettingType::BodyLong, CandleSettingType::ShadowVeryShort]);
        impl PatternDefinition for $config {
            type State = ();
            fn name(&self) -> &'static str { $name }
            fn settings(&self) -> CandleSettings { self.candle_settings }
            fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::BodyLong, CandleSettingType::ShadowVeryShort] }
            fn lookback(&self) -> usize { self.warm_up() }
            fn transition_start(&self) -> usize { self.lookback() }
            fn initial_state(&self) -> Self::State {}
            fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
                if is_kicking(context) { standard(($direction)(context)) } else { PatternSignal::NoMatch }
            }
        }
    };
}

define_kicking_config!(CDLKICKINGConfig, CDLKICKINGBatchRunner, CDLKICKINGStream, "CDLKICKING",
    |context: &RecognitionContext<'_>| direction(context.color(0)));
define_kicking_config!(CDLKICKINGBYLENGTHConfig, CDLKICKINGBYLENGTHBatchRunner, CDLKICKINGBYLENGTHStream,
    "CDLKICKINGBYLENGTH", |context: &RecognitionContext<'_>| {
        let selected = if context.real_body(0) > context.real_body(1) { 0 } else { 1 };
        direction(context.color(selected))
    });

define_two_candle_config!(CDLMATCHINGLOWConfig, CDLMATCHINGLOWBatchRunner, CDLMATCHINGLOWStream,
    [CandleSettingType::Equal]);

impl PatternDefinition for CDLMATCHINGLOWConfig {
    type State = ();
    fn name(&self) -> &'static str { "CDLMATCHINGLOW" }
    fn settings(&self) -> CandleSettings { self.candle_settings }
    fn referenced_settings(&self) -> &'static [CandleSettingType] { &[CandleSettingType::Equal] }
    fn lookback(&self) -> usize { self.warm_up() }
    fn transition_start(&self) -> usize { self.lookback() }
    fn initial_state(&self) -> Self::State {}
    fn transition(&self, context: &RecognitionContext<'_>, _state: &mut Self::State) -> PatternSignal {
        let current = context.candle(0);
        let previous = context.candle(1);
        let equal = context.average(CandleSettingType::Equal, 1);
        if context.color(1) == CandleColor::Black && context.color(0) == CandleColor::Black
            && current.close <= previous.close + equal && current.close >= previous.close - equal
        { standard(PatternDirection::Bullish) } else { PatternSignal::NoMatch }
    }
}
