//! Local definitions for the two-Candle body and containment family.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
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

define_pattern_config!(
    CDLCOUNTERATTACKConfig,
    CDLCOUNTERATTACKBatchRunner,
    CDLCOUNTERATTACKStream,
    1,
    [CandleSettingType::BodyLong, CandleSettingType::Equal]
);

impl PatternDefinition for CDLCOUNTERATTACKConfig {
    type State = ();
    fn name(&self) -> &'static str {
        "CDLCOUNTERATTACK"
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
        if context.color(1) != context.color(0)
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && current.close <= previous.close + equal
            && current.close >= previous.close - equal
        {
            standard(context.color(0).direction())
        } else {
            PatternSignal::NoMatch
        }
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
        self.candle_settings
            .setting(CandleSettingType::BodyLong)
            .average_period()
            + 1
    }
}

impl Default for CDLDARKCLOUDCOVERConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
            penetration: Penetration::new(0.5 as Float).expect("valid pinned default"),
        }
    }
}

impl PatternDefinition for CDLDARKCLOUDCOVERConfig {
    type State = ();
    fn name(&self) -> &'static str {
        "CDLDARKCLOUDCOVER"
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
        if context.color(1) == CandleColor::White
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.color(0) == CandleColor::Black
            && current.open > previous.high
            && current.close > previous.open
            && current.close < previous.close - context.real_body(1) * self.penetration.wide_value()
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

impl_pattern_execution!(
    CDLDARKCLOUDCOVERConfig,
    CDLDARKCLOUDCOVERBatchRunner,
    CDLDARKCLOUDCOVERStream
);

define_pattern_config!(
    CDLDOJISTARConfig,
    CDLDOJISTARBatchRunner,
    CDLDOJISTARStream,
    1,
    [CandleSettingType::BodyDoji, CandleSettingType::BodyLong]
);

impl PatternDefinition for CDLDOJISTARConfig {
    type State = ();
    fn name(&self) -> &'static str {
        "CDLDOJISTAR"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyDoji, CandleSettingType::BodyLong]
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
        let previous_color = context.color(1);
        let gap = match previous_color {
            CandleColor::White => context.real_body_gap_up(0, 1),
            CandleColor::Black => context.real_body_gap_down(0, 1),
        };
        if context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) <= context.average(CandleSettingType::BodyDoji, 0)
            && gap
        {
            standard(previous_color.opposite_direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

macro_rules! impl_harami_definition {
    ($config:ident, $short_setting:expr, $name:literal) => {
        impl PatternDefinition for $config {
            type State = ();
            fn name(&self) -> &'static str {
                $name
            }
            fn settings(&self) -> CandleSettings {
                self.candle_settings
            }
            fn referenced_settings(&self) -> &'static [CandleSettingType] {
                &[CandleSettingType::BodyLong, $short_setting]
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
                if context.real_body(1) <= context.average(CandleSettingType::BodyLong, 1)
                    || context.real_body(0) > context.average($short_setting, 0)
                {
                    return PatternSignal::NoMatch;
                }
                let direction = context.color(1).opposite_direction();
                if body_high(context, 0) < body_high(context, 1)
                    && body_low(context, 0) > body_low(context, 1)
                {
                    standard(direction)
                } else if body_high(context, 0) <= body_high(context, 1)
                    && body_low(context, 0) >= body_low(context, 1)
                {
                    PatternSignal::Match {
                        direction,
                        strength: PatternStrength::Partial,
                    }
                } else {
                    PatternSignal::NoMatch
                }
            }
        }
    };
}

define_pattern_config!(
    CDLHARAMIConfig,
    CDLHARAMIBatchRunner,
    CDLHARAMIStream,
    1,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);
impl_harami_definition!(CDLHARAMIConfig, CandleSettingType::BodyShort, "CDLHARAMI");

define_pattern_config!(
    CDLHARAMICROSSConfig,
    CDLHARAMICROSSBatchRunner,
    CDLHARAMICROSSStream,
    1,
    [CandleSettingType::BodyLong, CandleSettingType::BodyDoji]
);
impl_harami_definition!(
    CDLHARAMICROSSConfig,
    CandleSettingType::BodyDoji,
    "CDLHARAMICROSS"
);

define_pattern_config!(
    CDLHOMINGPIGEONConfig,
    CDLHOMINGPIGEONBatchRunner,
    CDLHOMINGPIGEONStream,
    1,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLHOMINGPIGEONConfig {
    type State = ();
    fn name(&self) -> &'static str {
        "CDLHOMINGPIGEON"
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
        let current = context.candle(0);
        let previous = context.candle(1);
        if context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::Black
            && context.real_body(1) > context.average(CandleSettingType::BodyLong, 1)
            && context.real_body(0) <= context.average(CandleSettingType::BodyShort, 0)
            && current.open < previous.open
            && current.close > previous.close
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
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

define_pattern_config!(
    CDLKICKINGConfig,
    CDLKICKINGBatchRunner,
    CDLKICKINGStream,
    1,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLKICKINGConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLKICKING"
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
        if is_kicking(context) {
            standard(context.color(0).direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLKICKINGBYLENGTHConfig,
    CDLKICKINGBYLENGTHBatchRunner,
    CDLKICKINGBYLENGTHStream,
    1,
    [
        CandleSettingType::BodyLong,
        CandleSettingType::ShadowVeryShort
    ]
);

impl PatternDefinition for CDLKICKINGBYLENGTHConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLKICKINGBYLENGTH"
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
        if !is_kicking(context) {
            return PatternSignal::NoMatch;
        }
        let selected = if context.real_body(0) > context.real_body(1) {
            0
        } else {
            1
        };
        standard(context.color(selected).direction())
    }
}

define_pattern_config!(
    CDLMATCHINGLOWConfig,
    CDLMATCHINGLOWBatchRunner,
    CDLMATCHINGLOWStream,
    1,
    [CandleSettingType::Equal]
);

impl PatternDefinition for CDLMATCHINGLOWConfig {
    type State = ();
    fn name(&self) -> &'static str {
        "CDLMATCHINGLOW"
    }
    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }
    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::Equal]
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
            && context.color(0) == CandleColor::Black
            && current.close <= previous.close + equal
            && current.close >= previous.close - equal
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}
