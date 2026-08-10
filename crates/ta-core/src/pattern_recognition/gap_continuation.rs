//! Local definitions for gap and continuation patterns.

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

#[inline]
const fn direction(color: CandleColor) -> PatternDirection {
    match color {
        CandleColor::White => PatternDirection::Bullish,
        CandleColor::Black => PatternDirection::Bearish,
    }
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

macro_rules! define_gap_config {
    ($config:ident, $runner:ident, $stream:ident, $span:expr, [$($setting:expr),* $(,)?]) => {
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
                maximum_average_period(self.candle_settings, &[$($setting),*]) + $span
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

define_gap_config!(
    CDL2CROWSConfig,
    CDL2CROWSBatchRunner,
    CDL2CROWSStream,
    2,
    [CandleSettingType::BodyLong]
);

impl PatternDefinition for CDL2CROWSConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDL2CROWS"
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
        let first = context.candle(2);
        let second = context.candle(1);
        let third = context.candle(0);
        if context.color(2) == CandleColor::White
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.color(1) == CandleColor::Black
            && context.real_body_gap_up(1, 2)
            && context.color(0) == CandleColor::Black
            && third.open < second.open
            && third.open > second.close
            && third.close > first.open
            && third.close < first.close
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDL3LINESTRIKEConfig,
    CDL3LINESTRIKEBatchRunner,
    CDL3LINESTRIKEStream,
    3,
    [CandleSettingType::Near]
);

impl PatternDefinition for CDL3LINESTRIKEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDL3LINESTRIKE"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::Near]
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
        let first = context.candle(3);
        let second = context.candle(2);
        let third = context.candle(1);
        let fourth = context.candle(0);
        let line_color = context.color(1);
        let fourth_is_opposite = match line_color {
            CandleColor::White => context.color(0) == CandleColor::Black,
            CandleColor::Black => context.color(0) == CandleColor::White,
        };
        let near_first = context.average(CandleSettingType::Near, 3);
        let near_second = context.average(CandleSettingType::Near, 2);
        let strike = match line_color {
            CandleColor::White => {
                third.close > second.close
                    && second.close > first.close
                    && fourth.open > third.close
                    && fourth.close < first.open
            }
            CandleColor::Black => {
                third.close < second.close
                    && second.close < first.close
                    && fourth.open < third.close
                    && fourth.close > first.open
            }
        };
        if context.color(3) == context.color(2)
            && context.color(2) == line_color
            && fourth_is_opposite
            && second.open >= body_low(context, 3) - near_first
            && second.open <= body_high(context, 3) + near_first
            && third.open >= body_low(context, 2) - near_second
            && third.open <= body_high(context, 2) + near_second
            && strike
        {
            standard(direction(line_color))
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLGAPSIDESIDEWHITEConfig,
    CDLGAPSIDESIDEWHITEBatchRunner,
    CDLGAPSIDESIDEWHITEStream,
    2,
    [CandleSettingType::Equal, CandleSettingType::Near]
);

impl PatternDefinition for CDLGAPSIDESIDEWHITEConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLGAPSIDESIDEWHITE"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::Equal, CandleSettingType::Near]
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
        let second = context.candle(1);
        let third = context.candle(0);
        let upside = context.real_body_gap_up(1, 2) && context.real_body_gap_up(0, 2);
        let downside = context.real_body_gap_down(1, 2) && context.real_body_gap_down(0, 2);
        let near = context.average(CandleSettingType::Near, 1);
        let equal = context.average(CandleSettingType::Equal, 1);
        if (upside || downside)
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::White
            && context.real_body(0) >= context.real_body(1) - near
            && context.real_body(0) <= context.real_body(1) + near
            && third.open >= second.open - equal
            && third.open <= second.open + equal
        {
            standard(if upside {
                PatternDirection::Bullish
            } else {
                PatternDirection::Bearish
            })
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLSTICKSANDWICHConfig,
    CDLSTICKSANDWICHBatchRunner,
    CDLSTICKSANDWICHStream,
    2,
    [CandleSettingType::Equal]
);

impl PatternDefinition for CDLSTICKSANDWICHConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLSTICKSANDWICH"
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
        let first = context.candle(2);
        let second = context.candle(1);
        let third = context.candle(0);
        let equal = context.average(CandleSettingType::Equal, 2);
        if context.color(2) == CandleColor::Black
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::Black
            && second.low > first.close
            && third.close <= first.close + equal
            && third.close >= first.close - equal
        {
            standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLTASUKIGAPConfig,
    CDLTASUKIGAPBatchRunner,
    CDLTASUKIGAPStream,
    2,
    [CandleSettingType::Near]
);

impl PatternDefinition for CDLTASUKIGAPConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLTASUKIGAP"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::Near]
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
        let near = context.average(CandleSettingType::Near, 1);
        let near_size = (context.real_body(1) - context.real_body(0)).abs() < near;
        let upside = context.real_body_gap_up(1, 2)
            && context.color(1) == CandleColor::White
            && context.color(0) == CandleColor::Black
            && third.open < second.close
            && third.open > second.open
            && third.close < second.open
            && third.close > first.open.max(first.close)
            && near_size;
        let downside = context.real_body_gap_down(1, 2)
            && context.color(1) == CandleColor::Black
            && context.color(0) == CandleColor::White
            && third.open < second.open
            && third.open > second.close
            && third.close > second.open
            && third.close < first.open.min(first.close)
            && near_size;
        if upside || downside {
            standard(direction(context.color(1)))
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLTRISTARConfig,
    CDLTRISTARBatchRunner,
    CDLTRISTARStream,
    2,
    [CandleSettingType::BodyDoji]
);

impl PatternDefinition for CDLTRISTARConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLTRISTAR"
    }

    fn settings(&self) -> CandleSettings {
        self.candle_settings
    }

    fn referenced_settings(&self) -> &'static [CandleSettingType] {
        &[CandleSettingType::BodyDoji]
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
        // Pinned TA-Lib deliberately uses one rolling total aligned to i-2 for all three doji.
        let threshold = context.average(CandleSettingType::BodyDoji, 2);
        if context.real_body(2) <= threshold
            && context.real_body(1) <= threshold
            && context.real_body(0) <= threshold
        {
            if context.real_body_gap_up(1, 2) && body_high(context, 0) < body_high(context, 1) {
                standard(PatternDirection::Bearish)
            } else if context.real_body_gap_down(1, 2)
                && body_low(context, 0) > body_low(context, 1)
            {
                standard(PatternDirection::Bullish)
            } else {
                PatternSignal::NoMatch
            }
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLUPSIDEGAP2CROWSConfig,
    CDLUPSIDEGAP2CROWSBatchRunner,
    CDLUPSIDEGAP2CROWSStream,
    2,
    [CandleSettingType::BodyLong, CandleSettingType::BodyShort]
);

impl PatternDefinition for CDLUPSIDEGAP2CROWSConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLUPSIDEGAP2CROWS"
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
        if context.color(2) == CandleColor::White
            && context.real_body(2) > context.average(CandleSettingType::BodyLong, 2)
            && context.color(1) == CandleColor::Black
            && context.real_body(1) <= context.average(CandleSettingType::BodyShort, 1)
            && context.real_body_gap_up(1, 2)
            && context.color(0) == CandleColor::Black
            && third.open > second.open
            && third.close < second.close
            && third.close > first.close
        {
            standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_gap_config!(
    CDLXSIDEGAP3METHODSConfig,
    CDLXSIDEGAP3METHODSBatchRunner,
    CDLXSIDEGAP3METHODSStream,
    2,
    []
);

impl PatternDefinition for CDLXSIDEGAP3METHODSConfig {
    type State = ();

    fn name(&self) -> &'static str {
        "CDLXSIDEGAP3METHODS"
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
        let first_color = context.color(2);
        let third = context.candle(0);
        if first_color == context.color(1)
            && first_color != context.color(0)
            && third.open < body_high(context, 1)
            && third.open > body_low(context, 1)
            && third.close < body_high(context, 2)
            && third.close > body_low(context, 2)
            && ((first_color == CandleColor::White && context.real_body_gap_up(1, 2))
                || (first_color == CandleColor::Black && context.real_body_gap_down(1, 2)))
        {
            standard(direction(first_color))
        } else {
            PatternSignal::NoMatch
        }
    }
}
