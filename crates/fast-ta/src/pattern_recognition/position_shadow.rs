//! Local definitions for the two-Candle position and shadow family.

use super::engine::{CandleColor, PatternDefinition, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal};

define_pattern_config!(CDLHAMMERConfig, CDLHAMMERBatchRunner, CDLHAMMERStream);

impl CDLHAMMERConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[
                CandleSettingType::BodyShort,
                CandleSettingType::ShadowLong,
                CandleSettingType::ShadowVeryShort,
                CandleSettingType::Near,
            ],
        ) + 1
    }
}

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
        let previous = context.candle(1);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && context.body_low(0) <= previous.low + context.average(CandleSettingType::Near, 1)
        {
            PatternSignal::standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLHANGINGMANConfig,
    CDLHANGINGMANBatchRunner,
    CDLHANGINGMANStream
);

impl CDLHANGINGMANConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[
                CandleSettingType::BodyShort,
                CandleSettingType::ShadowLong,
                CandleSettingType::ShadowVeryShort,
                CandleSettingType::Near,
            ],
        ) + 1
    }
}

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
        let previous = context.candle(1);
        if context.real_body(0) < context.average(CandleSettingType::BodyShort, 0)
            && context.lower_shadow(0) > context.average(CandleSettingType::ShadowLong, 0)
            && context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            && context.body_low(0) >= previous.high - context.average(CandleSettingType::Near, 1)
        {
            PatternSignal::standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(CDLINNECKConfig, CDLINNECKBatchRunner, CDLINNECKStream);

impl CDLINNECKConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[CandleSettingType::BodyLong, CandleSettingType::Equal],
        ) + 1
    }
}

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
            PatternSignal::standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLINVERTEDHAMMERConfig,
    CDLINVERTEDHAMMERBatchRunner,
    CDLINVERTEDHAMMERStream
);

impl CDLINVERTEDHAMMERConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[
                CandleSettingType::BodyShort,
                CandleSettingType::ShadowLong,
                CandleSettingType::ShadowVeryShort,
            ],
        ) + 1
    }
}

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
            && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            PatternSignal::standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(CDLONNECKConfig, CDLONNECKBatchRunner, CDLONNECKStream);

impl CDLONNECKConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[CandleSettingType::BodyLong, CandleSettingType::Equal],
        ) + 1
    }
}

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
            PatternSignal::standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(CDLPIERCINGConfig, CDLPIERCINGBatchRunner, CDLPIERCINGStream);

impl CDLPIERCINGConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(self.candle_settings, &[CandleSettingType::BodyLong])
            + 1
    }
}

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
            && current.close > previous.close + context.real_body(1) * 0.5
        {
            PatternSignal::standard(PatternDirection::Bullish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLSEPARATINGLINESConfig,
    CDLSEPARATINGLINESBatchRunner,
    CDLSEPARATINGLINESStream
);

impl CDLSEPARATINGLINESConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[
                CandleSettingType::BodyLong,
                CandleSettingType::Equal,
                CandleSettingType::ShadowVeryShort,
            ],
        ) + 1
    }
}

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
                context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            }
            CandleColor::Black => {
                context.upper_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
            }
        };
        if context.color(1) != color
            && current.open <= previous.open + equal
            && current.open >= previous.open - equal
            && context.real_body(0) > context.average(CandleSettingType::BodyLong, 0)
            && short_leading_shadow
        {
            PatternSignal::standard(color.direction())
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLSHOOTINGSTARConfig,
    CDLSHOOTINGSTARBatchRunner,
    CDLSHOOTINGSTARStream
);

impl CDLSHOOTINGSTARConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[
                CandleSettingType::BodyShort,
                CandleSettingType::ShadowLong,
                CandleSettingType::ShadowVeryShort,
            ],
        ) + 1
    }
}

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
            && context.lower_shadow(0) < context.average(CandleSettingType::ShadowVeryShort, 0)
        {
            PatternSignal::standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}

define_pattern_config!(
    CDLTHRUSTINGConfig,
    CDLTHRUSTINGBatchRunner,
    CDLTHRUSTINGStream
);

impl CDLTHRUSTINGConfig {
    /// Returns the Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        super::engine::maximum_average_period(
            self.candle_settings,
            &[CandleSettingType::BodyLong, CandleSettingType::Equal],
        ) + 1
    }
}

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
            && current.close > previous.close + context.average(CandleSettingType::Equal, 1)
            && current.close <= previous.close + context.real_body(1) * 0.5
        {
            PatternSignal::standard(PatternDirection::Bearish)
        } else {
            PatternSignal::NoMatch
        }
    }
}
