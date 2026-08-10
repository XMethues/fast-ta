//! Stateful local definitions for ordinary and modified Hikkake.

use super::engine::{PatternDefinition, PatternFloat, RecognitionContext};
use super::{CandleSettingType, CandleSettings, PatternDirection, PatternSignal, PatternStrength};
use crate::Result;

#[derive(Debug, Clone, Copy)]
struct PendingConfirmation {
    direction: PatternDirection,
    boundary: PatternFloat,
    remaining_positions: u8,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct State {
    pending: Option<PendingConfirmation>,
}

#[inline]
const fn signal(direction: PatternDirection, strength: PatternStrength) -> PatternSignal {
    PatternSignal::Match {
        direction,
        strength,
    }
}

#[inline]
fn transition(
    state: &mut State,
    formation: Option<(PatternDirection, PatternFloat)>,
    close: PatternFloat,
) -> PatternSignal {
    if let Some((direction, boundary)) = formation {
        state.pending = Some(PendingConfirmation {
            direction,
            boundary,
            remaining_positions: 3,
        });
        return signal(direction, PatternStrength::Standard);
    }

    let Some(pending) = state.pending else {
        return PatternSignal::NoMatch;
    };
    let confirmed = match pending.direction {
        PatternDirection::Bullish => close > pending.boundary,
        PatternDirection::Bearish => close < pending.boundary,
    };
    if confirmed {
        state.pending = None;
        return signal(pending.direction, PatternStrength::Confirmed);
    }

    let remaining_positions = pending.remaining_positions - 1;
    state.pending = (remaining_positions != 0).then_some(PendingConfirmation {
        remaining_positions,
        ..pending
    });
    PatternSignal::NoMatch
}

/// Immutable CDLHIKKAKE Indicator Configuration.
///
/// The ordinary definition is setting-free. Candle Settings are still owned to
/// preserve the common Pattern Recognition configuration seam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLHIKKAKEConfig {
    candle_settings: CandleSettings,
}

impl CDLHIKKAKEConfig {
    /// Creates CDLHIKKAKE with immutable Candle Settings.
    pub fn new(candle_settings: CandleSettings) -> Result<Self> {
        Ok(Self { candle_settings })
    }

    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings {
        self.candle_settings
    }

    /// Returns the fixed Warm-up tick count, identical to Lookback.
    #[inline]
    pub const fn warm_up(&self) -> usize {
        5
    }
}

impl Default for CDLHIKKAKEConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
        }
    }
}

impl PatternDefinition for CDLHIKKAKEConfig {
    type State = State;

    fn name(&self) -> &'static str {
        "CDLHIKKAKE"
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
        2
    }

    fn initial_state(&self) -> Self::State {
        State::default()
    }

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(2);
        let second = context.candle(1);
        let current = context.candle(0);
        let inside = second.high < first.high && second.low > first.low;
        let formation = if inside && current.high < second.high && current.low < second.low {
            Some((PatternDirection::Bullish, second.high))
        } else if inside && current.high > second.high && current.low > second.low {
            Some((PatternDirection::Bearish, second.low))
        } else {
            None
        };
        transition(state, formation, current.close)
    }
}

impl_pattern_execution!(CDLHIKKAKEConfig, CDLHIKKAKEBatchRunner, CDLHIKKAKEStream);

/// Immutable CDLHIKKAKEMOD Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CDLHIKKAKEMODConfig {
    candle_settings: CandleSettings,
}

impl CDLHIKKAKEMODConfig {
    /// Creates CDLHIKKAKEMOD with immutable Candle Settings.
    pub fn new(candle_settings: CandleSettings) -> Result<Self> {
        Ok(Self { candle_settings })
    }

    /// Returns the owned immutable Candle Settings value.
    #[inline]
    pub const fn candle_settings(&self) -> CandleSettings {
        self.candle_settings
    }

    /// Returns the dynamic Warm-up tick count, identical to Lookback.
    #[inline]
    pub fn warm_up(&self) -> usize {
        self.candle_settings
            .setting(CandleSettingType::Near)
            .average_period()
            .max(1)
            + 5
    }

    #[inline]
    fn first_transition(&self) -> usize {
        self.candle_settings
            .setting(CandleSettingType::Near)
            .average_period()
            .max(1)
            + 2
    }
}

impl Default for CDLHIKKAKEMODConfig {
    fn default() -> Self {
        Self {
            candle_settings: CandleSettings::default(),
        }
    }
}

impl PatternDefinition for CDLHIKKAKEMODConfig {
    type State = State;

    fn name(&self) -> &'static str {
        "CDLHIKKAKEMOD"
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
        self.first_transition()
    }

    fn initial_state(&self) -> Self::State {
        State::default()
    }

    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        state: &mut Self::State,
    ) -> PatternSignal {
        let first = context.candle(3);
        let second = context.candle(2);
        let third = context.candle(1);
        let current = context.candle(0);
        let nested_inside = second.high < first.high
            && second.low > first.low
            && third.high < second.high
            && third.low > second.low;
        let formation = if nested_inside
            && current.high < third.high
            && current.low < third.low
            && second.close <= second.low + context.average(CandleSettingType::Near, 2)
        {
            Some((PatternDirection::Bullish, third.high))
        } else if nested_inside
            && current.high > third.high
            && current.low > third.low
            && second.close >= second.high - context.average(CandleSettingType::Near, 2)
        {
            Some((PatternDirection::Bearish, third.low))
        } else {
            None
        };
        transition(state, formation, current.close)
    }
}

impl_pattern_execution!(
    CDLHIKKAKEMODConfig,
    CDLHIKKAKEMODBatchRunner,
    CDLHIKKAKEMODStream
);
