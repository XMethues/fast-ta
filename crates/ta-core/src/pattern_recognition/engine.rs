//! Crate-private generic execution engine for Pattern Recognition definitions.

use super::{
    Candle, CandleInput, CandleRangeKind, CandleSettingType, CandleSettings, PatternSignal,
};
use crate::{common::validate_finite_value, TalibError};
use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    CompactOutput, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Pinned C evaluates f32-rounded inputs in double-precision arithmetic.
pub(crate) type PatternFloat = f64;

#[inline]
pub(crate) fn widen(value: Float) -> PatternFloat {
    value as PatternFloat
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandleColor {
    White,
    Black,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct WideCandle {
    pub(crate) open: PatternFloat,
    pub(crate) high: PatternFloat,
    pub(crate) low: PatternFloat,
    pub(crate) close: PatternFloat,
}

impl From<Candle> for WideCandle {
    fn from(candle: Candle) -> Self {
        Self {
            open: widen(candle.open),
            high: widen(candle.high),
            low: widen(candle.low),
            close: widen(candle.close),
        }
    }
}

impl Candle {
    #[inline]
    fn real_body(self) -> PatternFloat {
        (widen(self.close) - widen(self.open)).abs()
    }

    #[inline]
    fn high_low_range(self) -> PatternFloat {
        widen(self.high) - widen(self.low)
    }

    #[inline]
    fn upper_shadow(self) -> PatternFloat {
        let body_high = if self.close >= self.open {
            self.close
        } else {
            self.open
        };
        widen(self.high) - widen(body_high)
    }

    #[inline]
    fn lower_shadow(self) -> PatternFloat {
        let body_low = if self.close >= self.open {
            self.open
        } else {
            self.close
        };
        widen(body_low) - widen(self.low)
    }

    #[inline]
    fn shadows(self) -> PatternFloat {
        self.upper_shadow() + self.lower_shadow()
    }

    #[inline]
    fn color(self) -> CandleColor {
        if self.close >= self.open {
            CandleColor::White
        } else {
            CandleColor::Black
        }
    }

    #[inline]
    fn range(self, range_kind: CandleRangeKind) -> PatternFloat {
        match range_kind {
            CandleRangeKind::RealBody => self.real_body(),
            CandleRangeKind::HighLow => self.high_low_range(),
            CandleRangeKind::Shadows => self.shadows(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CandleFrame {
    candle: Candle,
    averages: [PatternFloat; 11],
}

#[derive(Debug, Clone)]
struct CandleHistory {
    frames: Vec<CandleFrame>,
    first: usize,
    capacity: usize,
}

impl CandleHistory {
    fn new(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            first: 0,
            capacity,
        }
    }

    fn clear(&mut self) {
        self.frames.clear();
        self.first = 0;
    }

    fn push(&mut self, frame: CandleFrame) {
        if self.frames.len() < self.capacity {
            self.frames.push(frame);
        } else {
            self.frames[self.first] = frame;
            self.first = (self.first + 1) % self.capacity;
        }
    }

    #[inline]
    fn frame(&self, offset: usize) -> &CandleFrame {
        assert!(
            offset < self.frames.len(),
            "pattern requested unavailable Candle offset"
        );
        let newest = (self.first + self.frames.len() - 1) % self.capacity;
        let index = (newest + self.capacity - offset) % self.capacity;
        &self.frames[index]
    }
}

/// Offset-aware Candle geometry and Candle Average access for local definitions.
pub(crate) struct RecognitionContext<'a> {
    history: &'a CandleHistory,
}

#[allow(dead_code)]
impl RecognitionContext<'_> {
    #[inline]
    fn raw_candle(&self, offset: usize) -> Candle {
        self.history.frame(offset).candle
    }

    /// Returns a widened Candle, where offset zero is the current source position.
    #[inline]
    pub(crate) fn candle(&self, offset: usize) -> WideCandle {
        self.raw_candle(offset).into()
    }

    #[inline]
    pub(crate) fn open(&self, offset: usize) -> PatternFloat {
        self.candle(offset).open
    }

    #[inline]
    pub(crate) fn high(&self, offset: usize) -> PatternFloat {
        self.candle(offset).high
    }

    #[inline]
    pub(crate) fn low(&self, offset: usize) -> PatternFloat {
        self.candle(offset).low
    }

    #[inline]
    pub(crate) fn close(&self, offset: usize) -> PatternFloat {
        self.candle(offset).close
    }

    #[inline]
    pub(crate) fn body_high(&self, offset: usize) -> PatternFloat {
        let candle = self.raw_candle(offset);
        widen(if candle.close >= candle.open {
            candle.close
        } else {
            candle.open
        })
    }

    #[inline]
    pub(crate) fn body_low(&self, offset: usize) -> PatternFloat {
        let candle = self.raw_candle(offset);
        widen(if candle.close >= candle.open {
            candle.open
        } else {
            candle.close
        })
    }

    #[inline]
    pub(crate) fn real_body(&self, offset: usize) -> PatternFloat {
        self.raw_candle(offset).real_body()
    }

    #[inline]
    pub(crate) fn high_low_range(&self, offset: usize) -> PatternFloat {
        self.raw_candle(offset).high_low_range()
    }

    #[inline]
    pub(crate) fn upper_shadow(&self, offset: usize) -> PatternFloat {
        self.raw_candle(offset).upper_shadow()
    }

    #[inline]
    pub(crate) fn lower_shadow(&self, offset: usize) -> PatternFloat {
        self.raw_candle(offset).lower_shadow()
    }

    #[inline]
    pub(crate) fn shadows(&self, offset: usize) -> PatternFloat {
        self.raw_candle(offset).shadows()
    }

    #[inline]
    pub(crate) fn color(&self, offset: usize) -> CandleColor {
        self.raw_candle(offset).color()
    }

    /// Returns the setting's already source-aligned Candle Average.
    #[inline]
    pub(crate) fn average(&self, setting_type: CandleSettingType, offset: usize) -> PatternFloat {
        self.history.frame(offset).averages[setting_type.index()]
    }

    #[inline]
    pub(crate) fn real_body_gap_up(&self, second: usize, first: usize) -> bool {
        let second = self.candle(second);
        let first = self.candle(first);
        let second_low = if second.close >= second.open {
            second.open
        } else {
            second.close
        };
        let first_high = if first.close >= first.open {
            first.close
        } else {
            first.open
        };
        second_low > first_high
    }

    #[inline]
    pub(crate) fn real_body_gap_down(&self, second: usize, first: usize) -> bool {
        let second = self.candle(second);
        let first = self.candle(first);
        let second_high = if second.close >= second.open {
            second.close
        } else {
            second.open
        };
        let first_low = if first.close >= first.open {
            first.open
        } else {
            first.close
        };
        second_high < first_low
    }

    #[inline]
    pub(crate) fn candle_gap_up(&self, second: usize, first: usize) -> bool {
        self.candle(second).low > self.candle(first).high
    }

    #[inline]
    pub(crate) fn candle_gap_down(&self, second: usize, first: usize) -> bool {
        self.candle(second).high < self.candle(first).low
    }
}

/// Static, crate-owned definition seam. Predicates and formulas stay local.
pub(crate) trait PatternDefinition: Copy {
    type State;

    fn name(&self) -> &'static str;
    fn settings(&self) -> CandleSettings;
    fn referenced_settings(&self) -> &'static [CandleSettingType];
    fn lookback(&self) -> usize;
    fn transition_start(&self) -> usize;
    fn initial_state(&self) -> Self::State;
    fn transition(
        &self,
        context: &RecognitionContext<'_>,
        state: &mut Self::State,
    ) -> PatternSignal;
}

/// Shared rolling lifecycle used by all four public execution modes.
#[derive(Debug, Clone)]
pub(crate) struct RecognitionEngine<D: PatternDefinition> {
    definition: D,
    state: D::State,
    history: CandleHistory,
    totals: [PatternFloat; 11],
    position: usize,
}

impl<D> RecognitionEngine<D>
where
    D: PatternDefinition,
{
    pub(crate) fn new(definition: D) -> Self {
        let history_capacity = definition.lookback().saturating_add(1).max(1);
        Self {
            definition,
            state: definition.initial_state(),
            history: CandleHistory::new(history_capacity),
            totals: [0.0; 11],
            position: 0,
        }
    }

    #[inline]
    pub(crate) const fn definition(&self) -> D {
        self.definition
    }

    pub(crate) fn reset(&mut self) {
        self.state = self.definition.initial_state();
        self.history.clear();
        self.totals.fill(0.0);
        self.position = 0;
    }

    pub(crate) fn next(&mut self, candle: Candle) -> Result<Option<PatternSignal>> {
        validate_candle(candle)?;
        Ok(self.next_validated(candle))
    }

    fn next_validated(&mut self, candle: Candle) -> Option<PatternSignal> {
        let mut averages = [0.0; 11];
        let settings = self.definition.settings();
        for &setting_type in self.definition.referenced_settings() {
            let setting = settings.setting(setting_type);
            let range = if setting.average_period() == 0 {
                candle.range(setting.range_kind())
            } else {
                self.totals[setting_type.index()] / setting.average_period() as PatternFloat
            };
            let divisor = if setting.range_kind() == CandleRangeKind::Shadows {
                2.0
            } else {
                1.0
            };
            averages[setting_type.index()] = widen(setting.factor()) * range / divisor;
        }

        self.history.push(CandleFrame { candle, averages });

        let signal = if self.position >= self.definition.transition_start() {
            let context = RecognitionContext {
                history: &self.history,
            };
            self.definition.transition(&context, &mut self.state)
        } else {
            PatternSignal::NoMatch
        };

        for &setting_type in self.definition.referenced_settings() {
            let setting = settings.setting(setting_type);
            let period = setting.average_period();
            if period == 0 {
                continue;
            }
            let current_range = candle.range(setting.range_kind());
            let outgoing_range = if self.position >= period {
                self.history
                    .frame(period)
                    .candle
                    .range(setting.range_kind())
            } else {
                0.0
            };
            self.totals[setting_type.index()] += current_range - outgoing_range;
        }

        let source_position = self.position;
        self.position += 1;
        (source_position >= self.definition.lookback()).then_some(signal)
    }

    pub(crate) fn run_validated(
        &mut self,
        input: CandleInput<'_>,
        output: &mut [PatternSignal],
        shape: BatchShape,
    ) -> OutputRange {
        self.reset();
        let mut output_index = 0;
        for index in 0..shape.source_len {
            if let Some(signal) = self.next_validated(input.candle(index)) {
                output[output_index] = signal;
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, shape.output_count);
        if shape.output_count == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(shape.lookback, shape.output_count)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchShape {
    lookback: usize,
    source_len: usize,
    output_count: usize,
}

pub(crate) fn validate_batch<D: PatternDefinition>(
    definition: D,
    input: CandleInput<'_>,
) -> Result<BatchShape> {
    let source_len = validate_all_same_len(&[
        ("open", input.open.len()),
        ("high", input.high.len()),
        ("low", input.low.len()),
        ("close", input.close.len()),
    ])?;
    validate_finite_slices(&[
        ("open", input.open),
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    let lookback = definition.lookback();
    let output_count = validate_input_len(source_len, lookback)?;
    Ok(BatchShape {
        lookback,
        source_len,
        output_count,
    })
}

pub(crate) fn validate_prepared_capacity(
    input: CandleInput<'_>,
    max_input_len: usize,
) -> Result<()> {
    let actual_input_len = input.max_len();
    if actual_input_len > max_input_len {
        return Err(TalibError::prepared_capacity_exceeded(
            max_input_len,
            actual_input_len,
        ));
    }
    Ok(())
}

pub(crate) fn compute_owned<D: PatternDefinition>(
    definition: D,
    input: CandleInput<'_>,
) -> Result<CompactOutput<Vec<PatternSignal>>> {
    let shape = validate_batch(definition, input)?;
    let mut values = Vec::with_capacity(shape.output_count);
    values.resize(shape.output_count, PatternSignal::NoMatch);
    let mut engine = RecognitionEngine::new(definition);
    let range = engine.run_validated(input, &mut values, shape);
    CompactOutput::new(shape.source_len, range, values)
}

pub(crate) fn compute_into<D: PatternDefinition>(
    definition: D,
    input: CandleInput<'_>,
    output: &mut [PatternSignal],
) -> Result<OutputRange> {
    let shape = validate_batch(definition, input)?;
    validate_output_len(definition.name(), output.len(), shape.output_count)?;
    let mut engine = RecognitionEngine::new(definition);
    Ok(engine.run_validated(input, output, shape))
}

pub(crate) fn prepared_compute_into<D: PatternDefinition>(
    engine: &mut RecognitionEngine<D>,
    max_input_len: usize,
    input: CandleInput<'_>,
    output: &mut [PatternSignal],
) -> Result<OutputRange> {
    validate_prepared_capacity(input, max_input_len)?;
    let shape = validate_batch(engine.definition(), input)?;
    validate_output_len(engine.definition().name(), output.len(), shape.output_count)?;
    Ok(engine.run_validated(input, output, shape))
}

#[inline]
fn validate_candle(candle: Candle) -> Result<()> {
    validate_finite_value("open", 0, candle.open)?;
    validate_finite_value("high", 0, candle.high)?;
    validate_finite_value("low", 0, candle.low)?;
    validate_finite_value("close", 0, candle.close)
}
