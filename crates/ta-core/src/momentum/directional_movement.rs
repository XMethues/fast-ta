//! Wilder Directional Movement indicator system.
//!
//! All seven definitions share the same qualified one-period movement, true
//! range, Wilder recursive smoothing, and source-position state. `PLUS_DM` and
//! `MINUS_DM` accept Period 1 and have Lookback 1 at that Period; at larger
//! Periods their Lookback is `period - 1`. `PLUS_DI` and `MINUS_DI` accept
//! Period 1 and have Lookback 1 at that Period; at larger Periods their
//! Lookback is `period`. `DX` requires Period 2 or greater and has Lookback
//! `period`. `ADX` requires Period 2 or greater and has Lookback
//! `2 * period - 1`. `ADXR` requires Period 2 or greater and has Lookback
//! `3 * period - 2`. There is no additional Stabilization span. DM outputs are
//! non-negative; DI, DX, ADX, and ADXR outputs are in `[0, 100]`.

use crate::{
    validate_all_same_len, validate_finite_slices, validate_input_len, validate_output_len,
    validate_period, CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner,
    Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::{vec, vec::Vec};

/// Borrowed high/low/close inputs shared by the Directional Movement system.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick shared by the Directional Movement system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Definition {
    PlusDm,
    MinusDm,
    PlusDi,
    MinusDi,
    Dx,
    Adx,
    Adxr,
}

impl Definition {
    const fn name(self) -> &'static str {
        match self {
            Self::PlusDm => "PLUS_DM",
            Self::MinusDm => "MINUS_DM",
            Self::PlusDi => "PLUS_DI",
            Self::MinusDi => "MINUS_DI",
            Self::Dx => "DX",
            Self::Adx => "ADX",
            Self::Adxr => "ADXR",
        }
    }

    const fn minimum_period(self) -> usize {
        match self {
            Self::PlusDm | Self::MinusDm | Self::PlusDi | Self::MinusDi => 1,
            Self::Dx | Self::Adx | Self::Adxr => 2,
        }
    }

    fn lookback(self, period: usize) -> Result<usize> {
        validate_period("timeperiod", period)?;
        if period > 100_000 {
            return Err(TalibError::invalid_period(
                period,
                "timeperiod must not exceed 100000",
            ));
        }
        if period < self.minimum_period() {
            return Err(TalibError::invalid_period(
                period,
                "period must be at least 2 for DX, ADX, and ADXR",
            ));
        }
        match self {
            Self::PlusDm | Self::MinusDm => {
                if period == 1 {
                    Ok(1)
                } else {
                    Ok(period - 1)
                }
            }
            Self::PlusDi | Self::MinusDi | Self::Dx => Ok(period),
            Self::Adx => period
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or_else(|| TalibError::invalid_period(period, "ADX lookback would overflow")),
            Self::Adxr => period
                .checked_mul(3)
                .and_then(|value| value.checked_sub(2))
                .ok_or_else(|| TalibError::invalid_period(period, "ADXR lookback would overflow")),
        }
    }
}

fn validate_input(
    input: DirectionalInput<'_>,
    definition: Definition,
    period: usize,
) -> Result<(usize, usize, usize)> {
    let lookback = definition.lookback(period)?;
    let len = validate_all_same_len(&[
        ("high", input.high.len()),
        ("low", input.low.len()),
        ("close", input.close.len()),
    ])?;
    validate_finite_slices(&[
        ("high", input.high),
        ("low", input.low),
        ("close", input.close),
    ])?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count, len))
}

#[inline]
fn validate_tick(input: DirectionalTick) -> Result<()> {
    validate_finite_slices(&[
        ("high", &[input.high]),
        ("low", &[input.low]),
        ("close", &[input.close]),
    ])
}

#[inline]
fn qualified_movement(
    high: Float,
    low: Float,
    previous_high: Float,
    previous_low: Float,
) -> (Float, Float) {
    let up = high - previous_high;
    let down = previous_low - low;
    if up > 0.0 as Float && up > down {
        (up, 0.0 as Float)
    } else if down > 0.0 as Float && down > up {
        (0.0 as Float, down)
    } else {
        (0.0 as Float, 0.0 as Float)
    }
}

#[inline]
fn directional_indices(
    plus_dm: Float,
    minus_dm: Float,
    true_range: Float,
) -> (Float, Float, Float, bool) {
    if true_range == 0.0 as Float {
        return (0.0 as Float, 0.0 as Float, 0.0 as Float, false);
    }
    let plus_di = 100.0 as Float * plus_dm / true_range;
    let minus_di = 100.0 as Float * minus_dm / true_range;
    let di_sum = plus_di + minus_di;
    if di_sum == 0.0 as Float {
        (plus_di, minus_di, 0.0 as Float, false)
    } else {
        (
            plus_di,
            minus_di,
            100.0 as Float * (plus_di - minus_di).abs() / di_sum,
            true,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct DirectionalPoint {
    plus_dm: Float,
    minus_dm: Float,
    plus_di: Float,
    minus_di: Float,
    dx: Float,
    dx_defined: bool,
    dm_ready: bool,
    di_ready: bool,
}

#[derive(Debug, Clone)]
struct DirectionalState {
    period: usize,
    previous: Option<DirectionalTick>,
    movement_count: usize,
    plus_dm: Float,
    minus_dm: Float,
    true_range: Float,
}

impl DirectionalState {
    fn new(period: usize) -> Self {
        Self {
            period,
            previous: None,
            movement_count: 0,
            plus_dm: 0.0 as Float,
            minus_dm: 0.0 as Float,
            true_range: 0.0 as Float,
        }
    }

    fn reset(&mut self) {
        self.previous = None;
        self.movement_count = 0;
        self.plus_dm = 0.0 as Float;
        self.minus_dm = 0.0 as Float;
        self.true_range = 0.0 as Float;
    }

    fn update(&mut self, tick: DirectionalTick) -> Option<DirectionalPoint> {
        let Some(previous) = self.previous else {
            self.previous = Some(tick);
            return None;
        };

        let (raw_plus, raw_minus) =
            qualified_movement(tick.high, tick.low, previous.high, previous.low);
        let raw_range =
            super::super::volatility::directional_true_range(tick.high, tick.low, previous.close);
        self.previous = Some(tick);
        self.movement_count += 1;

        if self.period == 1 {
            self.plus_dm = raw_plus;
            self.minus_dm = raw_minus;
            self.true_range = raw_range;
        } else if self.movement_count < self.period {
            self.plus_dm += raw_plus;
            self.minus_dm += raw_minus;
            self.true_range += raw_range;
        } else {
            let period = self.period as Float;
            self.plus_dm = self.plus_dm - self.plus_dm / period + raw_plus;
            self.minus_dm = self.minus_dm - self.minus_dm / period + raw_minus;
            self.true_range = self.true_range - self.true_range / period + raw_range;
        }

        let dm_ready = self.period == 1 || self.movement_count >= self.period - 1;
        let di_ready = self.period == 1 || self.movement_count >= self.period;
        let (plus_di, minus_di, dx, dx_defined) = if di_ready {
            directional_indices(self.plus_dm, self.minus_dm, self.true_range)
        } else {
            (0.0 as Float, 0.0 as Float, 0.0 as Float, false)
        };

        Some(DirectionalPoint {
            plus_dm: self.plus_dm,
            minus_dm: self.minus_dm,
            plus_di,
            minus_di,
            dx,
            dx_defined,
            dm_ready,
            di_ready,
        })
    }
}

#[derive(Debug, Clone)]
struct AdxState {
    period: usize,
    directional: DirectionalState,
    dx_count: usize,
    dx_sum: Float,
    adx: Float,
}

impl AdxState {
    fn new(period: usize) -> Self {
        Self {
            period,
            directional: DirectionalState::new(period),
            dx_count: 0,
            dx_sum: 0.0 as Float,
            adx: 0.0 as Float,
        }
    }

    fn reset(&mut self) {
        self.directional.reset();
        self.dx_count = 0;
        self.dx_sum = 0.0 as Float;
        self.adx = 0.0 as Float;
    }

    fn update(&mut self, tick: DirectionalTick) -> Option<Float> {
        let point = self.directional.update(tick)?;
        if !point.di_ready {
            return None;
        }
        if self.dx_count < self.period {
            self.dx_sum += point.dx;
            self.dx_count += 1;
            if self.dx_count < self.period {
                return None;
            }
            self.adx = self.dx_sum / self.period as Float;
            return Some(self.adx);
        }
        if point.dx_defined {
            self.adx = (self.adx * (self.period - 1) as Float + point.dx) / self.period as Float;
        }
        Some(self.adx)
    }
}

fn tick_at(input: DirectionalInput<'_>, index: usize) -> DirectionalTick {
    DirectionalTick {
        high: input.high[index],
        low: input.low[index],
        close: input.close[index],
    }
}

// Batch ADX has three fixed phases. Keeping those phases explicit avoids the
// per-tick readiness/Option state machine while reusing the qualified movement,
// true-range, and DI/DX formulas used by streaming and neighboring definitions.
struct AdxBatchState {
    period: Float,
    previous: DirectionalTick,
    plus_dm: Float,
    minus_dm: Float,
    true_range: Float,
}

impl AdxBatchState {
    #[inline]
    fn new(period: usize, first: DirectionalTick) -> Self {
        Self {
            period: period as Float,
            previous: first,
            plus_dm: 0.0 as Float,
            minus_dm: 0.0 as Float,
            true_range: 0.0 as Float,
        }
    }

    #[inline]
    fn accumulate(&mut self, tick: DirectionalTick) {
        let (raw_plus, raw_minus) =
            qualified_movement(tick.high, tick.low, self.previous.high, self.previous.low);
        self.plus_dm += raw_plus;
        self.minus_dm += raw_minus;
        self.true_range += super::super::volatility::directional_true_range(
            tick.high,
            tick.low,
            self.previous.close,
        );
        self.previous = tick;
    }

    #[inline]
    fn smooth_dx(&mut self, tick: DirectionalTick) -> (Float, bool) {
        let (raw_plus, raw_minus) =
            qualified_movement(tick.high, tick.low, self.previous.high, self.previous.low);
        let raw_range = super::super::volatility::directional_true_range(
            tick.high,
            tick.low,
            self.previous.close,
        );
        self.plus_dm = self.plus_dm - self.plus_dm / self.period + raw_plus;
        self.minus_dm = self.minus_dm - self.minus_dm / self.period + raw_minus;
        self.true_range = self.true_range - self.true_range / self.period + raw_range;
        self.previous = tick;
        let (_, _, dx, dx_defined) =
            directional_indices(self.plus_dm, self.minus_dm, self.true_range);
        (dx, dx_defined)
    }
}

fn adx_batch_kernel(
    input: DirectionalInput<'_>,
    period: usize,
    count: usize,
    output: &mut [Float],
) {
    let mut ticks = input
        .high
        .iter()
        .copied()
        .zip(input.low.iter().copied())
        .zip(input.close.iter().copied())
        .map(|((high, low), close)| DirectionalTick { high, low, close });
    let mut state = AdxBatchState::new(
        period,
        ticks
            .next()
            .expect("non-empty ADX input after lookback validation"),
    );

    for _ in 1..period {
        state.accumulate(
            ticks
                .next()
                .expect("ADX input contains the initial Directional Movement span"),
        );
    }

    let mut dx_sum = 0.0 as Float;
    for _ in 0..period {
        let (dx, _) = state.smooth_dx(
            ticks
                .next()
                .expect("ADX input contains the initial Directional Index span"),
        );
        dx_sum += dx;
    }

    let mut adx = dx_sum / period as Float;
    output[0] = adx;
    let period_minus_one = (period - 1) as Float;
    for slot in &mut output[1..count] {
        let (dx, dx_defined) = state.smooth_dx(
            ticks
                .next()
                .expect("validated ADX output count matches the input"),
        );
        if dx_defined {
            adx = (adx * period_minus_one + dx) / period as Float;
        }
        *slot = adx;
    }
}

fn kernel(
    input: DirectionalInput<'_>,
    definition: Definition,
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    match definition {
        Definition::Adx => adx_batch_kernel(input, period, count, output),
        Definition::Adxr => {
            let lag = period - 1;
            let mut current = AdxState::new(period);
            let mut lagged = AdxState::new(period);
            let mut output_index = 0;
            for index in 0..input.high.len() {
                let current_value = current.update(tick_at(input, index));
                let lagged_value = if index >= lag {
                    lagged.update(tick_at(input, index - lag))
                } else {
                    None
                };
                if let (Some(current_adx), Some(lagged_adx)) = (current_value, lagged_value) {
                    output[output_index] = (current_adx + lagged_adx) / 2.0 as Float;
                    output_index += 1;
                }
            }
        }
        _ => {
            let mut state = DirectionalState::new(period);
            let mut output_index = 0;
            for index in 0..input.high.len() {
                let Some(point) = state.update(tick_at(input, index)) else {
                    continue;
                };
                let value = match definition {
                    Definition::PlusDm if point.dm_ready => Some(point.plus_dm),
                    Definition::MinusDm if point.dm_ready => Some(point.minus_dm),
                    Definition::PlusDi if point.di_ready => Some(point.plus_di),
                    Definition::MinusDi if point.di_ready => Some(point.minus_di),
                    Definition::Dx if point.di_ready => Some(point.dx),
                    _ => None,
                };
                if let Some(value) = value {
                    output[output_index] = value;
                    output_index += 1;
                }
            }
        }
    }
    OutputRange::new(lookback, count)
}

fn compute_function(
    input: DirectionalInput<'_>,
    definition: Definition,
    period: usize,
    output: &mut [Float],
) -> Result<OutputRange> {
    let (lookback, count, _) = validate_input(input, definition, period)?;
    validate_output_len(definition.name(), output.len(), count)?;
    Ok(kernel(input, definition, period, lookback, count, output))
}

macro_rules! directional_function {
    ($function:ident, $kind:expr, $doc:literal) => {
        #[doc = $doc]
        #[allow(non_snake_case)]
        pub fn $function(
            high: &[Float],
            low: &[Float],
            close: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            compute_function(
                DirectionalInput { high, low, close },
                $kind,
                timeperiod,
                out_real,
            )
        }
    };
}

directional_function!(
    PLUS_DM,
    Definition::PlusDm,
    "Computes Plus Directional Movement."
);
directional_function!(
    MINUS_DM,
    Definition::MinusDm,
    "Computes Minus Directional Movement."
);
directional_function!(
    PLUS_DI,
    Definition::PlusDi,
    "Computes Plus Directional Indicator."
);
directional_function!(
    MINUS_DI,
    Definition::MinusDi,
    "Computes Minus Directional Indicator."
);
directional_function!(DX, Definition::Dx, "Computes Directional Movement Index.");
directional_function!(
    ADX,
    Definition::Adx,
    "Computes Average Directional Movement Index."
);
directional_function!(
    ADXR,
    Definition::Adxr,
    "Computes Average Directional Movement Index Rating."
);

#[derive(Debug, Clone)]
enum StreamCore {
    Directional(DirectionalState),
    Adx(AdxState),
    Adxr {
        adx: AdxState,
        history: Vec<Float>,
        position: usize,
        seen: usize,
    },
}

impl StreamCore {
    fn new(definition: Definition, period: usize) -> Self {
        match definition {
            Definition::Adx => Self::Adx(AdxState::new(period)),
            Definition::Adxr => Self::Adxr {
                adx: AdxState::new(period),
                history: vec![0.0 as Float; period - 1],
                position: 0,
                seen: 0,
            },
            _ => Self::Directional(DirectionalState::new(period)),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Directional(state) => state.reset(),
            Self::Adx(state) => state.reset(),
            Self::Adxr {
                adx,
                history,
                position,
                seen,
            } => {
                adx.reset();
                history.fill(0.0 as Float);
                *position = 0;
                *seen = 0;
            }
        }
    }

    fn next(&mut self, definition: Definition, tick: DirectionalTick) -> Option<Float> {
        match self {
            Self::Directional(state) => {
                let point = state.update(tick)?;
                match definition {
                    Definition::PlusDm if point.dm_ready => Some(point.plus_dm),
                    Definition::MinusDm if point.dm_ready => Some(point.minus_dm),
                    Definition::PlusDi if point.di_ready => Some(point.plus_di),
                    Definition::MinusDi if point.di_ready => Some(point.minus_di),
                    Definition::Dx if point.di_ready => Some(point.dx),
                    _ => None,
                }
            }
            Self::Adx(state) => state.update(tick),
            Self::Adxr {
                adx,
                history,
                position,
                seen,
            } => {
                let value = adx.update(tick)?;
                if *seen < history.len() {
                    history[*position] = value;
                    *position = (*position + 1) % history.len();
                    *seen += 1;
                    return None;
                }
                let lagged = history[*position];
                history[*position] = value;
                *position = (*position + 1) % history.len();
                Some((lagged + value) / 2.0 as Float)
            }
        }
    }
}

macro_rules! directional_indicator {
    (
        $config:ident, $runner:ident, $stream:ident, $kind:expr,
        $config_doc:literal, $runner_doc:literal, $stream_doc:literal
    ) => {
        #[doc = $config_doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[allow(non_camel_case_types)]
        pub struct $config {
            period: usize,
        }

        impl $config {
            /// Creates an immutable Indicator Configuration.
            pub fn new(timeperiod: usize) -> Result<Self> {
                $kind.lookback(timeperiod)?;
                Ok(Self { period: timeperiod })
            }

            /// Returns the configured Period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Returns the Warm-up tick count, identical to Lookback.
            pub fn warm_up(&self) -> usize {
                $kind
                    .lookback(self.period)
                    .expect("validated Directional Movement period")
            }

            /// Returns the fixed additional Stabilization span (zero).
            pub const fn stabilization(&self) -> usize {
                0
            }
        }

        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = DirectionalInput<'a>;
            type Output = Vec<Float>;
            type OutputMut<'a> = &'a mut [Float];
            type BatchRunner = $runner;
            type Stream = $stream;

            fn lookback(&self) -> usize {
                $kind
                    .lookback(self.period)
                    .expect("validated Directional Movement period")
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let (lookback, count, len) = validate_input(input, $kind, self.period)?;
                let mut values = Vec::with_capacity(count);
                values.resize(count, 0.0 as Float);
                let range = kernel(input, $kind, self.period, lookback, count, &mut values);
                CompactOutput::new(len, range, values)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                compute_function(input, $kind, self.period, output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    core: StreamCore::new($kind, self.period),
                })
            }
        }

        #[doc = $runner_doc]
        #[derive(Debug, Clone)]
        #[allow(non_camel_case_types)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            fn compute_into<'a>(
                &mut self,
                input: <$config as IndicatorConfig>::Input<'a>,
                output: <$config as IndicatorConfig>::OutputMut<'a>,
            ) -> Result<OutputRange>
            where
                $config: 'a,
            {
                let actual_input_len = input.high.len().max(input.low.len()).max(input.close.len());
                if actual_input_len > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        actual_input_len,
                    ));
                }
                IndicatorConfig::compute_into(&self.config, input, output)
            }
        }

        #[doc = $stream_doc]
        #[derive(Debug, Clone)]
        #[allow(non_camel_case_types)]
        pub struct $stream {
            core: StreamCore,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = DirectionalTick;
            type TickOutput = Float;

            fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
                validate_tick(input)?;
                Ok(self.core.next($kind, input))
            }

            fn reset(&mut self) {
                self.core.reset();
            }
        }
    };
}

directional_indicator!(
    PLUS_DMConfig,
    PLUS_DMBatchRunner,
    PLUS_DMStream,
    Definition::PlusDm,
    "Immutable Plus Directional Movement Indicator Configuration.",
    "Reusable Prepared Batch Runner for PLUS_DM.",
    "Independent Streaming Computation for PLUS_DM."
);
directional_indicator!(
    MINUS_DMConfig,
    MINUS_DMBatchRunner,
    MINUS_DMStream,
    Definition::MinusDm,
    "Immutable Minus Directional Movement Indicator Configuration.",
    "Reusable Prepared Batch Runner for MINUS_DM.",
    "Independent Streaming Computation for MINUS_DM."
);
directional_indicator!(
    PLUS_DIConfig,
    PLUS_DIBatchRunner,
    PLUS_DIStream,
    Definition::PlusDi,
    "Immutable Plus Directional Indicator Configuration.",
    "Reusable Prepared Batch Runner for PLUS_DI.",
    "Independent Streaming Computation for PLUS_DI."
);
directional_indicator!(
    MINUS_DIConfig,
    MINUS_DIBatchRunner,
    MINUS_DIStream,
    Definition::MinusDi,
    "Immutable Minus Directional Indicator Configuration.",
    "Reusable Prepared Batch Runner for MINUS_DI.",
    "Independent Streaming Computation for MINUS_DI."
);
directional_indicator!(
    DXConfig,
    DXBatchRunner,
    DXStream,
    Definition::Dx,
    "Immutable Directional Movement Index Indicator Configuration.",
    "Reusable Prepared Batch Runner for DX.",
    "Independent Streaming Computation for DX."
);
directional_indicator!(
    ADXConfig,
    ADXBatchRunner,
    ADXStream,
    Definition::Adx,
    "Immutable Average Directional Movement Index Indicator Configuration.",
    "Reusable Prepared Batch Runner for ADX.",
    "Independent Streaming Computation for ADX."
);
directional_indicator!(
    ADXRConfig,
    ADXRBatchRunner,
    ADXRStream,
    Definition::Adxr,
    "Immutable Average Directional Movement Index Rating Configuration.",
    "Reusable Prepared Batch Runner for ADXR.",
    "Independent Streaming Computation for ADXR."
);
