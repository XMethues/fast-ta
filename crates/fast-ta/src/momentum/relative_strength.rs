//! Relative-strength momentum indicators: RSI, CMO, and IMI.
//!
//! RSI and CMO share the same Wilder-smoothed gain/loss transition and differ
//! only in their final projection. IMI uses the same gain/loss classification
//! over intraday close-minus-open movement, but retains its rolling Period and
//! source alignment.

use crate::{
    validate_finite_slice, validate_finite_slices, validate_input_len, validate_output_len,
    validate_same_len, CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner,
    Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec, vec::Vec};

const MIN_PERIOD: usize = 2;
const MAX_PERIOD: usize = 100_000;
const TA_EPSILON: Float = 1e-14 as Float;

#[inline]
fn relative_strength_period(timeperiod: usize) -> Result<()> {
    if !(MIN_PERIOD..=MAX_PERIOD).contains(&timeperiod) {
        return Err(TalibError::invalid_period(
            timeperiod,
            format!("timeperiod must be in {MIN_PERIOD}..={MAX_PERIOD}"),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct GainLoss {
    gain: Float,
    loss: Float,
}

impl GainLoss {
    #[inline]
    fn from_movement(movement: Float) -> Self {
        if movement < 0.0 as Float {
            Self {
                gain: 0.0 as Float,
                loss: -movement,
            }
        } else {
            Self {
                gain: movement,
                loss: 0.0 as Float,
            }
        }
    }

    #[inline]
    fn add(&mut self, other: Self) {
        self.gain += other.gain;
        self.loss += other.loss;
    }

    #[inline]
    fn replace(&mut self, outgoing: Self, incoming: Self) {
        // Mathematically both totals remain non-negative. Clamp subtraction
        // round-off so long rolling streams preserve the oscillator bounds.
        self.gain = (self.gain - outgoing.gain + incoming.gain).max(0.0 as Float);
        self.loss = (self.loss - outgoing.loss + incoming.loss).max(0.0 as Float);
    }
}

#[derive(Debug, Clone, Copy)]
enum Projection {
    Rsi,
    Cmo,
    Imi,
}

#[inline]
fn project(totals: GainLoss, projection: Projection) -> Float {
    let denominator = totals.gain + totals.loss;
    let denominator_is_zero = match projection {
        Projection::Imi => denominator == 0.0 as Float,
        Projection::Rsi | Projection::Cmo => denominator > -TA_EPSILON && denominator < TA_EPSILON,
    };
    if denominator_is_zero {
        return match projection {
            Projection::Imi => 50.0 as Float,
            Projection::Rsi | Projection::Cmo => 0.0 as Float,
        };
    }

    match projection {
        Projection::Rsi | Projection::Imi => 100.0 as Float * (totals.gain / denominator),
        Projection::Cmo => 100.0 as Float * ((totals.gain - totals.loss) / denominator),
    }
}

/// Wilder's shared recursive gain/loss smoothing model used by RSI and CMO.
#[derive(Debug, Clone, Copy)]
struct WilderGainLoss {
    period: usize,
    movement_count: usize,
    totals: GainLoss,
}

impl WilderGainLoss {
    #[inline]
    const fn new(period: usize) -> Self {
        Self {
            period,
            movement_count: 0,
            totals: GainLoss {
                gain: 0.0 as Float,
                loss: 0.0 as Float,
            },
        }
    }

    #[inline]
    fn next(&mut self, movement: Float) -> Option<GainLoss> {
        let current = GainLoss::from_movement(movement);
        if self.movement_count < self.period {
            self.totals.add(current);
            self.movement_count += 1;
            if self.movement_count < self.period {
                return None;
            }
            let period = self.period as Float;
            self.totals.gain /= period;
            self.totals.loss /= period;
            return Some(self.totals);
        }

        let previous_weight = (self.period - 1) as Float;
        let period = self.period as Float;
        self.totals.gain = (self.totals.gain * previous_weight + current.gain) / period;
        self.totals.loss = (self.totals.loss * previous_weight + current.loss) / period;
        Some(self.totals)
    }

    #[inline]
    fn reset(&mut self) {
        self.movement_count = 0;
        self.totals = GainLoss::default();
    }
}

fn validate_wilder_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    relative_strength_period(timeperiod)?;
    validate_finite_slice("real", real)?;
    let lookback = timeperiod;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

fn wilder_kernel(
    real: &[Float],
    timeperiod: usize,
    projection: Projection,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut state = WilderGainLoss::new(timeperiod);
    let mut output_idx = 0;
    for pair in real.windows(2) {
        if let Some(totals) = state.next(pair[1] - pair[0]) {
            out_real[output_idx] = project(totals, projection);
            output_idx += 1;
        }
    }
    debug_assert_eq!(output_idx, count);
    OutputRange::new(timeperiod, count)
}

fn wilder_compute_into(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    projection: Projection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let (_, count) = validate_wilder_input(real, timeperiod)?;
    validate_output_len(name, out_real.len(), count)?;
    Ok(wilder_kernel(real, timeperiod, projection, count, out_real))
}

/// Relative Strength Index using Wilder's recursive gain/loss smoothing.
#[allow(non_snake_case)]
pub fn RSI(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    wilder_compute_into("RSI", real, timeperiod, Projection::Rsi, out_real)
}

/// Chande Momentum Oscillator using Wilder's recursive gain/loss smoothing.
#[allow(non_snake_case)]
pub fn CMO(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    wilder_compute_into("CMO", real, timeperiod, Projection::Cmo, out_real)
}

macro_rules! define_wilder_indicator {
    (
        $config:ident,
        $runner:ident,
        $stream:ident,
        $function:ident,
        $projection:expr,
        $display:literal
    ) => {
        #[doc = concat!("Immutable ", $display, " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            period: usize,
        }

        impl $config {
            #[doc = concat!("Creates a ", $display, " configuration.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                relative_strength_period(timeperiod)?;
                Ok(Self { period: timeperiod })
            }

            /// Returns the configured Period.
            #[inline]
            pub const fn period(&self) -> usize {
                self.period
            }
        }

        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = &'a [Float];
            type Output = Vec<Float>;
            type OutputMut<'a> = &'a mut [Float];
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                self.period
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let (_, count) = validate_wilder_input(input, self.period)?;
                let mut values = Vec::with_capacity(count);
                values.resize(count, 0.0 as Float);
                let range = wilder_kernel(input, self.period, $projection, count, &mut values);
                CompactOutput::new(input.len(), range, values)
            }

            #[inline]
            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $function(input, self.period, output)
            }

            #[inline]
            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                })
            }

            #[inline]
            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    previous: None,
                    smoother: WilderGainLoss::new(self.period),
                })
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for ", $display, ".")]
        #[derive(Debug, Clone)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            #[inline]
            fn compute_into<'a>(
                &mut self,
                input: <$config as IndicatorConfig>::Input<'a>,
                output: <$config as IndicatorConfig>::OutputMut<'a>,
            ) -> Result<OutputRange>
            where
                $config: 'a,
            {
                if input.len() > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input.len(),
                    ));
                }
                IndicatorConfig::compute_into(&self.config, input, output)
            }
        }

        #[doc = concat!("Independent Streaming Computation state for ", $display, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            previous: Option<Float>,
            smoother: WilderGainLoss,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                let Some(previous) = self.previous else {
                    self.previous = Some(input);
                    return Ok(None);
                };
                let movement = input - previous;
                self.previous = Some(input);
                Ok(self
                    .smoother
                    .next(movement)
                    .map(|totals| project(totals, $projection)))
            }

            #[inline]
            fn reset(&mut self) {
                self.previous = None;
                self.smoother.reset();
            }
        }
    };
}

define_wilder_indicator!(
    RSIConfig,
    RSIBatchRunner,
    RSIStream,
    RSI,
    Projection::Rsi,
    "RSI"
);

impl RSIBatchRunner {
    /// Runs the kernel after a composed indicator has checked capacity.
    #[inline]
    pub(crate) fn compute_into_bounded<'a>(
        &mut self,
        input: <RSIConfig as IndicatorConfig>::Input<'a>,
        output: <RSIConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        RSIConfig: 'a,
    {
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}
define_wilder_indicator!(
    CMOConfig,
    CMOBatchRunner,
    CMOStream,
    CMO,
    Projection::Cmo,
    "CMO"
);

/// Borrowed aligned open/close observations for [`IMI`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct IMIInput<'a> {
    /// Open observation series.
    pub open: &'a [Float],
    /// Close observation series.
    pub close: &'a [Float],
}

/// One aligned open/close observation for [`IMI`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IMITick {
    /// Open observation.
    pub open: Float,
    /// Close observation.
    pub close: Float,
}

fn validate_imi_input(input: IMIInput<'_>, timeperiod: usize) -> Result<(usize, usize, usize)> {
    relative_strength_period(timeperiod)?;
    validate_same_len("open", input.open.len(), "close", input.close.len())?;
    validate_finite_slices(&[("open", input.open), ("close", input.close)])?;
    let lookback = timeperiod - 1;
    let count = validate_input_len(input.open.len(), lookback)?;
    Ok((lookback, count, input.open.len()))
}

#[inline]
fn intraday_movement(open: Float, close: Float) -> Float {
    close - open
}

fn imi_kernel(
    input: IMIInput<'_>,
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut totals = GainLoss::default();
    for idx in 0..timeperiod {
        totals.add(GainLoss::from_movement(intraday_movement(
            input.open[idx],
            input.close[idx],
        )));
    }
    out_real[0] = project(totals, Projection::Imi);

    for (output_idx, output_value) in out_real.iter_mut().enumerate().take(count).skip(1) {
        let incoming_idx = lookback + output_idx;
        let outgoing_idx = incoming_idx - timeperiod;
        let outgoing = GainLoss::from_movement(intraday_movement(
            input.open[outgoing_idx],
            input.close[outgoing_idx],
        ));
        let incoming = GainLoss::from_movement(intraday_movement(
            input.open[incoming_idx],
            input.close[incoming_idx],
        ));
        totals.replace(outgoing, incoming);
        *output_value = project(totals, Projection::Imi);
    }

    OutputRange::new(lookback, count)
}

/// Intraday Momentum Index over aligned open and close observations.
#[allow(non_snake_case)]
pub fn IMI(
    open: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = IMIInput { open, close };
    let (lookback, count, _) = validate_imi_input(input, timeperiod)?;
    validate_output_len("IMI", out_real.len(), count)?;
    Ok(imi_kernel(input, timeperiod, lookback, count, out_real))
}

/// Immutable Intraday Momentum Index Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IMIConfig {
    period: usize,
}

impl IMIConfig {
    /// Creates an IMI configuration.
    pub fn new(timeperiod: usize) -> Result<Self> {
        relative_strength_period(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for IMIConfig {}

impl IndicatorConfig for IMIConfig {
    type Input<'a> = IMIInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = IMIBatchRunner;
    type Stream = IMIStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count, len) = validate_imi_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = imi_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        IMI(input.open, input.close, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(IMIBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(IMIStream {
            period: self.period,
            movements: vec![0.0 as Float; self.period],
            position: 0,
            count: 0,
            totals: GainLoss::default(),
        })
    }
}

/// Reusable Prepared Batch Runner for IMI.
#[derive(Debug, Clone)]
pub struct IMIBatchRunner {
    config: IMIConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for IMIBatchRunner {}

impl PreparedBatchRunner<IMIConfig> for IMIBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <IMIConfig as IndicatorConfig>::Input<'a>,
        output: <IMIConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        IMIConfig: 'a,
    {
        let actual_len = input.open.len().max(input.close.len());
        if actual_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_len,
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for IMI.
#[derive(Debug, Clone)]
pub struct IMIStream {
    period: usize,
    movements: Vec<Float>,
    position: usize,
    count: usize,
    totals: GainLoss,
}

impl crate::traits::sealed::Sealed for IMIStream {}

impl StreamingComputation<IMIConfig> for IMIStream {
    type Tick = IMITick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Float>> {
        validate_finite_slices(&[("open", &[input.open]), ("close", &[input.close])])?;
        let movement = intraday_movement(input.open, input.close);
        let incoming = GainLoss::from_movement(movement);

        if self.count < self.period {
            self.movements[self.position] = movement;
            self.totals.add(incoming);
            self.position = (self.position + 1) % self.period;
            self.count += 1;
            if self.count < self.period {
                return Ok(None);
            }
            return Ok(Some(project(self.totals, Projection::Imi)));
        }

        let outgoing = GainLoss::from_movement(self.movements[self.position]);
        self.movements[self.position] = movement;
        self.position = (self.position + 1) % self.period;
        self.totals.replace(outgoing, incoming);
        Ok(Some(project(self.totals, Projection::Imi)))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.count = 0;
        self.totals = GainLoss::default();
    }
}
