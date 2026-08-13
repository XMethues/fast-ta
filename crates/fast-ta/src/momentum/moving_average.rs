//! Moving-average Momentum Indicator Definitions.
//!
//! `APO` and `PPO` compare aligned fast and slow Period-based Moving Averages.
//! `MACD`, `MACDEXT`, and `MACDFIX` return aligned, named MACD, signal, and
//! histogram columns. `TRIX` is the one-observation percentage rate of change
//! of a triple-smoothed EMA. All Periods are explicit and every accepted moving
//! average kind is a [`PeriodMAType`]; MAMA is therefore not selectable.

use crate::common::{validate_finite_value, CompactPayloadLen};
use crate::overlap::{ema_multiplier, ema_seed, ema_step, MAConfig, MAStream, PeriodMAType};
use crate::{
    validate_finite_slice, validate_input_len, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::{vec, vec::Vec};

const MAX_PERIOD: usize = 100_000;
const DEFAULT_FAST_PERIOD: usize = 12;
const DEFAULT_SLOW_PERIOD: usize = 26;
const DEFAULT_SIGNAL_PERIOD: usize = 9;
const DEFAULT_TRIX_PERIOD: usize = 30;
const TA_EPSILON: Float = 1e-14 as Float;

#[inline]
fn validate_bounded_period(name: &str, period: usize, minimum: usize) -> Result<()> {
    if !(minimum..=MAX_PERIOD).contains(&period) {
        return Err(TalibError::invalid_period(
            period,
            if minimum == 1 {
                "Period must be in 1..=100000"
            } else {
                "Period must be in 2..=100000"
            },
        ));
    }
    let _ = name;
    Ok(())
}

#[inline]
fn validate_period_order(fast_period: usize, slow_period: usize) -> Result<()> {
    if fast_period >= slow_period {
        return Err(TalibError::invalid_input(
            "fast Period must be strictly less than slow Period",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PairSpec {
    fast: MAConfig,
    slow: MAConfig,
    lookback: usize,
}

impl PairSpec {
    fn new(
        fast_period: usize,
        fast_type: PeriodMAType,
        slow_period: usize,
        slow_type: PeriodMAType,
    ) -> Result<Self> {
        validate_bounded_period("fast_period", fast_period, 2)?;
        validate_bounded_period("slow_period", slow_period, 2)?;
        validate_period_order(fast_period, slow_period)?;
        let fast = MAConfig::new(fast_period, fast_type)?;
        let slow = MAConfig::new(slow_period, slow_type)?;
        let lookback = fast.lookback().max(slow.lookback());
        Ok(Self {
            fast,
            slow,
            lookback,
        })
    }
}

#[derive(Debug, Clone)]
struct PairStreamCore {
    fast: MAStream,
    slow: MAStream,
    fast_start: usize,
    slow_start: usize,
    observations: usize,
}

impl PairStreamCore {
    fn new(spec: PairSpec, align_initial_windows: bool) -> Result<Self> {
        Ok(Self {
            fast: spec.fast.stream()?,
            slow: spec.slow.stream()?,
            fast_start: if align_initial_windows {
                spec.lookback - spec.fast.lookback()
            } else {
                0
            },
            slow_start: if align_initial_windows {
                spec.lookback - spec.slow.lookback()
            } else {
                0
            },
            observations: 0,
        })
    }

    fn next(&mut self, input: Float) -> Result<Option<(Float, Float)>> {
        validate_finite_value("input", self.observations, input)?;
        let index = self.observations;
        let fast = if index >= self.fast_start {
            self.fast.next(input)?
        } else {
            None
        };
        let slow = if index >= self.slow_start {
            self.slow.next(input)?
        } else {
            None
        };
        self.observations += 1;
        Ok(match (fast, slow) {
            (Some(fast), Some(slow)) => Some((fast, slow)),
            _ => None,
        })
    }

    fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.observations = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PriceOscillatorKind {
    Absolute,
    Percentage,
}

impl PriceOscillatorKind {
    #[inline]
    fn evaluate(self, fast: Float, slow: Float) -> Float {
        let difference = fast - slow;
        match self {
            Self::Absolute => difference,
            Self::Percentage if slow > -TA_EPSILON && slow < TA_EPSILON => 0.0 as Float,
            Self::Percentage => difference / slow * 100.0 as Float,
        }
    }
}

fn validate_single_output(
    real: &[Float],
    lookback: usize,
    output_name: &str,
    output_len: usize,
) -> Result<usize> {
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(output_name, output_len, count)?;
    Ok(count)
}

fn compute_pair_validated(
    real: &[Float],
    spec: PairSpec,
    kind: PriceOscillatorKind,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let mut core = PairStreamCore::new(spec, false)?;
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some((fast, slow)) = core.next(input)? {
            out_real[output_index] = kind.evaluate(fast, slow);
            output_index += 1;
        }
    }
    debug_assert_eq!(output_index, real.len().saturating_sub(spec.lookback));
    Ok(if output_index == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(spec.lookback, output_index)
    })
}

fn compute_pair_into(
    real: &[Float],
    spec: PairSpec,
    kind: PriceOscillatorKind,
    output_name: &str,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    validate_single_output(real, spec.lookback, output_name, out_real.len())?;
    compute_pair_validated(real, spec, kind, out_real)
}

macro_rules! define_price_oscillator {
    (
        $function:ident,
        $config:ident,
        $runner:ident,
        $stream:ident,
        $kind:expr,
        $title:literal
    ) => {
        #[doc = concat!("Immutable ", $title, " Indicator Configuration.")]
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            spec: PairSpec,
            ma_type: PeriodMAType,
        }

        impl $config {
            #[doc = concat!("Creates an explicit ", $title, " configuration.")]
            pub fn new(
                fast_period: usize,
                slow_period: usize,
                ma_type: PeriodMAType,
            ) -> Result<Self> {
                Ok(Self {
                    spec: PairSpec::new(fast_period, ma_type, slow_period, ma_type)?,
                    ma_type,
                })
            }

            /// Returns the fast Period.
            #[inline]
            pub const fn fast_period(&self) -> usize {
                self.spec.fast.period()
            }

            /// Returns the slow Period.
            #[inline]
            pub const fn slow_period(&self) -> usize {
                self.spec.slow.period()
            }

            /// Returns the selected Period-based Moving Average kind.
            #[inline]
            pub const fn ma_type(&self) -> PeriodMAType {
                self.ma_type
            }
        }

        impl Default for $config {
            fn default() -> Self {
                Self::new(DEFAULT_FAST_PERIOD, DEFAULT_SLOW_PERIOD, PeriodMAType::EMA)
                    .expect("default price-oscillator configuration must be valid")
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
                self.spec.lookback
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                validate_finite_slice("real", input)?;
                let count = validate_input_len(input.len(), self.spec.lookback)?;
                let mut values = vec![0.0 as Float; count];
                let range = compute_pair_validated(input, self.spec, $kind, &mut values)?;
                CompactOutput::new(input.len(), range, values)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                compute_pair_into(input, self.spec, $kind, stringify!($function), output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                    core: PairStreamCore::new(self.spec, false)?,
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    core: PairStreamCore::new(self.spec, false)?,
                })
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for ", $title, ".")]
        #[derive(Debug, Clone)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
            core: PairStreamCore,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
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
                if input.len() > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input.len(),
                    ));
                }
                let count = validate_single_output(
                    input,
                    self.config.spec.lookback,
                    stringify!($function),
                    output.len(),
                )?;
                self.core.reset();
                let mut output_index = 0;
                for value in input.iter().copied() {
                    if let Some((fast, slow)) = self.core.next(value)? {
                        output[output_index] = $kind.evaluate(fast, slow);
                        output_index += 1;
                    }
                }
                debug_assert_eq!(output_index, count);
                Ok(if count == 0 {
                    OutputRange::empty()
                } else {
                    OutputRange::new(self.config.spec.lookback, count)
                })
            }
        }

        #[doc = concat!("Independent Streaming Computation for ", $title, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            core: PairStreamCore,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                Ok(self
                    .core
                    .next(input)?
                    .map(|(fast, slow)| $kind.evaluate(fast, slow)))
            }

            #[inline]
            fn reset(&mut self) {
                self.core.reset();
            }
        }

        #[doc = concat!("Computes ", $title, " into caller-owned Compact Output storage.")]
        #[allow(non_snake_case)]
        pub fn $function(
            real: &[Float],
            fast_period: usize,
            slow_period: usize,
            ma_type: PeriodMAType,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            let config = $config::new(fast_period, slow_period, ma_type)?;
            config.compute_into(real, out_real)
        }
    };
}

define_price_oscillator!(
    APO,
    APOConfig,
    APOBatchRunner,
    APOStream,
    PriceOscillatorKind::Absolute,
    "Absolute Price Oscillator"
);
define_price_oscillator!(
    PPO,
    PPOConfig,
    PPOBatchRunner,
    PPOStream,
    PriceOscillatorKind::Percentage,
    "Percentage Price Oscillator"
);

/// One valid aligned MACD-family streaming output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MACDValue {
    /// Fast-minus-slow moving-average line.
    pub macd: Float,
    /// Moving average of the MACD line.
    pub signal: Float,
    /// Difference between `macd` and `signal`.
    pub histogram: Float,
}

/// Named aligned Compact Output columns for the MACD family.
#[derive(Debug, Clone, PartialEq)]
pub struct MACDValues {
    /// Fast-minus-slow moving-average column.
    pub macd: Vec<Float>,
    /// Signal moving-average column.
    pub signal: Vec<Float>,
    /// MACD-minus-signal histogram column.
    pub histogram: Vec<Float>,
}

impl CompactPayloadLen for MACDValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.macd.len() != self.signal.len() || self.macd.len() != self.histogram.len() {
            return Err(TalibError::invalid_input(
                "MACD Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.macd.len())
    }
}

/// Caller-owned named aligned Compact Output columns for the MACD family.
#[derive(Debug)]
pub struct MACDValuesMut<'a> {
    /// Fast-minus-slow moving-average output buffer.
    pub macd: &'a mut [Float],
    /// Signal moving-average output buffer.
    pub signal: &'a mut [Float],
    /// MACD-minus-signal histogram output buffer.
    pub histogram: &'a mut [Float],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MACDSpec {
    pair: PairSpec,
    signal: MAConfig,
    lookback: usize,
}

impl MACDSpec {
    fn new(
        fast_period: usize,
        fast_type: PeriodMAType,
        slow_period: usize,
        slow_type: PeriodMAType,
        signal_period: usize,
        signal_type: PeriodMAType,
    ) -> Result<Self> {
        let pair = PairSpec::new(fast_period, fast_type, slow_period, slow_type)?;
        validate_bounded_period("signal_period", signal_period, 1)?;
        let signal = MAConfig::new(signal_period, signal_type)?;
        let lookback = pair
            .lookback
            .checked_add(signal.lookback())
            .ok_or_else(|| {
                TalibError::invalid_period(signal_period, "MACD lookback would overflow")
            })?;
        Ok(Self {
            pair,
            signal,
            lookback,
        })
    }

    #[inline]
    fn uses_only_ema(self) -> bool {
        self.pair.fast.ma_type() == PeriodMAType::EMA
            && self.pair.slow.ma_type() == PeriodMAType::EMA
            && self.signal.ma_type() == PeriodMAType::EMA
    }
}

#[derive(Debug, Clone)]
struct MACDStreamCore {
    pair: PairStreamCore,
    signal: MAStream,
}

impl MACDStreamCore {
    fn new(spec: MACDSpec) -> Result<Self> {
        Ok(Self {
            pair: PairStreamCore::new(spec.pair, true)?,
            signal: spec.signal.stream()?,
        })
    }

    fn next(&mut self, input: Float) -> Result<Option<MACDValue>> {
        let Some((fast, slow)) = self.pair.next(input)? else {
            return Ok(None);
        };
        let macd = fast - slow;
        Ok(self.signal.next(macd)?.map(|signal| MACDValue {
            macd,
            signal,
            histogram: macd - signal,
        }))
    }

    fn reset(&mut self) {
        self.pair.reset();
        self.signal.reset();
    }
}

fn validate_macd_outputs(
    real: &[Float],
    lookback: usize,
    output_name: &str,
    output: &MACDValuesMut<'_>,
) -> Result<usize> {
    let names = match output_name {
        "MACDEXT" => ("MACDEXT MACD", "MACDEXT signal", "MACDEXT histogram"),
        "MACDFIX" => ("MACDFIX MACD", "MACDFIX signal", "MACDFIX histogram"),
        _ => ("MACD line", "MACD signal", "MACD histogram"),
    };
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(names.0, output.macd.len(), count)?;
    validate_output_len(names.1, output.signal.len(), count)?;
    validate_output_len(names.2, output.histogram.len(), count)?;
    Ok(count)
}

// Batch validation is complete before this fused path is entered. Keeping the
// three EMA states as scalars avoids repeating streaming validation, generic MA
// dispatch, and warm-up `Option` transitions for every observation.
fn compute_macd_ema_validated(
    real: &[Float],
    spec: MACDSpec,
    output: MACDValuesMut<'_>,
) -> OutputRange {
    let output_count = real.len().saturating_sub(spec.lookback);
    if output_count == 0 {
        return OutputRange::empty();
    }

    let fast_period = spec.pair.fast.period();
    let slow_period = spec.pair.slow.period();
    let signal_period = spec.signal.period();
    let pair_lookback = spec.pair.lookback;
    let fast_start = pair_lookback + 1 - fast_period;
    let fast_multiplier = ema_multiplier(fast_period);
    let slow_multiplier = ema_multiplier(slow_period);
    let signal_multiplier = ema_multiplier(signal_period);
    let mut fast = ema_seed(&real[fast_start..=pair_lookback], fast_period);
    let mut slow = ema_seed(real, slow_period);
    let mut signal_sum = 0.0 as Float;
    let mut macd = fast - slow;
    signal_sum += macd;

    for &input in &real[pair_lookback + 1..spec.lookback + 1] {
        fast = ema_step(fast, input, fast_multiplier);
        slow = ema_step(slow, input, slow_multiplier);
        macd = fast - slow;
        signal_sum += macd;
    }

    let mut signal = signal_sum / signal_period as Float;
    output.macd[0] = macd;
    output.signal[0] = signal;
    output.histogram[0] = macd - signal;

    for (((macd_output, signal_output), histogram_output), &input) in output.macd[1..output_count]
        .iter_mut()
        .zip(&mut output.signal[1..output_count])
        .zip(&mut output.histogram[1..output_count])
        .zip(&real[spec.lookback + 1..])
    {
        fast = ema_step(fast, input, fast_multiplier);
        slow = ema_step(slow, input, slow_multiplier);
        macd = fast - slow;
        signal = ema_step(signal, macd, signal_multiplier);
        *macd_output = macd;
        *signal_output = signal;
        *histogram_output = macd - signal;
    }

    OutputRange::new(spec.lookback, output_count)
}

fn compute_macd_stream_validated(
    real: &[Float],
    spec: MACDSpec,
    core: &mut MACDStreamCore,
    output: MACDValuesMut<'_>,
) -> Result<OutputRange> {
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(value) = core.next(input)? {
            output.macd[output_index] = value.macd;
            output.signal[output_index] = value.signal;
            output.histogram[output_index] = value.histogram;
            output_index += 1;
        }
    }
    debug_assert_eq!(output_index, real.len().saturating_sub(spec.lookback));
    Ok(if output_index == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(spec.lookback, output_index)
    })
}

fn compute_macd_validated(
    real: &[Float],
    spec: MACDSpec,
    output: MACDValuesMut<'_>,
) -> Result<OutputRange> {
    if spec.uses_only_ema() {
        return Ok(compute_macd_ema_validated(real, spec, output));
    }
    let mut core = MACDStreamCore::new(spec)?;
    compute_macd_stream_validated(real, spec, &mut core, output)
}

fn compute_macd_into(
    real: &[Float],
    spec: MACDSpec,
    output_name: &str,
    output: MACDValuesMut<'_>,
) -> Result<OutputRange> {
    validate_macd_outputs(real, spec.lookback, output_name, &output)?;
    compute_macd_validated(real, spec, output)
}

/// Immutable standard EMA MACD Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MACDConfig {
    spec: MACDSpec,
}

impl MACDConfig {
    /// Creates a standard EMA MACD configuration.
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Self> {
        Ok(Self {
            spec: MACDSpec::new(
                fast_period,
                PeriodMAType::EMA,
                slow_period,
                PeriodMAType::EMA,
                signal_period,
                PeriodMAType::EMA,
            )?,
        })
    }

    /// Returns the fast EMA Period.
    #[inline]
    pub const fn fast_period(&self) -> usize {
        self.spec.pair.fast.period()
    }

    /// Returns the slow EMA Period.
    #[inline]
    pub const fn slow_period(&self) -> usize {
        self.spec.pair.slow.period()
    }

    /// Returns the signal EMA Period.
    #[inline]
    pub const fn signal_period(&self) -> usize {
        self.spec.signal.period()
    }
}

impl Default for MACDConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_FAST_PERIOD,
            DEFAULT_SLOW_PERIOD,
            DEFAULT_SIGNAL_PERIOD,
        )
        .expect("default MACD configuration must be valid")
    }
}

/// Immutable extended MACD Indicator Configuration with explicit Period-based Moving Average kinds.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MACDEXTConfig {
    spec: MACDSpec,
}

impl MACDEXTConfig {
    /// Creates an extended MACD configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fast_period: usize,
        fast_type: PeriodMAType,
        slow_period: usize,
        slow_type: PeriodMAType,
        signal_period: usize,
        signal_type: PeriodMAType,
    ) -> Result<Self> {
        Ok(Self {
            spec: MACDSpec::new(
                fast_period,
                fast_type,
                slow_period,
                slow_type,
                signal_period,
                signal_type,
            )?,
        })
    }

    /// Returns the fast Period.
    #[inline]
    pub const fn fast_period(&self) -> usize {
        self.spec.pair.fast.period()
    }

    /// Returns the fast Period-based Moving Average kind.
    #[inline]
    pub const fn fast_type(&self) -> PeriodMAType {
        self.spec.pair.fast.ma_type()
    }

    /// Returns the slow Period.
    #[inline]
    pub const fn slow_period(&self) -> usize {
        self.spec.pair.slow.period()
    }

    /// Returns the slow Period-based Moving Average kind.
    #[inline]
    pub const fn slow_type(&self) -> PeriodMAType {
        self.spec.pair.slow.ma_type()
    }

    /// Returns the signal Period.
    #[inline]
    pub const fn signal_period(&self) -> usize {
        self.spec.signal.period()
    }

    /// Returns the signal Period-based Moving Average kind.
    #[inline]
    pub const fn signal_type(&self) -> PeriodMAType {
        self.spec.signal.ma_type()
    }
}

impl Default for MACDEXTConfig {
    fn default() -> Self {
        Self::new(
            DEFAULT_FAST_PERIOD,
            PeriodMAType::EMA,
            DEFAULT_SLOW_PERIOD,
            PeriodMAType::EMA,
            DEFAULT_SIGNAL_PERIOD,
            PeriodMAType::EMA,
        )
        .expect("default MACDEXT configuration must be valid")
    }
}

/// Immutable fixed-period (12, 26) EMA MACD Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MACDFIXConfig {
    spec: MACDSpec,
}

impl MACDFIXConfig {
    /// Creates a fixed 12/26 EMA MACD configuration with an explicit signal Period.
    pub fn new(signal_period: usize) -> Result<Self> {
        Ok(Self {
            spec: MACDSpec::new(
                DEFAULT_FAST_PERIOD,
                PeriodMAType::EMA,
                DEFAULT_SLOW_PERIOD,
                PeriodMAType::EMA,
                signal_period,
                PeriodMAType::EMA,
            )?,
        })
    }

    /// Returns the fixed fast EMA Period (12).
    #[inline]
    pub const fn fast_period(&self) -> usize {
        DEFAULT_FAST_PERIOD
    }

    /// Returns the fixed slow EMA Period (26).
    #[inline]
    pub const fn slow_period(&self) -> usize {
        DEFAULT_SLOW_PERIOD
    }

    /// Returns the signal EMA Period.
    #[inline]
    pub const fn signal_period(&self) -> usize {
        self.spec.signal.period()
    }
}

impl Default for MACDFIXConfig {
    fn default() -> Self {
        Self::new(DEFAULT_SIGNAL_PERIOD).expect("default MACDFIX configuration must be valid")
    }
}

macro_rules! define_macd_execution {
    ($config:ident, $runner:ident, $stream:ident, $title:literal) => {
        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = &'a [Float];
            type Output = MACDValues;
            type OutputMut<'a> = MACDValuesMut<'a>;
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                self.spec.lookback
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                validate_finite_slice("real", input)?;
                let count = validate_input_len(input.len(), self.spec.lookback)?;
                let mut values = MACDValues {
                    macd: vec![0.0 as Float; count],
                    signal: vec![0.0 as Float; count],
                    histogram: vec![0.0 as Float; count],
                };
                let range = compute_macd_validated(
                    input,
                    self.spec,
                    MACDValuesMut {
                        macd: &mut values.macd,
                        signal: &mut values.signal,
                        histogram: &mut values.histogram,
                    },
                )?;
                CompactOutput::new(input.len(), range, values)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                compute_macd_into(input, self.spec, $title, output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                    core: MACDStreamCore::new(self.spec)?,
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    core: MACDStreamCore::new(self.spec)?,
                })
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for ", $title, ".")]
        #[derive(Debug, Clone)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
            core: MACDStreamCore,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
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
                if input.len() > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input.len(),
                    ));
                }
                validate_macd_outputs(input, self.config.spec.lookback, $title, &output)?;
                if self.config.spec.uses_only_ema() {
                    return Ok(compute_macd_ema_validated(input, self.config.spec, output));
                }
                self.core.reset();
                compute_macd_stream_validated(input, self.config.spec, &mut self.core, output)
            }
        }

        #[doc = concat!("Independent Streaming Computation for ", $title, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            core: MACDStreamCore,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = MACDValue;

            #[inline]
            fn next(&mut self, input: Float) -> Result<Option<MACDValue>> {
                self.core.next(input)
            }

            #[inline]
            fn reset(&mut self) {
                self.core.reset();
            }
        }
    };
}

define_macd_execution!(MACDConfig, MACDBatchRunner, MACDStream, "MACD");
define_macd_execution!(MACDEXTConfig, MACDEXTBatchRunner, MACDEXTStream, "MACDEXT");
define_macd_execution!(MACDFIXConfig, MACDFIXBatchRunner, MACDFIXStream, "MACDFIX");

/// Computes standard EMA MACD into three caller-owned aligned Compact Output columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn MACD(
    real: &[Float],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
    out_macd: &mut [Float],
    out_signal: &mut [Float],
    out_histogram: &mut [Float],
) -> Result<OutputRange> {
    MACDConfig::new(fast_period, slow_period, signal_period)?.compute_into(
        real,
        MACDValuesMut {
            macd: out_macd,
            signal: out_signal,
            histogram: out_histogram,
        },
    )
}

/// Computes extended MACD into three caller-owned aligned Compact Output columns.
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn MACDEXT(
    real: &[Float],
    fast_period: usize,
    fast_type: PeriodMAType,
    slow_period: usize,
    slow_type: PeriodMAType,
    signal_period: usize,
    signal_type: PeriodMAType,
    out_macd: &mut [Float],
    out_signal: &mut [Float],
    out_histogram: &mut [Float],
) -> Result<OutputRange> {
    MACDEXTConfig::new(
        fast_period,
        fast_type,
        slow_period,
        slow_type,
        signal_period,
        signal_type,
    )?
    .compute_into(
        real,
        MACDValuesMut {
            macd: out_macd,
            signal: out_signal,
            histogram: out_histogram,
        },
    )
}

/// Computes fixed 12/26 EMA MACD into three caller-owned aligned Compact Output columns.
#[allow(non_snake_case)]
pub fn MACDFIX(
    real: &[Float],
    signal_period: usize,
    out_macd: &mut [Float],
    out_signal: &mut [Float],
    out_histogram: &mut [Float],
) -> Result<OutputRange> {
    MACDFIXConfig::new(signal_period)?.compute_into(
        real,
        MACDValuesMut {
            macd: out_macd,
            signal: out_signal,
            histogram: out_histogram,
        },
    )
}

#[inline]
fn trix_lookback(period: usize) -> Result<usize> {
    validate_bounded_period("timeperiod", period, 1)?;
    (period - 1)
        .checked_mul(3)
        .and_then(|lookback| lookback.checked_add(1))
        .ok_or_else(|| TalibError::invalid_period(period, "TRIX lookback would overflow"))
}

#[derive(Debug, Clone)]
struct TRIXStreamCore {
    ema1: MAStream,
    ema2: MAStream,
    ema3: MAStream,
    previous: Option<Float>,
    observations: usize,
}

impl TRIXStreamCore {
    fn new(period: usize) -> Result<Self> {
        let ema = MAConfig::new(period, PeriodMAType::EMA)?;
        Ok(Self {
            ema1: ema.stream()?,
            ema2: ema.stream()?,
            ema3: ema.stream()?,
            previous: None,
            observations: 0,
        })
    }

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_value("input", self.observations, input)?;
        self.observations += 1;
        let Some(first) = self.ema1.next(input)? else {
            return Ok(None);
        };
        let Some(second) = self.ema2.next(first)? else {
            return Ok(None);
        };
        let Some(third) = self.ema3.next(second)? else {
            return Ok(None);
        };
        let previous = self.previous.replace(third);
        Ok(previous.map(|previous| {
            if previous == 0.0 as Float {
                0.0 as Float
            } else {
                (third / previous - 1.0 as Float) * 100.0 as Float
            }
        }))
    }

    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
        self.previous = None;
        self.observations = 0;
    }
}

/// Computes the one-observation percentage change of a triple EMA into caller-owned storage.
#[allow(non_snake_case)]
pub fn TRIX(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = trix_lookback(timeperiod)?;
    let count = validate_single_output(real, lookback, "TRIX", out_real.len())?;
    let mut core = TRIXStreamCore::new(timeperiod)?;
    let mut output_index = 0;
    for input in real.iter().copied() {
        if let Some(value) = core.next(input)? {
            out_real[output_index] = value;
            output_index += 1;
        }
    }
    debug_assert_eq!(output_index, count);
    Ok(if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(lookback, count)
    })
}

/// Immutable Triple Exponential Average change Indicator Configuration.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TRIXConfig {
    period: usize,
    lookback: usize,
}

impl TRIXConfig {
    /// Creates a TRIX configuration for the supplied Period.
    pub fn new(timeperiod: usize) -> Result<Self> {
        Ok(Self {
            period: timeperiod,
            lookback: trix_lookback(timeperiod)?,
        })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl Default for TRIXConfig {
    fn default() -> Self {
        Self::new(DEFAULT_TRIX_PERIOD).expect("default TRIX configuration must be valid")
    }
}

impl crate::traits::sealed::Sealed for TRIXConfig {}

impl IndicatorConfig for TRIXConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = TRIXBatchRunner;
    type Stream = TRIXStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        validate_finite_slice("real", input)?;
        let count = validate_input_len(input.len(), self.lookback)?;
        let mut values = vec![0.0 as Float; count];
        let range = TRIX(input, self.period, &mut values)?;
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRIX(input, self.period, output)
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(TRIXBatchRunner {
            config: *self,
            max_input_len,
            core: TRIXStreamCore::new(self.period)?,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        Ok(TRIXStream {
            core: TRIXStreamCore::new(self.period)?,
        })
    }
}

/// Reusable Prepared Batch Runner for TRIX.
#[derive(Debug, Clone)]
pub struct TRIXBatchRunner {
    config: TRIXConfig,
    max_input_len: usize,
    core: TRIXStreamCore,
}

impl crate::traits::sealed::Sealed for TRIXBatchRunner {}

impl PreparedBatchRunner<TRIXConfig> for TRIXBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <TRIXConfig as IndicatorConfig>::Input<'a>,
        output: <TRIXConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        TRIXConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let count = validate_single_output(input, self.config.lookback, "TRIX", output.len())?;
        self.core.reset();
        let mut output_index = 0;
        for input in input.iter().copied() {
            if let Some(value) = self.core.next(input)? {
                output[output_index] = value;
                output_index += 1;
            }
        }
        debug_assert_eq!(output_index, count);
        Ok(if count == 0 {
            OutputRange::empty()
        } else {
            OutputRange::new(self.config.lookback, count)
        })
    }
}

/// Independent Streaming Computation for TRIX.
#[derive(Debug, Clone)]
pub struct TRIXStream {
    core: TRIXStreamCore,
}

impl crate::traits::sealed::Sealed for TRIXStream {}

impl StreamingComputation<TRIXConfig> for TRIXStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        self.core.next(input)
    }

    #[inline]
    fn reset(&mut self) {
        self.core.reset();
    }
}
