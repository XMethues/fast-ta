//! Price-change Momentum Indicator Definitions.
//!
//! All five definitions compare source position `i` with position `i - Period`,
//! so their Lookback and streaming Warm-up are exactly the configured Period.
//! The accepted Period range is `1..=100_000`.
//! `MOM` returns the point difference. `ROC`, `ROCP`, `ROCR`, and `ROCR100`
//! use the same trailing observation as their denominator and follow the pinned
//! TA-Lib definition exactly: an exactly zero denominator produces `0`, while
//! every nonzero denominator, including a near-zero value, is divided normally.
//! Consequently, whenever the denominator is nonzero,
//! `ROC = ROCP * 100`, `ROCR = ROCP + 1`, and `ROCR100 = ROCR * 100`.

use crate::{
    validate_finite_slice, validate_input_len, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

const MAX_PERIOD: usize = 100_000;

#[inline]
fn validate_change_period(period: usize) -> Result<()> {
    if !(1..=MAX_PERIOD).contains(&period) {
        return Err(TalibError::invalid_period(
            period,
            "timeperiod must be in 1..=100000",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Momentum,
    RateOfChange,
    RateOfChangePercentage,
    RateOfChangeRatio,
    RateOfChangeRatio100,
}

impl ChangeKind {
    #[inline(always)]
    fn evaluate(self, current: Float, trailing: Float) -> Float {
        if self == Self::Momentum {
            return current - trailing;
        }
        if trailing == 0.0 as Float {
            return 0.0 as Float;
        }

        match self {
            Self::Momentum => unreachable!(),
            Self::RateOfChange => (current / trailing - 1.0 as Float) * 100.0 as Float,
            Self::RateOfChangePercentage => (current - trailing) / trailing,
            Self::RateOfChangeRatio => current / trailing,
            Self::RateOfChangeRatio100 => current / trailing * 100.0 as Float,
        }
    }
}

#[inline]
fn validate_change_batch(real: &[Float], period: usize) -> Result<usize> {
    validate_change_period(period)?;
    validate_finite_slice("real", real)?;
    validate_input_len(real.len(), period)
}

#[inline]
fn change_kernel(
    real: &[Float],
    period: usize,
    count: usize,
    kind: ChangeKind,
    out_real: &mut [Float],
) -> OutputRange {
    for output_idx in 0..count {
        let input_idx = period + output_idx;
        out_real[output_idx] = kind.evaluate(real[input_idx], real[output_idx]);
    }

    if count == 0 {
        OutputRange::empty()
    } else {
        OutputRange::new(period, count)
    }
}

#[derive(Debug, Clone)]
struct ChangeStreamCore {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl ChangeStreamCore {
    fn new(period: usize) -> Result<Self> {
        validate_change_period(period)?;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Ok(Self {
            period,
            buffer,
            index: 0,
            count: 0,
        })
    }

    #[inline]
    fn next(&mut self, input: Float, kind: ChangeKind) -> Result<Option<Float>> {
        // Validation precedes every state write, preserving failure-before-mutation.
        validate_finite_slice("input", &[input])?;

        if self.count < self.period {
            self.buffer[self.index] = input;
            self.index = (self.index + 1) % self.period;
            self.count += 1;
            return Ok(None);
        }

        let trailing = self.buffer[self.index];
        let output = kind.evaluate(input, trailing);
        self.buffer[self.index] = input;
        self.index = (self.index + 1) % self.period;
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}

macro_rules! define_change_indicator {
    (
        $function:ident,
        $config:ident,
        $runner:ident,
        $stream:ident,
        $kind:expr,
        $definition:literal
    ) => {
        #[doc = concat!("Computes ", $definition, " into caller-owned Compact Output storage.")]
        #[allow(non_snake_case)]
        pub fn $function(
            real: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            let count = validate_change_batch(real, timeperiod)?;
            validate_output_len(stringify!($function), out_real.len(), count)?;
            Ok(change_kernel(real, timeperiod, count, $kind, out_real))
        }

        #[doc = concat!("Immutable configuration for ", $definition, ".")]
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            period: usize,
        }

        impl $config {
            #[doc = concat!("Creates a ", stringify!($function), " configuration.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                validate_change_period(timeperiod)?;
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
                let count = validate_change_batch(input, self.period)?;
                let mut values = Vec::with_capacity(count);
                values.resize(count, 0.0 as Float);
                let range = change_kernel(input, self.period, count, $kind, &mut values);
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
                    core: ChangeStreamCore::new(self.period)?,
                })
            }
        }

        #[doc = concat!("Prepared Batch Runner for ", $definition, ".")]
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

        #[doc = concat!("Independent Streaming Computation for ", $definition, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            core: ChangeStreamCore,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            #[inline]
            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                self.core.next(input, $kind)
            }

            #[inline]
            fn reset(&mut self) {
                self.core.reset();
            }
        }
    };
}

define_change_indicator!(
    MOM,
    MOMConfig,
    MOMBatchRunner,
    MOMStream,
    ChangeKind::Momentum,
    "Momentum (MOM)"
);
define_change_indicator!(
    ROC,
    ROCConfig,
    ROCBatchRunner,
    ROCStream,
    ChangeKind::RateOfChange,
    "Rate of Change (ROC)"
);
define_change_indicator!(
    ROCP,
    ROCPConfig,
    ROCPBatchRunner,
    ROCPStream,
    ChangeKind::RateOfChangePercentage,
    "Rate of Change Percentage (ROCP)"
);
define_change_indicator!(
    ROCR,
    ROCRConfig,
    ROCRBatchRunner,
    ROCRStream,
    ChangeKind::RateOfChangeRatio,
    "Rate of Change Ratio (ROCR)"
);
define_change_indicator!(
    ROCR100,
    ROCR100Config,
    ROCR100BatchRunner,
    ROCR100Stream,
    ChangeKind::RateOfChangeRatio100,
    "100-scaled Rate of Change Ratio (ROCR100)"
);
