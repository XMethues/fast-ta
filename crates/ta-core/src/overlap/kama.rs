//! Kaufman Adaptive Moving Average (KAMA).
//!
//! KAMA adapts its smoothing constant to the efficiency of price movement over
//! its configured Period. A Period of one is the identity definition, matching
//! the Period-based moving-average selector contract.

use crate::{
    validate_finite_slice, validate_input_len, validate_output_len, validate_period, CompactOutput,
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

const FAST_SMOOTHING: Float = 2.0 as Float / 3.0 as Float;
const SLOW_SMOOTHING: Float = 2.0 as Float / 31.0 as Float;
const SMOOTHING_DIFFERENCE: Float = FAST_SMOOTHING - SLOW_SMOOTHING;
const ZERO_TOLERANCE: Float = 1.0e-14 as Float;

#[inline]
pub(super) fn kama_lookback(timeperiod: usize) -> Result<usize> {
    validate_period("timeperiod", timeperiod)?;
    if timeperiod == usize::MAX {
        return Err(TalibError::invalid_period(
            timeperiod,
            "KAMA source requirement would overflow",
        ));
    }
    Ok(if timeperiod == 1 { 0 } else { timeperiod })
}

#[inline]
fn smoothing_constant(change: Float, volatility: Float) -> Float {
    let efficiency_ratio = if volatility.abs() < ZERO_TOLERANCE || change.abs() >= volatility {
        1.0 as Float
    } else {
        (change / volatility).abs()
    };
    let smoothing = efficiency_ratio.mul_add(SMOOTHING_DIFFERENCE, SLOW_SMOOTHING);
    smoothing * smoothing
}

fn validate_kama_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = kama_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[inline]
pub(super) fn kama_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }
    if timeperiod == 1 {
        out_real[..count].copy_from_slice(&real[..count]);
        return OutputRange::new(0, count);
    }

    let mut volatility = 0.0 as Float;
    for idx in 1..=timeperiod {
        volatility += (real[idx] - real[idx - 1]).abs();
    }

    let mut previous = real[timeperiod - 1];
    let first_change = real[timeperiod] - real[0];
    let smoothing = smoothing_constant(first_change, volatility);
    previous = (real[timeperiod] - previous).mul_add(smoothing, previous);
    out_real[0] = previous;

    for output_idx in 1..count {
        let source_idx = lookback + output_idx;
        volatility -= (real[source_idx - timeperiod] - real[source_idx - timeperiod - 1]).abs();
        volatility += (real[source_idx] - real[source_idx - 1]).abs();
        let change = real[source_idx] - real[source_idx - timeperiod];
        let smoothing = smoothing_constant(change, volatility);
        previous = (real[source_idx] - previous).mul_add(smoothing, previous);
        out_real[output_idx] = previous;
    }

    OutputRange::new(lookback, count)
}

/// Kaufman Adaptive Moving Average batch computation.
#[allow(non_snake_case)]
pub fn KAMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_kama_input(real, timeperiod)?;
    validate_output_len("KAMA", out_real.len(), count)?;
    Ok(kama_kernel(real, timeperiod, lookback, count, out_real))
}

/// Immutable Kaufman Adaptive Moving Average Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KAMAConfig {
    period: usize,
}

impl KAMAConfig {
    /// Creates a KAMA configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        kama_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for KAMAConfig {}

impl IndicatorConfig for KAMAConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = KAMABatchRunner;
    type Stream = KAMAStream;

    #[inline]
    fn lookback(&self) -> usize {
        if self.period == 1 {
            0
        } else {
            self.period
        }
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_kama_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = kama_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        KAMA(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(KAMABatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        KAMAStream::new(self.period)
    }
}

/// Prepared Batch Runner for KAMA.
///
/// KAMA is allocation-free in caller-owned batch execution, so preparation
/// stores only its immutable configuration and declared capacity.
#[derive(Debug, Clone)]
pub struct KAMABatchRunner {
    config: KAMAConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for KAMABatchRunner {}

impl PreparedBatchRunner<KAMAConfig> for KAMABatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <KAMAConfig as IndicatorConfig>::Input<'a>,
        output: <KAMAConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        KAMAConfig: 'a,
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

/// Independent Streaming Computation state for KAMA.
#[derive(Debug, Clone)]
pub struct KAMAStream {
    period: usize,
    observations: Vec<Float>,
    count: usize,
    volatility: Float,
    previous_input: Float,
    value: Float,
}

impl KAMAStream {
    pub(super) fn new(period: usize) -> Result<Self> {
        kama_lookback(period)?;
        let capacity = if period == 1 { 0 } else { period + 1 };
        let mut observations = Vec::new();
        observations.resize(capacity, 0.0 as Float);
        Ok(Self {
            period,
            observations,
            count: 0,
            volatility: 0.0 as Float,
            previous_input: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    #[inline]
    pub(super) fn next_unchecked(&mut self, input: Float) -> Option<Float> {
        if self.period == 1 {
            return Some(input);
        }

        let capacity = self.period + 1;
        let slot = self.count % capacity;
        let trailing = if self.count < capacity {
            self.observations[0]
        } else {
            let next_oldest = self.observations[(slot + 1) % capacity];
            self.volatility -= (next_oldest - self.observations[slot]).abs();
            next_oldest
        };
        if self.count > 0 {
            self.volatility += (input - self.previous_input).abs();
        }
        self.observations[slot] = input;
        self.count += 1;

        if self.count <= self.period {
            self.previous_input = input;
            return None;
        }

        if self.count == capacity {
            self.value = self.previous_input;
        }
        let smoothing = smoothing_constant(input - trailing, self.volatility);
        self.value = (input - self.value).mul_add(smoothing, self.value);
        self.previous_input = input;
        Some(self.value)
    }

    pub(super) fn reset_state(&mut self) {
        self.observations.fill(0.0 as Float);
        self.count = 0;
        self.volatility = 0.0 as Float;
        self.previous_input = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}

impl crate::traits::sealed::Sealed for KAMAStream {}

impl StreamingComputation<KAMAConfig> for KAMAStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;
        Ok(self.next_unchecked(input))
    }

    #[inline]
    fn reset(&mut self) {
        self.reset_state();
    }
}
