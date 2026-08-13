//! Rolling-window Math Operators.

use crate::common::validate_finite_value;
use crate::{
    period_lookback, validate_finite_slice, validate_input_len, validate_output_len, CompactOutput,
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::{vec, vec::Vec};

fn validate_rolling_window(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    output_len: usize,
) -> Result<(usize, usize)> {
    let (lookback, count) = validate_rolling_input(real, timeperiod)?;
    validate_output_len(name, output_len, count)?;
    Ok((lookback, count))
}

fn validate_rolling_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

fn rolling_extreme<F>(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
    mut is_better: F,
) -> Result<OutputRange>
where
    F: FnMut(Float, Float) -> bool,
{
    let (lookback, count) = validate_rolling_window(name, real, timeperiod, out_real.len())?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut deque = Vec::with_capacity(real.len());
    let mut head = 0usize;

    for idx in 0..real.len() {
        while head < deque.len() && deque[head] + timeperiod <= idx {
            head += 1;
        }

        while deque.len() > head {
            let &last_idx = deque.last().expect("deque has an element");
            if is_better(real[idx], real[last_idx]) {
                deque.pop();
            } else {
                break;
            }
        }

        deque.push(idx);

        if idx + 1 >= timeperiod {
            let output_idx = idx + 1 - timeperiod;
            out_real[output_idx] = real[deque[head]];
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// TA-Lib-style rolling sum.
#[allow(non_snake_case)]
pub fn SUM(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_rolling_window("SUM", real, timeperiod, out_real.len())?;
    Ok(sum_kernel(real, timeperiod, lookback, count, out_real))
}

#[inline(always)]
fn sum_kernel(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut window_sum: Float = real[..period].iter().copied().sum();
    output[0] = window_sum;
    for (output_idx, output_value) in output.iter_mut().enumerate().take(count).skip(1) {
        let new_idx = output_idx + period - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        *output_value = window_sum;
    }
    OutputRange::new(lookback, count)
}

/// Immutable Rolling Sum Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SUMConfig {
    period: usize,
}

impl SUMConfig {
    /// Creates a configuration for `timeperiod` observations.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for SUMConfig {}

impl IndicatorConfig for SUMConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = SUMBatchRunner;
    type Stream = SUMStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_rolling_input(input, self.period)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = sum_kernel(input, self.period, lookback, count, &mut values);
        CompactOutput::new(input.len(), range, values)
    }

    #[inline(always)]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        SUM(input, self.period, output)
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(SUMBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        SUMStream::new(self.period)
    }
}

/// Prepared Batch Runner for Rolling Sum.
#[derive(Debug, Clone)]
pub struct SUMBatchRunner {
    config: SUMConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for SUMBatchRunner {}

impl PreparedBatchRunner<SUMConfig> for SUMBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline(always)]
    fn compute_into<'a>(
        &mut self,
        input: <SUMConfig as IndicatorConfig>::Input<'a>,
        output: <SUMConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        SUMConfig: 'a,
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

/// Independent Streaming Computation state for Rolling Sum.
#[derive(Debug, Clone)]
pub struct SUMStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl SUMStream {
    fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Ok(Self {
            period,
            buffer,
            index: 0,
            count: 0,
        })
    }
}

impl crate::traits::sealed::Sealed for SUMStream {}

impl StreamingComputation<SUMConfig> for SUMStream {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_value("input", 0, input)?;

        if self.count < self.period {
            self.buffer[self.index] = input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;
            if self.count < self.period {
                return Ok(None);
            }
        } else {
            self.buffer[self.index] = input;
            self.index = (self.index + 1) % self.period;
        }

        Ok(Some(self.buffer.iter().copied().sum()))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}

/// TA-Lib-style rolling minimum.
#[allow(non_snake_case)]
pub fn MIN(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_extreme("MIN", real, timeperiod, out_real, |candidate, current| {
        candidate < current
    })
}

/// TA-Lib-style rolling maximum.
#[allow(non_snake_case)]
pub fn MAX(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_extreme("MAX", real, timeperiod, out_real, |candidate, current| {
        candidate > current
    })
}

#[inline(always)]
unsafe fn push_reserved_extreme_index(queue: &mut Vec<usize>, index: usize) {
    let len = queue.len();
    debug_assert!(len < queue.capacity());
    // SAFETY: callers reserve at least the source length before entering the
    // kernel, and the queue receives at most one push per source observation.
    unsafe {
        queue.as_mut_ptr().add(len).write(index);
        queue.set_len(len + 1);
    }
}

#[inline(always)]
fn rolling_single_extreme_append<const MINIMUM: bool, const RESERVED_PUSH: bool>(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
    queue: &mut Vec<usize>,
) -> OutputRange {
    queue.clear();
    if count == 0 {
        return OutputRange::empty();
    }

    let mut head = 0usize;
    for idx in 0..real.len() {
        if idx >= period {
            let expired_through = idx - period;
            while head < queue.len() && queue[head] <= expired_through {
                head += 1;
            }
        }

        while queue.len() > head {
            let last_idx = queue[queue.len() - 1];
            let is_better = if MINIMUM {
                real[idx] < real[last_idx]
            } else {
                real[idx] > real[last_idx]
            };
            if is_better {
                queue.pop();
            } else {
                break;
            }
        }
        if RESERVED_PUSH {
            // SAFETY: prepared queues have source-capacity headroom.
            unsafe { push_reserved_extreme_index(queue, idx) };
        } else {
            queue.push(idx);
        }

        if idx + 1 >= period {
            output[idx + 1 - period] = real[queue[head]];
        }
    }

    OutputRange::new(lookback, count)
}

macro_rules! define_extreme_execution {
    (
        $config:ident,
        $runner:ident,
        $stream:ident,
        $minimum:literal,
        $label:literal,
        $description:literal
    ) => {
        #[doc = concat!("Immutable ", $description, " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            period: usize,
        }

        impl $config {
            /// Creates a configuration for `timeperiod` observations.
            pub fn new(timeperiod: usize) -> Result<Self> {
                period_lookback("timeperiod", timeperiod)?;
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
                self.period - 1
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let (lookback, count) = validate_rolling_input(input, self.period)?;
                let mut values = vec![0.0 as Float; count];
                let mut queue = Vec::with_capacity(input.len());
                let range = rolling_single_extreme_append::<$minimum, false>(
                    input,
                    self.period,
                    lookback,
                    count,
                    &mut values,
                    &mut queue,
                );
                CompactOutput::new(input.len(), range, values)
            }

            #[doc = concat!(
                                "Computes into caller-owned Compact Output using one input-length ",
                                "index queue as algorithm scratch. Use [`",
                                stringify!($runner),
                                "`] to retain that queue across calls."
                            )]
            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                let (lookback, count) =
                    validate_rolling_window($label, input, self.period, output.len())?;
                let mut queue = Vec::with_capacity(input.len());
                Ok(rolling_single_extreme_append::<$minimum, false>(
                    input,
                    self.period,
                    lookback,
                    count,
                    output,
                    &mut queue,
                ))
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                    queue: Vec::with_capacity(max_input_len),
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                $stream::new(self.period)
            }
        }

        #[doc = concat!("Prepared Batch Runner for ", $description, ".")]
        #[derive(Debug)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
            queue: Vec<usize>,
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
                let (lookback, count) =
                    validate_rolling_window($label, input, self.config.period, output.len())?;
                Ok(rolling_single_extreme_append::<$minimum, true>(
                    input,
                    self.config.period,
                    lookback,
                    count,
                    output,
                    &mut self.queue,
                ))
            }
        }

        #[doc = concat!("Independent Streaming Computation state for ", $description, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            period: usize,
            buffer: Vec<Float>,
            index: usize,
            count: usize,
        }

        impl $stream {
            fn new(period: usize) -> Result<Self> {
                period_lookback("timeperiod", period)?;
                Ok(Self {
                    period,
                    buffer: vec![0.0 as Float; period],
                    index: 0,
                    count: 0,
                })
            }
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                self.buffer[self.index] = input;
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;
                if self.count < self.period {
                    return Ok(None);
                }

                let mut extreme = self.buffer[0];
                for &value in &self.buffer[1..self.count] {
                    if if $minimum {
                        value < extreme
                    } else {
                        value > extreme
                    } {
                        extreme = value;
                    }
                }
                Ok(Some(extreme))
            }

            fn reset(&mut self) {
                self.buffer.fill(0.0 as Float);
                self.index = 0;
                self.count = 0;
            }
        }
    };
}

define_extreme_execution!(
    MINConfig,
    MINBatchRunner,
    MINStream,
    true,
    "MIN",
    "rolling minimum"
);
define_extreme_execution!(
    MAXConfig,
    MAXBatchRunner,
    MAXStream,
    false,
    "MAX",
    "rolling maximum"
);
