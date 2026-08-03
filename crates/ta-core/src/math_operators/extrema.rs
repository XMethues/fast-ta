//! Extrema and extrema-index Math Operators.

use crate::common::CompactPayloadLen;
use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, CompactOutput, Float, Indicator, IndicatorConfig,
    OutputRange, PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec, vec::Vec};

/// Full-length MINMAX output vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct MINMAXOutput {
    /// Minimum values.
    pub min: Vec<Float>,
    /// Maximum values.
    pub max: Vec<Float>,
}

/// Borrowed compact MINMAX output buffers.
pub struct MINMAXOutputMut<'a> {
    /// Minimum output buffer.
    pub min: &'a mut [Float],
    /// Maximum output buffer.
    pub max: &'a mut [Float],
}

/// One valid streaming MINMAX output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MINMAXValue {
    /// Minimum value.
    pub min: Float,
    /// Maximum value.
    pub max: Float,
}

/// Full-length MINMAXINDEX output vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MINMAXINDEXOutput {
    /// Absolute minimum indexes.
    pub min_idx: Vec<i32>,
    /// Absolute maximum indexes.
    pub max_idx: Vec<i32>,
}

/// Borrowed compact MINMAXINDEX output buffers.
pub struct MINMAXINDEXOutputMut<'a> {
    /// Absolute minimum index output buffer.
    pub min_idx: &'a mut [i32],
    /// Absolute maximum index output buffer.
    pub max_idx: &'a mut [i32],
}

/// One valid streaming MINMAXINDEX output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MINMAXINDEXValue {
    /// Absolute minimum index.
    pub min_idx: i32,
    /// Absolute maximum index.
    pub max_idx: i32,
}

fn validate_window(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

fn push_extreme_index<F>(
    deque: &mut Vec<usize>,
    head: usize,
    values: &[Float],
    idx: usize,
    mut is_better: F,
) where
    F: FnMut(Float, Float) -> bool,
{
    while deque.len() > head {
        let &last_idx = deque.last().expect("deque has an active element");
        if is_better(values[idx], values[last_idx]) {
            deque.pop();
        } else {
            break;
        }
    }
    deque.push(idx);
}

fn rolling_min_max<F>(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    mut write: F,
) -> OutputRange
where
    F: FnMut(usize, Float, Float, i32, i32),
{
    if count == 0 {
        return OutputRange::empty();
    }

    let mut min_deque = Vec::new();
    min_deque.reserve(real.len());
    let mut max_deque = Vec::new();
    max_deque.reserve(real.len());
    let mut min_head = 0usize;
    let mut max_head = 0usize;

    for idx in 0..real.len() {
        while min_head < min_deque.len() && min_deque[min_head] + timeperiod <= idx {
            min_head += 1;
        }
        while max_head < max_deque.len() && max_deque[max_head] + timeperiod <= idx {
            max_head += 1;
        }

        push_extreme_index(&mut min_deque, min_head, real, idx, |candidate, current| {
            candidate < current
        });
        push_extreme_index(&mut max_deque, max_head, real, idx, |candidate, current| {
            candidate > current
        });

        if idx + 1 >= timeperiod {
            let output_idx = idx + 1 - timeperiod;
            let min_idx = min_deque[min_head];
            let max_idx = max_deque[max_head];
            write(
                output_idx,
                real[min_idx],
                real[max_idx],
                min_idx as i32,
                max_idx as i32,
            );
        }
    }

    OutputRange::new(lookback, count)
}

fn rolling_single_index<F>(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    out_integer: &mut [i32],
    mut is_better: F,
) -> OutputRange
where
    F: FnMut(Float, Float) -> bool,
{
    if count == 0 {
        return OutputRange::empty();
    }

    let mut deque = Vec::with_capacity(real.len());
    let mut head = 0usize;
    for idx in 0..real.len() {
        while head < deque.len() && deque[head] + timeperiod <= idx {
            head += 1;
        }
        push_extreme_index(&mut deque, head, real, idx, &mut is_better);
        if idx + 1 >= timeperiod {
            out_integer[idx + 1 - timeperiod] = deque[head] as i32;
        }
    }
    OutputRange::new(lookback, count)
}

/// TA-Lib-style rolling minimum index.
#[allow(non_snake_case)]
pub fn MININDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MININDEX", out_integer.len(), count)?;
    Ok(rolling_single_index(
        real,
        timeperiod,
        lookback,
        count,
        out_integer,
        |candidate, current| candidate < current,
    ))
}

/// Computes rolling minimum indexes into a full-length vector padded with zeroes.
#[allow(non_snake_case)]
pub fn MININDEX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<i32>> {
    let mut compact = compact_buffer::<i32>(real.len());
    let range = MININDEX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// TA-Lib-style rolling maximum index.
#[allow(non_snake_case)]
pub fn MAXINDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MAXINDEX", out_integer.len(), count)?;
    Ok(rolling_single_index(
        real,
        timeperiod,
        lookback,
        count,
        out_integer,
        |candidate, current| candidate > current,
    ))
}

/// Computes rolling maximum indexes into a full-length vector padded with zeroes.
#[allow(non_snake_case)]
pub fn MAXINDEX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<i32>> {
    let mut compact = compact_buffer::<i32>(real.len());
    let range = MAXINDEX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// TA-Lib-style rolling minimum and maximum.
#[allow(non_snake_case)]
pub fn MINMAX(
    real: &[Float],
    timeperiod: usize,
    out_min: &mut [Float],
    out_max: &mut [Float],
) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MINMAX min", out_min.len(), count)?;
    validate_output_len("MINMAX max", out_max.len(), count)?;
    Ok(rolling_min_max(
        real,
        timeperiod,
        lookback,
        count,
        |output_idx, min_value, max_value, _, _| {
            out_min[output_idx] = min_value;
            out_max[output_idx] = max_value;
        },
    ))
}

/// Computes rolling minimum and maximum into full-length vectors.
#[allow(non_snake_case)]
pub fn MINMAX_vec(real: &[Float], timeperiod: usize) -> Result<MINMAXOutput> {
    let mut min_compact = compact_buffer::<Float>(real.len());
    let mut max_compact = compact_buffer::<Float>(real.len());
    let range = MINMAX(real, timeperiod, &mut min_compact, &mut max_compact)?;
    Ok(MINMAXOutput {
        min: padded_from_compact(real.len(), range, &min_compact[..range.nb_element]),
        max: padded_from_compact(real.len(), range, &max_compact[..range.nb_element]),
    })
}

/// TA-Lib-style rolling minimum and maximum indexes.
#[allow(non_snake_case)]
pub fn MINMAXINDEX(
    real: &[Float],
    timeperiod: usize,
    out_min_idx: &mut [i32],
    out_max_idx: &mut [i32],
) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MINMAXINDEX min", out_min_idx.len(), count)?;
    validate_output_len("MINMAXINDEX max", out_max_idx.len(), count)?;
    Ok(rolling_min_max(
        real,
        timeperiod,
        lookback,
        count,
        |output_idx, _, _, min_idx, max_idx| {
            out_min_idx[output_idx] = min_idx;
            out_max_idx[output_idx] = max_idx;
        },
    ))
}

/// Computes rolling minimum and maximum indexes into full-length vectors.
#[allow(non_snake_case)]
pub fn MINMAXINDEX_vec(real: &[Float], timeperiod: usize) -> Result<MINMAXINDEXOutput> {
    let mut min_compact = compact_buffer::<i32>(real.len());
    let mut max_compact = compact_buffer::<i32>(real.len());
    let range = MINMAXINDEX(real, timeperiod, &mut min_compact, &mut max_compact)?;
    Ok(MINMAXINDEXOutput {
        min_idx: padded_from_compact(real.len(), range, &min_compact[..range.nb_element]),
        max_idx: padded_from_compact(real.len(), range, &max_compact[..range.nb_element]),
    })
}

/// Named compact rolling minimum and maximum value columns.
#[derive(Debug, Clone, PartialEq)]
pub struct MINMAXValues {
    /// Minimum values.
    pub min: Vec<Float>,
    /// Maximum values.
    pub max: Vec<Float>,
}

impl CompactPayloadLen for MINMAXValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.min.len() != self.max.len() {
            return Err(TalibError::invalid_input(
                "MINMAX Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.min.len())
    }
}

/// Caller-owned compact rolling minimum and maximum value columns.
#[derive(Debug)]
pub struct MINMAXValuesMut<'a> {
    /// Minimum output buffer.
    pub min: &'a mut [Float],
    /// Maximum output buffer.
    pub max: &'a mut [Float],
}

/// Named compact rolling minimum and maximum absolute index columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MINMAXINDEXValues {
    /// Absolute minimum indexes.
    pub min_idx: Vec<usize>,
    /// Absolute maximum indexes.
    pub max_idx: Vec<usize>,
}

impl CompactPayloadLen for MINMAXINDEXValues {
    fn compact_payload_len(&self) -> Result<usize> {
        if self.min_idx.len() != self.max_idx.len() {
            return Err(TalibError::invalid_input(
                "MINMAXINDEX Compact Output columns must have equal lengths",
            ));
        }
        Ok(self.min_idx.len())
    }
}

/// Caller-owned compact rolling minimum and maximum absolute index columns.
#[derive(Debug)]
pub struct MINMAXINDEXValuesMut<'a> {
    /// Absolute minimum index output buffer.
    pub min_idx: &'a mut [usize],
    /// Absolute maximum index output buffer.
    pub max_idx: &'a mut [usize],
}

#[derive(Debug, Clone)]
struct ExtremaAppendScratch {
    min: Vec<usize>,
    max: Vec<usize>,
}

impl ExtremaAppendScratch {
    fn with_capacity(max_input_len: usize) -> Self {
        Self {
            min: Vec::with_capacity(max_input_len),
            max: Vec::with_capacity(max_input_len),
        }
    }
}

fn validate_value_outputs(
    real: &[Float],
    period: usize,
    min_len: usize,
    max_len: usize,
) -> Result<(usize, usize)> {
    let (lookback, count) = validate_window(real, period)?;
    validate_output_len("MINMAX min", min_len, count)?;
    validate_output_len("MINMAX max", max_len, count)?;
    Ok((lookback, count))
}

fn validate_index_outputs(
    real: &[Float],
    period: usize,
    min_len: usize,
    max_len: usize,
) -> Result<(usize, usize)> {
    let (lookback, count) = validate_window(real, period)?;
    validate_output_len("MINMAXINDEX min", min_len, count)?;
    validate_output_len("MINMAXINDEX max", max_len, count)?;
    Ok((lookback, count))
}

// Preparation and one-shot setup reserve enough indexes for every push. This
// helper makes that private invariant explicit to the optimizer so retained
// scratch does not pay a dead growth branch on each observation.
#[inline(always)]
unsafe fn push_reserved_index(queue: &mut Vec<usize>, index: usize) {
    let len = queue.len();
    debug_assert!(len < queue.capacity());
    // SAFETY: every caller reserves at least the source length before entering
    // a kernel, and a queue receives at most one push per source observation.
    unsafe {
        queue.as_mut_ptr().add(len).write(index);
        queue.set_len(len + 1);
    }
}

// Performance seam: the legacy/current `rolling_min_max` loop above and these
// new value/index loops intentionally remain duplicated. Benchmark candidates
// that routed writes through a generic writer or queue state through a generic
// runner regressed the qualified paths; concrete direct-write kernels with local
// queue heads were the benchmark-selected shape. Prepared callers therefore pass
// disjoint scratch vectors instead of a mutable runner, keeping field borrowing
// and dispatch out of the hot loop. Do not contract this duplication until issue
// #15 removes the legacy/current execution surface; issue #15 is the explicit
// contraction point for the remaining legacy duplication.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // Concrete SoA slices keep the prepared hot loop closure-free.
fn rolling_min_max_values_append<const RESERVED_PUSH: bool>(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    out_min: &mut [Float],
    out_max: &mut [Float],
    min_queue: &mut Vec<usize>,
    max_queue: &mut Vec<usize>,
) -> OutputRange {
    min_queue.clear();
    max_queue.clear();
    if count == 0 {
        return OutputRange::empty();
    }

    let mut min_head = 0usize;
    let mut max_head = 0usize;
    for idx in 0..real.len() {
        if idx >= period {
            let expired_through = idx - period;
            while min_head < min_queue.len() && min_queue[min_head] <= expired_through {
                min_head += 1;
            }
            while max_head < max_queue.len() && max_queue[max_head] <= expired_through {
                max_head += 1;
            }
        }

        while min_queue.len() > min_head {
            let last_idx = min_queue[min_queue.len() - 1];
            if real[idx] < real[last_idx] {
                min_queue.pop();
            } else {
                break;
            }
        }
        while max_queue.len() > max_head {
            let last_idx = max_queue[max_queue.len() - 1];
            if real[idx] > real[last_idx] {
                max_queue.pop();
            } else {
                break;
            }
        }
        if RESERVED_PUSH {
            // SAFETY: prepared queues have source-capacity headroom and receive
            // at most one push for each `idx` in this loop.
            unsafe {
                push_reserved_index(min_queue, idx);
                push_reserved_index(max_queue, idx);
            }
        } else {
            min_queue.push(idx);
            max_queue.push(idx);
        }

        if idx + 1 >= period {
            let output_idx = idx + 1 - period;
            out_min[output_idx] = real[min_queue[min_head]];
            out_max[output_idx] = real[max_queue[max_head]];
        }
    }

    OutputRange::new(lookback, count)
}

// Keep a second concrete kernel for usize index output. A generic writer closure
// was deliberately avoided so prepared value and index loops write directly to
// their final Structure-of-Arrays columns.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // Concrete SoA slices keep the prepared hot loop closure-free.
fn rolling_min_max_indexes_append<const RESERVED_PUSH: bool>(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    out_min_idx: &mut [usize],
    out_max_idx: &mut [usize],
    min_queue: &mut Vec<usize>,
    max_queue: &mut Vec<usize>,
) -> OutputRange {
    min_queue.clear();
    max_queue.clear();
    if count == 0 {
        return OutputRange::empty();
    }

    let mut min_head = 0usize;
    let mut max_head = 0usize;
    for idx in 0..real.len() {
        if idx >= period {
            let expired_through = idx - period;
            while min_head < min_queue.len() && min_queue[min_head] <= expired_through {
                min_head += 1;
            }
            while max_head < max_queue.len() && max_queue[max_head] <= expired_through {
                max_head += 1;
            }
        }

        while min_queue.len() > min_head {
            let last_idx = min_queue[min_queue.len() - 1];
            if real[idx] < real[last_idx] {
                min_queue.pop();
            } else {
                break;
            }
        }
        while max_queue.len() > max_head {
            let last_idx = max_queue[max_queue.len() - 1];
            if real[idx] > real[last_idx] {
                max_queue.pop();
            } else {
                break;
            }
        }
        if RESERVED_PUSH {
            // SAFETY: prepared queues have source-capacity headroom and receive
            // at most one push for each `idx` in this loop.
            unsafe {
                push_reserved_index(min_queue, idx);
                push_reserved_index(max_queue, idx);
            }
        } else {
            min_queue.push(idx);
            max_queue.push(idx);
        }

        if idx + 1 >= period {
            let output_idx = idx + 1 - period;
            out_min_idx[output_idx] = min_queue[min_head];
            out_max_idx[output_idx] = max_queue[max_head];
        }
    }

    OutputRange::new(lookback, count)
}

#[inline(always)]
fn rolling_single_extreme_index_append<const MINIMUM: bool, const RESERVED_PUSH: bool>(
    real: &[Float],
    period: usize,
    lookback: usize,
    count: usize,
    output: &mut [usize],
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
            unsafe { push_reserved_index(queue, idx) };
        } else {
            queue.push(idx);
        }

        if idx + 1 >= period {
            output[idx + 1 - period] = queue[head];
        }
    }

    OutputRange::new(lookback, count)
}

macro_rules! define_single_index_execution {
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
            type Output = Vec<usize>;
            type OutputMut<'a> = &'a mut [usize];
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let (lookback, count) = validate_window(input, self.period)?;
                let mut values = vec![0; count];
                let mut queue = Vec::with_capacity(input.len());
                let range = rolling_single_extreme_index_append::<$minimum, false>(
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
                let (lookback, count) = validate_window(input, self.period)?;
                validate_output_len($label, output.len(), count)?;
                let mut queue = Vec::with_capacity(input.len());
                Ok(rolling_single_extreme_index_append::<$minimum, false>(
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
                let (lookback, count) = validate_window(input, self.config.period)?;
                validate_output_len($label, output.len(), count)?;
                Ok(rolling_single_extreme_index_append::<$minimum, true>(
                    input,
                    self.config.period,
                    lookback,
                    count,
                    output,
                    &mut self.queue,
                ))
            }
        }

        #[doc = concat!(
                            "Independent Streaming Computation state for ",
                            $description,
                            "."
                        )]
        #[derive(Debug, Clone)]
        pub struct $stream {
            period: usize,
            buffer: Vec<Float>,
            indexes: Vec<usize>,
            index: usize,
            count: usize,
            seen: usize,
        }

        impl $stream {
            fn new(period: usize) -> Result<Self> {
                period_lookback("timeperiod", period)?;
                Ok(Self {
                    period,
                    buffer: vec![0.0 as Float; period],
                    indexes: vec![0; period],
                    index: 0,
                    count: 0,
                    seen: 0,
                })
            }

            #[inline]
            const fn period(&self) -> usize {
                self.period
            }

            fn next_mapped<T, F>(
                &mut self,
                input: Float,
                max_position: usize,
                map_output: F,
            ) -> Result<Option<T>>
            where
                F: FnOnce(usize) -> Result<T>,
            {
                validate_finite_slice("input", &[input])?;
                if self.seen > max_position {
                    return Err(TalibError::computation_error(format!(
                        "{} stream position {} exceeds supported maximum {}",
                        $label, self.seen, max_position
                    )));
                }
                let next_seen = self.seen.checked_add(1).ok_or_else(|| {
                    TalibError::computation_error(concat!($label, " stream position overflow"))
                })?;
                let next_count = (self.count + 1).min(self.period);
                let output = if next_count < self.period {
                    None
                } else {
                    let index = select_pending_stream_index(
                        &self.buffer,
                        &self.indexes,
                        next_count,
                        self.index,
                        input,
                        self.seen,
                        |candidate, current| {
                            if $minimum {
                                candidate < current
                            } else {
                                candidate > current
                            }
                        },
                    );
                    Some(map_output(index)?)
                };

                self.buffer[self.index] = input;
                self.indexes[self.index] = self.seen;
                self.seen = next_seen;
                self.count = next_count;
                self.index = (self.index + 1) % self.period;
                Ok(output)
            }
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = usize;

            #[inline]
            fn next(&mut self, input: Float) -> Result<Option<usize>> {
                self.next_mapped(input, usize::MAX, Ok)
            }

            fn reset(&mut self) {
                self.buffer.fill(0.0 as Float);
                self.indexes.fill(0);
                self.index = 0;
                self.count = 0;
                self.seen = 0;
            }
        }
    };
}

define_single_index_execution!(
    MININDEXConfig,
    MININDEXBatchRunner,
    MININDEXStream,
    true,
    "MININDEX",
    "rolling minimum absolute-index"
);
define_single_index_execution!(
    MAXINDEXConfig,
    MAXINDEXBatchRunner,
    MAXINDEXStream,
    false,
    "MAXINDEX",
    "rolling maximum absolute-index"
);

/// Immutable rolling minimum/maximum value Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MINMAXConfig {
    period: usize,
}

impl MINMAXConfig {
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

impl crate::traits::sealed::Sealed for MINMAXConfig {}

impl IndicatorConfig for MINMAXConfig {
    type Input<'a> = &'a [Float];
    type Output = MINMAXValues;
    type OutputMut<'a> = MINMAXValuesMut<'a>;
    type BatchRunner = MINMAXBatchRunner;
    type Stream = MINMAXStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_window(input, self.period)?;
        let mut values = MINMAXValues {
            min: vec![0.0 as Float; count],
            max: vec![0.0 as Float; count],
        };
        let mut scratch = ExtremaAppendScratch::with_capacity(input.len());
        let range = rolling_min_max_values_append::<false>(
            input,
            self.period,
            lookback,
            count,
            &mut values.min,
            &mut values.max,
            &mut scratch.min,
            &mut scratch.max,
        );
        CompactOutput::new(input.len(), range, values)
    }

    /// Computes into caller-owned columns. This one-shot path allocates two
    /// input-length index queues as documented algorithm scratch; output is not
    /// allocated. Use [`MINMAXBatchRunner`] to retain that scratch.
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) =
            validate_value_outputs(input, self.period, output.min.len(), output.max.len())?;
        let mut scratch = ExtremaAppendScratch::with_capacity(input.len());
        Ok(rolling_min_max_values_append::<false>(
            input,
            self.period,
            lookback,
            count,
            output.min,
            output.max,
            &mut scratch.min,
            &mut scratch.max,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        let scratch = ExtremaAppendScratch::with_capacity(max_input_len);
        Ok(MINMAXBatchRunner {
            config: *self,
            max_input_len,
            min_scratch: scratch.min,
            max_scratch: scratch.max,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        MINMAXStream::new(self.period)
    }
}

/// Reusable Prepared Batch Runner for rolling minimum/maximum values.
#[derive(Debug)]
pub struct MINMAXBatchRunner {
    config: MINMAXConfig,
    max_input_len: usize,
    min_scratch: Vec<usize>,
    max_scratch: Vec<usize>,
}

impl crate::traits::sealed::Sealed for MINMAXBatchRunner {}

impl PreparedBatchRunner<MINMAXConfig> for MINMAXBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <MINMAXConfig as IndicatorConfig>::Input<'a>,
        output: <MINMAXConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MINMAXConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let (lookback, count) = validate_value_outputs(
            input,
            self.config.period,
            output.min.len(),
            output.max.len(),
        )?;
        let Self {
            min_scratch,
            max_scratch,
            ..
        } = self;
        Ok(rolling_min_max_values_append::<true>(
            input,
            self.config.period,
            lookback,
            count,
            output.min,
            output.max,
            min_scratch,
            max_scratch,
        ))
    }
}

/// Immutable rolling minimum/maximum absolute-index Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MINMAXINDEXConfig {
    period: usize,
}

impl MINMAXINDEXConfig {
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

impl crate::traits::sealed::Sealed for MINMAXINDEXConfig {}

impl IndicatorConfig for MINMAXINDEXConfig {
    type Input<'a> = &'a [Float];
    type Output = MINMAXINDEXValues;
    type OutputMut<'a> = MINMAXINDEXValuesMut<'a>;
    type BatchRunner = MINMAXINDEXBatchRunner;
    type Stream = MINMAXINDEXStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_window(input, self.period)?;
        let mut values = MINMAXINDEXValues {
            min_idx: vec![0; count],
            max_idx: vec![0; count],
        };
        let mut scratch = ExtremaAppendScratch::with_capacity(input.len());
        let range = rolling_min_max_indexes_append::<false>(
            input,
            self.period,
            lookback,
            count,
            &mut values.min_idx,
            &mut values.max_idx,
            &mut scratch.min,
            &mut scratch.max,
        );
        CompactOutput::new(input.len(), range, values)
    }

    /// Computes into caller-owned columns. This one-shot path allocates two
    /// input-length index queues as documented algorithm scratch; output is not
    /// allocated. Use [`MINMAXINDEXBatchRunner`] to retain that scratch.
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_index_outputs(
            input,
            self.period,
            output.min_idx.len(),
            output.max_idx.len(),
        )?;
        let mut scratch = ExtremaAppendScratch::with_capacity(input.len());
        Ok(rolling_min_max_indexes_append::<false>(
            input,
            self.period,
            lookback,
            count,
            output.min_idx,
            output.max_idx,
            &mut scratch.min,
            &mut scratch.max,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        let scratch = ExtremaAppendScratch::with_capacity(max_input_len);
        Ok(MINMAXINDEXBatchRunner {
            config: *self,
            max_input_len,
            min_scratch: scratch.min,
            max_scratch: scratch.max,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        MINMAXINDEXStream::new(self.period)
    }
}

/// Reusable Prepared Batch Runner for rolling absolute extrema indexes.
#[derive(Debug)]
pub struct MINMAXINDEXBatchRunner {
    config: MINMAXINDEXConfig,
    max_input_len: usize,
    min_scratch: Vec<usize>,
    max_scratch: Vec<usize>,
}

impl crate::traits::sealed::Sealed for MINMAXINDEXBatchRunner {}

impl PreparedBatchRunner<MINMAXINDEXConfig> for MINMAXINDEXBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <MINMAXINDEXConfig as IndicatorConfig>::Input<'a>,
        output: <MINMAXINDEXConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MINMAXINDEXConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let (lookback, count) = validate_index_outputs(
            input,
            self.config.period,
            output.min_idx.len(),
            output.max_idx.len(),
        )?;
        let Self {
            min_scratch,
            max_scratch,
            ..
        } = self;
        Ok(rolling_min_max_indexes_append::<true>(
            input,
            self.config.period,
            lookback,
            count,
            output.min_idx,
            output.max_idx,
            min_scratch,
            max_scratch,
        ))
    }
}

/// Independent Streaming Computation state for rolling minimum/maximum values.
#[derive(Debug, Clone)]
pub struct MINMAXStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl MINMAXStream {
    fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        Ok(Self {
            period,
            buffer: vec![0.0 as Float; period],
            index: 0,
            count: 0,
        })
    }

    #[inline]
    const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for MINMAXStream {}

impl StreamingComputation<MINMAXConfig> for MINMAXStream {
    type Tick = Float;
    type TickOutput = MINMAXValue;

    fn next(&mut self, input: Float) -> Result<Option<Self::TickOutput>> {
        validate_finite_slice("input", &[input])?;
        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;
        if self.count < self.period {
            return Ok(None);
        }

        let values = &self.buffer[..self.count];
        Ok(Some(MINMAXValue {
            min: values.iter().copied().fold(values[0], Float::min),
            max: values.iter().copied().fold(values[0], Float::max),
        }))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}

/// One valid streaming output with Rust-native absolute indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MINMAXINDEXStreamValue {
    /// Absolute minimum index.
    pub min_idx: usize,
    /// Absolute maximum index.
    pub max_idx: usize,
}

fn select_pending_stream_index<F>(
    values: &[Float],
    indexes: &[usize],
    count: usize,
    pending_slot: usize,
    pending_value: Float,
    pending_index: usize,
    mut is_better: F,
) -> usize
where
    F: FnMut(Float, Float) -> bool,
{
    let mut best_value = pending_value;
    let mut best_index = pending_index;
    let mut consider = |value: Float, index: usize| {
        if is_better(value, best_value) || (value == best_value && index < best_index) {
            best_value = value;
            best_index = index;
        }
    };

    for (&value, &index) in values[..pending_slot].iter().zip(&indexes[..pending_slot]) {
        consider(value, index);
    }
    for (&value, &index) in values[pending_slot + 1..count]
        .iter()
        .zip(&indexes[pending_slot + 1..count])
    {
        consider(value, index);
    }
    best_index
}

fn pending_stream_min_max_index(
    values: &[Float],
    indexes: &[usize],
    count: usize,
    pending_slot: usize,
    pending_value: Float,
    pending_index: usize,
) -> MINMAXINDEXStreamValue {
    MINMAXINDEXStreamValue {
        min_idx: select_pending_stream_index(
            values,
            indexes,
            count,
            pending_slot,
            pending_value,
            pending_index,
            |candidate, current| candidate < current,
        ),
        max_idx: select_pending_stream_index(
            values,
            indexes,
            count,
            pending_slot,
            pending_value,
            pending_index,
            |candidate, current| candidate > current,
        ),
    }
}

/// Independent Streaming Computation state for absolute extrema indexes.
#[derive(Debug, Clone)]
pub struct MINMAXINDEXStream {
    period: usize,
    buffer: Vec<Float>,
    indexes: Vec<usize>,
    index: usize,
    count: usize,
    seen: usize,
}

impl MINMAXINDEXStream {
    fn new(period: usize) -> Result<Self> {
        period_lookback("timeperiod", period)?;
        Ok(Self {
            period,
            buffer: vec![0.0 as Float; period],
            indexes: vec![0; period],
            index: 0,
            count: 0,
            seen: 0,
        })
    }

    #[inline]
    const fn period(&self) -> usize {
        self.period
    }

    fn next_mapped<T, F>(
        &mut self,
        input: Float,
        max_position: usize,
        map_output: F,
    ) -> Result<Option<T>>
    where
        F: FnOnce(MINMAXINDEXStreamValue) -> Result<T>,
    {
        validate_finite_slice("input", &[input])?;
        if self.seen > max_position {
            return Err(TalibError::computation_error(format!(
                "MINMAXINDEX stream position {} exceeds supported maximum {}",
                self.seen, max_position
            )));
        }
        let next_seen = self
            .seen
            .checked_add(1)
            .ok_or_else(|| TalibError::computation_error("MINMAXINDEX stream position overflow"))?;
        let next_count = (self.count + 1).min(self.period);
        let output = if next_count < self.period {
            None
        } else {
            let value = pending_stream_min_max_index(
                &self.buffer,
                &self.indexes,
                next_count,
                self.index,
                input,
                self.seen,
            );
            Some(map_output(value)?)
        };

        self.buffer[self.index] = input;
        self.indexes[self.index] = self.seen;
        self.seen = next_seen;
        self.count = next_count;
        self.index = (self.index + 1) % self.period;
        Ok(output)
    }
}

impl crate::traits::sealed::Sealed for MINMAXINDEXStream {}

impl StreamingComputation<MINMAXINDEXConfig> for MINMAXINDEXStream {
    type Tick = Float;
    type TickOutput = MINMAXINDEXStreamValue;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Self::TickOutput>> {
        self.next_mapped(input, usize::MAX, Ok)
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.indexes.fill(0);
        self.index = 0;
        self.count = 0;
        self.seen = 0;
    }
}

macro_rules! define_index_adapter {
    ($name:ident, $vec_name:ident, $config:ident, $stream:ident) => {
        #[doc = concat!("Legacy ", stringify!($name), " indicator.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            stream: $stream,
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                let config = $config::new(timeperiod)?;
                let stream = IndicatorConfig::stream(&config)?;
                Ok(Self { stream })
            }

            /// Returns the configured Period.
            #[inline]
            pub const fn period(&self) -> usize {
                self.stream.period()
            }

            /// Computes compact outputs.
            #[inline]
            pub fn compute(&self, real: &[Float], out_integer: &mut [i32]) -> Result<OutputRange> {
                $name(real, self.period(), out_integer)
            }

            /// Computes full-length outputs.
            #[inline]
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<i32>> {
                $vec_name(real, self.period())
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [i32];
            type OutputOwned = Vec<i32>;

            #[inline]
            fn lookback(&self) -> usize {
                self.period() - 1
            }

            #[inline]
            fn compute<'a>(
                &self,
                inputs: Self::Input<'a>,
                outputs: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(inputs, self.period(), outputs)
            }

            #[inline]
            fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(inputs, self.period())
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = i32;

            #[inline]
            fn next(&mut self, input: Float) -> Result<Option<i32>> {
                self.stream.next_mapped(input, i32::MAX as usize, |index| {
                    i32::try_from(index).map_err(|_| {
                        TalibError::computation_error(format!(
                            "{} stream index {index} does not fit legacy i32 output",
                            stringify!($name)
                        ))
                    })
                })
            }
        }

        impl Resettable for $name {
            #[inline]
            fn reset(&mut self) {
                StreamingComputation::<$config>::reset(&mut self.stream);
            }
        }
    };
}

define_index_adapter!(MININDEX, MININDEX_vec, MININDEXConfig, MININDEXStream);
define_index_adapter!(MAXINDEX, MAXINDEX_vec, MAXINDEXConfig, MAXINDEXStream);

/// Legacy MINMAX indicator.
///
/// This compatibility adapter preserves the historical combined batch and
/// streaming signatures while storing only its [`MINMAXStream`] execution state.
#[derive(Debug, Clone)]
pub struct MINMAX {
    stream: MINMAXStream,
}

impl MINMAX {
    /// Creates a MINMAX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = MINMAXConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.stream.period()
    }

    /// Computes compact outputs into parallel buffers.
    #[inline]
    pub fn compute(
        &self,
        real: &[Float],
        out_min: &mut [Float],
        out_max: &mut [Float],
    ) -> Result<OutputRange> {
        MINMAX(real, self.period(), out_min, out_max)
    }

    /// Computes full-length outputs.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXOutput> {
        MINMAX_vec(real, self.period())
    }
}

impl Indicator for MINMAX {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = MINMAXOutputMut<'a>;
    type OutputOwned = MINMAXOutput;

    #[inline]
    fn lookback(&self) -> usize {
        self.period() - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MINMAX(inputs, self.period(), outputs.min, outputs.max)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MINMAX_vec(inputs, self.period())
    }
}

impl StreamingIndicator for MINMAX {
    type Tick = Float;
    type TickOutput = MINMAXValue;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<MINMAXValue>> {
        StreamingComputation::<MINMAXConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for MINMAX {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<MINMAXConfig>::reset(&mut self.stream);
    }
}

fn legacy_stream_index(index: usize) -> Result<i32> {
    i32::try_from(index).map_err(|_| {
        TalibError::computation_error(format!(
            "MINMAXINDEX stream index {index} does not fit legacy i32 output"
        ))
    })
}

/// Legacy MINMAXINDEX indicator.
///
/// This compatibility adapter preserves the historical combined batch and
/// streaming signatures while storing only its [`MINMAXINDEXStream`] execution state.
#[derive(Debug, Clone)]
pub struct MINMAXINDEX {
    stream: MINMAXINDEXStream,
}

impl MINMAXINDEX {
    /// Creates a MINMAXINDEX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let config = MINMAXINDEXConfig::new(timeperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { stream })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.stream.period()
    }

    /// Computes compact outputs into parallel buffers.
    #[inline]
    pub fn compute(
        &self,
        real: &[Float],
        out_min_idx: &mut [i32],
        out_max_idx: &mut [i32],
    ) -> Result<OutputRange> {
        MINMAXINDEX(real, self.period(), out_min_idx, out_max_idx)
    }

    /// Computes full-length outputs.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXINDEXOutput> {
        MINMAXINDEX_vec(real, self.period())
    }
}

impl Indicator for MINMAXINDEX {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = MINMAXINDEXOutputMut<'a>;
    type OutputOwned = MINMAXINDEXOutput;

    #[inline]
    fn lookback(&self) -> usize {
        self.period() - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MINMAXINDEX(inputs, self.period(), outputs.min_idx, outputs.max_idx)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MINMAXINDEX_vec(inputs, self.period())
    }
}

impl StreamingIndicator for MINMAXINDEX {
    type Tick = Float;
    type TickOutput = MINMAXINDEXValue;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<MINMAXINDEXValue>> {
        self.stream.next_mapped(
            input,
            i32::MAX as usize,
            |value| -> Result<MINMAXINDEXValue> {
                Ok(MINMAXINDEXValue {
                    min_idx: legacy_stream_index(value.min_idx)?,
                    max_idx: legacy_stream_index(value.max_idx)?,
                })
            },
        )
    }
}

impl Resettable for MINMAXINDEX {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<MINMAXINDEXConfig>::reset(&mut self.stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmaxindex_stream_conversion_and_position_failures_preserve_state() {
        let mut conversion = MINMAXINDEX::new(2).unwrap();
        conversion
            .stream
            .buffer
            .copy_from_slice(&[0.0 as Float, 10.0]);
        conversion.stream.indexes[0] = i32::MAX as usize + 1;
        conversion.stream.indexes[1] = 0;
        conversion.stream.index = 1;
        conversion.stream.count = 2;
        conversion.stream.seen = 1;
        let before_conversion = conversion.stream.clone();
        assert!(StreamingIndicator::next(&mut conversion, 20.0).is_err());
        assert_eq!(conversion.stream.buffer, before_conversion.buffer);
        assert_eq!(conversion.stream.indexes, before_conversion.indexes);
        assert_eq!(conversion.stream.index, before_conversion.index);
        assert_eq!(conversion.stream.count, before_conversion.count);
        assert_eq!(conversion.stream.seen, before_conversion.seen);

        let mut legacy_position = MINMAXINDEX::new(1).unwrap();
        legacy_position.stream.seen = i32::MAX as usize + 1;
        let before_legacy_position = legacy_position.stream.clone();
        assert!(StreamingIndicator::next(&mut legacy_position, 1.0).is_err());
        assert_eq!(legacy_position.stream.buffer, before_legacy_position.buffer);
        assert_eq!(
            legacy_position.stream.indexes,
            before_legacy_position.indexes
        );
        assert_eq!(legacy_position.stream.index, before_legacy_position.index);
        assert_eq!(legacy_position.stream.count, before_legacy_position.count);
        assert_eq!(legacy_position.stream.seen, before_legacy_position.seen);

        let mut native_position = MINMAXINDEXStream::new(1).unwrap();
        native_position.seen = usize::MAX;
        let before_native_position = native_position.clone();
        assert!(
            StreamingComputation::<MINMAXINDEXConfig>::next(&mut native_position, 1.0,).is_err()
        );
        assert_eq!(native_position.buffer, before_native_position.buffer);
        assert_eq!(native_position.indexes, before_native_position.indexes);
        assert_eq!(native_position.index, before_native_position.index);
        assert_eq!(native_position.count, before_native_position.count);
        assert_eq!(native_position.seen, before_native_position.seen);
    }

    #[test]
    fn single_index_stream_conversion_and_position_failures_preserve_state() {
        let mut conversion = MININDEX::new(2).unwrap();
        conversion
            .stream
            .buffer
            .copy_from_slice(&[0.0 as Float, 10.0]);
        conversion.stream.indexes[0] = i32::MAX as usize + 1;
        conversion.stream.indexes[1] = 0;
        conversion.stream.index = 1;
        conversion.stream.count = 2;
        conversion.stream.seen = 1;
        let before_conversion = conversion.stream.clone();
        assert!(StreamingIndicator::next(&mut conversion, 20.0).is_err());
        assert_eq!(conversion.stream.buffer, before_conversion.buffer);
        assert_eq!(conversion.stream.indexes, before_conversion.indexes);
        assert_eq!(conversion.stream.index, before_conversion.index);
        assert_eq!(conversion.stream.count, before_conversion.count);
        assert_eq!(conversion.stream.seen, before_conversion.seen);

        let mut legacy_position = MAXINDEX::new(1).unwrap();
        legacy_position.stream.seen = i32::MAX as usize + 1;
        let before_legacy_position = legacy_position.stream.clone();
        assert!(StreamingIndicator::next(&mut legacy_position, 1.0).is_err());
        assert_eq!(legacy_position.stream.buffer, before_legacy_position.buffer);
        assert_eq!(
            legacy_position.stream.indexes,
            before_legacy_position.indexes
        );
        assert_eq!(legacy_position.stream.index, before_legacy_position.index);
        assert_eq!(legacy_position.stream.count, before_legacy_position.count);
        assert_eq!(legacy_position.stream.seen, before_legacy_position.seen);

        let mut native_position = MAXINDEXStream::new(1).unwrap();
        native_position.seen = usize::MAX;
        let before_native_position = native_position.clone();
        assert!(StreamingComputation::<MAXINDEXConfig>::next(&mut native_position, 1.0).is_err());
        assert_eq!(native_position.buffer, before_native_position.buffer);
        assert_eq!(native_position.indexes, before_native_position.indexes);
        assert_eq!(native_position.index, before_native_position.index);
        assert_eq!(native_position.count, before_native_position.count);
        assert_eq!(native_position.seen, before_native_position.seen);
    }

    #[test]
    fn named_compact_payloads_reject_unequal_columns() {
        let values_error = CompactOutput::new(
            3,
            OutputRange::new(1, 2),
            MINMAXValues {
                min: vec![1.0 as Float, 2.0],
                max: vec![3.0 as Float],
            },
        )
        .unwrap_err();
        assert!(matches!(values_error, TalibError::InvalidInput { .. }));

        let indexes_error = CompactOutput::new(
            3,
            OutputRange::new(1, 2),
            MINMAXINDEXValues {
                min_idx: vec![0, 1],
                max_idx: vec![1],
            },
        )
        .unwrap_err();
        assert!(matches!(indexes_error, TalibError::InvalidInput { .. }));
    }
}
