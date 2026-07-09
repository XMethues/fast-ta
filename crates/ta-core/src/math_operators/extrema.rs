//! Extrema and extrema-index Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

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

fn select_stream_index<F>(values: &[Float], indexes: &[usize], mut is_better: F) -> i32
where
    F: FnMut(Float, Float) -> bool,
{
    let mut best_value = values[0];
    let mut best_index = indexes[0];

    for (&value, &index) in values.iter().zip(indexes.iter()).skip(1) {
        if is_better(value, best_value) || (value == best_value && index < best_index) {
            best_value = value;
            best_index = index;
        }
    }

    best_index as i32
}

fn stream_min_max_index(values: &[Float], indexes: &[usize]) -> MINMAXINDEXValue {
    MINMAXINDEXValue {
        min_idx: select_stream_index(values, indexes, |candidate, current| candidate < current),
        max_idx: select_stream_index(values, indexes, |candidate, current| candidate > current),
    }
}

/// TA-Lib-style rolling minimum index.
#[allow(non_snake_case)]
pub fn MININDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MININDEX", out_integer.len(), count)?;
    Ok(rolling_min_max(
        real,
        timeperiod,
        lookback,
        count,
        |output_idx, _, _, min_idx, _| {
            out_integer[output_idx] = min_idx;
        },
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
    Ok(rolling_min_max(
        real,
        timeperiod,
        lookback,
        count,
        |output_idx, _, _, _, max_idx| {
            out_integer[output_idx] = max_idx;
        },
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

macro_rules! define_index_struct {
    ($name:ident, $vec_name:ident, $is_better:expr) => {
        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            buffer: Vec<Float>,
            indexes: Vec<usize>,
            index: usize,
            count: usize,
            seen: usize,
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                period_lookback("timeperiod", timeperiod)?;
                let mut buffer = Vec::new();
                buffer.resize(timeperiod, 0.0 as Float);
                let mut indexes = Vec::new();
                indexes.resize(timeperiod, 0);
                Ok(Self {
                    period: timeperiod,
                    buffer,
                    indexes,
                    index: 0,
                    count: 0,
                    seen: 0,
                })
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Computes compact outputs.
            pub fn compute(&self, real: &[Float], out_integer: &mut [i32]) -> Result<OutputRange> {
                $name(real, self.period, out_integer)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<i32>> {
                $vec_name(real, self.period)
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [i32];
            type OutputOwned = Vec<i32>;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(
                &self,
                inputs: Self::Input<'a>,
                outputs: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(inputs, self.period, outputs)
            }

            fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(inputs, self.period)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = i32;

            fn next(&mut self, input: Float) -> Result<Option<i32>> {
                validate_finite_slice("input", &[input])?;

                self.buffer[self.index] = input;
                self.indexes[self.index] = self.seen;
                self.seen = self.seen.saturating_add(1);
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;

                if self.count < self.period {
                    return Ok(None);
                }

                let is_better = $is_better;
                Ok(Some(select_stream_index(
                    &self.buffer[..self.count],
                    &self.indexes[..self.count],
                    is_better,
                )))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                for value in &mut self.buffer {
                    *value = 0.0 as Float;
                }
                for index in &mut self.indexes {
                    *index = 0;
                }
                self.index = 0;
                self.count = 0;
                self.seen = 0;
            }
        }
    };
}

define_index_struct!(
    MININDEX,
    MININDEX_vec,
    |candidate: Float, current: Float| candidate < current
);
define_index_struct!(
    MAXINDEX,
    MAXINDEX_vec,
    |candidate: Float, current: Float| candidate > current
);

/// MINMAX struct surface.
#[derive(Debug, Clone)]
pub struct MINMAX {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl MINMAX {
    /// Creates a MINMAX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);
        Ok(Self {
            period: timeperiod,
            buffer,
            index: 0,
            count: 0,
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs into parallel buffers.
    pub fn compute(
        &self,
        real: &[Float],
        out_min: &mut [Float],
        out_max: &mut [Float],
    ) -> Result<OutputRange> {
        MINMAX(real, self.period, out_min, out_max)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXOutput> {
        MINMAX_vec(real, self.period)
    }
}

impl Indicator for MINMAX {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = MINMAXOutputMut<'a>;
    type OutputOwned = MINMAXOutput;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MINMAX(inputs, self.period, outputs.min, outputs.max)
    }

    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MINMAX_vec(inputs, self.period)
    }
}

impl StreamingIndicator for MINMAX {
    type Tick = Float;
    type TickOutput = MINMAXValue;

    fn next(&mut self, input: Float) -> Result<Option<MINMAXValue>> {
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
}

impl Resettable for MINMAX {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}

/// MINMAXINDEX struct surface.
#[derive(Debug, Clone)]
pub struct MINMAXINDEX {
    period: usize,
    buffer: Vec<Float>,
    indexes: Vec<usize>,
    index: usize,
    count: usize,
    seen: usize,
}

impl MINMAXINDEX {
    /// Creates a MINMAXINDEX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);
        let mut indexes = Vec::new();
        indexes.resize(timeperiod, 0);
        Ok(Self {
            period: timeperiod,
            buffer,
            indexes,
            index: 0,
            count: 0,
            seen: 0,
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs into parallel buffers.
    pub fn compute(
        &self,
        real: &[Float],
        out_min_idx: &mut [i32],
        out_max_idx: &mut [i32],
    ) -> Result<OutputRange> {
        MINMAXINDEX(real, self.period, out_min_idx, out_max_idx)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXINDEXOutput> {
        MINMAXINDEX_vec(real, self.period)
    }
}

impl Indicator for MINMAXINDEX {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = MINMAXINDEXOutputMut<'a>;
    type OutputOwned = MINMAXINDEXOutput;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MINMAXINDEX(inputs, self.period, outputs.min_idx, outputs.max_idx)
    }

    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MINMAXINDEX_vec(inputs, self.period)
    }
}

impl StreamingIndicator for MINMAXINDEX {
    type Tick = Float;
    type TickOutput = MINMAXINDEXValue;

    fn next(&mut self, input: Float) -> Result<Option<MINMAXINDEXValue>> {
        validate_finite_slice("input", &[input])?;

        self.buffer[self.index] = input;
        self.indexes[self.index] = self.seen;
        self.seen = self.seen.saturating_add(1);
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Ok(None);
        }

        Ok(Some(stream_min_max_index(
            &self.buffer[..self.count],
            &self.indexes[..self.count],
        )))
    }
}

impl Resettable for MINMAXINDEX {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        for index in &mut self.indexes {
            *index = 0;
        }
        self.index = 0;
        self.count = 0;
        self.seen = 0;
    }
}
