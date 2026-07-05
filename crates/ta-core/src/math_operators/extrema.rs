//! Extrema and extrema-index Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
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

/// Full-length MINMAXINDEX output vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MINMAXINDEXOutput {
    /// Absolute minimum indexes.
    pub min_idx: Vec<i32>,
    /// Absolute maximum indexes.
    pub max_idx: Vec<i32>,
}

fn window_min_max(window: &[Float], offset: usize) -> (Float, Float, i32, i32) {
    let mut min_value = window[0];
    let mut max_value = window[0];
    let mut min_idx = offset;
    let mut max_idx = offset;

    for (local_idx, value) in window.iter().copied().enumerate().skip(1) {
        let absolute_idx = offset + local_idx;
        if value < min_value {
            min_value = value;
            min_idx = absolute_idx;
        }
        if value > max_value {
            max_value = value;
            max_idx = absolute_idx;
        }
    }

    (min_value, max_value, min_idx as i32, max_idx as i32)
}

fn validate_window(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
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

/// TA-Lib-style rolling minimum index.
#[allow(non_snake_case)]
pub fn MININDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MININDEX", out_integer.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, min_idx, _) =
            window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_integer[output_idx] = min_idx;
    }
    Ok(OutputRange::new(lookback, count))
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
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, _, max_idx) =
            window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_integer[output_idx] = max_idx;
    }
    Ok(OutputRange::new(lookback, count))
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
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (min_value, max_value, _, _) =
            window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_min[output_idx] = min_value;
        out_max[output_idx] = max_value;
    }
    Ok(OutputRange::new(lookback, count))
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
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, min_idx, max_idx) =
            window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_min_idx[output_idx] = min_idx;
        out_max_idx[output_idx] = max_idx;
    }
    Ok(OutputRange::new(lookback, count))
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
            type Input = Float;
            type Output = i32;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute(
                &self,
                inputs: &[Self::Input],
                outputs: &mut [Self::Output],
            ) -> Result<OutputRange> {
                $name(inputs, self.period, outputs)
            }

            fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                $vec_name(inputs, self.period)
            }

            fn next(&mut self, input: Float) -> i32 {
                self.buffer[self.index] = input;
                self.indexes[self.index] = self.seen;
                self.seen = self.seen.saturating_add(1);
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;

                if self.count < self.period {
                    return 0;
                }

                let is_better = $is_better;
                select_stream_index(
                    &self.buffer[..self.count],
                    &self.indexes[..self.count],
                    is_better,
                )
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
#[derive(Debug, Clone, Copy)]
pub struct MINMAX {
    period: usize,
}

impl MINMAX {
    /// Creates a MINMAX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
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

/// MINMAXINDEX struct surface.
#[derive(Debug, Clone, Copy)]
pub struct MINMAXINDEX {
    period: usize,
}

impl MINMAXINDEX {
    /// Creates a MINMAXINDEX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
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
