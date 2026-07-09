//! Rolling-window Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

fn validate_rolling_window(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    output_len: usize,
) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, output_len, count)?;
    Ok((lookback, count))
}

fn min_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::min)
}

fn max_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::max)
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

    let mut deque = Vec::new();
    deque.reserve(real.len());
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
    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut window_sum: Float = real[..timeperiod].iter().copied().sum();
    out_real[0] = window_sum;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = window_sum;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes rolling sum into a full-length vector.
#[allow(non_snake_case)]
pub fn SUM_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = SUM(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// TA-Lib-style rolling minimum.
#[allow(non_snake_case)]
pub fn MIN(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_extreme("MIN", real, timeperiod, out_real, |candidate, current| {
        candidate < current
    })
}

/// Computes rolling minimum into a full-length vector.
#[allow(non_snake_case)]
pub fn MIN_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MIN(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// TA-Lib-style rolling maximum.
#[allow(non_snake_case)]
pub fn MAX(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_extreme("MAX", real, timeperiod, out_real, |candidate, current| {
        candidate > current
    })
}

/// Computes rolling maximum into a full-length vector.
#[allow(non_snake_case)]
pub fn MAX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MAX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

macro_rules! define_rolling_struct {
    ($name:ident, $vec_name:ident, $aggregate:expr) => {
        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            buffer: Vec<Float>,
            index: usize,
            count: usize,
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
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

            /// Computes compact outputs.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, out_real)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period)
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [Float];
            type OutputOwned = Vec<Float>;

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

                Ok(Some($aggregate(&self.buffer)))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                for value in &mut self.buffer {
                    *value = 0.0 as Float;
                }
                self.index = 0;
                self.count = 0;
            }
        }
    };
}

define_rolling_struct!(SUM, SUM_vec, |window: &[Float]| window
    .iter()
    .copied()
    .sum());
define_rolling_struct!(MIN, MIN_vec, min_window);
define_rolling_struct!(MAX, MAX_vec, max_window);
