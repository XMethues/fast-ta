//! Rolling-window Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

fn rolling_apply<F>(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
    mut aggregate: F,
) -> Result<OutputRange>
where
    F: FnMut(&[Float]) -> Float,
{
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        out_real[output_idx] = aggregate(&real[output_idx..output_idx + timeperiod]);
    }

    Ok(OutputRange::new(lookback, count))
}

fn sum_window(window: &[Float]) -> Float {
    window.iter().copied().sum()
}

fn min_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::min)
}

fn max_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::max)
}

/// TA-Lib-style rolling sum.
#[allow(non_snake_case)]
pub fn SUM(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_apply("SUM", real, timeperiod, out_real, sum_window)
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
    rolling_apply("MIN", real, timeperiod, out_real, min_window)
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
    rolling_apply("MAX", real, timeperiod, out_real, max_window)
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
    ($name:ident, $vec_name:ident, $aggregate:ident) => {
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
            type Input = Float;
            type Output = Float;

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

            fn next(&mut self, input: Float) -> Float {
                self.buffer[self.index] = input;
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;

                if self.count < self.period {
                    return Float::NAN;
                }

                $aggregate(&self.buffer)
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

define_rolling_struct!(SUM, SUM_vec, sum_window);
define_rolling_struct!(MIN, MIN_vec, min_window);
define_rolling_struct!(MAX, MAX_vec, max_window);
