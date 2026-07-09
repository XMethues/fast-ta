//! Triangular Moving Average (TRIMA).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn trima_weight(index: usize, timeperiod: usize) -> usize {
    if timeperiod % 2 == 1 {
        let center = timeperiod / 2;
        if index <= center {
            index + 1
        } else {
            timeperiod - index
        }
    } else {
        let half = timeperiod / 2;
        if index < half {
            index + 1
        } else {
            timeperiod - index
        }
    }
}

#[inline]
fn trima_denominator(timeperiod: usize) -> Float {
    if timeperiod % 2 == 1 {
        let value = timeperiod / 2 + 1;
        (value * value) as Float
    } else {
        let half = timeperiod / 2;
        (half * (half + 1)) as Float
    }
}

fn trima_window(window: &[Float]) -> Float {
    let weighted_sum = window
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| trima_weight(idx, window.len()) as Float * value)
        .sum::<Float>();
    weighted_sum / trima_denominator(window.len())
}

/// TA-Lib-style Triangular Moving Average batch function.
#[allow(non_snake_case)]
pub fn TRIMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("TRIMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        out_real[output_idx] = trima_window(&real[output_idx..output_idx + timeperiod]);
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes TRIMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn TRIMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = TRIMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Triangular Moving Average indicator.
#[derive(Debug, Clone)]
pub struct TRIMA {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl TRIMA {
    /// Creates a new TRIMA indicator.
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
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact TRIMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        TRIMA(real, self.period, out_real)
    }

    /// Computes full-length padded TRIMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        TRIMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TRIMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRIMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TRIMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for TRIMA {
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

        let weighted_sum = (0..self.period)
            .map(|offset| {
                let ordered_idx = (self.index + offset) % self.period;
                trima_weight(offset, self.period) as Float * self.buffer[ordered_idx]
            })
            .sum::<Float>();
        Ok(Some(weighted_sum / trima_denominator(self.period)))
    }
}

impl Resettable for TRIMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}
