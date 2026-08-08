//! Rolling midpoint Overlap Studies: MIDPOINT and MIDPRICE.

use crate::common::validate_finite_value;
use crate::{
    validate_all_same_len, validate_finite_slice, validate_finite_slices, validate_input_len,
    validate_output_len, CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner,
    Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec, vec::Vec};

const MAX_PERIOD: usize = 100_000;

#[inline]
fn midpoint_lookback(timeperiod: usize) -> Result<usize> {
    if !(2..=MAX_PERIOD).contains(&timeperiod) {
        return Err(TalibError::invalid_period(
            timeperiod,
            format!("timeperiod must be in 2..={MAX_PERIOD}"),
        ));
    }
    Ok(timeperiod - 1)
}

#[derive(Debug, Clone)]
struct ExtremaScratch {
    min: Vec<usize>,
    max: Vec<usize>,
}

impl ExtremaScratch {
    fn with_capacity(max_input_len: usize) -> Self {
        Self {
            min: Vec::with_capacity(max_input_len),
            max: Vec::with_capacity(max_input_len),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn midpoint_kernel(
    low: &[Float],
    high: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    output: &mut [Float],
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
    for idx in 0..low.len() {
        if idx >= timeperiod {
            let expired_through = idx - timeperiod;
            while min_head < min_queue.len() && min_queue[min_head] <= expired_through {
                min_head += 1;
            }
            while max_head < max_queue.len() && max_queue[max_head] <= expired_through {
                max_head += 1;
            }
        }

        while min_queue.len() > min_head {
            let last = min_queue[min_queue.len() - 1];
            if low[idx] < low[last] {
                min_queue.pop();
            } else {
                break;
            }
        }
        while max_queue.len() > max_head {
            let last = max_queue[max_queue.len() - 1];
            if high[idx] > high[last] {
                max_queue.pop();
            } else {
                break;
            }
        }
        min_queue.push(idx);
        max_queue.push(idx);

        if idx >= lookback {
            let output_idx = idx - lookback;
            output[output_idx] =
                (low[min_queue[min_head]] + high[max_queue[max_head]]) / 2.0 as Float;
        }
    }

    OutputRange::new(lookback, count)
}

fn validate_midpoint(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = midpoint_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

/// TA-Lib-defined rolling midpoint of the highest and lowest real observations.
#[allow(non_snake_case)]
pub fn MIDPOINT(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let (lookback, count) = validate_midpoint(real, timeperiod)?;
    validate_output_len("MIDPOINT", out_real.len(), count)?;
    let mut scratch = ExtremaScratch::with_capacity(real.len());
    Ok(midpoint_kernel(
        real,
        real,
        timeperiod,
        lookback,
        count,
        out_real,
        &mut scratch.min,
        &mut scratch.max,
    ))
}

/// Immutable rolling MIDPOINT Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MIDPOINTConfig {
    period: usize,
}

impl MIDPOINTConfig {
    /// Creates a MIDPOINT configuration for a Period in `2..=100_000`.
    pub fn new(timeperiod: usize) -> Result<Self> {
        midpoint_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for MIDPOINTConfig {}

impl IndicatorConfig for MIDPOINTConfig {
    type Input<'a> = &'a [Float];
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MIDPOINTBatchRunner;
    type Stream = MIDPOINTStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_midpoint(input, self.period)?;
        let mut values = vec![0.0 as Float; count];
        let mut scratch = ExtremaScratch::with_capacity(input.len());
        let range = midpoint_kernel(
            input,
            input,
            self.period,
            lookback,
            count,
            &mut values,
            &mut scratch.min,
            &mut scratch.max,
        );
        CompactOutput::new(input.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MIDPOINT(input, self.period, output)
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        let scratch = ExtremaScratch::with_capacity(max_input_len);
        Ok(MIDPOINTBatchRunner {
            config: *self,
            max_input_len,
            min_scratch: scratch.min,
            max_scratch: scratch.max,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        MIDPOINTStream::new(self.period)
    }
}

/// Reusable Prepared Batch Runner for MIDPOINT.
#[derive(Debug)]
pub struct MIDPOINTBatchRunner {
    config: MIDPOINTConfig,
    max_input_len: usize,
    min_scratch: Vec<usize>,
    max_scratch: Vec<usize>,
}

impl crate::traits::sealed::Sealed for MIDPOINTBatchRunner {}

impl PreparedBatchRunner<MIDPOINTConfig> for MIDPOINTBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MIDPOINTConfig as IndicatorConfig>::Input<'a>,
        output: <MIDPOINTConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MIDPOINTConfig: 'a,
    {
        if input.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.len(),
            ));
        }
        let (lookback, count) = validate_midpoint(input, self.config.period)?;
        validate_output_len("MIDPOINT", output.len(), count)?;
        Ok(midpoint_kernel(
            input,
            input,
            self.config.period,
            lookback,
            count,
            output,
            &mut self.min_scratch,
            &mut self.max_scratch,
        ))
    }
}

/// Independent Streaming Computation state for MIDPOINT.
#[derive(Debug, Clone)]
pub struct MIDPOINTStream {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl MIDPOINTStream {
    fn new(period: usize) -> Result<Self> {
        midpoint_lookback(period)?;
        Ok(Self {
            period,
            buffer: vec![0.0 as Float; period],
            index: 0,
            count: 0,
        })
    }
}

impl crate::traits::sealed::Sealed for MIDPOINTStream {}

impl StreamingComputation<MIDPOINTConfig> for MIDPOINTStream {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("input", 0, input)?;
        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;
        if self.count < self.period {
            return Ok(None);
        }

        let values = &self.buffer[..self.count];
        let low = values.iter().copied().fold(values[0], Float::min);
        let high = values.iter().copied().fold(values[0], Float::max);
        Ok(Some((low + high) / 2.0 as Float))
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}

/// Borrowed aligned high/low Observation Series for MIDPRICE.
#[derive(Debug, Clone, Copy)]
pub struct MIDPRICEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
}

/// One aligned high/low MIDPRICE tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MIDPRICETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
}

fn validate_midprice(input: MIDPRICEInput<'_>, timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = midpoint_lookback(timeperiod)?;
    let len = validate_all_same_len(&[("high", input.high.len()), ("low", input.low.len())])?;
    validate_finite_slices(&[("high", input.high), ("low", input.low)])?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count))
}

/// TA-Lib-defined rolling midpoint of the highest high and lowest low.
#[allow(non_snake_case)]
pub fn MIDPRICE(
    high: &[Float],
    low: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = MIDPRICEInput { high, low };
    let (lookback, count) = validate_midprice(input, timeperiod)?;
    validate_output_len("MIDPRICE", out_real.len(), count)?;
    let mut scratch = ExtremaScratch::with_capacity(high.len());
    Ok(midpoint_kernel(
        low,
        high,
        timeperiod,
        lookback,
        count,
        out_real,
        &mut scratch.min,
        &mut scratch.max,
    ))
}

/// Immutable rolling MIDPRICE Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MIDPRICEConfig {
    period: usize,
}

impl MIDPRICEConfig {
    /// Creates a MIDPRICE configuration for a Period in `2..=100_000`.
    pub fn new(timeperiod: usize) -> Result<Self> {
        midpoint_lookback(timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured Period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }
}

impl crate::traits::sealed::Sealed for MIDPRICEConfig {}

impl IndicatorConfig for MIDPRICEConfig {
    type Input<'a> = MIDPRICEInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MIDPRICEBatchRunner;
    type Stream = MIDPRICEStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count) = validate_midprice(input, self.period)?;
        let mut values = vec![0.0 as Float; count];
        let mut scratch = ExtremaScratch::with_capacity(input.high.len());
        let range = midpoint_kernel(
            input.low,
            input.high,
            self.period,
            lookback,
            count,
            &mut values,
            &mut scratch.min,
            &mut scratch.max,
        );
        CompactOutput::new(input.high.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let (lookback, count) = validate_midprice(input, self.period)?;
        validate_output_len("MIDPRICE", output.len(), count)?;
        let mut scratch = ExtremaScratch::with_capacity(input.high.len());
        Ok(midpoint_kernel(
            input.low,
            input.high,
            self.period,
            lookback,
            count,
            output,
            &mut scratch.min,
            &mut scratch.max,
        ))
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        let scratch = ExtremaScratch::with_capacity(max_input_len);
        Ok(MIDPRICEBatchRunner {
            config: *self,
            max_input_len,
            min_scratch: scratch.min,
            max_scratch: scratch.max,
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        MIDPRICEStream::new(self.period)
    }
}

/// Reusable Prepared Batch Runner for MIDPRICE.
#[derive(Debug)]
pub struct MIDPRICEBatchRunner {
    config: MIDPRICEConfig,
    max_input_len: usize,
    min_scratch: Vec<usize>,
    max_scratch: Vec<usize>,
}

impl crate::traits::sealed::Sealed for MIDPRICEBatchRunner {}

impl PreparedBatchRunner<MIDPRICEConfig> for MIDPRICEBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MIDPRICEConfig as IndicatorConfig>::Input<'a>,
        output: <MIDPRICEConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MIDPRICEConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        let (lookback, count) = validate_midprice(input, self.config.period)?;
        validate_output_len("MIDPRICE", output.len(), count)?;
        Ok(midpoint_kernel(
            input.low,
            input.high,
            self.config.period,
            lookback,
            count,
            output,
            &mut self.min_scratch,
            &mut self.max_scratch,
        ))
    }
}

/// Independent Streaming Computation state for MIDPRICE.
#[derive(Debug, Clone)]
pub struct MIDPRICEStream {
    period: usize,
    highs: Vec<Float>,
    lows: Vec<Float>,
    index: usize,
    count: usize,
}

impl MIDPRICEStream {
    fn new(period: usize) -> Result<Self> {
        midpoint_lookback(period)?;
        Ok(Self {
            period,
            highs: vec![0.0 as Float; period],
            lows: vec![0.0 as Float; period],
            index: 0,
            count: 0,
        })
    }
}

impl crate::traits::sealed::Sealed for MIDPRICEStream {}

impl StreamingComputation<MIDPRICEConfig> for MIDPRICEStream {
    type Tick = MIDPRICETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_value("high", 0, input.high)?;
        validate_finite_value("low", 0, input.low)?;

        self.highs[self.index] = input.high;
        self.lows[self.index] = input.low;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;
        if self.count < self.period {
            return Ok(None);
        }

        let highs = &self.highs[..self.count];
        let lows = &self.lows[..self.count];
        let high = highs.iter().copied().fold(highs[0], Float::max);
        let low = lows.iter().copied().fold(lows[0], Float::min);
        Ok(Some((high + low) / 2.0 as Float))
    }

    fn reset(&mut self) {
        self.highs.fill(0.0 as Float);
        self.lows.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
    }
}
