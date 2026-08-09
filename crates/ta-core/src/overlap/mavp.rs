//! Moving Average with Variable Period (MAVP).
//!
//! Each Observation is paired with an aligned integer Period selection. The
//! selection is clamped to immutable configured bounds and evaluated with one
//! qualified [`PeriodMAType`](super::PeriodMAType).

use super::ma::{ma_kernel, ma_lookback, MAConfig, MAStream, PeriodMAType};
use crate::{
    validate_finite_slice, validate_input_len, validate_output_len, validate_same_len,
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
    StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed aligned input series for MAVP Batch Computation.
#[derive(Debug, Clone, Copy)]
pub struct MAVPInput<'a> {
    /// Observation Series.
    pub real: &'a [Float],
    /// Aligned integer Period Selection Series.
    pub periods: &'a [usize],
}

/// One aligned MAVP streaming Tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MAVPTick {
    /// Observation value.
    pub real: Float,
    /// Integer Period selection, clamped to the configured bounds.
    pub period: usize,
}

#[inline]
fn clamp_period(period: usize, minimum: usize, maximum: usize) -> usize {
    period.clamp(minimum, maximum)
}

#[inline]
fn requires_fallible_preflight(ma_type: PeriodMAType) -> bool {
    match ma_type {
        PeriodMAType::DEMA | PeriodMAType::TEMA | PeriodMAType::T3 => true,
        PeriodMAType::SMA
        | PeriodMAType::EMA
        | PeriodMAType::WMA
        | PeriodMAType::TRIMA
        | PeriodMAType::KAMA => false,
    }
}

fn validate_mavp_config(
    minimum_period: usize,
    maximum_period: usize,
    ma_type: PeriodMAType,
) -> Result<usize> {
    if minimum_period > maximum_period {
        return Err(TalibError::invalid_input(
            "minimum_period must not exceed maximum_period",
        ));
    }
    ma_lookback(minimum_period, ma_type)?;
    ma_lookback(maximum_period, ma_type)
}

fn validate_mavp_input(input: MAVPInput<'_>, lookback: usize) -> Result<usize> {
    validate_same_len("real", input.real.len(), "periods", input.periods.len())?;
    validate_finite_slice("real", input.real)?;
    validate_input_len(input.real.len(), lookback)
}

fn collect_used_periods(
    periods: &[usize],
    lookback: usize,
    minimum: usize,
    maximum: usize,
    used_periods: &mut Vec<usize>,
) {
    used_periods.clear();
    for &selection in &periods[lookback..] {
        let period = clamp_period(selection, minimum, maximum);
        if !used_periods.contains(&period) {
            used_periods.push(period);
        }
    }
}

fn mavp_kernel(
    input: MAVPInput<'_>,
    config: MAVPConfig,
    lookback: usize,
    count: usize,
    output: &mut [Float],
    ma_scratch: &mut [Float],
    used_periods: &mut Vec<usize>,
) -> Result<OutputRange> {
    if count == 0 {
        used_periods.clear();
        return Ok(OutputRange::empty());
    }

    collect_used_periods(
        input.periods,
        lookback,
        config.minimum_period,
        config.maximum_period,
        used_periods,
    );
    if requires_fallible_preflight(config.ma_type) {
        for &period in used_periods.iter() {
            let period_lookback = ma_lookback(period, config.ma_type)?;
            let start_offset = lookback - period_lookback;
            ma_kernel(
                &input.real[start_offset..],
                period,
                config.ma_type,
                period_lookback,
                count,
                &mut ma_scratch[..count],
            )?;
        }
    }

    for &period in used_periods.iter() {
        let period_lookback = ma_lookback(period, config.ma_type)?;
        let start_offset = lookback - period_lookback;
        ma_kernel(
            &input.real[start_offset..],
            period,
            config.ma_type,
            period_lookback,
            count,
            &mut ma_scratch[..count],
        )?;

        for source_idx in lookback..input.real.len() {
            if clamp_period(
                input.periods[source_idx],
                config.minimum_period,
                config.maximum_period,
            ) == period
            {
                output[source_idx - lookback] = ma_scratch[source_idx - lookback];
            }
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// Variable-period Period-based Moving Average Batch Computation.
///
/// `periods` must align exactly with `real`. Each integer selection is clamped
/// to `[minimum_period, maximum_period]` without floating-point conversion.
/// Each selected definition is seeded so its first result aligns with the
/// maximum-Period MAVP Lookback, matching recursive MAVP restart semantics.
/// This caller-owned convenience function allocates two algorithm scratch
/// buffers; use [`MAVPBatchRunner`] to reserve and reuse them.
#[allow(non_snake_case)]
pub fn MAVP(
    real: &[Float],
    periods: &[usize],
    minimum_period: usize,
    maximum_period: usize,
    ma_type: PeriodMAType,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let config = MAVPConfig::new(minimum_period, maximum_period, ma_type)?;
    let input = MAVPInput { real, periods };
    let lookback = config.lookback;
    let count = validate_mavp_input(input, lookback)?;
    validate_output_len("MAVP", out_real.len(), count)?;

    let mut ma_scratch = Vec::new();
    ma_scratch.resize(count, 0.0 as Float);
    let mut used_periods = Vec::with_capacity(count);
    mavp_kernel(
        input,
        config,
        lookback,
        count,
        out_real,
        &mut ma_scratch,
        &mut used_periods,
    )
}

/// Immutable MAVP Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MAVPConfig {
    minimum_period: usize,
    maximum_period: usize,
    ma_type: PeriodMAType,
    lookback: usize,
}

impl MAVPConfig {
    /// Creates a bounded variable-period configuration.
    pub fn new(
        minimum_period: usize,
        maximum_period: usize,
        ma_type: PeriodMAType,
    ) -> Result<Self> {
        let lookback = validate_mavp_config(minimum_period, maximum_period, ma_type)?;
        Ok(Self {
            minimum_period,
            maximum_period,
            ma_type,
            lookback,
        })
    }

    /// Returns the immutable minimum Period bound.
    #[inline]
    pub const fn minimum_period(&self) -> usize {
        self.minimum_period
    }

    /// Returns the immutable maximum Period bound.
    #[inline]
    pub const fn maximum_period(&self) -> usize {
        self.maximum_period
    }

    /// Returns the selected Period-based Moving Average definition.
    #[inline]
    pub const fn ma_type(&self) -> PeriodMAType {
        self.ma_type
    }
}

impl crate::traits::sealed::Sealed for MAVPConfig {}

impl IndicatorConfig for MAVPConfig {
    type Input<'a> = MAVPInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = MAVPBatchRunner;
    type Stream = MAVPStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let count = validate_mavp_input(input, self.lookback)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let mut ma_scratch = Vec::new();
        ma_scratch.resize(count, 0.0 as Float);
        let mut used_periods = Vec::with_capacity(count);
        let range = mavp_kernel(
            input,
            *self,
            self.lookback,
            count,
            &mut values,
            &mut ma_scratch,
            &mut used_periods,
        )?;
        CompactOutput::new(input.real.len(), range, values)
    }

    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        let count = validate_mavp_input(input, self.lookback)?;
        validate_output_len("MAVP", output.len(), count)?;
        let mut ma_scratch = Vec::new();
        ma_scratch.resize(count, 0.0 as Float);
        let mut used_periods = Vec::with_capacity(count);
        mavp_kernel(
            input,
            *self,
            self.lookback,
            count,
            output,
            &mut ma_scratch,
            &mut used_periods,
        )
    }

    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        let scratch_capacity = max_input_len.saturating_sub(self.lookback);
        let mut ma_scratch = Vec::new();
        ma_scratch.resize(scratch_capacity, 0.0 as Float);
        Ok(MAVPBatchRunner {
            config: *self,
            max_input_len,
            ma_scratch,
            used_periods: Vec::with_capacity(scratch_capacity),
        })
    }

    fn stream(&self) -> Result<Self::Stream> {
        MAVPStream::new(*self)
    }
}

/// Prepared Batch Runner with all MAVP algorithm scratch reserved at creation.
#[derive(Debug, Clone)]
pub struct MAVPBatchRunner {
    config: MAVPConfig,
    max_input_len: usize,
    ma_scratch: Vec<Float>,
    used_periods: Vec<usize>,
}

impl crate::traits::sealed::Sealed for MAVPBatchRunner {}

impl PreparedBatchRunner<MAVPConfig> for MAVPBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    fn compute_into<'a>(
        &mut self,
        input: <MAVPConfig as IndicatorConfig>::Input<'a>,
        output: <MAVPConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        MAVPConfig: 'a,
    {
        let actual_input_len = input.real.len().max(input.periods.len());
        if actual_input_len > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                actual_input_len,
            ));
        }
        let count = validate_mavp_input(input, self.config.lookback)?;
        validate_output_len("MAVP", output.len(), count)?;
        mavp_kernel(
            input,
            self.config,
            self.config.lookback,
            count,
            output,
            &mut self.ma_scratch,
            &mut self.used_periods,
        )
    }
}

/// Independent MAVP Streaming Computation.
///
/// One independent selected-MA stream is prepared for every Period in the
/// configured inclusive bounds. Creation owns that storage; ticks allocate no
/// heap memory.
#[derive(Debug, Clone)]
pub struct MAVPStream {
    config: MAVPConfig,
    streams: Vec<MAStream>,
    count: usize,
}

impl MAVPStream {
    fn new(config: MAVPConfig) -> Result<Self> {
        let period_count = config.maximum_period - config.minimum_period + 1;
        let mut streams = Vec::with_capacity(period_count);
        for period in config.minimum_period..=config.maximum_period {
            streams.push(IndicatorConfig::stream(&MAConfig::new(
                period,
                config.ma_type,
            )?)?);
        }
        Ok(Self {
            config,
            streams,
            count: 0,
        })
    }
}

impl crate::traits::sealed::Sealed for MAVPStream {}

impl StreamingComputation<MAVPConfig> for MAVPStream {
    type Tick = MAVPTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slice("real", &[input.real])?;
        let selected_period = clamp_period(
            input.period,
            self.config.minimum_period,
            self.config.maximum_period,
        );
        let selected_index = selected_period - self.config.minimum_period;
        let source_idx = self.count;
        let mut selected_value = None;
        for (index, stream) in self.streams.iter_mut().enumerate() {
            let period = self.config.minimum_period + index;
            let period_lookback = ma_lookback(period, self.config.ma_type)
                .expect("validated MAVP Period must retain a valid lookback");
            let start_offset = self.config.lookback - period_lookback;
            let value = if source_idx < start_offset {
                None
            } else {
                StreamingComputation::<MAConfig>::next(stream, input.real)?
            };
            if index == selected_index {
                selected_value = value;
            }
        }
        self.count += 1;
        if source_idx < self.config.lookback {
            Ok(None)
        } else {
            Ok(selected_value)
        }
    }

    fn reset(&mut self) {
        for stream in &mut self.streams {
            StreamingComputation::<MAConfig>::reset(stream);
        }
        self.count = 0;
    }
}
