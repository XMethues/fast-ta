//! Chaikin Accumulation/Distribution Oscillator (ADOSC).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, CompactOutput, Float, Indicator, IndicatorConfig,
    OutputRange, PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

/// Borrowed high/low/close/volume inputs for [`ADOSC`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct ADOSCInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One high/low/close/volume tick for [`ADOSC`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ADOSCTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

#[inline]
fn ema_multiplier(period: usize) -> Float {
    2.0 as Float / (period as Float + 1.0 as Float)
}

#[derive(Debug, Clone)]
struct EmaState {
    period: usize,
    multiplier: Float,
    count: usize,
    sum: Float,
    value: Float,
}

impl EmaState {
    fn new(period: usize) -> Self {
        Self {
            period,
            multiplier: ema_multiplier(period),
            count: 0,
            sum: 0.0 as Float,
            value: 0.0 as Float,
        }
    }

    fn next(&mut self, input: Float) -> Option<Float> {
        if self.count < self.period {
            self.sum += input;
            self.count += 1;

            if self.count < self.period {
                return None;
            }

            self.value = self.sum / self.period as Float;
            return Some(self.value);
        }

        self.value = (input - self.value) * self.multiplier + self.value;
        Some(self.value)
    }

    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}

fn adosc_lookback(fastperiod: usize, slowperiod: usize) -> Result<usize> {
    validate_period("fastperiod", fastperiod)?;
    validate_period("slowperiod", slowperiod)?;
    if fastperiod >= slowperiod {
        return Err(TalibError::invalid_parameter(
            "fastperiod".to_string(),
            format!("{fastperiod} (slowperiod={slowperiod})"),
            "fastperiod must be less than slowperiod".to_string(),
        ));
    }
    Ok(slowperiod - 1)
}
fn validate_adosc_input(
    input: ADOSCInput<'_>,
    fastperiod: usize,
    slowperiod: usize,
) -> Result<(usize, usize, usize)> {
    let lookback = adosc_lookback(fastperiod, slowperiod)?;
    let len = super::ad::validate_hlcv(input.high, input.low, input.close, input.volume)?;
    let count = validate_input_len(len, lookback)?;
    Ok((lookback, count, len))
}

fn adosc_kernel(
    input: ADOSCInput<'_>,
    fastperiod: usize,
    slowperiod: usize,
    lookback: usize,
    count: usize,
    len: usize,
    output: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut cumulative = 0.0 as Float;
    let mut fast = EmaState::new(fastperiod);
    let mut slow = EmaState::new(slowperiod);
    let mut output_idx = 0usize;
    for idx in 0..len {
        cumulative += super::ad::money_flow_volume(
            input.high[idx],
            input.low[idx],
            input.close[idx],
            input.volume[idx],
        );
        let fast_value = fast.next(cumulative);
        let slow_value = slow.next(cumulative);
        if let (Some(fast_value), Some(slow_value)) = (fast_value, slow_value) {
            output[output_idx] = fast_value - slow_value;
            output_idx += 1;
        }
    }
    OutputRange::new(lookback, count)
}

/// Chaikin Accumulation/Distribution Oscillator batch function.
#[allow(non_snake_case)]
pub fn ADOSC(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
    fastperiod: usize,
    slowperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let input = ADOSCInput {
        high,
        low,
        close,
        volume,
    };
    let (lookback, count, len) = validate_adosc_input(input, fastperiod, slowperiod)?;
    validate_output_len("ADOSC", out_real.len(), count)?;
    Ok(adosc_kernel(
        input, fastperiod, slowperiod, lookback, count, len, out_real,
    ))
}

/// Computes ADOSC into a full-length padded vector.
#[allow(non_snake_case)]
pub fn ADOSC_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
    fastperiod: usize,
    slowperiod: usize,
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = ADOSC(
        high,
        low,
        close,
        volume,
        fastperiod,
        slowperiod,
        &mut compact,
    )?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}
/// Immutable Chaikin Accumulation/Distribution Oscillator Indicator Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ADOSCConfig {
    fastperiod: usize,
    slowperiod: usize,
}

impl ADOSCConfig {
    /// Creates an ADOSC configuration.
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self> {
        adosc_lookback(fastperiod, slowperiod)?;
        Ok(Self {
            fastperiod,
            slowperiod,
        })
    }

    /// Returns the fast Period.
    pub const fn fastperiod(&self) -> usize {
        self.fastperiod
    }

    /// Returns the slow Period.
    pub const fn slowperiod(&self) -> usize {
        self.slowperiod
    }
}

impl crate::traits::sealed::Sealed for ADOSCConfig {}

impl IndicatorConfig for ADOSCConfig {
    type Input<'a> = ADOSCInput<'a>;
    type Output = Vec<Float>;
    type OutputMut<'a> = &'a mut [Float];
    type BatchRunner = ADOSCBatchRunner;
    type Stream = ADOSCStream;

    #[inline]
    fn lookback(&self) -> usize {
        self.slowperiod - 1
    }

    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
        let (lookback, count, len) = validate_adosc_input(input, self.fastperiod, self.slowperiod)?;
        let mut values = Vec::with_capacity(count);
        values.resize(count, 0.0 as Float);
        let range = adosc_kernel(
            input,
            self.fastperiod,
            self.slowperiod,
            lookback,
            count,
            len,
            &mut values,
        );
        CompactOutput::new(len, range, values)
    }

    #[inline]
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ADOSC(
            input.high,
            input.low,
            input.close,
            input.volume,
            self.fastperiod,
            self.slowperiod,
            output,
        )
    }

    #[inline]
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
        Ok(ADOSCBatchRunner {
            config: *self,
            max_input_len,
        })
    }

    #[inline]
    fn stream(&self) -> Result<Self::Stream> {
        Ok(ADOSCStream {
            cumulative: 0.0 as Float,
            fast: EmaState::new(self.fastperiod),
            slow: EmaState::new(self.slowperiod),
        })
    }
}

/// Reusable Prepared Batch Runner for ADOSC.
#[derive(Debug, Clone)]
pub struct ADOSCBatchRunner {
    config: ADOSCConfig,
    max_input_len: usize,
}

impl crate::traits::sealed::Sealed for ADOSCBatchRunner {}

impl PreparedBatchRunner<ADOSCConfig> for ADOSCBatchRunner {
    #[inline]
    fn max_input_len(&self) -> usize {
        self.max_input_len
    }

    #[inline]
    fn compute_into<'a>(
        &mut self,
        input: <ADOSCConfig as IndicatorConfig>::Input<'a>,
        output: <ADOSCConfig as IndicatorConfig>::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        ADOSCConfig: 'a,
    {
        if input.high.len() > self.max_input_len {
            return Err(TalibError::prepared_capacity_exceeded(
                self.max_input_len,
                input.high.len(),
            ));
        }
        IndicatorConfig::compute_into(&self.config, input, output)
    }
}

/// Independent Streaming Computation state for ADOSC.
#[derive(Debug, Clone)]
pub struct ADOSCStream {
    cumulative: Float,
    fast: EmaState,
    slow: EmaState,
}

impl crate::traits::sealed::Sealed for ADOSCStream {}

impl StreamingComputation<ADOSCConfig> for ADOSCStream {
    type Tick = ADOSCTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
            ("volume", &[input.volume]),
        ])?;
        self.cumulative +=
            super::ad::money_flow_volume(input.high, input.low, input.close, input.volume);
        let fast = self.fast.next(self.cumulative);
        let slow = self.slow.next(self.cumulative);
        Ok(match (fast, slow) {
            (Some(fast), Some(slow)) => Some(fast - slow),
            _ => None,
        })
    }

    #[inline]
    fn reset(&mut self) {
        self.cumulative = 0.0 as Float;
        self.fast.reset();
        self.slow.reset();
    }
}

/// Chaikin Accumulation/Distribution Oscillator compatibility adapter.
#[derive(Debug, Clone)]
pub struct ADOSC {
    config: ADOSCConfig,
    stream: ADOSCStream,
}

impl ADOSC {
    /// Creates a new ADOSC indicator.
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self> {
        let config = ADOSCConfig::new(fastperiod, slowperiod)?;
        let stream = IndicatorConfig::stream(&config)?;
        Ok(Self { config, stream })
    }

    /// Returns the fast period.
    pub const fn fastperiod(&self) -> usize {
        self.config.fastperiod()
    }

    /// Returns the slow period.
    pub const fn slowperiod(&self) -> usize {
        self.config.slowperiod()
    }

    /// Computes compact ADOSC outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        ADOSC(
            high,
            low,
            close,
            volume,
            self.config.fastperiod,
            self.config.slowperiod,
            out_real,
        )
    }

    /// Computes full-length padded ADOSC outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
    ) -> Result<Vec<Float>> {
        ADOSC_vec(
            high,
            low,
            close,
            volume,
            self.config.fastperiod,
            self.config.slowperiod,
        )
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: ADOSCTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for ADOSC {
    type Input<'a> = ADOSCInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        IndicatorConfig::lookback(&self.config)
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ADOSC(
            input.high,
            input.low,
            input.close,
            input.volume,
            self.config.fastperiod,
            self.config.slowperiod,
            output,
        )
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        ADOSC_vec(
            input.high,
            input.low,
            input.close,
            input.volume,
            self.config.fastperiod,
            self.config.slowperiod,
        )
    }
}

impl StreamingIndicator for ADOSC {
    type Tick = ADOSCTick;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        StreamingComputation::<ADOSCConfig>::next(&mut self.stream, input)
    }
}

impl Resettable for ADOSC {
    #[inline]
    fn reset(&mut self) {
        StreamingComputation::<ADOSCConfig>::reset(&mut self.stream);
    }
}
