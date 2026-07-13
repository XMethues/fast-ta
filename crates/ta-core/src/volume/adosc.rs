//! Chaikin Accumulation/Distribution Oscillator (ADOSC).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
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
    let lookback = adosc_lookback(fastperiod, slowperiod)?;
    let len = super::ad::validate_hlcv(high, low, close, volume)?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("ADOSC", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut cumulative = 0.0 as Float;
    let mut fast = EmaState::new(fastperiod);
    let mut slow = EmaState::new(slowperiod);
    let mut output_idx = 0usize;

    for idx in 0..len {
        cumulative += super::ad::money_flow_volume(high[idx], low[idx], close[idx], volume[idx]);
        let fast_value = fast.next(cumulative);
        let slow_value = slow.next(cumulative);
        if let (Some(fast_value), Some(slow_value)) = (fast_value, slow_value) {
            out_real[output_idx] = fast_value - slow_value;
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
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

/// Chaikin Accumulation/Distribution Oscillator indicator.
#[derive(Debug, Clone)]
pub struct ADOSC {
    fastperiod: usize,
    slowperiod: usize,
    cumulative: Float,
    fast: EmaState,
    slow: EmaState,
}

impl ADOSC {
    /// Creates a new ADOSC indicator.
    pub fn new(fastperiod: usize, slowperiod: usize) -> Result<Self> {
        adosc_lookback(fastperiod, slowperiod)?;
        Ok(Self {
            fastperiod,
            slowperiod,
            cumulative: 0.0 as Float,
            fast: EmaState::new(fastperiod),
            slow: EmaState::new(slowperiod),
        })
    }

    /// Returns the fast period.
    pub const fn fastperiod(&self) -> usize {
        self.fastperiod
    }

    /// Returns the slow period.
    pub const fn slowperiod(&self) -> usize {
        self.slowperiod
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
            self.fastperiod,
            self.slowperiod,
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
        ADOSC_vec(high, low, close, volume, self.fastperiod, self.slowperiod)
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
        self.slowperiod - 1
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
            self.fastperiod,
            self.slowperiod,
            output,
        )
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        ADOSC_vec(
            input.high,
            input.low,
            input.close,
            input.volume,
            self.fastperiod,
            self.slowperiod,
        )
    }
}

impl StreamingIndicator for ADOSC {
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
}

impl Resettable for ADOSC {
    fn reset(&mut self) {
        self.cumulative = 0.0 as Float;
        self.fast.reset();
        self.slow.reset();
    }
}
