//! Private rolling statistic engines.

use crate::{Float, Result, TalibError};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

pub(super) const DEFAULT_NBDEV: Float = 1.0 as Float;
pub(super) const TA_EPSILON: Float = 1e-14 as Float;
const MAX_PERIOD: usize = 100_000;

pub(super) fn statistic_lookback(timeperiod: usize, minimum: usize, extra: usize) -> Result<usize> {
    if !(minimum..=MAX_PERIOD).contains(&timeperiod) {
        return Err(TalibError::invalid_period(
            timeperiod,
            format!("timeperiod must be in {minimum}..={MAX_PERIOD}"),
        ));
    }

    Ok((timeperiod - 1) + extra)
}

pub(super) fn validate_nbdev(nbdev: Float) -> Result<()> {
    if !nbdev.is_finite() {
        return Err(TalibError::invalid_parameter(
            "nbdev".to_string(),
            format!("{nbdev}"),
            "finite number".to_string(),
        ));
    }
    Ok(())
}

#[inline]
pub(super) fn is_ta_zero(value: Float) -> bool {
    value > -TA_EPSILON && value < TA_EPSILON
}

#[derive(Debug, Clone)]
pub(super) struct RollingMoments {
    period: usize,
    trailing: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
    sum_sq: Float,
}

impl RollingMoments {
    pub(super) fn new(period: usize) -> Self {
        let mut trailing = Vec::new();
        trailing.resize(period.saturating_sub(1), 0.0 as Float);
        Self {
            period,
            trailing,
            index: 0,
            count: 0,
            sum: 0.0 as Float,
            sum_sq: 0.0 as Float,
        }
    }

    #[inline]
    pub(super) fn push(&mut self, input: Float) -> Option<Float> {
        self.sum += input;
        self.sum_sq += input * input;
        if self.count < self.period {
            self.count += 1;
        }

        if self.count < self.period {
            self.trailing[self.index] = input;
            self.index = (self.index + 1) % self.trailing.len();
            return None;
        }

        let mean = self.sum / self.period as Float;
        let variance = self.sum_sq / self.period as Float - mean * mean;
        self.remove_trailing(input);
        Some(variance)
    }

    #[inline]
    fn remove_trailing(&mut self, input: Float) {
        if self.trailing.is_empty() {
            self.sum -= input;
            self.sum_sq -= input * input;
            return;
        }

        let old = self.trailing[self.index];
        self.sum -= old;
        self.sum_sq -= old * old;
        self.trailing[self.index] = input;
        self.index = (self.index + 1) % self.trailing.len();
    }

    #[inline]
    pub(super) fn reset(&mut self) {
        self.trailing.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
        self.sum_sq = 0.0 as Float;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PairedSnapshot {
    n: Float,
    sum_x: Float,
    sum_y: Float,
    sum_x_sq: Float,
    sum_y_sq: Float,
    sum_xy: Float,
}

impl PairedSnapshot {
    pub(super) fn new(
        n: usize,
        sum_x: Float,
        sum_y: Float,
        sum_x_sq: Float,
        sum_y_sq: Float,
        sum_xy: Float,
    ) -> Self {
        Self {
            n: n as Float,
            sum_x,
            sum_y,
            sum_x_sq,
            sum_y_sq,
            sum_xy,
        }
    }
    pub(super) fn correlation(self) -> Float {
        let centered_x = self.sum_x_sq - self.sum_x * self.sum_x / self.n;
        let centered_y = self.sum_y_sq - self.sum_y * self.sum_y / self.n;
        let denominator_sq = centered_x * centered_y;
        if denominator_sq < TA_EPSILON {
            0.0 as Float
        } else {
            (self.sum_xy - self.sum_x * self.sum_y / self.n) / denominator_sq.sqrt()
        }
    }

    pub(super) fn beta(self) -> Float {
        // TA_BETA applies TA_IS_ZERO directly to the period-scaled centered
        // real0 variance: n * sum(x²) - sum(x)².
        let scaled_variance_x = self.n * self.sum_x_sq - self.sum_x * self.sum_x;
        if is_ta_zero(scaled_variance_x) {
            0.0 as Float
        } else {
            (self.n * self.sum_xy - self.sum_x * self.sum_y) / scaled_variance_x
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct RollingPairedMoments {
    period: usize,
    trailing_x: Vec<Float>,
    trailing_y: Vec<Float>,
    index: usize,
    count: usize,
    sum_x: Float,
    sum_y: Float,
    sum_x_sq: Float,
    sum_y_sq: Float,
    sum_xy: Float,
}

impl RollingPairedMoments {
    pub(super) fn new(period: usize) -> Self {
        let mut trailing_x = Vec::new();
        trailing_x.resize(period.saturating_sub(1), 0.0 as Float);
        let mut trailing_y = Vec::new();
        trailing_y.resize(period.saturating_sub(1), 0.0 as Float);
        Self {
            period,
            trailing_x,
            trailing_y,
            index: 0,
            count: 0,
            sum_x: 0.0 as Float,
            sum_y: 0.0 as Float,
            sum_x_sq: 0.0 as Float,
            sum_y_sq: 0.0 as Float,
            sum_xy: 0.0 as Float,
        }
    }

    pub(super) fn push(&mut self, x: Float, y: Float) -> Option<PairedSnapshot> {
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x_sq += x * x;
        self.sum_y_sq += y * y;
        self.sum_xy += x * y;
        if self.count < self.period {
            self.count += 1;
        }

        if self.count < self.period {
            self.trailing_x[self.index] = x;
            self.trailing_y[self.index] = y;
            self.index = (self.index + 1) % self.trailing_x.len();
            return None;
        }

        let snapshot = PairedSnapshot {
            n: self.period as Float,
            sum_x: self.sum_x,
            sum_y: self.sum_y,
            sum_x_sq: self.sum_x_sq,
            sum_y_sq: self.sum_y_sq,
            sum_xy: self.sum_xy,
        };
        self.remove_trailing(x, y);
        Some(snapshot)
    }

    fn remove_trailing(&mut self, x: Float, y: Float) {
        if self.trailing_x.is_empty() {
            self.sum_x -= x;
            self.sum_y -= y;
            self.sum_x_sq -= x * x;
            self.sum_y_sq -= y * y;
            self.sum_xy -= x * y;
            return;
        }

        let old_x = self.trailing_x[self.index];
        let old_y = self.trailing_y[self.index];
        self.sum_x -= old_x;
        self.sum_y -= old_y;
        self.sum_x_sq -= old_x * old_x;
        self.sum_y_sq -= old_y * old_y;
        self.sum_xy -= old_x * old_y;
        self.trailing_x[self.index] = x;
        self.trailing_y[self.index] = y;
        self.index = (self.index + 1) % self.trailing_x.len();
    }

    pub(super) fn reset(&mut self) {
        self.trailing_x.fill(0.0 as Float);
        self.trailing_y.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum_x = 0.0 as Float;
        self.sum_y = 0.0 as Float;
        self.sum_x_sq = 0.0 as Float;
        self.sum_y_sq = 0.0 as Float;
        self.sum_xy = 0.0 as Float;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RegressionFit {
    pub(super) slope: Float,
    pub(super) intercept: Float,
}

#[derive(Debug, Clone)]
pub(super) struct RollingRegression {
    period: usize,
    n: Float,
    sum_x: Float,
    divisor: Float,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
    sum_y: Float,
    sum_xy: Float,
}

impl RollingRegression {
    pub(super) fn new(period: usize) -> Self {
        let n = period as Float;
        let sum_x = n * (n - 1.0 as Float) * 0.5 as Float;
        let sum_x_sq = n * (n - 1.0 as Float) * (2.0 as Float * n - 1.0 as Float) / 6.0 as Float;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Self {
            period,
            n,
            sum_x,
            divisor: sum_x * sum_x - n * sum_x_sq,
            buffer,
            index: 0,
            count: 0,
            sum_y: 0.0 as Float,
            sum_xy: 0.0 as Float,
        }
    }

    pub(super) fn push(&mut self, input: Float) -> Option<RegressionFit> {
        if self.count < self.period {
            self.buffer[self.index] = input;
            self.sum_y += input;
            self.sum_xy += (self.period - 1 - self.count) as Float * input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;

            if self.count < self.period {
                return None;
            }
            return Some(self.fit());
        }

        let trailing = self.buffer[self.index];
        self.sum_xy = self.sum_xy + self.sum_y - self.n * trailing;
        self.sum_y = self.sum_y - trailing + input;
        self.buffer[self.index] = input;
        self.index = (self.index + 1) % self.period;
        Some(self.fit())
    }

    fn fit(&self) -> RegressionFit {
        let slope = (self.n * self.sum_xy - self.sum_x * self.sum_y) / self.divisor;
        let intercept = (self.sum_y - slope * self.sum_x) / self.n;
        RegressionFit { slope, intercept }
    }

    pub(super) fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum_y = 0.0 as Float;
        self.sum_xy = 0.0 as Float;
    }
}

#[cfg(test)]
mod tests {
    use super::{is_ta_zero, Float, TA_EPSILON};

    #[test]
    fn ta_zero_uses_strict_endpoints() {
        assert!(is_ta_zero(0.0 as Float));
        assert!(is_ta_zero(TA_EPSILON * 0.5 as Float));
        assert!(is_ta_zero(-TA_EPSILON * 0.5 as Float));
        assert!(!is_ta_zero(TA_EPSILON));
        assert!(!is_ta_zero(-TA_EPSILON));
    }
}
