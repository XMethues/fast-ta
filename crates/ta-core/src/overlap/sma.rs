//! Implementation of the Simple Moving Average (SMA) indicator.

use crate::{simd::LANES, Float, Indicator};
use wide;
/// SMA indicator
pub struct SMA {
    period: usize,
}

impl SMA {
    /// Create a new SMA indicator with the given period.
    pub fn new(period: usize) -> Self {
        SMA { period }
    }
}

impl Indicator for SMA {
    type Input = Float;

    type Output = Float;

    fn lookback(&self) -> usize {
        self.period.saturating_sub(1)
    }

    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) {
        let n = inputs.len();
        let window_size = self.period;
        let mut window_sum = 0.0;
        let mut i = 0;
        while i + LANES <= window_size {
            let slice = &inputs[i..i + LANES];
            let chunk = wide::f64x8::from(slice);
            window_sum += chunk.reduce_add();
            i += LANES;
        }
        while i < window_size {
            window_sum += inputs[i];
            i += 1;
        }
        // First window result
        outputs[window_size - 1] = window_sum / window_size as Float;
        // Use sliding window technique: subtract old element, add new element
        for i in window_size..n {
            window_sum = window_sum - inputs[i - window_size] + inputs[i];
            outputs[i] = window_sum / window_size as Float;
        }
    }

    fn compute_to_vec(&self, inputs: &[Self::Input]) -> crate::Result<Vec<Self::Output>> {
        let mut result = vec![Float::NAN; inputs.len()];
        self.compute(inputs, &mut result);
        Ok(result)
    }

    fn next(&mut self, _input: Self::Input) -> Option<Self::Output> {
        // For streaming computation, we would need to maintain internal state
        // This is a simplified implementation that returns None
        None
    }

    fn stream(&mut self, _inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
        // For streaming computation, we would need to maintain internal state
        // This is a simplified implementation that returns empty vector
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Indicator;

    #[test]
    fn test_sma_basic() {
        let sma = SMA::new(3);
        let inputs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma.compute_to_vec(&inputs).unwrap();

        // First two values should be NaN due to insufficient data
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());

        // SMA(3) of [1,2,3] = 2.0
        assert!((result[2] - 2.0).abs() < 1e-10);
        // SMA(3) of [2,3,4] = 3.0
        assert!((result[3] - 3.0).abs() < 1e-10);
        // SMA(3) of [3,4,5] = 4.0
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sma_lookback() {
        let sma = SMA::new(5);
        assert_eq!(sma.lookback(), 4); // 5 - 1 = 4

        let sma = SMA::new(1);
        assert_eq!(sma.lookback(), 0); // 1 - 1 = 0 (saturating_sub prevents underflow)
    }
}
