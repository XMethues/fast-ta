//! Simple Moving Average (SMA)
//!
//! The Simple Moving Average calculates the average price over a specified period.
//! It's one of the most commonly used technical indicators for trend identification.

use crate::{
    error::{Result, TalibError},
    traits::{Indicator, Resettable},
    Float,
};
use alloc::vec;

/// Simple Moving Average (SMA) indicator
///
/// SMA calculates the arithmetic mean of prices over a specified time period.
/// Each value in the output represents the average of the previous N data points.
///
/// # Formula
///
/// ```text
/// SMA = (P1 + P2 + ... + Pn) / n
///
/// Where:
/// - P1, P2, ..., Pn are the prices over the last n periods
/// - n is the period
/// ```
///
/// # Characteristics
///
/// - **Lagging indicator**: Based on past data, reacts slowly to price changes
/// - **Trend follower**: Smooths out price fluctuations to show the underlying trend
/// - **Equal weighting**: All data points in the period contribute equally
///
/// # Use Cases
///
/// - Identifying trend direction (upward, downward, or sideways)
/// - Providing dynamic support and resistance levels
/// - Generating trading signals (price crossovers, moving average crossovers)
/// - Smoothing noisy price data
///
/// # Example
///
/// ```rust,ignore
/// use ta_core::{overlap::Sma, traits::Indicator, error::Result};
///
/// fn example() -> Result<()> {
///     // Create a 20-period SMA
///     let sma = Sma::new(20)?;
///
///     // Batch computation
///     let prices = &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
///     let results = sma.compute_to_vec(prices)?;
///
///     // Streaming computation
///     let mut sma_stream = Sma::new(5)?;
///     for price in prices {
///         if let Some(value) = sma_stream.next(price) {
///             println!("SMA(5): {}", value);
///         }
///     }
///
///     Ok(())
/// }
/// ```
pub struct Sma {
    /// Number of periods for the moving average
    period: usize,

    /// Circular buffer for storing recent prices
    buffer: Vec<Float>,

    /// Running sum of values in the buffer
    sum: Float,

    /// Current position in the circular buffer
    index: usize,

    /// Number of valid values in the buffer (0 to period)
    count: usize,
}

impl Sma {
    /// Creates a new SMA indicator with the specified period
    ///
    /// # Arguments
    ///
    /// * `period` - Number of periods to average (must be > 0)
    ///
    /// # Returns
    ///
    /// Returns `Ok(Sma)` if the period is valid, `Err(TalibError)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns `TalibError::InvalidPeriod` if the period is zero.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ta_core::{overlap::Sma, error::Result};
    ///
    /// fn example() -> Result<()> {
    ///     let sma = Sma::new(20)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn new(period: usize) -> Result<Self> {
        if period == 0 {
            return Err(TalibError::invalid_period(
                period,
                "period must be greater than zero",
            ));
        }

        Ok(Sma {
            period,
            buffer: vec![0.0; period],
            sum: 0.0,
            index: 0,
            count: 0,
        })
    }

    /// Returns the period setting
    pub fn period(&self) -> usize {
        self.period
    }

    fn compute_to_vec_impl(&self, inputs: &[Float]) -> Result<Vec<Float>> {
        let lookback = self.lookback();
        if inputs.len() <= lookback {
            return Ok(Vec::new());
        }

        let mut outputs = Vec::with_capacity(inputs.len() - lookback);
        for i in lookback..inputs.len() {
            unsafe {
                outputs.set_len(i - lookback + 1);
            }
            let window = &inputs[i - lookback..=i];

            for &value in window {
                if !value.is_finite() {
                    return Err(TalibError::invalid_input(
                        "Input contains NaN or infinite values",
                    ));
                }
            }

            let sum: Float = window.iter().sum();
            outputs[i - lookback] = sum / self.period as Float;
        }

        Ok(outputs)
    }
}

impl Indicator<1> for Sma {
    type Input = Float;
    type Output = Float;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<usize> {
        let lookback = self.lookback();
        let inputs_len = inputs.len();

        if inputs_len <= lookback {
            return Ok(0);
        }

        let expected_outputs = inputs_len - lookback;
        if outputs.len() < expected_outputs {
            return Err(TalibError::InsufficientData {
                required: expected_outputs,
                actual: outputs.len(),
            });
        }

        for (i, output) in outputs.iter_mut().enumerate().take(expected_outputs) {
            let end = i + lookback + 1;
            let window = &inputs[i..end];

            for &value in window {
                if !value.is_finite() {
                    return Err(TalibError::invalid_input(
                        "Input contains NaN or infinite values",
                    ));
                }
            }

            let sum: Float = window.iter().sum();
            *output = sum / self.period as Float;
        }

        Ok(expected_outputs)
    }

    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        let lookback = self.lookback();
        if inputs.len() <= lookback {
            return Ok(Vec::new());
        }

        let mut outputs = Vec::with_capacity(inputs.len() - lookback);
        for i in lookback..inputs.len() {
            let end = i + 1;
            let window = &inputs[i..end];

            for &value in window {
                if !value.is_finite() {
                    return Err(TalibError::invalid_input(
                        "Input contains NaN or infinite values",
                    ));
                }
            }

            let sum: Float = window.iter().sum();
            outputs.push(sum / self.period as Float);
        }

        Ok(outputs)
    }

    fn next(&mut self, input: Self::Input) -> Option<Self::Output> {
        if !input.is_finite() {
            return None;
        }

        if self.count == self.period {
            self.sum -= self.buffer[self.index];
        } else {
            self.count += 1;
        }

        self.buffer[self.index] = input;
        self.sum += input;

        self.index = (self.index + 1) % self.period;

        if self.count == self.period {
            Some(self.sum / self.period as Float)
        } else {
            None
        }
    }

    fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
        let mut results = Vec::with_capacity(inputs.len());
        for &input in inputs {
            results.push(self.next(input));
        }
        results
    }
}

impl Resettable for Sma {
    fn reset(&mut self) {
        self.sum = 0.0;
        self.index = 0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linear_data(n: usize) -> Vec<Float> {
        (1..=n).map(|i| i as Float).collect()
    }

    #[test]
    fn test_new_valid_period() {
        let sma = Sma::new(5);
        assert!(sma.is_ok());
        let sma = sma.unwrap();
        assert_eq!(sma.period(), 5);
        assert_eq!(sma.lookback(), 4);
    }

    #[test]
    fn test_new_period_of_one() {
        let sma = Sma::new(1);
        assert!(sma.is_ok());
        let sma = sma.unwrap();
        assert_eq!(sma.period(), 1);
        assert_eq!(sma.lookback(), 0);
    }

    #[test]
    fn test_new_zero_period_fails() {
        let sma = Sma::new(0);
        assert!(sma.is_err());
        match sma.unwrap_err() {
            TalibError::InvalidPeriod { period, .. } => assert_eq!(period, 0),
            _ => panic!("Expected InvalidPeriod error"),
        }
    }

    #[test]
    fn test_compute_empty_inputs() {
        let sma = Sma::new(5).unwrap();
        let mut outputs = [0.0; 10];
        let count = sma.compute(&[], &mut outputs).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_compute_insufficient_data() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[1.0, 2.0, 3.0];
        let mut outputs = [0.0; 10];
        let count = sma.compute(inputs, &mut outputs).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_compute_minimum_data() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let mut outputs = [0.0; 10];
        let count = sma.compute(inputs, &mut outputs).unwrap();
        assert_eq!(count, 1);
        assert!((outputs[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_linear_data() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut outputs = [0.0; 10];
        let count = sma.compute(inputs, &mut outputs).unwrap();
        assert_eq!(count, 4);

        let expected = vec![2.0, 3.0, 4.0, 5.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!((outputs[i] - exp).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_to_vec_empty_inputs() {
        let sma = Sma::new(5).unwrap();
        let results = sma.compute_to_vec(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_compute_to_vec_insufficient_data() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[1.0, 2.0, 3.0];
        let results = sma.compute_to_vec(inputs).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_compute_to_vec_correct_size() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[1.0; 25];
        let results = sma.compute_to_vec(inputs).unwrap();
        assert_eq!(results.len(), 21);
    }

    #[test]
    fn test_compute_to_vec_values() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let results = sma.compute_to_vec(inputs).unwrap();
        assert_eq!(results, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_next_warmup_phase() {
        let mut sma = Sma::new(3).unwrap();

        assert_eq!(sma.next(1.0), None);
        assert_eq!(sma.next(2.0), None);

        let result = sma.next(3.0);
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_next_after_warmup() {
        let mut sma = Sma::new(3).unwrap();

        sma.next(1.0);
        sma.next(2.0);
        sma.next(3.0);

        assert_eq!(sma.next(4.0), Some(3.0));
        assert_eq!(sma.next(5.0), Some(4.0));
        assert_eq!(sma.next(6.0), Some(5.0));
    }

    #[test]
    fn test_next_nan_returns_none() {
        let mut sma = Sma::new(3).unwrap();

        sma.next(1.0);
        sma.next(2.0);

        assert_eq!(sma.next(Float::NAN), None);
        assert_eq!(sma.next(3.0), None);
    }

    #[test]
    fn test_next_infinity_returns_none() {
        let mut sma = Sma::new(3).unwrap();

        sma.next(1.0);
        sma.next(2.0);

        assert_eq!(sma.next(Float::INFINITY), None);
    }

    #[test]
    fn test_stream_empty() {
        let mut sma = Sma::new(3).unwrap();
        let results = sma.stream(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_stream_with_data() {
        let mut sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let results = sma.stream(inputs);

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], None);
        assert_eq!(results[1], None);
        assert_eq!(results[2], Some(2.0));
        assert_eq!(results[3], Some(3.0));
        assert_eq!(results[4], Some(4.0));
    }

    #[test]
    fn test_stream_filter_valid() {
        let mut sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let results = sma.stream(inputs);

        let valid: Vec<_> = results.into_iter().filter_map(|x| x).collect();
        assert_eq!(valid, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut sma = Sma::new(3).unwrap();

        sma.next(1.0);
        sma.next(2.0);
        sma.next(3.0);
        assert_eq!(sma.next(4.0), Some(3.0));

        sma.reset();

        assert_eq!(sma.next(1.0), None);
        assert_eq!(sma.next(2.0), None);
        assert_eq!(sma.next(3.0), Some(2.0));
    }

    #[test]
    fn test_compute_output_buffer_too_small() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut outputs = [0.0; 1];

        let result = sma.compute(inputs, &mut outputs);
        assert!(result.is_err());
        match result.unwrap_err() {
            TalibError::InsufficientData {
                required: 2,
                actual: 1,
            } => {}
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_compute_with_nan_input() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, Float::NAN, 4.0, 5.0];
        let mut outputs = [0.0; 10];

        let result = sma.compute(inputs, &mut outputs);
        assert!(result.is_err());
        match result.unwrap_err() {
            TalibError::InvalidInput { .. } => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_compute_with_infinity_input() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[1.0, 2.0, Float::INFINITY, 4.0, 5.0];
        let mut outputs = [0.0; 10];

        let result = sma.compute(inputs, &mut outputs);
        assert!(result.is_err());
        match result.unwrap_err() {
            TalibError::InvalidInput { .. } => {}
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_compute_large_dataset() {
        let sma = Sma::new(20).unwrap();
        let inputs: Vec<Float> = (1..=1000).map(|i| i as Float).collect();
        let mut outputs = vec![0.0; inputs.len()];
        let count = sma.compute(&inputs, &mut outputs).unwrap();
        assert_eq!(count, 981);
    }

    #[test]
    fn test_period_one_lookback() {
        let sma = Sma::new(1).unwrap();
        assert_eq!(sma.lookback(), 0);

        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let results = sma.compute_to_vec(inputs).unwrap();

        assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_period_one_next() {
        let mut sma = Sma::new(1).unwrap();

        assert_eq!(sma.next(1.0), Some(1.0));
        assert_eq!(sma.next(2.0), Some(2.0));
        assert_eq!(sma.next(3.0), Some(3.0));
    }

    #[test]
    fn test_compute_consistency() {
        let period = 5;
        let data = create_linear_data(20);

        let sma_batch = Sma::new(period).unwrap();
        let batch_results = sma_batch.compute_to_vec(&data).unwrap();

        let mut sma_stream = Sma::new(period).unwrap();
        let mut stream_results = Vec::new();
        for &value in &data {
            if let Some(result) = sma_stream.next(value) {
                stream_results.push(result);
            }
        }

        assert_eq!(batch_results.len(), stream_results.len());
        for (batch, stream) in batch_results.iter().zip(stream_results.iter()) {
            assert!((batch - stream).abs() < 1e-10);
        }
    }

    #[test]
    fn test_precision_accuracy() {
        let sma = Sma::new(10).unwrap();
        let inputs: Vec<Float> = (1..=20).map(|i| i as Float / 3.0).collect();
        let results = sma.compute_to_vec(&inputs).unwrap();

        assert_eq!(results.len(), 11);

        for &value in &results {
            assert!(value.is_finite());
            assert!(value > 0.0);
        }
    }

    #[test]
    fn test_multiple_resets() {
        let mut sma = Sma::new(3).unwrap();

        sma.next(1.0);
        sma.next(2.0);
        sma.next(3.0);
        assert_eq!(sma.next(4.0), Some(3.0));

        sma.reset();

        sma.next(10.0);
        sma.next(20.0);
        sma.next(30.0);
        assert_eq!(sma.next(40.0), Some(30.0));

        sma.reset();

        sma.next(100.0);
        assert_eq!(sma.next(200.0), None);
    }

    #[test]
    fn test_negative_values() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
        let results = sma.compute_to_vec(inputs).unwrap();

        assert_eq!(results, vec![-3.0, -1.0, 1.0, 3.0]);
    }

    #[test]
    fn test_mixed_sign_values() {
        let sma = Sma::new(3).unwrap();
        let inputs = &[-2.0, 0.0, 2.0, 4.0, 6.0, 8.0];
        let results = sma.compute_to_vec(inputs).unwrap();

        assert_eq!(results, vec![0.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_constant_values() {
        let sma = Sma::new(5).unwrap();
        let inputs = &[5.0; 20];
        let results = sma.compute_to_vec(inputs).unwrap();

        assert!(results.iter().all(|&x| (x - 5.0).abs() < 1e-10));
    }

    #[test]
    fn test_large_period() {
        let sma = Sma::new(100).unwrap();
        assert_eq!(sma.lookback(), 99);

        let inputs: Vec<Float> = (1..=200).map(|i| i as Float).collect();
        let results = sma.compute_to_vec(&inputs).unwrap();

        assert_eq!(results.len(), 101);

        let expected_first: Float = (1..=100).map(|i| i as Float).sum::<Float>() / 100.0;
        assert!((results[0] - expected_first).abs() < 1e-8);
    }

    #[test]
    fn test_next_and_stream_produce_same_results() {
        let mut sma1 = Sma::new(3).unwrap();
        let mut sma2 = Sma::new(3).unwrap();

        let inputs = &[1.0, 2.0, 3.0, 4.0, 5.0];

        let mut next_results = Vec::new();
        for &input in inputs {
            next_results.push(sma1.next(input));
        }

        let stream_results = sma2.stream(inputs);

        assert_eq!(next_results, stream_results);
    }
}
