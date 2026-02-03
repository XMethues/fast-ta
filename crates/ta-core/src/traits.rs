//! Core traits for technical analysis indicators
//!
//! This module defines the fundamental traits that all indicators must implement.
//! The design provides flexibility for different usage patterns while maintaining
//! high performance through zero-copy operations.

use crate::error::Result;
/// Unified trait for technical analysis indicators
///
/// This trait provides a unified interface that supports three usage modes:
/// - **Batch computation** with zero-copy performance (`compute`)
/// - **Convenient batch computation** with automatic memory management (`compute_to_vec`)
/// - **Streaming computation** for real-time data (`next`, `stream`)
///
/// # SIMD Acceleration Requirement
///
/// **All indicator implementations MUST use SIMD acceleration** for the `compute` method.
/// - When processing >1000 data points, SIMD implementations must achieve >2x speedup over scalar implementations
///
/// - Use the `wide` crate for portable SIMD operations
/// - Implement SIMD for the main computation path, with scalar fallback for remainder elements if necessary
/// - Performance benchmarks are required to verify SIMD acceleration effectiveness
///
/// # Generic Parameters
///
/// - `N`: Number of output values per input (default: 1). Multi-output indicators can specify a different value (e.g., Bollinger Bands might use `N=3`).
///
/// # Example
///
/// ```rust,ignore
/// use ta_core::{traits::Indicator, overlap::Sma, error::Result};
///
/// fn example() -> Result<()> {
///     // Create indicator
///     let mut sma = Sma::new(20)?;
///
///     // Mode 1: Zero-copy batch computation (high performance)
///     let prices: &[Float] = &[1.0, 2.0, 3.0, 4.0, 5.0];
///     let mut outputs = vec![0.0; prices.len()];
///     let count = sma.compute(prices, &mut outputs)?;
///
///     // Mode 2: Convenient batch computation (easy to use)
///     let results = sma.compute_to_vec(prices)?;
///
///     // Mode 3: Streaming (real-time processing)
///     for price in prices {
///         if let Some(value) = sma.next(price) {
///             println!("SMA: {}", value);
///         }
///     }
///
///     Ok(())
/// }
/// ```
#[allow(unused_variables)]
pub trait Indicator<const N: usize = 1> {
    /// Input type for this indicator
    ///
    /// Most indicators use `Float` (f64 or f32 depending on feature configuration),
    /// but custom indicators can use other types like `OHLC` structs.
    type Input;

    /// Output type for this indicator
    ///
    /// For single-output indicators (`N=1`), this is typically `Float`.
    /// For multi-output indicators, this could be an array or tuple type.
    type Output;

    /// Returns the amount of historical data required to produce the first valid output
    ///
    /// This is also known as "warm-up period" or "initialization period". For example:
    /// - SMA(20) returns 19 (needs 20 data points, first output on index 19)
    /// - EMA(20) typically returns 19 (same lookback)
    /// - RSI(14) returns 14 (needs at least 14 data points)
    ///
    /// # Returns
    ///
    /// The number of initial data points that will be skipped during computation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sma = Sma::new(20)?;
    /// assert_eq!(sma.lookback(), 19);
    ///
    /// // With 25 data points, we get 25 - 19 = 6 outputs
    /// let outputs = sma.compute_to_vec(&[0.0; 25])?;
    /// assert_eq!(outputs.len(), 6);
    /// ```
    fn lookback(&self) -> usize;

    /// Zero-copy batch computation (performance-optimized)
    ///
    /// This is the core computation method that writes results directly to a
    /// pre-allocated output buffer. It performs no heap allocation during computation,
    /// making it ideal for high-frequency scenarios.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of input data
    /// * `outputs` - Pre-allocated output buffer. Must have sufficient capacity:
    ///   `outputs.len() >= inputs.len() - self.lookback()`
    ///
    /// # Returns
    ///
    /// The number of output values actually written to `outputs`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Output buffer is too small
    /// - Input data contains invalid values (e.g., NaN)
    /// - Computation fails for other reasons
    ///
    /// # Performance
    ///
    /// - **Zero heap allocation**: Uses the provided output buffer
    /// - **Cache-friendly**: Linear access patterns
    /// - **Ideal for**: High-frequency trading, backtesting with large datasets
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use ta_core::{overlap::Sma, traits::Indicator, error::Result};
    ///
    /// fn backtest(prices: &[Float]) -> Result<Vec<Float>> {
    ///     let sma = Sma::new(20)?;
    ///     let lookback = sma.lookback();
    ///
    ///     if prices.len() <= lookback {
    ///         return Ok(Vec::new());
    ///     }
    ///
    ///     // Pre-allocate output buffer (can be reused across calls)
    ///     let mut outputs = vec![0.0; prices.len() - lookback];
    ///
    ///     // Zero-copy computation
    ///     let count = sma.compute(prices, &mut outputs)?;
    ///
    ///     // Trim to actual count (important if output varies)
    ///     outputs.truncate(count);
    ///
    ///     Ok(outputs)
    /// }
    /// ```
    fn compute(&self, inputs: &[Self::Input], _outputs: &mut [Self::Output]) -> Result<usize>;

    /// Convenient batch computation with automatic memory management
    ///
    /// This is a convenience wrapper around `compute` that automatically allocates
    /// and manages the output buffer. It's simpler to use but incurs one heap allocation.
    ///
    /// # Returns
    ///
    /// A `Vec` containing all computed output values.
    ///
    /// # Performance
    ///
    /// - **One heap allocation**: Creates the output Vec
    /// - **Convenient**: No manual buffer management
    /// - **Ideal for**: One-off computations, moderate-sized datasets, prototypes
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sma = Sma::new(20)?;
    /// let prices = &[1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// // Simple and direct
    /// let results = sma.compute_to_vec(prices)?;
    /// assert_eq!(results.len(), prices.len() - 20);
    /// ```
    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>;

    /// Process a single new value (streaming mode)
    ///
    /// This method processes one input value at a time, maintaining internal state
    /// across calls. Returns `Some(output)` when enough data has accumulated,
    /// or `None` if still in the warm-up phase.
    ///
    /// # Arguments
    ///
    /// * `input` - A single new input value
    ///
    /// # Returns
    ///
    /// - `Some(output)` - A valid output value when sufficient data is available
    /// - `None` - Not enough data yet (still in warm-up phase)
    ///
    /// # Use Cases
    ///
    /// - Real-time market data processing
    /// - Trading bot decision making
    /// - Incremental indicator updates
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sma = Sma::new(5)?;
    ///
    /// // First 4 calls return None (warm-up phase)
    /// assert_eq!(sma.next(1.0), None);
    /// assert_eq!(sma.next(2.0), None);
    /// assert_eq!(sma.next(3.0), None);
    /// assert_eq!(sma.next(4.0), None);
    ///
    /// // 5th call returns first valid output
    /// assert_eq!(sma.next(5.0), Some(3.0));  // (1+2+3+4+5)/5 = 3.0
    ///
    /// // Subsequent calls always return Some
    /// assert_eq!(sma.next(6.0), Some(4.0));  // (2+3+4+5+6)/5 = 4.0
    /// ```
    fn next(&mut self, input: Self::Input) -> Option<Self::Output>;

    /// Process multiple values in streaming mode
    ///
    /// This is a convenience method that processes a slice of inputs using `next`,
    /// returning `Vec<Option<Output>>` where each element corresponds to the result
    /// of processing that input.
    ///
    /// # Returns
    ///
    /// A vector where each element is:
    /// - `Some(output)` - Valid output for that input
    /// - `None` - Insufficient data at that point
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sma = Sma::new(3)?;
    /// let prices = &[1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// let results: Vec<_> = sma.stream(prices);
    /// // results: [None, None, Some(2.0), Some(3.0), Some(4.0)]
    ///
    /// // Filter valid results
    /// let valid: Vec<_> = results.into_iter().filter_map(|x| x).collect();
    /// // valid: [2.0, 3.0, 4.0]
    /// ```
    fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>>;
}

/// Trait for indicators that can reset their internal state
///
/// Some indicators accumulate state across calls (e.g., EMA with its exponential
/// smoothing factor). This trait allows resetting that state to initial conditions.
///
/// # Use Cases
///
/// - Backtesting multiple scenarios without reallocating indicators
/// - Restarting analysis when market conditions change
/// - Memory optimization by reusing indicator instances
///
/// # Example
///
/// ```rust,ignore
/// let mut ema = Ema::new(20)?;
///
/// // Process some data
/// ema.next(1.0);
/// ema.next(2.0);
/// ema.next(3.0);
///
/// // Reset to initial state
/// ema.reset();
///
/// // Now next() behaves as if called for the first time
/// assert_eq!(ema.next(1.0), None);
/// ```
pub trait Resettable {
    /// Reset the indicator to its initial state
    ///
    /// After calling `reset()`, the indicator behaves as if it were just created.
    /// All internal buffers and accumulated values are cleared or reset to defaults.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Float;

    // Mock indicator for testing trait defaults
    struct MockIndicator {
        lookback: usize,
    }

    impl Indicator<1> for MockIndicator {
        type Input = Float;
        type Output = Float;

        fn lookback(&self) -> usize {
            self.lookback
        }

        fn compute(
            &self,
            _inputs: &[Self::Input],
            result_buffer: &mut [Self::Output],
        ) -> Result<usize> {
            // Simply fill result_buffer with zeros
            for out in result_buffer.iter_mut() {
                *out = 0.0;
            }
            Ok(result_buffer.len())
        }

        fn next(&mut self, _input: Self::Input) -> Option<Self::Output> {
            None
        }

        fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
            let lookback = self.lookback();
            if inputs.len() <= lookback {
                return Ok(Vec::new());
            }

            let mut outputs = Vec::with_capacity(inputs.len() - lookback);
            self.compute(inputs, &mut outputs)?;
            Ok(outputs)
        }

        fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
            inputs.iter().map(|&input| self.next(input)).collect()
        }
    }

    #[test]
    fn test_compute_to_vec_empty_inputs() {
        let indicator = MockIndicator { lookback: 5 };
        let result = indicator.compute_to_vec(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_to_vec_insufficient_data() {
        let indicator = MockIndicator { lookback: 5 };
        let result = indicator.compute_to_vec(&[1.0, 2.0]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_to_vec_minimum_data() {
        let indicator = MockIndicator { lookback: 5 };
        let result = indicator.compute_to_vec(&[0.0; 6]).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_stream_empty_inputs() {
        struct StreamMock;

        impl Indicator<1> for StreamMock {
            type Input = Float;
            type Output = Float;

            fn lookback(&self) -> usize {
                0
            }

            fn compute(
                &self,
                inputs: &[Self::Input],
                __outputs: &mut [Self::Output],
            ) -> Result<usize> {
                Ok(inputs.len())
            }

            fn next(&mut self, input: Self::Input) -> Option<Self::Output> {
                Some(input)
            }

            fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                Ok(inputs.to_vec())
            }

            fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
                inputs.iter().map(|&input| Some(input)).collect()
            }
        }

        let mut indicator = StreamMock;
        let result = indicator.stream(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_stream_with_data() {
        struct SumMock {
            sum: Float,
        }

        impl Indicator<1> for SumMock {
            type Input = Float;
            type Output = Float;

            fn lookback(&self) -> usize {
                0
            }

            fn compute(
                &self,
                inputs: &[Self::Input],
                _outputs: &mut [Self::Output],
            ) -> Result<usize> {
                Ok(inputs.len())
            }

            fn next(&mut self, input: Self::Input) -> Option<Self::Output> {
                self.sum += input;
                Some(self.sum)
            }

            fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                let mut result = Vec::with_capacity(inputs.len());
                let mut sum = 0.0;
                for &input in inputs {
                    sum += input;
                    result.push(sum);
                }
                Ok(result)
            }

            fn stream(&mut self, inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
                let mut results = Vec::with_capacity(inputs.len());
                for &input in inputs {
                    self.sum += input;
                    results.push(Some(self.sum));
                }
                results
            }
        }

        let mut indicator = SumMock { sum: 0.0 };
        let result = indicator.stream(&[1.0, 2.0, 3.0]);
        assert_eq!(result, vec![Some(1.0), Some(3.0), Some(6.0)]);
    }

    #[test]
    fn test_resettable_trait_exists() {
        struct ResetMock;

        impl Resettable for ResetMock {
            fn reset(&mut self) {
                // Nothing to reset
            }
        }

        let mut mock = ResetMock;
        mock.reset(); // Just ensure it compiles
    }
}
