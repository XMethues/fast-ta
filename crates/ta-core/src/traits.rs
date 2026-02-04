//! Core traits for technical analysis indicators
//!
//! This module defines fundamental traits that all indicators must implement.
//! The design provides flexibility for different usage patterns while maintaining
//! high performance through zero-copy operations.
//!
//! ## NaN Value Semantics
//!
//! This library uses `Float::NAN` with context-specific meanings. Understanding these is crucial for correct usage.
//!
//! ### Contexts
//!
//! - **Input validation**: `Float::NAN` in input data → **Error, reject entire operation**
//! - **Batch warm-up**: `Float::NAN` in output arrays → **Normal placeholder** for warm-up period
//! - **Streaming warm-up**: `Float::NAN` from `next()` → **Normal state**, not an error
//!
//! ## Design Rationale
//!
//! `Float::NAN` is used instead of `Option<Float>` for:
//! - **~0.02ns lower overhead** per call (benchmark: < 1% difference)
//! - **Better SIMD compatibility** with `Float` types aligning with SIMD vector operations
//! - **Simpler cognitive model** for users (single return type vs Option wrapper)
//!
//! See [error.rs](../error/index.html) for detailed validation patterns.
//! Note that `stream()` uses `Option<Float>` where `None` indicates warm-up.

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

    /// Convenient batch computation with automatic memory management
    ///
    /// This method allocates an output vector and processes all inputs.
    /// The first `lookback()` elements will be `Float::NAN` (warm-up placeholders),
    /// followed by valid indicator values.
    ///
    /// # NaN Handling
    ///
    /// Output array uses `Float::NAN` for warm-up phase (first `lookback()` elements).
    /// Filter these out if you only want valid values:
    /// ```rust,ignore
    /// let outputs = sma.compute_to_vec(prices)?;
    /// let valid = outputs.iter().filter(|&v| !v.is_nan()).collect();
    /// ```
    ///
    /// # Returns
    ///
    /// A `Vec` containing all computed output values. Use `is_nan()` to
    /// distinguish warm-up placeholders from valid values.
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
    ///
    /// // Filter out warm-up values (only process valid data)
    /// let valid: Vec<_> = results.into_iter().filter_map(|x| x).collect();
    /// assert_eq!(valid.len(), 6);
    /// ```
    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>;

    /// Process a single new value (streaming mode)
    ///
    /// This method processes one input value at a time, maintaining internal state
    /// across calls. Returns `Float` (not `Option<Float>`).
    ///
    /// # NaN Handling
    ///
    /// - **Returns `Float::NAN`**: Indicates indicator is in **warm-up phase** (insufficient data)
    /// - **Returns valid value**: Sufficient data accumulated, computed indicator value
    ///
    /// ## Important
    ///
    /// **This is NOT an error state.** `Float::NAN` return indicates:
    /// - The indicator needs more data to produce a valid output
    /// - This is a normal, expected state during initial `next()` calls
    /// - Input validation should happen BEFORE calling `next()` (reject invalid data with error)
    ///
    /// ## Validation Pattern
    ///
    /// Check with `value.is_nan()` to distinguish warm-up from errors:
    /// ```rust,ignore
    /// let value = sma.next(price);
    /// if !value.is_nan() {
    ///     execute_trade(value);  // Valid output
    /// } else {
    ///     // Warm-up phase - indicator needs more data
    ///     // (First `lookback()` calls will return Float::NAN)
    /// }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `input` - A single new input value
    ///
    /// # Returns
    ///
    /// - `Float::NAN`: Indicator in warm-up phase (needs more data)
    /// - Valid `Float`: Computed indicator value
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sma = Sma::new(5)?;
    ///
    /// // First 4 calls return Float::NAN (warm-up phase)
    /// assert!(sma.next(1.0).is_nan());
    /// assert!(sma.next(2.0).is_nan());
    /// assert!(sma.next(3.0).is_nan());
    /// assert!(sma.next(4.0).is_nan());
    ///
    /// // 5th call returns first valid output
    /// assert_eq!(sma.next(5.0), 3.0);
    ///
    /// // Subsequent calls always return valid values
    /// assert_eq!(sma.next(6.0), 4.0);
    /// ```
    ///
    /// # Note: Difference from `stream()`
    ///
    /// The `stream()` method uses `Option<Float>` semantics:
    /// - `None` indicates warm-up
    /// - `Some(value)` indicates valid output
    ///
    /// This difference provides semantic flexibility:
    /// - `next()`: Best performance, `Float::NAN` for warm-up
    /// - `stream()`: Batch processing, `Option<Float>` for clear semantics
    fn next(&mut self, input: Self::Input) -> Self::Output;
    ///
    /// This is a convenience method that processes a slice of inputs using `next`,
    /// returning `Vec<Option<Output>>` where each element corresponds to the result
    /// of processing that input.
    ///
    /// # Returns
    ///
    /// A vector where each element is:
    /// - `Some(output)` - Valid output for that input
    /// - `None` - Insufficient data at that point (warm-up phase)
    ///
    /// # NaN Handling
    ///
    /// Unlike `next()` which returns `Float::NAN` for warm-up, this method
    /// uses `Option<Float>` where `None` indicates warm-up phase.
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
    /// // Filter only valid results
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
