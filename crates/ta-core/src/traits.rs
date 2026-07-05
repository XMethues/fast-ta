//! Core traits for technical analysis indicators.
//!
//! The core batch contract is intentionally close to TA-Lib C: implementations
//! can write valid values compactly into caller-provided output buffers and
//! return an [`OutputRange`](crate::OutputRange) describing where those values
//! belong in the original input. Existing indicators remain source-compatible
//! because [`Indicator::compute`] has a default implementation over
//! [`Indicator::compute_to_vec`]; performance-sensitive indicators should
//! override it.
//!
//! ## Validation semantics
//!
//! `ta-core` APIs are Rust-first by default: invalid parameters, mismatched
//! lengths, non-finite inputs, and undersized output buffers are reported with
//! [`TalibError`](crate::TalibError) through [`Result`](crate::Result). Warm-up
//! positions in convenience vectors are represented by [`PadValue`](crate::PadValue).

use crate::{
    common::{validate_input_len, validate_output_len},
    OutputRange, PadValue, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Unified trait for single-input technical-analysis indicators.
///
/// Multi-input and multi-output TA-Lib functions expose free functions and
/// inherent struct methods with separate slices/buffers for performance. This
/// trait covers the common single-input shape used by functions such as `SMA`,
/// `AVGDEV`, and unary math transforms.
pub trait Indicator<const N: usize = 1> {
    /// Input type for this indicator.
    type Input;

    /// Output type for this indicator.
    type Output: PadValue;

    /// Returns the number of input elements required before the first output.
    fn lookback(&self) -> usize;

    /// Computes valid values into a compact caller-provided buffer.
    ///
    /// The default implementation calls [`Indicator::compute_to_vec`], then
    /// copies non-padding values from the full-length vector into `outputs`.
    /// Indicator implementations should override this method to avoid the
    /// allocation on performance-sensitive paths.
    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<OutputRange> {
        let padded = self.compute_to_vec(inputs)?;
        let lookback = self.lookback();
        let count = validate_input_len(inputs.len(), lookback)?;
        validate_output_len("compute", outputs.len(), count)?;

        if count > 0 {
            outputs[..count].copy_from_slice(&padded[lookback..lookback + count]);
        }

        Ok(OutputRange::new(lookback, count))
    }

    /// Computes a full-length padded vector.
    ///
    /// Positions outside the valid output range are filled with
    /// [`PadValue::pad_value`]. For `Float` outputs this is `NaN`; for integer
    /// outputs this is `0`.
    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>;

    /// Processes one input value in streaming mode.
    ///
    /// Streaming warm-up returns the output type's padding value rather than an
    /// error. Callers that need strict streaming validation should validate the
    /// input before calling `next` or use an indicator-specific checked method.
    fn next(&mut self, input: Self::Input) -> Self::Output;
}

/// Trait for indicators that can reset their internal streaming state.
pub trait Resettable {
    /// Reset the indicator to its initial state.
    fn reset(&mut self);
}
