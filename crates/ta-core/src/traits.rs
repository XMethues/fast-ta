//! Core traits for technical analysis indicators.
//!
//! The batch contract is intentionally close to TA-Lib C: implementations write
//! valid values into caller-provided Structure-of-Arrays (SoA) output buffers and
//! return an [`OutputRange`](crate::OutputRange) describing where those compact
//! values belong in the original input. Single-output indicators use plain
//! slices; multi-input and multi-output indicators provide small named input and
//! output view structs.
//!
//! Streaming is modeled separately from batch computation. [`Indicator`] is the
//! common batch capability every indicator implements, while
//! [`StreamingIndicator`] covers per-tick real-time updates with explicit warm-up
//! state.

use crate::{OutputRange, Result};

/// Batch capability implemented by every technical-analysis indicator.
///
/// Associated input and output types are generic over the borrow lifetime so
/// implementations can preserve SoA layouts:
///
/// - single input/output: `&[Float]` and `&mut [Float]`
/// - multi input: an input view such as `AVGPRICEInput<'a>`
/// - multi output: an output view such as `MINMAXOutputMut<'a>`
///
/// Convenience wrappers return `OutputOwned`, usually a padded `Vec<Float>` or a
/// named struct containing several padded vectors.
pub trait Indicator {
    /// Borrowed input view used by batch computation.
    type Input<'a>
    where
        Self: 'a;

    /// Borrowed mutable output view used by zero-copy batch computation.
    type OutputMut<'a>
    where
        Self: 'a;

    /// Owned padded output returned by convenience computation.
    type OutputOwned;

    /// Returns the number of input elements required before the first output.
    fn lookback(&self) -> usize;

    /// Computes valid values into caller-provided compact output buffers.
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange>;

    /// Computes full-length padded output vectors.
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned>;
}

/// Streaming capability implemented by every indicator for per-tick updates.
///
/// `Ok(None)` represents warm-up before the first valid output. `Ok(Some(_))`
/// represents a valid streaming output, and `Err(_)` represents invalid input or
/// another computation failure.
pub trait StreamingIndicator {
    /// One streaming input tick.
    type Tick;

    /// One valid streaming output tick.
    type TickOutput;

    /// Processes one input tick.
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>>;
}

/// Trait for indicators that can reset their internal streaming state.
pub trait Resettable {
    /// Reset the indicator to its initial state.
    fn reset(&mut self);
}
