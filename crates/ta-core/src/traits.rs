//! Core traits for technical analysis indicators.
//!
//! [`IndicatorConfig`] is the Rust-first execution seam for owned Compact Output,
//! caller-owned Batch Computation, Prepared Batch Runners, and independent
//! Streaming Computation. Named borrowed views preserve Structure-of-Arrays
//! layouts for future multi-input and multi-output implementations.
use crate::{CompactOutput, OutputRange, Result};

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Immutable Indicator Configuration with owned, caller-owned, prepared, and
/// streaming execution modes.
///
/// Implementations contain parameters only. Prepared Batch Runner scratch and
/// Streaming Computation state live in the associated execution types.
///
/// This trait is intentionally sealed because this crate owns the full Indicator
/// Catalogue and keeps [`CompactOutput`] construction crate-private to preserve
/// its invariants. External implementations are not part of this API contract.
pub trait IndicatorConfig: Sized + sealed::Sealed {
    /// Borrowed input view used by Batch Computation.
    type Input<'a>
    where
        Self: 'a;

    /// Owned payload stored by [`CompactOutput`].
    type Output;

    /// Borrowed mutable output view used by caller-owned Batch Computation.
    type OutputMut<'a>
    where
        Self: 'a;

    /// Reusable Prepared Batch Runner for this exact configuration type.
    type BatchRunner: PreparedBatchRunner<Self>;

    /// Independent Streaming Computation created by this exact configuration type.
    type Stream: StreamingComputation<Self>;

    /// Returns the number of source positions before the first valid output.
    fn lookback(&self) -> usize;

    /// Computes an owned Compact Output.
    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>>;

    /// Computes into caller-owned compact output storage.
    fn compute_into<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange>;

    /// Prepares a reusable runner for inputs no longer than `max_input_len`.
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner>;

    /// Creates an independent Streaming Computation.
    fn stream(&self) -> Result<Self::Stream>;
}

/// Reusable Batch Computation with an explicit maximum input capacity.
///
/// The configuration parameter ties the accepted borrowed input and output
/// views to one exact [`IndicatorConfig`]. This catalogue trait is intentionally
/// sealed; runners are supplied only by this crate's indicator implementations.
pub trait PreparedBatchRunner<C: IndicatorConfig>: sealed::Sealed {
    /// Returns the maximum accepted source length.
    fn max_input_len(&self) -> usize;

    /// Computes into caller-owned output without growing prepared storage.
    fn compute_into<'a>(
        &mut self,
        input: C::Input<'a>,
        output: C::OutputMut<'a>,
    ) -> Result<OutputRange>
    where
        C: 'a;
}

/// Stateful per-tick execution created independently from Indicator Configuration.
///
/// The configuration parameter identifies the exact [`IndicatorConfig`] that
/// creates this stream. This catalogue trait is intentionally sealed; streams
/// are supplied only by this crate's indicator implementations.
pub trait StreamingComputation<C: IndicatorConfig>: sealed::Sealed {
    /// One input Tick.
    type Tick;

    /// One valid output Tick.
    type TickOutput;

    /// Processes one Tick, returning `None` during Warm-up.
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>>;

    /// Resets accumulated observations to the initial Warm-up state.
    fn reset(&mut self);
}
