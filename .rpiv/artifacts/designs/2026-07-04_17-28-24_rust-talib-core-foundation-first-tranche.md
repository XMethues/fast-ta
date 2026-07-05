---
date: 2026-07-04T17:28:24+0800
author: unknown
commit: 896d1d7
branch: main
repository: fast-ta
topic: "TA-Lib Rust core foundation + first tranche"
tags: [design, ta-core, talib, indicators, foundation, first-tranche]
status: ready
parent: .rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md
last_updated: 2026-07-04T17:28:24+0800
last_updated_by: unknown
---

# Design: TA-Lib Rust Core Foundation + First Tranche

## Summary
This design establishes the Rust `ta-core` contract for implementing the full TA-Lib inventory while delivering a first tranche of concrete functions. The core will expose both TA-Lib-style zero-copy free functions and uppercase struct surfaces, use Rust-style strict finite-input validation by default, and record all unfinished official TA-Lib functions for later group-by-group implementation.

## Requirements
- Implement every TA-Lib indicator/function algorithm in Rust `ta-core`; Python and WASM remain adapter layers.
- Use official TA-Lib groups as Rust module boundaries.
- Provide both free-function batch APIs and struct-based APIs as equal first-class surfaces.
- Prefer performance-oriented zero-copy output buffers; provide padded convenience `Vec` wrappers.
- Use separate input slices for multi-price and paired-series batch APIs.
- Use parallel output buffers for multi-output functions.
- Use `Result`/`TalibError` for invalid parameters, length mismatches, non-finite inputs, and output buffer errors.
- Preserve uppercase TA-Lib names in public APIs.
- Include a source-level inventory ledger for implemented and pending functions.
- First tranche includes foundation, SMA rewrite, Price Transform, Math Transform, Math Operators, inventory, and benchmarks.

## Current State Analysis
`ta-core` currently exports only one indicator family (`overlap`) and one indicator type (`SMA`). The current SMA demonstrates full-length NaN-padded output shape but has legacy constructor and streaming-state hazards, so it is a reference for output alignment only, not for error handling or buffer initialization.

### Key Discoveries
- `crates/ta-core/src/lib.rs:32-41` exposes `error`, `overlap`, `simd`, `traits`, `types`, and common aliases only.
- `crates/ta-core/src/overlap/mod.rs:4-6` privately declares `sma` and re-exports `SMA`; this is the family-module pattern to extend.
- `crates/ta-core/src/overlap/sma.rs:51-55` uses `assert!` and `AVec::with_capacity`, which must not be copied for new constructors/state.
- `crates/ta-core/src/traits.rs:76-219` defines only allocating batch and `next()` methods despite docs mentioning zero-copy `compute`.
- `crates/ta-core/src/error.rs:73-115` already has constructors for invalid input, invalid period, and insufficient data.
- `crates/ta-core/src/simd/dispatch.rs:35-40` uses contiguous `&[Float]` slices, supporting the separate-slice batch decision.
- `.sisyphus/IMPLEMENTATION_PLAN.md:3108-3111` misclassifies some official TA-Lib functions, so this design follows official groups instead.
- `README:13` and `.sisyphus/QUALITY_ASSURANCE_PLAN.md:65-73` require TA-Lib parity verification, but no `crates/ta-core/tests/` parity harness exists yet.

## Scope
### Building
- Core output and validation helpers in `ta-core`.
- Updated `Indicator`/`Resettable` traits for zero-copy + padded convenience usage.
- Rewritten `SMA` with fallible constructors, compact zero-copy output, padded wrapper, finite-input validation, and safe ring-buffer state.
- Price Transform functions: `AVGDEV`, `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`.
- Math Transform functions: `ACOS`, `ASIN`, `ATAN`, `CEIL`, `COS`, `COSH`, `EXP`, `FLOOR`, `LN`, `LOG10`, `SIN`, `SINH`, `SQRT`, `TAN`, `TANH`.
- Math Operators: `ADD`, `DIV`, `MULT`, `SUB`, `MAX`, `MAXINDEX`, `MIN`, `MININDEX`, `MINMAX`, `MINMAXINDEX`, `SUM`.
- Inventory ledger for the official 161-function TA-Lib inventory with first-tranche statuses and pending group records.
- Integration tests and Criterion benchmark target for first-tranche APIs.

### Not Building
- Python (`ta-py`) and WASM (`ta-wasm`) bindings in this design; they consume stable core APIs later.
- Complex TA-Lib groups/functions beyond first tranche: remaining Overlap, Momentum, Volume, Volatility, Cycle, Pattern Recognition, Statistic functions.
- TA-Lib C code generation or build-time XML ingestion.
- Full golden-vector fixture import; first tranche tests use deterministic expected values and helper behavior, while later group designs add full parity fixtures.
- Non-official extras (`WWMA`, `HMA`, `VWAP`) until the official 161-function inventory is complete.

## Decisions
### Module wiring follows family-private modules and public re-exports
Decision: Use private implementation files with family-level `pub use`, following `crates/ta-core/src/overlap/mod.rs:4-6`.

### Constructors use `Result<Self>`
Decision: New and rewritten indicators use fallible constructors and typed `TalibError`, replacing legacy panic constructors at `crates/ta-core/src/overlap/sma.rs:51-52`.

### Official TA-Lib groups define Rust module groups
Decision: Use official TA-Lib groups as modules because local planning docs misclassify functions such as `SAR`, `ATR/NATR/TRANGE`, and `VAR`.

### Dual API is the core output contract
Ambiguity: `README:21-25` prefers zero-copy buffers, while `crates/ta-core/src/overlap/sma.rs:104-107` returns full-length padded vectors.
Explored:
- Zero-copy compact buffers: best performance and closest to TA-Lib C.
- Full-length padded vectors: easiest for users and binding adapters.
Decision: Provide both. Core `compute`/free functions write compact outputs and return `OutputRange`; `compute_to_vec`/`*_vec` helpers return full-length padded output.

### Multi-price inputs use separate slices
Decision: Multi-input batch APIs take separate slices (`high`, `low`, `close`, `volume`, etc.) to match TA-Lib and preserve contiguous SIMD-friendly data.

### Multi-output functions use parallel buffers
Decision: Multi-output APIs write separate output buffers (struct-of-arrays) for performance. Convenience wrappers return named structs containing parallel `Vec`s.

### Rust-style validation is default
Decision: Default core APIs reject non-finite inputs with `TalibError::InvalidInput`. This follows the developer's “按照rust方式的” direction; TA-Lib/ta-lib-python NaN propagation can be added later as an explicit compatibility mode if needed.

### Public names remain uppercase TA-Lib names
Decision: Use uppercase public function/type names like current `SMA` (`crates/ta-core/src/overlap/mod.rs:6`) for TA-Lib discoverability, with local `#[allow(non_snake_case)]` where needed.

### Unfinished inventory is recorded in source and design
Decision: Add `ta-core` source metadata plus this design artifact's unfinished ledger so later sessions can implement group by group.

## Architecture
### crates/ta-core/src/common.rs — NEW
Shared output ranges, padding, validation, compact-to-padded helpers.
```rust
//! Shared TA-Lib core helpers.
//!
//! This module contains small, allocation-aware primitives shared by indicator
//! implementations: compact output ranges, padded-output conversion, and common
//! validation routines. Batch indicator kernels write compact TA-Lib-style output
//! buffers and return [`OutputRange`]; convenience wrappers use these helpers to
//! create full-length padded vectors.

use crate::{Float, Result, TalibError};

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec::Vec};

/// Location and length of valid compact output values within the original input.
///
/// TA-Lib C reports this as `outBegIdx` and `outNBElement`. Implementations in
/// this crate write valid values compactly starting at output index `0`, then
/// return the original beginning index and valid count in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputRange {
    /// Index in the original input where the first output value belongs.
    pub beg_idx: usize,
    /// Number of valid output elements written compactly from output index `0`.
    pub nb_element: usize,
}

impl OutputRange {
    /// Creates a new output range.
    #[inline]
    pub const fn new(beg_idx: usize, nb_element: usize) -> Self {
        Self { beg_idx, nb_element }
    }

    /// Returns an empty range at the start of the input.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            beg_idx: 0,
            nb_element: 0,
        }
    }

    /// Returns the exclusive end index in the original input.
    #[inline]
    pub const fn end_idx(&self) -> usize {
        self.beg_idx + self.nb_element
    }

    /// Returns true when no valid output elements were produced.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.nb_element == 0
    }
}

/// Padding value used by full-length convenience output vectors.
pub trait PadValue: Copy {
    /// Returns the value used for positions outside [`OutputRange`].
    fn pad_value() -> Self;
}

impl PadValue for Float {
    #[inline]
    fn pad_value() -> Self {
        Float::NAN
    }
}

impl PadValue for i32 {
    #[inline]
    fn pad_value() -> Self {
        0
    }
}

/// Returns the number of compact outputs for an input length and lookback.
#[inline]
pub fn output_count(input_len: usize, lookback: usize) -> usize {
    input_len.saturating_sub(lookback)
}

/// Validates that a period parameter is greater than zero.
#[inline]
pub fn validate_period(name: &str, period: usize) -> Result<()> {
    if period == 0 {
        return Err(TalibError::invalid_period(
            period,
            format!("{} must be greater than zero", name),
        ));
    }
    Ok(())
}

/// Validates a period parameter and returns its standard TA-Lib lookback.
#[inline]
pub fn period_lookback(name: &str, period: usize) -> Result<usize> {
    validate_period(name, period)?;
    Ok(period - 1)
}

/// Validates that an input slice is long enough for a lookback.
#[inline]
pub fn validate_input_len(input_len: usize, lookback: usize) -> Result<usize> {
    let count = output_count(input_len, lookback);
    if count == 0 && input_len > 0 {
        return Err(TalibError::insufficient_data(lookback + 1, input_len));
    }
    Ok(count)
}

/// Validates that a compact output buffer can hold `required` values.
#[inline]
pub fn validate_output_len(name: &str, output_len: usize, required: usize) -> Result<()> {
    if output_len < required {
        return Err(TalibError::invalid_input(format!(
            "{} output buffer too small: need {}, got {}",
            name, required, output_len
        )));
    }
    Ok(())
}

/// Validates that one slice length matches another.
#[inline]
pub fn validate_same_len(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> Result<()> {
    if left_len != right_len {
        return Err(TalibError::invalid_input(format!(
            "{} and {} must have the same length: got {} and {}",
            left_name, right_name, left_len, right_len
        )));
    }
    Ok(())
}

/// Validates that every named slice length matches the first entry.
pub fn validate_all_same_len(lengths: &[(&str, usize)]) -> Result<usize> {
    let Some((first_name, first_len)) = lengths.first().copied() else {
        return Ok(0);
    };

    for &(name, len) in &lengths[1..] {
        validate_same_len(first_name, first_len, name, len)?;
    }

    Ok(first_len)
}

/// Validates that every input value is finite.
pub fn validate_finite_slice(name: &str, values: &[Float]) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(TalibError::invalid_input(format!(
                "{}[{}] must be finite, got {}",
                name, idx, value
            )));
        }
    }
    Ok(())
}

/// Validates that all named input slices contain only finite values.
pub fn validate_finite_slices(slices: &[(&str, &[Float])]) -> Result<()> {
    for &(name, values) in slices {
        validate_finite_slice(name, values)?;
    }
    Ok(())
}

/// Builds a full-length padded vector from compact outputs and their range.
pub fn padded_from_compact<T>(input_len: usize, range: OutputRange, compact: &[T]) -> Vec<T>
where
    T: PadValue,
{
    let mut output = Vec::new();
    output.resize(input_len, T::pad_value());
    copy_compact_to_padded(range, compact, &mut output);
    output
}

/// Copies compact outputs into their full-length padded positions.
pub fn copy_compact_to_padded<T>(range: OutputRange, compact: &[T], padded: &mut [T])
where
    T: PadValue,
{
    let count = range.nb_element.min(compact.len());
    let end = range.beg_idx + count;
    if end <= padded.len() {
        padded[range.beg_idx..end].copy_from_slice(&compact[..count]);
    }
}

/// Allocates a compact output buffer large enough for any first-tranche function.
pub fn compact_buffer<T>(input_len: usize) -> Vec<T>
where
    T: PadValue,
{
    let mut output = Vec::new();
    output.resize(input_len, T::pad_value());
    output
}
```

### crates/ta-core/src/traits.rs:1-255 — MODIFY
Rewrite trait docs and add zero-copy `compute` plus padded convenience behavior.
```rust
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
    common::{output_count, validate_output_len},
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
        let count = output_count(inputs.len(), lookback);
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
```

### crates/ta-core/src/lib.rs:32-41 — MODIFY
Add new modules/re-exports as slices introduce them.
```rust
pub mod common;
pub mod error;
/// Official TA-Lib function inventory and implementation status.
pub mod inventory;
/// Math Operators functions.
pub mod math_operators;
/// Math Transform functions.
pub mod math_transform;
/// Overlap studies: Moving averages and other price overlay indicators.
pub mod overlap;
/// Price Transform functions.
pub mod price_transform;
pub mod simd;
pub mod traits;
pub mod types;

pub use common::{
    compact_buffer, copy_compact_to_padded, output_count, padded_from_compact, period_lookback,
    validate_all_same_len, validate_finite_slice, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, validate_same_len, OutputRange, PadValue,
};
pub use error::{Result, TalibError};
pub use traits::{Indicator, Resettable};
pub use types::Float;
```

### crates/ta-core/src/overlap/mod.rs:1-6 — MODIFY
Re-export rewritten SMA free function and struct surface.
```rust
//! Overlap Studies: moving averages and other price overlay indicators.

mod sma;

pub use sma::{SMA, SMA_vec};
```

### crates/ta-core/src/overlap/sma.rs:1-143 — MODIFY
Rewrite SMA on the new dual API contract.
```rust
//! Simple Moving Average (SMA).
//!
//! This module exposes both the TA-Lib-style zero-copy function [`SMA`] and the
//! stateful [`SMA`] struct. The free function writes compact valid outputs and
//! returns an [`OutputRange`](crate::OutputRange); [`SMA_vec`] returns a
//! full-length padded vector for convenience.

use crate::{
    compact_buffer, output_count, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Simple Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn SMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("SMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let inv_period = 1.0 as Float / timeperiod as Float;
    let mut window_sum: Float = real[..timeperiod].iter().copied().sum();
    out_real[0] = window_sum * inv_period;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = window_sum * inv_period;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes SMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn SMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = SMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Simple Moving Average indicator.
#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
    inv_period: Float,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
}

impl SMA {
    /// Creates a new SMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);

        Ok(Self {
            period: timeperiod,
            inv_period: 1.0 as Float / timeperiod as Float,
            buffer,
            index: 0,
            count: 0,
            sum: 0.0 as Float,
        })
    }

    /// Creates a new SMA indicator seeded from the most recent `timeperiod` values.
    pub fn from_data(timeperiod: usize, real: &[Float]) -> Result<Self> {
        validate_finite_slice("real", real)?;
        let mut sma = Self::new(timeperiod)?;
        let start = real.len().saturating_sub(timeperiod);
        for &value in &real[start..] {
            sma.next(value);
        }
        Ok(sma)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact SMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        SMA(real, self.period, out_real)
    }

    /// Computes full-length padded SMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        SMA_vec(real, self.period)
    }

    /// Checked streaming update that rejects non-finite inputs.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        if !input.is_finite() {
            return Err(TalibError::invalid_input("SMA input must be finite"));
        }
        Ok(self.next(input))
    }
}

impl Indicator for SMA {
    type Input = Float;
    type Output = Float;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<OutputRange> {
        SMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        SMA_vec(inputs, self.period)
    }

    #[inline]
    fn next(&mut self, input: Float) -> Float {
        if self.count < self.period {
            self.buffer[self.index] = input;
            self.sum += input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;

            if self.count < self.period {
                return Float::NAN;
            }

            return self.sum * self.inv_period;
        }

        let old = self.buffer[self.index];
        self.buffer[self.index] = input;
        self.sum += input - old;
        self.index = (self.index + 1) % self.period;
        self.sum * self.inv_period
    }
}

impl Resettable for SMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
    }
}
```

### crates/ta-core/tests/overlap_sma.rs — NEW
SMA integration coverage for zero-copy, padded wrapper, streaming, reset, and errors.
```rust
use ta_core::overlap::{SMA, SMA_vec};
use ta_core::{Float, Indicator, OutputRange, Resettable};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn sma_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0; 5];

    let range = SMA(&real, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 2.0);
    assert_close(output[1], 3.0);
    assert_close(output[2], 4.0);
}

#[test]
fn sma_vec_returns_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];

    let output = SMA_vec(&real, 3).unwrap();

    assert_eq!(output.len(), real.len());
    assert!(output[0].is_nan());
    assert!(output[1].is_nan());
    assert_close(output[2], 2.0);
    assert_close(output[3], 3.0);
    assert_close(output[4], 4.0);
}

#[test]
fn sma_struct_implements_indicator_compute() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sma = SMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&sma, &real, &mut compact).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 2.0);
    assert_close(compact[2], 4.0);
}

#[test]
fn sma_rejects_invalid_parameters_and_inputs() {
    assert!(SMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(SMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(SMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(SMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(SMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}

#[test]
fn sma_streaming_next_and_reset_are_safe() {
    let mut sma = SMA::new(3).unwrap();

    assert!(sma.next_checked(1.0).unwrap().is_nan());
    assert!(sma.next_checked(2.0).unwrap().is_nan());
    assert_close(sma.next_checked(3.0).unwrap(), 2.0);
    assert_close(sma.next_checked(4.0).unwrap(), 3.0);

    sma.reset();
    assert!(sma.next_checked(10.0).unwrap().is_nan());
    assert!(sma.next_checked(Float::NAN).is_err());
}
```

### crates/ta-core/src/price_transform/mod.rs — NEW
Price Transform module wiring and public re-exports.
```rust
//! Price Transform functions.
//!
//! These functions transform price-series inputs into derived real-valued series.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod avgdev;
mod avgprice;
mod medprice;
mod typprice;
mod wclprice;

pub use avgdev::{AVGDEV, AVGDEV_vec};
pub use avgprice::{AVGPRICE, AVGPRICE_vec};
pub use medprice::{MEDPRICE, MEDPRICE_vec};
pub use typprice::{TYPPRICE, TYPPRICE_vec};
pub use wclprice::{WCLPRICE, WCLPRICE_vec};
```

### crates/ta-core/src/price_transform/avgdev.rs — NEW
`AVGDEV` implementation.
```rust
//! Average Deviation (AVGDEV).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Average Deviation batch function.
#[allow(non_snake_case)]
pub fn AVGDEV(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("AVGDEV", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let period = timeperiod as Float;
    for output_idx in 0..count {
        let window = &real[output_idx..output_idx + timeperiod];
        let mean = window.iter().copied().sum::<Float>() / period;
        let deviation = window
            .iter()
            .map(|value| (*value - mean).abs())
            .sum::<Float>()
            / period;
        out_real[output_idx] = deviation;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Average Deviation into a full-length vector.
#[allow(non_snake_case)]
pub fn AVGDEV_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = AVGDEV(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

/// Average Deviation indicator.
#[derive(Debug, Clone)]
pub struct AVGDEV {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl AVGDEV {
    /// Creates a new Average Deviation indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        let mut buffer = Vec::new();
        buffer.resize(timeperiod, 0.0 as Float);
        Ok(Self {
            period: timeperiod,
            buffer,
            index: 0,
            count: 0,
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs.
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        AVGDEV(real, self.period, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        AVGDEV_vec(real, self.period)
    }
}

impl Indicator for AVGDEV {
    type Input = Float;
    type Output = Float;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute(&self, inputs: &[Self::Input], outputs: &mut [Self::Output]) -> Result<OutputRange> {
        AVGDEV(inputs, self.period, outputs)
    }

    fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        AVGDEV_vec(inputs, self.period)
    }

    fn next(&mut self, input: Float) -> Float {
        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Float::NAN;
        }

        let mean = self.buffer.iter().copied().sum::<Float>() / self.period as Float;
        self.buffer
            .iter()
            .map(|value| (*value - mean).abs())
            .sum::<Float>()
            / self.period as Float
    }
}
```

### crates/ta-core/src/price_transform/avgprice.rs — NEW
`AVGPRICE` implementation.
```rust
//! Average Price (AVGPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Average Price batch function: `(open + high + low + close) / 4`.
#[allow(non_snake_case)]
pub fn AVGPRICE(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_all_same_len(&[
        ("open", open.len()),
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("open", open), ("high", high), ("low", low), ("close", close)])?;
    validate_output_len("AVGPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (open[idx] + high[idx] + low[idx] + close[idx]) / 4.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Average Price into a full-length vector.
#[allow(non_snake_case)]
pub fn AVGPRICE_vec(
    open: &[Float],
    high: &[Float],
    low: &[Float],
    close: &[Float],
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(open.len());
    let range = AVGPRICE(open, high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        open.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Average Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct AVGPRICE {
    _private: (),
}

impl AVGPRICE {
    /// Creates an Average Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        AVGPRICE(open, high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        open: &[Float],
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        AVGPRICE_vec(open, high, low, close)
    }
}
```

### crates/ta-core/src/price_transform/medprice.rs — NEW
`MEDPRICE` implementation.
```rust
//! Median Price (MEDPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Median Price batch function: `(high + low) / 2`.
#[allow(non_snake_case)]
pub fn MEDPRICE(high: &[Float], low: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let len = validate_all_same_len(&[("high", high.len()), ("low", low.len())])?;
    validate_finite_slices(&[("high", high), ("low", low)])?;
    validate_output_len("MEDPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx]) / 2.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Median Price into a full-length vector.
#[allow(non_snake_case)]
pub fn MEDPRICE_vec(high: &[Float], low: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = MEDPRICE(high, low, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Median Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct MEDPRICE {
    _private: (),
}

impl MEDPRICE {
    /// Creates a Median Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        MEDPRICE(high, low, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, high: &[Float], low: &[Float]) -> Result<Vec<Float>> {
        MEDPRICE_vec(high, low)
    }
}
```

### crates/ta-core/src/price_transform/typprice.rs — NEW
`TYPPRICE` implementation.
```rust
//! Typical Price (TYPPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Typical Price batch function: `(high + low + close) / 3`.
#[allow(non_snake_case)]
pub fn TYPPRICE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    validate_output_len("TYPPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx] + close[idx]) / 3.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Typical Price into a full-length vector.
#[allow(non_snake_case)]
pub fn TYPPRICE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = TYPPRICE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Typical Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct TYPPRICE {
    _private: (),
}

impl TYPPRICE {
    /// Creates a Typical Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        TYPPRICE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        TYPPRICE_vec(high, low, close)
    }
}
```

### crates/ta-core/src/price_transform/wclprice.rs — NEW
`WCLPRICE` implementation.
```rust
//! Weighted Close Price (WCLPRICE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Weighted Close Price batch function: `(high + low + 2 * close) / 4`.
#[allow(non_snake_case)]
pub fn WCLPRICE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    validate_output_len("WCLPRICE", out_real.len(), len)?;

    for idx in 0..len {
        out_real[idx] = (high[idx] + low[idx] + 2.0 as Float * close[idx]) / 4.0 as Float;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes Weighted Close Price into a full-length vector.
#[allow(non_snake_case)]
pub fn WCLPRICE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = WCLPRICE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Weighted Close Price struct surface.
#[derive(Debug, Clone, Copy)]
pub struct WCLPRICE {
    _private: (),
}

impl WCLPRICE {
    /// Creates a Weighted Close Price calculator.
    pub fn new() -> Result<Self> {
        Ok(Self { _private: () })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        WCLPRICE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        WCLPRICE_vec(high, low, close)
    }
}
```

### crates/ta-core/tests/price_transform.rs — NEW
Price Transform integration tests.
```rust
use ta_core::price_transform::{
    AVGDEV, AVGDEV_vec, AVGPRICE, AVGPRICE_vec, MEDPRICE, MEDPRICE_vec, TYPPRICE, TYPPRICE_vec,
    WCLPRICE, WCLPRICE_vec,
};
use ta_core::{Float, Indicator, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn avgprice_medprice_typprice_wclprice_compute_expected_values() {
    let open = [1.0, 2.0, 3.0];
    let high = [2.0, 3.0, 4.0];
    let low = [0.0, 1.0, 2.0];
    let close = [1.5, 2.5, 3.5];
    let mut output = [0.0; 3];

    assert_eq!(
        AVGPRICE(&open, &high, &low, &close, &mut output).unwrap(),
        OutputRange::new(0, 3)
    );
    assert_close(output[0], 1.125);
    assert_close(output[2], 3.125);

    MEDPRICE(&high, &low, &mut output).unwrap();
    assert_close(output[0], 1.0);
    assert_close(output[2], 3.0);

    TYPPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.1666666666666667);
    assert_close(output[2], 3.1666666666666665);

    WCLPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.25);
    assert_close(output[2], 3.25);
}

#[test]
fn price_transform_vec_wrappers_preserve_length() {
    let open = [1.0, 2.0];
    let high = [2.0, 3.0];
    let low = [0.0, 1.0];
    let close = [1.5, 2.5];

    assert_eq!(AVGPRICE_vec(&open, &high, &low, &close).unwrap().len(), 2);
    assert_eq!(MEDPRICE_vec(&high, &low).unwrap().len(), 2);
    assert_eq!(TYPPRICE_vec(&high, &low, &close).unwrap().len(), 2);
    assert_eq!(WCLPRICE_vec(&high, &low, &close).unwrap().len(), 2);
}

#[test]
fn avgdev_computes_compact_and_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0; 5];

    let range = AVGDEV(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 2.0 / 3.0);
    assert_close(output[2], 2.0 / 3.0);

    let padded = AVGDEV_vec(&real, 3).unwrap();
    assert!(padded[0].is_nan());
    assert!(padded[1].is_nan());
    assert_close(padded[2], 2.0 / 3.0);
}

#[test]
fn price_transform_struct_surfaces_work() {
    let high = [2.0, 3.0, 4.0];
    let low = [0.0, 1.0, 2.0];
    let close = [1.5, 2.5, 3.5];
    let mut output = [0.0; 3];

    let typprice = TYPPRICE::new().unwrap();
    typprice.compute(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.1666666666666667);

    let avgdev = AVGDEV::new(3).unwrap();
    let range = Indicator::compute(&avgdev, &[1.0, 2.0, 3.0], &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 1));
}

#[test]
fn price_transform_rejects_bad_lengths_and_non_finite_inputs() {
    let mut output = [0.0; 3];
    assert!(MEDPRICE(&[1.0, 2.0], &[1.0], &mut output).is_err());
    assert!(TYPPRICE(&[1.0, Float::NAN], &[1.0, 2.0], &[1.0, 2.0], &mut output).is_err());
    assert!(AVGDEV(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(AVGDEV::new(0).is_err());
}
```

### crates/ta-core/src/math_transform/mod.rs — NEW
Math Transform unary function implementations and struct surfaces.
```rust
//! Math Transform functions.
//!
//! These are unary element-wise transforms over one real input series. They use
//! strict finite-input validation, compact zero-copy output buffers, and padded
//! convenience wrappers matching the first-tranche core contract.

mod functions {
    use crate::{
        compact_buffer, padded_from_compact, validate_finite_slice, validate_output_len, Float,
        Indicator, OutputRange, Result,
    };

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    macro_rules! define_transform {
        ($name:ident, $vec_name:ident, $operation:expr) => {
            #[doc = concat!("TA-Lib-style ", stringify!($name), " unary transform.")]
            #[allow(non_snake_case)]
            pub fn $name(real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                validate_finite_slice("real", real)?;
                validate_output_len(stringify!($name), out_real.len(), real.len())?;
                let operation = $operation;
                for (idx, value) in real.iter().copied().enumerate() {
                    out_real[idx] = operation(value);
                }
                Ok(OutputRange::new(0, real.len()))
            }

            #[doc = concat!("Computes ", stringify!($name), " into a full-length vector.")]
            #[allow(non_snake_case)]
            pub fn $vec_name(real: &[Float]) -> Result<Vec<Float>> {
                let mut compact = compact_buffer::<Float>(real.len());
                let range = $name(real, &mut compact)?;
                Ok(padded_from_compact(
                    real.len(),
                    range,
                    &compact[..range.nb_element],
                ))
            }

            #[doc = concat!(stringify!($name), " struct surface.")]
            #[derive(Debug, Clone, Copy)]
            pub struct $name {
                _private: (),
            }

            impl $name {
                #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
                pub fn new() -> Result<Self> {
                    Ok(Self { _private: () })
                }

                /// Computes compact outputs.
                pub fn compute(
                    &self,
                    real: &[Float],
                    out_real: &mut [Float],
                ) -> Result<OutputRange> {
                    $name(real, out_real)
                }

                /// Computes full-length outputs.
                pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                    $vec_name(real)
                }
            }

            impl Indicator for $name {
                type Input = Float;
                type Output = Float;

                fn lookback(&self) -> usize {
                    0
                }

                fn compute(
                    &self,
                    inputs: &[Self::Input],
                    outputs: &mut [Self::Output],
                ) -> Result<OutputRange> {
                    $name(inputs, outputs)
                }

                fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                    $vec_name(inputs)
                }

                fn next(&mut self, input: Float) -> Float {
                    let operation = $operation;
                    operation(input)
                }
            }
        };
    }

    define_transform!(ACOS, ACOS_vec, |value: Float| value.acos());
    define_transform!(ASIN, ASIN_vec, |value: Float| value.asin());
    define_transform!(ATAN, ATAN_vec, |value: Float| value.atan());
    define_transform!(CEIL, CEIL_vec, |value: Float| value.ceil());
    define_transform!(COS, COS_vec, |value: Float| value.cos());
    define_transform!(COSH, COSH_vec, |value: Float| value.cosh());
    define_transform!(EXP, EXP_vec, |value: Float| value.exp());
    define_transform!(FLOOR, FLOOR_vec, |value: Float| value.floor());
    define_transform!(LN, LN_vec, |value: Float| value.ln());
    define_transform!(LOG10, LOG10_vec, |value: Float| value.log10());
    define_transform!(SIN, SIN_vec, |value: Float| value.sin());
    define_transform!(SINH, SINH_vec, |value: Float| value.sinh());
    define_transform!(SQRT, SQRT_vec, |value: Float| value.sqrt());
    define_transform!(TAN, TAN_vec, |value: Float| value.tan());
    define_transform!(TANH, TANH_vec, |value: Float| value.tanh());
}

pub use functions::{
    ACOS, ACOS_vec, ASIN, ASIN_vec, ATAN, ATAN_vec, CEIL, CEIL_vec, COS, COSH, COSH_vec,
    COS_vec, EXP, EXP_vec, FLOOR, FLOOR_vec, LN, LN_vec, LOG10, LOG10_vec, SIN, SINH,
    SINH_vec, SIN_vec, SQRT, SQRT_vec, TAN, TANH, TANH_vec, TAN_vec,
};
```

### crates/ta-core/tests/math_transform.rs — NEW
Math Transform integration tests.
```rust
use ta_core::math_transform::{
    ACOS, ACOS_vec, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT,
    SQRT_vec, TAN, TANH,
};
use ta_core::{Float, Indicator, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn math_transform_functions_compute_expected_values() {
    let real = [0.5 as Float];
    let mut output = [0.0 as Float; 1];

    assert_eq!(SIN(&real, &mut output).unwrap(), OutputRange::new(0, 1));
    assert_close(output[0], (0.5 as Float).sin());

    COS(&real, &mut output).unwrap();
    assert_close(output[0], (0.5 as Float).cos());

    ACOS(&real, &mut output).unwrap();
    assert_close(output[0], (0.5 as Float).acos());

    SQRT(&[4.0 as Float], &mut output).unwrap();
    assert_close(output[0], 2.0 as Float);

    LN(&[(2.0 as Float).exp()], &mut output).unwrap();
    assert_close(output[0], 2.0 as Float);
}

#[test]
fn all_math_transform_functions_are_exported() {
    let real = [0.5 as Float];
    let mut output = [0.0 as Float; 1];

    let funcs: [fn(&[Float], &mut [Float]) -> ta_core::Result<OutputRange>; 15] = [
        ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
    ];

    for func in funcs {
        assert_eq!(func(&real, &mut output).unwrap(), OutputRange::new(0, 1));
    }
}

#[test]
fn math_transform_vec_wrappers_preserve_length() {
    let real = [1.0 as Float, 4.0 as Float, 9.0 as Float];

    assert_eq!(SQRT_vec(&real).unwrap().len(), 3);
    assert_eq!(ACOS_vec(&[0.5 as Float, 1.0 as Float]).unwrap().len(), 2);
}

#[test]
fn math_transform_struct_surface_implements_indicator() {
    let sqrt = SQRT::new().unwrap();
    let mut output = [0.0 as Float; 2];
    let range = Indicator::compute(&sqrt, &[4.0 as Float, 9.0 as Float], &mut output).unwrap();

    assert_eq!(range, OutputRange::new(0, 2));
    assert_close(output[0], 2.0 as Float);
    assert_close(output[1], 3.0 as Float);
}

#[test]
fn math_transform_rejects_non_finite_inputs() {
    let mut output = [0.0 as Float; 1];

    assert!(SIN(&[Float::NAN], &mut output).is_err());
    assert!(COS(&[Float::INFINITY], &mut output).is_err());
}
```

### crates/ta-core/src/math_operators/mod.rs — NEW
Math Operators module wiring and re-exports.
```rust
//! Math Operators functions.
//!
//! This module contains TA-Lib math operators over one or two real input series.
//! Rolling and multi-output functions use compact outputs and `OutputRange`.

mod arithmetic;
mod extrema;
mod rolling;

pub use arithmetic::{ADD, ADD_vec, DIV, DIV_vec, MULT, MULT_vec, SUB, SUB_vec};
pub use extrema::{
    MAXINDEX, MAXINDEX_vec, MININDEX, MININDEX_vec, MINMAX, MINMAX_vec, MINMAXINDEX,
    MINMAXINDEX_vec, MINMAXINDEXOutput, MINMAXOutput,
};
pub use rolling::{MAX, MAX_vec, MIN, MIN_vec, SUM, SUM_vec};
```

### crates/ta-core/src/math_operators/arithmetic.rs — NEW
`ADD`, `SUB`, `MULT`, `DIV` implementations.
```rust
//! Arithmetic Math Operators.

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

macro_rules! define_binary_operator {
    ($name:ident, $vec_name:ident, $operation:expr) => {
        #[doc = concat!("TA-Lib-style ", stringify!($name), " binary math operator.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real0: &[Float],
            real1: &[Float],
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
            validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
            validate_output_len(stringify!($name), out_real.len(), len)?;
            let operation = $operation;
            for idx in 0..len {
                out_real[idx] = operation(real0[idx], real1[idx]);
            }
            Ok(OutputRange::new(0, len))
        }

        #[doc = concat!("Computes ", stringify!($name), " into a full-length vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real0.len());
            let range = $name(real0, real1, &mut compact)?;
            Ok(padded_from_compact(
                real0.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            _private: (),
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new() -> Result<Self> {
                Ok(Self { _private: () })
            }

            /// Computes compact outputs.
            pub fn compute(
                &self,
                real0: &[Float],
                real1: &[Float],
                out_real: &mut [Float],
            ) -> Result<OutputRange> {
                $name(real0, real1, out_real)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real0, real1)
            }
        }
    };
}

define_binary_operator!(ADD, ADD_vec, |left: Float, right: Float| left + right);
define_binary_operator!(SUB, SUB_vec, |left: Float, right: Float| left - right);
define_binary_operator!(MULT, MULT_vec, |left: Float, right: Float| left * right);
define_binary_operator!(DIV, DIV_vec, |left: Float, right: Float| left / right);
```

### crates/ta-core/src/math_operators/rolling.rs — NEW
`SUM`, `MIN`, `MAX` rolling-window implementations.
```rust
//! Rolling-window Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

fn rolling_apply<F>(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
    mut aggregate: F,
) -> Result<OutputRange>
where
    F: FnMut(&[Float]) -> Float,
{
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        out_real[output_idx] = aggregate(&real[output_idx..output_idx + timeperiod]);
    }

    Ok(OutputRange::new(lookback, count))
}

fn sum_window(window: &[Float]) -> Float {
    window.iter().copied().sum()
}

fn min_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::min)
}

fn max_window(window: &[Float]) -> Float {
    window.iter().copied().fold(window[0], Float::max)
}

/// TA-Lib-style rolling sum.
#[allow(non_snake_case)]
pub fn SUM(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_apply("SUM", real, timeperiod, out_real, sum_window)
}

/// Computes rolling sum into a full-length vector.
#[allow(non_snake_case)]
pub fn SUM_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = SUM(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

/// TA-Lib-style rolling minimum.
#[allow(non_snake_case)]
pub fn MIN(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_apply("MIN", real, timeperiod, out_real, min_window)
}

/// Computes rolling minimum into a full-length vector.
#[allow(non_snake_case)]
pub fn MIN_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MIN(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

/// TA-Lib-style rolling maximum.
#[allow(non_snake_case)]
pub fn MAX(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    rolling_apply("MAX", real, timeperiod, out_real, max_window)
}

/// Computes rolling maximum into a full-length vector.
#[allow(non_snake_case)]
pub fn MAX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MAX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

macro_rules! define_rolling_struct {
    ($name:ident, $vec_name:ident, $aggregate:ident) => {
        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            buffer: Vec<Float>,
            index: usize,
            count: usize,
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                period_lookback("timeperiod", timeperiod)?;
                let mut buffer = Vec::new();
                buffer.resize(timeperiod, 0.0 as Float);
                Ok(Self {
                    period: timeperiod,
                    buffer,
                    index: 0,
                    count: 0,
                })
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Computes compact outputs.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, out_real)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period)
            }
        }

        impl Indicator for $name {
            type Input = Float;
            type Output = Float;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute(
                &self,
                inputs: &[Self::Input],
                outputs: &mut [Self::Output],
            ) -> Result<OutputRange> {
                $name(inputs, self.period, outputs)
            }

            fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                $vec_name(inputs, self.period)
            }

            fn next(&mut self, input: Float) -> Float {
                self.buffer[self.index] = input;
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;

                if self.count < self.period {
                    return Float::NAN;
                }

                $aggregate(&self.buffer)
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                for value in &mut self.buffer {
                    *value = 0.0 as Float;
                }
                self.index = 0;
                self.count = 0;
            }
        }
    };
}

define_rolling_struct!(SUM, SUM_vec, sum_window);
define_rolling_struct!(MIN, MIN_vec, min_window);
define_rolling_struct!(MAX, MAX_vec, max_window);
```

### crates/ta-core/src/math_operators/extrema.rs — NEW
`MININDEX`, `MAXINDEX`, `MINMAX`, `MINMAXINDEX` implementations.
```rust
//! Extrema and extrema-index Math Operators.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Full-length MINMAX output vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct MINMAXOutput {
    /// Minimum values.
    pub min: Vec<Float>,
    /// Maximum values.
    pub max: Vec<Float>,
}

/// Full-length MINMAXINDEX output vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MINMAXINDEXOutput {
    /// Absolute minimum indexes.
    pub min_idx: Vec<i32>,
    /// Absolute maximum indexes.
    pub max_idx: Vec<i32>,
}

fn window_min_max(window: &[Float], offset: usize) -> (Float, Float, i32, i32) {
    let mut min_value = window[0];
    let mut max_value = window[0];
    let mut min_idx = offset;
    let mut max_idx = offset;

    for (local_idx, value) in window.iter().copied().enumerate().skip(1) {
        let absolute_idx = offset + local_idx;
        if value < min_value {
            min_value = value;
            min_idx = absolute_idx;
        }
        if value > max_value {
            max_value = value;
            max_idx = absolute_idx;
        }
    }

    (min_value, max_value, min_idx as i32, max_idx as i32)
}

fn validate_window(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

fn select_stream_index<F>(values: &[Float], indexes: &[usize], mut is_better: F) -> i32
where
    F: FnMut(Float, Float) -> bool,
{
    let mut best_value = values[0];
    let mut best_index = indexes[0];

    for (&value, &index) in values.iter().zip(indexes.iter()).skip(1) {
        if is_better(value, best_value) || (value == best_value && index < best_index) {
            best_value = value;
            best_index = index;
        }
    }

    best_index as i32
}

/// TA-Lib-style rolling minimum index.
#[allow(non_snake_case)]
pub fn MININDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MININDEX", out_integer.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, min_idx, _) = window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_integer[output_idx] = min_idx;
    }
    Ok(OutputRange::new(lookback, count))
}

/// Computes rolling minimum indexes into a full-length vector padded with zeroes.
#[allow(non_snake_case)]
pub fn MININDEX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<i32>> {
    let mut compact = compact_buffer::<i32>(real.len());
    let range = MININDEX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

/// TA-Lib-style rolling maximum index.
#[allow(non_snake_case)]
pub fn MAXINDEX(real: &[Float], timeperiod: usize, out_integer: &mut [i32]) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MAXINDEX", out_integer.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, _, max_idx) = window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_integer[output_idx] = max_idx;
    }
    Ok(OutputRange::new(lookback, count))
}

/// Computes rolling maximum indexes into a full-length vector padded with zeroes.
#[allow(non_snake_case)]
pub fn MAXINDEX_vec(real: &[Float], timeperiod: usize) -> Result<Vec<i32>> {
    let mut compact = compact_buffer::<i32>(real.len());
    let range = MAXINDEX(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(real.len(), range, &compact[..range.nb_element]))
}

/// TA-Lib-style rolling minimum and maximum.
#[allow(non_snake_case)]
pub fn MINMAX(
    real: &[Float],
    timeperiod: usize,
    out_min: &mut [Float],
    out_max: &mut [Float],
) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MINMAX min", out_min.len(), count)?;
    validate_output_len("MINMAX max", out_max.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (min_value, max_value, _, _) = window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_min[output_idx] = min_value;
        out_max[output_idx] = max_value;
    }
    Ok(OutputRange::new(lookback, count))
}

/// Computes rolling minimum and maximum into full-length vectors.
#[allow(non_snake_case)]
pub fn MINMAX_vec(real: &[Float], timeperiod: usize) -> Result<MINMAXOutput> {
    let mut min_compact = compact_buffer::<Float>(real.len());
    let mut max_compact = compact_buffer::<Float>(real.len());
    let range = MINMAX(real, timeperiod, &mut min_compact, &mut max_compact)?;
    Ok(MINMAXOutput {
        min: padded_from_compact(real.len(), range, &min_compact[..range.nb_element]),
        max: padded_from_compact(real.len(), range, &max_compact[..range.nb_element]),
    })
}

/// TA-Lib-style rolling minimum and maximum indexes.
#[allow(non_snake_case)]
pub fn MINMAXINDEX(
    real: &[Float],
    timeperiod: usize,
    out_min_idx: &mut [i32],
    out_max_idx: &mut [i32],
) -> Result<OutputRange> {
    let (lookback, count) = validate_window(real, timeperiod)?;
    validate_output_len("MINMAXINDEX min", out_min_idx.len(), count)?;
    validate_output_len("MINMAXINDEX max", out_max_idx.len(), count)?;
    if count == 0 {
        return Ok(OutputRange::empty());
    }
    for output_idx in 0..count {
        let (_, _, min_idx, max_idx) = window_min_max(&real[output_idx..output_idx + timeperiod], output_idx);
        out_min_idx[output_idx] = min_idx;
        out_max_idx[output_idx] = max_idx;
    }
    Ok(OutputRange::new(lookback, count))
}

/// Computes rolling minimum and maximum indexes into full-length vectors.
#[allow(non_snake_case)]
pub fn MINMAXINDEX_vec(real: &[Float], timeperiod: usize) -> Result<MINMAXINDEXOutput> {
    let mut min_compact = compact_buffer::<i32>(real.len());
    let mut max_compact = compact_buffer::<i32>(real.len());
    let range = MINMAXINDEX(real, timeperiod, &mut min_compact, &mut max_compact)?;
    Ok(MINMAXINDEXOutput {
        min_idx: padded_from_compact(real.len(), range, &min_compact[..range.nb_element]),
        max_idx: padded_from_compact(real.len(), range, &max_compact[..range.nb_element]),
    })
}

macro_rules! define_index_struct {
    ($name:ident, $vec_name:ident, $is_better:expr) => {
        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            buffer: Vec<Float>,
            indexes: Vec<usize>,
            index: usize,
            count: usize,
            seen: usize,
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                period_lookback("timeperiod", timeperiod)?;
                let mut buffer = Vec::new();
                buffer.resize(timeperiod, 0.0 as Float);
                let mut indexes = Vec::new();
                indexes.resize(timeperiod, 0);
                Ok(Self {
                    period: timeperiod,
                    buffer,
                    indexes,
                    index: 0,
                    count: 0,
                    seen: 0,
                })
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Computes compact outputs.
            pub fn compute(&self, real: &[Float], out_integer: &mut [i32]) -> Result<OutputRange> {
                $name(real, self.period, out_integer)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<i32>> {
                $vec_name(real, self.period)
            }
        }

        impl Indicator for $name {
            type Input = Float;
            type Output = i32;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute(
                &self,
                inputs: &[Self::Input],
                outputs: &mut [Self::Output],
            ) -> Result<OutputRange> {
                $name(inputs, self.period, outputs)
            }

            fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                $vec_name(inputs, self.period)
            }

            fn next(&mut self, input: Float) -> i32 {
                self.buffer[self.index] = input;
                self.indexes[self.index] = self.seen;
                self.seen = self.seen.saturating_add(1);
                if self.count < self.period {
                    self.count += 1;
                }
                self.index = (self.index + 1) % self.period;

                if self.count < self.period {
                    return 0;
                }

                let is_better = $is_better;
                select_stream_index(&self.buffer[..self.count], &self.indexes[..self.count], is_better)
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                for value in &mut self.buffer {
                    *value = 0.0 as Float;
                }
                for index in &mut self.indexes {
                    *index = 0;
                }
                self.index = 0;
                self.count = 0;
                self.seen = 0;
            }
        }
    };
}

define_index_struct!(MININDEX, MININDEX_vec, |candidate: Float, current: Float| candidate < current);
define_index_struct!(MAXINDEX, MAXINDEX_vec, |candidate: Float, current: Float| candidate > current);

/// MINMAX struct surface.
#[derive(Debug, Clone, Copy)]
pub struct MINMAX {
    period: usize,
}

impl MINMAX {
    /// Creates a MINMAX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs into parallel buffers.
    pub fn compute(
        &self,
        real: &[Float],
        out_min: &mut [Float],
        out_max: &mut [Float],
    ) -> Result<OutputRange> {
        MINMAX(real, self.period, out_min, out_max)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXOutput> {
        MINMAX_vec(real, self.period)
    }
}

/// MINMAXINDEX struct surface.
#[derive(Debug, Clone, Copy)]
pub struct MINMAXINDEX {
    period: usize,
}

impl MINMAXINDEX {
    /// Creates a MINMAXINDEX calculator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self { period: timeperiod })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs into parallel buffers.
    pub fn compute(
        &self,
        real: &[Float],
        out_min_idx: &mut [i32],
        out_max_idx: &mut [i32],
    ) -> Result<OutputRange> {
        MINMAXINDEX(real, self.period, out_min_idx, out_max_idx)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<MINMAXINDEXOutput> {
        MINMAXINDEX_vec(real, self.period)
    }
}
```

### crates/ta-core/tests/math_operators.rs — NEW
Math Operators integration tests.
```rust
use ta_core::math_operators::{
    ADD, ADD_vec, DIV, MAX, MAXINDEX, MAXINDEX_vec, MIN, MININDEX, MINMAX, MINMAXINDEX,
    MINMAXINDEX_vec, MINMAX_vec, MULT, SUB, SUM, SUM_vec,
};
use ta_core::{Float, Indicator, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn arithmetic_operators_compute_expected_values() {
    let real0 = [8.0 as Float, 6.0 as Float, 4.0 as Float];
    let real1 = [2.0 as Float, 3.0 as Float, 4.0 as Float];
    let mut output = [0.0 as Float; 3];

    assert_eq!(ADD(&real0, &real1, &mut output).unwrap(), OutputRange::new(0, 3));
    assert_eq!(output, [10.0 as Float, 9.0 as Float, 8.0 as Float]);

    SUB(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [6.0 as Float, 3.0 as Float, 0.0 as Float]);

    MULT(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [16.0 as Float, 18.0 as Float, 16.0 as Float]);

    DIV(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [4.0 as Float, 2.0 as Float, 1.0 as Float]);
}

#[test]
fn rolling_operators_compute_compact_and_padded_outputs() {
    let real = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let mut output = [0.0 as Float; 4];

    assert_eq!(SUM(&real, 2, &mut output).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&output[..3], &[4.0 as Float, 5.0 as Float, 6.0 as Float]);

    MIN(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[1.0 as Float, 1.0 as Float, 2.0 as Float]);

    MAX(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[3.0 as Float, 4.0 as Float, 4.0 as Float]);

    let padded = SUM_vec(&real, 2).unwrap();
    assert!(padded[0].is_nan());
    assert_close(padded[1], 4.0 as Float);
    assert_close(padded[3], 6.0 as Float);
}

#[test]
fn extrema_index_functions_return_absolute_indexes() {
    let real = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let mut min_idx = [0; 4];
    let mut max_idx = [0; 4];

    assert_eq!(MININDEX(&real, 2, &mut min_idx).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&min_idx[..3], &[1, 1, 3]);

    assert_eq!(MAXINDEX(&real, 2, &mut max_idx).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&max_idx[..3], &[0, 2, 2]);

    let mut min = [0.0 as Float; 4];
    let mut max = [0.0 as Float; 4];
    assert_eq!(MINMAX(&real, 2, &mut min, &mut max).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&min[..3], &[1.0 as Float, 1.0 as Float, 2.0 as Float]);
    assert_eq!(&max[..3], &[3.0 as Float, 4.0 as Float, 4.0 as Float]);

    MINMAXINDEX(&real, 2, &mut min_idx, &mut max_idx).unwrap();
    assert_eq!(&min_idx[..3], &[1, 1, 3]);
    assert_eq!(&max_idx[..3], &[0, 2, 2]);
}

#[test]
fn vec_wrappers_preserve_length_and_padding() {
    let real0 = [1.0 as Float, 2.0 as Float, 3.0 as Float];
    let real1 = [3.0 as Float, 2.0 as Float, 1.0 as Float];

    assert_eq!(ADD_vec(&real0, &real1).unwrap(), vec![4.0 as Float, 4.0 as Float, 4.0 as Float]);

    let minmax = MINMAX_vec(&real0, 2).unwrap();
    assert_eq!(minmax.min.len(), 3);
    assert!(minmax.min[0].is_nan());
    assert_close(minmax.min[1], 1.0 as Float);
    assert_close(minmax.max[2], 3.0 as Float);

    let minmaxindex = MINMAXINDEX_vec(&real0, 2).unwrap();
    assert_eq!(minmaxindex.min_idx, vec![0, 0, 1]);
    assert_eq!(minmaxindex.max_idx, vec![0, 1, 2]);

    assert_eq!(MAXINDEX_vec(&real0, 2).unwrap(), vec![0, 1, 2]);
}

#[test]
fn struct_surfaces_work() {
    let real0 = [1.0 as Float, 2.0 as Float, 3.0 as Float];
    let real1 = [3.0 as Float, 2.0 as Float, 1.0 as Float];
    let mut output = [0.0 as Float; 3];

    let add = ADD::new().unwrap();
    add.compute(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [4.0 as Float, 4.0 as Float, 4.0 as Float]);

    let sum = SUM::new(2).unwrap();
    let range = Indicator::compute(&sum, &real0, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(1, 2));
    assert_eq!(&output[..2], &[3.0 as Float, 5.0 as Float]);

    let minindex = MININDEX::new(2).unwrap();
    let mut index_output = [0; 3];
    let range = Indicator::compute(&minindex, &real0, &mut index_output).unwrap();
    assert_eq!(range, OutputRange::new(1, 2));
    assert_eq!(&index_output[..2], &[0, 1]);

    let minmax = MINMAX::new(2).unwrap();
    let mut min = [0.0 as Float; 3];
    let mut max = [0.0 as Float; 3];
    minmax.compute(&real0, &mut min, &mut max).unwrap();
    assert_eq!(&min[..2], &[1.0 as Float, 2.0 as Float]);
    assert_eq!(&max[..2], &[2.0 as Float, 3.0 as Float]);
}

#[test]
fn operators_reject_bad_lengths_periods_and_non_finite_inputs() {
    let mut output = [0.0 as Float; 3];
    assert!(ADD(&[1.0 as Float, 2.0 as Float], &[1.0 as Float], &mut output).is_err());
    assert!(SUB(&[1.0 as Float, Float::NAN], &[1.0 as Float, 2.0 as Float], &mut output).is_err());
    assert!(SUM(&[1.0 as Float, 2.0 as Float], 0, &mut output).is_err());
    assert!(MAX(&[1.0 as Float], 2, &mut output).is_err());

    let mut min_output = [0.0 as Float; 3];
    let mut max_output = [0.0 as Float; 3];
    assert!(MINMAX(
        &[1.0 as Float, Float::INFINITY],
        2,
        &mut min_output,
        &mut max_output,
    )
    .is_err());

    assert!(MAXINDEX::new(0).is_err());
}
```

### crates/ta-core/src/inventory.rs — NEW
Source-level official TA-Lib function ledger and implementation status.
```rust
//! Official TA-Lib function inventory and implementation ledger.
//!
//! This source-level ledger records the 161-function TA-Lib surface that
//! `ta-core` intends to implement in Rust. It deliberately follows official
//! TA-Lib groups rather than the older local planning documents, so future work
//! can advance group-by-group without rediscovering scope.

/// Total official TA-Lib function count tracked by this ledger.
pub const FUNCTION_COUNT: usize = 161;

/// Number of functions implemented by the foundation + first-tranche design.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 32;

/// Official TA-Lib function group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionGroup {
    /// Overlap Studies.
    OverlapStudies,
    /// Momentum Indicators.
    MomentumIndicators,
    /// Volume Indicators.
    VolumeIndicators,
    /// Volatility Indicators.
    VolatilityIndicators,
    /// Price Transform.
    PriceTransform,
    /// Cycle Indicators.
    CycleIndicators,
    /// Pattern Recognition.
    PatternRecognition,
    /// Statistic Functions.
    StatisticFunctions,
    /// Math Transform.
    MathTransform,
    /// Math Operators.
    MathOperators,
}

impl FunctionGroup {
    /// All official groups in TA-Lib display order.
    pub const ALL: &'static [Self] = &[
        Self::OverlapStudies,
        Self::MomentumIndicators,
        Self::VolumeIndicators,
        Self::VolatilityIndicators,
        Self::PriceTransform,
        Self::CycleIndicators,
        Self::PatternRecognition,
        Self::StatisticFunctions,
        Self::MathTransform,
        Self::MathOperators,
    ];

    /// Human-readable official group label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OverlapStudies => "Overlap Studies",
            Self::MomentumIndicators => "Momentum Indicators",
            Self::VolumeIndicators => "Volume Indicators",
            Self::VolatilityIndicators => "Volatility Indicators",
            Self::PriceTransform => "Price Transform",
            Self::CycleIndicators => "Cycle Indicators",
            Self::PatternRecognition => "Pattern Recognition",
            Self::StatisticFunctions => "Statistic Functions",
            Self::MathTransform => "Math Transform",
            Self::MathOperators => "Math Operators",
        }
    }

    /// Expected official function count for this group.
    pub const fn expected_count(self) -> usize {
        match self {
            Self::OverlapStudies => 18,
            Self::MomentumIndicators => 31,
            Self::VolumeIndicators => 3,
            Self::VolatilityIndicators => 3,
            Self::PriceTransform => 5,
            Self::CycleIndicators => 5,
            Self::PatternRecognition => 61,
            Self::StatisticFunctions => 9,
            Self::MathTransform => 15,
            Self::MathOperators => 11,
        }
    }

    /// Rust module planned for this group.
    pub const fn rust_module(self) -> &'static str {
        match self {
            Self::OverlapStudies => "overlap",
            Self::MomentumIndicators => "momentum",
            Self::VolumeIndicators => "volume",
            Self::VolatilityIndicators => "volatility",
            Self::PriceTransform => "price_transform",
            Self::CycleIndicators => "cycle",
            Self::PatternRecognition => "pattern_recognition",
            Self::StatisticFunctions => "statistic",
            Self::MathTransform => "math_transform",
            Self::MathOperators => "math_operators",
        }
    }
}

/// Implementation state for a TA-Lib function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationStatus {
    /// Implemented in Rust `ta-core`.
    Implemented,
    /// Official TA-Lib function recorded for future Rust implementation.
    Planned,
}

/// One official TA-Lib function inventory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionInfo {
    /// Uppercase TA-Lib function name.
    pub name: &'static str,
    /// Official TA-Lib group.
    pub group: FunctionGroup,
    /// Current Rust implementation status.
    pub status: ImplementationStatus,
}

impl FunctionInfo {
    /// Returns true when this function is implemented in Rust `ta-core`.
    pub const fn is_implemented(self) -> bool {
        match self.status {
            ImplementationStatus::Implemented => true,
            ImplementationStatus::Planned => false,
        }
    }

    /// Rust module for this function's official group.
    pub const fn rust_module(self) -> &'static str {
        self.group.rust_module()
    }
}

macro_rules! function {
    ($name:literal, $group:ident, $status:ident) => {
        FunctionInfo {
            name: $name,
            group: FunctionGroup::$group,
            status: ImplementationStatus::$status,
        }
    };
}

/// Official TA-Lib function inventory in group order.
pub const TALIB_FUNCTIONS: &[FunctionInfo] = &[
    // Overlap Studies — 18 functions.
    function!("ACCBANDS", OverlapStudies, Planned),
    function!("BBANDS", OverlapStudies, Planned),
    function!("DEMA", OverlapStudies, Planned),
    function!("EMA", OverlapStudies, Planned),
    function!("HT_TRENDLINE", OverlapStudies, Planned),
    function!("KAMA", OverlapStudies, Planned),
    function!("MA", OverlapStudies, Planned),
    function!("MAMA", OverlapStudies, Planned),
    function!("MAVP", OverlapStudies, Planned),
    function!("MIDPOINT", OverlapStudies, Planned),
    function!("MIDPRICE", OverlapStudies, Planned),
    function!("SAR", OverlapStudies, Planned),
    function!("SAREXT", OverlapStudies, Planned),
    function!("SMA", OverlapStudies, Implemented),
    function!("T3", OverlapStudies, Planned),
    function!("TEMA", OverlapStudies, Planned),
    function!("TRIMA", OverlapStudies, Planned),
    function!("WMA", OverlapStudies, Planned),

    // Momentum Indicators — 31 functions.
    function!("ADX", MomentumIndicators, Planned),
    function!("ADXR", MomentumIndicators, Planned),
    function!("APO", MomentumIndicators, Planned),
    function!("AROON", MomentumIndicators, Planned),
    function!("AROONOSC", MomentumIndicators, Planned),
    function!("BOP", MomentumIndicators, Planned),
    function!("CCI", MomentumIndicators, Planned),
    function!("CMO", MomentumIndicators, Planned),
    function!("DX", MomentumIndicators, Planned),
    function!("IMI", MomentumIndicators, Planned),
    function!("MACD", MomentumIndicators, Planned),
    function!("MACDEXT", MomentumIndicators, Planned),
    function!("MACDFIX", MomentumIndicators, Planned),
    function!("MFI", MomentumIndicators, Planned),
    function!("MINUS_DI", MomentumIndicators, Planned),
    function!("MINUS_DM", MomentumIndicators, Planned),
    function!("MOM", MomentumIndicators, Planned),
    function!("PLUS_DI", MomentumIndicators, Planned),
    function!("PLUS_DM", MomentumIndicators, Planned),
    function!("PPO", MomentumIndicators, Planned),
    function!("ROC", MomentumIndicators, Planned),
    function!("ROCP", MomentumIndicators, Planned),
    function!("ROCR", MomentumIndicators, Planned),
    function!("ROCR100", MomentumIndicators, Planned),
    function!("RSI", MomentumIndicators, Planned),
    function!("STOCH", MomentumIndicators, Planned),
    function!("STOCHF", MomentumIndicators, Planned),
    function!("STOCHRSI", MomentumIndicators, Planned),
    function!("TRIX", MomentumIndicators, Planned),
    function!("ULTOSC", MomentumIndicators, Planned),
    function!("WILLR", MomentumIndicators, Planned),

    // Volume Indicators — 3 functions.
    function!("AD", VolumeIndicators, Planned),
    function!("ADOSC", VolumeIndicators, Planned),
    function!("OBV", VolumeIndicators, Planned),

    // Volatility Indicators — 3 functions.
    function!("ATR", VolatilityIndicators, Planned),
    function!("NATR", VolatilityIndicators, Planned),
    function!("TRANGE", VolatilityIndicators, Planned),

    // Price Transform — 5 functions.
    function!("AVGDEV", PriceTransform, Implemented),
    function!("AVGPRICE", PriceTransform, Implemented),
    function!("MEDPRICE", PriceTransform, Implemented),
    function!("TYPPRICE", PriceTransform, Implemented),
    function!("WCLPRICE", PriceTransform, Implemented),

    // Cycle Indicators — 5 functions.
    function!("HT_DCPERIOD", CycleIndicators, Planned),
    function!("HT_DCPHASE", CycleIndicators, Planned),
    function!("HT_PHASOR", CycleIndicators, Planned),
    function!("HT_SINE", CycleIndicators, Planned),
    function!("HT_TRENDMODE", CycleIndicators, Planned),

    // Pattern Recognition — 61 functions.
    function!("CDL2CROWS", PatternRecognition, Planned),
    function!("CDL3BLACKCROWS", PatternRecognition, Planned),
    function!("CDL3INSIDE", PatternRecognition, Planned),
    function!("CDL3LINESTRIKE", PatternRecognition, Planned),
    function!("CDL3OUTSIDE", PatternRecognition, Planned),
    function!("CDL3STARSINSOUTH", PatternRecognition, Planned),
    function!("CDL3WHITESOLDIERS", PatternRecognition, Planned),
    function!("CDLABANDONEDBABY", PatternRecognition, Planned),
    function!("CDLADVANCEBLOCK", PatternRecognition, Planned),
    function!("CDLBELTHOLD", PatternRecognition, Planned),
    function!("CDLBREAKAWAY", PatternRecognition, Planned),
    function!("CDLCLOSINGMARUBOZU", PatternRecognition, Planned),
    function!("CDLCONCEALBABYSWALL", PatternRecognition, Planned),
    function!("CDLCOUNTERATTACK", PatternRecognition, Planned),
    function!("CDLDARKCLOUDCOVER", PatternRecognition, Planned),
    function!("CDLDOJI", PatternRecognition, Planned),
    function!("CDLDOJISTAR", PatternRecognition, Planned),
    function!("CDLDRAGONFLYDOJI", PatternRecognition, Planned),
    function!("CDLENGULFING", PatternRecognition, Planned),
    function!("CDLEVENINGDOJISTAR", PatternRecognition, Planned),
    function!("CDLEVENINGSTAR", PatternRecognition, Planned),
    function!("CDLGAPSIDESIDEWHITE", PatternRecognition, Planned),
    function!("CDLGRAVESTONEDOJI", PatternRecognition, Planned),
    function!("CDLHAMMER", PatternRecognition, Planned),
    function!("CDLHANGINGMAN", PatternRecognition, Planned),
    function!("CDLHARAMI", PatternRecognition, Planned),
    function!("CDLHARAMICROSS", PatternRecognition, Planned),
    function!("CDLHIGHWAVE", PatternRecognition, Planned),
    function!("CDLHIKKAKE", PatternRecognition, Planned),
    function!("CDLHIKKAKEMOD", PatternRecognition, Planned),
    function!("CDLHOMINGPIGEON", PatternRecognition, Planned),
    function!("CDLIDENTICAL3CROWS", PatternRecognition, Planned),
    function!("CDLINNECK", PatternRecognition, Planned),
    function!("CDLINVERTEDHAMMER", PatternRecognition, Planned),
    function!("CDLKICKING", PatternRecognition, Planned),
    function!("CDLKICKINGBYLENGTH", PatternRecognition, Planned),
    function!("CDLLADDERBOTTOM", PatternRecognition, Planned),
    function!("CDLLONGLEGGEDDOJI", PatternRecognition, Planned),
    function!("CDLLONGLINE", PatternRecognition, Planned),
    function!("CDLMARUBOZU", PatternRecognition, Planned),
    function!("CDLMATCHINGLOW", PatternRecognition, Planned),
    function!("CDLMATHOLD", PatternRecognition, Planned),
    function!("CDLMORNINGDOJISTAR", PatternRecognition, Planned),
    function!("CDLMORNINGSTAR", PatternRecognition, Planned),
    function!("CDLONNECK", PatternRecognition, Planned),
    function!("CDLPIERCING", PatternRecognition, Planned),
    function!("CDLRICKSHAWMAN", PatternRecognition, Planned),
    function!("CDLRISEFALL3METHODS", PatternRecognition, Planned),
    function!("CDLSEPARATINGLINES", PatternRecognition, Planned),
    function!("CDLSHOOTINGSTAR", PatternRecognition, Planned),
    function!("CDLSHORTLINE", PatternRecognition, Planned),
    function!("CDLSPINNINGTOP", PatternRecognition, Planned),
    function!("CDLSTALLEDPATTERN", PatternRecognition, Planned),
    function!("CDLSTICKSANDWICH", PatternRecognition, Planned),
    function!("CDLTAKURI", PatternRecognition, Planned),
    function!("CDLTASUKIGAP", PatternRecognition, Planned),
    function!("CDLTHRUSTING", PatternRecognition, Planned),
    function!("CDLTRISTAR", PatternRecognition, Planned),
    function!("CDLUNIQUE3RIVER", PatternRecognition, Planned),
    function!("CDLUPSIDEGAP2CROWS", PatternRecognition, Planned),
    function!("CDLXSIDEGAP3METHODS", PatternRecognition, Planned),

    // Statistic Functions — 9 functions.
    function!("BETA", StatisticFunctions, Planned),
    function!("CORREL", StatisticFunctions, Planned),
    function!("LINEARREG", StatisticFunctions, Planned),
    function!("LINEARREG_ANGLE", StatisticFunctions, Planned),
    function!("LINEARREG_INTERCEPT", StatisticFunctions, Planned),
    function!("LINEARREG_SLOPE", StatisticFunctions, Planned),
    function!("STDDEV", StatisticFunctions, Planned),
    function!("TSF", StatisticFunctions, Planned),
    function!("VAR", StatisticFunctions, Planned),

    // Math Transform — 15 functions.
    function!("ACOS", MathTransform, Implemented),
    function!("ASIN", MathTransform, Implemented),
    function!("ATAN", MathTransform, Implemented),
    function!("CEIL", MathTransform, Implemented),
    function!("COS", MathTransform, Implemented),
    function!("COSH", MathTransform, Implemented),
    function!("EXP", MathTransform, Implemented),
    function!("FLOOR", MathTransform, Implemented),
    function!("LN", MathTransform, Implemented),
    function!("LOG10", MathTransform, Implemented),
    function!("SIN", MathTransform, Implemented),
    function!("SINH", MathTransform, Implemented),
    function!("SQRT", MathTransform, Implemented),
    function!("TAN", MathTransform, Implemented),
    function!("TANH", MathTransform, Implemented),

    // Math Operators — 11 functions.
    function!("ADD", MathOperators, Implemented),
    function!("DIV", MathOperators, Implemented),
    function!("MAX", MathOperators, Implemented),
    function!("MAXINDEX", MathOperators, Implemented),
    function!("MIN", MathOperators, Implemented),
    function!("MININDEX", MathOperators, Implemented),
    function!("MINMAX", MathOperators, Implemented),
    function!("MINMAXINDEX", MathOperators, Implemented),
    function!("MULT", MathOperators, Implemented),
    function!("SUB", MathOperators, Implemented),
    function!("SUM", MathOperators, Implemented),
];

/// Finds a TA-Lib function by uppercase name.
pub fn function(name: &str) -> Option<&'static FunctionInfo> {
    TALIB_FUNCTIONS.iter().find(|info| info.name == name)
}
```

### crates/ta-core/tests/inventory.rs — NEW
Inventory count/status tests.
```rust
use ta_core::inventory::{
    function, FunctionGroup, ImplementationStatus, FUNCTION_COUNT, IMPLEMENTED_FUNCTION_COUNT,
    TALIB_FUNCTIONS,
};

#[test]
fn inventory_contains_official_161_functions() {
    assert_eq!(FUNCTION_COUNT, 161);
    assert_eq!(TALIB_FUNCTIONS.len(), FUNCTION_COUNT);
}

#[test]
fn group_counts_match_official_talib_inventory() {
    let expected = [
        (FunctionGroup::OverlapStudies, 18),
        (FunctionGroup::MomentumIndicators, 31),
        (FunctionGroup::VolumeIndicators, 3),
        (FunctionGroup::VolatilityIndicators, 3),
        (FunctionGroup::PriceTransform, 5),
        (FunctionGroup::CycleIndicators, 5),
        (FunctionGroup::PatternRecognition, 61),
        (FunctionGroup::StatisticFunctions, 9),
        (FunctionGroup::MathTransform, 15),
        (FunctionGroup::MathOperators, 11),
    ];

    let mut total = 0;
    for (group, count) in expected {
        assert_eq!(group.expected_count(), count, "{:?} expected count", group);
        assert_eq!(
            TALIB_FUNCTIONS.iter().filter(|info| info.group == group).count(),
            count,
            "{:?} actual count",
            group
        );
        total += count;
    }

    assert_eq!(total, FUNCTION_COUNT);
    assert_eq!(FunctionGroup::ALL.len(), 10);
}

#[test]
fn function_names_are_unique() {
    for (idx, info) in TALIB_FUNCTIONS.iter().enumerate() {
        assert!(
            TALIB_FUNCTIONS[idx + 1..]
                .iter()
                .all(|other| other.name != info.name),
            "duplicate function name {}",
            info.name
        );
    }
}

#[test]
fn first_tranche_functions_are_marked_implemented() {
    let implemented = [
        "SMA",
        "AVGDEV",
        "AVGPRICE",
        "MEDPRICE",
        "TYPPRICE",
        "WCLPRICE",
        "ACOS",
        "ASIN",
        "ATAN",
        "CEIL",
        "COS",
        "COSH",
        "EXP",
        "FLOOR",
        "LN",
        "LOG10",
        "SIN",
        "SINH",
        "SQRT",
        "TAN",
        "TANH",
        "ADD",
        "DIV",
        "MAX",
        "MAXINDEX",
        "MIN",
        "MININDEX",
        "MINMAX",
        "MINMAXINDEX",
        "MULT",
        "SUB",
        "SUM",
    ];

    assert_eq!(IMPLEMENTED_FUNCTION_COUNT, implemented.len());
    assert_eq!(
        TALIB_FUNCTIONS
            .iter()
            .filter(|info| info.is_implemented())
            .count(),
        IMPLEMENTED_FUNCTION_COUNT
    );

    for name in implemented {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert!(!info.rust_module().is_empty());
    }
}

#[test]
fn deferred_functions_remain_planned() {
    for name in ["MACD", "BBANDS", "ATR", "OBV", "CDLDOJI", "VAR", "HT_SINE"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Planned, "{name}");
    }
}

#[test]
fn non_talib_local_plan_extras_are_not_in_official_inventory() {
    for name in ["WWMA", "HMA", "VWAP"] {
        assert!(function(name).is_none(), "{name} should not be in TA-Lib inventory");
    }
}
```

### crates/ta-benchmarks/Cargo.toml:13-15 — MODIFY
Register the first-tranche Criterion benchmark target.
```toml
[[bench]]
name = "basic"
harness = false

[[bench]]
name = "first_tranche"
harness = false
```

### crates/ta-benchmarks/benches/first_tranche.rs — NEW
Criterion benchmarks for SMA, price transform, math transform, and math operators.
```rust
//! First-tranche `ta-core` indicator benchmarks.
//!
//! These benchmarks exercise the public Rust APIs designed for the foundation
//! tranche: compact zero-copy kernels plus selected padded convenience wrappers.
//! Fixtures and reusable output buffers are allocated outside `b.iter()` unless
//! the wrapper allocation itself is the behavior under measurement.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ta_core::{
    math_operators::{ADD, MINMAX, SUM},
    math_transform::SQRT,
    overlap::{SMA, SMA_vec},
    price_transform::{AVGDEV, AVGPRICE},
    Float,
};

const SIZES: &[usize] = &[1_024, 16_384, 65_536];
const PERIOD: usize = 20;

fn series_fixture(size: usize) -> Vec<Float> {
    (0..size)
        .map(|idx| ((idx % 997) as Float + 1.0 as Float) * 0.5 as Float)
        .collect()
}

fn paired_fixture(size: usize) -> (Vec<Float>, Vec<Float>) {
    let left = series_fixture(size);
    let right = left
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 17) as Float + 1.0 as Float)
        .collect();
    (left, right)
}

fn ohlc_fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    let close = series_fixture(size);
    let open: Vec<Float> = close
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 5) as Float * 0.01 as Float)
        .collect();
    let high: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::max(*open, *close) + 1.0 as Float)
        .collect();
    let low: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::min(*open, *close) - 1.0 as Float)
        .collect();

    (open, high, low, close)
}

fn bench_overlap_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap/sma");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SMA_vec", size), &size, |b, &size| {
            let prices = series_fixture(size);

            b.iter(|| {
                let output = SMA_vec(black_box(prices.as_slice()), black_box(PERIOD))
                    .expect("valid SMA benchmark fixture");
                black_box(output);
            });
        });
    }

    group.finish();
}

fn bench_price_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/price_transform");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("AVGPRICE_compact", size), &size, |b, &size| {
            let (open, high, low, close) = ohlc_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = AVGPRICE(
                    black_box(open.as_slice()),
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid AVGPRICE benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("AVGDEV_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = AVGDEV(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid AVGDEV benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_math_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_transform");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SQRT_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SQRT(black_box(prices.as_slice()), black_box(output.as_mut_slice()))
                    .expect("valid SQRT benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_math_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_operators");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("ADD_compact", size), &size, |b, &size| {
            let (left, right) = paired_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = ADD(
                    black_box(left.as_slice()),
                    black_box(right.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid ADD benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SUM_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SUM(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SUM benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("MINMAX_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut min = vec![0.0 as Float; size];
            let mut max = vec![0.0 as Float; size];

            b.iter(|| {
                let range = MINMAX(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(min.as_mut_slice()),
                    black_box(max.as_mut_slice()),
                )
                .expect("valid MINMAX benchmark fixture");
                black_box(range);
                black_box(min.as_slice());
                black_box(max.as_slice());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overlap_sma,
    bench_price_transform,
    bench_math_transform,
    bench_math_operators
);
criterion_main!(benches);
```

## Slices
### Slice 1: Core contracts

**Files**: `crates/ta-core/src/common.rs`, `crates/ta-core/src/traits.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] Core crate compiles after helper and trait additions: `cargo test -p ta-core`
- [ ] f32 precision still compiles with the new helpers: `cargo test -p ta-core --features f32`
- [ ] Workspace module exports still check after `lib.rs` changes: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] New zero-copy trait method is present: `grep -n "fn compute(&self" crates/ta-core/src/traits.rs`

#### Manual Verification:
- [ ] Confirm `crates/ta-core/src/common.rs` documents compact output as `outBegIdx/outNBElement`-style.
- [ ] Confirm public validation helpers return `TalibError` rather than panicking.
- [ ] Confirm `crates/ta-core/src/lib.rs` re-exports `OutputRange` and `PadValue` for downstream indicator modules.
- [ ] Confirm `Indicator::compute` is defaulted so existing `SMA` compiles until Slice 2 overrides it.

### Slice 2: SMA rewrite

**Files**: `crates/ta-core/src/overlap/sma.rs`, `crates/ta-core/src/overlap/mod.rs`, `crates/ta-core/tests/overlap_sma.rs`

#### Automated Verification:
- [ ] SMA rewrite tests pass: `cargo test -p ta-core --test overlap_sma`
- [ ] Existing core tests still pass: `cargo test -p ta-core`
- [ ] f32 precision still compiles: `cargo test -p ta-core --features f32`
- [ ] Workspace public exports still check after `overlap/mod.rs` changes: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] No SMA period constructor panic remains: `grep -R "assert!(.*period" crates/ta-core/src/overlap/sma.rs` returns no matches

#### Manual Verification:
- [ ] Confirm `SMA` free function writes compact outputs and returns `OutputRange`.
- [ ] Confirm `SMA_vec` preserves full-length padded output shape.
- [ ] Confirm streaming buffer is initialized with real elements, not capacity-only allocation.
- [ ] Confirm `SMA::new(0)` returns `TalibError`, not a panic.

### Slice 3: Price Transform group

**Files**: `crates/ta-core/src/price_transform/mod.rs`, `crates/ta-core/src/price_transform/avgdev.rs`, `crates/ta-core/src/price_transform/avgprice.rs`, `crates/ta-core/src/price_transform/medprice.rs`, `crates/ta-core/src/price_transform/typprice.rs`, `crates/ta-core/src/price_transform/wclprice.rs`, `crates/ta-core/tests/price_transform.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] Price Transform tests pass: `cargo test -p ta-core --test price_transform`
- [ ] Existing SMA tests still pass: `cargo test -p ta-core --test overlap_sma`
- [ ] Core crate tests pass after `lib.rs` module addition: `cargo test -p ta-core`
- [ ] f32 precision still compiles: `cargo test -p ta-core --features f32`
- [ ] Workspace exports still check after adding `price_transform`: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] Public Price Transform module is wired: `grep -n "pub mod price_transform" crates/ta-core/src/lib.rs`
- [ ] Stateless constructors are fallible: `grep -R "pub const fn new() -> Self" crates/ta-core/src/price_transform` returns no matches

#### Manual Verification:
- [ ] Confirm `AVGPRICE` formula is `(open + high + low + close) / 4`, not the incorrect local-plan formula.
- [ ] Confirm `AVGDEV` uses lookback padding through `OutputRange::new(timeperiod - 1, count)`.
- [ ] Confirm multi-price functions use separate input slices and validate equal lengths.
- [ ] Confirm all public functions/types use uppercase TA-Lib names.

### Slice 4: Math Transform group

**Files**: `crates/ta-core/src/math_transform/mod.rs`, `crates/ta-core/tests/math_transform.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] Math Transform tests pass: `cargo test -p ta-core --test math_transform`
- [ ] Price Transform tests still pass: `cargo test -p ta-core --test price_transform`
- [ ] SMA tests still pass: `cargo test -p ta-core --test overlap_sma`
- [ ] Core crate tests pass after `lib.rs` module addition: `cargo test -p ta-core`
- [ ] f32 precision still compiles: `cargo test -p ta-core --features f32`
- [ ] Workspace exports still check after adding `math_transform`: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] Public Math Transform module is wired: `grep -n "pub mod math_transform" crates/ta-core/src/lib.rs`
- [ ] Math Transform follows private-module re-export direction: `grep -n "pub use functions" crates/ta-core/src/math_transform/mod.rs`

#### Manual Verification:
- [ ] Confirm all 15 official Math Transform functions are present in `crates/ta-core/src/math_transform/mod.rs`.
- [ ] Confirm unary functions use strict finite-input validation before applying math operations.
- [ ] Confirm each transform has free function, `*_vec` wrapper, and uppercase struct surface.

### Slice 5: Math Operators group

**Files**: `crates/ta-core/src/math_operators/mod.rs`, `crates/ta-core/src/math_operators/arithmetic.rs`, `crates/ta-core/src/math_operators/rolling.rs`, `crates/ta-core/src/math_operators/extrema.rs`, `crates/ta-core/tests/math_operators.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] Math Operators tests pass: `cargo test -p ta-core --test math_operators`
- [ ] Math Transform tests still pass: `cargo test -p ta-core --test math_transform`
- [ ] Price Transform tests still pass: `cargo test -p ta-core --test price_transform`
- [ ] SMA tests still pass: `cargo test -p ta-core --test overlap_sma`
- [ ] Core crate tests pass after `lib.rs` module addition: `cargo test -p ta-core`
- [ ] f32 precision still compiles for arithmetic/operators: `cargo test -p ta-core --features f32`
- [ ] Workspace exports still check after adding `math_operators`: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] Public Math Operators module is wired: `grep -n "pub mod math_operators" crates/ta-core/src/lib.rs`
- [ ] Math Operators follow private-module re-export direction: `grep -n "pub use arithmetic" crates/ta-core/src/math_operators/mod.rs`

#### Manual Verification:
- [ ] Confirm all 11 official Math Operators functions are present: `ADD`, `DIV`, `MULT`, `SUB`, `MAX`, `MAXINDEX`, `MIN`, `MININDEX`, `MINMAX`, `MINMAXINDEX`, `SUM`.
- [ ] Confirm binary operators use separate `real0`/`real1` slices and validate equal lengths.
- [ ] Confirm rolling functions use `OutputRange::new(timeperiod - 1, count)` and padded wrappers place warm-up values before compact outputs.
- [ ] Confirm `MINMAX` and `MINMAXINDEX` use parallel output buffers in compact APIs and named parallel vectors in convenience wrappers.
- [ ] Confirm integer index outputs use `i32` and padded wrappers use zero warm-up values.
- [ ] Confirm all struct constructors return `Result<Self>` and reject period `0` where applicable.

### Slice 6: Full inventory ledger

**Files**: `crates/ta-core/src/inventory.rs`, `crates/ta-core/tests/inventory.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] Inventory tests pass: `cargo test -p ta-core --test inventory`
- [ ] Math Operators tests still pass: `cargo test -p ta-core --test math_operators`
- [ ] Math Transform tests still pass: `cargo test -p ta-core --test math_transform`
- [ ] Price Transform tests still pass: `cargo test -p ta-core --test price_transform`
- [ ] SMA tests still pass: `cargo test -p ta-core --test overlap_sma`
- [ ] Core crate tests pass after `lib.rs` module addition: `cargo test -p ta-core`
- [ ] Workspace exports still check after adding `inventory`: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] Public inventory module is wired: `grep -n "pub mod inventory" crates/ta-core/src/lib.rs`
- [ ] Source ledger count remains explicit: `grep -n "FUNCTION_COUNT: usize = 161" crates/ta-core/src/inventory.rs`

#### Manual Verification:
- [ ] Confirm all 10 official TA-Lib groups are represented by `FunctionGroup`.
- [ ] Confirm `TALIB_FUNCTIONS` contains 161 records and no local non-TA-Lib extras.
- [ ] Confirm first-tranche functions are marked `Implemented`: `SMA`, all Price Transform, all Math Transform, all Math Operators.
- [ ] Confirm remaining official functions are marked `Planned` so future sessions can implement group-by-group.
- [ ] Confirm group counts match the research artifact: 18/31/3/3/5/5/61/9/15/11.

### Slice 7: First-tranche benchmarks

**Files**: `crates/ta-benchmarks/Cargo.toml`, `crates/ta-benchmarks/benches/first_tranche.rs`

#### Automated Verification:
- [ ] First-tranche benchmark target compiles: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`
- [ ] Existing basic benchmark target still compiles: `cargo bench -p ta-benchmarks --bench basic --no-run`
- [ ] Core crate tests still pass after benchmark additions: `cargo test -p ta-core`
- [ ] Workspace exports still check with benchmark target registered: `cargo check --workspace`
- [ ] Formatting passes: `cargo fmt --all -- --check`
- [ ] First-tranche benchmark target is registered: `grep -n "name = \"first_tranche\"" crates/ta-benchmarks/Cargo.toml`
- [ ] Benchmark file uses public ta-core APIs only: `grep -R "ta_core::.*src\|crate::" crates/ta-benchmarks/benches/first_tranche.rs` returns no matches

#### Manual Verification:
- [ ] Confirm fixtures and output buffers are allocated outside `b.iter()` except padded wrapper allocation benchmarks like `SMA_vec`.
- [ ] Confirm benchmark groups cover first-tranche families: SMA, Price Transform, Math Transform, and Math Operators.
- [ ] Confirm every `bench_*` function is registered in `criterion_group!`.
- [ ] Confirm benchmark code imports only public `ta_core` APIs and does not touch private core modules.

## Desired End State
```rust
use ta_core::{Float, OutputRange};
use ta_core::overlap::{SMA, SMA_vec};

let input: Vec<Float> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let mut compact = vec![0.0; input.len()];
let range: OutputRange = SMA(&input, 3, &mut compact)?;
assert_eq!(range.beg_idx, 2);
assert_eq!(&compact[..range.nb_element], &[2.0, 3.0, 4.0]);

let padded = SMA_vec(&input, 3)?;
assert!(padded[0].is_nan());
assert_eq!(&padded[2..], &[2.0, 3.0, 4.0]);

let mut sma = SMA::new(3)?;
assert!(sma.next(1.0)?.is_nan());
assert_eq!(sma.next(3.0)?, 2.0);
```

```rust
use ta_core::math_operators::{MINMAX, MINMAXINDEX};

let real = [3.0, 1.0, 4.0, 2.0];
let mut min = [0.0; 4];
let mut max = [0.0; 4];
let range = MINMAX(&real, 2, &mut min, &mut max)?;
assert_eq!(range.beg_idx, 1);

let mut min_idx = [0; 4];
let mut max_idx = [0; 4];
MINMAXINDEX(&real, 2, &mut min_idx, &mut max_idx)?;
```

## File Map
```text
crates/ta-core/src/common.rs  # NEW — shared range/padding/validation helpers
crates/ta-core/src/traits.rs  # MODIFY — zero-copy + padded trait contract
crates/ta-core/src/lib.rs  # MODIFY — module exports as slices land
crates/ta-core/src/overlap/mod.rs  # MODIFY — SMA re-export
crates/ta-core/src/overlap/sma.rs  # MODIFY — safe dual-API SMA
crates/ta-core/tests/overlap_sma.rs  # NEW — SMA tests
crates/ta-core/src/price_transform/  # NEW — 5 Price Transform functions
crates/ta-core/tests/price_transform.rs  # NEW — Price Transform tests
crates/ta-core/src/math_transform/mod.rs  # NEW — 15 Math Transform functions
crates/ta-core/tests/math_transform.rs  # NEW — Math Transform tests
crates/ta-core/src/math_operators/  # NEW — 11 Math Operators functions
crates/ta-core/tests/math_operators.rs  # NEW — Math Operators tests
crates/ta-core/src/inventory.rs  # NEW — 161-function implementation ledger
crates/ta-core/tests/inventory.rs  # NEW — inventory tests
crates/ta-benchmarks/Cargo.toml  # MODIFY — add first-tranche bench target
crates/ta-benchmarks/benches/first_tranche.rs  # NEW — Criterion first-tranche benchmarks
```

## Ordering Constraints
- Slice 1 must land before every implementation slice because it defines `OutputRange`, validation, and trait contracts.
- Slice 2 depends on Slice 1 and validates the pattern for moving-average indicators.
- Slices 3, 4, and 5 can be implemented after Slice 1, but are sequenced here to keep review and verification focused.
- Slice 6 depends on Slices 2-5 so status values can mark first-tranche functions implemented.
- Slice 7 depends on Slices 2-5 so benchmarks compile against real APIs.
- Python/WASM bindings should wait until at least the first-tranche core APIs stabilize.

## Verification Notes
- Run `cargo fmt --all -- --check` after each slice that modifies Rust source.
- Run `cargo test -p ta-core` after each core slice.
- Run `cargo test -p ta-core --features f32` after Slice 1 and after each arithmetic/transform slice.
- Run `cargo check --workspace` after module export changes; note `ta-py` may require a Python environment.
- Run `cargo bench -p ta-benchmarks --bench first_tranche --no-run` after Slice 7.
- Verify no new constructors use `assert!` for user parameters: `grep -R "assert!(.*period" crates/ta-core/src` should not find first-tranche constructors.
- Verify full inventory remains 161 records once Slice 6 lands.
- Preserve cross-platform SIMD dispatch boundaries; first-tranche functions may use scalar loops or public SIMD helpers but must not call private arch backends.

## Performance Considerations
- Zero-copy compact output buffers are the performance-critical API.
- Padded `Vec` wrappers allocate and fill warm-up sentinels for convenience only.
- Separate input slices avoid struct field extraction and preserve contiguous memory for future SIMD optimization.
- Multi-output functions use parallel buffers to avoid row-struct layout costs.
- Simple elementwise transform/operators use tight loops; future SIMD improvements can specialize these without changing public API.

## Migration Notes
- Existing `SMA::new(period) -> Self` changes to `SMA::new(period) -> Result<SMA>`.
- Existing `Indicator::compute_to_vec` behavior becomes a default padded convenience wrapper over zero-copy `compute`.
- Existing code using `ta_core::overlap::SMA` remains on the uppercase name, but constructor call sites must handle `Result`.
- No persisted schema or data migration is involved.

## Pattern References
- `crates/ta-core/src/overlap/mod.rs:4-6` — family-private module and public re-export pattern.
- `crates/ta-core/src/error.rs:73-115` — typed error constructors to use instead of panics.
- `crates/ta-core/src/types.rs:26-33` — `Float` precision boundary.
- `crates/ta-core/src/simd/mod.rs:37-47` — public SIMD helper boundary and `FastFloat`/`LANES` definitions.
- `crates/ta-benchmarks/benches/basic.rs:23-41` — Criterion benchmark registration pattern.

## Developer Context
**Q (`crates/ta-core/src/overlap/mod.rs:4-6`): About to follow the private implementation module + family re-export pattern across all TA-Lib groups. Confirm that direction, or are we moving off it?**
A: Follow module pattern.

**Q (`crates/ta-core/src/overlap/sma.rs:51-52`): About to move off legacy panic constructors and make new indicator constructors return `Result<Self>` using `TalibError` from `crates/ta-core/src/error.rs:73-115`. Confirm?**
A: Use Result constructors.

**Q (`crates/ta-core/src/lib.rs:32-37`): About to use official TA-Lib groups as Rust modules because the local plan misclassifies functions like `SAR`/`VAR` in `.sisyphus/IMPLEMENTATION_PLAN.md:3108-3111`. Confirm?**
A: Use TA-Lib groups.

**Q (`crates/ta-core/src/overlap/sma.rs:104-107`, `README:21-25`): Which core output contract should the full TA-Lib Rust design use?**
A: Dual API.

**Q (`crates/ta-core/src/traits.rs:79-81`, `crates/ta-core/src/simd/dispatch.rs:35-40`): How should multi-price indicators receive inputs in `ta-core`?**
A: Separate slices.

**Q (`crates/ta-core/src/traits.rs:44-46`, `crates/ta-core/src/traits.rs:151`): How should multi-output functions like MACD/BBANDS be represented?**
A: “按照性能最好的方式” — interpreted as parallel buffers / struct-of-arrays.

**Q (`crates/ta-core/src/overlap/mod.rs:4-6`): For this design artifact, how large should the copy-pasteable code slice be?**
A: Foundation + first tranche, with unfinished records for later group-by-group implementation.

**Q (`crates/ta-core/src/traits.rs:76-151`): Should `ta-core` be function-first or struct-first?**
A: Both equal surfaces.

**Q (`crates/ta-core/src/traits.rs:13-15`): Which default should core parity APIs follow for NaN behavior?**
A: “按照rust方式的” — strict finite-input validation by default.

**Q (`crates/ta-core/src/overlap/mod.rs:6`, `crates/ta-core/src/traits.rs:55`): Which naming should new public APIs use?**
A: TA-Lib uppercase.

**Q (`crates/ta-core/src/lib.rs:32-37`): Should unfinished work record live only in the design artifact, or also in `ta-core` source?**
A: Both.

**Q (decomposition): 7 slices for TA-Lib Rust core foundation + first tranche. Approve decomposition?**
A: Approve.

**Q (Slice 1 micro-checkpoint): Slice 1/7: Core contracts — adds shared output ranges, padding/validation helpers, zero-copy `Indicator::compute`, and root exports. Approve?**
A: Approve.

**Q (Slice 2 micro-checkpoint): Slice 2/7: SMA rewrite — rewrites SMA onto the dual API, fallible constructor, initialized streaming buffer, and tests. Approve?**
A: Approve.

**Q (Slice 3 micro-checkpoint): Slice 3/7: Price Transform group — adds AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE with compact APIs, padded wrappers, struct surfaces, tests, and module wiring. Approve?**
A: Approve.

**Q (Slice 4 micro-checkpoint): Slice 4/7: Math Transform group — adds all 15 unary transforms with compact functions, padded wrappers, uppercase structs, module wiring, and tests. Approve?**
A: Approve.

**Q (Slice 5 micro-checkpoint): Slice 5/7: Math Operators group — adds all 11 operators with compact APIs, padded wrappers, uppercase structs, tests, and module wiring. Approve?**
A: Approve.

**Q (Slice 6 micro-checkpoint): Slice 6/7: Full inventory ledger — adds a source-level 161-function TA-Lib ledger with group counts and first-tranche statuses. Approve?**
A: Approve.

**Q (Slice 7 micro-checkpoint): Slice 7/7: First-tranche benchmarks — registers `first_tranche` Criterion target and benchmarks public SMA, Price Transform, Math Transform, and Math Operators APIs. Approve?**
A: Approve.

## Design History
- Slice 1: Core contracts — approved as generated
- Slice 2: SMA rewrite — approved as generated
- Slice 3: Price Transform group — approved as generated
- Slice 4: Math Transform group — approved as generated
- Slice 5: Math Operators group — approved as generated
- Slice 6: Full inventory ledger — approved as generated
- Slice 7: First-tranche benchmarks — approved as generated

## References
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — parent research artifact.
- [Official TA-Lib function index](https://ta-lib.org/functions/) — canonical function inventory.
- [TA-Lib v0.7.1 function metadata XML](https://github.com/TA-Lib/ta-lib/blob/v0.7.1/ta_func_api.xml) — authoritative metadata reference.
- [TA-Lib v0.7.1 generated C function header](https://github.com/TA-Lib/ta-lib/blob/v0.7.1/include/ta_func.h) — zero-copy C API pattern.
- [ta-lib-python generated wrappers](https://github.com/TA-Lib/ta-lib-python/blob/master/tools/generate_func.py) — padded wrapper conventions.
