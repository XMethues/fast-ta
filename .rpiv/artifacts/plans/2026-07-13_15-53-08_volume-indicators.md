---
date: 2026-07-13T15:53:08+0800
author: unknown
commit: c6f3630
branch: main
repository: fast-ta
topic: "实现Volume分组指标"
tags: [plan, blueprint, ta-core, volume, ad, obv, adosc]
status: ready
parent: .rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md
phase_count: 4
phases:
  - { n: 1, title: AD foundation }
  - { n: 2, title: OBV }
  - { n: 3, title: ADOSC }
  - { n: 4, title: Benchmarks }
unresolved_phase_count: 0
last_updated: 2026-07-13T15:53:08+0800
last_updated_by: unknown
---

# Volume Indicators Implementation Plan

## Overview
Add the complete official TA-Lib Volume Indicators group to `ta-core`: `AD`, `OBV`, and `ADOSC`. The implementation follows the existing private-module/public-facade, compact `OutputRange`, padded `_vec`, struct, streaming, reset, inventory, integration-test, and Criterion patterns; OBV intentionally uses the developer-selected first-tick warm-up contract, while ADOSC requires `fastperiod < slowperiod`.

## Requirements
- Implement `ta_core::volume::{AD, OBV, ADOSC}` with uppercase free functions, `_vec` wrappers, structs, named SoA input views, streaming ticks, `Indicator`, `StreamingIndicator`, and reset support.
- Use `Float`, `Result`, `OutputRange`, and existing validators; do not call private SIMD/backend APIs.
- AD and ADOSC consume separate high/low/close/volume slices; OBV consumes close/volume slices.
- AD computes cumulative Chaikin accumulation/distribution money-flow volume and contributes zero when `high <= low`, matching TA-Lib's non-positive-range guard.
- OBV uses the selected first-tick warm-up behavior: lookback 1, no output for the first observation, then accumulate signed current volume based on close movement.
- ADOSC computes AD followed by fast and slow EMA states, returns fast EMA minus slow EMA, and rejects periods unless both are nonzero and `fastperiod < slowperiod`.
- Update inventory count/status and trait-conformance tests incrementally.
- Add integration tests for values, compact/padded alignment, traits, batch/streaming consistency, reset, invalid inputs, edge cases, and f32.
- Add Criterion benchmarks using public APIs and fixtures/output buffers prepared outside `b.iter()`.

## Current State Analysis
`ta-core` tracks the three official Volume functions but marks all as planned, and there is no `volume` module. Existing Price Transform, Volatility, and Overlap implementations provide the required multi-input, cumulative-state, EMA, compact-output, integration-test, and benchmark patterns.

### Key Discoveries
- `crates/ta-core/src/inventory.rs:12` reports 42 implemented functions; `crates/ta-core/src/inventory.rs:201-204` records AD, ADOSC, and OBV as planned.
- `crates/ta-core/src/inventory.rs:75,91` already fixes the Volume group count at 3 and maps it to Rust module `"volume"`.
- `crates/ta-core/src/price_transform/mod.rs:6-16` and `crates/ta-core/src/volatility/mod.rs:6-12` establish private child modules with explicit public re-exports.
- `crates/ta-core/src/price_transform/typprice.rs:13-55` establishes named multi-slice inputs, finite/length validation, compact output, and lookback-zero behavior.
- `crates/ta-core/src/volatility/trange.rs:94-181` establishes previous-value streaming state without deriving `Copy`.
- `crates/ta-core/src/overlap/ema.rs:19-32,156-186` establishes EMA multiplier/step, warm-up, and reset semantics; its helpers are `pub(super)` and cannot be reused from a sibling `volume` module.
- `crates/ta-core/src/common.rs:89-209` provides period, length, finite-value, compact-buffer, and padded-output helpers.
- `crates/ta-core/tests/inventory.rs:70-225` is the implemented-name and trait-conformance integration surface; OBV is currently in the deferred list.
- `crates/ta-benchmarks/benches/first_tranche.rs:284-347,435-443` is the public compact-kernel Criterion pattern.

## Desired End State

```rust
use ta_core::volume::{ADInput, ADTick, ADOSC, ADOSCInput, OBV_vec, AD};
use ta_core::{Indicator, StreamingIndicator};

let high = [10.0, 12.0, 11.0, 15.0];
let low = [8.0, 8.0, 9.0, 13.0];
let close = [10.0, 11.0, 9.0, 14.0];
let volume = [100.0, 200.0, 50.0, 300.0];
let mut compact = [0.0; 4];

let ad = AD::new()?;
let range = Indicator::compute(
    &ad,
    ADInput { high: &high, low: &low, close: &close, volume: &volume },
    &mut compact,
)?;
assert_eq!(range.beg_idx, 0);

let obv = OBV_vec(&close, &volume)?;
assert!(obv[0].is_nan());

let adosc = ADOSC::new(3, 10)?;
assert_eq!(adosc.lookback(), 9);

let mut streaming = AD::new()?;
let _ = streaming.next(ADTick {
    high: high[0], low: low[0], close: close[0], volume: volume[0],
})?;
# Ok::<(), ta_core::TalibError>(())
```

## What We're NOT Doing
- No Python (`ta-py`) or WASM (`ta-wasm`) bindings in this tranche.
- No SIMD specialization or private backend calls.
- No non-Volume functions such as SAR, ATR, NATR, or VWAP.
- No TA-Lib global compatibility/configuration state.
- No workspace membership, Cargo dependency, or benchmark-target changes.
- No automatic relaxation of the selected `fastperiod < slowperiod` rule.
- No TA-Lib-style OBV index-zero seed; the selected API intentionally warms up on the first tick and begins at index 1.

## Decisions

### Volume follows the existing group facade and core-only boundary
Create `crates/ta-core/src/volume/mod.rs` with private child modules and explicit re-exports, expose it from `lib.rs`, and keep adapters unchanged. This follows `crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/src/lib.rs:35-48`, and the root dependency boundary.

### Inputs use named SoA views and ticks
AD/ADOSC expose high/low/close/volume input and tick structs; OBV exposes close/volume structs. This follows the multi-input convention at `crates/ta-core/src/price_transform/typprice.rs:13-33`.

### Compact and padded outputs reuse shared helpers
All batch functions write valid values from `out_real[0]`, return `OutputRange`, and build full-length `_vec` results through `compact_buffer` and `padded_from_compact` (`crates/ta-core/src/common.rs:16-59,179-209`).

### ADOSC uses local EMA state and strict period ordering
The sibling-only helpers in `crates/ta-core/src/overlap/ema.rs:19-32` are not accessible from `volume`. ADOSC therefore models their formulas locally, with checked validation requiring nonzero periods and `fastperiod < slowperiod`; its lookback is `slowperiod - 1`.

### OBV deliberately warms up on the first tick
The developer selected a previous-close contract like `crates/ta-core/src/volatility/trange.rs:156-175`: the first observation stores close and returns no output, batch `beg_idx` is 1, padded index 0 is `Float::NAN`, and subsequent values start from zero before applying signed current volume. This is an intentional compatibility deviation from common TA-Lib OBV index-zero seeding.

### Benchmarks remain in scope
Extend `first_tranche.rs` with deterministic volume data and compact AD/OBV/ADOSC benchmarks; fixtures and output buffers remain outside `b.iter()`.

## Phase 1: AD foundation

### Overview
Creates the Volume facade and implements AD end-to-end; depends on nothing and establishes shared HLCV/money-flow helpers for ADOSC.

### Changes Required:

#### 1. crates/ta-core/src/volume/mod.rs
**File**: crates/ta-core/src/volume/mod.rs
**Changes**: NEW — Volume facade with AD export.

```rust
//! Volume Indicators.
//!
//! These functions derive cumulative price-volume measures from market data.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod ad;

pub use ad::{ADInput, ADTick, AD_vec, AD};
```

#### 2. crates/ta-core/src/volume/ad.rs
**File**: crates/ta-core/src/volume/ad.rs
**Changes**: NEW — AD batch, vec, struct, streaming, reset, and shared HLCV/money-flow helpers.

```rust
//! Chaikin Accumulation/Distribution Line (AD).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed high/low/close/volume inputs for [`AD`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct ADInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One high/low/close/volume tick for [`AD`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ADTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

/// Validates high/low/close/volume slices and returns their shared length.
pub(super) fn validate_hlcv(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
        ("volume", volume.len()),
    ])?;
    validate_finite_slices(&[
        ("high", high),
        ("low", low),
        ("close", close),
        ("volume", volume),
    ])?;
    Ok(len)
}

/// Computes one money-flow-volume contribution.
#[inline]
pub(super) fn money_flow_volume(
    high: Float,
    low: Float,
    close: Float,
    volume: Float,
) -> Float {
    let range = high - low;
    if range <= 0.0 as Float {
        0.0 as Float
    } else {
        (((close - low) - (high - close)) / range) * volume
    }
}

/// TA-Lib-style Chaikin Accumulation/Distribution Line batch function.
#[allow(non_snake_case)]
pub fn AD(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_hlcv(high, low, close, volume)?;
    validate_output_len("AD", out_real.len(), len)?;

    let mut cumulative = 0.0 as Float;
    for idx in 0..len {
        cumulative += money_flow_volume(high[idx], low[idx], close[idx], volume[idx]);
        out_real[idx] = cumulative;
    }

    Ok(OutputRange::new(0, len))
}

/// Computes AD into a full-length vector.
#[allow(non_snake_case)]
pub fn AD_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    volume: &[Float],
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = AD(high, low, close, volume, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Chaikin Accumulation/Distribution Line indicator.
#[derive(Debug, Clone, Default)]
pub struct AD {
    cumulative: Float,
}

impl AD {
    /// Creates a new AD indicator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            cumulative: 0.0 as Float,
        })
    }

    /// Computes compact AD outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        AD(high, low, close, volume, out_real)
    }

    /// Computes full-length AD outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        volume: &[Float],
    ) -> Result<Vec<Float>> {
        AD_vec(high, low, close, volume)
    }

    /// Checked streaming update.
    pub fn next_checked(&mut self, input: ADTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for AD {
    type Input<'a> = ADInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        0
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        AD(input.high, input.low, input.close, input.volume, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        AD_vec(input.high, input.low, input.close, input.volume)
    }
}

impl StreamingIndicator for AD {
    type Tick = ADTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
            ("volume", &[input.volume]),
        ])?;
        self.cumulative += money_flow_volume(input.high, input.low, input.close, input.volume);
        Ok(Some(self.cumulative))
    }
}

impl Resettable for AD {
    fn reset(&mut self) {
        self.cumulative = 0.0 as Float;
    }
}
```

#### 3. crates/ta-core/src/lib.rs:35-48
**File**: crates/ta-core/src/lib.rs
**Changes**: MODIFY — expose the Volume module from the crate root.

```rust
/// Volume Indicators functions.
pub mod volume;
/// Volatility Indicators functions.
pub mod volatility;
```

#### 4. crates/ta-core/src/inventory.rs:12,201-204
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark AD implemented and increment the implemented count to 43.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 43;

// Volume Indicators — 3 functions.
function!("AD", VolumeIndicators, Implemented),
function!("ADOSC", VolumeIndicators, Planned),
function!("OBV", VolumeIndicators, Planned),
```

#### 5. crates/ta-core/tests/inventory.rs:1-225
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — add AD to implemented and trait-conformance checks.

```rust
use ta_core::volume::AD;

// Add inside first_tranche_functions_are_marked_implemented:
"AD",

// Add inside first_tranche_structs_implement_batch_and_streaming_traits:
assert_indicator::<AD>();
assert_streaming::<AD>();
```

#### 6. crates/ta-core/tests/volume.rs
**File**: crates/ta-core/tests/volume.rs
**Changes**: NEW — AD compact, padded, trait, streaming/reset, edge-case, and invalid-input tests.

```rust
use ta_core::volume::{ADInput, ADTick, AD_vec, AD};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-5 as Float,
        "expected {expected}, got {actual}"
    );
}

fn fixture() -> ([Float; 4], [Float; 4], [Float; 4], [Float; 4]) {
    (
        [10.0, 12.0, 11.0, 15.0],
        [8.0, 8.0, 9.0, 13.0],
        [10.0, 11.0, 9.0, 14.0],
        [100.0, 200.0, 50.0, 300.0],
    )
}

#[test]
fn ad_function_writes_compact_outputs() {
    let (high, low, close, volume) = fixture();
    let mut output = [0.0; 4];

    let range = AD(&high, &low, &close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(0, 4));
    for (actual, expected) in output.into_iter().zip([100.0, 200.0, 150.0, 150.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn ad_vec_returns_full_length_outputs() {
    let (high, low, close, volume) = fixture();
    let output = AD_vec(&high, &low, &close, &volume).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output.iter().all(|value| value.is_finite()));
    assert_close(output[0], 100.0);
    assert_close(output[3], 150.0);
}

#[test]
fn ad_struct_implements_indicator_compute() {
    let (high, low, close, volume) = fixture();
    let ad = AD::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &ad,
        ADInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(ad.lookback(), 0);
    assert_eq!(range, OutputRange::new(0, 4));
    assert_close(output[2], 150.0);
}

#[test]
fn ad_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = fixture();
    let mut batch = [0.0; 4];
    AD(&high, &low, &close, &volume, &mut batch).unwrap();

    let mut ad = AD::new().unwrap();
    for idx in 0..high.len() {
        let streamed = ad
            .next(ADTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap()
            .unwrap();
        assert_close(streamed, batch[idx]);
    }

    ad.reset();
    assert_close(
        ad.next_checked(ADTick {
            high: high[0],
            low: low[0],
            close: close[0],
            volume: volume[0],
        })
        .unwrap(),
        100.0,
    );
}

#[test]
fn ad_non_positive_range_contributes_zero() {
    let high = [10.0, 10.0, 9.0];
    let low = [8.0, 10.0, 10.0];
    let close = [10.0, 10.0, 9.0];
    let volume = [100.0, 500.0, 700.0];
    let mut output = [0.0; 3];

    AD(&high, &low, &close, &volume, &mut output).unwrap();

    assert_close(output[0], 100.0);
    assert_close(output[1], 100.0);
    assert_close(output[2], 100.0);
}

#[test]
fn ad_rejects_bad_inputs() {
    let mut output = [0.0; 4];
    assert!(AD(&[1.0, 2.0], &[0.0], &[0.5, 1.5], &[10.0, 20.0], &mut output).is_err());
    assert!(AD(
        &[1.0, Float::NAN],
        &[0.0, 1.0],
        &[0.5, 1.5],
        &[10.0, 20.0],
        &mut output,
    )
    .is_err());

    let mut too_small = [0.0; 1];
    let (high, low, close, volume) = fixture();
    assert!(AD(&high, &low, &close, &volume, &mut too_small).is_err());

    let mut ad = AD::new().unwrap();
    assert!(ad
        .next(ADTick {
            high: 1.0,
            low: 0.0,
            close: 0.5,
            volume: Float::INFINITY,
        })
        .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] AD integration tests pass: `cargo test -p ta-core --test volume`
- [x] Inventory tests pass with AD implemented: `cargo test -p ta-core --test inventory`
- [x] AD compiles and tests under f32 precision: `cargo test -p ta-core --features f32 --test volume`

#### Manual Verification:
- [x] `crates/ta-core/src/lib.rs` exposes `pub mod volume;` only after the facade exists, with no adapter changes.
- [x] AD uses shared `Float`, `Result`, `OutputRange`, and validators; no private SIMD/backend APIs.
- [x] AD has lookback 0, emits on the first streaming tick, and treats every `high <= low` range as a zero contribution.
- [x] Stateful `AD` derives `Clone` but not `Copy`, and reset restores zero cumulative state.

## Phase 2: OBV

### Overview
Adds developer-selected warm-up-aligned OBV end-to-end; depends on Phase 1.

### Changes Required:

#### 1. crates/ta-core/src/volume/obv.rs
**File**: crates/ta-core/src/volume/obv.rs
**Changes**: NEW — OBV batch, vec, struct, streaming, and reset.

```rust
//! On-Balance Volume (OBV).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed close/volume inputs for [`OBV`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct OBVInput<'a> {
    /// Close price series.
    pub close: &'a [Float],
    /// Volume series.
    pub volume: &'a [Float],
}

/// One close/volume tick for [`OBV`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OBVTick {
    /// Close price.
    pub close: Float,
    /// Volume.
    pub volume: Float,
}

fn validate_close_volume(close: &[Float], volume: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[("close", close.len()), ("volume", volume.len())])?;
    validate_finite_slices(&[("close", close), ("volume", volume)])?;
    Ok(len)
}

#[inline]
fn update_obv(value: Float, current_close: Float, previous_close: Float, volume: Float) -> Float {
    if current_close > previous_close {
        value + volume
    } else if current_close < previous_close {
        value - volume
    } else {
        value
    }
}

/// On-Balance Volume batch function using first-observation warm-up.
#[allow(non_snake_case)]
pub fn OBV(close: &[Float], volume: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
    let len = validate_close_volume(close, volume)?;
    let lookback = 1;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("OBV", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut value = 0.0 as Float;
    for output_idx in 0..count {
        let input_idx = lookback + output_idx;
        value = update_obv(
            value,
            close[input_idx],
            close[input_idx - 1],
            volume[input_idx],
        );
        out_real[output_idx] = value;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes OBV into a full-length vector padded at the warm-up index.
#[allow(non_snake_case)]
pub fn OBV_vec(close: &[Float], volume: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(close.len());
    let range = OBV(close, volume, &mut compact)?;
    Ok(padded_from_compact(
        close.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// On-Balance Volume indicator using first-observation warm-up.
#[derive(Debug, Clone, Default)]
pub struct OBV {
    previous_close: Option<Float>,
    value: Float,
}

impl OBV {
    /// Creates a new OBV indicator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            previous_close: None,
            value: 0.0 as Float,
        })
    }

    /// Computes compact OBV outputs.
    pub fn compute(
        &self,
        close: &[Float],
        volume: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        OBV(close, volume, out_real)
    }

    /// Computes full-length padded OBV outputs.
    pub fn compute_to_vec(&self, close: &[Float], volume: &[Float]) -> Result<Vec<Float>> {
        OBV_vec(close, volume)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: OBVTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for OBV {
    type Input<'a> = OBVInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        1
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        OBV(input.close, input.volume, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        OBV_vec(input.close, input.volume)
    }
}

impl StreamingIndicator for OBV {
    type Tick = OBVTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[("close", &[input.close]), ("volume", &[input.volume])])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        self.value = update_obv(self.value, input.close, previous_close, input.volume);
        self.previous_close = Some(input.close);
        Ok(Some(self.value))
    }
}

impl Resettable for OBV {
    fn reset(&mut self) {
        self.previous_close = None;
        self.value = 0.0 as Float;
    }
}
```

#### 2. crates/ta-core/src/volume/mod.rs
**File**: crates/ta-core/src/volume/mod.rs
**Changes**: MODIFY — add OBV module and exports.

```rust
mod ad;
mod obv;

pub use ad::{ADInput, ADTick, AD_vec, AD};
pub use obv::{OBVInput, OBVTick, OBV_vec, OBV};
```

#### 3. crates/ta-core/src/inventory.rs:12,201-204
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark OBV implemented and increment the implemented count to 44.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 44;

// Volume Indicators — 3 functions.
function!("AD", VolumeIndicators, Implemented),
function!("ADOSC", VolumeIndicators, Planned),
function!("OBV", VolumeIndicators, Implemented),
```

#### 4. crates/ta-core/tests/inventory.rs:1-235
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — add OBV to implemented/trait checks and remove it from deferred functions.

```rust
// Replace the Phase 1 Volume import with:
use ta_core::volume::{AD, OBV};

// Add after the existing `"AD",` entry inside
// first_tranche_functions_are_marked_implemented:
"OBV",

// Add after the existing AD assertions inside
// first_tranche_structs_implement_batch_and_streaming_traits:
assert_indicator::<OBV>();
assert_streaming::<OBV>();

// Remove only `"OBV",` from the deferred_functions_remain_planned list.
```

#### 5. crates/ta-core/tests/volume.rs
**File**: crates/ta-core/tests/volume.rs
**Changes**: MODIFY — add OBV compact, padded, trait, streaming/reset, flat-close, and invalid-input tests.

```rust
// Replace the Volume import with:
use ta_core::volume::{ADInput, ADTick, AD_vec, OBVInput, OBVTick, OBV_vec, AD, OBV};

#[test]
fn obv_function_writes_compact_outputs() {
    let (_, _, close, volume) = fixture();
    let mut output = [0.0; 4];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 3));
    for (actual, expected) in output[..3]
        .iter()
        .copied()
        .zip([200.0, 150.0, 450.0])
    {
        assert_close(actual, expected);
    }
}

#[test]
fn obv_vec_returns_padded_outputs() {
    let (_, _, close, volume) = fixture();
    let output = OBV_vec(&close, &volume).unwrap();

    assert_eq!(output.len(), close.len());
    assert!(output[0].is_nan());
    assert_close(output[1], 200.0);
    assert_close(output[2], 150.0);
    assert_close(output[3], 450.0);
}

#[test]
fn obv_struct_implements_indicator_compute() {
    let (_, _, close, volume) = fixture();
    let obv = OBV::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &obv,
        OBVInput {
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(obv.lookback(), 1);
    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 200.0);
}

#[test]
fn obv_streaming_matches_batch_and_reset() {
    let (_, _, close, volume) = fixture();
    let mut batch = [0.0; 4];
    let range = OBV(&close, &volume, &mut batch).unwrap();
    let mut obv = OBV::new().unwrap();

    assert!(obv
        .next(OBVTick {
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_none());
    for idx in 1..close.len() {
        let streamed = obv
            .next(OBVTick {
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap()
            .unwrap();
        assert_close(streamed, batch[idx - range.beg_idx]);
    }

    obv.reset();
    assert!(obv
        .next_checked(OBVTick {
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_nan());
}

#[test]
fn obv_flat_close_leaves_value_unchanged() {
    let close = [10.0, 10.0, 9.0];
    let volume = [100.0, 50.0, 25.0];
    let mut output = [0.0; 3];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 2));
    assert_close(output[0], 0.0);
    assert_close(output[1], -25.0);
}

#[test]
fn obv_rejects_bad_inputs() {
    let mut output = [0.0; 4];
    assert!(OBV(&[1.0, 2.0], &[10.0], &mut output).is_err());
    assert!(OBV(&[1.0, Float::NAN], &[10.0, 20.0], &mut output).is_err());
    assert!(OBV(&[1.0], &[10.0], &mut output).is_err());

    let mut too_small = [0.0; 1];
    let (_, _, close, volume) = fixture();
    assert!(OBV(&close, &volume, &mut too_small).is_err());

    let mut obv = OBV::new().unwrap();
    assert!(obv
        .next(OBVTick {
            close: 1.0,
            volume: Float::INFINITY,
        })
        .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] Volume integration tests pass with OBV: `cargo test -p ta-core --test volume`
- [x] Inventory tests pass with AD and OBV implemented: `cargo test -p ta-core --test inventory`
- [x] OBV compiles and tests under f32 precision: `cargo test -p ta-core --features f32 --test volume`

#### Manual Verification:
- [x] OBV batch `OutputRange::beg_idx`, struct `lookback()`, streaming warm-up, and `_vec` NaN padding all equal 1.
- [x] OBV starts cumulative state at zero, ignores the first observation's volume, and applies current volume only from index 1.
- [x] Equal closes leave OBV unchanged; higher/lower closes add/subtract current volume.
- [x] Stateful `OBV` derives `Clone` but not `Copy`, and reset restores first-tick warm-up behavior.

## Phase 3: ADOSC

### Overview
Adds ADOSC by feeding cumulative AD values through local fast/slow EMA states; depends on Phases 1-2.

### Changes Required:

#### 1. crates/ta-core/src/volume/adosc.rs
**File**: crates/ta-core/src/volume/adosc.rs
**Changes**: NEW — ADOSC batch, vec, struct, dual-EMA streaming, strict period validation, and reset.

```rust
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
        cumulative +=
            super::ad::money_flow_volume(high[idx], low[idx], close[idx], volume[idx]);
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
        ADOSC_vec(
            high,
            low,
            close,
            volume,
            self.fastperiod,
            self.slowperiod,
        )
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
```

#### 2. crates/ta-core/src/volume/mod.rs
**File**: crates/ta-core/src/volume/mod.rs
**Changes**: MODIFY — add ADOSC module and exports.

```rust
mod ad;
mod adosc;
mod obv;

pub use ad::{ADInput, ADTick, AD_vec, AD};
pub use adosc::{ADOSCInput, ADOSCTick, ADOSC_vec, ADOSC};
pub use obv::{OBVInput, OBVTick, OBV_vec, OBV};
```

#### 3. crates/ta-core/src/inventory.rs:12,201-204
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark ADOSC implemented and increment the implemented count to 45.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 45;

// Volume Indicators — 3 functions.
function!("AD", VolumeIndicators, Implemented),
function!("ADOSC", VolumeIndicators, Implemented),
function!("OBV", VolumeIndicators, Implemented),
```

#### 4. crates/ta-core/tests/inventory.rs:1-235
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — add ADOSC to implemented and trait-conformance checks.

```rust
// Replace the Phase 2 Volume import with:
use ta_core::volume::{AD, ADOSC, OBV};

// Add after the existing `"AD",` entry inside
// first_tranche_functions_are_marked_implemented:
"ADOSC",

// Add after the existing AD assertions inside
// first_tranche_structs_implement_batch_and_streaming_traits:
assert_indicator::<ADOSC>();
assert_streaming::<ADOSC>();
```

#### 5. crates/ta-core/tests/volume.rs
**File**: crates/ta-core/tests/volume.rs
**Changes**: MODIFY — add ADOSC expected-value, compact/padded, trait, streaming/reset, period, and invalid-input tests.

```rust
// Replace the Volume import with:
use ta_core::volume::{
    ADInput, ADTick, AD_vec, ADOSCInput, ADOSCTick, ADOSC_vec, OBVInput, OBVTick, OBV_vec, AD,
    ADOSC, OBV,
};

fn adosc_fixture() -> ([Float; 5], [Float; 5], [Float; 5], [Float; 5]) {
    (
        [2.0; 5],
        [0.0; 5],
        [2.0; 5],
        [1.0, 2.0, 3.0, 4.0, 5.0],
    )
}

#[test]
fn adosc_function_writes_compact_outputs() {
    let (high, low, close, volume) = adosc_fixture();
    let mut output = [0.0; 5];

    let range = ADOSC(&high, &low, &close, &volume, 2, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 4.0 / 3.0);
    assert_close(output[1], 14.0 / 9.0);
    assert_close(output[2], 103.0 / 54.0);
}

#[test]
fn adosc_vec_returns_padded_outputs() {
    let (high, low, close, volume) = adosc_fixture();
    let output = ADOSC_vec(&high, &low, &close, &volume, 2, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..2].iter().all(|value| value.is_nan()));
    assert_close(output[2], 4.0 / 3.0);
    assert_close(output[4], 103.0 / 54.0);
}

#[test]
fn adosc_struct_implements_indicator_compute() {
    let (high, low, close, volume) = adosc_fixture();
    let adosc = ADOSC::new(2, 3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &adosc,
        ADOSCInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(adosc.fastperiod(), 2);
    assert_eq!(adosc.slowperiod(), 3);
    assert_eq!(adosc.lookback(), 2);
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 4.0 / 3.0);
}

#[test]
fn adosc_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = adosc_fixture();
    let mut batch = [0.0; 5];
    let range = ADOSC(&high, &low, &close, &volume, 2, 3, &mut batch).unwrap();
    let mut adosc = ADOSC::new(2, 3).unwrap();

    for idx in 0..high.len() {
        let streamed = adosc
            .next_checked(ADOSCTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(streamed.is_nan());
        } else {
            assert_close(streamed, batch[idx - range.beg_idx]);
        }
    }

    adosc.reset();
    assert!(adosc
        .next_checked(ADOSCTick {
            high: high[0],
            low: low[0],
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_nan());
}

#[test]
fn adosc_rejects_invalid_periods_and_inputs() {
    assert!(ADOSC::new(0, 3).is_err());
    assert!(ADOSC::new(2, 0).is_err());
    assert!(ADOSC::new(3, 3).is_err());
    let ordering_error = ADOSC::new(4, 3).unwrap_err().to_string();
    assert!(ordering_error.contains("4 (slowperiod=3)"));
    assert!(ADOSC::new(1, 2).is_ok());

    let (high, low, close, volume) = adosc_fixture();
    let mut output = [0.0; 5];
    assert!(ADOSC(
        &high[..2],
        &low[..2],
        &close[..2],
        &volume[..2],
        2,
        3,
        &mut output,
    )
    .is_err());

    let mut invalid_high = high;
    invalid_high[2] = Float::NAN;
    assert!(ADOSC(
        &invalid_high,
        &low,
        &close,
        &volume,
        2,
        3,
        &mut output,
    )
    .is_err());

    let mut too_small = [0.0; 1];
    assert!(ADOSC(
        &high,
        &low,
        &close,
        &volume,
        2,
        3,
        &mut too_small,
    )
    .is_err());

    let mut adosc = ADOSC::new(2, 3).unwrap();
    assert!(adosc
        .next(ADOSCTick {
            high: 2.0,
            low: 0.0,
            close: 2.0,
            volume: Float::INFINITY,
        })
        .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] Volume integration tests pass with all three indicators: `cargo test -p ta-core --test volume`
- [x] Inventory tests pass with the complete Volume group implemented: `cargo test -p ta-core --test inventory`
- [x] Volume tests pass under f32 precision: `cargo test -p ta-core --features f32 --test volume`

#### Manual Verification:
- [x] ADOSC reuses Phase 1 HLCV and money-flow helpers and does not allocate an intermediate AD vector.
- [x] ADOSC rejects zero periods and every `fastperiod >= slowperiod` configuration.
- [x] ADOSC batch lookback, struct `lookback()`, streaming warm-up, and `_vec` NaN prefix all equal `slowperiod - 1`.
- [x] Local EMA states use the same SMA seed and recursive multiplier formula as `overlap::EMA`, and reset clears cumulative AD plus both EMA states.
- [x] All three Volume functions are marked implemented and `IMPLEMENTED_FUNCTION_COUNT` equals 45.

## Phase 4: Benchmarks

### Overview
Adds public compact-kernel Criterion coverage for the completed Volume group; depends on Phase 3.

### Changes Required:

#### 1. crates/ta-benchmarks/benches/first_tranche.rs:10-443
**File**: crates/ta-benchmarks/benches/first_tranche.rs
**Changes**: MODIFY — add volume fixture, imports, benchmark group, and Criterion registration.

```rust
// Extend the ta_core imports with:
volume::{AD, ADOSC, OBV},

// Add next to PERIOD:
const ADOSC_FAST_PERIOD: usize = 3;
const ADOSC_SLOW_PERIOD: usize = 10;

fn hlcv_fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    let (_open, high, low, close) = ohlc_fixture(size);
    let volume = (0..size)
        .map(|idx| ((idx % 1_000) + 1) as Float * 10.0 as Float)
        .collect();
    (high, low, close, volume)
}

fn bench_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/volume");

    for &size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("AD_compact", size),
            &size,
            |b, &size| {
                let (high, low, close, volume) = hlcv_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = AD(
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(volume.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AD benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("OBV_compact", size),
            &size,
            |b, &size| {
                let (_high, _low, close, volume) = hlcv_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = OBV(
                        black_box(close.as_slice()),
                        black_box(volume.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid OBV benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ADOSC_compact", size),
            &size,
            |b, &size| {
                let (high, low, close, volume) = hlcv_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = ADOSC(
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(volume.as_slice()),
                        black_box(ADOSC_FAST_PERIOD),
                        black_box(ADOSC_SLOW_PERIOD),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid ADOSC benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );
    }

    group.finish();
}

// Register with the existing Criterion group:
criterion_group!(
    benches,
    bench_overlap_sma,
    bench_overlap_moving_averages,
    bench_price_transform,
    bench_volatility,
    bench_volume,
    bench_math_transform,
    bench_math_operators
);
```

### Success Criteria:

#### Automated Verification:
- [x] Benchmark target compiles with Volume group registered: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`
- [x] Workspace check passes after all Volume phases: `cargo check --workspace`
- [x] Workspace tests pass after all Volume phases: `cargo test --workspace`
- [x] f32 precision tests pass for ta-core: `cargo test -p ta-core --features f32`
- [x] Formatting is clean: `cargo fmt --all -- --check`

#### Manual Verification:
- [x] `bench_volume` imports and calls only public `ta_core::volume::{AD, OBV, ADOSC}` APIs.
- [x] HLCV fixtures and reusable output buffers are allocated outside every `b.iter()`.
- [x] `bench_volume` is registered in `criterion_group!` and all three compact kernels have size-matrix coverage.
- [x] No `ta-py`, `ta-wasm`, Cargo manifest, workspace membership, or SIMD backend files are changed by the plan.

## Ordering Constraints
- Phase 1 creates the facade and shared AD/HLCV helpers.
- Phase 2 extends the facade/tests/inventory after Phase 1; it does not depend on AD logic but remains sequential because shared files evolve incrementally.
- Phase 3 depends on Phase 1 helpers and the prior facade/test state.
- Phase 4 imports all three finalized public kernels and must run last.
- Implementation phases are sequential; inventory count/status and test imports intentionally advance per phase.

## Verification Notes
- Run the dedicated Volume integration test target after each core phase: `cargo test -p ta-core --test volume`.
- Run inventory tests after each ledger update: `cargo test -p ta-core --test inventory`.
- Verify f32 compatibility on the terminal core phase: `cargo test -p ta-core --features f32 --test volume`.
- Verify terminal workspace health: `cargo check --workspace` and `cargo test --workspace`.
- Verify formatting: `cargo fmt --all -- --check`.
- Compile Criterion coverage without executing it: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`.
- Inspect that `ta-py`, `ta-wasm`, Cargo manifests, and SIMD backends remain unchanged.
- Verify each batch `OutputRange` aligns with streaming warm-up and `_vec` padding.
- Verify stateful structs do not derive `Copy`, all constructors are fallible, and reset restores fresh-state behavior.
- Verify non-empty insufficient input uses `validate_input_len`, not raw `output_count`.

## Performance Considerations
- AD and OBV are single-pass O(n) kernels with O(1) state.
- ADOSC is single-pass O(n), computes AD and both EMA states without allocating an intermediate AD vector, and uses O(1) state.
- `_vec` wrappers allocate full-length outputs by design; compact kernels remain caller-buffer based.
- Criterion fixtures and reusable output buffers are allocated outside timed iterations.
- No SIMD specialization is introduced in this tranche.

## Migration Notes
No persisted data, schema, dependency, or workspace migration is required. This adds new Rust APIs and changes inventory statuses from planned to implemented. OBV's selected warm-up semantics are documented as an intentional API behavior before first release.

## Pattern References
- `crates/ta-core/src/price_transform/mod.rs:6-16` — private modules and explicit group exports.
- `crates/ta-core/src/price_transform/typprice.rs:13-138` — named multi-input views/ticks, validation, compact output, and traits.
- `crates/ta-core/src/volatility/trange.rs:37-181` — shared validators/helpers and previous-value streaming/reset state.
- `crates/ta-core/src/overlap/ema.rs:19-32,156-186` — EMA formulas, warm-up, and reset pattern.
- `crates/ta-core/src/overlap/dema.rs:33-48,133-152` — composed streaming states without intermediate allocation.
- `crates/ta-core/src/common.rs:89-209` — validation, compact buffer, and padded output helpers.
- `crates/ta-core/tests/volatility.rs:15-477` — granular compact/padded/streaming/reset/error tests.
- `crates/ta-benchmarks/benches/first_tranche.rs:284-347` — public compact multi-input Criterion group.

## Developer Context
**Q (`crates/ta-core/src/common.rs:89-103`, `crates/ta-core/src/overlap/ema.rs:84-95`): ADOSC 的 fast/slow 周期校验应该怎么做？现有 Rust EMA 只要求 period > 0，但 ADOSC 语义上通常 fast 应短于 slow。**
A: 要求 `fastperiod < slowperiod`。

**Q (root architecture boundary; `crates/ta-core/src/lib.rs:35-48`): Volume tranche 是否继续沿用 Volatility 的边界：只改 ta-core、inventory/tests 和 Criterion benchmark，不加 ta-py/ta-wasm 绑定？**
A: 只做 core + bench。

**Q (`crates/ta-core/src/volatility/trange.rs:156-175`): OBV 首个输出如何对齐？**
A: 首个 tick 预热；batch 从 index 1 输出，padded index 0 为 NaN。

**Q: Design: Volume Indicators。Approach: 新增 `ta_core::volume` facade，分 AD、OBV、ADOSC 三个核心 slice，最后补 inventory/tests/benchmarks；ADOSC 使用本地 EMA 状态，OBV 采用首 tick/index0 预热。准备进入分解吗？**
A: Proceed。

**Q: 4 个 slices：AD foundation、OBV、ADOSC、Criterion benchmarks。批准这个分解吗？**
A: Approve。

**Q: Slice 1/4: AD foundation — 6 个文件。建立 Volume facade 和可批处理/流式/reset 的 AD，并为 ADOSC 留下共享 HLCV/MFV helper。批准吗？**
A: Approve。

**Q: Slice 2/4: OBV — 5 个文件。按已选语义首 tick 预热、index 1 起输出，并已修正 inventory 为 OBV-only delta。批准吗？**
A: Approve。

**Q: Slice 3/4: ADOSC — 5 个文件。复用 AD helper，单遍双 EMA，严格 fast<slow，lookback=slow-1。批准吗？**
A: Approve。

**Q: Cross-slice: OK。Slice 4/4: Benchmarks — 1 个文件。为 AD/OBV/ADOSC 增加完整 size matrix，并承担终态 workspace/f32/fmt 验证。批准吗？**
A: Approve。

**Q: Concern 1：AD 在 `high < low` 时当前公式会产生反向贡献；TA-Lib 对非正 range 应贡献 0。如何处理？**
A: Applied — guard 改为 `range <= 0`，并增加 reversed-range 回归测试。

**Q: Concern 2：ADOSC 周期顺序错误目前只写表达式文本，没有把实际 fast/slow 值放进 `TalibError`。如何处理？**
A: Applied — invalid value 改为包含实际 `fastperiod` 与 `slowperiod` 数值，并增加诊断断言。

## Plan Review (Step 8)

_Independent post-finalization review by artifact-code-reviewer and artifact-coverage-reviewer subagents. Findings triaged at Step 9._

| source | plan-loc | codebase-loc | severity | dimension | finding | recommendation | resolution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code | Phase 1 §2 (ad.rs) | `<n/a>` | concern | code-quality | `money_flow_volume` guards only `range == 0.0`, so finite inputs with `high < low` produce an inverted contribution instead of TA-Lib’s zero contribution for a non-positive range. | Change the guard to `range <= 0.0 as Float` and add a reversed-range test. | applied: non-positive ranges now contribute zero and the test covers flat and reversed ranges |
| code | Phase 3 §1 (adosc.rs) | `crates/ta-core/src/error.rs:123` | concern | code-quality | `TalibError::invalid_parameter` defines its second argument as the invalid value, but ADOSC passes the expression text `"fastperiod >= slowperiod"` rather than the supplied periods, producing misleading diagnostics. | Pass the actual fast and slow period values in the error’s value field. | applied: invalid-parameter diagnostics now include actual fast and slow values with a regression assertion |

_Coverage reviewer found no uncovered verification or pattern-reference intent._

## Plan History
- Phase 1: AD foundation — revised after review: non-positive range guard and regression coverage applied
- Phase 2: OBV — revised: inventory test changes corrected to an OBV-only incremental delta, then approved
- Phase 3: ADOSC — revised after review: parameter-order diagnostics now include actual values
- Phase 4: Benchmarks — approved as generated

## References
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — official inventory and ta-core boundary research.
- `.rpiv/artifacts/plans/2026-07-09_21-49-41_volatility-indicators.md` — most recent complete-group facade, state, testing, inventory, and benchmark precedent.
- `.rpiv/artifacts/validation/2026-07-12_14-15-57_实现volatility分组指标.md` — validation lessons for the preceding group.
- `crates/ta-core/src/price_transform/typprice.rs` — multi-input API precedent.
- `crates/ta-core/src/overlap/ema.rs` — EMA state precedent.
