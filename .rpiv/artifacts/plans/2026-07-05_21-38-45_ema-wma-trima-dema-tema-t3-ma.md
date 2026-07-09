---
date: 2026-07-05T21:38:45+0800
author: unknown
commit: cfdf1e5
branch: main
repository: fast-ta
topic: "EMA/WMA/TRIMA/DEMA/TEMA/T3/MA"
tags: [plan, ta-core, talib, overlap, moving-averages]
status: ready
parent: ".rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md"
phase_count: 6
phases:
  - { n: 1, title: EMA foundation }
  - { n: 2, title: Weighted and triangular moving averages }
  - { n: 3, title: EMA-derived composites }
  - { n: 4, title: T3 with vfactor }
  - { n: 5, title: MA dispatcher }
  - { n: 6, title: Moving-average benchmarks }
unresolved_phase_count: 0
last_updated: 2026-07-05T21:38:45+0800
last_updated_by: unknown
---

# EMA/WMA/TRIMA/DEMA/TEMA/T3/MA Implementation Plan

## Overview

Implement the next overlap moving-average tranche in `ta-core`: standalone EMA, WMA, TRIMA, DEMA, TEMA, and T3 indicators, plus a TA-Lib-compatible `MA` dispatcher in the overlap module. The plan follows the existing `SMA` dual API pattern: compact uppercase functions returning `OutputRange`, padded `*_vec` wrappers, uppercase struct surfaces, `Indicator`/`StreamingIndicator` implementations where applicable, source inventory updates, integration tests, and Criterion benchmark coverage.

## Requirements

- Implement each concrete moving-average algorithm separately in Rust `ta-core`: `EMA`, `WMA`, `TRIMA`, `DEMA`, `TEMA`, and `T3`.
- Implement `overlap::MA` as the official TA-Lib Overlap Studies dispatcher, not as a separate formula.
- Keep algorithm logic out of `ta-py`, `ta-wasm`, and benchmark crates.
- Use strict Rust validation: invalid periods/parameters, non-finite inputs, and undersized outputs return `TalibError` via `Result`.
- Preserve compact TA-Lib-style output plus full-length padded wrapper output.
- Add integration tests for compact output alignment, padded NaN warm-up, invalid inputs, struct APIs, streaming/reset behavior, `MAType` dispatch, and `T3` vfactor validation.
- Update the official inventory ledger as each function becomes implemented.
- Extend Criterion benchmarks using only public `ta_core` APIs.

## Current State Analysis

`ta-core` already has the first-tranche compact-output infrastructure and one overlap moving average, `SMA`. The requested functions are recorded in inventory as planned, and adapters currently expose only placeholders, so this tranche should stay in core plus benchmarks.

### Key Discoveries

- `crates/ta-core/src/overlap/sma.rs:23` defines the compact uppercase `SMA(real, timeperiod, out_real) -> Result<OutputRange>` pattern to follow.
- `crates/ta-core/src/overlap/sma.rs:49` defines the padded `SMA_vec` convenience wrapper, currently optimized to write the padded vector directly.
- `crates/ta-core/src/overlap/sma.rs:76` and `:137` define the uppercase struct + `Indicator` implementation pattern for overlap indicators.
- `crates/ta-core/src/overlap/sma.rs:162` defines the `StreamingIndicator` warm-up pattern with `Ok(None)` before the first valid output.
- `crates/ta-core/src/overlap/mod.rs:3-5` shows family-private modules plus public re-exports; new overlap modules should be declared only in the phase where their files are created.
- `crates/ta-core/src/common.rs:101`, `:108`, `:118`, and `:159` provide period, input length, output length, and finite-input validators.
- `crates/ta-core/src/inventory.rs:153-168` lists `DEMA`, `EMA`, `MA`, `T3`, `TEMA`, `TRIMA`, and `WMA` as planned overlap functions while `SMA` is already implemented.
- `crates/ta-core/tests/overlap_sma.rs:12-75` is the closest test template for compact output, padded output, trait compute, validation, streaming, and reset.
- `crates/ta-core/tests/inventory.rs:69-127` is the inventory status and trait assertion registration point.
- `crates/ta-benchmarks/benches/first_tranche.rs:58-89` benchmarks overlap `SMA` using public APIs; `:228-235` registers benchmark groups.
- `crates/ta-py/src/lib.rs:12-20` and `crates/ta-wasm/src/lib.rs:9-17` are placeholder adapter surfaces and are intentionally out of scope for this plan.

## Desired End State

```rust
use ta_core::overlap::{EMA, EMA_vec, MAType, MA, T3};
use ta_core::{Float, OutputRange};

let real: Vec<Float> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let mut compact = vec![0.0 as Float; real.len()];
let range: OutputRange = EMA(&real, 3, &mut compact)?;
assert_eq!(range.beg_idx, 2);

let padded = EMA_vec(&real, 3)?;
assert!(padded[0].is_nan());
assert!(padded[1].is_nan());

let ma_range = MA(&real, 3, MAType::EMA, &mut compact)?;
assert_eq!(ma_range, range);

let t3 = T3::with_default_vfactor(5)?;
assert_eq!(t3.vfactor(), 0.7 as Float);
```

```rust
use ta_core::overlap::{DEMA, TEMA, TRIMA_vec, WMA_vec};

let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
let wma = WMA_vec(&real, 3)?;
let trima = TRIMA_vec(&real, 3)?;
let mut compact = [0.0; 7];
DEMA(&real, 3, &mut compact)?;
TEMA(&real, 3, &mut compact)?;
```

## What We're NOT Doing

- No Python or WASM bindings in this plan; `ta-py` and `ta-wasm` remain adapter-only placeholders for this tranche.
- No KAMA or MAMA algorithm implementation, even though their `MAType` variants are exposed for official compatibility; unsupported dispatcher branches return `TalibError::NotImplemented`.
- No SIMD rewrites or private architecture backend calls; moving-average kernels stay scalar/shared-helper based in this plan.
- No non-TA-Lib extras such as WWMA/HMA/VWAP.
- No persisted data/schema migration.

## Decisions

### Follow the existing overlap dual API

Decision: New indicators use compact uppercase free functions, padded `*_vec` wrappers, uppercase structs, and trait implementations, following `crates/ta-core/src/overlap/sma.rs:23`, `:49`, `:76`, `:137`, and `:162`.

### Keep strict Rust validation

Decision: Constructors and batch APIs return `Result`/`TalibError` for invalid periods, invalid `T3` vfactor, non-finite inputs, insufficient data, and undersized output buffers. This follows `crates/ta-core/src/common.rs:101`, `:118`, and `:159`, and the tests at `crates/ta-core/tests/overlap_sma.rs:52`.

### Scope is core plus benchmarks only

Decision: Implement `ta-core` algorithms, tests, inventory updates, and Criterion benchmarks now. Defer Python/WASM bindings because `crates/ta-py/src/lib.rs:12-20` and `crates/ta-wasm/src/lib.rs:9-17` are still placeholder adapters and the developer selected core+benchmarks scope.

### Implement `MA` as an overlap dispatcher

Ambiguity: `MA` appears in the requested list but is not a distinct moving-average formula. It is an official TA-Lib Overlap Studies function (`crates/ta-core/src/inventory.rs:157`).
Explored:
- Defer `MA`: keeps only standalone formulas but leaves a requested official function planned.
- Implement `MA`: adds a thin compatibility dispatcher over standalone implementations.
Decision: Implement `overlap::MA` in `crates/ta-core/src/overlap/ma.rs` and re-export it from overlap.

### `MAType` exposes full official variants

Ambiguity: `KAMA` and `MAMA` are still planned (`crates/ta-core/src/inventory.rs:156`, `:158`), but official TA-Lib `MAType` includes them.
Explored:
- Supported-only enum: avoids unsupported variants but diverges from official MAType.
- Full official enum: matches official MAType and returns explicit errors for missing algorithms.
Decision: Expose full official variants (`SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3`); `KAMA` and `MAMA` branches return `TalibError::NotImplemented` until their algorithms land.

### `T3` exposes explicit and default-vfactor APIs

Ambiguity: TA-Lib `T3` has `vfactor` default `0.7` and range `[0, 1]`, while existing `SMA` is period-only.
Explored:
- Explicit only: complete but less ergonomic.
- Default plus explicit: complete and mirrors TA-Lib defaults.
- Default only: simpler but incomplete.
Decision: Implement explicit `T3(real, timeperiod, vfactor, out_real)` and `T3::new(timeperiod, vfactor)`, plus default-vfactor convenience such as `T3_DEFAULT_VFACTOR`, `T3_with_default_vfactor`, `T3_vec_with_default_vfactor`, and `T3::with_default_vfactor(timeperiod)`. The `MAType::T3` branch uses default `0.7`.

## Phase 1: EMA foundation

### Overview

Adds EMA as the foundational moving average and internal EMA helper needed by later EMA-derived indicators; this phase does not declare future overlap modules. Depends on existing `SMA`/common infrastructure; later phases depend on this phase.

### Changes Required:

#### 1. crates/ta-core/src/overlap/ema.rs
**File**: `crates/ta-core/src/overlap/ema.rs`
**Changes**: NEW — EMA compact API, padded wrapper, internal reusable EMA helper, struct surface, streaming/reset support.

```rust
//! Exponential Moving Average (EMA).
//!
//! This module exposes both the TA-Lib-style zero-copy function [`EMA`] and the
//! stateful [`EMA`] struct. The free function writes compact valid outputs and
//! returns an [`OutputRange`](crate::OutputRange); [`EMA_vec`] returns a
//! full-length padded vector for convenience.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
pub(super) fn ema_multiplier(timeperiod: usize) -> Float {
    2.0 as Float / (timeperiod as Float + 1.0 as Float)
}

#[inline]
pub(super) fn ema_seed(real: &[Float], timeperiod: usize) -> Float {
    real[..timeperiod].iter().copied().sum::<Float>() / timeperiod as Float
}

#[inline]
pub(super) fn ema_step(previous: Float, input: Float, multiplier: Float) -> Float {
    (input - previous) * multiplier + previous
}

/// TA-Lib-style Exponential Moving Average batch function.
///
/// Valid outputs are written compactly starting at `out_real[0]`. The returned
/// range maps those compact values back to their original input positions.
#[allow(non_snake_case)]
pub fn EMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("EMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let multiplier = ema_multiplier(timeperiod);
    let mut ema = ema_seed(real, timeperiod);
    out_real[0] = ema;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        ema = ema_step(ema, real[input_idx], multiplier);
        out_real[output_idx] = ema;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes EMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn EMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = EMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    multiplier: Float,
    count: usize,
    sum: Float,
    value: Float,
}

impl EMA {
    /// Creates a new EMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self {
            period: timeperiod,
            multiplier: ema_multiplier(timeperiod),
            count: 0,
            sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    /// Creates a new EMA indicator seeded by processing `real` in order.
    pub fn from_data(timeperiod: usize, real: &[Float]) -> Result<Self> {
        validate_finite_slice("real", real)?;
        let mut ema = Self::new(timeperiod)?;
        for &value in real {
            let _ = ema.next(value)?;
        }
        Ok(ema)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact EMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        EMA(real, self.period, out_real)
    }

    /// Computes full-length padded EMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        EMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for EMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        EMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        EMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for EMA {
    type Tick = Float;
    type TickOutput = Float;

    #[inline]
    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        if self.count < self.period {
            self.sum += input;
            self.count += 1;

            if self.count < self.period {
                return Ok(None);
            }

            self.value = self.sum / self.period as Float;
            return Ok(Some(self.value));
        }

        self.value = ema_step(self.value, input, self.multiplier);
        Ok(Some(self.value))
    }
}

impl Resettable for EMA {
    fn reset(&mut self) {
        self.count = 0;
        self.sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
```

#### 2. crates/ta-core/src/overlap/mod.rs:3-5
**File**: `crates/ta-core/src/overlap/mod.rs`
**Changes**: MODIFY — declare `ema` and re-export `EMA`/`EMA_vec` alongside existing `SMA` exports.

```rust
mod ema;
mod sma;

pub use ema::{EMA_vec, EMA};
pub use sma::{SMA_vec, SMA};
```

#### 3. crates/ta-core/tests/overlap_moving_averages.rs
**File**: `crates/ta-core/tests/overlap_moving_averages.rs`
**Changes**: NEW — EMA compact/padded/struct/streaming/validation tests.

```rust
use ta_core::overlap::{EMA_vec, EMA};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-6 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn ema_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let mut output = [0.0; 5];

    let range = EMA(&real, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 7.0 as Float / 3.0 as Float);
    assert_close(output[1], 31.0 as Float / 6.0 as Float);
    assert_close(output[2], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_vec_returns_padded_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];

    let output = EMA_vec(&real, 3).unwrap();

    assert_eq!(output.len(), real.len());
    assert!(output[0].is_nan());
    assert!(output[1].is_nan());
    assert_close(output[2], 7.0 as Float / 3.0 as Float);
    assert_close(output[3], 31.0 as Float / 6.0 as Float);
    assert_close(output[4], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_struct_implements_indicator_compute() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let ema = EMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&ema, &real, &mut compact).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 7.0 as Float / 3.0 as Float);
    assert_close(compact[2], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_rejects_invalid_parameters_and_inputs() {
    assert!(EMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(EMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(EMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(EMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(EMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}

#[test]
fn ema_streaming_next_and_reset_are_safe() {
    let mut ema = EMA::new(3).unwrap();

    assert!(ema.next_checked(1.0).unwrap().is_nan());
    assert!(ema.next_checked(2.0).unwrap().is_nan());
    assert_close(ema.next_checked(4.0).unwrap(), 7.0 as Float / 3.0 as Float);
    assert_close(ema.next_checked(8.0).unwrap(), 31.0 as Float / 6.0 as Float);

    ema.reset();
    assert!(ema.next_checked(10.0).unwrap().is_nan());
    assert!(ema.next(Float::NAN).is_err());
}
```

#### 4. crates/ta-core/src/inventory.rs:12,153-168
**File**: `crates/ta-core/src/inventory.rs`
**Changes**: MODIFY — increment implemented count to 33 and mark `EMA` implemented.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 33;
```

```rust
function!("DEMA", OverlapStudies, Planned),
function!("EMA", OverlapStudies, Implemented),
function!("HT_TRENDLINE", OverlapStudies, Planned),
```

#### 5. crates/ta-core/tests/inventory.rs:9-13,69-127,196-201
**File**: `crates/ta-core/tests/inventory.rs`
**Changes**: MODIFY — import/assert `EMA`, include it in implemented status and trait checks, and keep non-implemented overlap functions planned.

```rust
use ta_core::overlap::{EMA, SMA};
```

```rust
let implemented = [
    "SMA",
    "EMA",
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
```

```rust
assert_indicator::<SMA>();
assert_streaming::<SMA>();
assert_indicator::<EMA>();
assert_streaming::<EMA>();
```

### Success Criteria:

#### Automated Verification:
- [x] EMA moving-average tests pass: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests pass after marking EMA implemented: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after adding EMA exports: `cargo test -p ta-core`
- [x] f32 precision compiles for EMA recursion: `cargo test -p ta-core --features f32`
- [x] Workspace checks after EMA public export: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] EMA is wired in overlap module: `grep -n "mod ema\|pub use ema" crates/ta-core/src/overlap/mod.rs`
- [x] EMA inventory status is implemented: `grep -n 'function!("EMA", OverlapStudies, Implemented)' crates/ta-core/src/inventory.rs`
- [x] No future overlap modules are declared early in Phase 1: `grep -n "mod wma\|mod trima\|mod dema\|mod tema\|mod t3\|mod ma" crates/ta-core/src/overlap/mod.rs` returns no matches

#### Manual Verification:
- [x] EMA batch seed is the SMA of the first `timeperiod` values and first output begins at `timeperiod - 1`.
- [x] `EMA_vec` preserves full input length and pads warm-up values with `Float::NAN`.
- [x] EMA streaming returns warm-up `None`/`Float::NAN` before the first valid output and resets all state.
- [x] `EMA::new(0)` and non-finite inputs return `TalibError` instead of panicking.

## Phase 2: Weighted and triangular moving averages

### Overview

Adds WMA and TRIMA as standalone rolling-window moving averages. Depends on Phase 1 test/export patterns; can be reviewed independently from EMA-derived phases after Phase 1.

### Changes Required:

#### 1. crates/ta-core/src/overlap/wma.rs
**File**: `crates/ta-core/src/overlap/wma.rs`
**Changes**: NEW — WMA compact API, padded wrapper, struct surface, streaming/reset support.

```rust
//! Weighted Moving Average (WMA).
//!
//! Valid batch outputs are written compactly. Padded wrappers preserve input
//! length and fill warm-up positions with `Float::NAN`.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn wma_denominator(timeperiod: usize) -> Float {
    (timeperiod * (timeperiod + 1) / 2) as Float
}

/// TA-Lib-style Weighted Moving Average batch function.
#[allow(non_snake_case)]
pub fn WMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("WMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let denominator = wma_denominator(timeperiod);
    let mut window_sum = real[..timeperiod].iter().copied().sum::<Float>();
    let mut weighted_sum = real[..timeperiod]
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx + 1) as Float * value)
        .sum::<Float>();
    out_real[0] = weighted_sum / denominator;

    for output_idx in 1..count {
        let new_idx = output_idx + timeperiod - 1;
        let old_idx = output_idx - 1;
        weighted_sum = weighted_sum - window_sum + timeperiod as Float * real[new_idx];
        window_sum += real[new_idx] - real[old_idx];
        out_real[output_idx] = weighted_sum / denominator;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes WMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn WMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = WMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Weighted Moving Average indicator.
#[derive(Debug, Clone)]
pub struct WMA {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl WMA {
    /// Creates a new WMA indicator.
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
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact WMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        WMA(real, self.period, out_real)
    }

    /// Computes full-length padded WMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        WMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for WMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        WMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        WMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for WMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Ok(None);
        }

        let weighted_sum = (0..self.period)
            .map(|offset| {
                let ordered_idx = (self.index + offset) % self.period;
                (offset + 1) as Float * self.buffer[ordered_idx]
            })
            .sum::<Float>();
        Ok(Some(weighted_sum / wma_denominator(self.period)))
    }
}

impl Resettable for WMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}
```

#### 2. crates/ta-core/src/overlap/trima.rs
**File**: `crates/ta-core/src/overlap/trima.rs`
**Changes**: NEW — TRIMA compact API, padded wrapper, struct surface, streaming/reset support.

```rust
//! Triangular Moving Average (TRIMA).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn trima_weight(index: usize, timeperiod: usize) -> usize {
    if timeperiod % 2 == 1 {
        let center = timeperiod / 2;
        if index <= center {
            index + 1
        } else {
            timeperiod - index
        }
    } else {
        let half = timeperiod / 2;
        if index < half {
            index + 1
        } else {
            timeperiod - index
        }
    }
}

#[inline]
fn trima_denominator(timeperiod: usize) -> Float {
    if timeperiod % 2 == 1 {
        let value = timeperiod / 2 + 1;
        (value * value) as Float
    } else {
        let half = timeperiod / 2;
        (half * (half + 1)) as Float
    }
}

fn trima_window(window: &[Float]) -> Float {
    let weighted_sum = window
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| trima_weight(idx, window.len()) as Float * value)
        .sum::<Float>();
    weighted_sum / trima_denominator(window.len())
}

/// TA-Lib-style Triangular Moving Average batch function.
#[allow(non_snake_case)]
pub fn TRIMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = period_lookback("timeperiod", timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("TRIMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        out_real[output_idx] = trima_window(&real[output_idx..output_idx + timeperiod]);
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes TRIMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn TRIMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = TRIMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Triangular Moving Average indicator.
#[derive(Debug, Clone)]
pub struct TRIMA {
    period: usize,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
}

impl TRIMA {
    /// Creates a new TRIMA indicator.
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
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact TRIMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        TRIMA(real, self.period, out_real)
    }

    /// Computes full-length padded TRIMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        TRIMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TRIMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period - 1
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TRIMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TRIMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for TRIMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        validate_finite_slice("input", &[input])?;

        self.buffer[self.index] = input;
        if self.count < self.period {
            self.count += 1;
        }
        self.index = (self.index + 1) % self.period;

        if self.count < self.period {
            return Ok(None);
        }

        let weighted_sum = (0..self.period)
            .map(|offset| {
                let ordered_idx = (self.index + offset) % self.period;
                trima_weight(offset, self.period) as Float * self.buffer[ordered_idx]
            })
            .sum::<Float>();
        Ok(Some(weighted_sum / trima_denominator(self.period)))
    }
}

impl Resettable for TRIMA {
    fn reset(&mut self) {
        for value in &mut self.buffer {
            *value = 0.0 as Float;
        }
        self.index = 0;
        self.count = 0;
    }
}
```

#### 3. crates/ta-core/src/overlap/mod.rs:3-5
**File**: `crates/ta-core/src/overlap/mod.rs`
**Changes**: MODIFY — declare `wma`/`trima` and re-export `WMA`, `WMA_vec`, `TRIMA`, and `TRIMA_vec`.

```rust
mod ema;
mod sma;
mod trima;
mod wma;

pub use ema::{EMA_vec, EMA};
pub use sma::{SMA_vec, SMA};
pub use trima::{TRIMA_vec, TRIMA};
pub use wma::{WMA_vec, WMA};
```

#### 4. crates/ta-core/tests/overlap_moving_averages.rs
**File**: `crates/ta-core/tests/overlap_moving_averages.rs`
**Changes**: MODIFY — append WMA/TRIMA expected-value, padded output, streaming/reset, and validation tests.

```rust
use ta_core::overlap::{EMA_vec, EMA, TRIMA_vec, TRIMA, WMA_vec, WMA};
```

```rust
#[test]
fn wma_and_trima_functions_write_compact_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let mut output = [0.0; 5];

    let range = WMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 17.0 as Float / 6.0 as Float);
    assert_close(output[1], 17.0 as Float / 3.0 as Float);
    assert_close(output[2], 34.0 as Float / 3.0 as Float);

    let range = TRIMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 9.0 as Float / 4.0 as Float);
    assert_close(output[1], 9.0 as Float / 2.0 as Float);
    assert_close(output[2], 9.0 as Float);

    let range = TRIMA(&real, 4, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 7.0 as Float / 2.0 as Float);
    assert_close(output[1], 7.0 as Float);
}

#[test]
fn wma_and_trima_vec_return_padded_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];

    let wma = WMA_vec(&real, 3).unwrap();
    assert_eq!(wma.len(), real.len());
    assert!(wma[0].is_nan());
    assert!(wma[1].is_nan());
    assert_close(wma[2], 17.0 as Float / 6.0 as Float);

    let trima = TRIMA_vec(&real, 3).unwrap();
    assert_eq!(trima.len(), real.len());
    assert!(trima[0].is_nan());
    assert!(trima[1].is_nan());
    assert_close(trima[2], 9.0 as Float / 4.0 as Float);
}

#[test]
fn wma_and_trima_structs_implement_indicator_compute() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let wma = WMA::new(3).unwrap();
    let trima = TRIMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&wma, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 17.0 as Float / 6.0 as Float);

    let range = Indicator::compute(&trima, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 9.0 as Float / 4.0 as Float);
}

#[test]
fn wma_and_trima_streaming_next_and_reset_are_safe() {
    let mut wma = WMA::new(3).unwrap();
    assert!(wma.next_checked(1.0).unwrap().is_nan());
    assert!(wma.next_checked(2.0).unwrap().is_nan());
    assert_close(wma.next_checked(4.0).unwrap(), 17.0 as Float / 6.0 as Float);
    assert_close(wma.next_checked(8.0).unwrap(), 17.0 as Float / 3.0 as Float);
    wma.reset();
    assert!(wma.next_checked(10.0).unwrap().is_nan());

    let mut trima = TRIMA::new(3).unwrap();
    assert!(trima.next_checked(1.0).unwrap().is_nan());
    assert!(trima.next_checked(2.0).unwrap().is_nan());
    assert_close(trima.next_checked(4.0).unwrap(), 9.0 as Float / 4.0 as Float);
    assert_close(trima.next_checked(8.0).unwrap(), 9.0 as Float / 2.0 as Float);
    trima.reset();
    assert!(trima.next(Float::NAN).is_err());
}

#[test]
fn wma_and_trima_reject_invalid_parameters_and_inputs() {
    assert!(WMA::new(0).is_err());
    assert!(TRIMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(WMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(TRIMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(WMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}
```

#### 5. crates/ta-core/src/inventory.rs:12,153-168
**File**: `crates/ta-core/src/inventory.rs`
**Changes**: MODIFY — increment implemented count to 35 and mark `WMA`/`TRIMA` implemented.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 35;
```

```rust
function!("TEMA", OverlapStudies, Planned),
function!("TRIMA", OverlapStudies, Implemented),
function!("WMA", OverlapStudies, Implemented),
```

#### 6. crates/ta-core/tests/inventory.rs:9-13,69-127,196-201
**File**: `crates/ta-core/tests/inventory.rs`
**Changes**: MODIFY — import/assert `WMA`/`TRIMA`, include them in implemented status and trait checks.

```rust
use ta_core::overlap::{EMA, SMA, TRIMA, WMA};
```

```rust
let implemented = [
    "SMA",
    "EMA",
    "TRIMA",
    "WMA",
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
```

```rust
assert_indicator::<EMA>();
assert_streaming::<EMA>();
assert_indicator::<TRIMA>();
assert_streaming::<TRIMA>();
assert_indicator::<WMA>();
assert_streaming::<WMA>();
```

### Success Criteria:

#### Automated Verification:
- [x] Moving-average tests pass with WMA/TRIMA coverage: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests pass after marking WMA/TRIMA implemented: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after WMA/TRIMA exports: `cargo test -p ta-core`
- [x] f32 precision compiles for weighted/triangular arithmetic: `cargo test -p ta-core --features f32`
- [x] Workspace checks after WMA/TRIMA public exports: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] WMA/TRIMA are wired in overlap module: `grep -n "mod wma\|mod trima\|pub use wma\|pub use trima" crates/ta-core/src/overlap/mod.rs`
- [x] WMA/TRIMA inventory statuses are implemented: `grep -n 'function!("TRIMA", OverlapStudies, Implemented)\|function!("WMA", OverlapStudies, Implemented)' crates/ta-core/src/inventory.rs`
- [x] No future overlap modules are declared early in Phase 2: `grep -n "mod dema\|mod tema\|mod t3\|mod ma" crates/ta-core/src/overlap/mod.rs` returns no matches

#### Manual Verification:
- [x] WMA weights oldest values as 1 and newest values as `timeperiod` with denominator `timeperiod * (timeperiod + 1) / 2`.
- [x] TRIMA odd/even triangular weights match the documented denominator formulas.
- [x] `WMA_vec` and `TRIMA_vec` preserve full input length and pad warm-up values with `Float::NAN`.
- [x] WMA/TRIMA streaming order uses oldest-to-newest windows after the ring buffer wraps without heap allocation.
- [x] WMA/TRIMA constructors and non-finite inputs return `TalibError` instead of panicking.

## Phase 3: EMA-derived composites

### Overview

Adds DEMA and TEMA as composite EMA indicators backed by the Phase 1 EMA helper. Depends on Phases 1 and 2 because this phase extends the overlap module, moving-average tests, and inventory state produced by both prior phases.

### Changes Required:

#### 1. crates/ta-core/src/overlap/dema.rs
**File**: `crates/ta-core/src/overlap/dema.rs`
**Changes**: NEW — DEMA compact API, padded wrapper, struct surface, streaming/reset support.

```rust
//! Double Exponential Moving Average (DEMA).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn dema_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(2)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "DEMA lookback would overflow"))
}

/// TA-Lib-style Double Exponential Moving Average batch function.
#[allow(non_snake_case)]
pub fn DEMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = dema_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("DEMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut ema1 = super::ema::EMA::new(timeperiod)?;
    let mut ema2 = super::ema::EMA::new(timeperiod)?;
    let mut output_idx = 0usize;

    for &value in real {
        let Some(ema1_value) = ema1.next(value)? else {
            continue;
        };
        let Some(ema2_value) = ema2.next(ema1_value)? else {
            continue;
        };
        out_real[output_idx] = 2.0 as Float * ema1_value - ema2_value;
        output_idx += 1;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes DEMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn DEMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = DEMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Double Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct DEMA {
    period: usize,
    lookback: usize,
    ema1: super::ema::EMA,
    ema2: super::ema::EMA,
}

impl DEMA {
    /// Creates a new DEMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let lookback = dema_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            lookback,
            ema1: super::ema::EMA::new(timeperiod)?,
            ema2: super::ema::EMA::new(timeperiod)?,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact DEMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        DEMA(real, self.period, out_real)
    }

    /// Computes full-length padded DEMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        DEMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for DEMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        DEMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        DEMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for DEMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next(input)? else {
            return Ok(None);
        };
        let Some(ema2) = self.ema2.next(ema1)? else {
            return Ok(None);
        };
        Ok(Some(2.0 as Float * ema1 - ema2))
    }
}

impl Resettable for DEMA {
    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
    }
}
```

#### 2. crates/ta-core/src/overlap/tema.rs
**File**: `crates/ta-core/src/overlap/tema.rs`
**Changes**: NEW — TEMA compact API, padded wrapper, struct surface, streaming/reset support.

```rust
//! Triple Exponential Moving Average (TEMA).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn tema_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(3)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "TEMA lookback would overflow"))
}

/// TA-Lib-style Triple Exponential Moving Average batch function.
#[allow(non_snake_case)]
pub fn TEMA(real: &[Float], timeperiod: usize, out_real: &mut [Float]) -> Result<OutputRange> {
    let lookback = tema_lookback(timeperiod)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("TEMA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut ema1 = super::ema::EMA::new(timeperiod)?;
    let mut ema2 = super::ema::EMA::new(timeperiod)?;
    let mut ema3 = super::ema::EMA::new(timeperiod)?;
    let mut output_idx = 0usize;

    for &value in real {
        let Some(ema1_value) = ema1.next(value)? else {
            continue;
        };
        let Some(ema2_value) = ema2.next(ema1_value)? else {
            continue;
        };
        let Some(ema3_value) = ema3.next(ema2_value)? else {
            continue;
        };
        out_real[output_idx] = 3.0 as Float * ema1_value
            - 3.0 as Float * ema2_value
            + ema3_value;
        output_idx += 1;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes TEMA into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn TEMA_vec(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = TEMA(real, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Triple Exponential Moving Average indicator.
#[derive(Debug, Clone)]
pub struct TEMA {
    period: usize,
    lookback: usize,
    ema1: super::ema::EMA,
    ema2: super::ema::EMA,
    ema3: super::ema::EMA,
}

impl TEMA {
    /// Creates a new TEMA indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        let lookback = tema_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            lookback,
            ema1: super::ema::EMA::new(timeperiod)?,
            ema2: super::ema::EMA::new(timeperiod)?,
            ema3: super::ema::EMA::new(timeperiod)?,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact TEMA outputs using this indicator's period.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        TEMA(real, self.period, out_real)
    }

    /// Computes full-length padded TEMA outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        TEMA_vec(real, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TEMA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        TEMA(inputs, self.period, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TEMA_vec(inputs, self.period)
    }
}

impl StreamingIndicator for TEMA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next(input)? else {
            return Ok(None);
        };
        let Some(ema2) = self.ema2.next(ema1)? else {
            return Ok(None);
        };
        let Some(ema3) = self.ema3.next(ema2)? else {
            return Ok(None);
        };
        Ok(Some(
            3.0 as Float * ema1 - 3.0 as Float * ema2 + ema3,
        ))
    }
}

impl Resettable for TEMA {
    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
    }
}
```

#### 3. crates/ta-core/src/overlap/mod.rs:3-5
**File**: `crates/ta-core/src/overlap/mod.rs`
**Changes**: MODIFY — declare `dema`/`tema` and re-export `DEMA`, `DEMA_vec`, `TEMA`, and `TEMA_vec`.

```rust
mod dema;
mod ema;
mod sma;
mod tema;
mod trima;
mod wma;

pub use dema::{DEMA_vec, DEMA};
pub use ema::{EMA_vec, EMA};
pub use sma::{SMA_vec, SMA};
pub use tema::{TEMA_vec, TEMA};
pub use trima::{TRIMA_vec, TRIMA};
pub use wma::{WMA_vec, WMA};
```

#### 4. crates/ta-core/tests/overlap_moving_averages.rs
**File**: `crates/ta-core/tests/overlap_moving_averages.rs`
**Changes**: MODIFY — append DEMA/TEMA lookback, padded output, struct/streaming, and validation tests.

```rust
use ta_core::overlap::{
    DEMA_vec, DEMA, EMA_vec, EMA, TEMA_vec, TEMA, TRIMA_vec, TRIMA, WMA_vec, WMA,
};
```

```rust
#[test]
fn dema_and_tema_functions_write_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut output = [0.0; 7];

    let range = DEMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(4, 3));
    assert_close(output[0], 5.0);
    assert_close(output[1], 6.0);
    assert_close(output[2], 7.0);

    let range = TEMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(6, 1));
    assert_close(output[0], 7.0);
}

#[test]
fn dema_and_tema_vec_return_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let dema = DEMA_vec(&real, 3).unwrap();
    assert_eq!(dema.len(), real.len());
    assert!(dema[..4].iter().all(|value| value.is_nan()));
    assert_close(dema[4], 5.0);
    assert_close(dema[6], 7.0);

    let tema = TEMA_vec(&real, 3).unwrap();
    assert_eq!(tema.len(), real.len());
    assert!(tema[..6].iter().all(|value| value.is_nan()));
    assert_close(tema[6], 7.0);
}

#[test]
fn dema_and_tema_structs_implement_indicator_compute() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let dema = DEMA::new(3).unwrap();
    let tema = TEMA::new(3).unwrap();
    let mut compact = [0.0; 7];

    let range = Indicator::compute(&dema, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(4, 3));
    assert_close(compact[0], 5.0);

    let range = Indicator::compute(&tema, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(6, 1));
    assert_close(compact[0], 7.0);
}

#[test]
fn dema_and_tema_streaming_next_and_reset_are_safe() {
    let mut dema = DEMA::new(3).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0] {
        assert!(dema.next_checked(value).unwrap().is_nan());
    }
    assert_close(dema.next_checked(5.0).unwrap(), 5.0);
    dema.reset();
    assert!(dema.next_checked(10.0).unwrap().is_nan());

    let mut tema = TEMA::new(3).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert!(tema.next_checked(value).unwrap().is_nan());
    }
    assert_close(tema.next_checked(7.0).unwrap(), 7.0);
    tema.reset();
    assert!(tema.next(Float::NAN).is_err());
}

#[test]
fn dema_and_tema_reject_invalid_parameters_and_inputs() {
    assert!(DEMA::new(0).is_err());
    assert!(TEMA::new(0).is_err());
    assert!(DEMA::new(usize::MAX).is_err());
    assert!(TEMA::new(usize::MAX).is_err());

    let mut output = [0.0; 7];
    assert!(DEMA(&[1.0, 2.0, 3.0], 3, &mut output).is_err());
    assert!(TEMA(&[1.0, Float::NAN, 3.0, 4.0, 5.0, 6.0, 7.0], 3, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(DEMA(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, &mut too_small).is_err());
}
```

#### 5. crates/ta-core/src/inventory.rs:12,153-168
**File**: `crates/ta-core/src/inventory.rs`
**Changes**: MODIFY — increment implemented count to 37 and mark `DEMA`/`TEMA` implemented.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 37;
```

```rust
function!("DEMA", OverlapStudies, Implemented),
function!("EMA", OverlapStudies, Implemented),
```

```rust
function!("T3", OverlapStudies, Planned),
function!("TEMA", OverlapStudies, Implemented),
function!("TRIMA", OverlapStudies, Implemented),
```

#### 6. crates/ta-core/tests/inventory.rs:9-13,69-127,196-201
**File**: `crates/ta-core/tests/inventory.rs`
**Changes**: MODIFY — import/assert `DEMA`/`TEMA`, include them in implemented status and trait checks.

```rust
use ta_core::overlap::{DEMA, EMA, SMA, TEMA, TRIMA, WMA};
```

```rust
let implemented = [
    "SMA",
    "DEMA",
    "EMA",
    "TEMA",
    "TRIMA",
    "WMA",
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
```

```rust
assert_indicator::<SMA>();
assert_streaming::<SMA>();
assert_indicator::<DEMA>();
assert_streaming::<DEMA>();
assert_indicator::<EMA>();
assert_streaming::<EMA>();
assert_indicator::<TEMA>();
assert_streaming::<TEMA>();
assert_indicator::<TRIMA>();
assert_streaming::<TRIMA>();
assert_indicator::<WMA>();
assert_streaming::<WMA>();
```

### Success Criteria:

#### Automated Verification:
- [x] Moving-average tests pass with DEMA/TEMA coverage: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests pass after marking DEMA/TEMA implemented: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after DEMA/TEMA exports: `cargo test -p ta-core`
- [x] f32 precision compiles for recursive DEMA/TEMA chains: `cargo test -p ta-core --features f32`
- [x] Workspace checks after DEMA/TEMA public exports: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] DEMA/TEMA are wired in overlap module: `grep -n "mod dema\|mod tema\|pub use dema\|pub use tema" crates/ta-core/src/overlap/mod.rs`
- [x] DEMA/TEMA inventory statuses are implemented: `grep -n 'function!("DEMA", OverlapStudies, Implemented)\|function!("TEMA", OverlapStudies, Implemented)' crates/ta-core/src/inventory.rs`
- [x] DEMA/TEMA lookback multiplication is checked: `grep -n "checked_mul(2)\|checked_mul(3)" crates/ta-core/src/overlap/dema.rs crates/ta-core/src/overlap/tema.rs`
- [x] No future overlap modules are declared early in Phase 3: `grep -n "mod t3\|mod ma" crates/ta-core/src/overlap/mod.rs` returns no matches

#### Manual Verification:
- [x] DEMA lookback is `2 * (timeperiod - 1)` and output formula is `2 * EMA1 - EMA2`.
- [x] TEMA lookback is `3 * (timeperiod - 1)` and output formula is `3 * EMA1 - 3 * EMA2 + EMA3`.
- [x] DEMA/TEMA compact batch paths stream through EMA state without intermediate compact-buffer allocations.
- [x] DEMA/TEMA streaming chains reuse EMA state and reset all nested EMA state.
- [x] DEMA/TEMA constructors and non-finite inputs return `TalibError` instead of panicking.
- [x] Extremely large DEMA/TEMA periods return `TalibError::InvalidPeriod` instead of panicking or wrapping lookback math.

## Phase 4: T3 with vfactor

### Overview

Adds T3 with explicit and default-vfactor APIs, including validation for `vfactor` in `[0, 1]`. Depends on Phases 1-3 because this phase extends the overlap module, moving-average tests, and inventory state produced by prior moving-average phases.

### Changes Required:

#### 1. crates/ta-core/src/overlap/t3.rs
**File**: `crates/ta-core/src/overlap/t3.rs`
**Changes**: NEW — T3 compact API, default-vfactor convenience APIs, padded wrappers, struct surface, streaming/reset support.

```rust
//! T3 Moving Average (T3).

use crate::{
    compact_buffer, padded_from_compact, period_lookback, validate_finite_slice,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

/// TA-Lib default T3 volume factor.
pub const T3_DEFAULT_VFACTOR: Float = 0.7 as Float;

#[inline]
fn t3_lookback(timeperiod: usize) -> Result<usize> {
    period_lookback("timeperiod", timeperiod)?
        .checked_mul(6)
        .ok_or_else(|| TalibError::invalid_period(timeperiod, "T3 lookback would overflow"))
}

fn validate_vfactor(vfactor: Float) -> Result<()> {
    if !vfactor.is_finite() || !(0.0 as Float..=1.0 as Float).contains(&vfactor) {
        return Err(TalibError::invalid_parameter(
            "vfactor".to_string(),
            format!("{}", vfactor),
            "value in [0.0, 1.0]".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn t3_coefficients(vfactor: Float) -> (Float, Float, Float, Float) {
    let v2 = vfactor * vfactor;
    let v3 = v2 * vfactor;
    let c1 = -v3;
    let c2 = 3.0 as Float * (v2 - c1);
    let c3 = -6.0 as Float * v2 - 3.0 as Float * (vfactor - c1);
    let c4 = 1.0 as Float + 3.0 as Float * vfactor - c1 + 3.0 as Float * v2;
    (c1, c2, c3, c4)
}

#[inline]
fn t3_value(
    ema3: Float,
    ema4: Float,
    ema5: Float,
    ema6: Float,
    coefficients: (Float, Float, Float, Float),
) -> Float {
    let (c1, c2, c3, c4) = coefficients;
    c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
}

/// TA-Lib-style T3 Moving Average batch function.
#[allow(non_snake_case)]
pub fn T3(
    real: &[Float],
    timeperiod: usize,
    vfactor: Float,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = t3_lookback(timeperiod)?;
    validate_vfactor(vfactor)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len("T3", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut t3 = T3::new(timeperiod, vfactor)?;
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(output) = t3.next(value)? {
            out_real[output_idx] = output;
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// TA-Lib-style T3 batch function using `T3_DEFAULT_VFACTOR`.
#[allow(non_snake_case)]
pub fn T3_with_default_vfactor(
    real: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    T3(real, timeperiod, T3_DEFAULT_VFACTOR, out_real)
}

/// Computes T3 into a full-length vector padded with `Float::NAN` before the lookback.
#[allow(non_snake_case)]
pub fn T3_vec(real: &[Float], timeperiod: usize, vfactor: Float) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = T3(real, timeperiod, vfactor, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Computes T3 with the default vfactor into a full-length vector.
#[allow(non_snake_case)]
pub fn T3_vec_with_default_vfactor(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    T3_vec(real, timeperiod, T3_DEFAULT_VFACTOR)
}

/// T3 Moving Average indicator.
#[derive(Debug, Clone)]
pub struct T3 {
    period: usize,
    lookback: usize,
    vfactor: Float,
    coefficients: (Float, Float, Float, Float),
    ema1: super::ema::EMA,
    ema2: super::ema::EMA,
    ema3: super::ema::EMA,
    ema4: super::ema::EMA,
    ema5: super::ema::EMA,
    ema6: super::ema::EMA,
}

impl T3 {
    /// Creates a new T3 indicator with an explicit vfactor.
    pub fn new(timeperiod: usize, vfactor: Float) -> Result<Self> {
        let lookback = t3_lookback(timeperiod)?;
        validate_vfactor(vfactor)?;
        Ok(Self {
            period: timeperiod,
            lookback,
            vfactor,
            coefficients: t3_coefficients(vfactor),
            ema1: super::ema::EMA::new(timeperiod)?,
            ema2: super::ema::EMA::new(timeperiod)?,
            ema3: super::ema::EMA::new(timeperiod)?,
            ema4: super::ema::EMA::new(timeperiod)?,
            ema5: super::ema::EMA::new(timeperiod)?,
            ema6: super::ema::EMA::new(timeperiod)?,
        })
    }

    /// Creates a new T3 indicator with TA-Lib's default vfactor.
    pub fn with_default_vfactor(timeperiod: usize) -> Result<Self> {
        Self::new(timeperiod, T3_DEFAULT_VFACTOR)
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured vfactor.
    #[inline]
    pub const fn vfactor(&self) -> Float {
        self.vfactor
    }

    /// Computes compact T3 outputs using this indicator's period and vfactor.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        T3(real, self.period, self.vfactor, out_real)
    }

    /// Computes full-length padded T3 outputs using this indicator's period and vfactor.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        T3_vec(real, self.period, self.vfactor)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for T3 {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.lookback
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        T3(inputs, self.period, self.vfactor, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        T3_vec(inputs, self.period, self.vfactor)
    }
}

impl StreamingIndicator for T3 {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        let Some(ema1) = self.ema1.next(input)? else {
            return Ok(None);
        };
        let Some(ema2) = self.ema2.next(ema1)? else {
            return Ok(None);
        };
        let Some(ema3) = self.ema3.next(ema2)? else {
            return Ok(None);
        };
        let Some(ema4) = self.ema4.next(ema3)? else {
            return Ok(None);
        };
        let Some(ema5) = self.ema5.next(ema4)? else {
            return Ok(None);
        };
        let Some(ema6) = self.ema6.next(ema5)? else {
            return Ok(None);
        };
        Ok(Some(t3_value(
            ema3,
            ema4,
            ema5,
            ema6,
            self.coefficients,
        )))
    }
}

impl Resettable for T3 {
    fn reset(&mut self) {
        self.ema1.reset();
        self.ema2.reset();
        self.ema3.reset();
        self.ema4.reset();
        self.ema5.reset();
        self.ema6.reset();
    }
}
```

#### 2. crates/ta-core/src/overlap/mod.rs:3-5
**File**: `crates/ta-core/src/overlap/mod.rs`
**Changes**: MODIFY — declare `t3` and re-export `T3`, `T3_vec`, default-vfactor helpers, and `T3_DEFAULT_VFACTOR`.

```rust
mod dema;
mod ema;
mod sma;
mod t3;
mod tema;
mod trima;
mod wma;

pub use dema::{DEMA_vec, DEMA};
pub use ema::{EMA_vec, EMA};
pub use sma::{SMA_vec, SMA};
pub use t3::{T3_vec, T3_vec_with_default_vfactor, T3_with_default_vfactor, T3_DEFAULT_VFACTOR, T3};
pub use tema::{TEMA_vec, TEMA};
pub use trima::{TRIMA_vec, TRIMA};
pub use wma::{WMA_vec, WMA};
```

#### 3. crates/ta-core/tests/overlap_moving_averages.rs
**File**: `crates/ta-core/tests/overlap_moving_averages.rs`
**Changes**: MODIFY — append T3 vfactor, default-vfactor, padded output, struct/streaming, and validation tests.

```rust
use ta_core::overlap::{
    DEMA_vec, DEMA, EMA_vec, EMA, T3_vec, T3_vec_with_default_vfactor,
    T3_with_default_vfactor, T3_DEFAULT_VFACTOR, T3, TEMA_vec, TEMA, TRIMA_vec, TRIMA, WMA_vec,
    WMA,
};
```

```rust
#[test]
fn t3_function_writes_compact_outputs_and_default_matches_explicit() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let mut explicit = [0.0; 4];
    let mut defaulted = [0.0; 4];

    let explicit_range = T3(&real, 1, T3_DEFAULT_VFACTOR, &mut explicit).unwrap();
    let default_range = T3_with_default_vfactor(&real, 1, &mut defaulted).unwrap();

    assert_eq!(explicit_range, OutputRange::new(0, 4));
    assert_eq!(default_range, explicit_range);
    for idx in 0..real.len() {
        assert_close(explicit[idx], real[idx]);
        assert_close(defaulted[idx], explicit[idx]);
    }
}

#[test]
fn t3_vec_returns_padded_outputs_for_recursive_lookback() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let explicit = T3_vec(&real, 2, T3_DEFAULT_VFACTOR).unwrap();
    let defaulted = T3_vec_with_default_vfactor(&real, 2).unwrap();

    assert_eq!(explicit.len(), real.len());
    assert!(explicit[..6].iter().all(|value| value.is_nan()));
    assert!(explicit[6].is_finite());
    assert!(explicit[7].is_finite());
    for idx in 0..real.len() {
        if explicit[idx].is_nan() {
            assert!(defaulted[idx].is_nan());
        } else {
            assert_close(defaulted[idx], explicit[idx]);
        }
    }
}

#[test]
fn t3_struct_implements_indicator_compute_and_streaming() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let t3 = T3::with_default_vfactor(1).unwrap();
    assert_close(t3.vfactor(), T3_DEFAULT_VFACTOR);

    let mut compact = [0.0; 4];
    let range = Indicator::compute(&t3, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(0, 4));
    assert_eq!(compact, real);

    let mut streaming = T3::new(2, T3_DEFAULT_VFACTOR).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert!(streaming.next_checked(value).unwrap().is_nan());
    }
    assert!(streaming.next_checked(7.0).unwrap().is_finite());
    streaming.reset();
    assert!(streaming.next(Float::NAN).is_err());
}

#[test]
fn t3_rejects_invalid_parameters_and_inputs() {
    assert!(T3::new(0, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3::new(usize::MAX, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3::new(3, -0.1 as Float).is_err());
    assert!(T3::new(3, 1.1 as Float).is_err());
    assert!(T3::new(3, Float::NAN).is_err());

    let mut output = [0.0; 8];
    assert!(T3(&[1.0, Float::NAN, 3.0], 1, T3_DEFAULT_VFACTOR, &mut output).is_err());
    let mut too_small = [0.0; 1];
    assert!(T3(&[1.0, 2.0, 3.0, 4.0], 1, T3_DEFAULT_VFACTOR, &mut too_small).is_err());
}
```

#### 4. crates/ta-core/src/inventory.rs:12,153-168
**File**: `crates/ta-core/src/inventory.rs`
**Changes**: MODIFY — increment implemented count to 38 and mark `T3` implemented.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 38;
```

```rust
function!("SMA", OverlapStudies, Implemented),
function!("T3", OverlapStudies, Implemented),
function!("TEMA", OverlapStudies, Implemented),
```

#### 5. crates/ta-core/tests/inventory.rs:9-13,69-127,196-201
**File**: `crates/ta-core/tests/inventory.rs`
**Changes**: MODIFY — import/assert `T3`, include it in implemented status and trait checks.

```rust
use ta_core::overlap::{DEMA, EMA, SMA, T3, TEMA, TRIMA, WMA};
```

```rust
let implemented = [
    "SMA",
    "DEMA",
    "EMA",
    "T3",
    "TEMA",
    "TRIMA",
    "WMA",
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
```

```rust
assert_indicator::<SMA>();
assert_streaming::<SMA>();
assert_indicator::<DEMA>();
assert_streaming::<DEMA>();
assert_indicator::<EMA>();
assert_streaming::<EMA>();
assert_indicator::<T3>();
assert_streaming::<T3>();
assert_indicator::<TEMA>();
assert_streaming::<TEMA>();
assert_indicator::<TRIMA>();
assert_streaming::<TRIMA>();
assert_indicator::<WMA>();
assert_streaming::<WMA>();
```

### Success Criteria:

#### Automated Verification:
- [x] Moving-average tests pass with T3 coverage: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests pass after marking T3 implemented: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after T3 exports: `cargo test -p ta-core`
- [x] f32 precision compiles for T3 coefficient arithmetic: `cargo test -p ta-core --features f32`
- [x] Workspace checks after T3 public exports: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] T3 is wired in overlap module: `grep -n "mod t3\|pub use t3" crates/ta-core/src/overlap/mod.rs`
- [x] T3 inventory status is implemented: `grep -n 'function!("T3", OverlapStudies, Implemented)' crates/ta-core/src/inventory.rs`
- [x] T3 lookback multiplication is checked: `grep -n "checked_mul(6)" crates/ta-core/src/overlap/t3.rs`
- [x] No future MA dispatcher module is declared early in Phase 4: `grep -n "mod ma" crates/ta-core/src/overlap/mod.rs` returns no matches

#### Manual Verification:
- [x] T3 lookback is `6 * (timeperiod - 1)`.
- [x] T3 validates `vfactor` as finite and within `[0, 1]` using `TalibError::InvalidParameter`.
- [x] Explicit and default-vfactor APIs produce identical output when `vfactor == T3_DEFAULT_VFACTOR`.
- [x] T3 output uses TA-Lib coefficients `c1*e6 + c2*e5 + c3*e4 + c4*e3`.
- [x] T3 streaming chains reset all six nested EMA states.
- [x] Extremely large T3 periods return `TalibError::InvalidPeriod` instead of panicking or wrapping lookback math.

## Phase 5: MA dispatcher

### Overview

Adds official `overlap::MA` as a dispatcher over implemented moving averages and full official `MAType` variants, with explicit NotImplemented errors for KAMA/MAMA. Depends on Phases 1-4 plus existing SMA.

### Changes Required:

#### 1. crates/ta-core/src/overlap/ma.rs
**File**: `crates/ta-core/src/overlap/ma.rs`
**Changes**: NEW — `MAType`, `MA`, `MA_vec`, dispatcher struct, trait implementation, and unsupported KAMA/MAMA error handling.

```rust
//! Generic Moving Average (MA) dispatcher.

use crate::{
    compact_buffer, padded_from_compact, period_lookback, Float, Indicator, OutputRange,
    Resettable, Result, StreamingIndicator, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec::Vec};

/// Official TA-Lib moving-average type selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MAType {
    /// Simple Moving Average.
    SMA,
    /// Exponential Moving Average.
    EMA,
    /// Weighted Moving Average.
    WMA,
    /// Double Exponential Moving Average.
    DEMA,
    /// Triple Exponential Moving Average.
    TEMA,
    /// Triangular Moving Average.
    TRIMA,
    /// Kaufman Adaptive Moving Average (not implemented in this tranche).
    KAMA,
    /// MESA Adaptive Moving Average (not implemented in this tranche).
    MAMA,
    /// T3 Moving Average.
    T3,
}

impl MAType {
    /// Official TA-Lib integer id for this moving-average type.
    pub const fn talib_id(self) -> usize {
        match self {
            Self::SMA => 0,
            Self::EMA => 1,
            Self::WMA => 2,
            Self::DEMA => 3,
            Self::TEMA => 4,
            Self::TRIMA => 5,
            Self::KAMA => 6,
            Self::MAMA => 7,
            Self::T3 => 8,
        }
    }

    /// Stable display label used in error messages.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SMA => "SMA",
            Self::EMA => "EMA",
            Self::WMA => "WMA",
            Self::DEMA => "DEMA",
            Self::TEMA => "TEMA",
            Self::TRIMA => "TRIMA",
            Self::KAMA => "KAMA",
            Self::MAMA => "MAMA",
            Self::T3 => "T3",
        }
    }
}

fn unsupported_ma_type(matype: MAType) -> TalibError {
    TalibError::not_implemented(format!("MAType::{}", matype.as_str()))
}

/// TA-Lib-style generic Moving Average dispatcher.
#[allow(non_snake_case)]
pub fn MA(
    real: &[Float],
    timeperiod: usize,
    matype: MAType,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    match matype {
        MAType::SMA => super::sma::SMA(real, timeperiod, out_real),
        MAType::EMA => super::ema::EMA(real, timeperiod, out_real),
        MAType::WMA => super::wma::WMA(real, timeperiod, out_real),
        MAType::DEMA => super::dema::DEMA(real, timeperiod, out_real),
        MAType::TEMA => super::tema::TEMA(real, timeperiod, out_real),
        MAType::TRIMA => super::trima::TRIMA(real, timeperiod, out_real),
        MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
        MAType::T3 => super::t3::T3_with_default_vfactor(real, timeperiod, out_real),
    }
}

/// Computes the selected moving average into a full-length padded vector.
#[allow(non_snake_case)]
pub fn MA_vec(real: &[Float], timeperiod: usize, matype: MAType) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real.len());
    let range = MA(real, timeperiod, matype, &mut compact)?;
    Ok(padded_from_compact(
        real.len(),
        range,
        &compact[..range.nb_element],
    ))
}

#[derive(Debug, Clone)]
enum MAInner {
    SMA(super::sma::SMA),
    EMA(super::ema::EMA),
    WMA(super::wma::WMA),
    DEMA(super::dema::DEMA),
    TEMA(super::tema::TEMA),
    TRIMA(super::trima::TRIMA),
    T3(super::t3::T3),
}

impl MAInner {
    fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        match matype {
            MAType::SMA => Ok(Self::SMA(super::sma::SMA::new(timeperiod)?)),
            MAType::EMA => Ok(Self::EMA(super::ema::EMA::new(timeperiod)?)),
            MAType::WMA => Ok(Self::WMA(super::wma::WMA::new(timeperiod)?)),
            MAType::DEMA => Ok(Self::DEMA(super::dema::DEMA::new(timeperiod)?)),
            MAType::TEMA => Ok(Self::TEMA(super::tema::TEMA::new(timeperiod)?)),
            MAType::TRIMA => Ok(Self::TRIMA(super::trima::TRIMA::new(timeperiod)?)),
            MAType::KAMA | MAType::MAMA => Err(unsupported_ma_type(matype)),
            MAType::T3 => Ok(Self::T3(super::t3::T3::with_default_vfactor(timeperiod)?)),
        }
    }

    fn lookback(&self) -> usize {
        match self {
            Self::SMA(inner) => inner.lookback(),
            Self::EMA(inner) => inner.lookback(),
            Self::WMA(inner) => inner.lookback(),
            Self::DEMA(inner) => inner.lookback(),
            Self::TEMA(inner) => inner.lookback(),
            Self::TRIMA(inner) => inner.lookback(),
            Self::T3(inner) => inner.lookback(),
        }
    }

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        match self {
            Self::SMA(inner) => inner.next(input),
            Self::EMA(inner) => inner.next(input),
            Self::WMA(inner) => inner.next(input),
            Self::DEMA(inner) => inner.next(input),
            Self::TEMA(inner) => inner.next(input),
            Self::TRIMA(inner) => inner.next(input),
            Self::T3(inner) => inner.next(input),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::SMA(inner) => inner.reset(),
            Self::EMA(inner) => inner.reset(),
            Self::WMA(inner) => inner.reset(),
            Self::DEMA(inner) => inner.reset(),
            Self::TEMA(inner) => inner.reset(),
            Self::TRIMA(inner) => inner.reset(),
            Self::T3(inner) => inner.reset(),
        }
    }
}

/// Generic Moving Average indicator dispatcher.
#[derive(Debug, Clone)]
pub struct MA {
    period: usize,
    matype: MAType,
    inner: MAInner,
}

impl MA {
    /// Creates a new MA dispatcher for the selected moving-average type.
    pub fn new(timeperiod: usize, matype: MAType) -> Result<Self> {
        period_lookback("timeperiod", timeperiod)?;
        Ok(Self {
            period: timeperiod,
            matype,
            inner: MAInner::new(timeperiod, matype)?,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Returns the configured moving-average type.
    #[inline]
    pub const fn ma_type(&self) -> MAType {
        self.matype
    }

    /// Computes compact MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
        MA(real, self.period, self.matype, out_real)
    }

    /// Computes full-length padded MA outputs using this dispatcher's period and type.
    #[inline]
    pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
        MA_vec(real, self.period, self.matype)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: Float) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for MA {
    type Input<'a> = &'a [Float];
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.inner.lookback()
    }

    #[inline]
    fn compute<'a>(
        &self,
        inputs: Self::Input<'a>,
        outputs: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        MA(inputs, self.period, self.matype, outputs)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, inputs: Self::Input<'a>) -> Result<Self::OutputOwned> {
        MA_vec(inputs, self.period, self.matype)
    }
}

impl StreamingIndicator for MA {
    type Tick = Float;
    type TickOutput = Float;

    fn next(&mut self, input: Float) -> Result<Option<Float>> {
        self.inner.next(input)
    }
}

impl Resettable for MA {
    fn reset(&mut self) {
        self.inner.reset();
    }
}
```

#### 2. crates/ta-core/src/overlap/mod.rs:3-5
**File**: `crates/ta-core/src/overlap/mod.rs`
**Changes**: MODIFY — declare `ma` and re-export `MA`, `MA_vec`, and `MAType`.

```rust
mod dema;
mod ema;
mod ma;
mod sma;
mod t3;
mod tema;
mod trima;
mod wma;

pub use dema::{DEMA_vec, DEMA};
pub use ema::{EMA_vec, EMA};
pub use ma::{MA_vec, MAType, MA};
pub use sma::{SMA_vec, SMA};
pub use t3::{T3_vec, T3_vec_with_default_vfactor, T3_with_default_vfactor, T3_DEFAULT_VFACTOR, T3};
pub use tema::{TEMA_vec, TEMA};
pub use trima::{TRIMA_vec, TRIMA};
pub use wma::{WMA_vec, WMA};
```

#### 3. crates/ta-core/tests/overlap_moving_averages.rs
**File**: `crates/ta-core/tests/overlap_moving_averages.rs`
**Changes**: MODIFY — append MA dispatch parity tests for implemented variants and NotImplemented tests for KAMA/MAMA.

```rust
use ta_core::overlap::{
    DEMA_vec, DEMA, EMA_vec, EMA, MA_vec, MAType, MA, T3_vec, T3_vec_with_default_vfactor,
    T3_with_default_vfactor, T3_DEFAULT_VFACTOR, T3, TEMA_vec, TEMA, TRIMA_vec, TRIMA, WMA_vec,
    WMA,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator, TalibError};
```

```rust
fn assert_vec_close_with_nans(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN at {idx}, got {actual}");
        } else {
            assert_close(*actual, *expected);
        }
    }
}
```

```rust
#[test]
fn ma_dispatches_to_implemented_moving_averages() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::SMA).unwrap(), &ta_core::overlap::SMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::EMA).unwrap(), &EMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::WMA).unwrap(), &WMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::DEMA).unwrap(), &DEMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::TEMA).unwrap(), &TEMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 3, MAType::TRIMA).unwrap(), &TRIMA_vec(&real, 3).unwrap());
    assert_vec_close_with_nans(&MA_vec(&real, 2, MAType::T3).unwrap(), &T3_vec_with_default_vfactor(&real, 2).unwrap());
}

#[test]
fn ma_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut ma_output = [0.0; 5];
    let mut ema_output = [0.0; 5];

    let ma_range = MA(&real, 3, MAType::EMA, &mut ma_output).unwrap();
    let ema_range = EMA(&real, 3, &mut ema_output).unwrap();

    assert_eq!(ma_range, ema_range);
    assert_close(ma_output[0], ema_output[0]);
    assert_close(ma_output[2], ema_output[2]);
}

#[test]
fn ma_struct_streams_selected_average() {
    let mut ma = MA::new(3, MAType::EMA).unwrap();
    assert_eq!(ma.ma_type(), MAType::EMA);
    assert_eq!(ma.period(), 3);

    assert!(ma.next_checked(1.0).unwrap().is_nan());
    assert!(ma.next_checked(2.0).unwrap().is_nan());
    assert_close(ma.next_checked(3.0).unwrap(), 2.0);
    assert_close(ma.next_checked(4.0).unwrap(), 3.0);

    ma.reset();
    assert!(ma.next(Float::NAN).is_err());
}

#[test]
fn ma_rejects_unsupported_kama_and_mama_until_implemented() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];

    let kama_err = MA(&real, 3, MAType::KAMA, &mut output).unwrap_err();
    assert!(matches!(kama_err, TalibError::NotImplemented { .. }));

    let mama_err = MA::new(3, MAType::MAMA).unwrap_err();
    assert!(matches!(mama_err, TalibError::NotImplemented { .. }));
}
```

#### 4. crates/ta-core/src/inventory.rs:12,153-168
**File**: `crates/ta-core/src/inventory.rs`
**Changes**: MODIFY — increment implemented count to 39 and mark `MA` implemented while KAMA/MAMA remain planned.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 39;
```

```rust
function!("KAMA", OverlapStudies, Planned),
function!("MA", OverlapStudies, Implemented),
function!("MAMA", OverlapStudies, Planned),
```

#### 5. crates/ta-core/tests/inventory.rs:9-13,69-127,196-201
**File**: `crates/ta-core/tests/inventory.rs`
**Changes**: MODIFY — import/assert dispatcher `MA`, include `MA` in implemented status and trait checks while keeping KAMA/MAMA planned.

```rust
use ta_core::overlap::{DEMA, EMA, MA, SMA, T3, TEMA, TRIMA, WMA};
```

```rust
let implemented = [
    "SMA",
    "DEMA",
    "EMA",
    "MA",
    "T3",
    "TEMA",
    "TRIMA",
    "WMA",
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
```

```rust
assert_indicator::<MA>();
assert_streaming::<MA>();
```

```rust
for name in ["KAMA", "MAMA", "MACD", "BBANDS", "ATR", "OBV", "CDLDOJI", "VAR", "HT_SINE"] {
    let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(info.status, ImplementationStatus::Planned, "{name}");
}
```

### Success Criteria:

#### Automated Verification:
- [x] Moving-average tests pass with MA dispatcher coverage: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests pass after marking MA implemented: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after MA exports: `cargo test -p ta-core`
- [x] f32 precision compiles for MA dispatch: `cargo test -p ta-core --features f32`
- [x] Workspace checks after MA public exports: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] MA is wired in overlap module: `grep -n "mod ma\|pub use ma" crates/ta-core/src/overlap/mod.rs`
- [x] Final moving-average inventory count is 39: `grep -n "IMPLEMENTED_FUNCTION_COUNT: usize = 39" crates/ta-core/src/inventory.rs`
- [x] MA is implemented while KAMA/MAMA remain planned: `grep -n 'function!("MA", OverlapStudies, Implemented)\|function!("KAMA", OverlapStudies, Planned)\|function!("MAMA", OverlapStudies, Planned)' crates/ta-core/src/inventory.rs`
- [x] Moving-average implementations avoid private arch backends: `grep -R "crate::simd::arch\|core::arch\|std::arch" crates/ta-core/src/overlap` returns no matches

#### Manual Verification:
- [x] `MAType` exposes official ids 0-8 for SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/MAMA/T3.
- [x] `MA` delegates implemented variants to the standalone functions without reimplementing algorithms.
- [x] `MAType::T3` uses `T3_DEFAULT_VFACTOR`.
- [x] `MAType::KAMA` and `MAType::MAMA` return `TalibError::NotImplemented` and remain planned in inventory.
- [x] `MA` struct streaming delegates to the selected inner indicator and resets inner state.

## Phase 6: Moving-average benchmarks

### Overview

Extends the existing first-tranche Criterion benchmark target with compact and selected padded-wrapper benchmarks for the new moving-average APIs. Depends on Phases 1-5.

### Changes Required:

#### 1. crates/ta-benchmarks/benches/first_tranche.rs:10-228
**File**: `crates/ta-benchmarks/benches/first_tranche.rs`
**Changes**: MODIFY — import new overlap APIs, add moving-average benchmark group, and register it in `criterion_group!`.

```rust
use ta_core::{
    math_operators::{ADD, MINMAX, SUM},
    math_transform::SQRT,
    overlap::{
        DEMA, EMA, EMA_vec, MAType, MA, SMA_vec, SMA, T3_with_default_vfactor, TEMA, TRIMA,
        WMA,
    },
    price_transform::{AVGDEV, AVGPRICE},
    Float,
};
```

```rust
fn bench_overlap_moving_averages(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap/moving_averages");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("EMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = EMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid EMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("EMA_vec", size), &size, |b, &size| {
            let prices = series_fixture(size);

            b.iter(|| {
                let output = EMA_vec(black_box(prices.as_slice()), black_box(PERIOD))
                    .expect("valid EMA benchmark fixture");
                black_box(output);
            });
        });

        group.bench_with_input(BenchmarkId::new("WMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = WMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid WMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("TRIMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = TRIMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid TRIMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("DEMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = DEMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid DEMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("TEMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = TEMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid TEMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("T3_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = T3_with_default_vfactor(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid T3 benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("MA_EMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = MA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(MAType::EMA),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid MA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}
```

```rust
criterion_group!(
    benches,
    bench_overlap_sma,
    bench_overlap_moving_averages,
    bench_price_transform,
    bench_math_transform,
    bench_math_operators
);
criterion_main!(benches);
```

### Success Criteria:

#### Automated Verification:
- [x] First-tranche benchmark target compiles with moving-average additions: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`
- [x] Existing basic benchmark target still compiles: `cargo bench -p ta-benchmarks --bench basic --no-run`
- [x] Moving-average core tests still pass after benchmark imports compile: `cargo test -p ta-core --test overlap_moving_averages`
- [x] Inventory tests still pass with final count: `cargo test -p ta-core --test inventory`
- [x] Core crate tests pass after all moving-average phases: `cargo test -p ta-core`
- [x] f32 precision tests pass after all moving-average phases: `cargo test -p ta-core --features f32`
- [x] Workspace checks after final public exports and benchmarks: `cargo check --workspace`
- [x] Formatting passes: `cargo fmt --all -- --check`
- [x] Benchmark group is registered: `grep -n "bench_overlap_moving_averages" crates/ta-benchmarks/benches/first_tranche.rs`
- [x] Benchmark file uses public ta-core APIs only: `grep -R "ta_core::.*src\|crate::" crates/ta-benchmarks/benches/first_tranche.rs` returns no matches
- [x] Python/WASM adapters remain untouched in this core+benchmarks plan: `git diff --name-only -- crates/ta-py crates/ta-wasm` returns no matches

#### Manual Verification:
- [x] Fixtures and output buffers are allocated outside `b.iter()` except intentional padded wrapper allocation benchmark `EMA_vec`.
- [x] Benchmark group covers EMA, WMA, TRIMA, DEMA, TEMA, T3, and MA dispatcher compact APIs.
- [x] Every new `bench_*` function is registered in `criterion_group!`.
- [x] Benchmark code imports only public `ta_core::overlap` APIs and does not touch private core modules.

## Ordering Constraints

- Phase 1 must land first because DEMA, TEMA, and T3 reuse the EMA helper and later test/inventory modifications build on its patterns.
- Phase 2 depends on Phase 1, Phase 3 depends on Phases 1-2, and Phase 4 depends on Phases 1-3 because each later phase extends the cumulative overlap module, moving-average test file, and inventory status list.
- Phase 5 depends on Phases 1-4 plus existing `SMA` because `MA` dispatches across all implemented moving-average variants.
- Phase 6 depends on Phases 1-5 because benchmark imports must reference public APIs that already exist.
- Do not declare future `overlap` modules before their files exist; each phase wires only the modules it creates.

## Verification Notes

- Run `cargo fmt --all -- --check` after every phase that changes Rust source.
- Run `cargo test -p ta-core --test overlap_moving_averages` after every core moving-average phase once the test file exists.
- Run `cargo test -p ta-core --test inventory` after every phase that changes inventory status/counts.
- Run `cargo test -p ta-core --features f32` after each phase adding arithmetic or recursive Float-heavy code.
- Run `cargo test -p ta-core` after each core phase to catch doc/unit/integration regressions.
- Run `cargo check --workspace` after public module/export changes; note workspace checks include `ta-py` and may require Python/PyO3 environment.
- Verify no future modules are declared early: `grep -n "mod wma\|mod trima\|mod dema\|mod tema\|mod t3\|mod ma" crates/ta-core/src/overlap/mod.rs` should match only modules whose files exist in the current phase.
- Verify no new constructors use `assert!` for user parameters: `grep -R "assert!(.*period\|assert!(.*vfactor" crates/ta-core/src/overlap` should not find new constructors.
- Verify moving-average implementations do not call private architecture backends: `grep -R "crate::simd::arch\|core::arch\|std::arch" crates/ta-core/src/overlap` should return no matches.
- Verify final inventory count is 39 after `EMA/WMA/TRIMA/DEMA/TEMA/T3/MA` are implemented.
- Run `cargo bench -p ta-benchmarks --bench first_tranche --no-run` after Phase 6.

## Precedents & Lessons

- First-tranche plan review caught compile blockers from declaring modules before files existed; this plan wires overlap modules only in the phase that creates each file.
- SMA/API churn previously centered on compact vs padded output, streaming warm-up, and `Result` constructors; this plan preserves the validated `SMA` contract rather than altering traits.
- Prior generic `wide` SIMD attempts regressed benchmark results; this plan avoids SIMD rewrites and keeps performance work at scalar algorithm level.
- Python/WASM adapters remain placeholders and no exact end-to-end indicator binding precedent exists; this plan defers bindings instead of inventing first real adapter APIs.
- Benchmarks should use public `ta_core` APIs and keep fixtures/output buffers outside `b.iter()`, following `crates/ta-benchmarks/benches/first_tranche.rs:58-89`.

## Performance Considerations

- EMA-derived indicators should compute recursively in O(n) and avoid allocating padded vectors internally when compact output is requested.
- WMA should use rolling weighted/window sums rather than rescanning each window where practical.
- TRIMA may use triangular weights or equivalent SMA-of-SMA logic, but compact output must remain aligned to `lookback = timeperiod - 1`.
- T3 chains six EMA states and should avoid per-output heap allocation.
- Padded `*_vec` wrappers are convenience APIs; compact functions remain the benchmarked performance-critical surface.
- No private SIMD/architecture backend calls are introduced in this plan.

## Migration Notes

- No persisted data or schema migration is involved.
- Public inventory count increases from 32 to 39 as the seven requested official functions become implemented.
- Existing `SMA` APIs remain unchanged.
- `MAType` will expose full official variants; `KAMA` and `MAMA` are valid enum values but return `TalibError::NotImplemented` through `MA` until their algorithms are implemented.

## Pattern References

- `crates/ta-core/src/overlap/sma.rs:23-44` — compact moving-average function shape.
- `crates/ta-core/src/overlap/sma.rs:49-72` — padded vector wrapper shape.
- `crates/ta-core/src/overlap/sma.rs:76-158` — uppercase struct and `Indicator` implementation shape.
- `crates/ta-core/src/overlap/sma.rs:162-200` — streaming and reset shape.
- `crates/ta-core/src/common.rs:101-118` — period/input/output validation helpers.
- `crates/ta-core/src/common.rs:159-169` — finite-input validation helper.
- `crates/ta-core/src/inventory.rs:153-168` — overlap function inventory records to update.
- `crates/ta-core/tests/overlap_sma.rs:12-75` — moving-average test pattern.
- `crates/ta-benchmarks/benches/first_tranche.rs:58-89` — Criterion benchmark pattern for compact and padded APIs.

## Developer Context

**Q (`crates/ta-core/src/overlap/sma.rs:23`, `:49`, `:76`, `:137`, `:162`; `crates/ta-core/src/overlap/mod.rs:3-5`): About to follow the existing overlap indicator surface for EMA/WMA/TRIMA/DEMA/TEMA/T3/MA: uppercase compact function, `*_vec` wrapper, uppercase struct, `Indicator` + `StreamingIndicator` impls, and family re-export. Confirm that's the direction, or moving off it?**
A: Follow dual API.

**Q (`crates/ta-core/src/overlap/sma.rs:87`, `crates/ta-core/src/common.rs:101`, `:118`, `:159`; `crates/ta-core/tests/overlap_sma.rs:52`): About to follow strict Rust validation for the new indicators: `Result<Self>` constructors, `period_lookback` / output checks, and finite-input rejection. Confirm that's the direction, or moving off it?**
A: Follow strict Result.

**Q (`crates/ta-py/src/lib.rs:12-20`, `crates/ta-wasm/src/lib.rs:9-17`, `crates/ta-benchmarks/benches/first_tranche.rs:58`, `:228`): Scope boundary — should this blueprint include Python/WASM bindings for EMA/WMA/TRIMA/DEMA/TEMA/T3/MA now?**
A: Core+benchmarks.

**Q (`crates/ta-core/src/inventory.rs:157`): `MA` is TA-Lib Overlap Studies' generic moving-average dispatcher, not a distinct formula. Should this plan include `overlap::MA`?**
A: Do `overlap::MA`.

**Q (`crates/ta-core/src/inventory.rs:156`, `:158`): Since `KAMA`/`MAMA` remain planned, should `MAType` expose only implemented variants or all official variants?**
A: Full official types.

**Q (`crates/ta-core/src/overlap/sma.rs:23`, `:87`): `T3` has a second parameter `vfactor`, default `0.7`, range `[0,1]`. How should Rust expose it?**
A: Default+explicit.

**Q (design summary): Ready to proceed to decomposition?**
A: Proceed.

**Q (decomposition): 6 slices for overlap moving averages. Slice 1: EMA foundation. Slices 2-6: WMA/TRIMA, DEMA/TEMA, T3, MA dispatcher, benchmarks. Approve decomposition?**
A: Approve.

## Plan History

- Phase 1: EMA foundation — approved as generated
- Phase 2: Weighted and triangular moving averages — approved as generated
- Phase 3: EMA-derived composites — approved as generated
- Phase 4: T3 with vfactor — approved as generated
- Phase 5: MA dispatcher — approved as generated
- Phase 6: Moving-average benchmarks — approved as generated

## Plan Review (Step 8)

_Independent post-finalization review by artifact-code-reviewer and artifact-coverage-reviewer subagents. Findings triaged at Step 9._

| source | plan-loc | codebase-loc | severity | dimension | finding | recommendation | resolution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coverage | ## Precedents & Lessons §4 | <n/a> | blocker | verification-coverage | Lesson "Python/WASM adapters remain placeholders and no exact end-to-end indicator binding precedent exists; this plan defers bindings instead of inventing first real adapter APIs." — criteria NOT FOUND; code NOT FOUND | Add an automated bullet under Phase 6's `#### Automated Verification:`: `git diff --name-only -- crates/ta-py crates/ta-wasm` returns no matches | applied: added Phase 6 automated adapter no-diff success criterion |
| code | Phase 3 | <n/a> | concern | actionability | Phase 3 says it “can run before or after Phase 2 once EMA exists,” but its fences already reference Phase 2’s `TRIMA`/`WMA` modules, exports, tests, and inventory statuses. | Declare Phase 3 as depending on Phase 2 or rewrite its fences so they do not mention WMA/TRIMA. | applied: updated Phase 3 overview and ordering constraints to depend on prior cumulative phases |
| code | Phase 3 §1 (dema.rs) | crates/ta-core/src/common.rs:89 | concern | code-quality | `period_lookback("timeperiod", timeperiod)? * 2` is unchecked while HEAD validation only rejects zero, so very large periods can panic or wrap instead of returning `TalibError`. | Use checked lookback multiplication during construction and batch validation, and store the checked lookback for `Indicator::lookback`. | applied: changed DEMA lookback to checked_mul, stored checked lookback on the struct, and added overflow-period validation coverage |
| code | Phase 3 §2 (tema.rs) | crates/ta-core/src/common.rs:89 | concern | code-quality | `period_lookback("timeperiod", timeperiod)? * 3` is unchecked while HEAD validation only rejects zero, so very large periods can panic or wrap instead of returning `TalibError`. | Use checked lookback multiplication during construction and batch validation, and store the checked lookback for `Indicator::lookback`. | applied: changed TEMA lookback to checked_mul, stored checked lookback on the struct, and added overflow-period validation coverage |
| code | Phase 4 | <n/a> | concern | actionability | Phase 4 says it depends on Phase 1 and “can run after Phase 1,” but its fences reference DEMA/TEMA/TRIMA/WMA symbols created in Phases 2-3. | Declare Phase 4 as depending on Phases 2 and 3 or rewrite its fences to include only symbols available after Phase 1. | applied: updated Phase 4 overview and ordering constraints to depend on prior cumulative phases |
| code | Phase 4 §1 (t3.rs) | crates/ta-core/src/common.rs:89 | concern | code-quality | `period_lookback("timeperiod", timeperiod)? * 6` is unchecked while HEAD validation only rejects zero, so very large periods can panic or wrap instead of returning `TalibError`. | Use checked lookback multiplication during construction and batch validation, and store the checked lookback for `Indicator::lookback`. | applied: changed T3 lookback to checked_mul, stored checked lookback on the struct, and added overflow-period validation coverage |

## References

- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — parent inventory and architecture research.
- `.rpiv/artifacts/designs/2026-07-04_17-28-24_rust-talib-core-foundation-first-tranche.md` — first-tranche design precedent.
- `.rpiv/artifacts/plans/2026-07-05_09-51-36_rust-talib-core-foundation-first-tranche.md` — first-tranche implementation plan precedent and plan-review lessons.
- `.rpiv/artifacts/validation/2026-07-05_11-35-24_ta-lib-rust-core-foundation-first-tranche.md` — validated first-tranche patterns and commands.
- `.rpiv/artifacts/benchmarks/2026-07-05_simd-attempt-comparison.md` — SIMD regression precedent.
- `https://ta-lib.org/functions/ema/` — official EMA reference.
- `https://ta-lib.org/functions/wma/` — official WMA reference.
- `https://ta-lib.org/functions/trima/` — official TRIMA reference.
- `https://ta-lib.org/functions/dema/` — official DEMA reference.
- `https://ta-lib.org/functions/tema/` — official TEMA reference.
- `https://ta-lib.org/functions/t3/` — official T3 reference.
- `https://ta-lib.org/functions/ma/` — official MA reference.
- `https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_EMA.c` — TA-Lib EMA C implementation notes.
- `https://raw.githubusercontent.com/TA-Lib/ta-lib/main/src/ta_func/ta_T3.c` — TA-Lib T3 C implementation notes.
