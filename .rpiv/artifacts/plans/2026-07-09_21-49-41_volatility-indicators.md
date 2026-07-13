---
date: 2026-07-09T21:49:41+0800
author: unknown
commit: c6f3630
branch: main
repository: fast-ta
topic: "实现Volatility分组指标"
tags: [plan, blueprint, ta-core, volatility, atr, natr, trange]
status: ready
parent: .rpiv/artifacts/research/2026-07-09_21-23-34_volatility-indicators.md
phase_count: 4
phases:
  - { n: 1, title: TRANGE foundation }
  - { n: 2, title: ATR }
  - { n: 3, title: NATR }
  - { n: 4, title: Benchmarks }
unresolved_phase_count: 0
last_updated: 2026-07-09T21:49:41+0800
last_updated_by: unknown
---

# Volatility Indicators Implementation Plan

## Overview
Add TA-Lib Volatility Indicators to `ta-core`: `TRANGE`, `ATR`, and `NATR`. The implementation follows the existing group facade and compact-output contracts, with `TRANGE` as the HLC/previous-close foundation, `ATR` as Wilder-smoothed true range, `NATR` as normalized ATR with exact TA-Lib period=1 behavior, and Criterion coverage for the public compact kernels.

## Requirements
- Implement `ta_core::volatility::{TRANGE, ATR, NATR}` with uppercase free functions, `_vec` wrappers, structs, HLC input views, streaming ticks, `Indicator`, `StreamingIndicator`, and reset support where stateful.
- Preserve TA-Lib compact output semantics via `OutputRange`, with padded `Vec<Float>` wrappers returning full-length vectors and `Float::NAN` warm-up values.
- Match TA-Lib default unstable period behavior for ATR/NATR: unstable period fixed at 0; no global unstable-period API in this tranche.
- Match exact TA-Lib period=1 behavior: `ATR` returns TRANGE and `NATR` follows TA-Lib’s TRANGE-like special path rather than normalizing.
- Update inventory status/counts and inventory trait-conformance tests incrementally as each indicator lands.
- Add integration tests for expected values, padding, struct/trait surfaces, streaming/reset, invalid inputs, and f32 compatibility through existing workspace checks.
- Add Criterion benchmarks using public APIs only and prepared fixtures outside `b.iter()`.

## Current State Analysis
`ta-core` already reserves the Volatility group and its three functions in inventory, but no `volatility` module exists. Existing Price Transform and Overlap indicators provide the API, validation, compact-output, streaming, and benchmark patterns needed for implementation.

### Key Discoveries
- `crates/ta-core/src/inventory.rs:206-208` records `ATR`, `NATR`, and `TRANGE` as `Planned` under `VolatilityIndicators`.
- `crates/ta-core/src/inventory.rs:12` reports `IMPLEMENTED_FUNCTION_COUNT = 39`; each phase increments this count as a new Volatility function becomes implemented.
- `crates/ta-core/src/inventory.rs:87-93` already maps the Volatility group to Rust module name `"volatility"`.
- `crates/ta-core/src/lib.rs:41-43` exposes `overlap` and `price_transform`, but no `pub mod volatility;` yet.
- `crates/ta-core/src/price_transform/mod.rs:6-16` is the group facade template: private implementation modules plus explicit `pub use` exports.
- `crates/ta-core/src/price_transform/typprice.rs:15-55` is the HLC SoA input and validation template.
- `crates/ta-core/src/common.rs:22-49` and `crates/ta-core/src/common.rs:180-209` define compact output and padded-vector conversion.
- `crates/ta-core/src/overlap/ema.rs:156-185` is the closest streaming/reset pattern for recursive smoothing.
- `crates/ta-core/tests/inventory.rs:217-222` currently asserts `ATR` remains planned, so ATR’s phase must remove it from the deferred list.
- `crates/ta-benchmarks/benches/first_tranche.rs:233-278` shows the public-API Criterion style to extend.

## Desired End State
Consumers can call compact kernels, vector wrappers, struct methods, and streaming APIs through `ta_core::volatility`:

```rust
use ta_core::volatility::{ATR, ATRInput, ATRTick, ATR_vec, NATR, TRANGE};
use ta_core::{Indicator, StreamingIndicator};

let high = [10.0, 12.0, 11.0, 15.0, 16.0];
let low = [8.0, 9.0, 10.0, 13.0, 14.0];
let close = [9.0, 11.0, 10.0, 14.0, 15.0];
let mut compact = [0.0; 5];

let range = TRANGE(&high, &low, &close, &mut compact)?;
assert_eq!(range.beg_idx, 1);

let atr = ATR::new(3)?;
let atr_range = Indicator::compute(
    &atr,
    ATRInput { high: &high, low: &low, close: &close },
    &mut compact,
)?;
assert_eq!(atr_range.beg_idx, 3);

let padded = ATR_vec(&high, &low, &close, 3)?;
assert!(padded[..3].iter().all(|value| value.is_nan()));

let mut streaming = ATR::new(3)?;
for tick in high.iter().zip(low.iter()).zip(close.iter()).map(|((high, low), close)| ATRTick {
    high: *high,
    low: *low,
    close: *close,
}) {
    let _ = streaming.next(tick)?;
}
# Ok::<(), ta_core::TalibError>(())
```

## What We're NOT Doing
- No Python (`ta-py`) or WASM (`ta-wasm`) bindings in this plan; adapters stay out of `ta-core` changes.
- No TA-Lib global unstable-period configuration; ATR/NATR implement default unstable period = 0 only.
- No SIMD specialization; Volatility kernels are scalar baseline public APIs in this tranche.
- No non-Volatility functions (`SAR`, momentum indicators, volume indicators) are pulled into scope.
- No workspace membership or crate dependency changes.

## Decisions

### New Volatility module follows existing group facade
Decision: create `crates/ta-core/src/volatility/mod.rs` with private implementation modules and explicit `pub use` exports, and expose it from root `lib.rs`.
Evidence: `crates/ta-core/src/price_transform/mod.rs:6-16` and `crates/ta-core/src/lib.rs:41-43`.
Developer checkpoint: Follow facade.

### HLC APIs use named SoA input/tick structs
Decision: each Volatility indicator exposes `*Input<'a>` and `*Tick` structs with `high`, `low`, and `close` fields, modeled after `TYPPRICEInput` and `TYPPRICETick`.
Evidence: `crates/ta-core/src/price_transform/typprice.rs:15-33`.

### Compact output and padding use existing helpers
Decision: all batch kernels write compact outputs from `out_real[0]` and return `OutputRange`; `_vec` wrappers use `compact_buffer` and `padded_from_compact`.
Evidence: `crates/ta-core/src/common.rs:22-49`, `crates/ta-core/src/common.rs:180-209`, `crates/ta-core/src/price_transform/avgdev.rs:43-50`.

### ATR/NATR unstable period stays default-only
Ambiguity: TA-Lib supports configurable unstable periods for ATR/NATR, but the current crate has no global unstable-period API.
Explored:
- Default-only — matches TA-Lib default unstable period = 0 without introducing global state; fits `crates/ta-core/src/common.rs:101-113` and `crates/ta-core/src/lib.rs:35-49`.
- Add unstable config — larger core API/state design, outside this tranche.
Decision: default-only; do not add global unstable-period configuration.
Developer checkpoint: Default only.

### Period=1 follows exact TA-Lib behavior
Ambiguity: TA-Lib C has a special `timeperiod <= 1` path where ATR returns TRANGE and NATR also follows the TRANGE-like path.
Explored:
- Exact TA-Lib — preserves external compatibility while still accepting `timeperiod > 0` like `crates/ta-core/src/common.rs:89-103`.
- Always normalize NATR — more intuitive, but diverges from TA-Lib C behavior.
- Reject period 1 — diverges from existing nonzero-period constructor convention in `crates/ta-core/src/price_transform/avgdev.rs:64-73`.
Decision: exact TA-Lib behavior.
Developer checkpoint: Exact TA-Lib.

### Benchmarks are in scope
Decision: extend `crates/ta-benchmarks/benches/first_tranche.rs` with a Volatility benchmark group using prepared HLC fixtures and output buffers outside `b.iter()`.
Evidence: `crates/ta-benchmarks/benches/first_tranche.rs:233-278`.
Developer checkpoint: Include bench.

## Phase 1: TRANGE foundation

### Overview
Adds the Volatility module foundation and TRANGE end-to-end; depends on nothing and establishes the facade/test pattern for later phases.

### Changes Required:

#### 1. crates/ta-core/src/volatility/mod.rs
**File**: crates/ta-core/src/volatility/mod.rs
**Changes**: NEW — Volatility facade with TRANGE export.

```rust
//! Volatility Indicators.
//!
//! These functions measure price range and volatility from high/low/close inputs.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod trange;

pub use trange::{TRANGEInput, TRANGETick, TRANGE_vec, TRANGE};
```

#### 2. crates/ta-core/src/volatility/trange.rs
**File**: crates/ta-core/src/volatility/trange.rs
**Changes**: NEW — TRANGE batch, vec, struct, streaming, reset, and shared HLC helper functions.

```rust
//! True Range (TRANGE).

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`TRANGE`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct TRANGEInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`TRANGE`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRANGETick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Validates high/low/close slices and returns their shared length.
pub(super) fn validate_hlc(high: &[Float], low: &[Float], close: &[Float]) -> Result<usize> {
    let len = validate_all_same_len(&[
        ("high", high.len()),
        ("low", low.len()),
        ("close", close.len()),
    ])?;
    validate_finite_slices(&[("high", high), ("low", low), ("close", close)])?;
    Ok(len)
}

/// Computes one TA-Lib true range value from current high/low and previous close.
#[inline]
pub(super) fn true_range(high: Float, low: Float, previous_close: Float) -> Float {
    let high_low = high - low;
    let high_close = (high - previous_close).abs();
    let low_close = (low - previous_close).abs();
    Float::max(high_low, Float::max(high_close, low_close))
}

/// TA-Lib-style True Range batch function.
#[allow(non_snake_case)]
pub fn TRANGE(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let len = validate_hlc(high, low, close)?;
    let lookback = 1;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("TRANGE", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    for output_idx in 0..count {
        let input_idx = output_idx + lookback;
        out_real[output_idx] = true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn TRANGE_vec(high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = TRANGE(high, low, close, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// True Range indicator.
#[derive(Debug, Clone, Default)]
pub struct TRANGE {
    previous_close: Option<Float>,
}

impl TRANGE {
    /// Creates a True Range calculator.
    pub fn new() -> Result<Self> {
        Ok(Self {
            previous_close: None,
        })
    }

    /// Computes compact outputs.
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        TRANGE(high, low, close, out_real)
    }

    /// Computes full-length outputs.
    pub fn compute_to_vec(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
    ) -> Result<Vec<Float>> {
        TRANGE_vec(high, low, close)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: TRANGETick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for TRANGE {
    type Input<'a> = TRANGEInput<'a>;
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
        TRANGE(input.high, input.low, input.close, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        TRANGE_vec(input.high, input.low, input.close)
    }
}

impl StreamingIndicator for TRANGE {
    type Tick = TRANGETick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let value = true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);
        Ok(Some(value))
    }
}

impl Resettable for TRANGE {
    fn reset(&mut self) {
        self.previous_close = None;
    }
}
```

#### 3. crates/ta-core/src/lib.rs:41-43
**File**: crates/ta-core/src/lib.rs
**Changes**: MODIFY — expose the new Volatility module from the crate root.

```rust
/// Overlap studies: Moving averages and other price overlay indicators.
pub mod overlap;
/// Price Transform functions.
pub mod price_transform;
/// Volatility Indicators functions.
pub mod volatility;
pub mod simd;
```

#### 4. crates/ta-core/src/inventory.rs:12,206-208
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark TRANGE implemented and increment implemented count to 40.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 40;

// Volatility Indicators — 3 functions.
function!("ATR", VolatilityIndicators, Planned),
function!("NATR", VolatilityIndicators, Planned),
function!("TRANGE", VolatilityIndicators, Implemented),
```

#### 5. crates/ta-core/tests/inventory.rs:1-222
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — import TRANGE, add it to implemented inventory and trait checks.

```rust
// Add this import next to the existing group imports.
use ta_core::volatility::TRANGE;

#[test]
fn first_tranche_functions_are_marked_implemented() {
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
        "TRANGE",
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
fn first_tranche_structs_implement_batch_and_streaming_traits() {
    fn assert_indicator<T: Indicator>() {}
    fn assert_streaming<T: StreamingIndicator>() {}

    // Existing assertions remain unchanged.
    assert_indicator::<TRANGE>();
    assert_streaming::<TRANGE>();
}
```

#### 6. crates/ta-core/tests/volatility.rs
**File**: crates/ta-core/tests/volatility.rs
**Changes**: NEW — TRANGE integration tests for compact output, padded output, struct/trait surface, streaming/reset, and invalid inputs.

```rust
use ta_core::volatility::{TRANGEInput, TRANGETick, TRANGE_vec, TRANGE};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn trange_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];
    let mut output = [0.0; 4];

    let range = TRANGE(&high, &low, &close, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 3.0);
    assert_close(output[1], 1.0);
    assert_close(output[2], 5.0);
}

#[test]
fn trange_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];

    let output = TRANGE_vec(&high, &low, &close).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[0].is_nan());
    assert_close(output[1], 3.0);
    assert_close(output[2], 1.0);
    assert_close(output[3], 5.0);
}

#[test]
fn trange_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];
    let trange = TRANGE::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &trange,
        TRANGEInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(trange.lookback(), 1);
    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 3.0);
}

#[test]
fn trange_streaming_next_and_reset_are_safe() {
    let mut trange = TRANGE::new().unwrap();

    assert!(trange
        .next_checked(TRANGETick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        })
        .unwrap()
        .is_nan());
    assert_close(
        trange
            .next_checked(TRANGETick {
                high: 12.0,
                low: 9.0,
                close: 11.0,
            })
            .unwrap(),
        3.0,
    );
    assert_close(
        trange
            .next_checked(TRANGETick {
                high: 11.0,
                low: 10.0,
                close: 10.0,
            })
            .unwrap(),
        1.0,
    );

    trange.reset();
    assert!(trange
        .next_checked(TRANGETick {
            high: 15.0,
            low: 13.0,
            close: 14.0,
        })
        .unwrap()
        .is_nan());
    assert!(trange
        .next(TRANGETick {
            high: Float::NAN,
            low: 13.0,
            close: 14.0,
        })
        .is_err());
}

#[test]
fn trange_rejects_bad_inputs() {
    let mut output = [0.0; 4];

    assert!(TRANGE(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &mut output).is_err());
    assert!(TRANGE(
        &[1.0, Float::NAN],
        &[0.0, 1.0],
        &[0.5, 1.5],
        &mut output,
    )
    .is_err());
    assert!(TRANGE(&[1.0], &[0.0], &[0.5], &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(TRANGE(
        &[1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0],
        &[0.5, 1.5, 2.5],
        &mut too_small,
    )
    .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] Volatility tests pass for TRANGE: `cargo test -p ta-core volatility`
- [x] Inventory tests pass with TRANGE implemented: `cargo test -p ta-core inventory`
- [x] TRANGE compiles under f32 precision: `cargo test -p ta-core --features f32 volatility`

#### Manual Verification:
- [x] `crates/ta-core/src/lib.rs` exposes `pub mod volatility;` without changing adapter crates.
- [x] `crates/ta-core/src/inventory.rs` marks only `TRANGE` as newly implemented in this phase; `ATR` and `NATR` remain planned.
- [x] TRANGE first output maps to original index 1 and padded vectors leave index 0 as `Float::NAN`.

## Phase 2: ATR

### Overview
Adds ATR on top of TRANGE’s HLC helper and facade; depends on Phase 1.

### Changes Required:

#### 1. crates/ta-core/src/volatility/atr.rs
**File**: crates/ta-core/src/volatility/atr.rs
**Changes**: NEW — ATR batch, vec, struct, Wilder streaming, reset, and shared ATR helpers.

```rust
//! Average True Range (ATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, validate_period, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for [`ATR`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct ATRInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`ATR`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ATRTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

/// Returns the default TA-Lib ATR lookback for a period.
pub(super) fn atr_lookback(timeperiod: usize) -> Result<usize> {
    validate_period("timeperiod", timeperiod)?;
    timeperiod.checked_add(1).ok_or_else(|| {
        crate::TalibError::invalid_period(timeperiod, "ATR lookback would overflow")
    })?;
    Ok(timeperiod)
}

/// Applies one Wilder smoothing step.
#[inline]
pub(super) fn wilder_smooth(previous: Float, true_range: Float, timeperiod: usize) -> Float {
    ((previous * (timeperiod - 1) as Float) + true_range) / timeperiod as Float
}

/// TA-Lib-style Average True Range batch function.
#[allow(non_snake_case)]
pub fn ATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if timeperiod == 1 {
        validate_period("timeperiod", timeperiod)?;
        return super::trange::TRANGE(high, low, close, out_real);
    }

    let lookback = atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(high, low, close)?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("ATR", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut atr = 0.0 as Float;
    for input_idx in 1..=timeperiod {
        atr += super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }
    atr /= timeperiod as Float;
    out_real[0] = atr;

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range = super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
        atr = wilder_smooth(atr, range, timeperiod);
        out_real[output_idx] = atr;
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Average True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn ATR_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = ATR(high, low, close, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Average True Range indicator.
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl ATR {
    /// Creates a new Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        atr_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            previous_close: None,
            count: 0,
            true_range_sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact ATR outputs using this indicator's period.
    #[inline]
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        ATR(high, low, close, self.period, out_real)
    }

    /// Computes full-length padded ATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
        ATR_vec(high, low, close, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: ATRTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for ATR {
    type Input<'a> = ATRInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    #[inline]
    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        ATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        ATR_vec(input.high, input.low, input.close, self.period)
    }
}

impl StreamingIndicator for ATR {
    type Tick = ATRTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let range = super::trange::true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);

        if self.period == 1 {
            self.value = range;
            return Ok(Some(range));
        }

        if self.count < self.period {
            self.true_range_sum += range;
            self.count += 1;

            if self.count < self.period {
                return Ok(None);
            }

            self.value = self.true_range_sum / self.period as Float;
            return Ok(Some(self.value));
        }

        self.value = wilder_smooth(self.value, range, self.period);
        Ok(Some(self.value))
    }
}

impl Resettable for ATR {
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
```

#### 2. crates/ta-core/src/volatility/mod.rs
**File**: crates/ta-core/src/volatility/mod.rs
**Changes**: MODIFY — add ATR module and exports.

```rust
//! Volatility Indicators.
//!
//! These functions measure price range and volatility from high/low/close inputs.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod atr;
mod trange;

pub use atr::{ATRInput, ATRTick, ATR_vec, ATR};
pub use trange::{TRANGEInput, TRANGETick, TRANGE_vec, TRANGE};
```

#### 3. crates/ta-core/src/inventory.rs:12,206-208
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark ATR implemented and increment implemented count to 41.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 41;

// Volatility Indicators — 3 functions.
function!("ATR", VolatilityIndicators, Implemented),
function!("NATR", VolatilityIndicators, Planned),
function!("TRANGE", VolatilityIndicators, Implemented),
```

#### 4. crates/ta-core/tests/inventory.rs:1-222
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — import ATR, add it to implemented inventory and trait checks, remove ATR from deferred list.

```rust
// Replace the Volatility import added in Phase 1.
use ta_core::volatility::{ATR, TRANGE};

#[test]
fn first_tranche_functions_are_marked_implemented() {
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
        "ATR",
        "TRANGE",
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
fn first_tranche_structs_implement_batch_and_streaming_traits() {
    fn assert_indicator<T: Indicator>() {}
    fn assert_streaming<T: StreamingIndicator>() {}

    // Existing assertions remain unchanged.
    assert_indicator::<ATR>();
    assert_streaming::<ATR>();
    assert_indicator::<TRANGE>();
    assert_streaming::<TRANGE>();
}

#[test]
fn deferred_functions_remain_planned() {
    for name in [
        "KAMA", "MAMA", "MACD", "BBANDS", "OBV", "CDLDOJI", "VAR", "HT_SINE",
    ] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Planned, "{name}");
    }
}
```

#### 5. crates/ta-core/tests/volatility.rs
**File**: crates/ta-core/tests/volatility.rs
**Changes**: MODIFY — add ATR expected value, padding, struct, streaming/reset, period=1, and invalid-input tests.

```rust
// Extend the Phase 1 import.
use ta_core::volatility::{
    ATRInput, ATRTick, ATR_vec, TRANGEInput, TRANGETick, TRANGE_vec, ATR, TRANGE,
};

#[test]
fn atr_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut output = [0.0; 5];

    let range = ATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 3.0);
    assert_close(output[1], 8.0 / 3.0);
}

#[test]
fn atr_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];

    let output = ATR_vec(&high, &low, &close, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..3].iter().all(|value| value.is_nan()));
    assert_close(output[3], 3.0);
    assert_close(output[4], 8.0 / 3.0);
}

#[test]
fn atr_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let atr = ATR::new(3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &atr,
        ATRInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(atr.period(), 3);
    assert_eq!(atr.lookback(), 3);
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 3.0);
}

#[test]
fn atr_streaming_next_and_reset_are_safe() {
    let mut atr = ATR::new(3).unwrap();

    for tick in [
        ATRTick { high: 10.0, low: 8.0, close: 9.0 },
        ATRTick { high: 12.0, low: 9.0, close: 11.0 },
        ATRTick { high: 11.0, low: 10.0, close: 10.0 },
    ] {
        assert!(atr.next_checked(tick).unwrap().is_nan());
    }

    assert_close(
        atr.next_checked(ATRTick { high: 15.0, low: 13.0, close: 14.0 })
            .unwrap(),
        3.0,
    );
    assert_close(
        atr.next_checked(ATRTick { high: 16.0, low: 14.0, close: 15.0 })
            .unwrap(),
        8.0 / 3.0,
    );

    atr.reset();
    assert!(atr
        .next_checked(ATRTick { high: 10.0, low: 8.0, close: 9.0 })
        .unwrap()
        .is_nan());
    assert!(atr
        .next(ATRTick { high: Float::NAN, low: 8.0, close: 9.0 })
        .is_err());
}

#[test]
fn atr_period_one_matches_trange() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut atr_output = [0.0; 5];
    let mut trange_output = [0.0; 5];

    let atr_range = ATR(&high, &low, &close, 1, &mut atr_output).unwrap();
    let trange_range = TRANGE(&high, &low, &close, &mut trange_output).unwrap();

    assert_eq!(atr_range, trange_range);
    for idx in 0..atr_range.nb_element {
        assert_close(atr_output[idx], trange_output[idx]);
    }
}

#[test]
fn atr_rejects_invalid_inputs() {
    assert!(ATR::new(0).is_err());
    assert!(ATR::new(usize::MAX).is_err());

    let mut output = [0.0; 5];
    assert!(ATR(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0], &[0.5, 1.5, 2.5], 3, &mut output).is_err());
    assert!(ATR(&[1.0, Float::NAN, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0], &[0.5, 1.5, 2.5, 3.5], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(ATR(
        &[10.0, 12.0, 11.0, 15.0, 16.0],
        &[8.0, 9.0, 10.0, 13.0, 14.0],
        &[9.0, 11.0, 10.0, 14.0, 15.0],
        3,
        &mut too_small,
    )
    .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] Volatility tests pass with ATR: `cargo test -p ta-core volatility`
- [x] Inventory tests pass with ATR implemented: `cargo test -p ta-core inventory`
- [x] ATR compiles under f32 precision: `cargo test -p ta-core --features f32 volatility`

#### Manual Verification:
- [x] `crates/ta-core/src/volatility/atr.rs` uses `super::trange::true_range` and does not duplicate alternate true-range formulas.
- [x] ATR `lookback()` and batch `OutputRange::beg_idx` are both `timeperiod` for default unstable period = 0.
- [x] ATR period=1 path delegates to TRANGE behavior and does not introduce unstable-period configuration.

## Phase 3: NATR

### Overview
Adds NATR normalization and exact TA-Lib period=1 behavior; depends on Phase 2.

### Changes Required:

#### 1. crates/ta-core/src/volatility/natr.rs
**File**: crates/ta-core/src/volatility/natr.rs
**Changes**: NEW — NATR batch, vec, struct, streaming, reset, and zero-close normalization handling.

```rust
//! Normalized Average True Range (NATR).

use crate::{
    compact_buffer, padded_from_compact, validate_finite_slices, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

const ZERO_TOLERANCE: Float = 1e-8 as Float;

/// Borrowed SoA inputs for [`NATR`] batch computation.
#[derive(Debug, Clone, Copy)]
pub struct NATRInput<'a> {
    /// High price series.
    pub high: &'a [Float],
    /// Low price series.
    pub low: &'a [Float],
    /// Close price series.
    pub close: &'a [Float],
}

/// One high/low/close tick for [`NATR`] streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NATRTick {
    /// High price.
    pub high: Float,
    /// Low price.
    pub low: Float,
    /// Close price.
    pub close: Float,
}

#[inline]
fn normalize(atr: Float, close: Float) -> Float {
    if close.abs() <= ZERO_TOLERANCE {
        0.0 as Float
    } else {
        (atr / close) * 100.0 as Float
    }
}

/// TA-Lib-style Normalized Average True Range batch function.
#[allow(non_snake_case)]
pub fn NATR(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    if timeperiod == 1 {
        crate::validate_period("timeperiod", timeperiod)?;
        return super::trange::TRANGE(high, low, close, out_real);
    }

    let lookback = super::atr::atr_lookback(timeperiod)?;
    let len = super::trange::validate_hlc(high, low, close)?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("NATR", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut atr = 0.0 as Float;
    for input_idx in 1..=timeperiod {
        atr += super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
    }
    atr /= timeperiod as Float;
    out_real[0] = normalize(atr, close[lookback]);

    for output_idx in 1..count {
        let input_idx = lookback + output_idx;
        let range = super::trange::true_range(high[input_idx], low[input_idx], close[input_idx - 1]);
        atr = super::atr::wilder_smooth(atr, range, timeperiod);
        out_real[output_idx] = normalize(atr, close[input_idx]);
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Normalized Average True Range into a full-length vector.
#[allow(non_snake_case)]
pub fn NATR_vec(
    high: &[Float],
    low: &[Float],
    close: &[Float],
    timeperiod: usize,
) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(high.len());
    let range = NATR(high, low, close, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        high.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Normalized Average True Range indicator.
#[derive(Debug, Clone)]
pub struct NATR {
    period: usize,
    previous_close: Option<Float>,
    count: usize,
    true_range_sum: Float,
    value: Float,
}

impl NATR {
    /// Creates a new Normalized Average True Range indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        super::atr::atr_lookback(timeperiod)?;
        Ok(Self {
            period: timeperiod,
            previous_close: None,
            count: 0,
            true_range_sum: 0.0 as Float,
            value: 0.0 as Float,
        })
    }

    /// Returns the configured period.
    #[inline]
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact NATR outputs using this indicator's period.
    #[inline]
    pub fn compute(
        &self,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        NATR(high, low, close, self.period, out_real)
    }

    /// Computes full-length padded NATR outputs using this indicator's period.
    #[inline]
    pub fn compute_to_vec(&self, high: &[Float], low: &[Float], close: &[Float]) -> Result<Vec<Float>> {
        NATR_vec(high, low, close, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: NATRTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for NATR {
    type Input<'a> = NATRInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    #[inline]
    fn lookback(&self) -> usize {
        self.period
    }

    #[inline]
    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        NATR(input.high, input.low, input.close, self.period, output)
    }

    #[inline]
    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        NATR_vec(input.high, input.low, input.close, self.period)
    }
}

impl StreamingIndicator for NATR {
    type Tick = NATRTick;
    type TickOutput = Float;

    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
        validate_finite_slices(&[
            ("high", &[input.high]),
            ("low", &[input.low]),
            ("close", &[input.close]),
        ])?;

        let Some(previous_close) = self.previous_close else {
            self.previous_close = Some(input.close);
            return Ok(None);
        };

        let range = super::trange::true_range(input.high, input.low, previous_close);
        self.previous_close = Some(input.close);

        if self.period == 1 {
            self.value = range;
            return Ok(Some(range));
        }

        if self.count < self.period {
            self.true_range_sum += range;
            self.count += 1;

            if self.count < self.period {
                return Ok(None);
            }

            self.value = self.true_range_sum / self.period as Float;
            return Ok(Some(normalize(self.value, input.close)));
        }

        self.value = super::atr::wilder_smooth(self.value, range, self.period);
        Ok(Some(normalize(self.value, input.close)))
    }
}

impl Resettable for NATR {
    fn reset(&mut self) {
        self.previous_close = None;
        self.count = 0;
        self.true_range_sum = 0.0 as Float;
        self.value = 0.0 as Float;
    }
}
```

#### 2. crates/ta-core/src/volatility/mod.rs
**File**: crates/ta-core/src/volatility/mod.rs
**Changes**: MODIFY — add NATR module and exports.

```rust
//! Volatility Indicators.
//!
//! These functions measure price range and volatility from high/low/close inputs.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod atr;
mod natr;
mod trange;

pub use atr::{ATRInput, ATRTick, ATR_vec, ATR};
pub use natr::{NATRInput, NATRTick, NATR_vec, NATR};
pub use trange::{TRANGEInput, TRANGETick, TRANGE_vec, TRANGE};
```

#### 3. crates/ta-core/src/inventory.rs:12,206-208
**File**: crates/ta-core/src/inventory.rs
**Changes**: MODIFY — mark NATR implemented and increment implemented count to 42.

```rust
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 42;

// Volatility Indicators — 3 functions.
function!("ATR", VolatilityIndicators, Implemented),
function!("NATR", VolatilityIndicators, Implemented),
function!("TRANGE", VolatilityIndicators, Implemented),
```

#### 4. crates/ta-core/tests/inventory.rs:1-222
**File**: crates/ta-core/tests/inventory.rs
**Changes**: MODIFY — import NATR, add it to implemented inventory and trait checks.

```rust
// Replace the Volatility import added in Phase 2.
use ta_core::volatility::{ATR, NATR, TRANGE};

#[test]
fn first_tranche_functions_are_marked_implemented() {
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
        "ATR",
        "NATR",
        "TRANGE",
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
fn first_tranche_structs_implement_batch_and_streaming_traits() {
    fn assert_indicator<T: Indicator>() {}
    fn assert_streaming<T: StreamingIndicator>() {}

    // Existing assertions remain unchanged.
    assert_indicator::<ATR>();
    assert_streaming::<ATR>();
    assert_indicator::<NATR>();
    assert_streaming::<NATR>();
    assert_indicator::<TRANGE>();
    assert_streaming::<TRANGE>();
}
```

#### 5. crates/ta-core/tests/volatility.rs
**File**: crates/ta-core/tests/volatility.rs
**Changes**: MODIFY — add NATR expected value, zero-close, struct, streaming/reset, period=1, and invalid-input tests.

```rust
// Extend the Phase 2 import.
use ta_core::volatility::{
    ATRInput, ATRTick, ATR_vec, NATRInput, NATRTick, NATR_vec, TRANGEInput, TRANGETick,
    TRANGE_vec, ATR, NATR, TRANGE,
};

#[test]
fn natr_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut output = [0.0; 5];

    let range = NATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], (3.0 / 14.0) * 100.0);
    assert_close(output[1], ((8.0 / 3.0) / 15.0) * 100.0);
}

#[test]
fn natr_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];

    let output = NATR_vec(&high, &low, &close, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..3].iter().all(|value| value.is_nan()));
    assert_close(output[3], (3.0 / 14.0) * 100.0);
    assert_close(output[4], ((8.0 / 3.0) / 15.0) * 100.0);
}

#[test]
fn natr_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let natr = NATR::new(3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &natr,
        NATRInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(natr.period(), 3);
    assert_eq!(natr.lookback(), 3);
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], (3.0 / 14.0) * 100.0);
}

#[test]
fn natr_streaming_next_and_reset_are_safe() {
    let mut natr = NATR::new(3).unwrap();

    for tick in [
        NATRTick { high: 10.0, low: 8.0, close: 9.0 },
        NATRTick { high: 12.0, low: 9.0, close: 11.0 },
        NATRTick { high: 11.0, low: 10.0, close: 10.0 },
    ] {
        assert!(natr.next_checked(tick).unwrap().is_nan());
    }

    assert_close(
        natr.next_checked(NATRTick { high: 15.0, low: 13.0, close: 14.0 })
            .unwrap(),
        (3.0 / 14.0) * 100.0,
    );
    assert_close(
        natr.next_checked(NATRTick { high: 16.0, low: 14.0, close: 15.0 })
            .unwrap(),
        ((8.0 / 3.0) / 15.0) * 100.0,
    );

    natr.reset();
    assert!(natr
        .next_checked(NATRTick { high: 10.0, low: 8.0, close: 9.0 })
        .unwrap()
        .is_nan());
    assert!(natr
        .next(NATRTick { high: Float::NAN, low: 8.0, close: 9.0 })
        .is_err());
}

#[test]
fn natr_zero_close_outputs_zero() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 0.0];
    let mut output = [1.0; 4];

    let range = NATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 1));
    assert_close(output[0], 0.0);
}

#[test]
fn natr_period_one_matches_trange() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut natr_output = [0.0; 5];
    let mut trange_output = [0.0; 5];

    let natr_range = NATR(&high, &low, &close, 1, &mut natr_output).unwrap();
    let trange_range = TRANGE(&high, &low, &close, &mut trange_output).unwrap();

    assert_eq!(natr_range, trange_range);
    for idx in 0..natr_range.nb_element {
        assert_close(natr_output[idx], trange_output[idx]);
    }
}

#[test]
fn natr_rejects_invalid_inputs() {
    assert!(NATR::new(0).is_err());
    assert!(NATR::new(usize::MAX).is_err());

    let mut output = [0.0; 5];
    assert!(NATR(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0], &[0.5, 1.5, 2.5], 3, &mut output).is_err());
    assert!(NATR(&[1.0, Float::NAN, 3.0, 4.0], &[0.0, 1.0, 2.0, 3.0], &[0.5, 1.5, 2.5, 3.5], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(NATR(
        &[10.0, 12.0, 11.0, 15.0, 16.0],
        &[8.0, 9.0, 10.0, 13.0, 14.0],
        &[9.0, 11.0, 10.0, 14.0, 15.0],
        3,
        &mut too_small,
    )
    .is_err());
}
```

### Success Criteria:

#### Automated Verification:
- [x] Volatility tests pass with NATR: `cargo test -p ta-core volatility`
- [x] Inventory tests pass with all Volatility functions implemented: `cargo test -p ta-core inventory`
- [x] NATR compiles under f32 precision: `cargo test -p ta-core --features f32 volatility`

#### Manual Verification:
- [x] `crates/ta-core/src/volatility/natr.rs` reuses ATR lookback/Wilder helpers and TRANGE true-range helper.
- [x] NATR `timeperiod == 1` returns TRANGE-like values rather than normalized values, matching the developer decision.
- [x] NATR close-zero handling returns `0.0` for normalized outputs without panics or infinities.

## Phase 4: Benchmarks

### Overview
Adds public API Criterion coverage for the completed Volatility group; depends on Phase 3.

### Changes Required:

#### 1. crates/ta-benchmarks/benches/first_tranche.rs
**File**: crates/ta-benchmarks/benches/first_tranche.rs
**Changes**: MODIFY — import Volatility kernels, add HLC benchmark group, and register it.

```rust
use ta_core::{
    math_operators::{ADD, MINMAX, SUM},
    math_transform::SQRT,
    overlap::{
        EMA_vec, MAType, SMA_vec, T3_with_default_vfactor, DEMA, EMA, MA, SMA, TEMA, TRIMA, WMA,
    },
    price_transform::{AVGDEV, AVGPRICE},
    volatility::{ATR, NATR, TRANGE},
    Float,
};

fn bench_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/volatility");

    for &size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("TRANGE_compact", size),
            &size,
            |b, &size| {
                let (_open, high, low, close) = ohlc_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = TRANGE(
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid TRANGE benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("ATR_compact", size), &size, |b, &size| {
            let (_open, high, low, close) = ohlc_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = ATR(
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid ATR benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("NATR_compact", size), &size, |b, &size| {
            let (_open, high, low, close) = ohlc_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = NATR(
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid NATR benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overlap_sma,
    bench_overlap_moving_averages,
    bench_price_transform,
    bench_volatility,
    bench_math_transform,
    bench_math_operators
);
```

### Success Criteria:

#### Automated Verification:
- [x] Benchmark target compiles with Volatility group registered: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`
- [x] Workspace check passes after all Volatility phases: `cargo check --workspace`
- [x] Workspace tests pass after all Volatility phases: `cargo test --workspace`
- [x] f32 precision tests pass for ta-core: `cargo test -p ta-core --features f32`
- [x] Formatting is clean: `cargo fmt --all -- --check`

#### Manual Verification:
- [x] `bench_volatility` uses public `ta_core::volatility::{TRANGE, ATR, NATR}` exports only.
- [x] Benchmark fixtures and output buffers are allocated outside `b.iter()` for compact kernels.
- [x] No `ta-py` or `ta-wasm` files are changed by the plan.

## Ordering Constraints
- Phase 1 must run first because it creates the `volatility` module and shared HLC/TRANGE helper surface.
- Phase 2 depends on Phase 1 because ATR uses TRANGE true-range semantics and extends the facade/tests.
- Phase 3 depends on Phase 2 because NATR reuses ATR lookback/smoothing helpers and extends the completed Volatility facade.
- Phase 4 depends on Phase 3 because benchmarks import all three public Volatility kernels.
- Phases are sequential; do not parallelize implementation because inventory count/status and tests intentionally change incrementally.

## Verification Notes
- Run `cargo test -p ta-core volatility` after each core phase to verify the new integration tests for that phase.
- Run `cargo test -p ta-core inventory` after each inventory change to verify implemented count/status and trait conformance.
- Run `cargo test -p ta-core --features f32` on the terminal phase to verify `Float = f32` compatibility.
- Run `cargo check --workspace` and `cargo test --workspace` on the terminal phase.
- Run `cargo fmt --all -- --check` on the terminal phase.
- Run `cargo bench -p ta-benchmarks --bench first_tranche --no-run` after benchmark wiring to verify Criterion target compilation without executing benchmarks.
- Inspect that no `ta-py` or `ta-wasm` files changed; this plan is core + benchmarks only.
- Verify `ATR/NATR` do not expose or depend on unstable-period global configuration.
- Verify all Volatility APIs use public `Float`, `Result`, `OutputRange`, and existing validators; no private SIMD/backend calls.

## Performance Considerations
- TRANGE, ATR, and NATR are single-pass scalar kernels over HLC slices with O(n) time and O(1) extra state beyond caller-provided output.
- `_vec` wrappers allocate full-length output by design, matching existing convenience wrappers.
- Benchmark fixtures and output buffers are prepared outside `b.iter()` so compact-kernel timings exclude allocation.
- No SIMD specialization is introduced in this tranche; future SIMD can be planned separately behind existing dispatch boundaries.

## Migration Notes
No persisted schema, data migration, or backwards-compatibility migration is required. This adds new public APIs and updates inventory statuses for previously planned functions.

## Pattern References
- `crates/ta-core/src/price_transform/mod.rs:6-16` — group facade and public re-export pattern.
- `crates/ta-core/src/price_transform/typprice.rs:15-137` — HLC input/tick, multi-input validation, trait, and streaming shape.
- `crates/ta-core/src/price_transform/avgdev.rs:16-148` — period validation, compact output, `_vec`, fallible constructor, streaming/reset pattern.
- `crates/ta-core/src/overlap/ema.rs:25-185` — recursive seed, streaming alignment, `next_checked`, and reset pattern.
- `crates/ta-core/src/overlap/dema.rs:15-19` — custom lookback helper with overflow-safe error pattern.
- `crates/ta-core/tests/price_transform.rs:1-93` — HLC integration-test style and invalid-input coverage.
- `crates/ta-core/tests/overlap_moving_averages.rs:27-90` — compact/padded/streaming/reset test style for recursive indicators.
- `crates/ta-benchmarks/benches/first_tranche.rs:233-278` — Criterion group using prepared fixtures and output buffers.

## Developer Context
**Q (`crates/ta-core/src/common.rs:101-113`, `crates/ta-core/src/lib.rs:35-49`): TA-Lib ATR/NATR have an optional unstable-period adjustment, but current helpers only model fixed lookback/count and the crate exposes no global unstable-period API. Should this tranche implement default TA-Lib behavior only or introduce unstable-period configuration now?**
A: Default only. Implement ATR/NATR with TA-Lib default unstable period = 0; do not add global unstable-period configuration in this tranche.

**Q (`crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/src/lib.rs:41-43`): About to follow existing group-module facade conventions for Volatility: private files plus `pub use` exports. Confirm this direction, or are we moving off it?**
A: Follow facade.

**Q (`crates/ta-benchmarks/benches/first_tranche.rs:233-278`): About to include Criterion benchmark coverage because new public indicators follow the benchmark pattern and root guidance expects performance coverage for end-to-end indicators. Confirm this direction, or are we moving off it?**
A: Include bench.

**Q (`crates/ta-core/src/common.rs:89-103`, `crates/ta-core/src/price_transform/avgdev.rs:64-73`): For Volatility, TA-Lib has a special `timeperiod <= 1` path where ATR returns TRANGE and NATR’s C path also returns TRANGE-like values. Should the plan implement this exact TA-Lib period=1 behavior, or make NATR always normalized?**
A: Exact TA-Lib.

**Q (`crates/ta-core/src/inventory.rs:206-208`, `crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/src/overlap/ema.rs:156-185`): Design summary ready to proceed to decomposition?**
A: Proceed.

**Q (`crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/src/overlap/ema.rs:156-185`): 4 slices for Volatility Indicators. Slice 1: TRANGE foundation. Slices 2-4: ATR, NATR, benchmarks. Approve decomposition?**
A: Continue; proceed with decomposition.

## Plan Review (Step 8)

_Independent post-finalization review by artifact-code-reviewer and artifact-coverage-reviewer subagents. Findings triaged at Step 9._

| source | plan-loc | codebase-loc | severity | dimension | finding | recommendation | resolution |
| --- | --- | --- | --- | --- | --- | --- | --- |
| code | Phase 2 §1 (atr.rs) | `crates/ta-core/src/common.rs:111` | concern | code-quality | `atr_lookback` accepts `usize::MAX` and returns it as `lookback`, so ATR/NATR can reach the live helper’s unchecked `lookback + 1` and panic instead of returning `TalibError`. | Reject periods whose `timeperiod.checked_add(1)` would overflow before calling `validate_input_len`. | applied: added overflow rejection in `atr_lookback` and invalid-period tests for ATR/NATR `usize::MAX`. |
| code | Phase 1 §2 (trange.rs) | `crates/ta-core/src/overlap/ema.rs:75` | concern | codebase-fit | Phase 1 derives `Copy` for stateful `TRANGE { previous_close: Option<Float> }`, while existing stateful streaming indicators such as EMA derive only `Clone`; implicit copies can silently fork streaming state. | Remove `Copy` from TRANGE’s derive list. | applied: removed `Copy` from TRANGE derive so stateful streaming instances are not implicitly copied. |

## Plan History
- Phase 1: TRANGE foundation — approved as generated
- Phase 2: ATR — approved as generated
- Phase 3: NATR — approved as generated
- Phase 4: Benchmarks — approved as generated

## References
- `.rpiv/artifacts/research/2026-07-09_21-23-34_volatility-indicators.md` — parent research artifact.
- `crates/ta-core/src/price_transform/typprice.rs` — HLC input and multi-input API template.
- `crates/ta-core/src/price_transform/avgdev.rs` — period and compact-output template.
- `crates/ta-core/src/overlap/ema.rs` — recursive smoothing and streaming template.
- `crates/ta-benchmarks/benches/first_tranche.rs` — benchmark template.
