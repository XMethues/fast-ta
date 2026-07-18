---
date: 2026-07-16T19:30:52+0800
author: unknown
commit: ffd2451
branch: main
repository: fast-ta
topic: "Statistic Functions"
tags: [design, ta-core, statistic-functions, rolling-statistics, linear-regression]
status: ready
parent: .rpiv/artifacts/research/2026-07-15_20-37-23_statistic-functions.md
last_updated: 2026-07-16T19:30:52+0800
last_updated_by: unknown
---

# Design: Statistic Functions

## Summary

Add all nine TA-Lib Statistic Functions to `ta-core` through a new `statistic` facade: VAR, STDDEV, CORREL, BETA, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, and TSF. The implementation uses group-private O(1) rolling moment/regression engines shared by batch and streaming surfaces, while preserving the existing compact-output, padded-vector, typed-error, reset, inventory, and Criterion contracts.

## Requirements

- Implement the nine official Statistic Functions with TA-Lib-compatible formulas, period bounds, lookbacks, zero thresholds, and degenerate outputs.
- Expose compact free functions, full-length `*_vec` wrappers, same-named indicator structs, `Indicator`, `StreamingIndicator`, `Resettable`, and `next_checked()` for every function.
- Keep `PairInput` and `PairTick` public and owned by the `statistic` facade for CORREL and BETA.
- Use O(1) rolling transitions after initial fill; do not rescan every window or introduce SIMD in this tranche.
- Make batch and streaming use the same private state transitions after batch-level validation.
- Provide explicit and default-`nbdev` VAR/STDDEV APIs; reject non-finite `nbdev` as `InvalidParameter`.
- Pin regression golden behavior to upstream TA-Lib commit `1bec05cf72fa790e2e3ecca40e6607de15fe0a30`.
- Add strict contract, numerical, degenerate, ring-wrap, reset, validation-precedence, f64, and f32 tests.
- Mark all nine inventory records implemented and update compile-time discoverability checks.
- Benchmark all nine compact and vec APIs across input sizes 1,024/16,384/65,536 and periods 5/20/100/500.

## Current State Analysis

`ta-core` already has the complete protocol needed by the new group, but no `statistic` source module or statistical test target. All nine names are present only as `Planned` inventory records, and neither adapters nor benchmarks currently import statistical APIs.

### Key Discoveries

- Compact kernels write from output index zero and return original-input coordinates through `OutputRange` (`crates/ta-core/src/common.rs:16-58`).
- Empty input succeeds with `OutputRange::empty()`, while non-empty input below one complete window returns `InsufficientData` (`crates/ta-core/src/common.rs:81-126`).
- Paired APIs validate equal lengths before finite values (`crates/ta-core/src/math_operators/arithmetic.rs:31-48`, `crates/ta-core/src/volume/obv.rs:31-35`).
- Recent groups use private leaves, explicit facade re-exports, and `pub(super)` shared helpers (`crates/ta-core/src/volume/mod.rs:6-12`, `crates/ta-core/src/volume/ad.rs:40-69`).
- ADOSC demonstrates a shared private state transition used by batch and streaming (`crates/ta-core/src/volume/adosc.rs:45-86,121-132,263-287`).
- SMA and WMA provide the O(1) rolling-sum and order-sensitive weighted-sum recurrence patterns (`crates/ta-core/src/overlap/sma.rs:23-46,164-204`, `crates/ta-core/src/overlap/wma.rs:34-49,135-162`).
- Previous-observation warm-up is already modeled with `Option` state (`crates/ta-core/src/volatility/trange.rs:89-91,138-166`, `crates/ta-core/src/volume/obv.rs:79-83,135-143`).
- `Float` is f64 by default and f32 under the feature boundary, so all constants, accumulators, and tolerance policy must remain precision-aware (`crates/ta-core/src/types.rs:25-33`).
- Inventory discoverability is manually synchronized across count, statuses, names, and trait assertions (`crates/ta-core/src/inventory.rs:8-12,283-292`, `crates/ta-core/tests/inventory.rs:68-245`).
- Criterion setup, caller-owned compact buffers, allocating vec measurements, and explicit registration already exist in one target (`crates/ta-benchmarks/benches/first_tranche.rs:22-41,72-103,512-522`).

## Scope

### Building

- New `ta_core::statistic` public facade and five private implementation/engine files.
- Public statistic-owned paired batch/tick views.
- Nine compact kernels, nine padded vec wrappers, nine stateful structs, and VAR/STDDEV default-`nbdev` convenience surfaces.
- Shared rolling univariate moments, paired moments, BETA return state, and rolling regression state.
- A new comprehensive `crates/ta-core/tests/statistic.rs` integration target.
- Inventory status/count/name/trait integration.
- Full compact+vec Criterion size/period matrix for all nine functions.

### Not Building

- Python or WebAssembly statistic bindings.
- SIMD kernels, dispatch-table changes, or architecture-specific code.
- Changes to `Indicator`, `StreamingIndicator`, `Resettable`, `TalibError`, `Float`, or common output helpers.
- New dependencies, Cargo features, workspace members, benchmark targets, or lockfile changes.
- Welford, compensated summation, two-pass variance, or per-window regression rescans.
- Repository-wide repair or certification of the pre-existing no-std configuration.
- Schema, persistence, migration, or backward-compatibility shims; no existing statistic API is being replaced.

## Decisions

### Use a private engine-grouped `statistic` facade

**Ambiguity:** Existing groups usually use one private file per indicator, while five regression exports share the same fit and differ only by projection.

**Explored:**
- One API per file follows the literal volume/volatility layout (`crates/ta-core/src/volume/mod.rs:6-12`) but spreads shared state and repeated trait wiring across more files.
- Engine grouping keeps `moments.rs`, `variance.rs`, `correl.rs`, `beta.rs`, and `regression.rs` private while preserving explicit facade exports and sibling-only helpers (`crates/ta-core/src/volume/ad.rs:40-69`).

**Decision:** Group by engine. Public surface is re-exported only from `statistic/mod.rs`; internal states remain private or `pub(super)`.

### Share private state transitions between batch and streaming

**Decision:** Batch validates complete slices first, creates fresh private state, then feeds the same unchecked `push` transition used after per-tick streaming validation. This follows the shared-state shape in ADOSC (`crates/ta-core/src/volume/adosc.rs:45-86,121-132,263-287`) and keeps batch independent of accumulated struct history.

### Preserve the established public indicator contract

**Decision:** Every function receives a compact kernel, padded vec wrapper, same-named struct, inherent compute methods, `Indicator`, `StreamingIndicator`, `Resettable`, and `next_checked()`. `PairInput`/`PairTick` mirror the binary view shape (`crates/ta-core/src/math_operators/arithmetic.rs:13-29`) but are owned and exported by `statistic`.

### Apply function-specific periods and legal degenerate outputs

**Decision:** VAR/CORREL/BETA accept `1..=100000`; STDDEV and all five regression functions accept `2..=100000`. Standard lookback is `period - 1`; BETA lookback is `period`. Period, pair length, finite input, input sufficiency, and output capacity are validated in that order. Constant windows and zero variance/denominators produce valid zero or constant projections rather than `TalibError`; no new error variant is added (`crates/ta-core/src/error.rs:13-50`).

### Expose explicit and default `nbdev` surfaces

**Ambiguity:** TA-Lib exposes `nbdev`, while local optional-parameter precedent T3 provides both explicit and default convenience APIs (`crates/ta-core/src/overlap/t3.rs:59-151`).

**Decision:** VAR/STDDEV expose explicit `nbdev` kernels, vec wrappers, and constructors plus `*_with_default_nbdev`, `*_vec_with_default_nbdev`, and `with_default_nbdev()` conveniences using `1.0`. Any finite `Float` is accepted, including negative values; NaN and infinities return `InvalidParameter`. VAR stores and validates `nbdev` for API consistency but intentionally ignores it in computation.

### Pin regression arithmetic and golden comparison policy

**Ambiguity:** Upstream TA-Lib `v0.7.1` rescans every regression window, while commit `1bec05c…` uses an O(1) recurrence and FMA with numerics-changing low bits.

**Decision:** Pin golden vectors and arithmetic order to `1bec05c…`, including `Float::mul_add` for regression projections. External ordinary-scale goldens use `abs_diff <= abs_tol + rel_tol * max(abs(actual), abs(expected))` with f64 `(1e-12, 1e-10)` and f32 `(1e-4, 1e-4)`. Shared-path batch/stream results use exact bit comparison; cancellation stress vectors use precision-specific expected behavior rather than the broad oracle tolerance.

### Use TA-Lib rolling formulas rather than numerically stable alternatives

**Decision:** VAR/STDDEV retain `E[x²] - E[x]²`; CORREL/BETA retain centered rolling sums and the `1e-14` TA-Lib zero threshold. `TA_IS_ZERO` uses strict endpoints (`-1e-14 < value < 1e-14`). BETA transforms adjacent prices into simple returns, treats an interior near-zero previous price as a zero return, and applies the threshold directly to TA-Lib's period-scaled denominator `n * sum(x²) - sum(x)²`. Regression maintains rolling `sum_y` and reversed-position `sum_xy` in O(1), matching the pinned upstream recurrence.

### Benchmark the full public size/period matrix

**Ambiguity:** The existing fixed period 20 measures throughput but cannot expose O(n×period) regressions (`crates/ta-benchmarks/benches/first_tranche.rs:22-23`).

**Decision:** Benchmark compact and vec APIs for all nine functions at periods 5, 20, 100, and 500 for every existing input size. Fixtures and compact buffers remain outside `b.iter()`; vec allocation remains inside; the statistic group is explicitly registered.

### Keep adapters and stable core contracts untouched

**Decision:** The work remains in `ta-core` plus `ta-benchmarks`. `ta-py`, `ta-wasm`, common helpers, traits, errors, types, SIMD, manifests, and lockfile remain unchanged because no current adapter exposes indicator families and the existing contracts already express the feature.

## Architecture

### crates/ta-core/src/statistic/mod.rs — NEW

Public statistic facade, paired input/tick views, private module declarations, and explicit re-exports.

```rust
//! Statistic Functions.
//!
//! These functions calculate rolling variance, paired statistics, and linear
//! regression projections. Batch APIs write compact outputs; convenience
//! wrappers return input-length vectors padded with `Float::NAN`.

mod beta;
mod correl;
mod moments;
mod regression;
mod variance;

use crate::Float;

pub use beta::{BETA_vec, BETA};
pub use correl::{CORREL_vec, CORREL};
pub use regression::{
    LINEARREG_ANGLE_vec, LINEARREG_INTERCEPT_vec, LINEARREG_SLOPE_vec, LINEARREG_vec, TSF_vec,
    LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, TSF,
};
pub use variance::{
    STDDEV_vec, STDDEV_vec_with_default_nbdev, STDDEV_with_default_nbdev, VAR_vec,
    VAR_vec_with_default_nbdev, VAR_with_default_nbdev, STDDEV, VAR,
};

/// Borrowed paired real-valued inputs for statistic batch computation.
#[derive(Debug, Clone, Copy)]
pub struct PairInput<'a> {
    /// First real-valued input series.
    pub real0: &'a [Float],
    /// Second real-valued input series.
    pub real1: &'a [Float],
}

/// One paired real-valued tick for statistic streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairTick {
    /// First real-valued input.
    pub real0: Float,
    /// Second real-valued input.
    pub real1: Float,
}
```

### crates/ta-core/src/statistic/moments.rs — NEW

Private validation, constants, univariate/paired rolling moments, and rolling regression engines.

```rust
//! Private rolling statistic engines.

use crate::{Float, Result, TalibError};

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec::Vec};
#[cfg(feature = "std")]
use std::{format, string::ToString, vec::Vec};

pub(super) const DEFAULT_NBDEV: Float = 1.0 as Float;
pub(super) const TA_EPSILON: Float = 1e-14 as Float;
const MAX_PERIOD: usize = 100_000;

pub(super) fn statistic_lookback(timeperiod: usize, minimum: usize, extra: usize) -> Result<usize> {
    if !(minimum..=MAX_PERIOD).contains(&timeperiod) {
        return Err(TalibError::invalid_period(
            timeperiod,
            format!("timeperiod must be in {minimum}..={MAX_PERIOD}"),
        ));
    }

    Ok((timeperiod - 1) + extra)
}

pub(super) fn validate_nbdev(nbdev: Float) -> Result<()> {
    if !nbdev.is_finite() {
        return Err(TalibError::invalid_parameter(
            "nbdev".to_string(),
            format!("{nbdev}"),
            "finite number".to_string(),
        ));
    }
    Ok(())
}

#[inline]
pub(super) fn is_ta_zero(value: Float) -> bool {
    value > -TA_EPSILON && value < TA_EPSILON
}

#[derive(Debug, Clone)]
pub(super) struct RollingMoments {
    period: usize,
    trailing: Vec<Float>,
    index: usize,
    count: usize,
    sum: Float,
    sum_sq: Float,
}

impl RollingMoments {
    pub(super) fn new(period: usize) -> Self {
        let mut trailing = Vec::new();
        trailing.resize(period.saturating_sub(1), 0.0 as Float);
        Self {
            period,
            trailing,
            index: 0,
            count: 0,
            sum: 0.0 as Float,
            sum_sq: 0.0 as Float,
        }
    }

    pub(super) fn push(&mut self, input: Float) -> Option<Float> {
        self.sum += input;
        self.sum_sq += input * input;
        if self.count < self.period {
            self.count += 1;
        }

        if self.count < self.period {
            self.trailing[self.index] = input;
            self.index = (self.index + 1) % self.trailing.len();
            return None;
        }

        let mean = self.sum / self.period as Float;
        let variance = self.sum_sq / self.period as Float - mean * mean;
        self.remove_trailing(input);
        Some(variance)
    }

    fn remove_trailing(&mut self, input: Float) {
        if self.trailing.is_empty() {
            self.sum -= input;
            self.sum_sq -= input * input;
            return;
        }

        let old = self.trailing[self.index];
        self.sum -= old;
        self.sum_sq -= old * old;
        self.trailing[self.index] = input;
        self.index = (self.index + 1) % self.trailing.len();
    }

    pub(super) fn reset(&mut self) {
        self.trailing.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum = 0.0 as Float;
        self.sum_sq = 0.0 as Float;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct PairedSnapshot {
    n: Float,
    sum_x: Float,
    sum_y: Float,
    sum_x_sq: Float,
    sum_y_sq: Float,
    sum_xy: Float,
}

impl PairedSnapshot {
    pub(super) fn correlation(self) -> Float {
        let centered_x = self.sum_x_sq - self.sum_x * self.sum_x / self.n;
        let centered_y = self.sum_y_sq - self.sum_y * self.sum_y / self.n;
        let denominator_sq = centered_x * centered_y;
        if denominator_sq < TA_EPSILON {
            0.0 as Float
        } else {
            (self.sum_xy - self.sum_x * self.sum_y / self.n) / denominator_sq.sqrt()
        }
    }

    pub(super) fn beta(self) -> Float {
        // TA_BETA applies TA_IS_ZERO directly to the period-scaled centered
        // real0 variance: n * sum(x²) - sum(x)².
        let scaled_variance_x = self.n * self.sum_x_sq - self.sum_x * self.sum_x;
        if is_ta_zero(scaled_variance_x) {
            0.0 as Float
        } else {
            (self.n * self.sum_xy - self.sum_x * self.sum_y) / scaled_variance_x
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct RollingPairedMoments {
    period: usize,
    trailing_x: Vec<Float>,
    trailing_y: Vec<Float>,
    index: usize,
    count: usize,
    sum_x: Float,
    sum_y: Float,
    sum_x_sq: Float,
    sum_y_sq: Float,
    sum_xy: Float,
}

impl RollingPairedMoments {
    pub(super) fn new(period: usize) -> Self {
        let mut trailing_x = Vec::new();
        trailing_x.resize(period.saturating_sub(1), 0.0 as Float);
        let mut trailing_y = Vec::new();
        trailing_y.resize(period.saturating_sub(1), 0.0 as Float);
        Self {
            period,
            trailing_x,
            trailing_y,
            index: 0,
            count: 0,
            sum_x: 0.0 as Float,
            sum_y: 0.0 as Float,
            sum_x_sq: 0.0 as Float,
            sum_y_sq: 0.0 as Float,
            sum_xy: 0.0 as Float,
        }
    }

    pub(super) fn push(&mut self, x: Float, y: Float) -> Option<PairedSnapshot> {
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x_sq += x * x;
        self.sum_y_sq += y * y;
        self.sum_xy += x * y;
        if self.count < self.period {
            self.count += 1;
        }

        if self.count < self.period {
            self.trailing_x[self.index] = x;
            self.trailing_y[self.index] = y;
            self.index = (self.index + 1) % self.trailing_x.len();
            return None;
        }

        let snapshot = PairedSnapshot {
            n: self.period as Float,
            sum_x: self.sum_x,
            sum_y: self.sum_y,
            sum_x_sq: self.sum_x_sq,
            sum_y_sq: self.sum_y_sq,
            sum_xy: self.sum_xy,
        };
        self.remove_trailing(x, y);
        Some(snapshot)
    }

    fn remove_trailing(&mut self, x: Float, y: Float) {
        if self.trailing_x.is_empty() {
            self.sum_x -= x;
            self.sum_y -= y;
            self.sum_x_sq -= x * x;
            self.sum_y_sq -= y * y;
            self.sum_xy -= x * y;
            return;
        }

        let old_x = self.trailing_x[self.index];
        let old_y = self.trailing_y[self.index];
        self.sum_x -= old_x;
        self.sum_y -= old_y;
        self.sum_x_sq -= old_x * old_x;
        self.sum_y_sq -= old_y * old_y;
        self.sum_xy -= old_x * old_y;
        self.trailing_x[self.index] = x;
        self.trailing_y[self.index] = y;
        self.index = (self.index + 1) % self.trailing_x.len();
    }

    pub(super) fn reset(&mut self) {
        self.trailing_x.fill(0.0 as Float);
        self.trailing_y.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum_x = 0.0 as Float;
        self.sum_y = 0.0 as Float;
        self.sum_x_sq = 0.0 as Float;
        self.sum_y_sq = 0.0 as Float;
        self.sum_xy = 0.0 as Float;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RegressionFit {
    pub(super) slope: Float,
    pub(super) intercept: Float,
}

#[derive(Debug, Clone)]
pub(super) struct RollingRegression {
    period: usize,
    n: Float,
    sum_x: Float,
    divisor: Float,
    buffer: Vec<Float>,
    index: usize,
    count: usize,
    sum_y: Float,
    sum_xy: Float,
}

impl RollingRegression {
    pub(super) fn new(period: usize) -> Self {
        let n = period as Float;
        let sum_x = n * (n - 1.0 as Float) * 0.5 as Float;
        let sum_x_sq = n * (n - 1.0 as Float) * (2.0 as Float * n - 1.0 as Float) / 6.0 as Float;
        let mut buffer = Vec::new();
        buffer.resize(period, 0.0 as Float);
        Self {
            period,
            n,
            sum_x,
            divisor: sum_x * sum_x - n * sum_x_sq,
            buffer,
            index: 0,
            count: 0,
            sum_y: 0.0 as Float,
            sum_xy: 0.0 as Float,
        }
    }

    pub(super) fn push(&mut self, input: Float) -> Option<RegressionFit> {
        if self.count < self.period {
            self.buffer[self.index] = input;
            self.sum_y += input;
            self.sum_xy += (self.period - 1 - self.count) as Float * input;
            self.count += 1;
            self.index = (self.index + 1) % self.period;

            if self.count < self.period {
                return None;
            }
            return Some(self.fit());
        }

        let trailing = self.buffer[self.index];
        self.sum_xy = self.sum_xy + self.sum_y - self.n * trailing;
        self.sum_y = self.sum_y - trailing + input;
        self.buffer[self.index] = input;
        self.index = (self.index + 1) % self.period;
        Some(self.fit())
    }

    fn fit(&self) -> RegressionFit {
        let slope = (self.n * self.sum_xy - self.sum_x * self.sum_y) / self.divisor;
        let intercept = (self.sum_y - slope * self.sum_x) / self.n;
        RegressionFit { slope, intercept }
    }

    pub(super) fn reset(&mut self) {
        self.buffer.fill(0.0 as Float);
        self.index = 0;
        self.count = 0;
        self.sum_y = 0.0 as Float;
        self.sum_xy = 0.0 as Float;
    }
}

#[cfg(test)]
mod tests {
    use super::{is_ta_zero, Float, TA_EPSILON};

    #[test]
    fn ta_zero_uses_strict_endpoints() {
        assert!(is_ta_zero(0.0 as Float));
        assert!(is_ta_zero(TA_EPSILON * 0.5 as Float));
        assert!(is_ta_zero(-TA_EPSILON * 0.5 as Float));
        assert!(!is_ta_zero(TA_EPSILON));
        assert!(!is_ta_zero(-TA_EPSILON));
    }
}
```

### crates/ta-core/src/lib.rs:32-50 — MODIFY

Register the new official `statistic` crate-root module.

```rust
/// Statistic Functions.
pub mod statistic;
```

### crates/ta-core/src/statistic/variance.rs — NEW

VAR and STDDEV compact/default/vec/struct/streaming/reset surfaces.

```rust
//! Variance (VAR) and Standard Deviation (STDDEV).

use super::moments::{
    statistic_lookback, validate_nbdev, RollingMoments, DEFAULT_NBDEV, TA_EPSILON,
};
use crate::{
    compact_buffer, padded_from_compact, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[derive(Debug, Clone, Copy)]
enum VarianceProjection {
    Variance,
    StandardDeviation,
}

#[inline]
fn project(variance: Float, nbdev: Float, projection: VarianceProjection) -> Float {
    match projection {
        VarianceProjection::Variance => variance,
        VarianceProjection::StandardDeviation if variance <= TA_EPSILON => 0.0 as Float,
        VarianceProjection::StandardDeviation => variance.sqrt() * nbdev,
    }
}

fn variance_batch(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    nbdev: Float,
    minimum_period: usize,
    projection: VarianceProjection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, minimum_period, 0)?;
    validate_nbdev(nbdev)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut moments = RollingMoments::new(timeperiod);
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(variance) = moments.push(value) {
            out_real[output_idx] = project(variance, nbdev, projection);
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

macro_rules! define_variance_indicator {
    (
        $name:ident,
        $vec_name:ident,
        $default_name:ident,
        $default_vec_name:ident,
        $minimum_period:expr,
        $projection:expr,
        $description:literal
    ) => {
        #[doc = concat!("TA-Lib-style ", $description, " batch function.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real: &[Float],
            timeperiod: usize,
            nbdev: Float,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            variance_batch(
                stringify!($name),
                real,
                timeperiod,
                nbdev,
                $minimum_period,
                $projection,
                out_real,
            )
        }

        #[doc = concat!("TA-Lib-style ", $description, " with default nbdev 1.0.")]
        #[allow(non_snake_case)]
        pub fn $default_name(
            real: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            $name(real, timeperiod, DEFAULT_NBDEV, out_real)
        }

        #[doc = concat!("Computes ", $description, " into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real: &[Float], timeperiod: usize, nbdev: Float) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real.len());
            let range = $name(real, timeperiod, nbdev, &mut compact)?;
            Ok(padded_from_compact(
                real.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!("Computes ", $description, " with default nbdev 1.0 into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $default_vec_name(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
            $vec_name(real, timeperiod, DEFAULT_NBDEV)
        }

        #[doc = concat!($description, " indicator.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            nbdev: Float,
            moments: RollingMoments,
        }

        impl $name {
            #[doc = concat!("Creates a new ", $description, " indicator.")]
            pub fn new(timeperiod: usize, nbdev: Float) -> Result<Self> {
                statistic_lookback(timeperiod, $minimum_period, 0)?;
                validate_nbdev(nbdev)?;
                Ok(Self {
                    period: timeperiod,
                    nbdev,
                    moments: RollingMoments::new(timeperiod),
                })
            }

            #[doc = concat!("Creates a new ", $description, " indicator with nbdev 1.0.")]
            pub fn with_default_nbdev(timeperiod: usize) -> Result<Self> {
                Self::new(timeperiod, DEFAULT_NBDEV)
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Returns the configured deviation multiplier.
            pub const fn nbdev(&self) -> Float {
                self.nbdev
            }

            /// Computes compact outputs using this indicator's configuration.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, self.nbdev, out_real)
            }

            /// Computes full-length padded outputs using this indicator's configuration.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period, self.nbdev)
            }

            /// Checked streaming update that returns `Float::NAN` during warm-up.
            pub fn next_checked(&mut self, input: Float) -> Result<Float> {
                Ok(self.next(input)?.unwrap_or(Float::NAN))
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [Float];
            type OutputOwned = Vec<Float>;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input, self.period, self.nbdev, output)
            }

            fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(input, self.period, self.nbdev)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                Ok(self
                    .moments
                    .push(input)
                    .map(|variance| project(variance, self.nbdev, $projection)))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                self.moments.reset();
            }
        }
    };
}

define_variance_indicator!(
    VAR,
    VAR_vec,
    VAR_with_default_nbdev,
    VAR_vec_with_default_nbdev,
    1,
    VarianceProjection::Variance,
    "Variance"
);
define_variance_indicator!(
    STDDEV,
    STDDEV_vec,
    STDDEV_with_default_nbdev,
    STDDEV_vec_with_default_nbdev,
    2,
    VarianceProjection::StandardDeviation,
    "Standard Deviation"
);
```

### crates/ta-core/tests/statistic.rs — NEW

Numerical, protocol, validation, streaming, reset, precision, and regression tests for the group.

```rust
use ta_core::statistic::{
    BETA_vec, CORREL_vec, LINEARREG_ANGLE_vec, LINEARREG_INTERCEPT_vec, LINEARREG_SLOPE_vec,
    LINEARREG_vec, PairInput, PairTick, STDDEV_vec, STDDEV_vec_with_default_nbdev, TSF_vec,
    VAR_vec, VAR_vec_with_default_nbdev, VAR_with_default_nbdev, BETA, CORREL, LINEARREG,
    LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator, TalibError};

#[cfg(feature = "f32")]
const ABS_TOLERANCE: Float = 1e-4;
#[cfg(not(feature = "f32"))]
const ABS_TOLERANCE: Float = 1e-12;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 1e-4;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 1e-10;

fn assert_close(actual: Float, expected: Float) {
    let tolerance = ABS_TOLERANCE + REL_TOLERANCE * Float::max(actual.abs(), expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

fn assert_vec_close_with_nans(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN at {idx}, got {actual}");
        } else {
            assert_close(actual, expected);
        }
    }
}

#[test]
fn var_and_stddev_match_population_and_nbdev_semantics() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut variance = [0.0; 3];
    let mut ignored_nbdev = [0.0; 3];
    let mut stddev = [0.0; 3];

    let var_range = VAR(&real, 3, 1.0, &mut variance).unwrap();
    let ignored_range = VAR(&real, 3, 7.0, &mut ignored_nbdev).unwrap();
    let stddev_range = STDDEV(&real, 3, 2.0, &mut stddev).unwrap();

    assert_eq!(var_range, OutputRange::new(2, 3));
    assert_eq!(ignored_range, var_range);
    assert_eq!(stddev_range, var_range);
    for idx in 0..3 {
        assert_close(variance[idx], 2.0 as Float / 3.0 as Float);
        assert_eq!(ignored_nbdev[idx].to_bits(), variance[idx].to_bits());
        assert_close(
            stddev[idx],
            (2.0 as Float / 3.0 as Float).sqrt() * 2.0 as Float,
        );
    }
}

#[test]
fn variance_vec_defaults_and_indicator_surfaces_preserve_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let explicit = VAR_vec(&real, 3, 1.0).unwrap();
    let defaulted = VAR_vec_with_default_nbdev(&real, 3).unwrap();
    assert_vec_close_with_nans(&defaulted, &explicit);
    assert!(explicit[..2].iter().all(|value| value.is_nan()));
    assert_close(explicit[2], 2.0 as Float / 3.0 as Float);

    let stddev = STDDEV_vec_with_default_nbdev(&real, 3).unwrap();
    assert!(stddev[..2].iter().all(|value| value.is_nan()));
    assert_close(stddev[2], (2.0 as Float / 3.0 as Float).sqrt());

    let indicator = VAR::with_default_nbdev(3).unwrap();
    let mut compact = [0.0; 2];
    let range = Indicator::compute(&indicator, &real, &mut compact).unwrap();
    assert_eq!(indicator.period(), 3);
    assert_close(indicator.nbdev(), 1.0);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], 2.0 as Float / 3.0 as Float);

    let stddev_indicator = STDDEV::new(3, 2.0).unwrap();
    assert_close(stddev_indicator.nbdev(), 2.0);
    let via_trait = Indicator::compute_to_vec(&stddev_indicator, &real).unwrap();
    assert_vec_close_with_nans(&via_trait, &STDDEV_vec(&real, 3, 2.0).unwrap());
}

#[test]
fn variance_period_one_and_degenerate_windows_are_valid() {
    let real = [2.0, 4.0, 8.0];
    let mut output = [1.0; 3];
    let range = VAR_with_default_nbdev(&real, 1, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(0, 3));
    assert!(output.iter().all(|&value| value == 0.0 as Float));

    let constant = [7.0; 4];
    let stddev = STDDEV_vec(&constant, 2, -3.0).unwrap();
    assert!(stddev[0].is_nan());
    assert!(stddev[1..].iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn variance_preserves_selected_cancellation_behavior() {
    let real = [1_000_000.0, 1_000_001.0, 1_000_002.0];
    let output = VAR_vec_with_default_nbdev(&real, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[2], -65_536.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[2], 0.6666259765625 as Float);
}

#[test]
fn variance_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        VAR::new(0, 1.0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(matches!(
        STDDEV::new(1, 1.0),
        Err(TalibError::InvalidPeriod { period: 1, .. })
    ));
    assert!(VAR::new(100_000, 1.0).is_ok());
    assert!(STDDEV::new(100_000, 1.0).is_ok());
    assert!(matches!(
        VAR::new(100_001, 1.0),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    for nbdev in [Float::NAN, Float::INFINITY, Float::NEG_INFINITY] {
        assert!(matches!(
            VAR::new(3, nbdev),
            Err(TalibError::InvalidParameter { .. })
        ));
    }

    assert_eq!(VAR(&[], 1, 1.0, &mut []).unwrap(), OutputRange::empty());
    assert_eq!(STDDEV(&[], 2, 1.0, &mut []).unwrap(), OutputRange::empty());
    assert!(matches!(
        STDDEV(&[1.0], 2, 1.0, &mut []),
        Err(TalibError::InsufficientData {
            required: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        STDDEV(&[Float::NAN], 2, 1.0, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));

    let mut too_small = [123.0];
    assert!(matches!(
        VAR(&[1.0, 2.0, 3.0], 2, 1.0, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [123.0]);
}

#[test]
fn variance_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
    let mut batch_var = [0.0; 6];
    let var_range = VAR(&real, 3, 1.0, &mut batch_var).unwrap();
    let mut streaming_var = VAR::with_default_nbdev(3).unwrap();

    for (idx, &value) in real.iter().enumerate() {
        let streamed = streaming_var.next(value).unwrap();
        if idx < var_range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_eq!(
                streamed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }

    streaming_var.reset();
    for (idx, &value) in real.iter().enumerate() {
        let replayed = streaming_var.next(value).unwrap();
        if idx < var_range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }
    streaming_var.reset();
    assert!(streaming_var.next_checked(real[0]).unwrap().is_nan());

    let mut dirty = VAR::with_default_nbdev(3).unwrap();
    let mut clean = VAR::with_default_nbdev(3).unwrap();
    assert!(dirty.next(1.0).unwrap().is_none());
    assert!(clean.next(1.0).unwrap().is_none());
    assert!(dirty.next(Float::NAN).is_err());
    assert!(dirty.next(2.0).unwrap().is_none());
    assert!(clean.next(2.0).unwrap().is_none());
    assert_eq!(
        dirty.next(3.0).unwrap().unwrap().to_bits(),
        clean.next(3.0).unwrap().unwrap().to_bits()
    );

    let mut batch_stddev = [0.0; 6];
    let stddev_range = STDDEV(&real, 3, 2.0, &mut batch_stddev).unwrap();
    let mut streaming_stddev = STDDEV::new(3, 2.0).unwrap();
    for (idx, &value) in real.iter().enumerate() {
        let streamed = streaming_stddev.next(value).unwrap();
        if idx < stddev_range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_eq!(
                streamed.unwrap().to_bits(),
                batch_stddev[idx - stddev_range.beg_idx].to_bits()
            );
        }
    }
}

#[test]
fn correl_matches_positive_negative_constant_and_period_one_semantics() {
    let real0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let positive = [2.0, 4.0, 6.0, 8.0, 10.0];
    let negative = [10.0, 8.0, 6.0, 4.0, 2.0];
    let constant = [7.0; 5];
    let mut output = [0.0; 5];

    let range = CORREL(&real0, &positive, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    for &value in &output[..range.nb_element] {
        assert_close(value, 1.0);
    }

    CORREL(&real0, &negative, 3, &mut output).unwrap();
    for &value in &output[..3] {
        assert_close(value, -1.0);
    }

    CORREL(&real0, &constant, 3, &mut output).unwrap();
    assert!(output[..3].iter().all(|&value| value == 0.0 as Float));

    let range = CORREL(&real0, &positive, 1, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(0, 5));
    assert!(output.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn correl_vec_and_indicator_surfaces_preserve_pair_alignment() {
    let real0 = [1.0, 2.0, 4.0, 8.0];
    let real1 = [2.0, 3.0, 5.0, 9.0];
    let padded = CORREL_vec(&real0, &real1, 3).unwrap();
    assert_eq!(padded.len(), real0.len());
    assert!(padded[..2].iter().all(|value| value.is_nan()));
    assert!(padded[2..].iter().all(|value| value.is_finite()));

    let indicator = CORREL::new(3).unwrap();
    let mut compact = [0.0; 3];
    let range = Indicator::compute(
        &indicator,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
        &mut compact,
    )
    .unwrap();
    assert_eq!(indicator.period(), 3);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], padded[2]);
    assert_close(compact[1], padded[3]);
}

#[test]
fn correl_preserves_selected_cancellation_behavior() {
    let real0 = [100_000.0, 100_001.0, 100_002.0];
    let real1 = [200_000.0, 200_002.0, 200_004.0];
    let output = CORREL_vec(&real0, &real1, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[2], -1.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[2], 1.0 as Float);
}

#[test]
fn correl_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        CORREL::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(CORREL::new(100_000).is_ok());
    assert!(matches!(
        CORREL::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(CORREL(&[], &[], 1, &mut []).unwrap(), OutputRange::empty());

    let period_first = CORREL(&[Float::NAN], &[], 0, &mut []).unwrap_err();
    assert!(matches!(
        period_first,
        TalibError::InvalidPeriod { period: 0, .. }
    ));
    let length_first = CORREL(&[Float::NAN], &[], 1, &mut []).unwrap_err();
    assert!(length_first
        .to_string()
        .contains("must have the same length"));
    let finite_first = CORREL(&[Float::NAN], &[1.0], 3, &mut []).unwrap_err();
    assert!(finite_first.to_string().contains("must be finite"));
    assert!(matches!(
        CORREL(&[Float::INFINITY], &[1.0], 1, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        CORREL(&[1.0, 2.0], &[1.0, 2.0], 3, &mut []),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));

    let mut too_small = [321.0];
    assert!(matches!(
        CORREL(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], 2, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [321.0]);
}

#[test]
fn correl_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real0 = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
    let real1 = [2.0, 1.0, 5.0, 3.0, 8.0, 4.0, 9.0, 6.0];
    let mut batch = [0.0; 6];
    let range = CORREL(&real0, &real1, 3, &mut batch).unwrap();
    let mut streaming = CORREL::new(3).unwrap();

    for idx in 0..real0.len() {
        let value = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(value.is_none());
        } else {
            assert_eq!(
                value.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }

    streaming.reset();
    for idx in 0..real0.len() {
        let replayed = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }
    streaming.reset();
    assert!(streaming
        .next_checked(PairTick {
            real0: real0[0],
            real1: real1[0],
        })
        .unwrap()
        .is_nan());

    let mut dirty = CORREL::new(2).unwrap();
    let mut clean = CORREL::new(2).unwrap();
    let first = PairTick {
        real0: 1.0,
        real1: 2.0,
    };
    assert!(dirty.next(first).unwrap().is_none());
    assert!(clean.next(first).unwrap().is_none());
    assert!(dirty
        .next(PairTick {
            real0: 3.0,
            real1: Float::NAN,
        })
        .is_err());
    let second = PairTick {
        real0: 2.0,
        real1: 4.0,
    };
    assert_eq!(
        dirty.next(second).unwrap().unwrap().to_bits(),
        clean.next(second).unwrap().unwrap().to_bits()
    );
}

#[test]
fn beta_uses_returns_real0_denominator_and_ta_zero_boundaries() {
    let market = [100.0, 110.0, 132.0, 118.8];
    let asset = [50.0, 60.0, 84.0, 67.2];
    let mut output = [0.0; 1];

    let range = BETA(&market, &asset, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(3, 1));
    assert_close(output[0], 2.0);

    BETA(&asset, &market, 3, &mut output).unwrap();
    assert_close(output[0], 0.5);

    let interior_near_zero = [5e-15, 10.0, 20.0];
    let comparison = [5.0, 6.0, 8.0];
    BETA(&interior_near_zero, &comparison, 2, &mut output).unwrap();
    assert_close(output[0], 2.0 as Float / 15.0 as Float);

    let positive_endpoint = [1e-14, 10.0, 20.0];
    BETA(&positive_endpoint, &comparison, 2, &mut output).unwrap();
    assert!(output[0].is_sign_negative() && output[0] != 0.0 as Float);

    let negative_endpoint = [-1e-14, 10.0, 20.0];
    BETA(&negative_endpoint, &comparison, 2, &mut output).unwrap();
    assert!(output[0].is_sign_positive() && output[0] != 0.0 as Float);
}

#[test]
fn beta_vec_indicator_and_period_one_use_extra_lookback() {
    let real0 = [10.0, 11.0, 13.0, 12.0];
    let real1 = [20.0, 21.0, 24.0, 22.0];
    let padded = BETA_vec(&real0, &real1, 2).unwrap();
    assert_eq!(padded.len(), real0.len());
    assert!(padded[..2].iter().all(|value| value.is_nan()));
    assert!(padded[2..].iter().all(|value| value.is_finite()));

    let indicator = BETA::new(2).unwrap();
    let mut compact = [0.0; 2];
    let range = Indicator::compute(
        &indicator,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
        &mut compact,
    )
    .unwrap();
    assert_eq!(indicator.period(), 2);
    assert_eq!(indicator.lookback(), 2);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], padded[2]);
    assert_close(compact[1], padded[3]);

    let mut period_one = [1.0; 3];
    let range = BETA(&real0, &real1, 1, &mut period_one).unwrap();
    assert_eq!(range, OutputRange::new(1, 3));
    assert!(period_one.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn beta_preserves_selected_cancellation_behavior() {
    let real0 = [20_000_000.0, 20_000_001.0, 20_000_003.0, 20_000_002.0];
    let real1 = [40_000_000.0, 40_000_004.0, 40_000_012.0, 40_000_008.0];
    let output = BETA_vec(&real0, &real1, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[3], 0.9285713 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[3], 1.999999835714329 as Float);
}

#[test]
fn beta_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        BETA::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(BETA::new(100_000).is_ok());
    assert!(matches!(
        BETA::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(BETA(&[], &[], 1, &mut []).unwrap(), OutputRange::empty());

    let period_first = BETA(&[Float::NAN], &[], 0, &mut []).unwrap_err();
    assert!(matches!(
        period_first,
        TalibError::InvalidPeriod { period: 0, .. }
    ));
    let length_first = BETA(&[Float::NAN], &[], 1, &mut []).unwrap_err();
    assert!(length_first
        .to_string()
        .contains("must have the same length"));
    let finite_first = BETA(&[Float::NAN], &[1.0], 2, &mut []).unwrap_err();
    assert!(finite_first.to_string().contains("must be finite"));
    assert!(matches!(
        BETA(&[Float::INFINITY, 2.0], &[1.0, 2.0], 1, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        BETA(&[1.0, 2.0], &[1.0, 2.0], 2, &mut []),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));

    let mut too_small = [456.0];
    assert!(matches!(
        BETA(
            &[1.0, 2.0, 3.0, 4.0],
            &[2.0, 3.0, 5.0, 8.0],
            2,
            &mut too_small
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [456.0]);
}

#[test]
fn beta_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real0 = [10.0, 11.0, 13.0, 12.0, 15.0, 14.0, 18.0, 17.0];
    let real1 = [20.0, 22.0, 23.0, 21.0, 26.0, 24.0, 29.0, 28.0];
    let mut batch = [0.0; 6];
    let range = BETA(&real0, &real1, 2, &mut batch).unwrap();
    let mut streaming = BETA::new(2).unwrap();

    for idx in 0..real0.len() {
        let value = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(value.is_none());
        } else {
            assert_eq!(
                value.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }

    streaming.reset();
    for idx in 0..real0.len() {
        let replayed = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }
    streaming.reset();
    assert!(streaming
        .next_checked(PairTick {
            real0: real0[0],
            real1: real1[0],
        })
        .unwrap()
        .is_nan());

    let mut dirty = BETA::new(2).unwrap();
    let mut clean = BETA::new(2).unwrap();
    let first = PairTick {
        real0: 10.0,
        real1: 20.0,
    };
    assert!(dirty.next(first).unwrap().is_none());
    assert!(clean.next(first).unwrap().is_none());
    assert!(dirty
        .next(PairTick {
            real0: Float::NAN,
            real1: 21.0,
        })
        .is_err());
    for tick in [
        PairTick {
            real0: 11.0,
            real1: 22.0,
        },
        PairTick {
            real0: 13.0,
            real1: 23.0,
        },
    ] {
        let dirty_value = dirty.next(tick).unwrap();
        let clean_value = clean.next(tick).unwrap();
        match (dirty_value, clean_value) {
            (Some(dirty_value), Some(clean_value)) => {
                assert_eq!(dirty_value.to_bits(), clean_value.to_bits());
            }
            (None, None) => {}
            pair => panic!("streaming state diverged after invalid tick: {pair:?}"),
        }
    }
}

#[test]
fn regression_family_matches_closed_form_projections() {
    let real = [1.0, 2.0, 3.0];
    let mut output = [0.0; 1];

    assert_eq!(
        LINEARREG(&real, 3, &mut output).unwrap(),
        OutputRange::new(2, 1)
    );
    assert_close(output[0], 3.0);
    LINEARREG_SLOPE(&real, 3, &mut output).unwrap();
    assert_close(output[0], 1.0);
    LINEARREG_INTERCEPT(&real, 3, &mut output).unwrap();
    assert_close(output[0], 1.0);
    LINEARREG_ANGLE(&real, 3, &mut output).unwrap();
    assert_close(output[0], 45.0);
    TSF(&real, 3, &mut output).unwrap();
    assert_close(output[0], 4.0);
}

#[test]
fn regression_vec_struct_and_constant_surfaces_preserve_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    for output in [
        LINEARREG_vec(&real, 3).unwrap(),
        LINEARREG_SLOPE_vec(&real, 3).unwrap(),
        LINEARREG_INTERCEPT_vec(&real, 3).unwrap(),
        LINEARREG_ANGLE_vec(&real, 3).unwrap(),
        TSF_vec(&real, 3).unwrap(),
    ] {
        assert_eq!(output.len(), real.len());
        assert!(output[..2].iter().all(|value| value.is_nan()));
        assert!(output[2..].iter().all(|value| value.is_finite()));
    }

    let indicator = LINEARREG::new(3).unwrap();
    let mut compact = [0.0; 3];
    let range = Indicator::compute(&indicator, &real, &mut compact).unwrap();
    assert_eq!(indicator.period(), 3);
    assert_eq!(range, OutputRange::new(2, 3));
    for (&compact, &padded) in compact.iter().zip(&LINEARREG_vec(&real, 3).unwrap()[2..]) {
        assert_close(compact, padded);
    }

    let constant = [7.0; 5];
    let endpoint = LINEARREG_vec(&constant, 3).unwrap();
    let slope = LINEARREG_SLOPE_vec(&constant, 3).unwrap();
    let intercept = LINEARREG_INTERCEPT_vec(&constant, 3).unwrap();
    let angle = LINEARREG_ANGLE_vec(&constant, 3).unwrap();
    let forecast = TSF_vec(&constant, 3).unwrap();
    for idx in 2..constant.len() {
        assert_close(endpoint[idx], 7.0);
        assert_close(slope[idx], 0.0);
        assert_close(intercept[idx], 7.0);
        assert_close(angle[idx], 0.0);
        assert_close(forecast[idx], 7.0);
    }
}

#[test]
fn regression_matches_pinned_rolling_fma_oracle() {
    let real = [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0];
    let endpoint = LINEARREG_vec(&real, 4).unwrap();
    let slope = LINEARREG_SLOPE_vec(&real, 4).unwrap();
    let intercept = LINEARREG_INTERCEPT_vec(&real, 4).unwrap();
    let angle = LINEARREG_ANGLE_vec(&real, 4).unwrap();
    let forecast = TSF_vec(&real, 4).unwrap();

    let expected_endpoint = [14.1, 14.5, 17.5, 17.5, 19.8];
    let expected_slope = [1.4, 1.0, 2.0, 1.0, 1.7];
    let expected_intercept = [9.9, 11.5, 11.5, 14.5, 14.7];
    let expected_angle = [
        54.46232220802562,
        45.0,
        63.43494882292202,
        45.0,
        59.53445508054013,
    ];
    let expected_forecast = [15.5, 15.5, 19.5, 18.5, 21.5];

    for idx in 0..5 {
        let output_idx = idx + 3;
        assert_close(endpoint[output_idx], expected_endpoint[idx]);
        assert_close(slope[output_idx], expected_slope[idx]);
        assert_close(intercept[output_idx], expected_intercept[idx]);
        assert_close(angle[output_idx], expected_angle[idx]);
        assert_close(forecast[output_idx], expected_forecast[idx]);
    }

    #[cfg(feature = "f32")]
    let expected_endpoint_bits = [
        0x4161_9999_u32,
        0x4168_0000,
        0x418c_0000,
        0x418c_0000,
        0x419e_6666,
    ];
    #[cfg(not(feature = "f32"))]
    let expected_endpoint_bits = [
        0x402c_3333_3333_3333_u64,
        0x402d_0000_0000_0000,
        0x4031_8000_0000_0000,
        0x4031_8000_0000_0000,
        0x4033_cccc_cccc_cccd,
    ];
    for (idx, &expected_bits) in expected_endpoint_bits.iter().enumerate() {
        assert_eq!(endpoint[idx + 3].to_bits(), expected_bits);
    }
}

#[test]
fn regression_preserves_large_baseline_cancellation_behavior() {
    let real = [
        10_000_000.0,
        10_000_001.0,
        10_000_002.0,
        10_000_003.0,
        10_000_004.0,
        10_000_005.0,
    ];
    let slope = LINEARREG_SLOPE_vec(&real, 3).unwrap();

    #[cfg(feature = "f32")]
    let expected_bits = [0x3faa_aaab_u32, 0x402a_aaab, 0x4080_0000, 0x4080_0000];
    #[cfg(not(feature = "f32"))]
    let expected_bits = [
        0x3ff0_0000_0000_0000_u64,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
    ];
    for (idx, &expected_bits) in expected_bits.iter().enumerate() {
        assert_eq!(slope[idx + 2].to_bits(), expected_bits);
    }
}

#[test]
fn regression_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        LINEARREG::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_SLOPE::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_INTERCEPT::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_ANGLE::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(TSF::new(1), Err(TalibError::InvalidPeriod { .. })));
    assert!(LINEARREG::new(100_000).is_ok());
    assert!(matches!(
        TSF::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(LINEARREG(&[], 2, &mut []).unwrap(), OutputRange::empty());

    let period_first = LINEARREG(&[Float::NAN], 1, &mut []).unwrap_err();
    assert!(matches!(period_first, TalibError::InvalidPeriod { .. }));
    assert!(matches!(
        LINEARREG(&[Float::INFINITY], 2, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        LINEARREG(&[1.0], 2, &mut []),
        Err(TalibError::InsufficientData {
            required: 2,
            actual: 1
        })
    ));

    let mut too_small = [789.0];
    assert!(matches!(
        LINEARREG(&[1.0, 2.0, 3.0], 2, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [789.0]);
}

#[test]
fn regression_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];

    macro_rules! assert_parity {
        ($function:ident, $indicator:ident) => {{
            let mut batch = [0.0; 6];
            let range = $function(&real, 3, &mut batch).unwrap();
            let mut streaming = $indicator::new(3).unwrap();
            for (idx, &value) in real.iter().enumerate() {
                let streamed = streaming.next(value).unwrap();
                if idx < range.beg_idx {
                    assert!(streamed.is_none());
                } else {
                    assert_eq!(
                        streamed.unwrap().to_bits(),
                        batch[idx - range.beg_idx].to_bits()
                    );
                }
            }
            streaming.reset();
            for (idx, &value) in real.iter().enumerate() {
                let replayed = streaming.next(value).unwrap();
                if idx < range.beg_idx {
                    assert!(replayed.is_none());
                } else {
                    assert_eq!(
                        replayed.unwrap().to_bits(),
                        batch[idx - range.beg_idx].to_bits()
                    );
                }
            }
        }};
    }

    assert_parity!(LINEARREG, LINEARREG);
    assert_parity!(LINEARREG_SLOPE, LINEARREG_SLOPE);
    assert_parity!(LINEARREG_INTERCEPT, LINEARREG_INTERCEPT);
    assert_parity!(LINEARREG_ANGLE, LINEARREG_ANGLE);
    assert_parity!(TSF, TSF);

    let mut dirty = LINEARREG::new(2).unwrap();
    let mut clean = LINEARREG::new(2).unwrap();
    assert!(dirty.next(1.0).unwrap().is_none());
    assert!(clean.next(1.0).unwrap().is_none());
    assert!(dirty.next(Float::NAN).is_err());
    assert_eq!(
        dirty.next(2.0).unwrap().unwrap().to_bits(),
        clean.next(2.0).unwrap().unwrap().to_bits()
    );
    dirty.reset();
    assert!(dirty.next_checked(1.0).unwrap().is_nan());
}
```

### crates/ta-core/src/statistic/correl.rs — NEW

CORREL paired rolling implementation and public indicator surfaces.

```rust
//! Pearson's Correlation Coefficient (CORREL).

use super::{
    moments::{statistic_lookback, RollingPairedMoments},
    PairInput, PairTick,
};
use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// TA-Lib-style Pearson's Correlation Coefficient batch function.
#[allow(non_snake_case)]
pub fn CORREL(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 1, 0)?;
    let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
    validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("CORREL", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut moments = RollingPairedMoments::new(timeperiod);
    let mut output_idx = 0usize;
    for (&real0, &real1) in real0.iter().zip(real1) {
        if let Some(snapshot) = moments.push(real0, real1) {
            out_real[output_idx] = snapshot.correlation();
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Pearson's Correlation Coefficient into a full-length padded vector.
#[allow(non_snake_case)]
pub fn CORREL_vec(real0: &[Float], real1: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real0.len());
    let range = CORREL(real0, real1, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real0.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Pearson's Correlation Coefficient indicator.
#[derive(Debug, Clone)]
pub struct CORREL {
    period: usize,
    moments: RollingPairedMoments,
}

impl CORREL {
    /// Creates a new Pearson's Correlation Coefficient indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 0)?;
        Ok(Self {
            period: timeperiod,
            moments: RollingPairedMoments::new(timeperiod),
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs using this indicator's period.
    pub fn compute(
        &self,
        real0: &[Float],
        real1: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        CORREL(real0, real1, self.period, out_real)
    }

    /// Computes full-length padded outputs using this indicator's period.
    pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
        CORREL_vec(real0, real1, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: PairTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for CORREL {
    type Input<'a> = PairInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        self.period - 1
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        CORREL(input.real0, input.real1, self.period, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        CORREL_vec(input.real0, input.real1, self.period)
    }
}

impl StreamingIndicator for CORREL {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_slices(&[("real0", &[input.real0]), ("real1", &[input.real1])])?;
        Ok(self
            .moments
            .push(input.real0, input.real1)
            .map(|snapshot| snapshot.correlation()))
    }
}

impl Resettable for CORREL {
    fn reset(&mut self) {
        self.moments.reset();
    }
}
```

### crates/ta-core/src/statistic/beta.rs — NEW

BETA adjacent-return transformation, extra warm-up, and public indicator surfaces.

```rust
//! Beta Coefficient (BETA).

use super::{
    moments::{is_ta_zero, statistic_lookback, RollingPairedMoments},
    PairInput, PairTick,
};
use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_input_len, validate_output_len, Float, Indicator, OutputRange, Resettable, Result,
    StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[inline]
fn price_return(current: Float, previous: Float) -> Float {
    if is_ta_zero(previous) {
        0.0 as Float
    } else {
        (current - previous) / previous
    }
}

#[derive(Debug, Clone)]
struct BetaState {
    previous: Option<PairTick>,
    moments: RollingPairedMoments,
}

impl BetaState {
    fn new(period: usize) -> Self {
        Self {
            previous: None,
            moments: RollingPairedMoments::new(period),
        }
    }

    fn push(&mut self, input: PairTick) -> Option<Float> {
        let Some(previous) = self.previous.replace(input) else {
            return None;
        };
        let real0_return = price_return(input.real0, previous.real0);
        let real1_return = price_return(input.real1, previous.real1);
        self.moments
            .push(real0_return, real1_return)
            .map(|snapshot| snapshot.beta())
    }

    fn reset(&mut self) {
        self.previous = None;
        self.moments.reset();
    }
}

/// TA-Lib-style Beta Coefficient batch function.
#[allow(non_snake_case)]
pub fn BETA(
    real0: &[Float],
    real1: &[Float],
    timeperiod: usize,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 1, 1)?;
    let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
    validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
    let count = validate_input_len(len, lookback)?;
    validate_output_len("BETA", out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut state = BetaState::new(timeperiod);
    let mut output_idx = 0usize;
    for (&real0, &real1) in real0.iter().zip(real1) {
        if let Some(value) = state.push(PairTick { real0, real1 }) {
            out_real[output_idx] = value;
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

/// Computes Beta Coefficient into a full-length padded vector.
#[allow(non_snake_case)]
pub fn BETA_vec(real0: &[Float], real1: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
    let mut compact = compact_buffer::<Float>(real0.len());
    let range = BETA(real0, real1, timeperiod, &mut compact)?;
    Ok(padded_from_compact(
        real0.len(),
        range,
        &compact[..range.nb_element],
    ))
}

/// Beta Coefficient indicator.
#[derive(Debug, Clone)]
pub struct BETA {
    period: usize,
    state: BetaState,
}

impl BETA {
    /// Creates a new Beta Coefficient indicator.
    pub fn new(timeperiod: usize) -> Result<Self> {
        statistic_lookback(timeperiod, 1, 1)?;
        Ok(Self {
            period: timeperiod,
            state: BetaState::new(timeperiod),
        })
    }

    /// Returns the configured period.
    pub const fn period(&self) -> usize {
        self.period
    }

    /// Computes compact outputs using this indicator's period.
    pub fn compute(
        &self,
        real0: &[Float],
        real1: &[Float],
        out_real: &mut [Float],
    ) -> Result<OutputRange> {
        BETA(real0, real1, self.period, out_real)
    }

    /// Computes full-length padded outputs using this indicator's period.
    pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
        BETA_vec(real0, real1, self.period)
    }

    /// Checked streaming update that returns `Float::NAN` during warm-up.
    pub fn next_checked(&mut self, input: PairTick) -> Result<Float> {
        Ok(self.next(input)?.unwrap_or(Float::NAN))
    }
}

impl Indicator for BETA {
    type Input<'a> = PairInput<'a>;
    type OutputMut<'a> = &'a mut [Float];
    type OutputOwned = Vec<Float>;

    fn lookback(&self) -> usize {
        self.period
    }

    fn compute<'a>(
        &self,
        input: Self::Input<'a>,
        output: Self::OutputMut<'a>,
    ) -> Result<OutputRange> {
        BETA(input.real0, input.real1, self.period, output)
    }

    fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
        BETA_vec(input.real0, input.real1, self.period)
    }
}

impl StreamingIndicator for BETA {
    type Tick = PairTick;
    type TickOutput = Float;

    fn next(&mut self, input: PairTick) -> Result<Option<Float>> {
        validate_finite_slices(&[("real0", &[input.real0]), ("real1", &[input.real1])])?;
        Ok(self.state.push(input))
    }
}

impl Resettable for BETA {
    fn reset(&mut self) {
        self.state.reset();
    }
}
```

### crates/ta-core/src/statistic/regression.rs — NEW

Macro-backed public surfaces for the five projections over one private rolling fit engine.

```rust
//! Linear Regression family and Time Series Forecast.

use super::moments::{statistic_lookback, RegressionFit, RollingRegression};
use crate::{
    compact_buffer, padded_from_compact, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(feature = "f32")]
const RAD_TO_DEG: Float = 180.0 as Float / core::f32::consts::PI;
#[cfg(not(feature = "f32"))]
const RAD_TO_DEG: Float = 180.0 as Float / core::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
enum RegressionProjection {
    Endpoint,
    Slope,
    Intercept,
    Angle,
    Forecast,
}

#[inline]
fn project(fit: RegressionFit, period: usize, projection: RegressionProjection) -> Float {
    match projection {
        RegressionProjection::Endpoint => fit.slope.mul_add((period - 1) as Float, fit.intercept),
        RegressionProjection::Slope => fit.slope,
        RegressionProjection::Intercept => fit.intercept,
        RegressionProjection::Angle => fit.slope.atan() * RAD_TO_DEG,
        RegressionProjection::Forecast => fit.slope.mul_add(period as Float, fit.intercept),
    }
}

fn regression_batch(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    projection: RegressionProjection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 2, 0)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut regression = RollingRegression::new(timeperiod);
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(fit) = regression.push(value) {
            out_real[output_idx] = project(fit, timeperiod, projection);
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

macro_rules! define_regression_indicator {
    ($name:ident, $vec_name:ident, $projection:expr, $description:literal) => {
        #[doc = concat!("TA-Lib-style ", $description, " batch function.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            regression_batch(stringify!($name), real, timeperiod, $projection, out_real)
        }

        #[doc = concat!("Computes ", $description, " into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real.len());
            let range = $name(real, timeperiod, &mut compact)?;
            Ok(padded_from_compact(
                real.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!($description, " indicator.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            regression: RollingRegression,
        }

        impl $name {
            #[doc = concat!("Creates a new ", $description, " indicator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                statistic_lookback(timeperiod, 2, 0)?;
                Ok(Self {
                    period: timeperiod,
                    regression: RollingRegression::new(timeperiod),
                })
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Computes compact outputs using this indicator's period.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, out_real)
            }

            /// Computes full-length padded outputs using this indicator's period.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period)
            }

            /// Checked streaming update that returns `Float::NAN` during warm-up.
            pub fn next_checked(&mut self, input: Float) -> Result<Float> {
                Ok(self.next(input)?.unwrap_or(Float::NAN))
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [Float];
            type OutputOwned = Vec<Float>;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input, self.period, output)
            }

            fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(input, self.period)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                Ok(self
                    .regression
                    .push(input)
                    .map(|fit| project(fit, self.period, $projection)))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                self.regression.reset();
            }
        }
    };
}

define_regression_indicator!(
    LINEARREG,
    LINEARREG_vec,
    RegressionProjection::Endpoint,
    "Linear Regression"
);
define_regression_indicator!(
    LINEARREG_SLOPE,
    LINEARREG_SLOPE_vec,
    RegressionProjection::Slope,
    "Linear Regression Slope"
);
define_regression_indicator!(
    LINEARREG_INTERCEPT,
    LINEARREG_INTERCEPT_vec,
    RegressionProjection::Intercept,
    "Linear Regression Intercept"
);
define_regression_indicator!(
    LINEARREG_ANGLE,
    LINEARREG_ANGLE_vec,
    RegressionProjection::Angle,
    "Linear Regression Angle"
);
define_regression_indicator!(
    TSF,
    TSF_vec,
    RegressionProjection::Forecast,
    "Time Series Forecast"
);
```

### crates/ta-core/src/inventory.rs:8-12,283-292 — MODIFY

Advance the implemented count and all nine Statistic Function records.

```rust
/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 54;

// Statistic Functions — 9 functions.
function!("BETA", StatisticFunctions, Implemented),
function!("CORREL", StatisticFunctions, Implemented),
function!("LINEARREG", StatisticFunctions, Implemented),
function!("LINEARREG_ANGLE", StatisticFunctions, Implemented),
function!("LINEARREG_INTERCEPT", StatisticFunctions, Implemented),
function!("LINEARREG_SLOPE", StatisticFunctions, Implemented),
function!("STDDEV", StatisticFunctions, Implemented),
function!("TSF", StatisticFunctions, Implemented),
function!("VAR", StatisticFunctions, Implemented),
```

### crates/ta-core/tests/inventory.rs:1-245 — MODIFY

Add implemented names, imports, trait assertions, and remove VAR from deferred representatives.

```rust
use ta_core::statistic::{
    BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF,
    VAR,
};

// Add to the `implemented` array after TRANGE:
"BETA",
"CORREL",
"LINEARREG",
"LINEARREG_ANGLE",
"LINEARREG_INTERCEPT",
"LINEARREG_SLOPE",
"STDDEV",
"TSF",
"VAR",

// Add after volatility trait assertions:
assert_indicator::<BETA>();
assert_streaming::<BETA>();
assert_indicator::<CORREL>();
assert_streaming::<CORREL>();
assert_indicator::<LINEARREG>();
assert_streaming::<LINEARREG>();
assert_indicator::<LINEARREG_ANGLE>();
assert_streaming::<LINEARREG_ANGLE>();
assert_indicator::<LINEARREG_INTERCEPT>();
assert_streaming::<LINEARREG_INTERCEPT>();
assert_indicator::<LINEARREG_SLOPE>();
assert_streaming::<LINEARREG_SLOPE>();
assert_indicator::<STDDEV>();
assert_streaming::<STDDEV>();
assert_indicator::<TSF>();
assert_streaming::<TSF>();
assert_indicator::<VAR>();
assert_streaming::<VAR>();

// Replace the deferred representative list:
for name in ["KAMA", "MAMA", "MACD", "BBANDS", "CDLDOJI", "HT_SINE"] {
    let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
    assert_eq!(info.status, ImplementationStatus::Planned, "{name}");
}
```

### crates/ta-benchmarks/benches/first_tranche.rs:9-522 — MODIFY

Import all statistic APIs, add the complete size/period benchmark matrix, and register it.

```rust
// Add inside the existing `use ta_core::{ ... }` tree:
statistic::{
    BETA_vec, CORREL_vec, LINEARREG_ANGLE_vec, LINEARREG_INTERCEPT_vec, LINEARREG_SLOPE_vec,
    LINEARREG_vec, STDDEV_vec, TSF_vec, VAR_vec, BETA, CORREL, LINEARREG, LINEARREG_ANGLE,
    LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,
},

// Add beside the existing benchmark constants:
const STATISTIC_PERIODS: &[usize] = &[5, 20, 100, 500];

fn bench_statistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/statistic");

    macro_rules! bench_variance {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let real = series_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(1.0 as Float),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let real = series_fixture($size);
                    b.iter(|| {
                        let output = $vec(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(1.0 as Float),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    macro_rules! bench_single {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let real = series_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let real = series_fixture($size);
                    b.iter(|| {
                        let output = $vec(black_box(real.as_slice()), black_box($period))
                            .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    macro_rules! bench_paired {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let (real0, real1) = paired_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real0.as_slice()),
                            black_box(real1.as_slice()),
                            black_box($period),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let (real0, real1) = paired_fixture($size);
                    b.iter(|| {
                        let output = $vec(
                            black_box(real0.as_slice()),
                            black_box(real1.as_slice()),
                            black_box($period),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    for &size in SIZES {
        for &period in STATISTIC_PERIODS {
            bench_variance!(VAR, VAR_vec, "VAR", size, period);
            bench_variance!(STDDEV, STDDEV_vec, "STDDEV", size, period);
            bench_paired!(CORREL, CORREL_vec, "CORREL", size, period);
            bench_paired!(BETA, BETA_vec, "BETA", size, period);
            bench_single!(LINEARREG, LINEARREG_vec, "LINEARREG", size, period);
            bench_single!(
                LINEARREG_SLOPE,
                LINEARREG_SLOPE_vec,
                "LINEARREG_SLOPE",
                size,
                period
            );
            bench_single!(
                LINEARREG_INTERCEPT,
                LINEARREG_INTERCEPT_vec,
                "LINEARREG_INTERCEPT",
                size,
                period
            );
            bench_single!(
                LINEARREG_ANGLE,
                LINEARREG_ANGLE_vec,
                "LINEARREG_ANGLE",
                size,
                period
            );
            bench_single!(TSF, TSF_vec, "TSF", size, period);
        }
    }

    group.finish();
}

// Add to the existing criterion_group! registration list:
bench_statistic,
```

## Slices

### Slice 1: Statistic foundation

**Files**: `crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/src/statistic/moments.rs`, `crates/ta-core/src/lib.rs`

#### Automated Verification:
- [ ] The revised foundation compiles with the public Pair views and private engines: `cargo check -p ta-core --lib`
- [ ] Existing core library tests remain green: `cargo test -p ta-core --lib`
- [ ] Strict TA-Lib zero endpoints pass under default precision: `cargo test -p ta-core --lib ta_zero_uses_strict_endpoints`
- [ ] Strict TA-Lib zero endpoints pass under f32: `cargo test -p ta-core --features f32 --lib ta_zero_uses_strict_endpoints`
- [ ] The crate root still registers exactly one statistic module: `grep -c '^pub mod statistic;' crates/ta-core/src/lib.rs` returns `1`

#### Manual Verification:
- [ ] `is_ta_zero` uses strict comparisons at both `±1e-14` endpoints, exactly matching pinned TA-Lib `TA_IS_ZERO`.
- [ ] `moments.rs` exposes no crate-root public helper and every post-warm-up `push` remains O(1).
- [ ] Allocation imports preserve the conditional `alloc`/`std` source boundary and all numeric state uses `Float`.

### Slice 2: VAR and STDDEV

**Files**: `crates/ta-core/src/statistic/variance.rs`, `crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/tests/statistic.rs`

#### Automated Verification:
- [ ] VAR/STDDEV public exports and shared-state implementations compile: `cargo check -p ta-core --lib`
- [ ] Variance integration tests pass under default precision: `cargo test -p ta-core --test statistic`
- [ ] Variance integration tests pass under the f32 feature boundary: `cargo test -p ta-core --features f32 --test statistic`
- [ ] Explicit/default surfaces are exported once: `grep -E 'pub use variance::' crates/ta-core/src/statistic/mod.rs | wc -l` returns `1`

#### Manual Verification:
- [ ] Validation order is period → nbdev → finite input → sufficiency → output capacity, and failures do not write output or advance streaming state.
- [ ] VAR ignores valid nbdev, STDDEV applies it only above TA epsilon, and batch/stream/reset replay uses exact shared-transition arithmetic after multiple wraps.

### Slice 3: CORREL

**Files**: `crates/ta-core/src/statistic/correl.rs`, `crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/tests/statistic.rs`

#### Automated Verification:
- [ ] CORREL and statistic-owned Pair views compile through the public facade: `cargo check -p ta-core --lib`
- [ ] CORREL and locked variance tests pass under default precision: `cargo test -p ta-core --test statistic`
- [ ] CORREL and locked variance tests pass under f32: `cargo test -p ta-core --features f32 --test statistic`
- [ ] CORREL is exported exactly once: `grep -E 'pub use correl::' crates/ta-core/src/statistic/mod.rs | wc -l` returns `1`

#### Manual Verification:
- [ ] Paired validation precedence is period → equal lengths → finite inputs → sufficiency → output capacity, with no writes/state mutation on failure.
- [ ] Positive, negative, constant-side, period-one, cancellation, ring-wrap, reset-replay, and exact batch/stream behavior match the fixed paired-moments semantics.

### Slice 4: BETA

**Files**: `crates/ta-core/src/statistic/beta.rs`, `crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/tests/statistic.rs`

#### Automated Verification:
- [ ] BETA compiles with statistic-owned Pair views and private return state: `cargo check -p ta-core --lib`
- [ ] BETA plus locked statistic tests pass under default precision: `cargo test -p ta-core --test statistic`
- [ ] BETA plus locked statistic tests pass under f32: `cargo test -p ta-core --features f32 --test statistic`
- [ ] Strict zero-endpoint unit tests remain green with BETA present: `cargo test -p ta-core --lib ta_zero_uses_strict_endpoints`
- [ ] BETA is exported exactly once: `grep -E 'pub use beta::' crates/ta-core/src/statistic/mod.rs | wc -l` returns `1`

#### Manual Verification:
- [ ] BETA validates period → equal lengths → finite inputs → period+1 sufficiency → capacity and leaves output/previous-price/moments unchanged on failure.
- [ ] Tests lock adjacent returns, real0 denominator direction, strict `±1e-14` endpoints, interior near-zero fallback, period-one zero, cancellation, extra lookback, ring-wrap, reset replay, and exact batch/stream parity.

### Slice 5: Linear regression family

**Files**: `crates/ta-core/src/statistic/regression.rs`, `crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/tests/statistic.rs`

#### Automated Verification:
- [ ] All five regression APIs compile through one rolling engine and facade: `cargo check -p ta-core --lib`
- [ ] The complete statistic integration suite passes under default precision: `cargo test -p ta-core --test statistic`
- [ ] The complete statistic integration suite passes under f32: `cargo test -p ta-core --features f32 --test statistic`
- [ ] All five regression compact/vec/struct families are exported: `grep -E 'pub use regression::' crates/ta-core/src/statistic/mod.rs | wc -l` returns `1`

#### Manual Verification:
- [ ] Period → finite input → sufficiency → capacity validation, empty semantics, constant projections, and all five public surfaces match existing contracts.
- [ ] `[1,2,3]` projections, pinned rolling/FMA bits, precision-specific large-baseline cancellation, angle degrees, TSF next-coordinate behavior, ring-wrap parity, reset replay, and invalid-tick non-mutation are locked.

### Slice 6: Inventory discoverability

**Files**: `crates/ta-core/src/inventory.rs`, `crates/ta-core/tests/inventory.rs`

#### Automated Verification:
- [ ] Inventory and all public statistic type consumers compile: `cargo check -p ta-core --tests`
- [ ] Inventory count, status, name, and trait assertions pass: `cargo test -p ta-core --test inventory`
- [ ] Inventory type assertions pass with `Float=f32`: `cargo test -p ta-core --features f32 --test inventory`
- [ ] All completed statistic numerical tests remain green: `cargo test -p ta-core --test statistic`
- [ ] Exactly nine Statistic records are implemented: `grep -c 'StatisticFunctions, Implemented' crates/ta-core/src/inventory.rs` returns `9`

#### Manual Verification:
- [ ] `IMPLEMENTED_FUNCTION_COUNT` is 54 and matches both the manual implemented-name list and dynamic status count.
- [ ] Every statistic struct has `Indicator` and `StreamingIndicator` assertions, all nine records are Implemented, and no statistic name remains in the deferred representative list.

### Slice 7: Full statistic benchmark matrix

**Files**: `crates/ta-benchmarks/benches/first_tranche.rs`

#### Automated Verification:
- [ ] All workspace crates compile with the completed statistic facade and benchmark imports: `cargo check --workspace`
- [ ] All workspace tests pass after inventory and benchmark wiring: `cargo test --workspace`
- [ ] The full core suite passes with `Float=f32`: `cargo test -p ta-core --features f32`
- [ ] All 216 statistic benchmark IDs compile in the existing target: `cargo bench -p ta-benchmarks --bench first_tranche --no-run`
- [ ] Formatting is clean across the workspace: `cargo fmt --all -- --check`
- [ ] The statistic benchmark group is defined and registered: `grep -c 'bench_statistic' crates/ta-benchmarks/benches/first_tranche.rs` returns `2`

#### Manual Verification:
- [ ] The matrix is 9 functions × compact/vec × 4 periods × 3 sizes = 216 unique IDs, with periods encoded in every `BenchmarkId`.
- [ ] Fixtures and reusable compact outputs are created outside `b.iter()`, vec allocation stays inside, and inputs/period/nbdev/outputs/results are black-boxed.
- [ ] No new benchmark target, production dependency, adapter wiring, shared-contract change, or SIMD path is introduced.

## Desired End State

```rust
use ta_core::statistic::{
    BETA_vec, CORREL_vec, LINEARREG_ANGLE_vec, STDDEV_vec_with_default_nbdev,
    PairInput, VAR,
};
use ta_core::{Float, Indicator, OutputRange};

let prices: [Float; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
let mut compact = [0.0; 3];
let range = VAR(&prices, 3, 1.0, &mut compact)?;
assert_eq!(range, OutputRange::new(2, 3));

let stddev = STDDEV_vec_with_default_nbdev(&prices, 3)?;
assert!(stddev[..2].iter().all(|value| value.is_nan()));

let market: [Float; 5] = [100.0, 101.0, 103.0, 102.0, 104.0];
let asset: [Float; 5] = [50.0, 51.0, 52.0, 51.5, 53.0];
let correlation = CORREL_vec(&market, &asset, 3)?;
let beta = BETA_vec(&market, &asset, 3)?;

let angle = LINEARREG_ANGLE_vec(&prices, 3)?;
assert!((angle[2] - 45.0).abs() < 1e-10);

let correl = ta_core::statistic::CORREL::new(3)?;
let via_trait = Indicator::compute_to_vec(
    &correl,
    PairInput {
        real0: &market,
        real1: &asset,
    },
)?;
assert_eq!(via_trait.len(), market.len());
# Ok::<(), ta_core::TalibError>(())
```

## File Map

```text
crates/ta-core/src/statistic/mod.rs                    # NEW — public facade and Pair types
crates/ta-core/src/statistic/moments.rs                # NEW — private rolling engines and validation
crates/ta-core/src/statistic/variance.rs               # NEW — VAR and STDDEV family
crates/ta-core/src/statistic/correl.rs                 # NEW — CORREL family
crates/ta-core/src/statistic/beta.rs                   # NEW — BETA family
crates/ta-core/src/statistic/regression.rs             # NEW — five regression projections
crates/ta-core/tests/statistic.rs                      # NEW — group integration tests
crates/ta-core/src/lib.rs                              # MODIFY — crate-root module registration
crates/ta-core/src/inventory.rs                        # MODIFY — count and statuses
crates/ta-core/tests/inventory.rs                      # MODIFY — names and trait consumers
crates/ta-benchmarks/benches/first_tranche.rs          # MODIFY — full public benchmark matrix
```

## Ordering Constraints

- The facade, Pair types, validators, and rolling engines must exist before any public statistic family.
- VAR/STDDEV establish the single-input macro and shared moments usage before paired or regression families.
- CORREL establishes statistic-owned paired trait wiring before BETA adds previous-observation transformation.
- BETA must preserve its extra lookback independently of CORREL despite shared paired moments.
- All five regression projections land together so no public projection temporarily diverges from the common fit engine.
- Inventory may advance only after all nine structs and functions exist.
- Benchmarks land last so every imported public API and default wrapper is available.
- No slices are parallelized in the generated plan; each locked slice is the source of truth for the next.

## Verification Notes

- Run `cargo fmt --all -- --check`; all new public items must satisfy crate-level `missing_docs`.
- Run `cargo check -p ta-core` after the foundation and each algorithm family to catch facade/export and no-std-style import mistakes early.
- Run `cargo test -p ta-core --test statistic` after each algorithm slice; the test target must cover exact `OutputRange`, NaN padding, empty input, insufficient data, short output, NaN/Infinity, exact error variants, validation precedence, and untouched state/output on errors.
- Run `cargo test -p ta-core --features f32 --test statistic`; external golden assertions must use the fixed f32 abs+rel tolerance rather than f64 assumptions.
- Verify batch and streaming results with `to_bits()` after multiple ring wraps because both surfaces share private transitions.
- Verify reset replay, invalid-tick non-mutation, period minima/maxima, and rejection of period `100001` for each period class.
- Verify VAR `[1,2,3]` is population variance `2/3`, changing valid VAR `nbdev` does not change output, and STDDEV applies `sqrt(variance) * nbdev` only above the `1e-14` threshold.
- Verify CORREL positive/negative/constant-side behavior and period-one zero output.
- Verify BETA uses adjacent returns, real0 as denominator, direction-sensitive input order, strict `±1e-14` endpoints, interior near-zero previous-price fallback, and lookback `period`.
- Verify `[1,2,3]` regression outputs LINEARREG `3`, SLOPE `1`, INTERCEPT `1`, ANGLE `45` degrees, and TSF `4`, plus pinned long-sequence oracle vectors from `1bec05c…`.
- Verify large-baseline near-constant vectors document the selected cancellation behavior separately for f64/f32 instead of substituting a stable formula.
- Run `cargo test -p ta-core --test inventory`; implemented count must be 54, all nine records must be Implemented, and all nine structs must satisfy batch/stream trait assertions.
- Run `cargo bench -p ta-benchmarks --bench first_tranche --no-run`; all 216 statistic IDs (9 functions × 2 surfaces × 4 periods × 3 sizes) must compile and `bench_statistic` must appear in `criterion_group!`.
- Inspect the benchmark body to ensure fixtures/compact outputs are outside `b.iter()`, vec allocation is inside, and every input/period/output is black-boxed.
- Run `cargo check --workspace` and `cargo test --workspace` only on the terminal slice after inventory and benchmark wiring are complete.
- Run `cargo test -p ta-core --features f32` on the terminal slice to catch cross-target precision regressions beyond the statistic test file.
- Do not claim repository-wide no-std support: current configuration has pre-existing dependency/CI gaps. Manually verify new files use conditional `alloc::vec::Vec`/`std::vec::Vec` imports and do not add new std-only APIs.
- Confirm `git diff -- crates/ta-py crates/ta-wasm crates/ta-core/src/common.rs crates/ta-core/src/traits.rs crates/ta-core/src/error.rs crates/ta-core/src/types.rs crates/ta-core/src/simd Cargo.toml Cargo.lock` is empty.

## Performance Considerations

- All rolling engines perform one initial O(period) fill and O(1) add/remove transitions per subsequent input, yielding O(n) batch and streaming work.
- Each public function owns independent state; related public calls do not cache across APIs.
- Regression uses the pinned reversed-position `sum_xy` recurrence and `Float::mul_add` projections; arithmetic order is part of the oracle contract.
- BETA stores previous raw prices plus a period-sized paired-return ring; CORREL stores only raw paired moments.
- Each state uses O(period) memory. The official maximum period of 100,000 bounds allocations before construction.
- Compact benchmarks exclude fixture and reusable-output allocation; vec benchmarks intentionally include compact allocation, padded allocation, initialization, and copy.
- The full 216-case matrix is intentionally larger than existing groups to expose runtime growth with period as well as input length.

## Migration Notes

Not applicable. There is no persisted schema and no existing public `statistic` implementation to migrate or roll back. Inventory changes can be reverted atomically with the module if the tranche is withdrawn.

## Pattern References

- `crates/ta-core/src/volume/mod.rs:6-12` — private leaves and explicit public group facade.
- `crates/ta-core/src/volume/ad.rs:40-69` — sibling-visible validation/computation helpers.
- `crates/ta-core/src/volume/adosc.rs:45-86,121-132,263-287` — shared private state transition across batch/streaming.
- `crates/ta-core/src/overlap/sma.rs:23-46,164-204` — O(1) rolling batch/ring state and reset.
- `crates/ta-core/src/overlap/wma.rs:34-49,135-162` — order-sensitive weighted recurrence and ring order.
- `crates/ta-core/src/math_operators/arithmetic.rs:13-29,86-118` — paired batch/tick views and trait forwarding.
- `crates/ta-core/src/volatility/trange.rs:89-91,138-166` — previous-observation warm-up and reset.
- `crates/ta-core/src/overlap/t3.rs:24-34,59-151` — finite optional-parameter validation and explicit/default surfaces.
- `crates/ta-core/tests/volume.rs:193-225,314-397` — paired/multi-input batch-stream parity, reset, and validation tests.
- `crates/ta-core/tests/volatility.rs:14-118,253-307,376-397` — compact/padded/trait/warm-up/degenerate/error test organization.
- `crates/ta-benchmarks/benches/first_tranche.rs:72-103,512-522` — compact/vec Criterion timing and registration.

## Developer Context

**Q (`crates/ta-core/src/volume/mod.rs:6-12`, `crates/ta-core/src/volume/ad.rs:40-69`): ❓ Question: 即将在 `statistic` 的 5 个新实现文件中延续“私有叶子模块 + facade 显式 re-export + `pub(super)` 共享 helper”的分组模式（`crates/ta-core/src/volume/mod.rs:6-12`、`crates/ta-core/src/volume/ad.rs:40-69`）。确认继续这个方向，还是项目准备离开该模式？**

A: Follow facade pattern.

**Q (`crates/ta-core/src/volume/adosc.rs:45-86,121-132,263-287`): ❓ Question: 即将在 VAR/STDDEV、CORREL/BETA 和回归族中延续“batch 与 streaming 共用同一个内部状态转移”的模式（`crates/ta-core/src/volume/adosc.rs:45-86,121-132,263-287`），覆盖 3 组 rolling state。确认继续，还是 batch/stream 分开实现？**

A: Share state path.

**Q (`crates/ta-core/src/overlap/wma.rs:103-110`, `crates/ta-core/src/volume/adosc.rs:215-218`): ❓ Question: 现有 stateful 指标普遍提供 `next_checked()`，用 NaN 表示 warm-up（如 `crates/ta-core/src/overlap/wma.rs:103-110`、`crates/ta-core/src/volume/adosc.rs:215-218`）。是否在 9 个新 struct 上继续传播这一 convenience API？**

A: Add next_checked.

**Q (`crates/ta-core/src/overlap/t3.rs:59-115,133-151`): ❓ Question: VAR/STDDEV 需要公开 `nbdev`；本地唯一同类可选参数先例 T3 同时提供显式参数和 default wrappers（`crates/ta-core/src/overlap/t3.rs:59-115,133-151`）。新统计 API 应采用哪种 surface？**

A: Explicit + default.

**Q (`crates/ta-core/src/overlap/t3.rs:24-34`, `crates/ta-core/src/math_transform/mod.rs:20-40`): ❓ Question: TA-Lib C 只替换默认 sentinel，并未拒绝非有限 `nbdev`；但本地参数先例会拒绝 NaN/Infinity（`crates/ta-core/src/overlap/t3.rs:24-34`），而结果可按 IEEE 传播（`crates/ta-core/src/math_transform/mod.rs:20-40`）。VAR/STDDEV 应采用哪种校验？**

A: Require finite.

**Q (`crates/ta-core/src/overlap/wma.rs:34-49`): ❓ Question: 明确为上游 C TA-Lib 后，回归 golden 应固定到哪个版本？**

A: Commit 1bec05c.

**Clarification requested:** “哪里的稳定版？”

**Clarification given:** `v0.7.1` means the upstream C TA-Lib release, not a fast-ta release; `1bec05c…` is the newer upstream commit containing the O(1)+FMA regression change.

**Q (`crates/ta-core/src/types.rs:25-33`, `crates/ta-core/tests/price_transform.rs:7-11`): ❓ Question: `Float` 在默认/f32 构建切换（`crates/ta-core/src/types.rs:25-33`），而现有测试只用固定绝对误差（如 `crates/ta-core/tests/price_transform.rs:7-11`）。统计 oracle 应采用哪套绝对+相对容差？**

A: Feature-aware was accepted after clarification: f64 uses abs `1e-12` + rel `1e-10`; f32 uses abs `1e-4` + rel `1e-4`; batch/stream exactness and cancellation stress vectors are tested separately.

**Q (`crates/ta-benchmarks/benches/first_tranche.rs:22-23`): ❓ Question: 当前 Criterion 只按输入长度、固定 `PERIOD=20`（`crates/ta-benchmarks/benches/first_tranche.rs:22-23`），无法暴露 O(n×period) 回退。统计 benchmark 应采用哪种 period 矩阵？**

A: Full period matrix.

**Q (`crates/ta-core/src/volume/mod.rs:6-12`): ❓ Question: 现有 group 多按指标拆私有文件（`crates/ta-core/src/volume/mod.rs:6-12`），但 5 个 regression exports 只在最终 projection 上不同。`statistic` 应如何分文件？**

A: Group by engine.

**Design confirmation:** Proceed to decomposition with 7 new and 4 modified files; no adapter, SIMD, shared-contract, or manifest scope.

**Decomposition confirmation:** Approved 7 slices: foundation, variance, CORREL, BETA, regression, inventory, benchmarks.

**Micro-checkpoint (`crates/ta-core/src/statistic/mod.rs`, `crates/ta-core/src/statistic/moments.rs`, `crates/ta-core/src/lib.rs`): Slice 1/7: Statistic foundation — 3 files affected. Adds Pair views, bounded validation, private O(1) engines, and root wiring. Approve?**

A: Approve. The slice was later reopened when Slice 4 exposed an epsilon-endpoint defect.

**Micro-checkpoint (`crates/ta-core/src/statistic/variance.rs`, `crates/ta-core/tests/statistic.rs`): Slice 2/7: VAR and STDDEV — 3 files affected. Adds full APIs, shared-state streaming, strict validation, precision/cancellation tests, and reset replay. Approve?**

A: Approve. The shared upper-bound validator made a separate STDDEV `100001` assertion intentionally redundant.

**Micro-checkpoint (`crates/ta-core/src/statistic/correl.rs`, `crates/ta-core/tests/statistic.rs`): Slice 3/7: CORREL — 3 files affected. Adds complete paired correlation APIs and strict parity/degeneracy/validation tests. Approve?**

A: Approve.

**Cascade checkpoint (`crates/ta-core/src/statistic/moments.rs:42`): Reopen approved Slice 1 to correct strict TA-Lib epsilon endpoints, then regenerate/reverify Slice 4?**

A: Reopen Slice 1.

**Reopened micro-checkpoint (`crates/ta-core/src/statistic/moments.rs:42`): Reopened Slice 1 — correct strict epsilon endpoints and add default/f32 boundary tests. Re-approve?**

A: Approve. `is_ta_zero` now uses strict comparisons and unit tests cover interior values plus both endpoints.

**Micro-checkpoint (`crates/ta-core/src/statistic/beta.rs`, `crates/ta-core/tests/statistic.rs`): Slice 4/7: BETA — 3 files affected. Adds return-based directional BETA, strict threshold endpoints, extra warm-up, and exact stream/reset tests. Approve?**

A: Approve.

**Micro-checkpoint (`crates/ta-core/src/statistic/regression.rs`, `crates/ta-core/tests/statistic.rs`): Slice 5/7: regression family — 3 files affected. Adds all five projection APIs over one pinned rolling/FMA engine and complete parity tests. Approve?**

A: Approve.

**Micro-checkpoint (`crates/ta-core/src/inventory.rs`, `crates/ta-core/tests/inventory.rs`): Slice 6/7: inventory discoverability — 2 files affected. Synchronizes count=54, nine statuses/names, trait consumers, and deferred list. Approve?**

A: Approve.

**Micro-checkpoint (`crates/ta-benchmarks/benches/first_tranche.rs`): Slice 7/7: full benchmark matrix — 1 file affected. Adds 216 compact/vec size-period IDs and terminal verification gates. Approve?**

A: Approve.

## Design History

- Slice 1: Statistic foundation — revised: strict TA-Lib epsilon endpoints and default/f32 boundary tests
- Slice 2: VAR and STDDEV — approved as generated
- Slice 3: CORREL — approved as generated
- Slice 4: BETA — approved as generated after Slice 1 epsilon cascade revision
- Slice 5: Linear regression family — approved as generated
- Slice 6: Inventory discoverability — approved as generated
- Slice 7: Full statistic benchmark matrix — approved as generated

## References

- `.rpiv/artifacts/research/2026-07-15_20-37-23_statistic-functions.md`
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md`
- `.rpiv/artifacts/research/2026-07-09_21-23-34_volatility-indicators.md`
- `.rpiv/artifacts/validation/2026-07-13_18-22-08_实现volume分组指标.md`
- [TA-Lib VAR at `1bec05c…`](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_VAR.c)
- [TA-Lib STDDEV at `1bec05c…`](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_STDDEV.c)
- [TA-Lib BETA source template at `1bec05c…`](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/ta_codegen/input/beta/beta.c)
- [TA-Lib LINEARREG at `1bec05c…`](https://github.com/TA-Lib/ta-lib/blob/1bec05cf72fa790e2e3ecca40e6607de15fe0a30/src/ta_func/ta_LINEARREG.c)
- [TA-Lib `v0.7.1` LINEARREG comparison](https://github.com/TA-Lib/ta-lib/blob/v0.7.1/src/ta_func/ta_LINEARREG.c)
