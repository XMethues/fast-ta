---
template_version: 1
date: 2026-07-05T11:35:24+0800
author: unknown
commit: 896d1d7
branch: main
repository: fast-ta
topic: "Validation of TA-Lib Rust core foundation + first tranche"
status: ready
verdict: pass
parent: ".rpiv/artifacts/plans/2026-07-05_09-51-36_rust-talib-core-foundation-first-tranche.md"
tags: [validation, ta-core, talib, indicators, foundation, first-tranche]
last_updated: 2026-07-05T11:35:24+0800
---

## Validation Report: TA-Lib Rust core foundation + first tranche

### Implementation Status

- ✓ Phase 1: Core contracts — Fully implemented
- ✓ Phase 2: SMA rewrite — Fully implemented
- ✓ Phase 3: Price Transform group — Fully implemented
- ✓ Phase 4: Math Transform group — Fully implemented
- ✓ Phase 5: Math Operators group — Fully implemented
- ✓ Phase 6: Full inventory ledger — Fully implemented
- ✓ Phase 7: First-tranche benchmarks — Fully implemented

### Automated Verification Results

- ✓ SMA rewrite tests: `cargo test -p ta-core --test overlap_sma` — 5 tests passed.
- ✓ Price Transform tests: `cargo test -p ta-core --test price_transform` — 5 tests passed.
- ✓ Math Transform tests: `cargo test -p ta-core --test math_transform` — 5 tests passed.
- ✓ Math Operators tests: `cargo test -p ta-core --test math_operators` — 6 tests passed.
- ✓ Inventory tests: `cargo test -p ta-core --test inventory` — 6 tests passed.
- ✓ Core crate tests: `cargo test -p ta-core` — unit, integration, and doc tests passed.
- ✓ f32 precision tests: `cargo test -p ta-core --features f32` — unit, integration, and doc tests passed under `Float = f32`.
- ✓ Workspace check: `cargo check --workspace` — completed successfully, including `ta-py`.
- ✓ Formatting: `cargo fmt --all -- --check` — no formatting changes required.
- ✓ Trait zero-copy method present: `grep -n "fn compute(&self" crates/ta-core/src/traits.rs` — matched `traits.rs:50`.
- ✓ No SMA assert-based period constructor remains: `grep -R "assert!(.*period" crates/ta-core/src/overlap/sma.rs` — no matches.
- ✓ Price Transform module wired: `grep -n "pub mod price_transform" crates/ta-core/src/lib.rs` — matched `lib.rs:43`.
- ✓ Stateless Price Transform constructors are fallible: `grep -R "pub const fn new() -> Self" crates/ta-core/src/price_transform` — no matches.
- ✓ Math Transform module wired: `grep -n "pub mod math_transform" crates/ta-core/src/lib.rs` — matched `lib.rs:39`.
- ✓ Math Transform private-module re-export present: `grep -n "pub use functions" crates/ta-core/src/math_transform/mod.rs` — matched `math_transform/mod.rs:116`.
- ✓ Math Operators module wired: `grep -n "pub mod math_operators" crates/ta-core/src/lib.rs` — matched `lib.rs:37`.
- ✓ Math Operators private-module re-export present: `grep -n "pub use arithmetic" crates/ta-core/src/math_operators/mod.rs` — matched `math_operators/mod.rs:10`.
- ✓ First-tranche SIMD boundary preserved: `grep -R "crate::simd::arch\|core::arch\|std::arch" crates/ta-core/src/overlap crates/ta-core/src/price_transform crates/ta-core/src/math_transform crates/ta-core/src/math_operators` — no matches.
- ✓ Inventory module wired: `grep -n "pub mod inventory" crates/ta-core/src/lib.rs` — matched `lib.rs:35`.
- ✓ Inventory count explicit: `grep -n "FUNCTION_COUNT: usize = 161" crates/ta-core/src/inventory.rs` — matched `inventory.rs:9`.
- ✓ First-tranche benchmark target compiles: `cargo bench -p ta-benchmarks --bench first_tranche --no-run` — compiled successfully; only `criterion::black_box` deprecation warnings.
- ✓ Existing basic benchmark target compiles: `cargo bench -p ta-benchmarks --bench basic --no-run` — compiled successfully; only existing `criterion::black_box` deprecation warnings.
- ✓ Benchmark uses public APIs only: `grep -R "ta_core::.*src\|crate::" crates/ta-benchmarks/benches/first_tranche.rs` — no matches.
- ✓ No regressions detected.

### Code Review Findings

#### Matches Plan:

- `crates/ta-core/src/common.rs:18` — `OutputRange` documents TA-Lib `outBegIdx/outNBElement` mapping and exposes `beg_idx` / `nb_element` compact-output metadata.
- `crates/ta-core/src/common.rs:91` — shared validators return `TalibError` through `Result` rather than panicking.
- `crates/ta-core/src/lib.rs:48` — `OutputRange`, `PadValue`, and shared helper functions are re-exported for downstream indicator modules.
- `crates/ta-core/src/traits.rs:50` — `Indicator::compute` has the planned default compact-output signature returning `OutputRange`.
- `crates/ta-core/src/overlap/sma.rs:24` — `SMA` free function validates inputs, writes compact outputs, and returns `OutputRange::new(lookback, count)`.
- `crates/ta-core/src/overlap/sma.rs:50` — `SMA_vec` preserves full-length padded output shape through `padded_from_compact`.
- `crates/ta-core/src/overlap/sma.rs:73` — `SMA::new` is fallible and initializes its streaming buffer.
- `crates/ta-core/src/price_transform/mod.rs:6` — all five Price Transform modules are wired and re-exported.
- `crates/ta-core/src/price_transform/avgprice.rs:15` — `AVGPRICE` implements `(open + high + low + close) / 4` with separate OHLC slices and equal-length validation.
- `crates/ta-core/src/price_transform/avgdev.rs:15` — `AVGDEV` uses rolling lookback output with padded wrapper and a `Resettable` stateful surface.
- `crates/ta-core/src/math_transform/mod.rs:99` — all 15 Math Transform functions are generated with free function, `*_vec` wrapper, uppercase struct, and `Indicator` implementation.
- `crates/ta-core/src/math_operators/mod.rs:6` — Math Operators are split into arithmetic, rolling, and extrema modules with public re-exports.
- `crates/ta-core/src/math_operators/extrema.rs:130` — `MINMAX` uses compact parallel output buffers; `MINMAX_vec` returns named parallel vectors.
- `crates/ta-core/src/math_operators/extrema.rs:15` — integer index output wrappers use `Vec<i32>` with zero padding via shared `PadValue` behavior.
- `crates/ta-core/src/inventory.rs:9` — inventory tracks the official 161-function count, 10 groups, and 32 first-tranche implemented functions.
- `crates/ta-benchmarks/Cargo.toml:17` — `first_tranche` Criterion target is registered.
- `crates/ta-benchmarks/benches/first_tranche.rs:57` — benchmark groups cover SMA, Price Transform, Math Transform, and Math Operators with public `ta_core` APIs.

#### Deviations from Plan:

- None. Implementation is a faithful realization of the plan.

#### Pattern Conformance:

- ✓ Module wiring, private file organization, integration test layout, uppercase TA-Lib names, `Result<Self>` constructors, and shared validator usage follow the established `ta-core` conventions.
- ✓ First-tranche indicator code uses scalar loops/shared helpers and does not call private SIMD architecture backends.
- ✓ Benchmark fixtures and reusable output buffers are allocated outside `b.iter()` except the intentional `SMA_vec` allocation benchmark.
- Minor observation: some public re-export lists place `*_vec` before the primary function/type. This is an acceptable ordering variation, not a semantic deviation.

### Manual Testing Required:

None outstanding — the plan's manual criteria are static code-inspection checks and were verified during validation:

1. Core contract and helpers:
   - [x] `common.rs` documents compact `outBegIdx/outNBElement`-style output.
   - [x] public validation helpers return `TalibError` through `Result`.
   - [x] `lib.rs` re-exports `OutputRange` and `PadValue`.
   - [x] `Indicator::compute` is defaulted while first-tranche indicators override compact kernels where needed.
2. First-tranche indicator APIs:
   - [x] compact APIs return `OutputRange`; padded wrappers preserve full input length and warm-up sentinels.
   - [x] multi-price/binary functions use separate input slices and validate equal lengths.
   - [x] all first-tranche public functions/types use uppercase TA-Lib names.
   - [x] period-based constructors return `Result<Self>` and reject period `0`.
3. Inventory and benchmarks:
   - [x] all 10 official TA-Lib groups and 161 records are represented; local extras are excluded.
   - [x] first-tranche functions are marked `Implemented`, with remaining official functions `Planned`.
   - [x] benchmark code imports only public `ta_core` APIs and registers every `bench_*` function in `criterion_group!`.

### Recommendations:

- Ready to commit — implementation is complete and validated.
- Optional maintenance follow-up: replace deprecated `criterion::black_box` imports with `std::hint::black_box` in benchmarks when convenient; current benchmark targets compile successfully.
