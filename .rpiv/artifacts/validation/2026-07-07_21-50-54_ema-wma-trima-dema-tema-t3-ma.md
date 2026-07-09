---
template_version: 1
date: 2026-07-07T21:50:54+0800
author: unknown
commit: cfdf1e5
branch: main
repository: fast-ta
topic: "Validation of EMA/WMA/TRIMA/DEMA/TEMA/T3/MA"
status: ready
verdict: pass
parent: ".rpiv/artifacts/plans/2026-07-05_21-38-45_ema-wma-trima-dema-tema-t3-ma.md"
tags: [validation, ta-core, talib, overlap, moving-averages]
last_updated: 2026-07-07T21:50:54+0800
---

## Validation Report: EMA/WMA/TRIMA/DEMA/TEMA/T3/MA

### Implementation Status

- ✓ Phase 1: EMA foundation — Fully implemented
- ✓ Phase 2: Weighted and triangular moving averages — Fully implemented
- ✓ Phase 3: EMA-derived composites — Fully implemented
- ✓ Phase 4: T3 with vfactor — Fully implemented
- ✓ Phase 5: MA dispatcher — Fully implemented
- ✓ Phase 6: Moving-average benchmarks — Fully implemented

### Automated Verification Results

- ✓ Moving-average tests: `cargo test -p ta-core --test overlap_moving_averages` — 23 tests passed.
- ✓ Inventory tests: `cargo test -p ta-core --test inventory` — 7 tests passed.
- ✓ Core crate tests: `cargo test -p ta-core` — unit, integration, and doc tests passed.
- ✓ f32 feature tests: `cargo test -p ta-core --features f32` — unit, integration, and doc tests passed with `Float = f32`.
- ✓ Workspace check: `cargo check --workspace` — finished without errors.
- ✓ Formatting: `cargo fmt --all -- --check` — passed.
- ✓ First-tranche benchmark compile: `cargo bench -p ta-benchmarks --bench first_tranche --no-run` — benchmark executable built.
- ✓ Basic benchmark compile: `cargo bench -p ta-benchmarks --bench basic --no-run` — benchmark executable built.
- ✓ Overlap module wiring: `grep -n "mod ema\|pub use ema" crates/ta-core/src/overlap/mod.rs` and corresponding WMA/TRIMA, DEMA/TEMA, T3, and MA wiring greps — all expected modules and re-exports found.
- ✓ Inventory statuses: `grep -n 'function!("DEMA", OverlapStudies, Implemented)\|function!("EMA", OverlapStudies, Implemented)\|function!("MA", OverlapStudies, Implemented)\|function!("T3", OverlapStudies, Implemented)\|function!("TEMA", OverlapStudies, Implemented)\|function!("TRIMA", OverlapStudies, Implemented)\|function!("WMA", OverlapStudies, Implemented)' crates/ta-core/src/inventory.rs` — all planned moving averages marked implemented.
- ✓ Deferred variants remain planned: `grep -n 'function!("KAMA", OverlapStudies, Planned)\|function!("MAMA", OverlapStudies, Planned)' crates/ta-core/src/inventory.rs` — both remain planned.
- ✓ Adapter scope: `git diff --name-only -- crates/ta-py crates/ta-wasm` plus `git status --porcelain -- crates/ta-py crates/ta-wasm` — no Python or WASM changes.
- ✓ Backend scope: `grep -R -n -E 'crate::simd::arch|core::arch|std::arch|wasm|python|pyo3|wasm_bindgen' crates/ta-core/src/overlap/{ema,wma,trima,dema,tema,t3,ma}.rs` — no private SIMD/backend/binding references.
- ✓ Plan checklist: plan artifact contains 89 checked success criteria and 0 unchecked criteria. Phase-local “no future module declared early” greps were treated as historical phase checkpoints because later phases intentionally add those modules in the final tree.
- ✓ No regressions detected.

### Code Review Findings

#### Matches Plan:

- `crates/ta-core/src/overlap/mod.rs:3-21` — all new overlap modules are declared and public APIs are re-exported for EMA, WMA, TRIMA, DEMA, TEMA, T3, and MA.
- `crates/ta-core/src/overlap/ema.rs:39`, `wma.rs:24`, `trima.rs:56`, `dema.rs:23`, `tema.rs:23`, `t3.rs:60`, `ma.rs:74` — compact uppercase functions implement the TA-Lib-style `Result<OutputRange>` API.
- `crates/ta-core/src/overlap/ema.rs:64`, `wma.rs:57`, `trima.rs:75`, `dema.rs:53`, `tema.rs:57`, `t3.rs:100`, `ma.rs:94` — padded vector wrappers preserve input length and use shared padding helpers.
- `crates/ta-core/src/common.rs:67-70` and `crates/ta-core/src/common.rs:180-198` — padded `Float` outputs use `Float::NAN` before valid compact output positions.
- `crates/ta-core/src/overlap/dema.rs:16-18`, `tema.rs:16-18`, `t3.rs:19-21` — composite lookbacks use checked multiplication and return `TalibError` on overflow.
- `crates/ta-core/src/overlap/t3.rs:15`, `t3.rs:24-31`, `t3.rs:90-95`, `t3.rs:112-113`, `t3.rs:151-152` — T3 exposes explicit vfactor validation and default-vfactor helpers using `0.7`.
- `crates/ta-core/src/overlap/ma.rs:38-48`, `ma.rs:81-88`, `ma.rs:124-125` — `MAType` exposes the official variant set, dispatches implemented averages, and returns `TalibError::NotImplemented` for KAMA/MAMA.
- `crates/ta-core/src/inventory.rs:12` and `crates/ta-core/src/inventory.rs:153-168` — implemented count is 39; new moving averages are implemented while KAMA/MAMA remain planned.
- `crates/ta-core/tests/overlap_moving_averages.rs:27-414` — tests cover compact output alignment, padded NaN warm-up, invalid inputs, struct APIs, streaming/reset, MA dispatch, and T3 defaults/validation.
- `crates/ta-core/tests/inventory.rs:71-78` and `crates/ta-core/tests/inventory.rs:133-148` — inventory tests include moving-average implementation statuses and trait conformance assertions.
- `crates/ta-benchmarks/benches/first_tranche.rs:94-224` — Criterion benchmarks cover public moving-average APIs and allocate reusable buffers outside `b.iter()` for compact benchmarks.

#### Deviations from Plan:

None. Implementation is a faithful realization of the plan.

#### Pattern Conformance:

- ✓ Batch API, struct naming, trait implementations, reset behavior, integration-test organization, and benchmark structure follow the existing SMA and first-tranche conventions.
- Acceptable variation, not a deviation: new padded wrappers use `compact_buffer` + `padded_from_compact` rather than SMA’s hand-written padded loop; this uses existing shared helpers and preserves the same public behavior.
- Acceptable variation, not a deviation: `from_data` is not uniformly implemented across every new moving-average struct; the approved plan did not require it, and all required constructor/compute/streaming surfaces are present.
- Acceptable variation, not a deviation: benchmark `_vec` coverage is selective (`EMA_vec` plus compact coverage for all new averages), matching the plan’s public-API benchmark requirement without expanding scope.

### Manual Testing Required:

None — the plan’s manual criteria were source/API behavior checks, and they were covered by integration tests, static inspection, inventory assertions, benchmark compilation, and adapter no-diff verification. No Python/WASM/manual runtime flow is in scope for this plan.

### Recommendations:

- Ready to commit — implementation is complete and validated.
- Before committing, review unrelated/pre-existing working-tree modifications and group commits so the validated moving-average tranche stays atomic.
