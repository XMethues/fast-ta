---
template_version: 1
date: 2026-07-18T16:10:37+0800
author: unknown
commit: ffd2451
branch: main
repository: fast-ta
topic: "Validation of Statistic Functions"
status: ready
verdict: pass
parent: ".rpiv/artifacts/plans/2026-07-18_14-01-08_statistic-functions.md"
tags: [validation, plan, ta-core, statistic-functions, rolling-statistics, linear-regression]
last_updated: 2026-07-18T16:10:37+0800
---

## Validation Report: Statistic Functions

### Implementation Status

- ✓ Phase 1: Statistic foundation — Fully implemented
- ✓ Phase 2: VAR and STDDEV — Fully implemented
- ✓ Phase 3: CORREL — Fully implemented
- ✓ Phase 4: BETA — Fully implemented
- ✓ Phase 5: Linear regression family — Fully implemented
- ✓ Phase 6: Inventory discoverability — Fully implemented
- ✓ Phase 7: Full statistic benchmark matrix — Fully implemented

### Automated Verification Results

- ✓ Phase 1 core compilation: `cargo check -p ta-core --lib` — passed.
- ✓ Phase 1 core library tests: `cargo test -p ta-core --lib` — 56 passed.
- ✓ Phase 1 strict-zero default test: `cargo test -p ta-core --lib ta_zero_uses_strict_endpoints` — 1 passed.
- ✓ Phase 1 strict-zero f32 test: `cargo test -p ta-core --features f32 --lib ta_zero_uses_strict_endpoints` — 1 passed.
- ✓ Phase 1 facade registration: `grep -c '^pub mod statistic;' crates/ta-core/src/lib.rs` — returned `1`.
- ✓ Phase 2 core compilation: `cargo check -p ta-core --lib` — passed.
- ✓ Phase 2 statistic suite: `cargo test -p ta-core --test statistic` — 22 passed.
- ✓ Phase 2 f32 statistic suite: `cargo test -p ta-core --features f32 --test statistic` — 22 passed.
- ✓ Phase 2 variance export count: `grep -E 'pub use variance::' crates/ta-core/src/statistic/mod.rs | wc -l` — returned `1`.
- ✓ Phase 2 STDDEV epsilon test: `cargo test -p ta-core --lib stddev_treats_epsilon_as_zero` — 1 passed.
- ✓ Phase 2 f32 STDDEV epsilon test: `cargo test -p ta-core --features f32 --lib stddev_treats_epsilon_as_zero` — 1 passed.
- ✓ Phase 3 core compilation: `cargo check -p ta-core --lib` — passed.
- ✓ Phase 3 statistic suite: `cargo test -p ta-core --test statistic` — 22 passed.
- ✓ Phase 3 f32 statistic suite: `cargo test -p ta-core --features f32 --test statistic` — 22 passed.
- ✓ Phase 3 CORREL export count: `grep -E 'pub use correl::' crates/ta-core/src/statistic/mod.rs | wc -l` — returned `1`.
- ✓ Phase 4 core compilation: `cargo check -p ta-core --lib` — passed.
- ✓ Phase 4 statistic suite: `cargo test -p ta-core --test statistic` — 22 passed.
- ✓ Phase 4 f32 statistic suite: `cargo test -p ta-core --features f32 --test statistic` — 22 passed.
- ✓ Phase 4 strict-zero regression test: `cargo test -p ta-core --lib ta_zero_uses_strict_endpoints` — 1 passed.
- ✓ Phase 4 BETA export count: `grep -E 'pub use beta::' crates/ta-core/src/statistic/mod.rs | wc -l` — returned `1`.
- ✓ Phase 5 core compilation: `cargo check -p ta-core --lib` — passed.
- ✓ Phase 5 statistic suite: `cargo test -p ta-core --test statistic` — 22 passed.
- ✓ Phase 5 f32 statistic suite: `cargo test -p ta-core --features f32 --test statistic` — 22 passed.
- ✓ Phase 5 regression export count: `grep -E 'pub use regression::' crates/ta-core/src/statistic/mod.rs | wc -l` — returned `1`.
- ✓ Phase 6 test-target compilation: `cargo check -p ta-core --tests` — passed.
- ✓ Phase 6 inventory suite: `cargo test -p ta-core --test inventory` — 7 passed.
- ✓ Phase 6 f32 inventory suite: `cargo test -p ta-core --features f32 --test inventory` — 7 passed.
- ✓ Phase 6 statistic regression suite: `cargo test -p ta-core --test statistic` — 22 passed.
- ✓ Phase 6 implemented statistic records: `grep -c 'StatisticFunctions, Implemented' crates/ta-core/src/inventory.rs` — returned `9`.
- ✓ Phase 7 workspace compilation: `cargo check --workspace` — passed.
- ✓ Phase 7 workspace tests: `cargo test --workspace` — all workspace unit, integration, and doc-test targets passed; one existing doc test remained ignored.
- ✓ Phase 7 f32 core suite: `cargo test -p ta-core --features f32` — all core unit, integration, and doc-test targets passed; one existing doc test remained ignored.
- ✓ Phase 7 benchmark compilation: `cargo bench -p ta-benchmarks --bench first_tranche --no-run` — Criterion executable built successfully.
- ✓ Phase 7 formatting: `cargo fmt --all -- --check` — clean.
- ✓ Phase 7 benchmark registration: `grep -c 'bench_statistic' crates/ta-benchmarks/benches/first_tranche.rs` — returned `2`.
- ✓ Phase 7 protected-path hygiene: `git diff -- crates/ta-py crates/ta-wasm crates/ta-core/src/common.rs crates/ta-core/src/traits.rs crates/ta-core/src/error.rs crates/ta-core/src/types.rs crates/ta-core/src/simd Cargo.toml Cargo.lock` — no output.
- ✓ No regressions detected.

### Code Review Findings

#### Matches Plan:

- `crates/ta-core/src/lib.rs:46` and `crates/ta-core/src/statistic/mod.rs:7-42` — the statistic facade is registered once, leaves remain private, approved APIs are explicitly re-exported, and statistic-owned `PairInput`/`PairTick` are public.
- `crates/ta-core/src/statistic/moments.rs:14-313` — bounded period validation, strict TA zero handling, and the univariate, paired, and regression rolling engines match the specified O(1) post-warm-up transitions using `Float` state.
- `crates/ta-core/src/statistic/variance.rs:23-230` — VAR/STDDEV compact, default, vec, indicator, streaming, checked-streaming, and reset surfaces match the plan, including `variance <= TA_EPSILON` for STDDEV.
- `crates/ta-core/src/statistic/correl.rs:20-142` and `crates/ta-core/src/statistic/beta.rs:19-178` — paired validation order, CORREL degeneracy behavior, BETA adjacent-return state, extra lookback, and failure-before-mutation behavior match the planned contracts.
- `crates/ta-core/src/statistic/regression.rs:29-203` — all five regression projections share the pinned rolling fit and expose the required compact, vec, struct, streaming, checked, and reset APIs.
- `crates/ta-core/tests/statistic.rs:38-859` — 22 tests cover numerical oracles, f64/f32 cancellation, validation precedence, output non-mutation, ring wraps, reset replay, invalid ticks, compact ranges, and padded alignment.
- `crates/ta-core/src/inventory.rs:12,284-292` and `crates/ta-core/tests/inventory.rs:73-276` — the count is 54, all nine statistic records are implemented, and every statistic struct has batch/streaming trait assertions.
- `crates/ta-benchmarks/benches/first_tranche.rs:27-29,518-683` — the registered matrix is 3 sizes × 4 periods × 9 functions × 2 surfaces = 216 unique cases with period-bearing IDs and the intended allocation/black-box boundaries.
- `.rpiv/artifacts/designs/2026-07-16_19-30-52_statistic-functions.md:547` — the parent design now uses the finalized `<= TA_EPSILON` equality guard and is synchronized with the plan, code, and boundary tests.

#### Deviations from Plan:

None. Implementation is a faithful realization of the plan.

#### Pattern Conformance:

- ✓ Compact kernels, padded wrappers, same-named structs, typed constructors, `Indicator`, `StreamingIndicator`, `Resettable`, and NaN warm-up behavior follow established `ta-core` indicator conventions.
- ✓ Validation completes before output writes or streaming state changes, matching existing moving-average, volatility, and volume implementations.
- ✓ Private leaf modules with explicit facade exports, synchronized inventory assertions, and registered Criterion groups follow repository conventions.
- Minor observation: statistic benchmarks use `bench_function` with composite IDs rather than older groups' `bench_with_input`; this is an acceptable variation, not a deviation.

### Manual Testing Required:

1. API and state contracts:
   - [x] Inspect validation precedence, untouched output/state on errors, NaN padding, lookbacks, reset replay, conditional `alloc`/`std` imports, and `Float`-typed state.
   - [x] Confirm strict threshold endpoints, VAR/STDDEV semantics, paired direction/degeneracy, and all regression projections against the locked tests.
2. Inventory and performance wiring:
   - [x] Confirm the dynamic implemented count is 54, all nine statistic types implement both traits, and the Criterion matrix contains 216 size/period/surface combinations.
   - [x] Confirm fixtures and reusable compact outputs are outside `b.iter()`, vec allocation remains inside, and benchmark inputs/results are black-boxed.
3. Artifact consistency:
   - [x] Confirm the parent design, final plan, implementation, and equality-boundary tests all use `<= TA_EPSILON` for STDDEV.

### Recommendations:

- Ready to commit — implementation is complete and validated.
