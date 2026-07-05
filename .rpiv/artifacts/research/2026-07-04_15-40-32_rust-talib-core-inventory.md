---
date: 2026-07-04T15:40:32+0800
author: unknown
commit: 896d1d7
branch: main
repository: fast-ta
topic: "继续完成rust版本的talib https://github.com/TA-Lib/ta-lib-python"
tags: [research, codebase, ta-core, talib, indicators, parity]
status: ready
last_updated: 2026-07-04T15:40:32+0800
last_updated_by: unknown
---

# Research: 继续完成rust版本的talib https://github.com/TA-Lib/ta-lib-python

## Research Question
继续完成 Rust 版本的 TA-Lib, using `TA-Lib/ta-lib-python` as the compatibility reference, with the developer clarification that every indicator algorithm should be implemented in Rust `ta-core`.

## Summary
`fast-ta` is currently a four-crate workspace where `ta-core` owns algorithm logic, while `ta-py`, `ta-wasm`, and `ta-benchmarks` are adapters/tooling. Live Rust-core indicator coverage is only `ta_core::overlap::SMA`: `ta-core` exposes `overlap`, and `overlap/mod.rs` only re-exports `SMA` (`crates/ta-core/src/lib.rs:32-41`, `crates/ta-core/src/overlap/mod.rs:4-6`). The official current TA-Lib inventory is 161 functions across 10 groups; local planning docs cover only part of that inventory and misclassify or omit several functions. The load-bearing path for continuing the project is to stabilize the Rust `Indicator`/error/parity contract first, then expand `ta-core` module families to cover the full TA-Lib function inventory; Python/WASM bindings remain adapter layers after core APIs exist.

## Detailed Findings

### Current Rust Core Shape
- The workspace members are `ta-core`, `ta-py`, `ta-wasm`, and `ta-benchmarks`; default members are `ta-core`, `ta-wasm`, and `ta-benchmarks`, not `ta-py` (`Cargo.toml:3-13`).
- `ta-core` public facade currently exports `error`, `overlap`, `simd`, `traits`, and `types`, plus `Result`, `TalibError`, `Indicator`, `Resettable`, and `Float` (`crates/ta-core/src/lib.rs:32-41`).
- `overlap` is the only indicator family wired into live source, and it only contains `mod sma; pub use sma::SMA;` (`crates/ta-core/src/overlap/mod.rs:4-6`).
- `SMA` is the only public indicator type in the source tree (`crates/ta-core/src/overlap/sma.rs:35`, `crates/ta-core/src/overlap/sma.rs:95-106`).
- SIMD helpers exist as primitives, not TA-Lib public indicator APIs: `simd::sum` and `simd::dot_product` are re-exported from dispatch (`crates/ta-core/src/simd/mod.rs:37-38`).

### Indicator Contract and SMA Template
- `Indicator<const N: usize = 1>` defines associated `Input` and `Output` types, `lookback()`, `compute_to_vec() -> Result<Vec<_>>`, and `next()` (`crates/ta-core/src/traits.rs:76-219`).
- The trait docs define three NaN contexts: input NaN is currently documented as an error, batch output NaN is warm-up, and streaming `next()` NaN is warm-up (`crates/ta-core/src/traits.rs:13-15`).
- `Float` is feature-selected: `f32` under feature `f32`, otherwise default `f64` (`crates/ta-core/src/types.rs:26-33`).
- `Result<T>` is `core::result::Result<T, TalibError>` (`crates/ta-core/src/error.rs:275`), and `TalibError` has variants for invalid input, invalid period, insufficient data, invalid parameter, computation error, and not implemented (`crates/ta-core/src/error.rs:13-55`).
- `SMA::compute_to_vec()` preserves full input length by pre-filling `vec![Float::NAN; inputs.len()]`, then calls `compute_sma()` and returns `Ok(result)` (`crates/ta-core/src/overlap/sma.rs:104-107`).
- `compute_sma()` writes the first valid output at `outputs[period - 1]`, leaving the warm-up prefix as NaN (`crates/ta-core/src/overlap/sma.rs:10-31`).
- `SMA::new()` currently returns `Self` and panics on zero period via `assert!`, instead of returning `TalibError::invalid_period()` (`crates/ta-core/src/overlap/sma.rs:51-55`, `crates/ta-core/src/error.rs:93-97`).
- `SMA::next()` reads and writes `self.buffer[self.index]`, but `AVec::with_capacity(64, period)` creates capacity without initialized length; current streaming state can panic on first indexed access (`crates/ta-core/src/overlap/sma.rs:55`, `crates/ta-core/src/overlap/sma.rs:111-140`).
- The trait docs mention a zero-copy `compute` method in prose, but the live trait only defines `compute_to_vec()` and `next()` (`crates/ta-core/src/traits.rs:35-42`, `crates/ta-core/src/traits.rs:151`, `crates/ta-core/src/traits.rs:219`).

### SIMD, Precision, and Feature Boundaries
- `simd::dispatch` exposes function pointers for `sum` and `dot_product`, initialized once through `OnceLock` (`crates/ta-core/src/simd/dispatch.rs:47-77`, `crates/ta-core/src/simd/dispatch.rs:174-240`).
- Runtime dispatch selects AVX-512/AVX2 on x86_64 with `std`, NEON on AArch64, SIMD128 on wasm32, or scalar fallback (`crates/ta-core/src/simd/dispatch.rs:95-156`).
- `SMA` bypasses runtime dispatch and directly uses `FastFloat`/`LANES` for the initial window, then scalar sliding-window updates (`crates/ta-core/src/overlap/sma.rs:3-5`, `crates/ta-core/src/overlap/sma.rs:16-31`).
- `FastFloat` is `wide::f32x16` in `f32` builds and `wide::f64x8` otherwise; `LANES` derives from SIMD vector size and `Float` size (`crates/ta-core/src/simd/mod.rs:40-47`).
- `ta-core` advertises `no_std` via `#![cfg_attr(not(feature = "std"), no_std)]`, but `once_cell` is optional and not enabled by a feature even though `dispatch.rs` imports it when `std` is disabled (`crates/ta-core/src/lib.rs:17`, `crates/ta-core/src/simd/dispatch.rs:12-13`, `crates/ta-core/Cargo.toml:13-23`).
- The default `Float` behavior is TA-Lib-compatible `f64`, because default features are `f64` and `std` (`crates/ta-core/Cargo.toml:14`, `crates/ta-core/src/types.rs:32-33`).

### Official TA-Lib Function Inventory for `ta-core`
The current official TA-Lib inventory is 161 functions across 10 groups. Current ta-lib-python and TA-Lib metadata sources include functions missing from some older indexes, notably `ACCBANDS`, `AVGDEV`, and `IMI`.

#### Overlap Studies — 18 functions
- `ACCBANDS` — HLC input, 3 outputs: upperband, middleband, lowerband.
- `BBANDS` — real input, 3 outputs: upperband, middleband, lowerband.
- `DEMA`, `EMA`, `HT_TRENDLINE`, `KAMA`, `MA`, `MAMA`, `MAVP`, `MIDPOINT`, `MIDPRICE`, `SAR`, `SAREXT`, `SMA`, `T3`, `TEMA`, `TRIMA`, `WMA`.
- Multi-output overlap functions: `ACCBANDS`, `BBANDS`, `MAMA`.

#### Momentum Indicators — 31 functions
- `ADX`, `ADXR`, `APO`, `AROON`, `AROONOSC`, `BOP`, `CCI`, `CMO`, `DX`, `IMI`, `MACD`, `MACDEXT`, `MACDFIX`, `MFI`, `MINUS_DI`, `MINUS_DM`, `MOM`, `PLUS_DI`, `PLUS_DM`, `PPO`, `ROC`, `ROCP`, `ROCR`, `ROCR100`, `RSI`, `STOCH`, `STOCHF`, `STOCHRSI`, `TRIX`, `ULTOSC`, `WILLR`.
- Multi-output momentum functions: `AROON`, `MACD`, `MACDEXT`, `MACDFIX`, `STOCH`, `STOCHF`, `STOCHRSI`.

#### Volume Indicators — 3 functions
- `AD`, `ADOSC`, `OBV`.
- Inputs are HLCV for `AD`/`ADOSC` and close+volume for `OBV`.

#### Volatility Indicators — 3 functions
- `ATR`, `NATR`, `TRANGE`.
- Inputs are high, low, close.

#### Price Transform — 5 functions
- `AVGDEV`, `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`.
- Inputs vary between real, OHLC, HL, and HLC.

#### Cycle Indicators — 5 functions
- `HT_DCPERIOD`, `HT_DCPHASE`, `HT_PHASOR`, `HT_SINE`, `HT_TRENDMODE`.
- Multi-output cycle functions: `HT_PHASOR`, `HT_SINE`; integer output: `HT_TRENDMODE`.

#### Pattern Recognition — 61 functions
- All take OHLC input and return integer signal arrays.
- Functions: `CDL2CROWS`, `CDL3BLACKCROWS`, `CDL3INSIDE`, `CDL3LINESTRIKE`, `CDL3OUTSIDE`, `CDL3STARSINSOUTH`, `CDL3WHITESOLDIERS`, `CDLABANDONEDBABY`, `CDLADVANCEBLOCK`, `CDLBELTHOLD`, `CDLBREAKAWAY`, `CDLCLOSINGMARUBOZU`, `CDLCONCEALBABYSWALL`, `CDLCOUNTERATTACK`, `CDLDARKCLOUDCOVER`, `CDLDOJI`, `CDLDOJISTAR`, `CDLDRAGONFLYDOJI`, `CDLENGULFING`, `CDLEVENINGDOJISTAR`, `CDLEVENINGSTAR`, `CDLGAPSIDESIDEWHITE`, `CDLGRAVESTONEDOJI`, `CDLHAMMER`, `CDLHANGINGMAN`, `CDLHARAMI`, `CDLHARAMICROSS`, `CDLHIGHWAVE`, `CDLHIKKAKE`, `CDLHIKKAKEMOD`, `CDLHOMINGPIGEON`, `CDLIDENTICAL3CROWS`, `CDLINNECK`, `CDLINVERTEDHAMMER`, `CDLKICKING`, `CDLKICKINGBYLENGTH`, `CDLLADDERBOTTOM`, `CDLLONGLEGGEDDOJI`, `CDLLONGLINE`, `CDLMARUBOZU`, `CDLMATCHINGLOW`, `CDLMATHOLD`, `CDLMORNINGDOJISTAR`, `CDLMORNINGSTAR`, `CDLONNECK`, `CDLPIERCING`, `CDLRICKSHAWMAN`, `CDLRISEFALL3METHODS`, `CDLSEPARATINGLINES`, `CDLSHOOTINGSTAR`, `CDLSHORTLINE`, `CDLSPINNINGTOP`, `CDLSTALLEDPATTERN`, `CDLSTICKSANDWICH`, `CDLTAKURI`, `CDLTASUKIGAP`, `CDLTHRUSTING`, `CDLTRISTAR`, `CDLUNIQUE3RIVER`, `CDLUPSIDEGAP2CROWS`, `CDLXSIDEGAP3METHODS`.

#### Statistic Functions — 9 functions
- `BETA`, `CORREL`, `LINEARREG`, `LINEARREG_ANGLE`, `LINEARREG_INTERCEPT`, `LINEARREG_SLOPE`, `STDDEV`, `TSF`, `VAR`.
- `BETA` and `CORREL` use paired real series; the others use real input.

#### Math Transform — 15 functions
- `ACOS`, `ASIN`, `ATAN`, `CEIL`, `COS`, `COSH`, `EXP`, `FLOOR`, `LN`, `LOG10`, `SIN`, `SINH`, `SQRT`, `TAN`, `TANH`.

#### Math Operators — 11 functions
- `ADD`, `DIV`, `MAX`, `MAXINDEX`, `MIN`, `MININDEX`, `MINMAX`, `MINMAXINDEX`, `MULT`, `SUB`, `SUM`.
- Multi-output math operators: `MINMAX`, `MINMAXINDEX`; integer outputs: `MAXINDEX`, `MININDEX`, `MINMAXINDEX`.

### Local Plan Coverage and Gaps
- `.sisyphus/IMPLEMENTATION_PLAN.md` claims 150+ indicators and 100% TA-Lib compatibility (`.sisyphus/IMPLEMENTATION_PLAN.md:49-58`), but source implements only `SMA` (`crates/ta-core/src/overlap/mod.rs:4-6`).
- Phase 2 lists 16 overlap/moving-average-like entries (`.sisyphus/IMPLEMENTATION_PLAN.md:2321-2342`), but official TA-Lib overlap has 18 functions and several listed Phase 2 entries are not TA-Lib overlap functions.
- The plan row for SMA maps it to `TA_MA`, but official TA-Lib has both `SMA` and generic `MA` semantics; live source implements `SMA` (`.sisyphus/IMPLEMENTATION_PLAN.md:2327`, `crates/ta-core/src/overlap/sma.rs:10`).
- Missing from main overlap plan: `ACCBANDS`, `BBANDS`, `MA`, `HT_TRENDLINE`, `MAVP`, `MIDPOINT`, `MIDPRICE`, `SAREXT`.
- `SAR` is listed under Phase 4 volatility, but TA-Lib classifies it under Overlap Studies (`.sisyphus/IMPLEMENTATION_PLAN.md:3111`).
- Phase 3 momentum lists 33 items, but includes `ATR`, `NATR`, and `TRANGE`, which are TA-Lib volatility functions (`.sisyphus/IMPLEMENTATION_PLAN.md:2698-2735`).
- Missing from the Phase 3 momentum list: `IMI` and `ROCR100` (`.sisyphus/IMPLEMENTATION_PLAN.md:2704-2732`).
- Phase 4 volume/volatility lists `OBV`, `AD`, `NATR`, `ATR`, `TRANGE`, and `SAR`; it omits `ADOSC` and duplicates volatility functions already listed in Phase 3 (`.sisyphus/IMPLEMENTATION_PLAN.md:3100-3111`).
- Price transforms `AVGPRICE`, `MEDPRICE`, and `TYPPRICE` are listed under overlap, while official TA-Lib groups them as Price Transform (`.sisyphus/IMPLEMENTATION_PLAN.md:2340-2342`).
- `WCLPRICE` and `AVGDEV` are missing from the local plan.
- `VAR` is described as “Variable Moving Average” in Phase 2, but official TA-Lib `VAR` is statistical variance (`.sisyphus/IMPLEMENTATION_PLAN.md:2339`, `.sisyphus/IMPLEMENTATION_PLAN.md:2481`).
- Non-TA-Lib extras in the plan include `WWMA`, `HMA`, and `VWAP` (`.sisyphus/IMPLEMENTATION_PLAN.md:2336-2338`).
- Completely missing local plan groups: Cycle Indicators, Pattern Recognition, Math Transform, Math Operators, and most Statistic Functions.
- Plan status is internally inconsistent: Phase 2/3/4 status markers say not started, while later acceptance checklists mark all indicators complete (`.sisyphus/IMPLEMENTATION_PLAN.md:2303-2307`, `.sisyphus/IMPLEMENTATION_PLAN.md:2500`, `.sisyphus/IMPLEMENTATION_PLAN.md:2680-2684`, `.sisyphus/IMPLEMENTATION_PLAN.md:2992`, `.sisyphus/IMPLEMENTATION_PLAN.md:3082-3086`, `.sisyphus/IMPLEMENTATION_PLAN.md:3206`).

### ta-lib-python Compatibility Facts Relevant to `ta-core`
- The Function API uses uppercase function names such as `SMA`, `EMA`, `RSI`, `MACD`, and `ATR`.
- Outputs are same-length arrays with leading NaNs for the lookback period; multi-output functions return multiple same-length arrays in documented order.
- `MACD` returns `macd`, `macdsignal`, and `macdhist`; `BBANDS` returns upper/middle/lower bands; `MAMA` returns `mama` and `fama`.
- Single-series functions use `real` in generated signatures, while the Abstract API typically maps single-price indicators to `close` input by default.
- TA-Lib/ta-lib-python NaN behavior is not the same as strict Rust validation: input NaNs can propagate through outputs rather than rejecting the whole call.
- Canonical parity sources include TA-Lib C regression tests and generated ta-lib-python wrappers; C API compact output through `outBegIdx`/`outNbElement` is expanded by ta-lib-python into same-length arrays with leading NaNs.

### Python and WASM Adapter State
- `ta-py` currently exposes only `hello_world()` through the module initializer (`crates/ta-py/src/lib.rs:12-20`).
- `ta-py` has dependencies needed for binding work: `ta-core`, `pyo3`, and `numpy` (`crates/ta-py/Cargo.toml:18-21`).
- `ta-wasm` currently exposes only `hello_world()` and `add(a: f64, b: f64)` (`crates/ta-wasm/src/lib.rs:9-17`).
- `ta-wasm` depends on `ta-core` with `std`, plus `wasm-bindgen`, `web-sys`, and optional panic hook (`crates/ta-wasm/Cargo.toml:21-25`).
- These crates are adapter layers; current research scope treats them as consumers of future `ta-core` APIs, not places for indicator algorithms.

### Verification Coverage and Parity Gap
- CI runs `cargo test --all-targets --verbose` and `cargo clippy --all-targets --verbose` from `crates/ta-core`, plus rustfmt from the workspace root (`.github/workflows/ci.yml:62-68`).
- The CI matrix labels multiple targets, but current cargo commands do not pass `--target`, so target labels are not directly wired into compilation (`.github/workflows/ci.yml:21-45`, `.github/workflows/ci.yml:62-68`).
- `crates/ta-core` has source-local tests for errors and SIMD primitives but no TA-Lib golden-vector integration tests (`crates/ta-core/src/error.rs:277`, `crates/ta-core/src/simd/dispatch.rs:243`).
- `crates/ta-benchmarks/benches/basic.rs` benchmarks plain Rust addition/vector sum, not `ta-core` indicators (`crates/ta-benchmarks/benches/basic.rs:23-41`).
- README claims 100% TA-Lib C numerical parity with ε < 1e-10 (`README:13`), and the QA plan requires every indicator to compare against TA-Lib C, including NaN/Inf behavior (`.sisyphus/QUALITY_ASSURANCE_PLAN.md:65-73`).
- No `crates/ta-core/tests/` parity fixture directory exists in the live source tree.

## Code References
- `Cargo.toml:1-16` — Workspace members, default members, and shared `wide` dependency.
- `crates/ta-core/Cargo.toml:13-28` — `ta-core` features, SIMD/aligned dependencies, and dev-dependencies.
- `crates/ta-core/src/lib.rs:17-41` — `no_std` attribute, public modules, and facade re-exports.
- `crates/ta-core/src/types.rs:26-33` — `Float` alias switches between `f32` and default `f64`.
- `crates/ta-core/src/error.rs:13-55` — `TalibError` variants.
- `crates/ta-core/src/error.rs:73-115` — invalid input/period and insufficient-data constructors.
- `crates/ta-core/src/error.rs:275` — `Result<T>` alias.
- `crates/ta-core/src/traits.rs:13-15` — Current NaN semantics documentation.
- `crates/ta-core/src/traits.rs:76-219` — Live `Indicator` trait shape.
- `crates/ta-core/src/overlap/mod.rs:4-6` — Only SMA is wired into the overlap family.
- `crates/ta-core/src/overlap/sma.rs:10-31` — Batch SMA kernel.
- `crates/ta-core/src/overlap/sma.rs:51-65` — `SMA::new()` constructor and state allocation.
- `crates/ta-core/src/overlap/sma.rs:95-107` — `SMA` batch implementation.
- `crates/ta-core/src/overlap/sma.rs:111-140` — `SMA` streaming implementation.
- `crates/ta-core/src/simd/mod.rs:37-47` — Public SIMD dispatch exports and `FastFloat`/`LANES` definitions.
- `crates/ta-core/src/simd/dispatch.rs:47-77` — Dispatch table and global initialization state.
- `crates/ta-core/src/simd/dispatch.rs:95-156` — Architecture dispatch selection.
- `crates/ta-core/src/simd/dispatch.rs:202-240` — Public `sum` and `dot_product` wrappers.
- `crates/ta-py/src/lib.rs:12-20` — Placeholder Python module exports.
- `crates/ta-py/Cargo.toml:18-21` — Python adapter dependencies.
- `crates/ta-wasm/src/lib.rs:9-17` — Placeholder WASM exports.
- `crates/ta-wasm/Cargo.toml:21-28` — WASM adapter dependencies and tests.
- `crates/ta-benchmarks/benches/basic.rs:23-41` — Current benchmark does not exercise indicators.
- `.github/workflows/ci.yml:62-68` — Current CI core test/clippy commands.
- `.sisyphus/IMPLEMENTATION_PLAN.md:2321-2342` — Local Phase 2 overlap list.
- `.sisyphus/IMPLEMENTATION_PLAN.md:2698-2735` — Local Phase 3 momentum list.
- `.sisyphus/IMPLEMENTATION_PLAN.md:3100-3111` — Local Phase 4 volume/volatility list.
- `.sisyphus/QUALITY_ASSURANCE_PLAN.md:65-73` — Stated TA-Lib C parity requirements.
- `README:13` — README precision-parity claim.

## Integration Points

### Inbound References
- `crates/ta-py/Cargo.toml:18-21` — Python adapter depends on `ta-core`, `pyo3`, and `numpy` but live source only exposes `hello_world()`.
- `crates/ta-wasm/Cargo.toml:21-25` — WASM adapter depends on `ta-core` and wasm-bindgen stack but live source only exposes placeholders.
- `crates/ta-benchmarks/Cargo.toml:18-22` — Benchmark crate depends on `ta-core`, though current benchmark file does not call it.
- `Cargo.toml:9-13` — Default workspace builds exercise `ta-core`, `ta-wasm`, and `ta-benchmarks`; `ta-py` is explicit-only.
- `.github/workflows/ci.yml:62-68` — CI currently gates `ta-core` tests/clippy and workspace formatting.

### Outbound Dependencies
- `crates/ta-core/Cargo.toml:21` and `Cargo.toml:16` — `wide` supplies portable SIMD vector types.
- `crates/ta-core/Cargo.toml:22` — `once_cell` is intended for no-std dispatch initialization but is optional and not feature-enabled.
- `crates/ta-core/Cargo.toml:23` — `aligned-vec` supplies `AVec`, currently used by `SMA` streaming state.
- `crates/ta-core/src/simd/dispatch.rs:12-18` — Dispatch chooses `OnceLock`/`OnceCell` depending on `std`.
- `crates/ta-core/src/simd/dispatch.rs:19-29` — Dispatch imports scalar and cfg-gated arch backends.

### Infrastructure Wiring
- `Cargo.toml:1-16` — Workspace and shared dependency wiring.
- `crates/ta-core/src/lib.rs:32-41` — Root module export surface; every new public Rust-core family must appear through this facade or a child module it exposes.
- `crates/ta-core/src/overlap/mod.rs:4-6` — Current family wiring pattern: private implementation file, public type re-export.
- `.github/workflows/ci.yml:62-68` — Current test/clippy gate where future `crates/ta-core/tests/` parity tests would be executed.

## Architecture Insights
- `ta-core` is the correct home for every TA-Lib algorithm; binding crates should translate inputs/outputs/errors only.
- Current public module layout is not sufficient for full TA-Lib coverage; only `overlap` exists, while full coverage needs additional Rust-core families for momentum, volume, volatility, price transform, cycle, pattern recognition, statistic, math transform, and math operators.
- The live `Indicator` trait already anticipates multi-output indicators through const generic `N` and associated `Output`, but no live multi-output implementation exists.
- Full TA-Lib coverage requires integer outputs as well as `Float` outputs because pattern recognition and index functions return integer signal/index arrays.
- Current `SMA` is useful as a batch-output shape precedent but not a safe constructor/streaming/error precedent until panic and buffer issues are resolved.
- The current strict input-NaN contract conflicts with ta-lib-python parity behavior; this is a design-level compatibility decision before golden tests are locked.
- Official TA-Lib group membership differs from local plans; `SAR`, `ATR/NATR/TRANGE`, price transforms, and `VAR` need canonical classification before module names become public API.
- README and QA docs already commit to TA-Lib C parity, but live tests do not enforce it; core integration tests with golden vectors are the missing verification surface.

## Precedents & Lessons
4 similar past change clusters analyzed.

### Precedent: Indicator traits and SMA implementation churn
**Commit(s)**: `3683ed3` — "任务 1.3: 实现核心 Traits" (2026-01-29); `491d9e8` — "任务 1.3: 实现核心 Traits (更新版-强制SIMD加速+零拷贝)" (2026-01-29)
**Blast radius**: 5 files across 3 layers
  core API/ — `traits.rs`, `lib.rs` exports changed
  overlap/ — `overlap/sma.rs`, `overlap/mod.rs` added/rewritten
  benchmarks/ — SMA benchmark added

**Follow-up fixes**:
- `bd5b5a3` — "1.9" (2026-02-02) — deleted `overlap/sma.rs` and SMA benchmark while adding CI/clippy.
- `6528112` — "sma" (2026-02-03) — re-added simplified SMA and changed `Indicator::compute` signature.
- `f833937` — "1" (2026-02-04) — rewrote SMA streaming/NaN semantics and added `aligned-vec`.
- `3267f43` — "1" (2026-02-04) — reintroduced `compute(...) -> Result<usize>` with `todo!()`.
- `896d1d7` — "sma" (2026-02-05) — removed compute/stream trait additions and deleted SIMD integration tests.

**Lessons from docs**:
- No existing `.rpiv/artifacts/` docs were found.

**Takeaway**: Stabilize `Indicator`, warm-up/NaN semantics, and TA-Lib parity tests before adding many indicators.

### Precedent: SIMD framework and runtime dispatch
**Commit(s)**: `90f894a` — "任务 1.3: SIMD 模块基础框架" (2026-01-30); `a4a9385` — "任务 1.4: SIMD 运行时调度系统" (2026-01-30); `51c3cad` — "任务 1.5: x86_64 SIMD 实现" (2026-01-30)
**Blast radius**: 16 files across 4 layers
  simd core/ — `simd/mod.rs`, `simd/types.rs`, `simd/dispatch.rs` added/rewritten
  arch/ — x86_64 AVX2/AVX512 modules added
  tests/ — x86 SIMD tests added
  cargo/ — `Cargo.toml`, `Cargo.lock`, `ta-core/Cargo.toml` changed

**Follow-up fixes**:
- `34741cb` — "Fix task 1.5: Remove dead cfg guards and enable runtime SIMD dispatch" (2026-02-01) — compile-time guards blocked runtime dispatch.
- `50ff6ee` — "1" (2026-01-31) — added dead-code/import allowances and replaced unavailable `wide::*::from_slice_unaligned`.
- `d36627c` — "1.7" (2026-02-01) — enabled ARM64/WASM dispatch paths after earlier scalar fallback.
- `29fb091` — "update" (2026-02-03) — large no_std/std and SIMD API simplification.
- `896d1d7` — "sma" (2026-02-05) — deleted aarch64/WASM/x86 SIMD integration tests.

**Lessons from docs**:
- No existing `.rpiv/artifacts/` docs were found.

**Takeaway**: Preserve cross-arch SIMD tests while changing dispatch or indicator kernels.

### Precedent: Workspace, bindings, benchmarks, and CI setup
**Commit(s)**: `c14f0f5` — "任务 1.1: 创建 Workspace 结构" (2026-01-29); `bd5b5a3` — "1.9" (2026-02-02)
**Blast radius**: 27 files across 6 layers
  workspace/ — root `Cargo.toml`, `Cargo.lock`
  core/ — `ta-core` lib/types/error/overlap touched
  py/ — `ta-py/src/lib.rs`, later `crates/ta-py/types.rs`
  wasm/ — `ta-wasm/src/lib.rs`, `ta-wasm/Cargo.toml`
  benchmarks/ — `basic.rs`, SMA bench added/removed
  ci/ — `.github/workflows/ci.yml`, rustfmt, clippy

**Follow-up fixes**:
- `ee346da` — "remove target from git" (2026-01-29) — build artifacts were accidentally committed.
- `29fb091` — "update" (2026-02-03) — removed `ta-py`/`ta-wasm` from workspace members/default-members.
- `bd5b5a3` — "1.9" (2026-02-02) — CI addition coincided with deleting SMA implementation and SMA bench.

**Lessons from docs**:
- No existing `.rpiv/artifacts/` docs were found.

**Takeaway**: Workspace-wide changes need explicit decisions about Python/WASM/benchmark inclusion.

### Precedent: Error system and no_std/std boundary
**Commit(s)**: `2335e94` — "任务 1.2: 实现错误类型系统" (2026-01-29)
**Blast radius**: 3 files across 2 layers
  core error/ — `error.rs` added
  core exports/ — `lib.rs` exports changed
  cargo/ — `ta-core/Cargo.toml` feature/dependency changes

**Follow-up fixes**:
- `29fb091` — "update" (2026-02-03) — changed `#![no_std]` to `cfg_attr`, moved alloc/std imports, adjusted error imports.
- `bd5b5a3` — "1.9" (2026-02-02) — CI/clippy introduced stricter validation surface.

**Lessons from docs**:
- No existing `.rpiv/artifacts/` docs were found.

**Takeaway**: Error/type changes need std and no-std feature checks.

### Composite Lessons
- Trait/SMA API churn recurred around `compute`, `stream`, `next`, NaN warm-up, and `Result` semantics.
- SIMD work repeatedly broke around cfg/runtime dispatch and cross-arch tests.
- CI/workspace changes caused large rollbacks/removals; keep py/wasm/bench scope explicit.
- TA-Lib parity and SIMD integration tests should not be deleted during refactors.

## Historical Context (from `.rpiv/artifacts/`)
- None found.

## External Sources
- [Official TA-Lib function index](https://ta-lib.org/functions/) — canonical current function inventory.
- [ta-lib-python function groups](https://ta-lib.github.io/ta-lib-python/funcs.html) — Python Function API grouping.
- [ta-lib-python overlap studies](https://ta-lib.github.io/ta-lib-python/func_groups/overlap_studies.html) — overlap function list and signatures.
- [ta-lib-python momentum indicators](https://ta-lib.github.io/ta-lib-python/func_groups/momentum_indicators.html) — momentum function list and signatures.
- [ta-lib-python volume indicators](https://ta-lib.github.io/ta-lib-python/func_groups/volume_indicators.html) — volume function list and signatures.
- [ta-lib-python volatility indicators](https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html) — volatility function list and signatures.
- [ta-lib-python Abstract API](https://ta-lib.github.io/ta-lib-python/abstract.html) — input naming/model conventions.
- [TA-Lib function metadata XML](https://raw.githubusercontent.com/TA-Lib/ta-lib/main/ta_func_api.xml) — current metadata source for functions/parameters.
- [ta-lib-python generated function wrappers](https://raw.githubusercontent.com/TA-Lib/ta-lib-python/master/talib/_func.pxi) — generated wrapper behavior/docstrings.
- [TA-Lib C/C++ API docs](https://ta-lib.org/api/) — C output model with `outBegIdx` and `outNbElement`.

## Developer Context
**Q (`crates/ta-core/src/traits.rs:13-15`): The current trait says input `NaN` should reject the whole operation, but ta-lib-python parity propagates input NaNs in outputs. Which behavior should new indicators and bindings treat as canonical?**
A: Developer clarified the primary requirement: “我要求是使用rust实现每一个指标。” This resolves the algorithm location (`ta-core`) but leaves exact NaN policy as an open compatibility decision.

**Q (`crates/ta-core/src/lib.rs:32-37`): The scan found only `overlap::SMA` live in Rust core, while the local plan covers only part of TA-Lib. Should the research add a full TA-Lib inventory and focus on core scope?**
A: Developer requested: “先补talib全函数清单，聚焦ta-core。py和wasm先补绑定”. This artifact incorporates the full TA-Lib inventory and treats `ta-core` as the primary algorithm scope; Python/WASM are documented as adapter surfaces.

**Q (`crates/ta-core/src/lib.rs:32-37`): Updated scan complete — write the research doc now, or adjust first?**
A: Write the doc.

## Related Research
- None found.

## Open Questions
- Should default `ta-core` indicator functions follow ta-lib-python NaN propagation exactly, or keep the current strict “input NaN is error” trait contract (`crates/ta-core/src/traits.rs:13-15`)?
- Should public Rust type names use current uppercase names like `SMA`, or plan-style names like `Sma` for all new indicators (`crates/ta-core/src/overlap/mod.rs:6`, `.sisyphus/IMPLEMENTATION_PLAN.md:2493-2495`)?
- Should non-TA-Lib extras already listed locally (`WWMA`, `HMA`, `VWAP`) remain in scope after the official 161-function TA-Lib inventory is completed (`.sisyphus/IMPLEMENTATION_PLAN.md:2336-2338`)?
- Should Python/WASM bindings be implemented immediately alongside each new core function, or after a stable Rust-core module/API layer is established?
