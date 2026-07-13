---
date: 2026-07-09T21:23:34+0800
author: unknown
commit: c6f3630
branch: main
repository: fast-ta
topic: "实现Volatility分组指标"
tags: [research, codebase, ta-core, volatility, atr, natr, trange]
status: ready
last_updated: 2026-07-09T21:23:34+0800
last_updated_by: unknown
---

# Research: 实现Volatility分组指标

## Research Question
实现 Volatility 分组指标：在 `ta-core` 中实现 TA-Lib Volatility Indicators 组的 `ATR`、`NATR`、`TRANGE`，并理解现有 inventory、module facade、HLC input、compact output、period/lookback、streaming/reset、tests 的集成契约。

## Summary
`ta-core` 已在官方 TA-Lib inventory 中预留 Volatility 分组和 `ATR/NATR/TRANGE` 三条记录，但没有实际 `volatility` 模块。实现该组应沿用现有 `price_transform` facade 形状：uppercase free function、`*_vec` padded wrapper、uppercase struct、`Indicator` + `StreamingIndicator` + `Resettable`（stateful 时）实现，并在 root `lib.rs` 暴露 `pub mod volatility;`。HLC SoA 输入、同长/finite 验证可以直接复用 `TYPPRICE` 模式；period、compact output、NaN padding 和 fallible constructors 可复用 `AVGDEV` 模式；ATR 的 Wilder smoothing 和 streaming alignment 可借鉴 `EMA` 的 seed + recursive-state pattern。TA-Lib 外部语义确认：`TRANGE` 首个有效输出从原始索引 `1` 开始；默认 unstable period 为 0 时，`ATR/NATR` 首个有效输出索引为 `timeperiod`；本研究记录开发者决定本 tranche 不新增 unstable-period 全局配置，仅实现 TA-Lib 默认行为。

## Detailed Findings

### Inventory and status ledger
- `FunctionGroup` already includes `VolatilityIndicators` (`crates/ta-core/src/inventory.rs:16-24`).
- Official group ordering already includes Volatility between Volume and Price Transform (`crates/ta-core/src/inventory.rs:41-47`).
- Official group count for Volatility is fixed at `3`, so implementing the group does not change `FUNCTION_COUNT` or group counts (`crates/ta-core/src/inventory.rs:71-78`).
- `FunctionGroup::rust_module()` already maps Volatility to `"volatility"` (`crates/ta-core/src/inventory.rs:87-93`).
- Inventory records for `ATR`, `NATR`, and `TRANGE` currently exist as `Planned` (`crates/ta-core/src/inventory.rs:206-208`).
- `IMPLEMENTED_FUNCTION_COUNT` is currently `39`; implementing all three Volatility functions makes the expected implemented count `42` (`crates/ta-core/src/inventory.rs:9-12`).
- `function(name)` performs exact uppercase lookup over `TALIB_FUNCTIONS`; after status changes, `function("ATR")`, `function("NATR")`, and `function("TRANGE")` will resolve to the same records with `Implemented` status (`crates/ta-core/src/inventory.rs:324-325`).
- Inventory tests lock both implemented count and status: implemented-name array length must match `IMPLEMENTED_FUNCTION_COUNT`, and actual `is_implemented()` count must match the constant (`crates/ta-core/tests/inventory.rs:69-124`).
- Current deferred test explicitly expects `ATR` to remain `Planned`; that assertion becomes stale once Volatility is implemented (`crates/ta-core/tests/inventory.rs:217-222`).

### Module facade and exports
- Existing group facade pattern is `price_transform`: private implementation modules plus public `pub use` re-exports (`crates/ta-core/src/price_transform/mod.rs:6-16`).
- Root crate currently exposes `math_operators`, `math_transform`, `overlap`, and `price_transform`, but no `volatility` module (`crates/ta-core/src/lib.rs:35-49`).
- Inventory metadata naming does not create Rust modules; `FunctionInfo::rust_module()` only delegates to the group string (`crates/ta-core/src/inventory.rs:133-134`).
- Integration tests discover group APIs through root module wiring and facade re-exports, as seen in `use ta_core::price_transform::{...}` (`crates/ta-core/tests/price_transform.rs:1-4`).
- The analogous Volatility shape is a new `crates/ta-core/src/volatility/mod.rs` facade re-exporting `ATR`, `NATR`, `TRANGE`, their input/tick types, and `*_vec` wrappers, plus root `pub mod volatility;`.

### HLC input contract and validation
- `TYPPRICEInput<'a>` is the existing HLC SoA input model with `high`, `low`, and `close` borrowed slices (`crates/ta-core/src/price_transform/typprice.rs:15-22`).
- `TYPPRICETick` is the existing HLC streaming tick shape (`crates/ta-core/src/price_transform/typprice.rs:26-33`).
- `TYPPRICE()` validates same lengths for `high/low/close`, validates finite values, checks output length, writes compact output, and returns `OutputRange::new(0, len)` (`crates/ta-core/src/price_transform/typprice.rs:37-55`).
- `validate_all_same_len()` uses the first slice length as the baseline and returns an invalid-input error on mismatch (`crates/ta-core/src/common.rs:130-155`).
- `validate_finite_slices()` delegates to `validate_finite_slice()` and reports the named slice/index for NaN or infinity (`crates/ta-core/src/common.rs:158-174`).
- Volatility shares the HLC shape but differs from price transforms because `TRANGE/ATR/NATR` need previous close (`close[idx - 1]`), so their first valid output cannot be at original index `0`.
- Existing tests cover HLC finite validation and expected value patterns for price transforms (`crates/ta-core/tests/price_transform.rs:13-41`, `crates/ta-core/tests/price_transform.rs:88-93`).

### OutputRange, compact output, and padding
- `OutputRange` records `beg_idx` and `nb_element`, mapping compact outputs back to original input positions (`crates/ta-core/src/common.rs:22-49`).
- `output_count(input_len, lookback)` is `input_len.saturating_sub(lookback)` (`crates/ta-core/src/common.rs:82-84`).
- `validate_input_len()` returns count or reports insufficient data when non-empty input cannot produce any output (`crates/ta-core/src/common.rs:108-113`).
- `validate_output_len()` reports invalid input when compact output buffers are too small (`crates/ta-core/src/common.rs:118-126`).
- `padded_from_compact()` creates full-length output, fills with `PadValue`, then copies compact outputs into `range.beg_idx..end` (`crates/ta-core/src/common.rs:180-198`).
- `Float` padding is `Float::NAN`, so correct `OutputRange.beg_idx` automatically gives warm-up NaNs in `*_vec` wrappers (`crates/ta-core/src/common.rs:56-60`).
- `compact_buffer()` allocates an input-length buffer using the type pad value (`crates/ta-core/src/common.rs:203-209`).
- `AVGDEV_vec()` demonstrates the canonical wrapper: allocate compact buffer, call free function, pass `compact[..range.nb_element]` to `padded_from_compact()` (`crates/ta-core/src/price_transform/avgdev.rs:43-50`).
- Price-transform tests assert compact range and leading NaN padding for period indicators (`crates/ta-core/tests/price_transform.rs:56-68`).

### Period and lookback semantics
- `period_lookback("timeperiod", period)` validates nonzero period and returns `period - 1` (`crates/ta-core/src/common.rs:89-103`).
- `AVGDEV()` demonstrates period validation order: compute lookback, validate finite input, compute output count, validate output length, then compute compact outputs (`crates/ta-core/src/price_transform/avgdev.rs:16-38`).
- `AVGDEV::new()` validates period before allocating state and returns `Result<Self>` (`crates/ta-core/src/price_transform/avgdev.rs:64-73`).
- For TA-Lib Volatility semantics, `TRANGE` lookback is `1`; default `ATR/NATR` lookback is `timeperiod` because it combines one bar of previous-close dependency with `timeperiod - 1` averaging warm-up.
- The current shared `period_lookback()` alone expresses `period - 1`; ATR/NATR need their own effective lookback calculation that accounts for previous-close dependency while still reusing validation primitives.
- Empty input behavior should follow current helpers: count `0` returns empty range, while non-empty but insufficient input errors through `validate_input_len()` (`crates/ta-core/src/common.rs:108-113`, `crates/ta-core/src/price_transform/avgdev.rs:22-23`).

### Smoothing and streaming alignment
- `StreamingIndicator` defines warm-up as `Ok(None)`, valid tick output as `Ok(Some(_))`, and bad input as `Err(_)` (`crates/ta-core/src/traits.rs:56-69`).
- `EMA` provides the recursive smoothing precedent: batch seeds from the first `timeperiod` inputs, writes the seed as first compact output, then applies `ema_step()` for later inputs (`crates/ta-core/src/overlap/ema.rs:25-59`).
- `EMA::next()` mirrors batch: accumulate seed until `count == period`, return first `Some(value)`, then recursively update state for each later input (`crates/ta-core/src/overlap/ema.rs:156-177`).
- `EMA::next_checked()` bridges streaming warm-up to padded-vector semantics by converting `None` to `Float::NAN` (`crates/ta-core/src/overlap/ema.rs:126-127`).
- `AVGDEV` provides rolling-buffer streaming/reset precedent for period state and reset to warm-up (`crates/ta-core/src/price_transform/avgdev.rs:114-148`).
- ATR’s Wilder smoothing must align four surfaces: batch first compact output, streaming first `Some`, padded vector first non-NaN, and struct `lookback()`.
- Streaming Volatility state needs previous close in addition to any period/smoothing state; `TYPPRICE` is stateless and outputs every tick, so it is a shape precedent but not a state precedent (`crates/ta-core/src/price_transform/typprice.rs:126-137`).

### Public struct and trait surface
- `Indicator` requires associated borrowed input, mutable compact output, owned padded output, plus `lookback`, `compute`, and `compute_to_vec` (`crates/ta-core/src/traits.rs:28-53`).
- `TYPPRICE` shows multi-input trait shape: named HLC input, `&mut [Float]` output, `Vec<Float>` owned output, `lookback() == 0`, and trait methods delegating to free functions (`crates/ta-core/src/price_transform/typprice.rs:104-123`).
- `AVGDEV` shows period-bearing struct shape: private period/state fields, `new(timeperiod) -> Result<Self>`, `period()`, inherent compute methods, and trait methods delegating to free functions (`crates/ta-core/src/price_transform/avgdev.rs:55-110`).
- Inventory trait-conformance tests compile-check every implemented struct as both `Indicator` and `StreamingIndicator` (`crates/ta-core/tests/inventory.rs:129-213`).
- Volatility structs should be covered analogously for `ATR`, `NATR`, and `TRANGE` once those inventory records become `Implemented`.

### External TA-Lib semantics
- TA-Lib C `TRANGE` computes `max(high-low, abs(prevClose-high), abs(prevClose-low))`, consumes `inClose[today - 1]`, and skips the first bar; lookback is `1`.
- TA-Lib C `ATR` uses True Range, seeds the first ATR with a simple average of `timeperiod` TR values, then applies Wilder smoothing: `((prevATR * (timeperiod - 1)) + todayTR) / timeperiod`.
- TA-Lib C `NATR` uses the same ATR path, then normalizes with `(ATR / Close) * 100` using the current output bar’s close; if close is considered zero, it outputs `0.0`.
- TA-Lib documents ATR and NATR as having an unstable period; default global unstable period is `0`, and this project currently has no global unstable-period API.
- External references: TA-Lib `ta_TRANGE.c` (<https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_TRANGE.c>), `ta_ATR.c` (<https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ATR.c>), `ta_NATR.c` (<https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_NATR.c>), Python volatility docs (<https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html>), API docs (<https://ta-lib.org/api/>).

## Code References
- `crates/ta-core/src/inventory.rs:9-12` — official function count and current implemented count.
- `crates/ta-core/src/inventory.rs:16-24` — `FunctionGroup` includes `VolatilityIndicators`.
- `crates/ta-core/src/inventory.rs:71-93` — official group counts and Rust module names, including `"volatility"`.
- `crates/ta-core/src/inventory.rs:125-134` — implemented-status and module-name helpers.
- `crates/ta-core/src/inventory.rs:206-208` — `ATR`, `NATR`, `TRANGE` inventory records currently marked `Planned`.
- `crates/ta-core/src/inventory.rs:324-325` — exact uppercase function lookup.
- `crates/ta-core/src/lib.rs:35-49` — current root module exports and public facade re-exports; no `volatility` module.
- `crates/ta-core/src/price_transform/mod.rs:6-16` — module facade pattern for private files plus `pub use` exports.
- `crates/ta-core/src/price_transform/typprice.rs:15-33` — HLC input and streaming tick shapes.
- `crates/ta-core/src/price_transform/typprice.rs:37-67` — HLC batch validation, compact output, and vector wrapper pattern.
- `crates/ta-core/src/price_transform/typprice.rs:104-137` — multi-input `Indicator` and stateless streaming implementation.
- `crates/ta-core/src/price_transform/avgdev.rs:16-50` — period-based compact output and padded wrapper pattern.
- `crates/ta-core/src/price_transform/avgdev.rs:55-110` — fallible period constructor and period-bearing trait surface.
- `crates/ta-core/src/price_transform/avgdev.rs:114-148` — rolling-buffer streaming and reset semantics.
- `crates/ta-core/src/overlap/ema.rs:25-59` — recursive smoothing seed and compact batch output pattern.
- `crates/ta-core/src/overlap/ema.rs:76-127` — EMA state, constructor, helper methods, and NaN warm-up bridge.
- `crates/ta-core/src/overlap/ema.rs:156-185` — recursive streaming and reset semantics.
- `crates/ta-core/src/common.rs:22-49` — `OutputRange` compact-output location contract.
- `crates/ta-core/src/common.rs:82-126` — period, input-length, and output-buffer validators.
- `crates/ta-core/src/common.rs:130-174` — same-length and finite-input validators.
- `crates/ta-core/src/common.rs:180-209` — compact-to-padded conversion and compact buffer allocation.
- `crates/ta-core/src/traits.rs:28-69` — `Indicator` and `StreamingIndicator` contracts.
- `crates/ta-core/tests/inventory.rs:69-124` — implemented inventory count/status assertions.
- `crates/ta-core/tests/inventory.rs:129-213` — trait-conformance compile checks for implemented structs.
- `crates/ta-core/tests/inventory.rs:217-222` — current deferred Planned assertion includes `ATR`.
- `crates/ta-core/tests/price_transform.rs:1-4` — integration import style through group facade.
- `crates/ta-core/tests/price_transform.rs:56-68` — compact range and padded NaN assertions.
- `crates/ta-core/tests/price_transform.rs:88-93` — bad length, non-finite, insufficient-data, and invalid-period tests.

## Integration Points

### Inbound References
- `crates/ta-core/tests/inventory.rs:69-124` — verifies implemented inventory names, implemented count, and `function(name)` status.
- `crates/ta-core/tests/inventory.rs:129-213` — verifies implemented structs satisfy `Indicator` and `StreamingIndicator`.
- `crates/ta-core/tests/inventory.rs:217-222` — currently asserts `ATR` is still planned.
- `crates/ta-core/tests/price_transform.rs:1-93` — existing integration-test pattern for group facade imports, expected values, vector wrappers, struct methods, and invalid inputs.
- Future `crates/ta-core/tests/volatility.rs` — natural integration-test location matching `price_transform.rs` style.

### Outbound Dependencies
- `crates/ta-core/src/common.rs:22-209` — Volatility implementations depend on `OutputRange`, compact/padded helpers, and shared validators.
- `crates/ta-core/src/traits.rs:28-69` — public batch/streaming trait contracts for Volatility structs.
- `crates/ta-core/src/types.rs` — Volatility APIs use crate-wide `Float` precision boundary through the public facade.
- `crates/ta-core/src/error.rs` — validation failures propagate through `Result` / `TalibError` via common helpers.
- `crates/ta-core/src/overlap/ema.rs:25-59` — recursive seed/smoothing pattern to model ATR batch/streaming alignment from, not a direct dependency unless helpers are reused.

### Infrastructure Wiring
- `crates/ta-core/src/lib.rs:35-49` — root module declaration needs to expose the new `volatility` facade.
- `crates/ta-core/src/volatility/mod.rs` — new facade file should mirror existing group-module re-export conventions.
- `crates/ta-core/src/inventory.rs:9-12` — implemented count must reflect three additional implemented functions.
- `crates/ta-core/src/inventory.rs:206-208` — `ATR/NATR/TRANGE` statuses are the inventory records to update.
- `crates/ta-core/tests/inventory.rs:69-222` — inventory tests must move Volatility functions from deferred/planned expectations into implemented and trait-conformance expectations.

## Architecture Insights
- Official TA-Lib group classification is already canonical in the inventory; Volatility scope is exactly `ATR`, `NATR`, and `TRANGE` (`crates/ta-core/src/inventory.rs:71-78`, `crates/ta-core/src/inventory.rs:206-208`).
- Group modules use private implementation files and public facade re-exports; tests should import through `ta_core::<group>` rather than private submodules (`crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/tests/price_transform.rs:1-4`).
- Batch kernels write compact outputs and return `OutputRange`; convenience wrappers return full-length padded vectors (`crates/ta-core/src/common.rs:22-49`, `crates/ta-core/src/common.rs:180-209`).
- New period constructors should return `Result<Self>` and validate before state allocation, matching current non-legacy indicators (`crates/ta-core/src/price_transform/avgdev.rs:64-73`).
- Multi-input Volatility indicators can share HLC SoA input/tick shapes with Price Transform, but previous-close state makes their lookback/streaming semantics different (`crates/ta-core/src/price_transform/typprice.rs:15-33`).
- Recursive smoothing indicators must keep batch seed, streaming seed, reset, padded vector, and `lookback()` consistent; `EMA` is the closest existing model (`crates/ta-core/src/overlap/ema.rs:39-59`, `crates/ta-core/src/overlap/ema.rs:156-185`).
- Developer decision: implement ATR/NATR with TA-Lib default unstable period fixed at 0; do not introduce global unstable-period configuration in this tranche.

## Precedents & Lessons
3 similar past changes analyzed.

### Precedent: TA-Lib first-tranche core APIs
**Commit(s)**: `3754ae9` — "Add TA-Lib first-tranche core APIs" (2026-07-05)
**Blast radius**: 22 files across 6 layers
  core contract/ — added common helpers, `OutputRange`, validators, traits/public exports
  inventory/ — added official 161-function ledger and statuses
  price_transform/ — added `AVGDEV`, `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`
  overlap/ — rewrote `SMA` to new dual API
  math_transform + math_operators/ — added first-tranche function families
  tests/ — added inventory, price transform, SMA, and math tests

**Follow-up fixes**:
- `f5c9ed1` — "Add TA-Lib moving-average core APIs" (2026-07-09) — follow-on touched `traits.rs`, `inventory.rs`, and existing `price_transform` files; no explicit bug-fix commit found.

**Lessons from docs**:
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — Volatility is exactly `ATR`, `NATR`, `TRANGE`; prior local plans misclassified some functions.
- `.rpiv/artifacts/designs/2026-07-04_17-28-24_rust-talib-core-foundation-first-tranche.md` — official TA-Lib groups become Rust modules; core APIs write compact outputs returning `OutputRange`; `*_vec` returns padded vectors.
- `.rpiv/artifacts/validation/2026-07-05_11-35-24_ta-lib-rust-core-foundation-first-tranche.md` — validation included module wiring, inventory count/status, f32 tests, and no private SIMD/backend leakage.

**Takeaway**: Treat Volatility as a new official group module with inventory/status/tests wired in the same change, not as an extension of momentum or overlap.

### Precedent: Moving-average tranche with dispatcher
**Commit(s)**: `f5c9ed1` — "Add TA-Lib moving-average core APIs" (2026-07-09)
**Blast radius**: 27 files across 6 layers
  overlap/ — added `EMA`, `WMA`, `TRIMA`, `DEMA`, `TEMA`, `T3`, `MA`
  core contract/ — adjusted `traits.rs` and `lib.rs`
  inventory/ — moved implemented count/statuses
  price_transform + math_*/ — follow-on adjustments to prior tranche modules
  tests/ — moving-average and inventory coverage
  benchmarks/ — public API benchmark coverage

**Follow-up fixes**:
- None found after 2026-07-09 through path search.

**Lessons from docs**:
- `.rpiv/artifacts/plans/2026-07-05_21-38-45_ema-wma-trima-dema-tema-t3-ma.md` — follow existing dual API; test compact alignment, padded NaN warm-up, invalid inputs, structs, streaming/reset, and dispatcher behavior.
- `.rpiv/artifacts/validation/2026-07-07_21-50-54_ema-wma-trima-dema-tema-t3-ma.md` — keep deferred functions planned unless implemented, verify adapter no-diff, and grep for private SIMD/backend leakage.

**Takeaway**: For Volatility, include `ATR/NATR/TRANGE` tests plus inventory assertions that only those statuses changed.

### Precedent: Core traits/SMA foundation churn
**Commit(s)**: `491d9e8` — "任务 1.3: 实现核心 Traits (更新版-强制SIMD加速+零拷贝)" (2026-01-29); `896d1d7` — "sma" (2026-02-05)
**Blast radius**: 5 files across 3 layers
  traits/ — `Indicator` API churn
  overlap/ — SMA batch/streaming implementation churn
  benchmarks/ — SMA benchmark added

**Follow-up fixes**:
- `9b397bf` — "Update traits.rs" (2026-02-02) — trait churn.
- `29fb091` — "update" (2026-02-03) — removed trait lines.
- `6528112` — "sma" (2026-02-03) — rewrote SMA/traits again.
- `3754ae9` — "Add TA-Lib first-tranche core APIs" (2026-07-05) — replaced panic/shape issues with fallible constructors, initialized state, and compact `OutputRange` APIs.

**Lessons from docs**:
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — legacy `SMA::new` panicked on zero period, streaming buffer used capacity without initialized length, and trait prose diverged from live trait.
- `.rpiv/artifacts/designs/2026-07-04_17-28-24_rust-talib-core-foundation-first-tranche.md` — constructors must return `Result<Self>` and use typed `TalibError`.

**Takeaway**: Do not add Volatility indicators with ad-hoc constructors or streaming state; use shared validators and initialized state from the current contract.

### Composite Lessons
- No direct `ATR`/`NATR`/`TRANGE` implementation commits exist; closest precedents are first-tranche group wiring and moving-average recursive/streaming work.
- Canonical group classification matters: Volatility is only `ATR`, `NATR`, and `TRANGE`; do not pull in `SAR` or momentum functions.
- Every group change must update source module wiring, `lib.rs`, inventory statuses/counts, integration tests, and `f32`/workspace checks together.
- Avoid expanding core-global state unless a design explicitly covers it; default TA-Lib behavior can be matched without unstable-period configuration.

## Historical Context (from `.rpiv/artifacts/`)
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md` — TA-Lib inventory and canonical function grouping.
- `.rpiv/artifacts/designs/2026-07-04_17-28-24_rust-talib-core-foundation-first-tranche.md` — first-tranche core API architecture and module pattern.
- `.rpiv/artifacts/plans/2026-07-05_09-51-36_rust-talib-core-foundation-first-tranche.md` — first-tranche implementation plan.
- `.rpiv/artifacts/validation/2026-07-05_11-35-24_ta-lib-rust-core-foundation-first-tranche.md` — validation of first-tranche core APIs.
- `.rpiv/artifacts/plans/2026-07-05_21-38-45_ema-wma-trima-dema-tema-t3-ma.md` — moving-average tranche implementation plan.
- `.rpiv/artifacts/validation/2026-07-07_21-50-54_ema-wma-trima-dema-tema-t3-ma.md` — moving-average validation report.

## Developer Context
**Q (`crates/ta-core/src/common.rs:101-113`, `crates/ta-core/src/lib.rs:35-49`): TA-Lib ATR/NATR have an optional unstable-period adjustment, but current helpers only model fixed lookback/count and the crate exposes no global unstable-period API. Should this tranche implement default TA-Lib behavior only or introduce unstable-period configuration now?**
A: Default only. Implement ATR/NATR with TA-Lib default unstable period = 0; do not add global unstable-period configuration in this tranche.

**Q (`crates/ta-core/src/inventory.rs:206-208`, `crates/ta-core/src/price_transform/mod.rs:6-16`, `crates/ta-core/src/overlap/ema.rs:156-185`): Scan complete — write the doc, add an area, or correct a finding?**
A: Write the doc.

## Related Research
- `.rpiv/artifacts/research/2026-07-04_15-40-32_rust-talib-core-inventory.md`

## Open Questions
None recorded. Developer resolved the only implementation-scope ambiguity by choosing TA-Lib default unstable period behavior without new global configuration.
