# Qualify the moving-average Momentum family

APO, PPO, MACD, MACDEXT, MACDFIX, and TRIX are accepted as the moving-average Momentum family. APO and PPO select one qualified `PeriodMAType` for their fast and slow Periods. MACDEXT selects an independent `PeriodMAType` for each fast, slow, and signal Period. Standard MACD and MACDFIX are the corresponding EMA-only definitions; MACDFIX fixes the price EMA Periods at 12 and 26 while retaining an explicit signal Period. `PeriodMAType` remains closed over implemented single-output Period-based definitions, including KAMA and excluding MAMA.

## Status

Accepted.

## Definition and alignment contract

Fast and slow Periods are bounded to `2..=100000` and the fast Period must be strictly less than the slow Period. Signal Periods are bounded to `1..=100000`; Period one is meaningful because every accepted Period-based Moving Average is an identity at Period one. APO is fast MA minus slow MA. PPO is `100 × (fast - slow) / slow`; a slow output within TA-Lib's open `(-1e-14, 1e-14)` zero tolerance produces zero. Their Lookback is the greater selected MA Lookback, and their streams retain each definition's natural history before aligned comparison.

MACD, MACDEXT, and MACDFIX own three equally sized named columns: `macd`, `signal`, and `histogram`. The MACD line is fast MA minus slow MA, the signal line is the selected MA of that line, and histogram is evaluated independently at every valid source position as `macd - signal`. Their Lookback is the greater price-MA Lookback plus the signal-MA Lookback. Price MAs are initialized so their first values are aligned at the greater price-MA Lookback; this preserves recursive restart/seed semantics rather than silently pairing values from different source positions. The signal definition consumes every aligned MACD-line value during its Warm-up. No leading MACD values are exposed without aligned signal and histogram values, and no output column is discarded or padded.

`MACDConfig::default()`, all-EMA `MACDEXTConfig::default()`, and `MACDFIXConfig::default()` are intentionally equivalent 12/26/9 configurations. MACDFIX does not introduce alternate smoothing constants. This Rust-first definition makes the fixed configuration a constrained standard MACD rather than a compatibility-mode selector.

TRIX uses three lockstep EMAs with the same Period. Its Lookback is `3 × (Period - 1) + 1`. At each valid source position it returns the exact one-position percentage change `(current_triple_ema / previous_triple_ema - 1) × 100`; an exactly zero previous triple EMA produces zero. Near-zero nonzero values are divided normally. Period one therefore reduces to a one-position ROC of the Observation Series.

## Execution and failure contract

Every configuration is immutable and creates distinct one-shot owned, caller-owned, Prepared Batch Runner, and independent Streaming Computation state. Owned results are Compact Outputs and allocate exactly one compact vector per returned column. All input, Period/order, output-capacity, and prepared-capacity validation precedes output mutation or state transitions. All three MACD-family output capacities are validated before any column is written. Prepared capacity rejection preserves reusable state; invalid streaming ticks preserve accumulated state. Reset returns streams to their original Warm-up and replay behavior.

One-shot execution may construct definition-specific rings required by a selected Period-based Moving Average. Preparation constructs those rings once; repeated in-capacity execution, reset, and steady-state ticks do not allocate. The all-EMA default profiles require no scratch allocation. This follows ADR-0001's separation of immutable Indicator Configuration from prepared and streaming state.

## Qualification evidence

- `crates/fast-ta/tests/fixtures/moving_average_momentum_reference.rs` pins TA-Lib 0.6.4 revision `43f9d5042ecc4bd367941846494ad907bf20ea50` and source archive SHA-256 `aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86`. Its checked-in generator builds that source and executes all six definitions through `ctypes` without adding a runtime dependency.
- Public-seam tests cover pinned default-`f64` vectors and supported-`f32` tolerances, all eight `PeriodMAType` kinds including KAMA, all-EMA extended equivalence, fixed/default equivalence, histogram identity, scaling, flat and linear-trend invariants, TRIX change semantics, Lookback and Warm-up, owned/caller-owned/prepared/streaming parity, reset/replay, and failure non-mutation/state preservation.
- The source selects `Vec` from `alloc` when `std` is absent and uses the crate's feature-selected `Float`, preserving the supported no-std `f64` and `f32` matrix. MAMA has no selector path.
- The allocation executable asserts exact allocation operations, gross requested bytes, incremental peak bytes, and retained bytes per compact column. At 4,096 default-`f64` observations the default APO/PPO column is 32,568 bytes, every named MACD column is 32,504 bytes, and the default TRIX column is 32,064 bytes. Caller-owned and prepared steady-state default paths allocate zero bytes.
- Criterion IDs under `indicator_execution/expanded/moving_average_momentum` and `indicator_execution/expanded/moving_average_momentum_workloads` cover one-shot sizes, a complete kind/representative-Period sweep, Universe execution, one Prepared Batch Runner per worker, and independent multi-stream ticks.

These are executable first-delivery contracts, not portable speed claims. Host-local records and retained benchmark identities live in `crates/ta-benchmarks/EXECUTION_BASELINES.md` and use ADR-0001's stable approximately-five-percent regression gate.
