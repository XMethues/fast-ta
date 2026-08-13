# Qualify the range-position Momentum family

AROON, AROONOSC, STOCH, STOCHF, STOCHRSI, and WILLR are accepted as the range-position Momentum family. AROON and AROONOSC share a single rolling extrema transition over aligned high/low observations. STOCH, STOCHF, STOCHRSI, and WILLR share one stochastic range projection over aligned high/low/close observations. STOCH and STOCHF select one qualified `PeriodMAType` for their smoothing kernels. STOCHRSI composes the qualified RSI transition before applying the same stochastic range projection.

## Status

Accepted.

## Definition and alignment contract

AROON's period is bounded to `2..=100000`. Its rolling window is `period + 1` observations. Down and Up are `100 * (period - (source_idx - extrema_idx)) / period` with `factor = 100 / period`, evaluated in default `Float` precision. AROONOSC returns the same extrema at the same window length and projects `100 * (up - down) / period`. Both definitions return Compact Output whose first valid position is `period` so every AROON result is paired with an AROONOSC result of identical range.

STOCHF's fast `%K` window is `fast_k_period` observations of aligned high/low/close. Flat-range inputs return zero percent. `%D` smooths `%K` through one qualified `PeriodMAType`. Its Lookback is `fast_k_period - 1 + fast_d.lookback()`. STOCH composes the same `%K` projection, smooths it through a qualified `PeriodMAType` to produce `%K`, then smooths `%K` through a second qualified `PeriodMAType` to produce `%D`. Its Lookback is `fast_k_period - 1 + slow_k.lookback() + slow_d.lookback()`.

STOCHRSI reuses the qualified RSI transition from issue #22 with RSI Period bounded to `2..=100000`. The RSI series is treated as one real Observation Series; its fast `%K` window is `fast_k_period` observations and `%D` smooths `%K` through one qualified `PeriodMAType`. Its Lookback is `rsi.lookback() + fast_k_period - 1 + fast_d.lookback()`.

WILLR's period is bounded to `2..=100000`. Its rolling window is `period` observations. The result is `-100 * (max - close) / (max - min)` with the same TA-zero denominator rule as the stochastic `%K`. Its Lookback is `period - 1`.

AROON's rolling extrema transition selects the newest source position on equal extrema. This is independent of the public Math Operators `rolling_range_extrema` default, which selects the oldest position. Issue #30 documents this divergence so future consumers of either contract are warned at the API boundary. The local `RangeStream` type records `index` alongside `(low, high)` so the newest-source rule is preserved without re-introducing a separate kernel. STOCH's fast `%K` then reuses the same selection rule for both fast and slow `%K` smoothing.

All six definitions return Compact Output. AROON and STOCH families return named, equal-length multi-output columns (`AROONValues { down, up }`, `STOCHFValues { fast_k, fast_d }`, `STOCHValues { slow_k, slow_d }`, `STOCHRSIValues { fast_k, fast_d }`); column-capacity failures mutate no output because the validate phase precedes any column write. AROONOSC and WILLR return a single `Vec<Float>` Compact Output.

## Composition with qualified seams

STOCH and STOCHF compose the qualified Period-based Moving Average dispatcher from ADR-0003 and issue #18, accepting every selectable `PeriodMAType` for their smoothing steps. STOCHRSI composes the qualified RSI configuration from issue #22. `MAMA` is not a `PeriodMAType` selector because its configuration and paired outputs are not Period-based, per ADR-0011 and ADR-0015; STOCH and STOCHF therefore never accept it. The MAMA exclusion therefore also excludes `MAMA`-driven stochastic smoothing. Future consumers that need MAMA-driven oscillators must follow the qualification already locked in by ADR-0013.

STOCHRSI's reuse of RSI is restricted to the qualified Wilder gain/loss model. It does not extend to CMO's signed projection or IMI's intraday Period-based summation. Reusing RSI in STOCHRSI does not change any decision in ADR-0006. Future consumers that want stochastic CMO or stochastic IMI must follow a different qualification.

## Execution and failure contract

Every configuration is immutable and creates distinct one-shot owned, caller-owned `compute_into`, Prepared Batch Runner, and independent Streaming Computation state. Owned results are Compact Outputs and allocate exactly the columns required by each definition. Caller-owned and prepared execution allocate zero bytes per call after initial scratch setup. Streams own at most one `(period + 1)`-observation ring for AROON/AROONOSC, one `period`-observation ring for WILLR, one `(fast_k_period)`-observation ring plus the qualified moving-average rings for STOCH/STOCHF, and the RSI recursive state plus those rings for STOCHRSI.

All input, Period, Period-order, output-capacity, and prepared-capacity validation precedes output mutation or state transitions. The multi-output capacity check for AROON, STOCH, STOCHF, and STOCHRSI validates both columns before any column is written. Prepared capacity rejection preserves reusable state; invalid streaming ticks preserve accumulated state. Reset returns streams to their original Warm-up and replay behavior.

## Consequences

The shared rolling extrema transition, the qualified Period-based Moving Average dispatcher, and the qualified RSI seam keep the six definitions thin and allocation-bounded. KAMA's selection in STOCH/STOCHF/STOCHRSI follows ADR-0011's KAMA qualification. MAMA and KAMA drivers for stochastic smoothing therefore diverge by ADR: KAMA qualifies because it is Period-based; MAMA does not because it is not. Future stochastic indicators that compose MAMA must follow ADR-0013 rather than this ADR.

## Qualification evidence

- `crates/fast-ta/tests/fixtures/momentum_range_position_reference.rs` pins TA-Lib 0.8.1 revision `e64d2ac896c595f38d65e44c812efbfdac8a64cf` for all six definitions, generated by `crates/fast-ta/tests/fixtures/generate_momentum_range_position.py` using 50-digit Python `Decimal` arithmetic. The generator uses independent decimal recurrences for AROON, AROONOSC, STOCH's SMA smoothing, and Wilder RSI; cross-definition identities (fast `%K` plus `%D` alignment, WILLR `= fast %K − 100`) are independent recurrence checks rather than TA-Lib runtime calls.
- Public-seam tests cover pinned default-`f64` vectors and supported-`f32` tolerances, AROON/AROONOSC tie policy, fast/slow stochastic equivalence at smoothing Period one, all eight qualified `PeriodMAType` kinds including KAMA, WILLR `= fast %K − 100`, STOCHRSI reuse of qualified RSI, source scaling invariance, flat-range zero outputs, every Period ordering, Lookback, Warm-up, reset/replay, batch/stream parity, failure non-mutation/state preservation, and column-capacity transactionality.
- The source selects `Vec` from `alloc` when `std` is absent and uses the crate's feature-selected `Float`, preserving the supported no-std `f64` and `f32` matrix.
- The allocation executable asserts exact allocation operations, gross requested bytes, incremental peak bytes, and retained bytes per compact column. At 4,096 default-`f64` observations with the locked Period 14 fast kernel, AROON owns exactly two `(4,083) × 8 = 32,664`-byte columns, AROONOSC owns one `32,664`-byte column, WILLR owns one `32,664`-byte column, STOCH owns two `32,608`-byte columns, STOCHF owns two `32,608`-byte columns, and STOCHRSI owns two `32,488`-byte columns. Caller-owned and prepared execution allocate zero bytes per call.
- Criterion IDs under `indicator_execution/expanded/range_position` and `indicator_execution/expanded/range_position_workloads` cover one-shot sizes 64, 4,096, and 65,536 for all six definitions, a complete kind/Period sweep across all eight `PeriodMAType` × four Periods, Universe prepared STOCHRSI runner reuse, per-worker STOCHF Prepared Batch Runners, and independent AROON multi-stream ticks.

These are executable first-delivery contracts, not portable speed claims. Host-local records and retained benchmark identities live in `crates/ta-benchmarks/EXECUTION_BASELINES.md` and use ADR-0001's stable approximately-five-percent regression gate.