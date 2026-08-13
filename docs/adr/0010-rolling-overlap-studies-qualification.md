# Qualify rolling bands and midpoint Overlap Studies

## Status

Accepted.

## Context

ACCBANDS, BBANDS, MIDPOINT, and MIDPRICE are rolling Overlap Studies with three distinct input and result shapes. ACCBANDS consumes aligned high, low, and close observations and returns upper, middle, and lower columns. BBANDS consumes one real series, selects a Period-based Moving Average for its middle column, and returns deviation bands around it. MIDPOINT consumes one real series; MIDPRICE consumes aligned high and low observations. Treating those shapes as one padded generic adapter would obscure source alignment, admit invalid moving-average selectors, and prevent pre-mutation validation of named columns.

The mathematical reference is the official TA-Lib definition, not its C call surface:

- ACCBANDS: <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ACCBANDS.c>
- BBANDS: <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_BBANDS.c>
- MIDPOINT: <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPOINT.c>
- MIDPRICE: <https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPRICE.c>

`crates/fast-ta/tests/fixtures/generate_overlap_rolling.py` records the independent Python TA-Lib reproduction, and `overlap_rolling_reference.rs` pins its compact vectors.

## Decision

Each definition has an immutable Indicator Configuration, owned Compact Output, caller-owned `compute_into`, reusable Prepared Batch Runner, and independent Streaming Computation.

- ACCBANDS uses typed high/low/close batch input and tick types. Its named output is `ACCBANDSValues { upper, middle, lower }`. All three columns are the same range: `period - 1..source_len`. The middle is `SMA(close)`. The upper and lower columns are rolling simple averages of the exact pointwise Acceleration Bands maps. A zero high-plus-low denominator degenerates to the unadjusted high and low.
- BBANDS uses `PeriodMAType`, so every constructible selector is implemented, single-output, and governed by the supplied Period. MAMA is unrepresentable. Its middle column is the selected moving average. Population standard deviation always uses the source window ending at the middle value's source position, so DEMA, TEMA, T3, KAMA, and later qualified Period-based definitions retain honest source alignment. Upper and lower multipliers remain independent finite values.
- MIDPOINT is `(rolling_max(real) + rolling_min(real)) / 2`.
- MIDPRICE is `(rolling_max(high) + rolling_min(low)) / 2` over aligned high/low observations; it does not average a per-observation price before finding extrema.

Band caller-owned columns must have equal lengths and sufficient capacity. Every input, parameter, source-length, prepared-capacity, and output-column check completes before any caller column is mutated. A rejected streaming tick leaves every rolling state unchanged. Reset restores warm-up without replacing the stream allocation.

MIDPOINT and MIDPRICE use two monotonic index queues in batch execution. One-shot execution allocates those algorithm-scratch queues locally; preparation reserves them once to the declared maximum source length and every in-capacity prepared call reuses them without allocation. Their streams use Period-bounded rings because streaming state must remain independent and source indexes are not output. ACCBANDS and BBANDS batch kernels require no algorithm scratch; BBANDS preparation retains the selected MA runner.

## Allocation and peak-memory contract

For source length $N$, valid compact count $C$, floating element size $F$, and index size $I$:

| Execution | Allocation operations | Gross / peak requested bytes | Retained bytes |
|---|---:|---:|---:|
| ACCBANDS owned | 3 | $3CF$ | $3CF$ |
| BBANDS owned | 3 | $3CF$ | $3CF$ |
| ACCBANDS or BBANDS caller-owned / prepared steady state | 0 | 0 | 0 |
| MIDPOINT or MIDPRICE caller-owned one-shot | 2 | $2NI$ | 0 |
| MIDPOINT or MIDPRICE owned | 3 | $CF + 2NI$ | $CF$ |
| Prepare MIDPOINT or MIDPRICE | 2 | $2NI$ | $2NI$ |
| MIDPOINT or MIDPRICE prepared steady state | 0 | 0 | 0 |
| Any streaming ticks after stream construction | 0 | 0 | 0 |

The allocation executable asserts these formulas at its representative 4,096-observation, Period-14 `f64` workload and records incremental requested-heap peak, not RSS or allocator metadata. The three band output allocations are exact per-column compact allocations; no padded staging column exists.

## Criterion workload contract

The durable benchmark IDs under `indicator_execution/expanded/rolling_overlap` cover representative one-shot owned, caller-owned, and prepared paths plus:

- a 128-instrument, 4,096-observation Universe;
- Period sweeps over 5, 14, 50, and 200;
- four independent Prepared Batch Runners, one per worker;
- sixteen independent Streaming Computations with 4,096 ticks each.

MIDPOINT uses the standard single-output matrix and repeated-workload topology. ACCBANDS, BBANDS, and MIDPRICE use their typed inputs and named output columns directly. Outputs and prepared state are allocated outside timed loops. No benchmark path constructs an aligned or padded adapter.

## Consequences

The four definitions are available in default `f64`, supported `f32`, `std`, and supported `no_std` builds through the same Rust-first public interfaces. Pinned references and public contract tests cover exact compact ranges, middle/extrema meaning, flat series, positive scaling, band order for positive deviation multipliers, every qualified Period-based MA, prepared reuse, batch/stream parity, stream independence, reset/replay, and transactional error behavior.
