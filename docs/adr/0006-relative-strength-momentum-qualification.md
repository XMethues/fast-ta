# Qualify relative-strength momentum as one shared gain/loss model

## Status

Accepted.

## Context

RSI, CMO, and IMI are three Indicator Definitions in the Momentum group. RSI and CMO consume one real Observation Series and use the same Wilder-recursive average gain and average loss. IMI consumes aligned open/close observations and uses a rolling Period of intraday movements. Treating them as unrelated implementations would duplicate the subtle movement classification, denominator behavior, validation order, warm-up, and streaming transitions. Treating IMI as Wilder-recursive would instead change its definition and source alignment.

The numerical reference is TA-Lib 0.8.1 at commit `e64d2ac896c595f38d65e44c812efbfdac8a64cf`, specifically `src/ta_func/ta_RSI.c`, `ta_CMO.c`, and `ta_IMI.c`. The checked-in fixture generator uses independent high-precision decimal arithmetic and records the revision beside its output.

## Decision

A private gain/loss value classifies every non-negative movement as gain and every negative movement by its magnitude as loss. RSI and CMO share one Wilder state transition: seed average gain and average loss from exactly `Period` consecutive source-to-source movements, then update each average as `(previous * (Period - 1) + current) / Period`. Their Lookback is therefore `Period`; RSI projects `100 * gain / (gain + loss)` and CMO projects `100 * (gain - loss) / (gain + loss)`.

IMI applies the same classification to `close - open` at each aligned observation, retains exactly `Period` classified movements, and replaces the outgoing contribution with the incoming contribution. Its Lookback is `Period - 1`; it does not inherit Wilder smoothing merely because the classification is shared.

RSI and CMO return zero when total gain plus loss is TA-zero. IMI returns its neutral center `50` for an all-flat Period. All-gain and all-loss Periods reach the applicable oscillator boundaries. Rolling subtraction round-off is clamped only at the mathematically non-negative gain/loss totals so a long IMI Streaming Computation remains bounded.

Each immutable configuration supplies owned Compact Output, caller-owned `compute_into`, a scratch-free Prepared Batch Runner, and an independent Streaming Computation. RSI and CMO streams retain constant-sized scalar state. An IMI stream allocates exactly one `Period`-element movement ring during stream creation and allocates nothing per tick. Rejected non-finite ticks are validated before any transition, and reset retains allocated IMI capacity while restoring warm-up state.

## Qualification evidence

`crates/ta-core/tests/momentum_relative_strength.rs` covers the pinned ordinary vectors, source ranges and first outputs, batch/stream parity, warm-up, reset/replay, independent streams, rejected-tick state preservation, flat/all-gain/all-loss/alternating boundaries, immutable configuration periods, caller-output non-mutation, insufficient data, aligned lengths, finite observations, output capacity, prepared reuse, and oversize rejection. The same tests compile for default `f64` and supported `f32` with precision-specific numerical tolerances.

The supported no-std and precision checks are:

```text
cargo check -p ta-core --no-default-features
cargo check -p ta-core --no-default-features --features f32
cargo test -p ta-core --no-default-features --features f32,std --test momentum_relative_strength
```

The allocation executable asserts these exact default-`f64`, Period-14 profiles at 4,096 observations:

| Operation | Allocation operations | Gross bytes | Peak incremental bytes | Retained bytes |
|---|---:|---:|---:|---:|
| construct RSI/CMO/IMI configuration | 0 | 0 | 0 | 0 |
| RSI/CMO/IMI caller-owned computation | 0 | 0 | 0 | 0 |
| RSI/CMO owned Compact Output (4,082 values) | 1 | 32,656 | 32,656 | 32,656 |
| IMI owned Compact Output (4,083 values) | 1 | 32,664 | 32,664 | 32,664 |
| empty owned Compact Output, any definition | 0 | 0 | 0 | 0 |
| prepare any runner at capacity 4,096 | 0 | 0 | 0 | 0 |
| first/repeated prepared call or oversize rejection | 0 | 0 | 0 | 0 |
| create RSI or CMO stream | 0 | 0 | 0 | 0 |
| create IMI stream | 1 | 112 | 112 | 112 |
| process ticks or reset after stream creation | 0 | 0 | 0 | 0 |

Under supported `f32`, the same one-allocation shapes retain 16,328 bytes for RSI/CMO output, 16,332 bytes for IMI output, and 56 bytes for the IMI stream ring; all zero-allocation rows remain zero.

Criterion registers representative observation/Period matrices for RSI and CMO and a representative Period-14 matrix for IMI. The workload groups cover a 128-instrument Universe, periods 5/14/50/200 for all three definitions, four independent per-worker runners, and 16 independent Streaming Computations. Stable IDs are rooted at:

```text
indicator_execution/expanded/momentum_relative_strength/{RSI,CMO,IMI}
indicator_execution/expanded/momentum_relative_strength_workloads/{RSI,CMO,IMI}
```

## Consequences

The shared model is deep enough to prevent RSI/CMO smoothing drift without erasing IMI's different observation shape or rolling semantics. Batch output remains compact and allocation-minimal, caller-owned and prepared execution remain allocation-free, and IMI pays only its required Period-bounded streaming state. No stochastic or Directional Movement definition is implied by this decision.
