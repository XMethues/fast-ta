# Qualify SAR and SAREXT as distinct stop-and-reverse definitions

`SAR` and `SAREXT` are the first recursive Overlap Studies without a Period in this repository. We accept their issue #20 Criterion workload IDs as absolute first-delivery baselines under ADR-0001 rather than presenting TA-Lib or an unrelated Indicator Definition as a speed predecessor. `SAR` retains one acceleration increment and limit. `SAREXT` retains signed position output, caller-selected initial direction and stop, reversal offset, and independent initial, increment, and limit parameters for long and short positions. The extended definition is not implemented as a SAR alias.

## Status

Accepted.

## Consequences

- Owned, caller-owned, prepared, and streaming modes execute one recurrence per Indicator Definition. Immutable configurations contain no recursive state; each Prepared Batch Runner invocation and Streaming Computation owns isolated state.
- The fixed Lookback is one Observation. Compact Output represents source range `1..source_len`; SAREXT uses positive stops for long positions and negative stops for short positions.
- The pinned numerical evidence is generated from TA-Lib 0.6.4 source revision `43f9d5042ecc4bd367941846494ad907bf20ea50`. Tests also enforce stop-direction invariants independently of those vectors at explicit `f64` and `f32` tolerances.
- Criterion IDs cover 64, 4,096, and 65,536-observation one-shot paths, Universe processing, one Prepared Batch Runner per worker, and multi-instrument streaming for both definitions. A stable regression greater than approximately five percent on these IDs blocks later changes unless its trade-off is explicitly accepted.
- Configuration construction, caller-owned execution, Prepared Batch Runner construction and reuse, over-capacity rejection, stream construction, and streaming ticks allocate zero heap bytes. Owned Compact Output performs one exact allocation of `(source_len - 1) × size_of::<Float>()`; peak incremental requested heap equals those retained output bytes.

## Qualification commands

The delivery matrix retains these definition-specific and supported-build
checks:

```text
cargo test -p ta-core --test overlap_sar
cargo test -p ta-core --no-default-features --features f32,std --test overlap_sar
cargo check -p ta-core --no-default-features
cargo check -p ta-core --no-default-features --features f32
cargo bench -p ta-benchmarks --bench execution_allocations
cargo bench -p ta-benchmarks --bench execution_baselines -- stop_and_reverse
```
