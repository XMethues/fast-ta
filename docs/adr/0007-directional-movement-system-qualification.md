# Qualify Directional Movement as one shared system

`PLUS_DM`, `MINUS_DM`, `PLUS_DI`, `MINUS_DI`, `DX`, `ADX`, and `ADXR` are mathematically dependent Indicator Definitions. Implementing them as isolated formulas would permit movement qualification, true range, recursive smoothing, zero-range behavior, and source alignment to drift between public execution modes. We therefore qualify them as one Directional Movement system with a single one-period movement rule, the shared volatility true-range primitive, one Wilder sum-smoothing state, and one directional-index dependency chain. Public configurations and output ranges remain definition-specific.

## Status

Accepted.

## Qualification basis

- Reference vectors are generated through `ctypes` from the pinned official TA-Lib 0.6.4 release archive at commit `43f9d5042ecc4bd367941846494ad907bf20ea50`, after verifying archive SHA-256 `aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86` and the loaded library version. The generator and generated vectors are retained under `crates/ta-core/tests/fixtures/`.
- The reference Observation Series includes a sustained rise, reversal, recovery, and oscillating decline. Independent tests additionally cover reflected-series symmetry, movement sign, `[0, 100]` strength ranges, the DI-to-DX dependency identity, a flat zero-true-range series, and Period 1 DM/DI behavior.
- Owned Compact Output, caller-owned Batch Computation, Prepared Batch Runner, and independent Streaming Computation are compared at their public seams against the same reference values. Invalid aligned inputs, insufficient observations, output and prepared capacity failures, stream error preservation, and reset are contract tests.
- Qualification covers default `f64`, supported `f32`, and no-std builds. Criterion IDs cover the seven definitions and representative caller-owned, owned, prepared, repeated-Universe, and streaming workloads. Allocation profiles require zero allocations for caller-owned and prepared steady-state Batch Computation, one exact output-column allocation for owned Compact Output, and one `(period - 1) * size_of::<Float>()` history allocation only when constructing an `ADXR` stream.

## Consequences

- The Period is immutable. `PLUS_DM` and `MINUS_DM` have Lookback 1 at Period 1 and `period - 1` otherwise; `PLUS_DI` and `MINUS_DI` have Lookback 1 at Period 1 and `period` otherwise; `DX`, `ADX`, and `ADXR` require Period 2 or greater and have Lookbacks `period`, `2 * period - 1`, and `3 * period - 2` respectively.
- Warm-up equals Lookback and additional Stabilization is zero for every definition. DM outputs are non-negative. DI, DX, ADX, and ADXR output values are in `[0, 100]`; a zero true range or zero DI sum produces finite zero.
- ADXR caller-owned Batch Computation stays allocation-free by advancing a current ADX accumulator and a second accumulator over the lagged source positions. Streaming ADXR retains the minimum `period - 1` ADX history values because earlier source observations are unavailable to an incremental computation.
- Every input failure is detected before caller output or streaming state is mutated. Reset returns a stream to its initial Warm-up phase.
