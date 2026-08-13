# Qualify MAMA and HT_TRENDLINE through the shared Hilbert transition

MESA Adaptive Moving Average (`MAMA`) and Hilbert Transform Instantaneous Trendline (`HT_TRENDLINE`) are accepted as independent Overlap Study Indicator Definitions. They reuse the qualified crate-private Hilbert transition from ADR-0002 and ADR-0009. The weighted-price smoother, parity-separated Hilbert transforms, delayed InPhase and Quadrature state, bounded period recurrence, and smoothed-period recurrence remain one implementation. The Overlap Studies retain only definition-specific projection state; no Batch Computation, Prepared Batch Runner, or Streaming Computation owns a second Hilbert recurrence.

## Status

Accepted.

## Definitions and public shape

`MAMAConfig` is immutable and contains a finite fast limit and slow limit, each in the inclusive interval `[0.01, 0.99]`. Defaults remain `0.5` and `0.05`. These limits are Indicator Configuration, not a Period. The definition has fixed Lookback and Stabilization 32. Every valid source position produces both named columns, `MAMAValues { mama, fama }`; caller-owned execution uses `MAMAValuesMut`, and streaming emits `MAMAValue`. FAMA is never discarded.

At each Hilbert transition, phase is `atan(Quadrature / InPhase)` in degrees, with zero phase when InPhase is zero. The phase delta is floored at one degree. The MAMA adaptation factor is the fast limit divided by that delta and floored at the slow limit; FAMA uses half the resulting factor. Both recurrences start from zero and are advanced from the qualified transition's canonical observation 12 onward before results become valid at source position 32.

`HT_TRENDLINEConfig` is parameter-free. Its fixed Lookback, Warm-up, and Stabilization are 63 observations. It rounds the shared smoothed dominant Period to the nearest integer, averages that many current and preceding raw observations, and applies the canonical weighted four-value instantaneous-trend smoother. The recurrence begins at the shared phase lead-in position 37 so its first compact result at source position 63 includes the complete fixed stabilization.

MAMA remains absent from `PeriodMAType`. Adding it would accept a meaningless Period and force a single-output consumer either to discard FAMA or invent an alternate result shape. ADR-0003 therefore continues to close the generic selector over implemented, single-output Period-based Moving Average definitions only.

## Execution and failure contracts

Both Indicator Definitions expose owned Compact Output, caller-owned `compute_into`, a reusable Prepared Batch Runner, and independent Streaming Computation. Batch and prepared paths validate all parameters, finite observations, source length, every output-column capacity, and prepared source capacity before writing any output. Prepared capacity is checked first. Rejected stream ticks are validated before either the shared Hilbert state or definition-specific state advances. Reset returns the complete recurrence to Warm-up without changing immutable configuration.

All definition-specific state is fixed-size: MAMA retains scalar phase and paired-average state, while HT_TRENDLINE retains a 50-element raw-observation history and three trend values. The same source supports default `f64`, supported `f32`, and `no_std`; both input precisions execute the recurrence internally in `f64`, matching the qualified Hilbert family and preventing precision-dependent recurrence branches.

## Numerical evidence

`crates/fast-ta/tests/fixtures/generate_hilbert_overlap.py` verifies the official TA-Lib 0.6.4 source archive SHA-256 `aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86`, builds commit `43f9d5042ecc4bd367941846494ad907bf20ea50`, and calls both definitions through `ctypes`. The checked-in vectors cover constant, linear-trend, 20-observation sine, chirp, and deterministic seeded-noise Observation Series. Tests apply explicit `f64` and `f32` tolerances to every public execution mode and independently cover compact alignment, limit boundaries and sensitivity, FAMA preservation, flat/trend/sine invariants, reset and replay, independent streams, input and parameter rejection, every output capacity, prepared capacity, and failure state preservation.

## Allocation and peak-heap contract

At 4,096 default-`f64` observations, MAMA has 4,064 valid paired outputs. Owned computation performs exactly two 32,512-byte allocations, 65,024 bytes gross, peak incremental, and retained. HT_TRENDLINE has 4,033 valid outputs and owned computation performs exactly one 32,264-byte allocation whose gross, peak incremental, and retained sizes are all 32,264 bytes. Empty owned results, configuration construction, caller-owned execution, Prepared Batch Runner construction and reuse, over-capacity rejection, stream construction, and streaming ticks perform zero allocations. There is no hidden scratch allocation.

## Representative performance gates

Criterion roots `indicator_execution/expanded/hilbert_overlap` and `indicator_execution/expanded/hilbert_overlap_workloads` are permanent first-delivery baselines. They cover caller-owned, owned Compact Output, and prepared one-shot execution at 64, 4,096, and 65,536 observations; a 128-by-4,096 Universe through caller-owned and prepared paths; a representative four-configuration MAMA limit sweep; four independent Prepared Batch Runners; and sixteen independent Streaming Computations over 4,096 ticks for both definitions. There is no retained Rust predecessor with the same mathematical meaning, so these absolute IDs are not presented as relative speedups. Later stable regressions greater than approximately five percent on the same host and workload block delivery unless an explicit trade-off is accepted.
