# Qualify composite-input Momentum definitions as first deliveries

BOP, CCI, MFI, and ULTOSC are the first Momentum Indicator Definitions in this repository whose observations contain multiple price or volume fields. They use explicit typed Structure-of-Arrays batch inputs and equally explicit streaming ticks: BOP and CCI consume aligned OHLC, MFI consumes aligned OHLCV, and ULTOSC consumes aligned HLC. CCI and MFI validate open alignment and finiteness even though their formulas use typical price from high, low, and close. Every column is aligned and finite before output or streaming state is mutated. This applies ADR-0001 without introducing a generic untyped observation bag or routing a Compact Output through a padded representation.

## Status

Accepted.

## Definition qualification

Numerical reference evidence is pinned to TA-Lib v0.6.4 commit `43f9d5042ecc4bd367941846494ad907bf20ea50`; the release archive SHA-256 is `aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86`. `crates/fast-ta/tests/fixtures/generate_momentum_composite.py` verifies that archive, builds it, calls all four C definitions through `ctypes`, checks each Lookback, and emits the checked-in ordinary-market vectors. The public-seam tests independently cover flat prices, a trend followed by reversal, zero volume, scale and translation invariants where definitionally valid, and boundedness for internally consistent non-negative OHLCV observations.

The edge contracts are definition-specific:

- BOP has Lookback zero and maps a zero, negative, or TA-Lib-epsilon high/low range to zero.
- CCI accepts Periods `2..=100000`, has Lookback `Period - 1`, and maps an exactly zero current deviation or mean-deviation sum to zero.
- MFI accepts Periods `2..=100000`, has Lookback `Period`, classifies unchanged typical price as zero flow, and maps a total positive-plus-negative flow below one to zero.
- ULTOSC accepts three Periods in `1..=100000`, canonicalizes them shortest-to-longest, has Lookback equal to the longest Period, and applies the definition's 4:2:1 weights independently of argument order. A zero true-range sum contributes zero for that horizon.

All configurations are immutable. Caller-owned and owned Batch Computation write directly to compact storage. Prepared Batch Runners retain only configuration and declared capacity because the source columns provide the outgoing rolling observations directly; they do not construct a padded path or allocate scratch. Streaming Computations retain independent Period-bounded rings, validate each tick before transition, and reset to the original Warm-up state.

## Allocation and peak-heap contract

`crates/ta-benchmarks/benches/execution_allocations.rs` contains executable assertions for the following default-`f64`, 4,096-observation profiles. Bytes are requested payload bytes; peak is incremental requested heap, excluding allocator metadata and fixtures created before measurement.

| Operation | Allocation operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Any configuration construction | 0 | 0 | 0 | 0 |
| Any caller-owned Batch Computation | 0 | 0 | 0 | 0 |
| BOP owned Compact Output | 1 | 32,768 | 32,768 | 32,768 |
| CCI(14) owned Compact Output | 1 | 32,664 | 32,664 | 32,664 |
| MFI(14) owned Compact Output | 1 | 32,656 | 32,656 | 32,656 |
| ULTOSC(7,14,28) owned Compact Output | 1 | 32,544 | 32,544 | 32,544 |
| Empty owned Compact Output, any definition | 0 | 0 | 0 | 0 |
| Any Prepared Batch Runner setup, first/repeated call, or oversize rejection | 0 | 0 | 0 | 0 |
| BOP stream setup | 0 | 0 | 0 | 0 |
| CCI(14) stream setup | 1 | 112 | 112 | 112 |
| MFI(14) stream setup | 1 | 224 | 224 | 224 |
| ULTOSC(7,14,28) stream setup | 1 | 448 | 448 | 448 |
| Streaming ticks after setup, any definition | 0 | 0 | 0 | 0 |

The owned byte counts are exactly `Compact Output count × size_of::<f64>()`. CCI retains one Float ring, while MFI and ULTOSC retain one ring whose entries each contain two Floats. No steady-state path grows retained storage.

## Criterion baseline contract

These definitions have no retained Rust predecessor with the same mathematical meaning. The stable IDs under `indicator_execution/expanded/composite_momentum` therefore establish absolute first-delivery one-shot baselines rather than relative speedup claims. They cross 64, 4,096, and 65,536 observations with caller-owned, owned Compact Output, and Prepared Batch Runner execution for every definition.

The IDs under `indicator_execution/expanded/composite_momentum_workloads` record caller-owned and prepared Universe execution at `128 × 4,096`, CCI/MFI/ULTOSC parameter sweeps, one independent Prepared Batch Runner per worker at `4 × 4,096`, and independent Streaming Computations at `16 × 4,096` for every definition. Later changes must retain these IDs and treat a stable regression greater than approximately five percent as an ADR-0001 gate. Host-local point estimates belong in `crates/ta-benchmarks/EXECUTION_BASELINES.md`; no comparison against TA-Lib or an unrelated Indicator Definition may be described as a speedup.

The supported qualification matrix remains default `f64`, `f32`, and `no_std` library checks. Reference tolerances are precision-aware, while configuration, source alignment, Lookback, output capacity, prepared capacity, reset, and failure-before-mutation contracts are exact in both precisions.
