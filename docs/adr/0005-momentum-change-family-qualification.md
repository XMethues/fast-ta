# Qualify the Momentum change family as one source-aligned definition set

## Status

Accepted.

## Decision

`MOM`, `ROC`, `ROCP`, `ROCR`, and `ROCR100` compare source position `i` with
source position `i - Period`. Their Lookback and streaming Warm-up are therefore
the configured Period, not `Period - 1`. The accepted Period range is
`1..=100_000`, matching the pinned definitions. They share only that
source-position primitive and ring history; each named Indicator Definition
retains its own immutable configuration, public execution types, output scaling, and benchmark
identity.

The denominator contract is pinned to TA-Lib revision
`e64d2ac896c595f38d65e44c812efbfdac8a64cf`, specifically
`src/ta_func/ta_ROC.c`, `ta_ROCP.c`, `ta_ROCR.c`, and `ta_ROCR100.c`. An exactly
zero trailing observation produces zero for every normalized definition. No
epsilon is used: every nonzero denominator, including a near-zero value, is
divided normally. On a nonzero denominator, the family relationships are
`ROC = ROCP × 100`, `ROCR = ROCP + 1`, and `ROCR100 = ROCR × 100`; `MOM` is the
unscaled point difference.

## Qualification

The independent fixture generator under `crates/ta-core/tests/fixtures` spells
out the five reference definitions separately from the Rust kernel and includes
ordinary, exact-zero, near-zero, and negative-denominator cases. Public-seam
tests qualify default `f64` and feature-selected `f32` execution, Compact Output
alignment, every execution mode, reset, validation order, capacity rejection,
and failure-before-mutation. The implementation uses only `core` plus the
feature-selected `alloc` vector and remains available without `std`.

Criterion IDs under `indicator_execution/expanded/momentum/{MOM,ROC,ROCP,ROCR,ROCR100}`
form the absolute first-delivery latency baseline. Each definition records
caller-owned, owned Compact Output, and prepared one-shot execution at 64,
4,096, and 65,536 observations; a Period sweep over 5, 14, 50, and 200;
Universe caller-owned and prepared execution at 128 × 4,096; per-worker
prepared execution at 4 × 4,096; and independent streaming at 16 × 4,096.
There is no retained Rust predecessor with these Indicator Definitions, so no
relative speedup is claimed.

The allocation harness enforces the exact default-`f64` profile. Configuration,
caller-owned execution, Prepared Batch Runner setup and reuse, capacity
rejection, and streaming ticks allocate zero bytes. Owned Compact Output makes
one exact allocation of `(input_len - Period) × size_of::<Float>()`. Each stream
makes one exact setup allocation of `Period × size_of::<Float>()` for its
independent ring and makes no allocation while consuming ticks. Peak incremental
requested heap equals retained bytes on the two owned paths and is zero on the
remaining paths. Universe, Period-sweep, per-worker, and multi-stream execution
are separately asserted allocation-free after caller-owned storage and stream
rings have been prepared.

## Consequences

Later Momentum work may reuse the source-position primitive only when its
mathematics truly compare the same two observations. It must not weaken the
exact-zero denominator rule, introduce a padded output path, share streaming
state with prepared batch execution, or collapse the five public Indicator
Definitions into a runtime selector.
