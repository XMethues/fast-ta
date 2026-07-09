---
date: 2026-07-05
repository: fast-ta
branch: main
base_commit: cfdf1e5
baseline: before-algo-simd-opt
command: "cargo bench -p ta-benchmarks --bench first_tranche -- --baseline before-algo-simd-opt"
status: captured
---

# First-tranche Benchmark Comparison After Algorithm Optimization

This comparison uses Criterion baseline `before-algo-simd-opt` captured before the rolling algorithm changes.

## Optimizations Applied

- `SUM` batch kernel changed from per-window rescan to sliding-window sum.
- `MIN`, `MAX`, `MININDEX`, `MAXINDEX`, `MINMAX`, and `MINMAXINDEX` batch kernels changed to monotonic deque extrema tracking.
- `SMA_vec` now writes padded output directly instead of allocating compact output then copying into a padded vector.

## Criterion Comparison Summary

Middle estimate from the post-optimization run:

| Benchmark | 1,024 | Change | 16,384 | Change | 65,536 | Change |
|---|---:|---:|---:|---:|---:|---:|
| `SMA_compact` | 1.5406 µs | no change | 25.363 µs | no change | 101.20 µs | noise threshold |
| `SMA_vec` | 1.7108 µs | -11.35% | 27.213 µs | -43.97% | 110.35 µs | -27.28% |
| `AVGPRICE_compact` | 2.0521 µs | noise threshold | 35.821 µs | no change | 145.56 µs | no change |
| `AVGDEV_compact` | 12.949 µs | no change | 212.04 µs | no change | 861.01 µs | no change |
| `SQRT_compact` | 815.74 ns | no change | 13.598 µs | no change | 54.423 µs | noise threshold |
| `ADD_compact` | 1.0290 µs | noise threshold | 18.072 µs | no change | 74.966 µs | +5.56% regression reported |
| `SUM_compact` | 1.4759 µs | -64.06% | 24.396 µs | -63.78% | 99.713 µs | -63.25% |
| `MINMAX_compact` | 5.0215 µs | -63.21% | 82.936 µs | -63.60% | 341.01 µs | -62.30% |

## Notes

- The targeted rolling kernels improved by roughly 62-64% across benchmark sizes.
- `SMA_vec` improved because it now avoids the compact temporary buffer and copy.
- `ADD_compact` was not changed by this optimization pass; the reported +5.56% at 65,536 is likely run-to-run noise or system variance, but should be watched when SIMD work starts.
- No explicit SIMD kernels were added in this pass; this was algorithmic optimization plus padded-wrapper allocation reduction.
