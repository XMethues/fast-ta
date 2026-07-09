---
date: 2026-07-05
repository: fast-ta
branch: main
base_commit: cfdf1e5
baseline: before-algo-simd-opt
command: "cargo bench -p ta-benchmarks --bench first_tranche -- --baseline before-algo-simd-opt"
status: captured
---

# SIMD Attempt + Final Benchmark Comparison

## SIMD Attempt Outcome

A portable `wide`-based SIMD facade was tried for:

- `ADD`, `SUB`, `MULT`, `DIV`
- `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, `WCLPRICE`
- `SQRT`

The attempt used unaligned `FastFloat` chunk load/store wrappers in the public `simd` facade and wired them into first-tranche compact kernels.

Criterion showed significant regressions against `before-algo-simd-opt`:

- `ADD_compact`: roughly +29% to +61% regression depending on size.
- `AVGPRICE_compact`: roughly +20% to +31% regression depending on size.
- `SQRT_compact`: roughly +29% to +52% regression depending on size.

Conclusion: this portable-wide SIMD layer was not retained. The final working tree keeps scalar elementwise kernels plus the rolling algorithm optimizations.

## Final Retained Optimizations

- `SUM` uses a sliding-window `O(n)` kernel.
- `MIN`, `MAX`, `MININDEX`, `MAXINDEX`, `MINMAX`, `MINMAXINDEX` use monotonic deque `O(n)` kernels.
- `SMA_vec` writes padded output directly instead of allocating a compact temporary then copying.
- Benchmark code uses `std::hint::black_box`, removing Criterion 0.8 deprecation warnings.

## Final Criterion Comparison

Final retained implementation compared against `before-algo-simd-opt` baseline. Middle estimate from Criterion:

| Benchmark | 1,024 | Change | 16,384 | Change | 65,536 | Change |
|---|---:|---:|---:|---:|---:|---:|
| `SMA_compact` | 1.6281 µs | +4.27% regression | 24.000 µs | -3.84% | 95.797 µs | -8.55% |
| `SMA_vec` | 1.6963 µs | -10.79% | 26.116 µs | -46.49% | 105.23 µs | -31.51% |
| `AVGPRICE_compact` | 1.9424 µs | -7.03% | 33.121 µs | -7.56% | 135.24 µs | -6.77% |
| `AVGDEV_compact` | 12.072 µs | -8.29% | 197.40 µs | -6.88% | 811.26 µs | -6.93% |
| `SQRT_compact` | 762.85 ns | -5.81% | 12.682 µs | -8.45% | 50.964 µs | -8.58% |
| `ADD_compact` | 970.70 ns | -6.50% | 16.672 µs | -6.64% | 67.383 µs | -6.86% |
| `SUM_compact` | 1.4146 µs | -65.58% | 22.930 µs | -65.64% | 92.314 µs | -66.00% |
| `MINMAX_compact` | 4.6965 µs | -65.58% | 76.069 µs | -66.50% | 304.25 µs | -66.18% |

## Recommendation

Future SIMD work should avoid the generic unaligned `wide` facade used in this attempt. Prefer dedicated arch-dispatched kernels under the existing `simd::dispatch` boundary, benchmarked one operation at a time before wiring into indicator modules.
