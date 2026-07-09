---
date: 2026-07-05
repository: fast-ta
branch: main
commit: cfdf1e5
topic: "First-tranche benchmark baseline before algorithm/SIMD optimization"
baseline: before-algo-simd-opt
command: "cargo bench -p ta-benchmarks --bench first_tranche -- --save-baseline before-algo-simd-opt"
status: captured
---

# First-tranche Benchmark Baseline: before-algo-simd-opt

Criterion named baseline saved as `before-algo-simd-opt` under `target/criterion`.

Note: this baseline was captured from the current working tree after the `Indicator` / `StreamingIndicator` split and benchmark `std::hint::black_box` cleanup.

## Results

Middle estimate from Criterion confidence interval:

| Benchmark | 1,024 | 16,384 | 65,536 |
|---|---:|---:|---:|
| `SMA_compact` | 1.5628 µs | 25.394 µs | 103.48 µs |
| `SMA_vec` | 1.9271 µs | 48.861 µs | 153.89 µs |
| `AVGPRICE_compact` | 2.0882 µs | 35.827 µs | 144.12 µs |
| `AVGDEV_compact` | 12.995 µs | 212.58 µs | 863.88 µs |
| `SQRT_compact` | 813.01 ns | 13.938 µs | 55.803 µs |
| `ADD_compact` | 1.0483 µs | 17.982 µs | 73.350 µs |
| `SUM_compact` | 4.0409 µs | 66.439 µs | 270.95 µs |
| `MINMAX_compact` | 13.504 µs | 224.49 µs | 905.48 µs |

## Compare After Optimization

Run optimized benchmark against this baseline:

```bash
cargo bench -p ta-benchmarks --bench first_tranche -- --baseline before-algo-simd-opt
```

To save a new optimized baseline:

```bash
cargo bench -p ta-benchmarks --bench first_tranche -- --save-baseline after-algo-simd-opt
```
