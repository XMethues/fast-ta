---
date: 2026-08-03
repository: fast-ta
branch: issue-3-rust-first-sma
base_commit: 082fa18
comparison_baseline: issue-2-final
issue_2_criterion_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- --save-baseline issue-2-final"
issue_3_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines"
allocation_command: "cargo bench -p ta-benchmarks --bench execution_allocations"
status: issue-3-final
---

# Indicator Execution Baselines

This is the pre-migration correctness and performance baseline for ADR-0001. It characterizes the current public `Indicator`, `StreamingIndicator`, and `Resettable` traits without changing production behavior. Later slices should run the same workloads and compare against the Criterion baseline or the middle estimates recorded here.

## Current seam and target reconciliation

The current and ADR-0001 interfaces intentionally differ. This contract matrix is descriptive only; issue #2 makes no production behavior change.

| Contract area | Issue #2 current behavior | Target ownership |
|---|---|---|
| Caller-owned Batch Computation | `Indicator::compute` writes Compact Output and returns an `OutputRange`. | Characterized here; unchanged by #2. |
| Owned Batch Computation | `Indicator::compute_to_vec` returns legacy source-length Aligned Output, using the existing NaN or zero unavailable-position representation. It is not the target owned Compact Output. | Issue #3 introduces the target Compact Output seam. |
| Reusable batch execution | Current execution has no Prepared Batch Runner, declared prepared input capacity, or prepared-capacity error. Capacity validation applies only to caller-owned output buffers. | Issue #3 implements the Prepared Batch Runner and input-capacity contract. |
| Per-worker execution | Independent full indicator instances are the legacy predecessor workload, not Prepared Batch Runners. | Issue #3 replaces this predecessor with the target per-worker runner seam. |
| Configuration and execution-state separation | Current indicator objects combine Indicator Configuration and Streaming Computation state. | Later migration work separates configuration from execution state. |

Issue #2 therefore records predecessor behavior and workloads. It does not fabricate `prepare_batch`, a Prepared Batch Runner, or Compact Output APIs that issue #3 is responsible for introducing. `crates/ta-core/tests/indicator_execution_contracts.rs` fixes the current values, ranges, legacy unavailable-position representations, absolute `i32` indexes, output-capacity behavior, pre-mutation validation, and operational independence of separate streaming instances.

## Reference environment

- Host: Apple M2 MacBook Air, 16 GiB
- OS: Darwin 25.5.0, arm64
- Rust: `rustc 1.93.0 (254b59607 2026-01-19)`, LLVM 21.1.8
- Cargo: `cargo 1.93.0 (083ac5135 2025-12-15)`
- Precision: default `f64`
- Profile: Cargo `bench` (optimized)
- Criterion: 0.8.2, full default sampling

The issue #2 absolute results below are the middle estimates from the confidence intervals. Owned-result timing uses batched measurement so result destruction is outside the operation boundary. Streaming setup and teardown are also outside timing. The original `target/criterion/issue-2-current` capture is host-local and ignored by git; this document is the durable record.

## One-shot latency and throughput

| Indicator / current path | Observations | Latency | Throughput |
|---|---:|---:|---:|
| SMA caller compact | 64 | 220.05 ns | 290.85 Melem/s |
| SMA owned legacy Aligned Output | 64 | 224.34 ns | 285.29 Melem/s |
| SMA caller compact | 4,096 | 13.137 µs | 311.79 Melem/s |
| SMA owned legacy Aligned Output | 4,096 | 14.859 µs | 275.66 Melem/s |
| SMA caller compact | 65,536 | 210.30 µs | 311.63 Melem/s |
| SMA owned legacy Aligned Output | 65,536 | 244.17 µs | 268.40 Melem/s |
| AVGPRICE caller compact | 64 | 290.47 ns | 220.33 Melem/s |
| AVGPRICE owned legacy Aligned Output | 64 | 388.01 ns | 164.94 Melem/s |
| AVGPRICE caller compact | 4,096 | 18.040 µs | 227.05 Melem/s |
| AVGPRICE owned legacy Aligned Output | 4,096 | 20.950 µs | 195.51 Melem/s |
| AVGPRICE caller compact | 65,536 | 444.41 µs | 147.47 Melem/s |
| AVGPRICE owned legacy Aligned Output | 65,536 | 316.13 µs | 207.30 Melem/s |
| MINMAX caller compact | 64 | 886.61 ns | 72.185 Melem/s |
| MINMAX owned legacy Aligned Output | 64 | 954.86 ns | 67.026 Melem/s |
| MINMAX caller compact | 4,096 | 52.181 µs | 78.496 Melem/s |
| MINMAX owned legacy Aligned Output | 4,096 | 47.036 µs | 87.083 Melem/s |
| MINMAX caller compact | 65,536 | 782.96 µs | 83.703 Melem/s |
| MINMAX owned legacy Aligned Output | 65,536 | 752.43 µs | 87.099 Melem/s |
| MINMAXINDEX caller compact | 64 | 767.01 ns | 83.440 Melem/s |
| MINMAXINDEX owned legacy Aligned Output | 64 | 865.15 ns | 73.975 Melem/s |
| MINMAXINDEX caller compact | 4,096 | 46.023 µs | 88.999 Melem/s |
| MINMAXINDEX owned legacy Aligned Output | 4,096 | 42.218 µs | 97.021 Melem/s |
| MINMAXINDEX caller compact | 65,536 | 1.0529 ms | 62.241 Melem/s |
| MINMAXINDEX owned legacy Aligned Output | 65,536 | 668.54 µs | 98.029 Melem/s |

These representatives cover single-output, multi-input, multi-output, and index-output shapes through the public trait seam. The mixed issue #3 suite additionally measures `SMAConfig::compute_into` as `caller_compact_config` at the same 64, 4,096, and 65,536 observation sizes, while retaining every historical ID.

## Repeated and streaming workloads

| Current workload | Work per iteration | Latency | Throughput |
|---|---:|---:|---:|
| SMA Universe, one instance and one reused output | 128 × 4,096 observations | 1.7856 ms | 293.62 Melem/s |
| SMA parameter sweep, periods 5/14/50/200 | 4 × 4,096 observations | 45.846 µs | 357.37 Melem/s |
| SMA per-worker predecessor, independent instances | 4 × 4,096 observations | 56.263 µs | 291.21 Melem/s |
| SMA multi-instrument Streaming Computation | 16 × 4,096 ticks | 563.93 µs | 116.21 Melem/s |

The per-worker case is deliberately named `no_prepared_runner/per_worker_instances`: construction is outside steady-state timing, but these are full current indicator instances rather than Prepared Batch Runners. It is the legacy predecessor benchmark; issue #3 will implement and measure the target per-worker Prepared Batch Runner seam. The streaming benchmark interleaves ticks across independent instances; its batched setup and instance destruction are outside timing.

## Allocation and incremental peak heap

The dedicated `execution_allocations` executable wraps `System` with a benchmark-local global allocator. Measurements are isolated and single-threaded. Definitions:

- **operations**: successful `alloc`, `alloc_zeroed`, or `realloc` calls;
- **gross bytes**: requested bytes summed across those operations;
- **peak bytes**: maximum incremental live requested bytes during the measured operation;
- **retained bytes**: incremental requested bytes still live when the operation returns.

Fixtures, caller buffers, and reusable instances created before a workload are excluded. These metrics do not include stack, allocator metadata, unrelated process memory, or operating-system RSS. Incremental requested-heap high-water is the practical repeatable peak-memory measure for these in-process microbenchmarks.

| Scenario | Operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Construct SMA(14) | 1 | 112 | 112 | 112 |
| Construct 4 per-worker SMA instances | 5 | 704 | 704 | 704 |
| Construct SMA parameter sweep | 5 | 2,408 | 2,408 | 2,408 |
| Construct 16 streaming SMA instances | 17 | 2,816 | 2,816 | 2,816 |
| SMA caller compact, 4,096 | 0 | 0 | 0 | 0 |
| SMA owned legacy Aligned Output, 4,096 | 1 | 32,768 | 32,768 | 32,768 |
| AVGPRICE caller compact, 4,096 | 0 | 0 | 0 | 0 |
| AVGPRICE owned legacy Aligned Output, 4,096 | 2 | 65,536 | 65,536 | 32,768 |
| MINMAX caller compact, 4,096 | 2 | 65,536 | 65,536 | 0 |
| MINMAX owned legacy Aligned Output, 4,096 | 6 | 196,608 | 131,072 | 65,536 |
| MINMAXINDEX caller compact, 4,096 | 2 | 65,536 | 65,536 | 0 |
| MINMAXINDEX owned legacy Aligned Output, 4,096 | 6 | 131,072 | 98,304 | 32,768 |
| SMA Universe caller compact, 128 × 4,096 | 0 | 0 | 0 | 0 |
| SMA parameter sweep caller compact, 4 × 4,096 | 0 | 0 | 0 | 0 |
| SMA per-worker predecessor, 4 × 4,096 | 0 | 0 | 0 | 0 |
| MINMAX caller compact, 8 × 4,096 | 16 | 524,288 | 65,536 | 0 |
| SMA streaming ticks, 16 × 4,096 | 0 | 0 | 0 | 0 |

The setup rows expose the current streaming-buffer allocation paid even by batch-only SMA instances. MINMAX and MINMAXINDEX demonstrate that caller-owned output does not imply allocation-free algorithm scratch. The eight-call MINMAX row shows scratch is allocated and released on every current call.

## Issue #3 SMA expansion results

Issue #3 was measured on 2026-08-03 in the same Apple M2 worktree environment. The allocation executable contains executable assertions for the new contracts and was run with:

```text
cargo bench -p ta-benchmarks --bench execution_allocations
```

| New SMA scenario | Operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Construct `SMAConfig(14)` | 0 | 0 | 0 | 0 |
| `SMAConfig::compute_into`, 4,096 | 0 | 0 | 0 | 0 |
| `SMAConfig::compute` owned Compact Output, 4,096 | 1 | 32,664 | 32,664 | 32,664 |
| `SMAConfig::compute` owned Compact Output, count 0 | 0 | 0 | 0 | 0 |
| `SMAConfig::compute` owned Compact Output, count 1 | 1 | 8 | 8 | 8 |
| `SMAConfig::compute` owned Compact Output, count 2 | 1 | 16 | 16 | 16 |
| `SMAConfig::compute` owned Compact Output, count 3 | 1 | 24 | 24 | 24 |
| Prepare `SMABatchRunner(4,096)` | 0 | 0 | 0 | 0 |
| First prepared call, 4,096 | 0 | 0 | 0 | 0 |
| Repeated prepared call, 4,096 | 0 | 0 | 0 | 0 |
| Oversize prepared rejection, 4,097 | 0 | 0 | 0 | 0 |
| New independent stream ticks, 16 × 4,096 | 0 | 0 | 0 | 0 |

The owned payload has 4,083 valid `f64` values, so 32,664 bytes is exactly one compact allocation. The isolated count 0/1/2/3 profiles assert the exact-allocation boundary at zero and at allocator-small payload sizes. Configuration and preparation retain no heap memory for SMA. Stream buffers are constructed outside the tick measurement. The issue #2 legacy rows remain present and asserted behavior remains unchanged: legacy `SMA(14)` construction allocates 112 bytes, caller Compact Output allocates zero, legacy owned Aligned Output allocates 32,768 bytes, and legacy stream ticks allocate zero.

### Full Criterion acceptance evidence

Two full Criterion runs were performed after the sealed-trait and legacy-adapter fixes for repeated and streaming workloads. After the exact-allocation fix and addition of the new caller-owned benchmark, two further full one-shot runs were performed. Every run used the default 100 samples per benchmark case. The tables report same-run median deltas; positive values mean the candidate was slower, and negative values mean it was faster.

The new `caller_compact_config` cases use `IndicatorConfig::compute_into(&SMAConfig, ...)`, the same deterministic fixture and `black_box` boundaries as the legacy `caller_compact` predecessor, and a reusable output allocated outside the measured loop:

| New caller-owned path | Same-run predecessor | Run A median delta | Run B median delta |
|---|---|---:|---:|
| `caller_compact_config`, 64 observations | `caller_compact`, 64 observations | -0.24% | -0.91% |
| `caller_compact_config`, 4,096 observations | `caller_compact`, 4,096 observations | +0.45% | -0.64% |
| `caller_compact_config`, 65,536 observations | `caller_compact`, 65,536 observations | -1.27% | +0.56% |

All other new paths are likewise compared with their predecessors from the same run, avoiding cross-run host drift:

| Candidate path | Same-run predecessor | Run A median delta | Run B median delta |
|---|---|---:|---:|
| Owned Compact Output, 64 observations | Legacy owned Aligned Output | +0.68% | +1.58% |
| Owned Compact Output, 4,096 observations | Legacy owned Aligned Output | -0.31% | -1.42% |
| Owned Compact Output, 65,536 observations | Legacy owned Aligned Output | -4.37% | -4.86% |
| Prepared Batch Runner, Universe | Legacy caller-owned Universe | +0.48% | -0.43% |
| Prepared Batch Runner, parameter sweep | Legacy caller-owned parameter sweep | +0.69% | +0.61% |
| Prepared Batch Runner, per-worker | Independent legacy indicator instances | +0.08% | +0.13% |
| Independent `SMAConfig` Streaming Computation | Independent legacy `SMA` Streaming Computation | -0.39% | -0.38% |

Preserved-current IDs were compared separately with the freshly recaptured `issue-2-final` baseline. Their observed median-drift ranges across the two full runs were:

| Preserved-current path | Observations / workload | Median drift range |
|---|---|---:|
| Caller-owned Compact Output | 64 | +4.29% to +4.68% |
| Caller-owned Compact Output | 4,096 | +3.15% to +3.33% |
| Caller-owned Compact Output | 65,536 | +2.33% to +4.83% |
| Legacy owned Aligned Output | 64 | +2.76% to +3.47% |
| Legacy owned Aligned Output | 4,096 | +2.62% to +4.51% |
| Legacy owned Aligned Output | 65,536 | +2.96% to +3.03% |
| Caller-owned Universe | 128 × 4,096 | -1.40% to -0.42% |
| Caller-owned parameter sweep | 4 × 4,096 | +1.71% to +3.16% |
| Per-worker legacy indicator instances | 4 × 4,096 | +1.24% to +3.46% |
| Independent legacy Streaming Computation | 16 × 4,096 ticks | +0.13% to +2.31% |

Gate conclusion: all new paths fall between -4.86% and +1.58% of same-run predecessors across both final runs. Every preserved-current path remains below +5% in both final runs. No path has a stable >~5% regression. These host-local comparisons establish acceptance against the regression gate; they do not establish portable speedups.

## Migration gate

For issues #3 and later:

1. Keep workload topology, fixture generation, sizes, periods, precision, toolchain, and host fixed.
2. On pre-change `main`, capture the existing suite with `cargo bench -p ta-benchmarks --bench execution_baselines -- --save-baseline <name>`.
3. After the change, run the new mixed suite twice at full sampling (100 samples per case) on an otherwise idle host. Do not apply `--baseline` to the whole mixed suite: Criterion 0.8 panics when the suite contains newly introduced benchmark IDs that are absent from the saved baseline.
4. Compare every new path with its predecessor from the same run.
5. For each preserved ID, compare it separately with `cargo bench -p ta-benchmarks --bench execution_baselines -- --exact --baseline <name> <full-existing-ID>`.
6. Treat only a regression reproduced across the full runs as stable. A stable regression greater than approximately 5% requires an explicit accepted trade-off under ADR-0001.
7. Run `cargo bench -p ta-benchmarks --bench execution_allocations` and evaluate operations, gross bytes, peak bytes, and retained bytes together with timing.
8. Extend the current contract test file beside the legacy assertions; preserve current owned results as legacy Aligned Output rather than relabeling them as Compact Output.

The rejected append and period-bounded ring scratch implementations from `prototype/output-interface-benchmark` are not included in production or in this baseline. Only their measurement lessons informed the benchmark design.
