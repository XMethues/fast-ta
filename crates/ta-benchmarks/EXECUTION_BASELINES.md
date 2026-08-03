---
date: 2026-08-02
repository: fast-ta
branch: feat/2-execution-baselines
base_commit: c3b28d3
baseline: issue-2-current
criterion_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- --save-baseline issue-2-current"
allocation_command: "cargo bench -p ta-benchmarks --bench execution_allocations"
status: captured
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

Criterion results below are the middle estimates from the confidence intervals. Owned-result timing uses batched measurement so result destruction is outside the operation boundary. Streaming setup and teardown are also outside timing. `target/criterion/issue-2-current` is host-local and ignored by git; this document is the durable record.

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

These representatives cover single-output, multi-input, multi-output, and index-output shapes through the public trait seam.

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

## Migration gate

For issues #3 and later:

1. Keep workload topology, fixture generation, sizes, periods, precision, toolchain, and host fixed.
2. Run `cargo bench -p ta-benchmarks --bench execution_baselines -- --baseline issue-2-current`.
3. Repeat on an otherwise idle host before treating a change as stable.
4. A stable regression greater than approximately 5% requires an explicit accepted trade-off under ADR-0001.
5. Run `execution_allocations` and compare operations, gross bytes, peak bytes, and retained bytes together with timing.
6. Extend the current contract test file beside the legacy assertions; preserve current owned results as legacy Aligned Output rather than relabeling them as Compact Output.

The rejected append and period-bounded ring scratch implementations from `prototype/output-interface-benchmark` are not included in production or in this baseline. Only their measurement lessons informed the benchmark design.
