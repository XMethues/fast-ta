---
date: 2026-08-03
repository: fast-ta
branch: issue-4-output-interface
base_commit: 8c99de0
comparison_baseline: issue-2-final
issue_2_criterion_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- --save-baseline issue-2-final"
issue_3_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines"
issue_4_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema"
issue_4_extrema_workloads_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema_workloads"
issue_4_streaming_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/extrema_workloads/.*/streaming'"
allocation_command: "cargo bench -p ta-benchmarks --bench execution_allocations"
status: issue-4-adapters-and-owned-qualified
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
- Precision: default `f64`; performance evidence in this document is host-local default-`f64`
- Correctness precision coverage: CI additionally runs `cargo test -p ta-core --no-default-features --features f32,std` on stable x86_64 Linux
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

## Issue #4 MINMAX and MINMAXINDEX qualification

Issue #4 adds parameter-only `MINMAXConfig` and `MINMAXINDEXConfig`, compact named value or `usize` index columns, caller-owned computation, independent streams, and one Prepared Batch Runner per worker. The production scratch is a specialized reusable append design: preparation reserves two `Vec<usize>` columns for `max_input_len`, each call validates before clearing them, queue heads remain local, and the concrete value/index kernels write directly to their final slices. A private proven-capacity append removes the otherwise-unnecessary `Vec::push` growth branch only for prepared execution; one-shot configuration paths retain ordinary local `Vec` scratch.

The durable commands are:

```text
cargo bench -p ta-benchmarks --bench execution_allocations
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema
```

The full command was run twice at Criterion's default 100 samples. The valid matrix is the cross-product of observations 64/4,096/65,536 and periods 14/512 where `observations >= period`; 64/512 is mathematically invalid and is intentionally absent. Each case rotates registration order across all five paths: legacy caller-owned Compact Output, configuration caller-owned Compact Output, Prepared Batch Runner, legacy owned Aligned Output, and configuration owned Compact Output. Existing issue #2/#3 benchmark IDs were not renamed.

### Exact allocation profiles

These executable assertions use 4,096 observations, period 14, `Float = f64`, and 64-bit `usize`:

| Scenario | Operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Construct `MINMAXConfig` | 0 | 0 | 0 | 0 |
| Construct `MINMAXINDEXConfig` | 0 | 0 | 0 | 0 |
| MINMAX caller-owned one-shot | 2 | 65,536 | 65,536 | 0 |
| MINMAXINDEX caller-owned one-shot | 2 | 65,536 | 65,536 | 0 |
| MINMAX owned Compact Output | 4 | 130,864 | 130,864 | 65,328 |
| MINMAXINDEX owned Compact Output | 4 | 130,864 | 130,864 | 65,328 |
| Empty MINMAX owned Compact Output | 0 | 0 | 0 | 0 |
| Empty MINMAXINDEX owned Compact Output | 0 | 0 | 0 | 0 |
| Prepare `MINMAXBatchRunner(4,096)` | 2 | 65,536 | 65,536 | 65,536 |
| Prepare `MINMAXINDEXBatchRunner(4,096)` | 2 | 65,536 | 65,536 | 65,536 |
| Prepared first call, either path | 0 | 0 | 0 | 0 |
| Prepared repeated call, either path | 0 | 0 | 0 | 0 |
| Prepared oversize rejection, either path | 0 | 0 | 0 | 0 |
| Stream ticks after construction, either path | 0 | 0 | 0 | 0 |

Caller-owned output allocation is zero; its two allocations are documented algorithm scratch. Owned computation requests exactly two compact output allocations plus those two scratch allocations. The retained 65,328 bytes are exactly two output columns of 4,083 elements. Preparation requests exactly `2 * max_input_len * size_of::<usize>()` bytes. Refactoring the uppercase compatibility objects does not change these profiles: each adapter owns exactly the corresponding stream state and introduces no additional allocation.

### Legacy compatibility adapters

The uppercase `MINMAX` and `MINMAXINDEX` objects now follow the issue #3 SMA pattern. Each stores only a `MINMAXStream` or `MINMAXINDEXStream`; constructors create the corresponding validated configuration and stream, streaming/reset calls delegate to that stream, and the historical batch methods continue to delegate to the unchanged legacy free functions. This removes the second copy of ring-buffer transitions while preserving periods, lookbacks, aligned owned outputs, compact caller-owned outputs, warm-up, oldest-tie behavior, cloning, reset/replay, and every public legacy signature. Public contract tests also assert that each adapter has the same size as its stream, clones diverge independently, rejected non-finite ticks preserve replay position, and reset reproduces the same output sequence.

`MINMAXINDEXStream` has one private transactional transition path shared by configured and legacy execution. It validates the tick and both the native position increment and caller-supplied maximum position, derives the pending extrema, and maps the output before committing any state. The legacy adapter supplies `i32::MAX` and converts both `usize` outputs with `i32::try_from` at the adapter boundary. Unit tests force conversion-limit and native-position-overflow failures and verify that buffer, indexes, cursor, count, and seen position are unchanged.

### Full timing acceptance evidence

The table reports median latency and prepared deltas against both same-run predecessors. Negative deltas mean prepared execution was faster.

| Path | Observations / period | Run A current / config / prepared | Run A deltas | Run B current / config / prepared | Run B deltas |
|---|---:|---:|---:|---:|---:|
| MINMAX | 64 / 14 | 623.0 / 582.8 / 426.8 ns | -31.49% / -26.77% | 595.6 / 584.7 / 438.2 ns | -26.42% / -25.05% |
| MINMAX | 4,096 / 14 | 33.394 / 31.901 / 28.402 µs | -14.95% / -10.97% | 33.609 / 32.108 / 28.471 µs | -15.29% / -11.33% |
| MINMAX | 4,096 / 512 | 33.160 / 32.037 / 28.079 µs | -15.32% / -12.35% | 33.298 / 32.209 / 28.243 µs | -15.18% / -12.31% |
| MINMAX | 65,536 / 14 | 529.59 / 509.72 / 469.31 µs | -11.38% / -7.93% | 528.07 / 518.98 / 458.64 µs | -13.15% / -11.63% |
| MINMAX | 65,536 / 512 | 546.25 / 534.44 / 465.17 µs | -14.84% / -12.96% | 552.80 / 539.59 / 474.14 µs | -14.23% / -12.13% |
| MINMAXINDEX | 64 / 14 | 581.9 / 550.4 / 411.4 ns | -29.31% / -25.26% | 581.0 / 556.9 / 422.7 ns | -27.24% / -24.09% |
| MINMAXINDEX | 4,096 / 14 | 32.243 / 30.700 / 27.912 µs | -13.43% / -9.08% | 33.327 / 31.412 / 27.607 µs | -17.16% / -12.11% |
| MINMAXINDEX | 4,096 / 512 | 31.990 / 30.722 / 26.898 µs | -15.92% / -12.45% | 32.923 / 30.650 / 27.263 µs | -17.19% / -11.05% |
| MINMAXINDEX | 65,536 / 14 | 519.95 / 496.28 / 447.48 µs | -13.94% / -9.83% | 521.22 / 497.61 / 448.71 µs | -13.91% / -9.83% |
| MINMAXINDEX | 65,536 / 512 | 528.35 / 514.74 / 461.57 µs | -12.64% / -10.33% | 529.57 / 505.33 / 453.01 µs | -14.46% / -10.35% |

The selected append implementation clears the approximately-five-percent gate in both full runs against both current and configuration one-shot predecessors. These are host-local results, not portable speedup claims.

### Owned one-shot qualification

Owned qualification uses the same deterministic fixture, valid observation/period matrix, `black_box` boundaries, `BatchSize::LargeInput`, and rotating registration order as the caller-owned and prepared paths. `legacy_owned_aligned` calls `Indicator::compute_to_vec`; `config_owned_compact` calls `IndicatorConfig::compute`. Medians below are legacy / configuration owned, with the configuration delta against the same-run legacy result.

| Path | Observations / period | Run A medians | Run A delta | Run B medians | Run B delta |
|---|---:|---:|---:|---:|---:|
| MINMAX | 64 / 14 | 837.67 / 648.39 ns | -22.60% | 831.03 / 636.10 ns | -23.46% |
| MINMAX | 4,096 / 14 | 41.940 / 35.590 µs | -15.14% | 40.301 / 34.073 µs | -15.45% |
| MINMAX | 4,096 / 512 | 40.919 / 34.898 µs | -14.71% | 40.290 / 35.187 µs | -12.67% |
| MINMAX | 65,536 / 14 | 644.655 / 546.720 µs | -15.19% | 659.438 / 566.574 µs | -14.08% |
| MINMAX | 65,536 / 512 | 657.955 / 569.532 µs | -13.44% | 678.788 / 569.619 µs | -16.08% |
| MINMAXINDEX | 64 / 14 | 734.85 / 620.78 ns | -15.52% | 770.47 / 651.61 ns | -15.43% |
| MINMAXINDEX | 4,096 / 14 | 37.163 / 33.882 µs | -8.83% | 37.550 / 34.606 µs | -7.84% |
| MINMAXINDEX | 4,096 / 512 | 38.087 / 33.853 µs | -11.12% | 37.916 / 34.379 µs | -9.33% |
| MINMAXINDEX | 65,536 / 14 | 590.111 / 529.161 µs | -10.33% | 610.147 / 555.317 µs | -8.99% |
| MINMAXINDEX | 65,536 / 512 | 590.894 / 545.674 µs | -7.65% | 589.335 / 557.069 µs | -5.47% |

Gate conclusion: configuration owned Compact Output was faster in every case in both 100-sample runs, ranging from -23.46% to -5.47%. There is no stable approximately-five-percent regression and no owned-path optimization or block is required. The allocation executable still confirms the exact four-allocation profile: two compact output columns and two input-length scratch queues.

### Repeated and streaming extrema qualification

The final-review expansion adds new IDs under `indicator_execution/expanded/extrema_workloads` without renaming any existing ID. Both MINMAX and MINMAXINDEX now cover:

- a 128-instrument Universe of 4,096-observation series, processed repeatedly by one current instance, one configuration, or one Prepared Batch Runner;
- a parameter sweep over periods 5/14/50/200 on one 4,096-observation fixture, with current instances, configurations, or one prepared runner per period;
- four independent worker fixtures, comparing four current instances with one Prepared Batch Runner per worker; and
- 16 independent streams over 4,096 ticks, comparing legacy instances with configured streams.

Fixtures are deterministic and identical across each comparison. Caller-owned output columns are allocated once outside the measured loops and reused; stream construction/reset is in Criterion's batched setup rather than the operation boundary. Every path uses matching `black_box` boundaries around execution state, input, output columns, and observed results. The 128-instrument Universe deliberately reuses the shared baseline constant: it is large enough to exercise repeated scratch reuse while remaining manageable at full sampling.

Development and acceptance commands were:

```text
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema_workloads --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema_workloads
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/extrema_workloads
cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/extrema_workloads/.*/streaming' --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/extrema_workloads/.*/streaming'
cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/extrema_workloads/.*/streaming'
```

The quick runs completed before the acceptance runs. Each full run used Criterion's default 100 samples. The tables report middle estimates from the same full run; negative deltas mean the candidate took less time. Universe and parameter-sweep timing columns are current / configuration / prepared. Per-worker columns are current / prepared, and streaming columns are legacy / configured. Non-streaming rows retain the prior final-review evidence because the adapter refactor does not affect batch execution; the streaming rows below supersede the earlier measurements with the two post-adapter full runs.

| Indicator / workload | Run A medians | Run A same-run deltas | Run B medians | Run B same-run deltas |
|---|---:|---:|---:|---:|
| MINMAX Universe, 128 × 4,096 | 4.4732 / 4.2599 / 3.8215 ms | config/current -4.77%; prepared/current -14.57%; prepared/config -10.29% | 4.4122 / 4.2912 / 3.8868 ms | config/current -2.74%; prepared/current -11.91%; prepared/config -9.42% |
| MINMAX sweep, 4 × 4,096 | 136.69 / 132.31 / 117.18 µs | config/current -3.20%; prepared/current -14.27%; prepared/config -11.44% | 139.73 / 138.03 / 124.23 µs | config/current -1.22%; prepared/current -11.09%; prepared/config -10.00% |
| MINMAX per-worker, 4 × 4,096 | 135.73 / 119.01 µs | prepared/current -12.32% | 143.28 / 123.75 µs | prepared/current -13.63% |
| MINMAX streaming, 16 × 4,096 | 1.1632 / 1.1785 ms | configured/legacy +1.31% | 1.1457 / 1.1415 ms | configured/legacy -0.37% |
| MINMAXINDEX Universe, 128 × 4,096 | 4.3869 / 4.1742 / 3.6496 ms | config/current -4.85%; prepared/current -16.81%; prepared/config -12.57% | 4.3786 / 4.1147 / 3.5547 ms | config/current -6.03%; prepared/current -18.82%; prepared/config -13.61% |
| MINMAXINDEX sweep, 4 × 4,096 | 132.33 / 125.24 / 108.43 µs | config/current -5.36%; prepared/current -18.06%; prepared/config -13.42% | 134.64 / 125.90 / 110.17 µs | config/current -6.49%; prepared/current -18.17%; prepared/config -12.49% |
| MINMAXINDEX per-worker, 4 × 4,096 | 134.94 / 111.50 µs | prepared/current -17.37% | 137.60 / 113.78 µs | prepared/current -17.31% |
| MINMAXINDEX streaming, 16 × 4,096 | 2.6099 / 2.4418 ms | configured/legacy -6.44% | 2.6044 / 2.3635 ms | configured/legacy -9.25% |

Gate conclusion: no new repeated or streaming candidate has a stable regression near the approximately-five-percent gate. In the post-adapter streaming reruns, MINMAX configured streams ranged from -0.37% to +1.31% versus the delegating legacy adapter, while MINMAXINDEX configured streams were -6.44% and -9.25% faster than the checked-`i32` legacy adapter. Prepared execution stayed below both same-run predecessors in both earlier runs. These measurements qualify regressions only on the recorded host and default `f64` build; they make no portable speedup claim. `f32` is a CI correctness configuration, not part of this performance evidence.

### Rejected fallback evidence

An ordinary reusable append initially exposed a stable approximately 5–6% index-path delta against the same-run configuration path, so the prescribed fallbacks were tested before the retained-capacity invariant was made optimizer-visible:

- Separate single-extrema append passes were rejected after the quick matrix showed broad regressions, including approximately +7% to +29% for representative value cases and up to +44.6% for index cases.
- An explicit-branch, no-modulo ring was rejected at +17.7% to +108.9% versus append across the valid index matrix.
- A bounded compacting deque won one 4,096/14 quick case by 8.3%, but regressed the other valid cases by +6.5% to +55.3%; it was rejected as non-clearing.

The fallback implementations were temporary benchmark candidates and are not retained. The selected production candidate remains reusable append scratch; its private reserved append is justified by the validated capacity invariant and produced the full-run results above.
