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
issue_7_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/recursive_overlap"
issue_7_repeat_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/recursive_overlap/MA_EMA'"
issue_8_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/price_transform"
issue_8_streaming_repeat_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- 'indicator_execution/expanded/price_transform_workloads/.*/streaming'"
issue_9_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/volume"
issue_9_streaming_repeat_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/volume_workloads/ADOSC/streaming"
issue_10_full_command: "cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/volatility"
allocation_command: "cargo bench -p ta-benchmarks --bench execution_allocations"
status: issue-10-volatility-qualified
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

- Separate single-extrema append passes as a replacement for each combined-indicator pass were rejected after the quick matrix showed broad regressions, including approximately +7% to +29% for representative value cases and up to +44.6% for index cases.
- An explicit-branch, no-modulo ring was rejected at +17.7% to +108.9% versus append across the valid index matrix.
- A bounded compacting deque won one 4,096/14 quick case by 8.3%, but regressed the other valid cases by +6.5% to +55.3%; it was rejected as non-clearing.

The fallback implementations were temporary benchmark candidates and are not retained. The selected production candidate remains reusable append scratch; its private reserved append is justified by the validated capacity invariant and produced the full-run results above.

## Issue #5 remaining extrema qualification

Issue #5 adds parameter-only `MINConfig`, `MAXConfig`, `MININDEXConfig`, and
`MAXINDEXConfig` types. Each configuration supports owned Compact Output,
caller-owned Batch Computation, a Prepared Batch Runner, and an independent
Streaming Computation. Value outputs are compact `Vec<Float>` payloads; index
outputs are compact `Vec<usize>` payloads containing absolute source indexes.
Warm-up remains represented by the output range or `Option::None`, never by a
numeric index.

Every single-sided batch kernel maintains exactly one monotonic index queue.
One-shot configured and legacy caller-owned paths allocate that one
input-length queue. Each Prepared Batch Runner reserves and retains one queue
for its declared maximum input length, then uses the same proven-capacity append
as the qualified combined-extrema runners. The uppercase compatibility objects
store only their corresponding configured stream and delegate streaming and
reset behavior through the Rust-first seam.

The durable commands are:

```text
cargo bench -p ta-benchmarks --bench execution_allocations
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/single_extrema --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/single_extrema
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/single_extrema_workloads --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/single_extrema_workloads
```

The one-shot matrix uses the same valid observation/period pairs and rotating
five-path registration order as issue #4. The repeated matrix covers a
128-instrument Universe, a four-period parameter sweep, one runner per each of
four workers, and 16 independent streams over 4,096 ticks. Fixtures and
`black_box` placement match the issue #4 workloads.

### Exact single-extrema allocation profiles

These executable assertions use 4,096 observations, period 14, `Float = f64`,
and 64-bit `usize`:

| Scenario | Operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Construct any single-extrema configuration | 0 | 0 | 0 | 0 |
| Legacy MIN/MAX/MININDEX/MAXINDEX caller-owned one-shot | 1 | 32,768 | 32,768 | 0 |
| Configured MIN/MAX/MININDEX/MAXINDEX caller-owned one-shot | 1 | 32,768 | 32,768 | 0 |
| Configured owned Compact Output, any single-extrema path | 2 | 65,432 | 65,432 | 32,664 |
| Prepare any single-extrema runner for 4,096 observations | 1 | 32,768 | 32,768 | 32,768 |
| Prepared first or repeated call, any single-extrema path | 0 | 0 | 0 | 0 |
| Stream ticks after construction, any single-extrema path | 0 | 0 | 0 | 0 |

Caller-owned output allocation is zero; the one allocation is documented
algorithm scratch. Owned computation performs exactly one compact output
allocation and one scratch allocation. The retained 32,664 bytes are exactly
one output column of 4,083 elements. Prepared construction retains exactly one
`max_input_len * size_of::<usize>()` queue. These profiles demonstrate that no
single-sided computation carries an unused opposite-extrema queue.

### Full timing evidence

The command above was run twice at Criterion's default 100 samples. Criterion recorded both latency and element throughput; the tables retain median latency because throughput is its inverse for each fixed-size workload. Negative deltas mean the candidate took less time. The caller-owned tables report current free function / configuration / Prepared Batch Runner.

| Indicator | Observations / period | Run A medians | Run A same-run deltas | Run B medians | Run B same-run deltas |
|---|---:|---:|---:|---:|---:|
| MIN | 64 / 14 | 220.80 ns / 233.22 ns / 172.35 ns | config/current +5.62%; prepared/current -21.94%; prepared/config -26.10% | 217.46 ns / 232.96 ns / 172.01 ns | config/current +7.13%; prepared/current -20.90%; prepared/config -26.16% |
| MIN | 4,096 / 14 | 12.903 µs / 12.461 µs / 11.140 µs | config/current -3.42%; prepared/current -13.66%; prepared/config -10.61% | 14.913 µs / 15.886 µs / 15.273 µs | config/current +6.53%; prepared/current +2.41%; prepared/config -3.86% |
| MIN | 4,096 / 512 | 15.253 µs / 16.270 µs / 11.017 µs | config/current +6.66%; prepared/current -27.77%; prepared/config -32.28% | 24.673 µs / 30.821 µs / 18.130 µs | config/current +24.92%; prepared/current -26.52%; prepared/config -41.18% |
| MIN | 65,536 / 14 | 276.798 µs / 314.620 µs / 268.186 µs | config/current +13.66%; prepared/current -3.11%; prepared/config -14.76% | 350.569 µs / 417.730 µs / 420.145 µs | config/current +19.16%; prepared/current +19.85%; prepared/config +0.58% |
| MIN | 65,536 / 512 | 348.219 µs / 386.586 µs / 303.586 µs | config/current +11.02%; prepared/current -12.82%; prepared/config -21.47% | 467.942 µs / 462.111 µs / 361.204 µs | config/current -1.25%; prepared/current -22.81%; prepared/config -21.84% |
| MAX | 64 / 14 | 388.37 ns / 367.23 ns / 238.47 ns | config/current -5.44%; prepared/current -38.60%; prepared/config -35.06% | 370.95 ns / 355.91 ns / 282.24 ns | config/current -4.06%; prepared/current -23.92%; prepared/config -20.70% |
| MAX | 4,096 / 14 | 20.954 µs / 17.933 µs / 18.935 µs | config/current -14.41%; prepared/current -9.64%; prepared/config +5.58% | 31.568 µs / 23.914 µs / 22.220 µs | config/current -24.25%; prepared/current -29.61%; prepared/config -7.08% |
| MAX | 4,096 / 512 | 20.175 µs / 21.623 µs / 15.228 µs | config/current +7.17%; prepared/current -24.52%; prepared/config -29.58% | 25.723 µs / 20.716 µs / 24.257 µs | config/current -19.46%; prepared/current -5.70%; prepared/config +17.09% |
| MAX | 65,536 / 14 | 300.102 µs / 569.993 µs / 257.819 µs | config/current +89.93%; prepared/current -14.09%; prepared/config -54.77% | 477.291 µs / 616.450 µs / 585.520 µs | config/current +29.16%; prepared/current +22.68%; prepared/config -5.02% |
| MAX | 65,536 / 512 | 330.461 µs / 333.746 µs / 301.359 µs | config/current +0.99%; prepared/current -8.81%; prepared/config -9.70% | 757.238 µs / 741.822 µs / 833.299 µs | config/current -2.04%; prepared/current +10.04%; prepared/config +12.33% |
| MININDEX | 64 / 14 | 316.29 ns / 338.54 ns / 249.04 ns | config/current +7.04%; prepared/current -21.26%; prepared/config -26.44% | 1.587 µs / 1.083 µs / 679.68 ns | config/current -31.77%; prepared/current -57.16%; prepared/config -37.21% |
| MININDEX | 4,096 / 14 | 20.905 µs / 15.143 µs / 13.198 µs | config/current -27.56%; prepared/current -36.87%; prepared/config -12.85% | 32.468 µs / 34.702 µs / 31.948 µs | config/current +6.88%; prepared/current -1.60%; prepared/config -7.93% |
| MININDEX | 4,096 / 512 | 16.162 µs / 19.218 µs / 15.210 µs | config/current +18.91%; prepared/current -5.89%; prepared/config -20.86% | 34.046 µs / 29.511 µs / 28.093 µs | config/current -13.32%; prepared/current -17.49%; prepared/config -4.81% |
| MININDEX | 65,536 / 14 | 324.021 µs / 309.220 µs / 277.086 µs | config/current -4.57%; prepared/current -14.49%; prepared/config -10.39% | 232.919 µs / 228.840 µs / 210.993 µs | config/current -1.75%; prepared/current -9.41%; prepared/config -7.80% |
| MININDEX | 65,536 / 512 | 336.086 µs / 274.701 µs / 214.607 µs | config/current -18.26%; prepared/current -36.15%; prepared/config -21.88% | 199.964 µs / 195.551 µs / 175.122 µs | config/current -2.21%; prepared/current -12.42%; prepared/config -10.45% |
| MAXINDEX | 64 / 14 | 241.50 ns / 258.40 ns / 198.24 ns | config/current +7.00%; prepared/current -17.91%; prepared/config -23.28% | 234.58 ns / 219.28 ns / 204.46 ns | config/current -6.52%; prepared/current -12.84%; prepared/config -6.76% |
| MAXINDEX | 4,096 / 14 | 11.354 µs / 12.987 µs / 11.301 µs | config/current +14.38%; prepared/current -0.46%; prepared/config -12.98% | 11.479 µs / 15.146 µs / 12.566 µs | config/current +31.95%; prepared/current +9.47%; prepared/config -17.03% |
| MAXINDEX | 4,096 / 512 | 18.423 µs / 17.181 µs / 13.087 µs | config/current -6.74%; prepared/current -28.96%; prepared/config -23.82% | 13.831 µs / 12.452 µs / 11.187 µs | config/current -9.97%; prepared/current -19.12%; prepared/config -10.16% |
| MAXINDEX | 65,536 / 14 | 225.749 µs / 267.170 µs / 212.291 µs | config/current +18.35%; prepared/current -5.96%; prepared/config -20.54% | 212.302 µs / 228.138 µs / 188.764 µs | config/current +7.46%; prepared/current -11.09%; prepared/config -17.26% |
| MAXINDEX | 65,536 / 512 | 224.260 µs / 254.226 µs / 221.136 µs | config/current +13.36%; prepared/current -1.39%; prepared/config -13.02% | 240.445 µs / 430.918 µs / 247.509 µs | config/current +79.22%; prepared/current +2.94%; prepared/config -42.56% |

Owned qualification uses legacy owned Aligned Output / configuration owned Compact Output:

| Indicator | Observations / period | Run A medians | Run A delta | Run B medians | Run B delta |
|---|---:|---:|---:|---:|---:|
| MIN | 64 / 14 | 288.79 ns / 248.84 ns | -13.83% | 316.94 ns / 293.99 ns | -7.24% |
| MIN | 4,096 / 14 | 13.860 µs / 13.181 µs | -4.90% | 15.400 µs / 16.044 µs | +4.18% |
| MIN | 4,096 / 512 | 14.183 µs / 13.543 µs | -4.51% | 27.243 µs / 28.722 µs | +5.43% |
| MIN | 65,536 / 14 | 329.475 µs / 251.645 µs | -23.62% | 541.806 µs / 345.153 µs | -36.30% |
| MIN | 65,536 / 512 | 507.022 µs / 350.458 µs | -30.88% | 396.060 µs / 393.615 µs | -0.62% |
| MAX | 64 / 14 | 413.33 ns / 352.67 ns | -14.68% | 589.04 ns / 404.01 ns | -31.41% |
| MAX | 4,096 / 14 | 24.433 µs / 18.687 µs | -23.52% | 31.935 µs / 28.956 µs | -9.33% |
| MAX | 4,096 / 512 | 23.232 µs / 22.222 µs | -4.35% | 35.805 µs / 30.109 µs | -15.91% |
| MAX | 65,536 / 14 | 411.472 µs / 286.341 µs | -30.41% | 379.376 µs / 362.215 µs | -4.52% |
| MAX | 65,536 / 512 | 398.959 µs / 345.911 µs | -13.30% | 1.6139 ms / 771.585 µs | -52.19% |
| MININDEX | 64 / 14 | 363.41 ns / 397.08 ns | +9.27% | 1.229 µs / 996.22 ns | -18.97% |
| MININDEX | 4,096 / 14 | 19.104 µs / 16.982 µs | -11.11% | 34.400 µs / 34.566 µs | +0.48% |
| MININDEX | 4,096 / 512 | 17.472 µs / 16.863 µs | -3.49% | 35.531 µs / 33.097 µs | -6.85% |
| MININDEX | 65,536 / 14 | 317.310 µs / 279.028 µs | -12.06% | 268.824 µs / 248.572 µs | -7.53% |
| MININDEX | 65,536 / 512 | 277.581 µs / 300.139 µs | +8.13% | 235.266 µs / 234.177 µs | -0.46% |
| MAXINDEX | 64 / 14 | 265.73 ns / 320.75 ns | +20.71% | 249.15 ns / 333.00 ns | +33.65% |
| MAXINDEX | 4,096 / 14 | 12.573 µs / 14.492 µs | +15.26% | 12.864 µs / 23.501 µs | +82.69% |
| MAXINDEX | 4,096 / 512 | 15.142 µs / 25.025 µs | +65.27% | 13.039 µs / 18.676 µs | +43.23% |
| MAXINDEX | 65,536 / 14 | 232.148 µs / 326.873 µs | +40.80% | 196.703 µs / 286.947 µs | +45.88% |
| MAXINDEX | 65,536 / 512 | 242.921 µs / 318.238 µs | +31.00% | 532.993 µs / 358.698 µs | -32.70% |

### Repeated and streaming evidence

Universe and parameter-sweep timing columns are current / configuration / prepared. Per-worker columns are current / prepared, and streaming columns are legacy / configured. Fixtures and output buffers are identical within each comparison; construction/reset stays outside the measured operation.

| Indicator / workload | Run A medians | Run A same-run deltas | Run B medians | Run B same-run deltas |
|---|---:|---:|---:|---:|
| MIN Universe, 128 × 4,096 | 1.9384 ms / 1.8868 ms / 1.6781 ms | config/current -2.66%; prepared/current -13.43%; prepared/config -11.06% | 1.9450 ms / 1.8736 ms / 1.9159 ms | config/current -3.67%; prepared/current -1.49%; prepared/config +2.26% |
| MIN sweep, 4 × 4,096 | 62.537 µs / 64.086 µs / 56.631 µs | config/current +2.48%; prepared/current -9.44%; prepared/config -11.63% | 59.589 µs / 58.610 µs / 56.998 µs | config/current -1.64%; prepared/current -4.35%; prepared/config -2.75% |
| MIN per-worker, 4 × 4,096 | 60.061 µs / 48.022 µs | candidate/reference -20.04% | 98.715 µs / 115.430 µs | candidate/reference +16.93% |
| MIN streaming, 16 × 4,096 | 577.029 µs / 604.292 µs | candidate/reference +4.72% | 924.289 µs / 755.175 µs | candidate/reference -18.30% |
| MAX Universe, 128 × 4,096 | 1.8344 ms / 1.8522 ms / 1.6408 ms | config/current +0.97%; prepared/current -10.55%; prepared/config -11.41% | 2.5447 ms / 2.5338 ms / 1.9395 ms | config/current -0.43%; prepared/current -23.78%; prepared/config -23.46% |
| MAX sweep, 4 × 4,096 | 59.211 µs / 61.308 µs / 51.467 µs | config/current +3.54%; prepared/current -13.08%; prepared/config -16.05% | 87.853 µs / 78.112 µs / 70.285 µs | config/current -11.09%; prepared/current -20.00%; prepared/config -10.02% |
| MAX per-worker, 4 × 4,096 | 64.009 µs / 56.230 µs | candidate/reference -12.15% | 60.538 µs / 56.216 µs | candidate/reference -7.14% |
| MAX streaming, 16 × 4,096 | 621.998 µs / 540.419 µs | candidate/reference -13.12% | 606.706 µs / 1.2450 ms | candidate/reference +105.21% |
| MININDEX Universe, 128 × 4,096 | 1.7480 ms / 1.8964 ms / 1.6191 ms | config/current +8.49%; prepared/current -7.38%; prepared/config -14.62% | 4.0954 ms / 4.4446 ms / 4.6486 ms | config/current +8.53%; prepared/current +13.51%; prepared/config +4.59% |
| MININDEX sweep, 4 × 4,096 | 58.169 µs / 57.410 µs / 50.835 µs | config/current -1.30%; prepared/current -12.61%; prepared/config -11.45% | 177.465 µs / 189.862 µs / 147.229 µs | config/current +6.99%; prepared/current -17.04%; prepared/config -22.45% |
| MININDEX per-worker, 4 × 4,096 | 55.749 µs / 47.566 µs | candidate/reference -14.68% | 171.045 µs / 150.498 µs | candidate/reference -12.01% |
| MININDEX streaming, 16 × 4,096 | 1.0649 ms / 1.2333 ms | candidate/reference +15.82% | 3.5267 ms / 3.9296 ms | candidate/reference +11.42% |
| MAXINDEX Universe, 128 × 4,096 | 1.5888 ms / 1.8323 ms / 1.7101 ms | config/current +15.32%; prepared/current +7.63%; prepared/config -6.67% | 4.8160 ms / 5.0314 ms / 4.2729 ms | config/current +4.47%; prepared/current -11.28%; prepared/config -15.08% |
| MAXINDEX sweep, 4 × 4,096 | 63.190 µs / 65.903 µs / 61.685 µs | config/current +4.29%; prepared/current -2.38%; prepared/config -6.40% | 144.822 µs / 151.814 µs / 129.199 µs | config/current +4.83%; prepared/current -10.79%; prepared/config -14.90% |
| MAXINDEX per-worker, 4 × 4,096 | 56.861 µs / 58.267 µs | candidate/reference +2.47% | 138.846 µs / 100.333 µs | candidate/reference -27.74% |
| MAXINDEX streaming, 16 × 4,096 | 1.1466 ms / 1.0550 ms | candidate/reference -7.99% | 1.2154 ms / 1.2430 ms | candidate/reference +2.27% |

### Gate conclusion

The exact allocation assertions clear the caller-owned, prepared-reuse, and streaming allocation gates. Each single-sided runner retains one index queue and never carries the opposite-extrema queue.

The two timing runs are recorded, but the host was not idle: a separate `huge-trading` workspace continuously compiled throughout both acceptance runs, and medians drifted substantially between Run A and Run B. The compared implementations also differ: current MIN/MAX free functions use a period-bounded queue, current index functions use their legacy single-index kernel, and configuration/prepared paths use append scratch. These results therefore do not isolate host contention from implementation differences. Prepared execution was usually faster but not uniformly so. The values qualify benchmark coverage, not portable speedups or a stable code-path regression.

Per the explicit issue #5 implementation decision, residual timing regressions and uncertainty are accepted in favor of the clean, idiomatic single-queue implementation. No fallback implementation or second execution convention is retained. Future uncontended qualification can rerun the two durable commands without changing benchmark IDs.

## Issue #6 WMA and TRIMA qualification

Issue #6 adds parameter-only `WMAConfig` and `TRIMAConfig` types, owned and caller-owned Compact Output, scratch-free Prepared Batch Runners, and independent Streaming Computations. The uppercase `WMA` and `TRIMA` compatibility objects now store only their corresponding stream while preserving all historical batch, padded-owned, streaming, reset, and `MA` dispatch behavior.

The durable commands are:

```text
cargo bench -p ta-benchmarks --bench execution_allocations
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/windowed_overlap --quick
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/windowed_overlap
cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/windowed_overlap
```

The valid timing matrix is observations 64/4,096/65,536 crossed with periods 14/512 where observations are at least the period. Both full commands used Criterion's default 100 samples and reported latency plus element throughput. Registration order rotates across current caller-owned, configuration caller-owned, prepared, legacy owned Aligned Output, and configuration owned Compact Output paths.

### Exact allocation profiles

These executable assertions use 4,096 observations, period 14, `Float = f64`, and a 64-bit host:

| Scenario | Operations | Gross bytes | Peak bytes | Retained bytes |
|---|---:|---:|---:|---:|
| Construct either configuration | 0 | 0 | 0 | 0 |
| Either caller-owned one-shot | 0 | 0 | 0 | 0 |
| Either owned Compact Output | 1 | 32,664 | 32,664 | 32,664 |
| Empty owned Compact Output | 0 | 0 | 0 | 0 |
| Prepare either runner | 0 | 0 | 0 | 0 |
| Prepared first/repeated call | 0 | 0 | 0 | 0 |
| Prepared oversize rejection | 0 | 0 | 0 | 0 |
| Construct either period-14 stream | 1 | 112 | 112 | 112 |
| Stream 4,096 ticks after construction | 0 | 0 | 0 | 0 |

Owned computation allocates only its exact 4,083-value Compact Output. Caller-owned and prepared kernels need no algorithm scratch. Stream construction retains exactly `period * size_of::<Float>()` bytes; ticks allocate nothing.

### Full timing evidence

The caller-owned table reports current free function / configuration / Prepared Batch Runner. Negative deltas mean the candidate took less time.

| Indicator | Observations / period | Run A medians | Run A same-run deltas | Run B medians | Run B same-run deltas |
|---|---:|---:|---:|---:|---:|
| WMA | 64 / 14 | 122.88 ns / 125.26 ns / 124.12 ns | config/current +1.94%; prepared/current +1.01%; prepared/config -0.91% | 128.23 ns / 124.82 ns / 130.23 ns | config/current -2.66%; prepared/current +1.56%; prepared/config +4.34% |
| WMA | 4,096 / 14 | 9.059 µs / 9.070 µs / 9.067 µs | config/current +0.11%; prepared/current +0.08%; prepared/config -0.03% | 9.925 µs / 9.043 µs / 9.418 µs | config/current -8.89%; prepared/current -5.10%; prepared/config +4.16% |
| WMA | 4,096 / 512 | 8.983 µs / 8.985 µs / 8.983 µs | config/current +0.02%; prepared/current +0.00%; prepared/config -0.02% | 8.981 µs / 8.980 µs / 8.976 µs | config/current -0.00%; prepared/current -0.05%; prepared/config -0.05% |
| WMA | 65,536 / 14 | 145.858 µs / 145.491 µs / 145.270 µs | config/current -0.25%; prepared/current -0.40%; prepared/config -0.15% | 145.563 µs / 144.855 µs / 145.827 µs | config/current -0.49%; prepared/current +0.18%; prepared/config +0.67% |
| WMA | 65,536 / 512 | 145.731 µs / 146.034 µs / 145.477 µs | config/current +0.21%; prepared/current -0.17%; prepared/config -0.38% | 145.122 µs / 147.149 µs / 149.910 µs | config/current +1.40%; prepared/current +3.30%; prepared/config +1.88% |
| TRIMA | 64 / 14 | 318.62 ns / 315.62 ns / 315.12 ns | config/current -0.94%; prepared/current -1.10%; prepared/config -0.16% | 316.72 ns / 315.04 ns / 315.55 ns | config/current -0.53%; prepared/current -0.37%; prepared/config +0.16% |
| TRIMA | 4,096 / 14 | 24.762 µs / 24.625 µs / 24.494 µs | config/current -0.55%; prepared/current -1.08%; prepared/config -0.53% | 24.355 µs / 25.054 µs / 24.396 µs | config/current +2.87%; prepared/current +0.17%; prepared/config -2.63% |
| TRIMA | 4,096 / 512 | 1.4816 ms / 1.5021 ms / 1.4907 ms | config/current +1.38%; prepared/current +0.61%; prepared/config -0.76% | 1.4735 ms / 1.4810 ms / 1.4793 ms | config/current +0.51%; prepared/current +0.39%; prepared/config -0.12% |
| TRIMA | 65,536 / 14 | 397.095 µs / 404.360 µs / 395.445 µs | config/current +1.83%; prepared/current -0.42%; prepared/config -2.20% | 391.387 µs / 391.586 µs / 392.310 µs | config/current +0.05%; prepared/current +0.24%; prepared/config +0.18% |
| TRIMA | 65,536 / 512 | 27.0043 ms / 27.0609 ms / 27.8810 ms | config/current +0.21%; prepared/current +3.25%; prepared/config +3.03% | 26.7989 ms / 26.8458 ms / 26.8144 ms | config/current +0.18%; prepared/current +0.06%; prepared/config -0.12% |

Owned qualification reports legacy owned Aligned Output / configuration owned Compact Output:

| Indicator | Observations / period | Run A medians | Run A delta | Run B medians | Run B delta |
|---|---:|---:|---:|---:|---:|
| WMA | 64 / 14 | 177.30 ns / 140.57 ns | -20.71% | 205.13 ns / 158.51 ns | -22.72% |
| WMA | 4,096 / 14 | 10.579 µs / 9.607 µs | -9.19% | 10.591 µs / 10.441 µs | -1.42% |
| WMA | 4,096 / 512 | 10.430 µs / 9.435 µs | -9.54% | 10.500 µs / 9.432 µs | -10.17% |
| WMA | 65,536 / 14 | 190.017 µs / 150.194 µs | -20.96% | 195.415 µs / 149.907 µs | -23.29% |
| WMA | 65,536 / 512 | 187.959 µs / 149.704 µs | -20.35% | 192.047 µs / 149.272 µs | -22.27% |
| TRIMA | 64 / 14 | 379.87 ns / 339.50 ns | -10.63% | 391.92 ns / 342.89 ns | -12.51% |
| TRIMA | 4,096 / 14 | 25.969 µs / 25.675 µs | -1.13% | 25.709 µs / 25.039 µs | -2.61% |
| TRIMA | 4,096 / 512 | 1.5043 ms / 1.5402 ms | +2.39% | 1.4770 ms / 1.4758 ms | -0.08% |
| TRIMA | 65,536 / 14 | 446.528 µs / 404.752 µs | -9.36% | 435.051 µs / 399.810 µs | -8.10% |
| TRIMA | 65,536 / 512 | 28.3352 ms / 26.8167 ms | -5.36% | 26.8794 ms / 26.7561 ms | -0.46% |

### Repeated and streaming evidence

Universe and parameter-sweep rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations.

| Indicator / workload | Run A medians | Run A same-run deltas | Run B medians | Run B same-run deltas |
|---|---:|---:|---:|---:|
| WMA Universe, 128 × 4,096 | 1.2243 ms / 1.3179 ms / 1.2114 ms | config/current +7.65%; prepared/current -1.05%; prepared/config -8.08% | 1.1622 ms / 1.1608 ms / 1.1839 ms | config/current -0.11%; prepared/current +1.87%; prepared/config +1.99% |
| WMA sweep, 4 × 4,096 | 36.385 µs / 36.865 µs / 37.034 µs | config/current +1.32%; prepared/current +1.78%; prepared/config +0.46% | 36.406 µs / 36.444 µs / 36.407 µs | config/current +0.11%; prepared/current +0.00%; prepared/config -0.10% |
| WMA per-worker, 4 × 4,096 | 37.099 µs / 37.797 µs | candidate/reference +1.88% | 36.556 µs / 36.613 µs | candidate/reference +0.16% |
| WMA streaming, 16 × 4,096 | 781.974 µs / 797.623 µs | candidate/reference +2.00% | 777.127 µs / 776.838 µs | candidate/reference -0.04% |
| TRIMA Universe, 128 × 4,096 | 3.2573 ms / 3.2642 ms / 3.5488 ms | config/current +0.21%; prepared/current +8.95%; prepared/config +8.72% | 3.1479 ms / 3.1456 ms / 3.1476 ms | config/current -0.07%; prepared/current -0.01%; prepared/config +0.07% |
| TRIMA sweep, 4 × 4,096 | 695.060 µs / 891.873 µs / 744.866 µs | config/current +28.32%; prepared/current +7.17%; prepared/config -16.48% | 659.811 µs / 656.885 µs / 658.220 µs | config/current -0.44%; prepared/current -0.24%; prepared/config +0.20% |
| TRIMA per-worker, 4 × 4,096 | 121.893 µs / 116.952 µs | candidate/reference -4.05% | 99.895 µs / 97.445 µs | candidate/reference -2.45% |
| TRIMA streaming, 16 × 4,096 | 1.0076 ms / 981.712 µs | candidate/reference -2.57% | 818.366 µs / 816.824 µs | candidate/reference -0.19% |

### Gate conclusion

Allocation gates clear exactly: both caller-owned and prepared paths are allocation-free, owned output allocates one exact compact payload, and streaming allocates only at construction. No timing regression above approximately five percent reproduced across both full runs. The non-reproduced Run A regressions were WMA Universe configuration/current (+7.65%), TRIMA Universe prepared/current (+8.95%) and prepared/configuration (+8.72%), and TRIMA sweep configuration/current (+28.32%) and prepared/current (+7.17%); their Run B counterparts ranged from -0.44% to +1.99%. They are not stable regressions. These are host-local `f64` results, not portable speedup claims.

## Issue #7 recursive overlap qualification

Issue #7 adds parameter-only `EMAConfig`, `DEMAConfig`, `TEMAConfig`, `T3Config`, and `MAConfig` types with owned and caller-owned Compact Output, reusable Prepared Batch Runners, and independent Streaming Computations. Uppercase compatibility indicators retain the existing legacy batch, padded-owned, streaming, reset, and `MAType` dispatch behavior. The qualified `MA` benchmark uses `MAType::EMA`; the other supported dispatch kinds are covered by correctness tests.

The benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/recursive_overlap`. The valid matrix crosses observations 64/4,096/65,536 with periods 14/512. T3 omits 64/14 because its 78-observation lookback exceeds that input. Values below are Criterion median point estimates from the full 100-sample qualification run.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports:

| Indicator | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| EMA | 0 / 0 B | 0 / 0 B | 1 / 32,664 B | 0 / 0 B in every phase | 0 / 0 B |
| DEMA | 0 / 0 B | 0 / 0 B | 1 / 32,560 B | 0 / 0 B in every phase | 0 / 0 B |
| TEMA | 0 / 0 B | 0 / 0 B | 1 / 32,456 B | 0 / 0 B in every phase | 0 / 0 B |
| T3 | 0 / 0 B | 0 / 0 B | 1 / 32,144 B | 0 / 0 B in every phase | 0 / 0 B |
| MA (EMA) | 0 / 0 B | 0 / 0 B | 1 / 32,664 B | 0 / 0 B in every phase | 0 / 0 B |

Owned byte counts are exactly `compact_count × size_of::<Float>()`. Caller-owned computation, prepared reuse and rejection, and streaming ticks allocate nothing. Recursive streams retain fixed scalar state, so construction also allocates nothing.

### Full timing evidence

Caller-owned fields report current / configuration / prepared. Owned fields report legacy Aligned Output / configuration Compact Output.

| Indicator | Observations / period | Caller-owned medians | Config / current | Prepared / current | Owned medians | Compact / legacy |
|---|---:|---:|---:|---:|---:|---:|
| EMA | 64 / 14 | 173.64 ns / 174.03 ns / 174.37 ns | +0.22% | +0.42% | 225.68 ns / 189.72 ns | -15.93% |
| EMA | 4096 / 14 | 13.721 µs / 13.711 µs / 14.001 µs | -0.07% | +2.05% | 15.200 µs / 14.207 µs | -6.54% |
| EMA | 4096 / 512 | 12.683 µs / 12.675 µs / 12.959 µs | -0.06% | +2.18% | 14.113 µs / 13.103 µs | -7.16% |
| EMA | 65536 / 14 | 220.281 µs / 220.199 µs / 224.543 µs | -0.04% | +1.93% | 264.577 µs / 225.242 µs | -14.87% |
| EMA | 65536 / 512 | 219.247 µs / 219.149 µs / 223.488 µs | -0.04% | +1.93% | 261.146 µs / 224.411 µs | -14.07% |
| DEMA | 64 / 14 | 185.77 ns / 185.75 ns / 185.60 ns | -0.01% | -0.09% | 235.56 ns / 201.41 ns | -14.49% |
| DEMA | 4096 / 14 | 14.273 µs / 14.280 µs / 14.285 µs | +0.05% | +0.09% | 15.727 µs / 14.994 µs | -4.67% |
| DEMA | 4096 / 512 | 13.188 µs / 13.192 µs / 13.203 µs | +0.03% | +0.12% | 14.541 µs / 13.774 µs | -5.27% |
| DEMA | 65536 / 14 | 229.245 µs / 229.211 µs / 231.862 µs | -0.01% | +1.14% | 270.407 µs / 234.286 µs | -13.36% |
| DEMA | 65536 / 512 | 228.105 µs / 228.247 µs / 230.655 µs | +0.06% | +1.12% | 268.503 µs / 233.192 µs | -13.15% |
| TEMA | 64 / 14 | 356.02 ns / 356.03 ns / 356.31 ns | +0.00% | +0.08% | 409.28 ns / 370.52 ns | -9.47% |
| TEMA | 4096 / 14 | 24.700 µs / 24.673 µs / 24.698 µs | -0.11% | -0.01% | 26.033 µs / 25.177 µs | -3.29% |
| TEMA | 4096 / 512 | 23.659 µs / 23.667 µs / 23.671 µs | +0.03% | +0.05% | 25.006 µs / 23.952 µs | -4.22% |
| TEMA | 65536 / 14 | 396.109 µs / 396.340 µs / 397.160 µs | +0.06% | +0.27% | 434.813 µs / 399.283 µs | -8.17% |
| TEMA | 65536 / 512 | 395.154 µs / 394.610 µs / 395.604 µs | -0.14% | +0.11% | 435.056 µs / 398.468 µs | -8.41% |
| T3 | 4096 / 14 | 26.654 µs / 26.608 µs / 26.598 µs | -0.17% | -0.21% | 27.890 µs / 26.867 µs | -3.67% |
| T3 | 4096 / 512 | 24.287 µs / 24.257 µs / 24.256 µs | -0.12% | -0.13% | 25.382 µs / 24.448 µs | -3.68% |
| T3 | 65536 / 14 | 429.153 µs / 428.730 µs / 426.604 µs | -0.10% | -0.59% | 464.896 µs / 426.233 µs | -8.32% |
| T3 | 65536 / 512 | 426.744 µs / 426.377 µs / 424.393 µs | -0.09% | -0.55% | 462.170 µs / 424.395 µs | -8.17% |
| MA (EMA) | 64 / 14 | 174.73 ns / 174.69 ns / 174.77 ns | -0.02% | +0.02% | 226.00 ns / 191.99 ns | -15.05% |
| MA (EMA) | 4096 / 14 | 13.754 µs / 13.754 µs / 13.872 µs | +0.01% | +0.86% | 15.420 µs / 14.212 µs | -7.83% |
| MA (EMA) | 4096 / 512 | 12.743 µs / 12.743 µs / 12.864 µs | -0.00% | +0.95% | 14.343 µs / 13.129 µs | -8.46% |
| MA (EMA) | 65536 / 14 | 221.097 µs / 220.860 µs / 223.448 µs | -0.11% | +1.06% | 261.834 µs / 224.692 µs | -14.19% |
| MA (EMA) | 65536 / 512 | 220.343 µs / 220.405 µs / 222.397 µs | +0.03% | +0.93% | 263.795 µs / 223.472 µs | -15.29% |

### Repeated and streaming evidence

Universe and parameter-sweep rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations.

| Indicator / workload | Run A medians | Same-run deltas |
|---|---:|---:|
| EMA Universe, 128 × 4,096 | 1.7578 ms / 1.7580 ms / 1.7571 ms | config/current +0.01%; prepared/current -0.04%; prepared/config -0.05% |
| EMA Sweep, 4 × 4,096 | 54.467 µs / 54.454 µs / 54.431 µs | config/current -0.02%; prepared/current -0.07%; prepared/config -0.04% |
| EMA Per-worker, 4 × 4,096 | 56.054 µs / 54.830 µs | candidate/reference -2.18% |
| EMA Streaming, 16 × 4,096 | 116.064 µs / 116.111 µs | candidate/reference +0.04% |
| DEMA Universe, 128 × 4,096 | 1.8374 ms / 1.8376 ms / 1.8299 ms | config/current +0.01%; prepared/current -0.41%; prepared/config -0.42% |
| DEMA Sweep, 4 × 4,096 | 56.622 µs / 56.629 µs / 56.662 µs | config/current +0.01%; prepared/current +0.07%; prepared/config +0.06% |
| DEMA Per-worker, 4 × 4,096 | 57.441 µs / 57.315 µs | candidate/reference -0.22% |
| DEMA Streaming, 16 × 4,096 | 230.173 µs / 229.674 µs | candidate/reference -0.22% |
| TEMA Universe, 128 × 4,096 | 3.1639 ms / 3.1651 ms / 3.1792 ms | config/current +0.04%; prepared/current +0.48%; prepared/config +0.44% |
| TEMA Sweep, 4 × 4,096 | 99.313 µs / 99.365 µs / 98.513 µs | config/current +0.05%; prepared/current -0.81%; prepared/config -0.86% |
| TEMA Per-worker, 4 × 4,096 | 99.219 µs / 99.278 µs | candidate/reference +0.06% |
| TEMA Streaming, 16 × 4,096 | 280.791 µs / 279.258 µs | candidate/reference -0.55% |
| T3 Universe, 128 × 4,096 | 3.4103 ms / 3.4104 ms / 3.4426 ms | config/current +0.00%; prepared/current +0.95%; prepared/config +0.94% |
| T3 Sweep, 4 × 4,096 | 105.542 µs / 105.534 µs / 105.728 µs | config/current -0.01%; prepared/current +0.18%; prepared/config +0.18% |
| T3 Per-worker, 4 × 4,096 | 106.633 µs / 107.211 µs | candidate/reference +0.54% |
| T3 Streaming, 16 × 4,096 | 578.938 µs / 579.006 µs | candidate/reference +0.01% |
| MA (EMA) Universe, 128 × 4,096 | 1.7642 ms / 1.7639 ms / 1.7572 ms | config/current -0.02%; prepared/current -0.40%; prepared/config -0.38% |
| MA (EMA) Sweep, 4 × 4,096 | 54.647 µs / 54.660 µs / 54.457 µs | config/current +0.02%; prepared/current -0.35%; prepared/config -0.37% |
| MA (EMA) Per-worker, 4 × 4,096 | 55.237 µs / 54.937 µs | candidate/reference -0.54% |
| MA (EMA) Streaming, 16 × 4,096 | 144.027 µs / 136.320 µs | candidate/reference -5.35% |

### Gate conclusion

All allocation gates clear exactly. In the full qualification run, no configuration caller-owned, prepared, repeated-workload, or streaming path regressed by approximately five percent against its same-run reference. Every owned Compact Output path improved over the legacy Aligned Output path. These are host-local default-`f64` qualification results, not portable speedup claims.

## Issue #8 price-transform qualification

Issue #8 adds parameter-only configurations for `AVGDEV`, `AVGPRICE`, `MEDPRICE`, `TYPPRICE`, and `WCLPRICE`, with owned and caller-owned Compact Output, reusable Prepared Batch Runners, and independent Streaming Computations. Uppercase compatibility indicators retain their existing batch, padded-owned, and streaming behavior; stateful `AVGDEV` also retains reset behavior. Named multi-series inputs use typed structures rather than positional tuples.

The full benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/price_transform`. The valid `AVGDEV` matrix crosses observations 64/4,096/65,536 with periods 14/512; the four zero-lookback price transforms use observations 64/4,096/65,536. A focused rerun of `indicator_execution/expanded/price_transform_workloads/.*/streaming` records the final tick-interleaved multi-instrument workload. Values below are Criterion median point estimates from the 100-sample qualification runs.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports:

| Indicator | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| AVGDEV | 0 / 0 B | 0 / 0 B | 1 / 32,664 B | 0 / 0 B in every phase | 1 / 112 B; 0 / 0 B |
| AVGPRICE | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |
| MEDPRICE | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |
| TYPPRICE | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |
| WCLPRICE | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |

Entries are allocation operations / gross allocated bytes. Owned byte counts are exactly `compact_count × size_of::<Float>()`; caller-owned computation, prepared reuse and rejection, and streaming ticks allocate nothing. Only the rolling `AVGDEV` stream allocates its fixed-period ring buffer during construction.

### Full timing evidence

Caller-owned fields report current / configuration / prepared. Owned fields report legacy Aligned Output / configuration Compact Output.

| Indicator | Observations / period | Caller-owned medians | Config / current | Prepared / current | Owned medians | Compact / legacy |
|---|---:|---:|---:|---:|---:|---:|
| AVGDEV | 64 / 14 | 392.59 ns / 392.49 ns / 392.17 ns | -0.02% | -0.11% | 449.75 ns / 417.94 ns | -7.07% |
| AVGDEV | 4,096 / 14 | 30.492 µs / 30.456 µs / 30.671 µs | -0.12% | +0.58% | 31.920 µs / 31.195 µs | -2.27% |
| AVGDEV | 4,096 / 512 | 2.8520 ms / 2.8543 ms / 2.8507 ms | +0.08% | -0.04% | 2.8559 ms / 2.8564 ms | +0.02% |
| AVGDEV | 65,536 / 14 | 491.210 µs / 491.119 µs / 495.015 µs | -0.02% | +0.77% | 536.180 µs / 496.486 µs | -7.40% |
| AVGDEV | 65,536 / 512 | 51.8864 ms / 51.8787 ms / 51.8470 ms | -0.01% | -0.08% | 51.9613 ms / 51.8978 ms | -0.12% |
| AVGPRICE | 64 | 124.68 ns / 116.28 ns / 117.54 ns | -6.74% | -5.73% | 182.01 ns / 140.42 ns | -22.85% |
| AVGPRICE | 4,096 | 7.504 µs / 7.473 µs / 7.604 µs | -0.42% | +1.34% | 10.592 µs / 8.106 µs | -23.48% |
| AVGPRICE | 65,536 | 129.554 µs / 129.451 µs / 130.723 µs | -0.08% | +0.90% | 166.859 µs / 135.027 µs | -19.08% |
| MEDPRICE | 64 | 66.68 ns / 63.15 ns / 63.05 ns | -5.29% | -5.44% | 123.35 ns / 73.90 ns | -40.09% |
| MEDPRICE | 4,096 | 3.725 µs / 3.880 µs / 3.722 µs | +4.17% | -0.09% | 5.646 µs / 4.407 µs | -21.95% |
| MEDPRICE | 65,536 | 67.360 µs / 68.538 µs / 67.661 µs | +1.75% | +0.45% | 92.676 µs / 72.378 µs | -21.90% |
| TYPPRICE | 64 | 82.65 ns / 83.75 ns / 82.90 ns | +1.33% | +0.31% | 146.54 ns / 103.44 ns | -29.41% |
| TYPPRICE | 4,096 | 5.748 µs / 5.808 µs / 5.648 µs | +1.03% | -1.74% | 7.937 µs / 6.344 µs | -20.07% |
| TYPPRICE | 65,536 | 100.391 µs / 99.705 µs / 98.485 µs | -0.68% | -1.90% | 129.457 µs / 103.850 µs | -19.78% |
| WCLPRICE | 64 | 82.35 ns / 83.11 ns / 83.13 ns | +0.92% | +0.95% | 145.64 ns / 107.10 ns | -26.46% |
| WCLPRICE | 4,096 | 5.695 µs / 5.552 µs / 5.556 µs | -2.50% | -2.45% | 7.764 µs / 5.984 µs | -22.93% |
| WCLPRICE | 65,536 | 100.887 µs / 98.512 µs / 98.521 µs | -2.35% | -2.35% | 129.582 µs / 103.727 µs | -19.95% |

### Repeated and streaming evidence

Universe and parameter-sweep rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations. Parameter sweep applies only to parameterized `AVGDEV`.

| Indicator / workload | Medians | Same-run deltas |
|---|---:|---:|
| AVGDEV Universe, 128 × 4,096 | 3.9150 ms / 3.9126 ms / 3.9118 ms | config/current -0.06%; prepared/current -0.08% |
| AVGDEV Sweep, 4 × 4,096 | 1.1518 ms / 1.1513 ms / 1.1522 ms | config/current -0.05%; prepared/current +0.03% |
| AVGDEV Per-worker, 4 × 4,096 | 122.393 µs / 122.089 µs | candidate/reference -0.25% |
| AVGDEV Streaming, 16 × 4,096 | 659.997 µs / 660.608 µs | candidate/reference +0.09% |
| AVGPRICE Universe, 128 × 4,096 | 984.175 µs / 987.446 µs / 979.345 µs | config/current +0.33%; prepared/current -0.49% |
| AVGPRICE Per-worker, 4 × 4,096 | 30.223 µs / 30.667 µs | candidate/reference +1.47% |
| AVGPRICE Streaming, 16 × 4,096 | 238.092 µs / 242.484 µs | candidate/reference +1.84% |
| MEDPRICE Universe, 128 × 4,096 | 486.662 µs / 502.929 µs / 486.485 µs | config/current +3.34%; prepared/current -0.04% |
| MEDPRICE Per-worker, 4 × 4,096 | 15.402 µs / 15.096 µs | candidate/reference -1.99% |
| MEDPRICE Streaming, 16 × 4,096 | 236.424 µs / 236.434 µs | candidate/reference +0.00% |
| TYPPRICE Universe, 128 × 4,096 | 763.223 µs / 750.199 µs / 751.567 µs | config/current -1.71%; prepared/current -1.53% |
| TYPPRICE Per-worker, 4 × 4,096 | 22.993 µs / 22.645 µs | candidate/reference -1.52% |
| TYPPRICE Streaming, 16 × 4,096 | 260.233 µs / 260.162 µs | candidate/reference -0.03% |
| WCLPRICE Universe, 128 × 4,096 | 757.675 µs / 777.185 µs / 740.212 µs | config/current +2.57%; prepared/current -2.30% |
| WCLPRICE Per-worker, 4 × 4,096 | 22.584 µs / 22.437 µs | candidate/reference -0.65% |
| WCLPRICE Streaming, 16 × 4,096 | 255.631 µs / 256.196 µs | candidate/reference +0.22% |

### Gate conclusion

All allocation gates clear exactly. No configuration caller-owned, prepared, repeated-workload, or streaming path regressed by approximately five percent against its same-run reference. Every owned Compact Output path improved over the legacy Aligned Output path except `AVGDEV` at the two period-512 cases, where it remained within +0.02%. These are host-local default-`f64` qualification results, not portable speedup claims.

## Issue #9 volume qualification

Issue #9 adds parameter-only configurations for `AD`, `ADOSC`, and `OBV`, with owned and caller-owned Compact Output, reusable Prepared Batch Runners, and independent Streaming Computations. Uppercase compatibility indicators retain their existing batch, padded-owned, streaming, warm-up, and reset behavior. Multi-series high/low/close/volume and close/volume inputs use typed structures.

The full benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/volume`. The matrix uses observations 64/4,096/65,536; `ADOSC` uses fast/slow periods 7/14 and its parameter sweep uses slow periods 5/14/50/200 with `fastperiod = max(1, slowperiod / 2)`. A focused rerun of `indicator_execution/expanded/volume_workloads/ADOSC/streaming` records the final direct configured stream and compatibility adapter. Values below are Criterion point estimates from the 100-sample qualification runs.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports:

| Indicator | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| AD | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |
| ADOSC | 0 / 0 B | 0 / 0 B | 1 / 32,664 B | 0 / 0 B in every phase | 0 / 0 B |
| OBV | 0 / 0 B | 0 / 0 B | 1 / 32,760 B | 0 / 0 B in every phase | 0 / 0 B |

Entries are allocation operations / gross allocated bytes. Owned byte counts are exactly `compact_count × size_of::<Float>()`; caller-owned computation, prepared reuse and rejection, stream construction, and streaming ticks allocate nothing.

### Full timing evidence

Caller-owned fields report current / configuration / prepared. Owned fields report legacy Aligned Output / configuration Compact Output.

| Indicator | Observations | Caller-owned point estimates | Config / current | Prepared / current | Owned point estimates | Compact / legacy |
|---|---:|---:|---:|---:|---:|---:|
| AD | 64 | 162.97 ns / 162.10 ns / 163.25 ns | -0.53% | +0.17% | 223.36 ns / 183.53 ns | -17.83% |
| AD | 4,096 | 11.098 µs / 11.115 µs / 11.113 µs | +0.15% | +0.14% | 14.150 µs / 11.764 µs | -16.87% |
| AD | 65,536 | 175.496 µs / 176.281 µs / 175.826 µs | +0.45% | +0.19% | 217.396 µs / 184.982 µs | -14.91% |
| ADOSC | 64 | 280.80 ns / 280.41 ns / 281.16 ns | -0.14% | +0.13% | 347.45 ns / 306.84 ns | -11.69% |
| ADOSC | 4,096 | 19.035 µs / 19.041 µs / 19.197 µs | +0.03% | +0.85% | 22.135 µs / 19.787 µs | -10.61% |
| ADOSC | 65,536 | 306.848 µs / 304.461 µs / 305.825 µs | -0.78% | -0.33% | 348.035 µs / 308.570 µs | -11.34% |
| OBV | 64 | 97.59 ns / 97.18 ns / 100.28 ns | -0.42% | +2.76% | 161.37 ns / 119.34 ns | -26.04% |
| OBV | 4,096 | 6.932 µs / 7.032 µs / 6.935 µs | +1.45% | +0.04% | 9.179 µs / 7.515 µs | -18.14% |
| OBV | 65,536 | 111.922 µs / 111.028 µs / 112.316 µs | -0.80% | +0.35% | 138.932 µs / 116.496 µs | -16.15% |

### Repeated and streaming evidence

Universe rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations. The ADOSC sweep row aggregates the four separately measured period point estimates.

| Indicator / workload | Point estimates | Same-run deltas |
|---|---:|---:|
| AD Universe, 128 × 4,096 | 1.4564 ms / 1.4571 ms / 1.4489 ms | config/current +0.05%; prepared/current -0.52% |
| AD Per-worker, 4 × 4,096 | 46.447 µs / 45.005 µs | prepared/current -3.11% |
| AD Streaming, 16 × 4,096 | 470.770 µs / 326.096 µs | configured/legacy -30.73% |
| ADOSC Universe, 128 × 4,096 | 2.4658 ms / 2.4536 ms / 2.4743 ms | config/current -0.49%; prepared/current +0.35% |
| ADOSC Sweep, 4 × 4,096 | 76.160 µs / 76.104 µs / 76.645 µs | config/current -0.07%; prepared/current +0.64% |
| ADOSC Per-worker, 4 × 4,096 | 77.436 µs / 76.389 µs | prepared/current -1.35% |
| ADOSC Streaming, 16 × 4,096 | 422.320 µs / 422.325 µs | configured/legacy +0.00% |
| OBV Universe, 128 × 4,096 | 890.584 µs / 892.538 µs / 893.511 µs | config/current +0.22%; prepared/current +0.33% |
| OBV Per-worker, 4 × 4,096 | 28.087 µs / 27.659 µs | prepared/current -1.52% |
| OBV Streaming, 16 × 4,096 | 245.730 µs / 244.982 µs | configured/legacy -0.30% |

### Gate conclusion

All allocation gates clear exactly. No configuration caller-owned, prepared, Universe, parameter-sweep, per-worker, or streaming path regressed by approximately five percent against its same-run reference. Every owned Compact Output path improved over the legacy Aligned Output path by 10.61%–26.04%. These are host-local default-`f64` qualification results, not portable speedup claims.

## Issue #10 volatility qualification

Issue #10 adds parameter-only configurations for `TRANGE`, `ATR`, and `NATR`, with owned and caller-owned Compact Output, reusable Prepared Batch Runners, and independent Streaming Computations. Uppercase compatibility indicators retain their existing batch, padded-owned, streaming, warm-up, and reset behavior. Multi-series high/low/close inputs use typed structures.

The full benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/volatility`. The matrix uses observations 64/4,096/65,536; `ATR` and `NATR` use period 14, and their parameter sweeps use periods 5/14/50/200. Values below are Criterion mean point estimates from the 100-sample qualification run.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports:

| Indicator | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| TRANGE | 0 / 0 B | 0 / 0 B | 1 / 32,760 B | 0 / 0 B in every phase | 0 / 0 B |
| ATR | 0 / 0 B | 0 / 0 B | 1 / 32,656 B | 0 / 0 B in every phase | 0 / 0 B |
| NATR | 0 / 0 B | 0 / 0 B | 1 / 32,656 B | 0 / 0 B in every phase | 0 / 0 B |

Entries are allocation operations / gross allocated bytes. Owned byte counts are exactly `compact_count × size_of::<Float>()`; caller-owned computation, prepared reuse and rejection, stream construction, and streaming ticks allocate nothing.

### Full timing evidence

Caller-owned fields report current / configuration / prepared. Owned fields report legacy Aligned Output / configuration Compact Output.

| Indicator | Observations | Caller-owned point estimates | Config / current | Prepared / current | Owned point estimates | Compact / legacy |
|---|---:|---:|---:|---:|---:|---:|
| TRANGE | 64 | 86.50 ns / 86.13 ns / 88.56 ns | -0.43% | +2.38% | 147.19 ns / 107.10 ns | -27.24% |
| TRANGE | 4,096 | 5.796 µs / 5.684 µs / 5.818 µs | -1.94% | +0.38% | 7.820 µs / 6.106 µs | -21.92% |
| TRANGE | 65,536 | 100.673 µs / 99.002 µs / 100.990 µs | -1.66% | +0.31% | 140.501 µs / 103.396 µs | -26.41% |
| ATR | 64 | 329.13 ns / 326.32 ns / 327.45 ns | -0.85% | -0.51% | 380.73 ns / 356.97 ns | -6.24% |
| ATR | 4,096 | 25.666 µs / 25.657 µs / 25.620 µs | -0.03% | -0.18% | 28.485 µs / 26.288 µs | -7.71% |
| ATR | 65,536 | 411.820 µs / 412.058 µs / 411.098 µs | +0.06% | -0.18% | 454.511 µs / 415.909 µs | -8.49% |
| NATR | 64 | 333.56 ns / 334.39 ns / 333.31 ns | +0.25% | -0.07% | 385.99 ns / 344.72 ns | -10.69% |
| NATR | 4,096 | 25.843 µs / 25.846 µs / 25.904 µs | +0.01% | +0.24% | 28.320 µs / 26.320 µs | -7.06% |
| NATR | 65,536 | 418.601 µs / 416.822 µs / 415.193 µs | -0.43% | -0.81% | 458.409 µs / 420.565 µs | -8.26% |

### Repeated and streaming evidence

Universe and parameter-sweep rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations. Sweep rows aggregate the four separately measured period point estimates.

| Indicator / workload | Point estimates | Same-run deltas |
|---|---:|---:|
| TRANGE Universe, 128 × 4,096 | 747.211 µs / 774.557 µs / 748.167 µs | config/current +3.66%; prepared/current +0.13% |
| TRANGE Per-worker, 4 × 4,096 | 22.923 µs / 23.728 µs | prepared/current +3.51% |
| TRANGE Streaming, 16 × 4,096 | 287.426 µs / 289.187 µs | configured/legacy +0.61% |
| ATR Universe, 128 × 4,096 | 3.2799 ms / 3.2795 ms / 3.3236 ms | config/current -0.01%; prepared/current +1.33% |
| ATR Sweep, 4 × 4,096 | 101.739 µs / 101.799 µs / 102.955 µs | config/current +0.06%; prepared/current +1.19% |
| ATR Per-worker, 4 × 4,096 | 103.450 µs / 102.429 µs | prepared/current -0.99% |
| ATR Streaming, 16 × 4,096 | 330.982 µs / 331.422 µs | configured/legacy +0.13% |
| NATR Universe, 128 × 4,096 | 3.3201 ms / 3.3159 ms / 3.3453 ms | config/current -0.13%; prepared/current +0.76% |
| NATR Sweep, 4 × 4,096 | 103.221 µs / 103.427 µs / 102.674 µs | config/current +0.20%; prepared/current -0.53% |
| NATR Per-worker, 4 × 4,096 | 103.534 µs / 103.497 µs | prepared/current -0.04% |
| NATR Streaming, 16 × 4,096 | 367.205 µs / 365.349 µs | configured/legacy -0.51% |

### Gate conclusion

All allocation gates clear exactly. No configuration caller-owned, prepared, Universe, parameter-sweep, per-worker, or streaming path regressed by approximately five percent against its same-run reference. Every owned Compact Output path improved over the legacy Aligned Output path by 6.24%–27.24%. These are host-local default-`f64` qualification results, not portable speedup claims.

## Issue #11 math-transform qualification

Issue #11 migrates all 15 macro-generated unary math transforms (`ACOS`, `ASIN`, `ATAN`, `CEIL`, `COS`, `COSH`, `EXP`, `FLOOR`, `LN`, `LOG10`, `SIN`, `SINH`, `SQRT`, `TAN`, and `TANH`) through one generated Rust-first execution surface. Each parameter-free configuration provides owned and caller-owned Compact Output, a reusable Prepared Batch Runner, and an independent Streaming Computation while the uppercase compatibility functions and structs retain their existing signatures and padded-owned behavior. Dense input series must remain finite. Finite values outside an operation's conventional real domain are still passed to the underlying IEEE-754 operation so operation-produced `NaN` and infinity results remain valid values.

The full benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/math_transform`. The matrix uses observations 64/4,096/65,536 and also records Universe (128 × 4,096), per-worker (4 × 4,096), and streaming (16 × 4,096) workloads. Prepared paths were refreshed with `cargo bench -p ta-benchmarks --bench execution_baselines -- 'math_transform.*/prepared_runner'` after the final zero-overhead runner dispatch was selected. Values below are Criterion mean point estimates from the 100-sample qualification runs.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports the same result for every transform:

| Indicators | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| All 15 unary transforms | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |

Entries are allocation operations / gross allocated bytes. The owned byte count is exactly `4,096 × size_of::<Float>()`; caller-owned computation, prepared construction, prepared reuse and rejection, stream construction, and streaming ticks allocate nothing.

### One-shot timing evidence

Caller-owned point estimates show current / configuration / prepared for 4,096 observations. Delta ranges include all three measured observation counts. Owned point estimates show legacy Aligned Output / configuration Compact Output for 4,096 observations.

| Indicator | 4,096 caller-owned point estimates | Config / current range | Prepared / current range | 4,096 owned point estimates | Compact / legacy range |
|---|---:|---:|---:|---:|---:|
| ACOS | 15.449 µs / 15.482 µs / 15.448 µs | -0.18% to +0.44% | -0.07% to +0.06% | 17.193 µs / 16.119 µs | -16.51% to -6.25% |
| ASIN | 17.435 µs / 17.452 µs / 17.443 µs | -0.19% to +0.14% | +0.01% to +0.04% | 18.892 µs / 17.908 µs | -14.01% to -5.21% |
| ATAN | 15.734 µs / 15.797 µs / 15.835 µs | +0.35% to +1.95% | +0.64% to +1.16% | 17.167 µs / 16.180 µs | -17.04% to -5.75% |
| CEIL | 2.269 µs / 2.136 µs / 2.110 µs | -6.21% to -3.54% | -7.04% to -0.01% | 3.601 µs / 2.358 µs | -49.86% to -34.52% |
| COS | 15.696 µs / 15.715 µs / 15.689 µs | -3.67% to +0.15% | -3.52% to -0.04% | 17.266 µs / 16.177 µs | -20.33% to -6.31% |
| COSH | 13.225 µs / 13.215 µs / 13.229 µs | -0.17% to +0.06% | -0.29% to +0.03% | 14.743 µs / 13.717 µs | -18.75% to -6.96% |
| EXP | 10.441 µs / 10.594 µs / 10.511 µs | +0.32% to +1.47% | +0.40% to +1.02% | 11.747 µs / 10.948 µs | -21.22% to -6.80% |
| FLOOR | 2.323 µs / 2.168 µs / 2.168 µs | -6.70% to -2.11% | -6.76% to -0.42% | 3.593 µs / 2.365 µs | -49.63% to -34.18% |
| LN | 12.144 µs / 11.990 µs / 12.270 µs | -1.46% to +0.86% | +0.60% to +2.86% | 13.385 µs / 12.573 µs | -21.46% to -6.07% |
| LOG10 | 13.225 µs / 13.503 µs / 13.227 µs | +1.39% to +2.10% | -0.10% to +0.44% | 14.732 µs / 13.706 µs | -19.12% to -6.97% |
| SIN | 15.583 µs / 15.564 µs / 15.556 µs | -0.73% to +0.10% | -0.63% to -0.17% | 17.005 µs / 16.098 µs | -16.88% to -5.34% |
| SINH | 13.313 µs / 13.300 µs / 13.307 µs | -0.62% to +0.28% | -0.56% to -0.05% | 14.868 µs / 14.574 µs | -12.93% to -1.97% |
| SQRT | 3.109 µs / 2.827 µs / 2.853 µs | -9.04% to -0.28% | -8.22% to +0.35% | 4.347 µs / 3.247 µs | -47.68% to -25.31% |
| TAN | 18.710 µs / 18.888 µs / 18.717 µs | +0.02% to +0.95% | -0.08% to +0.05% | 20.190 µs / 19.209 µs | -13.78% to -4.86% |
| TANH | 10.244 µs / 10.245 µs / 10.237 µs | -0.02% to +0.17% | -0.57% to +0.19% | 11.850 µs / 10.773 µs | -21.61% to -9.08% |

### Repeated and streaming evidence

Universe rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction remains outside measured operations.

| Indicator | Universe point estimates | Universe deltas | Per-worker point estimates | Prepared / current | Streaming point estimates | Configured / legacy |
|---|---:|---:|---:|---:|---:|---:|
| ACOS | 2.0011 ms / 2.0001 ms / 2.0339 ms | config -0.05%; prepared +1.64% | 62.278 µs / 62.847 µs | +0.91% | 237.273 µs / 236.984 µs | -0.12% |
| ASIN | 2.2424 ms / 2.2363 ms / 2.2664 ms | config -0.27%; prepared +1.07% | 70.057 µs / 70.594 µs | +0.77% | 265.598 µs / 265.411 µs | -0.07% |
| ATAN | 2.0481 ms / 2.0221 ms / 2.0632 ms | config -1.27%; prepared +0.74% | 62.971 µs / 64.187 µs | +1.93% | 259.667 µs / 259.525 µs | -0.05% |
| CEIL | 252.754 µs / 255.363 µs / 252.778 µs | config +1.03%; prepared +0.01% | 8.320 µs / 7.927 µs | -4.73% | 52.424 µs / 52.307 µs | -0.22% |
| COS | 2.0210 ms / 2.0246 ms / 2.0146 ms | config +0.18%; prepared -0.32% | 63.169 µs / 63.490 µs | +0.51% | 263.610 µs / 262.662 µs | -0.36% |
| COSH | 1.7133 ms / 1.6971 ms / 1.7307 ms | config -0.94%; prepared +1.01% | 53.248 µs / 53.554 µs | +0.58% | 200.128 µs / 200.476 µs | +0.17% |
| EXP | 1.3386 ms / 1.3541 ms / 1.3330 ms | config +1.16%; prepared -0.41% | 42.171 µs / 42.264 µs | +0.22% | 181.978 µs / 163.844 µs | -9.96% |
| FLOOR | 252.842 µs / 253.304 µs / 255.070 µs | config +0.18%; prepared +0.88% | 8.298 µs / 7.895 µs | -4.87% | 52.283 µs / 51.826 µs | -0.87% |
| LN | 1.5348 ms / 1.5358 ms / 1.5387 ms | config +0.06%; prepared +0.25% | 48.607 µs / 48.258 µs | -0.72% | 215.816 µs / 215.713 µs | -0.05% |
| LOG10 | 1.7300 ms / 1.7288 ms / 1.6948 ms | config -0.07%; prepared -2.04% | 53.087 µs / 54.049 µs | +1.81% | 216.961 µs / 198.440 µs | -8.54% |
| SIN | 2.0091 ms / 2.0116 ms / 2.0084 ms | config +0.12%; prepared -0.04% | 62.947 µs / 63.150 µs | +0.32% | 259.264 µs / 259.051 µs | -0.08% |
| SINH | 1.7299 ms / 1.7106 ms / 1.7458 ms | config -1.11%; prepared +0.92% | 53.694 µs / 54.038 µs | +0.64% | 205.375 µs / 205.392 µs | +0.01% |
| SQRT | 364.183 µs / 361.637 µs / 363.935 µs | config -0.70%; prepared -0.07% | 11.662 µs / 11.314 µs | -2.98% | 53.954 µs / 55.517 µs | +2.90% |
| TAN | 2.4084 ms / 2.3948 ms / 2.4359 ms | config -0.56%; prepared +1.14% | 74.848 µs / 75.324 µs | +0.64% | 304.392 µs / 303.676 µs | -0.24% |
| TANH | 1.3317 ms / 1.3182 ms / 1.3603 ms | config -1.01%; prepared +2.15% | 41.199 µs / 41.458 µs | +0.63% | 158.211 µs / 157.539 µs | -0.43% |

### Gate conclusion

All allocation gates clear exactly. No configuration caller-owned, prepared, Universe, per-worker, or streaming path regressed by approximately five percent against its reference. Every owned Compact Output path improved over the legacy Aligned Output path by 1.97%–49.86%. These are host-local default-`f64` qualification results, not portable speedup claims.

## Issue #12 arithmetic qualification

Issue #12 migrates the four paired arithmetic operators (`ADD`, `SUB`, `MULT`, and `DIV`) and rolling `SUM` through the Rust-first execution seam. The paired operators use one generated surface with typed borrowed batch inputs and typed ticks; `SUM` retains its Period configuration. Every configuration provides owned and caller-owned Compact Output, a reusable Prepared Batch Runner, and an independent Streaming Computation while the uppercase compatibility functions and structs retain their existing signatures and padded-owned behavior. Input Observation Series must be finite. Arithmetic results still follow the underlying IEEE-754 operations: finite operands may produce infinity on overflow or nonzero division by zero, and zero divided by zero produces `NaN`.

The full benchmark command is `cargo bench -p ta-benchmarks --bench execution_baselines -- indicator_execution/expanded/arithmetic`. The one-shot matrix uses observations 64/4,096/65,536; `SUM` uses periods 14 and 512 where the observation count permits. Workloads also record Universe (128 × 4,096), parameter sweep (four `SUM` periods × 4,096), per-worker (4 × 4,096), and streaming (16 × 4,096) execution. Values below are Criterion mean point estimates from the 100-sample qualification runs.

### Allocation evidence

`cargo bench -p ta-benchmarks --bench execution_allocations` reports:

| Indicators | Configuration | Caller-owned | Owned Compact Output | Prepared setup / first / repeated / oversize | Stream setup / ticks |
|---|---:|---:|---:|---:|---:|
| `ADD`, `SUB`, `MULT`, `DIV` | 0 / 0 B | 0 / 0 B | 1 / 32,768 B | 0 / 0 B in every phase | 0 / 0 B |
| `SUM`, period 14 | 0 / 0 B | 0 / 0 B | 1 / 32,664 B | 0 / 0 B in every phase | 1 / 112 B; 0 / 0 B |

Entries are allocation operations / gross allocated bytes for 4,096 observations. Owned byte counts are exactly `compact_count × size_of::<Float>()`; caller-owned computation, prepared construction, prepared reuse and rejection, and every streaming tick allocate nothing. The parameter-free paired-operator streams also require no setup allocation; `SUM` stream construction allocates its period-sized rolling buffer once.

### One-shot timing evidence

Caller-owned point estimates show current / configuration / prepared for 4,096 observations; the `SUM` row uses period 14. Delta ranges include every measured observation count and, for `SUM`, both measured periods. Owned point estimates show legacy Aligned Output / configuration Compact Output for 4,096 observations.

| Indicator | 4,096 caller-owned point estimates | Config / current range | Prepared / current range | 4,096 owned point estimates | Compact / legacy range |
|---|---:|---:|---:|---:|---:|
| ADD | 3.828 µs / 3.855 µs / 3.792 µs | -1.09% to +0.78% | -1.84% to -0.94% | 5.615 µs / 4.167 µs | -36.28% to -25.79% |
| SUB | 3.751 µs / 3.878 µs / 3.768 µs | -14.67% to +3.37% | -1.11% to +0.45% | 5.605 µs / 4.477 µs | -40.06% to -20.12% |
| MULT | 3.778 µs / 3.892 µs / 3.781 µs | -14.09% to +3.03% | -1.18% to +0.08% | 5.599 µs / 4.485 µs | -39.35% to -19.89% |
| DIV | 3.899 µs / 3.977 µs / 3.874 µs | -1.03% to +1.99% | -1.65% to -0.64% | 5.706 µs / 4.589 µs | -35.50% to -19.58% |
| SUM | 5.251 µs / 5.240 µs / 5.257 µs | -0.22% to +0.82% | +0.01% to +0.39% | 7.052 µs / 5.725 µs | -31.24% to -18.55% |

### Repeated and streaming evidence

Universe and parameter-sweep rows report current / configuration / prepared. Per-worker rows report current / prepared, and streaming rows report legacy / configured. Fixtures and caller-owned buffers are identical within each comparison; construction and reset remain outside measured operations.

| Indicator / workload | Point estimates | Same-run deltas |
|---|---:|---:|
| ADD Universe, 128 × 4,096 | 519.920 µs / 516.163 µs / 501.863 µs | config/current -0.72%; prepared/current -3.47% |
| ADD Per-worker, 4 × 4,096 | 15.668 µs / 15.777 µs | prepared/current +0.70% |
| ADD Streaming, 16 × 4,096 | 60.539 µs / 62.797 µs | configured/legacy +3.73% |
| SUB Universe, 128 × 4,096 | 518.964 µs / 516.716 µs / 501.977 µs | config/current -0.43%; prepared/current -3.27% |
| SUB Per-worker, 4 × 4,096 | 15.663 µs / 15.800 µs | prepared/current +0.88% |
| SUB Streaming, 16 × 4,096 | 59.715 µs / 59.618 µs | configured/legacy -0.16% |
| MULT Universe, 128 × 4,096 | 519.620 µs / 516.687 µs / 497.509 µs | config/current -0.56%; prepared/current -4.26% |
| MULT Per-worker, 4 × 4,096 | 15.622 µs / 15.737 µs | prepared/current +0.74% |
| MULT Streaming, 16 × 4,096 | 62.435 µs / 64.073 µs | configured/legacy +2.62% |
| DIV Universe, 128 × 4,096 | 536.870 µs / 536.452 µs / 520.556 µs | config/current -0.08%; prepared/current -3.04% |
| DIV Per-worker, 4 × 4,096 | 16.060 µs / 16.260 µs | prepared/current +1.25% |
| DIV Streaming, 16 × 4,096 | 149.213 µs / 155.569 µs | configured/legacy +4.26% |
| SUM Universe, 128 × 4,096 | 674.523 µs / 674.142 µs / 683.747 µs | config/current -0.06%; prepared/current +1.37% |
| SUM Sweep, 4 × 4,096 | 21.868 µs / 21.852 µs / 21.277 µs | config/current -0.08%; prepared/current -2.71% |
| SUM Per-worker, 4 × 4,096 | 21.087 µs / 21.269 µs | prepared/current +0.86% |
| SUM Streaming, 16 × 4,096 | 260.956 µs / 259.792 µs | configured/legacy -0.45% |

### Gate conclusion

All allocation gates clear exactly. No configuration caller-owned, prepared, Universe, parameter-sweep, per-worker, or streaming path regressed by approximately five percent against its same-run reference. Every owned Compact Output path improved over the legacy Aligned Output path by 18.55%–40.06%. These are host-local default-`f64` qualification results, not portable speedup claims.
