---
date: 2026-07-05
repository: fast-ta
branch: main
host: aarch64-apple-darwin
baseline: before-algo-simd-opt
status: reverted
---

# Platform SIMD ADD Attempt

## What Was Tried

A minimal platform-dispatched `add_into` kernel was prototyped:

- `simd::scalar::add_into` fallback
- `simd::dispatch::add_into` function pointer
- AArch64 NEON implementation using `vld1q_*`, `vaddq_*`, `vst1q_*`
- x86 AVX2/AVX512 implementations were sketched for portability
- `math_operators::ADD` was wired through `crate::simd::add_into`

The host was `aarch64-apple-darwin`, so the measured platform path was NEON.

## Result

Against the original `before-algo-simd-opt` baseline, `ADD_compact` showed:

| Size | Result with platform SIMD ADD | Criterion judgement |
|---:|---:|---|
| 1,024 | 1.0758 µs | +3.44% regression |
| 16,384 | 16.907 µs | -5.45% improvement vs original baseline |
| 65,536 | 68.658 µs | -5.40% improvement vs original baseline |

However, compared with the final retained scalar/LLVM-auto-vectorized algorithm-optimized run, the NEON path was slower:

| Size | Final scalar/auto-vectorized retained run | NEON dispatch attempt |
|---:|---:|---:|
| 1,024 | ~970.70 ns | 1.0758 µs |
| 16,384 | ~16.672 µs | 16.907 µs |
| 65,536 | ~67.383 µs | 68.658 µs |

## Decision

The platform SIMD ADD attempt was reverted. The current scalar loop is already optimized well by LLVM on this host, and the function-pointer dispatch + explicit NEON kernel did not beat it.

## Follow-up Ideas

If SIMD work continues, benchmark standalone kernels before wiring indicators:

1. direct scalar loop
2. `#[target_feature]` direct-call NEON/AVX kernel without dispatch
3. dispatch-call NEON/AVX kernel
4. aligned-buffer variants
5. larger benchmark sizes where dispatch overhead and memory bandwidth effects are clearer
