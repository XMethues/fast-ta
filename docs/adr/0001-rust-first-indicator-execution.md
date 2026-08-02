# Rust-first indicator interfaces and separated execution state

TA-Lib defines the indicator catalogue and default mathematical reference, but not this library's public interface. We choose a Rust-first execution architecture that separates immutable indicator configuration, one-shot batch computation, reusable prepared batch execution, and streaming state so the same indicator definitions support one-off analysis, universe-scale screening, backtesting, precomputation services, and multi-instrument streaming without coupling their state or storage policies.

## Consequences

- Ordinary owned batch results are compact and carry their input length and output range; padding is not stored in the core result.
- Performance-sensitive callers can provide compact output buffers through `compute_into`.
- A prepared batch runner retains private scratch for repeated calls under one indicator configuration. For inputs within its prepared capacity, steady-state computation must not allocate; one independent runner is used per concurrent worker.
- Streaming state is created separately and is never stored in immutable indicator configuration or reused as batch scratch.
- Index outputs use Rust-native absolute indexes in compact results. Python and WASM adapters perform checked host-type conversion and own aligned/masked presentation policies.
- Scratch representation remains an implementation detail. It may be replaced without changing the public interface and must be selected by representative benchmarks rather than memory-size intuition alone.
- A one-shot `compute_into` call guarantees no output allocation, but indicator-specific scratch allocation must be documented separately.

## Performance gates

- `compute_into` performs no output allocation; kernels that are otherwise allocation-free must remain allocation-free.
- Owned batch computation performs at most one exact allocation per output column and does not stage compact values through a second padded allocation.
- Benchmarks cover one-shot series, repeated universe processing, parameter sweeps, one prepared runner per worker, and multi-instrument streaming.
- Reviews consider latency, throughput, allocation count, allocated bytes, and peak memory together.
- A stable regression greater than approximately 5% on representative workloads blocks an interface migration unless an explicit trade-off is accepted.

## Evidence

The throwaway branch `prototype/output-interface-benchmark` contains the benchmark prototypes. Compact range-bearing output removed ambiguous index padding and reduced owned multi-output allocation. Prepared scratch prototypes proved steady-state zero allocation, but both the O(max-input) append implementation and O(period) ring implementation regressed throughput, so those implementations are rejected while the prepared-execution seam is retained.
