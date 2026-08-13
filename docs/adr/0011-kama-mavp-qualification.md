# Qualify KAMA and integer-selected MAVP

Kaufman Adaptive Moving Average (KAMA) is accepted as a single-output Period-based Moving Average and therefore joins `PeriodMAType`. Moving Average with Variable Period (MAVP) is accepted as a separate Indicator Definition whose Observation Series is aligned one-for-one with a `usize` Period Selection Series. Its immutable configuration contains inclusive minimum and maximum Period bounds and one `PeriodMAType`. Each selection is clamped as an integer before evaluation; floating-point periods and truncation are not representable. MESA Adaptive Moving Average (MAMA) remains outside `PeriodMAType` because its configuration and paired MAMA/FAMA output are not Period-based.

## Status

Accepted.

## Definition and output contract

For Period $p>1$, KAMA begins at source position $p$. Its initial previous value is observation $p-1$. At each valid position, efficiency is the absolute $p$-position change divided by the sum of absolute one-position changes, capped at one; zero-volatility input uses efficiency one. The adaptive factor is

$$
\left(\frac{2}{31}+\text{efficiency}\left(\frac{2}{3}-\frac{2}{31}\right)\right)^2.
$$

A Period of one is the identity definition with zero Lookback. MAVP's Lookback is the selected moving-average definition's Lookback at the configured maximum Period. Its Compact Output begins there. For a selected Period whose own Lookback is $l$, evaluation begins at source position $L-l$, where $L$ is MAVP's Lookback, so that every selected definition produces its first aligned result at $L$. This preserves the TA-Lib MAVP restart/seed semantics for recursive definitions while remaining identical to the natural aligned window for rolling definitions. Selections below or above the bounds clamp to the respective bound.

## Qualification evidence

- `crates/fast-ta/tests/fixtures/kama_mavp_reference.rs` pins TA-Lib KAMA/MAVP semantics at revision `e64d2ac896c595f38d65e44c812efbfdac8a64cf` and default-`f64` vectors generated independently with 50-digit Python `Decimal` arithmetic. `generate_kama_mavp.py` is the auditable generator. The behavior tests apply explicit `f64` and `f32` tolerances.
- The public-seam tests cover direct and selector KAMA, MAVP owned and caller-owned Compact Output, repeated Prepared Batch Runner execution, independent streaming, reset/replay, adaptivity, integer bound clamping, flat input, source alignment, every qualified `PeriodMAType`, and failure non-mutation/state preservation.
- Both modules select `Vec` from `alloc` without `std` and retain the crate's `f64`/`f32` feature-independent `Float` formulas. The supported checks are `cargo check -p ta-core --no-default-features --features f64` and `cargo check -p ta-core --no-default-features --features f32`; they exercise the same production definitions rather than alternate implementations.
- KAMA caller-owned and prepared Batch Computation allocate zero bytes. Owned KAMA allocates exactly one Compact Output column. A KAMA stream allocates one ring of $(p+1)\times\text{size_of::<Float>()}$ bytes for $p>1$.
- MAVP caller-owned Batch Computation allocates exactly two $(n-L)$-element temporary buffers: one `Float` moving-average output scratch vector and one `usize` used-Period scratch vector, where $n$ is source length and $L$ is Lookback. Owned MAVP adds exactly one $(n-L)\times\text{size_of::<Float>()}$ Compact Output allocation. At maximum source capacity $N$, a Prepared Batch Runner reserves exactly $\max(N-L,0)\times(\text{size_of::<Float>()}+\text{size_of::<usize>()})$ bytes and repeated calls allocate zero bytes.
- MAVP streaming prepares one independent moving-average stream for every Period in its inclusive configured bounds. Stream creation owns the selector vector and definition-specific stream storage; every aligned tick and reset allocate zero bytes.
- Criterion IDs under `indicator_execution/expanded/variable_period_overlap` cover KAMA caller-owned, owned, and prepared one-shot paths plus MAVP representative Period Selection Series, Universe prepared execution, one runner per worker, and independent multi-stream execution. The allocation harness asserts operation count, gross bytes, peak incremental bytes, and retained bytes for KAMA and MAVP paths.

These profiles are executable contracts in the benchmark harness rather than portable speed claims. Host-local point estimates belong in `crates/ta-benchmarks/EXECUTION_BASELINES.md` when the qualification harness is run on the reference host.
