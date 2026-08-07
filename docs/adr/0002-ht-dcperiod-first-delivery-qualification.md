# Qualify HT_DCPERIOD against an absolute first-delivery baseline

`HT_DCPERIOD` is the first Cycle Indicator Definition in this repository, so no retained Rust implementation can provide a truthful same-algorithm migration comparison. We accept its issue #19 Criterion measurements as the absolute first-delivery baseline: later work uses the same one-shot, Universe, per-worker prepared, and multi-instrument streaming benchmark IDs, and a stable regression greater than approximately five percent blocks delivery unless its trade-off is explicitly accepted. This applies ADR-0001 without presenting TA-Lib or unrelated Indicator Definitions as a relative speedup baseline.

## Status

Accepted.

## Consequences

- The qualified default-`f64` point estimates on the Apple M2 reference host are 230.08 µs for caller-owned, 231.64 µs for owned Compact Output, and 229.02 µs for prepared execution at 4,096 observations; Universe caller-owned and prepared execution are 29.837 ms and 29.877 ms for 128 × 4,096 observations; per-worker prepared execution is 932.71 µs for 4 × 4,096; and streaming is 1.6460 ms for 16 × 4,096.
- Caller-owned execution, Prepared Batch Runner setup and reuse, over-capacity rejection, stream setup, and streaming ticks allocate zero heap bytes. Owned Compact Output performs one exact 32,512-byte allocation for 4,096 observations. Peak incremental requested heap equals retained owned-output bytes and is zero on every other qualified path.
- These measurements are host-local qualification evidence, not portable speed claims. The full commands, allocation profiles, latency, and throughput records live in `crates/ta-benchmarks/EXECUTION_BASELINES.md`.
