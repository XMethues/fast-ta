# Complete the Cycle family with typed Trend Mode classification

`HT_TRENDMODE` completes the five-definition Cycle family by classifying each valid source position with the named `TrendMode::Cycle` or `TrendMode::Trend` variant. We reject Boolean and C-style integer results because they obscure which state a value names, and we reject direction, strength, and probability interpretations because the Indicator Definition computes none of them. The immutable parameter-free configuration, caller-owned and owned Compact Output, Prepared Batch Runner, and independent Streaming Computation all use the dominant-phase projection over the one private Hilbert transition qualified by ADR-0002 and ADR-0009.

## Status

Accepted.

## Consequences

- Fixed Stabilization and Lookback are 63 observations. Streaming Warm-up is 63 ticks and the first result belongs to source position 63. No runtime unstable-period setting changes this definition.
- The classification retains the canonical sine/lead-sine crossing, dominant-phase progression, dominant-period trendline, and 1.5-percent smoothed-price displacement rules. Its raw-price and smoothed-price histories are fixed-size execution state; configuration never owns accumulated observations.
- The checksum-pinned TA-Lib 0.6.4 source archive remains the auditable numerical authority. Constant, linear-trend, 20-observation sine, chirp, and seeded-noise vectors cover the two-variant domain and transitions under both supported input precisions. Reference integers are converted by the generator into a private typed fixture enum and never enter the public interface.
- Every failure is checked before mutation: batch input and capacity validation preserve caller storage, and rejected non-finite streaming ticks preserve state. Reset returns a stream to its initial state, so replay is deterministic. Prepared runners and streams are independent.
- At 4,096 default-`f64` observations, owned computation performs one exact 4,033-byte allocation for 4,033 `TrendMode` values; requested incremental peak and retained heap are both 4,033 bytes. Configuration, caller-owned, prepared, over-capacity rejection, stream construction, and streaming ticks allocate zero bytes.
- Criterion workloads retain caller-owned, owned, and prepared one-shot IDs at 64, 4,096, and 65,536 observations, a 128-instrument Universe through caller-owned and prepared execution, four independent per-worker runners, and sixteen independent streams.
- With `HT_DCPERIOD`, `HT_DCPHASE`, `HT_PHASOR`, and `HT_SINE`, all five Cycle Indicator Definitions now pass one shared-recurrence inventory, supported `no_std`, allocation, peak-memory, representative workload, and two-axis review qualification. Catalogue and group totals remain fixed; only implementation status changes.
