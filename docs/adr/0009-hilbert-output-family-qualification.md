# Qualify phase, phasor, and sine through one Hilbert transition

`HT_DCPHASE`, `HT_PHASOR`, and `HT_SINE` extend the Cycle transition qualified by ADR-0002. We keep the weighted-price smoother, parity-separated Hilbert transforms, bounded period recurrence, and delayed InPhase state in one private transition. `HT_PHASOR` exposes that transition's InPhase and Quadrature components after its fixed 32-observation Stabilization. `HT_DCPHASE` and `HT_SINE` use the canonical 31-observation phase lead-in and expose phase-derived results after the fixed 63-observation Stabilization. Batch and Streaming Computation call the same transition methods; no execution mode owns a second recurrence.

## Status

Accepted.

## Consequences

- Dominant Cycle Phase is a `Float` number of degrees with the canonical `(-45, 315]` definition wrap. Phasor columns retain input-value units. Sine and LeadSine are unitless, and LeadSine is the sine of phase advanced by 45 degrees.
- The checksum-pinned TA-Lib 0.6.4 source archive remains the auditable numerical authority. Constant, trend, 20-observation sine, chirp, and seeded-noise fixtures cover phase, amplitude, alignment, and `Sine = sin(phase)` / `LeadSine = sin(phase + 45 degrees)` relationships under explicit `f64` and `f32` tolerances.
- Caller-owned execution, Prepared Batch Runner construction and reuse, over-capacity rejection, stream construction, and streaming ticks allocate zero heap bytes. At 4,096 default-`f64` observations, owned `HT_DCPHASE` performs one exact 32,264-byte allocation, owned `HT_PHASOR` performs two exact 32,512-byte allocations (65,024 bytes total), and owned `HT_SINE` performs two exact 32,264-byte allocations (64,528 bytes total). Peak incremental requested heap equals retained owned-output bytes; every other qualified path has zero incremental requested heap.
- Criterion IDs cover caller-owned, owned Compact Output, and prepared one-shot series at 64, 4,096, and 65,536 observations, plus 128-by-4,096 Universe caller-owned and prepared execution, four independent prepared runners, and sixteen independent Streaming Computations. These are absolute first-delivery baselines for each new Indicator Definition; later stable regressions greater than approximately five percent on the same host and workload block delivery unless an explicit trade-off is accepted. Cross-indicator timing ratios are not speedup claims because phase projection performs additional trigonometric work.
