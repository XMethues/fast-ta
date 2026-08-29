# Migration guide: the Rust-first indicator execution seam

This guide is the consolidated reference for migrating callers, indicators,
adapters, and benchmark suites onto the Rust-first execution architecture
ratified by [ADR-0001](../adr/0001-rust-first-indicator-execution.md) and
issue #1, then qualified end-to-end by issue #16. It supersedes the per-issue
migration guidance scattered across #2–#15 and locks in five areas of
contractual change that every consumer must honour:

1. **Compact Output** replaces the source-length `XXX_vec`-style padded output.
2. **Output Range** carries source positions for every compact payload.
3. **Prepared Batch Runner capacity** is declared up front and never grows.
4. **Independent stream state** decouples Streaming Computation from
   Indicator Configuration.
5. **Removal of implicit padding** means Aligned Output is now an explicit
   caller or adapter choice, not a core invariant.

All 100 implemented indicators expose this seam. The inventory test
`every_implemented_indicator_exposes_the_full_execution_seam` in
`crates/fast-ta/tests/inventory.rs` cross-checks the catalogue and the
execution types at compile time so adding a future indicator cannot ship
without the seam. The fixed point of 100 implemented definitions covers
the full non-pattern numerical catalogue through issue #17; Pattern
Recognition's 61 functions remain Planned until a separate stage.

## Public seam at a glance

The three traits in `crates/fast-ta/src/traits.rs` form the only public
execution surface. They are sealed: external implementations are not part of
this API contract.

```rust
pub trait IndicatorConfig: Sized + sealed::Sealed {
    type Input<'a>;
    type Output;
    type OutputMut<'a>;
    type BatchRunner: PreparedBatchRunner<Self>;
    type Stream: StreamingComputation<Self>;

    fn lookback(&self) -> usize;
    fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>>;
    fn compute_into<'a>(&self, input: Self::Input<'a>, output: Self::OutputMut<'a>)
        -> Result<OutputRange>;
    fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner>;
    fn stream(&self) -> Result<Self::Stream>;
}

pub trait PreparedBatchRunner<C: IndicatorConfig>: sealed::Sealed {
    fn max_input_len(&self) -> usize;
    fn compute_into<'a>(&mut self, input: C::Input<'a>, output: C::OutputMut<'a>)
        -> Result<OutputRange>;
}

pub trait StreamingComputation<C: IndicatorConfig>: sealed::Sealed {
    type Tick;
    type TickOutput;
    fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>>;
    fn reset(&mut self);
}
```

The catalogue uses one `Config` per indicator. Each `Config` knows its
parameter set, its input borrow shape, and its owned-output payload type, but
holds no observations and no execution buffers.

## 1. Compact Output

A Compact Output carries exactly the valid indicator values together with
their location in the source Observation Series. It is produced by
`IndicatorConfig::compute` and is the only owned result type the core ships:

```rust
let config = SMAConfig::new(14)?;
let prices: &[Float] = /* your dense, oldest-to-newest series */;

let result: CompactOutput<Vec<Float>> = config.compute(prices)?;
let source_len: usize = result.source_len();
let range: OutputRange = result.range();
let values: &[Float] = result.values();
// values.len() == range.nb_element
// range.beg_idx + range.nb_element <= source_len
```

The key invariants are:

- `source_len()` equals the original input length, not the output length.
- `range().beg_idx + range().nb_element <= source_len`. Compact element
  zero corresponds to source position `range.beg_idx`.
- `values().len() == range().nb_element`. There is no implicit padding.
- Multi-output payloads (for example `MINMAXConfig`'s `MINMAXValues`) keep
  their named columns aligned: every column has length `range.nb_element`
  and shares the same range.

### Caller-owned output

`compute_into` writes Compact Output directly into caller-owned storage. It
performs **no output allocation** and returns the [`OutputRange`]:

```rust
let config = SMAConfig::new(14)?;
let lookback = config.lookback();
let count = prices.len().saturating_sub(lookback);
let mut owned = vec![0.0 as Float; count];

let range = config.compute_into(prices, &mut owned)?;
// owned.len() == range.nb_element; no source-length padding.
```

The output-capacity check happens before any kernel runs: if `owned.len()`
is short, `compute_into` returns
[`TalibError::InvalidInput`](../../crates/fast-ta/src/error.rs) without
mutating `owned` (the error message names the missing capacity). The same
pre-mutation rule applies to Prepared Batch Runners.

## 2. Output Range

`OutputRange { beg_idx, nb_element }` is the bridge between Compact Output and
the original Observation Series. Two helpers capture the most common cases:

```rust
let range = result.range();
let first_source_position = range.beg_idx;
let one_past_last_source_position = range.end_idx();
let is_empty = range.is_empty();
```

Walking the compact payload by source position is an explicit caller or
adapter responsibility. The simplest pattern, when you need a source-length
buffer, is:

```rust
let source_len = result.source_len();
let range = result.range();
let compact = result.values();

let mut aligned: Vec<Option<Float>> = vec![None; source_len];
for (i, value) in compact.iter().enumerate() {
    aligned[range.beg_idx + i] = Some(*value);
}
```

This pattern is the modern replacement for the removed `padded_from_compact`
helper and the legacy `XXX_vec` family of source-length convenience calls.
Each adapter is free to choose its own representation (masked `Vec<f32>`,
NumPy `ndarray` with `NaN`, JavaScript typed array with `null`, etc.) — the
core does not store any sentinel value.

Index outputs use Rust-native absolute `usize` values. `MINMAXINDEXConfig`'s
columns and `MAXINDEXConfig` / `MININDEXConfig` outputs are positioned the
same way: compact element `i` corresponds to source position
`range.beg_idx + i`. Adapters perform the `usize → host index type` checked
conversion at the boundary.

## 3. Prepared Batch Runner capacity

`prepare_batch(max_input_len)` declares a hard source capacity up front. The
runner reserves any algorithm scratch it needs for inputs no longer than
`max_input_len`, and rejects oversized inputs with a structured error
**before** any other validation or mutation:

```rust
let config = SMAConfig::new(14)?;
let mut runner = config.prepare_batch(4096)?;
assert_eq!(runner.max_input_len(), 4096);

let range = runner.compute_into(prices, &mut owned)?;
// Oversized input is a clean error:
//   runner.compute_into(&prices[..4097], &mut owned)
//   => Err(TalibError::prepared_capacity_exceeded(4096, 4097))
```

Concurrent workers each own an independent runner. The `IndicatorConfig` is
immutable and may be shared freely; the runner is the per-worker state.

```rust
let config = Arc::new(SMAConfig::new(14)?);
let handle = std::thread::spawn(move || {
    let mut runner = config.prepare_batch(4096).unwrap();
    runner.compute_into(&prices, &mut owned).unwrap();
});
```

Inside capacity, repeated calls allocate nothing on the steady-state hot
path. `execution_allocations` records the exact-zero profile per indicator
family in `crates/ta-benchmarks/EXECUTION_BASELINES.md`.

Algorithm scratch that some indicators (for example `MINMAXConfig`) need for
a one-shot `compute_into` call is documented separately; those indicators
move the scratch into the Prepared Batch Runner so repeated prepared calls
remain allocation-free.

## 4. Independent stream state

`stream()` constructs a fresh per-instrument Streaming Computation. The
stream is **not** stored inside the `IndicatorConfig`; each call returns an
independent state value:

```rust
let config = SMAConfig::new(14)?;
let mut aapl = config.stream()?;
let mut msft = config.stream()?;

// First 13 ticks on AAPL: warm-up, `Ok(None)`.
for tick in &aapl_ticks[..13] {
    assert!(aapl.next(*tick)?.is_none());
}

// 14th tick onwards: one valid output per tick.
for tick in &aapl_ticks[13..] {
    let value = aapl.next(*tick)?.unwrap();
    // ...
}

// MSFT ticks never influence AAPL state.
for tick in &msft_ticks {
    let _ = msft.next(*tick)?;
}
```

Rules of the streaming contract:

- Warm-up returns `Ok(None)`. It is **not** an error.
- `next` validates the tick first; rejected ticks preserve replay position
  (count, seen, internal buffers) so a caller can recover by sending the
  same tick again.
- `reset()` returns the stream to its initial warm-up state without
  reallocating buffers.
- Each `stream()` call produces a fresh, independent state. Two streams
  derived from one `Config` must never share scratch, so a market-data
  service that ingests N instruments builds N streams.
- Index outputs (`MINMAXINDEXConfig`, `MAXINDEXConfig`, `MININDEXConfig`)
  emit Rust-native absolute `usize` indexes. Adapter code performs
  `usize::try_into()` at the boundary.

Batch and streaming agree under each Indicator Definition's numerical
contract. Where the streaming path differs from the batch path (for example,
strict validation order on invalid ticks, or batch-specific lookback
enforcement), the difference is documented in the per-family contract
tests in `crates/fast-ta/tests/indicator_execution_contracts.rs`.

## 5. Removal of implicit padding

The core no longer stores warm-up `NaN` values, zero-index sentinels, or any
other padding inside owned output. Concrete consequences:

- `XXX_vec`-style source-length convenience calls are **gone**. Owning a
  source-length buffer is now a caller-side or adapter-side decision.
- The legacy `padded_from_compact` helper is **gone**. The walk-the-compact
  pattern in section 2 is the supported replacement.
- Aligned Output is **explicit**. Adapter crates are responsible for the
  nullability policy (NumPy `NaN`, Python `None`, JavaScript `null`,
  optional `Vec<Float>`, mask bit, etc.) and for converting absolute
  `usize` indexes to host types.
- `Indicator::compute_to_vec` and the `Indicator` / `StreamingIndicator`
  / `Resettable` legacy traits are **gone**. New code uses only the
  sealed seam.

If you previously wrote:

```rust
let mut out = vec![0.0; n];
indicator.compute_into(prices, &mut out)?;
// out[0..lookback] was unusable / sentinel-filled
```

you now write one of the patterns in sections 1–3 and never store padding
in the result.

## Workload examples

The five workloads called out in ADR-0001 and accepted by issue #16.

### One-shot owned Compact Output

The ordinary analyst path. One configuration, one Observation Series, one
owned result:

```rust
let config = SMAConfig::new(14)?;
let result = config.compute(prices)?;
// Source-aligned walk only if needed; otherwise consume `result.into_values()`.
```

### Caller-owned Compact Output

The performance-sensitive caller path. Allocate the output once, reuse it:

```rust
let config = SMAConfig::new(14)?;
let count = prices.len() - config.lookback();
let mut owned = vec![0.0 as Float; count];
let _range = config.compute_into(prices, &mut owned)?;
```

### Universe screening

One `Config` shared across many instruments, each with its own caller-owned
buffer:

```rust
let config = Arc::new(SMAConfig::new(14)?);
let count = instrument[0].len() - config.lookback();
let mut buffers: Vec<Vec<Float>> = instruments.iter()
    .map(|_| vec![0.0 as Float; count])
    .collect();
for (series, buf) in instruments.iter().zip(buffers.iter_mut()) {
    let _ = config.compute_into(series, buf)?;
}
```

### Parameter sweep

One `Config` per period, each in its own slot. Indicators that need
algorithm scratch (for example `MINMAXConfig`) move the scratch into the
Prepared Batch Runner:

```rust
let periods = [5, 14, 50, 200];
let configs: Vec<SMAConfig> = periods.iter()
    .map(|&p| SMAConfig::new(p).unwrap())
    .collect();
// Each config owns a buffer sized to its own lookback so the output
// capacity check passes for every sweep slot.
let mut buffers: Vec<Vec<Float>> = configs.iter()
    .map(|cfg| vec![0.0 as Float; prices.len() - cfg.lookback()])
    .collect();
for (cfg, buf) in configs.iter().zip(buffers.iter_mut()) {
    let _ = cfg.compute_into(prices, buf)?;
}
```

### Per-worker Prepared Batch Runner

Each thread owns its runner. `prepare_batch` happens on the worker, not the
dispatcher:

```rust
let config = Arc::new(MINMAXConfig::new(14)?);
let handles: Vec<_> = fixtures.into_iter().map(|prices| {
    let cfg = Arc::clone(&config);
    std::thread::spawn(move || {
        let mut runner = cfg.prepare_batch(prices.len()).unwrap();
        let count = prices.len() - cfg.lookback();
        let mut min = vec![0.0; count];
        let mut max = vec![0.0; count];
        let mut out = MINMAXValuesMut { min: &mut min, max: &mut max };
        runner.compute_into(&prices, &mut out).unwrap();
    })
}).collect();
for h in handles { h.join().unwrap(); }
```

Inside capacity, repeated prepared calls allocate nothing. The exact-zero
profile per indicator family is recorded in
`crates/ta-benchmarks/EXECUTION_BASELINES.md` under each issue's
`Allocation evidence` section.

### Multi-instrument Streaming Computation

One independent stream per instrument, all created from the same `Config`:

```rust
let config = SMAConfig::new(14)?;
let streams: Vec<SMAStream> = instruments.iter()
    .map(|_| config.stream().unwrap())
    .collect();
for tick_batch in market_data {
    for (stream, tick) in streams.iter_mut().zip(tick_batch.iter()) {
        if let Some(value) = stream.next(*tick)? {
            route_to_consumer(stream.id(), value);
        }
    }
}
```

The `BETAConfig` / `CORRELConfig` paired-input streams follow the same
pattern with a `PairTick { real0, real1 }` instead of a single `Float`.

## Adding a new indicator

The seam contract is the same for new indicators. Follow these steps:

1. **Inventory.** Add the function to
   `crates/fast-ta/src/inventory.rs` and mark it `Implemented` so
   `every_implemented_indicator_exposes_the_full_execution_seam` will
   require its seam coverage at compile time.
2. **Configuration.** Implement a `<NAME>Config` that holds only parameters.
   Implement `IndicatorConfig` with `Input`, `Output`, `OutputMut`,
   `BatchRunner`, and `Stream` associated types. The trait is sealed, so the
   blanket implementations live in the same module.
3. **Prepared Batch Runner.** Implement `<NAME>BatchRunner` with private
   scratch sized at `prepare_batch(max_input_len)` time. Reject oversized
   inputs with `TalibError::prepared_capacity_exceeded` before any other
   validation or mutation.
4. **Streaming state.** Implement `<NAME>Stream` so `next` returns
   `Ok(None)` during warm-up and `Ok(Some(value))` thereafter. `reset`
   must return the stream to its initial state without reallocating
   buffers. Per-instrument streams must be fully independent.
5. **Tests.** Extend the existing family suite under
   `crates/fast-ta/tests/` with the five contract checks called out in
   `indicator_execution_contracts.rs`: pre-mutation validation, owned
   vs. caller-owned equality, prepared-batch reuse and oversize rejection,
   independent streams, and batch/stream parity. Add an inventory assertion
   to `every_implemented_indicator_exposes_the_full_execution_seam` so a
   future macro regression cannot drop the new indicator.
6. **Benchmarks.** Extend the family's allocation and Criterion benches so
   `EXECUTION_BASELINES.md` records the new indicator's exact allocation
   profile, one-shot timing, and Universe / parameter-sweep / per-worker /
   streaming workload coverage. The shared support helpers live under
   `crates/ta-benchmarks/benches/support/`.

## Moving-average Momentum configurations

APO, PPO, and MACDEXT accept `PeriodMAType`, not a generic TA-Lib selector
number. The enum contains only implemented, single-output Period-based Moving
Averages (`SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `T3`, and `KAMA`);
MAMA has its own paired-output definition and cannot be selected here.

```rust
let config = MACDEXTConfig::new(
    12,
    PeriodMAType::EMA,
    26,
    PeriodMAType::KAMA,
    9,
    PeriodMAType::EMA,
)?;
let result = config.compute(prices)?;
let values: &MACDValues = result.values();
// values.macd, values.signal, and values.histogram have equal compact lengths.
```

Caller-owned MACD-family execution uses `MACDValuesMut`; supply all three
columns at `prices.len().saturating_sub(config.lookback())`. Capacity is
validated for every column before any is mutated. Standard `MACDConfig`
selects EMA for all three Periods, while `MACDFIXConfig` fixes the fast and
slow Periods to 12 and 26. `TRIXConfig` returns one compact percentage-change
column and does not expose or discard its internal EMA stages.

## Adding a new adapter

Adapters live outside the core seam. Each adapter is free to choose its own
nullability and alignment policy.

- **Compact payload shape.** Use `result.source_len()` to size the host
  buffer and `result.range()` to place valid values at
  `range.beg_idx + i`. Compact element zero corresponds to the first source
  position in the range, so absolute source positions are recoverable
  without inspecting padding sentinels.
- **Index conversion.** Rust index outputs use `usize`. Perform
  `usize::try_into()` at the adapter boundary and surface host-specific
  range errors as a structured error, not a panic.
- **Aligned Output.** Decide once per adapter whether to expose a
  source-length typed array, a masked container, a `(value, valid)`
  pair-vector, or a NumPy `ndarray` with `NaN`. The core never assumes the
  host representation.
- **No core leakage.** Do not import private core types. The only public
  surface is `IndicatorConfig`, `PreparedBatchRunner`,
  `StreamingComputation`, `CompactOutput`, `OutputRange`, `Result`, and
  `Float`.

## Acceptance gate summary

The issue #16 end-to-end acceptance gate requires every item below to be
green on the merged commit:

- `cargo test -p fast-ta --all-targets`
- `cargo test -p fast-ta --no-default-features --features f32,std`
- `cargo clippy -p fast-ta --all-targets` (no new warnings beyond the
  20-warning baseline)
- `cargo fmt --all -- --check`
- `cargo check -p ta-benchmarks --benches`
- `every_implemented_indicator_exposes_the_full_execution_seam` covering
  all 100 implemented indicators
- ADR-0001 performance gates (latency, throughput, allocation count,
  allocated bytes, peak memory) with explicit accepted trade-offs
  documented in `crates/ta-benchmarks/EXECUTION_BASELINES.md`
- Prototype branches (`prototype/output-interface-benchmark`) absent from
  the merged history