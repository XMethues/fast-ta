# ta-benchmarks

## Responsibility
Criterion benchmark crate for measuring `ta-core` algorithms. It is tooling-only: production logic stays in `ta-core`.

## Dependencies
- **Criterion**: benchmark groups, parameterized IDs, black-boxing, and executable entrypoints.
- **ta-core**: public API under test when benchmarking indicators/SIMD operations.

## Consumers
- **Cargo bench**: runs declared `[[bench]]` targets.
- **Developers/CI**: compare performance while changing core algorithms.

## Module Structure
```text
crates/ta-benchmarks/
├── Cargo.toml      # Criterion bench targets with harness=false
└── benches/        # one Criterion executable target per file
```

## Criterion Target Wiring

```toml
[[bench]]
name = "overlap"
harness = false # Criterion supplies main()
```

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_overlap(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap");
    group.finish();
}

criterion_group!(benches, bench_overlap);
criterion_main!(benches);
```

## Parameterized Benchmarks with Prepared Fixtures

```rust
use criterion::{black_box, BenchmarkId, Criterion};

fn bench_vector_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");
    for &size in &[128_usize, 1_024, 16_384] {
        group.bench_with_input(BenchmarkId::new("sum", size), &size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect(); // outside timing
            b.iter(|| {
                let sum: f64 = black_box(&data).iter().copied().sum();
                black_box(sum)
            });
        });
    }
    group.finish();
}
```

## ta-core Indicator Benchmark Boundary

```rust
use criterion::{black_box, BenchmarkId, Criterion};
use ta_core::{overlap::SMA, Float, Indicator};

fn bench_sma_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/sma");
    for &size in &[1_024_usize, 16_384] {
        group.bench_with_input(BenchmarkId::new("compute_to_vec", size), &size, |b, &size| {
            let prices: Vec<Float> = (0..size).map(|i| i as Float + 1.0).collect();
            let sma = SMA::new(20); // outside timing unless constructor cost is intentional
            b.iter(|| {
                let output = sma.compute_to_vec(black_box(&prices)).expect("valid fixture");
                black_box(output)
            });
        });
    }
    group.finish();
}
```

## Architectural Boundaries
- **NO production code**: benchmark helpers are for measurement only.
- **NO private core modules**: use public exports such as `Float`, `Indicator`, and indicator structs.
- **NO fixture allocation inside `b.iter()`** unless allocation cost is the target metric.
- **NO unregistered benchmark functions**: every `bench_*` must appear in `criterion_group!`.

<important if="you are adding a new benchmark">
## Adding a New Benchmark
1. Use `bench_function` for one fixed workload or `bench_with_input` with `BenchmarkId` for matrices.
2. Prepare deterministic fixtures outside `b.iter()`.
3. Wrap inputs and outputs with `black_box`.
4. Register the function in `criterion_group!`.
5. Add a new `[[bench]]` target if creating a new file under `benches/`.
6. Run `cargo bench -p ta-benchmarks --bench <target>`.
</important>
