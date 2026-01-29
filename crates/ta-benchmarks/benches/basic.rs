//! Basic benchmarks for TA indicators
//!
//! This module contains fundamental benchmarks to validate the benchmarking setup.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Simple addition benchmark to validate Criterion setup
fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_math");

    group.bench_function("add_two_numbers", |b| {
        b.iter(|| {
            let a: f64 = black_box(1.0);
            let b: f64 = black_box(2.0);
            a + b
        })
    });

    group.finish();
}

/// Vector sum benchmark
fn bench_vector_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    for size in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("sum", size), size, |b, &size| {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

            b.iter(|| {
                let sum: f64 = black_box(&data).iter().sum();
                black_box(sum)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_add, bench_vector_sum);
criterion_main!(benches);
