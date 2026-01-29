//! Benchmarks for SMA indicator
//!
//! This module contains performance benchmarks for SMA indicator,
//! comparing SIMD-accelerated implementation against scalar reference.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ta_core::{overlap::Sma, traits::Indicator};

fn bench_sma_compute_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma_compute");

    for size in [10, 100, 500, 1000, 2000, 5000, 10000].iter() {
        let period = 20;
        let sma = Sma::new(period).unwrap();
        let data: Vec<f64> = (0..*size).map(|i| i as f64 + 1.0).collect();
        let lookback = sma.lookback();

        if *size > lookback {
            let mut outputs = vec![0.0; *size - lookback];

            group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
                b.iter(|| {
                    let count = sma
                        .compute(black_box(&data), black_box(&mut outputs))
                        .unwrap();
                    black_box(count)
                })
            });
        }
    }

    group.finish();
}

fn bench_sma_compute_to_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma_compute_to_vec");

    for size in [10, 100, 500, 1000, 2000, 5000, 10000].iter() {
        let period = 20;
        let sma = Sma::new(period).unwrap();
        let data: Vec<f64> = (0..*size).map(|i| i as f64 + 1.0).collect();

        group.bench_with_input(BenchmarkId::new("simd", size), size, |b, &_size| {
            b.iter(|| {
                let result = sma.compute_to_vec(black_box(&data)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

fn bench_sma_next(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma_next");

    let iterations = 10000;
    let period = 20;
    let mut sma = Sma::new(period).unwrap();

    group.bench_function("streaming", |b| {
        b.iter(|| {
            let mut sma = Sma::new(period).unwrap();
            for i in 0..iterations {
                let value = (i % 100) as f64 + 1.0;
                black_box(sma.next(value));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sma_compute_simd,
    bench_sma_compute_to_vec,
    bench_sma_next
);
criterion_main!(benches);
