//! First-tranche `ta-core` indicator benchmarks.
//!
//! These benchmarks exercise the public Rust APIs designed for the foundation
//! tranche: compact zero-copy kernels plus selected padded convenience wrappers.
//! Fixtures and reusable output buffers are allocated outside `b.iter()` unless
//! the wrapper allocation itself is the behavior under measurement.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ta_core::{
    math_operators::{ADD, MINMAX, SUM},
    math_transform::SQRT,
    overlap::{SMA_vec, SMA},
    price_transform::{AVGDEV, AVGPRICE},
    Float,
};

const SIZES: &[usize] = &[1_024, 16_384, 65_536];
const PERIOD: usize = 20;

fn series_fixture(size: usize) -> Vec<Float> {
    (0..size)
        .map(|idx| ((idx % 997) as Float + 1.0 as Float) * 0.5 as Float)
        .collect()
}

fn paired_fixture(size: usize) -> (Vec<Float>, Vec<Float>) {
    let left = series_fixture(size);
    let right = left
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 17) as Float + 1.0 as Float)
        .collect();
    (left, right)
}

fn ohlc_fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    let close = series_fixture(size);
    let open: Vec<Float> = close
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 5) as Float * 0.01 as Float)
        .collect();
    let high: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::max(*open, *close) + 1.0 as Float)
        .collect();
    let low: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::min(*open, *close) - 1.0 as Float)
        .collect();

    (open, high, low, close)
}

fn bench_overlap_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap/sma");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SMA_vec", size), &size, |b, &size| {
            let prices = series_fixture(size);

            b.iter(|| {
                let output = SMA_vec(black_box(prices.as_slice()), black_box(PERIOD))
                    .expect("valid SMA benchmark fixture");
                black_box(output);
            });
        });
    }

    group.finish();
}

fn bench_price_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/price_transform");

    for &size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("AVGPRICE_compact", size),
            &size,
            |b, &size| {
                let (open, high, low, close) = ohlc_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = AVGPRICE(
                        black_box(open.as_slice()),
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AVGPRICE benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("AVGDEV_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = AVGDEV(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AVGDEV benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );
    }

    group.finish();
}

fn bench_math_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_transform");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SQRT_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SQRT(
                    black_box(prices.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SQRT benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_math_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_operators");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("ADD_compact", size), &size, |b, &size| {
            let (left, right) = paired_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = ADD(
                    black_box(left.as_slice()),
                    black_box(right.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid ADD benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SUM_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SUM(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SUM benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("MINMAX_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut min = vec![0.0 as Float; size];
                let mut max = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = MINMAX(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(min.as_mut_slice()),
                        black_box(max.as_mut_slice()),
                    )
                    .expect("valid MINMAX benchmark fixture");
                    black_box(range);
                    black_box(min.as_slice());
                    black_box(max.as_slice());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overlap_sma,
    bench_price_transform,
    bench_math_transform,
    bench_math_operators
);
criterion_main!(benches);
