//! Focused caller-owned Hilbert/Cycle regression measurements.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use ta_benchmarks::fixture::series_fixture;
use ta_core::{
    cycle::{
        HT_DCPERIOD, HT_DCPERIOD_LOOKBACK, HT_DCPHASE, HT_DCPHASE_LOOKBACK, HT_PHASOR,
        HT_PHASOR_LOOKBACK, HT_SINE, HT_SINE_LOOKBACK, HT_TRENDMODE, HT_TRENDMODE_LOOKBACK,
    },
    Float,
};

const SIZES: &[usize] = &[256, 4_096, 65_536];

fn bench_cycle_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("issue_61_cycle_regression/caller_owned");

    for &size in SIZES {
        let input = series_fixture(size, 0);
        group.throughput(Throughput::Elements(size as u64));

        let mut dcphase = vec![0.0 as Float; size.saturating_sub(HT_DCPHASE_LOOKBACK)];
        group.bench_with_input(BenchmarkId::new("HT_DCPHASE", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    HT_DCPHASE(
                        black_box(input.as_slice()),
                        black_box(dcphase.as_mut_slice()),
                    )
                    .expect("valid HT_DCPHASE fixture"),
                )
            });
        });

        let mut dcperiod = vec![0.0 as Float; size.saturating_sub(HT_DCPERIOD_LOOKBACK)];
        group.bench_with_input(BenchmarkId::new("HT_DCPERIOD", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    HT_DCPERIOD(
                        black_box(input.as_slice()),
                        black_box(dcperiod.as_mut_slice()),
                    )
                    .expect("valid HT_DCPERIOD fixture"),
                )
            });
        });

        let mut in_phase = vec![0.0 as Float; size.saturating_sub(HT_PHASOR_LOOKBACK)];
        let mut quadrature = vec![0.0 as Float; size.saturating_sub(HT_PHASOR_LOOKBACK)];
        group.bench_with_input(BenchmarkId::new("HT_PHASOR", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    HT_PHASOR(
                        black_box(input.as_slice()),
                        black_box(in_phase.as_mut_slice()),
                        black_box(quadrature.as_mut_slice()),
                    )
                    .expect("valid HT_PHASOR fixture"),
                )
            });
        });

        let mut sine = vec![0.0 as Float; size.saturating_sub(HT_SINE_LOOKBACK)];
        let mut lead_sine = vec![0.0 as Float; size.saturating_sub(HT_SINE_LOOKBACK)];
        group.bench_with_input(BenchmarkId::new("HT_SINE", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    HT_SINE(
                        black_box(input.as_slice()),
                        black_box(sine.as_mut_slice()),
                        black_box(lead_sine.as_mut_slice()),
                    )
                    .expect("valid HT_SINE fixture"),
                )
            });
        });

        let mut trendmode =
            vec![ta_core::cycle::TrendMode::Cycle; size.saturating_sub(HT_TRENDMODE_LOOKBACK)];
        group.bench_with_input(BenchmarkId::new("HT_TRENDMODE", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    HT_TRENDMODE(
                        black_box(input.as_slice()),
                        black_box(trendmode.as_mut_slice()),
                    )
                    .expect("valid HT_TRENDMODE fixture"),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_cycle_regression);
criterion_main!(benches);
