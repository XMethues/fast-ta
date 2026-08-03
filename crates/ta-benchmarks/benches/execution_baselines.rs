//! Current Indicator execution latency and throughput baselines for issue #2.
//!
//! This benchmark deliberately uses only the public traits and named borrowed
//! views. It characterizes the legacy seam before ADR-0001 is implemented; in
//! particular, owned results are currently legacy Aligned Output rather than
//! target Compact Output, and no Prepared Batch Runner or input-capacity
//! contract exists yet. The per-worker benchmark is the legacy predecessor to
//! the target runner seam that issue #3 will implement.

mod support;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use support::{
    ohlc_fixture, output_len, series_fixture, PERIOD, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS,
    SWEEP_PERIODS, UNIVERSE_INSTRUMENTS, WORKERS,
};
use ta_core::{
    math_operators::{MINMAXINDEXOutputMut, MINMAXOutputMut, MINMAX, MINMAXINDEX},
    overlap::SMA,
    price_transform::{AVGPRICEInput, AVGPRICE},
    Float, Indicator, StreamingIndicator,
};

const SIZES: &[usize] = &[64, 4_096, 65_536];

fn bench_sma_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/current/one_shot/SMA");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("caller_compact", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = SMA::new(black_box(PERIOD)).expect("valid period");
                let mut output = vec![0.0 as Float; output_len(size, PERIOD)];

                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid SMA fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("owned_legacy_aligned", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = SMA::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                        )
                        .expect("valid SMA fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_avgprice_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/current/one_shot/AVGPRICE");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("caller_compact", size),
            &size,
            |b, &size| {
                let ohlc = ohlc_fixture(size);
                let indicator = AVGPRICE::new().expect("valid AVGPRICE configuration");
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(AVGPRICEInput {
                            open: ohlc.open.as_slice(),
                            high: ohlc.high.as_slice(),
                            low: ohlc.low.as_slice(),
                            close: ohlc.close.as_slice(),
                        }),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AVGPRICE fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("owned_legacy_aligned", size),
            &size,
            |b, &size| {
                let ohlc = ohlc_fixture(size);
                let indicator = AVGPRICE::new().expect("valid AVGPRICE configuration");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(AVGPRICEInput {
                                open: ohlc.open.as_slice(),
                                high: ohlc.high.as_slice(),
                                low: ohlc.low.as_slice(),
                                close: ohlc.close.as_slice(),
                            }),
                        )
                        .expect("valid AVGPRICE fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_minmax_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/current/one_shot/MINMAX");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("caller_compact", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = MINMAX::new(black_box(PERIOD)).expect("valid period");
                let output_len = output_len(size, PERIOD);
                let mut min = vec![0.0 as Float; output_len];
                let mut max = vec![0.0 as Float; output_len];

                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(input.as_slice()),
                        MINMAXOutputMut {
                            min: black_box(min.as_mut_slice()),
                            max: black_box(max.as_mut_slice()),
                        },
                    )
                    .expect("valid MINMAX fixture");
                    black_box((range, min.as_slice(), max.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("owned_legacy_aligned", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = MINMAX::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                        )
                        .expect("valid MINMAX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_minmaxindex_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/current/one_shot/MINMAXINDEX");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("caller_compact", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = MINMAXINDEX::new(black_box(PERIOD)).expect("valid period");
                let output_len = output_len(size, PERIOD);
                let mut min_idx = vec![0_i32; output_len];
                let mut max_idx = vec![0_i32; output_len];

                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(input.as_slice()),
                        MINMAXINDEXOutputMut {
                            min_idx: black_box(min_idx.as_mut_slice()),
                            max_idx: black_box(max_idx.as_mut_slice()),
                        },
                    )
                    .expect("valid MINMAXINDEX fixture");
                    black_box((range, min_idx.as_slice(), max_idx.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("owned_legacy_aligned", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let indicator = MINMAXINDEX::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                        )
                        .expect("valid MINMAXINDEX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_universe(c: &mut Criterion) {
    let universe = (0..UNIVERSE_INSTRUMENTS)
        .map(|seed| series_fixture(REPEATED_SERIES_LEN, seed))
        .collect::<Vec<_>>();
    let indicator = SMA::new(PERIOD).expect("valid period");
    let mut output = vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)];
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("universe/SMA/caller_compact_reuse", |b| {
        b.iter(|| {
            for series in &universe {
                let range = Indicator::compute(
                    black_box(&indicator),
                    black_box(series.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid Universe fixture");
                black_box((range, output.as_slice()));
            }
        });
    });

    group.finish();
}

fn bench_parameter_sweep(c: &mut Criterion) {
    let input = series_fixture(REPEATED_SERIES_LEN, 0);
    let indicators = SWEEP_PERIODS
        .iter()
        .map(|&period| SMA::new(period).expect("valid sweep period"))
        .collect::<Vec<_>>();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("parameter_sweep/SMA/caller_compact_reuse", |b| {
        b.iter(|| {
            for indicator in &indicators {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid parameter-sweep fixture");
                black_box((range, output.as_slice()));
            }
        });
    });

    group.finish();
}

fn bench_per_worker_predecessor(c: &mut Criterion) {
    let indicators = (0..WORKERS)
        .map(|_| SMA::new(PERIOD).expect("valid period"))
        .collect::<Vec<_>>();
    let inputs = (0..WORKERS)
        .map(|seed| series_fixture(REPEATED_SERIES_LEN, seed))
        .collect::<Vec<_>>();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)])
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));

    group.bench_function("no_prepared_runner/per_worker_instances", |b| {
        b.iter(|| {
            for ((indicator, input), output) in
                indicators.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid per-worker fixture");
                black_box((range, output.as_slice()));
            }
        });
    });

    group.finish();
}

fn bench_multi_instrument_streaming(c: &mut Criterion) {
    let instrument_inputs = (0..STREAM_INSTRUMENTS)
        .map(|seed| series_fixture(REPEATED_SERIES_LEN, seed))
        .collect::<Vec<_>>();
    let inputs = (0..REPEATED_SERIES_LEN)
        .map(|tick_index| {
            instrument_inputs
                .iter()
                .map(|series| series[tick_index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("indicator_execution/current/streaming");
    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("SMA/independent_multi_instrument_instances", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| SMA::new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for tick in &inputs {
                    for (stream, &input) in streams.iter_mut().zip(tick.iter()) {
                        let output = StreamingIndicator::next(black_box(stream), black_box(input))
                            .expect("valid streaming fixture");
                        black_box(output);
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sma_one_shot,
    bench_avgprice_one_shot,
    bench_minmax_one_shot,
    bench_minmaxindex_one_shot,
    bench_universe,
    bench_parameter_sweep,
    bench_per_worker_predecessor,
    bench_multi_instrument_streaming,
);
criterion_main!(benches);
