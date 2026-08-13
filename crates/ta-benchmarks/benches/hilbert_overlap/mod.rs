use super::support::{
    series_fixture, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS, UNIVERSE_INSTRUMENTS, WORKERS,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use fast_ta::overlap::{HT_TRENDLINEConfig, MAMAConfig, MAMAValuesMut};
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const LIMIT_SWEEP: &[(Float, Float)] = &[
    (0.99 as Float, 0.5 as Float),
    (0.5 as Float, 0.05 as Float),
    (0.1 as Float, 0.01 as Float),
    (0.01 as Float, 0.01 as Float),
];

fn fixtures(count: usize, size: usize) -> Vec<Vec<Float>> {
    (0..count)
        .map(|instrument| series_fixture(size, instrument))
        .collect()
}

pub(crate) fn bench_hilbert_overlap_execution(c: &mut Criterion) {
    let mut one_shot = c.benchmark_group("indicator_execution/expanded/hilbert_overlap");
    for &size in SIZES {
        let input = series_fixture(size, 29);
        one_shot.throughput(Throughput::Elements(size as u64));

        let mama_config = MAMAConfig::default();
        let mama_count = size - mama_config.lookback();
        let mut mama = vec![0.0 as Float; mama_count];
        let mut fama = vec![0.0 as Float; mama_count];
        one_shot.bench_with_input(
            BenchmarkId::new("MAMA/caller_owned", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = mama_config
                        .compute_into(
                            black_box(input.as_slice()),
                            MAMAValuesMut {
                                mama: black_box(mama.as_mut_slice()),
                                fama: black_box(fama.as_mut_slice()),
                            },
                        )
                        .expect("valid MAMA fixture");
                    black_box((range, mama.as_slice(), fama.as_slice()));
                });
            },
        );
        one_shot.bench_with_input(
            BenchmarkId::new("MAMA/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| {
                        black_box(
                            mama_config
                                .compute(black_box(input.as_slice()))
                                .expect("valid MAMA fixture"),
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        let mut mama_runner = mama_config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("MAMA/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = mama_runner
                        .compute_into(
                            black_box(input.as_slice()),
                            MAMAValuesMut {
                                mama: black_box(mama.as_mut_slice()),
                                fama: black_box(fama.as_mut_slice()),
                            },
                        )
                        .expect("valid prepared MAMA fixture");
                    black_box((range, mama.as_slice(), fama.as_slice()));
                });
            },
        );

        let trendline_config = HT_TRENDLINEConfig::new();
        let mut trendline = vec![0.0 as Float; size - trendline_config.lookback()];
        one_shot.bench_with_input(
            BenchmarkId::new("HT_TRENDLINE/caller_owned", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = trendline_config
                        .compute_into(
                            black_box(input.as_slice()),
                            black_box(trendline.as_mut_slice()),
                        )
                        .expect("valid HT_TRENDLINE fixture");
                    black_box((range, trendline.as_slice()));
                });
            },
        );
        one_shot.bench_with_input(
            BenchmarkId::new("HT_TRENDLINE/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| {
                        black_box(
                            trendline_config
                                .compute(black_box(input.as_slice()))
                                .expect("valid HT_TRENDLINE fixture"),
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        let mut trendline_runner = trendline_config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("HT_TRENDLINE/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = trendline_runner
                        .compute_into(
                            black_box(input.as_slice()),
                            black_box(trendline.as_mut_slice()),
                        )
                        .expect("valid prepared HT_TRENDLINE fixture");
                    black_box((range, trendline.as_slice()));
                });
            },
        );
    }
    one_shot.finish();

    let universe = fixtures(UNIVERSE_INSTRUMENTS, REPEATED_SERIES_LEN);
    let mut workloads = c.benchmark_group("indicator_execution/expanded/hilbert_overlap_workloads");
    workloads.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    let mama_config = MAMAConfig::default();
    let mut mama = vec![0.0 as Float; REPEATED_SERIES_LEN - mama_config.lookback()];
    let mut fama = vec![0.0 as Float; REPEATED_SERIES_LEN - mama_config.lookback()];
    workloads.bench_function("MAMA/universe/caller_owned", |b| {
        b.iter(|| {
            for input in &universe {
                let range = mama_config
                    .compute_into(
                        black_box(input.as_slice()),
                        MAMAValuesMut {
                            mama: black_box(mama.as_mut_slice()),
                            fama: black_box(fama.as_mut_slice()),
                        },
                    )
                    .unwrap();
                black_box((range, mama.as_slice(), fama.as_slice()));
            }
        });
    });
    let mut mama_runner = mama_config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    workloads.bench_function("MAMA/universe/prepared_runner", |b| {
        b.iter(|| {
            for input in &universe {
                let range = mama_runner
                    .compute_into(
                        black_box(input.as_slice()),
                        MAMAValuesMut {
                            mama: black_box(mama.as_mut_slice()),
                            fama: black_box(fama.as_mut_slice()),
                        },
                    )
                    .unwrap();
                black_box((range, mama.as_slice(), fama.as_slice()));
            }
        });
    });

    let trendline_config = HT_TRENDLINEConfig::new();
    let mut trendline = vec![0.0 as Float; REPEATED_SERIES_LEN - trendline_config.lookback()];
    workloads.bench_function("HT_TRENDLINE/universe/caller_owned", |b| {
        b.iter(|| {
            for input in &universe {
                let range = trendline_config
                    .compute_into(
                        black_box(input.as_slice()),
                        black_box(trendline.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, trendline.as_slice()));
            }
        });
    });
    let mut trendline_runner = trendline_config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    workloads.bench_function("HT_TRENDLINE/universe/prepared_runner", |b| {
        b.iter(|| {
            for input in &universe {
                let range = trendline_runner
                    .compute_into(
                        black_box(input.as_slice()),
                        black_box(trendline.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, trendline.as_slice()));
            }
        });
    });

    workloads.throughput(Throughput::Elements(
        (LIMIT_SWEEP.len() * REPEATED_SERIES_LEN) as u64,
    ));
    let limit_configs = LIMIT_SWEEP
        .iter()
        .map(|&(fast, slow)| MAMAConfig::new(fast, slow).unwrap())
        .collect::<Vec<_>>();
    workloads.bench_function("MAMA/limit_sweep/caller_owned", |b| {
        b.iter(|| {
            for config in &limit_configs {
                let range = config
                    .compute_into(
                        black_box(universe[0].as_slice()),
                        MAMAValuesMut {
                            mama: black_box(mama.as_mut_slice()),
                            fama: black_box(fama.as_mut_slice()),
                        },
                    )
                    .unwrap();
                black_box((range, mama.as_slice(), fama.as_slice()));
            }
        });
    });
    let mut limit_runners = limit_configs
        .iter()
        .map(|config| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    workloads.bench_function("MAMA/limit_sweep/prepared_runner", |b| {
        b.iter(|| {
            for runner in &mut limit_runners {
                let range = runner
                    .compute_into(
                        black_box(universe[0].as_slice()),
                        MAMAValuesMut {
                            mama: black_box(mama.as_mut_slice()),
                            fama: black_box(fama.as_mut_slice()),
                        },
                    )
                    .unwrap();
                black_box((range, mama.as_slice(), fama.as_slice()));
            }
        });
    });

    let worker_inputs = fixtures(WORKERS, REPEATED_SERIES_LEN);
    let mut mama_workers = (0..WORKERS)
        .map(|_| mama_config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut mama_worker_outputs = (0..WORKERS)
        .map(|_| {
            (
                vec![0.0 as Float; REPEATED_SERIES_LEN - mama_config.lookback()],
                vec![0.0 as Float; REPEATED_SERIES_LEN - mama_config.lookback()],
            )
        })
        .collect::<Vec<_>>();
    workloads.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    workloads.bench_function("MAMA/per_worker/prepared_runner", |b| {
        b.iter(|| {
            for ((runner, input), (mama, fama)) in mama_workers
                .iter_mut()
                .zip(&worker_inputs)
                .zip(&mut mama_worker_outputs)
            {
                let range = runner
                    .compute_into(
                        black_box(input.as_slice()),
                        MAMAValuesMut {
                            mama: black_box(mama.as_mut_slice()),
                            fama: black_box(fama.as_mut_slice()),
                        },
                    )
                    .unwrap();
                black_box((range, mama.as_slice(), fama.as_slice()));
            }
        });
    });

    let mut trendline_workers = (0..WORKERS)
        .map(|_| trendline_config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut trendline_worker_outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN - trendline_config.lookback()])
        .collect::<Vec<_>>();
    workloads.bench_function("HT_TRENDLINE/per_worker/prepared_runner", |b| {
        b.iter(|| {
            for ((runner, input), output) in trendline_workers
                .iter_mut()
                .zip(&worker_inputs)
                .zip(&mut trendline_worker_outputs)
            {
                let range = runner
                    .compute_into(
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });

    let stream_inputs = fixtures(STREAM_INSTRUMENTS, REPEATED_SERIES_LEN);
    workloads.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("MAMA/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| mama_config.stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, input) in streams.iter_mut().zip(&stream_inputs) {
                        black_box(stream.next(black_box(input[index])).unwrap());
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    workloads.bench_function("HT_TRENDLINE/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| trendline_config.stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, input) in streams.iter_mut().zip(&stream_inputs) {
                        black_box(stream.next(black_box(input[index])).unwrap());
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    workloads.finish();
}
