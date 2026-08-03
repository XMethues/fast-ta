//! Indicator execution latency and throughput baselines.
//!
//! Historical issue #2 benchmark IDs remain unchanged for longitudinal
//! comparison. Migration workloads add owned and caller-owned Compact Output,
//! Prepared Batch Runner, repeated-series, and independent Streaming
//! Computation paths through the public traits.

mod support;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use support::{
    ohlc_fixture, output_len, series_fixture, PERIOD, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS,
    SWEEP_PERIODS, UNIVERSE_INSTRUMENTS, WORKERS,
};
use ta_core::{
    math_operators::{
        MAXConfig, MAXINDEXConfig, MINConfig, MININDEXConfig, MINMAXConfig, MINMAXINDEXConfig,
        MINMAXINDEXOutputMut, MINMAXINDEXValuesMut, MINMAXOutputMut, MINMAXValuesMut, MAX,
        MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX,
    },
    overlap::{SMAConfig, SMA},
    price_transform::{AVGPRICEInput, AVGPRICE},
    Float, Indicator, IndicatorConfig, PreparedBatchRunner, StreamingComputation,
    StreamingIndicator,
};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const EXTREMA_MATRIX: &[(usize, usize)] = &[
    (64, 14),
    (4_096, 14),
    (4_096, 512),
    (65_536, 14),
    (65_536, 512),
];

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
            BenchmarkId::new("caller_compact_config", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let config = SMAConfig::new(black_box(PERIOD)).expect("valid period");
                let mut output = vec![0.0 as Float; output_len(size, PERIOD)];

                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid SMA fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("owned_compact_config", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let config = SMAConfig::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
                            black_box(input.as_slice()),
                        )
                        .expect("valid SMA fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
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

#[derive(Clone, Copy)]
enum ExtremaPath {
    Current,
    Config,
    Prepared,
    LegacyOwned,
    ConfigOwned,
}

const EXTREMA_PATHS: [ExtremaPath; 5] = [
    ExtremaPath::Current,
    ExtremaPath::Config,
    ExtremaPath::Prepared,
    ExtremaPath::LegacyOwned,
    ExtremaPath::ConfigOwned,
];

fn rotated_extrema_paths(case_index: usize) -> [ExtremaPath; 5] {
    std::array::from_fn(|offset| EXTREMA_PATHS[(case_index + offset) % EXTREMA_PATHS.len()])
}

fn register_minmax_matrix_path(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    path: ExtremaPath,
    size: usize,
    period: usize,
) {
    let parameter = format!("n={size}/period={period}");
    match path {
        ExtremaPath::Current => group.bench_with_input(
            BenchmarkId::new("current_caller_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let indicator = MINMAX::new(black_box(period)).expect("valid period");
                let count = output_len(size, period);
                let mut min = vec![0.0 as Float; count];
                let mut max = vec![0.0 as Float; count];
                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(input.as_slice()),
                        MINMAXOutputMut {
                            min: black_box(min.as_mut_slice()),
                            max: black_box(max.as_mut_slice()),
                        },
                    )
                    .expect("valid current MINMAX fixture");
                    black_box((range, min.as_slice(), max.as_slice()));
                });
            },
        ),
        ExtremaPath::Config => group.bench_with_input(
            BenchmarkId::new("config_caller_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXConfig::new(black_box(period)).expect("valid period");
                let count = output_len(size, period);
                let mut min = vec![0.0 as Float; count];
                let mut max = vec![0.0 as Float; count];
                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        MINMAXValuesMut {
                            min: black_box(min.as_mut_slice()),
                            max: black_box(max.as_mut_slice()),
                        },
                    )
                    .expect("valid configured MINMAX fixture");
                    black_box((range, min.as_slice(), max.as_slice()));
                });
            },
        ),
        ExtremaPath::Prepared => group.bench_with_input(
            BenchmarkId::new("prepared_runner", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXConfig::new(black_box(period)).expect("valid period");
                let mut runner =
                    IndicatorConfig::prepare_batch(&config, size).expect("valid prepared capacity");
                let count = output_len(size, period);
                let mut min = vec![0.0 as Float; count];
                let mut max = vec![0.0 as Float; count];
                b.iter(|| {
                    let range = PreparedBatchRunner::<MINMAXConfig>::compute_into(
                        black_box(&mut runner),
                        black_box(input.as_slice()),
                        MINMAXValuesMut {
                            min: black_box(min.as_mut_slice()),
                            max: black_box(max.as_mut_slice()),
                        },
                    )
                    .expect("valid prepared MINMAX fixture");
                    black_box((range, min.as_slice(), max.as_slice()));
                });
            },
        ),
        ExtremaPath::LegacyOwned => group.bench_with_input(
            BenchmarkId::new("legacy_owned_aligned", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let indicator = MINMAX::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                        )
                        .expect("valid legacy owned MINMAX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        ),
        ExtremaPath::ConfigOwned => group.bench_with_input(
            BenchmarkId::new("config_owned_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXConfig::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
                            black_box(input.as_slice()),
                        )
                        .expect("valid configured owned MINMAX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        ),
    };
}

// Shared orchestration stops at registration; each typed timed body stays explicit so
// value/index output shapes and direct monomorphized calls remain visible to the optimizer.
macro_rules! define_extrema_matrix_benchmark {
    ($name:ident, $group_name:literal, $register:path) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);

            for (case_index, &(size, period)) in EXTREMA_MATRIX.iter().enumerate() {
                group.throughput(Throughput::Elements(size as u64));
                for path in rotated_extrema_paths(case_index) {
                    $register(&mut group, path, size, period);
                }
            }

            group.finish();
        }
    };
}

define_extrema_matrix_benchmark!(
    bench_minmax_qualified_scratch_matrix,
    "indicator_execution/expanded/extrema/MINMAX",
    register_minmax_matrix_path
);

fn register_minmaxindex_matrix_path(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    path: ExtremaPath,
    size: usize,
    period: usize,
) {
    let parameter = format!("n={size}/period={period}");
    match path {
        ExtremaPath::Current => group.bench_with_input(
            BenchmarkId::new("current_caller_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let indicator = MINMAXINDEX::new(black_box(period)).expect("valid period");
                let count = output_len(size, period);
                let mut min_idx = vec![0_i32; count];
                let mut max_idx = vec![0_i32; count];
                b.iter(|| {
                    let range = Indicator::compute(
                        black_box(&indicator),
                        black_box(input.as_slice()),
                        MINMAXINDEXOutputMut {
                            min_idx: black_box(min_idx.as_mut_slice()),
                            max_idx: black_box(max_idx.as_mut_slice()),
                        },
                    )
                    .expect("valid current MINMAXINDEX fixture");
                    black_box((range, min_idx.as_slice(), max_idx.as_slice()));
                });
            },
        ),
        ExtremaPath::Config => group.bench_with_input(
            BenchmarkId::new("config_caller_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXINDEXConfig::new(black_box(period)).expect("valid period");
                let count = output_len(size, period);
                let mut min_idx = vec![0_usize; count];
                let mut max_idx = vec![0_usize; count];
                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        MINMAXINDEXValuesMut {
                            min_idx: black_box(min_idx.as_mut_slice()),
                            max_idx: black_box(max_idx.as_mut_slice()),
                        },
                    )
                    .expect("valid configured MINMAXINDEX fixture");
                    black_box((range, min_idx.as_slice(), max_idx.as_slice()));
                });
            },
        ),
        ExtremaPath::Prepared => group.bench_with_input(
            BenchmarkId::new("prepared_runner", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXINDEXConfig::new(black_box(period)).expect("valid period");
                let mut runner =
                    IndicatorConfig::prepare_batch(&config, size).expect("valid prepared capacity");
                let count = output_len(size, period);
                let mut min_idx = vec![0_usize; count];
                let mut max_idx = vec![0_usize; count];
                b.iter(|| {
                    let range = PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
                        black_box(&mut runner),
                        black_box(input.as_slice()),
                        MINMAXINDEXValuesMut {
                            min_idx: black_box(min_idx.as_mut_slice()),
                            max_idx: black_box(max_idx.as_mut_slice()),
                        },
                    )
                    .expect("valid prepared MINMAXINDEX fixture");
                    black_box((range, min_idx.as_slice(), max_idx.as_slice()));
                });
            },
        ),
        ExtremaPath::LegacyOwned => group.bench_with_input(
            BenchmarkId::new("legacy_owned_aligned", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let indicator = MINMAXINDEX::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = Indicator::compute_to_vec(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                        )
                        .expect("valid legacy owned MINMAXINDEX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        ),
        ExtremaPath::ConfigOwned => group.bench_with_input(
            BenchmarkId::new("config_owned_compact", &parameter),
            &(size, period),
            |b, &(size, period)| {
                let input = series_fixture(size, 0);
                let config = MINMAXINDEXConfig::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
                            black_box(input.as_slice()),
                        )
                        .expect("valid configured owned MINMAXINDEX fixture");
                        black_box(output)
                    },
                    BatchSize::LargeInput,
                );
            },
        ),
    };
}

define_extrema_matrix_benchmark!(
    bench_minmaxindex_qualified_scratch_matrix,
    "indicator_execution/expanded/extrema/MINMAXINDEX",
    register_minmaxindex_matrix_path
);

fn repeated_series_fixtures(series_count: usize) -> Vec<Vec<Float>> {
    (0..series_count)
        .map(|seed| series_fixture(REPEATED_SERIES_LEN, seed))
        .collect()
}

fn universe_fixtures() -> Vec<Vec<Float>> {
    repeated_series_fixtures(UNIVERSE_INSTRUMENTS)
}

fn worker_fixtures() -> Vec<Vec<Float>> {
    repeated_series_fixtures(WORKERS)
}

fn extrema_sweep_outputs<T: Copy>(initial: T) -> Vec<(Vec<T>, Vec<T>)> {
    SWEEP_PERIODS
        .iter()
        .map(|&period| {
            let count = output_len(REPEATED_SERIES_LEN, period);
            (vec![initial; count], vec![initial; count])
        })
        .collect()
}

fn extrema_worker_outputs<T: Copy>(initial: T) -> Vec<(Vec<T>, Vec<T>)> {
    let count = output_len(REPEATED_SERIES_LEN, PERIOD);
    (0..WORKERS)
        .map(|_| (vec![initial; count], vec![initial; count]))
        .collect()
}

fn stream_inputs() -> Vec<Vec<Float>> {
    let instrument_inputs = repeated_series_fixtures(STREAM_INSTRUMENTS);
    (0..REPEATED_SERIES_LEN)
        .map(|tick_index| {
            instrument_inputs
                .iter()
                .map(|series| series[tick_index])
                .collect()
        })
        .collect()
}

macro_rules! for_each_stream_sample {
    ($streams:expr, $inputs:expr, |$stream:ident, $input:ident| $body:block) => {
        for tick in $inputs {
            for ($stream, &$input) in $streams.iter_mut().zip(tick.iter()) $body
        }
    };
}

fn bench_universe(c: &mut Criterion) {
    let universe = universe_fixtures();
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
    let inputs = worker_fixtures();
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

fn bench_prepared_universe(c: &mut Criterion) {
    let universe = universe_fixtures();
    let config = SMAConfig::new(PERIOD).expect("valid period");
    let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
        .expect("valid prepared capacity");
    let mut output = vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)];
    let mut group = c.benchmark_group("indicator_execution/expanded/repeated");
    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("universe/SMA/prepared_runner", |b| {
        b.iter(|| {
            for series in &universe {
                let range = PreparedBatchRunner::<SMAConfig>::compute_into(
                    black_box(&mut runner),
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

fn bench_prepared_parameter_sweep(c: &mut Criterion) {
    let input = series_fixture(REPEATED_SERIES_LEN, 0);
    let configs = SWEEP_PERIODS
        .iter()
        .map(|&period| SMAConfig::new(period).expect("valid sweep period"))
        .collect::<Vec<_>>();
    let mut runners = configs
        .iter()
        .map(|config| {
            IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                .expect("valid prepared capacity")
        })
        .collect::<Vec<_>>();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
    let mut group = c.benchmark_group("indicator_execution/expanded/repeated");
    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("parameter_sweep/SMA/prepared_runners", |b| {
        b.iter(|| {
            for runner in &mut runners {
                let range = PreparedBatchRunner::<SMAConfig>::compute_into(
                    black_box(runner),
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

fn bench_prepared_per_worker(c: &mut Criterion) {
    let config = SMAConfig::new(PERIOD).expect("valid period");
    let mut runners = (0..WORKERS)
        .map(|_| {
            IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                .expect("valid prepared capacity")
        })
        .collect::<Vec<_>>();
    let inputs = worker_fixtures();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)])
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("indicator_execution/expanded/repeated");
    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));

    group.bench_function("prepared_runner/per_worker", |b| {
        b.iter(|| {
            for ((runner, input), output) in runners
                .iter_mut()
                .zip(inputs.iter())
                .zip(outputs.iter_mut())
            {
                let range = PreparedBatchRunner::<SMAConfig>::compute_into(
                    black_box(runner),
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
    let inputs = stream_inputs();
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
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingIndicator::next(black_box(stream), black_box(input))
                        .expect("valid streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();

    let config = SMAConfig::new(PERIOD).expect("valid period");
    let mut group = c.benchmark_group("indicator_execution/expanded/streaming");
    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("SMA/independent_config_streams", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&config).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<SMAConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
                    .expect("valid streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_minmax_repeated_and_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/expanded/extrema_workloads/MINMAX");

    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("universe/current_caller_compact", |b| {
        let universe = universe_fixtures();
        let indicator = MINMAX::new(PERIOD).expect("valid period");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min = vec![0.0 as Float; count];
        let mut max = vec![0.0 as Float; count];
        b.iter(|| {
            for input in &universe {
                let range = Indicator::compute(
                    black_box(&indicator),
                    black_box(input.as_slice()),
                    MINMAXOutputMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAX Universe fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });
    group.bench_function("universe/config_caller_compact", |b| {
        let universe = universe_fixtures();
        let config = MINMAXConfig::new(PERIOD).expect("valid period");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min = vec![0.0 as Float; count];
        let mut max = vec![0.0 as Float; count];
        b.iter(|| {
            for input in &universe {
                let range = IndicatorConfig::compute_into(
                    black_box(&config),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid configured MINMAX Universe fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });
    group.bench_function("universe/prepared_runner", |b| {
        let universe = universe_fixtures();
        let config = MINMAXConfig::new(PERIOD).expect("valid period");
        let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
            .expect("valid prepared capacity");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min = vec![0.0 as Float; count];
        let mut max = vec![0.0 as Float; count];
        b.iter(|| {
            for input in &universe {
                let range = PreparedBatchRunner::<MINMAXConfig>::compute_into(
                    black_box(&mut runner),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAX Universe fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });

    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("parameter_sweep/current_caller_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let indicators = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAX::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0.0 as Float);
        b.iter(|| {
            for (indicator, (min, max)) in indicators.iter().zip(outputs.iter_mut()) {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    MINMAXOutputMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAX parameter-sweep fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });
    group.bench_function("parameter_sweep/config_caller_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAXConfig::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0.0 as Float);
        b.iter(|| {
            for (config, (min, max)) in configs.iter().zip(outputs.iter_mut()) {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid configured MINMAX parameter-sweep fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });
    group.bench_function("parameter_sweep/prepared_runners", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAXConfig::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut runners = configs
            .iter()
            .map(|config| {
                IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity")
            })
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0.0 as Float);
        b.iter(|| {
            for (runner, (min, max)) in runners.iter_mut().zip(outputs.iter_mut()) {
                let range = PreparedBatchRunner::<MINMAXConfig>::compute_into(
                    black_box(runner),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAX parameter-sweep fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });

    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    group.bench_function("per_worker/current_instances", |b| {
        let indicators = (0..WORKERS)
            .map(|_| MINMAX::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0.0 as Float);
        b.iter(|| {
            for ((indicator, input), (min, max)) in
                indicators.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    MINMAXOutputMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAX per-worker fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });
    group.bench_function("per_worker/prepared_runners", |b| {
        let config = MINMAXConfig::new(PERIOD).expect("valid period");
        let mut runners = (0..WORKERS)
            .map(|_| {
                IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity")
            })
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0.0 as Float);
        b.iter(|| {
            for ((runner, input), (min, max)) in runners
                .iter_mut()
                .zip(inputs.iter())
                .zip(outputs.iter_mut())
            {
                let range = PreparedBatchRunner::<MINMAXConfig>::compute_into(
                    black_box(runner),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
                        min: black_box(min.as_mut_slice()),
                        max: black_box(max.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAX per-worker fixture");
                black_box((range, min.as_slice(), max.as_slice()));
            }
        });
    });

    let inputs = stream_inputs();
    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("streaming/legacy_instances", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| MINMAX::new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingIndicator::next(black_box(stream), black_box(input))
                        .expect("valid legacy MINMAX streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });
    let config = MINMAXConfig::new(PERIOD).expect("valid period");
    group.bench_function("streaming/config_streams", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&config).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<MINMAXConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
                    .expect("valid configured MINMAX streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_minmaxindex_repeated_and_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/expanded/extrema_workloads/MINMAXINDEX");

    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("universe/current_caller_compact", |b| {
        let universe = universe_fixtures();
        let indicator = MINMAXINDEX::new(PERIOD).expect("valid period");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min_idx = vec![0_i32; count];
        let mut max_idx = vec![0_i32; count];
        b.iter(|| {
            for input in &universe {
                let range = Indicator::compute(
                    black_box(&indicator),
                    black_box(input.as_slice()),
                    MINMAXINDEXOutputMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAXINDEX Universe fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });
    group.bench_function("universe/config_caller_compact", |b| {
        let universe = universe_fixtures();
        let config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min_idx = vec![0_usize; count];
        let mut max_idx = vec![0_usize; count];
        b.iter(|| {
            for input in &universe {
                let range = IndicatorConfig::compute_into(
                    black_box(&config),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid configured MINMAXINDEX Universe fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });
    group.bench_function("universe/prepared_runner", |b| {
        let universe = universe_fixtures();
        let config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
        let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
            .expect("valid prepared capacity");
        let count = output_len(REPEATED_SERIES_LEN, PERIOD);
        let mut min_idx = vec![0_usize; count];
        let mut max_idx = vec![0_usize; count];
        b.iter(|| {
            for input in &universe {
                let range = PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
                    black_box(&mut runner),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAXINDEX Universe fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });

    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("parameter_sweep/current_caller_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let indicators = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAXINDEX::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0_i32);
        b.iter(|| {
            for (indicator, (min_idx, max_idx)) in indicators.iter().zip(outputs.iter_mut()) {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    MINMAXINDEXOutputMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAXINDEX parameter-sweep fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });
    group.bench_function("parameter_sweep/config_caller_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAXINDEXConfig::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0_usize);
        b.iter(|| {
            for (config, (min_idx, max_idx)) in configs.iter().zip(outputs.iter_mut()) {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid configured MINMAXINDEX parameter-sweep fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });
    group.bench_function("parameter_sweep/prepared_runners", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| MINMAXINDEXConfig::new(period).expect("valid sweep period"))
            .collect::<Vec<_>>();
        let mut runners = configs
            .iter()
            .map(|config| {
                IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity")
            })
            .collect::<Vec<_>>();
        let mut outputs = extrema_sweep_outputs(0_usize);
        b.iter(|| {
            for (runner, (min_idx, max_idx)) in runners.iter_mut().zip(outputs.iter_mut()) {
                let range = PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
                    black_box(runner),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAXINDEX parameter-sweep fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });

    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    group.bench_function("per_worker/current_instances", |b| {
        let indicators = (0..WORKERS)
            .map(|_| MINMAXINDEX::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0_i32);
        b.iter(|| {
            for ((indicator, input), (min_idx, max_idx)) in
                indicators.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = Indicator::compute(
                    black_box(indicator),
                    black_box(input.as_slice()),
                    MINMAXINDEXOutputMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid current MINMAXINDEX per-worker fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });
    group.bench_function("per_worker/prepared_runners", |b| {
        let config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
        let mut runners = (0..WORKERS)
            .map(|_| {
                IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity")
            })
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0_usize);
        b.iter(|| {
            for ((runner, input), (min_idx, max_idx)) in runners
                .iter_mut()
                .zip(inputs.iter())
                .zip(outputs.iter_mut())
            {
                let range = PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
                    black_box(runner),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
                        min_idx: black_box(min_idx.as_mut_slice()),
                        max_idx: black_box(max_idx.as_mut_slice()),
                    },
                )
                .expect("valid prepared MINMAXINDEX per-worker fixture");
                black_box((range, min_idx.as_slice(), max_idx.as_slice()));
            }
        });
    });

    let inputs = stream_inputs();
    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("streaming/legacy_instances", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| MINMAXINDEX::new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingIndicator::next(black_box(stream), black_box(input))
                        .expect("valid legacy MINMAXINDEX streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });
    let config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
    group.bench_function("streaming/config_streams", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&config).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<MINMAXINDEXConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
                    .expect("valid configured MINMAXINDEX streaming fixture");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn single_extrema_sweep_outputs<T: Copy>(initial: T) -> Vec<Vec<T>> {
    SWEEP_PERIODS
        .iter()
        .map(|&period| vec![initial; output_len(REPEATED_SERIES_LEN, period)])
        .collect()
}

fn single_extrema_worker_outputs<T: Copy>(initial: T) -> Vec<Vec<T>> {
    (0..WORKERS)
        .map(|_| vec![initial; output_len(REPEATED_SERIES_LEN, PERIOD)])
        .collect()
}

macro_rules! define_single_extrema_workloads {
    (
        $name:ident,
        $group_name:literal,
        $indicator:ident,
        $config:ident,
        $current_type:ty,
        $current_initial:expr,
        $config_type:ty,
        $config_initial:expr
    ) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);

            group.throughput(Throughput::Elements(
                (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("universe/current_caller_compact", |b| {
                let universe = universe_fixtures();
                let indicator = $indicator::new(PERIOD).expect("valid period");
                let mut output: Vec<$current_type> =
                    vec![$current_initial; output_len(REPEATED_SERIES_LEN, PERIOD)];
                b.iter(|| {
                    for input in &universe {
                        let range = Indicator::compute(
                            black_box(&indicator),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid current ",
                            stringify!($indicator),
                            " Universe"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/config_caller_compact", |b| {
                let universe = universe_fixtures();
                let config = $config::new(PERIOD).expect("valid period");
                let mut output: Vec<$config_type> =
                    vec![$config_initial; output_len(REPEATED_SERIES_LEN, PERIOD)];
                b.iter(|| {
                    for input in &universe {
                        let range = IndicatorConfig::compute_into(
                            black_box(&config),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid configured ",
                            stringify!($indicator),
                            " Universe"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/prepared_runner", |b| {
                let universe = universe_fixtures();
                let config = $config::new(PERIOD).expect("valid period");
                let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity");
                let mut output: Vec<$config_type> =
                    vec![$config_initial; output_len(REPEATED_SERIES_LEN, PERIOD)];
                b.iter(|| {
                    for input in &universe {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(&mut runner),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid prepared ",
                            stringify!($indicator),
                            " Universe"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements(
                (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("parameter_sweep/current_caller_compact", |b| {
                let input = series_fixture(REPEATED_SERIES_LEN, 0);
                let indicators = SWEEP_PERIODS
                    .iter()
                    .map(|&period| $indicator::new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut outputs = single_extrema_sweep_outputs::<$current_type>($current_initial);
                b.iter(|| {
                    for (indicator, output) in indicators.iter().zip(outputs.iter_mut()) {
                        let range = Indicator::compute(
                            black_box(indicator),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid current ",
                            stringify!($indicator),
                            " parameter sweep"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("parameter_sweep/config_caller_compact", |b| {
                let input = series_fixture(REPEATED_SERIES_LEN, 0);
                let configs = SWEEP_PERIODS
                    .iter()
                    .map(|&period| $config::new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut outputs = single_extrema_sweep_outputs::<$config_type>($config_initial);
                b.iter(|| {
                    for (config, output) in configs.iter().zip(outputs.iter_mut()) {
                        let range = IndicatorConfig::compute_into(
                            black_box(config),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid configured ",
                            stringify!($indicator),
                            " parameter sweep"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("parameter_sweep/prepared_runners", |b| {
                let input = series_fixture(REPEATED_SERIES_LEN, 0);
                let configs = SWEEP_PERIODS
                    .iter()
                    .map(|&period| $config::new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut runners = configs
                    .iter()
                    .map(|config| {
                        IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                            .expect("valid prepared capacity")
                    })
                    .collect::<Vec<_>>();
                let mut outputs = single_extrema_sweep_outputs::<$config_type>($config_initial);
                b.iter(|| {
                    for (runner, output) in runners.iter_mut().zip(outputs.iter_mut()) {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(runner),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid prepared ",
                            stringify!($indicator),
                            " parameter sweep"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
            group.bench_function("per_worker/current_instances", |b| {
                let indicators = (0..WORKERS)
                    .map(|_| $indicator::new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>();
                let inputs = worker_fixtures();
                let mut outputs = single_extrema_worker_outputs::<$current_type>($current_initial);
                b.iter(|| {
                    for ((indicator, input), output) in
                        indicators.iter().zip(inputs.iter()).zip(outputs.iter_mut())
                    {
                        let range = Indicator::compute(
                            black_box(indicator),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid current ",
                            stringify!($indicator),
                            " per worker"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("per_worker/prepared_runners", |b| {
                let config = $config::new(PERIOD).expect("valid period");
                let mut runners = (0..WORKERS)
                    .map(|_| {
                        IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                            .expect("valid prepared capacity")
                    })
                    .collect::<Vec<_>>();
                let inputs = worker_fixtures();
                let mut outputs = single_extrema_worker_outputs::<$config_type>($config_initial);
                b.iter(|| {
                    for ((runner, input), output) in runners
                        .iter_mut()
                        .zip(inputs.iter())
                        .zip(outputs.iter_mut())
                    {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(runner),
                            black_box(input.as_slice()),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!(
                            "valid prepared ",
                            stringify!($indicator),
                            " per worker"
                        ));
                        black_box((range, output.as_slice()));
                    }
                });
            });

            let inputs = stream_inputs();
            group.throughput(Throughput::Elements(
                (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("streaming/legacy_instances", |b| {
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| $indicator::new(PERIOD).expect("valid period"))
                            .collect::<Vec<_>>()
                    },
                    |streams| {
                        for_each_stream_sample!(streams, &inputs, |stream, input| {
                            let output =
                                StreamingIndicator::next(black_box(stream), black_box(input))
                                    .expect(concat!(
                                        "valid legacy ",
                                        stringify!($indicator),
                                        " stream"
                                    ));
                            black_box(output);
                        });
                    },
                    BatchSize::LargeInput,
                );
            });
            let config = $config::new(PERIOD).expect("valid period");
            group.bench_function("streaming/config_streams", |b| {
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| IndicatorConfig::stream(&config).expect("valid period"))
                            .collect::<Vec<_>>()
                    },
                    |streams| {
                        for_each_stream_sample!(streams, &inputs, |stream, input| {
                            let output = StreamingComputation::<$config>::next(
                                black_box(stream),
                                black_box(input),
                            )
                            .expect(concat!(
                                "valid configured ",
                                stringify!($indicator),
                                " stream"
                            ));
                            black_box(output);
                        });
                    },
                    BatchSize::LargeInput,
                );
            });

            group.finish();
        }
    };
}

define_single_extrema_workloads!(
    bench_min_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MIN",
    MIN,
    MINConfig,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_extrema_workloads!(
    bench_max_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MAX",
    MAX,
    MAXConfig,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_extrema_workloads!(
    bench_minindex_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MININDEX",
    MININDEX,
    MININDEXConfig,
    i32,
    0_i32,
    usize,
    0_usize
);
define_single_extrema_workloads!(
    bench_maxindex_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    i32,
    0_i32,
    usize,
    0_usize
);

macro_rules! define_single_extrema_benchmark {
    (
        $name:ident,
        $group_name:literal,
        $indicator:ident,
        $config:ident,
        $current_type:ty,
        $current_initial:expr,
        $config_type:ty,
        $config_initial:expr
    ) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            for (case_index, &(size, period)) in EXTREMA_MATRIX.iter().enumerate() {
                group.throughput(Throughput::Elements(size as u64));
                let parameter = format!("n={size}/period={period}");
                for path in rotated_extrema_paths(case_index) {
                    match path {
                        ExtremaPath::Current => group.bench_with_input(
                            BenchmarkId::new("current_caller_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let indicator =
                                    $indicator::new(black_box(period)).expect("valid period");
                                let mut output: Vec<$current_type> =
                                    vec![$current_initial; output_len(size, period)];
                                b.iter(|| {
                                    let range = Indicator::compute(
                                        black_box(&indicator),
                                        black_box(input.as_slice()),
                                        black_box(output.as_mut_slice()),
                                    )
                                    .expect(concat!("valid current ", stringify!($indicator)));
                                    black_box((range, output.as_slice()));
                                });
                            },
                        ),
                        ExtremaPath::Config => group.bench_with_input(
                            BenchmarkId::new("config_caller_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config::new(black_box(period)).expect("valid period");
                                let mut output: Vec<$config_type> =
                                    vec![$config_initial; output_len(size, period)];
                                b.iter(|| {
                                    let range = IndicatorConfig::compute_into(
                                        black_box(&config),
                                        black_box(input.as_slice()),
                                        black_box(output.as_mut_slice()),
                                    )
                                    .expect(concat!("valid configured ", stringify!($indicator)));
                                    black_box((range, output.as_slice()));
                                });
                            },
                        ),
                        ExtremaPath::Prepared => group.bench_with_input(
                            BenchmarkId::new("prepared_runner", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config::new(black_box(period)).expect("valid period");
                                let mut runner = IndicatorConfig::prepare_batch(&config, size)
                                    .expect("valid prepared capacity");
                                let mut output: Vec<$config_type> =
                                    vec![$config_initial; output_len(size, period)];
                                b.iter(|| {
                                    let range = PreparedBatchRunner::<$config>::compute_into(
                                        black_box(&mut runner),
                                        black_box(input.as_slice()),
                                        black_box(output.as_mut_slice()),
                                    )
                                    .expect(concat!("valid prepared ", stringify!($indicator)));
                                    black_box((range, output.as_slice()));
                                });
                            },
                        ),
                        ExtremaPath::LegacyOwned => group.bench_with_input(
                            BenchmarkId::new("legacy_owned_aligned", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let indicator =
                                    $indicator::new(black_box(period)).expect("valid period");
                                b.iter_batched(
                                    || (),
                                    |_| {
                                        let output = Indicator::compute_to_vec(
                                            black_box(&indicator),
                                            black_box(input.as_slice()),
                                        )
                                        .expect(concat!(
                                            "valid legacy owned ",
                                            stringify!($indicator)
                                        ));
                                        black_box(output)
                                    },
                                    BatchSize::LargeInput,
                                );
                            },
                        ),
                        ExtremaPath::ConfigOwned => group.bench_with_input(
                            BenchmarkId::new("config_owned_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config::new(black_box(period)).expect("valid period");
                                b.iter_batched(
                                    || (),
                                    |_| {
                                        let output = IndicatorConfig::compute(
                                            black_box(&config),
                                            black_box(input.as_slice()),
                                        )
                                        .expect(concat!(
                                            "valid configured owned ",
                                            stringify!($indicator)
                                        ));
                                        black_box(output)
                                    },
                                    BatchSize::LargeInput,
                                );
                            },
                        ),
                    };
                }
            }
            group.finish();
        }
    };
}

define_single_extrema_benchmark!(
    bench_min_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MIN",
    MIN,
    MINConfig,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_extrema_benchmark!(
    bench_max_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MAX",
    MAX,
    MAXConfig,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_extrema_benchmark!(
    bench_minindex_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MININDEX",
    MININDEX,
    MININDEXConfig,
    i32,
    0_i32,
    usize,
    0_usize
);
define_single_extrema_benchmark!(
    bench_maxindex_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    i32,
    0_i32,
    usize,
    0_usize
);

criterion_group!(
    benches,
    bench_sma_one_shot,
    bench_avgprice_one_shot,
    bench_minmax_one_shot,
    bench_minmaxindex_one_shot,
    bench_minmax_qualified_scratch_matrix,
    bench_minmaxindex_qualified_scratch_matrix,
    bench_min_qualified_scratch_matrix,
    bench_max_qualified_scratch_matrix,
    bench_minindex_qualified_scratch_matrix,
    bench_maxindex_qualified_scratch_matrix,
    bench_min_repeated_and_streaming,
    bench_max_repeated_and_streaming,
    bench_minindex_repeated_and_streaming,
    bench_maxindex_repeated_and_streaming,
    bench_minmax_repeated_and_streaming,
    bench_minmaxindex_repeated_and_streaming,
    bench_universe,
    bench_parameter_sweep,
    bench_per_worker_predecessor,
    bench_prepared_universe,
    bench_prepared_parameter_sweep,
    bench_prepared_per_worker,
    bench_multi_instrument_streaming,
);
criterion_main!(benches);
