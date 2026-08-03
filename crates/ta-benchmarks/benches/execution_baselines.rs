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
    overlap::{
        DEMAConfig, EMAConfig, MAConfig, MAType, SMAConfig, T3Config, TEMAConfig, TRIMAConfig,
        WMAConfig, DEMA, EMA, MA, SMA, T3, TEMA, TRIMA, WMA,
    },
    price_transform::{
        AVGDEVConfig, AVGPRICEConfig, AVGPRICEInput, AVGPRICETick, MEDPRICEConfig, MEDPRICEInput,
        MEDPRICETick, TYPPRICEConfig, TYPPRICEInput, TYPPRICETick, WCLPRICEConfig, WCLPRICEInput,
        WCLPRICETick, AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE,
    },
    Float, Indicator, IndicatorConfig, PreparedBatchRunner, StreamingComputation,
    StreamingIndicator,
};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const EXECUTION_MATRIX: &[(usize, usize)] = &[
    (64, 14),
    (4_096, 14),
    (4_096, 512),
    (65_536, 14),
    (65_536, 512),
];

fn ma_ema_indicator(period: usize) -> ta_core::Result<MA> {
    MA::new(period, MAType::EMA)
}

fn ma_ema_config(period: usize) -> ta_core::Result<MAConfig> {
    MAConfig::new(period, MAType::EMA)
}

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
enum ExecutionPath {
    Current,
    Config,
    Prepared,
    LegacyOwned,
    ConfigOwned,
}

const EXECUTION_PATHS: [ExecutionPath; 5] = [
    ExecutionPath::Current,
    ExecutionPath::Config,
    ExecutionPath::Prepared,
    ExecutionPath::LegacyOwned,
    ExecutionPath::ConfigOwned,
];

fn rotated_execution_paths(case_index: usize) -> [ExecutionPath; 5] {
    std::array::from_fn(|offset| EXECUTION_PATHS[(case_index + offset) % EXECUTION_PATHS.len()])
}

fn register_minmax_matrix_path(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    path: ExecutionPath,
    size: usize,
    period: usize,
) {
    let parameter = format!("n={size}/period={period}");
    match path {
        ExecutionPath::Current => group.bench_with_input(
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
        ExecutionPath::Config => group.bench_with_input(
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
        ExecutionPath::Prepared => group.bench_with_input(
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
        ExecutionPath::LegacyOwned => group.bench_with_input(
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
        ExecutionPath::ConfigOwned => group.bench_with_input(
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

            for (case_index, &(size, period)) in EXECUTION_MATRIX.iter().enumerate() {
                group.throughput(Throughput::Elements(size as u64));
                for path in rotated_execution_paths(case_index) {
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
    path: ExecutionPath,
    size: usize,
    period: usize,
) {
    let parameter = format!("n={size}/period={period}");
    match path {
        ExecutionPath::Current => group.bench_with_input(
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
        ExecutionPath::Config => group.bench_with_input(
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
        ExecutionPath::Prepared => group.bench_with_input(
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
        ExecutionPath::LegacyOwned => group.bench_with_input(
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
        ExecutionPath::ConfigOwned => group.bench_with_input(
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

fn single_output_len(input_len: usize, period: usize, lookback_multiplier: usize) -> usize {
    input_len.saturating_sub((period - 1).saturating_mul(lookback_multiplier))
}

fn single_output_sweep_outputs<T: Copy>(initial: T, lookback_multiplier: usize) -> Vec<Vec<T>> {
    SWEEP_PERIODS
        .iter()
        .map(|&period| {
            vec![initial; single_output_len(REPEATED_SERIES_LEN, period, lookback_multiplier)]
        })
        .collect()
}

fn single_output_worker_outputs<T: Copy>(initial: T, lookback_multiplier: usize) -> Vec<Vec<T>> {
    (0..WORKERS)
        .map(|_| vec![initial; single_output_len(REPEATED_SERIES_LEN, PERIOD, lookback_multiplier)])
        .collect()
}

macro_rules! define_single_output_workloads {
    (
        $name:ident,
        $group_name:literal,
        $indicator:ident,
        $config:ident,
        $current_new:path,
        $config_new:path,
        $lookback_multiplier:expr,
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
                let indicator = $current_new(PERIOD).expect("valid period");
                let mut output: Vec<$current_type> =
                    vec![
                        $current_initial;
                        single_output_len(REPEATED_SERIES_LEN, PERIOD, $lookback_multiplier,)
                    ];
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
                let config = $config_new(PERIOD).expect("valid period");
                let mut output: Vec<$config_type> =
                    vec![
                        $config_initial;
                        single_output_len(REPEATED_SERIES_LEN, PERIOD, $lookback_multiplier,)
                    ];
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
                let config = $config_new(PERIOD).expect("valid period");
                let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid prepared capacity");
                let mut output: Vec<$config_type> =
                    vec![
                        $config_initial;
                        single_output_len(REPEATED_SERIES_LEN, PERIOD, $lookback_multiplier,)
                    ];
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
                    .map(|&period| $current_new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut outputs = single_output_sweep_outputs::<$current_type>(
                    $current_initial,
                    $lookback_multiplier,
                );
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
                    .map(|&period| $config_new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut outputs = single_output_sweep_outputs::<$config_type>(
                    $config_initial,
                    $lookback_multiplier,
                );
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
                    .map(|&period| $config_new(period).expect("valid sweep period"))
                    .collect::<Vec<_>>();
                let mut runners = configs
                    .iter()
                    .map(|config| {
                        IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                            .expect("valid prepared capacity")
                    })
                    .collect::<Vec<_>>();
                let mut outputs = single_output_sweep_outputs::<$config_type>(
                    $config_initial,
                    $lookback_multiplier,
                );
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
                    .map(|_| $current_new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>();
                let inputs = worker_fixtures();
                let mut outputs = single_output_worker_outputs::<$current_type>(
                    $current_initial,
                    $lookback_multiplier,
                );
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
                let config = $config_new(PERIOD).expect("valid period");
                let mut runners = (0..WORKERS)
                    .map(|_| {
                        IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                            .expect("valid prepared capacity")
                    })
                    .collect::<Vec<_>>();
                let inputs = worker_fixtures();
                let mut outputs = single_output_worker_outputs::<$config_type>(
                    $config_initial,
                    $lookback_multiplier,
                );
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
                            .map(|_| $current_new(PERIOD).expect("valid period"))
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
            let config = $config_new(PERIOD).expect("valid period");
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

define_single_output_workloads!(
    bench_min_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MIN",
    MIN,
    MINConfig,
    MIN::new,
    MINConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_max_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MAX",
    MAX,
    MAXConfig,
    MAX::new,
    MAXConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_minindex_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MININDEX",
    MININDEX,
    MININDEXConfig,
    MININDEX::new,
    MININDEXConfig::new,
    1,
    i32,
    0_i32,
    usize,
    0_usize
);
define_single_output_workloads!(
    bench_maxindex_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    MAXINDEX::new,
    MAXINDEXConfig::new,
    1,
    i32,
    0_i32,
    usize,
    0_usize
);

macro_rules! define_single_output_benchmark {
    (
        $name:ident,
        $group_name:literal,
        $indicator:ident,
        $config:ident,
        $current_new:path,
        $config_new:path,
        $lookback_multiplier:expr,
        $current_type:ty,
        $current_initial:expr,
        $config_type:ty,
        $config_initial:expr
    ) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            for (case_index, &(size, period)) in EXECUTION_MATRIX.iter().enumerate() {
                if single_output_len(size, period, $lookback_multiplier) == 0 {
                    continue;
                }
                group.throughput(Throughput::Elements(size as u64));
                let parameter = format!("n={size}/period={period}");
                for path in rotated_execution_paths(case_index) {
                    match path {
                        ExecutionPath::Current => group.bench_with_input(
                            BenchmarkId::new("current_caller_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let indicator =
                                    $current_new(black_box(period)).expect("valid period");
                                let mut output: Vec<$current_type> =
                                    vec![
                                        $current_initial;
                                        single_output_len(size, period, $lookback_multiplier)
                                    ];
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
                        ExecutionPath::Config => group.bench_with_input(
                            BenchmarkId::new("config_caller_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config_new(black_box(period)).expect("valid period");
                                let mut output: Vec<$config_type> =
                                    vec![
                                        $config_initial;
                                        single_output_len(size, period, $lookback_multiplier)
                                    ];
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
                        ExecutionPath::Prepared => group.bench_with_input(
                            BenchmarkId::new("prepared_runner", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config_new(black_box(period)).expect("valid period");
                                let mut runner = IndicatorConfig::prepare_batch(&config, size)
                                    .expect("valid prepared capacity");
                                let mut output: Vec<$config_type> =
                                    vec![
                                        $config_initial;
                                        single_output_len(size, period, $lookback_multiplier)
                                    ];
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
                        ExecutionPath::LegacyOwned => group.bench_with_input(
                            BenchmarkId::new("legacy_owned_aligned", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let indicator =
                                    $current_new(black_box(period)).expect("valid period");
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
                        ExecutionPath::ConfigOwned => group.bench_with_input(
                            BenchmarkId::new("config_owned_compact", &parameter),
                            &(size, period),
                            |b, &(size, period)| {
                                let input = series_fixture(size, 0);
                                let config = $config_new(black_box(period)).expect("valid period");
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
macro_rules! define_named_price_benchmarks {
    (
        $matrix_name:ident,
        $workloads_name:ident,
        $matrix_group:literal,
        $workloads_group:literal,
        $indicator:ident,
        $config:ident,
        $input:ident,
        $tick:ident,
        [$($field:ident),+ $(,)?]
    ) => {
        fn $matrix_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($matrix_group);
            for &size in SIZES {
                group.throughput(Throughput::Elements(size as u64));

                group.bench_with_input(
                    BenchmarkId::new("current_caller_compact", size),
                    &size,
                    |b, &size| {
                        let ohlc = ohlc_fixture(size);
                        let indicator = $indicator::new().expect("valid price configuration");
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = Indicator::compute(
                                black_box(&indicator),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid current price fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("config_caller_compact", size),
                    &size,
                    |b, &size| {
                        let ohlc = ohlc_fixture(size);
                        let config = $config::new();
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = IndicatorConfig::compute_into(
                                black_box(&config),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid configured price fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("prepared_runner", size),
                    &size,
                    |b, &size| {
                        let ohlc = ohlc_fixture(size);
                        let config = $config::new();
                        let mut runner = IndicatorConfig::prepare_batch(&config, size)
                            .expect("valid prepared price capacity");
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = PreparedBatchRunner::<$config>::compute_into(
                                black_box(&mut runner),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid prepared price fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("legacy_owned_aligned", size),
                    &size,
                    |b, &size| {
                        let ohlc = ohlc_fixture(size);
                        let indicator = $indicator::new().expect("valid price configuration");
                        b.iter_batched(
                            || (),
                            |_| {
                                let output = Indicator::compute_to_vec(
                                    black_box(&indicator),
                                    black_box($input {
                                        $($field: ohlc.$field.as_slice()),+
                                    }),
                                )
                                .expect("valid legacy owned price fixture");
                                black_box(output)
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("config_owned_compact", size),
                    &size,
                    |b, &size| {
                        let ohlc = ohlc_fixture(size);
                        let config = $config::new();
                        b.iter_batched(
                            || (),
                            |_| {
                                let output = IndicatorConfig::compute(
                                    black_box(&config),
                                    black_box($input {
                                        $($field: ohlc.$field.as_slice()),+
                                    }),
                                )
                                .expect("valid configured owned price fixture");
                                black_box(output)
                            },
                            BatchSize::LargeInput,
                        );
                    },
                );
            }
            group.finish();
        }

        fn $workloads_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($workloads_group);

            group.throughput(Throughput::Elements(
                (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("universe/current_caller_compact", |b| {
                let universe = (0..UNIVERSE_INSTRUMENTS)
                    .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let indicator = $indicator::new().expect("valid price configuration");
                let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
                b.iter(|| {
                    for ohlc in &universe {
                        let range = Indicator::compute(
                            black_box(&indicator),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid current price Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/config_caller_compact", |b| {
                let universe = (0..UNIVERSE_INSTRUMENTS)
                    .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = $config::new();
                let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
                b.iter(|| {
                    for ohlc in &universe {
                        let range = IndicatorConfig::compute_into(
                            black_box(&config),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid configured price Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/prepared_runner", |b| {
                let universe = (0..UNIVERSE_INSTRUMENTS)
                    .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = $config::new();
                let mut runner =
                    IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN).unwrap();
                let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
                b.iter(|| {
                    for ohlc in &universe {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(&mut runner),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid prepared price Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements(
                (WORKERS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("per_worker/current_instances", |b| {
                let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
                let indicators = (0..WORKERS)
                    .map(|_| $indicator::new().expect("valid price configuration"))
                    .collect::<Vec<_>>();
                let mut outputs =
                    vec![vec![0.0 as Float; REPEATED_SERIES_LEN]; WORKERS];
                b.iter(|| {
                    for (indicator, output) in indicators.iter().zip(outputs.iter_mut()) {
                        let range = Indicator::compute(
                            black_box(indicator),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid current per-worker price fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("per_worker/prepared_runners", |b| {
                let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
                let config = $config::new();
                let mut runners = (0..WORKERS)
                    .map(|_| IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN).unwrap())
                    .collect::<Vec<_>>();
                let mut outputs =
                    vec![vec![0.0 as Float; REPEATED_SERIES_LEN]; WORKERS];
                b.iter(|| {
                    for (runner, output) in runners.iter_mut().zip(outputs.iter_mut()) {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(runner),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid prepared per-worker price fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements(
                (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("streaming/legacy_instances", |b| {
                let inputs = (0..STREAM_INSTRUMENTS)
                    .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| $indicator::new().expect("valid price configuration"))
                            .collect::<Vec<_>>()
                    },
                    |streams| {
                        for idx in 0..REPEATED_SERIES_LEN {
                            for (stream, ohlc) in streams.iter_mut().zip(inputs.iter()) {
                                let output = StreamingIndicator::next(
                                    black_box(stream),
                                    black_box($tick {
                                        $($field: ohlc.$field[idx]),+
                                    }),
                                )
                                .expect("valid legacy price stream");
                                black_box(output);
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
            group.bench_function("streaming/config_streams", |b| {
                let inputs = (0..STREAM_INSTRUMENTS)
                    .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = $config::new();
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| IndicatorConfig::stream(&config).unwrap())
                            .collect::<Vec<_>>()
                    },
                    |streams| {
                        for idx in 0..REPEATED_SERIES_LEN {
                            for (stream, ohlc) in streams.iter_mut().zip(inputs.iter()) {
                                let output = StreamingComputation::<$config>::next(
                                    black_box(stream),
                                    black_box($tick {
                                        $($field: ohlc.$field[idx]),+
                                    }),
                                )
                                .expect("valid configured price stream");
                                black_box(output);
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            });

            group.finish();
        }
    };
}

define_single_output_benchmark!(
    bench_min_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MIN",
    MIN,
    MINConfig,
    MIN::new,
    MINConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_max_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MAX",
    MAX,
    MAXConfig,
    MAX::new,
    MAXConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_minindex_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MININDEX",
    MININDEX,
    MININDEXConfig,
    MININDEX::new,
    MININDEXConfig::new,
    1,
    i32,
    0_i32,
    usize,
    0_usize
);
define_single_output_benchmark!(
    bench_maxindex_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    MAXINDEX::new,
    MAXINDEXConfig::new,
    1,
    i32,
    0_i32,
    usize,
    0_usize
);

define_single_output_workloads!(
    bench_wma_repeated_and_streaming,
    "indicator_execution/expanded/windowed_overlap_workloads/WMA",
    WMA,
    WMAConfig,
    WMA::new,
    WMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_trima_repeated_and_streaming,
    "indicator_execution/expanded/windowed_overlap_workloads/TRIMA",
    TRIMA,
    TRIMAConfig,
    TRIMA::new,
    TRIMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_wma_qualified_matrix,
    "indicator_execution/expanded/windowed_overlap/WMA",
    WMA,
    WMAConfig,
    WMA::new,
    WMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_trima_qualified_matrix,
    "indicator_execution/expanded/windowed_overlap/TRIMA",
    TRIMA,
    TRIMAConfig,
    TRIMA::new,
    TRIMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);

define_single_output_workloads!(
    bench_ema_repeated_and_streaming,
    "indicator_execution/expanded/recursive_overlap_workloads/EMA",
    EMA,
    EMAConfig,
    EMA::new,
    EMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_dema_repeated_and_streaming,
    "indicator_execution/expanded/recursive_overlap_workloads/DEMA",
    DEMA,
    DEMAConfig,
    DEMA::new,
    DEMAConfig::new,
    2,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_tema_repeated_and_streaming,
    "indicator_execution/expanded/recursive_overlap_workloads/TEMA",
    TEMA,
    TEMAConfig,
    TEMA::new,
    TEMAConfig::new,
    3,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_t3_repeated_and_streaming,
    "indicator_execution/expanded/recursive_overlap_workloads/T3",
    T3,
    T3Config,
    T3::with_default_vfactor,
    T3Config::with_default_vfactor,
    6,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_ma_ema_repeated_and_streaming,
    "indicator_execution/expanded/recursive_overlap_workloads/MA_EMA",
    MA,
    MAConfig,
    ma_ema_indicator,
    ma_ema_config,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_ema_qualified_matrix,
    "indicator_execution/expanded/recursive_overlap/EMA",
    EMA,
    EMAConfig,
    EMA::new,
    EMAConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_dema_qualified_matrix,
    "indicator_execution/expanded/recursive_overlap/DEMA",
    DEMA,
    DEMAConfig,
    DEMA::new,
    DEMAConfig::new,
    2,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_tema_qualified_matrix,
    "indicator_execution/expanded/recursive_overlap/TEMA",
    TEMA,
    TEMAConfig,
    TEMA::new,
    TEMAConfig::new,
    3,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_t3_qualified_matrix,
    "indicator_execution/expanded/recursive_overlap/T3",
    T3,
    T3Config,
    T3::with_default_vfactor,
    T3Config::with_default_vfactor,
    6,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_ma_ema_qualified_matrix,
    "indicator_execution/expanded/recursive_overlap/MA_EMA",
    MA,
    MAConfig,
    ma_ema_indicator,
    ma_ema_config,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);

define_single_output_benchmark!(
    bench_avgdev_qualified_matrix,
    "indicator_execution/expanded/price_transform/AVGDEV",
    AVGDEV,
    AVGDEVConfig,
    AVGDEV::new,
    AVGDEVConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_avgdev_repeated_and_streaming,
    "indicator_execution/expanded/price_transform_workloads/AVGDEV",
    AVGDEV,
    AVGDEVConfig,
    AVGDEV::new,
    AVGDEVConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_named_price_benchmarks!(
    bench_avgprice_qualified_matrix,
    bench_avgprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/AVGPRICE",
    "indicator_execution/expanded/price_transform_workloads/AVGPRICE",
    AVGPRICE,
    AVGPRICEConfig,
    AVGPRICEInput,
    AVGPRICETick,
    [open, high, low, close]
);
define_named_price_benchmarks!(
    bench_medprice_qualified_matrix,
    bench_medprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/MEDPRICE",
    "indicator_execution/expanded/price_transform_workloads/MEDPRICE",
    MEDPRICE,
    MEDPRICEConfig,
    MEDPRICEInput,
    MEDPRICETick,
    [high, low]
);
define_named_price_benchmarks!(
    bench_typprice_qualified_matrix,
    bench_typprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/TYPPRICE",
    "indicator_execution/expanded/price_transform_workloads/TYPPRICE",
    TYPPRICE,
    TYPPRICEConfig,
    TYPPRICEInput,
    TYPPRICETick,
    [high, low, close]
);
define_named_price_benchmarks!(
    bench_wclprice_qualified_matrix,
    bench_wclprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/WCLPRICE",
    "indicator_execution/expanded/price_transform_workloads/WCLPRICE",
    WCLPRICE,
    WCLPRICEConfig,
    WCLPRICEInput,
    WCLPRICETick,
    [high, low, close]
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
    bench_wma_qualified_matrix,
    bench_trima_qualified_matrix,
    bench_wma_repeated_and_streaming,
    bench_trima_repeated_and_streaming,
    bench_ema_qualified_matrix,
    bench_dema_qualified_matrix,
    bench_tema_qualified_matrix,
    bench_t3_qualified_matrix,
    bench_ma_ema_qualified_matrix,
    bench_ema_repeated_and_streaming,
    bench_dema_repeated_and_streaming,
    bench_tema_repeated_and_streaming,
    bench_t3_repeated_and_streaming,
    bench_ma_ema_repeated_and_streaming,
    bench_minmax_repeated_and_streaming,
    bench_minmaxindex_repeated_and_streaming,
    bench_universe,
    bench_parameter_sweep,
    bench_per_worker_predecessor,
    bench_prepared_universe,
    bench_prepared_parameter_sweep,
    bench_prepared_per_worker,
    bench_multi_instrument_streaming,
    bench_avgdev_qualified_matrix,
    bench_avgprice_qualified_matrix,
    bench_medprice_qualified_matrix,
    bench_typprice_qualified_matrix,
    bench_wclprice_qualified_matrix,
    bench_avgdev_repeated_and_streaming,
    bench_avgprice_repeated_and_streaming,
    bench_medprice_repeated_and_streaming,
    bench_typprice_repeated_and_streaming,
    bench_wclprice_repeated_and_streaming,
);
criterion_main!(benches);
