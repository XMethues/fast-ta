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
    cycle::{HT_DCPERIODConfig, HT_DCPERIOD_LOOKBACK},
    math_operators::{
        ADDConfig, BinaryInput, BinaryTick, DIVConfig, MAXConfig, MAXINDEXConfig, MINConfig,
        MININDEXConfig, MINMAXConfig, MINMAXINDEXConfig, MINMAXINDEXValuesMut, MINMAXValuesMut,
        MULTConfig, SUBConfig, SUMConfig, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX,
        MINMAXINDEX, MULT, SUB, SUM,
    },
    math_transform::{
        ACOSConfig, ASINConfig, ATANConfig, CEILConfig, COSConfig, COSHConfig, EXPConfig,
        FLOORConfig, LNConfig, LOG10Config, SINConfig, SINHConfig, SQRTConfig, TANConfig,
        TANHConfig, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN,
        TANH,
    },
    overlap::{
        DEMAConfig, EMAConfig, MAConfig, PeriodMAType, SMAConfig, T3Config, TEMAConfig,
        TRIMAConfig, WMAConfig, DEMA, EMA, MA, SMA, T3, TEMA, TRIMA, WMA,
    },
    price_transform::{
        AVGDEVConfig, AVGPRICEConfig, AVGPRICEInput, AVGPRICETick, MEDPRICEConfig, MEDPRICEInput,
        MEDPRICETick, TYPPRICEConfig, TYPPRICEInput, TYPPRICETick, WCLPRICEConfig, WCLPRICEInput,
        WCLPRICETick, AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE,
    },
    statistic::{
        BETAConfig, CORRELConfig, LINEARREGConfig, LINEARREG_ANGLEConfig,
        LINEARREG_INTERCEPTConfig, LINEARREG_SLOPEConfig, PairInput, PairTick, STDDEVConfig,
        TSFConfig, VARConfig, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT,
        LINEARREG_SLOPE, STDDEV, TSF, VAR,
    },
    volatility::{
        ATRConfig, ATRInput, ATRTick, NATRConfig, NATRInput, NATRTick, TRANGEConfig, TRANGEInput,
        TRANGETick, ATR, NATR, TRANGE,
    },
    volume::{
        ADConfig, ADInput, ADOSCConfig, ADOSCInput, ADOSCTick, ADTick, OBVConfig, OBVInput,
        OBVTick, AD, ADOSC, OBV,
    },
    Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation,
};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const EXECUTION_MATRIX: &[(usize, usize)] = &[
    (64, 14),
    (4_096, 14),
    (4_096, 512),
    (65_536, 14),
    (65_536, 512),
];

fn ma_ema_config(period: usize) -> ta_core::Result<MAConfig> {
    MAConfig::new(period, PeriodMAType::EMA)
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
                let config = AVGPRICEConfig::new();
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
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
                let config = AVGPRICEConfig::new();

                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
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
                let config = MINMAXConfig::new(black_box(PERIOD)).expect("valid period");
                let output_len = output_len(size, PERIOD);
                let mut min = vec![0.0 as Float; output_len];
                let mut max = vec![0.0 as Float; output_len];

                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        MINMAXValuesMut {
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
                let config = MINMAXConfig::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
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
                let config = MINMAXINDEXConfig::new(black_box(PERIOD)).expect("valid period");
                let output_len = output_len(size, PERIOD);
                let mut min_idx = vec![0_usize; output_len];
                let mut max_idx = vec![0_usize; output_len];

                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        MINMAXINDEXValuesMut {
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
                let config = MINMAXINDEXConfig::new(black_box(PERIOD)).expect("valid period");

                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
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
                let config = MINMAXConfig::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
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
                let config = MINMAXINDEXConfig::new(black_box(period)).expect("valid period");
                b.iter_batched(
                    || (),
                    |_| {
                        let output = IndicatorConfig::compute(
                            black_box(&config),
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

fn unit_domain_fixture(size: usize, seed: usize) -> Vec<Float> {
    (0..size)
        .map(|index| {
            let sample = ((index * 37 + seed * 17) % 201) as Float;
            sample / 100.0 as Float - 1.0 as Float
        })
        .collect()
}

fn math_transform_fixtures(
    series_count: usize,
    fixture: fn(usize, usize) -> Vec<Float>,
) -> Vec<Vec<Float>> {
    (0..series_count)
        .map(|seed| fixture(REPEATED_SERIES_LEN, seed))
        .collect()
}

fn math_transform_stream_inputs(fixture: fn(usize, usize) -> Vec<Float>) -> Vec<Vec<Float>> {
    let instrument_inputs = math_transform_fixtures(STREAM_INSTRUMENTS, fixture);
    (0..REPEATED_SERIES_LEN)
        .map(|tick_index| {
            instrument_inputs
                .iter()
                .map(|series| series[tick_index])
                .collect()
        })
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
    let config = SMAConfig::new(PERIOD).expect("valid period");
    let mut output = vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)];
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("universe/SMA/caller_compact_reuse", |b| {
        b.iter(|| {
            for series in &universe {
                let range = IndicatorConfig::compute_into(
                    black_box(&config),
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
    let configs = SWEEP_PERIODS
        .iter()
        .map(|&period| SMAConfig::new(period).expect("valid sweep period"))
        .collect::<Vec<_>>();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));

    group.bench_function("parameter_sweep/SMA/caller_compact_reuse", |b| {
        b.iter(|| {
            for config in &configs {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
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
    let configs = (0..WORKERS)
        .map(|_| SMAConfig::new(PERIOD).expect("valid period"))
        .collect::<Vec<_>>();
    let inputs = worker_fixtures();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; output_len(REPEATED_SERIES_LEN, PERIOD)])
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("indicator_execution/current/repeated");
    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));

    group.bench_function("no_prepared_runner/per_worker_instances", |b| {
        b.iter(|| {
            for ((config, input), output) in
                configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
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
        let config = SMAConfig::new(PERIOD).expect("valid period");
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
        let configs = (0..WORKERS)
            .map(|_| MINMAXConfig::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0.0 as Float);
        b.iter(|| {
            for ((config, input), (min, max)) in
                configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
                    black_box(input.as_slice()),
                    MINMAXValuesMut {
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
    let minmax_stream_config = MINMAXConfig::new(PERIOD).expect("valid period");
    group.bench_function("streaming/legacy_instances", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&minmax_stream_config).expect("valid period"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<MINMAXConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
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
        let configs = (0..WORKERS)
            .map(|_| MINMAXINDEXConfig::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>();
        let inputs = worker_fixtures();
        let mut outputs = extrema_worker_outputs(0_usize);
        b.iter(|| {
            for ((config, input), (min_idx, max_idx)) in
                configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
            {
                let range = IndicatorConfig::compute_into(
                    black_box(config),
                    black_box(input.as_slice()),
                    MINMAXINDEXValuesMut {
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
    let minmaxindex_stream_config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
    group.bench_function("streaming/legacy_instances", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| {
                        IndicatorConfig::stream(&minmaxindex_stream_config).expect("valid period")
                    })
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<MINMAXINDEXConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
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
                let configs = (0..WORKERS)
                    .map(|_| $config_new(PERIOD).expect("valid period"))
                    .collect::<Vec<_>>();
                let inputs = worker_fixtures();
                let mut outputs = single_output_worker_outputs::<$config_type>(
                    $config_initial,
                    $lookback_multiplier,
                );
                b.iter(|| {
                    for ((config, input), output) in
                        configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
                    {
                        let range = IndicatorConfig::compute_into(
                            black_box(config),
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
            let legacy_stream_config = $config_new(PERIOD).expect("valid period");
            group.bench_function("streaming/legacy_instances", |b| {
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| {
                                IndicatorConfig::stream(&legacy_stream_config)
                                    .expect("valid period")
                            })
                            .collect::<Vec<_>>()
                    },
                    |streams| {
                        for_each_stream_sample!(streams, &inputs, |stream, input| {
                            let output = StreamingComputation::<$config>::next(
                                black_box(stream),
                                black_box(input),
                            )
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
    MINConfig::new,
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
    MAXConfig::new,
    MAXConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_sum_repeated_and_streaming,
    "indicator_execution/expanded/arithmetic_workloads/SUM",
    SUM,
    SUMConfig,
    SUMConfig::new,
    SUMConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_var_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/VAR",
    VAR,
    VARConfig,
    VARConfig::with_default_nbdev,
    VARConfig::with_default_nbdev,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_stddev_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/STDDEV",
    STDDEV,
    STDDEVConfig,
    STDDEVConfig::with_default_nbdev,
    STDDEVConfig::with_default_nbdev,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_linearreg_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/LINEARREG",
    LINEARREG,
    LINEARREGConfig,
    LINEARREGConfig::new,
    LINEARREGConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_linearreg_slope_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/LINEARREG_SLOPE",
    LINEARREG_SLOPE,
    LINEARREG_SLOPEConfig,
    LINEARREG_SLOPEConfig::new,
    LINEARREG_SLOPEConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_linearreg_intercept_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/LINEARREG_INTERCEPT",
    LINEARREG_INTERCEPT,
    LINEARREG_INTERCEPTConfig,
    LINEARREG_INTERCEPTConfig::new,
    LINEARREG_INTERCEPTConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_linearreg_angle_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/LINEARREG_ANGLE",
    LINEARREG_ANGLE,
    LINEARREG_ANGLEConfig,
    LINEARREG_ANGLEConfig::new,
    LINEARREG_ANGLEConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_workloads!(
    bench_tsf_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics_workloads/TSF",
    TSF,
    TSFConfig,
    TSFConfig::new,
    TSFConfig::new,
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
    MININDEXConfig::new,
    MININDEXConfig::new,
    1,
    usize,
    0_usize,
    usize,
    0_usize
);
define_single_output_workloads!(
    bench_maxindex_repeated_and_streaming,
    "indicator_execution/expanded/single_extrema_workloads/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    MAXINDEXConfig::new,
    MAXINDEXConfig::new,
    1,
    usize,
    0_usize,
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
                                let config = $config_new(black_box(period)).expect("valid period");
                                b.iter_batched(
                                    || (),
                                    |_| {
                                        let output = IndicatorConfig::compute(
                                            black_box(&config),
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

macro_rules! bench_parameter_free_transform {
    ($criterion:expr, $indicator:ident, $config:ident, $fixture:path) => {{
        let mut group = $criterion.benchmark_group(concat!(
            "indicator_execution/expanded/math_transform/",
            stringify!($indicator)
        ));
        for (case_index, &size) in SIZES.iter().enumerate() {
            group.throughput(Throughput::Elements(size as u64));
            for path in rotated_execution_paths(case_index) {
                match path {
                    ExecutionPath::Current => group.bench_with_input(
                        BenchmarkId::new("current_caller_compact", size),
                        &size,
                        |b, &size| {
                            let input = $fixture(size, 0);
                            let config = $config::new();
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = IndicatorConfig::compute_into(
                                    black_box(&config),
                                    black_box(input.as_slice()),
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid current ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::Config => group.bench_with_input(
                        BenchmarkId::new("config_caller_compact", size),
                        &size,
                        |b, &size| {
                            let input = $fixture(size, 0);
                            let config = $config::new();
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = IndicatorConfig::compute_into(
                                    black_box(&config),
                                    black_box(input.as_slice()),
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid configured ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::Prepared => group.bench_with_input(
                        BenchmarkId::new("prepared_runner", size),
                        &size,
                        |b, &size| {
                            let input = $fixture(size, 0);
                            let config = $config::new();
                            let mut runner = IndicatorConfig::prepare_batch(&config, size)
                                .expect("valid prepared capacity");
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = PreparedBatchRunner::<$config>::compute_into(
                                    black_box(&mut runner),
                                    black_box(input.as_slice()),
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid prepared ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::LegacyOwned => group.bench_with_input(
                        BenchmarkId::new("legacy_owned_aligned", size),
                        &size,
                        |b, &size| {
                            let input = $fixture(size, 0);
                            let config = $config::new();
                            b.iter_batched(
                                || (),
                                |_| {
                                    let output = IndicatorConfig::compute(
                                        black_box(&config),
                                        black_box(input.as_slice()),
                                    )
                                    .expect(concat!(
                                        "valid legacy owned ",
                                        stringify!($indicator),
                                        " fixture"
                                    ));
                                    black_box(output)
                                },
                                BatchSize::LargeInput,
                            );
                        },
                    ),
                    ExecutionPath::ConfigOwned => group.bench_with_input(
                        BenchmarkId::new("config_owned_compact", size),
                        &size,
                        |b, &size| {
                            let input = $fixture(size, 0);
                            let config = $config::new();
                            b.iter_batched(
                                || (),
                                |_| {
                                    let output = IndicatorConfig::compute(
                                        black_box(&config),
                                        black_box(input.as_slice()),
                                    )
                                    .expect(concat!(
                                        "valid configured owned ",
                                        stringify!($indicator),
                                        " fixture"
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

        let mut group = $criterion.benchmark_group(concat!(
            "indicator_execution/expanded/math_transform_workloads/",
            stringify!($indicator)
        ));
        group.throughput(Throughput::Elements(
            (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
        ));
        group.bench_function("universe/current_caller_compact", |b| {
            let universe = math_transform_fixtures(UNIVERSE_INSTRUMENTS, $fixture);
            let config = $config::new();
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
            b.iter(|| {
                for input in &universe {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
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
            let universe = math_transform_fixtures(UNIVERSE_INSTRUMENTS, $fixture);
            let config = $config::new();
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
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
            let universe = math_transform_fixtures(UNIVERSE_INSTRUMENTS, $fixture);
            let config = $config::new();
            let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                .expect("valid prepared capacity");
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
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

        group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
        group.bench_function("per_worker/current_instances", |b| {
            let configs = (0..WORKERS).map(|_| $config::new()).collect::<Vec<_>>();
            let inputs = math_transform_fixtures(WORKERS, $fixture);
            let mut outputs = (0..WORKERS)
                .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN])
                .collect::<Vec<_>>();
            b.iter(|| {
                for ((config, input), output) in
                    configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
                {
                    let range = IndicatorConfig::compute_into(
                        black_box(config),
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
            let config = $config::new();
            let mut runners = (0..WORKERS)
                .map(|_| {
                    IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                        .expect("valid prepared capacity")
                })
                .collect::<Vec<_>>();
            let inputs = math_transform_fixtures(WORKERS, $fixture);
            let mut outputs = (0..WORKERS)
                .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN])
                .collect::<Vec<_>>();
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

        let inputs = math_transform_stream_inputs($fixture);
        group.throughput(Throughput::Elements(
            (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
        ));
        let legacy_stream_config = $config::new();
        group.bench_function("streaming/legacy_instances", |b| {
            b.iter_batched_ref(
                || {
                    (0..STREAM_INSTRUMENTS)
                        .map(|_| {
                            IndicatorConfig::stream(&legacy_stream_config)
                                .expect("valid parameter-free stream")
                        })
                        .collect::<Vec<_>>()
                },
                |streams| {
                    for_each_stream_sample!(streams, &inputs, |stream, input| {
                        let output = StreamingComputation::<$config>::next(
                            black_box(stream),
                            black_box(input),
                        )
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
        let config = $config::new();
        group.bench_function("streaming/config_streams", |b| {
            b.iter_batched_ref(
                || {
                    (0..STREAM_INSTRUMENTS)
                        .map(|_| {
                            IndicatorConfig::stream(&config).expect("valid parameter-free stream")
                        })
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
    }};
}

fn bench_math_transform_execution(c: &mut Criterion) {
    bench_parameter_free_transform!(c, ACOS, ACOSConfig, unit_domain_fixture);
    bench_parameter_free_transform!(c, ASIN, ASINConfig, unit_domain_fixture);
    bench_parameter_free_transform!(c, ATAN, ATANConfig, series_fixture);
    bench_parameter_free_transform!(c, CEIL, CEILConfig, series_fixture);
    bench_parameter_free_transform!(c, COS, COSConfig, series_fixture);
    bench_parameter_free_transform!(c, COSH, COSHConfig, series_fixture);
    bench_parameter_free_transform!(c, EXP, EXPConfig, series_fixture);
    bench_parameter_free_transform!(c, FLOOR, FLOORConfig, series_fixture);
    bench_parameter_free_transform!(c, LN, LNConfig, series_fixture);
    bench_parameter_free_transform!(c, LOG10, LOG10Config, series_fixture);
    bench_parameter_free_transform!(c, SIN, SINConfig, series_fixture);
    bench_parameter_free_transform!(c, SINH, SINHConfig, series_fixture);
    bench_parameter_free_transform!(c, SQRT, SQRTConfig, series_fixture);
}
struct PairFixture {
    real0: Vec<Float>,
    real1: Vec<Float>,
}

fn paired_fixture(size: usize) -> PairFixture {
    let (real0, real1) = binary_fixture(size, 0);
    PairFixture { real0, real1 }
}

fn binary_fixture(size: usize, seed: usize) -> (Vec<Float>, Vec<Float>) {
    (series_fixture(size, seed), series_fixture(size, seed + 1))
}

fn binary_fixtures(series_count: usize) -> Vec<(Vec<Float>, Vec<Float>)> {
    (0..series_count)
        .map(|seed| binary_fixture(REPEATED_SERIES_LEN, seed * 2))
        .collect()
}

fn binary_stream_inputs() -> Vec<Vec<BinaryTick>> {
    let instrument_inputs = binary_fixtures(STREAM_INSTRUMENTS);
    (0..REPEATED_SERIES_LEN)
        .map(|tick_index| {
            instrument_inputs
                .iter()
                .map(|(real0, real1)| BinaryTick {
                    real0: real0[tick_index],
                    real1: real1[tick_index],
                })
                .collect()
        })
        .collect()
}

macro_rules! bench_binary_operator {
    ($criterion:expr, $indicator:ident, $config:ident) => {{
        let mut group = $criterion.benchmark_group(concat!(
            "indicator_execution/expanded/arithmetic/",
            stringify!($indicator)
        ));
        for (case_index, &size) in SIZES.iter().enumerate() {
            group.throughput(Throughput::Elements(size as u64));
            for path in rotated_execution_paths(case_index) {
                match path {
                    ExecutionPath::Current => group.bench_with_input(
                        BenchmarkId::new("current_caller_compact", size),
                        &size,
                        |b, &size| {
                            let (real0, real1) = binary_fixture(size, 0);
                            let config = $config::new();
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = IndicatorConfig::compute_into(
                                    black_box(&config),
                                    BinaryInput {
                                        real0: black_box(real0.as_slice()),
                                        real1: black_box(real1.as_slice()),
                                    },
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid current ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::Config => group.bench_with_input(
                        BenchmarkId::new("config_caller_compact", size),
                        &size,
                        |b, &size| {
                            let (real0, real1) = binary_fixture(size, 0);
                            let config = $config::new();
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = IndicatorConfig::compute_into(
                                    black_box(&config),
                                    BinaryInput {
                                        real0: black_box(real0.as_slice()),
                                        real1: black_box(real1.as_slice()),
                                    },
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid configured ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::Prepared => group.bench_with_input(
                        BenchmarkId::new("prepared_runner", size),
                        &size,
                        |b, &size| {
                            let (real0, real1) = binary_fixture(size, 0);
                            let config = $config::new();
                            let mut runner = IndicatorConfig::prepare_batch(&config, size)
                                .expect("valid prepared capacity");
                            let mut output = vec![0.0 as Float; size];
                            b.iter(|| {
                                let range = PreparedBatchRunner::<$config>::compute_into(
                                    black_box(&mut runner),
                                    BinaryInput {
                                        real0: black_box(real0.as_slice()),
                                        real1: black_box(real1.as_slice()),
                                    },
                                    black_box(output.as_mut_slice()),
                                )
                                .expect(concat!(
                                    "valid prepared ",
                                    stringify!($indicator),
                                    " fixture"
                                ));
                                black_box((range, output.as_slice()));
                            });
                        },
                    ),
                    ExecutionPath::LegacyOwned => group.bench_with_input(
                        BenchmarkId::new("legacy_owned_aligned", size),
                        &size,
                        |b, &size| {
                            let (real0, real1) = binary_fixture(size, 0);
                            let config = $config::new();
                            b.iter_batched(
                                || (),
                                |_| {
                                    let output = IndicatorConfig::compute(
                                        black_box(&config),
                                        BinaryInput {
                                            real0: black_box(real0.as_slice()),
                                            real1: black_box(real1.as_slice()),
                                        },
                                    )
                                    .expect(concat!(
                                        "valid legacy owned ",
                                        stringify!($indicator),
                                        " fixture"
                                    ));
                                    black_box(output)
                                },
                                BatchSize::LargeInput,
                            );
                        },
                    ),
                    ExecutionPath::ConfigOwned => group.bench_with_input(
                        BenchmarkId::new("config_owned_compact", size),
                        &size,
                        |b, &size| {
                            let (real0, real1) = binary_fixture(size, 0);
                            let config = $config::new();
                            b.iter_batched(
                                || (),
                                |_| {
                                    let output = IndicatorConfig::compute(
                                        black_box(&config),
                                        BinaryInput {
                                            real0: black_box(real0.as_slice()),
                                            real1: black_box(real1.as_slice()),
                                        },
                                    )
                                    .expect(concat!(
                                        "valid configured owned ",
                                        stringify!($indicator),
                                        " fixture"
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

        let mut group = $criterion.benchmark_group(concat!(
            "indicator_execution/expanded/arithmetic_workloads/",
            stringify!($indicator)
        ));
        group.throughput(Throughput::Elements(
            (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
        ));
        group.bench_function("universe/current_caller_compact", |b| {
            let universe = binary_fixtures(UNIVERSE_INSTRUMENTS);
            let config = $config::new();
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
            b.iter(|| {
                for (real0, real1) in &universe {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        BinaryInput {
                            real0: black_box(real0.as_slice()),
                            real1: black_box(real1.as_slice()),
                        },
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
            let universe = binary_fixtures(UNIVERSE_INSTRUMENTS);
            let config = $config::new();
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
            b.iter(|| {
                for (real0, real1) in &universe {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        BinaryInput {
                            real0: black_box(real0.as_slice()),
                            real1: black_box(real1.as_slice()),
                        },
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
            let universe = binary_fixtures(UNIVERSE_INSTRUMENTS);
            let config = $config::new();
            let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                .expect("valid prepared capacity");
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
            b.iter(|| {
                for (real0, real1) in &universe {
                    let range = PreparedBatchRunner::<$config>::compute_into(
                        black_box(&mut runner),
                        BinaryInput {
                            real0: black_box(real0.as_slice()),
                            real1: black_box(real1.as_slice()),
                        },
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

        group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
        group.bench_function("per_worker/current_instances", |b| {
            let configs = (0..WORKERS).map(|_| $config::new()).collect::<Vec<_>>();
            let inputs = binary_fixtures(WORKERS);
            let mut outputs = (0..WORKERS)
                .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN])
                .collect::<Vec<_>>();
            b.iter(|| {
                for ((config, (real0, real1)), output) in
                    configs.iter().zip(inputs.iter()).zip(outputs.iter_mut())
                {
                    let range = IndicatorConfig::compute_into(
                        black_box(config),
                        BinaryInput {
                            real0: black_box(real0.as_slice()),
                            real1: black_box(real1.as_slice()),
                        },
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
            let config = $config::new();
            let mut runners = (0..WORKERS)
                .map(|_| {
                    IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                        .expect("valid prepared capacity")
                })
                .collect::<Vec<_>>();
            let inputs = binary_fixtures(WORKERS);
            let mut outputs = (0..WORKERS)
                .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN])
                .collect::<Vec<_>>();
            b.iter(|| {
                for ((runner, (real0, real1)), output) in runners
                    .iter_mut()
                    .zip(inputs.iter())
                    .zip(outputs.iter_mut())
                {
                    let range = PreparedBatchRunner::<$config>::compute_into(
                        black_box(runner),
                        BinaryInput {
                            real0: black_box(real0.as_slice()),
                            real1: black_box(real1.as_slice()),
                        },
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

        let inputs = binary_stream_inputs();
        group.throughput(Throughput::Elements(
            (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
        ));
        let legacy_stream_config = $config::new();
        group.bench_function("streaming/legacy_instances", |b| {
            b.iter_batched_ref(
                || {
                    (0..STREAM_INSTRUMENTS)
                        .map(|_| {
                            IndicatorConfig::stream(&legacy_stream_config)
                                .expect("valid parameter-free stream")
                        })
                        .collect::<Vec<_>>()
                },
                |streams| {
                    for_each_stream_sample!(streams, &inputs, |stream, input| {
                        let output = StreamingComputation::<$config>::next(
                            black_box(stream),
                            black_box(input),
                        )
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
        let config = $config::new();
        group.bench_function("streaming/config_streams", |b| {
            b.iter_batched_ref(
                || {
                    (0..STREAM_INSTRUMENTS)
                        .map(|_| {
                            IndicatorConfig::stream(&config).expect("valid parameter-free stream")
                        })
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
    }};
}

fn bench_binary_operator_execution(c: &mut Criterion) {
    bench_binary_operator!(c, ADD, ADDConfig);
    bench_binary_operator!(c, SUB, SUBConfig);
    bench_binary_operator!(c, MULT, MULTConfig);
    bench_binary_operator!(c, DIV, DIVConfig);
}

macro_rules! define_named_input_benchmarks {
    (
        $matrix_name:ident,
        $workloads_name:ident,
        $matrix_group:literal,
        $workloads_group:literal,
        $indicator:ident,
        $config:ident,
        $indicator_ctor:expr,
        $config_ctor:expr,
        $fixture_ctor:path,
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
                        let ohlc = ($fixture_ctor)(size);
                        let config = ($config_ctor)();
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = IndicatorConfig::compute_into(
                                black_box(&config),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid current multi-series fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("config_caller_compact", size),
                    &size,
                    |b, &size| {
                        let ohlc = ($fixture_ctor)(size);
                        let config = ($config_ctor)();
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = IndicatorConfig::compute_into(
                                black_box(&config),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid configured multi-series fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("prepared_runner", size),
                    &size,
                    |b, &size| {
                        let ohlc = ($fixture_ctor)(size);
                        let config = ($config_ctor)();
                        let mut runner = IndicatorConfig::prepare_batch(&config, size)
                            .expect("valid prepared multi-series capacity");
                        let mut output = vec![0.0 as Float; size];
                        b.iter(|| {
                            let range = PreparedBatchRunner::<$config>::compute_into(
                                black_box(&mut runner),
                                black_box($input {
                                    $($field: ohlc.$field.as_slice()),+
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid prepared multi-series fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("legacy_owned_aligned", size),
                    &size,
                    |b, &size| {
                        let ohlc = ($fixture_ctor)(size);
                        let config = ($config_ctor)();
                        b.iter_batched(
                            || (),
                            |_| {
                                let output = IndicatorConfig::compute(
                                    black_box(&config),
                                    black_box($input {
                                        $($field: ohlc.$field.as_slice()),+
                                    }),
                                )
                                .expect("valid legacy owned multi-series fixture");
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
                        let ohlc = ($fixture_ctor)(size);
                        let config = ($config_ctor)();
                        b.iter_batched(
                            || (),
                            |_| {
                                let output = IndicatorConfig::compute(
                                    black_box(&config),
                                    black_box($input {
                                        $($field: ohlc.$field.as_slice()),+
                                    }),
                                )
                                .expect("valid configured owned multi-series fixture");
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
                    .map(|_| ($fixture_ctor)(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = ($config_ctor)();
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
                        .expect("valid current multi-series Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/config_caller_compact", |b| {
                let universe = (0..UNIVERSE_INSTRUMENTS)
                    .map(|_| ($fixture_ctor)(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = ($config_ctor)();
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
                        .expect("valid configured multi-series Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("universe/prepared_runner", |b| {
                let universe = (0..UNIVERSE_INSTRUMENTS)
                    .map(|_| ($fixture_ctor)(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                let config = ($config_ctor)();
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
                        .expect("valid prepared multi-series Universe");
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements(
                (WORKERS * REPEATED_SERIES_LEN) as u64,
            ));
            group.bench_function("per_worker/current_instances", |b| {
                let ohlc = ($fixture_ctor)(REPEATED_SERIES_LEN);
                let configs = (0..WORKERS)
                    .map(|_| ($config_ctor)())
                    .collect::<Vec<_>>();
                let mut outputs =
                    vec![vec![0.0 as Float; REPEATED_SERIES_LEN]; WORKERS];
                b.iter(|| {
                    for (config, output) in configs.iter().zip(outputs.iter_mut()) {
                        let range = IndicatorConfig::compute_into(
                            black_box(config),
                            black_box($input {
                                $($field: ohlc.$field.as_slice()),+
                            }),
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid current per-worker multi-series fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("per_worker/prepared_runners", |b| {
                let ohlc = ($fixture_ctor)(REPEATED_SERIES_LEN);
                let config = ($config_ctor)();
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
                        .expect("valid prepared per-worker multi-series fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });

            group.throughput(Throughput::Elements(
                (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
            ));
            let legacy_stream_config = ($config_ctor)();
            group.bench_function("streaming/legacy_instances", |b| {
                let inputs = (0..STREAM_INSTRUMENTS)
                    .map(|_| ($fixture_ctor)(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
                b.iter_batched_ref(
                    || {
                        (0..STREAM_INSTRUMENTS)
                            .map(|_| {
                                IndicatorConfig::stream(&legacy_stream_config)
                                    .expect("valid multi-series configuration")
                            })
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
                                .expect("valid legacy multi-series stream");
                                black_box(output);
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
            let config = ($config_ctor)();
            group.bench_function("streaming/config_streams", |b| {
                let inputs = (0..STREAM_INSTRUMENTS)
                    .map(|_| ($fixture_ctor)(REPEATED_SERIES_LEN))
                    .collect::<Vec<_>>();
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
                                .expect("valid configured multi-series stream");
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
    MINConfig::new,
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
    MAXConfig::new,
    MAXConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_sum_qualified_matrix,
    "indicator_execution/expanded/arithmetic/SUM",
    SUM,
    SUMConfig,
    SUMConfig::new,
    SUMConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_var_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/VAR",
    VAR,
    VARConfig,
    VARConfig::with_default_nbdev,
    VARConfig::with_default_nbdev,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_stddev_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/STDDEV",
    STDDEV,
    STDDEVConfig,
    STDDEVConfig::with_default_nbdev,
    STDDEVConfig::with_default_nbdev,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_linearreg_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/LINEARREG",
    LINEARREG,
    LINEARREGConfig,
    LINEARREGConfig::new,
    LINEARREGConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_linearreg_slope_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/LINEARREG_SLOPE",
    LINEARREG_SLOPE,
    LINEARREG_SLOPEConfig,
    LINEARREG_SLOPEConfig::new,
    LINEARREG_SLOPEConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_linearreg_intercept_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/LINEARREG_INTERCEPT",
    LINEARREG_INTERCEPT,
    LINEARREG_INTERCEPTConfig,
    LINEARREG_INTERCEPTConfig::new,
    LINEARREG_INTERCEPTConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_linearreg_angle_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/LINEARREG_ANGLE",
    LINEARREG_ANGLE,
    LINEARREG_ANGLEConfig,
    LINEARREG_ANGLEConfig::new,
    LINEARREG_ANGLEConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
define_single_output_benchmark!(
    bench_tsf_qualified_matrix,
    "indicator_execution/expanded/rolling_statistics/TSF",
    TSF,
    TSFConfig,
    TSFConfig::new,
    TSFConfig::new,
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
    MININDEXConfig::new,
    MININDEXConfig::new,
    1,
    usize,
    0_usize,
    usize,
    0_usize
);
define_single_output_benchmark!(
    bench_maxindex_qualified_scratch_matrix,
    "indicator_execution/expanded/single_extrema/MAXINDEX",
    MAXINDEX,
    MAXINDEXConfig,
    MAXINDEXConfig::new,
    MAXINDEXConfig::new,
    1,
    usize,
    0_usize,
    usize,
    0_usize
);

define_single_output_workloads!(
    bench_wma_repeated_and_streaming,
    "indicator_execution/expanded/windowed_overlap_workloads/WMA",
    WMA,
    WMAConfig,
    WMAConfig::new,
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
    TRIMAConfig::new,
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
    WMAConfig::new,
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
    TRIMAConfig::new,
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
    EMAConfig::new,
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
    DEMAConfig::new,
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
    TEMAConfig::new,
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
    T3Config::with_default_vfactor,
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
    ma_ema_config,
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
    EMAConfig::new,
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
    DEMAConfig::new,
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
    TEMAConfig::new,
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
    T3Config::with_default_vfactor,
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
    ma_ema_config,
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
    AVGDEVConfig::new,
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
    AVGDEVConfig::new,
    AVGDEVConfig::new,
    1,
    Float,
    0.0 as Float,
    Float,
    0.0 as Float
);
macro_rules! define_hlc_parameter_sweep {
    (
        $name:ident,
        $group_name:literal,
        $indicator:ident,
        $config:ident,
        $input:ident
    ) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];

            for &period in SWEEP_PERIODS {
                let config = $config::new(period).expect("valid volatility sweep period");
                let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid volatility sweep capacity");
                group.throughput(Throughput::Elements(REPEATED_SERIES_LEN as u64));

                group.bench_with_input(
                    BenchmarkId::new("current_caller_compact", period),
                    &period,
                    |b, _| {
                        b.iter(|| {
                            let range = IndicatorConfig::compute_into(
                                black_box(&config),
                                black_box($input {
                                    high: ohlc.high.as_slice(),
                                    low: ohlc.low.as_slice(),
                                    close: ohlc.close.as_slice(),
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid current volatility sweep fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("config_caller_compact", period),
                    &period,
                    |b, _| {
                        b.iter(|| {
                            let range = IndicatorConfig::compute_into(
                                black_box(&config),
                                black_box($input {
                                    high: ohlc.high.as_slice(),
                                    low: ohlc.low.as_slice(),
                                    close: ohlc.close.as_slice(),
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid configured volatility sweep fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new("prepared_runner", period),
                    &period,
                    |b, _| {
                        b.iter(|| {
                            let range = PreparedBatchRunner::<$config>::compute_into(
                                black_box(&mut runner),
                                black_box($input {
                                    high: ohlc.high.as_slice(),
                                    low: ohlc.low.as_slice(),
                                    close: ohlc.close.as_slice(),
                                }),
                                black_box(output.as_mut_slice()),
                            )
                            .expect("valid prepared volatility sweep fixture");
                            black_box((range, output.as_slice()));
                        });
                    },
                );
            }
            group.finish();
        }
    };
}

define_hlc_parameter_sweep!(
    bench_atr_parameter_sweep,
    "indicator_execution/expanded/volatility_workloads/ATR/parameter_sweep",
    ATR,
    ATRConfig,
    ATRInput
);
define_hlc_parameter_sweep!(
    bench_natr_parameter_sweep,
    "indicator_execution/expanded/volatility_workloads/NATR/parameter_sweep",
    NATR,
    NATRConfig,
    NATRInput
);

macro_rules! define_paired_statistic_parameter_sweep {
    ($name:ident, $group_name:literal, $indicator:ident, $config:ident) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);
            let fixture = paired_fixture(REPEATED_SERIES_LEN);
            let configs = SWEEP_PERIODS
                .iter()
                .map(|&period| $config::new(period).expect("valid paired-statistic sweep period"))
                .collect::<Vec<_>>();
            let mut runners = configs
                .iter()
                .map(|config| {
                    IndicatorConfig::prepare_batch(config, REPEATED_SERIES_LEN)
                        .expect("valid paired-statistic sweep capacity")
                })
                .collect::<Vec<_>>();
            let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
            group.throughput(Throughput::Elements(
                (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
            ));

            group.bench_function("current_caller_compact", |b| {
                b.iter(|| {
                    for config in &configs {
                        let range = IndicatorConfig::compute_into(
                            black_box(config),
                            PairInput {
                                real0: black_box(fixture.real0.as_slice()),
                                real1: black_box(fixture.real1.as_slice()),
                            },
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid current paired-statistic sweep fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("config_caller_compact", |b| {
                b.iter(|| {
                    for config in &configs {
                        let range = IndicatorConfig::compute_into(
                            black_box(config),
                            PairInput {
                                real0: black_box(fixture.real0.as_slice()),
                                real1: black_box(fixture.real1.as_slice()),
                            },
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid configured paired-statistic sweep fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.bench_function("prepared_runners", |b| {
                b.iter(|| {
                    for runner in &mut runners {
                        let range = PreparedBatchRunner::<$config>::compute_into(
                            black_box(runner),
                            PairInput {
                                real0: black_box(fixture.real0.as_slice()),
                                real1: black_box(fixture.real1.as_slice()),
                            },
                            black_box(output.as_mut_slice()),
                        )
                        .expect("valid prepared paired-statistic sweep fixture");
                        black_box((range, output.as_slice()));
                    }
                });
            });
            group.finish();
        }
    };
}

define_paired_statistic_parameter_sweep!(
    bench_correl_parameter_sweep,
    "indicator_execution/expanded/rolling_statistics_workloads/CORREL/parameter_sweep",
    CORREL,
    CORRELConfig
);
define_paired_statistic_parameter_sweep!(
    bench_beta_parameter_sweep,
    "indicator_execution/expanded/rolling_statistics_workloads/BETA/parameter_sweep",
    BETA,
    BETAConfig
);

fn bench_adosc_parameter_sweep(c: &mut Criterion) {
    let mut group =
        c.benchmark_group("indicator_execution/expanded/volume_workloads/ADOSC/parameter_sweep");
    let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];

    for &slowperiod in SWEEP_PERIODS {
        let fastperiod = core::cmp::max(1, slowperiod / 2);
        let config =
            ADOSCConfig::new(fastperiod, slowperiod).expect("valid ADOSC sweep parameters");
        let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
            .expect("valid ADOSC sweep capacity");
        group.throughput(Throughput::Elements(REPEATED_SERIES_LEN as u64));

        group.bench_with_input(
            BenchmarkId::new("current_caller_compact", slowperiod),
            &slowperiod,
            |b, _| {
                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(ADOSCInput {
                            high: ohlc.high.as_slice(),
                            low: ohlc.low.as_slice(),
                            close: ohlc.close.as_slice(),
                            volume: ohlc.volume.as_slice(),
                        }),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid ADOSC sweep fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("config_caller_compact", slowperiod),
            &slowperiod,
            |b, _| {
                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(ADOSCInput {
                            high: ohlc.high.as_slice(),
                            low: ohlc.low.as_slice(),
                            close: ohlc.close.as_slice(),
                            volume: ohlc.volume.as_slice(),
                        }),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid configured ADOSC sweep fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("prepared_runner", slowperiod),
            &slowperiod,
            |b, _| {
                b.iter(|| {
                    let range = PreparedBatchRunner::<ADOSCConfig>::compute_into(
                        black_box(&mut runner),
                        black_box(ADOSCInput {
                            high: ohlc.high.as_slice(),
                            low: ohlc.low.as_slice(),
                            close: ohlc.close.as_slice(),
                            volume: ohlc.volume.as_slice(),
                        }),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid prepared ADOSC sweep fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );
    }
    group.finish();
}

define_named_input_benchmarks!(
    bench_avgprice_qualified_matrix,
    bench_avgprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/AVGPRICE",
    "indicator_execution/expanded/price_transform_workloads/AVGPRICE",
    AVGPRICE,
    AVGPRICEConfig,
    AVGPRICEConfig::new,
    AVGPRICEConfig::new,
    ohlc_fixture,
    AVGPRICEInput,
    AVGPRICETick,
    [open, high, low, close]
);
define_named_input_benchmarks!(
    bench_medprice_qualified_matrix,
    bench_medprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/MEDPRICE",
    "indicator_execution/expanded/price_transform_workloads/MEDPRICE",
    MEDPRICE,
    MEDPRICEConfig,
    MEDPRICEConfig::new,
    MEDPRICEConfig::new,
    ohlc_fixture,
    MEDPRICEInput,
    MEDPRICETick,
    [high, low]
);
define_named_input_benchmarks!(
    bench_typprice_qualified_matrix,
    bench_typprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/TYPPRICE",
    "indicator_execution/expanded/price_transform_workloads/TYPPRICE",
    TYPPRICE,
    TYPPRICEConfig,
    TYPPRICEConfig::new,
    TYPPRICEConfig::new,
    ohlc_fixture,
    TYPPRICEInput,
    TYPPRICETick,
    [high, low, close]
);
define_named_input_benchmarks!(
    bench_wclprice_qualified_matrix,
    bench_wclprice_repeated_and_streaming,
    "indicator_execution/expanded/price_transform/WCLPRICE",
    "indicator_execution/expanded/price_transform_workloads/WCLPRICE",
    WCLPRICE,
    WCLPRICEConfig,
    WCLPRICEConfig::new,
    WCLPRICEConfig::new,
    ohlc_fixture,
    WCLPRICEInput,
    WCLPRICETick,
    [high, low, close]
);

define_named_input_benchmarks!(
    bench_ad_qualified_matrix,
    bench_ad_repeated_and_streaming,
    "indicator_execution/expanded/volume/AD",
    "indicator_execution/expanded/volume_workloads/AD",
    AD,
    ADConfig,
    ADConfig::new,
    ADConfig::new,
    ohlc_fixture,
    ADInput,
    ADTick,
    [high, low, close, volume]
);
define_named_input_benchmarks!(
    bench_adosc_qualified_matrix,
    bench_adosc_repeated_and_streaming,
    "indicator_execution/expanded/volume/ADOSC",
    "indicator_execution/expanded/volume_workloads/ADOSC",
    ADOSC,
    ADOSCConfig,
    || ADOSCConfig::new(PERIOD / 2, PERIOD).expect("valid ADOSC parameters"),
    || ADOSCConfig::new(PERIOD / 2, PERIOD).expect("valid ADOSC parameters"),
    ohlc_fixture,
    ADOSCInput,
    ADOSCTick,
    [high, low, close, volume]
);
define_named_input_benchmarks!(
    bench_obv_qualified_matrix,
    bench_obv_repeated_and_streaming,
    "indicator_execution/expanded/volume/OBV",
    "indicator_execution/expanded/volume_workloads/OBV",
    OBV,
    OBVConfig,
    OBVConfig::new,
    OBVConfig::new,
    ohlc_fixture,
    OBVInput,
    OBVTick,
    [close, volume]
);

define_named_input_benchmarks!(
    bench_trange_qualified_matrix,
    bench_trange_repeated_and_streaming,
    "indicator_execution/expanded/volatility/TRANGE",
    "indicator_execution/expanded/volatility_workloads/TRANGE",
    TRANGE,
    TRANGEConfig,
    TRANGEConfig::new,
    TRANGEConfig::new,
    ohlc_fixture,
    TRANGEInput,
    TRANGETick,
    [high, low, close]
);
define_named_input_benchmarks!(
    bench_atr_qualified_matrix,
    bench_atr_repeated_and_streaming,
    "indicator_execution/expanded/volatility/ATR",
    "indicator_execution/expanded/volatility_workloads/ATR",
    ATR,
    ATRConfig,
    || ATRConfig::new(PERIOD).expect("valid ATR period"),
    || ATRConfig::new(PERIOD).expect("valid ATR period"),
    ohlc_fixture,
    ATRInput,
    ATRTick,
    [high, low, close]
);
define_named_input_benchmarks!(
    bench_natr_qualified_matrix,
    bench_natr_repeated_and_streaming,
    "indicator_execution/expanded/volatility/NATR",
    "indicator_execution/expanded/volatility_workloads/NATR",
    NATR,
    NATRConfig,
    || NATRConfig::new(PERIOD).expect("valid NATR period"),
    || NATRConfig::new(PERIOD).expect("valid NATR period"),
    ohlc_fixture,
    NATRInput,
    NATRTick,
    [high, low, close]
);
define_named_input_benchmarks!(
    bench_correl_qualified_matrix,
    bench_correl_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics/CORREL",
    "indicator_execution/expanded/rolling_statistics_workloads/CORREL",
    CORREL,
    CORRELConfig,
    || CORRELConfig::new(PERIOD).expect("valid CORREL period"),
    || CORRELConfig::new(PERIOD).expect("valid CORREL period"),
    paired_fixture,
    PairInput,
    PairTick,
    [real0, real1]
);
define_named_input_benchmarks!(
    bench_beta_qualified_matrix,
    bench_beta_repeated_and_streaming,
    "indicator_execution/expanded/rolling_statistics/BETA",
    "indicator_execution/expanded/rolling_statistics_workloads/BETA",
    BETA,
    BETAConfig,
    || BETAConfig::new(PERIOD).expect("valid BETA period"),
    || BETAConfig::new(PERIOD).expect("valid BETA period"),
    paired_fixture,
    PairInput,
    PairTick,
    [real0, real1]
);

fn bench_ht_dcperiod_execution(c: &mut Criterion) {
    let config = HT_DCPERIODConfig::new();
    let mut group = c.benchmark_group("indicator_execution/expanded/cycle/HT_DCPERIOD");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("one_shot/caller_compact", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let mut output = vec![0.0 as Float; size.saturating_sub(HT_DCPERIOD_LOOKBACK)];
                b.iter(|| {
                    let range = IndicatorConfig::compute_into(
                        black_box(&config),
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid HT_DCPERIOD fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("one_shot/owned_compact", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                b.iter_batched(
                    || (),
                    |_| {
                        black_box(
                            IndicatorConfig::compute(
                                black_box(&config),
                                black_box(input.as_slice()),
                            )
                            .expect("valid HT_DCPERIOD fixture"),
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("one_shot/prepared_runner", size),
            &size,
            |b, &size| {
                let input = series_fixture(size, 0);
                let mut runner = IndicatorConfig::prepare_batch(&config, size)
                    .expect("valid HT_DCPERIOD capacity");
                let mut output = vec![0.0 as Float; size.saturating_sub(HT_DCPERIOD_LOOKBACK)];
                b.iter(|| {
                    let range = PreparedBatchRunner::<HT_DCPERIODConfig>::compute_into(
                        black_box(&mut runner),
                        black_box(input.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid prepared HT_DCPERIOD fixture");
                    black_box((range, output.as_slice()));
                });
            },
        );
    }

    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("universe/config_caller_compact", |b| {
        let universe = universe_fixtures();
        let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN - HT_DCPERIOD_LOOKBACK];
        b.iter(|| {
            for input in &universe {
                let range = IndicatorConfig::compute_into(
                    black_box(&config),
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid HT_DCPERIOD Universe");
                black_box((range, output.as_slice()));
            }
        });
    });
    group.bench_function("universe/prepared_runner", |b| {
        let universe = universe_fixtures();
        let mut runner = IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
            .expect("valid HT_DCPERIOD capacity");
        let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN - HT_DCPERIOD_LOOKBACK];
        b.iter(|| {
            for input in &universe {
                let range = PreparedBatchRunner::<HT_DCPERIODConfig>::compute_into(
                    black_box(&mut runner),
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid prepared HT_DCPERIOD Universe");
                black_box((range, output.as_slice()));
            }
        });
    });

    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    group.bench_function("per_worker/prepared_runners", |b| {
        let inputs = worker_fixtures();
        let mut runners = (0..WORKERS)
            .map(|_| {
                IndicatorConfig::prepare_batch(&config, REPEATED_SERIES_LEN)
                    .expect("valid HT_DCPERIOD capacity")
            })
            .collect::<Vec<_>>();
        let mut outputs = (0..WORKERS)
            .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN - HT_DCPERIOD_LOOKBACK])
            .collect::<Vec<_>>();
        b.iter(|| {
            for ((runner, input), output) in runners
                .iter_mut()
                .zip(inputs.iter())
                .zip(outputs.iter_mut())
            {
                let range = PreparedBatchRunner::<HT_DCPERIODConfig>::compute_into(
                    black_box(runner),
                    black_box(input.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid per-worker HT_DCPERIOD fixture");
                black_box((range, output.as_slice()));
            }
        });
    });

    let inputs = stream_inputs();
    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("streaming/config_streams", |b| {
        b.iter_batched_ref(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&config).expect("valid HT_DCPERIOD stream"))
                    .collect::<Vec<_>>()
            },
            |streams| {
                for_each_stream_sample!(streams, &inputs, |stream, input| {
                    let output = StreamingComputation::<HT_DCPERIODConfig>::next(
                        black_box(stream),
                        black_box(input),
                    )
                    .expect("valid HT_DCPERIOD stream tick");
                    black_box(output);
                });
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sma_one_shot,
    bench_ht_dcperiod_execution,
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
    bench_ad_qualified_matrix,
    bench_adosc_qualified_matrix,
    bench_obv_qualified_matrix,
    bench_ad_repeated_and_streaming,
    bench_adosc_repeated_and_streaming,
    bench_obv_repeated_and_streaming,
    bench_adosc_parameter_sweep,
    bench_trange_qualified_matrix,
    bench_atr_qualified_matrix,
    bench_natr_qualified_matrix,
    bench_trange_repeated_and_streaming,
    bench_atr_repeated_and_streaming,
    bench_natr_repeated_and_streaming,
    bench_atr_parameter_sweep,
    bench_natr_parameter_sweep,
    bench_math_transform_execution,
    bench_binary_operator_execution,
    bench_sum_qualified_matrix,
    bench_sum_repeated_and_streaming,
    bench_var_qualified_matrix,
    bench_stddev_qualified_matrix,
    bench_var_repeated_and_streaming,
    bench_stddev_repeated_and_streaming,
    bench_correl_qualified_matrix,
    bench_beta_qualified_matrix,
    bench_correl_repeated_and_streaming,
    bench_beta_repeated_and_streaming,
    bench_correl_parameter_sweep,
    bench_beta_parameter_sweep,
    bench_linearreg_qualified_matrix,
    bench_linearreg_slope_qualified_matrix,
    bench_linearreg_intercept_qualified_matrix,
    bench_linearreg_angle_qualified_matrix,
    bench_tsf_qualified_matrix,
    bench_linearreg_repeated_and_streaming,
    bench_linearreg_slope_repeated_and_streaming,
    bench_linearreg_intercept_repeated_and_streaming,
    bench_linearreg_angle_repeated_and_streaming,
    bench_tsf_repeated_and_streaming,
);
criterion_main!(benches);
