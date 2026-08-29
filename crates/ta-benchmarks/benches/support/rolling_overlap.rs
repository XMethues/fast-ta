use crate::support::{
    ohlc_fixture, series_fixture, PERIOD, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS, SWEEP_PERIODS,
    UNIVERSE_INSTRUMENTS, WORKERS,
};
use criterion::{BatchSize, Criterion, Throughput};
use fast_ta::overlap::{
    ACCBANDSConfig, ACCBANDSInput, ACCBANDSTick, ACCBANDSValuesMut, BBANDSConfig, BBANDSValuesMut,
    PeriodMAType,
};
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};
use std::hint::black_box;

fn band_columns(count: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    (
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
    )
}

fn bench_accbands(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/expanded/rolling_overlap/ACCBANDS");
    group.throughput(Throughput::Elements(REPEATED_SERIES_LEN as u64));

    group.bench_function("one_shot/caller_compact", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            black_box(
                config
                    .compute_into(
                        ACCBANDSInput {
                            high: black_box(ohlc.high.as_slice()),
                            low: black_box(ohlc.low.as_slice()),
                            close: black_box(ohlc.close.as_slice()),
                        },
                        ACCBANDSValuesMut {
                            upper: black_box(upper.as_mut_slice()),
                            middle: black_box(middle.as_mut_slice()),
                            lower: black_box(lower.as_mut_slice()),
                        },
                    )
                    .unwrap(),
            )
        });
    });
    group.bench_function("one_shot/owned_compact", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        b.iter_batched(
            || (),
            |_| {
                black_box(
                    config
                        .compute(ACCBANDSInput {
                            high: black_box(ohlc.high.as_slice()),
                            low: black_box(ohlc.low.as_slice()),
                            close: black_box(ohlc.close.as_slice()),
                        })
                        .unwrap(),
                )
            },
            BatchSize::LargeInput,
        );
    });
    group.bench_function("one_shot/prepared_runner", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            black_box(
                runner
                    .compute_into(
                        ACCBANDSInput {
                            high: black_box(ohlc.high.as_slice()),
                            low: black_box(ohlc.low.as_slice()),
                            close: black_box(ohlc.close.as_slice()),
                        },
                        ACCBANDSValuesMut {
                            upper: black_box(upper.as_mut_slice()),
                            middle: black_box(middle.as_mut_slice()),
                            lower: black_box(lower.as_mut_slice()),
                        },
                    )
                    .unwrap(),
            )
        });
    });

    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("universe/prepared_runner", |b| {
        let universe = (0..UNIVERSE_INSTRUMENTS)
            .map(|_| ohlc_fixture(REPEATED_SERIES_LEN))
            .collect::<Vec<_>>();
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            for ohlc in &universe {
                black_box(
                    runner
                        .compute_into(
                            ACCBANDSInput {
                                high: ohlc.high.as_slice(),
                                low: ohlc.low.as_slice(),
                                close: ohlc.close.as_slice(),
                            },
                            ACCBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("parameter_sweep/prepared_runners", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| ACCBANDSConfig::new(period).unwrap())
            .collect::<Vec<_>>();
        let mut runners = configs
            .iter()
            .map(|config| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect::<Vec<_>>();
        let mut outputs = configs
            .iter()
            .map(|config| band_columns(REPEATED_SERIES_LEN - config.lookback()))
            .collect::<Vec<_>>();
        b.iter(|| {
            for (runner, (upper, middle, lower)) in runners.iter_mut().zip(outputs.iter_mut()) {
                black_box(
                    runner
                        .compute_into(
                            ACCBANDSInput {
                                high: ohlc.high.as_slice(),
                                low: ohlc.low.as_slice(),
                                close: ohlc.close.as_slice(),
                            },
                            ACCBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    group.bench_function("per_worker/prepared_runners", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        let mut runners = (0..WORKERS)
            .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect::<Vec<_>>();
        let mut outputs = (0..WORKERS)
            .map(|_| band_columns(REPEATED_SERIES_LEN - config.lookback()))
            .collect::<Vec<_>>();
        b.iter(|| {
            for (runner, (upper, middle, lower)) in runners.iter_mut().zip(outputs.iter_mut()) {
                black_box(
                    runner
                        .compute_into(
                            ACCBANDSInput {
                                high: ohlc.high.as_slice(),
                                low: ohlc.low.as_slice(),
                                close: ohlc.close.as_slice(),
                            },
                            ACCBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("streaming/independent_streams", |b| {
        let ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
        let config = ACCBANDSConfig::new(PERIOD).unwrap();
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| config.stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    let tick = ACCBANDSTick {
                        high: ohlc.high[index],
                        low: ohlc.low[index],
                        close: ohlc.close[index],
                    };
                    for stream in &mut streams {
                        black_box(stream.next(black_box(tick)).unwrap());
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_bbands(c: &mut Criterion) {
    let mut group = c.benchmark_group("indicator_execution/expanded/rolling_overlap/BBANDS");
    group.throughput(Throughput::Elements(REPEATED_SERIES_LEN as u64));

    group.bench_function("one_shot/caller_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::SMA).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            black_box(
                config
                    .compute_into(
                        black_box(input.as_slice()),
                        BBANDSValuesMut {
                            upper: black_box(upper.as_mut_slice()),
                            middle: black_box(middle.as_mut_slice()),
                            lower: black_box(lower.as_mut_slice()),
                        },
                    )
                    .unwrap(),
            )
        });
    });
    group.bench_function("one_shot/owned_compact", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::SMA).unwrap();
        b.iter_batched(
            || (),
            |_| black_box(config.compute(black_box(input.as_slice())).unwrap()),
            BatchSize::LargeInput,
        );
    });
    group.bench_function("one_shot/prepared_runner", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::SMA).unwrap();
        let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            black_box(
                runner
                    .compute_into(
                        black_box(input.as_slice()),
                        BBANDSValuesMut {
                            upper: black_box(upper.as_mut_slice()),
                            middle: black_box(middle.as_mut_slice()),
                            lower: black_box(lower.as_mut_slice()),
                        },
                    )
                    .unwrap(),
            )
        });
    });

    group.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("universe/prepared_runner", |b| {
        let universe = (0..UNIVERSE_INSTRUMENTS)
            .map(|seed| series_fixture(REPEATED_SERIES_LEN, seed))
            .collect::<Vec<_>>();
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::SMA).unwrap();
        let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        let (mut upper, mut middle, mut lower) =
            band_columns(REPEATED_SERIES_LEN - config.lookback());
        b.iter(|| {
            for input in &universe {
                black_box(
                    runner
                        .compute_into(
                            input.as_slice(),
                            BBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("parameter_sweep/prepared_runners", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let configs = SWEEP_PERIODS
            .iter()
            .map(|&period| {
                BBANDSConfig::with_default_deviations(period, PeriodMAType::EMA).unwrap()
            })
            .collect::<Vec<_>>();
        let mut runners = configs
            .iter()
            .map(|config| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect::<Vec<_>>();
        let mut outputs = configs
            .iter()
            .map(|config| band_columns(REPEATED_SERIES_LEN - config.lookback()))
            .collect::<Vec<_>>();
        b.iter(|| {
            for (runner, (upper, middle, lower)) in runners.iter_mut().zip(outputs.iter_mut()) {
                black_box(
                    runner
                        .compute_into(
                            input.as_slice(),
                            BBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    group.bench_function("per_worker/prepared_runners", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::SMA).unwrap();
        let mut runners = (0..WORKERS)
            .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect::<Vec<_>>();
        let mut outputs = (0..WORKERS)
            .map(|_| band_columns(REPEATED_SERIES_LEN - config.lookback()))
            .collect::<Vec<_>>();
        b.iter(|| {
            for (runner, (upper, middle, lower)) in runners.iter_mut().zip(outputs.iter_mut()) {
                black_box(
                    runner
                        .compute_into(
                            input.as_slice(),
                            BBANDSValuesMut {
                                upper: upper.as_mut_slice(),
                                middle: middle.as_mut_slice(),
                                lower: lower.as_mut_slice(),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    group.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    group.bench_function("streaming/independent_streams", |b| {
        let input = series_fixture(REPEATED_SERIES_LEN, 0);
        let config = BBANDSConfig::with_default_deviations(PERIOD, PeriodMAType::EMA).unwrap();
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| config.stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for &value in &input {
                    for stream in &mut streams {
                        black_box(stream.next(black_box(value)).unwrap());
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

pub(crate) fn bench_bands(c: &mut Criterion) {
    bench_accbands(c);
    bench_bbands(c);
}
