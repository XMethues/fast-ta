use super::support::{
    ohlc_fixture, OhlcFixture, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS, SWEEP_PERIODS,
    UNIVERSE_INSTRUMENTS, WORKERS,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use fast_ta::momentum::{
    BOPConfig, BOPInput, BOPTick, CCIConfig, CCIInput, CCITick, MFIConfig, MFIInput, MFITick,
    ULTOSCConfig, ULTOSCInput, ULTOSCTick,
};
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

const SIZES: &[usize] = &[64, 4_096, 65_536];

fn fixtures(count: usize, size: usize) -> Vec<OhlcFixture> {
    (0..count)
        .map(|instrument| {
            let mut fixture = ohlc_fixture(size);
            let offset = instrument as Float * 0.01 as Float;
            for value in &mut fixture.open {
                *value += offset;
            }
            for value in &mut fixture.high {
                *value += offset;
            }
            for value in &mut fixture.low {
                *value += offset;
            }
            for value in &mut fixture.close {
                *value += offset;
            }
            fixture
        })
        .collect()
}

#[inline]
fn bop_input(fixture: &OhlcFixture) -> BOPInput<'_> {
    BOPInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

#[inline]
fn cci_input(fixture: &OhlcFixture) -> CCIInput<'_> {
    CCIInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

#[inline]
fn mfi_input(fixture: &OhlcFixture) -> MFIInput<'_> {
    MFIInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
        volume: &fixture.volume,
    }
}

#[inline]
fn ultosc_input(fixture: &OhlcFixture) -> ULTOSCInput<'_> {
    ULTOSCInput {
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

pub(crate) fn bench_composite_momentum_execution(c: &mut Criterion) {
    let mut one_shot = c.benchmark_group("indicator_execution/expanded/composite_momentum");
    for &size in SIZES {
        let fixture = ohlc_fixture(size);
        one_shot.throughput(Throughput::Elements(size as u64));

        let config = BOPConfig::new();
        let mut caller = vec![0.0 as Float; size];
        one_shot.bench_with_input(BenchmarkId::new("BOP/caller_owned", size), &size, |b, _| {
            b.iter(|| {
                let range = IndicatorConfig::compute_into(
                    &config,
                    black_box(bop_input(&fixture)),
                    black_box(caller.as_mut_slice()),
                )
                .expect("valid BOP fixture");
                black_box((range, caller.as_slice()));
            });
        });
        one_shot.bench_with_input(
            BenchmarkId::new("BOP/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| {
                        black_box(
                            IndicatorConfig::compute(&config, black_box(bop_input(&fixture)))
                                .unwrap(),
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );
        let mut runner = config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("BOP/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = PreparedBatchRunner::<BOPConfig>::compute_into(
                        black_box(&mut runner),
                        black_box(bop_input(&fixture)),
                        black_box(caller.as_mut_slice()),
                    )
                    .unwrap();
                    black_box((range, caller.as_slice()));
                });
            },
        );

        let config = CCIConfig::new(14).unwrap();
        let mut caller = vec![0.0 as Float; size - config.lookback()];
        one_shot.bench_with_input(BenchmarkId::new("CCI/caller_owned", size), &size, |b, _| {
            b.iter(|| {
                let range = config
                    .compute_into(
                        black_box(cci_input(&fixture)),
                        black_box(caller.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, caller.as_slice()));
            });
        });
        one_shot.bench_with_input(
            BenchmarkId::new("CCI/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| black_box(config.compute(black_box(cci_input(&fixture))).unwrap()),
                    BatchSize::LargeInput,
                );
            },
        );
        let mut runner = config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("CCI/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = runner
                        .compute_into(
                            black_box(cci_input(&fixture)),
                            black_box(caller.as_mut_slice()),
                        )
                        .unwrap();
                    black_box((range, caller.as_slice()));
                });
            },
        );

        let config = MFIConfig::new(14).unwrap();
        let mut caller = vec![0.0 as Float; size - config.lookback()];
        one_shot.bench_with_input(BenchmarkId::new("MFI/caller_owned", size), &size, |b, _| {
            b.iter(|| {
                let range = config
                    .compute_into(
                        black_box(mfi_input(&fixture)),
                        black_box(caller.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, caller.as_slice()));
            });
        });
        one_shot.bench_with_input(
            BenchmarkId::new("MFI/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| black_box(config.compute(black_box(mfi_input(&fixture))).unwrap()),
                    BatchSize::LargeInput,
                );
            },
        );
        let mut runner = config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("MFI/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = runner
                        .compute_into(
                            black_box(mfi_input(&fixture)),
                            black_box(caller.as_mut_slice()),
                        )
                        .unwrap();
                    black_box((range, caller.as_slice()));
                });
            },
        );

        let config = ULTOSCConfig::new(7, 14, 28).unwrap();
        let mut caller = vec![0.0 as Float; size - config.lookback()];
        one_shot.bench_with_input(
            BenchmarkId::new("ULTOSC/caller_owned", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = config
                        .compute_into(
                            black_box(ultosc_input(&fixture)),
                            black_box(caller.as_mut_slice()),
                        )
                        .unwrap();
                    black_box((range, caller.as_slice()));
                });
            },
        );
        one_shot.bench_with_input(
            BenchmarkId::new("ULTOSC/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| black_box(config.compute(black_box(ultosc_input(&fixture))).unwrap()),
                    BatchSize::LargeInput,
                );
            },
        );
        let mut runner = config.prepare_batch(size).unwrap();
        one_shot.bench_with_input(
            BenchmarkId::new("ULTOSC/prepared_runner", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let range = runner
                        .compute_into(
                            black_box(ultosc_input(&fixture)),
                            black_box(caller.as_mut_slice()),
                        )
                        .unwrap();
                    black_box((range, caller.as_slice()));
                });
            },
        );
    }
    one_shot.finish();

    let universe = fixtures(UNIVERSE_INSTRUMENTS, REPEATED_SERIES_LEN);
    let mut repeated =
        c.benchmark_group("indicator_execution/expanded/composite_momentum_workloads");
    repeated.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    let config = BOPConfig::new();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN];
    repeated.bench_function("BOP/universe/caller_owned", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = config
                    .compute_into(
                        black_box(bop_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });
    let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    repeated.bench_function("BOP/universe/prepared_runner", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = runner
                    .compute_into(
                        black_box(bop_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });

    let config = CCIConfig::new(14).unwrap();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()];
    repeated.bench_function("CCI/universe/caller_owned", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = config
                    .compute_into(
                        black_box(cci_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });
    let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    repeated.bench_function("CCI/universe/prepared_runner", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = runner
                    .compute_into(
                        black_box(cci_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });

    let config = MFIConfig::new(14).unwrap();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()];
    repeated.bench_function("MFI/universe/caller_owned", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = config
                    .compute_into(
                        black_box(mfi_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });
    let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    repeated.bench_function("MFI/universe/prepared_runner", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = runner
                    .compute_into(
                        black_box(mfi_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });

    let config = ULTOSCConfig::new(7, 14, 28).unwrap();
    let mut output = vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()];
    repeated.bench_function("ULTOSC/universe/caller_owned", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = config
                    .compute_into(
                        black_box(ultosc_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });
    let mut runner = config.prepare_batch(REPEATED_SERIES_LEN).unwrap();
    repeated.bench_function("ULTOSC/universe/prepared_runner", |b| {
        b.iter(|| {
            for fixture in &universe {
                let range = runner
                    .compute_into(
                        black_box(ultosc_input(fixture)),
                        black_box(output.as_mut_slice()),
                    )
                    .unwrap();
                black_box((range, output.as_slice()));
            }
        });
    });

    let fixture = ohlc_fixture(REPEATED_SERIES_LEN);
    repeated.throughput(Throughput::Elements(
        (SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    let cci_configs = SWEEP_PERIODS
        .iter()
        .map(|&period| CCIConfig::new(period).unwrap())
        .collect::<Vec<_>>();
    let mut cci_outputs = cci_configs
        .iter()
        .map(|config| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("CCI/parameter_sweep/caller_owned", |b| {
        b.iter(|| {
            for (config, output) in cci_configs.iter().zip(&mut cci_outputs) {
                black_box(
                    config
                        .compute_into(black_box(cci_input(&fixture)), output)
                        .unwrap(),
                );
            }
        });
    });
    let mfi_configs = SWEEP_PERIODS
        .iter()
        .map(|&period| MFIConfig::new(period).unwrap())
        .collect::<Vec<_>>();
    let mut mfi_outputs = mfi_configs
        .iter()
        .map(|config| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("MFI/parameter_sweep/caller_owned", |b| {
        b.iter(|| {
            for (config, output) in mfi_configs.iter().zip(&mut mfi_outputs) {
                black_box(
                    config
                        .compute_into(black_box(mfi_input(&fixture)), output)
                        .unwrap(),
                );
            }
        });
    });
    let ultosc_configs = SWEEP_PERIODS
        .iter()
        .map(|&period| ULTOSCConfig::new(period, period * 2, period * 4).unwrap())
        .collect::<Vec<_>>();
    let mut ultosc_outputs = ultosc_configs
        .iter()
        .map(|config| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("ULTOSC/parameter_sweep/caller_owned", |b| {
        b.iter(|| {
            for (config, output) in ultosc_configs.iter().zip(&mut ultosc_outputs) {
                black_box(
                    config
                        .compute_into(black_box(ultosc_input(&fixture)), output)
                        .unwrap(),
                );
            }
        });
    });

    let workers = fixtures(WORKERS, REPEATED_SERIES_LEN);
    repeated.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));

    let config = BOPConfig::new();
    let mut runners = (0..WORKERS)
        .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN])
        .collect::<Vec<_>>();
    repeated.bench_function("BOP/per_worker/prepared_runners", |b| {
        b.iter(|| {
            for ((runner, fixture), output) in runners.iter_mut().zip(&workers).zip(&mut outputs) {
                black_box(
                    runner
                        .compute_into(black_box(bop_input(fixture)), output)
                        .unwrap(),
                );
            }
        });
    });

    let config = CCIConfig::new(14).unwrap();
    let mut runners = (0..WORKERS)
        .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("CCI/per_worker/prepared_runners", |b| {
        b.iter(|| {
            for ((runner, fixture), output) in runners.iter_mut().zip(&workers).zip(&mut outputs) {
                black_box(
                    runner
                        .compute_into(black_box(cci_input(fixture)), output)
                        .unwrap(),
                );
            }
        });
    });

    let config = MFIConfig::new(14).unwrap();
    let mut runners = (0..WORKERS)
        .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("MFI/per_worker/prepared_runners", |b| {
        b.iter(|| {
            for ((runner, fixture), output) in runners.iter_mut().zip(&workers).zip(&mut outputs) {
                black_box(
                    runner
                        .compute_into(black_box(mfi_input(fixture)), output)
                        .unwrap(),
                );
            }
        });
    });

    let config = ULTOSCConfig::new(7, 14, 28).unwrap();
    let mut runners = (0..WORKERS)
        .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
        .collect::<Vec<_>>();
    let mut outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; REPEATED_SERIES_LEN - config.lookback()])
        .collect::<Vec<_>>();
    repeated.bench_function("ULTOSC/per_worker/prepared_runners", |b| {
        b.iter(|| {
            for ((runner, fixture), output) in runners.iter_mut().zip(&workers).zip(&mut outputs) {
                black_box(
                    runner
                        .compute_into(black_box(ultosc_input(fixture)), output)
                        .unwrap(),
                );
            }
        });
    });

    let streams_fixture = fixtures(STREAM_INSTRUMENTS, REPEATED_SERIES_LEN);
    repeated.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));

    repeated.bench_function("BOP/streaming/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| BOPConfig::new().stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, fixture) in streams.iter_mut().zip(&streams_fixture) {
                        black_box(
                            stream
                                .next(BOPTick {
                                    open: fixture.open[index],
                                    high: fixture.high[index],
                                    low: fixture.low[index],
                                    close: fixture.close[index],
                                })
                                .unwrap(),
                        );
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    repeated.bench_function("CCI/streaming/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| CCIConfig::new(14).unwrap().stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, fixture) in streams.iter_mut().zip(&streams_fixture) {
                        black_box(
                            stream
                                .next(CCITick {
                                    open: fixture.open[index],
                                    high: fixture.high[index],
                                    low: fixture.low[index],
                                    close: fixture.close[index],
                                })
                                .unwrap(),
                        );
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    repeated.bench_function("MFI/streaming/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| MFIConfig::new(14).unwrap().stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, fixture) in streams.iter_mut().zip(&streams_fixture) {
                        black_box(
                            stream
                                .next(MFITick {
                                    open: fixture.open[index],
                                    high: fixture.high[index],
                                    low: fixture.low[index],
                                    close: fixture.close[index],
                                    volume: fixture.volume[index],
                                })
                                .unwrap(),
                        );
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });
    repeated.bench_function("ULTOSC/streaming/independent_streams", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| ULTOSCConfig::new(7, 14, 28).unwrap().stream().unwrap())
                    .collect::<Vec<_>>()
            },
            |mut streams| {
                for index in 0..REPEATED_SERIES_LEN {
                    for (stream, fixture) in streams.iter_mut().zip(&streams_fixture) {
                        black_box(
                            stream
                                .next(ULTOSCTick {
                                    high: fixture.high[index],
                                    low: fixture.low[index],
                                    close: fixture.close[index],
                                })
                                .unwrap(),
                        );
                    }
                }
            },
            BatchSize::LargeInput,
        );
    });

    repeated.finish();
}
