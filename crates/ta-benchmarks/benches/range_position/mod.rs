use super::support::{
    ohlc_fixture, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS, SWEEP_PERIODS, UNIVERSE_INSTRUMENTS,
    WORKERS,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use ta_core::momentum::{
    AROONConfig, AROONOSCConfig, AROONStream, AROONValuesMut, AroonInput, AroonTick, STOCHConfig,
    STOCHFConfig, STOCHFValuesMut, STOCHRSIConfig, STOCHRSIValuesMut, STOCHValuesMut,
    StochasticInput, WILLRConfig,
};
use ta_core::overlap::PeriodMAType;
use ta_core::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const KIND_SWEEP: &[PeriodMAType] = &[
    PeriodMAType::SMA,
    PeriodMAType::EMA,
    PeriodMAType::WMA,
    PeriodMAType::DEMA,
    PeriodMAType::TEMA,
    PeriodMAType::TRIMA,
    PeriodMAType::T3,
    PeriodMAType::KAMA,
];

pub(crate) fn bench_range_position_execution(c: &mut Criterion) {
    let mut one_shot = c.benchmark_group("indicator_execution/expanded/range_position");

    for &size in SIZES {
        let ohlc = ohlc_fixture(size);
        let high = ohlc.high.as_slice();
        let low = ohlc.low.as_slice();
        let close = ohlc.close.as_slice();

        let aroon = AROONConfig::new(14).unwrap();
        let aroon_count = size - aroon.lookback();
        let mut down = vec![0.0 as Float; aroon_count];
        let mut up = vec![0.0 as Float; aroon_count];
        one_shot.throughput(Throughput::Elements(aroon_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("AROON/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        aroon
                            .compute_into(
                                AroonInput {
                                    high: black_box(high),
                                    low: black_box(low),
                                },
                                AROONValuesMut {
                                    down: black_box(&mut down),
                                    up: black_box(&mut up),
                                },
                            )
                            .unwrap(),
                    );
                });
            },
        );

        let aroon_osc = AROONOSCConfig::new(14).unwrap();
        let osc_count = size - aroon_osc.lookback();
        let mut osc = vec![0.0 as Float; osc_count];
        one_shot.throughput(Throughput::Elements(osc_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("AROONOSC/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        aroon_osc
                            .compute_into(
                                AroonInput {
                                    high: black_box(high),
                                    low: black_box(low),
                                },
                                black_box(&mut osc),
                            )
                            .unwrap(),
                    );
                });
            },
        );

        let stoch = STOCHConfig::new(14, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA).unwrap();
        let stoch_count = size - stoch.lookback();
        let mut slow_k = vec![0.0 as Float; stoch_count];
        let mut slow_d = vec![0.0 as Float; stoch_count];
        one_shot.throughput(Throughput::Elements(stoch_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("STOCH/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        stoch
                            .compute_into(
                                StochasticInput {
                                    high: black_box(high),
                                    low: black_box(low),
                                    close: black_box(close),
                                },
                                STOCHValuesMut {
                                    slow_k: black_box(&mut slow_k),
                                    slow_d: black_box(&mut slow_d),
                                },
                            )
                            .unwrap(),
                    );
                });
            },
        );

        let stochf = STOCHFConfig::new(14, 3, PeriodMAType::SMA).unwrap();
        let stochf_count = size - stochf.lookback();
        let mut fast_k = vec![0.0 as Float; stochf_count];
        let mut fast_d = vec![0.0 as Float; stochf_count];
        one_shot.throughput(Throughput::Elements(stochf_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("STOCHF/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        stochf
                            .compute_into(
                                StochasticInput {
                                    high: black_box(high),
                                    low: black_box(low),
                                    close: black_box(close),
                                },
                                STOCHFValuesMut {
                                    fast_k: black_box(&mut fast_k),
                                    fast_d: black_box(&mut fast_d),
                                },
                            )
                            .unwrap(),
                    );
                });
            },
        );

        let stochrsi = STOCHRSIConfig::new(14, 14, 3, PeriodMAType::SMA).unwrap();
        let rsi_count = size - stochrsi.lookback();
        let mut rsi_k = vec![0.0 as Float; rsi_count];
        let mut rsi_d = vec![0.0 as Float; rsi_count];
        one_shot.throughput(Throughput::Elements(rsi_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("STOCHRSI/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        stochrsi
                            .compute_into(
                                black_box(close),
                                STOCHRSIValuesMut {
                                    fast_k: black_box(&mut rsi_k),
                                    fast_d: black_box(&mut rsi_d),
                                },
                            )
                            .unwrap(),
                    );
                });
            },
        );

        let willr = WILLRConfig::new(14).unwrap();
        let willr_count = size - willr.lookback();
        let mut willr_out = vec![0.0 as Float; willr_count];
        one_shot.throughput(Throughput::Elements(willr_count as u64));
        one_shot.bench_with_input(
            BenchmarkId::new("WILLR/caller_compact", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        willr
                            .compute_into(
                                StochasticInput {
                                    high: black_box(high),
                                    low: black_box(low),
                                    close: black_box(close),
                                },
                                black_box(&mut willr_out),
                            )
                            .unwrap(),
                    );
                });
            },
        );
    }
    one_shot.finish();

    // Build long-lived OHLC vectors so StochasticInput references stay valid
    // for the duration of each benchmark closure.
    let repeated_ohlc = ohlc_fixture(REPEATED_SERIES_LEN);
    let stochastic_repeated = StochasticInput {
        high: repeated_ohlc.high.as_slice(),
        low: repeated_ohlc.low.as_slice(),
        close: repeated_ohlc.close.as_slice(),
    };

    let stochf_ohlc: Vec<(Vec<Float>, Vec<Float>, Vec<Float>)> = (0..WORKERS)
        .map(|_| {
            let o = ohlc_fixture(REPEATED_SERIES_LEN);
            (o.high, o.low, o.close)
        })
        .collect();
    let stochf_inputs: Vec<StochasticInput<'_>> = stochf_ohlc
        .iter()
        .map(|(high, low, close)| StochasticInput {
            high: high.as_slice(),
            low: low.as_slice(),
            close: close.as_slice(),
        })
        .collect();

    let stream_inputs: Vec<Vec<AroonTick>> = (0..STREAM_INSTRUMENTS)
        .map(|_| {
            let o = ohlc_fixture(REPEATED_SERIES_LEN);
            o.high
                .iter()
                .zip(&o.low)
                .map(|(&high, &low)| AroonTick { high, low })
                .collect()
        })
        .collect();

    let close_universe: Vec<Vec<Float>> = (0..UNIVERSE_INSTRUMENTS)
        .map(|_| ohlc_fixture(REPEATED_SERIES_LEN).close)
        .collect();

    let mut workloads = c.benchmark_group("indicator_execution/expanded/range_position_workloads");

    // Kind/Period sweep across all eight PeriodMAType kinds × four Periods,
    // exercising the qualified MA dispatcher for STOCH (slow_k + slow_d).
    workloads.throughput(Throughput::Elements(
        (KIND_SWEEP.len() * SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("STOCH/kind_period_sweep/owned_compact", |b| {
        b.iter(|| {
            for &kind in KIND_SWEEP {
                for &slow_k in SWEEP_PERIODS {
                    let slow_d = slow_k;
                    let config = STOCHConfig::new(5, slow_k, kind, slow_d, kind).unwrap();
                    black_box(config.compute(stochastic_repeated).unwrap());
                }
            }
        });
    });

    // Universe prepared STOCHRSI runner reusing one Prepared Batch Runner.
    let stochrsi = STOCHRSIConfig::new(14, 14, 3, PeriodMAType::SMA).unwrap();
    let stochrsi_count = REPEATED_SERIES_LEN - stochrsi.lookback();
    let mut rsi_k = vec![0.0 as Float; stochrsi_count];
    let mut rsi_d = vec![0.0 as Float; stochrsi_count];
    workloads.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("STOCHRSI/universe/prepared_runner", |b| {
        let mut runner = stochrsi.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        b.iter(|| {
            for instrument in &close_universe {
                black_box(
                    runner
                        .compute_into(
                            black_box(instrument.as_slice()),
                            STOCHRSIValuesMut {
                                fast_k: black_box(&mut rsi_k),
                                fast_d: black_box(&mut rsi_d),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });
    // Per-worker STOCHF Prepared Batch Runners.

    workloads.bench_function("STOCHF/per_worker/prepared_runners", |b| {
        let stochf = STOCHFConfig::new(14, 3, PeriodMAType::SMA).unwrap();
        let count = REPEATED_SERIES_LEN - stochf.lookback();
        let mut runners: Vec<_> = (0..WORKERS)
            .map(|_| stochf.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect();
        let mut outputs: Vec<_> = (0..WORKERS)
            .map(|_| (vec![0.0 as Float; count], vec![0.0 as Float; count]))
            .collect();
        b.iter(|| {
            for (worker, runner) in runners.iter_mut().enumerate() {
                let output = &mut outputs[worker];
                black_box(
                    runner
                        .compute_into(
                            black_box(stochf_inputs[worker]),
                            STOCHFValuesMut {
                                fast_k: black_box(&mut output.0),
                                fast_d: black_box(&mut output.1),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    // Independent AROON Streaming Computations for multi-instrument tick flood.
    workloads.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("AROON/multi_stream/ticks", |b| {
        b.iter_batched(
            || {
                (0..STREAM_INSTRUMENTS)
                    .map(|_| IndicatorConfig::stream(&AROONConfig::new(14).unwrap()).unwrap())
                    .collect::<Vec<AROONStream>>()
            },
            |mut streams| {
                for inputs in (0..REPEATED_SERIES_LEN)
                    .map(|tick_index| stream_inputs.iter().map(move |series| series[tick_index]))
                {
                    for (stream, input) in streams.iter_mut().zip(inputs) {
                        black_box(
                            StreamingComputation::<AROONConfig>::next(
                                black_box(stream),
                                black_box(input),
                            )
                            .unwrap(),
                        );
                    }
                }
                for stream in &mut streams {
                    stream.reset();
                }
            },
            BatchSize::LargeInput,
        );
    });
    workloads.finish();
}
