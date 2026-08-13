use super::support::{
    series_fixture, REPEATED_SERIES_LEN, STREAM_INSTRUMENTS, SWEEP_PERIODS, UNIVERSE_INSTRUMENTS,
    WORKERS,
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use fast_ta::momentum::{
    APOConfig, MACDConfig, MACDEXTConfig, MACDValuesMut, PPOConfig, TRIXConfig,
};
use fast_ta::overlap::PeriodMAType;
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

const SIZES: &[usize] = &[64, 4_096, 65_536];
const KINDS: &[PeriodMAType] = &[
    PeriodMAType::SMA,
    PeriodMAType::EMA,
    PeriodMAType::WMA,
    PeriodMAType::DEMA,
    PeriodMAType::TEMA,
    PeriodMAType::TRIMA,
    PeriodMAType::T3,
    PeriodMAType::KAMA,
];

pub(crate) fn bench_moving_average_momentum_execution(c: &mut Criterion) {
    let mut one_shot = c.benchmark_group("indicator_execution/expanded/moving_average_momentum");
    for &size in SIZES {
        let real = series_fixture(size, 31);
        one_shot.throughput(Throughput::Elements(size as u64));

        let apo = APOConfig::default();
        let mut pair_output = vec![0.0 as Float; size - apo.lookback()];
        one_shot.bench_with_input(BenchmarkId::new("APO/caller_owned", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    apo.compute_into(black_box(&real), black_box(&mut pair_output))
                        .unwrap(),
                )
            });
        });
        one_shot.bench_with_input(
            BenchmarkId::new("PPO/owned_compact", size),
            &size,
            |b, _| {
                let ppo = PPOConfig::default();
                b.iter_batched(
                    || (),
                    |_| black_box(ppo.compute(black_box(&real)).unwrap()),
                    BatchSize::LargeInput,
                );
            },
        );

        let macd = MACDConfig::default();
        let count = size - macd.lookback();
        let mut line = vec![0.0 as Float; count];
        let mut signal = vec![0.0 as Float; count];
        let mut histogram = vec![0.0 as Float; count];
        one_shot.bench_with_input(
            BenchmarkId::new("MACD/caller_owned", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        macd.compute_into(
                            black_box(&real),
                            MACDValuesMut {
                                macd: black_box(&mut line),
                                signal: black_box(&mut signal),
                                histogram: black_box(&mut histogram),
                            },
                        )
                        .unwrap(),
                    )
                });
            },
        );
        one_shot.bench_with_input(
            BenchmarkId::new("MACD/owned_compact", size),
            &size,
            |b, _| {
                b.iter_batched(
                    || (),
                    |_| black_box(macd.compute(black_box(&real)).unwrap()),
                    BatchSize::LargeInput,
                );
            },
        );

        let trix = TRIXConfig::new(14).expect("valid representative TRIX Period");
        let mut trix_output = vec![0.0 as Float; size - trix.lookback()];
        one_shot.bench_with_input(
            BenchmarkId::new("TRIX/caller_owned", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        trix.compute_into(black_box(&real), black_box(&mut trix_output))
                            .unwrap(),
                    )
                });
            },
        );
    }
    one_shot.finish();

    let real = series_fixture(REPEATED_SERIES_LEN, 31);
    let mut workloads =
        c.benchmark_group("indicator_execution/expanded/moving_average_momentum_workloads");

    workloads.throughput(Throughput::Elements(
        (KINDS.len() * SWEEP_PERIODS.len() * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("APO/kind_period_sweep/owned_compact", |b| {
        b.iter(|| {
            for &kind in KINDS {
                for &slow in SWEEP_PERIODS {
                    let fast = (slow / 2).max(2);
                    let config = APOConfig::new(fast, slow, kind).unwrap();
                    black_box(config.compute(black_box(&real)).unwrap());
                }
            }
        });
    });

    let universe: Vec<Vec<Float>> = (0..UNIVERSE_INSTRUMENTS)
        .map(|instrument| series_fixture(REPEATED_SERIES_LEN, 31 + instrument))
        .collect();
    let macd = MACDConfig::default();
    let count = REPEATED_SERIES_LEN - macd.lookback();
    let mut line = vec![0.0 as Float; count];
    let mut signal = vec![0.0 as Float; count];
    let mut histogram = vec![0.0 as Float; count];
    workloads.throughput(Throughput::Elements(
        (UNIVERSE_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("MACD/universe/prepared_runner", |b| {
        let mut runner = macd.prepare_batch(REPEATED_SERIES_LEN).unwrap();
        b.iter(|| {
            for instrument in &universe {
                black_box(
                    runner
                        .compute_into(
                            black_box(instrument),
                            MACDValuesMut {
                                macd: black_box(&mut line),
                                signal: black_box(&mut signal),
                                histogram: black_box(&mut histogram),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    workloads.throughput(Throughput::Elements((WORKERS * REPEATED_SERIES_LEN) as u64));
    workloads.bench_function("MACDEXT/per_worker/prepared_runners", |b| {
        let config = MACDEXTConfig::default();
        let mut runners: Vec<_> = (0..WORKERS)
            .map(|_| config.prepare_batch(REPEATED_SERIES_LEN).unwrap())
            .collect();
        let mut outputs: Vec<_> = (0..WORKERS)
            .map(|_| {
                (
                    vec![0.0 as Float; count],
                    vec![0.0 as Float; count],
                    vec![0.0 as Float; count],
                )
            })
            .collect();
        b.iter(|| {
            for (worker, runner) in runners.iter_mut().enumerate() {
                let output = &mut outputs[worker];
                black_box(
                    runner
                        .compute_into(
                            black_box(&universe[worker]),
                            MACDValuesMut {
                                macd: black_box(&mut output.0),
                                signal: black_box(&mut output.1),
                                histogram: black_box(&mut output.2),
                            },
                        )
                        .unwrap(),
                );
            }
        });
    });

    workloads.throughput(Throughput::Elements(
        (STREAM_INSTRUMENTS * REPEATED_SERIES_LEN) as u64,
    ));
    workloads.bench_function("TRIX/multi_stream/ticks", |b| {
        let config = TRIXConfig::default();
        let mut streams: Vec<_> = (0..STREAM_INSTRUMENTS)
            .map(|_| config.stream().unwrap())
            .collect();
        b.iter(|| {
            for inputs in (0..REPEATED_SERIES_LEN)
                .map(|index| universe.iter().map(move |series| series[index]))
            {
                for (stream, input) in streams.iter_mut().zip(inputs) {
                    black_box(stream.next(black_box(input)).unwrap());
                }
            }
            for stream in &mut streams {
                stream.reset();
            }
        });
    });

    workloads.finish();
}
