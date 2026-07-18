//! First-tranche `ta-core` indicator benchmarks.
//!
//! These benchmarks exercise the public Rust APIs designed for the foundation
//! tranche: compact zero-copy kernels plus selected padded convenience wrappers.
//! Fixtures and reusable output buffers are allocated outside `b.iter()` unless
//! the wrapper allocation itself is the behavior under measurement.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use ta_core::{
    math_operators::{ADD, MINMAX, SUM},
    math_transform::SQRT,
    overlap::{
        EMA_vec, MAType, SMA_vec, T3_with_default_vfactor, DEMA, EMA, MA, SMA, TEMA, TRIMA, WMA,
    },
    price_transform::{AVGDEV, AVGPRICE},
    statistic::{
        BETA_vec, CORREL_vec, LINEARREG_ANGLE_vec, LINEARREG_INTERCEPT_vec, LINEARREG_SLOPE_vec,
        LINEARREG_vec, STDDEV_vec, TSF_vec, VAR_vec, BETA, CORREL, LINEARREG, LINEARREG_ANGLE,
        LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,
    },
    volatility::{ATR, NATR, TRANGE},
    volume::{AD, ADOSC, OBV},
    Float,
};

const SIZES: &[usize] = &[1_024, 16_384, 65_536];
const PERIOD: usize = 20;
const STATISTIC_PERIODS: &[usize] = &[5, 20, 100, 500];
const ADOSC_FAST_PERIOD: usize = 3;
const ADOSC_SLOW_PERIOD: usize = 10;

fn series_fixture(size: usize) -> Vec<Float> {
    (0..size)
        .map(|idx| ((idx % 997) as Float + 1.0 as Float) * 0.5 as Float)
        .collect()
}

fn paired_fixture(size: usize) -> (Vec<Float>, Vec<Float>) {
    let left = series_fixture(size);
    let right = left
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 17) as Float + 1.0 as Float)
        .collect();
    (left, right)
}

fn ohlc_fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    let close = series_fixture(size);
    let open: Vec<Float> = close
        .iter()
        .enumerate()
        .map(|(idx, value)| *value + (idx % 5) as Float * 0.01 as Float)
        .collect();
    let high: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::max(*open, *close) + 1.0 as Float)
        .collect();
    let low: Vec<Float> = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::min(*open, *close) - 1.0 as Float)
        .collect();

    (open, high, low, close)
}

fn hlcv_fixture(size: usize) -> (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>) {
    let (_open, high, low, close) = ohlc_fixture(size);
    let volume = (0..size)
        .map(|idx| ((idx % 1_000) + 1) as Float * 10.0 as Float)
        .collect();
    (high, low, close, volume)
}

fn bench_overlap_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap/sma");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SMA_vec", size), &size, |b, &size| {
            let prices = series_fixture(size);

            b.iter(|| {
                let output = SMA_vec(black_box(prices.as_slice()), black_box(PERIOD))
                    .expect("valid SMA benchmark fixture");
                black_box(output);
            });
        });
    }

    group.finish();
}

fn bench_overlap_moving_averages(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/overlap/moving_averages");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("EMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = EMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid EMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("EMA_vec", size), &size, |b, &size| {
            let prices = series_fixture(size);

            b.iter(|| {
                let output = EMA_vec(black_box(prices.as_slice()), black_box(PERIOD))
                    .expect("valid EMA benchmark fixture");
                black_box(output);
            });
        });

        group.bench_with_input(BenchmarkId::new("WMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = WMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid WMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("TRIMA_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = TRIMA(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid TRIMA benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("DEMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = DEMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid DEMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("TEMA_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = TEMA(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid TEMA benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("T3_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = T3_with_default_vfactor(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid T3 benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("MA_EMA_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = MA(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(MAType::EMA),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid MA benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );
    }

    group.finish();
}

fn bench_price_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/price_transform");

    for &size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("AVGPRICE_compact", size),
            &size,
            |b, &size| {
                let (open, high, low, close) = ohlc_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = AVGPRICE(
                        black_box(open.as_slice()),
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AVGPRICE benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("AVGDEV_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = AVGDEV(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid AVGDEV benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );
    }

    group.finish();
}

fn bench_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/volatility");

    for &size in SIZES {
        group.bench_with_input(
            BenchmarkId::new("TRANGE_compact", size),
            &size,
            |b, &size| {
                let (_open, high, low, close) = ohlc_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = TRANGE(
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid TRANGE benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("ATR_compact", size), &size, |b, &size| {
            let (_open, high, low, close) = ohlc_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = ATR(
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid ATR benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("NATR_compact", size), &size, |b, &size| {
            let (_open, high, low, close) = ohlc_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = NATR(
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid NATR benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/volume");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("AD_compact", size), &size, |b, &size| {
            let (high, low, close, volume) = hlcv_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = AD(
                    black_box(high.as_slice()),
                    black_box(low.as_slice()),
                    black_box(close.as_slice()),
                    black_box(volume.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid AD benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("OBV_compact", size), &size, |b, &size| {
            let (_high, _low, close, volume) = hlcv_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = OBV(
                    black_box(close.as_slice()),
                    black_box(volume.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid OBV benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("ADOSC_compact", size),
            &size,
            |b, &size| {
                let (high, low, close, volume) = hlcv_fixture(size);
                let mut output = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = ADOSC(
                        black_box(high.as_slice()),
                        black_box(low.as_slice()),
                        black_box(close.as_slice()),
                        black_box(volume.as_slice()),
                        black_box(ADOSC_FAST_PERIOD),
                        black_box(ADOSC_SLOW_PERIOD),
                        black_box(output.as_mut_slice()),
                    )
                    .expect("valid ADOSC benchmark fixture");
                    black_box(range);
                    black_box(output.as_slice());
                });
            },
        );
    }

    group.finish();
}

fn bench_math_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_transform");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("SQRT_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SQRT(
                    black_box(prices.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SQRT benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });
    }

    group.finish();
}

fn bench_math_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/math_operators");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::new("ADD_compact", size), &size, |b, &size| {
            let (left, right) = paired_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = ADD(
                    black_box(left.as_slice()),
                    black_box(right.as_slice()),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid ADD benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(BenchmarkId::new("SUM_compact", size), &size, |b, &size| {
            let prices = series_fixture(size);
            let mut output = vec![0.0 as Float; size];

            b.iter(|| {
                let range = SUM(
                    black_box(prices.as_slice()),
                    black_box(PERIOD),
                    black_box(output.as_mut_slice()),
                )
                .expect("valid SUM benchmark fixture");
                black_box(range);
                black_box(output.as_slice());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("MINMAX_compact", size),
            &size,
            |b, &size| {
                let prices = series_fixture(size);
                let mut min = vec![0.0 as Float; size];
                let mut max = vec![0.0 as Float; size];

                b.iter(|| {
                    let range = MINMAX(
                        black_box(prices.as_slice()),
                        black_box(PERIOD),
                        black_box(min.as_mut_slice()),
                        black_box(max.as_mut_slice()),
                    )
                    .expect("valid MINMAX benchmark fixture");
                    black_box(range);
                    black_box(min.as_slice());
                    black_box(max.as_slice());
                });
            },
        );
    }

    group.finish();
}

fn bench_statistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("ta_core/statistic");

    macro_rules! bench_variance {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let real = series_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(1.0 as Float),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let real = series_fixture($size);
                    b.iter(|| {
                        let output = $vec(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(1.0 as Float),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    macro_rules! bench_single {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let real = series_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real.as_slice()),
                            black_box($period),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let real = series_fixture($size);
                    b.iter(|| {
                        let output = $vec(black_box(real.as_slice()), black_box($period))
                            .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    macro_rules! bench_paired {
        ($compact:ident, $vec:ident, $label:literal, $size:expr, $period:expr) => {
            group.bench_function(
                BenchmarkId::new(
                    concat!($label, "_compact"),
                    format!("{}/p{}", $size, $period),
                ),
                move |b| {
                    let (real0, real1) = paired_fixture($size);
                    let mut output = vec![0.0 as Float; $size];
                    b.iter(|| {
                        let range = $compact(
                            black_box(real0.as_slice()),
                            black_box(real1.as_slice()),
                            black_box($period),
                            black_box(output.as_mut_slice()),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(range);
                        black_box(output.as_slice());
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!($label, "_vec"), format!("{}/p{}", $size, $period)),
                move |b| {
                    let (real0, real1) = paired_fixture($size);
                    b.iter(|| {
                        let output = $vec(
                            black_box(real0.as_slice()),
                            black_box(real1.as_slice()),
                            black_box($period),
                        )
                        .expect(concat!("valid ", $label, " benchmark fixture"));
                        black_box(output);
                    });
                },
            );
        };
    }

    for &size in SIZES {
        for &period in STATISTIC_PERIODS {
            bench_variance!(VAR, VAR_vec, "VAR", size, period);
            bench_variance!(STDDEV, STDDEV_vec, "STDDEV", size, period);
            bench_paired!(CORREL, CORREL_vec, "CORREL", size, period);
            bench_paired!(BETA, BETA_vec, "BETA", size, period);
            bench_single!(LINEARREG, LINEARREG_vec, "LINEARREG", size, period);
            bench_single!(
                LINEARREG_SLOPE,
                LINEARREG_SLOPE_vec,
                "LINEARREG_SLOPE",
                size,
                period
            );
            bench_single!(
                LINEARREG_INTERCEPT,
                LINEARREG_INTERCEPT_vec,
                "LINEARREG_INTERCEPT",
                size,
                period
            );
            bench_single!(
                LINEARREG_ANGLE,
                LINEARREG_ANGLE_vec,
                "LINEARREG_ANGLE",
                size,
                period
            );
            bench_single!(TSF, TSF_vec, "TSF", size, period);
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overlap_sma,
    bench_overlap_moving_averages,
    bench_price_transform,
    bench_volatility,
    bench_volume,
    bench_math_transform,
    bench_math_operators,
    bench_statistic
);
criterion_main!(benches);
