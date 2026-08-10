//! Non-blocking construction and throughput baselines for representative Pattern Recognition definitions.

mod support;

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use support::ohlc_fixture;
use ta_core::pattern_recognition::{
    CDL3WHITESOLDIERSConfig, CDLDOJIConfig, CDLENGULFINGConfig, CDLHIKKAKEConfig,
    CDLHIKKAKEMODConfig, CDLMORNINGSTARConfig, Candle, CandleInput, CandleSetting,
    CandleSettingType, CandleSettings, PatternSignal, Penetration,
};
use ta_core::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

const SIZES: &[usize] = &[256, 4_096, 65_536];
const LARGE_AVERAGE_PERIOD: usize = 200;
const PROVENANCE_FILE: &str = "pattern-recognition-provenance.txt";

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .replace('\n', " | ")
        })
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn git_dirty_state() -> &'static str {
    match Command::new("git").args(["status", "--porcelain"]).output() {
        Ok(output) if output.status.success() && output.stdout.is_empty() => "false",
        Ok(output) if output.status.success() => "true",
        _ => "unavailable",
    }
}

fn record_environment_provenance() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let commit = command_output("git", &["rev-parse", "HEAD"]);
        let dirty = git_dirty_state();
        let rustc = command_output("rustc", &["--version", "--verbose"]);
        let host = command_output("uname", &["-a"]);
        let parallelism = std::thread::available_parallelism()
            .map(|value| value.get().to_string())
            .unwrap_or_else(|_| "unavailable".to_owned());
        let provenance = format!(
            "suite=pattern_recognition\ncommit={commit}\ndirty={}\nrustc={rustc}\nhost={host}\nos={}\narch={}\nparallelism={parallelism}\nfloat_bits={}\ncriterion=0.8.2\nprofile=bench\nsizes=256,4096,65536\nlarge_average_period={LARGE_AVERAGE_PERIOD}\n",
            dirty,
            std::env::consts::OS,
            std::env::consts::ARCH,
            core::mem::size_of::<Float>() * 8,
        );
        eprintln!("{provenance}");

        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("target"));
        let criterion_dir = target_dir.join("criterion");
        if std::fs::create_dir_all(&criterion_dir).is_ok() {
            let _ = std::fs::write(criterion_dir.join(PROVENANCE_FILE), provenance);
        }
    });
}

fn large_period_settings() -> CandleSettings {
    let defaults = CandleSettings::default();
    let mut settings = defaults;
    for setting_type in CandleSettingType::ALL {
        let default = defaults.setting(setting_type);
        settings = settings.with_setting(
            setting_type,
            CandleSetting::new(default.range_kind(), LARGE_AVERAGE_PERIOD, default.factor())
                .expect("valid large-period Candle Setting"),
        );
    }
    settings
}

fn benchmark_construction<C, F>(c: &mut Criterion, name: &str, variant: &str, make_config: F)
where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C:
        IndicatorConfig<Input<'a> = CandleInput<'a>, OutputMut<'a> = &'a mut [PatternSignal]>,
    C::BatchRunner: PreparedBatchRunner<C>,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal>,
    F: Copy + Fn() -> C,
{
    let mut group = c.benchmark_group(format!("pattern_recognition/construction/{name}/{variant}"));
    group.bench_function("config", |b| b.iter(|| black_box(make_config())));
    group.bench_function("prepared_65536", |b| {
        b.iter(|| {
            black_box(
                make_config()
                    .prepare_batch(black_box(65_536))
                    .expect("valid prepared capacity"),
            )
        })
    });
    group.bench_function("stream", |b| {
        b.iter(|| black_box(make_config().stream().expect("valid Stream")))
    });
    group.finish();
}

fn benchmark_throughput<C>(c: &mut Criterion, name: &str, variant: &str, config: C)
where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C:
        IndicatorConfig<Input<'a> = CandleInput<'a>, OutputMut<'a> = &'a mut [PatternSignal]>,
    C::BatchRunner: PreparedBatchRunner<C>,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal>,
{
    let mut group = c.benchmark_group(format!("pattern_recognition/throughput/{name}/{variant}"));

    for &size in SIZES {
        let fixture = ohlc_fixture(size);
        let input = || CandleInput {
            open: fixture.open.as_slice(),
            high: fixture.high.as_slice(),
            low: fixture.low.as_slice(),
            close: fixture.close.as_slice(),
        };
        let output_len = size - config.lookback();
        let mut caller_output = vec![PatternSignal::NoMatch; output_len];
        let mut prepared_output = vec![PatternSignal::NoMatch; output_len];
        let mut prepared = config
            .prepare_batch(size)
            .expect("valid prepared Pattern Recognition capacity");
        let candles: Vec<_> = (0..size)
            .map(|index| Candle {
                open: fixture.open[index],
                high: fixture.high[index],
                low: fixture.low[index],
                close: fixture.close[index],
            })
            .collect();
        let mut stream = config.stream().expect("valid Pattern Recognition Stream");

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("owned", size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    config
                        .compute(black_box(input()))
                        .expect("valid owned Pattern Recognition fixture"),
                )
            })
        });
        group.bench_with_input(BenchmarkId::new("caller_owned", size), &size, |b, _| {
            b.iter(|| {
                let range = config
                    .compute_into(black_box(input()), black_box(caller_output.as_mut_slice()))
                    .expect("valid caller-owned Pattern Recognition fixture");
                black_box((range, caller_output.as_slice()));
            })
        });
        group.bench_with_input(BenchmarkId::new("prepared", size), &size, |b, _| {
            b.iter(|| {
                let range = prepared
                    .compute_into(
                        black_box(input()),
                        black_box(prepared_output.as_mut_slice()),
                    )
                    .expect("valid prepared Pattern Recognition fixture");
                black_box((range, prepared_output.as_slice()));
            })
        });
        group.bench_with_input(BenchmarkId::new("streaming", size), &size, |b, _| {
            b.iter(|| {
                stream.reset();
                for &candle in &candles {
                    black_box(
                        stream
                            .next(black_box(candle))
                            .expect("valid streaming Pattern Recognition fixture"),
                    );
                }
            })
        });
    }
    group.finish();
}

fn bench_pattern_recognition(c: &mut Criterion) {
    record_environment_provenance();

    benchmark_construction(c, "CDLENGULFING", "default", CDLENGULFINGConfig::default);
    benchmark_throughput(c, "CDLENGULFING", "default", CDLENGULFINGConfig::default());

    benchmark_construction(c, "CDLDOJI", "default", CDLDOJIConfig::default);
    benchmark_throughput(c, "CDLDOJI", "default", CDLDOJIConfig::default());
    benchmark_construction(c, "CDLDOJI", "period_200", || {
        CDLDOJIConfig::new(large_period_settings()).expect("valid large-period CDLDOJI")
    });
    benchmark_throughput(
        c,
        "CDLDOJI",
        "period_200",
        CDLDOJIConfig::new(large_period_settings()).expect("valid large-period CDLDOJI"),
    );

    benchmark_construction(
        c,
        "CDL3WHITESOLDIERS",
        "default",
        CDL3WHITESOLDIERSConfig::default,
    );
    benchmark_throughput(
        c,
        "CDL3WHITESOLDIERS",
        "default",
        CDL3WHITESOLDIERSConfig::default(),
    );
    benchmark_construction(c, "CDL3WHITESOLDIERS", "period_200", || {
        CDL3WHITESOLDIERSConfig::new(large_period_settings())
            .expect("valid large-period CDL3WHITESOLDIERS")
    });
    benchmark_throughput(
        c,
        "CDL3WHITESOLDIERS",
        "period_200",
        CDL3WHITESOLDIERSConfig::new(large_period_settings())
            .expect("valid large-period CDL3WHITESOLDIERS"),
    );

    benchmark_construction(
        c,
        "CDLMORNINGSTAR",
        "default",
        CDLMORNINGSTARConfig::default,
    );
    benchmark_throughput(
        c,
        "CDLMORNINGSTAR",
        "default",
        CDLMORNINGSTARConfig::default(),
    );
    benchmark_construction(c, "CDLMORNINGSTAR", "period_200", || {
        CDLMORNINGSTARConfig::new(
            large_period_settings(),
            Penetration::new(0.3 as Float).expect("valid pinned Penetration"),
        )
        .expect("valid large-period CDLMORNINGSTAR")
    });
    benchmark_throughput(
        c,
        "CDLMORNINGSTAR",
        "period_200",
        CDLMORNINGSTARConfig::new(
            large_period_settings(),
            Penetration::new(0.3 as Float).expect("valid pinned Penetration"),
        )
        .expect("valid large-period CDLMORNINGSTAR"),
    );

    benchmark_construction(c, "CDLHIKKAKE", "default", CDLHIKKAKEConfig::default);
    benchmark_throughput(c, "CDLHIKKAKE", "default", CDLHIKKAKEConfig::default());

    benchmark_construction(c, "CDLHIKKAKEMOD", "default", CDLHIKKAKEMODConfig::default);
    benchmark_throughput(
        c,
        "CDLHIKKAKEMOD",
        "default",
        CDLHIKKAKEMODConfig::default(),
    );
    benchmark_construction(c, "CDLHIKKAKEMOD", "period_200", || {
        CDLHIKKAKEMODConfig::new(large_period_settings()).expect("valid large-period CDLHIKKAKEMOD")
    });
    benchmark_throughput(
        c,
        "CDLHIKKAKEMOD",
        "period_200",
        CDLHIKKAKEMODConfig::new(large_period_settings())
            .expect("valid large-period CDLHIKKAKEMOD"),
    );
}

criterion_group!(benches, bench_pattern_recognition);
criterion_main!(benches);
