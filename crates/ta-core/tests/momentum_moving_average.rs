#[path = "fixtures/moving_average_momentum_reference.rs"]
mod reference;

use ta_core::inventory::{function, ImplementationStatus};
use ta_core::momentum::{
    APOBatchRunner, APOConfig, APOStream, MACDBatchRunner, MACDConfig, MACDEXTBatchRunner,
    MACDEXTConfig, MACDEXTStream, MACDFIXBatchRunner, MACDFIXConfig, MACDFIXStream, MACDStream,
    MACDValuesMut, PPOBatchRunner, PPOConfig, PPOStream, TRIXBatchRunner, TRIXConfig, TRIXStream,
    APO, MACD, MACDEXT, MACDFIX, PPO, TRIX,
};
use ta_core::overlap::{MAConfig, PeriodMAType};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const TOLERANCE: Float = 2e-3;
#[cfg(not(feature = "f32"))]
const TOLERANCE: Float = 2e-11;

fn real_fixture() -> Vec<Float> {
    reference::REAL
        .iter()
        .map(|&value| value as Float)
        .collect()
}

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

fn assert_slice_close(actual: &[Float], expected: &[f64]) {
    assert!(actual.len() >= expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected as Float);
    }
}

fn assert_full_seam<C, B, S>()
where
    C: IndicatorConfig<BatchRunner = B, Stream = S>,
    B: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn inventory_and_public_types_cover_all_six_definitions() {
    for name in ["APO", "PPO", "MACD", "MACDEXT", "MACDFIX", "TRIX"] {
        assert_eq!(
            function(name).expect("catalogue entry").status,
            ImplementationStatus::Implemented
        );
    }
    assert_full_seam::<APOConfig, APOBatchRunner, APOStream>();
    assert_full_seam::<PPOConfig, PPOBatchRunner, PPOStream>();
    assert_full_seam::<MACDConfig, MACDBatchRunner, MACDStream>();
    assert_full_seam::<MACDEXTConfig, MACDEXTBatchRunner, MACDEXTStream>();
    assert_full_seam::<MACDFIXConfig, MACDFIXBatchRunner, MACDFIXStream>();
    assert_full_seam::<TRIXConfig, TRIXBatchRunner, TRIXStream>();
}

#[test]
fn pinned_reference_prefixes_match_all_result_shapes() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    let real = real_fixture();
    let apo = APOConfig::new(
        reference::FAST_PERIOD,
        reference::SLOW_PERIOD,
        PeriodMAType::EMA,
    )
    .unwrap()
    .compute(&real)
    .unwrap();
    let ppo = PPOConfig::new(
        reference::FAST_PERIOD,
        reference::SLOW_PERIOD,
        PeriodMAType::EMA,
    )
    .unwrap()
    .compute(&real)
    .unwrap();
    let macd = MACDConfig::new(
        reference::FAST_PERIOD,
        reference::SLOW_PERIOD,
        reference::SIGNAL_PERIOD,
    )
    .unwrap()
    .compute(&real)
    .unwrap();
    let trix = TRIXConfig::new(reference::TRIX_PERIOD)
        .unwrap()
        .compute(&real)
        .unwrap();

    assert_eq!(apo.range(), OutputRange::new(10, 30));
    assert_eq!(ppo.range(), OutputRange::new(10, 30));
    assert_eq!(macd.range(), OutputRange::new(13, 27));
    assert_eq!(trix.range(), OutputRange::new(13, 27));
    assert_slice_close(apo.values(), reference::APO_EMA_PREFIX);
    assert_slice_close(ppo.values(), reference::PPO_EMA_PREFIX);
    assert_slice_close(&macd.values().macd, reference::MACD_PREFIX);
    assert_slice_close(&macd.values().signal, reference::MACD_SIGNAL_PREFIX);
    assert_slice_close(&macd.values().histogram, reference::MACD_HISTOGRAM_PREFIX);
    assert_slice_close(trix.values(), reference::TRIX_PREFIX);
}

#[test]
fn default_extended_and_fixed_macd_configurations_are_equivalent() {
    let real = real_fixture();
    let standard = MACDConfig::default().compute(&real).unwrap();
    let extended = MACDEXTConfig::default().compute(&real).unwrap();
    let fixed = MACDFIXConfig::default().compute(&real).unwrap();
    assert_eq!(standard.range(), extended.range());
    assert_eq!(standard.range(), fixed.range());
    assert_eq!(standard.values(), extended.values());
    assert_eq!(standard.values(), fixed.values());
}

#[test]
fn explicit_configurations_preserve_periods_kinds_and_lookbacks() {
    let apo = APOConfig::new(5, 11, PeriodMAType::KAMA).unwrap();
    let ppo = PPOConfig::new(5, 11, PeriodMAType::T3).unwrap();
    let macd = MACDConfig::new(5, 11, 4).unwrap();
    let extended = MACDEXTConfig::new(
        5,
        PeriodMAType::KAMA,
        11,
        PeriodMAType::TRIMA,
        4,
        PeriodMAType::WMA,
    )
    .unwrap();
    let fixed = MACDFIXConfig::new(4).unwrap();
    let trix = TRIXConfig::new(5).unwrap();

    assert_eq!(
        (apo.fast_period(), apo.slow_period(), apo.ma_type()),
        (5, 11, PeriodMAType::KAMA)
    );
    assert_eq!(
        (ppo.fast_period(), ppo.slow_period(), ppo.ma_type()),
        (5, 11, PeriodMAType::T3)
    );
    assert_eq!(
        (macd.fast_period(), macd.slow_period(), macd.signal_period()),
        (5, 11, 4)
    );
    assert_eq!(extended.fast_type(), PeriodMAType::KAMA);
    assert_eq!(extended.slow_type(), PeriodMAType::TRIMA);
    assert_eq!(extended.signal_type(), PeriodMAType::WMA);
    assert_eq!(
        (
            fixed.fast_period(),
            fixed.slow_period(),
            fixed.signal_period()
        ),
        (12, 26, 4)
    );
    assert_eq!(trix.period(), 5);
    assert_eq!(apo.lookback(), 11);
    assert_eq!(ppo.lookback(), 60);
    assert_eq!(macd.lookback(), 13);
    assert_eq!(extended.lookback(), 13);
    assert_eq!(trix.lookback(), 13);
}

#[test]
fn every_period_ma_kind_executes_in_pair_and_extended_macd_sweeps() {
    let real = real_fixture();
    let kinds = [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
        PeriodMAType::T3,
        PeriodMAType::KAMA,
    ];
    for kind in kinds {
        let apo = APOConfig::new(2, 3, kind).unwrap();
        let output = apo.compute(&real).unwrap();
        assert_eq!(output.range().beg_idx, apo.lookback());
        let extended = MACDEXTConfig::new(2, kind, 3, kind, 2, kind).unwrap();
        let output = extended.compute(&real).unwrap();
        assert_eq!(output.range().beg_idx, extended.lookback());
    }
}

#[test]
fn histogram_scaling_flat_and_trend_invariants_hold() {
    let real = real_fixture();
    let scaled: Vec<Float> = real.iter().map(|value| value * 3.5 as Float).collect();
    let apo = APOConfig::new(5, 11, PeriodMAType::EMA)
        .unwrap()
        .compute(&real)
        .unwrap();
    let scaled_apo = APOConfig::new(5, 11, PeriodMAType::EMA)
        .unwrap()
        .compute(&scaled)
        .unwrap();
    let ppo = PPOConfig::new(5, 11, PeriodMAType::EMA)
        .unwrap()
        .compute(&real)
        .unwrap();
    let scaled_ppo = PPOConfig::new(5, 11, PeriodMAType::EMA)
        .unwrap()
        .compute(&scaled)
        .unwrap();
    for ((&absolute, &scaled_absolute), (&percentage, &scaled_percentage)) in apo
        .values()
        .iter()
        .zip(scaled_apo.values())
        .zip(ppo.values().iter().zip(scaled_ppo.values()))
    {
        assert_close(scaled_absolute, absolute * 3.5 as Float);
        assert_close(scaled_percentage, percentage);
    }

    let macd = MACDConfig::new(5, 11, 4).unwrap().compute(&real).unwrap();
    for ((&line, &signal), &histogram) in macd
        .values()
        .macd
        .iter()
        .zip(&macd.values().signal)
        .zip(&macd.values().histogram)
    {
        assert_close(histogram, line - signal);
    }

    let flat = vec![7.0 as Float; 80];
    assert!(APOConfig::default()
        .compute(&flat)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value == 0.0));
    assert!(PPOConfig::default()
        .compute(&flat)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value == 0.0));
    let flat_macd = MACDConfig::default().compute(&flat).unwrap();
    assert!(flat_macd.values().macd.iter().all(|&value| value == 0.0));
    assert!(flat_macd.values().signal.iter().all(|&value| value == 0.0));
    assert!(flat_macd
        .values()
        .histogram
        .iter()
        .all(|&value| value == 0.0));
    assert!(TRIXConfig::new(5)
        .unwrap()
        .compute(&flat)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value == 0.0));

    let trend: Vec<Float> = (0..80)
        .map(|index| 50.0 as Float + index as Float)
        .collect();
    assert!(APOConfig::default()
        .compute(&trend)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value > 0.0));
    assert!(PPOConfig::default()
        .compute(&trend)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value > 0.0));
    assert!(TRIXConfig::new(5)
        .unwrap()
        .compute(&trend)
        .unwrap()
        .values()
        .iter()
        .all(|&value| value > 0.0));
}

#[test]
fn caller_owned_macd_rejects_each_short_column_before_mutation() {
    let real = real_fixture();
    let count = real.len() - 13;
    for short_column in 0..3 {
        let mut macd = vec![-7.0 as Float; count];
        let mut signal = vec![-7.0 as Float; count];
        let mut histogram = vec![-7.0 as Float; count];
        match short_column {
            0 => macd.pop(),
            1 => signal.pop(),
            _ => histogram.pop(),
        };
        let before = (macd.clone(), signal.clone(), histogram.clone());
        assert!(MACD(&real, 5, 11, 4, &mut macd, &mut signal, &mut histogram).is_err());
        assert_eq!((macd, signal, histogram), before);
    }
}

#[test]
fn input_parameter_and_insufficient_data_errors_do_not_mutate_outputs() {
    let real = real_fixture();
    let mut output = [-3.0 as Float; 40];
    let before = output;
    assert!(APO(&real, 0, 11, PeriodMAType::EMA, &mut output).is_err());
    assert_eq!(output, before);
    assert!(PPO(&real, 11, 5, PeriodMAType::EMA, &mut output).is_err());
    assert_eq!(output, before);
    assert!(TRIX(&real, 100_001, &mut output).is_err());
    assert_eq!(output, before);

    let mut invalid = real.clone();
    invalid[9] = Float::NAN;
    assert!(APO(&invalid, 5, 11, PeriodMAType::EMA, &mut output).is_err());
    assert_eq!(output, before);
    assert!(TRIX(&invalid, 5, &mut output).is_err());
    assert_eq!(output, before);

    let short = [1.0 as Float; 4];
    assert!(matches!(
        TRIX(&short, 5, &mut output),
        Err(TalibError::InsufficientData { .. })
    ));
    assert_eq!(output, before);
}

#[test]
fn prepared_runners_replay_and_preserve_state_on_capacity_errors() {
    let real = real_fixture();
    let config = MACDConfig::new(5, 11, 4).unwrap();
    let count = real.len() - config.lookback();
    let mut runner = config.prepare_batch(real.len()).unwrap();
    let mut first = (
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
    );
    let range = runner
        .compute_into(
            &real,
            MACDValuesMut {
                macd: &mut first.0,
                signal: &mut first.1,
                histogram: &mut first.2,
            },
        )
        .unwrap();
    let mut second = (
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
    );
    runner
        .compute_into(
            &real,
            MACDValuesMut {
                macd: &mut second.0,
                signal: &mut second.1,
                histogram: &mut second.2,
            },
        )
        .unwrap();
    assert_eq!(range, OutputRange::new(config.lookback(), count));
    assert_eq!(first, second);

    let oversized = vec![1.0 as Float; real.len() + 1];
    let before = second.clone();
    assert!(matches!(
        runner.compute_into(
            &oversized,
            MACDValuesMut {
                macd: &mut second.0,
                signal: &mut second.1,
                histogram: &mut second.2,
            }
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert_eq!(second, before);
}

#[test]
fn every_definition_has_prepared_streaming_and_reset_parity() {
    let real = real_fixture();

    macro_rules! assert_single_output_paths {
        ($config:expr) => {{
            let config = $config;
            let batch = config.compute(&real).unwrap();
            let mut prepared = config.prepare_batch(real.len()).unwrap();
            let mut caller = vec![0.0 as Float; batch.range().nb_element];
            let range = prepared.compute_into(&real, &mut caller).unwrap();
            assert_eq!(range, batch.range());
            assert_eq!(caller, *batch.values());
            let before = caller.clone();
            let oversized = vec![1.0 as Float; real.len() + 1];
            assert!(matches!(
                prepared.compute_into(&oversized, &mut caller),
                Err(TalibError::PreparedCapacityExceeded { .. })
            ));
            assert_eq!(caller, before);

            let mut stream = config.stream().unwrap();
            let streamed: Vec<_> = real
                .iter()
                .copied()
                .filter_map(|input| stream.next(input).unwrap())
                .collect();
            assert_eq!(streamed, *batch.values());
            stream.reset();
            let replay: Vec<_> = real
                .iter()
                .copied()
                .filter_map(|input| stream.next(input).unwrap())
                .collect();
            assert_eq!(replay, streamed);
            stream.reset();
            for &input in &real[..20] {
                let _ = stream.next(input).unwrap();
            }
            let mut expected_stream = stream.clone();
            assert!(stream.next(Float::NAN).is_err());
            assert_eq!(
                stream.next(real[20]).unwrap(),
                expected_stream.next(real[20]).unwrap()
            );
        }};
    }

    assert_single_output_paths!(APOConfig::new(5, 11, PeriodMAType::EMA).unwrap());
    assert_single_output_paths!(PPOConfig::new(5, 11, PeriodMAType::EMA).unwrap());
    assert_single_output_paths!(TRIXConfig::new(5).unwrap());

    macro_rules! assert_macd_paths {
        ($config:expr) => {{
            let config = $config;
            let batch = config.compute(&real).unwrap();
            let count = batch.range().nb_element;
            let mut prepared = config.prepare_batch(real.len()).unwrap();
            let mut line = vec![0.0 as Float; count];
            let mut signal = vec![0.0 as Float; count];
            let mut histogram = vec![0.0 as Float; count];
            let range = prepared
                .compute_into(
                    &real,
                    MACDValuesMut {
                        macd: &mut line,
                        signal: &mut signal,
                        histogram: &mut histogram,
                    },
                )
                .unwrap();
            assert_eq!(range, batch.range());
            assert_eq!(line, batch.values().macd);
            assert_eq!(signal, batch.values().signal);
            assert_eq!(histogram, batch.values().histogram);
            let before = (line.clone(), signal.clone(), histogram.clone());
            let oversized = vec![1.0 as Float; real.len() + 1];
            assert!(matches!(
                prepared.compute_into(
                    &oversized,
                    MACDValuesMut {
                        macd: &mut line,
                        signal: &mut signal,
                        histogram: &mut histogram,
                    }
                ),
                Err(TalibError::PreparedCapacityExceeded { .. })
            ));
            assert_eq!((line.clone(), signal.clone(), histogram.clone()), before);

            let mut stream = config.stream().unwrap();
            let streamed: Vec<_> = real
                .iter()
                .copied()
                .filter_map(|input| stream.next(input).unwrap())
                .collect();
            assert_eq!(streamed.len(), count);
            for (index, value) in streamed.iter().enumerate() {
                assert_eq!(value.macd, batch.values().macd[index]);
                assert_eq!(value.signal, batch.values().signal[index]);
                assert_eq!(value.histogram, batch.values().histogram[index]);
            }
            stream.reset();
            let replay_count = real
                .iter()
                .copied()
                .filter_map(|input| stream.next(input).unwrap())
                .count();
            assert_eq!(replay_count, count);
            stream.reset();
            for &input in &real[..20] {
                let _ = stream.next(input).unwrap();
            }
            let mut expected_stream = stream.clone();
            assert!(stream.next(Float::NAN).is_err());
            assert_eq!(
                stream.next(real[20]).unwrap(),
                expected_stream.next(real[20]).unwrap()
            );
        }};
    }

    assert_macd_paths!(MACDConfig::new(5, 11, 4).unwrap());
    assert_macd_paths!(MACDEXTConfig::new(
        5,
        PeriodMAType::EMA,
        11,
        PeriodMAType::EMA,
        4,
        PeriodMAType::EMA,
    )
    .unwrap());
    assert_macd_paths!(MACDFIXConfig::new(4).unwrap());
}

fn collect_macd_stream(config: &MACDConfig, real: &[Float]) -> Vec<(Float, Float, Float)> {
    let mut stream = config.stream().unwrap();
    real.iter()
        .copied()
        .filter_map(|input| {
            stream
                .next(input)
                .unwrap()
                .map(|value| (value.macd, value.signal, value.histogram))
        })
        .collect()
}

#[test]
fn streams_match_batch_reset_replay_and_preserve_state_on_error() {
    let real = real_fixture();
    let config = MACDConfig::new(5, 11, 4).unwrap();
    let batch = config.compute(&real).unwrap();
    let streamed = collect_macd_stream(&config, &real);
    assert_eq!(streamed.len(), batch.range().nb_element);
    for (stream, ((&macd, &signal), &histogram)) in streamed.iter().zip(
        batch
            .values()
            .macd
            .iter()
            .zip(&batch.values().signal)
            .zip(&batch.values().histogram),
    ) {
        assert_close(stream.0, macd);
        assert_close(stream.1, signal);
        assert_close(stream.2, histogram);
    }

    let mut stream = config.stream().unwrap();
    for &input in &real[..20] {
        let _ = stream.next(input).unwrap();
    }
    let before_error = stream.clone();
    assert!(stream.next(Float::NAN).is_err());
    let expected = before_error.clone().next(real[20]).unwrap();
    let actual = stream.next(real[20]).unwrap();
    assert_eq!(actual, expected);
    stream.reset();
    let replay: Vec<_> = real
        .iter()
        .copied()
        .filter_map(|input| stream.next(input).unwrap())
        .collect();
    assert_eq!(replay.len(), batch.range().nb_element);

    let trix = TRIXConfig::new(5).unwrap();
    let batch = trix.compute(&real).unwrap();
    let mut stream = trix.stream().unwrap();
    let streamed: Vec<_> = real
        .iter()
        .copied()
        .filter_map(|input| stream.next(input).unwrap())
        .collect();
    assert_eq!(streamed, *batch.values());
}

#[test]
fn free_functions_and_extended_output_columns_are_all_observable() {
    let real = real_fixture();
    let pair_count = real.len() - 10;
    let mut apo = vec![0.0 as Float; pair_count];
    let mut ppo = vec![0.0 as Float; pair_count];
    assert_eq!(
        APO(&real, 5, 11, PeriodMAType::EMA, &mut apo).unwrap(),
        OutputRange::new(10, pair_count)
    );
    assert_eq!(
        PPO(&real, 5, 11, PeriodMAType::EMA, &mut ppo).unwrap(),
        OutputRange::new(10, pair_count)
    );

    let count = real.len() - 13;
    let mut standard = (
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
        vec![0.0 as Float; count],
    );
    let mut extended = standard.clone();
    let mut fixed = standard.clone();
    MACD(
        &real,
        5,
        11,
        4,
        &mut standard.0,
        &mut standard.1,
        &mut standard.2,
    )
    .unwrap();
    MACDEXT(
        &real,
        5,
        PeriodMAType::EMA,
        11,
        PeriodMAType::EMA,
        4,
        PeriodMAType::EMA,
        &mut extended.0,
        &mut extended.1,
        &mut extended.2,
    )
    .unwrap();
    assert_eq!(standard, extended);

    let fixed_count = real.len() - 28;
    fixed.0.resize(fixed_count, 0.0);
    fixed.1.resize(fixed_count, 0.0);
    fixed.2.resize(fixed_count, 0.0);
    assert_eq!(
        MACDFIX(&real, 4, &mut fixed.0, &mut fixed.1, &mut fixed.2).unwrap(),
        OutputRange::new(28, fixed_count)
    );
}

#[test]
fn period_one_signal_and_trix_have_exact_change_semantics() {
    let real = real_fixture();
    let signal_one = MACDConfig::new(5, 11, 1).unwrap().compute(&real).unwrap();
    assert_eq!(signal_one.values().macd, signal_one.values().signal);
    assert!(signal_one
        .values()
        .histogram
        .iter()
        .all(|&value| value == 0.0));

    let source = [2.0 as Float, 4.0, 2.0, 0.0, 3.0];
    let trix = TRIXConfig::new(1).unwrap().compute(&source).unwrap();
    assert_eq!(trix.range(), OutputRange::new(1, 4));
    assert_eq!(trix.values(), &[100.0 as Float, -50.0, -100.0, 0.0]);
}

#[test]
fn ma_selector_has_only_qualified_period_based_variants() {
    for kind in [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
        PeriodMAType::T3,
        PeriodMAType::KAMA,
    ] {
        assert!(MAConfig::new(5, kind).is_ok());
    }
}
