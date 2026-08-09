#[path = "fixtures/kama_mavp_reference.rs"]
mod reference;

use ta_core::overlap::{
    KAMABatchRunner, KAMAConfig, KAMAStream, MABatchRunner, MAConfig, MAVPBatchRunner, MAVPConfig,
    MAVPInput, MAVPStream, MAVPTick, PeriodMAType, KAMA, MA, MAVP,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const ABS_TOLERANCE: Float = 2.0e-4;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 2.0e-5;
#[cfg(not(feature = "f32"))]
const ABS_TOLERANCE: Float = 2.0e-12;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 2.0e-13;

fn assert_close(actual: Float, expected: f64, context: &str) {
    let expected = expected as Float;
    let difference = (actual - expected).abs();
    let tolerance = ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs());
    assert!(
        difference <= tolerance,
        "{context}: expected {expected}, got {actual}, difference {difference}, tolerance {tolerance}"
    );
}

fn assert_values_close(actual: &[Float], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(
            actual,
            expected,
            &format!("{context}, compact index {index}"),
        );
    }
}

fn floats(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn assert_execution_types<C, B, S>()
where
    C: IndicatorConfig<BatchRunner = B, Stream = S>,
    B: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn kama_matches_independent_decimal_reference_in_every_execution_mode_and_selector() {
    assert_eq!(
        reference::REFERENCE_METHOD,
        "Python Decimal, precision=50, direct definition"
    );
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "e64d2ac896c595f38d65e44c812efbfdac8a64cf"
    );
    assert_eq!(reference::TALIB_KAMA_SOURCE, "src/ta_func/ta_KAMA.c");
    assert_eq!(reference::TALIB_MAVP_SOURCE, "src/ta_func/ta_MAVP.c");
    assert_execution_types::<KAMAConfig, KAMABatchRunner, KAMAStream>();
    assert_execution_types::<MAConfig, MABatchRunner, ta_core::overlap::MAStream>();

    let input = floats(reference::INPUT);
    let config = KAMAConfig::new(reference::KAMA_PERIOD).unwrap();
    let selector = MAConfig::new(reference::KAMA_PERIOD, PeriodMAType::KAMA).unwrap();
    let range = OutputRange::new(reference::KAMA_PERIOD, reference::KAMA_EXPECTED.len());
    assert_eq!(config.period(), reference::KAMA_PERIOD);
    assert_eq!(config.lookback(), reference::KAMA_PERIOD);
    assert_eq!(selector.lookback(), config.lookback());

    let owned = config.compute(&input).unwrap();
    assert_eq!(owned.source_len(), input.len());
    assert_eq!(owned.range(), range);
    assert_values_close(owned.values(), reference::KAMA_EXPECTED, "KAMA owned");

    let mut caller = vec![-91.0 as Float; reference::KAMA_EXPECTED.len() + 1];
    assert_eq!(
        KAMA(&input, reference::KAMA_PERIOD, &mut caller).unwrap(),
        range
    );
    assert_values_close(
        &caller[..reference::KAMA_EXPECTED.len()],
        reference::KAMA_EXPECTED,
        "KAMA caller-owned",
    );
    assert_eq!(caller[reference::KAMA_EXPECTED.len()], -91.0 as Float);

    let mut selected = vec![0.0 as Float; reference::KAMA_EXPECTED.len()];
    assert_eq!(selector.compute_into(&input, &mut selected).unwrap(), range);
    assert_values_close(&selected, reference::KAMA_EXPECTED, "KAMA selector");
    let selector_owned = selector.compute(&input).unwrap();
    assert_eq!(selector_owned.range(), range);
    assert_values_close(
        selector_owned.values(),
        reference::KAMA_EXPECTED,
        "KAMA selector owned",
    );
    let mut selector_runner = selector.prepare_batch(input.len()).unwrap();
    selected.fill(0.0 as Float);
    assert_eq!(
        selector_runner.compute_into(&input, &mut selected).unwrap(),
        range
    );
    assert_values_close(
        &selected,
        reference::KAMA_EXPECTED,
        "KAMA selector prepared",
    );

    let mut runner = config.prepare_batch(input.len()).unwrap();
    let mut prepared = vec![0.0 as Float; reference::KAMA_EXPECTED.len()];
    for pass in ["first", "repeated"] {
        assert_eq!(runner.compute_into(&input, &mut prepared).unwrap(), range);
        assert_values_close(
            &prepared,
            reference::KAMA_EXPECTED,
            &format!("KAMA prepared {pass}"),
        );
    }

    let mut direct_stream = config.stream().unwrap();
    let streamed = input
        .iter()
        .copied()
        .filter_map(|tick| direct_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(&streamed, reference::KAMA_EXPECTED, "KAMA stream");
    direct_stream.reset();
    let replayed = input
        .iter()
        .copied()
        .filter_map(|tick| direct_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(streamed, replayed);

    let mut selector_stream = selector.stream().unwrap();
    let selected_streamed = input
        .iter()
        .copied()
        .filter_map(|tick| selector_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(
        &selected_streamed,
        reference::KAMA_EXPECTED,
        "KAMA selector stream",
    );
}

#[test]
fn kama_adaptivity_identity_and_flat_series_are_explicit() {
    let mut identity = [0.0 as Float; 4];
    let source = [1.0 as Float, 3.0, 2.0, 4.0];
    assert_eq!(
        KAMA(&source, 1, &mut identity).unwrap(),
        OutputRange::new(0, 4)
    );
    assert_eq!(identity, source);

    let mut trending = [0.0 as Float; 1];
    assert_eq!(
        KAMA(&[1.0 as Float, 2.0, 3.0, 4.0], 3, &mut trending).unwrap(),
        OutputRange::new(3, 1)
    );
    assert_close(trending[0], 31.0 / 9.0, "KAMA efficiency ratio one");

    let slow = 2.0_f64 / 31.0;
    let difference = 2.0_f64 / 3.0 - slow;
    let choppy_smoothing = (3.0_f64 / 5.0 * difference + slow).powi(2);
    let expected_choppy = 2.0 + (4.0 - 2.0) * choppy_smoothing;
    let mut choppy = [0.0 as Float; 1];
    KAMA(&source, 3, &mut choppy).unwrap();
    assert_close(
        choppy[0],
        expected_choppy,
        "KAMA fractional efficiency ratio",
    );
    assert!(trending[0] > choppy[0]);

    let flat = [42.0 as Float; 12];
    let flat_output = KAMAConfig::new(5).unwrap().compute(&flat).unwrap();
    assert!(flat_output
        .values()
        .iter()
        .all(|&value| value == 42.0 as Float));
}

#[test]
fn mavp_matches_independent_bounded_integer_reference_in_every_execution_mode() {
    assert_execution_types::<MAVPConfig, MAVPBatchRunner, MAVPStream>();
    let real = floats(reference::INPUT);
    let periods = reference::PERIOD_SELECTIONS;
    let config = MAVPConfig::new(
        reference::MAVP_MINIMUM_PERIOD,
        reference::MAVP_MAXIMUM_PERIOD,
        PeriodMAType::EMA,
    )
    .unwrap();
    assert_eq!(config.minimum_period(), 2);
    assert_eq!(config.maximum_period(), 5);
    assert_eq!(config.ma_type(), PeriodMAType::EMA);
    assert_eq!(config.lookback(), 4);
    let input = MAVPInput {
        real: &real,
        periods,
    };
    let range = OutputRange::new(4, reference::MAVP_EMA_EXPECTED.len());

    let owned = config.compute(input).unwrap();
    assert_eq!(owned.source_len(), real.len());
    assert_eq!(owned.range(), range);
    assert_values_close(owned.values(), reference::MAVP_EMA_EXPECTED, "MAVP owned");

    let mut caller = vec![-81.0 as Float; reference::MAVP_EMA_EXPECTED.len() + 1];
    assert_eq!(
        MAVP(&real, periods, 2, 5, PeriodMAType::EMA, &mut caller).unwrap(),
        range
    );
    assert_values_close(
        &caller[..reference::MAVP_EMA_EXPECTED.len()],
        reference::MAVP_EMA_EXPECTED,
        "MAVP caller-owned",
    );
    assert_eq!(caller[reference::MAVP_EMA_EXPECTED.len()], -81.0 as Float);

    let mut runner = config.prepare_batch(real.len()).unwrap();
    let mut prepared = vec![0.0 as Float; reference::MAVP_EMA_EXPECTED.len()];
    for pass in ["first", "repeated"] {
        assert_eq!(runner.compute_into(input, &mut prepared).unwrap(), range);
        assert_values_close(
            &prepared,
            reference::MAVP_EMA_EXPECTED,
            &format!("MAVP prepared {pass}"),
        );
    }

    let ticks = real.iter().copied().zip(periods.iter().copied());
    let mut stream = config.stream().unwrap();
    let streamed = ticks
        .clone()
        .filter_map(|(real, period)| stream.next(MAVPTick { real, period }).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(&streamed, reference::MAVP_EMA_EXPECTED, "MAVP stream");
    stream.reset();
    let replayed = ticks
        .filter_map(|(real, period)| stream.next(MAVPTick { real, period }).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(streamed, replayed);
}

#[test]
fn mavp_supports_every_period_based_selector_kind_including_kama() {
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
    let real = floats(reference::INPUT);

    for kind in kinds {
        let maximum = 3;
        let config = MAVPConfig::new(2, maximum, kind).unwrap();
        let periods = vec![maximum; real.len()];
        let actual = config
            .compute(MAVPInput {
                real: &real,
                periods: &periods,
            })
            .unwrap();
        let mut direct = vec![0.0 as Float; real.len()];
        let direct_range = MA(&real, maximum, kind, &mut direct).unwrap();
        assert_eq!(actual.range().beg_idx, config.lookback(), "{kind:?}");
        let direct_start = config.lookback() - direct_range.beg_idx;
        assert_eq!(
            actual.values().as_slice(),
            &direct[direct_start..direct_start + actual.values().len()],
            "{kind:?}"
        );
    }
}

#[test]
fn mavp_clamps_bounds_without_truncation_and_preserves_exact_alignment() {
    let real = [1.0 as Float, 2.0, 4.0, 8.0, 16.0, 32.0];
    let below = [0usize; 6];
    let minimum = [2usize; 6];
    let above = [usize::MAX; 6];
    let maximum = [3usize; 6];
    let config = MAVPConfig::new(2, 3, PeriodMAType::SMA).unwrap();

    let below_output = config
        .compute(MAVPInput {
            real: &real,
            periods: &below,
        })
        .unwrap();
    let minimum_output = config
        .compute(MAVPInput {
            real: &real,
            periods: &minimum,
        })
        .unwrap();
    assert_eq!(below_output, minimum_output);
    let above_output = config
        .compute(MAVPInput {
            real: &real,
            periods: &above,
        })
        .unwrap();
    let maximum_output = config
        .compute(MAVPInput {
            real: &real,
            periods: &maximum,
        })
        .unwrap();
    assert_eq!(above_output, maximum_output);

    let aligned_periods = [2usize, 2, 2, 2, 3, 2];
    let aligned = config
        .compute(MAVPInput {
            real: &real,
            periods: &aligned_periods,
        })
        .unwrap();
    assert_eq!(aligned.range(), OutputRange::new(2, 4));
    assert_values_close(aligned.values(), &[3.0, 6.0, 28.0 / 3.0, 24.0], "alignment");
}

#[test]
fn kama_and_mavp_errors_do_not_mutate_outputs_or_stream_state() {
    assert!(KAMAConfig::new(0).is_err());
    assert!(KAMAConfig::new(usize::MAX).is_err());
    let mut kama_output = [-7.0 as Float; 4];
    assert!(KAMA(&[1.0 as Float, Float::NAN, 3.0, 4.0], 2, &mut kama_output).is_err());
    assert_eq!(kama_output, [-7.0 as Float; 4]);
    assert!(KAMA(&[1.0 as Float, 2.0, 3.0], 2, &mut []).is_err());
    assert!(KAMA(&[1.0 as Float, 2.0], 2, &mut kama_output).is_err());

    assert!(MAVPConfig::new(0, 3, PeriodMAType::EMA).is_err());
    assert!(MAVPConfig::new(3, 2, PeriodMAType::EMA).is_err());
    let config = MAVPConfig::new(2, 3, PeriodMAType::EMA).unwrap();
    let real = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let periods = [2usize, 3, 2, 3, 2];
    let mut untouched = [-13.0 as Float; 4];
    assert!(config
        .compute_into(
            MAVPInput {
                real: &real,
                periods: &periods[..4],
            },
            &mut untouched,
        )
        .is_err());
    assert_eq!(untouched, [-13.0 as Float; 4]);
    assert!(config
        .compute(MAVPInput {
            real: &[1.0 as Float, 2.0],
            periods: &[2usize, 3],
        })
        .is_err());
    let empty = config
        .compute(MAVPInput {
            real: &[],
            periods: &[],
        })
        .unwrap();
    assert_eq!(empty.range(), OutputRange::empty());
    let non_finite = [1.0 as Float, 2.0, Float::INFINITY, 4.0, 5.0];
    assert!(config
        .compute_into(
            MAVPInput {
                real: &non_finite,
                periods: &periods,
            },
            &mut untouched,
        )
        .is_err());
    assert_eq!(untouched, [-13.0 as Float; 4]);
    assert!(config
        .compute_into(
            MAVPInput {
                real: &real,
                periods: &periods
            },
            &mut [0.0 as Float; 2]
        )
        .is_err());
    let recursive = MAVPConfig::new(2, 2, PeriodMAType::DEMA).unwrap();
    let recursive_real = [Float::MAX; 5];
    let recursive_periods = [2usize; 5];
    let mut recursive_output = [-23.0 as Float; 3];
    assert!(recursive
        .compute_into(
            MAVPInput {
                real: &recursive_real,
                periods: &recursive_periods,
            },
            &mut recursive_output,
        )
        .is_err());
    assert_eq!(recursive_output, [-23.0 as Float; 3]);

    let mut runner = config.prepare_batch(4).unwrap();
    assert!(matches!(
        runner.compute_into(
            MAVPInput {
                real: &real,
                periods: &periods
            },
            &mut untouched
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert_eq!(untouched, [-13.0 as Float; 4]);

    let within_capacity = [1.0 as Float; 2];
    let oversized_periods = [2usize; 3];
    let mut no_output = [];
    let mut runner = config.prepare_batch(within_capacity.len()).unwrap();
    assert_eq!(
        runner
            .compute_into(
                MAVPInput {
                    real: &within_capacity,
                    periods: &oversized_periods,
                },
                &mut no_output,
            )
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: within_capacity.len(),
            actual_input_len: oversized_periods.len(),
        }
    );

    let mut kama_stream = KAMAConfig::new(3).unwrap().stream().unwrap();
    let mut kama_control = KAMAConfig::new(3).unwrap().stream().unwrap();
    for tick in [1.0 as Float, 2.0, 3.0] {
        assert_eq!(
            kama_stream.next(tick).unwrap(),
            kama_control.next(tick).unwrap()
        );
    }
    assert!(kama_stream.next(Float::NAN).is_err());
    assert_eq!(
        kama_stream.next(4.0).unwrap(),
        kama_control.next(4.0).unwrap()
    );

    let mut mavp_stream = config.stream().unwrap();
    let mut mavp_control = config.stream().unwrap();
    for tick in [
        MAVPTick {
            real: 1.0,
            period: 2,
        },
        MAVPTick {
            real: 2.0,
            period: 3,
        },
    ] {
        assert_eq!(
            mavp_stream.next(tick).unwrap(),
            mavp_control.next(tick).unwrap()
        );
    }
    assert!(mavp_stream
        .next(MAVPTick {
            real: Float::NAN,
            period: 2
        })
        .is_err());
    let next = MAVPTick {
        real: 3.0,
        period: 2,
    };
    assert_eq!(
        mavp_stream.next(next).unwrap(),
        mavp_control.next(next).unwrap()
    );
}
