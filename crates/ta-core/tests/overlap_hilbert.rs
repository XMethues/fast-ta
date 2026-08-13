// This shared generated input fixture contains provenance fields used by sibling suites.
#[allow(dead_code)]
#[path = "fixtures/ht_dcperiod_reference.rs"]
mod inputs;
#[path = "fixtures/hilbert_overlap_reference.rs"]
mod reference;

use fast_ta::{
    overlap::{
        HT_TRENDLINEBatchRunner, HT_TRENDLINEConfig, HT_TRENDLINEStream, MAMABatchRunner,
        MAMAConfig, MAMAStream, MAMAValuesMut, HT_TRENDLINE, HT_TRENDLINE_LOOKBACK, MAMA,
        MAMA_DEFAULT_FAST_LIMIT, MAMA_DEFAULT_SLOW_LIMIT, MAMA_LOOKBACK,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const ABS_TOLERANCE: Float = 5.0e-3;
#[cfg(not(feature = "f32"))]
const ABS_TOLERANCE: Float = 1.0e-9;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 5.0e-4;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 1.0e-12;

fn as_float(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn assert_value_close(actual: Float, expected: f64, context: &str) {
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
        assert_value_close(
            actual,
            expected,
            &format!("{context}, compact index {index}"),
        );
    }
}

#[test]
fn hilbert_overlap_matches_checksum_pinned_talib_in_every_execution_mode() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    assert_eq!(reference::SOURCE_LENGTH, 256);
    assert_eq!(reference::MAMA_OUTPUT_BEGIN, MAMA_LOOKBACK);
    assert_eq!(reference::HT_TRENDLINE_OUTPUT_BEGIN, HT_TRENDLINE_LOOKBACK);
    assert_eq!(reference::FAST_LIMIT as Float, MAMA_DEFAULT_FAST_LIMIT);
    assert_eq!(reference::SLOW_LIMIT as Float, MAMA_DEFAULT_SLOW_LIMIT);
    assert_eq!(inputs::CASES.len(), reference::CASES.len());

    for (input_case, expected) in inputs::CASES.iter().zip(reference::CASES) {
        assert_eq!(input_case.name, expected.name);
        let input = as_float(input_case.input);
        let mama_range = OutputRange::new(MAMA_LOOKBACK, expected.mama.len());
        let trendline_range = OutputRange::new(HT_TRENDLINE_LOOKBACK, expected.trendline.len());
        let context = format!("{} ({})", input_case.name, input_case.definition);

        let mama_config = MAMAConfig::default();
        assert_eq!(mama_config.lookback(), MAMA_LOOKBACK);
        assert_eq!(mama_config.fast_limit(), MAMA_DEFAULT_FAST_LIMIT);
        assert_eq!(mama_config.slow_limit(), MAMA_DEFAULT_SLOW_LIMIT);
        let owned_mama = mama_config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned_mama.source_len(),
            input.len(),
            "{context}, MAMA owned"
        );
        assert_eq!(owned_mama.range(), mama_range, "{context}, MAMA owned");
        assert_values_close(
            &owned_mama.values().mama,
            expected.mama,
            &format!("{context}, MAMA owned"),
        );
        assert_values_close(
            &owned_mama.values().fama,
            expected.fama,
            &format!("{context}, FAMA owned"),
        );

        let mut direct_mama = vec![0.0 as Float; expected.mama.len()];
        let mut direct_fama = vec![0.0 as Float; expected.fama.len()];
        assert_eq!(
            MAMA(
                &input,
                MAMA_DEFAULT_FAST_LIMIT,
                MAMA_DEFAULT_SLOW_LIMIT,
                &mut direct_mama,
                &mut direct_fama,
            )
            .unwrap(),
            mama_range
        );
        assert_values_close(
            &direct_mama,
            expected.mama,
            &format!("{context}, MAMA caller-owned"),
        );
        assert_values_close(
            &direct_fama,
            expected.fama,
            &format!("{context}, FAMA caller-owned"),
        );

        let mut config_mama = vec![0.0 as Float; expected.mama.len()];
        let mut config_fama = vec![0.0 as Float; expected.fama.len()];
        assert_eq!(
            mama_config
                .compute_into(
                    &input,
                    MAMAValuesMut {
                        mama: &mut config_mama,
                        fama: &mut config_fama,
                    },
                )
                .unwrap(),
            mama_range
        );
        assert_eq!(
            config_mama, direct_mama,
            "{context}, MAMA config caller-owned"
        );
        assert_eq!(
            config_fama, direct_fama,
            "{context}, FAMA config caller-owned"
        );

        let mut mama_runner = mama_config.prepare_batch(input.len()).unwrap();
        let mut prepared_mama = vec![0.0 as Float; expected.mama.len()];
        let mut prepared_fama = vec![0.0 as Float; expected.fama.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                mama_runner
                    .compute_into(
                        &input,
                        MAMAValuesMut {
                            mama: &mut prepared_mama,
                            fama: &mut prepared_fama,
                        },
                    )
                    .unwrap(),
                mama_range,
                "{context}, MAMA prepared {pass}"
            );
            assert_values_close(
                &prepared_mama,
                expected.mama,
                &format!("{context}, MAMA prepared {pass}"),
            );
            assert_values_close(
                &prepared_fama,
                expected.fama,
                &format!("{context}, FAMA prepared {pass}"),
            );
        }

        let mut mama_stream = mama_config.stream().unwrap();
        let streamed_mama = input
            .iter()
            .copied()
            .filter_map(|tick| mama_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_values_close(
            &streamed_mama
                .iter()
                .map(|value| value.mama)
                .collect::<Vec<_>>(),
            expected.mama,
            &format!("{context}, MAMA stream"),
        );
        assert_values_close(
            &streamed_mama
                .iter()
                .map(|value| value.fama)
                .collect::<Vec<_>>(),
            expected.fama,
            &format!("{context}, FAMA stream"),
        );
        mama_stream.reset();
        let replayed_mama = input
            .iter()
            .copied()
            .filter_map(|tick| mama_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(streamed_mama, replayed_mama, "{context}, MAMA reset replay");

        let trendline_config = HT_TRENDLINEConfig::new();
        assert_eq!(trendline_config.lookback(), HT_TRENDLINE_LOOKBACK);
        let owned_trendline = trendline_config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned_trendline.source_len(),
            input.len(),
            "{context}, trendline owned"
        );
        assert_eq!(
            owned_trendline.range(),
            trendline_range,
            "{context}, trendline owned"
        );
        assert_values_close(
            owned_trendline.values(),
            expected.trendline,
            &format!("{context}, trendline owned"),
        );

        let mut direct_trendline = vec![0.0 as Float; expected.trendline.len()];
        assert_eq!(
            HT_TRENDLINE(&input, &mut direct_trendline).unwrap(),
            trendline_range
        );
        assert_values_close(
            &direct_trendline,
            expected.trendline,
            &format!("{context}, trendline caller-owned"),
        );

        let mut trendline_runner = trendline_config.prepare_batch(input.len()).unwrap();
        let mut prepared_trendline = vec![0.0 as Float; expected.trendline.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                trendline_runner
                    .compute_into(&input, &mut prepared_trendline)
                    .unwrap(),
                trendline_range,
                "{context}, trendline prepared {pass}"
            );
            assert_values_close(
                &prepared_trendline,
                expected.trendline,
                &format!("{context}, trendline prepared {pass}"),
            );
        }

        let mut trendline_stream = trendline_config.stream().unwrap();
        let streamed_trendline = input
            .iter()
            .copied()
            .filter_map(|tick| trendline_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_values_close(
            &streamed_trendline,
            expected.trendline,
            &format!("{context}, trendline stream"),
        );
        trendline_stream.reset();
        let replayed_trendline = input
            .iter()
            .copied()
            .filter_map(|tick| trendline_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            streamed_trendline, replayed_trendline,
            "{context}, trendline reset replay"
        );
    }
}

#[test]
fn mama_limits_are_explicit_validated_and_observably_control_adaptation() {
    let default = MAMAConfig::default();
    assert_eq!(
        default,
        MAMAConfig::new(0.5 as Float, 0.05 as Float).unwrap()
    );
    assert_eq!(
        MAMAConfig::new(0.01 as Float, 0.01 as Float)
            .unwrap()
            .fast_limit(),
        0.01 as Float
    );
    assert_eq!(
        MAMAConfig::new(0.99 as Float, 0.99 as Float)
            .unwrap()
            .slow_limit(),
        0.99 as Float
    );

    for (fast, slow) in [
        (0.009 as Float, 0.05 as Float),
        (1.0 as Float, 0.05 as Float),
        (0.5 as Float, 0.009 as Float),
        (0.5 as Float, 1.0 as Float),
        (Float::NAN, 0.05 as Float),
        (0.5 as Float, Float::INFINITY),
    ] {
        assert!(matches!(
            MAMAConfig::new(fast, slow),
            Err(TalibError::InvalidParameter { .. })
        ));
    }

    let sine = as_float(
        inputs::CASES
            .iter()
            .find(|case| case.name == "sine")
            .unwrap()
            .input,
    );
    let fast = MAMAConfig::new(0.99 as Float, 0.99 as Float)
        .unwrap()
        .compute(&sine)
        .unwrap();
    let slow = MAMAConfig::new(0.01 as Float, 0.01 as Float)
        .unwrap()
        .compute(&sine)
        .unwrap();
    assert!(
        fast.values()
            .mama
            .iter()
            .zip(&slow.values().mama)
            .any(|(&left, &right)| (left - right).abs() > 1.0 as Float),
        "fast/slow limits must change the MAMA result"
    );
    assert!(
        fast.values()
            .mama
            .iter()
            .zip(&fast.values().fama)
            .any(|(&mama, &fama)| (mama - fama).abs() > ABS_TOLERANCE),
        "the paired FAMA result must remain independently observable"
    );
}

#[test]
fn flat_trend_and_sine_invariants_hold() {
    let constant = inputs::CASES
        .iter()
        .find(|case| case.name == "constant")
        .unwrap();
    let constant_input = as_float(constant.input);
    let constant_mama = MAMAConfig::default().compute(&constant_input).unwrap();
    let constant_trendline = HT_TRENDLINEConfig::new().compute(&constant_input).unwrap();
    assert_value_close(
        *constant_mama.values().mama.last().unwrap(),
        100.0,
        "constant MAMA tail",
    );
    assert_value_close(
        *constant_mama.values().fama.last().unwrap(),
        100.0,
        "constant FAMA tail",
    );
    for &value in constant_trendline.values() {
        assert_value_close(value, 100.0, "constant HT_TRENDLINE");
    }

    let trend = inputs::CASES
        .iter()
        .find(|case| case.name == "trend")
        .unwrap();
    let trend_output = HT_TRENDLINEConfig::new()
        .compute(&as_float(trend.input))
        .unwrap();
    assert!(trend_output
        .values()
        .windows(2)
        .all(|pair| pair[1] >= pair[0]));

    let sine = inputs::CASES
        .iter()
        .find(|case| case.name == "sine")
        .unwrap();
    let sine_output = HT_TRENDLINEConfig::new()
        .compute(&as_float(sine.input))
        .unwrap();
    let minimum = sine_output
        .values()
        .iter()
        .copied()
        .fold(Float::INFINITY, Float::min);
    let maximum = sine_output
        .values()
        .iter()
        .copied()
        .fold(Float::NEG_INFINITY, Float::max);
    // The shared Hilbert recurrence accumulates floating-point drift over
    // the lookback warm-up. The Hilbert-based trendline is a heavy low-pass
    // filter; verify the output remains responsive to a non-trivial
    // amplitude sine without requiring a fixed spread that would couple
    // the test to the precise reference rounding.
    let range = maximum - minimum;
    assert!(
        range > 0.1 as Float,
        "sine HT_TRENDLINE range {range} too small to be responsive"
    );
}

#[test]
fn batch_failures_are_transactional_for_every_output_column() {
    let input = as_float(inputs::CASES[2].input);
    let mama_count = input.len() - MAMA_LOOKBACK;
    let trendline_count = input.len() - HT_TRENDLINE_LOOKBACK;
    let mut mama = vec![123.0 as Float; mama_count];
    let mut short_fama = vec![456.0 as Float; mama_count - 1];
    assert!(matches!(
        MAMA(
            &input,
            MAMA_DEFAULT_FAST_LIMIT,
            MAMA_DEFAULT_SLOW_LIMIT,
            &mut mama,
            &mut short_fama,
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(mama.iter().all(|&value| value == 123.0 as Float));
    assert!(short_fama.iter().all(|&value| value == 456.0 as Float));

    let mut short_mama = vec![123.0 as Float; mama_count - 1];
    let mut fama = vec![456.0 as Float; mama_count];
    assert!(matches!(
        MAMA(
            &input,
            MAMA_DEFAULT_FAST_LIMIT,
            MAMA_DEFAULT_SLOW_LIMIT,
            &mut short_mama,
            &mut fama,
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(short_mama.iter().all(|&value| value == 123.0 as Float));
    assert!(fama.iter().all(|&value| value == 456.0 as Float));

    let mut invalid = input.clone();
    invalid[MAMA_LOOKBACK + 1] = Float::NAN;
    let mut out_mama = vec![123.0 as Float; mama_count];
    let mut out_fama = vec![456.0 as Float; mama_count];
    assert!(matches!(
        MAMA(
            &invalid,
            MAMA_DEFAULT_FAST_LIMIT,
            MAMA_DEFAULT_SLOW_LIMIT,
            &mut out_mama,
            &mut out_fama,
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(out_mama.iter().all(|&value| value == 123.0 as Float));
    assert!(out_fama.iter().all(|&value| value == 456.0 as Float));

    assert!(matches!(
        MAMA(
            &input,
            0.0 as Float,
            MAMA_DEFAULT_SLOW_LIMIT,
            &mut out_mama,
            &mut out_fama,
        ),
        Err(TalibError::InvalidParameter { .. })
    ));
    assert!(out_mama.iter().all(|&value| value == 123.0 as Float));
    assert!(out_fama.iter().all(|&value| value == 456.0 as Float));

    let mut trendline = vec![789.0 as Float; trendline_count - 1];
    assert!(matches!(
        HT_TRENDLINE(&input, &mut trendline),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(trendline.iter().all(|&value| value == 789.0 as Float));

    let mut invalid_trendline = vec![789.0 as Float; trendline_count];
    assert!(matches!(
        HT_TRENDLINE(&invalid, &mut invalid_trendline),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(invalid_trendline
        .iter()
        .all(|&value| value == 789.0 as Float));

    assert_eq!(
        MAMAConfig::default()
            .compute(&input[..MAMA_LOOKBACK])
            .unwrap_err(),
        TalibError::InsufficientData {
            required: MAMA_LOOKBACK + 1,
            actual: MAMA_LOOKBACK,
        }
    );
    assert_eq!(
        HT_TRENDLINEConfig::new()
            .compute(&input[..HT_TRENDLINE_LOOKBACK])
            .unwrap_err(),
        TalibError::InsufficientData {
            required: HT_TRENDLINE_LOOKBACK + 1,
            actual: HT_TRENDLINE_LOOKBACK,
        }
    );
}

#[test]
fn prepared_capacity_and_stream_errors_preserve_state() {
    let input = as_float(inputs::CASES[3].input);
    let mut mama_runner = MAMAConfig::default()
        .prepare_batch(input.len() - 1)
        .unwrap();
    let mut mama = vec![123.0 as Float; input.len() - MAMA_LOOKBACK];
    let mut fama = vec![456.0 as Float; input.len() - MAMA_LOOKBACK];
    assert_eq!(
        mama_runner
            .compute_into(
                &input,
                MAMAValuesMut {
                    mama: &mut mama,
                    fama: &mut fama,
                },
            )
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: input.len() - 1,
            actual_input_len: input.len(),
        }
    );
    assert!(mama.iter().all(|&value| value == 123.0 as Float));
    assert!(fama.iter().all(|&value| value == 456.0 as Float));

    let mama_config = MAMAConfig::default();
    let mut output_runner = mama_config.prepare_batch(input.len()).unwrap();
    let mut full_mama = vec![123.0 as Float; input.len() - MAMA_LOOKBACK];
    let mut short_fama = vec![456.0 as Float; input.len() - MAMA_LOOKBACK - 1];
    assert!(matches!(
        output_runner.compute_into(
            &input,
            MAMAValuesMut {
                mama: &mut full_mama,
                fama: &mut short_fama,
            },
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(full_mama.iter().all(|&value| value == 123.0 as Float));
    assert!(short_fama.iter().all(|&value| value == 456.0 as Float));
    let mut full_fama = vec![0.0 as Float; input.len() - MAMA_LOOKBACK];
    output_runner
        .compute_into(
            &input,
            MAMAValuesMut {
                mama: &mut full_mama,
                fama: &mut full_fama,
            },
        )
        .unwrap();
    let expected_mama = mama_config.compute(&input).unwrap();
    assert_eq!(full_mama, expected_mama.values().mama);
    assert_eq!(full_fama, expected_mama.values().fama);

    let mut trendline_runner = HT_TRENDLINEConfig::new()
        .prepare_batch(input.len() - 1)
        .unwrap();
    let mut trendline = vec![789.0 as Float; input.len() - HT_TRENDLINE_LOOKBACK];
    assert_eq!(
        trendline_runner
            .compute_into(&input, &mut trendline)
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: input.len() - 1,
            actual_input_len: input.len(),
        }
    );
    assert!(trendline.iter().all(|&value| value == 789.0 as Float));

    let trendline_config = HT_TRENDLINEConfig::new();
    let mut output_runner = trendline_config.prepare_batch(input.len()).unwrap();
    let mut short_trendline = vec![789.0 as Float; input.len() - HT_TRENDLINE_LOOKBACK - 1];
    assert!(matches!(
        output_runner.compute_into(&input, &mut short_trendline),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(short_trendline.iter().all(|&value| value == 789.0 as Float));
    let mut full_trendline = vec![0.0 as Float; input.len() - HT_TRENDLINE_LOOKBACK];
    output_runner
        .compute_into(&input, &mut full_trendline)
        .unwrap();
    assert_eq!(
        full_trendline,
        trendline_config.compute(&input).unwrap().into_values()
    );

    let mut mama_stream = mama_config.stream().unwrap();
    for &tick in &input[..MAMA_LOOKBACK + 8] {
        mama_stream.next(tick).unwrap();
    }
    let mut mama_control = mama_stream;
    assert!(matches!(
        mama_stream.next(Float::NAN),
        Err(TalibError::InvalidInput { .. })
    ));
    for &tick in &input[MAMA_LOOKBACK + 8..] {
        assert_eq!(
            mama_stream.next(tick).unwrap(),
            mama_control.next(tick).unwrap()
        );
    }

    let mut trendline_stream = HT_TRENDLINEConfig::new().stream().unwrap();
    for &tick in &input[..HT_TRENDLINE_LOOKBACK + 8] {
        trendline_stream.next(tick).unwrap();
    }
    let mut trendline_control = trendline_stream;
    assert!(matches!(
        trendline_stream.next(Float::INFINITY),
        Err(TalibError::InvalidInput { .. })
    ));
    for &tick in &input[HT_TRENDLINE_LOOKBACK + 8..] {
        assert_eq!(
            trendline_stream.next(tick).unwrap(),
            trendline_control.next(tick).unwrap()
        );
    }
}

#[test]
fn empty_execution_is_compact_and_independent_streams_do_not_share_state() {
    let mama = MAMAConfig::default().compute(&[]).unwrap();
    assert_eq!(mama.source_len(), 0);
    assert_eq!(mama.range(), OutputRange::empty());
    assert!(mama.values().mama.is_empty());
    assert!(mama.values().fama.is_empty());
    let trendline = HT_TRENDLINEConfig::new().compute(&[]).unwrap();
    assert_eq!(trendline.source_len(), 0);
    assert_eq!(trendline.range(), OutputRange::empty());
    assert!(trendline.values().is_empty());

    let left = as_float(inputs::CASES[1].input);
    let right = as_float(inputs::CASES[2].input);
    let config = MAMAConfig::default();
    let mut interleaved_left = config.stream().unwrap();
    let mut interleaved_right = config.stream().unwrap();
    let mut left_values = Vec::new();
    let mut right_values = Vec::new();
    for (&left_tick, &right_tick) in left.iter().zip(&right) {
        if let Some(value) = interleaved_left.next(left_tick).unwrap() {
            left_values.push(value);
        }
        if let Some(value) = interleaved_right.next(right_tick).unwrap() {
            right_values.push(value);
        }
    }
    let mut isolated_left = config.stream().unwrap();
    let expected_left = left
        .iter()
        .copied()
        .filter_map(|tick| isolated_left.next(tick).unwrap())
        .collect::<Vec<_>>();
    let mut isolated_right = config.stream().unwrap();
    let expected_right = right
        .iter()
        .copied()
        .filter_map(|tick| isolated_right.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(left_values, expected_left);
    assert_eq!(right_values, expected_right);
}

#[test]
fn public_execution_types_are_sealed_and_complete() {
    fn assert_execution_types<C, R, S>()
    where
        C: IndicatorConfig<BatchRunner = R, Stream = S>,
        R: PreparedBatchRunner<C>,
        S: StreamingComputation<C>,
    {
    }

    assert_execution_types::<MAMAConfig, MAMABatchRunner, MAMAStream>();
    assert_execution_types::<HT_TRENDLINEConfig, HT_TRENDLINEBatchRunner, HT_TRENDLINEStream>();
}
