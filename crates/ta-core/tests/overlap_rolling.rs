#![allow(clippy::type_complexity)]

// Pinned f64 oracle values intentionally retain more precision than f32 builds consume.
#[allow(clippy::excessive_precision)]
#[path = "fixtures/overlap_rolling_reference.rs"]
mod reference;

use ta_core::inventory::{function, ImplementationStatus};
use ta_core::overlap::{
    ACCBANDSBatchRunner, ACCBANDSConfig, ACCBANDSInput, ACCBANDSStream, ACCBANDSTick,
    ACCBANDSValuesMut, BBANDSBatchRunner, BBANDSConfig, BBANDSStream, BBANDSValuesMut, MAConfig,
    MIDPOINTBatchRunner, MIDPOINTConfig, MIDPOINTStream, MIDPRICEBatchRunner, MIDPRICEConfig,
    MIDPRICEInput, MIDPRICEStream, MIDPRICETick, PeriodMAType, ACCBANDS, BBANDS, MIDPOINT,
    MIDPRICE,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

const SENTINEL: Float = -98_765.0 as Float;

fn float_vec(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn assert_close(actual: Float, expected: Float) {
    let scale = expected.abs().max(1.0 as Float);
    let tolerance = Float::EPSILON * 512.0 as Float * scale;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:?}, expected {expected:?}, tolerance {tolerance:?}"
    );
}

fn assert_slice_close(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected as Float);
    }
}

fn sample() -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    (
        float_vec(reference::HIGH),
        float_vec(reference::LOW),
        float_vec(reference::REAL),
    )
}

#[test]
fn pinned_talib_reference_vectors_preserve_each_definition_and_source_range() {
    let (high, low, real) = sample();
    let count = real.len() - reference::PERIOD + 1;

    let mut acc_upper = vec![0.0 as Float; count];
    let mut acc_middle = vec![0.0 as Float; count];
    let mut acc_lower = vec![0.0 as Float; count];
    let acc_range = ACCBANDS(
        &high,
        &low,
        &real,
        reference::PERIOD,
        &mut acc_upper,
        &mut acc_middle,
        &mut acc_lower,
    )
    .expect("valid ACCBANDS reference input");
    assert_eq!(acc_range, OutputRange::new(2, count));
    assert_slice_close(&acc_upper, reference::ACCBANDS_UPPER);
    assert_slice_close(&acc_middle, reference::ACCBANDS_MIDDLE);
    assert_slice_close(&acc_lower, reference::ACCBANDS_LOWER);

    let mut bb_upper = vec![0.0 as Float; count];
    let mut bb_middle = vec![0.0 as Float; count];
    let mut bb_lower = vec![0.0 as Float; count];
    let bb_range = BBANDS(
        &real,
        reference::PERIOD,
        2.0 as Float,
        2.0 as Float,
        PeriodMAType::SMA,
        &mut bb_upper,
        &mut bb_middle,
        &mut bb_lower,
    )
    .expect("valid BBANDS reference input");
    assert_eq!(bb_range, OutputRange::new(2, count));
    assert_slice_close(&bb_upper, reference::BBANDS_UPPER);
    assert_slice_close(&bb_middle, reference::BBANDS_MIDDLE);
    assert_slice_close(&bb_lower, reference::BBANDS_LOWER);

    let mut midpoint = vec![0.0 as Float; count];
    let midpoint_range = MIDPOINT(&real, reference::PERIOD, &mut midpoint).unwrap();
    assert_eq!(midpoint_range, OutputRange::new(2, count));
    assert_slice_close(&midpoint, reference::MIDPOINT);

    let mut midprice = vec![0.0 as Float; count];
    let midprice_range = MIDPRICE(&high, &low, reference::PERIOD, &mut midprice).unwrap();
    assert_eq!(midprice_range, OutputRange::new(2, count));
    assert_slice_close(&midprice, reference::MIDPRICE);
}

#[test]
fn band_owned_caller_owned_and_prepared_paths_have_exact_named_compact_columns() {
    let (high, low, real) = sample();
    let acc = ACCBANDSConfig::new(3).unwrap();
    let owned = acc
        .compute(ACCBANDSInput {
            high: &high,
            low: &low,
            close: &real,
        })
        .unwrap();
    assert_eq!(owned.source_len(), real.len());
    assert_eq!(owned.range(), OutputRange::new(2, 8));
    assert_eq!(owned.values().upper.len(), 8);
    assert_eq!(owned.values().middle.len(), 8);
    assert_eq!(owned.values().lower.len(), 8);

    let mut upper = vec![SENTINEL; 10];
    let mut middle = vec![SENTINEL; 10];
    let mut lower = vec![SENTINEL; 10];
    let range = acc
        .compute_into(
            ACCBANDSInput {
                high: &high,
                low: &low,
                close: &real,
            },
            ACCBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .unwrap();
    assert_eq!(range, owned.range());
    assert_eq!(&upper[..8], owned.values().upper.as_slice());
    assert_eq!(&middle[..8], owned.values().middle.as_slice());
    assert_eq!(&lower[..8], owned.values().lower.as_slice());
    assert_eq!(
        (&upper[8..], &middle[8..], &lower[8..]),
        (&[SENTINEL; 2][..], &[SENTINEL; 2][..], &[SENTINEL; 2][..])
    );

    let mut acc_runner = acc.prepare_batch(real.len()).unwrap();
    upper.fill(SENTINEL);
    middle.fill(SENTINEL);
    lower.fill(SENTINEL);
    let prepared_range = acc_runner
        .compute_into(
            ACCBANDSInput {
                high: &high,
                low: &low,
                close: &real,
            },
            ACCBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .unwrap();
    assert_eq!(prepared_range, owned.range());
    assert_eq!(&upper[..8], owned.values().upper.as_slice());

    let bb = BBANDSConfig::with_default_deviations(3, PeriodMAType::SMA).unwrap();
    let bb_owned = bb.compute(&real).unwrap();
    let mut bb_runner = bb.prepare_batch(real.len()).unwrap();
    let mut bb_upper = vec![SENTINEL; 8];
    let mut bb_middle = vec![SENTINEL; 8];
    let mut bb_lower = vec![SENTINEL; 8];
    let bb_range = bb_runner
        .compute_into(
            &real,
            BBANDSValuesMut {
                upper: &mut bb_upper,
                middle: &mut bb_middle,
                lower: &mut bb_lower,
            },
        )
        .unwrap();
    assert_eq!(bb_range, bb_owned.range());
    assert_eq!(bb_upper, bb_owned.values().upper);
    assert_eq!(bb_middle, bb_owned.values().middle);
    assert_eq!(bb_lower, bb_owned.values().lower);
}

#[test]
fn short_or_mismatched_band_columns_fail_before_any_column_is_mutated() {
    let (high, low, real) = sample();
    let acc = ACCBANDSConfig::new(3).unwrap();
    let bb = BBANDSConfig::with_default_deviations(3, PeriodMAType::SMA).unwrap();

    let mut upper = vec![SENTINEL; 8];
    let mut middle = vec![SENTINEL; 9];
    let mut lower = vec![SENTINEL; 8];
    assert!(acc
        .compute_into(
            ACCBANDSInput {
                high: &high,
                low: &low,
                close: &real,
            },
            ACCBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .is_err());
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));
    assert!(bb
        .compute_into(
            &real,
            BBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .is_err());
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));

    upper.truncate(7);
    middle.truncate(7);
    lower.truncate(7);
    assert!(bb
        .compute_into(
            &real,
            BBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .is_err());
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));
}

#[test]
fn every_qualified_period_ma_is_a_valid_bbands_path_with_honest_alignment() {
    let real = (0..96)
        .map(|idx| (idx as Float * 0.17 as Float).sin() + idx as Float * 0.03 as Float)
        .collect::<Vec<_>>();
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
        let config = BBANDSConfig::new(5, 1.5 as Float, 2.5 as Float, kind).unwrap();
        let bands = config.compute(&real).unwrap();
        let ma = MAConfig::new(5, kind).unwrap().compute(&real).unwrap();
        assert_eq!(bands.range(), ma.range(), "{kind:?}");
        assert_eq!(bands.values().middle.len(), ma.values().len(), "{kind:?}");
        for (&middle, &expected) in bands.values().middle.iter().zip(ma.values()) {
            assert_close(middle, expected);
        }
        for ((&upper, &middle), &lower) in bands
            .values()
            .upper
            .iter()
            .zip(&bands.values().middle)
            .zip(&bands.values().lower)
        {
            assert!(upper >= middle || (upper - middle).abs() <= Float::EPSILON * 64.0);
            assert!(middle >= lower || (middle - lower).abs() <= Float::EPSILON * 64.0);
        }
    }
}

#[test]
fn midpoint_and_midprice_cover_owned_caller_owned_prepared_and_extrema_semantics() {
    let (high, low, real) = sample();
    let midpoint = MIDPOINTConfig::new(3).unwrap();
    let midpoint_owned = midpoint.compute(&real).unwrap();
    assert_slice_close(midpoint_owned.values(), reference::MIDPOINT);

    let mut output = vec![SENTINEL; 10];
    let mut midpoint_runner = midpoint.prepare_batch(real.len()).unwrap();
    let range = midpoint_runner.compute_into(&real, &mut output).unwrap();
    assert_eq!(range, midpoint_owned.range());
    assert_eq!(&output[..8], midpoint_owned.values());
    assert!(output[8..].iter().all(|&value| value == SENTINEL));

    let midprice = MIDPRICEConfig::new(3).unwrap();
    let midprice_owned = midprice
        .compute(MIDPRICEInput {
            high: &high,
            low: &low,
        })
        .unwrap();
    assert_slice_close(midprice_owned.values(), reference::MIDPRICE);
    let mut midprice_runner = midprice.prepare_batch(high.len()).unwrap();
    output.fill(SENTINEL);
    let range = midprice_runner
        .compute_into(
            MIDPRICEInput {
                high: &high,
                low: &low,
            },
            &mut output,
        )
        .unwrap();
    assert_eq!(range, midprice_owned.range());
    assert_eq!(&output[..8], midprice_owned.values());

    let observations = [4.0 as Float, -2.0, 7.0, 3.0];
    let values = MIDPOINTConfig::new(3)
        .unwrap()
        .compute(&observations)
        .unwrap();
    assert_eq!(values.values(), &[2.5 as Float, 2.5]);
    let highs = [10.0 as Float, 50.0, 20.0, 30.0];
    let lows = [8.0 as Float, 9.0, -40.0, 7.0];
    let values = MIDPRICEConfig::new(3)
        .unwrap()
        .compute(MIDPRICEInput {
            high: &highs,
            low: &lows,
        })
        .unwrap();
    assert_eq!(values.values(), &[5.0 as Float, 5.0]);
}

#[test]
fn streams_are_independent_resettable_transactional_and_match_batch() {
    let (high, low, real) = sample();
    let acc_config = ACCBANDSConfig::new(3).unwrap();
    let acc_batch = acc_config
        .compute(ACCBANDSInput {
            high: &high,
            low: &low,
            close: &real,
        })
        .unwrap();
    let mut acc_stream = acc_config.stream().unwrap();
    let mut acc_control = acc_config.stream().unwrap();
    assert!(acc_stream
        .next(ACCBANDSTick {
            high: high[0],
            low: low[0],
            close: Float::NAN,
        })
        .is_err());
    let mut acc_outputs = Vec::new();
    for idx in 0..real.len() {
        let tick = ACCBANDSTick {
            high: high[idx],
            low: low[idx],
            close: real[idx],
        };
        let actual = acc_stream.next(tick).unwrap();
        let expected = acc_control.next(tick).unwrap();
        assert_eq!(actual, expected);
        if let Some(value) = actual {
            acc_outputs.push(value);
        }
    }
    for (idx, value) in acc_outputs.iter().enumerate() {
        assert_close(value.upper, acc_batch.values().upper[idx]);
        assert_close(value.middle, acc_batch.values().middle[idx]);
        assert_close(value.lower, acc_batch.values().lower[idx]);
    }
    acc_stream.reset();
    assert!(acc_stream
        .next(ACCBANDSTick {
            high: high[0],
            low: low[0],
            close: real[0]
        })
        .unwrap()
        .is_none());

    let bb_config = BBANDSConfig::new(3, 2.0, 2.0, PeriodMAType::EMA).unwrap();
    let bb_batch = bb_config.compute(&real).unwrap();
    let mut bb_stream = bb_config.stream().unwrap();
    let mut control = bb_config.stream().unwrap();
    assert!(matches!(
        bb_stream.next(Float::NAN),
        Err(TalibError::InvalidInput { .. })
    ));
    for &input in &real {
        let actual = bb_stream.next(input).unwrap();
        let expected = control.next(input).unwrap();
        assert_eq!(actual.is_some(), expected.is_some());
        if let (Some(actual), Some(expected)) = (actual, expected) {
            assert_close(actual.upper, expected.upper);
            assert_close(actual.middle, expected.middle);
            assert_close(actual.lower, expected.lower);
        }
    }
    bb_stream.reset();
    let replay = real
        .iter()
        .filter_map(|&input| bb_stream.next(input).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(replay.len(), bb_batch.values().upper.len());
    for (idx, output) in replay.iter().enumerate() {
        assert_close(output.upper, bb_batch.values().upper[idx]);
        assert_close(output.middle, bb_batch.values().middle[idx]);
        assert_close(output.lower, bb_batch.values().lower[idx]);
    }

    let midpoint_config = MIDPOINTConfig::new(3).unwrap();
    let mut midpoint_stream = midpoint_config.stream().unwrap();
    let mut midpoint_control = midpoint_config.stream().unwrap();
    assert!(midpoint_stream.next(Float::INFINITY).is_err());
    for &input in &real {
        assert_eq!(
            midpoint_stream.next(input).unwrap(),
            midpoint_control.next(input).unwrap()
        );
    }
    midpoint_stream.reset();
    assert!(midpoint_stream.next(real[0]).unwrap().is_none());

    let midprice_config = MIDPRICEConfig::new(3).unwrap();
    let mut midprice_stream = midprice_config.stream().unwrap();
    let mut midprice_control = midprice_config.stream().unwrap();
    assert!(midprice_stream
        .next(MIDPRICETick {
            high: high[0],
            low: Float::NAN
        })
        .is_err());
    for idx in 0..high.len() {
        let tick = MIDPRICETick {
            high: high[idx],
            low: low[idx],
        };
        assert_eq!(
            midprice_stream.next(tick).unwrap(),
            midprice_control.next(tick).unwrap()
        );
    }
    midprice_stream.reset();
    assert!(midprice_stream
        .next(MIDPRICETick {
            high: high[0],
            low: low[0],
        })
        .unwrap()
        .is_none());
}

#[test]
fn empty_inputs_return_empty_compact_ranges_without_payload_columns() {
    let empty: [Float; 0] = [];
    let acc = ACCBANDSConfig::new(2)
        .unwrap()
        .compute(ACCBANDSInput {
            high: &empty,
            low: &empty,
            close: &empty,
        })
        .unwrap();
    assert_eq!(acc.range(), OutputRange::empty());
    assert!(acc.values().upper.is_empty());
    assert!(acc.values().middle.is_empty());
    assert!(acc.values().lower.is_empty());

    let bb = BBANDSConfig::with_default_deviations(2, PeriodMAType::SMA)
        .unwrap()
        .compute(&empty)
        .unwrap();
    assert_eq!(bb.range(), OutputRange::empty());
    assert!(bb.values().upper.is_empty());
    assert!(bb.values().middle.is_empty());
    assert!(bb.values().lower.is_empty());

    let midpoint = MIDPOINTConfig::new(2).unwrap().compute(&empty).unwrap();
    assert_eq!(midpoint.range(), OutputRange::empty());
    assert!(midpoint.values().is_empty());
    let midprice = MIDPRICEConfig::new(2)
        .unwrap()
        .compute(MIDPRICEInput {
            high: &empty,
            low: &empty,
        })
        .unwrap();
    assert_eq!(midprice.range(), OutputRange::empty());
    assert!(midprice.values().is_empty());
}

#[test]
fn validation_and_prepared_capacity_failures_preserve_all_caller_state() {
    assert!(matches!(
        MIDPOINTConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        MIDPRICEConfig::new(100_001),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        ACCBANDSConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        BBANDSConfig::new(3, Float::NAN, 2.0, PeriodMAType::SMA),
        Err(TalibError::InvalidParameter { .. })
    ));

    let (high, low, real) = sample();
    let mut output = vec![SENTINEL; 8];
    assert!(matches!(
        MIDPOINTConfig::new(20)
            .unwrap()
            .compute_into(&real, &mut output),
        Err(TalibError::InsufficientData { .. })
    ));
    assert!(output.iter().all(|&value| value == SENTINEL));

    let midprice = MIDPRICEConfig::new(3).unwrap();
    assert!(midprice
        .compute_into(
            MIDPRICEInput {
                high: &high,
                low: &low[..low.len() - 1],
            },
            &mut output,
        )
        .is_err());
    assert!(output.iter().all(|&value| value == SENTINEL));

    let mut bad_real = real.clone();
    bad_real[4] = Float::NAN;
    assert!(MIDPOINTConfig::new(3)
        .unwrap()
        .compute_into(&bad_real, &mut output)
        .is_err());
    assert!(output.iter().all(|&value| value == SENTINEL));

    let mut midpoint_runner = MIDPOINTConfig::new(3).unwrap().prepare_batch(9).unwrap();
    assert!(matches!(
        midpoint_runner.compute_into(&real, &mut output),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(output.iter().all(|&value| value == SENTINEL));

    let mut upper = vec![SENTINEL; 8];
    let mut middle = vec![SENTINEL; 8];
    let mut lower = vec![SENTINEL; 8];
    let acc = ACCBANDSConfig::new(3).unwrap();
    assert!(acc
        .compute_into(
            ACCBANDSInput {
                high: &high,
                low: &low[..low.len() - 1],
                close: &real,
            },
            ACCBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .is_err());
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));

    assert!(BBANDSConfig::with_default_deviations(3, PeriodMAType::SMA)
        .unwrap()
        .compute_into(
            &bad_real,
            BBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        )
        .is_err());
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));

    let mut acc_runner = acc.prepare_batch(9).unwrap();
    assert!(matches!(
        acc_runner.compute_into(
            ACCBANDSInput {
                high: &high,
                low: &low,
                close: &real,
            },
            ACCBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            },
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    let mut midprice_runner = midprice.prepare_batch(9).unwrap();
    assert!(matches!(
        midprice_runner.compute_into(
            MIDPRICEInput {
                high: &high,
                low: &low,
            },
            &mut output,
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    let mut bb_runner = BBANDSConfig::with_default_deviations(3, PeriodMAType::SMA)
        .unwrap()
        .prepare_batch(9)
        .unwrap();
    assert!(matches!(
        bb_runner.compute_into(
            &real,
            BBANDSValuesMut {
                upper: &mut upper,
                middle: &mut middle,
                lower: &mut lower,
            }
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(upper
        .iter()
        .chain(&middle)
        .chain(&lower)
        .all(|&value| value == SENTINEL));
}

#[test]
fn prepared_capacity_precedes_overlap_input_alignment() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let capacity_error = TalibError::PreparedCapacityExceeded {
        max_input_len: within.len(),
        actual_input_len: oversized.len(),
    };
    let mut upper = [];
    let mut middle = [];
    let mut lower = [];

    let mut accbands = ACCBANDSConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        accbands
            .compute_into(
                ACCBANDSInput {
                    high: &within,
                    low: &oversized,
                    close: &within,
                },
                ACCBANDSValuesMut {
                    upper: &mut upper,
                    middle: &mut middle,
                    lower: &mut lower,
                },
            )
            .unwrap_err(),
        capacity_error
    );

    let mut output = [];
    let mut midprice = MIDPRICEConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        midprice
            .compute_into(
                MIDPRICEInput {
                    high: &within,
                    low: &oversized,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );
}

#[test]
fn flat_series_band_order_and_positive_scaling_invariants_hold() {
    let flat = vec![7.5 as Float; 32];
    let bb = BBANDSConfig::with_default_deviations(5, PeriodMAType::SMA)
        .unwrap()
        .compute(&flat)
        .unwrap();
    for ((&upper, &middle), &lower) in bb
        .values()
        .upper
        .iter()
        .zip(&bb.values().middle)
        .zip(&bb.values().lower)
    {
        assert_close(upper, 7.5);
        assert_close(middle, 7.5);
        assert_close(lower, 7.5);
    }
    let zero_denominator = ACCBANDSConfig::new(2)
        .unwrap()
        .compute(ACCBANDSInput {
            high: &[1.0 as Float, 1.0, 1.0],
            low: &[-1.0 as Float, -1.0, -1.0],
            close: &[0.0 as Float, 0.0, 0.0],
        })
        .unwrap();
    assert_eq!(
        zero_denominator.values().upper.as_slice(),
        &[1.0 as Float, 1.0]
    );
    assert_eq!(
        zero_denominator.values().middle.as_slice(),
        &[0.0 as Float, 0.0]
    );
    assert_eq!(
        zero_denominator.values().lower.as_slice(),
        &[-1.0 as Float, -1.0]
    );

    let (high, low, real) = sample();
    let scale = 3.25 as Float;
    let scaled_high = high.iter().map(|value| value * scale).collect::<Vec<_>>();
    let scaled_low = low.iter().map(|value| value * scale).collect::<Vec<_>>();
    let scaled_real = real.iter().map(|value| value * scale).collect::<Vec<_>>();

    let acc_config = ACCBANDSConfig::new(3).unwrap();
    let base = acc_config
        .compute(ACCBANDSInput {
            high: &high,
            low: &low,
            close: &real,
        })
        .unwrap();
    let scaled = acc_config
        .compute(ACCBANDSInput {
            high: &scaled_high,
            low: &scaled_low,
            close: &scaled_real,
        })
        .unwrap();
    for idx in 0..base.values().upper.len() {
        assert_close(scaled.values().upper[idx], base.values().upper[idx] * scale);
        assert_close(
            scaled.values().middle[idx],
            base.values().middle[idx] * scale,
        );
        assert_close(scaled.values().lower[idx], base.values().lower[idx] * scale);
        assert!(base.values().upper[idx] >= base.values().middle[idx]);
        assert!(base.values().middle[idx] >= base.values().lower[idx]);
    }

    let bb_config = BBANDSConfig::with_default_deviations(3, PeriodMAType::SMA).unwrap();
    let bb_base = bb_config.compute(&real).unwrap();
    let bb_scaled = bb_config.compute(&scaled_real).unwrap();
    for idx in 0..bb_base.values().upper.len() {
        assert_close(
            bb_scaled.values().upper[idx],
            bb_base.values().upper[idx] * scale,
        );
        assert_close(
            bb_scaled.values().middle[idx],
            bb_base.values().middle[idx] * scale,
        );
        assert_close(
            bb_scaled.values().lower[idx],
            bb_base.values().lower[idx] * scale,
        );
    }

    let midpoint_base = MIDPOINTConfig::new(3).unwrap().compute(&real).unwrap();
    let midpoint_scaled = MIDPOINTConfig::new(3)
        .unwrap()
        .compute(&scaled_real)
        .unwrap();
    for (&actual, &base) in midpoint_scaled.values().iter().zip(midpoint_base.values()) {
        assert_close(actual, base * scale);
    }
    let midprice_base = MIDPRICEConfig::new(3)
        .unwrap()
        .compute(MIDPRICEInput {
            high: &high,
            low: &low,
        })
        .unwrap();
    let midprice_scaled = MIDPRICEConfig::new(3)
        .unwrap()
        .compute(MIDPRICEInput {
            high: &scaled_high,
            low: &scaled_low,
        })
        .unwrap();
    for (&actual, &base) in midprice_scaled.values().iter().zip(midprice_base.values()) {
        assert_close(actual, base * scale);
    }
}

#[test]
fn issue_26_inventory_and_execution_types_are_public() {
    for name in ["ACCBANDS", "BBANDS", "MIDPOINT", "MIDPRICE"] {
        assert_eq!(
            function(name).unwrap().status,
            ImplementationStatus::Implemented
        );
    }

    fn assert_execution<C, R, S>()
    where
        C: IndicatorConfig<BatchRunner = R, Stream = S>,
        R: PreparedBatchRunner<C>,
        S: StreamingComputation<C>,
    {
    }
    assert_execution::<ACCBANDSConfig, ACCBANDSBatchRunner, ACCBANDSStream>();
    assert_execution::<BBANDSConfig, BBANDSBatchRunner, BBANDSStream>();
    assert_execution::<MIDPOINTConfig, MIDPOINTBatchRunner, MIDPOINTStream>();
    assert_execution::<MIDPRICEConfig, MIDPRICEBatchRunner, MIDPRICEStream>();
}
