use ta_core::statistic::{
    BETA_vec, CORREL_vec, LINEARREG_ANGLE_vec, LINEARREG_INTERCEPT_vec, LINEARREG_SLOPE_vec,
    LINEARREG_vec, PairInput, PairTick, STDDEV_vec, STDDEV_vec_with_default_nbdev, TSF_vec,
    VAR_vec, VAR_vec_with_default_nbdev, VAR_with_default_nbdev, BETA, CORREL, LINEARREG,
    LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator, TalibError};

#[cfg(feature = "f32")]
const ABS_TOLERANCE: Float = 1e-4;
#[cfg(not(feature = "f32"))]
const ABS_TOLERANCE: Float = 1e-12;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 1e-4;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 1e-10;

fn assert_close(actual: Float, expected: Float) {
    let tolerance = ABS_TOLERANCE + REL_TOLERANCE * Float::max(actual.abs(), expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

fn assert_vec_close_with_nans(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN at {idx}, got {actual}");
        } else {
            assert_close(actual, expected);
        }
    }
}

#[test]
fn var_and_stddev_match_population_and_nbdev_semantics() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut variance = [0.0; 3];
    let mut ignored_nbdev = [0.0; 3];
    let mut stddev = [0.0; 3];

    let var_range = VAR(&real, 3, 1.0, &mut variance).unwrap();
    let ignored_range = VAR(&real, 3, 7.0, &mut ignored_nbdev).unwrap();
    let stddev_range = STDDEV(&real, 3, 2.0, &mut stddev).unwrap();

    assert_eq!(var_range, OutputRange::new(2, 3));
    assert_eq!(ignored_range, var_range);
    assert_eq!(stddev_range, var_range);
    for idx in 0..3 {
        assert_close(variance[idx], 2.0 as Float / 3.0 as Float);
        assert_eq!(ignored_nbdev[idx].to_bits(), variance[idx].to_bits());
        assert_close(
            stddev[idx],
            (2.0 as Float / 3.0 as Float).sqrt() * 2.0 as Float,
        );
    }
}

#[test]
fn variance_vec_defaults_and_indicator_surfaces_preserve_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let explicit = VAR_vec(&real, 3, 1.0).unwrap();
    let defaulted = VAR_vec_with_default_nbdev(&real, 3).unwrap();
    assert_vec_close_with_nans(&defaulted, &explicit);
    assert!(explicit[..2].iter().all(|value| value.is_nan()));
    assert_close(explicit[2], 2.0 as Float / 3.0 as Float);

    let stddev = STDDEV_vec_with_default_nbdev(&real, 3).unwrap();
    assert!(stddev[..2].iter().all(|value| value.is_nan()));
    assert_close(stddev[2], (2.0 as Float / 3.0 as Float).sqrt());

    let indicator = VAR::with_default_nbdev(3).unwrap();
    let mut compact = [0.0; 2];
    let range = Indicator::compute(&indicator, &real, &mut compact).unwrap();
    assert_eq!(indicator.period(), 3);
    assert_close(indicator.nbdev(), 1.0);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], 2.0 as Float / 3.0 as Float);

    let stddev_indicator = STDDEV::new(3, 2.0).unwrap();
    assert_close(stddev_indicator.nbdev(), 2.0);
    let via_trait = Indicator::compute_to_vec(&stddev_indicator, &real).unwrap();
    assert_vec_close_with_nans(&via_trait, &STDDEV_vec(&real, 3, 2.0).unwrap());
}

#[test]
fn variance_period_one_and_degenerate_windows_are_valid() {
    let real = [2.0, 4.0, 8.0];
    let mut output = [1.0; 3];
    let range = VAR_with_default_nbdev(&real, 1, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(0, 3));
    assert!(output.iter().all(|&value| value == 0.0 as Float));

    let constant = [7.0; 4];
    let stddev = STDDEV_vec(&constant, 2, -3.0).unwrap();
    assert!(stddev[0].is_nan());
    assert!(stddev[1..].iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn variance_preserves_selected_cancellation_behavior() {
    let real = [1_000_000.0, 1_000_001.0, 1_000_002.0];
    let output = VAR_vec_with_default_nbdev(&real, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[2], -65_536.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[2], 0.6666259765625 as Float);
}

#[test]
fn variance_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        VAR::new(0, 1.0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(matches!(
        STDDEV::new(1, 1.0),
        Err(TalibError::InvalidPeriod { period: 1, .. })
    ));
    assert!(VAR::new(100_000, 1.0).is_ok());
    assert!(STDDEV::new(100_000, 1.0).is_ok());
    assert!(matches!(
        VAR::new(100_001, 1.0),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    for nbdev in [Float::NAN, Float::INFINITY, Float::NEG_INFINITY] {
        assert!(matches!(
            VAR::new(3, nbdev),
            Err(TalibError::InvalidParameter { .. })
        ));
    }

    assert_eq!(VAR(&[], 1, 1.0, &mut []).unwrap(), OutputRange::empty());
    assert_eq!(STDDEV(&[], 2, 1.0, &mut []).unwrap(), OutputRange::empty());
    assert!(matches!(
        STDDEV(&[1.0], 2, 1.0, &mut []),
        Err(TalibError::InsufficientData {
            required: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        STDDEV(&[Float::NAN], 2, 1.0, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));

    let mut too_small = [123.0];
    assert!(matches!(
        VAR(&[1.0, 2.0, 3.0], 2, 1.0, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [123.0]);
}

#[test]
fn variance_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
    let mut batch_var = [0.0; 6];
    let var_range = VAR(&real, 3, 1.0, &mut batch_var).unwrap();
    let mut streaming_var = VAR::with_default_nbdev(3).unwrap();

    for (idx, &value) in real.iter().enumerate() {
        let streamed = streaming_var.next(value).unwrap();
        if idx < var_range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_eq!(
                streamed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }

    streaming_var.reset();
    for (idx, &value) in real.iter().enumerate() {
        let replayed = streaming_var.next(value).unwrap();
        if idx < var_range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }
    streaming_var.reset();
    assert!(streaming_var.next_checked(real[0]).unwrap().is_nan());

    let mut dirty = VAR::with_default_nbdev(3).unwrap();
    let mut clean = VAR::with_default_nbdev(3).unwrap();
    assert!(dirty.next(1.0).unwrap().is_none());
    assert!(clean.next(1.0).unwrap().is_none());
    assert!(dirty.next(Float::NAN).is_err());
    assert!(dirty.next(2.0).unwrap().is_none());
    assert!(clean.next(2.0).unwrap().is_none());
    assert_eq!(
        dirty.next(3.0).unwrap().unwrap().to_bits(),
        clean.next(3.0).unwrap().unwrap().to_bits()
    );

    let mut batch_stddev = [0.0; 6];
    let stddev_range = STDDEV(&real, 3, 2.0, &mut batch_stddev).unwrap();
    let mut streaming_stddev = STDDEV::new(3, 2.0).unwrap();
    for (idx, &value) in real.iter().enumerate() {
        let streamed = streaming_stddev.next(value).unwrap();
        if idx < stddev_range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_eq!(
                streamed.unwrap().to_bits(),
                batch_stddev[idx - stddev_range.beg_idx].to_bits()
            );
        }
    }
}

#[test]
fn correl_matches_positive_negative_constant_and_period_one_semantics() {
    let real0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let positive = [2.0, 4.0, 6.0, 8.0, 10.0];
    let negative = [10.0, 8.0, 6.0, 4.0, 2.0];
    let constant = [7.0; 5];
    let mut output = [0.0; 5];

    let range = CORREL(&real0, &positive, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    for &value in &output[..range.nb_element] {
        assert_close(value, 1.0);
    }

    CORREL(&real0, &negative, 3, &mut output).unwrap();
    for &value in &output[..3] {
        assert_close(value, -1.0);
    }

    CORREL(&real0, &constant, 3, &mut output).unwrap();
    assert!(output[..3].iter().all(|&value| value == 0.0 as Float));

    let range = CORREL(&real0, &positive, 1, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(0, 5));
    assert!(output.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn correl_vec_and_indicator_surfaces_preserve_pair_alignment() {
    let real0 = [1.0, 2.0, 4.0, 8.0];
    let real1 = [2.0, 3.0, 5.0, 9.0];
    let padded = CORREL_vec(&real0, &real1, 3).unwrap();
    assert_eq!(padded.len(), real0.len());
    assert!(padded[..2].iter().all(|value| value.is_nan()));
    assert!(padded[2..].iter().all(|value| value.is_finite()));

    let indicator = CORREL::new(3).unwrap();
    let mut compact = [0.0; 3];
    let range = Indicator::compute(
        &indicator,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
        &mut compact,
    )
    .unwrap();
    assert_eq!(indicator.period(), 3);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], padded[2]);
    assert_close(compact[1], padded[3]);
}

#[test]
fn correl_preserves_selected_cancellation_behavior() {
    let real0 = [100_000.0, 100_001.0, 100_002.0];
    let real1 = [200_000.0, 200_002.0, 200_004.0];
    let output = CORREL_vec(&real0, &real1, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[2], -1.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[2], 1.0 as Float);
}

#[test]
fn correl_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        CORREL::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(CORREL::new(100_000).is_ok());
    assert!(matches!(
        CORREL::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(CORREL(&[], &[], 1, &mut []).unwrap(), OutputRange::empty());

    let period_first = CORREL(&[Float::NAN], &[], 0, &mut []).unwrap_err();
    assert!(matches!(
        period_first,
        TalibError::InvalidPeriod { period: 0, .. }
    ));
    let length_first = CORREL(&[Float::NAN], &[], 1, &mut []).unwrap_err();
    assert!(length_first
        .to_string()
        .contains("must have the same length"));
    let finite_first = CORREL(&[Float::NAN], &[1.0], 3, &mut []).unwrap_err();
    assert!(finite_first.to_string().contains("must be finite"));
    assert!(matches!(
        CORREL(&[Float::INFINITY], &[1.0], 1, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        CORREL(&[1.0, 2.0], &[1.0, 2.0], 3, &mut []),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));

    let mut too_small = [321.0];
    assert!(matches!(
        CORREL(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], 2, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [321.0]);
}

#[test]
fn correl_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real0 = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
    let real1 = [2.0, 1.0, 5.0, 3.0, 8.0, 4.0, 9.0, 6.0];
    let mut batch = [0.0; 6];
    let range = CORREL(&real0, &real1, 3, &mut batch).unwrap();
    let mut streaming = CORREL::new(3).unwrap();

    for idx in 0..real0.len() {
        let value = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(value.is_none());
        } else {
            assert_eq!(
                value.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }

    streaming.reset();
    for idx in 0..real0.len() {
        let replayed = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }
    streaming.reset();
    assert!(streaming
        .next_checked(PairTick {
            real0: real0[0],
            real1: real1[0],
        })
        .unwrap()
        .is_nan());

    let mut dirty = CORREL::new(2).unwrap();
    let mut clean = CORREL::new(2).unwrap();
    let first = PairTick {
        real0: 1.0,
        real1: 2.0,
    };
    assert!(dirty.next(first).unwrap().is_none());
    assert!(clean.next(first).unwrap().is_none());
    assert!(dirty
        .next(PairTick {
            real0: 3.0,
            real1: Float::NAN,
        })
        .is_err());
    let second = PairTick {
        real0: 2.0,
        real1: 4.0,
    };
    assert_eq!(
        dirty.next(second).unwrap().unwrap().to_bits(),
        clean.next(second).unwrap().unwrap().to_bits()
    );
}

#[test]
fn beta_uses_returns_real0_denominator_and_ta_zero_boundaries() {
    let market = [100.0, 110.0, 132.0, 118.8];
    let asset = [50.0, 60.0, 84.0, 67.2];
    let mut output = [0.0; 1];

    let range = BETA(&market, &asset, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(3, 1));
    assert_close(output[0], 2.0);

    BETA(&asset, &market, 3, &mut output).unwrap();
    assert_close(output[0], 0.5);

    let interior_near_zero = [5e-15, 10.0, 20.0];
    let comparison = [5.0, 6.0, 8.0];
    BETA(&interior_near_zero, &comparison, 2, &mut output).unwrap();
    assert_close(output[0], 2.0 as Float / 15.0 as Float);

    let positive_endpoint = [1e-14, 10.0, 20.0];
    BETA(&positive_endpoint, &comparison, 2, &mut output).unwrap();
    assert!(output[0].is_sign_negative() && output[0] != 0.0 as Float);

    let negative_endpoint = [-1e-14, 10.0, 20.0];
    BETA(&negative_endpoint, &comparison, 2, &mut output).unwrap();
    assert!(output[0].is_sign_positive() && output[0] != 0.0 as Float);
}

#[test]
fn beta_vec_indicator_and_period_one_use_extra_lookback() {
    let real0 = [10.0, 11.0, 13.0, 12.0];
    let real1 = [20.0, 21.0, 24.0, 22.0];
    let padded = BETA_vec(&real0, &real1, 2).unwrap();
    assert_eq!(padded.len(), real0.len());
    assert!(padded[..2].iter().all(|value| value.is_nan()));
    assert!(padded[2..].iter().all(|value| value.is_finite()));

    let indicator = BETA::new(2).unwrap();
    let mut compact = [0.0; 2];
    let range = Indicator::compute(
        &indicator,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
        &mut compact,
    )
    .unwrap();
    assert_eq!(indicator.period(), 2);
    assert_eq!(indicator.lookback(), 2);
    assert_eq!(range, OutputRange::new(2, 2));
    assert_close(compact[0], padded[2]);
    assert_close(compact[1], padded[3]);

    let mut period_one = [1.0; 3];
    let range = BETA(&real0, &real1, 1, &mut period_one).unwrap();
    assert_eq!(range, OutputRange::new(1, 3));
    assert!(period_one.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn beta_preserves_selected_cancellation_behavior() {
    let real0 = [20_000_000.0, 20_000_001.0, 20_000_003.0, 20_000_002.0];
    let real1 = [40_000_000.0, 40_000_004.0, 40_000_012.0, 40_000_008.0];
    let output = BETA_vec(&real0, &real1, 3).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[3], 0.9285713 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[3], 1.999999835714329 as Float);
}

#[test]
fn beta_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        BETA::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(BETA::new(100_000).is_ok());
    assert!(matches!(
        BETA::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(BETA(&[], &[], 1, &mut []).unwrap(), OutputRange::empty());

    let period_first = BETA(&[Float::NAN], &[], 0, &mut []).unwrap_err();
    assert!(matches!(
        period_first,
        TalibError::InvalidPeriod { period: 0, .. }
    ));
    let length_first = BETA(&[Float::NAN], &[], 1, &mut []).unwrap_err();
    assert!(length_first
        .to_string()
        .contains("must have the same length"));
    let finite_first = BETA(&[Float::NAN], &[1.0], 2, &mut []).unwrap_err();
    assert!(finite_first.to_string().contains("must be finite"));
    assert!(matches!(
        BETA(&[Float::INFINITY, 2.0], &[1.0, 2.0], 1, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        BETA(&[1.0, 2.0], &[1.0, 2.0], 2, &mut []),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));

    let mut too_small = [456.0];
    assert!(matches!(
        BETA(
            &[1.0, 2.0, 3.0, 4.0],
            &[2.0, 3.0, 5.0, 8.0],
            2,
            &mut too_small
        ),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [456.0]);
}

#[test]
fn beta_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real0 = [10.0, 11.0, 13.0, 12.0, 15.0, 14.0, 18.0, 17.0];
    let real1 = [20.0, 22.0, 23.0, 21.0, 26.0, 24.0, 29.0, 28.0];
    let mut batch = [0.0; 6];
    let range = BETA(&real0, &real1, 2, &mut batch).unwrap();
    let mut streaming = BETA::new(2).unwrap();

    for idx in 0..real0.len() {
        let value = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(value.is_none());
        } else {
            assert_eq!(
                value.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }

    streaming.reset();
    for idx in 0..real0.len() {
        let replayed = streaming
            .next(PairTick {
                real0: real0[idx],
                real1: real1[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch[idx - range.beg_idx].to_bits()
            );
        }
    }
    streaming.reset();
    assert!(streaming
        .next_checked(PairTick {
            real0: real0[0],
            real1: real1[0],
        })
        .unwrap()
        .is_nan());

    let mut dirty = BETA::new(2).unwrap();
    let mut clean = BETA::new(2).unwrap();
    let first = PairTick {
        real0: 10.0,
        real1: 20.0,
    };
    assert!(dirty.next(first).unwrap().is_none());
    assert!(clean.next(first).unwrap().is_none());
    assert!(dirty
        .next(PairTick {
            real0: Float::NAN,
            real1: 21.0,
        })
        .is_err());
    for tick in [
        PairTick {
            real0: 11.0,
            real1: 22.0,
        },
        PairTick {
            real0: 13.0,
            real1: 23.0,
        },
    ] {
        let dirty_value = dirty.next(tick).unwrap();
        let clean_value = clean.next(tick).unwrap();
        match (dirty_value, clean_value) {
            (Some(dirty_value), Some(clean_value)) => {
                assert_eq!(dirty_value.to_bits(), clean_value.to_bits());
            }
            (None, None) => {}
            pair => panic!("streaming state diverged after invalid tick: {pair:?}"),
        }
    }
}

#[test]
fn regression_family_matches_closed_form_projections() {
    let real = [1.0, 2.0, 3.0];
    let mut output = [0.0; 1];

    assert_eq!(
        LINEARREG(&real, 3, &mut output).unwrap(),
        OutputRange::new(2, 1)
    );
    assert_close(output[0], 3.0);
    LINEARREG_SLOPE(&real, 3, &mut output).unwrap();
    assert_close(output[0], 1.0);
    LINEARREG_INTERCEPT(&real, 3, &mut output).unwrap();
    assert_close(output[0], 1.0);
    LINEARREG_ANGLE(&real, 3, &mut output).unwrap();
    assert_close(output[0], 45.0);
    TSF(&real, 3, &mut output).unwrap();
    assert_close(output[0], 4.0);
}

#[test]
fn regression_vec_struct_and_constant_surfaces_preserve_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    for output in [
        LINEARREG_vec(&real, 3).unwrap(),
        LINEARREG_SLOPE_vec(&real, 3).unwrap(),
        LINEARREG_INTERCEPT_vec(&real, 3).unwrap(),
        LINEARREG_ANGLE_vec(&real, 3).unwrap(),
        TSF_vec(&real, 3).unwrap(),
    ] {
        assert_eq!(output.len(), real.len());
        assert!(output[..2].iter().all(|value| value.is_nan()));
        assert!(output[2..].iter().all(|value| value.is_finite()));
    }

    let indicator = LINEARREG::new(3).unwrap();
    let mut compact = [0.0; 3];
    let range = Indicator::compute(&indicator, &real, &mut compact).unwrap();
    assert_eq!(indicator.period(), 3);
    assert_eq!(range, OutputRange::new(2, 3));
    for (&compact, &padded) in compact.iter().zip(&LINEARREG_vec(&real, 3).unwrap()[2..]) {
        assert_close(compact, padded);
    }

    let constant = [7.0; 5];
    let endpoint = LINEARREG_vec(&constant, 3).unwrap();
    let slope = LINEARREG_SLOPE_vec(&constant, 3).unwrap();
    let intercept = LINEARREG_INTERCEPT_vec(&constant, 3).unwrap();
    let angle = LINEARREG_ANGLE_vec(&constant, 3).unwrap();
    let forecast = TSF_vec(&constant, 3).unwrap();
    for idx in 2..constant.len() {
        assert_close(endpoint[idx], 7.0);
        assert_close(slope[idx], 0.0);
        assert_close(intercept[idx], 7.0);
        assert_close(angle[idx], 0.0);
        assert_close(forecast[idx], 7.0);
    }
}

#[test]
fn regression_matches_pinned_rolling_fma_oracle() {
    let real = [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0];
    let endpoint = LINEARREG_vec(&real, 4).unwrap();
    let slope = LINEARREG_SLOPE_vec(&real, 4).unwrap();
    let intercept = LINEARREG_INTERCEPT_vec(&real, 4).unwrap();
    let angle = LINEARREG_ANGLE_vec(&real, 4).unwrap();
    let forecast = TSF_vec(&real, 4).unwrap();

    let expected_endpoint = [14.1, 14.5, 17.5, 17.5, 19.8];
    let expected_slope = [1.4, 1.0, 2.0, 1.0, 1.7];
    let expected_intercept = [9.9, 11.5, 11.5, 14.5, 14.7];
    let expected_angle = [
        54.46232220802562,
        45.0,
        63.43494882292202,
        45.0,
        59.53445508054013,
    ];
    let expected_forecast = [15.5, 15.5, 19.5, 18.5, 21.5];

    for idx in 0..5 {
        let output_idx = idx + 3;
        assert_close(endpoint[output_idx], expected_endpoint[idx]);
        assert_close(slope[output_idx], expected_slope[idx]);
        assert_close(intercept[output_idx], expected_intercept[idx]);
        assert_close(angle[output_idx], expected_angle[idx]);
        assert_close(forecast[output_idx], expected_forecast[idx]);
    }

    #[cfg(feature = "f32")]
    let expected_endpoint_bits = [
        0x4161_9999_u32,
        0x4168_0000,
        0x418c_0000,
        0x418c_0000,
        0x419e_6666,
    ];
    #[cfg(not(feature = "f32"))]
    let expected_endpoint_bits = [
        0x402c_3333_3333_3333_u64,
        0x402d_0000_0000_0000,
        0x4031_8000_0000_0000,
        0x4031_8000_0000_0000,
        0x4033_cccc_cccc_cccd,
    ];
    for (idx, &expected_bits) in expected_endpoint_bits.iter().enumerate() {
        assert_eq!(endpoint[idx + 3].to_bits(), expected_bits);
    }
}

#[test]
fn regression_preserves_large_baseline_cancellation_behavior() {
    let real = [
        10_000_000.0,
        10_000_001.0,
        10_000_002.0,
        10_000_003.0,
        10_000_004.0,
        10_000_005.0,
    ];
    let slope = LINEARREG_SLOPE_vec(&real, 3).unwrap();

    #[cfg(feature = "f32")]
    let expected_bits = [0x3faa_aaab_u32, 0x402a_aaab, 0x4080_0000, 0x4080_0000];
    #[cfg(not(feature = "f32"))]
    let expected_bits = [
        0x3ff0_0000_0000_0000_u64,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
    ];
    for (idx, &expected_bits) in expected_bits.iter().enumerate() {
        assert_eq!(slope[idx + 2].to_bits(), expected_bits);
    }
}

#[test]
fn regression_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        LINEARREG::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_SLOPE::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_INTERCEPT::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_ANGLE::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(TSF::new(1), Err(TalibError::InvalidPeriod { .. })));
    assert!(LINEARREG::new(100_000).is_ok());
    assert!(matches!(
        TSF::new(100_001),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    assert_eq!(LINEARREG(&[], 2, &mut []).unwrap(), OutputRange::empty());

    let period_first = LINEARREG(&[Float::NAN], 1, &mut []).unwrap_err();
    assert!(matches!(period_first, TalibError::InvalidPeriod { .. }));
    assert!(matches!(
        LINEARREG(&[Float::INFINITY], 2, &mut []),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        LINEARREG(&[1.0], 2, &mut []),
        Err(TalibError::InsufficientData {
            required: 2,
            actual: 1
        })
    ));

    let mut too_small = [789.0];
    assert!(matches!(
        LINEARREG(&[1.0, 2.0, 3.0], 2, &mut too_small),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(too_small, [789.0]);
}

#[test]
fn regression_streaming_matches_batch_across_wrap_reset_and_invalid_tick() {
    let real = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];

    macro_rules! assert_parity {
        ($function:ident, $indicator:ident) => {{
            let mut batch = [0.0; 6];
            let range = $function(&real, 3, &mut batch).unwrap();
            let mut streaming = $indicator::new(3).unwrap();
            for (idx, &value) in real.iter().enumerate() {
                let streamed = streaming.next(value).unwrap();
                if idx < range.beg_idx {
                    assert!(streamed.is_none());
                } else {
                    assert_eq!(
                        streamed.unwrap().to_bits(),
                        batch[idx - range.beg_idx].to_bits()
                    );
                }
            }
            streaming.reset();
            for (idx, &value) in real.iter().enumerate() {
                let replayed = streaming.next(value).unwrap();
                if idx < range.beg_idx {
                    assert!(replayed.is_none());
                } else {
                    assert_eq!(
                        replayed.unwrap().to_bits(),
                        batch[idx - range.beg_idx].to_bits()
                    );
                }
            }
        }};
    }

    assert_parity!(LINEARREG, LINEARREG);
    assert_parity!(LINEARREG_SLOPE, LINEARREG_SLOPE);
    assert_parity!(LINEARREG_INTERCEPT, LINEARREG_INTERCEPT);
    assert_parity!(LINEARREG_ANGLE, LINEARREG_ANGLE);
    assert_parity!(TSF, TSF);

    let mut dirty = LINEARREG::new(2).unwrap();
    let mut clean = LINEARREG::new(2).unwrap();
    assert!(dirty.next(1.0).unwrap().is_none());
    assert!(clean.next(1.0).unwrap().is_none());
    assert!(dirty.next(Float::NAN).is_err());
    assert_eq!(
        dirty.next(2.0).unwrap().unwrap().to_bits(),
        clean.next(2.0).unwrap().unwrap().to_bits()
    );
    dirty.reset();
    assert!(dirty.next_checked(1.0).unwrap().is_nan());
}
