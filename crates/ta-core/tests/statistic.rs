use ta_core::statistic::{
    BETAConfig, CORRELConfig, LINEARREGConfig, LINEARREG_ANGLEConfig, LINEARREG_INTERCEPTConfig,
    LINEARREG_SLOPEConfig, PairInput, PairTick, STDDEVConfig, TSFConfig, VARConfig,
    VAR_with_default_nbdev, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT,
    LINEARREG_SLOPE, STDDEV, TSF, VAR,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

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
fn variance_config_default_nbdev_matches_explicit_and_compact_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let mut explicit = [0.0; 2];
    let mut defaulted = [0.0; 2];

    let explicit_range = VAR(&real, 3, 1.0, &mut explicit).unwrap();
    let default_range = VAR_with_default_nbdev(&real, 3, &mut defaulted).unwrap();
    assert_eq!(explicit_range, OutputRange::new(2, 2));
    assert_eq!(default_range, explicit_range);
    assert_eq!(explicit, defaulted);

    let config = VARConfig::with_default_nbdev(3).unwrap();
    assert_eq!(config.period(), 3);
    assert_close(config.nbdev(), 1.0);
    let owned = IndicatorConfig::compute(&config, &real).unwrap();
    assert_eq!(owned.range(), explicit_range);
    assert_close(owned.values()[0], 2.0 as Float / 3.0 as Float);

    let stddev = STDDEVConfig::new(3, 2.0).unwrap();
    assert_close(stddev.nbdev(), 2.0);
    let owned_stddev = IndicatorConfig::compute(&stddev, &real).unwrap();
    assert_eq!(owned_stddev.range(), OutputRange::new(2, 2));
    assert_close(
        owned_stddev.values()[0],
        (2.0 as Float / 3.0 as Float).sqrt() * 2.0 as Float,
    );
}

#[test]
fn variance_period_one_and_degenerate_windows_are_valid() {
    let real = [2.0, 4.0, 8.0];
    let mut output = [1.0; 3];
    let range = VAR_with_default_nbdev(&real, 1, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(0, 3));
    assert!(output.iter().all(|&value| value == 0.0 as Float));

    let constant = [7.0; 4];
    let mut stddev = [1.0; 3];
    let range = STDDEV(&constant, 2, -3.0, &mut stddev).unwrap();
    assert_eq!(range, OutputRange::new(1, 3));
    assert!(stddev.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn variance_preserves_selected_cancellation_behavior() {
    let real = [1_000_000.0, 1_000_001.0, 1_000_002.0];
    let mut output = [0.0; 1];
    VAR_with_default_nbdev(&real, 3, &mut output).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[0], -65_536.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[0], 0.6666259765625 as Float);
}

#[test]
fn variance_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        VARConfig::new(0, 1.0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(matches!(
        STDDEVConfig::new(1, 1.0),
        Err(TalibError::InvalidPeriod { period: 1, .. })
    ));
    assert!(VARConfig::new(100_000, 1.0).is_ok());
    assert!(STDDEVConfig::new(100_000, 1.0).is_ok());
    assert!(matches!(
        VARConfig::new(100_001, 1.0),
        Err(TalibError::InvalidPeriod {
            period: 100_001,
            ..
        })
    ));
    for nbdev in [Float::NAN, Float::INFINITY, Float::NEG_INFINITY] {
        assert!(matches!(
            VARConfig::new(3, nbdev),
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
    let var_config = VARConfig::with_default_nbdev(3).unwrap();
    let mut streaming_var = IndicatorConfig::stream(&var_config).unwrap();

    for (idx, &value) in real.iter().enumerate() {
        let streamed = StreamingComputation::<VARConfig>::next(&mut streaming_var, value).unwrap();
        if idx < var_range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_eq!(
                streamed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }

    StreamingComputation::<VARConfig>::reset(&mut streaming_var);
    for (idx, &value) in real.iter().enumerate() {
        let replayed = StreamingComputation::<VARConfig>::next(&mut streaming_var, value).unwrap();
        if idx < var_range.beg_idx {
            assert!(replayed.is_none());
        } else {
            assert_eq!(
                replayed.unwrap().to_bits(),
                batch_var[idx - var_range.beg_idx].to_bits()
            );
        }
    }
    StreamingComputation::<VARConfig>::reset(&mut streaming_var);
    assert!(
        StreamingComputation::<VARConfig>::next(&mut streaming_var, real[0])
            .unwrap()
            .is_none()
    );

    let mut dirty = IndicatorConfig::stream(&var_config).unwrap();
    let mut clean = IndicatorConfig::stream(&var_config).unwrap();
    assert!(StreamingComputation::<VARConfig>::next(&mut dirty, 1.0)
        .unwrap()
        .is_none());
    assert!(StreamingComputation::<VARConfig>::next(&mut clean, 1.0)
        .unwrap()
        .is_none());
    assert!(StreamingComputation::<VARConfig>::next(&mut dirty, Float::NAN).is_err());
    assert!(StreamingComputation::<VARConfig>::next(&mut dirty, 2.0)
        .unwrap()
        .is_none());
    assert!(StreamingComputation::<VARConfig>::next(&mut clean, 2.0)
        .unwrap()
        .is_none());
    assert_eq!(
        StreamingComputation::<VARConfig>::next(&mut dirty, 3.0)
            .unwrap()
            .unwrap()
            .to_bits(),
        StreamingComputation::<VARConfig>::next(&mut clean, 3.0)
            .unwrap()
            .unwrap()
            .to_bits()
    );

    let mut batch_stddev = [0.0; 6];
    let stddev_range = STDDEV(&real, 3, 2.0, &mut batch_stddev).unwrap();
    let stddev_config = STDDEVConfig::new(3, 2.0).unwrap();
    let mut streaming_stddev = IndicatorConfig::stream(&stddev_config).unwrap();
    for (idx, &value) in real.iter().enumerate() {
        let streamed =
            StreamingComputation::<STDDEVConfig>::next(&mut streaming_stddev, value).unwrap();
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
fn correl_config_surfaces_preserve_pair_alignment() {
    let real0 = [1.0, 2.0, 4.0, 8.0];
    let real1 = [2.0, 3.0, 5.0, 9.0];

    let config = CORRELConfig::new(3).unwrap();
    assert_eq!(config.period(), 3);
    let owned = IndicatorConfig::compute(
        &config,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
    )
    .unwrap();
    assert_eq!(owned.range(), OutputRange::new(2, 2));
    assert!(owned.values().iter().all(|value| value.is_finite()));

    let mut compact = [0.0; 2];
    let range = IndicatorConfig::compute_into(
        &config,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
        &mut compact,
    )
    .unwrap();
    assert_eq!(range, owned.range());
    assert_eq!(&compact[..2], owned.values());
}

#[test]
fn correl_preserves_selected_cancellation_behavior() {
    let real0 = [100_000.0, 100_001.0, 100_002.0];
    let real1 = [200_000.0, 200_002.0, 200_004.0];
    let mut output = [0.0; 1];
    CORREL(&real0, &real1, 3, &mut output).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[0], -1.0 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[0], 1.0 as Float);
}

#[test]
fn correl_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        CORRELConfig::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(CORRELConfig::new(100_000).is_ok());
    assert!(matches!(
        CORRELConfig::new(100_001),
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
    let config = CORRELConfig::new(3).unwrap();
    let mut streaming = IndicatorConfig::stream(&config).unwrap();

    for idx in 0..real0.len() {
        let value = StreamingComputation::<CORRELConfig>::next(
            &mut streaming,
            PairTick {
                real0: real0[idx],
                real1: real1[idx],
            },
        )
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

    StreamingComputation::<CORRELConfig>::reset(&mut streaming);
    for idx in 0..real0.len() {
        let replayed = StreamingComputation::<CORRELConfig>::next(
            &mut streaming,
            PairTick {
                real0: real0[idx],
                real1: real1[idx],
            },
        )
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

    let mut dirty = IndicatorConfig::stream(&CORRELConfig::new(2).unwrap()).unwrap();
    let mut clean = IndicatorConfig::stream(&CORRELConfig::new(2).unwrap()).unwrap();
    let first = PairTick {
        real0: 1.0,
        real1: 2.0,
    };
    assert!(
        StreamingComputation::<CORRELConfig>::next(&mut dirty, first)
            .unwrap()
            .is_none()
    );
    assert!(
        StreamingComputation::<CORRELConfig>::next(&mut clean, first)
            .unwrap()
            .is_none()
    );
    assert!(StreamingComputation::<CORRELConfig>::next(
        &mut dirty,
        PairTick {
            real0: 3.0,
            real1: Float::NAN,
        }
    )
    .is_err());
    let second = PairTick {
        real0: 2.0,
        real1: 4.0,
    };
    assert_eq!(
        StreamingComputation::<CORRELConfig>::next(&mut dirty, second)
            .unwrap()
            .unwrap()
            .to_bits(),
        StreamingComputation::<CORRELConfig>::next(&mut clean, second)
            .unwrap()
            .unwrap()
            .to_bits()
    );
}

#[test]
fn prepared_capacity_precedes_paired_statistic_input_alignment() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let mut output = [];
    let capacity_error = TalibError::PreparedCapacityExceeded {
        max_input_len: within.len(),
        actual_input_len: oversized.len(),
    };
    let input = PairInput {
        real0: &within,
        real1: &oversized,
    };

    let mut beta = BETAConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        beta.compute_into(input, &mut output).unwrap_err(),
        capacity_error
    );

    let mut correl = CORRELConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        correl.compute_into(input, &mut output).unwrap_err(),
        capacity_error
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
fn beta_config_and_period_one_use_extra_lookback() {
    let real0 = [10.0, 11.0, 13.0, 12.0];
    let real1 = [20.0, 21.0, 24.0, 22.0];

    let config = BETAConfig::new(2).unwrap();
    let owned = IndicatorConfig::compute(
        &config,
        PairInput {
            real0: &real0,
            real1: &real1,
        },
    )
    .unwrap();
    assert_eq!(config.period(), 2);
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert_eq!(owned.range(), OutputRange::new(2, 2));
    assert!(owned.values().iter().all(|value| value.is_finite()));

    let mut period_one = [1.0; 3];
    let range = BETA(&real0, &real1, 1, &mut period_one).unwrap();
    assert_eq!(range, OutputRange::new(1, 3));
    assert!(period_one.iter().all(|&value| value == 0.0 as Float));
}

#[test]
fn beta_preserves_selected_cancellation_behavior() {
    let real0 = [20_000_000.0, 20_000_001.0, 20_000_003.0, 20_000_002.0];
    let real1 = [40_000_000.0, 40_000_004.0, 40_000_012.0, 40_000_008.0];
    let mut output = [0.0; 1];
    BETA(&real0, &real1, 3, &mut output).unwrap();

    #[cfg(feature = "f32")]
    assert_eq!(output[0], 0.9285713 as Float);
    #[cfg(not(feature = "f32"))]
    assert_close(output[0], 1.999999835714329 as Float);
}

#[test]
fn beta_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        BETAConfig::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(BETAConfig::new(100_000).is_ok());
    assert!(matches!(
        BETAConfig::new(100_001),
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
    let config = BETAConfig::new(2).unwrap();
    let mut streaming = IndicatorConfig::stream(&config).unwrap();

    for idx in 0..real0.len() {
        let value = StreamingComputation::<BETAConfig>::next(
            &mut streaming,
            PairTick {
                real0: real0[idx],
                real1: real1[idx],
            },
        )
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

    StreamingComputation::<BETAConfig>::reset(&mut streaming);
    for idx in 0..real0.len() {
        let replayed = StreamingComputation::<BETAConfig>::next(
            &mut streaming,
            PairTick {
                real0: real0[idx],
                real1: real1[idx],
            },
        )
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

    let mut dirty = IndicatorConfig::stream(&config).unwrap();
    let mut clean = IndicatorConfig::stream(&config).unwrap();
    let first = PairTick {
        real0: 10.0,
        real1: 20.0,
    };
    assert!(StreamingComputation::<BETAConfig>::next(&mut dirty, first)
        .unwrap()
        .is_none());
    assert!(StreamingComputation::<BETAConfig>::next(&mut clean, first)
        .unwrap()
        .is_none());
    assert!(StreamingComputation::<BETAConfig>::next(
        &mut dirty,
        PairTick {
            real0: Float::NAN,
            real1: 21.0,
        }
    )
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
        let dirty_value = StreamingComputation::<BETAConfig>::next(&mut dirty, tick).unwrap();
        let clean_value = StreamingComputation::<BETAConfig>::next(&mut clean, tick).unwrap();
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
fn regression_config_and_constant_surfaces_preserve_alignment() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];

    let config = LINEARREGConfig::new(3).unwrap();
    assert_eq!(config.period(), 3);
    let owned = IndicatorConfig::compute(&config, &real).unwrap();
    assert_eq!(owned.range(), OutputRange::new(2, 3));
    let mut compact = [0.0; 3];
    let range = IndicatorConfig::compute_into(&config, &real, &mut compact).unwrap();
    assert_eq!(range, owned.range());
    assert_eq!(&compact[..3], owned.values());

    let constant = [7.0; 5];
    let mut output = [0.0; 3];

    LINEARREG(&constant, 3, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 7.0 as Float));
    LINEARREG_SLOPE(&constant, 3, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 0.0 as Float));
    LINEARREG_INTERCEPT(&constant, 3, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 7.0 as Float));
    LINEARREG_ANGLE(&constant, 3, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 0.0 as Float));
    TSF(&constant, 3, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 7.0 as Float));
}

#[test]
fn regression_matches_pinned_rolling_fma_oracle() {
    let real = [10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0];
    let mut endpoint = [0.0; 5];
    let mut slope = [0.0; 5];
    let mut intercept = [0.0; 5];
    let mut angle = [0.0; 5];
    let mut forecast = [0.0; 5];

    LINEARREG(&real, 4, &mut endpoint).unwrap();
    LINEARREG_SLOPE(&real, 4, &mut slope).unwrap();
    LINEARREG_INTERCEPT(&real, 4, &mut intercept).unwrap();
    LINEARREG_ANGLE(&real, 4, &mut angle).unwrap();
    TSF(&real, 4, &mut forecast).unwrap();

    let expected_endpoint = [14.1, 14.5, 17.5, 17.5, 19.8];
    let expected_slope = [1.4, 1.0, 2.0, 1.0, 1.7];
    let expected_intercept = [9.9, 11.5, 11.5, 14.5, 14.7];
    let expected_angle: [f64; 5] = [
        54.46232220802562,
        45.0,
        63.43494882292202,
        45.0,
        59.53445508054013,
    ];
    let expected_forecast = [15.5, 15.5, 19.5, 18.5, 21.5];

    for idx in 0..5 {
        assert_close(endpoint[idx], expected_endpoint[idx]);
        assert_close(slope[idx], expected_slope[idx]);
        assert_close(intercept[idx], expected_intercept[idx]);
        assert_close(angle[idx], expected_angle[idx] as Float);
        assert_close(forecast[idx], expected_forecast[idx]);
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
        assert_eq!(endpoint[idx].to_bits(), expected_bits);
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
    let mut slope = [0.0; 4];
    LINEARREG_SLOPE(&real, 3, &mut slope).unwrap();

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
        assert_eq!(slope[idx].to_bits(), expected_bits);
    }
}

#[test]
fn regression_validation_is_typed_ordered_and_non_mutating() {
    assert!(matches!(
        LINEARREGConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_SLOPEConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_INTERCEPTConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        LINEARREG_ANGLEConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        TSFConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(LINEARREGConfig::new(100_000).is_ok());
    assert!(matches!(
        TSFConfig::new(100_001),
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
        ($function:ident, $config:ident) => {{
            let mut batch = [0.0; 6];
            let range = $function(&real, 3, &mut batch).unwrap();
            let config = $config::new(3).unwrap();
            let mut streaming = IndicatorConfig::stream(&config).unwrap();
            for (idx, &value) in real.iter().enumerate() {
                let streamed =
                    StreamingComputation::<$config>::next(&mut streaming, value).unwrap();
                if idx < range.beg_idx {
                    assert!(streamed.is_none());
                } else {
                    assert_eq!(
                        streamed.unwrap().to_bits(),
                        batch[idx - range.beg_idx].to_bits()
                    );
                }
            }
            StreamingComputation::<$config>::reset(&mut streaming);
            for (idx, &value) in real.iter().enumerate() {
                let replayed =
                    StreamingComputation::<$config>::next(&mut streaming, value).unwrap();
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

    assert_parity!(LINEARREG, LINEARREGConfig);
    assert_parity!(LINEARREG_SLOPE, LINEARREG_SLOPEConfig);
    assert_parity!(LINEARREG_INTERCEPT, LINEARREG_INTERCEPTConfig);
    assert_parity!(LINEARREG_ANGLE, LINEARREG_ANGLEConfig);
    assert_parity!(TSF, TSFConfig);

    let dirty_config = LINEARREGConfig::new(2).unwrap();
    let mut dirty = IndicatorConfig::stream(&dirty_config).unwrap();
    let mut clean = IndicatorConfig::stream(&dirty_config).unwrap();
    assert!(
        StreamingComputation::<LINEARREGConfig>::next(&mut dirty, 1.0)
            .unwrap()
            .is_none()
    );
    assert!(
        StreamingComputation::<LINEARREGConfig>::next(&mut clean, 1.0)
            .unwrap()
            .is_none()
    );
    assert!(StreamingComputation::<LINEARREGConfig>::next(&mut dirty, Float::NAN).is_err());
    assert_eq!(
        StreamingComputation::<LINEARREGConfig>::next(&mut dirty, 2.0)
            .unwrap()
            .unwrap()
            .to_bits(),
        StreamingComputation::<LINEARREGConfig>::next(&mut clean, 2.0)
            .unwrap()
            .unwrap()
            .to_bits()
    );
    StreamingComputation::<LINEARREGConfig>::reset(&mut dirty);
    assert!(
        StreamingComputation::<LINEARREGConfig>::next(&mut dirty, 1.0)
            .unwrap()
            .is_none()
    );
}
