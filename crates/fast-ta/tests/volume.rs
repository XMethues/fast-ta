use fast_ta::volume::{
    ADConfig, ADInput, ADOSCConfig, ADOSCInput, ADOSCTick, ADTick, OBVConfig, OBVInput, OBVTick,
    AD, ADOSC, OBV,
};
use fast_ta::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-5 as Float,
        "expected {expected}, got {actual}"
    );
}

fn fixture() -> ([Float; 4], [Float; 4], [Float; 4], [Float; 4]) {
    (
        [10.0, 12.0, 11.0, 15.0],
        [8.0, 8.0, 9.0, 13.0],
        [10.0, 11.0, 9.0, 14.0],
        [100.0, 200.0, 50.0, 300.0],
    )
}

#[test]
fn ad_function_writes_compact_outputs() {
    let (high, low, close, volume) = fixture();
    let mut output = [0.0; 4];

    let range = AD(&high, &low, &close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(0, 4));
    for (actual, expected) in output.into_iter().zip([100.0, 200.0, 150.0, 150.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn ad_config_implements_indicator_compute() {
    let (high, low, close, volume) = fixture();
    let config = ADConfig::new();
    let mut output = [0.0; 4];

    let range = IndicatorConfig::compute_into(
        &config,
        ADInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(IndicatorConfig::lookback(&config), 0);
    assert_eq!(range, OutputRange::new(0, 4));
    assert_close(output[2], 150.0);
}

#[test]
fn ad_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = fixture();
    let mut batch = [0.0; 4];
    AD(&high, &low, &close, &volume, &mut batch).unwrap();

    let config = ADConfig::new();
    let mut stream = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..high.len() {
        let streamed = StreamingComputation::<ADConfig>::next(
            &mut stream,
            ADTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            },
        )
        .unwrap()
        .unwrap();
        assert_close(streamed, batch[idx]);
    }

    StreamingComputation::<ADConfig>::reset(&mut stream);
    assert_close(
        StreamingComputation::<ADConfig>::next(
            &mut stream,
            ADTick {
                high: high[0],
                low: low[0],
                close: close[0],
                volume: volume[0],
            },
        )
        .unwrap()
        .unwrap(),
        100.0,
    );
}

#[test]
fn ad_non_positive_range_contributes_zero() {
    let high = [10.0, 10.0, 9.0];
    let low = [8.0, 10.0, 10.0];
    let close = [10.0, 10.0, 9.0];
    let volume = [100.0, 500.0, 700.0];
    let mut output = [0.0; 3];

    AD(&high, &low, &close, &volume, &mut output).unwrap();

    assert_close(output[0], 100.0);
    assert_close(output[1], 100.0);
    assert_close(output[2], 100.0);
}

#[test]
fn ad_rejects_bad_inputs() {
    let mut output = [0.0; 4];
    assert!(AD(&[1.0, 2.0], &[0.0], &[0.5, 1.5], &[10.0, 20.0], &mut output).is_err());
    assert!(AD(
        &[1.0, Float::NAN],
        &[0.0, 1.0],
        &[0.5, 1.5],
        &[10.0, 20.0],
        &mut output,
    )
    .is_err());

    let mut too_small = [0.0; 1];
    let (high, low, close, volume) = fixture();
    assert!(AD(&high, &low, &close, &volume, &mut too_small).is_err());

    let mut stream = IndicatorConfig::stream(&ADConfig::new()).unwrap();
    assert!(StreamingComputation::<ADConfig>::next(
        &mut stream,
        ADTick {
            high: 1.0,
            low: 0.0,
            close: 0.5,
            volume: Float::INFINITY,
        }
    )
    .is_err());
}

#[test]
fn obv_function_writes_compact_outputs() {
    let (_, _, close, volume) = fixture();
    let mut output = [0.0; 4];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(0, 4));
    for (actual, expected) in output.into_iter().zip([100.0, 300.0, 250.0, 550.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn obv_config_implements_indicator_compute() {
    let (_, _, close, volume) = fixture();
    let config = OBVConfig::new();
    let mut output = [0.0; 4];

    let range = IndicatorConfig::compute_into(
        &config,
        OBVInput {
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(IndicatorConfig::lookback(&config), 0);
    assert_eq!(range, OutputRange::new(0, 4));
    assert_close(output[0], 100.0);
}

#[test]
fn obv_streaming_matches_batch_and_reset() {
    let (_, _, close, volume) = fixture();
    let mut batch = [0.0; 4];
    let range = OBV(&close, &volume, &mut batch).unwrap();
    let config = OBVConfig::new();
    let mut stream = IndicatorConfig::stream(&config).unwrap();

    for idx in 0..close.len() {
        let streamed = StreamingComputation::<OBVConfig>::next(
            &mut stream,
            OBVTick {
                close: close[idx],
                volume: volume[idx],
            },
        )
        .unwrap()
        .unwrap();
        assert_close(streamed, batch[idx - range.beg_idx]);
    }

    StreamingComputation::<OBVConfig>::reset(&mut stream);
    assert_close(
        StreamingComputation::<OBVConfig>::next(
            &mut stream,
            OBVTick {
                close: close[0],
                volume: volume[0],
            },
        )
        .unwrap()
        .unwrap(),
        batch[0],
    );
}

#[test]
fn obv_flat_close_leaves_value_unchanged() {
    let close = [10.0, 10.0, 9.0];
    let volume = [100.0, 50.0, 25.0];
    let mut output = [0.0; 3];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(0, 3));
    assert_close(output[0], 100.0);
    assert_close(output[1], 100.0);
    assert_close(output[2], 75.0);
}

#[test]
fn obv_rejects_bad_inputs() {
    let mut output = [0.0; 4];
    assert!(OBV(&[1.0, 2.0], &[10.0], &mut output).is_err());
    assert!(OBV(&[1.0, Float::NAN], &[10.0, 20.0], &mut output).is_err());
    assert_eq!(
        OBV(&[1.0], &[10.0], &mut output).unwrap(),
        OutputRange::new(0, 1)
    );
    assert_close(output[0], 10.0);

    let mut too_small = [0.0; 1];
    let (_, _, close, volume) = fixture();
    assert!(OBV(&close, &volume, &mut too_small).is_err());

    let mut stream = IndicatorConfig::stream(&OBVConfig::new()).unwrap();
    assert!(StreamingComputation::<OBVConfig>::next(
        &mut stream,
        OBVTick {
            close: 1.0,
            volume: Float::INFINITY,
        }
    )
    .is_err());
}

fn adosc_fixture() -> ([Float; 5], [Float; 5], [Float; 5], [Float; 5]) {
    ([2.0; 5], [0.0; 5], [2.0; 5], [1.0, 2.0, 3.0, 4.0, 5.0])
}

#[test]
fn adosc_function_writes_compact_outputs() {
    let (high, low, close, volume) = adosc_fixture();
    let mut output = [0.0; 5];

    let range = ADOSC(&high, &low, &close, &volume, 2, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 4.0 / 3.0);
    assert_close(output[1], 14.0 / 9.0);
    assert_close(output[2], 103.0 / 54.0);
}

#[test]
fn adosc_config_implements_indicator_compute() {
    let (high, low, close, volume) = adosc_fixture();
    let config = ADOSCConfig::new(2, 3).unwrap();
    let mut output = [0.0; 5];

    let range = IndicatorConfig::compute_into(
        &config,
        ADOSCInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(config.fastperiod(), 2);
    assert_eq!(config.slowperiod(), 3);
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 4.0 / 3.0);
}

#[test]
fn adosc_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = adosc_fixture();
    let mut batch = [0.0; 5];
    let range = ADOSC(&high, &low, &close, &volume, 2, 3, &mut batch).unwrap();
    let config = ADOSCConfig::new(2, 3).unwrap();
    let mut stream = IndicatorConfig::stream(&config).unwrap();

    for idx in 0..high.len() {
        let streamed = StreamingComputation::<ADOSCConfig>::next(
            &mut stream,
            ADOSCTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            },
        )
        .unwrap();
        if idx < range.beg_idx {
            assert!(streamed.is_none());
        } else {
            assert_close(streamed.unwrap(), batch[idx - range.beg_idx]);
        }
    }

    StreamingComputation::<ADOSCConfig>::reset(&mut stream);
    assert!(StreamingComputation::<ADOSCConfig>::next(
        &mut stream,
        ADOSCTick {
            high: high[0],
            low: low[0],
            close: close[0],
            volume: volume[0],
        }
    )
    .unwrap()
    .is_none());
}

#[test]
fn adosc_rejects_invalid_periods_and_inputs() {
    assert!(ADOSCConfig::new(0, 3).is_err());
    assert!(ADOSCConfig::new(2, 0).is_err());
    assert!(ADOSCConfig::new(3, 3).is_err());
    let ordering_error = ADOSCConfig::new(4, 3).unwrap_err().to_string();
    assert!(ordering_error.contains("4 (slowperiod=3)"));
    assert!(ADOSCConfig::new(1, 2).is_ok());

    let (high, low, close, volume) = adosc_fixture();
    let mut output = [0.0; 5];
    assert!(ADOSC(
        &high[..2],
        &low[..2],
        &close[..2],
        &volume[..2],
        2,
        3,
        &mut output,
    )
    .is_err());

    let mut invalid_high = high;
    invalid_high[2] = Float::NAN;
    assert!(ADOSC(&invalid_high, &low, &close, &volume, 2, 3, &mut output,).is_err());

    let mut too_small = [0.0; 1];
    assert!(ADOSC(&high, &low, &close, &volume, 2, 3, &mut too_small,).is_err());

    let mut stream = IndicatorConfig::stream(&ADOSCConfig::new(2, 3).unwrap()).unwrap();
    assert!(StreamingComputation::<ADOSCConfig>::next(
        &mut stream,
        ADOSCTick {
            high: 2.0,
            low: 0.0,
            close: 2.0,
            volume: Float::INFINITY,
        }
    )
    .is_err());
}

#[test]
fn prepared_capacity_precedes_volume_input_alignment() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let mut output = [];
    let capacity_error = TalibError::PreparedCapacityExceeded {
        max_input_len: within.len(),
        actual_input_len: oversized.len(),
    };

    let mut ad = ADConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        ad.compute_into(
            ADInput {
                high: &within,
                low: &within,
                close: &within,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        capacity_error
    );

    let mut adosc = ADOSCConfig::new(1, 2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        adosc
            .compute_into(
                ADOSCInput {
                    high: &within,
                    low: &oversized,
                    close: &within,
                    volume: &within,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );

    let mut obv = OBVConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        obv.compute_into(
            OBVInput {
                close: &within,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        capacity_error
    );
}
