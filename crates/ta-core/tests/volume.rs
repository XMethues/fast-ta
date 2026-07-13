use ta_core::volume::{
    ADInput, ADOSCInput, ADOSCTick, ADOSC_vec, ADTick, AD_vec, OBVInput, OBVTick, OBV_vec, AD,
    ADOSC, OBV,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator};

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
fn ad_vec_returns_full_length_outputs() {
    let (high, low, close, volume) = fixture();
    let output = AD_vec(&high, &low, &close, &volume).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output.iter().all(|value| value.is_finite()));
    assert_close(output[0], 100.0);
    assert_close(output[3], 150.0);
}

#[test]
fn ad_struct_implements_indicator_compute() {
    let (high, low, close, volume) = fixture();
    let ad = AD::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &ad,
        ADInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(ad.lookback(), 0);
    assert_eq!(range, OutputRange::new(0, 4));
    assert_close(output[2], 150.0);
}

#[test]
fn ad_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = fixture();
    let mut batch = [0.0; 4];
    AD(&high, &low, &close, &volume, &mut batch).unwrap();

    let mut ad = AD::new().unwrap();
    for idx in 0..high.len() {
        let streamed = ad
            .next(ADTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap()
            .unwrap();
        assert_close(streamed, batch[idx]);
    }

    ad.reset();
    assert_close(
        ad.next_checked(ADTick {
            high: high[0],
            low: low[0],
            close: close[0],
            volume: volume[0],
        })
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

    let mut ad = AD::new().unwrap();
    assert!(ad
        .next(ADTick {
            high: 1.0,
            low: 0.0,
            close: 0.5,
            volume: Float::INFINITY,
        })
        .is_err());
}

#[test]
fn obv_function_writes_compact_outputs() {
    let (_, _, close, volume) = fixture();
    let mut output = [0.0; 4];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 3));
    for (actual, expected) in output[..3].iter().copied().zip([200.0, 150.0, 450.0]) {
        assert_close(actual, expected);
    }
}

#[test]
fn obv_vec_returns_padded_outputs() {
    let (_, _, close, volume) = fixture();
    let output = OBV_vec(&close, &volume).unwrap();

    assert_eq!(output.len(), close.len());
    assert!(output[0].is_nan());
    assert_close(output[1], 200.0);
    assert_close(output[2], 150.0);
    assert_close(output[3], 450.0);
}

#[test]
fn obv_struct_implements_indicator_compute() {
    let (_, _, close, volume) = fixture();
    let obv = OBV::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &obv,
        OBVInput {
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(obv.lookback(), 1);
    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 200.0);
}

#[test]
fn obv_streaming_matches_batch_and_reset() {
    let (_, _, close, volume) = fixture();
    let mut batch = [0.0; 4];
    let range = OBV(&close, &volume, &mut batch).unwrap();
    let mut obv = OBV::new().unwrap();

    assert!(obv
        .next(OBVTick {
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_none());
    for idx in 1..close.len() {
        let streamed = obv
            .next(OBVTick {
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap()
            .unwrap();
        assert_close(streamed, batch[idx - range.beg_idx]);
    }

    obv.reset();
    assert!(obv
        .next_checked(OBVTick {
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_nan());
}

#[test]
fn obv_flat_close_leaves_value_unchanged() {
    let close = [10.0, 10.0, 9.0];
    let volume = [100.0, 50.0, 25.0];
    let mut output = [0.0; 3];

    let range = OBV(&close, &volume, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 2));
    assert_close(output[0], 0.0);
    assert_close(output[1], -25.0);
}

#[test]
fn obv_rejects_bad_inputs() {
    let mut output = [0.0; 4];
    assert!(OBV(&[1.0, 2.0], &[10.0], &mut output).is_err());
    assert!(OBV(&[1.0, Float::NAN], &[10.0, 20.0], &mut output).is_err());
    assert!(OBV(&[1.0], &[10.0], &mut output).is_err());

    let mut too_small = [0.0; 1];
    let (_, _, close, volume) = fixture();
    assert!(OBV(&close, &volume, &mut too_small).is_err());

    let mut obv = OBV::new().unwrap();
    assert!(obv
        .next(OBVTick {
            close: 1.0,
            volume: Float::INFINITY,
        })
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
fn adosc_vec_returns_padded_outputs() {
    let (high, low, close, volume) = adosc_fixture();
    let output = ADOSC_vec(&high, &low, &close, &volume, 2, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..2].iter().all(|value| value.is_nan()));
    assert_close(output[2], 4.0 / 3.0);
    assert_close(output[4], 103.0 / 54.0);
}

#[test]
fn adosc_struct_implements_indicator_compute() {
    let (high, low, close, volume) = adosc_fixture();
    let adosc = ADOSC::new(2, 3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &adosc,
        ADOSCInput {
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(adosc.fastperiod(), 2);
    assert_eq!(adosc.slowperiod(), 3);
    assert_eq!(adosc.lookback(), 2);
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 4.0 / 3.0);
}

#[test]
fn adosc_streaming_matches_batch_and_reset() {
    let (high, low, close, volume) = adosc_fixture();
    let mut batch = [0.0; 5];
    let range = ADOSC(&high, &low, &close, &volume, 2, 3, &mut batch).unwrap();
    let mut adosc = ADOSC::new(2, 3).unwrap();

    for idx in 0..high.len() {
        let streamed = adosc
            .next_checked(ADOSCTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
                volume: volume[idx],
            })
            .unwrap();
        if idx < range.beg_idx {
            assert!(streamed.is_nan());
        } else {
            assert_close(streamed, batch[idx - range.beg_idx]);
        }
    }

    adosc.reset();
    assert!(adosc
        .next_checked(ADOSCTick {
            high: high[0],
            low: low[0],
            close: close[0],
            volume: volume[0],
        })
        .unwrap()
        .is_nan());
}

#[test]
fn adosc_rejects_invalid_periods_and_inputs() {
    assert!(ADOSC::new(0, 3).is_err());
    assert!(ADOSC::new(2, 0).is_err());
    assert!(ADOSC::new(3, 3).is_err());
    let ordering_error = ADOSC::new(4, 3).unwrap_err().to_string();
    assert!(ordering_error.contains("4 (slowperiod=3)"));
    assert!(ADOSC::new(1, 2).is_ok());

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

    let mut adosc = ADOSC::new(2, 3).unwrap();
    assert!(adosc
        .next(ADOSCTick {
            high: 2.0,
            low: 0.0,
            close: 2.0,
            volume: Float::INFINITY,
        })
        .is_err());
}
