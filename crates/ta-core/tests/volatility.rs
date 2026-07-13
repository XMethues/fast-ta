use ta_core::volatility::{
    ATRInput, ATRTick, ATR_vec, NATRInput, NATRTick, NATR_vec, TRANGEInput, TRANGETick, TRANGE_vec,
    ATR, NATR, TRANGE,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn trange_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];
    let mut output = [0.0; 4];

    let range = TRANGE(&high, &low, &close, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 3.0);
    assert_close(output[1], 1.0);
    assert_close(output[2], 5.0);
}

#[test]
fn trange_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];

    let output = TRANGE_vec(&high, &low, &close).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[0].is_nan());
    assert_close(output[1], 3.0);
    assert_close(output[2], 1.0);
    assert_close(output[3], 5.0);
}

#[test]
fn trange_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 14.0];
    let trange = TRANGE::new().unwrap();
    let mut output = [0.0; 4];

    let range = Indicator::compute(
        &trange,
        TRANGEInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(trange.lookback(), 1);
    assert_eq!(range, OutputRange::new(1, 3));
    assert_close(output[0], 3.0);
}

#[test]
fn trange_streaming_next_and_reset_are_safe() {
    let mut trange = TRANGE::new().unwrap();

    assert!(trange
        .next_checked(TRANGETick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        })
        .unwrap()
        .is_nan());
    assert_close(
        trange
            .next_checked(TRANGETick {
                high: 12.0,
                low: 9.0,
                close: 11.0,
            })
            .unwrap(),
        3.0,
    );
    assert_close(
        trange
            .next_checked(TRANGETick {
                high: 11.0,
                low: 10.0,
                close: 10.0,
            })
            .unwrap(),
        1.0,
    );

    trange.reset();
    assert!(trange
        .next_checked(TRANGETick {
            high: 15.0,
            low: 13.0,
            close: 14.0,
        })
        .unwrap()
        .is_nan());
    assert!(trange
        .next(TRANGETick {
            high: Float::NAN,
            low: 13.0,
            close: 14.0,
        })
        .is_err());
}

#[test]
fn trange_rejects_bad_inputs() {
    let mut output = [0.0; 4];

    assert!(TRANGE(&[1.0, 2.0], &[1.0], &[1.0, 2.0], &mut output).is_err());
    assert!(TRANGE(&[1.0, Float::NAN], &[0.0, 1.0], &[0.5, 1.5], &mut output,).is_err());
    assert!(TRANGE(&[1.0], &[0.0], &[0.5], &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(TRANGE(
        &[1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0],
        &[0.5, 1.5, 2.5],
        &mut too_small,
    )
    .is_err());
}

#[test]
fn atr_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut output = [0.0; 5];

    let range = ATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 3.0);
    assert_close(output[1], 8.0 / 3.0);
}

#[test]
fn atr_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];

    let output = ATR_vec(&high, &low, &close, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..3].iter().all(|value| value.is_nan()));
    assert_close(output[3], 3.0);
    assert_close(output[4], 8.0 / 3.0);
}

#[test]
fn atr_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let atr = ATR::new(3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &atr,
        ATRInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(atr.period(), 3);
    assert_eq!(atr.lookback(), 3);
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 3.0);
}

#[test]
fn atr_streaming_next_and_reset_are_safe() {
    let mut atr = ATR::new(3).unwrap();

    for tick in [
        ATRTick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        },
        ATRTick {
            high: 12.0,
            low: 9.0,
            close: 11.0,
        },
        ATRTick {
            high: 11.0,
            low: 10.0,
            close: 10.0,
        },
    ] {
        assert!(atr.next_checked(tick).unwrap().is_nan());
    }

    assert_close(
        atr.next_checked(ATRTick {
            high: 15.0,
            low: 13.0,
            close: 14.0,
        })
        .unwrap(),
        3.0,
    );
    assert_close(
        atr.next_checked(ATRTick {
            high: 16.0,
            low: 14.0,
            close: 15.0,
        })
        .unwrap(),
        8.0 / 3.0,
    );

    atr.reset();
    assert!(atr
        .next_checked(ATRTick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        })
        .unwrap()
        .is_nan());
    assert!(atr
        .next(ATRTick {
            high: Float::NAN,
            low: 8.0,
            close: 9.0,
        })
        .is_err());
}

#[test]
fn atr_period_one_matches_trange() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut atr_output = [0.0; 5];
    let mut trange_output = [0.0; 5];

    let atr_range = ATR(&high, &low, &close, 1, &mut atr_output).unwrap();
    let trange_range = TRANGE(&high, &low, &close, &mut trange_output).unwrap();

    assert_eq!(atr_range, trange_range);
    for idx in 0..atr_range.nb_element {
        assert_close(atr_output[idx], trange_output[idx]);
    }
}

#[test]
fn atr_rejects_invalid_inputs() {
    assert!(ATR::new(0).is_err());
    assert!(ATR::new(usize::MAX).is_err());

    let mut output = [0.0; 5];
    assert!(ATR(
        &[1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0],
        &[0.5, 1.5, 2.5],
        3,
        &mut output
    )
    .is_err());
    assert!(ATR(
        &[1.0, Float::NAN, 3.0, 4.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.5, 1.5, 2.5, 3.5],
        2,
        &mut output
    )
    .is_err());

    let mut too_small = [0.0; 1];
    assert!(ATR(
        &[10.0, 12.0, 11.0, 15.0, 16.0],
        &[8.0, 9.0, 10.0, 13.0, 14.0],
        &[9.0, 11.0, 10.0, 14.0, 15.0],
        3,
        &mut too_small,
    )
    .is_err());
}

#[test]
fn natr_function_writes_compact_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut output = [0.0; 5];

    let range = NATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], (3.0 / 14.0) * 100.0);
    assert_close(output[1], ((8.0 / 3.0) / 15.0) * 100.0);
}

#[test]
fn natr_vec_returns_padded_outputs() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];

    let output = NATR_vec(&high, &low, &close, 3).unwrap();

    assert_eq!(output.len(), high.len());
    assert!(output[..3].iter().all(|value| value.is_nan()));
    assert_close(output[3], (3.0 / 14.0) * 100.0);
    assert_close(output[4], ((8.0 / 3.0) / 15.0) * 100.0);
}

#[test]
fn natr_struct_implements_indicator_compute() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let natr = NATR::new(3).unwrap();
    let mut output = [0.0; 5];

    let range = Indicator::compute(
        &natr,
        NATRInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut output,
    )
    .unwrap();

    assert_eq!(natr.period(), 3);
    assert_eq!(natr.lookback(), 3);
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], (3.0 / 14.0) * 100.0);
}

#[test]
fn natr_streaming_next_and_reset_are_safe() {
    let mut natr = NATR::new(3).unwrap();

    for tick in [
        NATRTick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        },
        NATRTick {
            high: 12.0,
            low: 9.0,
            close: 11.0,
        },
        NATRTick {
            high: 11.0,
            low: 10.0,
            close: 10.0,
        },
    ] {
        assert!(natr.next_checked(tick).unwrap().is_nan());
    }

    assert_close(
        natr.next_checked(NATRTick {
            high: 15.0,
            low: 13.0,
            close: 14.0,
        })
        .unwrap(),
        (3.0 / 14.0) * 100.0,
    );
    assert_close(
        natr.next_checked(NATRTick {
            high: 16.0,
            low: 14.0,
            close: 15.0,
        })
        .unwrap(),
        ((8.0 / 3.0) / 15.0) * 100.0,
    );

    natr.reset();
    assert!(natr
        .next_checked(NATRTick {
            high: 10.0,
            low: 8.0,
            close: 9.0,
        })
        .unwrap()
        .is_nan());
    assert!(natr
        .next(NATRTick {
            high: Float::NAN,
            low: 8.0,
            close: 9.0,
        })
        .is_err());
}

#[test]
fn natr_zero_close_outputs_zero() {
    let high = [10.0, 12.0, 11.0, 15.0];
    let low = [8.0, 9.0, 10.0, 13.0];
    let close = [9.0, 11.0, 10.0, 0.0];
    let mut output = [1.0; 4];

    let range = NATR(&high, &low, &close, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(3, 1));
    assert_close(output[0], 0.0);
}

#[test]
fn natr_period_one_matches_trange() {
    let high = [10.0, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0, 9.0, 10.0, 13.0, 14.0];
    let close = [9.0, 11.0, 10.0, 14.0, 15.0];
    let mut natr_output = [0.0; 5];
    let mut trange_output = [0.0; 5];

    let natr_range = NATR(&high, &low, &close, 1, &mut natr_output).unwrap();
    let trange_range = TRANGE(&high, &low, &close, &mut trange_output).unwrap();

    assert_eq!(natr_range, trange_range);
    for idx in 0..natr_range.nb_element {
        assert_close(natr_output[idx], trange_output[idx]);
    }
}

#[test]
fn natr_rejects_invalid_inputs() {
    assert!(NATR::new(0).is_err());
    assert!(NATR::new(usize::MAX).is_err());

    let mut output = [0.0; 5];
    assert!(NATR(
        &[1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.0],
        &[0.5, 1.5, 2.5],
        3,
        &mut output
    )
    .is_err());
    assert!(NATR(
        &[1.0, Float::NAN, 3.0, 4.0],
        &[0.0, 1.0, 2.0, 3.0],
        &[0.5, 1.5, 2.5, 3.5],
        2,
        &mut output
    )
    .is_err());

    let mut too_small = [0.0; 1];
    assert!(NATR(
        &[10.0, 12.0, 11.0, 15.0, 16.0],
        &[8.0, 9.0, 10.0, 13.0, 14.0],
        &[9.0, 11.0, 10.0, 14.0, 15.0],
        3,
        &mut too_small,
    )
    .is_err());
}
