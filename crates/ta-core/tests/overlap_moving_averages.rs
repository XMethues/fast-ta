use ta_core::overlap::{
    DEMA_vec, EMA_vec, MAType, MA_vec, T3_vec, T3_vec_with_default_vfactor,
    T3_with_default_vfactor, TEMA_vec, TRIMA_vec, WMA_vec, DEMA, EMA, MA, T3, T3_DEFAULT_VFACTOR,
    TEMA, TRIMA, WMA,
};
use ta_core::{Float, Indicator, OutputRange, Resettable, StreamingIndicator, TalibError};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-6 as Float,
        "expected {expected}, got {actual}"
    );
}

fn assert_vec_close_with_nans(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "expected NaN at {idx}, got {actual}");
        } else {
            assert_close(*actual, *expected);
        }
    }
}

#[test]
fn ema_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let mut output = [0.0; 5];

    let range = EMA(&real, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 7.0 as Float / 3.0 as Float);
    assert_close(output[1], 31.0 as Float / 6.0 as Float);
    assert_close(output[2], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_vec_returns_padded_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];

    let output = EMA_vec(&real, 3).unwrap();

    assert_eq!(output.len(), real.len());
    assert!(output[0].is_nan());
    assert!(output[1].is_nan());
    assert_close(output[2], 7.0 as Float / 3.0 as Float);
    assert_close(output[3], 31.0 as Float / 6.0 as Float);
    assert_close(output[4], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_struct_implements_indicator_compute() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let ema = EMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&ema, &real, &mut compact).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 7.0 as Float / 3.0 as Float);
    assert_close(compact[2], 127.0 as Float / 12.0 as Float);
}

#[test]
fn ema_rejects_invalid_parameters_and_inputs() {
    assert!(EMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(EMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(EMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(EMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(EMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}

#[test]
fn ema_streaming_next_and_reset_are_safe() {
    let mut ema = EMA::new(3).unwrap();

    assert!(ema.next_checked(1.0).unwrap().is_nan());
    assert!(ema.next_checked(2.0).unwrap().is_nan());
    assert_close(ema.next_checked(4.0).unwrap(), 7.0 as Float / 3.0 as Float);
    assert_close(ema.next_checked(8.0).unwrap(), 31.0 as Float / 6.0 as Float);

    ema.reset();
    assert!(ema.next_checked(10.0).unwrap().is_nan());
    assert!(ema.next(Float::NAN).is_err());
}

#[test]
fn wma_and_trima_functions_write_compact_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let mut output = [0.0; 5];

    let range = WMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 17.0 as Float / 6.0 as Float);
    assert_close(output[1], 17.0 as Float / 3.0 as Float);
    assert_close(output[2], 34.0 as Float / 3.0 as Float);

    let range = TRIMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 9.0 as Float / 4.0 as Float);
    assert_close(output[1], 9.0 as Float / 2.0 as Float);
    assert_close(output[2], 9.0 as Float);

    let range = TRIMA(&real, 4, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(3, 2));
    assert_close(output[0], 7.0 as Float / 2.0 as Float);
    assert_close(output[1], 7.0 as Float);
}

#[test]
fn wma_and_trima_vec_return_padded_outputs() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];

    let wma = WMA_vec(&real, 3).unwrap();
    assert_eq!(wma.len(), real.len());
    assert!(wma[0].is_nan());
    assert!(wma[1].is_nan());
    assert_close(wma[2], 17.0 as Float / 6.0 as Float);

    let trima = TRIMA_vec(&real, 3).unwrap();
    assert_eq!(trima.len(), real.len());
    assert!(trima[0].is_nan());
    assert!(trima[1].is_nan());
    assert_close(trima[2], 9.0 as Float / 4.0 as Float);
}

#[test]
fn wma_and_trima_structs_implement_indicator_compute() {
    let real = [1.0, 2.0, 4.0, 8.0, 16.0];
    let wma = WMA::new(3).unwrap();
    let trima = TRIMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&wma, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 17.0 as Float / 6.0 as Float);

    let range = Indicator::compute(&trima, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 9.0 as Float / 4.0 as Float);
}

#[test]
fn wma_and_trima_streaming_next_and_reset_are_safe() {
    let mut wma = WMA::new(3).unwrap();
    assert!(wma.next_checked(1.0).unwrap().is_nan());
    assert!(wma.next_checked(2.0).unwrap().is_nan());
    assert_close(wma.next_checked(4.0).unwrap(), 17.0 as Float / 6.0 as Float);
    assert_close(wma.next_checked(8.0).unwrap(), 17.0 as Float / 3.0 as Float);
    wma.reset();
    assert!(wma.next_checked(10.0).unwrap().is_nan());

    let mut trima = TRIMA::new(3).unwrap();
    assert!(trima.next_checked(1.0).unwrap().is_nan());
    assert!(trima.next_checked(2.0).unwrap().is_nan());
    assert_close(
        trima.next_checked(4.0).unwrap(),
        9.0 as Float / 4.0 as Float,
    );
    assert_close(
        trima.next_checked(8.0).unwrap(),
        9.0 as Float / 2.0 as Float,
    );
    trima.reset();
    assert!(trima.next(Float::NAN).is_err());
}

#[test]
fn wma_and_trima_reject_invalid_parameters_and_inputs() {
    assert!(WMA::new(0).is_err());
    assert!(TRIMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(WMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(TRIMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(WMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}

#[test]
fn dema_and_tema_functions_write_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut output = [0.0; 7];

    let range = DEMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(4, 3));
    assert_close(output[0], 5.0);
    assert_close(output[1], 6.0);
    assert_close(output[2], 7.0);

    let range = TEMA(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(6, 1));
    assert_close(output[0], 7.0);
}

#[test]
fn dema_and_tema_vec_return_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

    let dema = DEMA_vec(&real, 3).unwrap();
    assert_eq!(dema.len(), real.len());
    assert!(dema[..4].iter().all(|value| value.is_nan()));
    assert_close(dema[4], 5.0);
    assert_close(dema[6], 7.0);

    let tema = TEMA_vec(&real, 3).unwrap();
    assert_eq!(tema.len(), real.len());
    assert!(tema[..6].iter().all(|value| value.is_nan()));
    assert_close(tema[6], 7.0);
}

#[test]
fn dema_and_tema_structs_implement_indicator_compute() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let dema = DEMA::new(3).unwrap();
    let tema = TEMA::new(3).unwrap();
    let mut compact = [0.0; 7];

    let range = Indicator::compute(&dema, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(4, 3));
    assert_close(compact[0], 5.0);

    let range = Indicator::compute(&tema, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(6, 1));
    assert_close(compact[0], 7.0);
}

#[test]
fn dema_and_tema_streaming_next_and_reset_are_safe() {
    let mut dema = DEMA::new(3).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0] {
        assert!(dema.next_checked(value).unwrap().is_nan());
    }
    assert_close(dema.next_checked(5.0).unwrap(), 5.0);
    dema.reset();
    assert!(dema.next_checked(10.0).unwrap().is_nan());

    let mut tema = TEMA::new(3).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert!(tema.next_checked(value).unwrap().is_nan());
    }
    assert_close(tema.next_checked(7.0).unwrap(), 7.0);
    tema.reset();
    assert!(tema.next(Float::NAN).is_err());
}

#[test]
fn dema_and_tema_reject_invalid_parameters_and_inputs() {
    assert!(DEMA::new(0).is_err());
    assert!(TEMA::new(0).is_err());
    assert!(DEMA::new(usize::MAX).is_err());
    assert!(TEMA::new(usize::MAX).is_err());

    let mut output = [0.0; 7];
    assert!(DEMA(&[1.0, 2.0, 3.0], 3, &mut output).is_err());
    assert!(TEMA(&[1.0, Float::NAN, 3.0, 4.0, 5.0, 6.0, 7.0], 3, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(DEMA(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, &mut too_small).is_err());
}

#[test]
fn t3_function_writes_compact_outputs_and_default_matches_explicit() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let mut explicit = [0.0; 4];
    let mut defaulted = [0.0; 4];

    let explicit_range = T3(&real, 1, T3_DEFAULT_VFACTOR, &mut explicit).unwrap();
    let default_range = T3_with_default_vfactor(&real, 1, &mut defaulted).unwrap();

    assert_eq!(explicit_range, OutputRange::new(0, 4));
    assert_eq!(default_range, explicit_range);
    for idx in 0..real.len() {
        assert_close(explicit[idx], real[idx]);
        assert_close(defaulted[idx], explicit[idx]);
    }
}

#[test]
fn t3_vec_returns_padded_outputs_for_recursive_lookback() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let explicit = T3_vec(&real, 2, T3_DEFAULT_VFACTOR).unwrap();
    let defaulted = T3_vec_with_default_vfactor(&real, 2).unwrap();

    assert_eq!(explicit.len(), real.len());
    assert!(explicit[..6].iter().all(|value| value.is_nan()));
    assert!(explicit[6].is_finite());
    assert!(explicit[7].is_finite());
    for idx in 0..real.len() {
        if explicit[idx].is_nan() {
            assert!(defaulted[idx].is_nan());
        } else {
            assert_close(defaulted[idx], explicit[idx]);
        }
    }
}

#[test]
fn t3_struct_implements_indicator_compute_and_streaming() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let t3 = T3::with_default_vfactor(1).unwrap();
    assert_close(t3.vfactor(), T3_DEFAULT_VFACTOR);

    let mut compact = [0.0; 4];
    let range = Indicator::compute(&t3, &real, &mut compact).unwrap();
    assert_eq!(range, OutputRange::new(0, 4));
    for idx in 0..real.len() {
        assert_close(compact[idx], real[idx]);
    }

    let mut streaming = T3::new(2, T3_DEFAULT_VFACTOR).unwrap();
    for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
        assert!(streaming.next_checked(value).unwrap().is_nan());
    }
    assert!(streaming.next_checked(7.0).unwrap().is_finite());
    streaming.reset();
    assert!(streaming.next(Float::NAN).is_err());
}

#[test]
fn t3_rejects_invalid_parameters_and_inputs() {
    assert!(T3::new(0, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3::new(usize::MAX, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3::new(3, -0.1 as Float).is_err());
    assert!(T3::new(3, 1.1 as Float).is_err());
    assert!(T3::new(3, Float::NAN).is_err());

    let mut output = [0.0; 8];
    assert!(T3(&[1.0, Float::NAN, 3.0], 1, T3_DEFAULT_VFACTOR, &mut output).is_err());
    let mut too_small = [0.0; 1];
    assert!(T3(&[1.0, 2.0, 3.0, 4.0], 1, T3_DEFAULT_VFACTOR, &mut too_small).is_err());
}

#[test]
fn ma_dispatches_to_implemented_moving_averages() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::SMA).unwrap(),
        &ta_core::overlap::SMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::EMA).unwrap(),
        &EMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::WMA).unwrap(),
        &WMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::DEMA).unwrap(),
        &DEMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::TEMA).unwrap(),
        &TEMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 3, MAType::TRIMA).unwrap(),
        &TRIMA_vec(&real, 3).unwrap(),
    );
    assert_vec_close_with_nans(
        &MA_vec(&real, 2, MAType::T3).unwrap(),
        &T3_vec_with_default_vfactor(&real, 2).unwrap(),
    );
}

#[test]
fn ma_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut ma_output = [0.0; 5];
    let mut ema_output = [0.0; 5];

    let ma_range = MA(&real, 3, MAType::EMA, &mut ma_output).unwrap();
    let ema_range = EMA(&real, 3, &mut ema_output).unwrap();

    assert_eq!(ma_range, ema_range);
    assert_close(ma_output[0], ema_output[0]);
    assert_close(ma_output[2], ema_output[2]);
}

#[test]
fn ma_struct_streams_selected_average() {
    let mut ma = MA::new(3, MAType::EMA).unwrap();
    assert_eq!(ma.ma_type(), MAType::EMA);
    assert_eq!(ma.period(), 3);

    assert!(ma.next_checked(1.0).unwrap().is_nan());
    assert!(ma.next_checked(2.0).unwrap().is_nan());
    assert_close(ma.next_checked(3.0).unwrap(), 2.0);
    assert_close(ma.next_checked(4.0).unwrap(), 3.0);

    ma.reset();
    assert!(ma.next(Float::NAN).is_err());
}

#[test]
fn ma_rejects_unsupported_kama_and_mama_until_implemented() {
    let real = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];

    let kama_err = MA(&real, 3, MAType::KAMA, &mut output).unwrap_err();
    assert!(matches!(kama_err, TalibError::NotImplemented { .. }));

    let mama_err = MA::new(3, MAType::MAMA).unwrap_err();
    assert!(matches!(mama_err, TalibError::NotImplemented { .. }));
}
