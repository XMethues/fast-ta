use ta_core::overlap::{
    DEMAConfig, EMAConfig, MAConfig, PeriodMAType, T3Config, T3_with_default_vfactor, TEMAConfig,
    TRIMAConfig, WMAConfig, DEMA, EMA, MA, T3, T3_DEFAULT_VFACTOR, TEMA, TRIMA, WMA,
};
use ta_core::{Float, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-6 as Float,
        "expected {expected}, got {actual}"
    );
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
fn ema_rejects_invalid_parameters_and_inputs() {
    assert!(EMAConfig::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(EMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(EMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(EMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(EMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
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
fn wma_and_trima_reject_invalid_parameters_and_inputs() {
    assert!(WMAConfig::new(0).is_err());
    assert!(TRIMAConfig::new(0).is_err());

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
fn dema_and_tema_reject_invalid_parameters_and_inputs() {
    assert!(DEMAConfig::new(0).is_err());
    assert!(TEMAConfig::new(0).is_err());
    assert!(DEMAConfig::new(usize::MAX).is_err());
    assert!(TEMAConfig::new(usize::MAX).is_err());

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
fn t3_rejects_invalid_parameters_and_inputs() {
    assert!(T3Config::new(0, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3Config::new(usize::MAX, T3_DEFAULT_VFACTOR).is_err());
    assert!(T3Config::new(3, -0.1 as Float).is_err());
    assert!(T3Config::new(3, 1.1 as Float).is_err());
    assert!(T3Config::new(3, Float::NAN).is_err());

    let mut output = [0.0; 8];
    assert!(T3(&[1.0, Float::NAN, 3.0], 1, T3_DEFAULT_VFACTOR, &mut output).is_err());
    let mut too_small = [0.0; 1];
    assert!(T3(&[1.0, 2.0, 3.0, 4.0], 1, T3_DEFAULT_VFACTOR, &mut too_small).is_err());
}

#[test]
fn ma_dispatches_to_implemented_moving_averages() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut dispatched = [0.0; 8];
    let mut direct = [0.0; 8];

    let cases: [PeriodMAType; 6] = [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
    ];
    for matype in cases {
        dispatched.fill(0.0);
        direct.fill(0.0);
        let dispatched_range = MA(&real, 3, matype, &mut dispatched).unwrap();
        let direct_range = match matype {
            PeriodMAType::SMA => ta_core::overlap::SMA(&real, 3, &mut direct).unwrap(),
            PeriodMAType::EMA => EMA(&real, 3, &mut direct).unwrap(),
            PeriodMAType::WMA => WMA(&real, 3, &mut direct).unwrap(),
            PeriodMAType::DEMA => DEMA(&real, 3, &mut direct).unwrap(),
            PeriodMAType::TEMA => TEMA(&real, 3, &mut direct).unwrap(),
            PeriodMAType::TRIMA => TRIMA(&real, 3, &mut direct).unwrap(),
            _ => unreachable!("cases only contain implemented types"),
        };
        assert_eq!(dispatched_range, direct_range);
        assert_eq!(
            &dispatched[..direct_range.nb_element],
            &direct[..direct_range.nb_element]
        );
    }

    dispatched.fill(0.0);
    direct.fill(0.0);
    let t3_dispatched = MA(&real, 2, PeriodMAType::T3, &mut dispatched).unwrap();
    let t3_direct = T3_with_default_vfactor(&real, 2, &mut direct).unwrap();
    assert_eq!(t3_dispatched, t3_direct);
    assert_eq!(
        &dispatched[..t3_direct.nb_element],
        &direct[..t3_direct.nb_element]
    );
}

#[test]
fn ma_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut ma_output = [0.0; 5];
    let mut ema_output = [0.0; 5];

    let ma_range = MA(&real, 3, PeriodMAType::EMA, &mut ma_output).unwrap();
    let ema_range = EMA(&real, 3, &mut ema_output).unwrap();

    assert_eq!(ma_range, ema_range);
    assert_close(ma_output[0], ema_output[0]);
    assert_close(ma_output[2], ema_output[2]);
}

#[test]
fn every_selectable_moving_average_uses_its_period() {
    let cases = [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
        PeriodMAType::T3,
    ];

    for ma_type in cases {
        let short = MAConfig::new(2, ma_type).unwrap();
        let long = MAConfig::new(3, ma_type).unwrap();
        assert!(
            ta_core::IndicatorConfig::lookback(&short) < ta_core::IndicatorConfig::lookback(&long),
            "{ma_type:?} must apply its configured Period"
        );
    }
}
