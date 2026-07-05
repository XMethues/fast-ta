use ta_core::price_transform::{
    AVGDEV_vec, AVGPRICE_vec, MEDPRICE_vec, TYPPRICE_vec, WCLPRICE_vec, AVGDEV, AVGPRICE, MEDPRICE,
    TYPPRICE, WCLPRICE,
};
use ta_core::{Float, Indicator, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn avgprice_medprice_typprice_wclprice_compute_expected_values() {
    let open = [1.0, 2.0, 3.0];
    let high = [2.0, 3.0, 4.0];
    let low = [0.0, 1.0, 2.0];
    let close = [1.5, 2.5, 3.5];
    let mut output = [0.0; 3];

    assert_eq!(
        AVGPRICE(&open, &high, &low, &close, &mut output).unwrap(),
        OutputRange::new(0, 3)
    );
    assert_close(output[0], 1.125);
    assert_close(output[2], 3.125);

    MEDPRICE(&high, &low, &mut output).unwrap();
    assert_close(output[0], 1.0);
    assert_close(output[2], 3.0);

    TYPPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.1666666666666667);
    assert_close(output[2], 3.1666666666666665);

    WCLPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.25);
    assert_close(output[2], 3.25);
}

#[test]
fn price_transform_vec_wrappers_preserve_length() {
    let open = [1.0, 2.0];
    let high = [2.0, 3.0];
    let low = [0.0, 1.0];
    let close = [1.5, 2.5];

    assert_eq!(AVGPRICE_vec(&open, &high, &low, &close).unwrap().len(), 2);
    assert_eq!(MEDPRICE_vec(&high, &low).unwrap().len(), 2);
    assert_eq!(TYPPRICE_vec(&high, &low, &close).unwrap().len(), 2);
    assert_eq!(WCLPRICE_vec(&high, &low, &close).unwrap().len(), 2);
}

#[test]
fn avgdev_computes_compact_and_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0; 5];

    let range = AVGDEV(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 2.0 / 3.0);
    assert_close(output[2], 2.0 / 3.0);

    let padded = AVGDEV_vec(&real, 3).unwrap();
    assert!(padded[0].is_nan());
    assert!(padded[1].is_nan());
    assert_close(padded[2], 2.0 / 3.0);
}

#[test]
fn price_transform_struct_surfaces_work() {
    let high = [2.0, 3.0, 4.0];
    let low = [0.0, 1.0, 2.0];
    let close = [1.5, 2.5, 3.5];
    let mut output = [0.0; 3];

    let typprice = TYPPRICE::new().unwrap();
    typprice.compute(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.1666666666666667);

    let avgdev = AVGDEV::new(3).unwrap();
    let range = Indicator::compute(&avgdev, &[1.0, 2.0, 3.0], &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 1));
}

#[test]
fn price_transform_rejects_bad_lengths_and_non_finite_inputs() {
    let mut output = [0.0; 3];
    assert!(MEDPRICE(&[1.0, 2.0], &[1.0], &mut output).is_err());
    assert!(TYPPRICE(&[1.0, Float::NAN], &[1.0, 2.0], &[1.0, 2.0], &mut output).is_err());
    assert!(AVGDEV(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(AVGDEV::new(0).is_err());
}
