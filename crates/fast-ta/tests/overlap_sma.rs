use fast_ta::overlap::{SMAConfig, SMA};
use fast_ta::{Float, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn sma_function_writes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0; 5];

    let range = SMA(&real, 3, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 2.0);
    assert_close(output[1], 3.0);
    assert_close(output[2], 4.0);
}

#[test]
fn sma_rejects_invalid_parameters_and_inputs() {
    assert!(SMAConfig::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(SMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(SMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(SMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(SMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}
