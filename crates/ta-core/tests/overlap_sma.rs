use ta_core::overlap::{SMA_vec, SMA};
use ta_core::{Float, Indicator, OutputRange, Resettable};

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
fn sma_vec_returns_padded_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];

    let output = SMA_vec(&real, 3).unwrap();

    assert_eq!(output.len(), real.len());
    assert!(output[0].is_nan());
    assert!(output[1].is_nan());
    assert_close(output[2], 2.0);
    assert_close(output[3], 3.0);
    assert_close(output[4], 4.0);
}

#[test]
fn sma_struct_implements_indicator_compute() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let sma = SMA::new(3).unwrap();
    let mut compact = [0.0; 5];

    let range = Indicator::compute(&sma, &real, &mut compact).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(compact[0], 2.0);
    assert_close(compact[2], 4.0);
}

#[test]
fn sma_rejects_invalid_parameters_and_inputs() {
    assert!(SMA::new(0).is_err());

    let mut output = [0.0; 4];
    assert!(SMA(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(SMA(&[1.0, Float::NAN, 3.0], 2, &mut output).is_err());
    assert!(SMA(&[1.0, Float::INFINITY, 3.0], 2, &mut output).is_err());

    let mut too_small = [0.0; 1];
    assert!(SMA(&[1.0, 2.0, 3.0], 2, &mut too_small).is_err());
}

#[test]
fn sma_streaming_next_and_reset_are_safe() {
    let mut sma = SMA::new(3).unwrap();

    assert!(sma.next_checked(1.0).unwrap().is_nan());
    assert!(sma.next_checked(2.0).unwrap().is_nan());
    assert_close(sma.next_checked(3.0).unwrap(), 2.0);
    assert_close(sma.next_checked(4.0).unwrap(), 3.0);

    sma.reset();
    assert!(sma.next_checked(10.0).unwrap().is_nan());
    assert!(sma.next_checked(Float::NAN).is_err());
}
