use ta_core::math_transform::{
    ACOS, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
};
use ta_core::{Float, IndicatorConfig, OutputRange};

type TransformFn = fn(&[Float], &mut [Float]) -> ta_core::Result<OutputRange>;

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn math_transform_functions_compute_expected_values() {
    let real = [0.5 as Float];
    let mut output = [0.0 as Float; 1];

    assert_eq!(SIN(&real, &mut output).unwrap(), OutputRange::new(0, 1));
    assert_close(output[0], (0.5 as Float).sin());

    COS(&real, &mut output).unwrap();
    assert_close(output[0], (0.5 as Float).cos());

    ACOS(&real, &mut output).unwrap();
    assert_close(output[0], (0.5 as Float).acos());

    SQRT(&[4.0 as Float], &mut output).unwrap();
    assert_close(output[0], 2.0 as Float);

    LN(&[(2.0 as Float).exp()], &mut output).unwrap();
    assert_close(output[0], 2.0 as Float);
}

#[test]
fn all_math_transform_functions_are_exported() {
    let real = [0.5 as Float];
    let mut output = [0.0 as Float; 1];

    let funcs: [TransformFn; 13] = [
        ACOS, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
    ];

    for func in funcs {
        assert_eq!(func(&real, &mut output).unwrap(), OutputRange::new(0, 1));
    }
}

#[test]
fn math_transform_configs_compute_expected_values() {
    let sqrt_config = ta_core::math_transform::SQRTConfig::new();
    let owned = IndicatorConfig::compute(&sqrt_config, &[4.0 as Float, 9.0 as Float]).unwrap();
    assert_eq!(owned.range(), OutputRange::new(0, 2));
    assert_close(owned.values()[0], 2.0 as Float);
    assert_close(owned.values()[1], 3.0 as Float);
}

#[test]
fn math_transform_rejects_non_finite_inputs() {
    let mut output = [0.0 as Float; 1];

    assert!(SIN(&[Float::NAN], &mut output).is_err());
    assert!(COS(&[Float::INFINITY], &mut output).is_err());
}
