use ta_core::math_operators::{
    MAXINDEXConfig, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX, MULT, SUB, SUM,
};
use ta_core::{Float, OutputRange};

#[test]
fn arithmetic_operators_compute_expected_values() {
    let real0 = [8.0 as Float, 6.0 as Float, 4.0 as Float];
    let real1 = [2.0 as Float, 3.0 as Float, 4.0 as Float];
    let mut output = [0.0 as Float; 3];

    assert_eq!(
        ADD(&real0, &real1, &mut output).unwrap(),
        OutputRange::new(0, 3)
    );
    assert_eq!(output, [10.0 as Float, 9.0 as Float, 8.0 as Float]);

    SUB(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [6.0 as Float, 3.0 as Float, 0.0 as Float]);

    MULT(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [16.0 as Float, 18.0 as Float, 16.0 as Float]);

    DIV(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [4.0 as Float, 2.0 as Float, 1.0 as Float]);
}

#[test]
fn rolling_operators_compute_compact_outputs() {
    let real = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let mut output = [0.0 as Float; 4];

    assert_eq!(SUM(&real, 2, &mut output).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&output[..3], &[4.0 as Float, 5.0 as Float, 6.0 as Float]);

    MIN(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[1.0 as Float, 1.0 as Float, 2.0 as Float]);

    MAX(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[3.0 as Float, 4.0 as Float, 4.0 as Float]);
}

#[test]
fn extrema_index_functions_return_absolute_indexes() {
    let real = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let mut min_idx = [0; 4];
    let mut max_idx = [0; 4];

    assert_eq!(
        MININDEX(&real, 2, &mut min_idx).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(&min_idx[..3], &[1, 1, 3]);

    assert_eq!(
        MAXINDEX(&real, 2, &mut max_idx).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(&max_idx[..3], &[0, 2, 2]);

    let mut min = [0.0 as Float; 4];
    let mut max = [0.0 as Float; 4];
    assert_eq!(
        MINMAX(&real, 2, &mut min, &mut max).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(&min[..3], &[1.0 as Float, 1.0 as Float, 2.0 as Float]);
    assert_eq!(&max[..3], &[3.0 as Float, 4.0 as Float, 4.0 as Float]);

    MINMAXINDEX(&real, 2, &mut min_idx, &mut max_idx).unwrap();
    assert_eq!(&min_idx[..3], &[1, 1, 3]);
    assert_eq!(&max_idx[..3], &[0, 2, 2]);
}

#[test]
fn operators_reject_bad_lengths_periods_and_non_finite_inputs() {
    let mut output = [0.0 as Float; 3];
    assert!(ADD(&[1.0 as Float, 2.0 as Float], &[1.0 as Float], &mut output).is_err());
    assert!(SUB(
        &[1.0 as Float, Float::NAN],
        &[1.0 as Float, 2.0 as Float],
        &mut output
    )
    .is_err());
    assert!(SUM(&[1.0 as Float, 2.0 as Float], 0, &mut output).is_err());
    assert!(MAX(&[1.0 as Float], 2, &mut output).is_err());

    let mut min_output = [0.0 as Float; 3];
    let mut max_output = [0.0 as Float; 3];
    assert!(MINMAX(
        &[1.0 as Float, Float::INFINITY],
        2,
        &mut min_output,
        &mut max_output,
    )
    .is_err());

    assert!(MAXINDEXConfig::new(0).is_err());
}
