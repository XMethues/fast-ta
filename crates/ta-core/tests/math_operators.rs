use ta_core::math_operators::{
    ADD_vec, MAXINDEX_vec, MINMAXINDEX_vec, MINMAX_vec, SUM_vec, ADD, DIV, MAX, MAXINDEX, MIN,
    MININDEX, MINMAX, MINMAXINDEX, MULT, SUB, SUM,
};
use ta_core::{Float, Indicator, OutputRange};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

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
fn rolling_operators_compute_compact_and_padded_outputs() {
    let real = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let mut output = [0.0 as Float; 4];

    assert_eq!(SUM(&real, 2, &mut output).unwrap(), OutputRange::new(1, 3));
    assert_eq!(&output[..3], &[4.0 as Float, 5.0 as Float, 6.0 as Float]);

    MIN(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[1.0 as Float, 1.0 as Float, 2.0 as Float]);

    MAX(&real, 2, &mut output).unwrap();
    assert_eq!(&output[..3], &[3.0 as Float, 4.0 as Float, 4.0 as Float]);

    let padded = SUM_vec(&real, 2).unwrap();
    assert!(padded[0].is_nan());
    assert_close(padded[1], 4.0 as Float);
    assert_close(padded[3], 6.0 as Float);
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
fn vec_wrappers_preserve_length_and_padding() {
    let real0 = [1.0 as Float, 2.0 as Float, 3.0 as Float];
    let real1 = [3.0 as Float, 2.0 as Float, 1.0 as Float];

    assert_eq!(
        ADD_vec(&real0, &real1).unwrap(),
        vec![4.0 as Float, 4.0 as Float, 4.0 as Float]
    );

    let minmax = MINMAX_vec(&real0, 2).unwrap();
    assert_eq!(minmax.min.len(), 3);
    assert!(minmax.min[0].is_nan());
    assert_close(minmax.min[1], 1.0 as Float);
    assert_close(minmax.max[2], 3.0 as Float);

    let minmaxindex = MINMAXINDEX_vec(&real0, 2).unwrap();
    assert_eq!(minmaxindex.min_idx, vec![0, 0, 1]);
    assert_eq!(minmaxindex.max_idx, vec![0, 1, 2]);

    assert_eq!(MAXINDEX_vec(&real0, 2).unwrap(), vec![0, 1, 2]);
}

#[test]
fn struct_surfaces_work() {
    let real0 = [1.0 as Float, 2.0 as Float, 3.0 as Float];
    let real1 = [3.0 as Float, 2.0 as Float, 1.0 as Float];
    let mut output = [0.0 as Float; 3];

    let add = ADD::new().unwrap();
    add.compute(&real0, &real1, &mut output).unwrap();
    assert_eq!(output, [4.0 as Float, 4.0 as Float, 4.0 as Float]);

    let sum = SUM::new(2).unwrap();
    let range = Indicator::compute(&sum, &real0, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(1, 2));
    assert_eq!(&output[..2], &[3.0 as Float, 5.0 as Float]);

    let minindex = MININDEX::new(2).unwrap();
    let mut index_output = [0; 3];
    let range = Indicator::compute(&minindex, &real0, &mut index_output).unwrap();
    assert_eq!(range, OutputRange::new(1, 2));
    assert_eq!(&index_output[..2], &[0, 1]);

    let minmax = MINMAX::new(2).unwrap();
    let mut min = [0.0 as Float; 3];
    let mut max = [0.0 as Float; 3];
    minmax.compute(&real0, &mut min, &mut max).unwrap();
    assert_eq!(&min[..2], &[1.0 as Float, 2.0 as Float]);
    assert_eq!(&max[..2], &[2.0 as Float, 3.0 as Float]);
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

    assert!(MAXINDEX::new(0).is_err());
}
