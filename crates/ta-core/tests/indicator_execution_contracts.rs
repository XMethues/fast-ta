//! Characterization contracts for the pre-migration Indicator execution seam.
//!
//! These tests intentionally describe the current public behavior that issues
//! #3 and later migration slices compare against. Current owned results are
//! legacy Aligned Output rather than target Compact Output, and current
//! execution has no Prepared Batch Runner or input-capacity contract. These
//! tests do not introduce the target APIs that issue #3 will implement.

use ta_core::{
    math_operators::{MINMAXINDEXOutputMut, MINMAXOutputMut, MINMAX, MINMAXINDEX},
    overlap::SMA,
    price_transform::{AVGPRICEInput, AVGPRICE},
    Float, Indicator, OutputRange, Resettable, StreamingIndicator, TalibError,
};

const FLOAT_SENTINEL: Float = -9_876.5 as Float;

fn assert_float_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

fn assert_float_slice_close(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_float_close(actual, expected);
    }
}

fn assert_some_float_close(actual: Option<Float>, expected: Float) {
    assert_float_close(actual.expect("expected a computed value"), expected);
}

#[test]
fn current_owned_sma_is_legacy_aligned_and_matches_caller_compact_output() {
    let input = [
        1.0 as Float,
        2.0 as Float,
        3.0 as Float,
        4.0 as Float,
        5.0 as Float,
    ];
    let indicator = SMA::new(3).unwrap();
    let mut compact = [FLOAT_SENTINEL; 5];

    let range = Indicator::compute(&indicator, &input, &mut compact).unwrap();
    let owned = Indicator::compute_to_vec(&indicator, &input).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_float_slice_close(&compact[..range.nb_element], &[2.0, 3.0, 4.0]);
    assert_eq!(&compact[range.nb_element..], &[FLOAT_SENTINEL; 2]);

    assert_eq!(owned.len(), input.len());
    assert!(owned[..range.beg_idx].iter().all(|value| value.is_nan()));
    assert_float_slice_close(
        &owned[range.beg_idx..range.end_idx()],
        &compact[..range.nb_element],
    );
}

#[test]
fn current_multi_input_avgprice_uses_named_input_view() {
    let open = [10.0 as Float, 20.0 as Float, 30.0 as Float];
    let high = [14.0 as Float, 24.0 as Float, 34.0 as Float];
    let low = [8.0 as Float, 18.0 as Float, 28.0 as Float];
    let close = [12.0 as Float, 22.0 as Float, 32.0 as Float];
    let input = AVGPRICEInput {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
    };
    let indicator = AVGPRICE::new().unwrap();
    let mut compact = [FLOAT_SENTINEL; 3];

    let range = Indicator::compute(&indicator, input, &mut compact).unwrap();
    let owned = Indicator::compute_to_vec(&indicator, input).unwrap();

    assert_eq!(range, OutputRange::new(0, input.open.len()));
    assert_float_slice_close(&compact, &[11.0, 21.0, 31.0]);
    assert_float_slice_close(&owned, &compact);
}

#[test]
fn current_multi_output_minmax_columns_share_one_range() {
    let input = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let indicator = MINMAX::new(2).unwrap();
    let mut min = [FLOAT_SENTINEL; 4];
    let mut max = [FLOAT_SENTINEL; 4];

    let range = Indicator::compute(
        &indicator,
        &input,
        MINMAXOutputMut {
            min: &mut min,
            max: &mut max,
        },
    )
    .unwrap();
    let owned = Indicator::compute_to_vec(&indicator, &input).unwrap();

    assert_eq!(range, OutputRange::new(1, 3));
    assert_float_slice_close(&min[..range.nb_element], &[1.0, 1.0, 2.0]);
    assert_float_slice_close(&max[..range.nb_element], &[3.0, 4.0, 4.0]);
    assert_eq!(min[range.nb_element], FLOAT_SENTINEL);
    assert_eq!(max[range.nb_element], FLOAT_SENTINEL);

    assert_eq!(owned.min.len(), input.len());
    assert_eq!(owned.max.len(), input.len());
    assert!(owned.min[..range.beg_idx]
        .iter()
        .all(|value| value.is_nan()));
    assert!(owned.max[..range.beg_idx]
        .iter()
        .all(|value| value.is_nan()));
    assert_float_slice_close(
        &owned.min[range.beg_idx..range.end_idx()],
        &min[..range.nb_element],
    );
    assert_float_slice_close(
        &owned.max[range.beg_idx..range.end_idx()],
        &max[..range.nb_element],
    );
}

#[test]
fn current_legacy_aligned_index_output_has_ambiguous_zero_unavailable_value() {
    let input = [5.0 as Float, 4.0 as Float, 3.0 as Float, 2.0 as Float];
    let indicator = MINMAXINDEX::new(3).unwrap();
    let mut min_idx = [i32::MIN; 4];
    let mut max_idx = [i32::MIN; 4];

    let range = Indicator::compute(
        &indicator,
        &input,
        MINMAXINDEXOutputMut {
            min_idx: &mut min_idx,
            max_idx: &mut max_idx,
        },
    )
    .unwrap();
    let owned = Indicator::compute_to_vec(&indicator, &input).unwrap();

    assert_eq!(range, OutputRange::new(2, 2));
    assert_eq!(&min_idx[..range.nb_element], &[2_i32, 3_i32]);
    assert_eq!(&max_idx[..range.nb_element], &[0_i32, 1_i32]);
    assert_eq!(owned.min_idx, vec![0_i32, 0_i32, 2_i32, 3_i32]);
    assert_eq!(owned.max_idx, vec![0_i32, 0_i32, 0_i32, 1_i32]);

    // Zero represents both a legacy unavailable position and a valid absolute output index.
    assert_eq!(owned.max_idx[0], owned.max_idx[range.beg_idx]);
}

#[test]
fn current_execution_has_output_capacity_but_no_prepared_input_capacity() {
    let indicator = SMA::new(3).unwrap();
    let short = [1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float];
    let long = [
        1.0 as Float,
        2.0 as Float,
        3.0 as Float,
        4.0 as Float,
        5.0 as Float,
        6.0 as Float,
    ];

    let mut exact = [FLOAT_SENTINEL; 2];
    assert_eq!(
        Indicator::compute(&indicator, &short, &mut exact).unwrap(),
        OutputRange::new(2, 2)
    );
    assert_float_slice_close(&exact, &[2.0, 3.0]);

    let mut oversized = [FLOAT_SENTINEL; 6];
    assert_eq!(
        Indicator::compute(&indicator, &long, &mut oversized).unwrap(),
        OutputRange::new(2, 4)
    );
    assert_float_slice_close(&oversized[..4], &[2.0, 3.0, 4.0, 5.0]);
    assert_eq!(&oversized[4..], &[FLOAT_SENTINEL; 2]);

    let mut too_small = [FLOAT_SENTINEL; 3];
    let error = Indicator::compute(&indicator, &long, &mut too_small).unwrap_err();
    assert!(matches!(error, TalibError::InvalidInput { .. }));
    assert_eq!(too_small, [FLOAT_SENTINEL; 3]);
}

#[test]
fn current_multi_output_capacity_failure_does_not_partially_mutate_columns() {
    let input = [3.0 as Float, 1.0 as Float, 4.0 as Float, 2.0 as Float];
    let indicator = MINMAX::new(2).unwrap();
    let mut sufficient_min = [FLOAT_SENTINEL; 3];
    let mut insufficient_max = [FLOAT_SENTINEL; 2];

    let error = Indicator::compute(
        &indicator,
        &input,
        MINMAXOutputMut {
            min: &mut sufficient_min,
            max: &mut insufficient_max,
        },
    )
    .unwrap_err();

    assert!(matches!(error, TalibError::InvalidInput { .. }));
    assert_eq!(sufficient_min, [FLOAT_SENTINEL; 3]);
    assert_eq!(insufficient_max, [FLOAT_SENTINEL; 2]);
}

#[test]
fn current_streaming_instances_are_operationally_independent() {
    let mut left = SMA::new(3).unwrap();
    let mut right = SMA::new(3).unwrap();

    assert_eq!(StreamingIndicator::next(&mut left, 1.0).unwrap(), None);
    assert_eq!(StreamingIndicator::next(&mut right, 10.0).unwrap(), None);
    assert_eq!(StreamingIndicator::next(&mut left, 2.0).unwrap(), None);
    assert_eq!(StreamingIndicator::next(&mut right, 20.0).unwrap(), None);

    // Batch computation borrows the instance immutably and does not consume its
    // accumulated streaming observations.
    let mut compact = [FLOAT_SENTINEL; 3];
    Indicator::compute(
        &left,
        &[100.0 as Float, 200.0 as Float, 300.0 as Float],
        &mut compact,
    )
    .unwrap();

    assert_some_float_close(StreamingIndicator::next(&mut left, 3.0).unwrap(), 2.0);
    assert_some_float_close(StreamingIndicator::next(&mut right, 30.0).unwrap(), 20.0);
    assert_some_float_close(StreamingIndicator::next(&mut left, 4.0).unwrap(), 3.0);

    Resettable::reset(&mut left);
    assert_some_float_close(StreamingIndicator::next(&mut right, 40.0).unwrap(), 30.0);
    assert_eq!(StreamingIndicator::next(&mut left, 10.0).unwrap(), None);
}
