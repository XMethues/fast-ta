//! Public contracts for the Indicator execution seam.
//!
//! Historical tests keep the legacy `Indicator`, `StreamingIndicator`, and
//! `Resettable` behavior fixed. Migration tests exercise each Indicator
//! Configuration through owned Compact Output, caller-owned output, Prepared
//! Batch Runners, and independent Streaming Computation.

use ta_core::{
    math_operators::{
        MAXConfig, MAXINDEXConfig, MAXINDEXStream, MAXStream, MINConfig, MININDEXConfig,
        MININDEXStream, MINMAXConfig, MINMAXINDEXConfig, MINMAXINDEXOutputMut, MINMAXINDEXStream,
        MINMAXINDEXStreamValue, MINMAXINDEXValuesMut, MINMAXOutputMut, MINMAXStream,
        MINMAXValuesMut, MINStream, MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX,
    },
    overlap::{
        SMAConfig, SMAStream, TRIMAConfig, TRIMAStream, WMAConfig, WMAStream, SMA, TRIMA, WMA,
    },
    price_transform::{AVGPRICEInput, AVGPRICE},
    Float, Indicator, IndicatorConfig, OutputRange, PreparedBatchRunner, Resettable,
    StreamingComputation, StreamingIndicator, TalibError,
};

const FLOAT_SENTINEL: Float = -9_876.5 as Float;

fn assert_float_close(actual: Float, expected: Float) {
    #[cfg(feature = "f32")]
    const TOLERANCE: Float = 1e-5;
    #[cfg(not(feature = "f32"))]
    const TOLERANCE: Float = 1e-10;

    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

fn assert_float_slice_close(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_float_close(actual, expected);
    }
}

fn compute_with_prepared_runner<'a, C: IndicatorConfig + 'a>(
    config: &C,
    max_input_len: usize,
    input: C::Input<'a>,
    output: C::OutputMut<'a>,
) -> ta_core::Result<OutputRange> {
    let mut runner: C::BatchRunner = IndicatorConfig::prepare_batch(config, max_input_len)?;
    PreparedBatchRunner::<C>::compute_into(&mut runner, input, output)
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

fn assert_sma_owned_compact_shape(
    input: &[Float],
    expected_range: OutputRange,
    expected_values: &[Float],
) {
    let config = SMAConfig::new(3).unwrap();

    let output = IndicatorConfig::compute(&config, input).unwrap();

    assert_eq!(output.source_len(), input.len());
    assert_eq!(output.range(), expected_range);
    assert_float_slice_close(output.values(), expected_values);

    // ADR-0001 makes exact owned-output allocation part of the public
    // performance contract, so capacity is intentionally asserted here.
    let payload = output.into_values();
    assert_eq!(payload.len(), expected_values.len());
    assert_eq!(payload.capacity(), expected_values.len());
}

#[test]
fn sma_config_owned_compact_count_zero_has_exact_payload_and_metadata() {
    assert_sma_owned_compact_shape(&[], OutputRange::empty(), &[]);
}

#[test]
fn sma_config_owned_compact_count_one_has_exact_payload_and_metadata() {
    assert_sma_owned_compact_shape(&[1.0 as Float, 2.0, 3.0], OutputRange::new(2, 1), &[2.0]);
}

#[test]
fn sma_config_owned_compact_count_two_has_exact_payload_and_metadata() {
    assert_sma_owned_compact_shape(
        &[1.0 as Float, 2.0, 3.0, 4.0],
        OutputRange::new(2, 2),
        &[2.0, 3.0],
    );
}

#[test]
fn sma_config_owned_compact_count_three_has_exact_payload_and_metadata() {
    assert_sma_owned_compact_shape(
        &[1.0 as Float, 2.0, 3.0, 4.0, 5.0],
        OutputRange::new(2, 3),
        &[2.0, 3.0, 4.0],
    );
}

#[test]
fn sma_config_owned_output_is_compact_and_range_bearing() {
    let input = [
        1.0 as Float,
        2.0 as Float,
        3.0 as Float,
        4.0 as Float,
        5.0 as Float,
    ];
    let config = SMAConfig::new(3).unwrap();

    let output = IndicatorConfig::compute(&config, &input).unwrap();

    assert_eq!(output.source_len(), input.len());
    assert_eq!(output.range(), OutputRange::new(2, 3));
    assert_float_slice_close(output.values(), &[2.0, 3.0, 4.0]);
    assert_float_slice_close(&output.into_values(), &[2.0, 3.0, 4.0]);
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert_eq!(config.period(), 3);
}

#[test]
fn sma_config_compute_into_matches_owned_and_leaves_tail_untouched() {
    let input = [
        1.0 as Float,
        2.0 as Float,
        3.0 as Float,
        4.0 as Float,
        5.0 as Float,
    ];
    let config = SMAConfig::new(3).unwrap();
    let owned = IndicatorConfig::compute(&config, &input).unwrap();
    let mut output = [FLOAT_SENTINEL; 5];

    let range = IndicatorConfig::compute_into(&config, &input, &mut output).unwrap();

    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..range.nb_element], owned.values());
    assert_eq!(&output[range.nb_element..], &[FLOAT_SENTINEL; 2]);
}

#[test]
fn sma_config_validation_never_mutates_caller_output() {
    let config = SMAConfig::new(3).unwrap();
    let mut too_small = [FLOAT_SENTINEL; 2];
    let capacity_error =
        IndicatorConfig::compute_into(&config, &[1.0 as Float, 2.0, 3.0, 4.0, 5.0], &mut too_small)
            .unwrap_err();
    assert!(matches!(capacity_error, TalibError::InvalidInput { .. }));
    assert_eq!(too_small, [FLOAT_SENTINEL; 2]);

    let mut nonfinite_output = [FLOAT_SENTINEL; 3];
    let nonfinite_error = IndicatorConfig::compute_into(
        &config,
        &[1.0 as Float, Float::NAN, 3.0, 4.0, 5.0],
        &mut nonfinite_output,
    )
    .unwrap_err();
    assert!(matches!(nonfinite_error, TalibError::InvalidInput { .. }));
    assert_eq!(nonfinite_output, [FLOAT_SENTINEL; 3]);
}

#[test]
fn generic_config_ties_prepared_runner_to_config_views() {
    let config = SMAConfig::new(3).unwrap();
    let input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let mut output = [FLOAT_SENTINEL; 3];

    let range = compute_with_prepared_runner(&config, input.len(), &input, &mut output).unwrap();

    assert_eq!(range, OutputRange::new(2, 3));
    assert_float_slice_close(&output, &[2.0, 3.0, 4.0]);
}

#[test]
fn prepared_sma_runner_supports_exact_capacity_and_reuse() {
    let config = SMAConfig::new(3).unwrap();
    let mut runner = IndicatorConfig::prepare_batch(&config, 5).unwrap();
    let mut output = [FLOAT_SENTINEL; 3];

    assert_eq!(PreparedBatchRunner::<SMAConfig>::max_input_len(&runner), 5);
    let first = PreparedBatchRunner::<SMAConfig>::compute_into(
        &mut runner,
        &[1.0 as Float, 2.0, 3.0, 4.0, 5.0],
        &mut output,
    )
    .unwrap();
    assert_eq!(first, OutputRange::new(2, 3));
    assert_float_slice_close(&output, &[2.0, 3.0, 4.0]);

    output.fill(FLOAT_SENTINEL);
    let repeated = PreparedBatchRunner::<SMAConfig>::compute_into(
        &mut runner,
        &[5.0 as Float, 4.0, 3.0, 2.0, 1.0],
        &mut output,
    )
    .unwrap();
    assert_eq!(repeated, first);
    assert_float_slice_close(&output, &[4.0, 3.0, 2.0]);
}

#[test]
fn prepared_sma_runner_rejects_oversize_before_other_validation_or_mutation() {
    let config = SMAConfig::new(3).unwrap();
    let mut runner = IndicatorConfig::prepare_batch(&config, 4).unwrap();
    let mut output = [FLOAT_SENTINEL; 1];

    let error = PreparedBatchRunner::<SMAConfig>::compute_into(
        &mut runner,
        &[1.0 as Float, Float::NAN, 3.0, 4.0, 5.0],
        &mut output,
    )
    .unwrap_err();

    assert_eq!(
        error,
        TalibError::PreparedCapacityExceeded {
            max_input_len: 4,
            actual_input_len: 5,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 1]);
}

#[test]
fn prepared_sma_runners_are_independent_per_worker() {
    let config = SMAConfig::new(2).unwrap();
    let mut left = IndicatorConfig::prepare_batch(&config, 4).unwrap();
    let mut right = IndicatorConfig::prepare_batch(&config, 4).unwrap();
    let mut left_output = [FLOAT_SENTINEL; 3];
    let mut right_output = [FLOAT_SENTINEL; 3];

    PreparedBatchRunner::<SMAConfig>::compute_into(
        &mut left,
        &[1.0 as Float, 3.0, 5.0, 7.0],
        &mut left_output,
    )
    .unwrap();
    PreparedBatchRunner::<SMAConfig>::compute_into(
        &mut right,
        &[10.0 as Float, 20.0, 30.0, 40.0],
        &mut right_output,
    )
    .unwrap();

    assert_float_slice_close(&left_output, &[2.0, 4.0, 6.0]);
    assert_float_slice_close(&right_output, &[15.0, 25.0, 35.0]);
}

#[test]
fn sma_config_streams_are_independent_warm_up_and_reset() {
    let config = SMAConfig::new(3).unwrap();
    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();

    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut left, 1.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut right, 10.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut left, 2.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut right, 20.0).unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<SMAConfig>::next(&mut left, 3.0).unwrap(),
        2.0,
    );
    assert_some_float_close(
        StreamingComputation::<SMAConfig>::next(&mut right, 30.0).unwrap(),
        20.0,
    );

    StreamingComputation::<SMAConfig>::reset(&mut left);
    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut left, 7.0).unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<SMAConfig>::next(&mut right, 40.0).unwrap(),
        30.0,
    );
}

#[test]
fn rejected_sma_stream_tick_does_not_change_state() {
    let config = SMAConfig::new(3).unwrap();
    let mut stream = IndicatorConfig::stream(&config).unwrap();

    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut stream, 1.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<SMAConfig>::next(&mut stream, 2.0).unwrap(),
        None
    );
    assert!(StreamingComputation::<SMAConfig>::next(&mut stream, Float::NAN).is_err());
    assert_some_float_close(
        StreamingComputation::<SMAConfig>::next(&mut stream, 3.0).unwrap(),
        2.0,
    );
}

#[test]
fn sma_config_batch_and_stream_outputs_match() {
    let input = [
        1.0 as Float,
        2.0 as Float,
        4.0 as Float,
        8.0 as Float,
        16.0 as Float,
    ];
    let config = SMAConfig::new(3).unwrap();
    let batch = IndicatorConfig::compute(&config, &input).unwrap();
    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let streamed = input
        .iter()
        .filter_map(|&tick| StreamingComputation::<SMAConfig>::next(&mut stream, tick).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(batch.range(), OutputRange::new(2, streamed.len()));
    assert_float_slice_close(batch.values(), &streamed);
}

#[test]
fn sma_config_preserves_invalid_empty_short_and_nonfinite_semantics() {
    assert!(matches!(
        SMAConfig::new(0).unwrap_err(),
        TalibError::InvalidPeriod { period: 0, .. }
    ));

    let config = SMAConfig::new(3).unwrap();
    let empty = IndicatorConfig::compute(&config, &[]).unwrap();
    assert_eq!(empty.source_len(), 0);
    assert_eq!(empty.range(), OutputRange::empty());
    assert!(empty.values().is_empty());

    let mut untouched = [FLOAT_SENTINEL; 1];
    assert!(matches!(
        IndicatorConfig::compute_into(&config, &[1.0 as Float, 2.0], &mut untouched),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));
    assert_eq!(untouched, [FLOAT_SENTINEL; 1]);

    assert!(matches!(
        IndicatorConfig::compute(&config, &[1.0 as Float, Float::INFINITY, 3.0]),
        Err(TalibError::InvalidInput { .. })
    ));
}

#[test]
fn legacy_sma_from_data_clone_reset_and_replay_remain_compatible() {
    let mut seeded = SMA::from_data(3, &[1.0 as Float, 2.0]).unwrap();
    let mut cloned = seeded.clone();

    assert_float_close(seeded.next_checked(3.0).unwrap(), 2.0);
    assert_float_close(cloned.next_checked(3.0).unwrap(), 2.0);
    assert_float_close(seeded.next_checked(4.0).unwrap(), 3.0);
    assert_float_close(cloned.next_checked(4.0).unwrap(), 3.0);

    Resettable::reset(&mut seeded);
    assert!(seeded.next_checked(1.0).unwrap().is_nan());
    assert!(seeded.next_checked(2.0).unwrap().is_nan());
    assert_float_close(seeded.next_checked(3.0).unwrap(), 2.0);

    assert!(matches!(
        SMA::from_data(0, &[Float::NAN]),
        Err(TalibError::InvalidInput { .. })
    ));
}

#[test]
fn wma_and_trima_configs_cover_owned_caller_owned_and_prepared_execution() {
    let input = [1.0 as Float, 2.0, 4.0, 8.0, 16.0];
    let wma_config = WMAConfig::new(3).unwrap();
    let trima_config = TRIMAConfig::new(3).unwrap();
    assert_eq!(wma_config.period(), 3);
    assert_eq!(trima_config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&wma_config), 2);
    assert_eq!(IndicatorConfig::lookback(&trima_config), 2);
    assert_eq!(
        core::mem::size_of::<WMAConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<TRIMAConfig>(),
        core::mem::size_of::<usize>()
    );

    let wma_owned = IndicatorConfig::compute(&wma_config, &input).unwrap();
    let trima_owned = IndicatorConfig::compute(&trima_config, &input).unwrap();
    assert_eq!(wma_owned.source_len(), input.len());
    assert_eq!(trima_owned.source_len(), input.len());
    assert_eq!(wma_owned.range(), OutputRange::new(2, 3));
    assert_eq!(trima_owned.range(), wma_owned.range());
    assert_float_slice_close(wma_owned.values(), &[17.0 / 6.0, 34.0 / 6.0, 68.0 / 6.0]);
    assert_float_slice_close(trima_owned.values(), &[9.0 / 4.0, 18.0 / 4.0, 36.0 / 4.0]);

    let mut wma_output = [FLOAT_SENTINEL; 4];
    let mut trima_output = [FLOAT_SENTINEL; 4];
    let wma_range = IndicatorConfig::compute_into(&wma_config, &input, &mut wma_output).unwrap();
    let trima_range =
        IndicatorConfig::compute_into(&trima_config, &input, &mut trima_output).unwrap();
    assert_eq!(wma_range, wma_owned.range());
    assert_eq!(trima_range, trima_owned.range());
    assert_float_slice_close(&wma_output[..3], wma_owned.values());
    assert_float_slice_close(&trima_output[..3], trima_owned.values());
    assert_eq!(wma_output[3], FLOAT_SENTINEL);
    assert_eq!(trima_output[3], FLOAT_SENTINEL);

    let mut wma_runner = IndicatorConfig::prepare_batch(&wma_config, input.len()).unwrap();
    let mut trima_runner = IndicatorConfig::prepare_batch(&trima_config, input.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<WMAConfig>::max_input_len(&wma_runner),
        input.len()
    );
    assert_eq!(
        PreparedBatchRunner::<TRIMAConfig>::max_input_len(&trima_runner),
        input.len()
    );
    wma_output.fill(FLOAT_SENTINEL);
    trima_output.fill(FLOAT_SENTINEL);
    PreparedBatchRunner::<WMAConfig>::compute_into(&mut wma_runner, &input, &mut wma_output)
        .unwrap();
    PreparedBatchRunner::<TRIMAConfig>::compute_into(&mut trima_runner, &input, &mut trima_output)
        .unwrap();
    assert_float_slice_close(&wma_output[..3], wma_owned.values());
    assert_float_slice_close(&trima_output[..3], trima_owned.values());
    assert_eq!(wma_output[3], FLOAT_SENTINEL);
    assert_eq!(trima_output[3], FLOAT_SENTINEL);

    let alternate_input = [16.0 as Float, 8.0, 4.0, 2.0, 1.0];
    let alternate_wma = IndicatorConfig::compute(&wma_config, &alternate_input).unwrap();
    let alternate_trima = IndicatorConfig::compute(&trima_config, &alternate_input).unwrap();
    let mut second_wma_runner =
        IndicatorConfig::prepare_batch(&wma_config, alternate_input.len()).unwrap();
    let mut second_trima_runner =
        IndicatorConfig::prepare_batch(&trima_config, alternate_input.len()).unwrap();
    let mut alternate_wma_output = [FLOAT_SENTINEL; 3];
    let mut alternate_trima_output = [FLOAT_SENTINEL; 3];
    PreparedBatchRunner::<WMAConfig>::compute_into(
        &mut second_wma_runner,
        &alternate_input,
        &mut alternate_wma_output,
    )
    .unwrap();
    PreparedBatchRunner::<TRIMAConfig>::compute_into(
        &mut second_trima_runner,
        &alternate_input,
        &mut alternate_trima_output,
    )
    .unwrap();
    assert_float_slice_close(&alternate_wma_output, alternate_wma.values());
    assert_float_slice_close(&alternate_trima_output, alternate_trima.values());
    assert_float_slice_close(&wma_output[..3], wma_owned.values());
    assert_float_slice_close(&trima_output[..3], trima_owned.values());

    let oversized = [Float::NAN; 6];
    assert!(matches!(
        PreparedBatchRunner::<WMAConfig>::compute_into(
            &mut wma_runner,
            &oversized,
            &mut wma_output
        ),
        Err(TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6
        })
    ));
    assert_eq!(wma_output[0], wma_owned.values()[0]);
}

#[test]
fn wma_and_trima_streams_are_independent_and_preserve_reset_batch_parity() {
    let input = [1.0 as Float, 2.0, 4.0, 8.0, 16.0];
    let wma_config = WMAConfig::new(3).unwrap();
    let trima_config = TRIMAConfig::new(3).unwrap();
    let batch_wma = IndicatorConfig::compute(&wma_config, &input).unwrap();
    let batch_trima = IndicatorConfig::compute(&trima_config, &input).unwrap();
    let mut wma_stream = IndicatorConfig::stream(&wma_config).unwrap();
    let mut trima_stream = IndicatorConfig::stream(&trima_config).unwrap();
    let mut independent_wma = IndicatorConfig::stream(&wma_config).unwrap();
    let mut independent_trima = IndicatorConfig::stream(&trima_config).unwrap();
    let mut legacy_wma = WMA::new(3).unwrap();
    let mut legacy_trima = TRIMA::new(3).unwrap();
    let mut streamed_wma = Vec::new();
    let mut streamed_trima = Vec::new();

    for &tick in &input {
        let wma_value = StreamingComputation::<WMAConfig>::next(&mut wma_stream, tick).unwrap();
        let trima_value =
            StreamingComputation::<TRIMAConfig>::next(&mut trima_stream, tick).unwrap();
        assert_eq!(
            StreamingIndicator::next(&mut legacy_wma, tick).unwrap(),
            wma_value
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_trima, tick).unwrap(),
            trima_value
        );
        streamed_wma.extend(wma_value);
        streamed_trima.extend(trima_value);
    }
    assert_float_slice_close(&streamed_wma, batch_wma.values());
    assert_float_slice_close(&streamed_trima, batch_trima.values());

    assert_eq!(
        StreamingComputation::<WMAConfig>::next(&mut independent_wma, 10.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<TRIMAConfig>::next(&mut independent_trima, 10.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<WMAConfig>::next(&mut independent_wma, 20.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<TRIMAConfig>::next(&mut independent_trima, 20.0).unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<WMAConfig>::next(&mut independent_wma, 30.0).unwrap(),
        140.0 / 6.0,
    );
    assert_some_float_close(
        StreamingComputation::<TRIMAConfig>::next(&mut independent_trima, 30.0).unwrap(),
        20.0,
    );
    StreamingComputation::<WMAConfig>::reset(&mut wma_stream);
    StreamingComputation::<TRIMAConfig>::reset(&mut trima_stream);
    Resettable::reset(&mut legacy_wma);
    Resettable::reset(&mut legacy_trima);
    assert_some_float_close(
        StreamingComputation::<WMAConfig>::next(&mut independent_wma, 40.0).unwrap(),
        200.0 / 6.0,
    );
    assert_some_float_close(
        StreamingComputation::<TRIMAConfig>::next(&mut independent_trima, 40.0).unwrap(),
        30.0,
    );

    let replayed_wma = input
        .iter()
        .filter_map(|&tick| {
            let value = StreamingComputation::<WMAConfig>::next(&mut wma_stream, tick).unwrap();
            assert_eq!(
                StreamingIndicator::next(&mut legacy_wma, tick).unwrap(),
                value
            );
            value
        })
        .collect::<Vec<_>>();
    let replayed_trima = input
        .iter()
        .filter_map(|&tick| {
            let value = StreamingComputation::<TRIMAConfig>::next(&mut trima_stream, tick).unwrap();
            assert_eq!(
                StreamingIndicator::next(&mut legacy_trima, tick).unwrap(),
                value
            );
            value
        })
        .collect::<Vec<_>>();
    assert_float_slice_close(&replayed_wma, batch_wma.values());
    assert_float_slice_close(&replayed_trima, batch_trima.values());
    assert_eq!(
        core::mem::size_of::<WMA>(),
        core::mem::size_of::<WMAStream>()
    );
    assert_eq!(
        core::mem::size_of::<TRIMA>(),
        core::mem::size_of::<TRIMAStream>()
    );
}

#[test]
fn minmax_configs_are_parameter_only_and_return_exact_named_compact_payloads() {
    let input = [3.0 as Float, 1.0, 4.0, 2.0];
    let values_config = MINMAXConfig::new(2).unwrap();
    let indexes_config = MINMAXINDEXConfig::new(2).unwrap();

    let values = IndicatorConfig::compute(&values_config, &input).unwrap();
    assert_eq!(values.source_len(), input.len());
    assert_eq!(values.range(), OutputRange::new(1, 3));
    assert_float_slice_close(&values.values().min, &[1.0, 1.0, 2.0]);
    assert_float_slice_close(&values.values().max, &[3.0, 4.0, 4.0]);
    assert_eq!(values.values().min.capacity(), 3);
    assert_eq!(values.values().max.capacity(), 3);

    let indexes = IndicatorConfig::compute(&indexes_config, &input).unwrap();
    assert_eq!(indexes.source_len(), input.len());
    assert_eq!(indexes.range(), values.range());
    assert_eq!(indexes.values().min_idx, vec![1_usize, 1, 3]);
    assert_eq!(indexes.values().max_idx, vec![0_usize, 2, 2]);
    assert_eq!(indexes.values().min_idx.capacity(), 3);
    assert_eq!(indexes.values().max_idx.capacity(), 3);

    assert_eq!(values_config.period(), 2);
    assert_eq!(indexes_config.period(), 2);
    assert_eq!(IndicatorConfig::lookback(&values_config), 1);
    assert_eq!(IndicatorConfig::lookback(&indexes_config), 1);
    assert_eq!(
        core::mem::size_of::<MINMAXConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<MINMAXINDEXConfig>(),
        core::mem::size_of::<usize>()
    );
}

#[test]
fn minmax_config_caller_owned_paths_match_owned_and_leave_tails_untouched() {
    let input = [3.0 as Float, 1.0, 4.0, 2.0];
    let values_config = MINMAXConfig::new(2).unwrap();
    let indexes_config = MINMAXINDEXConfig::new(2).unwrap();
    let owned_values = IndicatorConfig::compute(&values_config, &input).unwrap();
    let owned_indexes = IndicatorConfig::compute(&indexes_config, &input).unwrap();
    let mut min = [FLOAT_SENTINEL; 5];
    let mut max = [FLOAT_SENTINEL; 5];
    let mut min_idx = [usize::MAX; 5];
    let mut max_idx = [usize::MAX; 5];

    let value_range = IndicatorConfig::compute_into(
        &values_config,
        &input,
        MINMAXValuesMut {
            min: &mut min,
            max: &mut max,
        },
    )
    .unwrap();
    let index_range = IndicatorConfig::compute_into(
        &indexes_config,
        &input,
        MINMAXINDEXValuesMut {
            min_idx: &mut min_idx,
            max_idx: &mut max_idx,
        },
    )
    .unwrap();

    assert_eq!(value_range, owned_values.range());
    assert_eq!(index_range, owned_indexes.range());
    assert_float_slice_close(&min[..3], &owned_values.values().min);
    assert_float_slice_close(&max[..3], &owned_values.values().max);
    assert_eq!(&min[3..], &[FLOAT_SENTINEL; 2]);
    assert_eq!(&max[3..], &[FLOAT_SENTINEL; 2]);
    assert_eq!(&min_idx[..3], owned_indexes.values().min_idx.as_slice());
    assert_eq!(&max_idx[..3], owned_indexes.values().max_idx.as_slice());
    assert_eq!(&min_idx[3..], &[usize::MAX; 2]);
    assert_eq!(&max_idx[3..], &[usize::MAX; 2]);
}

#[test]
fn minmax_config_validation_is_pre_mutation_for_both_columns() {
    let values_config = MINMAXConfig::new(3).unwrap();
    let indexes_config = MINMAXINDEXConfig::new(3).unwrap();
    let input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let mut min = [FLOAT_SENTINEL; 3];
    let mut short_max = [FLOAT_SENTINEL; 2];
    let mut min_idx = [usize::MAX; 3];
    let mut short_max_idx = [usize::MAX; 2];

    assert!(IndicatorConfig::compute_into(
        &values_config,
        &input,
        MINMAXValuesMut {
            min: &mut min,
            max: &mut short_max,
        },
    )
    .is_err());
    assert_eq!(min, [FLOAT_SENTINEL; 3]);
    assert_eq!(short_max, [FLOAT_SENTINEL; 2]);

    assert!(IndicatorConfig::compute_into(
        &indexes_config,
        &[1.0 as Float, Float::NAN, 3.0, 4.0, 5.0],
        MINMAXINDEXValuesMut {
            min_idx: &mut min_idx,
            max_idx: &mut short_max_idx,
        },
    )
    .is_err());
    assert_eq!(min_idx, [usize::MAX; 3]);
    assert_eq!(short_max_idx, [usize::MAX; 2]);
}

#[test]
fn prepared_minmax_runners_reuse_scratch_and_prioritize_oversize_rejection() {
    let values_config = MINMAXConfig::new(3).unwrap();
    let indexes_config = MINMAXINDEXConfig::new(3).unwrap();
    let mut values_runner = IndicatorConfig::prepare_batch(&values_config, 5).unwrap();
    let mut indexes_runner = IndicatorConfig::prepare_batch(&indexes_config, 5).unwrap();
    let input = [2.0 as Float, 1.0, 1.0, 3.0, 3.0];
    let mut min = [FLOAT_SENTINEL; 3];
    let mut max = [FLOAT_SENTINEL; 3];
    let mut min_idx = [usize::MAX; 3];
    let mut max_idx = [usize::MAX; 3];

    PreparedBatchRunner::<MINMAXConfig>::compute_into(
        &mut values_runner,
        &input,
        MINMAXValuesMut {
            min: &mut min,
            max: &mut max,
        },
    )
    .unwrap();
    PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut indexes_runner,
        &input,
        MINMAXINDEXValuesMut {
            min_idx: &mut min_idx,
            max_idx: &mut max_idx,
        },
    )
    .unwrap();
    assert_float_slice_close(&min, &[1.0, 1.0, 1.0]);
    assert_float_slice_close(&max, &[2.0, 3.0, 3.0]);
    assert_eq!(min_idx, [1, 1, 2]);
    assert_eq!(max_idx, [0, 3, 3]);

    let oversized = [Float::NAN; 6];
    let mut untouched_min = [FLOAT_SENTINEL; 1];
    let mut untouched_max = [FLOAT_SENTINEL; 1];
    let error = PreparedBatchRunner::<MINMAXConfig>::compute_into(
        &mut values_runner,
        &oversized,
        MINMAXValuesMut {
            min: &mut untouched_min,
            max: &mut untouched_max,
        },
    )
    .unwrap_err();
    assert_eq!(
        error,
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(untouched_min, [FLOAT_SENTINEL]);
    assert_eq!(untouched_max, [FLOAT_SENTINEL]);

    let mut untouched_min_idx = [usize::MAX; 1];
    let mut untouched_max_idx = [usize::MAX; 1];
    let index_error = PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut indexes_runner,
        &oversized,
        MINMAXINDEXValuesMut {
            min_idx: &mut untouched_min_idx,
            max_idx: &mut untouched_max_idx,
        },
    )
    .unwrap_err();
    assert_eq!(index_error, error);
    assert_eq!(untouched_min_idx, [usize::MAX]);
    assert_eq!(untouched_max_idx, [usize::MAX]);

    // A rejected within-capacity call must not poison retained scratch semantics.
    let mut too_short_max = [FLOAT_SENTINEL; 2];
    assert!(PreparedBatchRunner::<MINMAXConfig>::compute_into(
        &mut values_runner,
        &input,
        MINMAXValuesMut {
            min: &mut min,
            max: &mut too_short_max,
        },
    )
    .is_err());
    min.fill(FLOAT_SENTINEL);
    max.fill(FLOAT_SENTINEL);
    PreparedBatchRunner::<MINMAXConfig>::compute_into(
        &mut values_runner,
        &[5.0 as Float, 4.0, 3.0, 2.0, 1.0],
        MINMAXValuesMut {
            min: &mut min,
            max: &mut max,
        },
    )
    .unwrap();
    assert_float_slice_close(&min, &[3.0, 2.0, 1.0]);
    assert_float_slice_close(&max, &[5.0, 4.0, 3.0]);

    let prior_min_idx = min_idx;
    let prior_max_idx = max_idx;
    let mut too_short_max_idx = [usize::MAX; 2];
    assert!(PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut indexes_runner,
        &input,
        MINMAXINDEXValuesMut {
            min_idx: &mut min_idx,
            max_idx: &mut too_short_max_idx,
        },
    )
    .is_err());
    assert_eq!(min_idx, prior_min_idx);
    assert_eq!(max_idx, prior_max_idx);
    assert_eq!(too_short_max_idx, [usize::MAX; 2]);

    PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut indexes_runner,
        &[5.0 as Float, 4.0, 3.0, 2.0, 1.0],
        MINMAXINDEXValuesMut {
            min_idx: &mut min_idx,
            max_idx: &mut max_idx,
        },
    )
    .unwrap();
    assert_eq!(min_idx, [2, 3, 4]);
    assert_eq!(max_idx, [0, 1, 2]);
}

#[test]
fn prepared_minmax_runners_are_independent_per_worker() {
    let config = MINMAXINDEXConfig::new(2).unwrap();
    let mut left = IndicatorConfig::prepare_batch(&config, 4).unwrap();
    let mut right = IndicatorConfig::prepare_batch(&config, 4).unwrap();
    let mut left_min = [usize::MAX; 3];
    let mut left_max = [usize::MAX; 3];
    let mut right_min = [usize::MAX; 3];
    let mut right_max = [usize::MAX; 3];

    PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut left,
        &[1.0 as Float, 3.0, 2.0, 4.0],
        MINMAXINDEXValuesMut {
            min_idx: &mut left_min,
            max_idx: &mut left_max,
        },
    )
    .unwrap();
    PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
        &mut right,
        &[40.0 as Float, 30.0, 20.0, 10.0],
        MINMAXINDEXValuesMut {
            min_idx: &mut right_min,
            max_idx: &mut right_max,
        },
    )
    .unwrap();

    assert_eq!(left_min, [0, 2, 2]);
    assert_eq!(left_max, [1, 1, 3]);
    assert_eq!(right_min, [1, 2, 3]);
    assert_eq!(right_max, [0, 1, 2]);
}

#[test]
fn minmax_configs_preserve_period_one_empty_short_nonfinite_and_oldest_ties() {
    assert!(matches!(
        MINMAXConfig::new(0),
        Err(TalibError::InvalidPeriod { period: 0, .. })
    ));
    assert!(MINMAXINDEXConfig::new(0).is_err());

    for output in [
        IndicatorConfig::compute(&MINMAXConfig::new(3).unwrap(), &[]).unwrap(),
        IndicatorConfig::compute(&MINMAXConfig::new(1).unwrap(), &[]).unwrap(),
    ] {
        assert_eq!(output.source_len(), 0);
        assert_eq!(output.range(), OutputRange::empty());
        assert!(output.values().min.is_empty());
        assert!(output.values().max.is_empty());
    }

    assert!(matches!(
        IndicatorConfig::compute(&MINMAXConfig::new(3).unwrap(), &[1.0 as Float, 2.0]),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));
    assert!(IndicatorConfig::compute(
        &MINMAXINDEXConfig::new(2).unwrap(),
        &[1.0 as Float, Float::INFINITY],
    )
    .is_err());

    let input = [4.0 as Float, -1.0, 7.0, 7.0];
    let values = IndicatorConfig::compute(&MINMAXConfig::new(1).unwrap(), &input).unwrap();
    let indexes = IndicatorConfig::compute(&MINMAXINDEXConfig::new(1).unwrap(), &input).unwrap();
    assert_float_slice_close(&values.values().min, &input);
    assert_float_slice_close(&values.values().max, &input);
    assert_eq!(indexes.values().min_idx, vec![0, 1, 2, 3]);
    assert_eq!(indexes.values().max_idx, vec![0, 1, 2, 3]);
}

#[test]
fn minmax_config_streams_are_independent_reject_invalid_ticks_reset_and_match_batch() {
    let input = [2.0 as Float, 1.0, 1.0, 3.0, 3.0];
    let values_config = MINMAXConfig::new(3).unwrap();
    let indexes_config = MINMAXINDEXConfig::new(3).unwrap();
    let batch_values = IndicatorConfig::compute(&values_config, &input).unwrap();
    let batch_indexes = IndicatorConfig::compute(&indexes_config, &input).unwrap();
    let mut values_stream = IndicatorConfig::stream(&values_config).unwrap();
    let mut indexes_stream = IndicatorConfig::stream(&indexes_config).unwrap();
    let mut streamed_values = Vec::new();
    let mut streamed_indexes: Vec<MINMAXINDEXStreamValue> = Vec::new();

    assert_eq!(
        StreamingComputation::<MINMAXConfig>::next(&mut values_stream, 2.0).unwrap(),
        None
    );
    assert!(StreamingComputation::<MINMAXConfig>::next(&mut values_stream, Float::NAN).is_err());
    StreamingComputation::<MINMAXConfig>::reset(&mut values_stream);

    for &tick in &input {
        if let Some(value) =
            StreamingComputation::<MINMAXConfig>::next(&mut values_stream, tick).unwrap()
        {
            streamed_values.push(value);
        }
        if let Some(value) =
            StreamingComputation::<MINMAXINDEXConfig>::next(&mut indexes_stream, tick).unwrap()
        {
            streamed_indexes.push(value);
        }
    }

    assert_eq!(streamed_values.len(), batch_values.range().nb_element);
    for (index, value) in streamed_values.iter().enumerate() {
        assert_float_close(value.min, batch_values.values().min[index]);
        assert_float_close(value.max, batch_values.values().max[index]);
    }
    assert_eq!(
        streamed_indexes
            .iter()
            .map(|value| value.min_idx)
            .collect::<Vec<_>>(),
        batch_indexes.values().min_idx
    );
    assert_eq!(
        streamed_indexes
            .iter()
            .map(|value| value.max_idx)
            .collect::<Vec<_>>(),
        batch_indexes.values().max_idx
    );

    let mut independent = IndicatorConfig::stream(&indexes_config).unwrap();
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut independent, 10.0).unwrap(),
        None
    );
    StreamingComputation::<MINMAXINDEXConfig>::reset(&mut indexes_stream);
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut independent, 20.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut independent, 30.0).unwrap(),
        Some(MINMAXINDEXStreamValue {
            min_idx: 0,
            max_idx: 2,
        })
    );

    let mut rejected_tick = IndicatorConfig::stream(&indexes_config).unwrap();
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut rejected_tick, 5.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut rejected_tick, 4.0).unwrap(),
        None
    );
    assert!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut rejected_tick, Float::INFINITY,)
            .is_err()
    );
    assert_eq!(
        StreamingComputation::<MINMAXINDEXConfig>::next(&mut rejected_tick, 3.0).unwrap(),
        Some(MINMAXINDEXStreamValue {
            min_idx: 2,
            max_idx: 0,
        })
    );
}

#[test]
fn legacy_minmax_adapters_match_config_streams_and_preserve_ties_and_warmup() {
    let input = [2.0 as Float, 1.0, 1.0, 3.0, 3.0];
    let mut legacy_values = MINMAX::new(3).unwrap();
    let mut legacy_indexes = MINMAXINDEX::new(3).unwrap();
    let value_config = MINMAXConfig::new(3).unwrap();
    let index_config = MINMAXINDEXConfig::new(3).unwrap();
    let mut config_values = IndicatorConfig::stream(&value_config).unwrap();
    let mut config_indexes = IndicatorConfig::stream(&index_config).unwrap();

    for &tick in &input {
        assert_eq!(
            StreamingIndicator::next(&mut legacy_values, tick).unwrap(),
            StreamingComputation::<MINMAXConfig>::next(&mut config_values, tick).unwrap()
        );
        let expected = StreamingComputation::<MINMAXINDEXConfig>::next(&mut config_indexes, tick)
            .unwrap()
            .map(|value| ta_core::math_operators::MINMAXINDEXValue {
                min_idx: i32::try_from(value.min_idx).unwrap(),
                max_idx: i32::try_from(value.max_idx).unwrap(),
            });
        assert_eq!(
            StreamingIndicator::next(&mut legacy_indexes, tick).unwrap(),
            expected
        );
    }

    assert_eq!(legacy_values.period(), 3);
    assert_eq!(legacy_indexes.period(), 3);
    assert_eq!(Indicator::lookback(&legacy_values), 2);
    assert_eq!(Indicator::lookback(&legacy_indexes), 2);
    assert_eq!(
        core::mem::size_of::<MINMAX>(),
        core::mem::size_of::<MINMAXStream>()
    );
    assert_eq!(
        core::mem::size_of::<MINMAXINDEX>(),
        core::mem::size_of::<MINMAXINDEXStream>()
    );
}

#[test]
fn legacy_minmax_adapters_clone_independently_preserve_invalid_ticks_and_reset_replay() {
    let mut values = MINMAX::new(3).unwrap();
    let mut indexes = MINMAXINDEX::new(3).unwrap();
    for tick in [5.0 as Float, 4.0] {
        assert_eq!(StreamingIndicator::next(&mut values, tick).unwrap(), None);
        assert_eq!(StreamingIndicator::next(&mut indexes, tick).unwrap(), None);
    }
    let mut values_clone = values.clone();
    let mut indexes_clone = indexes.clone();

    assert!(StreamingIndicator::next(&mut values, Float::NAN).is_err());
    assert!(StreamingIndicator::next(&mut indexes, Float::INFINITY).is_err());
    assert_eq!(
        StreamingIndicator::next(&mut values, 3.0).unwrap(),
        Some(ta_core::math_operators::MINMAXValue { min: 3.0, max: 5.0 })
    );
    assert_eq!(
        StreamingIndicator::next(&mut indexes, 3.0).unwrap(),
        Some(ta_core::math_operators::MINMAXINDEXValue {
            min_idx: 2,
            max_idx: 0,
        })
    );
    assert_eq!(
        StreamingIndicator::next(&mut values_clone, 10.0).unwrap(),
        Some(ta_core::math_operators::MINMAXValue {
            min: 4.0,
            max: 10.0,
        })
    );
    assert_eq!(
        StreamingIndicator::next(&mut indexes_clone, 10.0).unwrap(),
        Some(ta_core::math_operators::MINMAXINDEXValue {
            min_idx: 1,
            max_idx: 2,
        })
    );

    let replay = [2.0 as Float, 1.0, 4.0, 3.0];
    Resettable::reset(&mut values);
    Resettable::reset(&mut indexes);
    let first_values = replay
        .iter()
        .map(|&tick| StreamingIndicator::next(&mut values, tick).unwrap())
        .collect::<Vec<_>>();
    let first_indexes = replay
        .iter()
        .map(|&tick| StreamingIndicator::next(&mut indexes, tick).unwrap())
        .collect::<Vec<_>>();
    Resettable::reset(&mut values);
    Resettable::reset(&mut indexes);
    let second_values = replay
        .iter()
        .map(|&tick| StreamingIndicator::next(&mut values, tick).unwrap())
        .collect::<Vec<_>>();
    let second_indexes = replay
        .iter()
        .map(|&tick| StreamingIndicator::next(&mut indexes, tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first_values, second_values);
    assert_eq!(first_indexes, second_indexes);
}

#[test]
fn single_extrema_configs_cover_owned_caller_owned_and_prepared_execution() {
    let input = [3.0 as Float, 1.0, 1.0, 4.0, 2.0];
    let min_config = MINConfig::new(3).unwrap();
    let max_config = MAXConfig::new(3).unwrap();
    let min_index_config = MININDEXConfig::new(3).unwrap();
    let max_index_config = MAXINDEXConfig::new(3).unwrap();
    assert_eq!(min_config.period(), 3);
    assert_eq!(max_config.period(), 3);
    assert_eq!(min_index_config.period(), 3);
    assert_eq!(max_index_config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&min_config), 2);
    assert_eq!(IndicatorConfig::lookback(&max_config), 2);
    assert_eq!(IndicatorConfig::lookback(&min_index_config), 2);
    assert_eq!(IndicatorConfig::lookback(&max_index_config), 2);
    assert_eq!(
        core::mem::size_of::<MINConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<MAXConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<MININDEXConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<MAXINDEXConfig>(),
        core::mem::size_of::<usize>()
    );

    let min = IndicatorConfig::compute(&min_config, &input).unwrap();
    let max = IndicatorConfig::compute(&max_config, &input).unwrap();
    let min_index = IndicatorConfig::compute(&min_index_config, &input).unwrap();
    let max_index = IndicatorConfig::compute(&max_index_config, &input).unwrap();
    assert_eq!(min.source_len(), input.len());
    assert_eq!(max.source_len(), input.len());
    assert_eq!(min_index.source_len(), input.len());
    assert_eq!(max_index.source_len(), input.len());
    assert_eq!(min.range(), OutputRange::new(2, 3));
    assert_eq!(max.range(), min.range());
    assert_eq!(min_index.range(), min.range());
    assert_eq!(max_index.range(), min.range());
    assert_float_slice_close(min.values(), &[1.0, 1.0, 1.0]);
    assert_float_slice_close(max.values(), &[3.0, 4.0, 4.0]);
    assert_eq!(min_index.values(), &[1_usize, 1, 2]);
    assert_eq!(max_index.values(), &[0_usize, 3, 3]);

    let mut min_out = [FLOAT_SENTINEL; 4];
    let mut max_out = [FLOAT_SENTINEL; 4];
    let mut min_index_out = [usize::MAX; 4];
    let mut max_index_out = [usize::MAX; 4];
    let min_range = IndicatorConfig::compute_into(&min_config, &input, &mut min_out).unwrap();
    let max_range = IndicatorConfig::compute_into(&max_config, &input, &mut max_out).unwrap();
    let min_index_range =
        IndicatorConfig::compute_into(&min_index_config, &input, &mut min_index_out).unwrap();
    let max_index_range =
        IndicatorConfig::compute_into(&max_index_config, &input, &mut max_index_out).unwrap();
    assert_eq!(min_range, min.range());
    assert_eq!(max_range, max.range());
    assert_eq!(min_index_range, min_index.range());
    assert_eq!(max_index_range, max_index.range());
    assert_float_slice_close(&min_out[..3], min.values());
    assert_float_slice_close(&max_out[..3], max.values());
    assert_eq!(&min_index_out[..3], min_index.values());
    assert_eq!(&max_index_out[..3], max_index.values());
    assert_eq!(min_out[3], FLOAT_SENTINEL);
    assert_eq!(max_out[3], FLOAT_SENTINEL);
    assert_eq!(min_index_out[3], usize::MAX);
    assert_eq!(max_index_out[3], usize::MAX);

    min_out.fill(FLOAT_SENTINEL);
    max_out.fill(FLOAT_SENTINEL);
    min_index_out.fill(usize::MAX);
    max_index_out.fill(usize::MAX);
    let mut min_runner = IndicatorConfig::prepare_batch(&min_config, input.len()).unwrap();
    let mut max_runner = IndicatorConfig::prepare_batch(&max_config, input.len()).unwrap();
    let mut min_index_runner =
        IndicatorConfig::prepare_batch(&min_index_config, input.len()).unwrap();
    let mut max_index_runner =
        IndicatorConfig::prepare_batch(&max_index_config, input.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<MINConfig>::max_input_len(&min_runner),
        5
    );
    assert_eq!(
        PreparedBatchRunner::<MAXConfig>::max_input_len(&max_runner),
        5
    );
    assert_eq!(
        PreparedBatchRunner::<MININDEXConfig>::max_input_len(&min_index_runner),
        5
    );
    assert_eq!(
        PreparedBatchRunner::<MAXINDEXConfig>::max_input_len(&max_index_runner),
        5
    );

    PreparedBatchRunner::<MINConfig>::compute_into(&mut min_runner, &input, &mut min_out).unwrap();
    PreparedBatchRunner::<MAXConfig>::compute_into(&mut max_runner, &input, &mut max_out).unwrap();
    PreparedBatchRunner::<MININDEXConfig>::compute_into(
        &mut min_index_runner,
        &input,
        &mut min_index_out,
    )
    .unwrap();
    PreparedBatchRunner::<MAXINDEXConfig>::compute_into(
        &mut max_index_runner,
        &input,
        &mut max_index_out,
    )
    .unwrap();
    assert_float_slice_close(&min_out[..3], min.values());
    assert_float_slice_close(&max_out[..3], max.values());
    assert_eq!(&min_index_out[..3], min_index.values());
    assert_eq!(&max_index_out[..3], max_index.values());
    assert_eq!(min_out[3], FLOAT_SENTINEL);
    assert_eq!(max_out[3], FLOAT_SENTINEL);
    assert_eq!(min_index_out[3], usize::MAX);
    assert_eq!(max_index_out[3], usize::MAX);

    let oversized = [Float::NAN; 6];
    assert!(matches!(
        PreparedBatchRunner::<MINConfig>::compute_into(&mut min_runner, &oversized, &mut min_out),
        Err(TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6
        })
    ));
    assert_eq!(min_out[0], 1.0);

    for config_result in [
        MINConfig::new(0).map(|_| ()),
        MAXConfig::new(0).map(|_| ()),
        MININDEXConfig::new(0).map(|_| ()),
        MAXINDEXConfig::new(0).map(|_| ()),
    ] {
        assert!(matches!(
            config_result,
            Err(TalibError::InvalidPeriod { period: 0, .. })
        ));
    }
    assert!(
        IndicatorConfig::compute(&MININDEXConfig::new(2).unwrap(), &[])
            .unwrap()
            .values()
            .is_empty()
    );
    assert!(IndicatorConfig::compute(
        &MAXINDEXConfig::new(2).unwrap(),
        &[1.0 as Float, Float::NAN]
    )
    .is_err());
}

#[test]
fn single_extrema_streams_preserve_option_reset_parity_and_legacy_adapters() {
    let input = [3.0 as Float, 1.0, 1.0, 4.0, 2.0];
    let min_config = MINConfig::new(3).unwrap();
    let max_config = MAXConfig::new(3).unwrap();
    let min_index_config = MININDEXConfig::new(3).unwrap();
    let max_index_config = MAXINDEXConfig::new(3).unwrap();
    let batch_min = IndicatorConfig::compute(&min_config, &input).unwrap();
    let batch_max = IndicatorConfig::compute(&max_config, &input).unwrap();
    let batch_min_index = IndicatorConfig::compute(&min_index_config, &input).unwrap();
    let batch_max_index = IndicatorConfig::compute(&max_index_config, &input).unwrap();
    let mut min_stream = IndicatorConfig::stream(&min_config).unwrap();
    let mut max_stream = IndicatorConfig::stream(&max_config).unwrap();
    let mut min_index_stream = IndicatorConfig::stream(&min_index_config).unwrap();
    let mut max_index_stream = IndicatorConfig::stream(&max_index_config).unwrap();
    let mut legacy_min = MIN::new(3).unwrap();
    let mut legacy_max = MAX::new(3).unwrap();
    let mut legacy_min_index = MININDEX::new(3).unwrap();
    let mut legacy_max_index = MAXINDEX::new(3).unwrap();
    let mut streamed_min = Vec::new();
    let mut streamed_max = Vec::new();
    let mut streamed_min_index = Vec::new();
    let mut streamed_max_index = Vec::new();

    for &tick in &input {
        let min_value = StreamingComputation::<MINConfig>::next(&mut min_stream, tick).unwrap();
        let max_value = StreamingComputation::<MAXConfig>::next(&mut max_stream, tick).unwrap();
        let min_index_value =
            StreamingComputation::<MININDEXConfig>::next(&mut min_index_stream, tick).unwrap();
        let max_index_value =
            StreamingComputation::<MAXINDEXConfig>::next(&mut max_index_stream, tick).unwrap();
        assert_eq!(
            StreamingIndicator::next(&mut legacy_min, tick).unwrap(),
            min_value
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_max, tick).unwrap(),
            max_value
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_min_index, tick).unwrap(),
            min_index_value.map(|index| i32::try_from(index).unwrap())
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_max_index, tick).unwrap(),
            max_index_value.map(|index| i32::try_from(index).unwrap())
        );
        streamed_min.extend(min_value);
        streamed_max.extend(max_value);
        streamed_min_index.extend(min_index_value);
        streamed_max_index.extend(max_index_value);
    }
    assert_float_slice_close(&streamed_min, batch_min.values());
    assert_float_slice_close(&streamed_max, batch_max.values());
    assert_eq!(streamed_min_index, batch_min_index.values().as_slice());
    assert_eq!(streamed_max_index, batch_max_index.values().as_slice());

    assert!(
        StreamingComputation::<MININDEXConfig>::next(&mut min_index_stream, Float::INFINITY)
            .is_err()
    );
    StreamingComputation::<MINConfig>::reset(&mut min_stream);
    StreamingComputation::<MAXConfig>::reset(&mut max_stream);
    StreamingComputation::<MININDEXConfig>::reset(&mut min_index_stream);
    StreamingComputation::<MAXINDEXConfig>::reset(&mut max_index_stream);
    Resettable::reset(&mut legacy_min);
    Resettable::reset(&mut legacy_max);
    Resettable::reset(&mut legacy_min_index);
    Resettable::reset(&mut legacy_max_index);
    let mut replayed_min = Vec::new();
    let mut replayed_max = Vec::new();
    let mut replayed_min_index = Vec::new();
    let mut replayed_max_index = Vec::new();
    for &tick in &input {
        let min_value = StreamingComputation::<MINConfig>::next(&mut min_stream, tick).unwrap();
        let max_value = StreamingComputation::<MAXConfig>::next(&mut max_stream, tick).unwrap();
        let min_index_value =
            StreamingComputation::<MININDEXConfig>::next(&mut min_index_stream, tick).unwrap();
        let max_index_value =
            StreamingComputation::<MAXINDEXConfig>::next(&mut max_index_stream, tick).unwrap();
        assert_eq!(
            StreamingIndicator::next(&mut legacy_min, tick).unwrap(),
            min_value
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_max, tick).unwrap(),
            max_value
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_min_index, tick).unwrap(),
            min_index_value.map(|index| i32::try_from(index).unwrap())
        );
        assert_eq!(
            StreamingIndicator::next(&mut legacy_max_index, tick).unwrap(),
            max_index_value.map(|index| i32::try_from(index).unwrap())
        );
        replayed_min.extend(min_value);
        replayed_max.extend(max_value);
        replayed_min_index.extend(min_index_value);
        replayed_max_index.extend(max_index_value);
    }
    assert_float_slice_close(&replayed_min, batch_min.values());
    assert_float_slice_close(&replayed_max, batch_max.values());
    assert_eq!(replayed_min_index, batch_min_index.values().as_slice());
    assert_eq!(replayed_max_index, batch_max_index.values().as_slice());
    assert_eq!(
        core::mem::size_of::<MIN>(),
        core::mem::size_of::<MINStream>()
    );
    assert_eq!(
        core::mem::size_of::<MAX>(),
        core::mem::size_of::<MAXStream>()
    );
    assert_eq!(
        core::mem::size_of::<MININDEX>(),
        core::mem::size_of::<MININDEXStream>()
    );
    assert_eq!(
        core::mem::size_of::<MAXINDEX>(),
        core::mem::size_of::<MAXINDEXStream>()
    );
}

#[test]
fn legacy_sma_names_and_signatures_remain_source_compatible() {
    use ta_core::overlap::{SMA_vec, SMA};

    let input = [1.0 as Float, 2.0, 3.0, 4.0];
    let mut compact = [FLOAT_SENTINEL; 2];
    let range = SMA(&input, 3, &mut compact).unwrap();
    let aligned = SMA_vec(&input, 3).unwrap();
    let mut indicator = SMA::from_data(3, &input[..2]).unwrap();
    let mut empty_output = [];

    assert_eq!(
        SMA(&[], 3, &mut empty_output).unwrap(),
        OutputRange::empty()
    );
    assert!(SMA_vec(&[], 3).unwrap().is_empty());
    assert_eq!(range, OutputRange::new(2, 2));
    assert!(aligned[..2].iter().all(|value| value.is_nan()));
    assert_float_close(indicator.next_checked(3.0).unwrap(), 2.0);
    assert_eq!(indicator.period(), 3);
    assert_eq!(Indicator::lookback(&indicator), 2);
    assert_eq!(
        core::mem::size_of::<SMA>(),
        core::mem::size_of::<SMAStream>()
    );
}
