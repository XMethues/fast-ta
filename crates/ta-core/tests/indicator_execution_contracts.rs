//! Public contracts for the Indicator execution seam.
//!
//! Migration tests exercise each Indicator Configuration through owned
//! Compact Output, caller-owned output, Prepared Batch Runners, and
//! independent Streaming Computation.

use ta_core::{
    math_operators::{
        ADDConfig, BinaryInput, BinaryTick, DIVConfig, MAXConfig, MAXINDEXConfig, MINConfig,
        MININDEXConfig, MINMAXConfig, MINMAXINDEXConfig, MINMAXINDEXStreamValue,
        MINMAXINDEXValuesMut, MINMAXValuesMut, MULTConfig, SUBConfig, SUMConfig,
    },
    math_transform::{
        ACOSConfig, ACOSStream, ASINConfig, ASINStream, ATANConfig, ATANStream, CEILConfig,
        CEILStream, COSConfig, COSHConfig, COSHStream, COSStream, EXPConfig, EXPStream,
        FLOORConfig, FLOORStream, LNConfig, LNStream, LOG10Config, LOG10Stream, SINConfig,
        SINHConfig, SINHStream, SINStream, SQRTConfig, SQRTStream, TANConfig, TANHConfig,
        TANHStream, TANStream,
    },
    overlap::{
        DEMAConfig, EMAConfig, MAConfig, PeriodMAType, SMAConfig, T3Config, TEMAConfig,
        TRIMAConfig, WMAConfig, T3_DEFAULT_VFACTOR,
    },
    price_transform::{
        AVGDEVConfig, AVGPRICEConfig, AVGPRICEInput, AVGPRICETick, MEDPRICEConfig, MEDPRICEInput,
        MEDPRICETick, TYPPRICEConfig, TYPPRICEInput, TYPPRICETick, WCLPRICEConfig, WCLPRICEInput,
        WCLPRICETick,
    },
    statistic::{
        BETAConfig, CORRELConfig, LINEARREGConfig, LINEARREG_ANGLEConfig,
        LINEARREG_INTERCEPTConfig, LINEARREG_SLOPEConfig, PairInput, PairTick, STDDEVConfig,
        TSFConfig, VARConfig,
    },
    volatility::{
        ATRConfig, ATRInput, ATRTick, NATRConfig, NATRInput, NATRTick, TRANGEConfig, TRANGEInput,
        TRANGETick,
    },
    volume::{
        ADConfig, ADInput, ADOSCConfig, ADOSCInput, ADOSCTick, ADTick, OBVConfig, OBVInput, OBVTick,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
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
fn ema_config_separates_batch_and_streaming_execution() {
    let input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let config = EMAConfig::new(3).unwrap();
    assert_eq!(config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert_eq!(
        core::mem::size_of::<EMAConfig>(),
        core::mem::size_of::<usize>()
    );

    let owned = IndicatorConfig::compute(&config, &input).unwrap();
    assert_eq!(owned.source_len(), input.len());
    assert_eq!(owned.range(), OutputRange::new(2, 3));
    assert_float_slice_close(owned.values(), &[2.0, 3.0, 4.0]);

    let mut output = [FLOAT_SENTINEL; 4];
    let range = IndicatorConfig::compute_into(&config, &input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(output[3], FLOAT_SENTINEL);

    let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
    output.fill(FLOAT_SENTINEL);
    PreparedBatchRunner::<EMAConfig>::compute_into(&mut runner, &input, &mut output).unwrap();
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(output[3], FLOAT_SENTINEL);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let mut independent = IndicatorConfig::stream(&config).unwrap();
    let streamed = input
        .iter()
        .filter_map(|&tick| StreamingComputation::<EMAConfig>::next(&mut stream, tick).unwrap())
        .collect::<Vec<_>>();

    assert_float_slice_close(&streamed, owned.values());

    assert_eq!(
        StreamingComputation::<EMAConfig>::next(&mut independent, 10.0).unwrap(),
        None
    );
    assert!(StreamingComputation::<EMAConfig>::next(&mut independent, Float::NAN).is_err());
    assert_eq!(
        StreamingComputation::<EMAConfig>::next(&mut independent, 20.0).unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<EMAConfig>::next(&mut independent, 30.0).unwrap(),
        20.0,
    );
    StreamingComputation::<EMAConfig>::reset(&mut stream);
    let replayed = input
        .iter()
        .filter_map(|&tick| StreamingComputation::<EMAConfig>::next(&mut stream, tick).unwrap())
        .collect::<Vec<_>>();

    assert_float_slice_close(&replayed, owned.values());
}

#[test]
fn dema_and_tema_configs_preserve_recursive_seeds_and_execution_modes() {
    let input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let dema_config = DEMAConfig::new(3).unwrap();
    let tema_config = TEMAConfig::new(3).unwrap();
    assert_eq!(dema_config.period(), 3);
    assert_eq!(tema_config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&dema_config), 4);
    assert_eq!(IndicatorConfig::lookback(&tema_config), 6);
    assert_eq!(
        core::mem::size_of::<DEMAConfig>(),
        core::mem::size_of::<usize>()
    );
    assert_eq!(
        core::mem::size_of::<TEMAConfig>(),
        core::mem::size_of::<usize>()
    );

    let dema_owned = IndicatorConfig::compute(&dema_config, &input).unwrap();
    let tema_owned = IndicatorConfig::compute(&tema_config, &input).unwrap();
    assert_eq!(dema_owned.range(), OutputRange::new(4, 3));
    assert_eq!(tema_owned.range(), OutputRange::new(6, 1));
    assert_float_slice_close(dema_owned.values(), &[5.0, 6.0, 7.0]);
    assert_float_slice_close(tema_owned.values(), &[7.0]);

    let mut dema_output = [FLOAT_SENTINEL; 4];
    let mut tema_output = [FLOAT_SENTINEL; 2];
    let mut dema_runner = IndicatorConfig::prepare_batch(&dema_config, input.len()).unwrap();
    let mut tema_runner = IndicatorConfig::prepare_batch(&tema_config, input.len()).unwrap();
    PreparedBatchRunner::<DEMAConfig>::compute_into(&mut dema_runner, &input, &mut dema_output)
        .unwrap();
    PreparedBatchRunner::<TEMAConfig>::compute_into(&mut tema_runner, &input, &mut tema_output)
        .unwrap();
    assert_float_slice_close(&dema_output[..3], dema_owned.values());
    assert_float_slice_close(&tema_output[..1], tema_owned.values());
    assert_eq!(dema_output[3], FLOAT_SENTINEL);
    assert_eq!(tema_output[1], FLOAT_SENTINEL);

    let mut dema_stream = IndicatorConfig::stream(&dema_config).unwrap();
    let mut tema_stream = IndicatorConfig::stream(&tema_config).unwrap();
    let mut streamed_dema = Vec::new();
    let mut streamed_tema = Vec::new();
    for &tick in &input {
        let dema_value = StreamingComputation::<DEMAConfig>::next(&mut dema_stream, tick).unwrap();
        let tema_value = StreamingComputation::<TEMAConfig>::next(&mut tema_stream, tick).unwrap();
        streamed_dema.extend(dema_value);
        streamed_tema.extend(tema_value);
    }
    assert_float_slice_close(&streamed_dema, dema_owned.values());
    assert_float_slice_close(&streamed_tema, tema_owned.values());

    StreamingComputation::<DEMAConfig>::reset(&mut dema_stream);
    StreamingComputation::<TEMAConfig>::reset(&mut tema_stream);
    assert_eq!(
        StreamingComputation::<DEMAConfig>::next(&mut dema_stream, 1.0).unwrap(),
        None
    );
    assert!(StreamingComputation::<TEMAConfig>::next(&mut tema_stream, Float::INFINITY).is_err());
    assert_eq!(
        StreamingComputation::<TEMAConfig>::next(&mut tema_stream, 1.0).unwrap(),
        None
    );
    StreamingComputation::<DEMAConfig>::reset(&mut dema_stream);
    StreamingComputation::<TEMAConfig>::reset(&mut tema_stream);
    let replayed_dema = input
        .iter()
        .filter_map(|&tick| {
            StreamingComputation::<DEMAConfig>::next(&mut dema_stream, tick).unwrap()
        })
        .collect::<Vec<_>>();
    let replayed_tema = input
        .iter()
        .filter_map(|&tick| {
            StreamingComputation::<TEMAConfig>::next(&mut tema_stream, tick).unwrap()
        })
        .collect::<Vec<_>>();
    assert_float_slice_close(&replayed_dema, dema_owned.values());
    assert_float_slice_close(&replayed_tema, tema_owned.values());
}

#[test]
fn t3_and_ma_configs_preserve_parameters_dispatch_and_execution_modes() {
    let input = [1.0 as Float, 2.0, 3.0, 4.0];
    let t3_config = T3Config::new(1, 0.5 as Float).unwrap();
    let default_t3 = T3Config::with_default_vfactor(1).unwrap();
    assert_eq!(t3_config.period(), 1);
    assert_float_close(t3_config.vfactor(), 0.5);
    assert_float_close(default_t3.vfactor(), T3_DEFAULT_VFACTOR);
    assert_eq!(IndicatorConfig::lookback(&t3_config), 0);

    let t3_owned = IndicatorConfig::compute(&t3_config, &input).unwrap();
    assert_eq!(t3_owned.range(), OutputRange::new(0, input.len()));
    assert_float_slice_close(t3_owned.values(), &input);
    let mut t3_output = [FLOAT_SENTINEL; 5];
    let mut t3_runner = IndicatorConfig::prepare_batch(&t3_config, input.len()).unwrap();
    PreparedBatchRunner::<T3Config>::compute_into(&mut t3_runner, &input, &mut t3_output).unwrap();
    assert_float_slice_close(&t3_output[..input.len()], &input);
    assert_eq!(t3_output[input.len()], FLOAT_SENTINEL);

    let mut t3_stream = IndicatorConfig::stream(&t3_config).unwrap();
    for &tick in &input {
        let value = StreamingComputation::<T3Config>::next(&mut t3_stream, tick).unwrap();
        assert_some_float_close(value, tick);
    }
    StreamingComputation::<T3Config>::reset(&mut t3_stream);
    assert!(StreamingComputation::<T3Config>::next(&mut t3_stream, Float::NAN).is_err());
    let recursive_input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let recursive_t3 = T3Config::with_default_vfactor(2).unwrap();
    assert_eq!(IndicatorConfig::lookback(&recursive_t3), 6);
    let recursive_batch = IndicatorConfig::compute(&recursive_t3, &recursive_input).unwrap();
    let mut recursive_stream = IndicatorConfig::stream(&recursive_t3).unwrap();
    let recursive_values = recursive_input
        .iter()
        .filter_map(|&tick| {
            StreamingComputation::<T3Config>::next(&mut recursive_stream, tick).unwrap()
        })
        .collect::<Vec<_>>();
    assert_float_slice_close(&recursive_values, recursive_batch.values());
    StreamingComputation::<T3Config>::reset(&mut recursive_stream);
    let replayed_recursive = recursive_input
        .iter()
        .filter_map(|&tick| {
            StreamingComputation::<T3Config>::next(&mut recursive_stream, tick).unwrap()
        })
        .collect::<Vec<_>>();
    assert_float_slice_close(&replayed_recursive, recursive_batch.values());

    const SUPPORTED: [PeriodMAType; 8] = [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
        PeriodMAType::T3,
        PeriodMAType::KAMA,
    ];
    for ma_type in SUPPORTED {
        let config = MAConfig::new(1, ma_type).unwrap();
        assert_eq!(config.period(), 1);
        assert_eq!(config.ma_type(), ma_type);
        assert_eq!(IndicatorConfig::lookback(&config), 0);
        let owned = IndicatorConfig::compute(&config, &input).unwrap();
        assert_eq!(owned.range(), OutputRange::new(0, input.len()));
        assert_float_slice_close(owned.values(), &input);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        for &tick in &input {
            let value = StreamingComputation::<MAConfig>::next(&mut stream, tick).unwrap();
            assert_some_float_close(value, tick);
        }
        StreamingComputation::<MAConfig>::reset(&mut stream);
        let replayed = input
            .iter()
            .filter_map(|&tick| StreamingComputation::<MAConfig>::next(&mut stream, tick).unwrap())
            .collect::<Vec<_>>();
        assert_float_slice_close(&replayed, owned.values());
    }

    let ema_dispatch = MAConfig::new(3, PeriodMAType::EMA).unwrap();
    assert_eq!(IndicatorConfig::lookback(&ema_dispatch), 2);
    let mut ma_runner = IndicatorConfig::prepare_batch(&ema_dispatch, input.len()).unwrap();
    let mut ma_output = [FLOAT_SENTINEL; 3];
    PreparedBatchRunner::<MAConfig>::compute_into(&mut ma_runner, &input, &mut ma_output).unwrap();
    assert_float_slice_close(&ma_output[..2], &[2.0, 3.0]);
    assert_eq!(ma_output[2], FLOAT_SENTINEL);
}

#[test]
fn recursive_configs_reject_non_finite_ema_intermediates() {
    let input = [Float::MAX; 7];
    let mut output = [FLOAT_SENTINEL; 7];
    let dema_config = DEMAConfig::new(2).unwrap();
    let tema_config = TEMAConfig::new(2).unwrap();
    let t3_config = T3Config::with_default_vfactor(2).unwrap();
    let ma_config = MAConfig::new(2, PeriodMAType::T3).unwrap();

    assert!(IndicatorConfig::compute_into(&dema_config, &input, &mut output).is_err());
    assert!(IndicatorConfig::compute_into(&tema_config, &input, &mut output).is_err());
    assert!(IndicatorConfig::compute_into(&t3_config, &input, &mut output).is_err());
    assert!(IndicatorConfig::compute_into(&ma_config, &input, &mut output).is_err());

    let mut dema_stream = IndicatorConfig::stream(&dema_config).unwrap();
    let mut tema_stream = IndicatorConfig::stream(&tema_config).unwrap();
    let mut t3_stream = IndicatorConfig::stream(&t3_config).unwrap();
    let mut ma_stream = IndicatorConfig::stream(&ma_config).unwrap();
    assert_eq!(
        StreamingComputation::<DEMAConfig>::next(&mut dema_stream, Float::MAX).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<TEMAConfig>::next(&mut tema_stream, Float::MAX).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<T3Config>::next(&mut t3_stream, Float::MAX).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<MAConfig>::next(&mut ma_stream, Float::MAX).unwrap(),
        None
    );
    assert!(StreamingComputation::<DEMAConfig>::next(&mut dema_stream, Float::MAX).is_err());
    assert!(StreamingComputation::<TEMAConfig>::next(&mut tema_stream, Float::MAX).is_err());
    assert!(StreamingComputation::<T3Config>::next(&mut t3_stream, Float::MAX).is_err());
    assert!(StreamingComputation::<MAConfig>::next(&mut ma_stream, Float::MAX).is_err());
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
    let mut streamed_wma = Vec::new();
    let mut streamed_trima = Vec::new();

    for &tick in &input {
        let wma_value = StreamingComputation::<WMAConfig>::next(&mut wma_stream, tick).unwrap();
        let trima_value =
            StreamingComputation::<TRIMAConfig>::next(&mut trima_stream, tick).unwrap();
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
        .filter_map(|&tick| StreamingComputation::<WMAConfig>::next(&mut wma_stream, tick).unwrap())
        .collect::<Vec<_>>();
    let replayed_trima = input
        .iter()
        .filter_map(|&tick| {
            StreamingComputation::<TRIMAConfig>::next(&mut trima_stream, tick).unwrap()
        })
        .collect::<Vec<_>>();

    assert_float_slice_close(&replayed_wma, batch_wma.values());
    assert_float_slice_close(&replayed_trima, batch_trima.values());
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
        replayed_min.extend(min_value);
        replayed_max.extend(max_value);
        replayed_min_index.extend(min_index_value);
        replayed_max_index.extend(max_index_value);
    }
    assert_float_slice_close(&replayed_min, batch_min.values());
    assert_float_slice_close(&replayed_max, batch_max.values());
    assert_eq!(replayed_min_index, batch_min_index.values().as_slice());
    assert_eq!(replayed_max_index, batch_max_index.values().as_slice());
}

#[test]
fn named_price_configs_cover_owned_and_caller_owned_compact_execution() {
    let open = [1.0 as Float, 2.0, 3.0];
    let high = [2.0 as Float, 4.0, 6.0];
    let low = [0.0 as Float, 1.0, 2.0];
    let close = [1.0 as Float, 3.0, 5.0];
    let avg_input = AVGPRICEInput {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
    };
    let med_input = MEDPRICEInput {
        high: &high,
        low: &low,
    };
    let typ_input = TYPPRICEInput {
        high: &high,
        low: &low,
        close: &close,
    };
    let wcl_input = WCLPRICEInput {
        high: &high,
        low: &low,
        close: &close,
    };

    let avg_config = AVGPRICEConfig::new();
    let med_config = MEDPRICEConfig::new();
    let typ_config = TYPPRICEConfig::new();
    let wcl_config = WCLPRICEConfig::new();
    let avg_owned = IndicatorConfig::compute(&avg_config, avg_input).unwrap();
    let med_owned = IndicatorConfig::compute(&med_config, med_input).unwrap();
    let typ_owned = IndicatorConfig::compute(&typ_config, typ_input).unwrap();
    let wcl_owned = IndicatorConfig::compute(&wcl_config, wcl_input).unwrap();

    for output in [&avg_owned, &med_owned, &typ_owned, &wcl_owned] {
        assert_eq!(output.source_len(), 3);
        assert_eq!(output.range(), OutputRange::new(0, 3));
    }
    assert_float_slice_close(avg_owned.values(), &[1.0, 2.5, 4.0]);
    assert_float_slice_close(med_owned.values(), &[1.0, 2.5, 4.0]);
    assert_float_slice_close(typ_owned.values(), &[1.0, 8.0 / 3.0, 13.0 / 3.0]);
    assert_float_slice_close(wcl_owned.values(), &[1.0, 2.75, 4.5]);

    let mut avg_output = [FLOAT_SENTINEL; 4];
    let mut med_output = [FLOAT_SENTINEL; 4];
    let mut typ_output = [FLOAT_SENTINEL; 4];
    let mut wcl_output = [FLOAT_SENTINEL; 4];
    assert_eq!(
        IndicatorConfig::compute_into(&avg_config, avg_input, &mut avg_output).unwrap(),
        avg_owned.range()
    );
    assert_eq!(
        IndicatorConfig::compute_into(&med_config, med_input, &mut med_output).unwrap(),
        med_owned.range()
    );
    assert_eq!(
        IndicatorConfig::compute_into(&typ_config, typ_input, &mut typ_output).unwrap(),
        typ_owned.range()
    );
    assert_eq!(
        IndicatorConfig::compute_into(&wcl_config, wcl_input, &mut wcl_output).unwrap(),
        wcl_owned.range()
    );
    assert_float_slice_close(&avg_output[..3], avg_owned.values());
    assert_float_slice_close(&med_output[..3], med_owned.values());
    assert_float_slice_close(&typ_output[..3], typ_owned.values());
    assert_float_slice_close(&wcl_output[..3], wcl_owned.values());
    assert_eq!(
        [avg_output[3], med_output[3], typ_output[3], wcl_output[3]],
        [FLOAT_SENTINEL; 4]
    );
}

#[test]
fn named_price_configs_cover_prepared_and_streaming_execution() {
    let open = [1.0 as Float, 2.0, 3.0];
    let high = [2.0 as Float, 4.0, 6.0];
    let low = [0.0 as Float, 1.0, 2.0];
    let close = [1.0 as Float, 3.0, 5.0];
    let avg_config = AVGPRICEConfig::new();
    let med_config = MEDPRICEConfig::new();
    let typ_config = TYPPRICEConfig::new();
    let wcl_config = WCLPRICEConfig::new();
    let mut avg_runner = IndicatorConfig::prepare_batch(&avg_config, 3).unwrap();
    let mut med_runner = IndicatorConfig::prepare_batch(&med_config, 3).unwrap();
    let mut typ_runner = IndicatorConfig::prepare_batch(&typ_config, 3).unwrap();
    let mut wcl_runner = IndicatorConfig::prepare_batch(&wcl_config, 3).unwrap();
    let mut avg_output = [FLOAT_SENTINEL; 3];
    let mut med_output = [FLOAT_SENTINEL; 3];
    let mut typ_output = [FLOAT_SENTINEL; 3];
    let mut wcl_output = [FLOAT_SENTINEL; 3];

    PreparedBatchRunner::<AVGPRICEConfig>::compute_into(
        &mut avg_runner,
        AVGPRICEInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        },
        &mut avg_output,
    )
    .unwrap();
    PreparedBatchRunner::<MEDPRICEConfig>::compute_into(
        &mut med_runner,
        MEDPRICEInput {
            high: &high,
            low: &low,
        },
        &mut med_output,
    )
    .unwrap();
    PreparedBatchRunner::<TYPPRICEConfig>::compute_into(
        &mut typ_runner,
        TYPPRICEInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut typ_output,
    )
    .unwrap();
    PreparedBatchRunner::<WCLPRICEConfig>::compute_into(
        &mut wcl_runner,
        WCLPRICEInput {
            high: &high,
            low: &low,
            close: &close,
        },
        &mut wcl_output,
    )
    .unwrap();
    assert_float_slice_close(&avg_output, &[1.0, 2.5, 4.0]);
    assert_float_slice_close(&med_output, &[1.0, 2.5, 4.0]);
    assert_float_slice_close(&typ_output, &[1.0, 8.0 / 3.0, 13.0 / 3.0]);
    assert_float_slice_close(&wcl_output, &[1.0, 2.75, 4.5]);

    let mut avg_stream = IndicatorConfig::stream(&avg_config).unwrap();
    let mut med_stream = IndicatorConfig::stream(&med_config).unwrap();
    let mut typ_stream = IndicatorConfig::stream(&typ_config).unwrap();
    let mut wcl_stream = IndicatorConfig::stream(&wcl_config).unwrap();
    for idx in 0..3 {
        assert_float_close(
            StreamingComputation::<AVGPRICEConfig>::next(
                &mut avg_stream,
                AVGPRICETick {
                    open: open[idx],
                    high: high[idx],
                    low: low[idx],
                    close: close[idx],
                },
            )
            .unwrap()
            .unwrap(),
            avg_output[idx],
        );
        assert_float_close(
            StreamingComputation::<MEDPRICEConfig>::next(
                &mut med_stream,
                MEDPRICETick {
                    high: high[idx],
                    low: low[idx],
                },
            )
            .unwrap()
            .unwrap(),
            med_output[idx],
        );
        assert_float_close(
            StreamingComputation::<TYPPRICEConfig>::next(
                &mut typ_stream,
                TYPPRICETick {
                    high: high[idx],
                    low: low[idx],
                    close: close[idx],
                },
            )
            .unwrap()
            .unwrap(),
            typ_output[idx],
        );
        assert_float_close(
            StreamingComputation::<WCLPRICEConfig>::next(
                &mut wcl_stream,
                WCLPRICETick {
                    high: high[idx],
                    low: low[idx],
                    close: close[idx],
                },
            )
            .unwrap()
            .unwrap(),
            wcl_output[idx],
        );
    }
}

#[test]
fn named_price_streams_match_legacy_reject_ticks_reset_replay_and_are_independent() {
    let open = [1.0 as Float, 2.0, 3.0];
    let high = [2.0 as Float, 4.0, 6.0];
    let low = [0.0 as Float, 1.0, 2.0];
    let close = [1.0 as Float, 3.0, 5.0];
    let avg_config = AVGPRICEConfig::new();
    let med_config = MEDPRICEConfig::new();
    let typ_config = TYPPRICEConfig::new();
    let wcl_config = WCLPRICEConfig::new();
    let mut avg_stream = IndicatorConfig::stream(&avg_config).unwrap();
    let mut med_stream = IndicatorConfig::stream(&med_config).unwrap();
    let mut typ_stream = IndicatorConfig::stream(&typ_config).unwrap();
    let mut wcl_stream = IndicatorConfig::stream(&wcl_config).unwrap();

    macro_rules! assert_tick_value {
        ($config_ty:ty, $stream:expr, $tick:expr) => {{
            let tick = $tick;
            let configured = StreamingComputation::<$config_ty>::next($stream, tick)
                .unwrap()
                .unwrap();
            configured
        }};
    }

    for idx in 0..open.len() {
        let _ = assert_tick_value!(
            AVGPRICEConfig,
            &mut avg_stream,
            AVGPRICETick {
                open: open[idx],
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
        let _ = assert_tick_value!(
            MEDPRICEConfig,
            &mut med_stream,
            MEDPRICETick {
                high: high[idx],
                low: low[idx],
            }
        );
        let _ = assert_tick_value!(
            TYPPRICEConfig,
            &mut typ_stream,
            TYPPRICETick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
        let _ = assert_tick_value!(
            WCLPRICEConfig,
            &mut wcl_stream,
            WCLPRICETick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
    }

    assert!(StreamingComputation::<AVGPRICEConfig>::next(
        &mut avg_stream,
        AVGPRICETick {
            open: Float::NAN,
            high: 1.0,
            low: 1.0,
            close: 1.0,
        },
    )
    .is_err());
    assert!(StreamingComputation::<MEDPRICEConfig>::next(
        &mut med_stream,
        MEDPRICETick {
            high: 1.0,
            low: Float::INFINITY,
        },
    )
    .is_err());
    assert!(StreamingComputation::<TYPPRICEConfig>::next(
        &mut typ_stream,
        TYPPRICETick {
            high: 1.0,
            low: 1.0,
            close: Float::NAN,
        },
    )
    .is_err());
    assert!(StreamingComputation::<WCLPRICEConfig>::next(
        &mut wcl_stream,
        WCLPRICETick {
            high: Float::INFINITY,
            low: 1.0,
            close: 1.0,
        },
    )
    .is_err());

    assert_some_float_close(
        StreamingComputation::<AVGPRICEConfig>::next(
            &mut avg_stream,
            AVGPRICETick {
                open: 2.0,
                high: 2.0,
                low: 2.0,
                close: 2.0,
            },
        )
        .unwrap(),
        2.0,
    );
    assert_some_float_close(
        StreamingComputation::<MEDPRICEConfig>::next(
            &mut med_stream,
            MEDPRICETick {
                high: 2.0,
                low: 2.0,
            },
        )
        .unwrap(),
        2.0,
    );
    assert_some_float_close(
        StreamingComputation::<TYPPRICEConfig>::next(
            &mut typ_stream,
            TYPPRICETick {
                high: 2.0,
                low: 2.0,
                close: 2.0,
            },
        )
        .unwrap(),
        2.0,
    );
    assert_some_float_close(
        StreamingComputation::<WCLPRICEConfig>::next(
            &mut wcl_stream,
            WCLPRICETick {
                high: 2.0,
                low: 2.0,
                close: 2.0,
            },
        )
        .unwrap(),
        2.0,
    );

    StreamingComputation::<AVGPRICEConfig>::reset(&mut avg_stream);
    StreamingComputation::<MEDPRICEConfig>::reset(&mut med_stream);
    StreamingComputation::<TYPPRICEConfig>::reset(&mut typ_stream);
    StreamingComputation::<WCLPRICEConfig>::reset(&mut wcl_stream);
    for idx in 0..open.len() {
        let _ = assert_tick_value!(
            AVGPRICEConfig,
            &mut avg_stream,
            AVGPRICETick {
                open: open[idx],
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
        let _ = assert_tick_value!(
            MEDPRICEConfig,
            &mut med_stream,
            MEDPRICETick {
                high: high[idx],
                low: low[idx],
            }
        );
        let _ = assert_tick_value!(
            TYPPRICEConfig,
            &mut typ_stream,
            TYPPRICETick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
        let _ = assert_tick_value!(
            WCLPRICEConfig,
            &mut wcl_stream,
            WCLPRICETick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            }
        );
    }

    let mut left = IndicatorConfig::stream(&avg_config).unwrap();
    let mut right = IndicatorConfig::stream(&avg_config).unwrap();
    assert_some_float_close(
        StreamingComputation::<AVGPRICEConfig>::next(
            &mut left,
            AVGPRICETick {
                open: 1.0,
                high: 1.0,
                low: 1.0,
                close: 1.0,
            },
        )
        .unwrap(),
        1.0,
    );
    assert_some_float_close(
        StreamingComputation::<AVGPRICEConfig>::next(
            &mut right,
            AVGPRICETick {
                open: 10.0,
                high: 10.0,
                low: 10.0,
                close: 10.0,
            },
        )
        .unwrap(),
        10.0,
    );
    assert_some_float_close(
        StreamingComputation::<AVGPRICEConfig>::next(
            &mut left,
            AVGPRICETick {
                open: 3.0,
                high: 3.0,
                low: 3.0,
                close: 3.0,
            },
        )
        .unwrap(),
        3.0,
    );
}

#[test]
fn named_price_prepared_runners_reuse_shorter_series_and_reject_oversize() {
    let avg_config = AVGPRICEConfig::new();
    let med_config = MEDPRICEConfig::new();
    let typ_config = TYPPRICEConfig::new();
    let wcl_config = WCLPRICEConfig::new();
    let mut avg_runner = IndicatorConfig::prepare_batch(&avg_config, 3).unwrap();
    let mut med_runner = IndicatorConfig::prepare_batch(&med_config, 3).unwrap();
    let mut typ_runner = IndicatorConfig::prepare_batch(&typ_config, 3).unwrap();
    let mut wcl_runner = IndicatorConfig::prepare_batch(&wcl_config, 3).unwrap();
    let open = [8.0 as Float, 4.0];
    let high = [10.0 as Float, 6.0];
    let low = [4.0 as Float, 2.0];
    let close = [6.0 as Float, 3.0];

    assert_eq!(
        PreparedBatchRunner::<AVGPRICEConfig>::max_input_len(&avg_runner),
        3
    );
    assert_eq!(
        PreparedBatchRunner::<MEDPRICEConfig>::max_input_len(&med_runner),
        3
    );
    assert_eq!(
        PreparedBatchRunner::<TYPPRICEConfig>::max_input_len(&typ_runner),
        3
    );
    assert_eq!(
        PreparedBatchRunner::<WCLPRICEConfig>::max_input_len(&wcl_runner),
        3
    );

    macro_rules! assert_reuse {
        ($config_ty:ty, $config:expr, $runner:expr, $input:expr) => {{
            let input = $input;
            let owned = IndicatorConfig::compute($config, input).unwrap();
            let mut output = [FLOAT_SENTINEL; 3];
            let range =
                PreparedBatchRunner::<$config_ty>::compute_into($runner, input, &mut output)
                    .unwrap();
            assert_eq!(range, owned.range());
            assert_float_slice_close(&output[..owned.values().len()], owned.values());
            assert_eq!(output[owned.values().len()], FLOAT_SENTINEL);
        }};
    }

    assert_reuse!(
        AVGPRICEConfig,
        &avg_config,
        &mut avg_runner,
        AVGPRICEInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        }
    );
    assert_reuse!(
        MEDPRICEConfig,
        &med_config,
        &mut med_runner,
        MEDPRICEInput {
            high: &high,
            low: &low,
        }
    );
    assert_reuse!(
        TYPPRICEConfig,
        &typ_config,
        &mut typ_runner,
        TYPPRICEInput {
            high: &high,
            low: &low,
            close: &close,
        }
    );
    assert_reuse!(
        WCLPRICEConfig,
        &wcl_config,
        &mut wcl_runner,
        WCLPRICEInput {
            high: &high,
            low: &low,
            close: &close,
        }
    );

    let oversized = [Float::NAN; 4];
    let expected = TalibError::PreparedCapacityExceeded {
        max_input_len: 3,
        actual_input_len: 4,
    };
    macro_rules! assert_oversize {
        ($config_ty:ty, $runner:expr, $input:expr) => {{
            let mut output = [FLOAT_SENTINEL; 4];
            let error =
                PreparedBatchRunner::<$config_ty>::compute_into($runner, $input, &mut output)
                    .unwrap_err();
            assert_eq!(error, expected);
            assert_eq!(output, [FLOAT_SENTINEL; 4]);
        }};
    }

    assert_oversize!(
        AVGPRICEConfig,
        &mut avg_runner,
        AVGPRICEInput {
            open: &oversized,
            high: &oversized,
            low: &oversized,
            close: &oversized,
        }
    );
    assert_oversize!(
        MEDPRICEConfig,
        &mut med_runner,
        MEDPRICEInput {
            high: &oversized,
            low: &oversized,
        }
    );
    assert_oversize!(
        TYPPRICEConfig,
        &mut typ_runner,
        TYPPRICEInput {
            high: &oversized,
            low: &oversized,
            close: &oversized,
        }
    );
    assert_oversize!(
        WCLPRICEConfig,
        &mut wcl_runner,
        WCLPRICEInput {
            high: &oversized,
            low: &oversized,
            close: &oversized,
        }
    );
}

#[test]
fn named_price_configs_preserve_length_finite_and_output_validation_order() {
    let valid = [1.0 as Float, 2.0];
    let short = [1.0 as Float];
    let invalid = [1.0 as Float, Float::NAN];
    let mut output = [FLOAT_SENTINEL; 2];
    let avg_config = AVGPRICEConfig::new();
    let med_config = MEDPRICEConfig::new();
    let typ_config = TYPPRICEConfig::new();
    let wcl_config = WCLPRICEConfig::new();

    macro_rules! assert_unchanged_error {
        ($call:expr, $message:literal) => {{
            assert_eq!($call.unwrap_err().to_string(), $message);
            assert_eq!(output, [FLOAT_SENTINEL; 2]);
        }};
    }

    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &avg_config,
            AVGPRICEInput {
                open: &valid,
                high: &short,
                low: &invalid,
                close: &valid,
            },
            &mut output,
        ),
        "Invalid input: open and high must have the same length: got 2 and 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &med_config,
            MEDPRICEInput {
                high: &valid,
                low: &short,
            },
            &mut output,
        ),
        "Invalid input: high and low must have the same length: got 2 and 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &typ_config,
            TYPPRICEInput {
                high: &valid,
                low: &short,
                close: &invalid,
            },
            &mut output,
        ),
        "Invalid input: high and low must have the same length: got 2 and 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &wcl_config,
            WCLPRICEInput {
                high: &valid,
                low: &valid,
                close: &short,
            },
            &mut output,
        ),
        "Invalid input: high and close must have the same length: got 2 and 1"
    );

    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &avg_config,
            AVGPRICEInput {
                open: &valid,
                high: &valid,
                low: &invalid,
                close: &valid,
            },
            &mut output[..0],
        ),
        "Invalid input: low[1] must be finite, got NaN"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &med_config,
            MEDPRICEInput {
                high: &valid,
                low: &invalid,
            },
            &mut output[..0],
        ),
        "Invalid input: low[1] must be finite, got NaN"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &typ_config,
            TYPPRICEInput {
                high: &valid,
                low: &valid,
                close: &invalid,
            },
            &mut output[..0],
        ),
        "Invalid input: close[1] must be finite, got NaN"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &wcl_config,
            WCLPRICEInput {
                high: &valid,
                low: &valid,
                close: &invalid,
            },
            &mut output[..0],
        ),
        "Invalid input: close[1] must be finite, got NaN"
    );

    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &avg_config,
            AVGPRICEInput {
                open: &valid,
                high: &valid,
                low: &valid,
                close: &valid,
            },
            &mut output[..1],
        ),
        "Invalid input: AVGPRICE output buffer too small: need 2, got 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &med_config,
            MEDPRICEInput {
                high: &valid,
                low: &valid,
            },
            &mut output[..1],
        ),
        "Invalid input: MEDPRICE output buffer too small: need 2, got 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &typ_config,
            TYPPRICEInput {
                high: &valid,
                low: &valid,
                close: &valid,
            },
            &mut output[..1],
        ),
        "Invalid input: TYPPRICE output buffer too small: need 2, got 1"
    );
    assert_unchanged_error!(
        IndicatorConfig::compute_into(
            &wcl_config,
            WCLPRICEInput {
                high: &valid,
                low: &valid,
                close: &valid,
            },
            &mut output[..1],
        ),
        "Invalid input: WCLPRICE output buffer too small: need 2, got 1"
    );
}

#[test]
fn avgdev_config_covers_reusable_prepared_validated_and_independent_stream_execution() {
    let input = [1.0 as Float, 2.0, 4.0, 8.0];
    let config = AVGDEVConfig::new(2).unwrap();
    assert_eq!(config.period(), 2);
    assert_eq!(IndicatorConfig::lookback(&config), 1);

    let owned = IndicatorConfig::compute(&config, &input).unwrap();
    assert_eq!(owned.source_len(), input.len());
    assert_eq!(owned.range(), OutputRange::new(1, 3));
    assert_float_slice_close(owned.values(), &[0.5, 1.0, 2.0]);

    let mut output = [FLOAT_SENTINEL; 4];
    let range = IndicatorConfig::compute_into(&config, &input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(output[3], FLOAT_SENTINEL);

    let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<AVGDEVConfig>::max_input_len(&runner),
        input.len()
    );
    output.fill(FLOAT_SENTINEL);
    let prepared_range =
        PreparedBatchRunner::<AVGDEVConfig>::compute_into(&mut runner, &input, &mut output)
            .unwrap();
    assert_eq!(prepared_range, owned.range());
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(output[3], FLOAT_SENTINEL);

    let shorter = [2.0 as Float, 6.0, 10.0];
    let shorter_owned = IndicatorConfig::compute(&config, &shorter).unwrap();
    output.fill(FLOAT_SENTINEL);
    let shorter_range =
        PreparedBatchRunner::<AVGDEVConfig>::compute_into(&mut runner, &shorter, &mut output)
            .unwrap();
    assert_eq!(shorter_range, shorter_owned.range());
    assert_float_slice_close(&output[..2], shorter_owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 2]);

    let oversized = [Float::NAN; 5];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<AVGDEVConfig>::compute_into(&mut runner, &oversized, &mut output)
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 4,
            actual_input_len: 5,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 4]);

    let invalid = [1.0 as Float, Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(&config, &invalid, &mut output[..0])
            .unwrap_err()
            .to_string(),
        "Invalid input: real[1] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 4]);
    assert_eq!(
        IndicatorConfig::compute_into(&config, &[1.0 as Float, 2.0], &mut output[..0])
            .unwrap_err()
            .to_string(),
        "Invalid input: AVGDEV output buffer too small: need 1, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 4]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let mut streamed = Vec::new();
    for tick in input {
        if let Some(value) = StreamingComputation::<AVGDEVConfig>::next(&mut stream, tick).unwrap()
        {
            streamed.push(value);
        }
    }
    assert_float_slice_close(&streamed, owned.values());

    assert!(StreamingComputation::<AVGDEVConfig>::next(&mut stream, Float::NAN).is_err());
    let configured_after_rejection = StreamingComputation::<AVGDEVConfig>::next(&mut stream, 10.0)
        .unwrap()
        .unwrap();
    assert_float_close(configured_after_rejection, 1.0);

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    assert_eq!(
        StreamingComputation::<AVGDEVConfig>::next(&mut left, 10.0).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<AVGDEVConfig>::next(&mut right, 100.0).unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<AVGDEVConfig>::next(&mut left, 20.0).unwrap(),
        5.0,
    );
    assert_some_float_close(
        StreamingComputation::<AVGDEVConfig>::next(&mut right, 200.0).unwrap(),
        50.0,
    );
    assert_some_float_close(
        StreamingComputation::<AVGDEVConfig>::next(&mut left, 30.0).unwrap(),
        5.0,
    );

    StreamingComputation::<AVGDEVConfig>::reset(&mut stream);
    let replayed = input
        .iter()
        .filter_map(|&tick| StreamingComputation::<AVGDEVConfig>::next(&mut stream, tick).unwrap())
        .collect::<Vec<_>>();
    assert_float_slice_close(&replayed, owned.values());

    assert_eq!(
        core::mem::size_of::<AVGDEVConfig>(),
        core::mem::size_of::<usize>()
    );
}

#[test]
fn ad_config_covers_owned_caller_prepared_validation_and_independent_streams() {
    let high = [2.0 as Float; 4];
    let low = [0.0 as Float; 4];
    let close = [2.0 as Float; 4];
    let volume = [1.0 as Float, 2.0, 3.0, 4.0];
    let input = ADInput {
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };
    let config = ADConfig::new();
    assert_eq!(IndicatorConfig::lookback(&config), 0);

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(0, 4));
    assert_float_slice_close(owned.values(), &[1.0, 3.0, 6.0, 10.0]);

    let mut output = [FLOAT_SENTINEL; 5];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..4], owned.values());
    assert_eq!(output[4], FLOAT_SENTINEL);

    let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<ADConfig>::max_input_len(&runner),
        high.len()
    );
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<ADConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..4], owned.values());
    assert_eq!(output[4], FLOAT_SENTINEL);

    let second_high = [4.0 as Float; 3];
    let second_low = [0.0 as Float; 3];
    let second_close = [0.0 as Float; 3];
    let second_volume = [2.0 as Float, 4.0, 6.0];
    let second_input = ADInput {
        high: &second_high,
        low: &second_low,
        close: &second_close,
        volume: &second_volume,
    };
    let second_owned = IndicatorConfig::compute(&config, second_input).unwrap();
    output.fill(FLOAT_SENTINEL);
    let second_range =
        PreparedBatchRunner::<ADConfig>::compute_into(&mut runner, second_input, &mut output)
            .unwrap();
    assert_eq!(second_range, second_owned.range());
    assert_float_slice_close(&output[..3], second_owned.values());
    assert_eq!(&output[3..], &[FLOAT_SENTINEL; 2]);

    let oversized = [Float::NAN; 5];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<ADConfig>::compute_into(
            &mut runner,
            ADInput {
                high: &oversized,
                low: &oversized,
                close: &oversized,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 4,
            actual_input_len: 5,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 5]);

    let valid = [1.0 as Float, 2.0];
    let short = [1.0 as Float];
    let invalid = [1.0 as Float, Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADInput {
                high: &valid,
                low: &short,
                close: &invalid,
                volume: &valid,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high and low must have the same length: got 2 and 1"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 5]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADInput {
                high: &invalid,
                low: &valid,
                close: &valid,
                volume: &valid,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high[1] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 5]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADInput {
                high: &valid,
                low: &valid,
                close: &valid,
                volume: &valid,
            },
            &mut output[..1],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: AD output buffer too small: need 2, got 1"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 5]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..high.len() {
        let tick = ADTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
            volume: volume[idx],
        };
        let configured = StreamingComputation::<ADConfig>::next(&mut stream, tick)
            .unwrap()
            .unwrap();
        assert_float_close(configured, owned.values()[idx]);
    }

    let invalid_tick = ADTick {
        high: 2.0,
        low: 0.0,
        close: 2.0,
        volume: Float::NAN,
    };
    assert!(StreamingComputation::<ADConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = ADTick {
        high: 2.0,
        low: 0.0,
        close: 2.0,
        volume: 1.0,
    };
    assert_some_float_close(
        StreamingComputation::<ADConfig>::next(&mut stream, next_tick).unwrap(),
        11.0,
    );

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    assert_some_float_close(
        StreamingComputation::<ADConfig>::next(
            &mut left,
            ADTick {
                volume: 1.0,
                ..next_tick
            },
        )
        .unwrap(),
        1.0,
    );
    assert_some_float_close(
        StreamingComputation::<ADConfig>::next(
            &mut right,
            ADTick {
                volume: 10.0,
                ..next_tick
            },
        )
        .unwrap(),
        10.0,
    );
    assert_some_float_close(
        StreamingComputation::<ADConfig>::next(
            &mut left,
            ADTick {
                volume: 2.0,
                ..next_tick
            },
        )
        .unwrap(),
        3.0,
    );

    StreamingComputation::<ADConfig>::reset(&mut stream);
    for idx in 0..high.len() {
        let tick = ADTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
            volume: volume[idx],
        };
        assert_some_float_close(
            StreamingComputation::<ADConfig>::next(&mut stream, tick).unwrap(),
            owned.values()[idx],
        );
    }
}

#[test]
fn adosc_config_covers_owned_caller_prepared_validation_and_independent_streams() {
    let high = [2.0 as Float; 5];
    let low = [0.0 as Float; 5];
    let close = [2.0 as Float; 5];
    let volume = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let input = ADOSCInput {
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };
    let config = ADOSCConfig::new(2, 3).unwrap();
    assert_eq!(config.fastperiod(), 2);
    assert_eq!(config.slowperiod(), 3);
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert!(ADOSCConfig::new(0, 3).is_err());
    assert!(ADOSCConfig::new(3, 3).is_err());

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(2, 3));
    assert_float_slice_close(owned.values(), &[4.0 / 3.0, 14.0 / 9.0, 103.0 / 54.0]);

    let mut output = [FLOAT_SENTINEL; 6];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(&output[3..], &[FLOAT_SENTINEL; 3]);

    let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<ADOSCConfig>::max_input_len(&runner),
        high.len()
    );
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<ADOSCConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..3], owned.values());
    assert_eq!(&output[3..], &[FLOAT_SENTINEL; 3]);

    let second_high = [4.0 as Float; 4];
    let second_low = [0.0 as Float; 4];
    let second_close = [0.0 as Float; 4];
    let second_volume = [2.0 as Float, 4.0, 6.0, 8.0];
    let second_input = ADOSCInput {
        high: &second_high,
        low: &second_low,
        close: &second_close,
        volume: &second_volume,
    };
    let second_owned = IndicatorConfig::compute(&config, second_input).unwrap();
    output.fill(FLOAT_SENTINEL);
    let second_range =
        PreparedBatchRunner::<ADOSCConfig>::compute_into(&mut runner, second_input, &mut output)
            .unwrap();
    assert_eq!(second_range, second_owned.range());
    assert_float_slice_close(&output[..2], second_owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 4]);

    let oversized = [Float::NAN; 6];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<ADOSCConfig>::compute_into(
            &mut runner,
            ADOSCInput {
                high: &oversized,
                low: &oversized,
                close: &oversized,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let valid_three = [1.0 as Float, 2.0, 3.0];
    let valid_two = [1.0 as Float, 2.0];
    let invalid_two = [1.0 as Float, Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADOSCInput {
                high: &valid_three,
                low: &valid_two,
                close: &valid_three,
                volume: &valid_three,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high and low must have the same length: got 3 and 2"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADOSCInput {
                high: &invalid_two,
                low: &valid_two,
                close: &valid_two,
                volume: &valid_two,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high[1] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADOSCInput {
                high: &valid_two,
                low: &valid_two,
                close: &valid_two,
                volume: &valid_two,
            },
            &mut output[..0],
        )
        .unwrap_err(),
        TalibError::InsufficientData {
            required: 3,
            actual: 2,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ADOSCInput {
                high: &valid_three,
                low: &valid_three,
                close: &valid_three,
                volume: &valid_three,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: ADOSC output buffer too small: need 1, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let mut configured_values = Vec::new();
    for idx in 0..high.len() {
        let tick = ADOSCTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
            volume: volume[idx],
        };
        let configured = StreamingComputation::<ADOSCConfig>::next(&mut stream, tick).unwrap();
        if let Some(value) = configured {
            configured_values.push(value);
        }
    }
    assert_float_slice_close(&configured_values, owned.values());

    let invalid_tick = ADOSCTick {
        high: 2.0,
        low: 0.0,
        close: 2.0,
        volume: Float::NAN,
    };
    assert!(StreamingComputation::<ADOSCConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = ADOSCTick {
        volume: 6.0,
        ..invalid_tick
    };
    let _configured_after_rejection =
        StreamingComputation::<ADOSCConfig>::next(&mut stream, next_tick)
            .unwrap()
            .unwrap();

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    for _ in 0..2 {
        assert_eq!(
            StreamingComputation::<ADOSCConfig>::next(
                &mut left,
                ADOSCTick {
                    volume: 1.0,
                    ..next_tick
                },
            )
            .unwrap(),
            None
        );
        assert_eq!(
            StreamingComputation::<ADOSCConfig>::next(
                &mut right,
                ADOSCTick {
                    volume: 10.0,
                    ..next_tick
                },
            )
            .unwrap(),
            None
        );
    }
    assert_some_float_close(
        StreamingComputation::<ADOSCConfig>::next(
            &mut left,
            ADOSCTick {
                volume: 1.0,
                ..next_tick
            },
        )
        .unwrap(),
        0.5,
    );
    assert_some_float_close(
        StreamingComputation::<ADOSCConfig>::next(
            &mut right,
            ADOSCTick {
                volume: 10.0,
                ..next_tick
            },
        )
        .unwrap(),
        5.0,
    );

    StreamingComputation::<ADOSCConfig>::reset(&mut stream);
    let replayed = high
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| {
            StreamingComputation::<ADOSCConfig>::next(
                &mut stream,
                ADOSCTick {
                    high: high[idx],
                    low: low[idx],
                    close: close[idx],
                    volume: volume[idx],
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    assert_float_slice_close(&replayed, owned.values());
}

#[test]
fn obv_config_covers_owned_caller_prepared_validation_and_independent_streams() {
    let close = [10.0 as Float, 12.0, 11.0, 15.0, 15.0];
    let volume = [100.0 as Float, 200.0, 50.0, 300.0, 400.0];
    let input = OBVInput {
        close: &close,
        volume: &volume,
    };
    let config = OBVConfig::new();
    assert_eq!(IndicatorConfig::lookback(&config), 0);

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), close.len());
    assert_eq!(owned.range(), OutputRange::new(0, 5));
    assert_float_slice_close(owned.values(), &[100.0, 300.0, 250.0, 550.0, 550.0]);

    let mut output = [FLOAT_SENTINEL; 6];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..5], owned.values());
    assert_eq!(&output[5..], &[FLOAT_SENTINEL; 1]);

    let mut runner = IndicatorConfig::prepare_batch(&config, close.len()).unwrap();
    assert_eq!(
        PreparedBatchRunner::<OBVConfig>::max_input_len(&runner),
        close.len()
    );
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<OBVConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..5], owned.values());
    assert_eq!(&output[5..], &[FLOAT_SENTINEL; 1]);

    let second_close = [5.0 as Float, 4.0, 6.0, 6.0];
    let second_volume = [9.0 as Float, 3.0, 7.0, 2.0];
    let second_input = OBVInput {
        close: &second_close,
        volume: &second_volume,
    };
    let second_owned = IndicatorConfig::compute(&config, second_input).unwrap();
    output.fill(FLOAT_SENTINEL);
    let second_range =
        PreparedBatchRunner::<OBVConfig>::compute_into(&mut runner, second_input, &mut output)
            .unwrap();
    assert_eq!(second_range, second_owned.range());
    assert_float_slice_close(&output[..4], second_owned.values());
    assert_eq!(&output[4..], &[FLOAT_SENTINEL; 2]);

    let oversized = [Float::NAN; 6];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<OBVConfig>::compute_into(
            &mut runner,
            OBVInput {
                close: &oversized,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let valid_two = [1.0 as Float, 2.0];
    let valid_one = [1.0 as Float];
    let invalid_one = [Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            OBVInput {
                close: &valid_two,
                volume: &valid_one,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: close and volume must have the same length: got 2 and 1"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            OBVInput {
                close: &invalid_one,
                volume: &valid_one,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: close[0] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            OBVInput {
                close: &valid_one,
                volume: &valid_one,
            },
            &mut output[..1],
        )
        .unwrap(),
        OutputRange::new(0, 1)
    );
    assert_float_close(output[0], 1.0);
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            OBVInput {
                close: &valid_two,
                volume: &valid_two,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: OBV output buffer too small: need 2, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let first_tick = OBVTick {
        close: close[0],
        volume: volume[0],
    };
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(&mut stream, first_tick).unwrap(),
        owned.values()[0],
    );
    for idx in 1..close.len() {
        let tick = OBVTick {
            close: close[idx],
            volume: volume[idx],
        };
        let configured = StreamingComputation::<OBVConfig>::next(&mut stream, tick)
            .unwrap()
            .unwrap();
        assert_float_close(configured, owned.values()[idx]);
    }

    let invalid_tick = OBVTick {
        close: Float::NAN,
        volume: 1.0,
    };
    assert!(StreamingComputation::<OBVConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = OBVTick {
        close: 16.0,
        volume: 10.0,
    };
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(&mut stream, next_tick).unwrap(),
        560.0,
    );

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(
            &mut left,
            OBVTick {
                close: 10.0,
                volume: 1.0,
            },
        )
        .unwrap(),
        1.0,
    );
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(
            &mut right,
            OBVTick {
                close: 100.0,
                volume: 10.0,
            },
        )
        .unwrap(),
        10.0,
    );
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(
            &mut left,
            OBVTick {
                close: 11.0,
                volume: 2.0,
            },
        )
        .unwrap(),
        3.0,
    );
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(
            &mut right,
            OBVTick {
                close: 90.0,
                volume: 20.0,
            },
        )
        .unwrap(),
        -10.0,
    );
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(
            &mut left,
            OBVTick {
                close: 10.0,
                volume: 3.0,
            },
        )
        .unwrap(),
        0.0,
    );

    StreamingComputation::<OBVConfig>::reset(&mut stream);
    assert_some_float_close(
        StreamingComputation::<OBVConfig>::next(&mut stream, first_tick).unwrap(),
        owned.values()[0],
    );
    for idx in 1..close.len() {
        let tick = OBVTick {
            close: close[idx],
            volume: volume[idx],
        };
        assert_some_float_close(
            StreamingComputation::<OBVConfig>::next(&mut stream, tick).unwrap(),
            owned.values()[idx],
        );
    }
}

#[test]
fn trange_config_covers_owned_prepared_validation_warmup_and_independent_streams() {
    let high = [10.0 as Float, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0 as Float, 8.0, 9.0, 13.0, 14.0];
    let close = [9.0 as Float, 11.0, 10.0, 14.0, 15.0];
    let input = TRANGEInput {
        high: &high,
        low: &low,
        close: &close,
    };
    let config = TRANGEConfig::new();
    assert_eq!(IndicatorConfig::lookback(&config), 1);

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(1, 4));
    assert_float_slice_close(owned.values(), &[4.0, 2.0, 5.0, 2.0]);

    let mut output = [FLOAT_SENTINEL; 6];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..4], owned.values());
    assert_eq!(&output[4..], &[FLOAT_SENTINEL; 2]);

    let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<TRANGEConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..4], owned.values());
    assert_eq!(&output[4..], &[FLOAT_SENTINEL; 2]);

    let short_input = TRANGEInput {
        high: &high[..4],
        low: &low[..4],
        close: &close[..4],
    };
    let short_owned = IndicatorConfig::compute(&config, short_input).unwrap();
    output.fill(FLOAT_SENTINEL);
    let short_range =
        PreparedBatchRunner::<TRANGEConfig>::compute_into(&mut runner, short_input, &mut output)
            .unwrap();
    assert_eq!(short_range, short_owned.range());
    assert_float_slice_close(&output[..3], short_owned.values());
    assert_eq!(&output[3..], &[FLOAT_SENTINEL; 3]);

    let oversized = [Float::NAN; 6];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<TRANGEConfig>::compute_into(
            &mut runner,
            TRANGEInput {
                high: &oversized,
                low: &oversized,
                close: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let valid_two = [1.0 as Float, 2.0];
    let valid_one = [1.0 as Float];
    let invalid_one = [Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            TRANGEInput {
                high: &valid_two,
                low: &valid_one,
                close: &valid_two,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high and low must have the same length: got 2 and 1"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            TRANGEInput {
                high: &invalid_one,
                low: &valid_one,
                close: &valid_one,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high[0] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            TRANGEInput {
                high: &valid_one,
                low: &valid_one,
                close: &valid_one,
            },
            &mut output[..0],
        )
        .unwrap_err(),
        TalibError::InsufficientData {
            required: 2,
            actual: 1,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            TRANGEInput {
                high: &valid_two,
                low: &valid_two,
                close: &valid_two,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: TRANGE output buffer too small: need 1, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    let first_tick = TRANGETick {
        high: high[0],
        low: low[0],
        close: close[0],
    };
    assert_eq!(
        StreamingComputation::<TRANGEConfig>::next(&mut stream, first_tick).unwrap(),
        None
    );
    for idx in 1..high.len() {
        let tick = TRANGETick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        assert_some_float_close(
            StreamingComputation::<TRANGEConfig>::next(&mut stream, tick).unwrap(),
            owned.values()[idx - 1],
        );
    }

    let invalid_tick = TRANGETick {
        high: Float::NAN,
        low: 0.0,
        close: 1.0,
    };
    assert!(StreamingComputation::<TRANGEConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = TRANGETick {
        high: 18.0,
        low: 16.0,
        close: 17.0,
    };
    assert_some_float_close(
        StreamingComputation::<TRANGEConfig>::next(&mut stream, next_tick).unwrap(),
        3.0,
    );

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    assert_eq!(
        StreamingComputation::<TRANGEConfig>::next(
            &mut left,
            TRANGETick {
                high: 11.0,
                low: 9.0,
                close: 10.0,
            },
        )
        .unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<TRANGEConfig>::next(
            &mut right,
            TRANGETick {
                high: 101.0,
                low: 99.0,
                close: 100.0,
            },
        )
        .unwrap(),
        None
    );
    assert_some_float_close(
        StreamingComputation::<TRANGEConfig>::next(
            &mut left,
            TRANGETick {
                high: 13.0,
                low: 11.0,
                close: 12.0,
            },
        )
        .unwrap(),
        3.0,
    );
    assert_some_float_close(
        StreamingComputation::<TRANGEConfig>::next(
            &mut right,
            TRANGETick {
                high: 95.0,
                low: 93.0,
                close: 94.0,
            },
        )
        .unwrap(),
        7.0,
    );

    StreamingComputation::<TRANGEConfig>::reset(&mut stream);
    assert_eq!(
        StreamingComputation::<TRANGEConfig>::next(&mut stream, first_tick).unwrap(),
        None
    );
    for idx in 1..high.len() {
        let tick = TRANGETick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        assert_some_float_close(
            StreamingComputation::<TRANGEConfig>::next(&mut stream, tick).unwrap(),
            owned.values()[idx - 1],
        );
    }
}

#[test]
fn atr_config_preserves_warmup_period_one_validation_and_recursive_stream_state() {
    let high = [10.0 as Float, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0 as Float, 8.0, 9.0, 13.0, 14.0];
    let close = [9.0 as Float, 11.0, 10.0, 14.0, 15.0];
    let input = ATRInput {
        high: &high,
        low: &low,
        close: &close,
    };
    let config = ATRConfig::new(3).unwrap();
    assert_eq!(config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&config), 3);
    assert!(ATRConfig::new(0).is_err());

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(3, 2));
    assert_float_slice_close(owned.values(), &[11.0 / 3.0, 28.0 / 9.0]);

    let mut output = [FLOAT_SENTINEL; 6];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..2], owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 4]);

    let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<ATRConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..2], owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 4]);

    let short_input = ATRInput {
        high: &high[..4],
        low: &low[..4],
        close: &close[..4],
    };
    output.fill(FLOAT_SENTINEL);
    let short_owned = IndicatorConfig::compute(&config, short_input).unwrap();
    let short_range =
        PreparedBatchRunner::<ATRConfig>::compute_into(&mut runner, short_input, &mut output)
            .unwrap();
    assert_eq!(short_range, short_owned.range());
    assert_float_slice_close(&output[..1], short_owned.values());
    assert_eq!(&output[1..], &[FLOAT_SENTINEL; 5]);

    let oversized = [Float::NAN; 6];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<ATRConfig>::compute_into(
            &mut runner,
            ATRInput {
                high: &oversized,
                low: &oversized,
                close: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let valid_four = [1.0 as Float, 2.0, 3.0, 4.0];
    let valid_three = [1.0 as Float, 2.0, 3.0];
    let invalid_three = [1.0 as Float, Float::NAN, 3.0];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ATRInput {
                high: &valid_four,
                low: &valid_three,
                close: &valid_four,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high and low must have the same length: got 4 and 3"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ATRInput {
                high: &invalid_three,
                low: &valid_three,
                close: &valid_three,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high[1] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ATRInput {
                high: &valid_three,
                low: &valid_three,
                close: &valid_three,
            },
            &mut output[..0],
        )
        .unwrap_err(),
        TalibError::InsufficientData {
            required: 4,
            actual: 3,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            ATRInput {
                high: &valid_four,
                low: &valid_four,
                close: &valid_four,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: ATR output buffer too small: need 1, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..high.len() {
        let tick = ATRTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        let configured = StreamingComputation::<ATRConfig>::next(&mut stream, tick).unwrap();
        if idx < 3 {
            assert_eq!(configured, None);
        } else {
            assert_some_float_close(configured, owned.values()[idx - 3]);
        }
    }

    let invalid_tick = ATRTick {
        high: Float::NAN,
        low: 0.0,
        close: 1.0,
    };
    assert!(StreamingComputation::<ATRConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = ATRTick {
        high: 18.0,
        low: 16.0,
        close: 17.0,
    };
    let configured_after_rejection =
        StreamingComputation::<ATRConfig>::next(&mut stream, next_tick)
            .unwrap()
            .unwrap();
    assert_float_close(configured_after_rejection, 83.0 / 27.0);

    let period_one = ATRConfig::new(1).unwrap();
    let trange_owned = IndicatorConfig::compute(
        &TRANGEConfig::new(),
        TRANGEInput {
            high: &high,
            low: &low,
            close: &close,
        },
    )
    .unwrap();
    let period_one_owned = IndicatorConfig::compute(&period_one, input).unwrap();
    assert_eq!(period_one_owned.range(), trange_owned.range());
    assert_float_slice_close(period_one_owned.values(), trange_owned.values());

    let mut period_one_stream = IndicatorConfig::stream(&period_one).unwrap();
    assert_eq!(
        StreamingComputation::<ATRConfig>::next(
            &mut period_one_stream,
            ATRTick {
                high: high[0],
                low: low[0],
                close: close[0],
            },
        )
        .unwrap(),
        None
    );
    for idx in 1..high.len() {
        assert_some_float_close(
            StreamingComputation::<ATRConfig>::next(
                &mut period_one_stream,
                ATRTick {
                    high: high[idx],
                    low: low[idx],
                    close: close[idx],
                },
            )
            .unwrap(),
            trange_owned.values()[idx - 1],
        );
    }

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..4 {
        let left_value = StreamingComputation::<ATRConfig>::next(
            &mut left,
            ATRTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            },
        )
        .unwrap();
        let right_value = StreamingComputation::<ATRConfig>::next(
            &mut right,
            ATRTick {
                high: high[idx] * 10.0,
                low: low[idx] * 10.0,
                close: close[idx] * 10.0,
            },
        )
        .unwrap();
        if idx < 3 {
            assert_eq!(left_value, None);
            assert_eq!(right_value, None);
        } else {
            assert_some_float_close(left_value, 11.0 / 3.0);
            assert_some_float_close(right_value, 110.0 / 3.0);
        }
    }

    StreamingComputation::<ATRConfig>::reset(&mut stream);
    for idx in 0..high.len() {
        let tick = ATRTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        let configured = StreamingComputation::<ATRConfig>::next(&mut stream, tick).unwrap();
        if idx >= 3 {
            assert_some_float_close(configured, owned.values()[idx - 3]);
        }
    }
}

#[test]
fn natr_config_preserves_normalization_period_one_and_recursive_stream_state() {
    let high = [10.0 as Float, 12.0, 11.0, 15.0, 16.0];
    let low = [8.0 as Float, 8.0, 9.0, 13.0, 14.0];
    let close = [9.0 as Float, 11.0, 10.0, 14.0, 15.0];
    let input = NATRInput {
        high: &high,
        low: &low,
        close: &close,
    };
    let config = NATRConfig::new(3).unwrap();
    assert_eq!(config.period(), 3);
    assert_eq!(IndicatorConfig::lookback(&config), 3);
    assert!(NATRConfig::new(0).is_err());

    let owned = IndicatorConfig::compute(&config, input).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(3, 2));
    assert_float_slice_close(owned.values(), &[550.0 / 21.0, 560.0 / 27.0]);

    let mut output = [FLOAT_SENTINEL; 6];
    let range = IndicatorConfig::compute_into(&config, input, &mut output).unwrap();
    assert_eq!(range, owned.range());
    assert_float_slice_close(&output[..2], owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 4]);

    let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
    output.fill(FLOAT_SENTINEL);
    let prepared =
        PreparedBatchRunner::<NATRConfig>::compute_into(&mut runner, input, &mut output).unwrap();
    assert_eq!(prepared, owned.range());
    assert_float_slice_close(&output[..2], owned.values());
    assert_eq!(&output[2..], &[FLOAT_SENTINEL; 4]);

    let short_input = NATRInput {
        high: &high[..4],
        low: &low[..4],
        close: &close[..4],
    };
    output.fill(FLOAT_SENTINEL);
    let short_owned = IndicatorConfig::compute(&config, short_input).unwrap();
    let short_range =
        PreparedBatchRunner::<NATRConfig>::compute_into(&mut runner, short_input, &mut output)
            .unwrap();
    assert_eq!(short_range, short_owned.range());
    assert_float_slice_close(&output[..1], short_owned.values());
    assert_eq!(&output[1..], &[FLOAT_SENTINEL; 5]);

    let oversized = [Float::NAN; 6];
    output.fill(FLOAT_SENTINEL);
    assert_eq!(
        PreparedBatchRunner::<NATRConfig>::compute_into(
            &mut runner,
            NATRInput {
                high: &oversized,
                low: &oversized,
                close: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: 5,
            actual_input_len: 6,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let valid_four = [1.0 as Float, 2.0, 3.0, 4.0];
    let valid_three = [1.0 as Float, 2.0, 3.0];
    let invalid_three = [1.0 as Float, Float::NAN, 3.0];
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            NATRInput {
                high: &valid_four,
                low: &valid_three,
                close: &valid_four,
            },
            &mut output,
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high and low must have the same length: got 4 and 3"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            NATRInput {
                high: &invalid_three,
                low: &valid_three,
                close: &valid_three,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: high[1] must be finite, got NaN"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            NATRInput {
                high: &valid_three,
                low: &valid_three,
                close: &valid_three,
            },
            &mut output[..0],
        )
        .unwrap_err(),
        TalibError::InsufficientData {
            required: 4,
            actual: 3,
        }
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);
    assert_eq!(
        IndicatorConfig::compute_into(
            &config,
            NATRInput {
                high: &valid_four,
                low: &valid_four,
                close: &valid_four,
            },
            &mut output[..0],
        )
        .unwrap_err()
        .to_string(),
        "Invalid input: NATR output buffer too small: need 1, got 0"
    );
    assert_eq!(output, [FLOAT_SENTINEL; 6]);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..high.len() {
        let tick = NATRTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        let configured = StreamingComputation::<NATRConfig>::next(&mut stream, tick).unwrap();
        if idx < 3 {
            assert_eq!(configured, None);
        } else {
            assert_some_float_close(configured, owned.values()[idx - 3]);
        }
    }

    let invalid_tick = NATRTick {
        high: Float::NAN,
        low: 0.0,
        close: 1.0,
    };
    assert!(StreamingComputation::<NATRConfig>::next(&mut stream, invalid_tick).is_err());
    let next_tick = NATRTick {
        high: 18.0,
        low: 16.0,
        close: 17.0,
    };
    let configured_after_rejection =
        StreamingComputation::<NATRConfig>::next(&mut stream, next_tick)
            .unwrap()
            .unwrap();
    assert_float_close(configured_after_rejection, (83.0 / 27.0) / 17.0 * 100.0);

    let period_one = NATRConfig::new(1).unwrap();
    let trange_owned = IndicatorConfig::compute(
        &TRANGEConfig::new(),
        TRANGEInput {
            high: &high,
            low: &low,
            close: &close,
        },
    )
    .unwrap();
    let period_one_owned = IndicatorConfig::compute(&period_one, input).unwrap();
    assert_eq!(period_one_owned.range(), trange_owned.range());
    assert_float_slice_close(period_one_owned.values(), trange_owned.values());

    let near_zero_high = [1.0 as Float, 2.0, 3.0, 4.0];
    let near_zero_low = [0.0 as Float, 1.0, 2.0, 3.0];
    let near_zero_close = [0.5 as Float, 1.5, 2.5, 1e-9 as Float];
    let near_zero_input = NATRInput {
        high: &near_zero_high,
        low: &near_zero_low,
        close: &near_zero_close,
    };
    let near_zero_owned = IndicatorConfig::compute(&config, near_zero_input).unwrap();
    assert_eq!(near_zero_owned.range(), OutputRange::new(3, 1));
    assert_float_slice_close(near_zero_owned.values(), &[0.0]);
    let mut near_zero_stream = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..3 {
        assert_eq!(
            StreamingComputation::<NATRConfig>::next(
                &mut near_zero_stream,
                NATRTick {
                    high: near_zero_high[idx],
                    low: near_zero_low[idx],
                    close: near_zero_close[idx],
                },
            )
            .unwrap(),
            None
        );
    }
    assert_some_float_close(
        StreamingComputation::<NATRConfig>::next(
            &mut near_zero_stream,
            NATRTick {
                high: near_zero_high[3],
                low: near_zero_low[3],
                close: near_zero_close[3],
            },
        )
        .unwrap(),
        0.0,
    );

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    for idx in 0..4 {
        let left_value = StreamingComputation::<NATRConfig>::next(
            &mut left,
            NATRTick {
                high: high[idx],
                low: low[idx],
                close: close[idx],
            },
        )
        .unwrap();
        let right_value = StreamingComputation::<NATRConfig>::next(
            &mut right,
            NATRTick {
                high: high[idx] * 10.0,
                low: low[idx] * 10.0,
                close: close[idx] * 10.0,
            },
        )
        .unwrap();
        assert_eq!(left_value.is_some(), right_value.is_some());
        if let (Some(left_value), Some(right_value)) = (left_value, right_value) {
            assert_float_close(left_value, right_value);
        }
    }

    StreamingComputation::<NATRConfig>::reset(&mut stream);
    for idx in 0..high.len() {
        let tick = NATRTick {
            high: high[idx],
            low: low[idx],
            close: close[idx],
        };
        let configured = StreamingComputation::<NATRConfig>::next(&mut stream, tick).unwrap();
        if idx >= 3 {
            assert_some_float_close(configured, owned.values()[idx - 3]);
        }
    }
}

macro_rules! assert_math_transform_config {
    ($config:ty, $stream:ty, $operation:expr) => {{
        let config = <$config>::new();
        let input = [0.25 as Float, 0.5, 0.75];
        let expected = input.map($operation);

        assert_eq!(IndicatorConfig::lookback(&config), 0);
        assert_eq!(core::mem::size_of::<$config>(), 0);
        assert_eq!(core::mem::size_of::<$stream>(), 0);

        let owned = IndicatorConfig::compute(&config, &input).unwrap();
        assert_eq!(owned.range(), OutputRange::new(0, input.len()));
        assert_eq!(owned.values(), expected.as_slice());

        let mut caller_output = [FLOAT_SENTINEL; 4];
        let range = IndicatorConfig::compute_into(&config, &input, &mut caller_output).unwrap();
        assert_eq!(range, owned.range());
        assert_eq!(&caller_output[..input.len()], expected.as_slice());
        assert_eq!(caller_output[input.len()], FLOAT_SENTINEL);

        let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
        let mut prepared_output = [FLOAT_SENTINEL; 4];
        let range =
            PreparedBatchRunner::<$config>::compute_into(&mut runner, &input, &mut prepared_output)
                .unwrap();
        assert_eq!(range, owned.range());
        assert_eq!(&prepared_output[..input.len()], expected.as_slice());
        assert_eq!(prepared_output[input.len()], FLOAT_SENTINEL);

        let oversized = [0.1 as Float, 0.2, 0.3, 0.4];
        let mut unchanged = [FLOAT_SENTINEL; 4];
        assert_eq!(
            PreparedBatchRunner::<$config>::compute_into(&mut runner, &oversized, &mut unchanged,)
                .unwrap_err(),
            TalibError::PreparedCapacityExceeded {
                max_input_len: input.len(),
                actual_input_len: oversized.len(),
            }
        );
        assert_eq!(unchanged, [FLOAT_SENTINEL; 4]);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        for (&value, &expected_value) in input.iter().zip(&expected) {
            assert_eq!(
                StreamingComputation::<$config>::next(&mut stream, value).unwrap(),
                Some(expected_value)
            );
        }
        StreamingComputation::<$config>::reset(&mut stream);
        assert_eq!(
            StreamingComputation::<$config>::next(&mut stream, input[0]).unwrap(),
            Some(expected[0])
        );
    }};
}

#[test]
fn math_transform_configs_cover_all_generated_execution_modes() {
    assert_math_transform_config!(ACOSConfig, ACOSStream, |value: Float| value.acos());
    assert_math_transform_config!(ASINConfig, ASINStream, |value: Float| value.asin());
    assert_math_transform_config!(ATANConfig, ATANStream, |value: Float| value.atan());
    assert_math_transform_config!(CEILConfig, CEILStream, |value: Float| value.ceil());
    assert_math_transform_config!(COSConfig, COSStream, |value: Float| value.cos());
    assert_math_transform_config!(COSHConfig, COSHStream, |value: Float| value.cosh());
    assert_math_transform_config!(EXPConfig, EXPStream, |value: Float| value.exp());
    assert_math_transform_config!(FLOORConfig, FLOORStream, |value: Float| value.floor());
    assert_math_transform_config!(LNConfig, LNStream, |value: Float| value.ln());
    assert_math_transform_config!(LOG10Config, LOG10Stream, |value: Float| value.log10());
    assert_math_transform_config!(SINConfig, SINStream, |value: Float| value.sin());
    assert_math_transform_config!(SINHConfig, SINHStream, |value: Float| value.sinh());
    assert_math_transform_config!(SQRTConfig, SQRTStream, |value: Float| value.sqrt());
    assert_math_transform_config!(TANConfig, TANStream, |value: Float| value.tan());
    assert_math_transform_config!(TANHConfig, TANHStream, |value: Float| value.tanh());
}

#[test]
fn math_transform_configs_preserve_ieee_domain_outputs_and_validation_order() {
    let outside_unit = [-2.0 as Float, 2.0];
    for values in [
        IndicatorConfig::compute(&ACOSConfig::new(), &outside_unit).unwrap(),
        IndicatorConfig::compute(&ASINConfig::new(), &outside_unit).unwrap(),
    ] {
        assert_eq!(values.range(), OutputRange::new(0, 2));
        assert!(values.values().iter().all(|value| value.is_nan()));
    }

    let logarithms = [-1.0 as Float, 0.0];
    for values in [
        IndicatorConfig::compute(&LNConfig::new(), &logarithms).unwrap(),
        IndicatorConfig::compute(&LOG10Config::new(), &logarithms).unwrap(),
    ] {
        assert!(values.values()[0].is_nan());
        assert_eq!(values.values()[1], Float::NEG_INFINITY);
    }

    let square_root = IndicatorConfig::compute(&SQRTConfig::new(), &[-1.0 as Float]).unwrap();
    assert_eq!(square_root.range(), OutputRange::new(0, 1));
    assert!(square_root.values()[0].is_nan());

    let mut no_output = [];
    let invalid_input = [0.0 as Float, Float::NAN];
    assert_eq!(
        IndicatorConfig::compute_into(&SINConfig::new(), &invalid_input, &mut no_output)
            .unwrap_err()
            .to_string(),
        "Invalid input: real[1] must be finite, got NaN"
    );
    assert_eq!(
        IndicatorConfig::compute_into(&SINConfig::new(), &[0.0 as Float], &mut no_output)
            .unwrap_err()
            .to_string(),
        "Invalid input: SIN output buffer too small: need 1, got 0"
    );

    let mut stream = IndicatorConfig::stream(&SINConfig::new()).unwrap();
    assert!(StreamingComputation::<SINConfig>::next(&mut stream, Float::INFINITY).is_err());
}

macro_rules! assert_binary_operator_config {
    ($config:ty, $operation:expr) => {{
        let config = <$config>::new();
        let left = [8.0 as Float, 6.0, 4.0];
        let right = [2.0 as Float, 3.0, 4.0];
        let input = BinaryInput {
            real0: &left,
            real1: &right,
        };
        let expected = [
            $operation(left[0], right[0]),
            $operation(left[1], right[1]),
            $operation(left[2], right[2]),
        ];

        assert_eq!(IndicatorConfig::lookback(&config), 0);
        let owned = IndicatorConfig::compute(&config, input).unwrap();
        assert_eq!(owned.range(), OutputRange::new(0, 3));
        assert_eq!(owned.values(), expected.as_slice());

        let mut caller_output = [FLOAT_SENTINEL; 4];
        let range = IndicatorConfig::compute_into(&config, input, &mut caller_output).unwrap();
        assert_eq!(range, owned.range());
        assert_eq!(&caller_output[..3], expected.as_slice());
        assert_eq!(caller_output[3], FLOAT_SENTINEL);

        let mut runner = IndicatorConfig::prepare_batch(&config, 3).unwrap();
        let mut prepared_output = [FLOAT_SENTINEL; 4];
        let range =
            PreparedBatchRunner::<$config>::compute_into(&mut runner, input, &mut prepared_output)
                .unwrap();
        assert_eq!(range, owned.range());
        assert_eq!(&prepared_output[..3], expected.as_slice());
        assert_eq!(prepared_output[3], FLOAT_SENTINEL);

        let oversized_left = [1.0 as Float; 4];
        let oversized_right = [2.0 as Float; 4];
        let mut unchanged = [FLOAT_SENTINEL; 4];
        assert_eq!(
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                BinaryInput {
                    real0: &oversized_left,
                    real1: &oversized_right,
                },
                &mut unchanged,
            )
            .unwrap_err(),
            TalibError::PreparedCapacityExceeded {
                max_input_len: 3,
                actual_input_len: 4,
            }
        );
        assert_eq!(unchanged, [FLOAT_SENTINEL; 4]);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        assert_eq!(
            StreamingComputation::<$config>::next(
                &mut stream,
                BinaryTick {
                    real0: left[0],
                    real1: right[0],
                },
            )
            .unwrap(),
            Some(expected[0])
        );
        StreamingComputation::<$config>::reset(&mut stream);
        assert_eq!(
            StreamingComputation::<$config>::next(
                &mut stream,
                BinaryTick {
                    real0: left[1],
                    real1: right[1],
                },
            )
            .unwrap(),
            Some(expected[1])
        );

        let mut unchanged = [FLOAT_SENTINEL; 3];
        assert!(IndicatorConfig::compute_into(
            &config,
            BinaryInput {
                real0: &left,
                real1: &right[..2],
            },
            &mut unchanged,
        )
        .is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 3]);

        let invalid = [2.0 as Float, Float::NAN, 4.0];
        assert!(IndicatorConfig::compute_into(
            &config,
            BinaryInput {
                real0: &left,
                real1: &invalid,
            },
            &mut unchanged,
        )
        .is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 3]);
    }};
}

#[test]
fn binary_operator_configs_cover_all_execution_modes_and_validation() {
    assert_binary_operator_config!(ADDConfig, |left, right| left + right);
    assert_binary_operator_config!(SUBConfig, |left, right| left - right);
    assert_binary_operator_config!(MULTConfig, |left, right| left * right);
    assert_binary_operator_config!(DIVConfig, |left, right| left / right);
}

#[test]
fn binary_operator_configs_preserve_operation_produced_non_finite_values() {
    let config = DIVConfig::new();
    let numerators = [1.0 as Float, 0.0, -1.0];
    let zeroes = [0.0 as Float; 3];
    let divided = IndicatorConfig::compute(
        &config,
        BinaryInput {
            real0: &numerators,
            real1: &zeroes,
        },
    )
    .unwrap();
    assert_eq!(divided.values()[0], Float::INFINITY);
    assert!(divided.values()[1].is_nan());
    assert_eq!(divided.values()[2], Float::NEG_INFINITY);

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    assert_eq!(
        StreamingComputation::<DIVConfig>::next(
            &mut stream,
            BinaryTick {
                real0: 1.0,
                real1: 0.0,
            },
        )
        .unwrap(),
        Some(Float::INFINITY)
    );

    let multiplied = IndicatorConfig::compute(
        &MULTConfig::new(),
        BinaryInput {
            real0: &[Float::MAX],
            real1: &[2.0 as Float],
        },
    )
    .unwrap();
    assert_eq!(multiplied.values(), &[Float::INFINITY]);
}

#[test]
fn sum_config_covers_owned_caller_prepared_validation_and_independent_streams() {
    let config = SUMConfig::new(3).unwrap();
    let input = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let owned = IndicatorConfig::compute(&config, &input).unwrap();
    assert_eq!(IndicatorConfig::lookback(&config), 2);
    assert_eq!(owned.range(), OutputRange::new(2, 3));
    assert_eq!(owned.values(), &[6.0 as Float, 9.0, 12.0]);

    let mut caller_output = [FLOAT_SENTINEL; 4];
    assert_eq!(
        IndicatorConfig::compute_into(&config, &input, &mut caller_output).unwrap(),
        owned.range()
    );
    assert_eq!(&caller_output[..3], owned.values());
    assert_eq!(caller_output[3], FLOAT_SENTINEL);

    let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
    let mut prepared_output = [FLOAT_SENTINEL; 4];
    assert_eq!(
        PreparedBatchRunner::<SUMConfig>::compute_into(&mut runner, &input, &mut prepared_output,)
            .unwrap(),
        owned.range()
    );
    assert_eq!(&prepared_output[..3], owned.values());

    let mut left = IndicatorConfig::stream(&config).unwrap();
    let mut right = IndicatorConfig::stream(&config).unwrap();
    for (idx, &value) in input.iter().enumerate() {
        let expected = if idx < 2 {
            None
        } else {
            Some(owned.values()[idx - 2])
        };
        assert_eq!(
            StreamingComputation::<SUMConfig>::next(&mut left, value).unwrap(),
            expected
        );
        assert_eq!(
            StreamingComputation::<SUMConfig>::next(&mut right, value).unwrap(),
            expected
        );
    }
    StreamingComputation::<SUMConfig>::reset(&mut left);
    assert_eq!(
        StreamingComputation::<SUMConfig>::next(&mut left, input[0]).unwrap(),
        None
    );

    let invalid = [1.0 as Float, Float::NAN, 3.0];
    let mut unchanged = [FLOAT_SENTINEL; 2];
    assert!(IndicatorConfig::compute_into(&config, &invalid, &mut unchanged).is_err());
    assert_eq!(unchanged, [FLOAT_SENTINEL; 2]);
    assert!(SUMConfig::new(0).is_err());
}

#[test]
fn sum_config_stream_preserves_legacy_overflow_recovery() {
    let config = SUMConfig::new(2).unwrap();
    let input = [Float::MAX, Float::MAX, 0.0 as Float];

    let mut stream = IndicatorConfig::stream(&config).unwrap();
    assert_eq!(
        StreamingComputation::<SUMConfig>::next(&mut stream, input[0]).unwrap(),
        None
    );
    assert_eq!(
        StreamingComputation::<SUMConfig>::next(&mut stream, input[1]).unwrap(),
        Some(Float::INFINITY)
    );
    assert_eq!(
        StreamingComputation::<SUMConfig>::next(&mut stream, input[2]).unwrap(),
        Some(Float::MAX)
    );
}

macro_rules! assert_single_statistic_config {
    ($config:expr, $config_type:ty) => {{
        let config = $config;
        let input = [1.0 as Float, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
        let owned = IndicatorConfig::compute(&config, &input).unwrap();

        let mut caller_output = [FLOAT_SENTINEL; 8];
        let range = IndicatorConfig::compute_into(&config, &input, &mut caller_output).unwrap();
        assert_eq!(range, owned.range());
        for (&actual, &expected) in caller_output.iter().zip(owned.values()) {
            assert_float_close(actual, expected);
        }
        assert!(caller_output[owned.values().len()..]
            .iter()
            .all(|&value| value == FLOAT_SENTINEL));

        let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
        let mut prepared_output = [FLOAT_SENTINEL; 8];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                &input,
                &mut prepared_output,
            )
            .unwrap(),
            owned.range()
        );
        for (&actual, &expected) in prepared_output.iter().zip(owned.values()) {
            assert_float_close(actual, expected);
        }

        let replay = [7.0 as Float, 5.0, 9.0, 3.0, 8.0, 2.0, 4.0, 1.0];
        let replay_owned = IndicatorConfig::compute(&config, &replay).unwrap();
        PreparedBatchRunner::<$config_type>::compute_into(
            &mut runner,
            &replay,
            &mut prepared_output,
        )
        .unwrap();
        for (&actual, &expected) in prepared_output.iter().zip(replay_owned.values()) {
            assert_float_close(actual, expected);
        }

        let oversized = [1.0 as Float; 9];
        let mut unchanged = [FLOAT_SENTINEL; 9];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                &oversized,
                &mut unchanged,
            )
            .unwrap_err(),
            TalibError::PreparedCapacityExceeded {
                max_input_len: input.len(),
                actual_input_len: oversized.len(),
            }
        );
        assert_eq!(unchanged, [FLOAT_SENTINEL; 9]);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        for (idx, &tick) in input.iter().enumerate() {
            let actual = StreamingComputation::<$config_type>::next(&mut stream, tick).unwrap();
            if idx < owned.range().beg_idx {
                assert_eq!(actual, None);
            } else {
                assert_float_close(actual.unwrap(), owned.values()[idx - owned.range().beg_idx]);
            }
        }
        StreamingComputation::<$config_type>::reset(&mut stream);
        assert_eq!(
            StreamingComputation::<$config_type>::next(&mut stream, input[0]).unwrap(),
            None
        );

        let invalid = [1.0 as Float, Float::NAN, 3.0];
        let mut unchanged = [FLOAT_SENTINEL; 3];
        assert!(IndicatorConfig::compute_into(&config, &invalid, &mut unchanged).is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 3]);
    }};
}

#[test]
fn variance_configs_cover_owned_caller_prepared_and_streaming_execution() {
    assert_single_statistic_config!(VARConfig::new(3, 1.0).unwrap(), VARConfig);
    assert_single_statistic_config!(STDDEVConfig::new(3, 2.0).unwrap(), STDDEVConfig);
    assert!(VARConfig::new(0, 1.0).is_err());
    assert!(STDDEVConfig::new(1, 1.0).is_err());
    assert!(VARConfig::new(3, Float::NAN).is_err());
}

macro_rules! assert_paired_statistic_config {
    ($config:expr, $config_type:ty) => {{
        let config = $config;
        let real0 = [10.0 as Float, 11.0, 13.0, 12.0, 15.0, 14.0, 18.0, 17.0];
        let real1 = [20.0 as Float, 22.0, 23.0, 21.0, 26.0, 24.0, 29.0, 28.0];
        let input = PairInput {
            real0: &real0,
            real1: &real1,
        };
        let owned = IndicatorConfig::compute(&config, input).unwrap();

        let mut caller_output = [FLOAT_SENTINEL; 8];
        let range = IndicatorConfig::compute_into(&config, input, &mut caller_output).unwrap();
        assert_eq!(range, owned.range());
        for (&actual, &expected) in caller_output.iter().zip(owned.values()) {
            assert_float_close(actual, expected);
        }

        let mut runner = IndicatorConfig::prepare_batch(&config, real0.len()).unwrap();
        let mut prepared_output = [FLOAT_SENTINEL; 8];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                input,
                &mut prepared_output,
            )
            .unwrap(),
            owned.range()
        );
        for (&actual, &expected) in prepared_output.iter().zip(owned.values()) {
            assert_float_close(actual, expected);
        }

        let oversized0 = [1.0 as Float; 9];
        let oversized1 = [2.0 as Float; 9];
        let mut unchanged = [FLOAT_SENTINEL; 9];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                PairInput {
                    real0: &oversized0,
                    real1: &oversized1,
                },
                &mut unchanged,
            )
            .unwrap_err(),
            TalibError::PreparedCapacityExceeded {
                max_input_len: real0.len(),
                actual_input_len: oversized0.len(),
            }
        );
        assert_eq!(unchanged, [FLOAT_SENTINEL; 9]);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        for idx in 0..real0.len() {
            let actual = StreamingComputation::<$config_type>::next(
                &mut stream,
                PairTick {
                    real0: real0[idx],
                    real1: real1[idx],
                },
            )
            .unwrap();
            if idx < owned.range().beg_idx {
                assert_eq!(actual, None);
            } else {
                assert_float_close(actual.unwrap(), owned.values()[idx - owned.range().beg_idx]);
            }
        }
        StreamingComputation::<$config_type>::reset(&mut stream);

        let mut unchanged = [FLOAT_SENTINEL; 8];
        assert!(IndicatorConfig::compute_into(
            &config,
            PairInput {
                real0: &real0,
                real1: &real1[..7],
            },
            &mut unchanged,
        )
        .is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 8]);
        let invalid = [Float::NAN; 8];
        assert!(IndicatorConfig::compute_into(
            &config,
            PairInput {
                real0: &invalid,
                real1: &real1,
            },
            &mut unchanged,
        )
        .is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 8]);
    }};
}

#[test]
fn paired_statistic_configs_cover_owned_caller_prepared_and_streaming_execution() {
    assert_paired_statistic_config!(CORRELConfig::new(3).unwrap(), CORRELConfig);
    assert_paired_statistic_config!(BETAConfig::new(3).unwrap(), BETAConfig);
    assert!(CORRELConfig::new(0).is_err());
    assert!(BETAConfig::new(0).is_err());
}

macro_rules! assert_regression_config {
    ($config:expr, $config_type:ty) => {{
        let config = $config;
        assert_eq!(IndicatorConfig::lookback(&config), config.period() - 1);
        let input = [1.0 as Float, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0];
        let owned = IndicatorConfig::compute(&config, &input).unwrap();
        assert_eq!(owned.source_len(), input.len());
        assert_eq!(
            owned.range(),
            OutputRange::new(config.period() - 1, input.len() - (config.period() - 1))
        );
        assert_eq!(owned.values().len(), owned.range().nb_element);

        let mut caller_output = [FLOAT_SENTINEL; 8];
        let range = IndicatorConfig::compute_into(&config, &input, &mut caller_output).unwrap();
        assert_eq!(range, owned.range());
        assert_float_slice_close(&caller_output[..owned.values().len()], owned.values());
        assert!(caller_output[owned.values().len()..]
            .iter()
            .all(|&value| value == FLOAT_SENTINEL));

        let mut runner = IndicatorConfig::prepare_batch(&config, input.len()).unwrap();
        assert_eq!(
            PreparedBatchRunner::<$config_type>::max_input_len(&runner),
            input.len()
        );
        let mut prepared_output = [FLOAT_SENTINEL; 8];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                &input,
                &mut prepared_output,
            )
            .unwrap(),
            owned.range()
        );
        assert_float_slice_close(&prepared_output[..owned.values().len()], owned.values());

        let replay = [7.0 as Float, 5.0, 9.0, 3.0, 8.0, 2.0, 4.0, 1.0];
        let replay_owned = IndicatorConfig::compute(&config, &replay).unwrap();
        PreparedBatchRunner::<$config_type>::compute_into(
            &mut runner,
            &replay,
            &mut prepared_output,
        )
        .unwrap();
        assert_float_slice_close(
            &prepared_output[..replay_owned.values().len()],
            replay_owned.values(),
        );

        let shorter = [2.0 as Float, 6.0, 4.0, 5.0];
        let shorter_owned = IndicatorConfig::compute(&config, &shorter).unwrap();
        let mut shorter_output = [FLOAT_SENTINEL; 4];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                &shorter,
                &mut shorter_output,
            )
            .unwrap(),
            shorter_owned.range()
        );
        assert_float_slice_close(
            &shorter_output[..shorter_owned.values().len()],
            shorter_owned.values(),
        );

        let oversized = [1.0 as Float; 9];
        let mut unchanged = [FLOAT_SENTINEL; 9];
        assert_eq!(
            PreparedBatchRunner::<$config_type>::compute_into(
                &mut runner,
                &oversized,
                &mut unchanged,
            )
            .unwrap_err(),
            TalibError::PreparedCapacityExceeded {
                max_input_len: input.len(),
                actual_input_len: oversized.len(),
            }
        );
        assert_eq!(unchanged, [FLOAT_SENTINEL; 9]);

        let mut stream = IndicatorConfig::stream(&config).unwrap();
        for (idx, &tick) in input.iter().enumerate() {
            let actual = StreamingComputation::<$config_type>::next(&mut stream, tick).unwrap();
            if idx < owned.range().beg_idx {
                assert_eq!(actual, None);
            } else {
                assert_float_close(actual.unwrap(), owned.values()[idx - owned.range().beg_idx]);
            }
        }
        StreamingComputation::<$config_type>::reset(&mut stream);
        assert_eq!(
            StreamingComputation::<$config_type>::next(&mut stream, input[0]).unwrap(),
            None
        );

        let mut rejected_tick_stream = IndicatorConfig::stream(&config).unwrap();
        StreamingComputation::<$config_type>::next(&mut rejected_tick_stream, input[0]).unwrap();
        assert!(
            StreamingComputation::<$config_type>::next(&mut rejected_tick_stream, Float::NAN)
                .is_err()
        );
        for (offset, &tick) in input[1..].iter().enumerate() {
            let idx = offset + 1;
            let actual =
                StreamingComputation::<$config_type>::next(&mut rejected_tick_stream, tick)
                    .unwrap();
            if idx < owned.range().beg_idx {
                assert_eq!(actual, None);
            } else {
                assert_float_close(actual.unwrap(), owned.values()[idx - owned.range().beg_idx]);
            }
        }

        let invalid = [1.0 as Float, Float::NAN, 3.0];
        let mut unchanged = [FLOAT_SENTINEL; 3];
        assert!(IndicatorConfig::compute_into(&config, &invalid, &mut unchanged).is_err());
        assert_eq!(unchanged, [FLOAT_SENTINEL; 3]);
    }};
}

#[test]
fn regression_configs_cover_owned_caller_prepared_and_streaming_execution() {
    assert_regression_config!(LINEARREGConfig::new(3).unwrap(), LINEARREGConfig);
    assert_regression_config!(
        LINEARREG_SLOPEConfig::new(3).unwrap(),
        LINEARREG_SLOPEConfig
    );
    assert_regression_config!(
        LINEARREG_INTERCEPTConfig::new(3).unwrap(),
        LINEARREG_INTERCEPTConfig
    );
    assert_regression_config!(
        LINEARREG_ANGLEConfig::new(3).unwrap(),
        LINEARREG_ANGLEConfig
    );
    assert_regression_config!(TSFConfig::new(3).unwrap(), TSFConfig);
}

#[test]
fn regression_configs_preserve_short_input_and_parameter_validation() {
    for period in [0usize, 1] {
        assert!(LINEARREGConfig::new(period).is_err());
        assert!(LINEARREG_SLOPEConfig::new(period).is_err());
        assert!(LINEARREG_INTERCEPTConfig::new(period).is_err());
        assert!(LINEARREG_ANGLEConfig::new(period).is_err());
        assert!(TSFConfig::new(period).is_err());
    }
    assert!(LINEARREGConfig::new(100_001).is_err());

    let config = LINEARREGConfig::new(4).unwrap();
    let empty = IndicatorConfig::compute(&config, &[]).unwrap();
    assert_eq!(empty.source_len(), 0);
    assert_eq!(empty.range(), OutputRange::empty());
    assert!(empty.values().is_empty());

    let too_small_output = [1.0 as Float, 2.0, 3.0];
    assert!(matches!(
        IndicatorConfig::compute_into(&config, &too_small_output, &mut [0.0; 0]),
        Err(TalibError::InsufficientData {
            required: 4,
            actual: 3
        })
    ));
}

#[test]
fn regression_config_matches_legacy_projection_semantics() {
    let input = [10.0 as Float, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0, 20.0];
    let period = 4usize;

    let linearreg_owned =
        IndicatorConfig::compute(&LINEARREGConfig::new(period).unwrap(), &input).unwrap();
    let slope_owned =
        IndicatorConfig::compute(&LINEARREG_SLOPEConfig::new(period).unwrap(), &input).unwrap();
    let intercept_owned =
        IndicatorConfig::compute(&LINEARREG_INTERCEPTConfig::new(period).unwrap(), &input).unwrap();
    let angle_owned =
        IndicatorConfig::compute(&LINEARREG_ANGLEConfig::new(period).unwrap(), &input).unwrap();
    let tsf_owned = IndicatorConfig::compute(&TSFConfig::new(period).unwrap(), &input).unwrap();

    assert_eq!(linearreg_owned.range(), OutputRange::new(3, 5));
    assert_float_slice_close(
        linearreg_owned.values(),
        &[14.1 as Float, 14.5, 17.5, 17.5, 19.8],
    );
    assert_float_slice_close(slope_owned.values(), &[1.4 as Float, 1.0, 2.0, 1.0, 1.7]);
    assert_float_slice_close(
        intercept_owned.values(),
        &[9.9 as Float, 11.5, 11.5, 14.5, 14.7],
    );
    for (&angle, &slope) in angle_owned.values().iter().zip(slope_owned.values()) {
        assert_float_close(
            angle,
            slope.atan() * (180.0 as Float / core::f64::consts::PI as Float),
        );
    }
    assert_float_slice_close(tsf_owned.values(), &[15.5 as Float, 15.5, 19.5, 18.5, 21.5]);
}
