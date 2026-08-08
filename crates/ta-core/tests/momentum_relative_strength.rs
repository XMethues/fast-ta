#[path = "fixtures/relative_strength_reference.rs"]
mod reference;

use ta_core::inventory::{function, ImplementationStatus};
use ta_core::momentum::{
    CMOBatchRunner, CMOConfig, CMOStream, IMIBatchRunner, IMIConfig, IMIInput, IMIStream, IMITick,
    RSIBatchRunner, RSIConfig, RSIStream, CMO, IMI, RSI,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const TOLERANCE: Float = 3e-4;
#[cfg(not(feature = "f32"))]
const TOLERANCE: Float = 1e-11;

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "expected {expected}, got {actual}"
    );
}

fn assert_slice_close(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected as Float);
    }
}

fn real_fixture() -> Vec<Float> {
    reference::REAL
        .iter()
        .map(|&value| value as Float)
        .collect()
}

fn open_fixture() -> Vec<Float> {
    reference::OPEN
        .iter()
        .map(|&value| value as Float)
        .collect()
}

fn close_fixture() -> Vec<Float> {
    reference::CLOSE
        .iter()
        .map(|&value| value as Float)
        .collect()
}

fn assert_full_seam<C, B, S>()
where
    C: IndicatorConfig<BatchRunner = B, Stream = S>,
    B: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn inventory_and_public_types_cover_all_three_definitions() {
    for name in ["RSI", "CMO", "IMI"] {
        assert_eq!(
            function(name).expect("catalogue entry").status,
            ImplementationStatus::Implemented
        );
    }
    assert_full_seam::<RSIConfig, RSIBatchRunner, RSIStream>();
    assert_full_seam::<CMOConfig, CMOBatchRunner, CMOStream>();
    assert_full_seam::<IMIConfig, IMIBatchRunner, IMIStream>();
}

#[test]
fn pinned_reference_vectors_preserve_definition_and_source_alignment() {
    assert_eq!(reference::TALIB_VERSION, "0.8.1");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "e64d2ac896c595f38d65e44c812efbfdac8a64cf"
    );
    let real = real_fixture();
    let open = open_fixture();
    let close = close_fixture();
    let mut rsi = [-99.0 as Float; 12];
    let mut cmo = [-99.0 as Float; 12];
    let mut imi = [-99.0 as Float; 10];

    let rsi_range = RSI(&real, reference::PERIOD, &mut rsi).unwrap();
    let cmo_range = CMO(&real, reference::PERIOD, &mut cmo).unwrap();
    let imi_range = IMI(&open, &close, reference::PERIOD, &mut imi).unwrap();

    assert_eq!(rsi_range, OutputRange::new(5, 7));
    assert_eq!(cmo_range, OutputRange::new(5, 7));
    assert_eq!(imi_range, OutputRange::new(4, 6));
    assert_slice_close(&rsi[..rsi_range.nb_element], reference::RSI_EXPECTED);
    assert_slice_close(&cmo[..cmo_range.nb_element], reference::CMO_EXPECTED);
    assert_slice_close(&imi[..imi_range.nb_element], reference::IMI_EXPECTED);
    assert_eq!(rsi[rsi_range.nb_element], -99.0 as Float);
    assert_eq!(cmo[cmo_range.nb_element], -99.0 as Float);
    assert_eq!(imi[imi_range.nb_element], -99.0 as Float);
}

#[test]
fn immutable_configs_return_exact_owned_compact_outputs() {
    let real = real_fixture();
    let open = open_fixture();
    let close = close_fixture();
    let rsi = RSIConfig::new(5).unwrap();
    let cmo = CMOConfig::new(5).unwrap();
    let imi = IMIConfig::new(5).unwrap();

    assert_eq!(rsi.period(), 5);
    assert_eq!(cmo.period(), 5);
    assert_eq!(imi.period(), 5);
    assert_eq!(IndicatorConfig::lookback(&rsi), 5);
    assert_eq!(IndicatorConfig::lookback(&cmo), 5);
    assert_eq!(IndicatorConfig::lookback(&imi), 4);

    let rsi_output = IndicatorConfig::compute(&rsi, real.as_slice()).unwrap();
    let cmo_output = IndicatorConfig::compute(&cmo, real.as_slice()).unwrap();
    let imi_output = IndicatorConfig::compute(
        &imi,
        IMIInput {
            open: &open,
            close: &close,
        },
    )
    .unwrap();

    assert_eq!(rsi_output.source_len(), real.len());
    assert_eq!(rsi_output.range(), OutputRange::new(5, 7));
    assert_slice_close(rsi_output.values(), reference::RSI_EXPECTED);
    assert_eq!(cmo_output.source_len(), real.len());
    assert_eq!(cmo_output.range(), OutputRange::new(5, 7));
    assert_slice_close(cmo_output.values(), reference::CMO_EXPECTED);
    assert_eq!(imi_output.source_len(), open.len());
    assert_eq!(imi_output.range(), OutputRange::new(4, 6));
    assert_slice_close(imi_output.values(), reference::IMI_EXPECTED);
}

#[test]
fn prepared_runners_reuse_capacity_and_reject_oversize_before_mutation() {
    let real = real_fixture();
    let open = open_fixture();
    let close = close_fixture();
    let rsi = RSIConfig::new(5).unwrap();
    let cmo = CMOConfig::new(5).unwrap();
    let imi = IMIConfig::new(5).unwrap();
    let mut rsi_runner = IndicatorConfig::prepare_batch(&rsi, real.len()).unwrap();
    let mut cmo_runner = IndicatorConfig::prepare_batch(&cmo, real.len()).unwrap();
    let mut imi_runner = IndicatorConfig::prepare_batch(&imi, open.len()).unwrap();
    let mut rsi_output = [-7.0 as Float; 12];
    let mut cmo_output = [-7.0 as Float; 12];
    let mut imi_output = [-7.0 as Float; 10];

    for _ in 0..2 {
        let rsi_range =
            PreparedBatchRunner::<RSIConfig>::compute_into(&mut rsi_runner, &real, &mut rsi_output)
                .unwrap();
        let cmo_range =
            PreparedBatchRunner::<CMOConfig>::compute_into(&mut cmo_runner, &real, &mut cmo_output)
                .unwrap();
        let imi_range = PreparedBatchRunner::<IMIConfig>::compute_into(
            &mut imi_runner,
            IMIInput {
                open: &open,
                close: &close,
            },
            &mut imi_output,
        )
        .unwrap();
        assert_eq!(rsi_range, OutputRange::new(5, 7));
        assert_eq!(cmo_range, OutputRange::new(5, 7));
        assert_eq!(imi_range, OutputRange::new(4, 6));
    }

    let oversized_real = vec![1.0 as Float; real.len() + 1];
    let before = rsi_output;
    assert!(matches!(
        PreparedBatchRunner::<RSIConfig>::compute_into(
            &mut rsi_runner,
            &oversized_real,
            &mut rsi_output
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert_eq!(rsi_output, before);
    let before = cmo_output;
    assert!(matches!(
        PreparedBatchRunner::<CMOConfig>::compute_into(
            &mut cmo_runner,
            &oversized_real,
            &mut cmo_output
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert_eq!(cmo_output, before);

    let oversized_close = vec![1.0 as Float; close.len() + 1];
    let before = imi_output;
    assert!(matches!(
        PreparedBatchRunner::<IMIConfig>::compute_into(
            &mut imi_runner,
            IMIInput {
                open: &open,
                close: &oversized_close,
            },
            &mut imi_output,
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert_eq!(imi_output, before);
}

#[test]
fn streaming_warmup_batch_parity_reset_replay_and_independence_hold() {
    let real = real_fixture();
    let open = open_fixture();
    let close = close_fixture();
    let rsi_config = RSIConfig::new(5).unwrap();
    let cmo_config = CMOConfig::new(5).unwrap();
    let imi_config = IMIConfig::new(5).unwrap();
    let mut rsi = IndicatorConfig::stream(&rsi_config).unwrap();
    let mut cmo = IndicatorConfig::stream(&cmo_config).unwrap();
    let mut imi = IndicatorConfig::stream(&imi_config).unwrap();
    let mut rsi_values = Vec::new();
    let mut cmo_values = Vec::new();
    let mut imi_values = Vec::new();

    for (index, &value) in real.iter().enumerate() {
        let next = StreamingComputation::<RSIConfig>::next(&mut rsi, value).unwrap();
        let next_cmo = StreamingComputation::<CMOConfig>::next(&mut cmo, value).unwrap();
        if index < 5 {
            assert!(next.is_none());
            assert!(next_cmo.is_none());
        }
        if let Some(value) = next {
            rsi_values.push(value);
        }
        if let Some(value) = next_cmo {
            cmo_values.push(value);
        }
    }
    for index in 0..open.len() {
        let next = StreamingComputation::<IMIConfig>::next(
            &mut imi,
            IMITick {
                open: open[index],
                close: close[index],
            },
        )
        .unwrap();
        if index < 4 {
            assert!(next.is_none());
        }
        if let Some(value) = next {
            imi_values.push(value);
        }
    }
    assert_slice_close(&rsi_values, reference::RSI_EXPECTED);
    assert_slice_close(&cmo_values, reference::CMO_EXPECTED);
    assert_slice_close(&imi_values, reference::IMI_EXPECTED);

    StreamingComputation::<RSIConfig>::reset(&mut rsi);
    StreamingComputation::<CMOConfig>::reset(&mut cmo);
    StreamingComputation::<IMIConfig>::reset(&mut imi);
    let mut independent_rsi = IndicatorConfig::stream(&rsi_config).unwrap();
    let mut independent_cmo = IndicatorConfig::stream(&cmo_config).unwrap();
    let mut independent_imi = IndicatorConfig::stream(&imi_config).unwrap();
    let mut cmo_replay = Vec::new();
    let mut imi_replay = Vec::new();
    for &value in &real {
        assert_eq!(
            StreamingComputation::<RSIConfig>::next(&mut rsi, value).unwrap(),
            StreamingComputation::<RSIConfig>::next(&mut independent_rsi, value).unwrap()
        );
        let replay = StreamingComputation::<CMOConfig>::next(&mut cmo, value).unwrap();
        assert_eq!(
            replay,
            StreamingComputation::<CMOConfig>::next(&mut independent_cmo, value).unwrap()
        );
        if let Some(value) = replay {
            cmo_replay.push(value);
        }
    }
    for index in 0..open.len() {
        let tick = IMITick {
            open: open[index],
            close: close[index],
        };
        let replay = StreamingComputation::<IMIConfig>::next(&mut imi, tick).unwrap();
        assert_eq!(
            replay,
            StreamingComputation::<IMIConfig>::next(&mut independent_imi, tick).unwrap()
        );
        if let Some(value) = replay {
            imi_replay.push(value);
        }
    }
    assert_slice_close(&cmo_replay, reference::CMO_EXPECTED);
    assert_slice_close(&imi_replay, reference::IMI_EXPECTED);
}

#[test]
fn rejected_stream_ticks_preserve_recursive_and_rolling_state() {
    let real = real_fixture();
    let rsi_config = RSIConfig::new(5).unwrap();
    let cmo_config = CMOConfig::new(5).unwrap();
    let imi_config = IMIConfig::new(5).unwrap();
    let mut rsi = IndicatorConfig::stream(&rsi_config).unwrap();
    let mut cmo = IndicatorConfig::stream(&cmo_config).unwrap();
    let mut imi = IndicatorConfig::stream(&imi_config).unwrap();
    for &value in &real[..4] {
        StreamingComputation::<RSIConfig>::next(&mut rsi, value).unwrap();
        StreamingComputation::<CMOConfig>::next(&mut cmo, value).unwrap();
    }
    for index in 0..3 {
        StreamingComputation::<IMIConfig>::next(
            &mut imi,
            IMITick {
                open: reference::OPEN[index] as Float,
                close: reference::CLOSE[index] as Float,
            },
        )
        .unwrap();
    }
    let mut rsi_control = rsi.clone();
    let mut cmo_control = cmo.clone();
    let mut imi_control = imi.clone();

    assert!(StreamingComputation::<RSIConfig>::next(&mut rsi, Float::NAN).is_err());
    assert!(StreamingComputation::<CMOConfig>::next(&mut cmo, Float::NAN).is_err());
    assert!(StreamingComputation::<IMIConfig>::next(
        &mut imi,
        IMITick {
            open: Float::INFINITY,
            close: 1.0,
        }
    )
    .is_err());
    assert_eq!(
        StreamingComputation::<RSIConfig>::next(&mut rsi, real[4]).unwrap(),
        StreamingComputation::<RSIConfig>::next(&mut rsi_control, real[4]).unwrap()
    );
    assert_eq!(
        StreamingComputation::<CMOConfig>::next(&mut cmo, real[4]).unwrap(),
        StreamingComputation::<CMOConfig>::next(&mut cmo_control, real[4]).unwrap()
    );
    let tick = IMITick {
        open: reference::OPEN[3] as Float,
        close: reference::CLOSE[3] as Float,
    };
    assert_eq!(
        StreamingComputation::<IMIConfig>::next(&mut imi, tick).unwrap(),
        StreamingComputation::<IMIConfig>::next(&mut imi_control, tick).unwrap()
    );
    assert_eq!(
        StreamingComputation::<RSIConfig>::next(&mut rsi, real[5]).unwrap(),
        StreamingComputation::<RSIConfig>::next(&mut rsi_control, real[5]).unwrap()
    );
    assert_eq!(
        StreamingComputation::<CMOConfig>::next(&mut cmo, real[5]).unwrap(),
        StreamingComputation::<CMOConfig>::next(&mut cmo_control, real[5]).unwrap()
    );
    let next_tick = IMITick {
        open: reference::OPEN[4] as Float,
        close: reference::CLOSE[4] as Float,
    };
    assert_eq!(
        StreamingComputation::<IMIConfig>::next(&mut imi, next_tick).unwrap(),
        StreamingComputation::<IMIConfig>::next(&mut imi_control, next_tick).unwrap()
    );
}

#[test]
fn denominator_edges_and_bounded_range_invariants_are_explicit() {
    let period = 3;
    let flat = [10.0 as Float; 7];
    let gains = [1.0 as Float, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let losses = [7.0 as Float, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let alternating = [10.0 as Float, 12.0, 9.0, 13.0, 8.0, 14.0, 7.0];
    let mut output = [0.0 as Float; 7];

    for (input, rsi_expected, cmo_expected) in [
        (flat.as_slice(), 0.0 as Float, 0.0 as Float),
        (gains.as_slice(), 100.0 as Float, 100.0 as Float),
        (losses.as_slice(), 0.0 as Float, -100.0 as Float),
    ] {
        let range = RSI(input, period, &mut output).unwrap();
        assert!(output[..range.nb_element]
            .iter()
            .all(|&value| value == rsi_expected));
        let range = CMO(input, period, &mut output).unwrap();
        assert!(output[..range.nb_element]
            .iter()
            .all(|&value| value == cmo_expected));
    }
    let range = RSI(&alternating, period, &mut output).unwrap();
    assert!(output[..range.nb_element]
        .iter()
        .all(|&value| (0.0..=100.0).contains(&value)));
    let range = CMO(&alternating, period, &mut output).unwrap();
    assert!(output[..range.nb_element]
        .iter()
        .all(|&value| (-100.0..=100.0).contains(&value)));

    let open = [10.0 as Float; 7];
    for (close, expected) in [
        ([10.0 as Float; 7], 50.0 as Float),
        ([11.0 as Float; 7], 100.0 as Float),
        ([9.0 as Float; 7], 0.0 as Float),
    ] {
        let range = IMI(&open, &close, period, &mut output).unwrap();
        assert!(output[..range.nb_element]
            .iter()
            .all(|&value| value == expected));
    }
    let zero_open = [0.0 as Float; 7];
    let tiny_gain = [Float::MIN_POSITIVE; 7];
    let range = IMI(&zero_open, &tiny_gain, period, &mut output).unwrap();
    assert!(output[..range.nb_element]
        .iter()
        .all(|&value| value == 100.0 as Float));
    let mixed_close = [11.0 as Float, 9.0, 10.0, 12.0, 8.0, 11.0, 9.0];
    let range = IMI(&open, &mixed_close, period, &mut output).unwrap();
    assert!(output[..range.nb_element]
        .iter()
        .all(|&value| (0.0..=100.0).contains(&value)));
}

#[test]
fn validation_and_capacity_errors_never_mutate_caller_outputs() {
    assert!(RSIConfig::new(1).is_err());
    assert!(CMOConfig::new(100_001).is_err());
    assert!(IMIConfig::new(0).is_err());
    assert!(RSIConfig::new(100_000).is_ok());

    let mut output = [123.0 as Float; 8];
    let before = output;
    assert!(RSI(&[1.0, 2.0, 3.0], 1, &mut output).is_err());
    assert_eq!(output, before);
    assert!(RSI(&[1.0, 2.0, 3.0], 3, &mut output).is_err());
    assert_eq!(output, before);
    assert!(CMO(&[1.0, Float::NAN, 3.0, 4.0], 2, &mut output).is_err());
    assert_eq!(output, before);
    assert!(IMI(&[1.0, 2.0], &[1.0], 2, &mut output).is_err());
    assert_eq!(output, before);
    assert!(IMI(&[1.0, 2.0], &[1.0, Float::INFINITY], 2, &mut output).is_err());
    assert_eq!(output, before);

    let real = real_fixture();
    let mut too_small = [77.0 as Float; 1];
    let before_small = too_small;
    assert!(RSI(&real, 5, &mut too_small).is_err());
    assert_eq!(too_small, before_small);
    assert!(CMO(&real, 5, &mut too_small).is_err());
    assert_eq!(too_small, before_small);
    let open = open_fixture();
    let close = close_fixture();
    assert!(IMI(&open, &close, 5, &mut too_small).is_err());
    assert_eq!(too_small, before_small);

    let rsi_empty = IndicatorConfig::compute(&RSIConfig::new(5).unwrap(), &[]).unwrap();
    assert_eq!(rsi_empty.range(), OutputRange::empty());
    assert!(rsi_empty.values().is_empty());
    let imi_empty = IndicatorConfig::compute(
        &IMIConfig::new(5).unwrap(),
        IMIInput {
            open: &[],
            close: &[],
        },
    )
    .unwrap();
    assert_eq!(imi_empty.range(), OutputRange::empty());
    assert!(imi_empty.values().is_empty());
}
