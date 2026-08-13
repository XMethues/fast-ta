// Pinned f64 oracle values intentionally retain more precision than f32 builds consume.
#[allow(clippy::excessive_precision)]
#[path = "fixtures/momentum_change_reference.rs"]
mod reference;

use reference::{
    INPUT, MOM_EXPECTED, PERIOD, ROCP_EXPECTED, ROCR100_EXPECTED, ROCR_EXPECTED, ROC_EXPECTED,
};
use fast_ta::{
    momentum::{
        MOMConfig, ROCConfig, ROCPConfig, ROCR100Config, ROCRConfig, MOM, ROC, ROCP, ROCR, ROCR100,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const TOLERANCE: Float = 2e-5 as Float;
#[cfg(not(feature = "f32"))]
const TOLERANCE: Float = 1e-12 as Float;

fn assert_close(actual: Float, expected: Float) {
    let scale = expected.abs().max(1.0 as Float);
    assert!(
        (actual - expected).abs() <= TOLERANCE * scale,
        "actual {actual:?}, expected {expected:?}, tolerance {:?}",
        TOLERANCE * scale
    );
}

fn assert_slice_close(actual: &[Float], expected: &[Float]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected);
    }
}

macro_rules! assert_execution_modes {
    ($config:ty, $function:ident, $expected:expr) => {{
        let config = <$config>::new(PERIOD).expect("valid Period");
        assert_eq!(config.period(), PERIOD);
        assert_eq!(IndicatorConfig::lookback(&config), PERIOD);

        let mut caller = vec![-777.0 as Float; $expected.len()];
        let range = $function(INPUT, PERIOD, &mut caller).expect("valid caller-owned execution");
        assert_eq!(range, OutputRange::new(PERIOD, $expected.len()));
        assert_slice_close(&caller, $expected);

        let owned = IndicatorConfig::compute(&config, INPUT).expect("valid owned execution");
        assert_eq!(owned.source_len(), INPUT.len());
        assert_eq!(owned.range(), range);
        assert_slice_close(owned.values(), $expected);

        let empty = IndicatorConfig::compute(&config, &[]).expect("valid empty execution");
        assert_eq!(empty.source_len(), 0);
        assert_eq!(empty.range(), OutputRange::empty());
        assert!(empty.values().is_empty());

        let mut prepared = IndicatorConfig::prepare_batch(&config, INPUT.len())
            .expect("valid Prepared Batch Runner");
        assert_eq!(prepared.max_input_len(), INPUT.len());
        let mut prepared_output = vec![-999.0 as Float; $expected.len()];
        let prepared_range = PreparedBatchRunner::<$config>::compute_into(
            &mut prepared,
            INPUT,
            &mut prepared_output,
        )
        .expect("valid prepared execution");
        assert_eq!(prepared_range, range);
        assert_slice_close(&prepared_output, $expected);
        prepared_output.fill(-333.0 as Float);
        let repeated_range = prepared
            .compute_into(INPUT, &mut prepared_output)
            .expect("valid repeated prepared execution");
        assert_eq!(repeated_range, range);
        assert_slice_close(&prepared_output, $expected);

        let mut stream = IndicatorConfig::stream(&config).expect("valid stream");
        let mut streamed = Vec::new();
        for &tick in INPUT {
            if let Some(value) = StreamingComputation::<$config>::next(&mut stream, tick)
                .expect("valid streaming Tick")
            {
                streamed.push(value);
            }
        }
        assert_slice_close(&streamed, $expected);

        StreamingComputation::<$config>::reset(&mut stream);
        let mut replayed = Vec::new();
        for &tick in INPUT {
            if let Some(value) =
                StreamingComputation::<$config>::next(&mut stream, tick).expect("valid replay Tick")
            {
                replayed.push(value);
            }
        }
        assert_slice_close(&replayed, $expected);

        let mut first = config.stream().expect("valid independent stream");
        let mut second = config.stream().expect("valid independent stream");
        let mut first_values = Vec::new();
        let mut second_values = Vec::new();
        for &tick in INPUT {
            if let Some(value) = first.next(tick).expect("valid first stream Tick") {
                first_values.push(value);
            }
            if let Some(value) = second.next(tick).expect("valid second stream Tick") {
                second_values.push(value);
            }
        }
        assert_slice_close(&first_values, $expected);
        assert_slice_close(&second_values, $expected);
    }};
}

#[test]
fn independent_f64_and_f32_reference_vectors_match_every_execution_mode() {
    assert_execution_modes!(MOMConfig, MOM, MOM_EXPECTED);
    assert_execution_modes!(ROCConfig, ROC, ROC_EXPECTED);
    assert_execution_modes!(ROCPConfig, ROCP, ROCP_EXPECTED);
    assert_execution_modes!(ROCRConfig, ROCR, ROCR_EXPECTED);
    assert_execution_modes!(ROCR100Config, ROCR100, ROCR100_EXPECTED);
}

#[test]
fn family_scaling_invariants_share_one_source_position_relationship() {
    let mom = MOMConfig::new(PERIOD).unwrap().compute(INPUT).unwrap();
    let roc = ROCConfig::new(PERIOD).unwrap().compute(INPUT).unwrap();
    let rocp = ROCPConfig::new(PERIOD).unwrap().compute(INPUT).unwrap();
    let rocr = ROCRConfig::new(PERIOD).unwrap().compute(INPUT).unwrap();
    let rocr100 = ROCR100Config::new(PERIOD).unwrap().compute(INPUT).unwrap();

    for (output_idx, (&previous, &current)) in
        INPUT.iter().zip(INPUT.iter().skip(PERIOD)).enumerate()
    {
        assert_close(mom.values()[output_idx], current - previous);
        if INPUT[output_idx] != 0.0 as Float {
            assert_close(
                roc.values()[output_idx],
                rocp.values()[output_idx] * 100.0 as Float,
            );
            assert_close(
                rocr.values()[output_idx],
                rocp.values()[output_idx] + 1.0 as Float,
            );
            assert_close(
                rocr100.values()[output_idx],
                rocr.values()[output_idx] * 100.0 as Float,
            );
        }
    }
}

#[test]
fn exact_zero_and_near_zero_denominators_are_distinct() {
    let input = [0.0 as Float, 2.0 as Float, 1e-12 as Float, 3.0 as Float];
    let mut roc = [0.0 as Float; 3];
    let mut rocp = [0.0 as Float; 3];
    let mut rocr = [0.0 as Float; 3];
    let mut rocr100 = [0.0 as Float; 3];

    ROC(&input, 1, &mut roc).unwrap();
    ROCP(&input, 1, &mut rocp).unwrap();
    ROCR(&input, 1, &mut rocr).unwrap();
    ROCR100(&input, 1, &mut rocr100).unwrap();

    assert_eq!(roc[0], 0.0 as Float);
    assert_eq!(rocp[0], 0.0 as Float);
    assert_eq!(rocr[0], 0.0 as Float);
    assert_eq!(rocr100[0], 0.0 as Float);

    assert_close(
        roc[1],
        (input[2] / input[1] - 1.0 as Float) * 100.0 as Float,
    );
    assert_close(rocp[1], (input[2] - input[1]) / input[1]);
    assert_close(rocr[1], input[2] / input[1]);
    assert_close(rocr100[1], input[2] / input[1] * 100.0 as Float);

    assert!(roc[2].abs() > 1e12 as Float);
    assert!(rocp[2].abs() > 1e12 as Float);
    assert!(rocr[2].abs() > 1e12 as Float);
    assert!(rocr100[2].abs() > 1e12 as Float);
}

macro_rules! assert_failure_contracts {
    ($config:ty, $function:ident, $expected_after_invalid:expr) => {{
        assert!(matches!(
            <$config>::new(0),
            Err(TalibError::InvalidPeriod { .. })
        ));
        assert!(matches!(
            <$config>::new(100_001),
            Err(TalibError::InvalidPeriod { .. })
        ));

        let mut untouched = [91.0 as Float];
        let error = $function(&[Float::NAN], 0, &mut untouched).unwrap_err();
        assert!(matches!(error, TalibError::InvalidPeriod { .. }));
        assert_eq!(untouched, [91.0 as Float]);
        let error = $function(&[Float::NAN], 100_001, &mut untouched).unwrap_err();
        assert!(matches!(error, TalibError::InvalidPeriod { .. }));
        assert_eq!(untouched, [91.0 as Float]);

        let error = $function(&[1.0 as Float, Float::NAN], 1, &mut untouched).unwrap_err();
        assert!(matches!(error, TalibError::InvalidInput { .. }));
        assert_eq!(untouched, [91.0 as Float]);

        let error = $function(&[Float::NAN], 2, &mut untouched).unwrap_err();
        assert!(matches!(error, TalibError::InvalidInput { .. }));
        assert_eq!(untouched, [91.0 as Float]);

        let error = $function(&[1.0 as Float], 1, &mut untouched).unwrap_err();
        assert!(matches!(
            error,
            TalibError::InsufficientData {
                required: 2,
                actual: 1
            }
        ));
        assert_eq!(untouched, [91.0 as Float]);

        let error = $function(&[1.0 as Float, 2.0 as Float], 1, &mut []).unwrap_err();
        assert!(matches!(error, TalibError::InvalidInput { .. }));

        let config = <$config>::new(2).unwrap();
        let mut prepared = config.prepare_batch(2).unwrap();
        let mut prepared_output = [37.0 as Float];
        let error = PreparedBatchRunner::<$config>::compute_into(
            &mut prepared,
            &[1.0 as Float, 2.0 as Float, Float::NAN],
            &mut prepared_output,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TalibError::PreparedCapacityExceeded {
                max_input_len: 2,
                actual_input_len: 3
            }
        ));
        assert_eq!(prepared_output, [37.0 as Float]);

        let mut failed_stream = config.stream().unwrap();
        assert_eq!(failed_stream.next(1.0 as Float).unwrap(), None);
        assert!(matches!(
            failed_stream.next(Float::INFINITY),
            Err(TalibError::InvalidInput { .. })
        ));
        assert_eq!(failed_stream.next(2.0 as Float).unwrap(), None);
        let after_failure = failed_stream.next(3.0 as Float).unwrap().unwrap();
        assert_close(after_failure, $expected_after_invalid);
    }};
}

#[test]
fn validation_order_capacity_and_failure_before_mutation_are_stable() {
    assert_failure_contracts!(MOMConfig, MOM, 2.0 as Float);
    assert_failure_contracts!(ROCConfig, ROC, 200.0 as Float);
    assert_failure_contracts!(ROCPConfig, ROCP, 2.0 as Float);
    assert_failure_contracts!(ROCRConfig, ROCR, 3.0 as Float);
    assert_failure_contracts!(ROCR100Config, ROCR100, 300.0 as Float);
}
