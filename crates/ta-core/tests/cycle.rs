#[path = "fixtures/ht_dcperiod_reference.rs"]
mod reference;

use fast_ta::{
    cycle::{
        HT_DCPERIODBatchRunner, HT_DCPERIODConfig, HT_DCPERIODStream, HT_DCPERIOD,
        HT_DCPERIOD_LOOKBACK,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const ABS_TOLERANCE: Float = 5.0e-3;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 5.0e-4;
#[cfg(not(feature = "f32"))]
const ABS_TOLERANCE: Float = 1.0e-9;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 1.0e-12;

fn assert_close(actual: Float, expected: f64, context: &str) {
    let expected = expected as Float;
    let difference = (actual - expected).abs();
    let tolerance = ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs());
    assert!(
        difference <= tolerance,
        "{context}: expected {expected}, got {actual}, difference {difference}, tolerance {tolerance}"
    );
}

fn assert_values_close(actual: &[Float], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_close(
            actual,
            expected,
            &format!("{context}, compact index {index}"),
        );
    }
}

fn as_float(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

#[test]
fn ht_dcperiod_matches_pinned_talib_vectors_in_every_execution_mode() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    assert_eq!(reference::OUTPUT_BEGIN, HT_DCPERIOD_LOOKBACK);
    assert_eq!(reference::SOURCE_LENGTH, 256);
    assert_eq!(reference::NOISE_SEED, 0x5EED_C0DE);

    let config = HT_DCPERIODConfig::new();
    assert_eq!(config.lookback(), HT_DCPERIOD_LOOKBACK);

    for case in reference::CASES {
        let input = as_float(case.input);
        let range = OutputRange::new(HT_DCPERIOD_LOOKBACK, case.expected.len());
        let context = format!("{} ({})", case.name, case.definition);

        let owned = config.compute(input.as_slice()).unwrap();
        assert_eq!(owned.source_len(), input.len(), "{context}, owned");
        assert_eq!(owned.range(), range, "{context}, owned");
        assert_values_close(owned.values(), case.expected, &format!("{context}, owned"));

        let mut caller_output = vec![0.0 as Float; case.expected.len()];
        assert_eq!(
            HT_DCPERIOD(input.as_slice(), caller_output.as_mut_slice()).unwrap(),
            range,
            "{context}, caller-owned"
        );
        assert_values_close(
            &caller_output,
            case.expected,
            &format!("{context}, caller-owned"),
        );

        let mut runner = config.prepare_batch(input.len()).unwrap();
        assert_eq!(runner.max_input_len(), input.len());
        let mut prepared_output = vec![0.0 as Float; case.expected.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                runner
                    .compute_into(input.as_slice(), prepared_output.as_mut_slice())
                    .unwrap(),
                range,
                "{context}, prepared {pass}"
            );
            assert_values_close(
                &prepared_output,
                case.expected,
                &format!("{context}, prepared {pass}"),
            );
        }

        let mut stream = config.stream().unwrap();
        let streamed = input
            .iter()
            .copied()
            .filter_map(|tick| stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_values_close(&streamed, case.expected, &format!("{context}, streaming"));

        stream.reset();
        let replayed = input
            .iter()
            .copied()
            .filter_map(|tick| stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(streamed, replayed, "{context}, reset replay");
    }
}

#[test]
fn ht_dcperiod_reports_validation_and_capacity_failures_without_mutating_state() {
    let config = HT_DCPERIODConfig::new();
    let empty = config.compute(&[]).unwrap();
    assert_eq!(empty.range(), OutputRange::empty());
    assert!(empty.values().is_empty());

    assert_eq!(
        config
            .compute(&[1.0 as Float; HT_DCPERIOD_LOOKBACK])
            .unwrap_err(),
        TalibError::InsufficientData {
            required: HT_DCPERIOD_LOOKBACK + 1,
            actual: HT_DCPERIOD_LOOKBACK,
        }
    );

    let valid = [1.0 as Float; HT_DCPERIOD_LOOKBACK + 1];
    let mut missing_output = [];
    assert!(matches!(
        config
            .compute_into(&valid, &mut missing_output)
            .unwrap_err(),
        TalibError::InvalidInput { .. }
    ));

    let mut runner = config.prepare_batch(valid.len() - 1).unwrap();
    let mut output = [123.0 as Float; 1];
    assert_eq!(
        runner.compute_into(&valid, &mut output).unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: valid.len() - 1,
            actual_input_len: valid.len(),
        }
    );
    assert_eq!(output, [123.0 as Float; 1]);

    let case = &reference::CASES[2];
    let input = as_float(case.input);
    let mut stream = config.stream().unwrap();
    assert!(matches!(
        stream.next(Float::NAN).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    let streamed = input
        .iter()
        .copied()
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(
        &streamed,
        case.expected,
        "streaming after rejected non-finite tick",
    );

    let mut batch_output = vec![123.0 as Float; case.expected.len()];
    let mut non_finite = input;
    non_finite[40] = Float::INFINITY;
    assert!(matches!(
        config
            .compute_into(&non_finite, batch_output.as_mut_slice())
            .unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(batch_output.iter().all(|&value| value == 123.0 as Float));
}

#[test]
fn ht_dcperiod_converges_on_known_stationary_cycles() {
    let config = HT_DCPERIODConfig::new();
    for expected_period in [10.0 as Float, 20.0 as Float, 40.0 as Float] {
        let input = (0..512)
            .map(|index| {
                100.0 as Float
                    + 7.0 as Float
                        * (2.0 as Float * core::f64::consts::PI as Float * index as Float
                            / expected_period)
                            .sin()
            })
            .collect::<Vec<_>>();
        let output = config.compute(input.as_slice()).unwrap();
        let settled = &output.values()[output.values().len() - 64..];
        let estimated = settled.iter().sum::<Float>() / settled.len() as Float;
        assert!(
            (estimated - expected_period).abs() <= 1.5 as Float,
            "expected period {expected_period}, estimated {estimated}"
        );
        assert!(
            output
                .values()
                .iter()
                .all(|&period| (6.0 as Float..=50.0 as Float).contains(&period)),
            "period bounds for {expected_period}"
        );
    }
}

fn assert_execution_types<C, R, S>()
where
    C: IndicatorConfig<BatchRunner = R, Stream = S>,
    R: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn ht_dcperiod_execution_types_are_publicly_wired() {
    assert_execution_types::<HT_DCPERIODConfig, HT_DCPERIODBatchRunner, HT_DCPERIODStream>();
}
