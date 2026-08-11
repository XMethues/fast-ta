// This shared generated input fixture contains provenance fields used by sibling suites.
#[allow(dead_code)]
#[path = "fixtures/ht_dcperiod_reference.rs"]
mod inputs;
#[path = "fixtures/ht_hilbert_outputs_reference.rs"]
mod reference;

use ta_core::{
    cycle::{
        HT_DCPHASEBatchRunner, HT_DCPHASEConfig, HT_DCPHASEStream, HT_PHASORBatchRunner,
        HT_PHASORConfig, HT_PHASORStream, HT_PHASORValue, HT_PHASORValuesMut, HT_SINEBatchRunner,
        HT_SINEConfig, HT_SINEStream, HT_SINEValue, HT_SINEValuesMut, HT_DCPHASE,
        HT_DCPHASE_LOOKBACK, HT_PHASOR, HT_PHASOR_LOOKBACK, HT_SINE, HT_SINE_LOOKBACK,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

#[cfg(feature = "f32")]
const VALUE_ABS_TOLERANCE: Float = 5.0e-3;
#[cfg(not(feature = "f32"))]
const VALUE_ABS_TOLERANCE: Float = 1.0e-9;
#[cfg(feature = "f32")]
const PHASE_ABS_TOLERANCE_DEGREES: Float = 5.0e-3;
#[cfg(not(feature = "f32"))]
const PHASE_ABS_TOLERANCE_DEGREES: Float = 1.0e-9;
#[cfg(feature = "f32")]
const REL_TOLERANCE: Float = 5.0e-4;
#[cfg(not(feature = "f32"))]
const REL_TOLERANCE: Float = 1.0e-12;

fn as_float(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn as_f64(value: Float) -> f64 {
    #[cfg(feature = "f32")]
    {
        f64::from(value)
    }
    #[cfg(not(feature = "f32"))]
    {
        value
    }
}

fn assert_value_close(actual: Float, expected: f64, context: &str) {
    let expected = expected as Float;
    let difference = (actual - expected).abs();
    let tolerance = VALUE_ABS_TOLERANCE.max(REL_TOLERANCE * expected.abs());
    assert!(
        difference <= tolerance,
        "{context}: expected {expected}, got {actual}, difference {difference}, tolerance {tolerance}"
    );
}

fn assert_values_close(actual: &[Float], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_value_close(
            actual,
            expected,
            &format!("{context}, compact index {index}"),
        );
    }
}

fn assert_phase_close(actual: Float, expected: f64, context: &str) {
    let expected = expected as Float;
    let difference =
        (actual - expected + 180.0 as Float).rem_euclid(360.0 as Float) - 180.0 as Float;
    assert!(
        difference.abs() <= PHASE_ABS_TOLERANCE_DEGREES,
        "{context}: expected {expected} degrees, got {actual}, circular difference {difference}"
    );
}

fn assert_phases_close(actual: &[Float], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_phase_close(
            actual,
            expected,
            &format!("{context}, compact index {index}"),
        );
    }
}

const LONG_PHASE_SOURCE_LEN: usize = 4_096;
const LONG_PHASE_ORACLE_INDICES: [usize; 8] = [0, 1, 31, 127, 511, 1_023, 2_047, 4_032];
// Selected compact outputs from the checksum-pinned TA-Lib 0.6.4 C functions
// over catalogue_fixture_v1; the f32 feature uses TA_S_HT_DCPHASE.

#[cfg(not(feature = "f32"))]
const LONG_PHASE_ORACLE: [f64; 8] = [
    260.21125582216257,
    307.88798701174403,
    -34.03849402095011,
    218.8648329912454,
    206.3619733224249,
    134.16938554420938,
    184.15718428019892,
    68.00605503619721,
];

#[cfg(feature = "f32")]
const LONG_PHASE_ORACLE: [f64; 8] = [
    260.2112592713675,
    307.88799025499213,
    -34.03849812173746,
    218.8648333548112,
    206.3619686339956,
    134.16938112475552,
    184.1571875799217,
    68.00605342058999,
];

fn long_catalogue_phase_input() -> Vec<Float> {
    (0..LONG_PHASE_SOURCE_LEN)
        .map(|index| (index as f64 * 0.001 + ((index * 37) % 101) as f64 + 1.0) as Float)
        .collect()
}

#[test]
fn dynamic_period_history_matches_long_talib_phase_oracle() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    let input = long_catalogue_phase_input();
    let output = HT_DCPHASEConfig::new().compute(&input).unwrap();
    assert_eq!(
        output.range(),
        OutputRange::new(
            HT_DCPHASE_LOOKBACK,
            LONG_PHASE_SOURCE_LEN - HT_DCPHASE_LOOKBACK,
        )
    );

    for (&index, &expected) in LONG_PHASE_ORACLE_INDICES.iter().zip(&LONG_PHASE_ORACLE) {
        assert_phase_close(
            output.values()[index],
            expected,
            &format!("long TA-Lib phase oracle, compact index {index}"),
        );
    }
}

#[test]
fn long_phase_history_preserves_execution_state_and_reset() {
    let input = long_catalogue_phase_input();
    let config = HT_DCPHASEConfig::new();
    let owned = config.compute(&input).unwrap();

    let mut caller_owned = vec![0.0 as Float; owned.values().len()];
    assert_eq!(
        HT_DCPHASE(&input, &mut caller_owned).unwrap(),
        owned.range()
    );
    assert_eq!(caller_owned.as_slice(), owned.values());

    let mut runner = config.prepare_batch(input.len()).unwrap();
    let mut prepared = vec![0.0 as Float; owned.values().len()];
    for _ in 0..2 {
        assert_eq!(
            runner.compute_into(&input, &mut prepared).unwrap(),
            owned.range()
        );
        assert_eq!(prepared.as_slice(), owned.values());
    }

    let mut stream = config.stream().unwrap();
    let streamed = input
        .iter()
        .copied()
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(streamed.as_slice(), owned.values());
    stream.reset();
    let replayed = input
        .iter()
        .copied()
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(replayed.as_slice(), owned.values());
}

#[test]
fn hilbert_outputs_match_checksum_pinned_talib_vectors_in_every_execution_mode() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    assert_eq!(reference::SOURCE_LENGTH, 256);
    assert_eq!(reference::NOISE_SEED, 0x5EED_C0DE);
    assert_eq!(reference::PHASE_OUTPUT_BEGIN, HT_DCPHASE_LOOKBACK);
    assert_eq!(reference::PHASOR_OUTPUT_BEGIN, HT_PHASOR_LOOKBACK);
    assert_eq!(HT_DCPHASE_LOOKBACK, HT_SINE_LOOKBACK);
    assert_eq!(inputs::CASES.len(), reference::CASES.len());

    for (input_case, expected) in inputs::CASES.iter().zip(reference::CASES) {
        assert_eq!(input_case.name, expected.name);
        assert_eq!(input_case.definition, expected.definition);
        let input = as_float(input_case.input);
        let phase_range = OutputRange::new(HT_DCPHASE_LOOKBACK, expected.phase.len());
        let phasor_range = OutputRange::new(HT_PHASOR_LOOKBACK, expected.in_phase.len());
        let context = format!("{} ({})", expected.name, expected.definition);

        let phase_config = HT_DCPHASEConfig::new();
        assert_eq!(phase_config.lookback(), HT_DCPHASE_LOOKBACK);
        let owned_phase = phase_config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned_phase.source_len(),
            input.len(),
            "{context}, phase owned"
        );
        assert_eq!(owned_phase.range(), phase_range, "{context}, phase owned");
        assert_phases_close(
            owned_phase.values(),
            expected.phase,
            &format!("{context}, phase owned"),
        );

        let mut caller_phase = vec![0.0 as Float; expected.phase.len()];
        assert_eq!(HT_DCPHASE(&input, &mut caller_phase).unwrap(), phase_range);
        assert_phases_close(
            &caller_phase,
            expected.phase,
            &format!("{context}, phase caller-owned"),
        );

        let mut phase_runner = phase_config.prepare_batch(input.len()).unwrap();
        let mut prepared_phase = vec![0.0 as Float; expected.phase.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                phase_runner
                    .compute_into(&input, &mut prepared_phase)
                    .unwrap(),
                phase_range,
                "{context}, phase prepared {pass}"
            );
            assert_phases_close(
                &prepared_phase,
                expected.phase,
                &format!("{context}, phase prepared {pass}"),
            );
        }

        let mut phase_stream = phase_config.stream().unwrap();
        let streamed_phase = input
            .iter()
            .copied()
            .filter_map(|tick| phase_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_phases_close(
            &streamed_phase,
            expected.phase,
            &format!("{context}, phase stream"),
        );
        phase_stream.reset();
        let replayed_phase = input
            .iter()
            .copied()
            .filter_map(|tick| phase_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            streamed_phase, replayed_phase,
            "{context}, phase reset replay"
        );

        let phasor_config = HT_PHASORConfig::new();
        assert_eq!(phasor_config.lookback(), HT_PHASOR_LOOKBACK);
        let owned_phasor = phasor_config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned_phasor.source_len(),
            input.len(),
            "{context}, phasor owned"
        );
        assert_eq!(
            owned_phasor.range(),
            phasor_range,
            "{context}, phasor owned"
        );
        assert_values_close(
            &owned_phasor.values().in_phase,
            expected.in_phase,
            &format!("{context}, InPhase owned"),
        );
        assert_values_close(
            &owned_phasor.values().quadrature,
            expected.quadrature,
            &format!("{context}, Quadrature owned"),
        );

        let mut caller_in_phase = vec![0.0 as Float; expected.in_phase.len()];
        let mut caller_quadrature = vec![0.0 as Float; expected.quadrature.len()];
        assert_eq!(
            HT_PHASOR(&input, &mut caller_in_phase, &mut caller_quadrature).unwrap(),
            phasor_range
        );
        assert_values_close(
            &caller_in_phase,
            expected.in_phase,
            &format!("{context}, InPhase caller-owned"),
        );
        assert_values_close(
            &caller_quadrature,
            expected.quadrature,
            &format!("{context}, Quadrature caller-owned"),
        );

        let mut phasor_runner = phasor_config.prepare_batch(input.len()).unwrap();
        let mut prepared_in_phase = vec![0.0 as Float; expected.in_phase.len()];
        let mut prepared_quadrature = vec![0.0 as Float; expected.quadrature.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                phasor_runner
                    .compute_into(
                        &input,
                        HT_PHASORValuesMut {
                            in_phase: &mut prepared_in_phase,
                            quadrature: &mut prepared_quadrature,
                        },
                    )
                    .unwrap(),
                phasor_range,
                "{context}, phasor prepared {pass}"
            );
            assert_values_close(
                &prepared_in_phase,
                expected.in_phase,
                &format!("{context}, InPhase prepared {pass}"),
            );
            assert_values_close(
                &prepared_quadrature,
                expected.quadrature,
                &format!("{context}, Quadrature prepared {pass}"),
            );
        }

        let mut phasor_stream = phasor_config.stream().unwrap();
        let streamed_phasor = input
            .iter()
            .copied()
            .filter_map(|tick| phasor_stream.next(tick).unwrap())
            .collect::<Vec<HT_PHASORValue>>();
        assert_values_close(
            &streamed_phasor
                .iter()
                .map(|value| value.in_phase)
                .collect::<Vec<_>>(),
            expected.in_phase,
            &format!("{context}, InPhase stream"),
        );
        assert_values_close(
            &streamed_phasor
                .iter()
                .map(|value| value.quadrature)
                .collect::<Vec<_>>(),
            expected.quadrature,
            &format!("{context}, Quadrature stream"),
        );
        phasor_stream.reset();
        let replayed_phasor = input
            .iter()
            .copied()
            .filter_map(|tick| phasor_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            streamed_phasor, replayed_phasor,
            "{context}, phasor reset replay"
        );

        let sine_config = HT_SINEConfig::new();
        assert_eq!(sine_config.lookback(), HT_SINE_LOOKBACK);
        let owned_sine = sine_config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned_sine.source_len(),
            input.len(),
            "{context}, sine owned"
        );
        assert_eq!(owned_sine.range(), phase_range, "{context}, sine owned");
        assert_values_close(
            &owned_sine.values().sine,
            expected.sine,
            &format!("{context}, Sine owned"),
        );
        assert_values_close(
            &owned_sine.values().lead_sine,
            expected.lead_sine,
            &format!("{context}, LeadSine owned"),
        );

        let mut caller_sine = vec![0.0 as Float; expected.sine.len()];
        let mut caller_lead_sine = vec![0.0 as Float; expected.lead_sine.len()];
        assert_eq!(
            HT_SINE(&input, &mut caller_sine, &mut caller_lead_sine).unwrap(),
            phase_range
        );
        assert_values_close(
            &caller_sine,
            expected.sine,
            &format!("{context}, Sine caller-owned"),
        );
        assert_values_close(
            &caller_lead_sine,
            expected.lead_sine,
            &format!("{context}, LeadSine caller-owned"),
        );

        let mut sine_runner = sine_config.prepare_batch(input.len()).unwrap();
        let mut prepared_sine = vec![0.0 as Float; expected.sine.len()];
        let mut prepared_lead_sine = vec![0.0 as Float; expected.lead_sine.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                sine_runner
                    .compute_into(
                        &input,
                        HT_SINEValuesMut {
                            sine: &mut prepared_sine,
                            lead_sine: &mut prepared_lead_sine,
                        },
                    )
                    .unwrap(),
                phase_range,
                "{context}, sine prepared {pass}"
            );
            assert_values_close(
                &prepared_sine,
                expected.sine,
                &format!("{context}, Sine prepared {pass}"),
            );
            assert_values_close(
                &prepared_lead_sine,
                expected.lead_sine,
                &format!("{context}, LeadSine prepared {pass}"),
            );
        }

        let mut sine_stream = sine_config.stream().unwrap();
        let streamed_sine = input
            .iter()
            .copied()
            .filter_map(|tick| sine_stream.next(tick).unwrap())
            .collect::<Vec<HT_SINEValue>>();
        assert_values_close(
            &streamed_sine
                .iter()
                .map(|value| value.sine)
                .collect::<Vec<_>>(),
            expected.sine,
            &format!("{context}, Sine stream"),
        );
        assert_values_close(
            &streamed_sine
                .iter()
                .map(|value| value.lead_sine)
                .collect::<Vec<_>>(),
            expected.lead_sine,
            &format!("{context}, LeadSine stream"),
        );
        sine_stream.reset();
        let replayed_sine = input
            .iter()
            .copied()
            .filter_map(|tick| sine_stream.next(tick).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(streamed_sine, replayed_sine, "{context}, sine reset replay");
    }
}

#[test]
fn phase_units_wrap_and_sine_relationships_are_explicit() {
    let input = as_float(inputs::CASES[2].input);
    let phase = HT_DCPHASEConfig::new().compute(&input).unwrap();
    let sine = HT_SINEConfig::new().compute(&input).unwrap();
    assert_eq!(phase.range(), sine.range());

    for (index, (&phase_degrees, (&actual_sine, &actual_lead))) in phase
        .values()
        .iter()
        .zip(sine.values().sine.iter().zip(&sine.values().lead_sine))
        .enumerate()
    {
        assert!(
            phase_degrees > -45.0 as Float && phase_degrees <= 315.0 as Float,
            "canonical degree wrap at compact index {index}: {phase_degrees}"
        );
        assert!(
            actual_sine.abs() <= 1.0 as Float,
            "Sine amplitude at {index}"
        );
        assert!(
            actual_lead.abs() <= 1.0 as Float,
            "LeadSine amplitude at {index}"
        );
        let radians = phase_degrees * core::f64::consts::PI as Float / 180.0 as Float;
        assert_value_close(
            actual_sine,
            as_f64(radians.sin()),
            &format!("Sine relationship at {index}"),
        );
        assert_value_close(
            actual_lead,
            as_f64((radians + core::f64::consts::FRAC_PI_4 as Float).sin()),
            &format!("LeadSine relationship at {index}"),
        );
    }
}

#[test]
fn validation_capacity_and_stream_failures_do_not_mutate_outputs_or_state() {
    let empty_phase = HT_DCPHASEConfig::new().compute(&[]).unwrap();
    assert_eq!(empty_phase.range(), OutputRange::empty());
    assert!(empty_phase.values().is_empty());
    let empty_phasor = HT_PHASORConfig::new().compute(&[]).unwrap();
    assert!(empty_phasor.values().in_phase.is_empty());
    assert!(empty_phasor.values().quadrature.is_empty());
    let empty_sine = HT_SINEConfig::new().compute(&[]).unwrap();
    assert!(empty_sine.values().sine.is_empty());
    assert!(empty_sine.values().lead_sine.is_empty());

    assert_eq!(
        HT_DCPHASEConfig::new()
            .compute(&[1.0 as Float; HT_DCPHASE_LOOKBACK])
            .unwrap_err(),
        TalibError::InsufficientData {
            required: HT_DCPHASE_LOOKBACK + 1,
            actual: HT_DCPHASE_LOOKBACK,
        }
    );

    let input = as_float(inputs::CASES[4].input);
    let phase_count = input.len() - HT_DCPHASE_LOOKBACK;
    let mut phase_output = vec![789.0 as Float; phase_count];
    let mut invalid_phase_input = input.clone();
    invalid_phase_input[100] = Float::INFINITY;
    assert!(matches!(
        HT_DCPHASE(&invalid_phase_input, &mut phase_output).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(phase_output.iter().all(|&value| value == 789.0 as Float));
    let mut phase_runner = HT_DCPHASEConfig::new()
        .prepare_batch(input.len() - 1)
        .unwrap();
    assert!(matches!(
        phase_runner
            .compute_into(&input, &mut phase_output)
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded { .. }
    ));
    assert!(phase_output.iter().all(|&value| value == 789.0 as Float));
    let phasor_count = input.len() - HT_PHASOR_LOOKBACK;
    let mut in_phase = vec![123.0 as Float; phasor_count];
    let mut short_quadrature = vec![456.0 as Float; phasor_count - 1];
    assert!(matches!(
        HT_PHASOR(&input, &mut in_phase, &mut short_quadrature).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(in_phase.iter().all(|&value| value == 123.0 as Float));
    assert!(short_quadrature
        .iter()
        .all(|&value| value == 456.0 as Float));
    let mut phasor_runner = HT_PHASORConfig::new()
        .prepare_batch(input.len() - 1)
        .unwrap();
    assert!(matches!(
        phasor_runner
            .compute_into(
                &input,
                HT_PHASORValuesMut {
                    in_phase: &mut in_phase,
                    quadrature: &mut short_quadrature,
                },
            )
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded { .. }
    ));
    assert!(in_phase.iter().all(|&value| value == 123.0 as Float));
    assert!(short_quadrature
        .iter()
        .all(|&value| value == 456.0 as Float));

    let sine_count = input.len() - HT_SINE_LOOKBACK;
    let mut sine = vec![123.0 as Float; sine_count];
    let mut lead_sine = vec![456.0 as Float; sine_count];
    let mut invalid = input.clone();
    invalid[100] = Float::NAN;
    assert!(matches!(
        HT_SINE(&invalid, &mut sine, &mut lead_sine).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(sine.iter().all(|&value| value == 123.0 as Float));
    assert!(lead_sine.iter().all(|&value| value == 456.0 as Float));

    let mut runner = HT_SINEConfig::new().prepare_batch(input.len() - 1).unwrap();
    assert_eq!(
        runner
            .compute_into(
                &input,
                HT_SINEValuesMut {
                    sine: &mut sine,
                    lead_sine: &mut lead_sine,
                },
            )
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: input.len() - 1,
            actual_input_len: input.len(),
        }
    );
    assert!(sine.iter().all(|&value| value == 123.0 as Float));
    assert!(lead_sine.iter().all(|&value| value == 456.0 as Float));

    let expected = &reference::CASES[4];
    let mut phase_stream = HT_DCPHASEConfig::new().stream().unwrap();
    assert!(matches!(
        phase_stream.next(Float::INFINITY).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    let phase = input
        .iter()
        .copied()
        .filter_map(|tick| phase_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_phases_close(&phase, expected.phase, "phase stream after rejected tick");

    let mut phasor_stream = HT_PHASORConfig::new().stream().unwrap();
    assert!(matches!(
        phasor_stream.next(Float::NAN).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));

    let phasor = input
        .iter()
        .copied()
        .filter_map(|tick| phasor_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(
        &phasor
            .iter()
            .map(|value| value.in_phase)
            .collect::<Vec<_>>(),
        expected.in_phase,
        "phasor stream after rejected tick",
    );

    let mut sine_stream = HT_SINEConfig::new().stream().unwrap();
    assert!(matches!(
        sine_stream.next(Float::NEG_INFINITY).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    let streamed_sine = input
        .iter()
        .copied()
        .filter_map(|tick| sine_stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_values_close(
        &streamed_sine
            .iter()
            .map(|value| value.sine)
            .collect::<Vec<_>>(),
        expected.sine,
        "sine stream after rejected tick",
    );
}
#[test]
fn warmup_and_first_output_are_source_aligned() {
    let input = [100.0 as Float; HT_DCPHASE_LOOKBACK + 2];
    let mut phase = HT_DCPHASEConfig::new().stream().unwrap();
    let phase_positions = input
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, tick)| phase.next(tick).unwrap().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(
        phase_positions,
        (HT_DCPHASE_LOOKBACK..input.len()).collect::<Vec<_>>()
    );
    let mut phasor = HT_PHASORConfig::new().stream().unwrap();
    let phasor_positions = input
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, tick)| phasor.next(tick).unwrap().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(
        phasor_positions,
        (HT_PHASOR_LOOKBACK..input.len()).collect::<Vec<_>>()
    );
    let mut sine = HT_SINEConfig::new().stream().unwrap();
    let sine_positions = input
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, tick)| sine.next(tick).unwrap().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(
        sine_positions,
        (HT_SINE_LOOKBACK..input.len()).collect::<Vec<_>>()
    );
}

fn assert_execution_types<C, R, S>()
where
    C: IndicatorConfig<BatchRunner = R, Stream = S>,
    R: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn execution_types_are_publicly_wired() {
    assert_execution_types::<HT_DCPHASEConfig, HT_DCPHASEBatchRunner, HT_DCPHASEStream>();
    assert_execution_types::<HT_PHASORConfig, HT_PHASORBatchRunner, HT_PHASORStream>();
    assert_execution_types::<HT_SINEConfig, HT_SINEBatchRunner, HT_SINEStream>();
}
