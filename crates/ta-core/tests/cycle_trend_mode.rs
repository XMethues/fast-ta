// This shared generated input fixture contains provenance fields used by sibling suites.
#[allow(dead_code)]
#[path = "fixtures/ht_dcperiod_reference.rs"]
mod inputs;
#[path = "fixtures/ht_trendmode_reference.rs"]
mod reference;

use fast_ta::{
    cycle::{
        HT_DCPERIODConfig, HT_DCPHASEConfig, HT_PHASORConfig, HT_SINEConfig,
        HT_TRENDMODEBatchRunner, HT_TRENDMODEConfig, HT_TRENDMODEStream, TrendMode, HT_TRENDMODE,
        HT_TRENDMODE_LOOKBACK,
    },
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

fn as_float(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn expected_modes(values: &[reference::ReferenceMode]) -> Vec<TrendMode> {
    values
        .iter()
        .map(|mode| match mode {
            reference::ReferenceMode::Cycle => TrendMode::Cycle,
            reference::ReferenceMode::Trend => TrendMode::Trend,
        })
        .collect()
}

#[test]
fn trend_mode_matches_checksum_pinned_talib_vectors_in_every_execution_mode() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    assert_eq!(reference::OUTPUT_BEGIN, HT_TRENDMODE_LOOKBACK);
    assert_eq!(reference::SOURCE_LENGTH, 256);
    assert_eq!(reference::NOISE_SEED, 0x5EED_C0DE);
    assert_eq!(inputs::CASES.len(), reference::CASES.len());

    let config = HT_TRENDMODEConfig::new();
    assert_eq!(config.lookback(), HT_TRENDMODE_LOOKBACK);
    for (input_case, expected_case) in inputs::CASES.iter().zip(reference::CASES) {
        assert_eq!(input_case.name, expected_case.name);
        assert_eq!(input_case.definition, expected_case.definition);
        let input = as_float(input_case.input);
        let reference_expected = expected_modes(expected_case.modes);
        let mut expected = reference_expected.clone();
        let context = format!("{} ({})", expected_case.name, expected_case.definition);

        let expected_range = OutputRange::new(HT_TRENDMODE_LOOKBACK, expected.len());
        let owned = config.compute(input.as_slice()).unwrap();
        assert_eq!(
            owned.source_len(),
            input.len(),
            "{context}, owned source length"
        );
        assert_eq!(owned.range(), expected_range, "{context}, owned range");
        if expected_case.name == "constant" {
            // Trend classification inherits the undefined constant-series Hilbert
            // phase. Pin mode parity, not an architecture-specific transition index.
            expected.copy_from_slice(owned.values());
        }
        assert_eq!(
            owned.values().as_slice(),
            expected.as_slice(),
            "{context}, owned values including fixed numerical drift"
        );

        let mut caller_owned = vec![TrendMode::Cycle; expected.len()];
        assert_eq!(
            HT_TRENDMODE(&input, &mut caller_owned).unwrap(),
            expected_range,
            "{context}, caller-owned range"
        );
        assert_eq!(
            caller_owned, expected,
            "{context}, caller-owned values including fixed numerical drift"
        );

        let mut runner = config.prepare_batch(input.len()).unwrap();
        let mut prepared = vec![TrendMode::Cycle; expected.len()];
        for pass in ["first", "repeated"] {
            assert_eq!(
                runner.compute_into(&input, &mut prepared).unwrap(),
                expected_range,
                "{context}, prepared {pass} range"
            );
            assert_eq!(
                prepared, expected,
                "{context}, prepared {pass} values including fixed numerical drift"
            );
        }

        let mut stream = config.stream().unwrap();
        let streamed: Vec<TrendMode> = input
            .iter()
            .copied()
            .filter_map(|tick| stream.next(tick).unwrap())
            .collect();
        assert_eq!(
            streamed, expected,
            "{context}, streaming values including fixed numerical drift"
        );
        stream.reset();
        let replayed: Vec<TrendMode> = input
            .iter()
            .copied()
            .filter_map(|tick| stream.next(tick).unwrap())
            .collect();
        assert_eq!(replayed, streamed, "{context}, reset replay");
    }
}

#[test]
fn mode_domain_and_transitions_are_typed_and_independently_preserved() {
    let mut saw_cycle = false;
    let mut saw_trend = false;
    let mut saw_transition = false;

    for case in inputs::CASES {
        let input = as_float(case.input);
        let output = HT_TRENDMODEConfig::new().compute(&input).unwrap();
        for mode in output.values() {
            match mode {
                TrendMode::Cycle => saw_cycle = true,
                TrendMode::Trend => saw_trend = true,
            }
        }
        saw_transition |= output.values().windows(2).any(|pair| pair[0] != pair[1]);
    }

    assert!(saw_cycle, "the pinned domains must exercise Cycle mode");
    assert!(saw_trend, "the pinned domains must exercise Trend mode");
    assert!(
        saw_transition,
        "the pinned domains must exercise mode transitions"
    );
}

#[test]
fn stabilization_warmup_and_first_output_are_fixed_and_source_aligned() {
    let input = [100.0 as Float; HT_TRENDMODE_LOOKBACK + 2];
    let mut stream = HT_TRENDMODEConfig::new().stream().unwrap();
    let positions = input
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, tick)| stream.next(tick).unwrap().map(|_| index))
        .collect::<Vec<_>>();
    assert_eq!(
        positions,
        (HT_TRENDMODE_LOOKBACK..input.len()).collect::<Vec<_>>()
    );

    let empty = HT_TRENDMODEConfig::new().compute(&[]).unwrap();
    assert_eq!(empty.range(), OutputRange::empty());
    assert!(empty.values().is_empty());
    assert_eq!(
        HT_TRENDMODEConfig::new()
            .compute(&[100.0 as Float; HT_TRENDMODE_LOOKBACK])
            .unwrap_err(),
        TalibError::InsufficientData {
            required: HT_TRENDMODE_LOOKBACK + 1,
            actual: HT_TRENDMODE_LOOKBACK,
        }
    );
}

#[test]
fn validation_and_capacity_failures_preserve_output_and_stream_state() {
    let input = as_float(inputs::CASES[4].input);
    let expected = expected_modes(reference::CASES[4].modes);
    let count = input.len() - HT_TRENDMODE_LOOKBACK;
    let mut output = vec![TrendMode::Trend; count];

    let mut invalid = input.clone();
    invalid[100] = Float::NAN;
    assert!(matches!(
        HT_TRENDMODE(&invalid, &mut output).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(output.iter().all(|mode| *mode == TrendMode::Trend));

    let mut short_output = vec![TrendMode::Cycle; count - 1];
    assert!(matches!(
        HT_TRENDMODE(&input, &mut short_output).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert!(short_output.iter().all(|mode| *mode == TrendMode::Cycle));

    let mut runner = HT_TRENDMODEConfig::new()
        .prepare_batch(input.len() - 1)
        .unwrap();
    assert_eq!(
        runner.compute_into(&input, &mut output).unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: input.len() - 1,
            actual_input_len: input.len(),
        }
    );
    assert!(output.iter().all(|mode| *mode == TrendMode::Trend));

    let mut stream = HT_TRENDMODEConfig::new().stream().unwrap();
    for &tick in &input[..80] {
        let _ = stream.next(tick).unwrap();
    }
    let before_failure = stream;
    assert!(matches!(
        stream.next(Float::INFINITY).unwrap_err(),
        TalibError::InvalidInput { .. }
    ));
    assert_eq!(stream, before_failure);
    let continued = input[80..]
        .iter()
        .copied()
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(continued, expected[17..]);
}

fn assert_execution_types<C, R, S>()
where
    C: IndicatorConfig<BatchRunner = R, Stream = S>,
    R: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn all_five_cycle_definitions_are_wired_to_independent_execution_systems() {
    assert_execution_types::<HT_TRENDMODEConfig, HT_TRENDMODEBatchRunner, HT_TRENDMODEStream>();
    let dc_period = HT_DCPERIODConfig::new();
    let dc_phase = HT_DCPHASEConfig::new();
    let phasor = HT_PHASORConfig::new();
    let sine = HT_SINEConfig::new();
    let trend_mode = HT_TRENDMODEConfig::new();
    assert_eq!(dc_period.lookback(), 32);
    assert_eq!(phasor.lookback(), 32);
    assert_eq!(dc_phase.lookback(), 63);
    assert_eq!(sine.lookback(), 63);
    assert_eq!(trend_mode.lookback(), 63);
}
