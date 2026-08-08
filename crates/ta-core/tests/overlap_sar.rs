#[path = "fixtures/sar_reference.rs"]
mod reference;

use ta_core::inventory::{function, ImplementationStatus};
use ta_core::overlap::{
    SARConfig, SAREXTConfig, SARInput, SARTick, SAR, SAREXT, SAR_DEFAULT_ACCELERATION,
    SAR_DEFAULT_MAXIMUM,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

fn prices(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn tolerance(expected: f64) -> Float {
    #[cfg(feature = "f32")]
    {
        (2.0e-5_f64 * expected.abs().max(1.0)) as Float
    }
    #[cfg(not(feature = "f32"))]
    {
        (2.0e-12_f64 * expected.abs().max(1.0)) as Float
    }
}

fn assert_reference(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected as Float).abs() <= tolerance(expected),
            "index {index}: actual={actual}, expected={expected}"
        );
    }
}

fn input<'a>(high: &'a [Float], low: &'a [Float]) -> SARInput<'a> {
    SARInput { high, low }
}

fn ticks<'a>(high: &'a [Float], low: &'a [Float]) -> impl Iterator<Item = SARTick> + 'a {
    high.iter()
        .zip(low)
        .map(|(&high, &low)| SARTick { high, low })
}

#[test]
fn sar_matches_pinned_talib_wilder_vector_in_every_execution_mode() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(
        reference::TALIB_SOURCE_ARCHIVE_SHA256,
        "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
    );
    let high = prices(reference::HIGH);
    let low = prices(reference::LOW);
    let config = SARConfig::default();
    assert_eq!(config.acceleration(), SAR_DEFAULT_ACCELERATION);
    assert_eq!(config.maximum(), SAR_DEFAULT_MAXIMUM);
    assert_eq!(config.lookback(), reference::OUTPUT_BEGIN);

    let owned = config.compute(input(&high, &low)).unwrap();
    assert_eq!(owned.source_len(), high.len());
    assert_eq!(owned.range(), OutputRange::new(1, high.len() - 1));
    assert_reference(owned.values(), reference::SAR_DEFAULT);

    let mut caller = vec![-777.0 as Float; high.len()];
    let range = config
        .compute_into(input(&high, &low), caller.as_mut_slice())
        .unwrap();
    assert_eq!(range, owned.range());
    assert_reference(&caller[..range.nb_element], reference::SAR_DEFAULT);
    assert_eq!(caller[range.nb_element], -777.0 as Float);

    let mut direct = vec![0.0 as Float; high.len() - 1];
    assert_eq!(
        SAR(
            &high,
            &low,
            SAR_DEFAULT_ACCELERATION,
            SAR_DEFAULT_MAXIMUM,
            &mut direct,
        )
        .unwrap(),
        owned.range()
    );
    assert_reference(&direct, reference::SAR_DEFAULT);

    let mut runner = config.prepare_batch(high.len()).unwrap();
    let mut prepared = vec![0.0 as Float; high.len() - 1];
    assert_eq!(
        runner
            .compute_into(input(&high, &low), prepared.as_mut_slice())
            .unwrap(),
        owned.range()
    );
    assert_eq!(prepared, direct);

    let mut stream = config.stream().unwrap();
    let streamed = ticks(&high, &low)
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(streamed, direct);
    stream.reset();
    let replay = ticks(&high, &low)
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(replay, direct);

    for (source_index, &stop) in direct.iter().enumerate() {
        let source_index = source_index + 1;
        assert!(
            stop <= low[source_index] || stop >= high[source_index],
            "SAR at {source_index} must remain outside the current high/low interval"
        );
    }
}

#[test]
fn sarext_custom_long_short_dynamics_match_pinned_reference() {
    let high = prices(reference::HIGH);
    let low = prices(reference::LOW);
    let [start, offset, init_long, long, max_long, init_short, short, max_short] =
        reference::SAREXT_PARAMETERS.map(|value| value as Float);
    let config = SAREXTConfig::new(
        start, offset, init_long, long, max_long, init_short, short, max_short,
    )
    .unwrap();

    assert_eq!(config.start_value(), start);
    assert_eq!(config.offset_on_reverse(), offset);
    assert_eq!(config.acceleration_init_long(), init_long);
    assert_eq!(config.acceleration_long(), long);
    assert_eq!(config.acceleration_max_long(), max_long);
    assert_eq!(config.acceleration_init_short(), init_short);
    assert_eq!(config.acceleration_short(), short);
    assert_eq!(config.acceleration_max_short(), max_short);

    let owned = config.compute(input(&high, &low)).unwrap();
    assert_eq!(owned.range(), OutputRange::new(1, high.len() - 1));
    assert_reference(owned.values(), reference::SAREXT_CUSTOM);

    let mut direct = vec![0.0 as Float; high.len() - 1];
    SAREXT(
        &high,
        &low,
        start,
        offset,
        init_long,
        long,
        max_long,
        init_short,
        short,
        max_short,
        &mut direct,
    )
    .unwrap();
    assert_eq!(&direct, owned.values());

    let mut runner = config.prepare_batch(high.len()).unwrap();
    let mut prepared = vec![0.0 as Float; high.len() - 1];
    runner
        .compute_into(input(&high, &low), prepared.as_mut_slice())
        .unwrap();
    assert_eq!(prepared, direct);

    let mut stream = config.stream().unwrap();
    let streamed = ticks(&high, &low)
        .filter_map(|tick| stream.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(streamed, direct);

    for (output_index, &signed_stop) in direct.iter().enumerate() {
        let source_index = output_index + 1;
        if signed_stop >= 0.0 as Float {
            assert!(signed_stop <= low[source_index]);
        } else {
            assert!(-signed_stop >= high[source_index]);
        }
    }
}

#[test]
fn sarext_is_not_a_shallow_sar_alias() {
    let high = prices(reference::HIGH);
    let low = prices(reference::LOW);
    let sar = SARConfig::default()
        .compute(input(&high, &low))
        .unwrap()
        .into_values();
    let sarext_default = SAREXTConfig::default()
        .compute(input(&high, &low))
        .unwrap()
        .into_values();

    assert_eq!(sar.len(), sarext_default.len());
    assert!(sarext_default.iter().any(|value| *value < 0.0 as Float));
    for (&sar, &signed_sarext) in sar.iter().zip(&sarext_default) {
        assert!((sar - signed_sarext.abs()).abs() <= tolerance(sar as f64));
    }

    let custom = SAREXTConfig::new(
        49.0 as Float,
        0.03 as Float,
        0.01 as Float,
        0.01 as Float,
        0.1 as Float,
        0.08 as Float,
        0.03 as Float,
        0.3 as Float,
    )
    .unwrap()
    .compute(input(&high, &low))
    .unwrap()
    .into_values();
    assert_ne!(custom, sarext_default);
}

#[test]
fn configurations_reject_nonfinite_negative_and_incoherent_dynamics() {
    assert!(SARConfig::new(Float::NAN, 0.2 as Float).is_err());
    assert!(SARConfig::new(-0.01 as Float, 0.2 as Float).is_err());
    assert!(SARConfig::new(0.21 as Float, 0.2 as Float).is_err());
    assert!(SAREXTConfig::new(
        Float::INFINITY,
        0.0 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float
    )
    .is_err());
    assert!(SAREXTConfig::new(
        0.0 as Float,
        -0.01 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float
    )
    .is_err());
    assert!(SAREXTConfig::new(
        0.0 as Float,
        0.0 as Float,
        0.21 as Float,
        0.02 as Float,
        0.2 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float
    )
    .is_err());
    assert!(SAREXTConfig::new(
        0.0 as Float,
        0.0 as Float,
        0.02 as Float,
        0.21 as Float,
        0.2 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float
    )
    .is_err());
    assert!(SAREXTConfig::new(
        0.0 as Float,
        0.0 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float,
        0.21 as Float,
        0.02 as Float,
        0.2 as Float
    )
    .is_err());
    assert!(SAREXTConfig::new(
        0.0 as Float,
        0.0 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float,
        0.02 as Float,
        0.21 as Float,
        0.2 as Float
    )
    .is_err());
}

#[test]
fn batch_failures_preserve_caller_output() {
    let config = SARConfig::default();
    let mut output = [91.0 as Float; 4];
    let sentinel = output;

    assert!(config
        .compute_into(input(&[2.0 as Float], &[1.0 as Float]), &mut output)
        .is_err());
    assert_eq!(output, sentinel);
    assert!(config
        .compute_into(
            input(&[2.0 as Float, 3.0 as Float], &[1.0 as Float]),
            &mut output
        )
        .is_err());
    assert_eq!(output, sentinel);
    assert!(config
        .compute_into(
            input(&[2.0 as Float, Float::NAN], &[1.0 as Float, 2.0 as Float]),
            &mut output
        )
        .is_err());
    assert_eq!(output, sentinel);

    let mut too_small: [Float; 0] = [];
    assert!(config
        .compute_into(
            input(&[2.0 as Float, 3.0 as Float], &[1.0 as Float, 2.0 as Float]),
            &mut too_small
        )
        .is_err());
    assert_eq!(output, sentinel);
    let ext = SAREXTConfig::default();
    assert!(ext
        .compute_into(
            input(
                &[2.0 as Float, Float::INFINITY],
                &[1.0 as Float, 2.0 as Float],
            ),
            &mut output,
        )
        .is_err());
    assert_eq!(output, sentinel);
    assert!(ext
        .compute_into(input(&[2.0 as Float], &[1.0 as Float]), &mut output)
        .is_err());
    assert_eq!(output, sentinel);
    assert!(ext
        .compute_into(
            input(&[2.0 as Float, 3.0 as Float], &[1.0 as Float]),
            &mut output,
        )
        .is_err());
    assert_eq!(output, sentinel);
    assert!(ext
        .compute_into(
            input(&[2.0 as Float, 3.0 as Float], &[1.0 as Float, 2.0 as Float],),
            &mut too_small,
        )
        .is_err());

    let mut direct = output;
    assert!(SAR(
        &[2.0 as Float, 3.0 as Float],
        &[1.0 as Float, 2.0 as Float],
        0.3 as Float,
        0.2 as Float,
        &mut direct
    )
    .is_err());
    assert_eq!(direct, sentinel);
    assert!(SAREXT(
        &[2.0 as Float, 3.0 as Float],
        &[1.0 as Float, 2.0 as Float],
        0.0 as Float,
        0.0 as Float,
        0.3 as Float,
        0.02 as Float,
        0.2 as Float,
        0.02 as Float,
        0.02 as Float,
        0.2 as Float,
        &mut direct,
    )
    .is_err());
    assert_eq!(direct, sentinel);
}

#[test]
fn prepared_capacity_and_stream_errors_preserve_execution_state() {
    let high = prices(reference::HIGH);
    let low = prices(reference::LOW);
    let config = SARConfig::default();
    let mut runner = config.prepare_batch(high.len()).unwrap();
    let mut output = vec![37.0 as Float; high.len() - 1];
    let before = output.clone();
    let oversized_high = vec![1.0 as Float; high.len() + 1];
    let oversized_low = vec![0.0 as Float; low.len() + 1];
    assert_eq!(
        runner.compute_into(input(&oversized_high, &oversized_low), &mut output),
        Err(TalibError::prepared_capacity_exceeded(
            high.len(),
            high.len() + 1
        ))
    );
    assert_eq!(output, before);
    runner
        .compute_into(input(&high, &low), &mut output)
        .unwrap();
    assert_reference(&output, reference::SAR_DEFAULT);

    let first = SARTick {
        high: high[0],
        low: low[0],
    };
    let second = SARTick {
        high: high[1],
        low: low[1],
    };
    let mut clean = config.stream().unwrap();
    let mut rejected = config.stream().unwrap();
    assert_eq!(clean.next(first).unwrap(), None);
    assert_eq!(rejected.next(first).unwrap(), None);
    assert!(rejected
        .next(SARTick {
            high: Float::NAN,
            low: low[1]
        })
        .is_err());
    assert_eq!(rejected.next(second).unwrap(), clean.next(second).unwrap());

    let ext = SAREXTConfig::default();
    let mut ext_runner = ext.prepare_batch(high.len()).unwrap();
    let mut ext_output = before.clone();
    assert_eq!(
        ext_runner.compute_into(input(&oversized_high, &oversized_low), &mut ext_output),
        Err(TalibError::prepared_capacity_exceeded(
            high.len(),
            high.len() + 1
        ))
    );
    assert_eq!(ext_output, before);
    ext_runner
        .compute_into(input(&high, &low), &mut ext_output)
        .unwrap();
    assert_eq!(
        ext_output,
        ext.compute(input(&high, &low)).unwrap().into_values()
    );

    let mut ext_clean = ext.stream().unwrap();
    let mut ext_rejected = ext.stream().unwrap();
    assert_eq!(ext_clean.next(first).unwrap(), None);
    assert_eq!(ext_rejected.next(first).unwrap(), None);
    assert!(ext_rejected
        .next(SARTick {
            high: high[1],
            low: Float::NEG_INFINITY,
        })
        .is_err());
    assert_eq!(
        ext_rejected.next(second).unwrap(),
        ext_clean.next(second).unwrap()
    );
}

#[test]
fn runners_streams_and_reset_have_isolated_recursive_state() {
    let high = prices(reference::HIGH);
    let low = prices(reference::LOW);
    let config = SAREXTConfig::default();
    let mut left = config.stream().unwrap();
    let mut right = config.stream().unwrap();

    let first_pass = ticks(&high, &low)
        .filter_map(|tick| left.next(tick).unwrap())
        .collect::<Vec<_>>();
    let right_pass = ticks(&high, &low)
        .filter_map(|tick| right.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(first_pass, right_pass);

    left.reset();
    let replay = ticks(&high, &low)
        .filter_map(|tick| left.next(tick).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(replay, first_pass);

    let mut runner_a = config.prepare_batch(high.len()).unwrap();
    let mut runner_b = runner_a.clone();
    let mut output_a = vec![0.0 as Float; high.len() - 1];
    let mut output_b = vec![0.0 as Float; high.len() - 1];
    runner_a
        .compute_into(input(&high, &low), &mut output_a)
        .unwrap();
    runner_b
        .compute_into(input(&high, &low), &mut output_b)
        .unwrap();
    assert_eq!(output_a, output_b);
}

#[test]
fn catalogue_inventory_marks_both_definitions_implemented() {
    for name in ["SAR", "SAREXT"] {
        assert_eq!(
            function(name).unwrap().status,
            ImplementationStatus::Implemented
        );
    }
}
