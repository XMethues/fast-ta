#[path = "fixtures/directional_movement_reference.rs"]
mod reference;

use ta_core::momentum::{
    ADXConfig, ADXRConfig, DXConfig, DirectionalInput, DirectionalTick, MINUS_DIConfig,
    MINUS_DMConfig, PLUS_DIConfig, PLUS_DMConfig, ADX, ADXR, DX, MINUS_DI, MINUS_DM, PLUS_DI,
    PLUS_DM,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

fn observations() -> (Vec<Float>, Vec<Float>, Vec<Float>) {
    (
        reference::HIGH
            .iter()
            .map(|&value| value as Float)
            .collect(),
        reference::LOW.iter().map(|&value| value as Float).collect(),
        reference::CLOSE
            .iter()
            .map(|&value| value as Float)
            .collect(),
    )
}

fn tolerance() -> Float {
    if core::mem::size_of::<Float>() == 4 {
        3e-4 as Float
    } else {
        3e-10 as Float
    }
}

fn assert_values(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected as Float).abs() <= tolerance(),
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

macro_rules! reference_contract {
    ($test:ident, $config:ty, $function:ident, $expected:ident, $lookback:expr) => {
        #[test]
        fn $test() {
            let (high, low, close) = observations();
            let input = DirectionalInput {
                high: &high,
                low: &low,
                close: &close,
            };
            let config = <$config>::new(reference::PERIOD).unwrap();
            assert_eq!(config.period(), reference::PERIOD);
            assert_eq!(config.warm_up(), $lookback);
            assert_eq!(config.stabilization(), 0);
            assert_eq!(IndicatorConfig::lookback(&config), $lookback);

            let owned = IndicatorConfig::compute(&config, input).unwrap();
            assert_eq!(
                owned.range(),
                OutputRange::new($lookback, reference::$expected.len())
            );
            assert_eq!(owned.source_len(), high.len());
            assert_values(owned.values(), reference::$expected);

            let mut caller = vec![-1.0 as Float; reference::$expected.len()];
            let range = $function(&high, &low, &close, reference::PERIOD, &mut caller).unwrap();
            assert_eq!(range, owned.range());
            assert_values(&caller, reference::$expected);
            assert_eq!(caller.as_slice(), owned.values().as_slice());

            let mut runner = IndicatorConfig::prepare_batch(&config, high.len()).unwrap();
            assert_eq!(
                PreparedBatchRunner::<$config>::max_input_len(&runner),
                high.len()
            );
            let mut prepared = vec![-1.0 as Float; reference::$expected.len()];
            let prepared_range = PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                input,
                prepared.as_mut_slice(),
            )
            .unwrap();
            assert_eq!(prepared_range, range);
            assert_values(&prepared, reference::$expected);
            assert_eq!(prepared, caller);

            let mut stream = IndicatorConfig::stream(&config).unwrap();
            let streamed: Vec<Float> = high
                .iter()
                .zip(&low)
                .zip(&close)
                .filter_map(|((&high, &low), &close)| {
                    StreamingComputation::<$config>::next(
                        &mut stream,
                        DirectionalTick { high, low, close },
                    )
                    .unwrap()
                })
                .collect();
            assert_values(&streamed, reference::$expected);
            assert_eq!(streamed, caller);

            StreamingComputation::<$config>::reset(&mut stream);
            let after_reset: Vec<Float> = high
                .iter()
                .zip(&low)
                .zip(&close)
                .filter_map(|((&high, &low), &close)| {
                    StreamingComputation::<$config>::next(
                        &mut stream,
                        DirectionalTick { high, low, close },
                    )
                    .unwrap()
                })
                .collect();
            assert_values(&after_reset, reference::$expected);
            assert_eq!(after_reset, caller);
        }
    };
}

reference_contract!(
    plus_dm_reference_and_modes,
    PLUS_DMConfig,
    PLUS_DM,
    PLUS_DM,
    4
);
reference_contract!(
    minus_dm_reference_and_modes,
    MINUS_DMConfig,
    MINUS_DM,
    MINUS_DM,
    4
);
reference_contract!(
    plus_di_reference_and_modes,
    PLUS_DIConfig,
    PLUS_DI,
    PLUS_DI,
    5
);
reference_contract!(
    minus_di_reference_and_modes,
    MINUS_DIConfig,
    MINUS_DI,
    MINUS_DI,
    5
);
reference_contract!(dx_reference_and_modes, DXConfig, DX, DX, 5);
reference_contract!(adx_reference_and_modes, ADXConfig, ADX, ADX, 9);
reference_contract!(adxr_reference_and_modes, ADXRConfig, ADXR, ADXR, 13);

#[test]
fn period_one_preserves_definition_specific_dm_and_di_alignment() {
    let high = [10.0 as Float, 12.0, 11.0, 11.0];
    let low = [8.0 as Float, 9.0, 8.0, 8.0];
    let close = [9.0 as Float, 10.0, 9.0, 9.0];
    let mut plus_dm = [0.0 as Float; 3];
    let mut minus_dm = [0.0 as Float; 3];
    let mut plus_di = [0.0 as Float; 3];
    let mut minus_di = [0.0 as Float; 3];

    assert_eq!(
        PLUS_DM(&high, &low, &close, 1, &mut plus_dm).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(
        MINUS_DM(&high, &low, &close, 1, &mut minus_dm).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(
        PLUS_DI(&high, &low, &close, 1, &mut plus_di).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_eq!(
        MINUS_DI(&high, &low, &close, 1, &mut minus_di).unwrap(),
        OutputRange::new(1, 3)
    );
    assert_values(&plus_dm, &[2.0, 0.0, 0.0]);
    assert_values(&minus_dm, &[0.0, 1.0, 0.0]);
    assert_values(&plus_di, &[200.0 / 3.0, 0.0, 0.0]);
    assert_values(&minus_di, &[0.0, 100.0 / 3.0, 0.0]);
}

#[test]
fn sign_range_symmetry_and_dependency_invariants_hold() {
    let (high, low, close) = observations();
    let reflected_high: Vec<Float> = low.iter().map(|value| -*value).collect();
    let reflected_low: Vec<Float> = high.iter().map(|value| -*value).collect();
    let reflected_close: Vec<Float> = close.iter().map(|value| -*value).collect();
    let count_dm = high.len() - 4;
    let count_di = high.len() - 5;
    let mut plus = vec![0.0 as Float; count_dm];
    let mut minus_reflected = vec![0.0 as Float; count_dm];
    let mut plus_di = vec![0.0 as Float; count_di];
    let mut minus_di = vec![0.0 as Float; count_di];
    let mut dx = vec![0.0 as Float; count_di];
    let mut reflected_dx = vec![0.0 as Float; count_di];

    PLUS_DM(&high, &low, &close, 5, &mut plus).unwrap();
    MINUS_DM(
        &reflected_high,
        &reflected_low,
        &reflected_close,
        5,
        &mut minus_reflected,
    )
    .unwrap();
    PLUS_DI(&high, &low, &close, 5, &mut plus_di).unwrap();
    MINUS_DI(&high, &low, &close, 5, &mut minus_di).unwrap();
    DX(&high, &low, &close, 5, &mut dx).unwrap();
    DX(
        &reflected_high,
        &reflected_low,
        &reflected_close,
        5,
        &mut reflected_dx,
    )
    .unwrap();

    for (&left, &right) in plus.iter().zip(&minus_reflected) {
        assert!((left - right).abs() <= tolerance());
        assert!(left >= 0.0 as Float);
    }
    for index in 0..count_di {
        let sum = plus_di[index] + minus_di[index];
        let dependent_dx = if sum == 0.0 as Float {
            0.0 as Float
        } else {
            100.0 as Float * (plus_di[index] - minus_di[index]).abs() / sum
        };
        assert!((dx[index] - dependent_dx).abs() <= tolerance());
        assert!((dx[index] - reflected_dx[index]).abs() <= tolerance());
        assert!((0.0 as Float..=100.0 as Float).contains(&plus_di[index]));
        assert!((0.0 as Float..=100.0 as Float).contains(&minus_di[index]));
        assert!((0.0 as Float..=100.0 as Float).contains(&dx[index]));
    }
}

#[test]
fn adx_batch_specialization_matches_streaming_across_periods_and_regimes() {
    let mut high = Vec::with_capacity(192);
    let mut low = Vec::with_capacity(192);
    let mut close = Vec::with_capacity(192);
    let mut previous_close = 100.0 as Float;
    for index in 0..192 {
        if index % 29 >= 25 {
            high.push(previous_close);
            low.push(previous_close);
            close.push(previous_close);
            continue;
        }
        let trend = index as Float * 0.037 as Float;
        let cycle = ((index * 17) % 23) as Float * 0.041 as Float;
        let center = 100.0 as Float + trend + cycle;
        let next_high = center + (1 + index % 5) as Float * 0.13 as Float;
        let next_low = center - (1 + index % 7) as Float * 0.11 as Float;
        let next_close = next_low + (next_high - next_low) * 0.43 as Float;
        high.push(next_high);
        low.push(next_low);
        close.push(next_close);
        previous_close = next_close;
    }

    for period in [2, 3, 14, 31] {
        let config = ADXConfig::new(period).unwrap();
        let lookback = config.lookback();
        let mut batch = vec![-1.0 as Float; high.len() - lookback];
        let range = config
            .compute_into(
                DirectionalInput {
                    high: &high,
                    low: &low,
                    close: &close,
                },
                &mut batch,
            )
            .unwrap();
        assert_eq!(range, OutputRange::new(lookback, batch.len()));

        let mut stream = config.stream().unwrap();
        let streamed: Vec<Float> = high
            .iter()
            .zip(&low)
            .zip(&close)
            .filter_map(|((&high, &low), &close)| {
                stream.next(DirectionalTick { high, low, close }).unwrap()
            })
            .collect();
        assert_eq!(batch, streamed, "period {period}");
    }
}

#[test]
fn flat_zero_true_range_produces_finite_zero_strength() {
    let values = [42.0 as Float; 20];
    let mut dx = [1.0 as Float; 15];
    let mut adx = [1.0 as Float; 11];
    let mut adxr = [1.0 as Float; 7];
    DX(&values, &values, &values, 5, &mut dx).unwrap();
    ADX(&values, &values, &values, 5, &mut adx).unwrap();
    ADXR(&values, &values, &values, 5, &mut adxr).unwrap();
    assert!(dx
        .iter()
        .chain(&adx)
        .chain(&adxr)
        .all(|&value| value == 0.0 as Float));
}

#[test]
fn validation_fails_before_output_mutation_and_preserves_stream_state() {
    let (high, low, close) = observations();
    let mut output = [7.0 as Float; 4];
    assert!(matches!(
        ADX(&high[..10], &low[..9], &close[..10], 5, &mut output),
        Err(TalibError::InvalidInput { .. })
    ));
    assert_eq!(output, [7.0 as Float; 4]);
    let mut non_finite_high = high[..10].to_vec();
    non_finite_high[5] = Float::INFINITY;
    let mut finite_output = [7.0 as Float; 1];
    assert!(ADX(
        &non_finite_high,
        &low[..10],
        &close[..10],
        5,
        &mut finite_output,
    )
    .is_err());
    assert_eq!(finite_output, [7.0 as Float; 1]);
    assert!(ADX(&high[..9], &low[..9], &close[..9], 5, &mut output).is_err());
    assert_eq!(output, [7.0 as Float; 4]);
    assert!(ADX(&high[..14], &low[..14], &close[..14], 5, &mut output[..3]).is_err());
    assert_eq!(output, [7.0 as Float; 4]);

    let config = ADXConfig::new(5).unwrap();
    let mut actual = IndicatorConfig::stream(&config).unwrap();
    let mut expected = IndicatorConfig::stream(&config).unwrap();
    for index in 0..8 {
        let tick = DirectionalTick {
            high: high[index],
            low: low[index],
            close: close[index],
        };
        assert_eq!(
            StreamingComputation::<ADXConfig>::next(&mut actual, tick).unwrap(),
            StreamingComputation::<ADXConfig>::next(&mut expected, tick).unwrap()
        );
    }
    assert!(StreamingComputation::<ADXConfig>::next(
        &mut actual,
        DirectionalTick {
            high: Float::NAN,
            low: low[8],
            close: close[8]
        },
    )
    .is_err());
    for index in 8..high.len() {
        let tick = DirectionalTick {
            high: high[index],
            low: low[index],
            close: close[index],
        };
        assert_eq!(
            StreamingComputation::<ADXConfig>::next(&mut actual, tick).unwrap(),
            StreamingComputation::<ADXConfig>::next(&mut expected, tick).unwrap()
        );
    }
}

#[test]
fn prepared_capacity_failure_does_not_touch_output() {
    let (high, low, close) = observations();
    let config = ADXRConfig::new(5).unwrap();
    let mut runner = IndicatorConfig::prepare_batch(&config, high.len() - 1).unwrap();
    let mut output = vec![9.0 as Float; reference::ADXR.len()];
    assert!(matches!(
        PreparedBatchRunner::<ADXRConfig>::compute_into(
            &mut runner,
            DirectionalInput {
                high: &high,
                low: &low,
                close: &close
            },
            output.as_mut_slice(),
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(output.iter().all(|&value| value == 9.0 as Float));

    let accepted_len = high.len() - 1;
    let mut recovered = vec![0.0 as Float; accepted_len - IndicatorConfig::lookback(&config)];
    let recovered_range = PreparedBatchRunner::<ADXRConfig>::compute_into(
        &mut runner,
        DirectionalInput {
            high: &high[..accepted_len],
            low: &low[..accepted_len],
            close: &close[..accepted_len],
        },
        recovered.as_mut_slice(),
    )
    .unwrap();
    assert_eq!(
        recovered_range,
        OutputRange::new(IndicatorConfig::lookback(&config), recovered.len())
    );
    assert_values(&recovered, &reference::ADXR[..recovered.len()]);
}

#[test]
fn prepared_capacity_precedes_directional_input_alignment() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let mut output = [];
    let config = ADXRConfig::new(2).unwrap();
    let mut runner = config.prepare_batch(within.len()).unwrap();

    assert_eq!(
        runner
            .compute_into(
                DirectionalInput {
                    high: &within,
                    low: &oversized,
                    close: &within,
                },
                &mut output,
            )
            .unwrap_err(),
        TalibError::PreparedCapacityExceeded {
            max_input_len: within.len(),
            actual_input_len: oversized.len(),
        }
    );
}

#[test]
fn invalid_periods_and_overflow_are_rejected() {
    assert!(PLUS_DMConfig::new(0).is_err());
    assert!(PLUS_DIConfig::new(usize::MAX).is_err());
    assert!(DXConfig::new(1).is_err());
    assert!(ADXConfig::new(1).is_err());
    assert!(ADXRConfig::new(1).is_err());
    assert!(ADXConfig::new(usize::MAX).is_err());
    assert!(ADXRConfig::new(usize::MAX).is_err());
}

#[test]
fn independent_streams_do_not_share_state() {
    let (high, low, close) = observations();
    let config = DXConfig::new(5).unwrap();
    let mut first = IndicatorConfig::stream(&config).unwrap();
    let mut second = IndicatorConfig::stream(&config).unwrap();
    for index in 0..high.len() {
        let first_value = StreamingComputation::<DXConfig>::next(
            &mut first,
            DirectionalTick {
                high: high[index],
                low: low[index],
                close: close[index],
            },
        )
        .unwrap();
        let reflected = DirectionalTick {
            high: -low[index],
            low: -high[index],
            close: -close[index],
        };
        let second_value = StreamingComputation::<DXConfig>::next(&mut second, reflected).unwrap();
        if let (Some(first_value), Some(second_value)) = (first_value, second_value) {
            assert!((first_value - second_value).abs() <= tolerance());
        }
    }
}

#[test]
fn fixture_provenance_is_pinned() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(reference::TALIB_GIT_REVISION.len(), 40);
    assert_eq!(reference::TALIB_SOURCE_ARCHIVE_SHA256.len(), 64);
}
