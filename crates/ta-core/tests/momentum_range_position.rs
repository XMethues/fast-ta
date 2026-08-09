use ta_core::momentum::{
    AROONConfig, AROONOSCConfig, AROONValuesMut, AroonInput, AroonTick, STOCHConfig, STOCHFConfig,
    STOCHFValuesMut, STOCHRSIConfig, STOCHRSIValuesMut, STOCHValuesMut, StochasticInput,
    StochasticTick, WILLRConfig, AROON, AROONOSC, STOCH, STOCHF, STOCHRSI, WILLR,
};
use ta_core::overlap::PeriodMAType;
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

mod reference {
    include!("fixtures/momentum_range_position_reference.rs");
}

#[cfg(feature = "f32")]
const REFERENCE_TOLERANCE: Float = 4e-3;
#[cfg(not(feature = "f32"))]
const REFERENCE_TOLERANCE: Float = 2e-10;

fn floats(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn assert_close(actual: Float, expected: Float) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= REFERENCE_TOLERANCE,
        "actual={actual:?}, expected={expected:?}, difference={difference:?}"
    );
}

fn assert_slice_close(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected as Float);
    }
}

fn input<'a>(high: &'a [Float], low: &'a [Float], close: &'a [Float]) -> StochasticInput<'a> {
    StochasticInput { high, low, close }
}

#[test]
fn pinned_default_references_have_compact_ranges_and_named_equal_columns() {
    assert_eq!(reference::TALIB_VERSION, "0.8.1");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "e64d2ac896c595f38d65e44c812efbfdac8a64cf"
    );
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let aroon_input = AroonInput {
        high: &high,
        low: &low,
    };
    let stochastic_input = input(&high, &low, &close);

    let aroon = AROONConfig::new(reference::PERIOD)
        .unwrap()
        .compute(aroon_input)
        .unwrap();
    assert_eq!(aroon.source_len(), high.len());
    assert_eq!(
        aroon.range(),
        OutputRange::new(reference::PERIOD, reference::AROON_UP_EXPECTED.len())
    );
    assert_eq!(aroon.values().down.len(), aroon.values().up.len());
    assert_slice_close(&aroon.values().down, reference::AROON_DOWN_EXPECTED);
    assert_slice_close(&aroon.values().up, reference::AROON_UP_EXPECTED);

    let oscillator = AROONOSCConfig::new(reference::PERIOD)
        .unwrap()
        .compute(aroon_input)
        .unwrap();
    assert_eq!(oscillator.range(), aroon.range());
    assert_slice_close(oscillator.values(), reference::AROONOSC_EXPECTED);

    let stoch = STOCHConfig::new(
        reference::PERIOD,
        reference::SMOOTHING_PERIOD,
        PeriodMAType::SMA,
        reference::SMOOTHING_PERIOD,
        PeriodMAType::SMA,
    )
    .unwrap()
    .compute(stochastic_input)
    .unwrap();
    assert_eq!(stoch.range(), OutputRange::new(8, 22));
    assert_eq!(stoch.values().slow_k.len(), stoch.values().slow_d.len());
    assert_slice_close(&stoch.values().slow_k, reference::STOCH_SLOW_K_EXPECTED);
    assert_slice_close(&stoch.values().slow_d, reference::STOCH_SLOW_D_EXPECTED);

    let stochf = STOCHFConfig::new(
        reference::PERIOD,
        reference::SMOOTHING_PERIOD,
        PeriodMAType::SMA,
    )
    .unwrap()
    .compute(stochastic_input)
    .unwrap();
    assert_eq!(stochf.range(), OutputRange::new(6, 24));
    assert_eq!(stochf.values().fast_k.len(), stochf.values().fast_d.len());
    assert_slice_close(&stochf.values().fast_k, reference::STOCHF_FAST_K_EXPECTED);
    assert_slice_close(&stochf.values().fast_d, reference::STOCHF_FAST_D_EXPECTED);

    let stochrsi = STOCHRSIConfig::new(
        reference::PERIOD,
        reference::PERIOD,
        reference::SMOOTHING_PERIOD,
        PeriodMAType::SMA,
    )
    .unwrap()
    .compute(&close)
    .unwrap();
    assert_eq!(stochrsi.range(), OutputRange::new(11, 19));
    assert_eq!(
        stochrsi.values().fast_k.len(),
        stochrsi.values().fast_d.len()
    );
    assert_slice_close(
        &stochrsi.values().fast_k,
        reference::STOCHRSI_FAST_K_EXPECTED,
    );
    assert_slice_close(
        &stochrsi.values().fast_d,
        reference::STOCHRSI_FAST_D_EXPECTED,
    );

    let willr = WILLRConfig::new(reference::PERIOD)
        .unwrap()
        .compute(stochastic_input)
        .unwrap();
    assert_eq!(willr.range(), OutputRange::new(4, 26));
    assert_slice_close(willr.values(), reference::WILLR_EXPECTED);
}

#[test]
fn uppercase_caller_owned_functions_match_owned_results() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);

    let mut down = vec![0.0 as Float; 25];
    let mut up = vec![0.0 as Float; 25];
    assert_eq!(
        AROON(&high, &low, 5, &mut down, &mut up).unwrap(),
        OutputRange::new(5, 25)
    );
    assert_slice_close(&down, reference::AROON_DOWN_EXPECTED);
    assert_slice_close(&up, reference::AROON_UP_EXPECTED);

    let mut oscillator = vec![0.0 as Float; 25];
    AROONOSC(&high, &low, 5, &mut oscillator).unwrap();
    assert_slice_close(&oscillator, reference::AROONOSC_EXPECTED);

    let mut slow_k = vec![0.0 as Float; 22];
    let mut slow_d = vec![0.0 as Float; 22];
    STOCH(
        &high,
        &low,
        &close,
        5,
        3,
        PeriodMAType::SMA,
        3,
        PeriodMAType::SMA,
        &mut slow_k,
        &mut slow_d,
    )
    .unwrap();
    assert_slice_close(&slow_k, reference::STOCH_SLOW_K_EXPECTED);
    assert_slice_close(&slow_d, reference::STOCH_SLOW_D_EXPECTED);

    let mut fast_k = vec![0.0 as Float; 24];
    let mut fast_d = vec![0.0 as Float; 24];
    STOCHF(
        &high,
        &low,
        &close,
        5,
        3,
        PeriodMAType::SMA,
        &mut fast_k,
        &mut fast_d,
    )
    .unwrap();
    assert_slice_close(&fast_k, reference::STOCHF_FAST_K_EXPECTED);
    assert_slice_close(&fast_d, reference::STOCHF_FAST_D_EXPECTED);

    let mut rsi_k = vec![0.0 as Float; 19];
    let mut rsi_d = vec![0.0 as Float; 19];
    STOCHRSI(&close, 5, 5, 3, PeriodMAType::SMA, &mut rsi_k, &mut rsi_d).unwrap();
    assert_slice_close(&rsi_k, reference::STOCHRSI_FAST_K_EXPECTED);
    assert_slice_close(&rsi_d, reference::STOCHRSI_FAST_D_EXPECTED);

    let mut willr = vec![0.0 as Float; 26];
    WILLR(&high, &low, &close, 5, &mut willr).unwrap();
    assert_slice_close(&willr, reference::WILLR_EXPECTED);
}

#[test]
fn prepared_runners_reuse_capacity_and_match_owned_for_all_definitions() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let aroon_input = AroonInput {
        high: &high,
        low: &low,
    };
    let stochastic_input = input(&high, &low, &close);

    let aroon_config = AROONConfig::new(5).unwrap();
    let expected_aroon = aroon_config.compute(aroon_input).unwrap().into_values();
    let mut aroon_runner = aroon_config.prepare_batch(high.len()).unwrap();
    let mut down = vec![0.0 as Float; expected_aroon.down.len()];
    let mut up = vec![0.0 as Float; expected_aroon.up.len()];
    for _ in 0..2 {
        aroon_runner
            .compute_into(
                aroon_input,
                AROONValuesMut {
                    down: &mut down,
                    up: &mut up,
                },
            )
            .unwrap();
        assert_eq!(down, expected_aroon.down);
        assert_eq!(up, expected_aroon.up);
    }

    let oscillator_config = AROONOSCConfig::new(5).unwrap();
    let expected_oscillator = oscillator_config
        .compute(aroon_input)
        .unwrap()
        .into_values();
    let mut oscillator_runner = oscillator_config.prepare_batch(high.len()).unwrap();
    let mut oscillator = vec![0.0 as Float; expected_oscillator.len()];
    oscillator_runner
        .compute_into(aroon_input, &mut oscillator)
        .unwrap();
    assert_eq!(oscillator, expected_oscillator);

    let stoch_config = STOCHConfig::new(5, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA).unwrap();
    let expected_stoch = stoch_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut stoch_runner = stoch_config.prepare_batch(high.len()).unwrap();
    let mut slow_k = vec![0.0 as Float; expected_stoch.slow_k.len()];
    let mut slow_d = vec![0.0 as Float; expected_stoch.slow_d.len()];
    stoch_runner
        .compute_into(
            stochastic_input,
            STOCHValuesMut {
                slow_k: &mut slow_k,
                slow_d: &mut slow_d,
            },
        )
        .unwrap();
    assert_eq!(slow_k, expected_stoch.slow_k);
    assert_eq!(slow_d, expected_stoch.slow_d);

    let stochf_config = STOCHFConfig::new(5, 3, PeriodMAType::SMA).unwrap();
    let expected_stochf = stochf_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut stochf_runner = stochf_config.prepare_batch(high.len()).unwrap();
    let mut fast_k = vec![0.0 as Float; expected_stochf.fast_k.len()];
    let mut fast_d = vec![0.0 as Float; expected_stochf.fast_d.len()];
    stochf_runner
        .compute_into(
            stochastic_input,
            STOCHFValuesMut {
                fast_k: &mut fast_k,
                fast_d: &mut fast_d,
            },
        )
        .unwrap();
    assert_eq!(fast_k, expected_stochf.fast_k);
    assert_eq!(fast_d, expected_stochf.fast_d);

    let stochrsi_config = STOCHRSIConfig::new(5, 5, 3, PeriodMAType::SMA).unwrap();
    let expected_stochrsi = stochrsi_config.compute(&close).unwrap().into_values();
    let mut stochrsi_runner = stochrsi_config.prepare_batch(close.len()).unwrap();
    let mut rsi_k = vec![0.0 as Float; expected_stochrsi.fast_k.len()];
    let mut rsi_d = vec![0.0 as Float; expected_stochrsi.fast_d.len()];
    stochrsi_runner
        .compute_into(
            &close,
            STOCHRSIValuesMut {
                fast_k: &mut rsi_k,
                fast_d: &mut rsi_d,
            },
        )
        .unwrap();
    assert_eq!(rsi_k, expected_stochrsi.fast_k);
    assert_eq!(rsi_d, expected_stochrsi.fast_d);

    let willr_config = WILLRConfig::new(5).unwrap();
    let expected_willr = willr_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut willr_runner = willr_config.prepare_batch(high.len()).unwrap();
    let mut willr = vec![0.0 as Float; expected_willr.len()];
    willr_runner
        .compute_into(stochastic_input, &mut willr)
        .unwrap();
    assert_eq!(willr, expected_willr);

    let oversized_high = vec![1.0 as Float; high.len() + 1];
    let oversized_low = vec![0.0 as Float; high.len() + 1];
    let error = aroon_runner
        .compute_into(
            AroonInput {
                high: &oversized_high,
                low: &oversized_low,
            },
            AROONValuesMut {
                down: &mut down,
                up: &mut up,
            },
        )
        .unwrap_err();
    assert!(matches!(error, TalibError::PreparedCapacityExceeded { .. }));
}

#[test]
fn streaming_matches_batch_then_reset_replays_for_every_definition() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let aroon_input = AroonInput {
        high: &high,
        low: &low,
    };
    let stochastic_input = input(&high, &low, &close);

    let aroon_config = AROONConfig::new(5).unwrap();
    let expected_aroon = aroon_config.compute(aroon_input).unwrap().into_values();
    let mut aroon_stream = aroon_config.stream().unwrap();
    let collect_aroon = |stream: &mut ta_core::momentum::AROONStream| {
        high.iter()
            .zip(&low)
            .filter_map(|(&high, &low)| stream.next(AroonTick { high, low }).unwrap())
            .collect::<Vec<_>>()
    };
    let first_aroon = collect_aroon(&mut aroon_stream);
    assert_eq!(first_aroon.len(), expected_aroon.up.len());
    for (actual, (&down, &up)) in first_aroon
        .iter()
        .zip(expected_aroon.down.iter().zip(&expected_aroon.up))
    {
        assert_close(actual.down, down);
        assert_close(actual.up, up);
    }
    aroon_stream.reset();
    assert_eq!(collect_aroon(&mut aroon_stream), first_aroon);

    let oscillator_config = AROONOSCConfig::new(5).unwrap();
    let expected_oscillator = oscillator_config
        .compute(aroon_input)
        .unwrap()
        .into_values();
    let mut oscillator_stream = oscillator_config.stream().unwrap();
    let mut streamed_oscillator = Vec::new();
    for (&high, &low) in high.iter().zip(&low) {
        if let Some(value) = oscillator_stream.next(AroonTick { high, low }).unwrap() {
            streamed_oscillator.push(value);
        }
    }
    assert_eq!(streamed_oscillator, expected_oscillator);
    oscillator_stream.reset();
    let replayed_oscillator: Vec<_> = high
        .iter()
        .zip(&low)
        .filter_map(|(&high, &low)| oscillator_stream.next(AroonTick { high, low }).unwrap())
        .collect();
    assert_eq!(replayed_oscillator, streamed_oscillator);

    let stoch_config = STOCHConfig::new(5, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA).unwrap();
    let expected_stoch = stoch_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut stoch_stream = stoch_config.stream().unwrap();
    let collect_stoch = |stream: &mut ta_core::momentum::STOCHStream| {
        (0..high.len())
            .filter_map(|index| {
                stream
                    .next(StochasticTick {
                        high: high[index],
                        low: low[index],
                        close: close[index],
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let first_stoch = collect_stoch(&mut stoch_stream);
    assert_eq!(first_stoch.len(), expected_stoch.slow_k.len());
    for (actual, (&slow_k, &slow_d)) in first_stoch
        .iter()
        .zip(expected_stoch.slow_k.iter().zip(&expected_stoch.slow_d))
    {
        assert_close(actual.slow_k, slow_k);
        assert_close(actual.slow_d, slow_d);
    }
    stoch_stream.reset();
    assert_eq!(collect_stoch(&mut stoch_stream), first_stoch);

    let stochf_config = STOCHFConfig::new(5, 3, PeriodMAType::SMA).unwrap();
    let expected_stochf = stochf_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut stochf_stream = stochf_config.stream().unwrap();
    let mut streamed_stochf = Vec::new();
    for index in 0..high.len() {
        if let Some(value) = stochf_stream
            .next(StochasticTick {
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap()
        {
            streamed_stochf.push(value);
        }
    }
    for (actual, (&fast_k, &fast_d)) in streamed_stochf
        .iter()
        .zip(expected_stochf.fast_k.iter().zip(&expected_stochf.fast_d))
    {
        assert_close(actual.fast_k, fast_k);
        assert_close(actual.fast_d, fast_d);
    }
    stochf_stream.reset();
    let replayed_stochf: Vec<_> = (0..high.len())
        .filter_map(|index| {
            stochf_stream
                .next(StochasticTick {
                    high: high[index],
                    low: low[index],
                    close: close[index],
                })
                .unwrap()
        })
        .collect();
    assert_eq!(replayed_stochf, streamed_stochf);

    let stochrsi_config = STOCHRSIConfig::new(5, 5, 3, PeriodMAType::SMA).unwrap();
    let expected_stochrsi = stochrsi_config.compute(&close).unwrap().into_values();
    let mut stochrsi_stream = stochrsi_config.stream().unwrap();
    let streamed_stochrsi: Vec<_> = close
        .iter()
        .filter_map(|&value| stochrsi_stream.next(value).unwrap())
        .collect();
    for (actual, (&fast_k, &fast_d)) in streamed_stochrsi.iter().zip(
        expected_stochrsi
            .fast_k
            .iter()
            .zip(&expected_stochrsi.fast_d),
    ) {
        assert_close(actual.fast_k, fast_k);
        assert_close(actual.fast_d, fast_d);
    }
    stochrsi_stream.reset();
    let replayed_stochrsi: Vec<_> = close
        .iter()
        .filter_map(|&value| stochrsi_stream.next(value).unwrap())
        .collect();
    assert_eq!(replayed_stochrsi, streamed_stochrsi);

    let willr_config = WILLRConfig::new(5).unwrap();
    let expected_willr = willr_config
        .compute(stochastic_input)
        .unwrap()
        .into_values();
    let mut willr_stream = willr_config.stream().unwrap();
    let streamed_willr: Vec<_> = (0..high.len())
        .filter_map(|index| {
            willr_stream
                .next(StochasticTick {
                    high: high[index],
                    low: low[index],
                    close: close[index],
                })
                .unwrap()
        })
        .collect();
    assert_eq!(streamed_willr, expected_willr);
    willr_stream.reset();
    let replayed_willr: Vec<_> = (0..high.len())
        .filter_map(|index| {
            willr_stream
                .next(StochasticTick {
                    high: high[index],
                    low: low[index],
                    close: close[index],
                })
                .unwrap()
        })
        .collect();
    assert_eq!(replayed_willr, streamed_willr);
}

#[test]
fn independent_streams_and_rejected_ticks_preserve_state() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);

    let aroon_config = AROONConfig::new(5).unwrap();
    let mut aroon = aroon_config.stream().unwrap();
    for index in 0..8 {
        aroon
            .next(AroonTick {
                high: high[index],
                low: low[index],
            })
            .unwrap();
    }
    let mut aroon_control = aroon.clone();
    assert!(aroon
        .next(AroonTick {
            high: Float::NAN,
            low: low[8],
        })
        .is_err());
    assert_eq!(
        aroon
            .next(AroonTick {
                high: high[8],
                low: low[8],
            })
            .unwrap(),
        aroon_control
            .next(AroonTick {
                high: high[8],
                low: low[8],
            })
            .unwrap()
    );

    let oscillator_config = AROONOSCConfig::new(5).unwrap();
    let mut oscillator = oscillator_config.stream().unwrap();
    for index in 0..8 {
        oscillator
            .next(AroonTick {
                high: high[index],
                low: low[index],
            })
            .unwrap();
    }
    let mut oscillator_control = oscillator.clone();
    assert!(oscillator
        .next(AroonTick {
            high: high[8],
            low: Float::INFINITY,
        })
        .is_err());
    let next_aroon_tick = AroonTick {
        high: high[8],
        low: low[8],
    };
    assert_eq!(
        oscillator.next(next_aroon_tick).unwrap(),
        oscillator_control.next(next_aroon_tick).unwrap()
    );

    let tick = |index: usize| StochasticTick {
        high: high[index],
        low: low[index],
        close: close[index],
    };
    let invalid_tick = StochasticTick {
        high: high[10],
        low: low[10],
        close: Float::NAN,
    };

    let stoch_config = STOCHConfig::new(5, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA).unwrap();
    let mut stoch = stoch_config.stream().unwrap();
    for index in 0..10 {
        stoch.next(tick(index)).unwrap();
    }
    let mut stoch_control = stoch.clone();
    assert!(stoch.next(invalid_tick).is_err());
    assert_eq!(
        stoch.next(tick(10)).unwrap(),
        stoch_control.next(tick(10)).unwrap()
    );

    let stochf_config = STOCHFConfig::new(5, 3, PeriodMAType::SMA).unwrap();
    let mut stochf = stochf_config.stream().unwrap();
    for index in 0..10 {
        stochf.next(tick(index)).unwrap();
    }
    let mut stochf_control = stochf.clone();
    assert!(stochf.next(invalid_tick).is_err());
    assert_eq!(
        stochf.next(tick(10)).unwrap(),
        stochf_control.next(tick(10)).unwrap()
    );

    let stochrsi_config = STOCHRSIConfig::new(5, 5, 3, PeriodMAType::SMA).unwrap();
    let mut stochrsi = stochrsi_config.stream().unwrap();
    for &value in &close[..14] {
        stochrsi.next(value).unwrap();
    }
    let mut stochrsi_control = stochrsi.clone();
    assert!(stochrsi.next(Float::NAN).is_err());
    assert_eq!(
        stochrsi.next(close[14]).unwrap(),
        stochrsi_control.next(close[14]).unwrap()
    );

    let willr_config = WILLRConfig::new(5).unwrap();
    let mut willr = willr_config.stream().unwrap();
    for index in 0..10 {
        willr.next(tick(index)).unwrap();
    }
    let mut willr_control = willr.clone();
    assert!(willr.next(invalid_tick).is_err());
    assert_eq!(
        willr.next(tick(10)).unwrap(),
        willr_control.next(tick(10)).unwrap()
    );

    let mut independent_a = stochf_config.stream().unwrap();
    let mut independent_b = stochf_config.stream().unwrap();
    independent_a.next(tick(0)).unwrap();
    independent_a.next(tick(1)).unwrap();
    assert_eq!(independent_b.next(tick(0)).unwrap(), None);
}

#[test]
fn extrema_ties_flat_ranges_and_cross_indicator_relationships_are_pinned() {
    let high = [10.0 as Float; 8];
    let low = [10.0 as Float; 8];
    let close = [10.0 as Float; 8];
    let aroon = AROONConfig::new(2)
        .unwrap()
        .compute(AroonInput {
            high: &high,
            low: &low,
        })
        .unwrap();
    assert!(aroon.values().down.iter().all(|&value| value == 100.0));
    assert!(aroon.values().up.iter().all(|&value| value == 100.0));
    let oscillator = AROONOSCConfig::new(2)
        .unwrap()
        .compute(AroonInput {
            high: &high,
            low: &low,
        })
        .unwrap();
    assert!(oscillator.values().iter().all(|&value| value == 0.0));

    let flat_input = input(&high, &low, &close);
    let stochf = STOCHFConfig::new(3, 1, PeriodMAType::SMA)
        .unwrap()
        .compute(flat_input)
        .unwrap();
    assert!(stochf.values().fast_k.iter().all(|&value| value == 0.0));
    assert!(stochf.values().fast_d.iter().all(|&value| value == 0.0));
    let willr = WILLRConfig::new(3).unwrap().compute(flat_input).unwrap();
    assert!(willr.values().iter().all(|&value| value == 0.0));

    let market_high = floats(reference::HIGH);
    let market_low = floats(reference::LOW);
    let market_close = floats(reference::CLOSE);
    let market = input(&market_high, &market_low, &market_close);
    let raw_stochastic = STOCHFConfig::new(5, 1, PeriodMAType::SMA)
        .unwrap()
        .compute(market)
        .unwrap();
    let willr = WILLRConfig::new(5).unwrap().compute(market).unwrap();
    assert_eq!(raw_stochastic.range(), willr.range());
    for (&fast_k, &willr) in raw_stochastic.values().fast_k.iter().zip(willr.values()) {
        assert_close(fast_k - 100.0 as Float, willr);
    }

    let stoch = STOCHConfig::new(5, 3, PeriodMAType::SMA, 1, PeriodMAType::SMA)
        .unwrap()
        .compute(market)
        .unwrap();
    let stochf = STOCHFConfig::new(5, 3, PeriodMAType::SMA)
        .unwrap()
        .compute(market)
        .unwrap();
    assert_eq!(stoch.range(), stochf.range());
    for ((&slow_k, &slow_d), (&fast_k, &fast_d)) in stoch
        .values()
        .slow_k
        .iter()
        .zip(&stoch.values().slow_d)
        .zip(stochf.values().fast_k.iter().zip(&stochf.values().fast_d))
    {
        assert_close(slow_k, fast_d);
        assert_close(slow_d, fast_d);
        let _ = fast_k;
    }
}

#[test]
fn positive_affine_scaling_preserves_range_positions_and_bounds() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let scaled_high: Vec<_> = high.iter().map(|&value| value * 3.0 + 17.0).collect();
    let scaled_low: Vec<_> = low.iter().map(|&value| value * 3.0 + 17.0).collect();
    let scaled_close: Vec<_> = close.iter().map(|&value| value * 3.0 + 17.0).collect();
    let original = input(&high, &low, &close);
    let scaled = input(&scaled_high, &scaled_low, &scaled_close);

    let original_stoch = STOCHFConfig::new(5, 3, PeriodMAType::SMA)
        .unwrap()
        .compute(original)
        .unwrap();
    let scaled_stoch = STOCHFConfig::new(5, 3, PeriodMAType::SMA)
        .unwrap()
        .compute(scaled)
        .unwrap();
    for (&actual, &expected) in scaled_stoch
        .values()
        .fast_k
        .iter()
        .zip(&original_stoch.values().fast_k)
    {
        assert_close(actual, expected);
        assert!((0.0 as Float..=100.0 as Float).contains(&actual));
    }

    let original_willr = WILLRConfig::new(5).unwrap().compute(original).unwrap();
    let scaled_willr = WILLRConfig::new(5).unwrap().compute(scaled).unwrap();
    for (&actual, &expected) in scaled_willr.values().iter().zip(original_willr.values()) {
        assert_close(actual, expected);
        assert!((-100.0 as Float..=0.0 as Float).contains(&actual));
    }

    let original_aroon = AROONConfig::new(5)
        .unwrap()
        .compute(AroonInput {
            high: &high,
            low: &low,
        })
        .unwrap();
    let scaled_aroon = AROONConfig::new(5)
        .unwrap()
        .compute(AroonInput {
            high: &scaled_high,
            low: &scaled_low,
        })
        .unwrap();
    assert_eq!(original_aroon, scaled_aroon);
    for &value in original_aroon
        .values()
        .down
        .iter()
        .chain(&original_aroon.values().up)
    {
        assert!((0.0 as Float..=100.0 as Float).contains(&value));
    }
}

#[test]
fn every_qualified_period_ma_kind_executes_and_configuration_is_immutable() {
    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let stochastic_input = input(&high, &low, &close);
    let kinds = [
        PeriodMAType::SMA,
        PeriodMAType::EMA,
        PeriodMAType::WMA,
        PeriodMAType::DEMA,
        PeriodMAType::TEMA,
        PeriodMAType::TRIMA,
        PeriodMAType::T3,
        PeriodMAType::KAMA,
    ];
    for kind in kinds {
        let stoch = STOCHConfig::new(3, 3, kind, 2, PeriodMAType::SMA).unwrap();
        assert_eq!(stoch.fast_k_period(), 3);
        assert_eq!(stoch.slow_k_period(), 3);
        assert_eq!(stoch.slow_k_ma_type(), kind);
        assert_eq!(stoch.slow_d_period(), 2);
        assert_eq!(stoch.slow_d_ma_type(), PeriodMAType::SMA);
        assert!(!stoch.compute(stochastic_input).unwrap().range().is_empty());

        let stochf = STOCHFConfig::new(3, 3, kind).unwrap();
        assert_eq!(stochf.fast_d_ma_type(), kind);
        assert!(!stochf.compute(stochastic_input).unwrap().range().is_empty());

        let stochrsi = STOCHRSIConfig::new(3, 3, 3, kind).unwrap();
        assert_eq!(stochrsi.rsi_period(), 3);
        assert_eq!(stochrsi.fast_d_ma_type(), kind);
        assert!(!stochrsi.compute(&close).unwrap().range().is_empty());
    }

    let aroon = AROONConfig::new(5).unwrap();
    let copied = aroon;
    assert_eq!(aroon, copied);
    assert_eq!(aroon.period(), 5);
    assert_eq!(AROONOSCConfig::new(5).unwrap().period(), 5);
    assert_eq!(WILLRConfig::new(5).unwrap().period(), 5);
}

#[test]
fn validation_order_capacity_and_failure_before_mutation_are_transactional() {
    assert!(matches!(
        AROONConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        AROONOSCConfig::new(100_001),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        STOCHConfig::new(0, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        STOCHFConfig::new(5, 0, PeriodMAType::SMA),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        STOCHRSIConfig::new(1, 5, 3, PeriodMAType::SMA),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        WILLRConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));

    let high = floats(reference::HIGH);
    let low = floats(reference::LOW);
    let close = floats(reference::CLOSE);
    let short_low = &low[..low.len() - 1];
    assert!(matches!(
        AROONConfig::new(5).unwrap().compute(AroonInput {
            high: &high,
            low: short_low
        }),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        STOCHFConfig::new(5, 3, PeriodMAType::SMA)
            .unwrap()
            .compute(StochasticInput {
                high: &high,
                low: &low,
                close: &close[..close.len() - 1]
            }),
        Err(TalibError::InvalidInput { .. })
    ));

    let mut nonfinite = high.clone();
    nonfinite[7] = Float::NAN;
    assert!(matches!(
        WILLRConfig::new(5)
            .unwrap()
            .compute(input(&nonfinite, &low, &close)),
        Err(TalibError::InvalidInput { .. })
    ));
    assert!(matches!(
        STOCHRSIConfig::new(5, 5, 3, PeriodMAType::SMA)
            .unwrap()
            .compute(&close[..11]),
        Err(TalibError::InsufficientData { .. })
    ));

    let sentinel = -9_999.0 as Float;
    let aroon_config = AROONConfig::new(5).unwrap();
    let mut down = vec![sentinel; 24];
    let mut up = vec![sentinel; 25];
    assert!(aroon_config
        .compute_into(
            AroonInput {
                high: &high,
                low: &low
            },
            AROONValuesMut {
                down: &mut down,
                up: &mut up
            },
        )
        .is_err());
    assert!(down.iter().all(|&value| value == sentinel));
    assert!(up.iter().all(|&value| value == sentinel));

    let stochastic_input = input(&high, &low, &close);
    let stoch_config = STOCHConfig::new(5, 3, PeriodMAType::SMA, 3, PeriodMAType::SMA).unwrap();
    let mut slow_k = vec![sentinel; 22];
    let mut slow_d = vec![sentinel; 21];
    assert!(stoch_config
        .compute_into(
            stochastic_input,
            STOCHValuesMut {
                slow_k: &mut slow_k,
                slow_d: &mut slow_d
            },
        )
        .is_err());
    assert!(slow_k.iter().all(|&value| value == sentinel));
    assert!(slow_d.iter().all(|&value| value == sentinel));

    let stochf_config = STOCHFConfig::new(5, 3, PeriodMAType::SMA).unwrap();
    let mut fast_k = vec![sentinel; 24];
    let mut fast_d = vec![sentinel; 23];
    assert!(stochf_config
        .compute_into(
            stochastic_input,
            STOCHFValuesMut {
                fast_k: &mut fast_k,
                fast_d: &mut fast_d
            },
        )
        .is_err());
    assert!(fast_k.iter().all(|&value| value == sentinel));
    assert!(fast_d.iter().all(|&value| value == sentinel));

    let stochrsi_config = STOCHRSIConfig::new(5, 5, 3, PeriodMAType::SMA).unwrap();
    let mut rsi_k = vec![sentinel; 19];
    let mut rsi_d = vec![sentinel; 18];
    assert!(stochrsi_config
        .compute_into(
            &close,
            STOCHRSIValuesMut {
                fast_k: &mut rsi_k,
                fast_d: &mut rsi_d
            },
        )
        .is_err());
    assert!(rsi_k.iter().all(|&value| value == sentinel));
    assert!(rsi_d.iter().all(|&value| value == sentinel));

    let willr_config = WILLRConfig::new(5).unwrap();
    let mut willr = vec![sentinel; 25];
    assert!(willr_config
        .compute_into(stochastic_input, &mut willr)
        .is_err());
    assert!(willr.iter().all(|&value| value == sentinel));

    let empty: &[Float] = &[];
    let empty_aroon = aroon_config
        .compute(AroonInput {
            high: empty,
            low: empty,
        })
        .unwrap();
    assert_eq!(empty_aroon.range(), OutputRange::empty());
    let empty_stoch = stoch_config
        .compute(StochasticInput {
            high: empty,
            low: empty,
            close: empty,
        })
        .unwrap();
    assert_eq!(empty_stoch.range(), OutputRange::empty());
}
