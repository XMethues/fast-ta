//! Public execution and numerical contracts for composite-input Momentum definitions.

#[path = "fixtures/momentum_composite_reference.rs"]
mod reference;

use fast_ta::momentum::{
    BOPConfig, BOPInput, BOPTick, CCIConfig, CCIInput, CCITick, MFIConfig, MFIInput, MFITick,
    ULTOSCConfig, ULTOSCInput, ULTOSCTick, BOP, CCI, MFI, ULTOSC,
};
use fast_ta::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

type Ohlcv = (Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>, Vec<Float>);

fn floats(values: &[f64]) -> Vec<Float> {
    values.iter().map(|&value| value as Float).collect()
}

fn tolerance() -> Float {
    if core::mem::size_of::<Float>() == core::mem::size_of::<f32>() {
        5e-4 as Float
    } else {
        1e-10 as Float
    }
}

fn assert_close(actual: &[Float], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let epsilon = tolerance();
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected as Float;
        let scale = Float::max(1.0 as Float, expected.abs());
        assert!(
            (actual - expected).abs() <= epsilon * scale,
            "value {index}: expected {expected}, got {actual}"
        );
    }
}

fn ordinary() -> Ohlcv {
    (
        floats(reference::OPEN),
        floats(reference::HIGH),
        floats(reference::LOW),
        floats(reference::CLOSE),
        floats(reference::VOLUME),
    )
}

fn assert_invalid_input(error: TalibError) {
    assert!(matches!(error, TalibError::InvalidInput { .. }));
}

#[test]
fn pinned_talib_reference_vectors_match_all_execution_modes() {
    assert_eq!(reference::TALIB_VERSION, "0.6.4");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "43f9d5042ecc4bd367941846494ad907bf20ea50"
    );
    assert_eq!(reference::TALIB_SOURCE_ARCHIVE_SHA256.len(), 64);

    let (open, high, low, close, volume) = ordinary();

    let bop = BOPConfig::new();
    let bop_input = BOPInput {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
    };
    let bop_owned = bop.compute(bop_input).unwrap();
    assert_eq!(bop_owned.range(), OutputRange::new(0, open.len()));
    assert_close(bop_owned.values(), reference::BOP_EXPECTED);
    let mut bop_caller = vec![-999.0 as Float; open.len()];
    assert_eq!(
        bop.compute_into(bop_input, &mut bop_caller).unwrap(),
        bop_owned.range()
    );
    assert_close(&bop_caller, reference::BOP_EXPECTED);
    let mut bop_runner = bop.prepare_batch(open.len()).unwrap();
    bop_caller.fill(-999.0 as Float);
    bop_runner.compute_into(bop_input, &mut bop_caller).unwrap();
    assert_close(&bop_caller, reference::BOP_EXPECTED);

    let cci = CCIConfig::new(reference::CCI_PERIOD).unwrap();
    let cci_input = CCIInput {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
    };
    let cci_owned = cci.compute(cci_input).unwrap();
    assert_eq!(
        cci_owned.range(),
        OutputRange::new(reference::CCI_PERIOD - 1, reference::CCI_EXPECTED.len())
    );
    assert_close(cci_owned.values(), reference::CCI_EXPECTED);
    let mut cci_caller = vec![-999.0 as Float; reference::CCI_EXPECTED.len()];
    cci.compute_into(cci_input, &mut cci_caller).unwrap();
    assert_close(&cci_caller, reference::CCI_EXPECTED);
    let mut cci_runner = cci.prepare_batch(high.len()).unwrap();
    cci_caller.fill(-999.0 as Float);
    cci_runner.compute_into(cci_input, &mut cci_caller).unwrap();
    assert_close(&cci_caller, reference::CCI_EXPECTED);

    let mfi = MFIConfig::new(reference::MFI_PERIOD).unwrap();
    let mfi_input = MFIInput {
        open: &open,
        high: &high,
        low: &low,
        close: &close,
        volume: &volume,
    };
    let mfi_owned = mfi.compute(mfi_input).unwrap();
    assert_eq!(
        mfi_owned.range(),
        OutputRange::new(reference::MFI_PERIOD, reference::MFI_EXPECTED.len())
    );
    assert_close(mfi_owned.values(), reference::MFI_EXPECTED);
    let mut mfi_caller = vec![-999.0 as Float; reference::MFI_EXPECTED.len()];
    mfi.compute_into(mfi_input, &mut mfi_caller).unwrap();
    assert_close(&mfi_caller, reference::MFI_EXPECTED);
    let mut mfi_runner = mfi.prepare_batch(high.len()).unwrap();
    mfi_caller.fill(-999.0 as Float);
    mfi_runner.compute_into(mfi_input, &mut mfi_caller).unwrap();
    assert_close(&mfi_caller, reference::MFI_EXPECTED);

    let periods = reference::ULTOSC_PERIODS;
    let ultosc = ULTOSCConfig::new(periods[0], periods[1], periods[2]).unwrap();
    let ultosc_input = ULTOSCInput {
        high: &high,
        low: &low,
        close: &close,
    };
    let ultosc_owned = ultosc.compute(ultosc_input).unwrap();
    assert_eq!(
        ultosc_owned.range(),
        OutputRange::new(periods[2], reference::ULTOSC_EXPECTED.len())
    );
    assert_close(ultosc_owned.values(), reference::ULTOSC_EXPECTED);
    let mut ultosc_caller = vec![-999.0 as Float; reference::ULTOSC_EXPECTED.len()];
    ultosc
        .compute_into(ultosc_input, &mut ultosc_caller)
        .unwrap();
    assert_close(&ultosc_caller, reference::ULTOSC_EXPECTED);
    let mut ultosc_runner = ultosc.prepare_batch(high.len()).unwrap();
    ultosc_caller.fill(-999.0 as Float);
    ultosc_runner
        .compute_into(ultosc_input, &mut ultosc_caller)
        .unwrap();
    assert_close(&ultosc_caller, reference::ULTOSC_EXPECTED);
}

#[test]
fn streaming_matches_compact_batch_alignment_and_reset_replay() {
    let (open, high, low, close, volume) = ordinary();

    let bop = BOPConfig::new();
    let mut bop_stream = bop.stream().unwrap();
    let mut bop_values = Vec::new();
    for index in 0..open.len() {
        bop_values.push(
            bop_stream
                .next(BOPTick {
                    open: open[index],
                    high: high[index],
                    low: low[index],
                    close: close[index],
                })
                .unwrap()
                .unwrap(),
        );
    }
    assert_close(&bop_values, reference::BOP_EXPECTED);
    bop_stream.reset();
    let replayed_bop = bop_stream
        .next(BOPTick {
            open: open[0],
            high: high[0],
            low: low[0],
            close: close[0],
        })
        .unwrap()
        .unwrap();
    assert_close(&[replayed_bop], &reference::BOP_EXPECTED[..1]);

    let cci = CCIConfig::new(reference::CCI_PERIOD).unwrap();
    let mut cci_stream = cci.stream().unwrap();
    let mut cci_values = Vec::new();
    for index in 0..high.len() {
        if let Some(value) = cci_stream
            .next(CCITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap()
        {
            cci_values.push(value);
        }
    }
    assert_close(&cci_values, reference::CCI_EXPECTED);
    cci_stream.reset();
    for index in 0..reference::CCI_PERIOD - 1 {
        assert!(cci_stream
            .next(CCITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap()
            .is_none());
    }
    assert!(cci_stream
        .next(CCITick {
            open: open[reference::CCI_PERIOD - 1],
            high: high[reference::CCI_PERIOD - 1],
            low: low[reference::CCI_PERIOD - 1],
            close: close[reference::CCI_PERIOD - 1],
        })
        .unwrap()
        .is_some());

    let mfi = MFIConfig::new(reference::MFI_PERIOD).unwrap();
    let mut mfi_stream = mfi.stream().unwrap();
    let mut mfi_values = Vec::new();
    for index in 0..high.len() {
        if let Some(value) = mfi_stream
            .next(MFITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
                volume: volume[index],
            })
            .unwrap()
        {
            mfi_values.push(value);
        }
    }
    assert_close(&mfi_values, reference::MFI_EXPECTED);
    mfi_stream.reset();
    for index in 0..reference::MFI_PERIOD {
        assert!(mfi_stream
            .next(MFITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
                volume: volume[index],
            })
            .unwrap()
            .is_none());
    }
    assert!(mfi_stream
        .next(MFITick {
            open: open[reference::MFI_PERIOD],
            high: high[reference::MFI_PERIOD],
            low: low[reference::MFI_PERIOD],
            close: close[reference::MFI_PERIOD],
            volume: volume[reference::MFI_PERIOD],
        })
        .unwrap()
        .is_some());

    let periods = reference::ULTOSC_PERIODS;
    let ultosc = ULTOSCConfig::new(periods[0], periods[1], periods[2]).unwrap();
    let mut ultosc_stream = ultosc.stream().unwrap();
    let mut ultosc_values = Vec::new();
    for index in 0..high.len() {
        if let Some(value) = ultosc_stream
            .next(ULTOSCTick {
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap()
        {
            ultosc_values.push(value);
        }
    }
    assert_close(&ultosc_values, reference::ULTOSC_EXPECTED);
    ultosc_stream.reset();
    for index in 0..periods[2] {
        assert!(ultosc_stream
            .next(ULTOSCTick {
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap()
            .is_none());
    }
    assert!(ultosc_stream
        .next(ULTOSCTick {
            high: high[periods[2]],
            low: low[periods[2]],
            close: close[periods[2]],
        })
        .unwrap()
        .is_some());
}

#[test]
fn observation_alignment_finite_values_and_output_capacity_fail_before_mutation() {
    let (open, high, low, close, volume) = ordinary();
    let sentinel = 12_345.0 as Float;

    let mut output = vec![sentinel; high.len()];
    assert_invalid_input(BOP(&open[..11], &high, &low, &close, &mut output).unwrap_err());
    assert!(output.iter().all(|&value| value == sentinel));
    assert_invalid_input(CCI(&open, &high, &low[..11], &close, 5, &mut output).unwrap_err());
    assert!(output.iter().all(|&value| value == sentinel));
    assert_invalid_input(
        MFI(&open, &high, &low, &close, &volume[..11], 5, &mut output).unwrap_err(),
    );
    assert!(output.iter().all(|&value| value == sentinel));
    assert_invalid_input(ULTOSC(&high, &low, &close[..11], 3, 5, 7, &mut output).unwrap_err());
    assert!(output.iter().all(|&value| value == sentinel));
    assert_invalid_input(CCI(&open[..11], &high, &low, &close, 5, &mut output).unwrap_err());
    assert_invalid_input(
        MFI(&open[..11], &high, &low, &close, &volume, 5, &mut output).unwrap_err(),
    );
    assert!(output.iter().all(|&value| value == sentinel));

    let mut invalid = high.clone();
    invalid[4] = Float::NAN;
    assert_invalid_input(BOP(&open, &invalid, &low, &close, &mut output).unwrap_err());
    assert_invalid_input(CCI(&open, &invalid, &low, &close, 5, &mut output).unwrap_err());
    assert_invalid_input(MFI(&open, &invalid, &low, &close, &volume, 5, &mut output).unwrap_err());
    assert_invalid_input(ULTOSC(&invalid, &low, &close, 3, 5, 7, &mut output).unwrap_err());
    assert!(output.iter().all(|&value| value == sentinel));
    let mut invalid_open = open.clone();
    invalid_open[4] = Float::NAN;
    assert_invalid_input(CCI(&invalid_open, &high, &low, &close, 5, &mut output).unwrap_err());
    assert_invalid_input(
        MFI(&invalid_open, &high, &low, &close, &volume, 5, &mut output).unwrap_err(),
    );
    assert!(output.iter().all(|&value| value == sentinel));

    let mut too_small = vec![sentinel; 1];
    assert_invalid_input(BOP(&open, &high, &low, &close, &mut too_small).unwrap_err());
    assert_eq!(too_small, vec![sentinel]);
    assert_invalid_input(CCI(&open, &high, &low, &close, 5, &mut too_small).unwrap_err());
    assert_eq!(too_small, vec![sentinel]);
    assert_invalid_input(MFI(&open, &high, &low, &close, &volume, 5, &mut too_small).unwrap_err());
    assert_eq!(too_small, vec![sentinel]);
    assert_invalid_input(ULTOSC(&high, &low, &close, 3, 5, 7, &mut too_small).unwrap_err());
    assert_eq!(too_small, vec![sentinel]);
}

#[test]
fn parameters_lookbacks_insufficient_data_and_empty_series_follow_each_definition() {
    assert!(matches!(
        CCIConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        CCIConfig::new(100_001),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        MFIConfig::new(1),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        MFIConfig::new(100_001),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        ULTOSCConfig::new(0, 5, 7),
        Err(TalibError::InvalidPeriod { .. })
    ));
    assert!(matches!(
        ULTOSCConfig::new(3, 5, 100_001),
        Err(TalibError::InvalidPeriod { .. })
    ));

    assert_eq!(BOPConfig::new().lookback(), 0);
    assert_eq!(CCIConfig::new(14).unwrap().lookback(), 13);
    assert_eq!(MFIConfig::new(14).unwrap().lookback(), 14);
    let ultosc = ULTOSCConfig::new(28, 7, 14).unwrap();
    assert_eq!(ultosc.periods(), [7, 14, 28]);
    assert_eq!(ultosc.lookback(), 28);

    let mut none = [];
    assert_eq!(
        BOP(&[], &[], &[], &[], &mut none).unwrap(),
        OutputRange::empty()
    );
    assert_eq!(
        CCI(&[], &[], &[], &[], 2, &mut none).unwrap(),
        OutputRange::empty()
    );
    assert_eq!(
        MFI(&[], &[], &[], &[], &[], 2, &mut none).unwrap(),
        OutputRange::empty()
    );
    assert_eq!(
        ULTOSC(&[], &[], &[], 2, 3, 4, &mut none).unwrap(),
        OutputRange::empty()
    );

    let one = [1.0 as Float];
    assert!(matches!(
        CCI(&one, &one, &one, &one, 2, &mut none),
        Err(TalibError::InsufficientData {
            required: 2,
            actual: 1
        })
    ));
    let two = [1.0 as Float; 2];
    assert!(matches!(
        MFI(&two, &two, &two, &two, &two, 2, &mut none),
        Err(TalibError::InsufficientData {
            required: 3,
            actual: 2
        })
    ));
    let four = [1.0 as Float; 4];
    assert!(matches!(
        ULTOSC(&four, &four, &four, 2, 3, 4, &mut none),
        Err(TalibError::InsufficientData {
            required: 5,
            actual: 4
        })
    ));
}

#[test]
fn prepared_capacity_rejections_preserve_output_and_runner_reuse() {
    let (open, high, low, close, volume) = ordinary();
    let sentinel = -777.0 as Float;

    let mut bop_runner = BOPConfig::new().prepare_batch(open.len() - 1).unwrap();
    let mut bop_out = vec![sentinel; open.len()];
    assert!(matches!(
        bop_runner.compute_into(
            BOPInput {
                open: &open,
                high: &high,
                low: &low,
                close: &close
            },
            &mut bop_out
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(bop_out.iter().all(|&value| value == sentinel));

    let cci = CCIConfig::new(5).unwrap();
    let mut cci_runner = cci.prepare_batch(high.len() - 1).unwrap();
    let mut cci_out = vec![sentinel; high.len() - cci.lookback()];
    assert!(matches!(
        cci_runner.compute_into(
            CCIInput {
                open: &open,
                high: &high,
                low: &low,
                close: &close
            },
            &mut cci_out
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(cci_out.iter().all(|&value| value == sentinel));

    let mfi = MFIConfig::new(5).unwrap();
    let mut mfi_runner = mfi.prepare_batch(high.len() - 1).unwrap();
    let mut mfi_out = vec![sentinel; high.len() - mfi.lookback()];
    assert!(matches!(
        mfi_runner.compute_into(
            MFIInput {
                open: &open,
                high: &high,
                low: &low,
                close: &close,
                volume: &volume
            },
            &mut mfi_out
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(mfi_out.iter().all(|&value| value == sentinel));

    let ultosc = ULTOSCConfig::new(3, 5, 7).unwrap();
    let mut ultosc_runner = ultosc.prepare_batch(high.len() - 1).unwrap();
    let mut ultosc_out = vec![sentinel; high.len() - ultosc.lookback()];
    assert!(matches!(
        ultosc_runner.compute_into(
            ULTOSCInput {
                high: &high,
                low: &low,
                close: &close
            },
            &mut ultosc_out
        ),
        Err(TalibError::PreparedCapacityExceeded { .. })
    ));
    assert!(ultosc_out.iter().all(|&value| value == sentinel));

    let mut runner = ultosc.prepare_batch(high.len()).unwrap();
    runner
        .compute_into(
            ULTOSCInput {
                high: &high,
                low: &low,
                close: &close,
            },
            &mut ultosc_out,
        )
        .unwrap();
    let first = ultosc_out.clone();
    ultosc_out.fill(sentinel);
    runner
        .compute_into(
            ULTOSCInput {
                high: &high,
                low: &low,
                close: &close,
            },
            &mut ultosc_out,
        )
        .unwrap();
    assert_eq!(ultosc_out, first);
}

#[test]
fn prepared_capacity_precedes_alignment_for_every_composite_input_column() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let mut output = [];
    let capacity_error = TalibError::PreparedCapacityExceeded {
        max_input_len: within.len(),
        actual_input_len: oversized.len(),
    };

    let mut bop = BOPConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        bop.compute_into(
            BOPInput {
                open: &within,
                high: &oversized,
                low: &within,
                close: &within,
            },
            &mut output,
        )
        .unwrap_err(),
        capacity_error
    );

    let mut cci = CCIConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        cci.compute_into(
            CCIInput {
                open: &within,
                high: &oversized,
                low: &within,
                close: &within,
            },
            &mut output,
        )
        .unwrap_err(),
        capacity_error
    );

    let mut mfi = MFIConfig::new(2)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        mfi.compute_into(
            MFIInput {
                open: &within,
                high: &within,
                low: &within,
                close: &within,
                volume: &oversized,
            },
            &mut output,
        )
        .unwrap_err(),
        capacity_error
    );

    let mut ultosc = ULTOSCConfig::new(2, 3, 4)
        .unwrap()
        .prepare_batch(within.len())
        .unwrap();
    assert_eq!(
        ultosc
            .compute_into(
                ULTOSCInput {
                    high: &within,
                    low: &within,
                    close: &oversized,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );
}

#[test]
fn rejected_stream_ticks_preserve_independent_state() {
    let (open, high, low, close, volume) = ordinary();
    let mut bop_stream = BOPConfig::new().stream().unwrap();
    assert_invalid_input(
        bop_stream
            .next(BOPTick {
                open: Float::NAN,
                high: high[0],
                low: low[0],
                close: close[0],
            })
            .unwrap_err(),
    );
    let valid_bop = bop_stream
        .next(BOPTick {
            open: open[0],
            high: high[0],
            low: low[0],
            close: close[0],
        })
        .unwrap()
        .unwrap();
    assert_close(&[valid_bop], &reference::BOP_EXPECTED[..1]);

    let cci = CCIConfig::new(5).unwrap();
    let mut cci_stream = cci.stream().unwrap();
    for index in 0..4 {
        cci_stream
            .next(CCITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap();
    }
    let mut cci_control = cci_stream.clone();
    assert_invalid_input(
        cci_stream
            .next(CCITick {
                open: open[4],
                high: Float::NAN,
                low: low[4],
                close: close[4],
            })
            .unwrap_err(),
    );
    let valid_cci = CCITick {
        open: open[4],
        high: high[4],
        low: low[4],
        close: close[4],
    };
    assert_eq!(
        cci_stream.next(valid_cci).unwrap(),
        cci_control.next(valid_cci).unwrap()
    );

    let mfi = MFIConfig::new(5).unwrap();
    let mut mfi_stream = mfi.stream().unwrap();
    for index in 0..5 {
        mfi_stream
            .next(MFITick {
                open: open[index],
                high: high[index],
                low: low[index],
                close: close[index],
                volume: volume[index],
            })
            .unwrap();
    }
    let mut mfi_control = mfi_stream.clone();
    assert_invalid_input(
        mfi_stream
            .next(MFITick {
                open: open[5],
                high: high[5],
                low: low[5],
                close: close[5],
                volume: Float::INFINITY,
            })
            .unwrap_err(),
    );
    let valid_mfi = MFITick {
        open: open[5],
        high: high[5],
        low: low[5],
        close: close[5],
        volume: volume[5],
    };
    assert_eq!(
        mfi_stream.next(valid_mfi).unwrap(),
        mfi_control.next(valid_mfi).unwrap()
    );

    let ultosc = ULTOSCConfig::new(3, 5, 7).unwrap();
    let mut ultosc_stream = ultosc.stream().unwrap();
    for index in 0..7 {
        ultosc_stream
            .next(ULTOSCTick {
                high: high[index],
                low: low[index],
                close: close[index],
            })
            .unwrap();
    }
    let mut ultosc_control = ultosc_stream.clone();
    assert_invalid_input(
        ultosc_stream
            .next(ULTOSCTick {
                high: high[7],
                low: Float::NEG_INFINITY,
                close: close[7],
            })
            .unwrap_err(),
    );
    let valid_ultosc = ULTOSCTick {
        high: high[7],
        low: low[7],
        close: close[7],
    };
    assert_eq!(
        ultosc_stream.next(valid_ultosc).unwrap(),
        ultosc_control.next(valid_ultosc).unwrap()
    );

    let mut independent = ultosc.stream().unwrap();
    assert!(independent
        .next(ULTOSCTick {
            high: high[0],
            low: low[0],
            close: close[0],
        })
        .unwrap()
        .is_none());
    assert!(ultosc_stream.next(valid_ultosc).unwrap().is_some());
}

#[test]
fn flat_zero_volume_trend_reversal_and_scaling_edges_are_explicit() {
    let flat = vec![100.0 as Float; 10];
    let zero_volume = vec![0.0 as Float; 10];
    let mut output = vec![Float::NAN; 10];

    BOP(&flat, &flat, &flat, &flat, &mut output).unwrap();
    assert!(output.iter().all(|&value| value == 0.0 as Float));
    let cci_range = CCI(&flat, &flat, &flat, &flat, 3, &mut output).unwrap();
    assert!(output[..cci_range.nb_element]
        .iter()
        .all(|&value| value == 0.0 as Float));
    let mfi_range = MFI(&flat, &flat, &flat, &flat, &flat, 3, &mut output).unwrap();
    assert!(output[..mfi_range.nb_element]
        .iter()
        .all(|&value| value == 0.0 as Float));
    let ultosc_range = ULTOSC(&flat, &flat, &flat, 2, 3, 4, &mut output).unwrap();
    assert!(output[..ultosc_range.nb_element]
        .iter()
        .all(|&value| value == 0.0 as Float));

    let close: Vec<Float> = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0]
        .iter()
        .map(|&value| value as Float)
        .collect();
    let high: Vec<Float> = close.iter().map(|&value| value + 1.0 as Float).collect();
    let low: Vec<Float> = close.iter().map(|&value| value - 1.0 as Float).collect();
    let open: Vec<Float> = close.iter().map(|&value| value - 0.5 as Float).collect();
    let volume = vec![100.0 as Float; close.len()];

    let bop = BOPConfig::new()
        .compute(BOPInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    assert!(bop
        .values()
        .iter()
        .all(|&value| (-1.0 as Float..=1.0 as Float).contains(&value)));

    let mfi = MFIConfig::new(3)
        .unwrap()
        .compute(MFIInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
            volume: &volume,
        })
        .unwrap();
    assert!(mfi
        .values()
        .iter()
        .all(|&value| (0.0 as Float..=100.0 as Float).contains(&value)));
    let zero_mfi = MFIConfig::new(3)
        .unwrap()
        .compute(MFIInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
            volume: &zero_volume,
        })
        .unwrap();
    assert!(zero_mfi.values().iter().all(|&value| value == 0.0 as Float));

    let ultosc = ULTOSCConfig::new(2, 3, 4)
        .unwrap()
        .compute(ULTOSCInput {
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    assert!(ultosc
        .values()
        .iter()
        .all(|&value| (0.0 as Float..=100.0 as Float).contains(&value)));

    let scaled_open: Vec<Float> = open
        .iter()
        .map(|&value| value * 10.0 as Float + 500.0 as Float)
        .collect();
    let scaled_high: Vec<Float> = high
        .iter()
        .map(|&value| value * 10.0 as Float + 500.0 as Float)
        .collect();
    let scaled_low: Vec<Float> = low
        .iter()
        .map(|&value| value * 10.0 as Float + 500.0 as Float)
        .collect();
    let scaled_close: Vec<Float> = close
        .iter()
        .map(|&value| value * 10.0 as Float + 500.0 as Float)
        .collect();
    let scaled_bop = BOPConfig::new()
        .compute(BOPInput {
            open: &scaled_open,
            high: &scaled_high,
            low: &scaled_low,
            close: &scaled_close,
        })
        .unwrap();
    for (&left, &right) in bop.values().iter().zip(scaled_bop.values()) {
        assert!((left - right).abs() <= tolerance());
    }
    let scaled_volume: Vec<Float> = volume.iter().map(|&value| value * 10.0 as Float).collect();
    let scaled_mfi = MFIConfig::new(3)
        .unwrap()
        .compute(MFIInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
            volume: &scaled_volume,
        })
        .unwrap();
    for (&left, &right) in mfi.values().iter().zip(scaled_mfi.values()) {
        assert!((left - right).abs() <= tolerance() * Float::max(1.0 as Float, left.abs()));
    }
    let cci = CCIConfig::new(3).unwrap();
    let original_cci = cci
        .compute(CCIInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    let scaled_cci = cci
        .compute(CCIInput {
            open: &scaled_open,
            high: &scaled_high,
            low: &scaled_low,
            close: &scaled_close,
        })
        .unwrap();
    for (&left, &right) in original_cci.values().iter().zip(scaled_cci.values()) {
        assert!((left - right).abs() <= tolerance() * Float::max(1.0 as Float, left.abs()));
    }
    let scaled_ultosc = ULTOSCConfig::new(4, 2, 3)
        .unwrap()
        .compute(ULTOSCInput {
            high: &scaled_high,
            low: &scaled_low,
            close: &scaled_close,
        })
        .unwrap();
    for (&left, &right) in ultosc.values().iter().zip(scaled_ultosc.values()) {
        assert!((left - right).abs() <= tolerance() * Float::max(1.0 as Float, left.abs()));
    }
}

#[test]
fn ultimate_oscillator_period_order_is_definitionally_irrelevant() {
    let (_, high, low, close, _) = ordinary();
    let permutations = [
        [3, 5, 7],
        [3, 7, 5],
        [5, 3, 7],
        [5, 7, 3],
        [7, 3, 5],
        [7, 5, 3],
    ];
    let expected = ULTOSCConfig::new(3, 5, 7)
        .unwrap()
        .compute(ULTOSCInput {
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    for periods in permutations {
        let config = ULTOSCConfig::new(periods[0], periods[1], periods[2]).unwrap();
        assert_eq!(config.periods(), [3, 5, 7]);
        let actual = config
            .compute(ULTOSCInput {
                high: &high,
                low: &low,
                close: &close,
            })
            .unwrap();
        assert_eq!(actual.values(), expected.values());
    }
}
