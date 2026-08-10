//! Public-seam qualification for the Pattern Recognition foundation tracers.

#[path = "fixtures/pattern_recognition_reference.rs"]
mod reference;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ta_core::pattern_recognition::{
    Candle, CandleInput, CandleRangeKind, CandleSetting, CandleSettingType, CandleSettings,
    PatternDirection, PatternSignal, PatternStrength, Penetration, CDL2CROWSConfig,
    CDL3BLACKCROWSConfig, CDL3INSIDEConfig, CDL3LINESTRIKEConfig, CDL3OUTSIDEConfig,
    CDL3STARSINSOUTHConfig, CDL3WHITESOLDIERSConfig, CDLADVANCEBLOCKConfig,
    CDLABANDONEDBABYConfig, CDLBELTHOLDConfig, CDLCLOSINGMARUBOZUConfig,
    CDLCONCEALBABYSWALLConfig, CDLCOUNTERATTACKConfig,
    CDLDARKCLOUDCOVERConfig, CDLDOJIConfig, CDLDOJISTARConfig, CDLDRAGONFLYDOJIConfig,
    CDLENGULFINGConfig, CDLEVENINGDOJISTARConfig, CDLEVENINGSTARConfig,
    CDLGAPSIDESIDEWHITEConfig, CDLGRAVESTONEDOJIConfig, CDLHAMMERConfig,
    CDLHANGINGMANConfig, CDLHARAMIConfig, CDLHARAMICROSSConfig, CDLHIGHWAVEConfig,
    CDLHOMINGPIGEONConfig, CDLIDENTICAL3CROWSConfig, CDLINNECKConfig,
    CDLINVERTEDHAMMERConfig,
    CDLKICKINGBYLENGTHConfig, CDLKICKINGConfig, CDLLONGLEGGEDDOJIConfig,
    CDLLONGLINEConfig, CDLMARUBOZUConfig, CDLMATCHINGLOWConfig, CDLMORNINGDOJISTARConfig,
    CDLMORNINGSTARConfig, CDLONNECKConfig, CDLPIERCINGConfig, CDLRICKSHAWMANConfig,
    CDLSEPARATINGLINESConfig, CDLSHOOTINGSTARConfig, CDLSHORTLINEConfig,
    CDLSPINNINGTOPConfig, CDLSTALLEDPATTERNConfig, CDLSTICKSANDWICHConfig, CDLTAKURIConfig,
    CDLTASUKIGAPConfig,
    CDLTHRUSTINGConfig, CDLTRISTARConfig, CDLUNIQUE3RIVERConfig, CDLUPSIDEGAP2CROWSConfig,
    CDLXSIDEGAP3METHODSConfig, CDLBREAKAWAYConfig, CDLHIKKAKEConfig, CDLHIKKAKEMODConfig,
    CDLLADDERBOTTOMConfig, CDLMATHOLDConfig, CDLRISEFALL3METHODSConfig,
};
use ta_core::{
    Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation, TalibError,
};

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_EVENTS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() {
                ALLOCATION_EVENTS.with(|events| events.set(events.get() + 1));
            }
        });
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() {
                ALLOCATION_EVENTS.with(|events| events.set(events.get() + 1));
            }
        });
        System.realloc(pointer, layout, new_size)
    }
}

fn allocation_events_during(operation: impl FnOnce()) -> usize {
    ALLOCATION_EVENTS.with(|events| events.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    operation();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    ALLOCATION_EVENTS.with(Cell::get)
}

#[derive(Debug)]
struct Series {
    open: Vec<Float>,
    high: Vec<Float>,
    low: Vec<Float>,
    close: Vec<Float>,
}

impl Series {
    fn from_fixture(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Self {
        Self {
            open: open.iter().map(|&value| value as Float).collect(),
            high: high.iter().map(|&value| value as Float).collect(),
            low: low.iter().map(|&value| value as Float).collect(),
            close: close.iter().map(|&value| value as Float).collect(),
        }
    }

    fn input(&self) -> CandleInput<'_> {
        CandleInput {
            open: &self.open,
            high: &self.high,
            low: &self.low,
            close: &self.close,
        }
    }

    fn candles(&self) -> Vec<Candle> {
        (0..self.open.len())
            .map(|index| Candle {
                open: self.open[index],
                high: self.high[index],
                low: self.low[index],
                close: self.close[index],
            })
            .collect()
    }
}

fn expected_codes(f64_codes: &[i32], f32_codes: &[i32]) -> Vec<PatternSignal> {
    #[cfg(feature = "f32")]
    let codes = f32_codes;
    #[cfg(not(feature = "f32"))]
    let codes = {
        let _ = f32_codes;
        f64_codes
    };
    codes
        .iter()
        .map(|&code| PatternSignal::from_talib_code(code).unwrap())
        .collect()
}

fn qualify_pattern_fixture<C>(
    config: C,
    series: &Series,
    lookback: usize,
    expected: &[PatternSignal],
) where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C: IndicatorConfig<
        Input<'a> = CandleInput<'a>,
        OutputMut<'a> = &'a mut [PatternSignal],
    >,
    C::BatchRunner: PreparedBatchRunner<C>,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal>,
{
    assert_eq!(config.lookback(), lookback);

    let owned = config.compute(series.input()).unwrap();
    assert_eq!(owned.source_len(), series.open.len());
    assert_eq!(
        owned.range(),
        OutputRange::new(lookback, series.open.len() - lookback)
    );
    assert_eq!(owned.values(), expected);
    let payload = owned.into_values();
    assert_eq!(payload.len(), expected.len());
    assert_eq!(payload.capacity(), expected.len());

    let sentinel = PatternSignal::Match {
        direction: PatternDirection::Bearish,
        strength: PatternStrength::Confirmed,
    };
    let misaligned = CandleInput {
        open: &series.open,
        high: &series.high[..series.high.len() - 1],
        low: &series.low,
        close: &series.close,
    };
    let mut validation_output = vec![sentinel; expected.len() + 1];
    assert!(config
        .compute_into(misaligned, &mut validation_output)
        .is_err());
    assert!(validation_output.iter().all(|&value| value == sentinel));
    let mut caller_output = vec![sentinel; expected.len() + 2];
    let range = config
        .compute_into(series.input(), &mut caller_output)
        .unwrap();
    assert_eq!(range, OutputRange::new(lookback, expected.len()));
    assert_eq!(&caller_output[..expected.len()], expected);
    assert_eq!(&caller_output[expected.len()..], &[sentinel, sentinel]);

    let mut prepared = config.prepare_batch(series.open.len()).unwrap();
    assert!(prepared
        .compute_into(misaligned, &mut validation_output)
        .is_err());
    assert!(validation_output.iter().all(|&value| value == sentinel));
    let mut prepared_output = vec![sentinel; expected.len() + 1];
    let prepared_range = prepared
        .compute_into(series.input(), &mut prepared_output)
        .unwrap();
    assert_eq!(prepared_range, range);
    assert_eq!(&prepared_output[..expected.len()], expected);
    assert_eq!(prepared_output[expected.len()], sentinel);
    prepared_output.fill(sentinel);
    assert_eq!(
        prepared
            .compute_into(series.input(), &mut prepared_output)
            .unwrap(),
        range
    );
    assert_eq!(&prepared_output[..expected.len()], expected);

    let mut independent = config.prepare_batch(series.open.len()).unwrap();
    let mut independent_output = vec![sentinel; expected.len()];
    assert_eq!(
        independent
            .compute_into(series.input(), &mut independent_output)
            .unwrap(),
        range
    );
    assert_eq!(independent_output, expected);

    let candles = series.candles();
    let mut stream = config.stream().unwrap();
    let mut streamed = Vec::with_capacity(expected.len());
    for (index, candle) in candles.iter().copied().enumerate() {
        let result = stream.next(candle).unwrap();
        if index < lookback {
            assert_eq!(result, None, "source position {index}");
        } else {
            streamed.push(result.expect("valid post-Lookback Pattern Signal"));
        }
    }
    assert_eq!(streamed, expected);

    stream.reset();
    let replay: Vec<_> = candles
        .iter()
        .copied()
        .filter_map(|candle| stream.next(candle).unwrap())
        .collect();
    assert_eq!(replay, expected);
}

#[test]
fn candle_settings_defaults_and_validation_match_pinned_talib() {
    use CandleRangeKind::{HighLow, RealBody, Shadows};
    let expected = [
        (CandleSettingType::BodyLong, RealBody, 10, 1.0),
        (CandleSettingType::BodyVeryLong, RealBody, 10, 3.0),
        (CandleSettingType::BodyShort, RealBody, 10, 1.0),
        (CandleSettingType::BodyDoji, HighLow, 10, 0.1),
        (CandleSettingType::ShadowLong, RealBody, 0, 1.0),
        (CandleSettingType::ShadowVeryLong, RealBody, 0, 2.0),
        (CandleSettingType::ShadowShort, Shadows, 10, 1.0),
        (CandleSettingType::ShadowVeryShort, HighLow, 10, 0.1),
        (CandleSettingType::Near, HighLow, 5, 0.2),
        (CandleSettingType::Far, HighLow, 5, 0.6),
        (CandleSettingType::Equal, HighLow, 5, 0.05),
    ];
    let settings = CandleSettings::default();
    assert_eq!(CandleSettingType::ALL.len(), 11);
    for (setting_type, range_kind, period, factor) in expected {
        let setting = settings.setting(setting_type);
        assert_eq!(setting.range_kind(), range_kind);
        assert_eq!(setting.average_period(), period);
        assert_eq!(setting.factor(), factor as Float);
    }

    assert!(matches!(
        CandleSetting::new(RealBody, 100_001, 1.0 as Float),
        Err(TalibError::InvalidPeriod { .. })
    ));
    for factor in [Float::NAN, Float::INFINITY, -1.0 as Float] {
        assert!(matches!(
            CandleSetting::new(RealBody, 10, factor),
            Err(TalibError::InvalidInput { .. })
        ));
    }
    assert!(CandleSetting::new(Shadows, 0, 0.0 as Float).is_ok());
}

#[test]
fn penetration_accepts_all_and_only_finite_nonnegative_values() {
    for value in [0.0 as Float, 0.3 as Float, 1.0 as Float, 4.0 as Float] {
        assert_eq!(Penetration::new(value).unwrap().value(), value);
    }
    for value in [-0.01 as Float, Float::NAN, Float::INFINITY] {
        assert!(matches!(
            Penetration::new(value),
            Err(TalibError::InvalidInput { .. })
        ));
    }
}

#[test]
fn pattern_signal_talib_projection_is_exact_and_checked() {
    for code in [0, -200, -100, -80, 80, 100, 200] {
        let signal = PatternSignal::from_talib_code(code).unwrap();
        assert_eq!(signal.to_talib_code(), code);
        assert_eq!(PatternSignal::try_from(code).unwrap(), signal);
    }
    for code in [1, -79, 79, 81, 180, 201, i32::MIN, i32::MAX] {
        assert!(PatternSignal::from_talib_code(code).is_err(), "{code}");
    }
}

#[test]
fn pinned_default_and_custom_doji_fixtures_qualify_all_four_modes() {
    assert_eq!(reference::TALIB_VERSION, "0.7.1");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "2247d599bddf37ed37e3a709371517e46efc66f6"
    );
    assert_eq!(reference::TALIB_SOURCE_ARCHIVE_SHA256.len(), 64);

    let default_series = Series::from_fixture(
        reference::DOJI_DEFAULT_OPEN,
        reference::DOJI_DEFAULT_HIGH,
        reference::DOJI_DEFAULT_LOW,
        reference::DOJI_DEFAULT_CLOSE,
    );
    let default_expected = expected_codes(
        reference::DOJI_DEFAULT_F64_CODES,
        reference::DOJI_DEFAULT_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLDOJIConfig::default(),
        &default_series,
        reference::DOJI_DEFAULT_LOOKBACK,
        &default_expected,
    );

    let custom_settings = CandleSettings::default().with_setting(
        CandleSettingType::BodyDoji,
        CandleSetting::new(CandleRangeKind::RealBody, 3, 0.5 as Float).unwrap(),
    );
    let custom_series = Series::from_fixture(
        reference::DOJI_CUSTOM_OPEN,
        reference::DOJI_CUSTOM_HIGH,
        reference::DOJI_CUSTOM_LOW,
        reference::DOJI_CUSTOM_CLOSE,
    );
    let custom_expected = expected_codes(
        reference::DOJI_CUSTOM_F64_CODES,
        reference::DOJI_CUSTOM_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLDOJIConfig::new(custom_settings).unwrap(),
        &custom_series,
        reference::DOJI_CUSTOM_LOOKBACK,
        &custom_expected,
    );
}

#[test]
fn pinned_engulfing_fixture_qualifies_partial_standard_and_both_directions() {
    let series = Series::from_fixture(
        reference::ENGULFING_OPEN,
        reference::ENGULFING_HIGH,
        reference::ENGULFING_LOW,
        reference::ENGULFING_CLOSE,
    );
    let expected = expected_codes(
        reference::ENGULFING_F64_CODES,
        reference::ENGULFING_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLENGULFINGConfig::default(),
        &series,
        reference::ENGULFING_LOOKBACK,
        &expected,
    );
}

#[test]
fn independently_reasoned_boundaries_lock_doji_and_engulfing_semantics() {
    let period_zero = CandleSettings::default().with_setting(
        CandleSettingType::BodyDoji,
        CandleSetting::new(CandleRangeKind::Shadows, 0, 1.0 as Float).unwrap(),
    );
    let doji = CDLDOJIConfig::new(period_zero).unwrap();
    let open = [10.0 as Float, 10.0];
    let high = [12.0 as Float, 10.5];
    let low = [8.0 as Float, 9.5];
    let close = [11.0 as Float, 12.0];
    let output = doji
        .compute(CandleInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    assert_eq!(output.range(), OutputRange::new(0, 2));
    assert_eq!(output.values()[0].to_talib_code(), 100);
    assert_eq!(output.values()[1], PatternSignal::NoMatch);

    // Exact doji is White when color is consulted, so the next black body
    // engulfs it and emits a bearish Standard result.
    let open = [8.0 as Float, 10.0, 11.0];
    let high = [9.0 as Float, 10.0, 12.0];
    let low = [7.0 as Float, 10.0, 8.0];
    let close = [8.0 as Float, 10.0, 9.0];
    let engulfing = CDLENGULFINGConfig::default()
        .compute(CandleInput {
            open: &open,
            high: &high,
            low: &low,
            close: &close,
        })
        .unwrap();
    assert_eq!(engulfing.values()[0].to_talib_code(), -100);

    let inert_settings = CandleSettings::default().with_setting(
        CandleSettingType::BodyDoji,
        CandleSetting::new(CandleRangeKind::HighLow, 100_000, 99.0 as Float).unwrap(),
    );
    assert_eq!(
        CDLENGULFINGConfig::new(inert_settings).unwrap().lookback(),
        2
    );
}

#[test]
fn validation_failures_precede_mutation_and_stream_retries_exactly() {
    let config = CDLDOJIConfig::default();
    let empty = config
        .compute(CandleInput {
            open: &[],
            high: &[],
            low: &[],
            close: &[],
        })
        .unwrap();
    assert_eq!(empty.source_len(), 0);
    assert_eq!(empty.range(), OutputRange::empty());
    assert_eq!(empty.values().len(), 0);
    assert_eq!(empty.values().capacity(), 0);

    let series = Series::from_fixture(
        reference::DOJI_DEFAULT_OPEN,
        reference::DOJI_DEFAULT_HIGH,
        reference::DOJI_DEFAULT_LOW,
        reference::DOJI_DEFAULT_CLOSE,
    );
    let sentinel = PatternSignal::Match {
        direction: PatternDirection::Bearish,
        strength: PatternStrength::Confirmed,
    };
    let mut output = vec![sentinel; 8];
    let misaligned = CandleInput {
        open: &series.open,
        high: &series.high[..series.high.len() - 1],
        low: &series.low,
        close: &series.close,
    };
    assert!(config.compute_into(misaligned, &mut output).is_err());
    assert!(output.iter().all(|&value| value == sentinel));

    for column in 0..4 {
        let mut open = series.open.clone();
        let mut high = series.high.clone();
        let mut low = series.low.clone();
        let mut close = series.close.clone();
        match column {
            0 => open[3] = Float::NAN,
            1 => high[3] = Float::INFINITY,
            2 => low[3] = Float::NEG_INFINITY,
            3 => close[3] = Float::NAN,
            _ => unreachable!(),
        }
        output.fill(sentinel);
        assert!(config
            .compute_into(
                CandleInput {
                    open: &open,
                    high: &high,
                    low: &low,
                    close: &close,
                },
                &mut output,
            )
            .is_err());
        assert!(output.iter().all(|&value| value == sentinel));
    }

    let insufficient = CandleInput {
        open: &series.open[..config.lookback()],
        high: &series.high[..config.lookback()],
        low: &series.low[..config.lookback()],
        close: &series.close[..config.lookback()],
    };
    output.fill(sentinel);
    assert!(matches!(
        config.compute_into(insufficient, &mut output),
        Err(TalibError::InsufficientData { .. })
    ));
    assert!(output.iter().all(|&value| value == sentinel));

    let mut short_output = [sentinel; 1];
    assert!(config
        .compute_into(series.input(), &mut short_output)
        .is_err());
    assert_eq!(short_output, [sentinel]);

    let mut prepared = config.prepare_batch(series.open.len() - 1).unwrap();
    let error = prepared.compute_into(misaligned, &mut []).unwrap_err();
    assert!(matches!(error, TalibError::PreparedCapacityExceeded { .. }));

    let candles = series.candles();
    let mut tested = config.stream().unwrap();
    let mut control = config.stream().unwrap();
    for candle in candles.iter().copied().take(5) {
        assert_eq!(tested.next(candle).unwrap(), control.next(candle).unwrap());
    }
    let invalid = Candle {
        close: Float::NAN,
        ..candles[5]
    };
    assert!(tested.next(invalid).is_err());
    for candle in candles.iter().copied().skip(5) {
        assert_eq!(tested.next(candle).unwrap(), control.next(candle).unwrap());
    }
}

#[test]
fn prepared_and_streaming_steady_state_do_not_allocate_or_grow() {
    let series = Series::from_fixture(
        reference::DOJI_DEFAULT_OPEN,
        reference::DOJI_DEFAULT_HIGH,
        reference::DOJI_DEFAULT_LOW,
        reference::DOJI_DEFAULT_CLOSE,
    );
    let config = CDLDOJIConfig::default();
    let mut prepared = config.prepare_batch(series.open.len()).unwrap();
    let mut output = vec![PatternSignal::NoMatch; reference::DOJI_DEFAULT_F64_CODES.len()];
    assert_eq!(
        allocation_events_during(|| {
            prepared.compute_into(series.input(), &mut output).unwrap();
            prepared.compute_into(series.input(), &mut output).unwrap();
        }),
        0
    );

    let candles = series.candles();
    let mut stream = config.stream().unwrap();
    assert_eq!(
        allocation_events_during(|| {
            for candle in candles.iter().copied() {
                stream.next(candle).unwrap();
            }
            stream.reset();
            for candle in candles.iter().copied() {
                stream.next(candle).unwrap();
            }
        }),
        0
    );

    let engulfing_series = Series::from_fixture(
        reference::ENGULFING_OPEN,
        reference::ENGULFING_HIGH,
        reference::ENGULFING_LOW,
        reference::ENGULFING_CLOSE,
    );
    let engulfing_config = CDLENGULFINGConfig::default();
    let mut engulfing_runner = engulfing_config
        .prepare_batch(engulfing_series.open.len())
        .unwrap();
    let mut engulfing_output =
        vec![PatternSignal::NoMatch; reference::ENGULFING_F64_CODES.len()];
    assert_eq!(
        allocation_events_during(|| {
            engulfing_runner
                .compute_into(engulfing_series.input(), &mut engulfing_output)
                .unwrap();
        }),
        0
    );

    let engulfing_candles = engulfing_series.candles();
    let mut engulfing_stream = engulfing_config.stream().unwrap();
    assert_eq!(
        allocation_events_during(|| {
            for candle in engulfing_candles.iter().copied() {
                engulfing_stream.next(candle).unwrap();
            }
            engulfing_stream.reset();
        }),
        0
    );
}

fn single_candle_series() -> Series {
    Series::from_fixture(
        reference::SINGLE_CANDLE_OPEN,
        reference::SINGLE_CANDLE_HIGH,
        reference::SINGLE_CANDLE_LOW,
        reference::SINGLE_CANDLE_CLOSE,
    )
}

fn custom_single_candle_settings() -> CandleSettings {
    use CandleRangeKind::{HighLow, RealBody, Shadows};
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(RealBody, 3, 1.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(RealBody, 3, 2.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyDoji,
            CandleSetting::new(HighLow, 3, 0.125 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowLong,
            CandleSetting::new(RealBody, 0, 1.25 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryLong,
            CandleSetting::new(RealBody, 0, 3.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowShort,
            CandleSetting::new(Shadows, 3, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(HighLow, 3, 0.0625 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Near,
            CandleSetting::new(HighLow, 3, 0.125 as Float).unwrap(),
        )
}

macro_rules! qualify_single_candle {
    (
        $series:expr, $default:expr, $custom:expr,
        $default_lookback:ident, $default_f64:ident, $default_f32:ident,
        $custom_lookback:ident, $custom_f64:ident, $custom_f32:ident
    ) => {{
        let default_expected =
            expected_codes(reference::$default_f64, reference::$default_f32);
        assert!(default_expected
            .iter()
            .any(|signal| matches!(signal, PatternSignal::Match { .. })));
        qualify_pattern_fixture(
            $default,
            $series,
            reference::$default_lookback,
            &default_expected,
        );

        let custom_expected = expected_codes(reference::$custom_f64, reference::$custom_f32);
        assert!(custom_expected
            .iter()
            .any(|signal| matches!(signal, PatternSignal::Match { .. })));
        assert_eq!($custom.warm_up(), reference::$custom_lookback);
        qualify_pattern_fixture(
            $custom,
            $series,
            reference::$custom_lookback,
            &custom_expected,
        );
    }};
}

#[test]
fn pinned_single_candle_oracles_qualify_every_definition_through_the_public_seam() {
    let series = single_candle_series();
    let settings = custom_single_candle_settings();
    qualify_single_candle!(
        &series, CDLBELTHOLDConfig::default(), CDLBELTHOLDConfig::new(settings).unwrap(),
        BELTHOLD_DEFAULT_LOOKBACK, BELTHOLD_DEFAULT_F64_CODES, BELTHOLD_DEFAULT_F32_CODES,
        BELTHOLD_CUSTOM_LOOKBACK, BELTHOLD_CUSTOM_F64_CODES, BELTHOLD_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLCLOSINGMARUBOZUConfig::default(),
        CDLCLOSINGMARUBOZUConfig::new(settings).unwrap(),
        CLOSINGMARUBOZU_DEFAULT_LOOKBACK, CLOSINGMARUBOZU_DEFAULT_F64_CODES,
        CLOSINGMARUBOZU_DEFAULT_F32_CODES, CLOSINGMARUBOZU_CUSTOM_LOOKBACK,
        CLOSINGMARUBOZU_CUSTOM_F64_CODES, CLOSINGMARUBOZU_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLDRAGONFLYDOJIConfig::default(), CDLDRAGONFLYDOJIConfig::new(settings).unwrap(),
        DRAGONFLYDOJI_DEFAULT_LOOKBACK, DRAGONFLYDOJI_DEFAULT_F64_CODES,
        DRAGONFLYDOJI_DEFAULT_F32_CODES, DRAGONFLYDOJI_CUSTOM_LOOKBACK,
        DRAGONFLYDOJI_CUSTOM_F64_CODES, DRAGONFLYDOJI_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLGRAVESTONEDOJIConfig::default(),
        CDLGRAVESTONEDOJIConfig::new(settings).unwrap(),
        GRAVESTONEDOJI_DEFAULT_LOOKBACK, GRAVESTONEDOJI_DEFAULT_F64_CODES,
        GRAVESTONEDOJI_DEFAULT_F32_CODES, GRAVESTONEDOJI_CUSTOM_LOOKBACK,
        GRAVESTONEDOJI_CUSTOM_F64_CODES, GRAVESTONEDOJI_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLHIGHWAVEConfig::default(), CDLHIGHWAVEConfig::new(settings).unwrap(),
        HIGHWAVE_DEFAULT_LOOKBACK, HIGHWAVE_DEFAULT_F64_CODES, HIGHWAVE_DEFAULT_F32_CODES,
        HIGHWAVE_CUSTOM_LOOKBACK, HIGHWAVE_CUSTOM_F64_CODES, HIGHWAVE_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLLONGLEGGEDDOJIConfig::default(),
        CDLLONGLEGGEDDOJIConfig::new(settings).unwrap(),
        LONGLEGGEDDOJI_DEFAULT_LOOKBACK, LONGLEGGEDDOJI_DEFAULT_F64_CODES,
        LONGLEGGEDDOJI_DEFAULT_F32_CODES, LONGLEGGEDDOJI_CUSTOM_LOOKBACK,
        LONGLEGGEDDOJI_CUSTOM_F64_CODES, LONGLEGGEDDOJI_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLLONGLINEConfig::default(), CDLLONGLINEConfig::new(settings).unwrap(),
        LONGLINE_DEFAULT_LOOKBACK, LONGLINE_DEFAULT_F64_CODES, LONGLINE_DEFAULT_F32_CODES,
        LONGLINE_CUSTOM_LOOKBACK, LONGLINE_CUSTOM_F64_CODES, LONGLINE_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLMARUBOZUConfig::default(), CDLMARUBOZUConfig::new(settings).unwrap(),
        MARUBOZU_DEFAULT_LOOKBACK, MARUBOZU_DEFAULT_F64_CODES, MARUBOZU_DEFAULT_F32_CODES,
        MARUBOZU_CUSTOM_LOOKBACK, MARUBOZU_CUSTOM_F64_CODES, MARUBOZU_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLRICKSHAWMANConfig::default(), CDLRICKSHAWMANConfig::new(settings).unwrap(),
        RICKSHAWMAN_DEFAULT_LOOKBACK, RICKSHAWMAN_DEFAULT_F64_CODES,
        RICKSHAWMAN_DEFAULT_F32_CODES, RICKSHAWMAN_CUSTOM_LOOKBACK,
        RICKSHAWMAN_CUSTOM_F64_CODES, RICKSHAWMAN_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLSHORTLINEConfig::default(), CDLSHORTLINEConfig::new(settings).unwrap(),
        SHORTLINE_DEFAULT_LOOKBACK, SHORTLINE_DEFAULT_F64_CODES, SHORTLINE_DEFAULT_F32_CODES,
        SHORTLINE_CUSTOM_LOOKBACK, SHORTLINE_CUSTOM_F64_CODES, SHORTLINE_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLSPINNINGTOPConfig::default(), CDLSPINNINGTOPConfig::new(settings).unwrap(),
        SPINNINGTOP_DEFAULT_LOOKBACK, SPINNINGTOP_DEFAULT_F64_CODES,
        SPINNINGTOP_DEFAULT_F32_CODES, SPINNINGTOP_CUSTOM_LOOKBACK,
        SPINNINGTOP_CUSTOM_F64_CODES, SPINNINGTOP_CUSTOM_F32_CODES
    );
    qualify_single_candle!(
        &series, CDLTAKURIConfig::default(), CDLTAKURIConfig::new(settings).unwrap(),
        TAKURI_DEFAULT_LOOKBACK, TAKURI_DEFAULT_F64_CODES, TAKURI_DEFAULT_F32_CODES,
        TAKURI_CUSTOM_LOOKBACK, TAKURI_CUSTOM_F64_CODES, TAKURI_CUSTOM_F32_CODES
    );
}

#[test]
fn single_candle_evidence_rows_cover_pinned_sources_and_canonical_near_misses() {
    const ROWS: [(&str, &str); 12] = [
        ("CDLBELTHOLD", "cdlbelthold/cdlbelthold.c"),
        ("CDLCLOSINGMARUBOZU", "cdlclosingmarubozu/cdlclosingmarubozu.c"),
        ("CDLDRAGONFLYDOJI", "cdldragonflydoji/cdldragonflydoji.c"),
        ("CDLGRAVESTONEDOJI", "cdlgravestonedoji/cdlgravestonedoji.c"),
        ("CDLHIGHWAVE", "cdlhighwave/cdlhighwave.c"),
        ("CDLLONGLEGGEDDOJI", "cdllongleggeddoji/cdllongleggeddoji.c"),
        ("CDLLONGLINE", "cdllongline/cdllongline.c"),
        ("CDLMARUBOZU", "cdlmarubozu/cdlmarubozu.c"),
        ("CDLRICKSHAWMAN", "cdlrickshawman/cdlrickshawman.c"),
        ("CDLSHORTLINE", "cdlshortline/cdlshortline.c"),
        ("CDLSPINNINGTOP", "cdlspinningtop/cdlspinningtop.c"),
        ("CDLTAKURI", "cdltakuri/cdltakuri.c"),
    ];
    assert_eq!(ROWS.len(), 12);
    assert!(ROWS.iter().all(|(_, source)| source.ends_with(".c")));
}

fn custom_two_candle_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 3, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 3, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowLong,
            CandleSetting::new(CandleRangeKind::RealBody, 3, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyDoji,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.1 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.1 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Near,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.2 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Equal,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.05 as Float).unwrap(),
        )
}

macro_rules! qualify_two_candle {
    (
        $default:expr, $custom:expr,
        $open:ident, $high:ident, $low:ident, $close:ident,
        $default_lookback:ident, $default_f64:ident, $default_f32:ident,
        $custom_lookback:ident, $custom_f64:ident, $custom_f32:ident
    ) => {{
        let series = Series::from_fixture(
            reference::$open,
            reference::$high,
            reference::$low,
            reference::$close,
        );
        qualify_pattern_fixture(
            $default,
            &series,
            reference::$default_lookback,
            &expected_codes(reference::$default_f64, reference::$default_f32),
        );
        qualify_pattern_fixture(
            $custom,
            &series,
            reference::$custom_lookback,
            &expected_codes(reference::$custom_f64, reference::$custom_f32),
        );
    }};
}

#[test]
fn pinned_two_candle_oracles_qualify_every_definition_through_the_public_seam() {
    let settings = custom_two_candle_settings();
    assert_eq!(
        CDLDARKCLOUDCOVERConfig::default().penetration().value(),
        0.5 as Float
    );
    qualify_two_candle!(
        CDLCOUNTERATTACKConfig::default(), CDLCOUNTERATTACKConfig::new(settings).unwrap(),
        COUNTERATTACK_OPEN, COUNTERATTACK_HIGH, COUNTERATTACK_LOW, COUNTERATTACK_CLOSE,
        COUNTERATTACK_DEFAULT_LOOKBACK, COUNTERATTACK_DEFAULT_F64_CODES, COUNTERATTACK_DEFAULT_F32_CODES,
        COUNTERATTACK_CUSTOM_LOOKBACK, COUNTERATTACK_CUSTOM_F64_CODES, COUNTERATTACK_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLDARKCLOUDCOVERConfig::default(),
        CDLDARKCLOUDCOVERConfig::new(settings, Penetration::new(0.25 as Float).unwrap()).unwrap(),
        DARKCLOUDCOVER_OPEN, DARKCLOUDCOVER_HIGH, DARKCLOUDCOVER_LOW, DARKCLOUDCOVER_CLOSE,
        DARKCLOUDCOVER_DEFAULT_LOOKBACK, DARKCLOUDCOVER_DEFAULT_F64_CODES, DARKCLOUDCOVER_DEFAULT_F32_CODES,
        DARKCLOUDCOVER_CUSTOM_LOOKBACK, DARKCLOUDCOVER_CUSTOM_F64_CODES, DARKCLOUDCOVER_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLDOJISTARConfig::default(), CDLDOJISTARConfig::new(settings).unwrap(),
        DOJISTAR_OPEN, DOJISTAR_HIGH, DOJISTAR_LOW, DOJISTAR_CLOSE,
        DOJISTAR_DEFAULT_LOOKBACK, DOJISTAR_DEFAULT_F64_CODES, DOJISTAR_DEFAULT_F32_CODES,
        DOJISTAR_CUSTOM_LOOKBACK, DOJISTAR_CUSTOM_F64_CODES, DOJISTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLHARAMIConfig::default(), CDLHARAMIConfig::new(settings).unwrap(),
        HARAMI_OPEN, HARAMI_HIGH, HARAMI_LOW, HARAMI_CLOSE,
        HARAMI_DEFAULT_LOOKBACK, HARAMI_DEFAULT_F64_CODES, HARAMI_DEFAULT_F32_CODES,
        HARAMI_CUSTOM_LOOKBACK, HARAMI_CUSTOM_F64_CODES, HARAMI_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLHARAMICROSSConfig::default(), CDLHARAMICROSSConfig::new(settings).unwrap(),
        HARAMICROSS_OPEN, HARAMICROSS_HIGH, HARAMICROSS_LOW, HARAMICROSS_CLOSE,
        HARAMICROSS_DEFAULT_LOOKBACK, HARAMICROSS_DEFAULT_F64_CODES, HARAMICROSS_DEFAULT_F32_CODES,
        HARAMICROSS_CUSTOM_LOOKBACK, HARAMICROSS_CUSTOM_F64_CODES, HARAMICROSS_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLHOMINGPIGEONConfig::default(), CDLHOMINGPIGEONConfig::new(settings).unwrap(),
        HOMINGPIGEON_OPEN, HOMINGPIGEON_HIGH, HOMINGPIGEON_LOW, HOMINGPIGEON_CLOSE,
        HOMINGPIGEON_DEFAULT_LOOKBACK, HOMINGPIGEON_DEFAULT_F64_CODES, HOMINGPIGEON_DEFAULT_F32_CODES,
        HOMINGPIGEON_CUSTOM_LOOKBACK, HOMINGPIGEON_CUSTOM_F64_CODES, HOMINGPIGEON_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLKICKINGConfig::default(), CDLKICKINGConfig::new(settings).unwrap(),
        KICKING_OPEN, KICKING_HIGH, KICKING_LOW, KICKING_CLOSE,
        KICKING_DEFAULT_LOOKBACK, KICKING_DEFAULT_F64_CODES, KICKING_DEFAULT_F32_CODES,
        KICKING_CUSTOM_LOOKBACK, KICKING_CUSTOM_F64_CODES, KICKING_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLKICKINGBYLENGTHConfig::default(), CDLKICKINGBYLENGTHConfig::new(settings).unwrap(),
        KICKINGBYLENGTH_OPEN, KICKINGBYLENGTH_HIGH, KICKINGBYLENGTH_LOW, KICKINGBYLENGTH_CLOSE,
        KICKINGBYLENGTH_DEFAULT_LOOKBACK, KICKINGBYLENGTH_DEFAULT_F64_CODES, KICKINGBYLENGTH_DEFAULT_F32_CODES,
        KICKINGBYLENGTH_CUSTOM_LOOKBACK, KICKINGBYLENGTH_CUSTOM_F64_CODES, KICKINGBYLENGTH_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLMATCHINGLOWConfig::default(), CDLMATCHINGLOWConfig::new(settings).unwrap(),
        MATCHINGLOW_OPEN, MATCHINGLOW_HIGH, MATCHINGLOW_LOW, MATCHINGLOW_CLOSE,
        MATCHINGLOW_DEFAULT_LOOKBACK, MATCHINGLOW_DEFAULT_F64_CODES, MATCHINGLOW_DEFAULT_F32_CODES,
        MATCHINGLOW_CUSTOM_LOOKBACK, MATCHINGLOW_CUSTOM_F64_CODES, MATCHINGLOW_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLHAMMERConfig::default(), CDLHAMMERConfig::new(settings).unwrap(),
        HAMMER_OPEN, HAMMER_HIGH, HAMMER_LOW, HAMMER_CLOSE,
        HAMMER_DEFAULT_LOOKBACK, HAMMER_DEFAULT_F64_CODES, HAMMER_DEFAULT_F32_CODES,
        HAMMER_CUSTOM_LOOKBACK, HAMMER_CUSTOM_F64_CODES, HAMMER_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLHANGINGMANConfig::default(), CDLHANGINGMANConfig::new(settings).unwrap(),
        HANGINGMAN_OPEN, HANGINGMAN_HIGH, HANGINGMAN_LOW, HANGINGMAN_CLOSE,
        HANGINGMAN_DEFAULT_LOOKBACK, HANGINGMAN_DEFAULT_F64_CODES, HANGINGMAN_DEFAULT_F32_CODES,
        HANGINGMAN_CUSTOM_LOOKBACK, HANGINGMAN_CUSTOM_F64_CODES, HANGINGMAN_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLINNECKConfig::default(), CDLINNECKConfig::new(settings).unwrap(),
        INNECK_OPEN, INNECK_HIGH, INNECK_LOW, INNECK_CLOSE,
        INNECK_DEFAULT_LOOKBACK, INNECK_DEFAULT_F64_CODES, INNECK_DEFAULT_F32_CODES,
        INNECK_CUSTOM_LOOKBACK, INNECK_CUSTOM_F64_CODES, INNECK_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLINVERTEDHAMMERConfig::default(), CDLINVERTEDHAMMERConfig::new(settings).unwrap(),
        INVERTEDHAMMER_OPEN, INVERTEDHAMMER_HIGH, INVERTEDHAMMER_LOW, INVERTEDHAMMER_CLOSE,
        INVERTEDHAMMER_DEFAULT_LOOKBACK, INVERTEDHAMMER_DEFAULT_F64_CODES,
        INVERTEDHAMMER_DEFAULT_F32_CODES, INVERTEDHAMMER_CUSTOM_LOOKBACK,
        INVERTEDHAMMER_CUSTOM_F64_CODES, INVERTEDHAMMER_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLONNECKConfig::default(), CDLONNECKConfig::new(settings).unwrap(),
        ONNECK_OPEN, ONNECK_HIGH, ONNECK_LOW, ONNECK_CLOSE,
        ONNECK_DEFAULT_LOOKBACK, ONNECK_DEFAULT_F64_CODES, ONNECK_DEFAULT_F32_CODES,
        ONNECK_CUSTOM_LOOKBACK, ONNECK_CUSTOM_F64_CODES, ONNECK_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLPIERCINGConfig::default(), CDLPIERCINGConfig::new(settings).unwrap(),
        PIERCING_OPEN, PIERCING_HIGH, PIERCING_LOW, PIERCING_CLOSE,
        PIERCING_DEFAULT_LOOKBACK, PIERCING_DEFAULT_F64_CODES, PIERCING_DEFAULT_F32_CODES,
        PIERCING_CUSTOM_LOOKBACK, PIERCING_CUSTOM_F64_CODES, PIERCING_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLSEPARATINGLINESConfig::default(), CDLSEPARATINGLINESConfig::new(settings).unwrap(),
        SEPARATINGLINES_OPEN, SEPARATINGLINES_HIGH, SEPARATINGLINES_LOW, SEPARATINGLINES_CLOSE,
        SEPARATINGLINES_DEFAULT_LOOKBACK, SEPARATINGLINES_DEFAULT_F64_CODES,
        SEPARATINGLINES_DEFAULT_F32_CODES, SEPARATINGLINES_CUSTOM_LOOKBACK,
        SEPARATINGLINES_CUSTOM_F64_CODES, SEPARATINGLINES_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLSHOOTINGSTARConfig::default(), CDLSHOOTINGSTARConfig::new(settings).unwrap(),
        SHOOTINGSTAR_OPEN, SHOOTINGSTAR_HIGH, SHOOTINGSTAR_LOW, SHOOTINGSTAR_CLOSE,
        SHOOTINGSTAR_DEFAULT_LOOKBACK, SHOOTINGSTAR_DEFAULT_F64_CODES,
        SHOOTINGSTAR_DEFAULT_F32_CODES, SHOOTINGSTAR_CUSTOM_LOOKBACK,
        SHOOTINGSTAR_CUSTOM_F64_CODES, SHOOTINGSTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLTHRUSTINGConfig::default(), CDLTHRUSTINGConfig::new(settings).unwrap(),
        THRUSTING_OPEN, THRUSTING_HIGH, THRUSTING_LOW, THRUSTING_CLOSE,
        THRUSTING_DEFAULT_LOOKBACK, THRUSTING_DEFAULT_F64_CODES, THRUSTING_DEFAULT_F32_CODES,
        THRUSTING_CUSTOM_LOOKBACK, THRUSTING_CUSTOM_F64_CODES, THRUSTING_CUSTOM_F32_CODES
    );
}

#[test]
fn pinned_three_candle_oracles_qualify_every_definition_through_the_public_seam() {
    let settings = custom_two_candle_settings();
    let custom_penetration = Penetration::new(0.6 as Float).unwrap();
    qualify_two_candle!(
        CDL3INSIDEConfig::default(), CDL3INSIDEConfig::new(settings).unwrap(),
        THREE_INSIDE_OPEN, THREE_INSIDE_HIGH, THREE_INSIDE_LOW, THREE_INSIDE_CLOSE,
        THREE_INSIDE_DEFAULT_LOOKBACK, THREE_INSIDE_DEFAULT_F64_CODES,
        THREE_INSIDE_DEFAULT_F32_CODES, THREE_INSIDE_CUSTOM_LOOKBACK,
        THREE_INSIDE_CUSTOM_F64_CODES, THREE_INSIDE_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDL3OUTSIDEConfig::default(), CDL3OUTSIDEConfig::new(settings).unwrap(),
        THREE_OUTSIDE_OPEN, THREE_OUTSIDE_HIGH, THREE_OUTSIDE_LOW, THREE_OUTSIDE_CLOSE,
        THREE_OUTSIDE_DEFAULT_LOOKBACK, THREE_OUTSIDE_DEFAULT_F64_CODES,
        THREE_OUTSIDE_DEFAULT_F32_CODES, THREE_OUTSIDE_CUSTOM_LOOKBACK,
        THREE_OUTSIDE_CUSTOM_F64_CODES, THREE_OUTSIDE_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLABANDONEDBABYConfig::default(),
        CDLABANDONEDBABYConfig::new(settings, custom_penetration).unwrap(),
        ABANDONEDBABY_OPEN, ABANDONEDBABY_HIGH, ABANDONEDBABY_LOW, ABANDONEDBABY_CLOSE,
        ABANDONEDBABY_DEFAULT_LOOKBACK, ABANDONEDBABY_DEFAULT_F64_CODES,
        ABANDONEDBABY_DEFAULT_F32_CODES, ABANDONEDBABY_CUSTOM_LOOKBACK,
        ABANDONEDBABY_CUSTOM_F64_CODES, ABANDONEDBABY_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLEVENINGDOJISTARConfig::default(),
        CDLEVENINGDOJISTARConfig::new(settings, custom_penetration).unwrap(),
        EVENINGDOJISTAR_OPEN, EVENINGDOJISTAR_HIGH, EVENINGDOJISTAR_LOW, EVENINGDOJISTAR_CLOSE,
        EVENINGDOJISTAR_DEFAULT_LOOKBACK, EVENINGDOJISTAR_DEFAULT_F64_CODES,
        EVENINGDOJISTAR_DEFAULT_F32_CODES, EVENINGDOJISTAR_CUSTOM_LOOKBACK,
        EVENINGDOJISTAR_CUSTOM_F64_CODES, EVENINGDOJISTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLEVENINGSTARConfig::default(),
        CDLEVENINGSTARConfig::new(settings, custom_penetration).unwrap(),
        EVENINGSTAR_OPEN, EVENINGSTAR_HIGH, EVENINGSTAR_LOW, EVENINGSTAR_CLOSE,
        EVENINGSTAR_DEFAULT_LOOKBACK, EVENINGSTAR_DEFAULT_F64_CODES,
        EVENINGSTAR_DEFAULT_F32_CODES, EVENINGSTAR_CUSTOM_LOOKBACK,
        EVENINGSTAR_CUSTOM_F64_CODES, EVENINGSTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLMORNINGDOJISTARConfig::default(),
        CDLMORNINGDOJISTARConfig::new(settings, custom_penetration).unwrap(),
        MORNINGDOJISTAR_OPEN, MORNINGDOJISTAR_HIGH, MORNINGDOJISTAR_LOW, MORNINGDOJISTAR_CLOSE,
        MORNINGDOJISTAR_DEFAULT_LOOKBACK, MORNINGDOJISTAR_DEFAULT_F64_CODES,
        MORNINGDOJISTAR_DEFAULT_F32_CODES, MORNINGDOJISTAR_CUSTOM_LOOKBACK,
        MORNINGDOJISTAR_CUSTOM_F64_CODES, MORNINGDOJISTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLMORNINGSTARConfig::default(),
        CDLMORNINGSTARConfig::new(settings, custom_penetration).unwrap(),
        MORNINGSTAR_OPEN, MORNINGSTAR_HIGH, MORNINGSTAR_LOW, MORNINGSTAR_CLOSE,
        MORNINGSTAR_DEFAULT_LOOKBACK, MORNINGSTAR_DEFAULT_F64_CODES,
        MORNINGSTAR_DEFAULT_F32_CODES, MORNINGSTAR_CUSTOM_LOOKBACK,
        MORNINGSTAR_CUSTOM_F64_CODES, MORNINGSTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLUNIQUE3RIVERConfig::default(), CDLUNIQUE3RIVERConfig::new(settings).unwrap(),
        UNIQUE3RIVER_OPEN, UNIQUE3RIVER_HIGH, UNIQUE3RIVER_LOW, UNIQUE3RIVER_CLOSE,
        UNIQUE3RIVER_DEFAULT_LOOKBACK, UNIQUE3RIVER_DEFAULT_F64_CODES,
        UNIQUE3RIVER_DEFAULT_F32_CODES, UNIQUE3RIVER_CUSTOM_LOOKBACK,
        UNIQUE3RIVER_CUSTOM_F64_CODES, UNIQUE3RIVER_CUSTOM_F32_CODES
    );
}

#[test]
fn pinned_gap_continuation_oracles_qualify_every_definition_through_the_public_seam() {
    let settings = custom_two_candle_settings();
    qualify_two_candle!(
        CDL2CROWSConfig::default(), CDL2CROWSConfig::new(settings).unwrap(),
        TWO_CROWS_OPEN, TWO_CROWS_HIGH, TWO_CROWS_LOW, TWO_CROWS_CLOSE,
        TWO_CROWS_DEFAULT_LOOKBACK, TWO_CROWS_DEFAULT_F64_CODES, TWO_CROWS_DEFAULT_F32_CODES,
        TWO_CROWS_CUSTOM_LOOKBACK, TWO_CROWS_CUSTOM_F64_CODES, TWO_CROWS_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDL3LINESTRIKEConfig::default(), CDL3LINESTRIKEConfig::new(settings).unwrap(),
        THREE_LINE_STRIKE_OPEN, THREE_LINE_STRIKE_HIGH, THREE_LINE_STRIKE_LOW,
        THREE_LINE_STRIKE_CLOSE, THREE_LINE_STRIKE_DEFAULT_LOOKBACK,
        THREE_LINE_STRIKE_DEFAULT_F64_CODES, THREE_LINE_STRIKE_DEFAULT_F32_CODES,
        THREE_LINE_STRIKE_CUSTOM_LOOKBACK, THREE_LINE_STRIKE_CUSTOM_F64_CODES,
        THREE_LINE_STRIKE_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLGAPSIDESIDEWHITEConfig::default(),
        CDLGAPSIDESIDEWHITEConfig::new(settings).unwrap(),
        GAP_SIDE_SIDE_WHITE_OPEN, GAP_SIDE_SIDE_WHITE_HIGH, GAP_SIDE_SIDE_WHITE_LOW,
        GAP_SIDE_SIDE_WHITE_CLOSE, GAP_SIDE_SIDE_WHITE_DEFAULT_LOOKBACK,
        GAP_SIDE_SIDE_WHITE_DEFAULT_F64_CODES, GAP_SIDE_SIDE_WHITE_DEFAULT_F32_CODES,
        GAP_SIDE_SIDE_WHITE_CUSTOM_LOOKBACK, GAP_SIDE_SIDE_WHITE_CUSTOM_F64_CODES,
        GAP_SIDE_SIDE_WHITE_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLSTICKSANDWICHConfig::default(), CDLSTICKSANDWICHConfig::new(settings).unwrap(),
        STICK_SANDWICH_OPEN, STICK_SANDWICH_HIGH, STICK_SANDWICH_LOW, STICK_SANDWICH_CLOSE,
        STICK_SANDWICH_DEFAULT_LOOKBACK, STICK_SANDWICH_DEFAULT_F64_CODES,
        STICK_SANDWICH_DEFAULT_F32_CODES, STICK_SANDWICH_CUSTOM_LOOKBACK,
        STICK_SANDWICH_CUSTOM_F64_CODES, STICK_SANDWICH_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLTASUKIGAPConfig::default(), CDLTASUKIGAPConfig::new(settings).unwrap(),
        TASUKI_GAP_OPEN, TASUKI_GAP_HIGH, TASUKI_GAP_LOW, TASUKI_GAP_CLOSE,
        TASUKI_GAP_DEFAULT_LOOKBACK, TASUKI_GAP_DEFAULT_F64_CODES,
        TASUKI_GAP_DEFAULT_F32_CODES, TASUKI_GAP_CUSTOM_LOOKBACK,
        TASUKI_GAP_CUSTOM_F64_CODES, TASUKI_GAP_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLTRISTARConfig::default(), CDLTRISTARConfig::new(settings).unwrap(),
        TRISTAR_OPEN, TRISTAR_HIGH, TRISTAR_LOW, TRISTAR_CLOSE,
        TRISTAR_DEFAULT_LOOKBACK, TRISTAR_DEFAULT_F64_CODES, TRISTAR_DEFAULT_F32_CODES,
        TRISTAR_CUSTOM_LOOKBACK, TRISTAR_CUSTOM_F64_CODES, TRISTAR_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLUPSIDEGAP2CROWSConfig::default(), CDLUPSIDEGAP2CROWSConfig::new(settings).unwrap(),
        UPSIDE_GAP_TWO_CROWS_OPEN, UPSIDE_GAP_TWO_CROWS_HIGH, UPSIDE_GAP_TWO_CROWS_LOW,
        UPSIDE_GAP_TWO_CROWS_CLOSE, UPSIDE_GAP_TWO_CROWS_DEFAULT_LOOKBACK,
        UPSIDE_GAP_TWO_CROWS_DEFAULT_F64_CODES, UPSIDE_GAP_TWO_CROWS_DEFAULT_F32_CODES,
        UPSIDE_GAP_TWO_CROWS_CUSTOM_LOOKBACK, UPSIDE_GAP_TWO_CROWS_CUSTOM_F64_CODES,
        UPSIDE_GAP_TWO_CROWS_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLXSIDEGAP3METHODSConfig::default(), CDLXSIDEGAP3METHODSConfig::new(settings).unwrap(),
        X_SIDE_GAP_THREE_METHODS_OPEN, X_SIDE_GAP_THREE_METHODS_HIGH,
        X_SIDE_GAP_THREE_METHODS_LOW, X_SIDE_GAP_THREE_METHODS_CLOSE,
        X_SIDE_GAP_THREE_METHODS_DEFAULT_LOOKBACK, X_SIDE_GAP_THREE_METHODS_DEFAULT_F64_CODES,
        X_SIDE_GAP_THREE_METHODS_DEFAULT_F32_CODES, X_SIDE_GAP_THREE_METHODS_CUSTOM_LOOKBACK,
        X_SIDE_GAP_THREE_METHODS_CUSTOM_F64_CODES, X_SIDE_GAP_THREE_METHODS_CUSTOM_F32_CODES
    );
}

#[test]
fn pinned_crow_soldier_oracles_qualify_every_definition_through_the_public_seam() {
    let settings = custom_two_candle_settings();
    qualify_two_candle!(CDL3BLACKCROWSConfig::default(), CDL3BLACKCROWSConfig::new(settings).unwrap(),
        THREE_BLACK_CROWS_OPEN, THREE_BLACK_CROWS_HIGH, THREE_BLACK_CROWS_LOW, THREE_BLACK_CROWS_CLOSE,
        THREE_BLACK_CROWS_DEFAULT_LOOKBACK, THREE_BLACK_CROWS_DEFAULT_F64_CODES, THREE_BLACK_CROWS_DEFAULT_F32_CODES,
        THREE_BLACK_CROWS_CUSTOM_LOOKBACK, THREE_BLACK_CROWS_CUSTOM_F64_CODES, THREE_BLACK_CROWS_CUSTOM_F32_CODES);
    qualify_two_candle!(CDL3STARSINSOUTHConfig::default(), CDL3STARSINSOUTHConfig::new(settings).unwrap(),
        THREE_STARS_IN_SOUTH_OPEN, THREE_STARS_IN_SOUTH_HIGH, THREE_STARS_IN_SOUTH_LOW, THREE_STARS_IN_SOUTH_CLOSE,
        THREE_STARS_IN_SOUTH_DEFAULT_LOOKBACK, THREE_STARS_IN_SOUTH_DEFAULT_F64_CODES, THREE_STARS_IN_SOUTH_DEFAULT_F32_CODES,
        THREE_STARS_IN_SOUTH_CUSTOM_LOOKBACK, THREE_STARS_IN_SOUTH_CUSTOM_F64_CODES, THREE_STARS_IN_SOUTH_CUSTOM_F32_CODES);
    qualify_two_candle!(CDL3WHITESOLDIERSConfig::default(), CDL3WHITESOLDIERSConfig::new(settings).unwrap(),
        THREE_WHITE_SOLDIERS_OPEN, THREE_WHITE_SOLDIERS_HIGH, THREE_WHITE_SOLDIERS_LOW, THREE_WHITE_SOLDIERS_CLOSE,
        THREE_WHITE_SOLDIERS_DEFAULT_LOOKBACK, THREE_WHITE_SOLDIERS_DEFAULT_F64_CODES, THREE_WHITE_SOLDIERS_DEFAULT_F32_CODES,
        THREE_WHITE_SOLDIERS_CUSTOM_LOOKBACK, THREE_WHITE_SOLDIERS_CUSTOM_F64_CODES, THREE_WHITE_SOLDIERS_CUSTOM_F32_CODES);
    qualify_two_candle!(CDLADVANCEBLOCKConfig::default(), CDLADVANCEBLOCKConfig::new(settings).unwrap(),
        ADVANCE_BLOCK_OPEN, ADVANCE_BLOCK_HIGH, ADVANCE_BLOCK_LOW, ADVANCE_BLOCK_CLOSE,
        ADVANCE_BLOCK_DEFAULT_LOOKBACK, ADVANCE_BLOCK_DEFAULT_F64_CODES, ADVANCE_BLOCK_DEFAULT_F32_CODES,
        ADVANCE_BLOCK_CUSTOM_LOOKBACK, ADVANCE_BLOCK_CUSTOM_F64_CODES, ADVANCE_BLOCK_CUSTOM_F32_CODES);
    qualify_two_candle!(CDLCONCEALBABYSWALLConfig::default(), CDLCONCEALBABYSWALLConfig::new(settings).unwrap(),
        CONCEAL_BABY_SWALLOW_OPEN, CONCEAL_BABY_SWALLOW_HIGH, CONCEAL_BABY_SWALLOW_LOW, CONCEAL_BABY_SWALLOW_CLOSE,
        CONCEAL_BABY_SWALLOW_DEFAULT_LOOKBACK, CONCEAL_BABY_SWALLOW_DEFAULT_F64_CODES, CONCEAL_BABY_SWALLOW_DEFAULT_F32_CODES,
        CONCEAL_BABY_SWALLOW_CUSTOM_LOOKBACK, CONCEAL_BABY_SWALLOW_CUSTOM_F64_CODES, CONCEAL_BABY_SWALLOW_CUSTOM_F32_CODES);
    qualify_two_candle!(CDLIDENTICAL3CROWSConfig::default(), CDLIDENTICAL3CROWSConfig::new(settings).unwrap(),
        IDENTICAL_THREE_CROWS_OPEN, IDENTICAL_THREE_CROWS_HIGH, IDENTICAL_THREE_CROWS_LOW, IDENTICAL_THREE_CROWS_CLOSE,
        IDENTICAL_THREE_CROWS_DEFAULT_LOOKBACK, IDENTICAL_THREE_CROWS_DEFAULT_F64_CODES, IDENTICAL_THREE_CROWS_DEFAULT_F32_CODES,
        IDENTICAL_THREE_CROWS_CUSTOM_LOOKBACK, IDENTICAL_THREE_CROWS_CUSTOM_F64_CODES, IDENTICAL_THREE_CROWS_CUSTOM_F32_CODES);
    qualify_two_candle!(CDLSTALLEDPATTERNConfig::default(), CDLSTALLEDPATTERNConfig::new(settings).unwrap(),
        STALLED_PATTERN_OPEN, STALLED_PATTERN_HIGH, STALLED_PATTERN_LOW, STALLED_PATTERN_CLOSE,
        STALLED_PATTERN_DEFAULT_LOOKBACK, STALLED_PATTERN_DEFAULT_F64_CODES, STALLED_PATTERN_DEFAULT_F32_CODES,
        STALLED_PATTERN_CUSTOM_LOOKBACK, STALLED_PATTERN_CUSTOM_F64_CODES, STALLED_PATTERN_CUSTOM_F32_CODES);
}

#[test]
fn crow_soldier_single_boundary_near_misses_lock_literal_comparisons() {
    fn check<C>(config: C, open: &[f64], high: &[f64], low: &[f64], close: &[f64], mutate: impl Fn(&mut Series))
    where
        C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
        for<'a> C: IndicatorConfig<Input<'a> = CandleInput<'a>>,
    {
        let canonical = Series::from_fixture(open, high, low, close);
        assert_ne!(config.compute(canonical.input()).unwrap().values().last().unwrap().to_talib_code(), 0);
        let mut boundary = Series::from_fixture(open, high, low, close);
        mutate(&mut boundary);
        assert_eq!(config.compute(boundary.input()).unwrap().values().last().unwrap().to_talib_code(), 0);
    }
    check(CDL3BLACKCROWSConfig::default(), reference::THREE_BLACK_CROWS_OPEN, reference::THREE_BLACK_CROWS_HIGH, reference::THREE_BLACK_CROWS_LOW, reference::THREE_BLACK_CROWS_CLOSE,
        |s| s.high[10] = s.close[11]);
    check(CDL3STARSINSOUTHConfig::default(), reference::THREE_STARS_IN_SOUTH_OPEN, reference::THREE_STARS_IN_SOUTH_HIGH, reference::THREE_STARS_IN_SOUTH_LOW, reference::THREE_STARS_IN_SOUTH_CLOSE,
        |s| s.high[12] = s.high[11]);
    check(CDL3WHITESOLDIERSConfig::default(), reference::THREE_WHITE_SOLDIERS_OPEN, reference::THREE_WHITE_SOLDIERS_HIGH, reference::THREE_WHITE_SOLDIERS_LOW, reference::THREE_WHITE_SOLDIERS_CLOSE,
        |s| s.close[12] = s.close[11]);
    check(CDLADVANCEBLOCKConfig::default(), reference::ADVANCE_BLOCK_OPEN, reference::ADVANCE_BLOCK_HIGH, reference::ADVANCE_BLOCK_LOW, reference::ADVANCE_BLOCK_CLOSE,
        |s| s.high[10] = s.close[10] + 2.0 as Float);
    check(CDLCONCEALBABYSWALLConfig::default(), reference::CONCEAL_BABY_SWALLOW_OPEN, reference::CONCEAL_BABY_SWALLOW_HIGH, reference::CONCEAL_BABY_SWALLOW_LOW, reference::CONCEAL_BABY_SWALLOW_CLOSE,
        |s| s.high[13] = s.high[12]);
    check(CDLIDENTICAL3CROWSConfig::default(), reference::IDENTICAL_THREE_CROWS_OPEN, reference::IDENTICAL_THREE_CROWS_HIGH, reference::IDENTICAL_THREE_CROWS_LOW, reference::IDENTICAL_THREE_CROWS_CLOSE,
        |s| s.open[12] = s.close[11] + 1.0 as Float);
    check(CDLSTALLEDPATTERNConfig::default(), reference::STALLED_PATTERN_OPEN, reference::STALLED_PATTERN_HIGH, reference::STALLED_PATTERN_LOW, reference::STALLED_PATTERN_CLOSE,
        |s| s.close[12] = s.close[11]);
}

#[test]
fn crow_soldier_evidence_rows_cover_pinned_offsets_settings_and_boundaries() {
    const ROWS: [(&str, &str, &str, &str, usize); 7] = [
        ("CDL3BLACKCROWS", "cdl3blackcrows/cdl3blackcrows.c", "four-Candle do-loop", "prior white at i-3; ShadowVeryShort at i-2..i; strict opens/closes", 13),
        ("CDL3STARSINSOUTH", "cdl3starsinsouth/cdl3starsinsouth.c", "three-Candle do-loop", "BodyLong, BodyShort, ShadowLong, ShadowVeryShort; strict engulfment", 12),
        ("CDL3WHITESOLDIERS", "cdl3whitesoldiers/cdl3whitesoldiers.c", "three-Candle do-loop", "Near/Far at literal prior offsets; BodyShort and ShadowVeryShort", 12),
        ("CDLADVANCEBLOCK", "cdladvanceblock/cdladvanceblock.c", "three-Candle do-loop", "Near/Far weakening disjunction; BodyLong, ShadowShort, ShadowLong", 12),
        ("CDLCONCEALBABYSWALL", "cdlconcealbabyswall/cdlconcealbabyswall.c", "four-Candle do-loop", "ShadowVeryShort at i-3..i-1; strict range engulfment", 13),
        ("CDLIDENTICAL3CROWS", "cdlidentical3crows/cdlidentical3crows.c", "three-Candle do-loop", "Equal inclusive opening bands; strict ShadowVeryShort and declines", 12),
        ("CDLSTALLEDPATTERN", "cdlstalledpattern/cdlstalledpattern.c", "three-Candle do-loop", "BodyLong/BodyShort, Near shoulder, ShadowVeryShort", 12),
    ];
    assert_eq!(ROWS.len(), 7);
    assert!(ROWS.iter().all(|(name, source, loop_shape, settings, lookback)|
        name.starts_with("CDL") && source.ends_with(".c") && loop_shape.contains("Candle") && !settings.is_empty() && *lookback >= 12));
}

#[test]
fn star_definitions_own_pinned_penetration_defaults_and_accept_values_above_one() {
    let settings = CandleSettings::default();
    let above_one = Penetration::new(4.0 as Float).unwrap();
    assert_eq!(CDLABANDONEDBABYConfig::default().penetration().value(), 0.3 as Float);
    assert_eq!(CDLEVENINGDOJISTARConfig::default().penetration().value(), 0.3 as Float);
    assert_eq!(CDLEVENINGSTARConfig::default().penetration().value(), 0.3 as Float);
    assert_eq!(CDLMORNINGDOJISTARConfig::default().penetration().value(), 0.3 as Float);
    assert_eq!(CDLMORNINGSTARConfig::default().penetration().value(), 0.3 as Float);
    assert_eq!(
        CDLABANDONEDBABYConfig::new(settings, above_one).unwrap().penetration(),
        above_one
    );
    assert_eq!(
        CDLEVENINGDOJISTARConfig::new(settings, above_one).unwrap().penetration(),
        above_one
    );
    assert_eq!(
        CDLEVENINGSTARConfig::new(settings, above_one).unwrap().penetration(),
        above_one
    );
    assert_eq!(
        CDLMORNINGDOJISTARConfig::new(settings, above_one).unwrap().penetration(),
        above_one
    );
    assert_eq!(
        CDLMORNINGSTARConfig::new(settings, above_one).unwrap().penetration(),
        above_one
    );
}

fn boundary_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyDoji,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Equal,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.0 as Float).unwrap(),
        )
}

fn boundary_code<C>(config: C, candles: &[Candle]) -> i32
where
    C: 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C: IndicatorConfig<Input<'a> = CandleInput<'a>>,
{
    let series = Series {
        open: candles.iter().map(|candle| candle.open).collect(),
        high: candles.iter().map(|candle| candle.high).collect(),
        low: candles.iter().map(|candle| candle.low).collect(),
        close: candles.iter().map(|candle| candle.close).collect(),
    };
    config.compute(series.input()).unwrap().values().last().unwrap().to_talib_code()
}

const fn candle(open: Float, high: Float, low: Float, close: Float) -> Candle {
    Candle { open, high, low, close }
}

#[test]
fn independently_reasoned_two_candle_boundaries_lock_exact_pinned_predicates() {
    let settings = boundary_settings();
    let white = candle(10.0, 20.0, 10.0, 20.0);
    let black = candle(20.0, 20.0, 10.0, 10.0);

    let harami = CDLHARAMIConfig::new(settings).unwrap();
    assert_eq!(boundary_code(harami, &[white, candle(16.0, 16.0, 14.0, 14.0)]), -100);
    assert_eq!(boundary_code(harami, &[white, candle(20.0, 20.0, 18.0, 18.0)]), -80);
    assert_eq!(boundary_code(harami, &[black, candle(14.0, 16.0, 14.0, 16.0)]), 100);
    assert_eq!(boundary_code(harami, &[black, candle(10.0, 12.0, 10.0, 12.0)]), 80);

    let cross = CDLHARAMICROSSConfig::new(settings).unwrap();
    assert_eq!(boundary_code(cross, &[white, candle(15.0, 16.0, 14.0, 15.0)]), -100);
    assert_eq!(boundary_code(cross, &[white, candle(20.0, 20.0, 20.0, 20.0)]), -80);
    assert_eq!(boundary_code(cross, &[black, candle(15.0, 16.0, 14.0, 15.0)]), 100);
    assert_eq!(boundary_code(cross, &[black, candle(10.0, 10.0, 10.0, 10.0)]), 80);

    let counter = CDLCOUNTERATTACKConfig::new(settings).unwrap();
    assert_eq!(boundary_code(counter, &[black, candle(0.0, 10.0, 0.0, 10.0)]), 100);
    assert_eq!(boundary_code(counter, &[black, candle(0.0, 11.0, 0.0, 11.0)]), 0);

    let matching = CDLMATCHINGLOWConfig::new(settings).unwrap();
    assert_eq!(boundary_code(matching, &[black, candle(18.0, 18.0, 10.0, 10.0)]), 100);
    assert_eq!(boundary_code(matching, &[black, candle(18.0, 18.0, 9.0, 9.0)]), 0);

    let doji_star = CDLDOJISTARConfig::new(settings).unwrap();
    assert_eq!(boundary_code(doji_star, &[white, candle(21.0, 21.0, 21.0, 21.0)]), -100);
    assert_eq!(boundary_code(doji_star, &[white, candle(20.0, 20.0, 20.0, 20.0)]), 0);

    let dark = CDLDARKCLOUDCOVERConfig::new(settings, Penetration::new(0.5 as Float).unwrap()).unwrap();
    assert_eq!(boundary_code(dark, &[white, candle(22.0, 22.0, 14.0, 14.0)]), -100);
    assert_eq!(boundary_code(dark, &[white, candle(22.0, 22.0, 15.0, 15.0)]), 0);
    let above_one = CDLDARKCLOUDCOVERConfig::new(settings, Penetration::new(4.0 as Float).unwrap()).unwrap();
    assert_eq!(above_one.penetration().value(), 4.0 as Float);

    let homing = CDLHOMINGPIGEONConfig::new(settings).unwrap();
    assert_eq!(boundary_code(homing, &[black, candle(17.0, 17.0, 16.0, 16.0)]), 100);
    assert_eq!(boundary_code(homing, &[black, candle(20.0, 20.0, 16.0, 16.0)]), 0);

    let kicking_pair = [black, candle(22.0, 32.0, 22.0, 32.0)];
    assert_eq!(boundary_code(CDLKICKINGConfig::new(settings).unwrap(), &kicking_pair), 100);
    assert_eq!(boundary_code(CDLKICKINGBYLENGTHConfig::new(settings).unwrap(), &kicking_pair), -100);
    let touching = [black, candle(20.0, 30.0, 20.0, 30.0)];
    assert_eq!(boundary_code(CDLKICKINGConfig::new(settings).unwrap(), &touching), 0);
}

fn position_shadow_boundary_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 2.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowLong,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Near,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Equal,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.1 as Float).unwrap(),
        )
}

#[test]
fn position_shadow_boundaries_lock_strict_and_inclusive_pinned_predicates() {
    let settings = position_shadow_boundary_settings();
    let positioned_previous = candle(20.0, 30.0, 10.0, 10.0);

    let hammer = CDLHAMMERConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(hammer, &[positioned_previous, candle(20.0, 23.0, 15.0, 22.0)]),
        100
    );
    assert_eq!(
        boundary_code(hammer, &[positioned_previous, candle(21.0, 24.0, 16.0, 23.0)]),
        0
    );

    let hanging_man = CDLHANGINGMANConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(hanging_man, &[positioned_previous, candle(20.0, 23.0, 15.0, 22.0)]),
        -100
    );
    assert_eq!(
        boundary_code(hanging_man, &[positioned_previous, candle(19.0, 22.0, 14.0, 21.0)]),
        0
    );

    let inverted_previous = candle(20.0, 30.0, 10.0, 30.0);
    let inverted = CDLINVERTEDHAMMERConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(inverted, &[inverted_previous, candle(10.0, 16.0, 9.0, 12.0)]),
        100
    );
    assert_eq!(
        boundary_code(inverted, &[inverted_previous, candle(10.0, 14.0, 9.0, 12.0)]),
        0
    );

    let shooting_previous = candle(20.0, 20.0, 10.0, 10.0);
    let shooting = CDLSHOOTINGSTARConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(shooting, &[shooting_previous, candle(30.0, 36.0, 29.0, 32.0)]),
        -100
    );
    assert_eq!(
        boundary_code(shooting, &[shooting_previous, candle(30.0, 34.0, 29.0, 32.0)]),
        0
    );

    let black = candle(30.0, 30.0, 10.0, 10.0);
    let in_neck = CDLINNECKConfig::new(settings).unwrap();
    assert_eq!(boundary_code(in_neck, &[black, candle(9.0, 12.0, 9.0, 12.0)]), -100);
    assert_eq!(boundary_code(in_neck, &[black, candle(9.0, 13.0, 9.0, 13.0)]), 0);

    let on_neck = CDLONNECKConfig::new(settings).unwrap();
    assert_eq!(boundary_code(on_neck, &[black, candle(9.0, 12.0, 9.0, 12.0)]), -100);
    assert_eq!(boundary_code(on_neck, &[black, candle(9.0, 13.0, 9.0, 13.0)]), 0);

    let piercing = CDLPIERCINGConfig::new(settings).unwrap();
    assert_eq!(boundary_code(piercing, &[black, candle(9.0, 21.0, 9.0, 21.0)]), 100);
    assert_eq!(boundary_code(piercing, &[black, candle(9.0, 20.0, 9.0, 20.0)]), 0);

    let thrusting = CDLTHRUSTINGConfig::new(settings).unwrap();
    assert_eq!(boundary_code(thrusting, &[black, candle(9.0, 20.0, 9.0, 20.0)]), -100);
    assert_eq!(boundary_code(thrusting, &[black, candle(9.0, 12.0, 9.0, 12.0)]), 0);

    let separating = CDLSEPARATINGLINESConfig::new(settings).unwrap();
    let separating_previous = candle(20.0, 30.0, 10.0, 10.0);
    assert_eq!(
        boundary_code(separating, &[separating_previous, candle(22.0, 32.0, 21.0, 32.0)]),
        100
    );
    assert_eq!(
        boundary_code(separating, &[separating_previous, candle(23.0, 33.0, 22.0, 33.0)]),
        0
    );
    assert_eq!(
        boundary_code(separating, &[separating_previous, candle(20.0, 30.0, 10.0, 30.0)]),
        0
    );
    assert_eq!(
        boundary_code(separating, &[separating_previous, candle(20.0, 30.0, 11.0, 30.0)]),
        100
    );
    assert_eq!(
        boundary_code(
            separating,
            &[candle(20.0, 30.0, 10.0, 30.0), candle(20.0, 21.0, 10.0, 10.0)]
        ),
        -100
    );
}

#[test]
fn two_candle_evidence_rows_cover_pinned_sources_and_qualification_scenarios() {
    const ROWS: [(&str, &str, &str); 18] = [
        ("CDLCOUNTERATTACK", "cdlcounterattack/cdlcounterattack.c", "exact equal closes"),
        ("CDLDARKCLOUDCOVER", "cdldarkcloudcover/cdldarkcloudcover.c", "Penetration"),
        ("CDLDOJISTAR", "cdldojistar/cdldojistar.c", "strict real-body gap"),
        ("CDLHARAMI", "cdlharami/cdlharami.c", "Standard and Partial"),
        ("CDLHARAMICROSS", "cdlharamicross/cdlharamicross.c", "Standard and Partial"),
        ("CDLHOMINGPIGEON", "cdlhomingpigeon/cdlhomingpigeon.c", "strict containment"),
        ("CDLKICKING", "cdlkicking/cdlkicking.c", "strict Candle gap"),
        ("CDLKICKINGBYLENGTH", "cdlkickingbylength/cdlkickingbylength.c", "first-Candle tie"),
        ("CDLMATCHINGLOW", "cdlmatchinglow/cdlmatchinglow.c", "inclusive Equal"),
        ("CDLHAMMER", "cdlhammer/cdlhammer.c", "inclusive Near and strict shadows"),
        ("CDLHANGINGMAN", "cdlhangingman/cdlhangingman.c", "inclusive Near"),
        ("CDLINNECK", "cdlinneck/cdlinneck.c", "inclusive Equal upper bound"),
        ("CDLINVERTEDHAMMER", "cdlinvertedhammer/cdlinvertedhammer.c", "strict upper shadow"),
        ("CDLONNECK", "cdlonneck/cdlonneck.c", "inclusive Equal interval"),
        ("CDLPIERCING", "cdlpiercing/cdlpiercing.c", "fixed strict fifty percent"),
        ("CDLSEPARATINGLINES", "cdlseparatinglines/cdlseparatinglines.c", "inclusive Equal and strict leading shadow"),
        ("CDLSHOOTINGSTAR", "cdlshootingstar/cdlshootingstar.c", "strict upper shadow"),
        ("CDLTHRUSTING", "cdlthrusting/cdlthrusting.c", "strict Equal and inclusive midpoint"),
    ];
    assert_eq!(reference::TALIB_GIT_REVISION, "2247d599bddf37ed37e3a709371517e46efc66f6");
    assert!(ROWS.iter().all(|(_, source, scenario)| source.ends_with(".c") && !scenario.is_empty()));
}

fn three_candle_boundary_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 1, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 1, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyDoji,
            CandleSetting::new(CandleRangeKind::HighLow, 1, 0.1 as Float).unwrap(),
        )
}

#[test]
fn independently_reasoned_three_candle_boundaries_lock_exact_pinned_predicates() {
    let settings = three_candle_boundary_settings();
    let penetration = Penetration::new(0.3 as Float).unwrap();
    let seed = candle(10.0, 12.0, 9.0, 11.0);
    let black = candle(20.0, 21.0, 9.0, 10.0);
    let white = candle(10.0, 21.0, 9.0, 20.0);

    let inside = CDL3INSIDEConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(
            inside,
            &[seed, black, candle(17.0, 18.0, 15.0, 16.0), candle(15.0, 22.0, 14.0, 21.0)]
        ),
        100
    );
    assert_eq!(
        boundary_code(
            inside,
            &[seed, black, candle(11.0, 12.0, 9.5, 10.0), candle(15.0, 22.0, 14.0, 21.0)]
        ),
        0
    );

    let outside = CDL3OUTSIDEConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(
            outside,
            &[seed, black, candle(9.0, 22.0, 8.0, 21.0), candle(21.0, 23.0, 20.0, 22.0)]
        ),
        100
    );
    assert_eq!(
        boundary_code(
            outside,
            &[seed, black, candle(10.0, 22.0, 9.0, 21.0), candle(21.0, 23.0, 20.0, 22.0)]
        ),
        0
    );

    let abandoned = CDLABANDONEDBABYConfig::new(settings, penetration).unwrap();
    let abandoned_match = [
        seed,
        black,
        candle(7.0, 8.0, 6.0, 7.0),
        candle(9.0, 19.0, 8.5, 18.0),
    ];
    assert_eq!(boundary_code(abandoned, &abandoned_match), 100);
    assert_eq!(
        boundary_code(
            abandoned,
            &[
                seed,
                black,
                candle(7.0, 9.0, 6.0, 7.0),
                candle(10.0, 19.0, 9.5, 18.0),
            ]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            abandoned,
            &[
                seed,
                black,
                candle(7.0, 8.0, 6.0, 7.0),
                candle(9.0, 19.0, 8.0, 18.0),
            ]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            abandoned,
            &[
                seed,
                black,
                candle(7.0, 8.0, 6.0, 7.0),
                candle(9.0, 19.0, 8.5, 13.0),
            ]
        ),
        0
    );

    let evening_doji = CDLEVENINGDOJISTARConfig::new(settings, penetration).unwrap();
    assert_eq!(
        boundary_code(
            evening_doji,
            &[seed, white, candle(21.0, 22.0, 20.5, 21.0), candle(21.0, 21.0, 12.0, 13.0)]
        ),
        -100
    );
    assert_eq!(
        boundary_code(
            evening_doji,
            &[seed, white, candle(20.0, 21.0, 19.5, 20.0), candle(21.0, 21.0, 12.0, 13.0)]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            evening_doji,
            &[seed, white, candle(21.0, 22.0, 20.5, 21.0), candle(21.0, 21.0, 12.0, 17.0)]
        ),
        0
    );

    let evening = CDLEVENINGSTARConfig::new(settings, penetration).unwrap();
    assert_eq!(
        boundary_code(
            evening,
            &[seed, white, candle(21.0, 23.0, 20.5, 22.0), candle(21.0, 21.0, 12.0, 13.0)]
        ),
        -100
    );
    assert_eq!(
        boundary_code(
            evening,
            &[seed, white, candle(20.0, 22.0, 19.5, 21.0), candle(21.0, 21.0, 12.0, 13.0)]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            evening,
            &[seed, white, candle(21.0, 23.0, 20.5, 22.0), candle(21.0, 21.0, 12.0, 17.0)]
        ),
        0
    );

    let morning_doji = CDLMORNINGDOJISTARConfig::new(settings, penetration).unwrap();
    assert_eq!(
        boundary_code(
            morning_doji,
            &[seed, black, candle(8.0, 8.5, 7.0, 8.0), candle(9.0, 18.0, 8.5, 17.0)]
        ),
        100
    );
    assert_eq!(
        boundary_code(
            morning_doji,
            &[seed, black, candle(10.0, 10.0, 9.0, 10.0), candle(9.0, 18.0, 8.5, 17.0)]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            morning_doji,
            &[seed, black, candle(8.0, 8.5, 7.0, 8.0), candle(9.0, 14.0, 8.5, 13.0)]
        ),
        0
    );

    let morning = CDLMORNINGSTARConfig::new(settings, penetration).unwrap();
    assert_eq!(
        boundary_code(
            morning,
            &[seed, black, candle(8.0, 9.0, 6.0, 7.0), candle(9.0, 18.0, 8.5, 17.0)]
        ),
        100
    );
    assert_eq!(
        boundary_code(
            morning,
            &[seed, black, candle(10.0, 10.0, 8.0, 9.0), candle(9.0, 18.0, 8.5, 17.0)]
        ),
        0
    );
    assert_eq!(
        boundary_code(
            morning,
            &[seed, black, candle(8.0, 9.0, 6.0, 7.0), candle(9.0, 14.0, 8.5, 13.0)]
        ),
        0
    );

    let unique = CDLUNIQUE3RIVERConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(
            unique,
            &[seed, black, candle(18.0, 19.0, 8.0, 12.0), candle(9.0, 10.0, 8.5, 9.5)]
        ),
        100
    );
    assert_eq!(
        boundary_code(
            unique,
            &[seed, black, candle(18.0, 19.0, 8.0, 12.0), candle(8.0, 9.0, 7.5, 8.5)]
        ),
        0
    );
}

#[test]
fn three_candle_evidence_rows_cover_pinned_sources_and_qualification_scenarios() {
    const ROWS: [(&str, &str, &str); 8] = [
        ("CDL3INSIDE", "cdl3inside/cdl3inside.c", "strict containment and first-Candle sign"),
        ("CDL3OUTSIDE", "cdl3outside/cdl3outside.c", "strict engulfing and second-Candle sign"),
        ("CDLABANDONEDBABY", "cdlabandonedbaby/cdlabandonedbaby.c", "two strict Candle gaps and Penetration"),
        ("CDLEVENINGDOJISTAR", "cdleveningdojistar/cdleveningdojistar.c", "strict real-body gap and Penetration"),
        ("CDLEVENINGSTAR", "cdleveningstar/cdleveningstar.c", "BodyShort offsets and Penetration"),
        ("CDLMORNINGDOJISTAR", "cdlmorningdojistar/cdlmorningdojistar.c", "strict real-body gap and Penetration"),
        ("CDLMORNINGSTAR", "cdlmorningstar/cdlmorningstar.c", "BodyShort offsets and Penetration"),
        ("CDLUNIQUE3RIVER", "cdlunique3river/cdlunique3river.c", "strict third open and fixed bullish sign"),
    ];
    assert_eq!(reference::TALIB_GIT_REVISION, "2247d599bddf37ed37e3a709371517e46efc66f6");
    assert!(ROWS.iter().all(|(_, source, scenario)| source.ends_with(".c") && !scenario.is_empty()));
}

fn gap_boundary_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 1.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Near,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.2 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::Equal,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.05 as Float).unwrap(),
        )
}

#[test]
fn independently_reasoned_gap_continuation_boundaries_lock_pinned_predicates_and_signs() {
    let settings = gap_boundary_settings();

    let two_crows = CDL2CROWSConfig::new(settings).unwrap();
    assert_eq!(
        boundary_code(
            two_crows,
            &[
                candle(10.0, 21.0, 9.0, 20.0),
                candle(24.0, 25.0, 21.5, 22.0),
                candle(23.0, 24.0, 14.0, 15.0),
            ],
        ),
        -100
    );
    assert_eq!(
        boundary_code(
            two_crows,
            &[
                candle(10.0, 21.0, 9.0, 20.0),
                candle(24.0, 25.0, 20.0, 20.0),
                candle(23.0, 24.0, 14.0, 15.0),
            ],
        ),
        0
    );

    let line_strike = CDL3LINESTRIKEConfig::new(settings).unwrap();
    let bullish_strike = [
        candle(10.0, 12.5, 9.5, 12.0),
        candle(11.0, 14.5, 10.5, 14.0),
        candle(13.0, 16.5, 12.5, 16.0),
        candle(17.0, 18.0, 8.0, 9.0),
    ];
    let bearish_strike = [
        candle(20.0, 20.5, 17.5, 18.0),
        candle(19.0, 19.5, 15.5, 16.0),
        candle(17.0, 17.5, 13.5, 14.0),
        candle(13.0, 22.0, 12.0, 21.0),
    ];
    assert_eq!(boundary_code(line_strike, &bullish_strike), 100);
    assert_eq!(boundary_code(line_strike, &bearish_strike), -100);
    assert_eq!(
        boundary_code(
            line_strike,
            &[
                bullish_strike[0],
                bullish_strike[1],
                bullish_strike[2],
                candle(17.0, 18.0, 9.0, 10.0),
            ],
        ),
        0
    );

    let side_by_side = CDLGAPSIDESIDEWHITEConfig::new(settings).unwrap();
    let upside_side_by_side = [
        candle(10.0, 12.5, 9.5, 12.0),
        candle(15.0, 17.5, 14.5, 17.0),
        candle(15.0, 17.5, 14.5, 17.0),
    ];
    let downside_side_by_side = [
        candle(20.0, 20.5, 17.5, 18.0),
        candle(13.0, 15.5, 12.5, 15.0),
        candle(13.0, 15.5, 12.5, 15.0),
    ];
    assert_eq!(boundary_code(side_by_side, &upside_side_by_side), 100);
    assert_eq!(boundary_code(side_by_side, &downside_side_by_side), -100);
    assert_eq!(
        boundary_code(
            side_by_side,
            &[
                upside_side_by_side[0],
                candle(12.0, 14.5, 11.5, 14.0),
                candle(12.0, 14.5, 11.5, 14.0),
            ],
        ),
        0
    );
    assert_eq!(
        boundary_code(
            side_by_side,
            &[
                downside_side_by_side[0],
                candle(16.0, 18.5, 15.5, 18.0),
                candle(16.0, 18.5, 15.5, 18.0),
            ],
        ),
        0
    );

    let stick_sandwich = CDLSTICKSANDWICHConfig::new(settings).unwrap();
    let stick_match = [
        candle(20.0, 21.0, 9.0, 10.0),
        candle(12.0, 18.5, 11.0, 18.0),
        candle(20.0, 21.0, 9.0, 10.0),
    ];
    assert_eq!(boundary_code(stick_sandwich, &stick_match), 100);
    assert_eq!(
        boundary_code(
            stick_sandwich,
            &[
                stick_match[0],
                candle(12.0, 18.5, 10.0, 18.0),
                stick_match[2],
            ],
        ),
        0
    );

    let tasuki = CDLTASUKIGAPConfig::new(settings).unwrap();
    let upside_tasuki = [
        candle(10.0, 12.5, 9.5, 12.0),
        candle(15.0, 18.5, 14.5, 18.0),
        candle(17.0, 17.5, 13.5, 14.0),
    ];
    let downside_tasuki = [
        candle(20.0, 20.5, 17.5, 18.0),
        candle(15.0, 15.5, 11.5, 12.0),
        candle(13.0, 16.5, 12.5, 16.0),
    ];
    assert_eq!(boundary_code(tasuki, &upside_tasuki), 100);
    assert_eq!(boundary_code(tasuki, &downside_tasuki), -100);
    assert_eq!(
        boundary_code(
            tasuki,
            &[
                candle(10.0, 15.5, 9.5, 15.0),
                upside_tasuki[1],
                upside_tasuki[2],
            ],
        ),
        0
    );
    assert_eq!(
        boundary_code(
            tasuki,
            &[
                candle(20.0, 20.5, 14.5, 15.0),
                downside_tasuki[1],
                downside_tasuki[2],
            ],
        ),
        0
    );

    let upside_gap_two_crows = CDLUPSIDEGAP2CROWSConfig::new(settings).unwrap();
    let upside_gap_match = [
        candle(10.0, 21.0, 9.0, 20.0),
        candle(23.0, 23.5, 21.5, 22.0),
        candle(24.0, 24.5, 20.5, 21.0),
    ];
    assert_eq!(boundary_code(upside_gap_two_crows, &upside_gap_match), -100);
    assert_eq!(
        boundary_code(
            upside_gap_two_crows,
            &[
                candle(10.0, 23.0, 9.0, 22.0),
                upside_gap_match[1],
                upside_gap_match[2],
            ],
        ),
        0
    );

    let x_side = CDLXSIDEGAP3METHODSConfig::new(settings).unwrap();
    let upside_x_side = [
        candle(10.0, 12.5, 9.5, 12.0),
        candle(15.0, 18.5, 14.5, 18.0),
        candle(17.0, 17.5, 10.5, 11.0),
    ];
    let downside_x_side = [
        candle(20.0, 20.5, 17.5, 18.0),
        candle(15.0, 15.5, 11.5, 12.0),
        candle(13.0, 20.0, 12.5, 19.0),
    ];
    assert_eq!(boundary_code(x_side, &upside_x_side), 100);
    assert_eq!(boundary_code(x_side, &downside_x_side), -100);
    assert_eq!(
        boundary_code(
            x_side,
            &[
                candle(10.0, 15.5, 9.5, 15.0),
                upside_x_side[1],
                upside_x_side[2],
            ],
        ),
        0
    );
    assert_eq!(
        boundary_code(
            x_side,
            &[
                candle(20.0, 20.5, 14.5, 15.0),
                downside_x_side[1],
                downside_x_side[2],
            ],
        ),
        0
    );
}

#[test]
fn tristar_uses_one_i_minus_two_body_doji_threshold_and_strict_gap_direction() {
    let settings = CandleSettings::default().with_setting(
        CandleSettingType::BodyDoji,
        CandleSetting::new(CandleRangeKind::HighLow, 3, 0.1 as Float).unwrap(),
    );
    let tristar = CDLTRISTARConfig::new(settings).unwrap();
    let seeds = [
        candle(10.0, 15.0, 5.0, 11.0),
        candle(10.0, 15.0, 5.0, 11.0),
        candle(10.0, 15.0, 5.0, 11.0),
    ];
    let bearish = [
        seeds[0],
        seeds[1],
        seeds[2],
        candle(10.0, 60.0, 0.0, 10.5),
        candle(12.0, 13.0, 11.0, 12.5),
        candle(11.0, 12.0, 10.0, 11.5),
    ];
    let bullish = [
        seeds[0],
        seeds[1],
        seeds[2],
        candle(20.0, 60.0, 0.0, 20.5),
        candle(18.0, 19.0, 17.0, 18.5),
        candle(19.0, 20.0, 18.0, 19.5),
    ];
    assert_eq!(boundary_code(tristar, &bearish), -100);
    assert_eq!(boundary_code(tristar, &bullish), 100);

    let mut per_candle_recalculation_trap = bearish;
    per_candle_recalculation_trap[4] = candle(12.0, 15.0, 11.0, 14.0);
    assert_eq!(boundary_code(tristar, &per_candle_recalculation_trap), 0);

    let mut touching_real_bodies = bearish;
    touching_real_bodies[4] = candle(10.5, 11.5, 10.0, 11.0);
    touching_real_bodies[5] = candle(10.0, 11.0, 9.5, 10.5);
    assert_eq!(boundary_code(tristar, &touching_real_bodies), 0);
}

#[test]
fn gap_continuation_evidence_rows_cover_pinned_spans_settings_and_boundaries() {
    const ROWS: [(&str, &str, &str, &str, usize); 8] = [
        ("CDL2CROWS", "cdl2crows/cdl2crows.c", "three-Candle do-loop", "BodyLong; strict real-body gap; fixed bearish sign", 12),
        ("CDL3LINESTRIKE", "cdl3linestrike/cdl3linestrike.c", "four-Candle do-loop", "Near at i-3/i-2; sign from i-1", 8),
        ("CDLGAPSIDESIDEWHITE", "cdlgapsidesidewhite/cdlgapsidesidewhite.c", "three-Candle do-loop", "Near and Equal at i-1; sign from gap direction", 7),
        ("CDLSTICKSANDWICH", "cdlsticksandwich/cdlsticksandwich.c", "three-Candle do-loop", "Equal at i-2; fixed bullish sign", 7),
        ("CDLTASUKIGAP", "cdltasukigap/cdltasukigap.c", "three-Candle do-loop", "Near at i-1; strict gap and sign from i-1", 7),
        ("CDLTRISTAR", "cdltristar/cdltristar.c", "three-Candle do-loop", "one BodyDoji total at i-2; sign from gap direction", 12),
        ("CDLUPSIDEGAP2CROWS", "cdlupsidegap2crows/cdlupsidegap2crows.c", "three-Candle do-loop", "BodyLong at i-2 and BodyShort at i-1; fixed bearish sign", 12),
        ("CDLXSIDEGAP3METHODS", "cdlxsidegap3methods/cdlxsidegap3methods.c", "three-Candle do-loop", "no settings; fixed Lookback 2; sign from i-2", 2),
    ];
    assert_eq!(reference::TALIB_GIT_REVISION, "2247d599bddf37ed37e3a709371517e46efc66f6");
    assert!(ROWS.iter().all(|(_, source, span, settings, lookback)| {
        source.ends_with(".c") && !span.is_empty() && !settings.is_empty() && *lookback >= 2
    }));
}

#[test]
fn pinned_long_formation_oracles_qualify_every_definition_through_the_public_seam() {
    let settings = custom_two_candle_settings();
    let custom_penetration = Penetration::new(1.5 as Float).unwrap();

    qualify_two_candle!(
        CDLBREAKAWAYConfig::default(),
        CDLBREAKAWAYConfig::new(settings).unwrap(),
        BREAKAWAY_OPEN,
        BREAKAWAY_HIGH,
        BREAKAWAY_LOW,
        BREAKAWAY_CLOSE,
        BREAKAWAY_DEFAULT_LOOKBACK,
        BREAKAWAY_DEFAULT_F64_CODES,
        BREAKAWAY_DEFAULT_F32_CODES,
        BREAKAWAY_CUSTOM_LOOKBACK,
        BREAKAWAY_CUSTOM_F64_CODES,
        BREAKAWAY_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLLADDERBOTTOMConfig::default(),
        CDLLADDERBOTTOMConfig::new(settings).unwrap(),
        LADDERBOTTOM_OPEN,
        LADDERBOTTOM_HIGH,
        LADDERBOTTOM_LOW,
        LADDERBOTTOM_CLOSE,
        LADDERBOTTOM_DEFAULT_LOOKBACK,
        LADDERBOTTOM_DEFAULT_F64_CODES,
        LADDERBOTTOM_DEFAULT_F32_CODES,
        LADDERBOTTOM_CUSTOM_LOOKBACK,
        LADDERBOTTOM_CUSTOM_F64_CODES,
        LADDERBOTTOM_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLMATHOLDConfig::default(),
        CDLMATHOLDConfig::new(settings, custom_penetration).unwrap(),
        MATHOLD_OPEN,
        MATHOLD_HIGH,
        MATHOLD_LOW,
        MATHOLD_CLOSE,
        MATHOLD_DEFAULT_LOOKBACK,
        MATHOLD_DEFAULT_F64_CODES,
        MATHOLD_DEFAULT_F32_CODES,
        MATHOLD_CUSTOM_LOOKBACK,
        MATHOLD_CUSTOM_F64_CODES,
        MATHOLD_CUSTOM_F32_CODES
    );
    qualify_two_candle!(
        CDLRISEFALL3METHODSConfig::default(),
        CDLRISEFALL3METHODSConfig::new(settings).unwrap(),
        RISEFALL3METHODS_OPEN,
        RISEFALL3METHODS_HIGH,
        RISEFALL3METHODS_LOW,
        RISEFALL3METHODS_CLOSE,
        RISEFALL3METHODS_DEFAULT_LOOKBACK,
        RISEFALL3METHODS_DEFAULT_F64_CODES,
        RISEFALL3METHODS_DEFAULT_F32_CODES,
        RISEFALL3METHODS_CUSTOM_LOOKBACK,
        RISEFALL3METHODS_CUSTOM_F64_CODES,
        RISEFALL3METHODS_CUSTOM_F32_CODES
    );

    for (f64_codes, f32_codes) in [
        (
            reference::BREAKAWAY_DEFAULT_F64_CODES,
            reference::BREAKAWAY_DEFAULT_F32_CODES,
        ),
        (
            reference::BREAKAWAY_CUSTOM_F64_CODES,
            reference::BREAKAWAY_CUSTOM_F32_CODES,
        ),
        (
            reference::LADDERBOTTOM_DEFAULT_F64_CODES,
            reference::LADDERBOTTOM_DEFAULT_F32_CODES,
        ),
        (
            reference::LADDERBOTTOM_CUSTOM_F64_CODES,
            reference::LADDERBOTTOM_CUSTOM_F32_CODES,
        ),
        (
            reference::MATHOLD_DEFAULT_F64_CODES,
            reference::MATHOLD_DEFAULT_F32_CODES,
        ),
        (
            reference::MATHOLD_CUSTOM_F64_CODES,
            reference::MATHOLD_CUSTOM_F32_CODES,
        ),
        (
            reference::RISEFALL3METHODS_DEFAULT_F64_CODES,
            reference::RISEFALL3METHODS_DEFAULT_F32_CODES,
        ),
        (
            reference::RISEFALL3METHODS_CUSTOM_F64_CODES,
            reference::RISEFALL3METHODS_CUSTOM_F32_CODES,
        ),
    ] {
        assert_eq!(f64_codes, f32_codes);
        assert!(f64_codes.iter().any(|&code| code != 0));
    }
    assert!(reference::BREAKAWAY_DEFAULT_F64_CODES.contains(&100));
    assert!(reference::BREAKAWAY_DEFAULT_F64_CODES.contains(&-100));
    assert!(reference::RISEFALL3METHODS_DEFAULT_F64_CODES.contains(&100));
    assert!(reference::RISEFALL3METHODS_DEFAULT_F64_CODES.contains(&-100));
}

fn long_formation_boundary_settings() -> CandleSettings {
    CandleSettings::default()
        .with_setting(
            CandleSettingType::BodyLong,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 0.5 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::BodyShort,
            CandleSetting::new(CandleRangeKind::RealBody, 0, 2.0 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(CandleRangeKind::HighLow, 0, 0.1 as Float).unwrap(),
        )
}

#[test]
fn long_formation_scenarios_isolate_first_middle_final_and_strict_boundaries() {
    let settings = long_formation_boundary_settings();

    let breakaway = CDLBREAKAWAYConfig::new(settings).unwrap();
    let breakaway_match = [
        candle(20.0, 21.0, 9.0, 10.0),
        candle(8.0, 9.0, 5.0, 6.0),
        candle(5.0, 8.0, 4.0, 7.0),
        candle(6.0, 7.0, 3.0, 4.0),
        candle(4.0, 10.0, 3.5, 9.0),
    ];
    assert_eq!(boundary_code(breakaway, &breakaway_match), 100);
    let mut first_miss = breakaway_match;
    first_miss[0].close = first_miss[0].open;
    assert_eq!(boundary_code(breakaway, &first_miss), 0);
    let mut middle_miss = breakaway_match;
    middle_miss[2].high = middle_miss[1].high;
    assert_eq!(boundary_code(breakaway, &middle_miss), 0);
    let mut final_boundary = breakaway_match;
    final_boundary[4].close = final_boundary[1].open;
    assert_eq!(boundary_code(breakaway, &final_boundary), 0);

    let ladder = CDLLADDERBOTTOMConfig::new(settings).unwrap();
    let ladder_match = [
        candle(20.0, 20.5, 17.5, 18.0),
        candle(19.0, 19.5, 16.5, 17.0),
        candle(18.0, 18.5, 15.5, 16.0),
        candle(17.0, 19.0, 14.5, 15.0),
        candle(18.0, 20.5, 17.5, 20.0),
    ];
    assert_eq!(boundary_code(ladder, &ladder_match), 100);
    let mut first_miss = ladder_match;
    first_miss[0].close = first_miss[0].open;
    assert_eq!(boundary_code(ladder, &first_miss), 0);
    let mut middle_boundary = ladder_match;
    middle_boundary[2].open = middle_boundary[1].open;
    middle_boundary[2].high = middle_boundary[1].high;
    assert_eq!(boundary_code(ladder, &middle_boundary), 0);
    let mut final_boundary = ladder_match;
    final_boundary[4].close = final_boundary[3].high;
    assert_eq!(boundary_code(ladder, &final_boundary), 0);

    let mathold = CDLMATHOLDConfig::new(settings, Penetration::new(0.5 as Float).unwrap()).unwrap();
    let mathold_match = [
        candle(10.0, 21.0, 9.0, 20.0),
        candle(23.0, 23.5, 22.0, 22.5),
        candle(20.0, 20.5, 19.0, 19.5),
        candle(19.5, 20.0, 18.5, 19.0),
        candle(19.5, 24.5, 19.0, 24.0),
    ];
    assert_eq!(boundary_code(mathold, &mathold_match), 100);
    let mut first_miss = mathold_match;
    first_miss[0].close = first_miss[0].open;
    assert_eq!(boundary_code(mathold, &first_miss), 0);
    let mut middle_boundary = mathold_match;
    middle_boundary[3].open = middle_boundary[2].open;
    assert_eq!(boundary_code(mathold, &middle_boundary), 0);
    let mut final_boundary = mathold_match;
    final_boundary[4].close = final_boundary[1].high;
    assert_eq!(boundary_code(mathold, &final_boundary), 0);

    let rise_fall = CDLRISEFALL3METHODSConfig::new(settings).unwrap();
    let rise_fall_match = [
        candle(10.0, 21.0, 9.0, 20.0),
        candle(19.0, 19.5, 18.0, 18.5),
        candle(18.5, 19.0, 17.5, 18.0),
        candle(18.0, 18.5, 17.0, 17.5),
        candle(18.0, 21.5, 17.5, 21.0),
    ];
    assert_eq!(boundary_code(rise_fall, &rise_fall_match), 100);
    let mut first_miss = rise_fall_match;
    first_miss[0].close = first_miss[0].open;
    assert_eq!(boundary_code(rise_fall, &first_miss), 0);
    let mut middle_boundary = rise_fall_match;
    middle_boundary[2].open = 19.0 as Float;
    middle_boundary[2].close = middle_boundary[1].close;
    assert_eq!(boundary_code(rise_fall, &middle_boundary), 0);
    let mut final_boundary = rise_fall_match;
    final_boundary[4].close = final_boundary[0].close;
    assert_eq!(boundary_code(rise_fall, &final_boundary), 0);
}

#[test]
fn mathold_owns_pinned_penetration_and_accepts_nondefault_values_above_one() {
    let settings = CandleSettings::default();
    let above_one = Penetration::new(4.0 as Float).unwrap();
    assert_eq!(
        CDLMATHOLDConfig::default().penetration().value(),
        0.5 as Float
    );
    assert_eq!(
        CDLMATHOLDConfig::new(settings, above_one)
            .unwrap()
            .penetration(),
        above_one
    );
}

#[test]
fn long_formation_evidence_rows_cover_pinned_offsets_settings_signs_and_lookbacks() {
    const ROWS: [(&str, &str, &str, &str, &str, &str, usize); 4] = [
        (
            "CDLBREAKAWAY",
            "cdlbreakaway/cdlbreakaway.c",
            "five-Candle do-loop at i-4..i",
            "BodyLong at i-4; exact real-body gap and strict high/low staircase",
            "branch from i-4 color; sign from i",
            "BodyLong average period + 4",
            14,
        ),
        (
            "CDLLADDERBOTTOM",
            "cdlladderbottom/cdlladderbottom.c",
            "five-Candle do-loop at i-4..i",
            "ShadowVeryShort at i-1; strict descending opens/closes and final breakout",
            "fixed bullish sign",
            "ShadowVeryShort average period + 4",
            14,
        ),
        (
            "CDLMATHOLD",
            "cdlmathold/cdlmathold.c",
            "five-Candle do-loop at i-4..i",
            "BodyLong at i-4; BodyShort at i-3..i-1; Penetration default 0.5",
            "fixed bullish sign",
            "max(BodyLong, BodyShort) average period + 4",
            14,
        ),
        (
            "CDLRISEFALL3METHODS",
            "cdlrisefall3methods/cdlrisefall3methods.c",
            "five-Candle do-loop at i-4..i",
            "BodyLong at i-4/i; BodyShort at i-3..i-1; strict range intersections",
            "comparisons and sign from i-4 color",
            "max(BodyLong, BodyShort) average period + 4",
            14,
        ),
    ];
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "2247d599bddf37ed37e3a709371517e46efc66f6"
    );
    assert!(ROWS.iter().all(
        |(name, source, span, predicate, direction, lookback_formula, default_lookback)| {
            name.starts_with("CDL")
                && source.ends_with(".c")
                && span.contains("five-Candle")
                && !predicate.is_empty()
                && !direction.is_empty()
                && lookback_formula.ends_with("+ 4")
                && *default_lookback == 14
        }
    ));
}

fn hikkake_series(candles: &[Candle]) -> Series {
    Series {
        open: candles.iter().map(|candle| candle.open).collect(),
        high: candles.iter().map(|candle| candle.high).collect(),
        low: candles.iter().map(|candle| candle.low).collect(),
        close: candles.iter().map(|candle| candle.close).collect(),
    }
}

fn hikkake_codes<C>(config: C, candles: &[Candle]) -> Vec<i32>
where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C: IndicatorConfig<Input<'a> = CandleInput<'a>>,
{
    let series = hikkake_series(candles);
    config
        .compute(series.input())
        .unwrap()
        .values()
        .iter()
        .map(|signal| signal.to_talib_code())
        .collect()
}

fn ordinary_formation(direction: PatternDirection) -> Vec<Candle> {
    let mut candles = vec![candle(10.0, 12.0, 8.0, 10.0); 6];
    candles.extend(match direction {
        PatternDirection::Bullish => [
            candle(10.0, 15.0, 5.0, 10.0),
            candle(10.0, 14.0, 6.0, 10.0),
            candle(9.0, 13.0, 4.0, 8.0),
        ],
        PatternDirection::Bearish => [
            candle(10.0, 15.0, 5.0, 10.0),
            candle(10.0, 14.0, 6.0, 10.0),
            candle(11.0, 16.0, 7.0, 12.0),
        ],
    });
    candles
}

fn modified_formation(direction: PatternDirection) -> Vec<Candle> {
    let mut candles = vec![candle(10.0, 15.0, 5.0, 10.0); 10];
    candles.extend(match direction {
        PatternDirection::Bullish => [
            candle(10.0, 16.0, 4.0, 10.0),
            candle(10.0, 15.0, 5.0, 6.0),
            candle(10.0, 14.0, 6.0, 10.0),
            candle(9.0, 13.0, 5.0, 8.0),
        ],
        PatternDirection::Bearish => [
            candle(10.0, 16.0, 4.0, 10.0),
            candle(10.0, 15.0, 5.0, 14.0),
            candle(10.0, 14.0, 6.0, 10.0),
            candle(11.0, 15.0, 7.0, 12.0),
        ],
    });
    candles
}

fn hikkake_filler() -> Candle {
    candle(10.0, 17.0, 4.0, 10.0)
}

fn ordinary_confirmation(direction: PatternDirection) -> Candle {
    match direction {
        PatternDirection::Bullish => candle(14.0, 17.0, 4.0, 15.0),
        PatternDirection::Bearish => candle(6.0, 17.0, 4.0, 5.0),
    }
}

fn modified_confirmation(direction: PatternDirection) -> Candle {
    match direction {
        PatternDirection::Bullish => candle(14.0, 17.0, 4.0, 15.0),
        PatternDirection::Bearish => candle(6.0, 17.0, 4.0, 5.0),
    }
}

fn with_confirmation_age(
    mut candles: Vec<Candle>,
    direction: PatternDirection,
    age: usize,
    confirmation: fn(PatternDirection) -> Candle,
) -> Vec<Candle> {
    candles.extend((1..age).map(|_| hikkake_filler()));
    candles.push(confirmation(direction));
    candles
}

#[test]
fn pinned_hikkake_fixtures_qualify_default_and_custom_f64_f32_in_all_four_modes() {
    assert_eq!(reference::TALIB_VERSION, "0.7.1");
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "2247d599bddf37ed37e3a709371517e46efc66f6"
    );

    let ordinary = Series::from_fixture(
        reference::HIKKAKE_OPEN,
        reference::HIKKAKE_HIGH,
        reference::HIKKAKE_LOW,
        reference::HIKKAKE_CLOSE,
    );
    let ordinary_default = expected_codes(
        reference::HIKKAKE_DEFAULT_F64_CODES,
        reference::HIKKAKE_DEFAULT_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLHIKKAKEConfig::default(),
        &ordinary,
        reference::HIKKAKE_DEFAULT_LOOKBACK,
        &ordinary_default,
    );

    let custom_settings = CandleSettings::default().with_setting(
        CandleSettingType::Near,
        CandleSetting::new(CandleRangeKind::HighLow, 3, 0.125 as Float).unwrap(),
    );
    let ordinary_custom = expected_codes(
        reference::HIKKAKE_CUSTOM_F64_CODES,
        reference::HIKKAKE_CUSTOM_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLHIKKAKEConfig::new(custom_settings).unwrap(),
        &ordinary,
        reference::HIKKAKE_CUSTOM_LOOKBACK,
        &ordinary_custom,
    );

    let modified = Series::from_fixture(
        reference::HIKKAKEMOD_OPEN,
        reference::HIKKAKEMOD_HIGH,
        reference::HIKKAKEMOD_LOW,
        reference::HIKKAKEMOD_CLOSE,
    );
    let modified_default = expected_codes(
        reference::HIKKAKEMOD_DEFAULT_F64_CODES,
        reference::HIKKAKEMOD_DEFAULT_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLHIKKAKEMODConfig::default(),
        &modified,
        reference::HIKKAKEMOD_DEFAULT_LOOKBACK,
        &modified_default,
    );
    let modified_custom = expected_codes(
        reference::HIKKAKEMOD_CUSTOM_F64_CODES,
        reference::HIKKAKEMOD_CUSTOM_F32_CODES,
    );
    qualify_pattern_fixture(
        CDLHIKKAKEMODConfig::new(custom_settings).unwrap(),
        &modified,
        reference::HIKKAKEMOD_CUSTOM_LOOKBACK,
        &modified_custom,
    );
}

#[test]
fn independent_hikkake_scenarios_cover_both_directions_and_confirmation_ages_one_to_three() {
    for direction in [PatternDirection::Bullish, PatternDirection::Bearish] {
        let standard = match direction {
            PatternDirection::Bullish => 100,
            PatternDirection::Bearish => -100,
        };
        let confirmed = standard * 2;
        for age in 1..=3 {
            let ordinary = with_confirmation_age(
                ordinary_formation(direction),
                direction,
                age,
                ordinary_confirmation,
            );
            let ordinary_codes = hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary);
            assert_eq!(ordinary_codes[3], standard, "{direction:?} age {age}");
            assert_eq!(ordinary_codes[3 + age], confirmed, "{direction:?} age {age}");
            assert_eq!(
                ordinary_codes
                    .iter()
                    .filter(|&&code| code == confirmed)
                    .count(),
                1
            );

            let modified = with_confirmation_age(
                modified_formation(direction),
                direction,
                age,
                modified_confirmation,
            );
            let modified_codes = hikkake_codes(CDLHIKKAKEMODConfig::default(), &modified);
            assert_eq!(modified_codes[3], standard, "{direction:?} age {age}");
            assert_eq!(modified_codes[3 + age], confirmed, "{direction:?} age {age}");
            assert_eq!(
                modified_codes
                    .iter()
                    .filter(|&&code| code == confirmed)
                    .count(),
                1
            );
        }
    }
}

#[test]
fn hikkake_confirmation_is_strict_uses_the_pinned_boundary_and_expires_after_age_three() {
    for direction in [PatternDirection::Bullish, PatternDirection::Bearish] {
        let mut ordinary = ordinary_formation(direction);
        ordinary.push(match direction {
            PatternDirection::Bullish => candle(13.0, 17.0, 4.0, 14.0),
            PatternDirection::Bearish => candle(7.0, 17.0, 4.0, 6.0),
        });
        ordinary.push(ordinary_confirmation(direction));
        let ordinary_codes = hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary);
        assert_eq!(ordinary_codes[4], 0, "ordinary boundary contact");
        assert_eq!(
            ordinary_codes[5],
            if direction == PatternDirection::Bullish { 200 } else { -200 }
        );

        let mut modified = modified_formation(direction);
        modified.push(match direction {
            PatternDirection::Bullish => candle(13.0, 17.0, 4.0, 13.5),
            PatternDirection::Bearish => candle(7.0, 17.0, 4.0, 6.5),
        });
        modified.push(match direction {
            PatternDirection::Bullish => candle(13.0, 17.0, 4.0, 14.0),
            PatternDirection::Bearish => candle(7.0, 17.0, 4.0, 6.0),
        });
        modified.push(modified_confirmation(direction));
        let modified_codes = hikkake_codes(CDLHIKKAKEMODConfig::default(), &modified);
        assert_eq!(&modified_codes[4..=5], &[0, 0]);
        assert_eq!(
            modified_codes[6],
            if direction == PatternDirection::Bullish { 200 } else { -200 }
        );

        let mut ordinary_expired = ordinary_formation(direction);
        ordinary_expired.extend([hikkake_filler(); 3]);
        ordinary_expired.push(ordinary_confirmation(direction));
        assert_eq!(
            hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary_expired)
                .last()
                .copied(),
            Some(0)
        );

        let mut modified_expired = modified_formation(direction);
        modified_expired.extend([hikkake_filler(); 3]);
        modified_expired.push(modified_confirmation(direction));
        assert_eq!(
            hikkake_codes(CDLHIKKAKEMODConfig::default(), &modified_expired)
                .last()
                .copied(),
            Some(0)
        );
    }
}

#[test]
fn newer_hikkake_formation_replaces_pending_and_wins_same_position_precedence() {
    let mut ordinary = vec![candle(10.0, 12.0, 8.0, 10.0); 6];
    ordinary.extend([
        candle(10.0, 20.0, 0.0, 10.0),
        candle(10.0, 18.0, 2.0, 10.0),
        candle(9.0, 16.0, 1.0, 8.0),
        candle(10.0, 15.0, 2.0, 10.0),
        candle(10.0, 20.0, 3.0, 19.0),
        candle(5.0, 17.0, 1.0, 1.0),
    ]);
    let ordinary_codes = hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary);
    assert_eq!(&ordinary_codes[3..=6], &[100, 0, -100, -200]);
    assert!(!ordinary_codes.contains(&200));

    let mut modified = vec![candle(10.0, 15.0, 5.0, 10.0); 10];
    modified.extend([
        candle(10.0, 20.0, 0.0, 10.0),
        candle(10.0, 18.0, 2.0, 3.0),
        candle(10.0, 16.0, 4.0, 10.0),
        candle(9.0, 15.0, 3.0, 8.0),
        candle(10.0, 14.0, 4.0, 13.0),
        candle(10.0, 13.0, 5.0, 10.0),
        candle(10.0, 18.0, 6.0, 17.0),
        candle(5.0, 17.0, 4.0, 4.0),
    ]);
    let modified_codes = hikkake_codes(CDLHIKKAKEMODConfig::default(), &modified);
    assert_eq!(&modified_codes[3..=7], &[100, 0, 0, -100, -200]);
    assert!(!modified_codes.contains(&200));
}

#[test]
fn modified_hikkake_owns_near_average_and_its_exact_non_strict_formation_boundary() {
    let settings = CandleSettings::default().with_setting(
        CandleSettingType::Near,
        CandleSetting::new(CandleRangeKind::HighLow, 0, 0.2 as Float).unwrap(),
    );
    let config = CDLHIKKAKEMODConfig::new(settings).unwrap();
    assert_eq!(config.warm_up(), 6);
    assert_eq!(CDLHIKKAKEMODConfig::default().warm_up(), 10);

    let mut bullish = vec![candle(10.0, 15.0, 5.0, 10.0); 6];
    bullish.extend([
        candle(10.0, 16.0, 4.0, 10.0),
        candle(10.0, 15.0, 5.0, 7.0),
        candle(10.0, 14.0, 6.0, 10.0),
        candle(9.0, 13.0, 5.0, 8.0),
    ]);
    assert_eq!(hikkake_codes(config, &bullish).last().copied(), Some(100));
    bullish[7].close = 7.01 as Float;
    assert_eq!(hikkake_codes(config, &bullish).last().copied(), Some(0));

    let mut bearish = vec![candle(10.0, 15.0, 5.0, 10.0); 6];
    bearish.extend([
        candle(10.0, 16.0, 4.0, 10.0),
        candle(10.0, 15.0, 5.0, 13.0),
        candle(10.0, 14.0, 6.0, 10.0),
        candle(11.0, 15.0, 7.0, 12.0),
    ]);
    assert_eq!(hikkake_codes(config, &bearish).last().copied(), Some(-100));
    bearish[7].close = 12.99 as Float;
    assert_eq!(hikkake_codes(config, &bearish).last().copied(), Some(0));
}

#[test]
fn silent_hikkake_transitions_reconstruct_and_consume_pending_before_lookback() {
    let ordinary_reconstructed = [
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 14.0, 6.0, 10.0),
        candle(9.0, 13.0, 4.0, 8.0),
        hikkake_filler(),
        hikkake_filler(),
        ordinary_confirmation(PatternDirection::Bullish),
    ];
    assert_eq!(
        hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary_reconstructed),
        [200]
    );

    let ordinary_consumed = [
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 14.0, 6.0, 10.0),
        candle(9.0, 13.0, 4.0, 8.0),
        ordinary_confirmation(PatternDirection::Bullish),
        hikkake_filler(),
        hikkake_filler(),
    ];
    assert_eq!(
        hikkake_codes(CDLHIKKAKEConfig::default(), &ordinary_consumed),
        [0]
    );

    let modified_reconstructed = [
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 15.0, 5.0, 10.0),
        candle(10.0, 16.0, 4.0, 10.0),
        candle(10.0, 15.0, 5.0, 6.0),
        candle(10.0, 14.0, 6.0, 10.0),
        candle(9.0, 13.0, 5.0, 8.0),
        hikkake_filler(),
        hikkake_filler(),
        modified_confirmation(PatternDirection::Bullish),
    ];
    assert_eq!(
        hikkake_codes(CDLHIKKAKEMODConfig::default(), &modified_reconstructed),
        [200]
    );
}

fn assert_prepared_hikkake_isolation<C>(config: C, pending: &[Candle])
where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C: IndicatorConfig<
        Input<'a> = CandleInput<'a>,
        OutputMut<'a> = &'a mut [PatternSignal],
    >,
    C::BatchRunner: PreparedBatchRunner<C>,
{
    let pending = hikkake_series(pending);
    let neutral_candles = vec![candle(10.0, 15.0, 5.0, 10.0); pending.open.len()];
    let neutral = hikkake_series(&neutral_candles);
    let output_len = pending.open.len() - config.lookback();
    let mut runner = config.prepare_batch(pending.open.len()).unwrap();
    let mut output = vec![PatternSignal::NoMatch; output_len];
    runner.compute_into(pending.input(), &mut output).unwrap();
    output.fill(PatternSignal::Match {
        direction: PatternDirection::Bearish,
        strength: PatternStrength::Confirmed,
    });
    runner.compute_into(neutral.input(), &mut output).unwrap();
    assert!(output.iter().all(|signal| *signal == PatternSignal::NoMatch));
}

fn assert_streaming_hikkake_lifecycle<C>(
    config: C,
    candles: &[Candle],
    formation_index: usize,
    expected_confirmation: i32,
) where
    C: Copy + 'static + IndicatorConfig<Output = Vec<PatternSignal>>,
    for<'a> C: IndicatorConfig<Input<'a> = CandleInput<'a>>,
    C::Stream: StreamingComputation<C, Tick = Candle, TickOutput = PatternSignal>,
{
    let expected = hikkake_codes(config, candles);
    for split in (formation_index + 1)..candles.len() {
        let mut stream = config.stream().unwrap();
        let mut actual = Vec::new();
        for candle in &candles[..split] {
            if let Some(signal) = stream.next(*candle).unwrap() {
                actual.push(signal.to_talib_code());
            }
        }
        for candle in &candles[split..] {
            if let Some(signal) = stream.next(*candle).unwrap() {
                actual.push(signal.to_talib_code());
            }
        }
        assert_eq!(actual, expected, "split after source position {}", split - 1);
    }

    let mut stream = config.stream().unwrap();
    for candle in &candles[..=formation_index] {
        stream.next(*candle).unwrap();
    }
    let invalid = Candle {
        close: Float::NAN,
        ..hikkake_filler()
    };
    assert!(stream.next(invalid).is_err());
    let mut last = PatternSignal::NoMatch;
    for candle in &candles[formation_index + 1..] {
        if let Some(signal) = stream.next(*candle).unwrap() {
            last = signal;
        }
    }
    assert_eq!(last.to_talib_code(), expected_confirmation);

    let mut reset = config.stream().unwrap();
    for candle in &candles[..=formation_index] {
        reset.next(*candle).unwrap();
    }
    reset.reset();
    for _ in 0..config.lookback() {
        assert_eq!(reset.next(hikkake_filler()).unwrap(), None);
    }
    assert_eq!(
        reset.next(*candles.last().unwrap()).unwrap(),
        Some(PatternSignal::NoMatch)
    );
}

#[test]
fn prepared_and_streaming_hikkake_state_is_isolated_retained_reset_and_retry_safe() {
    let ordinary_pending = ordinary_formation(PatternDirection::Bullish);
    assert_prepared_hikkake_isolation(CDLHIKKAKEConfig::default(), &ordinary_pending);
    let ordinary_age_three = with_confirmation_age(
        ordinary_pending,
        PatternDirection::Bullish,
        3,
        ordinary_confirmation,
    );
    assert_streaming_hikkake_lifecycle(
        CDLHIKKAKEConfig::default(),
        &ordinary_age_three,
        8,
        200,
    );

    let modified_pending = modified_formation(PatternDirection::Bearish);
    assert_prepared_hikkake_isolation(CDLHIKKAKEMODConfig::default(), &modified_pending);
    let modified_age_three = with_confirmation_age(
        modified_pending,
        PatternDirection::Bearish,
        3,
        modified_confirmation,
    );
    assert_streaming_hikkake_lifecycle(
        CDLHIKKAKEMODConfig::default(),
        &modified_age_three,
        13,
        -200,
    );
}

#[test]
fn hikkake_evidence_rows_cover_pinned_state_predicates_boundaries_and_transitions() {
    const ROWS: [(&str, &str, &str, &str, &str, &str, usize, usize); 2] = [
        (
            "CDLHIKKAKE",
            "cdlhikkake/cdlhikkake.c",
            "inside i-1 within i-2; i lower/lower or higher/higher",
            "no Candle Average",
            "strict close beyond the high/low of i-1 at formation",
            "fixed Lookback 5; transition starts at Lookback - 3",
            2,
            5,
        ),
        (
            "CDLHIKKAKEMOD",
            "cdlhikkakemod/cdlhikkakemod.c",
            "nested inside i-2/i-3 and i-1/i-2; i lower/lower or higher/higher",
            "Near at i-2 with non-strict close boundary",
            "strict close beyond the high/low of i-1 at formation",
            "Lookback max(1, Near period) + 5; transition starts three earlier",
            7,
            10,
        ),
    ];
    assert_eq!(
        reference::TALIB_GIT_REVISION,
        "2247d599bddf37ed37e3a709371517e46efc66f6"
    );
    assert_eq!(ROWS[0].6, CDLHIKKAKEConfig::default().warm_up() - 3);
    assert_eq!(ROWS[0].7, CDLHIKKAKEConfig::default().warm_up());
    assert_eq!(ROWS[1].6, CDLHIKKAKEMODConfig::default().warm_up() - 3);
    assert_eq!(ROWS[1].7, CDLHIKKAKEMODConfig::default().warm_up());
    assert!(ROWS.iter().all(
        |(name, source, formation, average, confirmation, formula, _, _)| {
            name.starts_with("CDLHIKKAKE")
                && source.ends_with(".c")
                && formation.contains("inside")
                && !average.is_empty()
                && confirmation.contains("strict")
                && formula.contains("transition starts")
        }
    ));
}
