//! Public-seam qualification for the Pattern Recognition foundation tracers.

#[path = "fixtures/pattern_recognition_reference.rs"]
mod reference;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use ta_core::pattern_recognition::{
    Candle, CandleInput, CandleRangeKind, CandleSetting, CandleSettingType, CandleSettings,
    PatternDirection, PatternSignal, PatternStrength, Penetration, CDLBELTHOLDConfig,
    CDLCLOSINGMARUBOZUConfig, CDLCOUNTERATTACKConfig, CDLDARKCLOUDCOVERConfig, CDLDOJIConfig,
    CDLDOJISTARConfig, CDLDRAGONFLYDOJIConfig, CDLENGULFINGConfig, CDLGRAVESTONEDOJIConfig,
    CDLHARAMIConfig, CDLHARAMICROSSConfig, CDLHIGHWAVEConfig, CDLHOMINGPIGEONConfig,
    CDLKICKINGBYLENGTHConfig, CDLKICKINGConfig, CDLLONGLEGGEDDOJIConfig, CDLLONGLINEConfig,
    CDLMARUBOZUConfig, CDLMATCHINGLOWConfig, CDLRICKSHAWMANConfig, CDLSHORTLINEConfig,
    CDLSPINNINGTOPConfig, CDLTAKURIConfig,
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
    let mut caller_output = vec![sentinel; expected.len() + 2];
    let range = config
        .compute_into(series.input(), &mut caller_output)
        .unwrap();
    assert_eq!(range, OutputRange::new(lookback, expected.len()));
    assert_eq!(&caller_output[..expected.len()], expected);
    assert_eq!(&caller_output[expected.len()..], &[sentinel, sentinel]);

    let mut prepared = config.prepare_batch(series.open.len()).unwrap();
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
            CandleSettingType::BodyDoji,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.1 as Float).unwrap(),
        )
        .with_setting(
            CandleSettingType::ShadowVeryShort,
            CandleSetting::new(CandleRangeKind::HighLow, 3, 0.1 as Float).unwrap(),
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

#[test]
fn two_candle_evidence_rows_cover_pinned_sources_and_qualification_scenarios() {
    const ROWS: [(&str, &str, &str); 9] = [
        ("CDLCOUNTERATTACK", "cdlcounterattack/cdlcounterattack.c", "exact equal closes"),
        ("CDLDARKCLOUDCOVER", "cdldarkcloudcover/cdldarkcloudcover.c", "Penetration"),
        ("CDLDOJISTAR", "cdldojistar/cdldojistar.c", "strict real-body gap"),
        ("CDLHARAMI", "cdlharami/cdlharami.c", "Standard and Partial"),
        ("CDLHARAMICROSS", "cdlharamicross/cdlharamicross.c", "Standard and Partial"),
        ("CDLHOMINGPIGEON", "cdlhomingpigeon/cdlhomingpigeon.c", "strict containment"),
        ("CDLKICKING", "cdlkicking/cdlkicking.c", "strict Candle gap"),
        ("CDLKICKINGBYLENGTH", "cdlkickingbylength/cdlkickingbylength.c", "first-Candle tie"),
        ("CDLMATCHINGLOW", "cdlmatchinglow/cdlmatchinglow.c", "exact equal closes"),
    ];
    assert_eq!(reference::TALIB_GIT_REVISION, "2247d599bddf37ed37e3a709371517e46efc66f6");
    assert!(ROWS.iter().all(|(_, source, scenario)| source.ends_with(".c") && !scenario.is_empty()));
}
