use ta_core::inventory::{
    function, FunctionGroup, ImplementationStatus, FUNCTION_COUNT, IMPLEMENTED_FUNCTION_COUNT,
    TALIB_FUNCTIONS,
};
use ta_core::math_operators::{
    ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, MINMAXINDEX, MULT, SUB, SUM,
};
use ta_core::math_transform::{
    ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
};
use ta_core::overlap::{DEMA, EMA, MA, SMA, T3, TEMA, TRIMA, WMA};
use ta_core::price_transform::{AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE};
use ta_core::statistic::{
    BETABatchRunner, BETAConfig, BETAStream, CORRELBatchRunner, CORRELConfig, CORRELStream,
    LINEARREGBatchRunner, LINEARREGConfig, LINEARREGStream, LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEConfig, LINEARREG_ANGLEStream, LINEARREG_INTERCEPTBatchRunner,
    LINEARREG_INTERCEPTConfig, LINEARREG_INTERCEPTStream, LINEARREG_SLOPEBatchRunner,
    LINEARREG_SLOPEConfig, LINEARREG_SLOPEStream, STDDEVBatchRunner, STDDEVConfig, STDDEVStream,
    TSFBatchRunner, TSFConfig, TSFStream, VARBatchRunner, VARConfig, VARStream, BETA, CORREL,
    LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR,
};
use ta_core::volatility::{ATR, NATR, TRANGE};
use ta_core::volume::{AD, ADOSC, OBV};
use ta_core::{
    Indicator, IndicatorConfig, PreparedBatchRunner, StreamingComputation, StreamingIndicator,
};

#[test]
fn inventory_contains_official_161_functions() {
    assert_eq!(FUNCTION_COUNT, 161);
    assert_eq!(TALIB_FUNCTIONS.len(), FUNCTION_COUNT);
}

#[test]
fn group_counts_match_official_talib_inventory() {
    let expected = [
        (FunctionGroup::OverlapStudies, 18),
        (FunctionGroup::MomentumIndicators, 31),
        (FunctionGroup::VolumeIndicators, 3),
        (FunctionGroup::VolatilityIndicators, 3),
        (FunctionGroup::PriceTransform, 5),
        (FunctionGroup::CycleIndicators, 5),
        (FunctionGroup::PatternRecognition, 61),
        (FunctionGroup::StatisticFunctions, 9),
        (FunctionGroup::MathTransform, 15),
        (FunctionGroup::MathOperators, 11),
    ];

    let mut total = 0;
    for (group, count) in expected {
        assert_eq!(group.expected_count(), count, "{:?} expected count", group);
        assert_eq!(
            TALIB_FUNCTIONS
                .iter()
                .filter(|info| info.group == group)
                .count(),
            count,
            "{:?} actual count",
            group
        );
        total += count;
    }

    assert_eq!(total, FUNCTION_COUNT);
    assert_eq!(FunctionGroup::ALL.len(), 10);
}

#[test]
fn function_names_are_unique() {
    for (idx, info) in TALIB_FUNCTIONS.iter().enumerate() {
        assert!(
            TALIB_FUNCTIONS[idx + 1..]
                .iter()
                .all(|other| other.name != info.name),
            "duplicate function name {}",
            info.name
        );
    }
}

#[test]
fn first_tranche_functions_are_marked_implemented() {
    let implemented = [
        "SMA",
        "DEMA",
        "EMA",
        "MA",
        "T3",
        "TEMA",
        "TRIMA",
        "WMA",
        "AVGDEV",
        "AVGPRICE",
        "MEDPRICE",
        "TYPPRICE",
        "WCLPRICE",
        "AD",
        "ADOSC",
        "OBV",
        "ATR",
        "NATR",
        "TRANGE",
        "BETA",
        "CORREL",
        "LINEARREG",
        "LINEARREG_ANGLE",
        "LINEARREG_INTERCEPT",
        "LINEARREG_SLOPE",
        "STDDEV",
        "TSF",
        "VAR",
        "ACOS",
        "ASIN",
        "ATAN",
        "CEIL",
        "COS",
        "COSH",
        "EXP",
        "FLOOR",
        "LN",
        "LOG10",
        "SIN",
        "SINH",
        "SQRT",
        "TAN",
        "TANH",
        "ADD",
        "DIV",
        "MAX",
        "MAXINDEX",
        "MIN",
        "MININDEX",
        "MINMAX",
        "MINMAXINDEX",
        "MULT",
        "SUB",
        "SUM",
    ];

    assert_eq!(IMPLEMENTED_FUNCTION_COUNT, implemented.len());
    assert_eq!(
        TALIB_FUNCTIONS
            .iter()
            .filter(|info| info.is_implemented())
            .count(),
        IMPLEMENTED_FUNCTION_COUNT
    );

    for name in implemented {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert!(!info.rust_module().is_empty());
    }
}

#[test]
fn first_tranche_structs_implement_batch_and_streaming_traits() {
    fn assert_indicator<T: Indicator>() {}
    fn assert_streaming<T: StreamingIndicator>() {}

    assert_indicator::<SMA>();
    assert_streaming::<SMA>();
    assert_indicator::<DEMA>();
    assert_streaming::<DEMA>();
    assert_indicator::<EMA>();
    assert_streaming::<EMA>();
    assert_indicator::<MA>();
    assert_streaming::<MA>();
    assert_indicator::<T3>();
    assert_streaming::<T3>();
    assert_indicator::<TEMA>();
    assert_streaming::<TEMA>();
    assert_indicator::<TRIMA>();
    assert_streaming::<TRIMA>();
    assert_indicator::<WMA>();
    assert_streaming::<WMA>();

    assert_indicator::<AVGDEV>();
    assert_streaming::<AVGDEV>();
    assert_indicator::<AVGPRICE>();
    assert_streaming::<AVGPRICE>();
    assert_indicator::<MEDPRICE>();
    assert_streaming::<MEDPRICE>();
    assert_indicator::<TYPPRICE>();
    assert_streaming::<TYPPRICE>();
    assert_indicator::<WCLPRICE>();
    assert_streaming::<WCLPRICE>();

    assert_indicator::<AD>();
    assert_streaming::<AD>();
    assert_indicator::<ADOSC>();
    assert_streaming::<ADOSC>();
    assert_indicator::<OBV>();
    assert_streaming::<OBV>();

    assert_indicator::<ATR>();
    assert_streaming::<ATR>();
    assert_indicator::<NATR>();
    assert_streaming::<NATR>();
    assert_indicator::<TRANGE>();
    assert_streaming::<TRANGE>();

    assert_indicator::<BETA>();
    assert_streaming::<BETA>();
    assert_indicator::<CORREL>();
    assert_streaming::<CORREL>();
    assert_indicator::<LINEARREG>();
    assert_streaming::<LINEARREG>();
    assert_indicator::<LINEARREG_ANGLE>();
    assert_streaming::<LINEARREG_ANGLE>();
    assert_indicator::<LINEARREG_INTERCEPT>();
    assert_streaming::<LINEARREG_INTERCEPT>();
    assert_indicator::<LINEARREG_SLOPE>();
    assert_streaming::<LINEARREG_SLOPE>();
    assert_indicator::<STDDEV>();
    assert_streaming::<STDDEV>();
    assert_indicator::<TSF>();
    assert_streaming::<TSF>();
    assert_indicator::<VAR>();
    assert_streaming::<VAR>();

    assert_indicator::<ACOS>();
    assert_streaming::<ACOS>();
    assert_indicator::<ASIN>();
    assert_streaming::<ASIN>();
    assert_indicator::<ATAN>();
    assert_streaming::<ATAN>();
    assert_indicator::<CEIL>();
    assert_streaming::<CEIL>();
    assert_indicator::<COS>();
    assert_streaming::<COS>();
    assert_indicator::<COSH>();
    assert_streaming::<COSH>();
    assert_indicator::<EXP>();
    assert_streaming::<EXP>();
    assert_indicator::<FLOOR>();
    assert_streaming::<FLOOR>();
    assert_indicator::<LN>();
    assert_streaming::<LN>();
    assert_indicator::<LOG10>();
    assert_streaming::<LOG10>();
    assert_indicator::<SIN>();
    assert_streaming::<SIN>();
    assert_indicator::<SINH>();
    assert_streaming::<SINH>();
    assert_indicator::<SQRT>();
    assert_streaming::<SQRT>();
    assert_indicator::<TAN>();
    assert_streaming::<TAN>();
    assert_indicator::<TANH>();
    assert_streaming::<TANH>();

    assert_indicator::<ADD>();
    assert_streaming::<ADD>();
    assert_indicator::<DIV>();
    assert_streaming::<DIV>();
    assert_indicator::<MAX>();
    assert_streaming::<MAX>();
    assert_indicator::<MAXINDEX>();
    assert_streaming::<MAXINDEX>();
    assert_indicator::<MIN>();
    assert_streaming::<MIN>();
    assert_indicator::<MININDEX>();
    assert_streaming::<MININDEX>();
    assert_indicator::<MINMAX>();
    assert_streaming::<MINMAX>();
    assert_indicator::<MINMAXINDEX>();
    assert_streaming::<MINMAXINDEX>();
    assert_indicator::<MULT>();
    assert_streaming::<MULT>();
    assert_indicator::<SUB>();
    assert_streaming::<SUB>();
    assert_indicator::<SUM>();
    assert_streaming::<SUM>();
}

#[test]
fn migrated_rolling_statistics_are_in_the_public_execution_catalogue() {
    fn assert_execution_types<C, R, S>()
    where
        C: IndicatorConfig<BatchRunner = R, Stream = S>,
        R: PreparedBatchRunner<C>,
        S: StreamingComputation<C>,
    {
    }

    assert_execution_types::<VARConfig, VARBatchRunner, VARStream>();
    assert_execution_types::<STDDEVConfig, STDDEVBatchRunner, STDDEVStream>();
    assert_execution_types::<CORRELConfig, CORRELBatchRunner, CORRELStream>();
    assert_execution_types::<BETAConfig, BETABatchRunner, BETAStream>();
    assert_execution_types::<LINEARREGConfig, LINEARREGBatchRunner, LINEARREGStream>();
    assert_execution_types::<
        LINEARREG_SLOPEConfig,
        LINEARREG_SLOPEBatchRunner,
        LINEARREG_SLOPEStream,
    >();
    assert_execution_types::<
        LINEARREG_INTERCEPTConfig,
        LINEARREG_INTERCEPTBatchRunner,
        LINEARREG_INTERCEPTStream,
    >();
    assert_execution_types::<
        LINEARREG_ANGLEConfig,
        LINEARREG_ANGLEBatchRunner,
        LINEARREG_ANGLEStream,
    >();
    assert_execution_types::<TSFConfig, TSFBatchRunner, TSFStream>();
}

#[test]
fn deferred_functions_remain_planned() {
    for name in ["KAMA", "MAMA", "MACD", "BBANDS", "CDLDOJI", "HT_SINE"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Planned, "{name}");
    }
}

#[test]
fn non_talib_local_plan_extras_are_not_in_official_inventory() {
    for name in ["WWMA", "HMA", "VWAP"] {
        assert!(
            function(name).is_none(),
            "{name} should not be in TA-Lib inventory"
        );
    }
}
