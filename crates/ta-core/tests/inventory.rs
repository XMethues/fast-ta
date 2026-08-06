use ta_core::inventory::{
    function, FunctionGroup, ImplementationStatus, FUNCTION_COUNT, IMPLEMENTED_FUNCTION_COUNT,
    TALIB_FUNCTIONS,
};
use ta_core::statistic::{
    BETABatchRunner, BETAConfig, BETAStream, CORRELBatchRunner, CORRELConfig, CORRELStream,
    LINEARREGBatchRunner, LINEARREGConfig, LINEARREGStream, LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEConfig, LINEARREG_ANGLEStream, LINEARREG_INTERCEPTBatchRunner,
    LINEARREG_INTERCEPTConfig, LINEARREG_INTERCEPTStream, LINEARREG_SLOPEBatchRunner,
    LINEARREG_SLOPEConfig, LINEARREG_SLOPEStream, STDDEVBatchRunner, STDDEVConfig, STDDEVStream,
    TSFBatchRunner, TSFConfig, TSFStream, VARBatchRunner, VARConfig, VARStream,
};

use ta_core::{IndicatorConfig, PreparedBatchRunner, StreamingComputation};

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
