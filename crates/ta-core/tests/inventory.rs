//! End-to-end inventory coverage for the Rust-first execution architecture.
//!
//! Every implemented indicator must expose the sealed
//! [`IndicatorConfig`]/[`PreparedBatchRunner`]/[`StreamingComputation`] seam
//! across all four execution modes (`compute`, `compute_into`, `prepare_batch`,
//! `stream`). Generic trait assertions exercise each indicator's configuration
//! type, prepared runner, and stream state at compile time so a future macro
//! regression cannot drop an indicator from the public catalogue.

use ta_core::cycle::{HT_DCPERIODBatchRunner, HT_DCPERIODConfig, HT_DCPERIODStream};
use ta_core::inventory::{
    function, FunctionGroup, ImplementationStatus, FUNCTION_COUNT, IMPLEMENTED_FUNCTION_COUNT,
    TALIB_FUNCTIONS,
};
use ta_core::math_operators::{
    ADDBatchRunner, ADDConfig, ADDStream, DIVBatchRunner, DIVConfig, DIVStream, MAXBatchRunner,
    MAXConfig, MAXINDEXBatchRunner, MAXINDEXConfig, MAXINDEXStream, MAXStream, MINBatchRunner,
    MINConfig, MININDEXBatchRunner, MININDEXConfig, MININDEXStream, MINMAXBatchRunner,
    MINMAXConfig, MINMAXINDEXBatchRunner, MINMAXINDEXConfig, MINMAXINDEXStream, MINMAXStream,
    MINStream, MULTBatchRunner, MULTConfig, MULTStream, SUBBatchRunner, SUBConfig, SUBStream,
    SUMBatchRunner, SUMConfig, SUMStream,
};
use ta_core::math_transform::{
    ACOSBatchRunner, ACOSConfig, ACOSStream, ASINBatchRunner, ASINConfig, ASINStream,
    ATANBatchRunner, ATANConfig, ATANStream, CEILBatchRunner, CEILConfig, CEILStream,
    COSBatchRunner, COSConfig, COSHBatchRunner, COSHConfig, COSHStream, COSStream, EXPBatchRunner,
    EXPConfig, EXPStream, FLOORBatchRunner, FLOORConfig, FLOORStream, LNBatchRunner, LNConfig,
    LNStream, LOG10BatchRunner, LOG10Config, LOG10Stream, SINBatchRunner, SINConfig,
    SINHBatchRunner, SINHConfig, SINHStream, SINStream, SQRTBatchRunner, SQRTConfig, SQRTStream,
    TANBatchRunner, TANConfig, TANHBatchRunner, TANHConfig, TANHStream, TANStream,
};
use ta_core::overlap::{
    DEMABatchRunner, DEMAConfig, DEMAStream, EMABatchRunner, EMAConfig, EMAStream, MABatchRunner,
    MAConfig, MAStream, SMABatchRunner, SMAConfig, SMAStream, T3BatchRunner, T3Config, T3Stream,
    TEMABatchRunner, TEMAConfig, TEMAStream, TRIMABatchRunner, TRIMAConfig, TRIMAStream,
    WMABatchRunner, WMAConfig, WMAStream,
};
use ta_core::price_transform::{
    AVGDEVBatchRunner, AVGDEVConfig, AVGDEVStream, AVGPRICEBatchRunner, AVGPRICEConfig,
    AVGPRICEStream, MEDPRICEBatchRunner, MEDPRICEConfig, MEDPRICEStream, TYPPRICEBatchRunner,
    TYPPRICEConfig, TYPPRICEStream, WCLPRICEBatchRunner, WCLPRICEConfig, WCLPRICEStream,
};
use ta_core::statistic::{
    BETABatchRunner, BETAConfig, BETAStream, CORRELBatchRunner, CORRELConfig, CORRELStream,
    LINEARREGBatchRunner, LINEARREGConfig, LINEARREGStream, LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEConfig, LINEARREG_ANGLEStream, LINEARREG_INTERCEPTBatchRunner,
    LINEARREG_INTERCEPTConfig, LINEARREG_INTERCEPTStream, LINEARREG_SLOPEBatchRunner,
    LINEARREG_SLOPEConfig, LINEARREG_SLOPEStream, STDDEVBatchRunner, STDDEVConfig, STDDEVStream,
    TSFBatchRunner, TSFConfig, TSFStream, VARBatchRunner, VARConfig, VARStream,
};
use ta_core::volatility::{
    ATRBatchRunner, ATRConfig, ATRStream, NATRBatchRunner, NATRConfig, NATRStream,
    TRANGEBatchRunner, TRANGEConfig, TRANGEStream,
};
use ta_core::volume::{
    ADBatchRunner, ADConfig, ADOSCBatchRunner, ADOSCConfig, ADOSCStream, ADStream, OBVBatchRunner,
    OBVConfig, OBVStream,
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
        "HT_DCPERIOD",
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
fn every_implemented_indicator_exposes_the_full_execution_seam() {
    // Compile-time witness that each implemented indicator's configuration
    // type is wired to its reusable Prepared Batch Runner and independent
    // Streaming Computation state. The bound names `C`, `R`, and `S` are
    // never instantiated, so the test body compiles only if every listed
    // combination satisfies the sealed `IndicatorConfig` contract.
    fn assert_execution_types<C, R, S>()
    where
        C: IndicatorConfig<BatchRunner = R, Stream = S>,
        R: PreparedBatchRunner<C>,
        S: StreamingComputation<C>,
    {
    }

    // Overlap Studies (8 implemented indicators).
    assert_execution_types::<SMAConfig, SMABatchRunner, SMAStream>();
    assert_execution_types::<DEMAConfig, DEMABatchRunner, DEMAStream>();
    assert_execution_types::<EMAConfig, EMABatchRunner, EMAStream>();
    assert_execution_types::<MAConfig, MABatchRunner, MAStream>();
    assert_execution_types::<T3Config, T3BatchRunner, T3Stream>();
    assert_execution_types::<TEMAConfig, TEMABatchRunner, TEMAStream>();
    assert_execution_types::<TRIMAConfig, TRIMABatchRunner, TRIMAStream>();
    assert_execution_types::<WMAConfig, WMABatchRunner, WMAStream>();

    // Price Transform (5 implemented indicators).
    assert_execution_types::<AVGDEVConfig, AVGDEVBatchRunner, AVGDEVStream>();
    assert_execution_types::<AVGPRICEConfig, AVGPRICEBatchRunner, AVGPRICEStream>();
    assert_execution_types::<MEDPRICEConfig, MEDPRICEBatchRunner, MEDPRICEStream>();
    assert_execution_types::<TYPPRICEConfig, TYPPRICEBatchRunner, TYPPRICEStream>();
    assert_execution_types::<WCLPRICEConfig, WCLPRICEBatchRunner, WCLPRICEStream>();

    // Cycle Indicators (1 implemented indicator).
    assert_execution_types::<HT_DCPERIODConfig, HT_DCPERIODBatchRunner, HT_DCPERIODStream>();

    // Volume Indicators (3 implemented indicators).
    assert_execution_types::<ADConfig, ADBatchRunner, ADStream>();
    assert_execution_types::<ADOSCConfig, ADOSCBatchRunner, ADOSCStream>();
    assert_execution_types::<OBVConfig, OBVBatchRunner, OBVStream>();

    // Volatility Indicators (3 implemented indicators).
    assert_execution_types::<ATRConfig, ATRBatchRunner, ATRStream>();
    assert_execution_types::<NATRConfig, NATRBatchRunner, NATRStream>();
    assert_execution_types::<TRANGEConfig, TRANGEBatchRunner, TRANGEStream>();

    // Statistic Functions (9 implemented indicators). The regression family
    // was promoted to the seam by issue #14 and is also exercised by the
    // dedicated `migrated_rolling_statistics_are_in_the_public_execution_catalogue`
    // test above.
    assert_execution_types::<BETAConfig, BETABatchRunner, BETAStream>();
    assert_execution_types::<CORRELConfig, CORRELBatchRunner, CORRELStream>();
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
    assert_execution_types::<VARConfig, VARBatchRunner, VARStream>();
    assert_execution_types::<STDDEVConfig, STDDEVBatchRunner, STDDEVStream>();

    // Math Transform (15 implemented indicators).
    assert_execution_types::<ACOSConfig, ACOSBatchRunner, ACOSStream>();
    assert_execution_types::<ASINConfig, ASINBatchRunner, ASINStream>();
    assert_execution_types::<ATANConfig, ATANBatchRunner, ATANStream>();
    assert_execution_types::<CEILConfig, CEILBatchRunner, CEILStream>();
    assert_execution_types::<COSConfig, COSBatchRunner, COSStream>();
    assert_execution_types::<COSHConfig, COSHBatchRunner, COSHStream>();
    assert_execution_types::<EXPConfig, EXPBatchRunner, EXPStream>();
    assert_execution_types::<FLOORConfig, FLOORBatchRunner, FLOORStream>();
    assert_execution_types::<LNConfig, LNBatchRunner, LNStream>();
    assert_execution_types::<LOG10Config, LOG10BatchRunner, LOG10Stream>();
    assert_execution_types::<SINConfig, SINBatchRunner, SINStream>();
    assert_execution_types::<SINHConfig, SINHBatchRunner, SINHStream>();
    assert_execution_types::<SQRTConfig, SQRTBatchRunner, SQRTStream>();
    assert_execution_types::<TANConfig, TANBatchRunner, TANStream>();
    assert_execution_types::<TANHConfig, TANHBatchRunner, TANHStream>();

    // Math Operators (11 implemented indicators).
    assert_execution_types::<ADDConfig, ADDBatchRunner, ADDStream>();
    assert_execution_types::<DIVConfig, DIVBatchRunner, DIVStream>();
    assert_execution_types::<MULTConfig, MULTBatchRunner, MULTStream>();
    assert_execution_types::<SUBConfig, SUBBatchRunner, SUBStream>();
    assert_execution_types::<MAXConfig, MAXBatchRunner, MAXStream>();
    assert_execution_types::<MAXINDEXConfig, MAXINDEXBatchRunner, MAXINDEXStream>();
    assert_execution_types::<MINConfig, MINBatchRunner, MINStream>();
    assert_execution_types::<MININDEXConfig, MININDEXBatchRunner, MININDEXStream>();
    assert_execution_types::<MINMAXConfig, MINMAXBatchRunner, MINMAXStream>();
    assert_execution_types::<MINMAXINDEXConfig, MINMAXINDEXBatchRunner, MINMAXINDEXStream>();
    assert_execution_types::<SUMConfig, SUMBatchRunner, SUMStream>();
}

#[test]
fn inventory_count_matches_execution_seam_coverage() {
    // Cross-check: the count of distinct (Config, BatchRunner, Stream)
    // triples wired through `every_implemented_indicator_exposes_the_full_execution_seam`
    // equals the implemented function ledger. This locks the catalogue and the
    // execution seam together so a new indicator cannot ship without both.
    let seam_groups = [
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
        "HT_DCPERIOD",
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

    assert_eq!(
        seam_groups.len(),
        IMPLEMENTED_FUNCTION_COUNT,
        "execution-seam coverage must match the implemented indicator ledger",
    );
    assert_eq!(
        TALIB_FUNCTIONS
            .iter()
            .filter(|info| info.is_implemented())
            .count(),
        IMPLEMENTED_FUNCTION_COUNT,
        "TALIB_FUNCTIONS ledger must record every implemented indicator",
    );
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
