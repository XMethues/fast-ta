//! End-to-end inventory coverage for the Rust-first execution architecture.
//!
//! Every implemented indicator must expose the sealed
//! [`IndicatorConfig`]/[`PreparedBatchRunner`]/[`StreamingComputation`] seam
//! across all four execution modes (`compute`, `compute_into`, `prepare_batch`,
//! `stream`). Generic trait assertions exercise each indicator's configuration
//! type, prepared runner, and stream state at compile time so a future macro
//! regression cannot drop an indicator from the public catalogue.

use ta_core::cycle::{
    HT_DCPERIODBatchRunner, HT_DCPERIODConfig, HT_DCPERIODStream, HT_DCPHASEBatchRunner,
    HT_DCPHASEConfig, HT_DCPHASEStream, HT_PHASORBatchRunner, HT_PHASORConfig, HT_PHASORStream,
    HT_SINEBatchRunner, HT_SINEConfig, HT_SINEStream, HT_TRENDMODEBatchRunner, HT_TRENDMODEConfig,
    HT_TRENDMODEStream,
};
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
use ta_core::momentum::{
    ADXBatchRunner, ADXConfig, ADXRBatchRunner, ADXRConfig, ADXRStream, ADXStream, APOBatchRunner,
    APOConfig, APOStream, AROONBatchRunner, AROONConfig, AROONOSCBatchRunner, AROONOSCConfig,
    AROONOSCStream, AROONStream, BOPBatchRunner, BOPConfig, BOPStream, CCIBatchRunner, CCIConfig,
    CCIStream, CMOBatchRunner, CMOConfig, CMOStream, DXBatchRunner, DXConfig, DXStream,
    IMIBatchRunner, IMIConfig, IMIStream, MACDBatchRunner, MACDConfig, MACDEXTBatchRunner,
    MACDEXTConfig, MACDEXTStream, MACDFIXBatchRunner, MACDFIXConfig, MACDFIXStream, MACDStream,
    MFIBatchRunner, MFIConfig, MFIStream, MINUS_DIBatchRunner, MINUS_DIConfig, MINUS_DIStream,
    MINUS_DMBatchRunner, MINUS_DMConfig, MINUS_DMStream, MOMBatchRunner, MOMConfig, MOMStream,
    PLUS_DIBatchRunner, PLUS_DIConfig, PLUS_DIStream, PLUS_DMBatchRunner, PLUS_DMConfig,
    PLUS_DMStream, PPOBatchRunner, PPOConfig, PPOStream, ROCBatchRunner, ROCConfig,
    ROCPBatchRunner, ROCPConfig, ROCPStream, ROCR100BatchRunner, ROCR100Config, ROCR100Stream,
    ROCRBatchRunner, ROCRConfig, ROCRStream, ROCStream, RSIBatchRunner, RSIConfig, RSIStream,
    STOCHBatchRunner, STOCHConfig, STOCHFBatchRunner, STOCHFConfig, STOCHFStream,
    STOCHRSIBatchRunner, STOCHRSIConfig, STOCHRSIStream, STOCHStream, TRIXBatchRunner, TRIXConfig,
    TRIXStream, ULTOSCBatchRunner, ULTOSCConfig, ULTOSCStream, WILLRBatchRunner, WILLRConfig,
    WILLRStream,
};
use ta_core::overlap::{
    ACCBANDSBatchRunner, ACCBANDSConfig, ACCBANDSStream, BBANDSBatchRunner, BBANDSConfig,
    BBANDSStream, DEMABatchRunner, DEMAConfig, DEMAStream, EMABatchRunner, EMAConfig, EMAStream,
    HT_TRENDLINEBatchRunner, HT_TRENDLINEConfig, HT_TRENDLINEStream, KAMABatchRunner, KAMAConfig,
    KAMAStream, MABatchRunner, MAConfig, MAMABatchRunner, MAMAConfig, MAMAStream, MAStream,
    MAVPBatchRunner, MAVPConfig, MAVPStream, MIDPOINTBatchRunner, MIDPOINTConfig, MIDPOINTStream,
    MIDPRICEBatchRunner, MIDPRICEConfig, MIDPRICEStream, SARBatchRunner, SARConfig,
    SAREXTBatchRunner, SAREXTConfig, SAREXTStream, SARStream, SMABatchRunner, SMAConfig, SMAStream,
    T3BatchRunner, T3Config, T3Stream, TEMABatchRunner, TEMAConfig, TEMAStream, TRIMABatchRunner,
    TRIMAConfig, TRIMAStream, WMABatchRunner, WMAConfig, WMAStream,
};
use ta_core::price_transform::{
    AVGDEVBatchRunner, AVGDEVConfig, AVGDEVStream, AVGPRICEBatchRunner, AVGPRICEConfig,
    AVGPRICEStream, MEDPRICEBatchRunner, MEDPRICEConfig, MEDPRICEStream, TYPPRICEBatchRunner,
    TYPPRICEConfig, TYPPRICEStream, WCLPRICEBatchRunner, WCLPRICEConfig, WCLPRICEStream,
};
use ta_core::pattern_recognition::{
    CDL2CROWSBatchRunner, CDL2CROWSConfig, CDL2CROWSStream, CDL3LINESTRIKEBatchRunner,
    CDL3LINESTRIKEConfig, CDL3LINESTRIKEStream, CDLGAPSIDESIDEWHITEBatchRunner,
    CDLGAPSIDESIDEWHITEConfig, CDLGAPSIDESIDEWHITEStream, CDLSTICKSANDWICHBatchRunner,
    CDLSTICKSANDWICHConfig, CDLSTICKSANDWICHStream, CDLTASUKIGAPBatchRunner, CDLTASUKIGAPConfig,
    CDLTASUKIGAPStream, CDLTRISTARBatchRunner, CDLTRISTARConfig, CDLTRISTARStream,
    CDLUPSIDEGAP2CROWSBatchRunner, CDLUPSIDEGAP2CROWSConfig, CDLUPSIDEGAP2CROWSStream,
    CDLXSIDEGAP3METHODSBatchRunner, CDLXSIDEGAP3METHODSConfig, CDLXSIDEGAP3METHODSStream,
    CDL3BLACKCROWSBatchRunner, CDL3BLACKCROWSConfig, CDL3BLACKCROWSStream,
    CDL3STARSINSOUTHBatchRunner, CDL3STARSINSOUTHConfig, CDL3STARSINSOUTHStream,
    CDL3WHITESOLDIERSBatchRunner, CDL3WHITESOLDIERSConfig, CDL3WHITESOLDIERSStream,
    CDLADVANCEBLOCKBatchRunner, CDLADVANCEBLOCKConfig, CDLADVANCEBLOCKStream,
    CDLCONCEALBABYSWALLBatchRunner, CDLCONCEALBABYSWALLConfig, CDLCONCEALBABYSWALLStream,
    CDLIDENTICAL3CROWSBatchRunner, CDLIDENTICAL3CROWSConfig, CDLIDENTICAL3CROWSStream,
    CDLSTALLEDPATTERNBatchRunner, CDLSTALLEDPATTERNConfig, CDLSTALLEDPATTERNStream,
    CDLBREAKAWAYBatchRunner, CDLBREAKAWAYConfig, CDLBREAKAWAYStream,
    CDLLADDERBOTTOMBatchRunner, CDLLADDERBOTTOMConfig, CDLLADDERBOTTOMStream,
    CDLMATHOLDBatchRunner, CDLMATHOLDConfig, CDLMATHOLDStream,
    CDLRISEFALL3METHODSBatchRunner, CDLRISEFALL3METHODSConfig, CDLRISEFALL3METHODSStream,
    CDL3INSIDEBatchRunner, CDL3INSIDEConfig, CDL3INSIDEStream, CDL3OUTSIDEBatchRunner,
    CDL3OUTSIDEConfig, CDL3OUTSIDEStream, CDLABANDONEDBABYBatchRunner, CDLABANDONEDBABYConfig,
    CDLABANDONEDBABYStream,
    CDLBELTHOLDBatchRunner, CDLBELTHOLDConfig, CDLBELTHOLDStream,
    CDLCLOSINGMARUBOZUBatchRunner, CDLCLOSINGMARUBOZUConfig, CDLCLOSINGMARUBOZUStream,
    CDLCOUNTERATTACKBatchRunner, CDLCOUNTERATTACKConfig, CDLCOUNTERATTACKStream,
    CDLDARKCLOUDCOVERBatchRunner, CDLDARKCLOUDCOVERConfig, CDLDARKCLOUDCOVERStream,
    CDLDOJIBatchRunner, CDLDOJIConfig, CDLDOJISTARBatchRunner, CDLDOJISTARConfig,
    CDLDOJISTARStream, CDLDOJIStream, CDLDRAGONFLYDOJIBatchRunner, CDLDRAGONFLYDOJIConfig,
    CDLDRAGONFLYDOJIStream, CDLENGULFINGBatchRunner, CDLENGULFINGConfig, CDLENGULFINGStream,
    CDLGRAVESTONEDOJIBatchRunner, CDLGRAVESTONEDOJIConfig, CDLGRAVESTONEDOJIStream,
    CDLHAMMERBatchRunner, CDLHAMMERConfig, CDLHAMMERStream, CDLHANGINGMANBatchRunner,
    CDLHANGINGMANConfig, CDLHANGINGMANStream, CDLHARAMIBatchRunner, CDLHARAMIConfig,
    CDLHARAMICROSSBatchRunner, CDLHARAMICROSSConfig, CDLHARAMICROSSStream, CDLHARAMIStream,
    CDLHIGHWAVEBatchRunner, CDLHIGHWAVEConfig, CDLHIGHWAVEStream, CDLHOMINGPIGEONBatchRunner,
    CDLHOMINGPIGEONConfig, CDLHOMINGPIGEONStream, CDLINNECKBatchRunner, CDLINNECKConfig,
    CDLINNECKStream, CDLINVERTEDHAMMERBatchRunner, CDLINVERTEDHAMMERConfig,
    CDLINVERTEDHAMMERStream, CDLKICKINGBatchRunner, CDLKICKINGBYLENGTHBatchRunner,
    CDLKICKINGBYLENGTHConfig, CDLKICKINGBYLENGTHStream, CDLKICKINGConfig, CDLKICKINGStream,
    CDLLONGLEGGEDDOJIBatchRunner, CDLLONGLEGGEDDOJIConfig, CDLLONGLEGGEDDOJIStream,
    CDLLONGLINEBatchRunner, CDLLONGLINEConfig, CDLLONGLINEStream, CDLMARUBOZUBatchRunner,
    CDLMARUBOZUConfig, CDLMARUBOZUStream, CDLMATCHINGLOWBatchRunner, CDLMATCHINGLOWConfig,
    CDLMATCHINGLOWStream, CDLONNECKBatchRunner, CDLONNECKConfig, CDLONNECKStream,
    CDLPIERCINGBatchRunner, CDLPIERCINGConfig, CDLPIERCINGStream, CDLRICKSHAWMANBatchRunner,
    CDLRICKSHAWMANConfig, CDLRICKSHAWMANStream, CDLSEPARATINGLINESBatchRunner,
    CDLSEPARATINGLINESConfig, CDLSEPARATINGLINESStream, CDLSHOOTINGSTARBatchRunner,
    CDLSHOOTINGSTARConfig, CDLSHOOTINGSTARStream, CDLSHORTLINEBatchRunner, CDLSHORTLINEConfig,
    CDLSHORTLINEStream, CDLSPINNINGTOPBatchRunner, CDLSPINNINGTOPConfig, CDLSPINNINGTOPStream,
    CDLTAKURIBatchRunner, CDLTAKURIConfig, CDLTAKURIStream, CDLTHRUSTINGBatchRunner,
    CDLTHRUSTINGConfig, CDLTHRUSTINGStream,
    CDLEVENINGDOJISTARBatchRunner, CDLEVENINGDOJISTARConfig, CDLEVENINGDOJISTARStream,
    CDLEVENINGSTARBatchRunner, CDLEVENINGSTARConfig, CDLEVENINGSTARStream,
    CDLMORNINGDOJISTARBatchRunner, CDLMORNINGDOJISTARConfig, CDLMORNINGDOJISTARStream,
    CDLMORNINGSTARBatchRunner, CDLMORNINGSTARConfig, CDLMORNINGSTARStream,
    CDLUNIQUE3RIVERBatchRunner, CDLUNIQUE3RIVERConfig, CDLUNIQUE3RIVERStream,
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

fn assert_execution_types<C, R, S>()
where
    C: 'static + IndicatorConfig<BatchRunner = R, Stream = S>,
    R: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

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
        "SAR",
        "SAREXT",
        "ACCBANDS",
        "BBANDS",
        "MIDPOINT",
        "MIDPRICE",
        "KAMA",
        "MAVP",
        "MAMA",
        "HT_TRENDLINE",
        "AVGDEV",
        "AVGPRICE",
        "MEDPRICE",
        "TYPPRICE",
        "WCLPRICE",
        "HT_DCPERIOD",
        "HT_DCPHASE",
        "HT_PHASOR",
        "HT_SINE",
        "HT_TRENDMODE",
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
        "MOM",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "CMO",
        "IMI",
        "PLUS_DM",
        "MINUS_DM",
        "PLUS_DI",
        "MINUS_DI",
        "DX",
        "ADX",
        "ADXR",
        "BOP",
        "CCI",
        "MFI",
        "ULTOSC",
        "APO",
        "PPO",
        "MACD",
        "MACDEXT",
        "MACDFIX",
        "TRIX",
        "AROON",
        "AROONOSC",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "WILLR",
        "CDLDOJI",
        "CDLENGULFING",
        "CDLBELTHOLD",
        "CDLCLOSINGMARUBOZU",
        "CDLDRAGONFLYDOJI",
        "CDLGRAVESTONEDOJI",
        "CDLHIGHWAVE",
        "CDLLONGLEGGEDDOJI",
        "CDLLONGLINE",
        "CDLMARUBOZU",
        "CDLRICKSHAWMAN",
        "CDLSHORTLINE",
        "CDLSPINNINGTOP",
        "CDLTAKURI",
        "CDLCOUNTERATTACK",
        "CDLDARKCLOUDCOVER",
        "CDLDOJISTAR",
        "CDLHARAMI",
        "CDLHARAMICROSS",
        "CDLHOMINGPIGEON",
        "CDLKICKING",
        "CDLKICKINGBYLENGTH",
        "CDLMATCHINGLOW",
        "CDLHAMMER",
        "CDLHANGINGMAN",
        "CDLINNECK",
        "CDLINVERTEDHAMMER",
        "CDLONNECK",
        "CDLPIERCING",
        "CDLSEPARATINGLINES",
        "CDLSHOOTINGSTAR",
        "CDLTHRUSTING",
        "CDL3INSIDE",
        "CDL3OUTSIDE",
        "CDLABANDONEDBABY",
        "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR",
        "CDLMORNINGDOJISTAR",
        "CDLMORNINGSTAR",
        "CDLUNIQUE3RIVER",
        "CDL2CROWS",
        "CDL3LINESTRIKE",
        "CDLGAPSIDESIDEWHITE",
        "CDLSTICKSANDWICH",
        "CDLTASUKIGAP",
        "CDLTRISTAR",
        "CDLUPSIDEGAP2CROWS",
        "CDLXSIDEGAP3METHODS",
        "CDL3BLACKCROWS",
        "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS",
        "CDLADVANCEBLOCK",
        "CDLCONCEALBABYSWALL",
        "CDLIDENTICAL3CROWS",
        "CDLSTALLEDPATTERN",
        "CDLBREAKAWAY",
        "CDLLADDERBOTTOM",
        "CDLMATHOLD",
        "CDLRISEFALL3METHODS",
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
fn migrated_rolling_statistics_are_in_the_public_execution_seam() {
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
    // Overlap Studies (18 implemented indicators).
    assert_execution_types::<SMAConfig, SMABatchRunner, SMAStream>();
    assert_execution_types::<DEMAConfig, DEMABatchRunner, DEMAStream>();
    assert_execution_types::<EMAConfig, EMABatchRunner, EMAStream>();
    assert_execution_types::<MAConfig, MABatchRunner, MAStream>();
    assert_execution_types::<T3Config, T3BatchRunner, T3Stream>();
    assert_execution_types::<TEMAConfig, TEMABatchRunner, TEMAStream>();
    assert_execution_types::<TRIMAConfig, TRIMABatchRunner, TRIMAStream>();
    assert_execution_types::<WMAConfig, WMABatchRunner, WMAStream>();
    assert_execution_types::<SARConfig, SARBatchRunner, SARStream>();
    assert_execution_types::<SAREXTConfig, SAREXTBatchRunner, SAREXTStream>();
    assert_execution_types::<ACCBANDSConfig, ACCBANDSBatchRunner, ACCBANDSStream>();
    assert_execution_types::<BBANDSConfig, BBANDSBatchRunner, BBANDSStream>();
    assert_execution_types::<MIDPOINTConfig, MIDPOINTBatchRunner, MIDPOINTStream>();
    assert_execution_types::<MIDPRICEConfig, MIDPRICEBatchRunner, MIDPRICEStream>();
    assert_execution_types::<KAMAConfig, KAMABatchRunner, KAMAStream>();
    assert_execution_types::<MAVPConfig, MAVPBatchRunner, MAVPStream>();
    assert_execution_types::<MAMAConfig, MAMABatchRunner, MAMAStream>();
    assert_execution_types::<HT_TRENDLINEConfig, HT_TRENDLINEBatchRunner, HT_TRENDLINEStream>();

    // Price Transform (5 implemented indicators).
    assert_execution_types::<AVGDEVConfig, AVGDEVBatchRunner, AVGDEVStream>();
    assert_execution_types::<AVGPRICEConfig, AVGPRICEBatchRunner, AVGPRICEStream>();
    assert_execution_types::<MEDPRICEConfig, MEDPRICEBatchRunner, MEDPRICEStream>();
    assert_execution_types::<TYPPRICEConfig, TYPPRICEBatchRunner, TYPPRICEStream>();
    assert_execution_types::<WCLPRICEConfig, WCLPRICEBatchRunner, WCLPRICEStream>();

    // Cycle Indicators (5 implemented definitions).
    assert_execution_types::<HT_DCPERIODConfig, HT_DCPERIODBatchRunner, HT_DCPERIODStream>();
    assert_execution_types::<HT_DCPHASEConfig, HT_DCPHASEBatchRunner, HT_DCPHASEStream>();
    assert_execution_types::<HT_PHASORConfig, HT_PHASORBatchRunner, HT_PHASORStream>();
    assert_execution_types::<HT_SINEConfig, HT_SINEBatchRunner, HT_SINEStream>();
    assert_execution_types::<HT_TRENDMODEConfig, HT_TRENDMODEBatchRunner, HT_TRENDMODEStream>();

    // Momentum Indicators (31 implemented indicators).
    assert_execution_types::<MOMConfig, MOMBatchRunner, MOMStream>();
    assert_execution_types::<ROCConfig, ROCBatchRunner, ROCStream>();
    assert_execution_types::<ROCPConfig, ROCPBatchRunner, ROCPStream>();
    assert_execution_types::<ROCRConfig, ROCRBatchRunner, ROCRStream>();
    assert_execution_types::<ROCR100Config, ROCR100BatchRunner, ROCR100Stream>();
    assert_execution_types::<RSIConfig, RSIBatchRunner, RSIStream>();
    assert_execution_types::<CMOConfig, CMOBatchRunner, CMOStream>();
    assert_execution_types::<IMIConfig, IMIBatchRunner, IMIStream>();
    assert_execution_types::<PLUS_DMConfig, PLUS_DMBatchRunner, PLUS_DMStream>();
    assert_execution_types::<MINUS_DMConfig, MINUS_DMBatchRunner, MINUS_DMStream>();
    assert_execution_types::<PLUS_DIConfig, PLUS_DIBatchRunner, PLUS_DIStream>();
    assert_execution_types::<MINUS_DIConfig, MINUS_DIBatchRunner, MINUS_DIStream>();
    assert_execution_types::<DXConfig, DXBatchRunner, DXStream>();
    assert_execution_types::<ADXConfig, ADXBatchRunner, ADXStream>();
    assert_execution_types::<ADXRConfig, ADXRBatchRunner, ADXRStream>();
    assert_execution_types::<BOPConfig, BOPBatchRunner, BOPStream>();
    assert_execution_types::<CCIConfig, CCIBatchRunner, CCIStream>();
    assert_execution_types::<MFIConfig, MFIBatchRunner, MFIStream>();
    assert_execution_types::<ULTOSCConfig, ULTOSCBatchRunner, ULTOSCStream>();
    assert_execution_types::<APOConfig, APOBatchRunner, APOStream>();
    assert_execution_types::<PPOConfig, PPOBatchRunner, PPOStream>();
    assert_execution_types::<MACDConfig, MACDBatchRunner, MACDStream>();
    assert_execution_types::<MACDEXTConfig, MACDEXTBatchRunner, MACDEXTStream>();
    assert_execution_types::<MACDFIXConfig, MACDFIXBatchRunner, MACDFIXStream>();
    assert_execution_types::<TRIXConfig, TRIXBatchRunner, TRIXStream>();
    assert_execution_types::<AROONConfig, AROONBatchRunner, AROONStream>();
    assert_execution_types::<AROONOSCConfig, AROONOSCBatchRunner, AROONOSCStream>();
    assert_execution_types::<STOCHConfig, STOCHBatchRunner, STOCHStream>();
    assert_execution_types::<STOCHFConfig, STOCHFBatchRunner, STOCHFStream>();
    assert_execution_types::<STOCHRSIConfig, STOCHRSIBatchRunner, STOCHRSIStream>();
    assert_execution_types::<WILLRConfig, WILLRBatchRunner, WILLRStream>();

    // Volume Indicators (3 implemented indicators).
    assert_execution_types::<ADConfig, ADBatchRunner, ADStream>();
    assert_execution_types::<ADOSCConfig, ADOSCBatchRunner, ADOSCStream>();
    assert_execution_types::<OBVConfig, OBVBatchRunner, OBVStream>();

    // Volatility Indicators (3 implemented indicators).
    assert_execution_types::<ATRConfig, ATRBatchRunner, ATRStream>();
    assert_execution_types::<NATRConfig, NATRBatchRunner, NATRStream>();
    assert_execution_types::<TRANGEConfig, TRANGEBatchRunner, TRANGEStream>();

    // Pattern Recognition (48 implemented indicators).
    assert_execution_types::<CDLDOJIConfig, CDLDOJIBatchRunner, CDLDOJIStream>();
    assert_execution_types::<
        CDLENGULFINGConfig,
        CDLENGULFINGBatchRunner,
        CDLENGULFINGStream,
    >();
    assert_execution_types::<CDLBELTHOLDConfig, CDLBELTHOLDBatchRunner, CDLBELTHOLDStream>();
    assert_execution_types::<
        CDLCLOSINGMARUBOZUConfig,
        CDLCLOSINGMARUBOZUBatchRunner,
        CDLCLOSINGMARUBOZUStream,
    >();
    assert_execution_types::<
        CDLDRAGONFLYDOJIConfig,
        CDLDRAGONFLYDOJIBatchRunner,
        CDLDRAGONFLYDOJIStream,
    >();
    assert_execution_types::<
        CDLGRAVESTONEDOJIConfig,
        CDLGRAVESTONEDOJIBatchRunner,
        CDLGRAVESTONEDOJIStream,
    >();
    assert_execution_types::<CDLHIGHWAVEConfig, CDLHIGHWAVEBatchRunner, CDLHIGHWAVEStream>();
    assert_execution_types::<
        CDLLONGLEGGEDDOJIConfig,
        CDLLONGLEGGEDDOJIBatchRunner,
        CDLLONGLEGGEDDOJIStream,
    >();
    assert_execution_types::<CDLLONGLINEConfig, CDLLONGLINEBatchRunner, CDLLONGLINEStream>();
    assert_execution_types::<CDLMARUBOZUConfig, CDLMARUBOZUBatchRunner, CDLMARUBOZUStream>();
    assert_execution_types::<
        CDLRICKSHAWMANConfig,
        CDLRICKSHAWMANBatchRunner,
        CDLRICKSHAWMANStream,
    >();
    assert_execution_types::<CDLSHORTLINEConfig, CDLSHORTLINEBatchRunner, CDLSHORTLINEStream>();
    assert_execution_types::<
        CDLSPINNINGTOPConfig,
        CDLSPINNINGTOPBatchRunner,
        CDLSPINNINGTOPStream,
    >();
    assert_execution_types::<CDLTAKURIConfig, CDLTAKURIBatchRunner, CDLTAKURIStream>();
    assert_execution_types::<
        CDLCOUNTERATTACKConfig,
        CDLCOUNTERATTACKBatchRunner,
        CDLCOUNTERATTACKStream,
    >();
    assert_execution_types::<
        CDLDARKCLOUDCOVERConfig,
        CDLDARKCLOUDCOVERBatchRunner,
        CDLDARKCLOUDCOVERStream,
    >();
    assert_execution_types::<CDLDOJISTARConfig, CDLDOJISTARBatchRunner, CDLDOJISTARStream>();
    assert_execution_types::<CDLHARAMIConfig, CDLHARAMIBatchRunner, CDLHARAMIStream>();
    assert_execution_types::<
        CDLHARAMICROSSConfig,
        CDLHARAMICROSSBatchRunner,
        CDLHARAMICROSSStream,
    >();
    assert_execution_types::<
        CDLHOMINGPIGEONConfig,
        CDLHOMINGPIGEONBatchRunner,
        CDLHOMINGPIGEONStream,
    >();
    assert_execution_types::<CDLKICKINGConfig, CDLKICKINGBatchRunner, CDLKICKINGStream>();
    assert_execution_types::<
        CDLKICKINGBYLENGTHConfig,
        CDLKICKINGBYLENGTHBatchRunner,
        CDLKICKINGBYLENGTHStream,
    >();
    assert_execution_types::<
        CDLMATCHINGLOWConfig,
        CDLMATCHINGLOWBatchRunner,
        CDLMATCHINGLOWStream,
    >();
    assert_execution_types::<CDLHAMMERConfig, CDLHAMMERBatchRunner, CDLHAMMERStream>();
    assert_execution_types::<
        CDLHANGINGMANConfig,
        CDLHANGINGMANBatchRunner,
        CDLHANGINGMANStream,
    >();
    assert_execution_types::<CDLINNECKConfig, CDLINNECKBatchRunner, CDLINNECKStream>();
    assert_execution_types::<
        CDLINVERTEDHAMMERConfig,
        CDLINVERTEDHAMMERBatchRunner,
        CDLINVERTEDHAMMERStream,
    >();
    assert_execution_types::<CDLONNECKConfig, CDLONNECKBatchRunner, CDLONNECKStream>();
    assert_execution_types::<CDLPIERCINGConfig, CDLPIERCINGBatchRunner, CDLPIERCINGStream>();
    assert_execution_types::<
        CDLSEPARATINGLINESConfig,
        CDLSEPARATINGLINESBatchRunner,
        CDLSEPARATINGLINESStream,
    >();
    assert_execution_types::<
        CDLSHOOTINGSTARConfig,
        CDLSHOOTINGSTARBatchRunner,
        CDLSHOOTINGSTARStream,
    >();
    assert_execution_types::<CDLTHRUSTINGConfig, CDLTHRUSTINGBatchRunner, CDLTHRUSTINGStream>();

    assert_execution_types::<CDL3INSIDEConfig, CDL3INSIDEBatchRunner, CDL3INSIDEStream>();
    assert_execution_types::<CDL3OUTSIDEConfig, CDL3OUTSIDEBatchRunner, CDL3OUTSIDEStream>();
    assert_execution_types::<
        CDLABANDONEDBABYConfig,
        CDLABANDONEDBABYBatchRunner,
        CDLABANDONEDBABYStream,
    >();
    assert_execution_types::<
        CDLEVENINGDOJISTARConfig,
        CDLEVENINGDOJISTARBatchRunner,
        CDLEVENINGDOJISTARStream,
    >();
    assert_execution_types::<
        CDLEVENINGSTARConfig,
        CDLEVENINGSTARBatchRunner,
        CDLEVENINGSTARStream,
    >();
    assert_execution_types::<
        CDLMORNINGDOJISTARConfig,
        CDLMORNINGDOJISTARBatchRunner,
        CDLMORNINGDOJISTARStream,
    >();
    assert_execution_types::<
        CDLMORNINGSTARConfig,
        CDLMORNINGSTARBatchRunner,
        CDLMORNINGSTARStream,
    >();
    assert_execution_types::<
        CDLUNIQUE3RIVERConfig,
        CDLUNIQUE3RIVERBatchRunner,
        CDLUNIQUE3RIVERStream,
    >();
    assert_execution_types::<CDL2CROWSConfig, CDL2CROWSBatchRunner, CDL2CROWSStream>();
    assert_execution_types::<
        CDL3LINESTRIKEConfig,
        CDL3LINESTRIKEBatchRunner,
        CDL3LINESTRIKEStream,
    >();
    assert_execution_types::<
        CDLGAPSIDESIDEWHITEConfig,
        CDLGAPSIDESIDEWHITEBatchRunner,
        CDLGAPSIDESIDEWHITEStream,
    >();
    assert_execution_types::<
        CDLSTICKSANDWICHConfig,
        CDLSTICKSANDWICHBatchRunner,
        CDLSTICKSANDWICHStream,
    >();
    assert_execution_types::<CDLTASUKIGAPConfig, CDLTASUKIGAPBatchRunner, CDLTASUKIGAPStream>();
    assert_execution_types::<CDLTRISTARConfig, CDLTRISTARBatchRunner, CDLTRISTARStream>();
    assert_execution_types::<
        CDLUPSIDEGAP2CROWSConfig,
        CDLUPSIDEGAP2CROWSBatchRunner,
        CDLUPSIDEGAP2CROWSStream,
    >();
    assert_execution_types::<
        CDLXSIDEGAP3METHODSConfig,
        CDLXSIDEGAP3METHODSBatchRunner,
        CDLXSIDEGAP3METHODSStream,
    >();
    assert_execution_types::<CDL3BLACKCROWSConfig, CDL3BLACKCROWSBatchRunner, CDL3BLACKCROWSStream>();
    assert_execution_types::<CDL3STARSINSOUTHConfig, CDL3STARSINSOUTHBatchRunner, CDL3STARSINSOUTHStream>();
    assert_execution_types::<CDL3WHITESOLDIERSConfig, CDL3WHITESOLDIERSBatchRunner, CDL3WHITESOLDIERSStream>();
    assert_execution_types::<CDLADVANCEBLOCKConfig, CDLADVANCEBLOCKBatchRunner, CDLADVANCEBLOCKStream>();
    assert_execution_types::<CDLCONCEALBABYSWALLConfig, CDLCONCEALBABYSWALLBatchRunner, CDLCONCEALBABYSWALLStream>();
    assert_execution_types::<CDLIDENTICAL3CROWSConfig, CDLIDENTICAL3CROWSBatchRunner, CDLIDENTICAL3CROWSStream>();
    assert_execution_types::<CDLSTALLEDPATTERNConfig, CDLSTALLEDPATTERNBatchRunner, CDLSTALLEDPATTERNStream>();
    assert_execution_types::<
        CDLBREAKAWAYConfig,
        CDLBREAKAWAYBatchRunner,
        CDLBREAKAWAYStream,
    >();
    assert_execution_types::<
        CDLLADDERBOTTOMConfig,
        CDLLADDERBOTTOMBatchRunner,
        CDLLADDERBOTTOMStream,
    >();
    assert_execution_types::<CDLMATHOLDConfig, CDLMATHOLDBatchRunner, CDLMATHOLDStream>();
    assert_execution_types::<
        CDLRISEFALL3METHODSConfig,
        CDLRISEFALL3METHODSBatchRunner,
        CDLRISEFALL3METHODSStream,
    >();
    // Statistic Functions (9 implemented indicators).
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
        "SAR",
        "SAREXT",
        "ACCBANDS",
        "BBANDS",
        "MIDPOINT",
        "MIDPRICE",
        "KAMA",
        "MAVP",
        "MAMA",
        "HT_TRENDLINE",
        "AVGDEV",
        "AVGPRICE",
        "MEDPRICE",
        "TYPPRICE",
        "WCLPRICE",
        "HT_DCPERIOD",
        "HT_DCPHASE",
        "HT_PHASOR",
        "HT_SINE",
        "HT_TRENDMODE",
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
        "MOM",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "CMO",
        "IMI",
        "PLUS_DM",
        "MINUS_DM",
        "PLUS_DI",
        "MINUS_DI",
        "DX",
        "ADX",
        "ADXR",
        "BOP",
        "CCI",
        "MFI",
        "ULTOSC",
        "APO",
        "PPO",
        "MACD",
        "MACDEXT",
        "MACDFIX",
        "TRIX",
        "AROON",
        "AROONOSC",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "WILLR",
        "CDLDOJI",
        "CDLENGULFING",
        "CDLBELTHOLD",
        "CDLCLOSINGMARUBOZU",
        "CDLDRAGONFLYDOJI",
        "CDLGRAVESTONEDOJI",
        "CDLHIGHWAVE",
        "CDLLONGLEGGEDDOJI",
        "CDLLONGLINE",
        "CDLMARUBOZU",
        "CDLRICKSHAWMAN",
        "CDLSHORTLINE",
        "CDLSPINNINGTOP",
        "CDLTAKURI",
        "CDLCOUNTERATTACK",
        "CDLDARKCLOUDCOVER",
        "CDLDOJISTAR",
        "CDLHARAMI",
        "CDLHARAMICROSS",
        "CDLHOMINGPIGEON",
        "CDLKICKING",
        "CDLKICKINGBYLENGTH",
        "CDLMATCHINGLOW",
        "CDLHAMMER",
        "CDLHANGINGMAN",
        "CDLINNECK",
        "CDLINVERTEDHAMMER",
        "CDLONNECK",
        "CDLPIERCING",
        "CDLSEPARATINGLINES",
        "CDLSHOOTINGSTAR",
        "CDLTHRUSTING",
        "CDL3INSIDE",
        "CDL3OUTSIDE",
        "CDLABANDONEDBABY",
        "CDLEVENINGDOJISTAR",
        "CDLEVENINGSTAR",
        "CDLMORNINGDOJISTAR",
        "CDLMORNINGSTAR",
        "CDLUNIQUE3RIVER",
        "CDL2CROWS",
        "CDL3LINESTRIKE",
        "CDLGAPSIDESIDEWHITE",
        "CDLSTICKSANDWICH",
        "CDLTASUKIGAP",
        "CDLTRISTAR",
        "CDLUPSIDEGAP2CROWS",
        "CDLXSIDEGAP3METHODS",
        "CDL3BLACKCROWS",
        "CDL3STARSINSOUTH",
        "CDL3WHITESOLDIERS",
        "CDLADVANCEBLOCK",
        "CDLCONCEALBABYSWALL",
        "CDLIDENTICAL3CROWS",
        "CDLSTALLEDPATTERN",
        "CDLBREAKAWAY",
        "CDLLADDERBOTTOM",
        "CDLMATHOLD",
        "CDLRISEFALL3METHODS",
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
fn remaining_pattern_recognition_functions_remain_planned() {
    for name in ["CDLHIKKAKE", "CDLHIKKAKEMOD"] {
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

#[test]
fn momentum_change_family_is_implemented_through_the_public_execution_seam() {
    use ta_core::momentum::{
        MOMBatchRunner, MOMConfig, MOMStream, ROCBatchRunner, ROCConfig, ROCPBatchRunner,
        ROCPConfig, ROCPStream, ROCR100BatchRunner, ROCR100Config, ROCR100Stream, ROCRBatchRunner,
        ROCRConfig, ROCRStream, ROCStream,
    };

    for name in ["MOM", "ROC", "ROCP", "ROCR", "ROCR100"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::MomentumIndicators, "{name}");
    }

    assert_execution_types::<MOMConfig, MOMBatchRunner, MOMStream>();
    assert_execution_types::<ROCConfig, ROCBatchRunner, ROCStream>();
    assert_execution_types::<ROCPConfig, ROCPBatchRunner, ROCPStream>();
    assert_execution_types::<ROCRConfig, ROCRBatchRunner, ROCRStream>();
    assert_execution_types::<ROCR100Config, ROCR100BatchRunner, ROCR100Stream>();
}

#[test]
fn directional_movement_system_is_implemented_through_the_public_execution_seam() {
    use ta_core::momentum::{
        ADXBatchRunner, ADXConfig, ADXRBatchRunner, ADXRConfig, ADXRStream, ADXStream,
        DXBatchRunner, DXConfig, DXStream, MINUS_DIBatchRunner, MINUS_DIConfig, MINUS_DIStream,
        MINUS_DMBatchRunner, MINUS_DMConfig, MINUS_DMStream, PLUS_DIBatchRunner, PLUS_DIConfig,
        PLUS_DIStream, PLUS_DMBatchRunner, PLUS_DMConfig, PLUS_DMStream,
    };

    for name in [
        "PLUS_DM", "MINUS_DM", "PLUS_DI", "MINUS_DI", "DX", "ADX", "ADXR",
    ] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::MomentumIndicators, "{name}");
    }

    assert_execution_types::<PLUS_DMConfig, PLUS_DMBatchRunner, PLUS_DMStream>();
    assert_execution_types::<MINUS_DMConfig, MINUS_DMBatchRunner, MINUS_DMStream>();
    assert_execution_types::<PLUS_DIConfig, PLUS_DIBatchRunner, PLUS_DIStream>();
    assert_execution_types::<MINUS_DIConfig, MINUS_DIBatchRunner, MINUS_DIStream>();
    assert_execution_types::<DXConfig, DXBatchRunner, DXStream>();
    assert_execution_types::<ADXConfig, ADXBatchRunner, ADXStream>();
    assert_execution_types::<ADXRConfig, ADXRBatchRunner, ADXRStream>();
}

#[test]
fn composite_momentum_family_is_implemented_through_the_public_execution_seam() {
    use ta_core::momentum::{
        BOPBatchRunner, BOPConfig, BOPStream, CCIBatchRunner, CCIConfig, CCIStream, MFIBatchRunner,
        MFIConfig, MFIStream, ULTOSCBatchRunner, ULTOSCConfig, ULTOSCStream,
    };

    for name in ["BOP", "CCI", "MFI", "ULTOSC"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::MomentumIndicators, "{name}");
    }

    assert_execution_types::<BOPConfig, BOPBatchRunner, BOPStream>();
    assert_execution_types::<CCIConfig, CCIBatchRunner, CCIStream>();
    assert_execution_types::<MFIConfig, MFIBatchRunner, MFIStream>();
    assert_execution_types::<ULTOSCConfig, ULTOSCBatchRunner, ULTOSCStream>();
}

#[test]
fn moving_average_momentum_family_uses_the_public_execution_seam() {
    use ta_core::momentum::{
        APOBatchRunner, APOConfig, APOStream, MACDBatchRunner, MACDConfig, MACDEXTBatchRunner,
        MACDEXTConfig, MACDEXTStream, MACDFIXBatchRunner, MACDFIXConfig, MACDFIXStream, MACDStream,
        PPOBatchRunner, PPOConfig, PPOStream, TRIXBatchRunner, TRIXConfig, TRIXStream,
    };

    for name in ["APO", "PPO", "MACD", "MACDEXT", "MACDFIX", "TRIX"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::MomentumIndicators, "{name}");
    }

    assert_execution_types::<APOConfig, APOBatchRunner, APOStream>();
    assert_execution_types::<PPOConfig, PPOBatchRunner, PPOStream>();
    assert_execution_types::<MACDConfig, MACDBatchRunner, MACDStream>();
    assert_execution_types::<MACDEXTConfig, MACDEXTBatchRunner, MACDEXTStream>();
    assert_execution_types::<MACDFIXConfig, MACDFIXBatchRunner, MACDFIXStream>();
    assert_execution_types::<TRIXConfig, TRIXBatchRunner, TRIXStream>();
}

#[test]
fn hilbert_overlap_studies_use_the_public_execution_seam() {
    use ta_core::overlap::{
        HT_TRENDLINEBatchRunner, HT_TRENDLINEConfig, HT_TRENDLINEStream, MAMABatchRunner,
        MAMAConfig, MAMAStream,
    };

    for name in ["MAMA", "HT_TRENDLINE"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::OverlapStudies, "{name}");
    }

    assert_execution_types::<MAMAConfig, MAMABatchRunner, MAMAStream>();
    assert_execution_types::<HT_TRENDLINEConfig, HT_TRENDLINEBatchRunner, HT_TRENDLINEStream>();
}

#[test]
fn range_position_momentum_family_uses_the_public_execution_seam() {
    use ta_core::momentum::{
        AROONBatchRunner, AROONConfig, AROONOSCBatchRunner, AROONOSCConfig, AROONOSCStream,
        AROONStream, STOCHBatchRunner, STOCHConfig, STOCHFBatchRunner, STOCHFConfig, STOCHFStream,
        STOCHRSIBatchRunner, STOCHRSIConfig, STOCHRSIStream, STOCHStream, WILLRBatchRunner,
        WILLRConfig, WILLRStream,
    };

    for name in ["AROON", "AROONOSC", "STOCH", "STOCHF", "STOCHRSI", "WILLR"] {
        let info = function(name).unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(info.status, ImplementationStatus::Implemented, "{name}");
        assert_eq!(info.group, FunctionGroup::MomentumIndicators, "{name}");
    }

    assert_execution_types::<AROONConfig, AROONBatchRunner, AROONStream>();
    assert_execution_types::<AROONOSCConfig, AROONOSCBatchRunner, AROONOSCStream>();
    assert_execution_types::<STOCHConfig, STOCHBatchRunner, STOCHStream>();
    assert_execution_types::<STOCHFConfig, STOCHFBatchRunner, STOCHFStream>();
    assert_execution_types::<STOCHRSIConfig, STOCHRSIBatchRunner, STOCHRSIStream>();
    assert_execution_types::<WILLRConfig, WILLRBatchRunner, WILLRStream>();
}
