//! Public-seam coverage for the Indicator Catalogue and execution architecture.
//!
//! Inventory tests exercise allocation-free catalogue queries and their
//! invariants without maintaining a second definition registry. Compile-time
//! trait assertions retain coverage of each implemented Indicator Definition's
//! [`IndicatorConfig`]/[`PreparedBatchRunner`]/[`StreamingComputation`] seam
//! across all four execution modes (`compute`, `compute_into`, `prepare_batch`,
//! `stream`).

use fast_ta::cycle::{
    HT_DCPERIODBatchRunner, HT_DCPERIODConfig, HT_DCPERIODStream, HT_DCPHASEBatchRunner,
    HT_DCPHASEConfig, HT_DCPHASEStream, HT_PHASORBatchRunner, HT_PHASORConfig, HT_PHASORStream,
    HT_SINEBatchRunner, HT_SINEConfig, HT_SINEStream, HT_TRENDMODEBatchRunner, HT_TRENDMODEConfig,
    HT_TRENDMODEStream,
};
use fast_ta::inventory::{
    function, FunctionGroup, ImplementationStatus, FUNCTION_COUNT, IMPLEMENTED_FUNCTION_COUNT,
    INDICATOR_CATALOGUE,
};
use fast_ta::math_operators::{
    ADDBatchRunner, ADDConfig, ADDStream, DIVBatchRunner, DIVConfig, DIVStream, MAXBatchRunner,
    MAXConfig, MAXINDEXBatchRunner, MAXINDEXConfig, MAXINDEXStream, MAXStream, MINBatchRunner,
    MINConfig, MININDEXBatchRunner, MININDEXConfig, MININDEXStream, MINMAXBatchRunner,
    MINMAXConfig, MINMAXINDEXBatchRunner, MINMAXINDEXConfig, MINMAXINDEXStream, MINMAXStream,
    MINStream, MULTBatchRunner, MULTConfig, MULTStream, SUBBatchRunner, SUBConfig, SUBStream,
    SUMBatchRunner, SUMConfig, SUMStream,
};
use fast_ta::math_transform::{
    ACOSBatchRunner, ACOSConfig, ACOSStream, ASINBatchRunner, ASINConfig, ASINStream,
    ATANBatchRunner, ATANConfig, ATANStream, CEILBatchRunner, CEILConfig, CEILStream,
    COSBatchRunner, COSConfig, COSHBatchRunner, COSHConfig, COSHStream, COSStream, EXPBatchRunner,
    EXPConfig, EXPStream, FLOORBatchRunner, FLOORConfig, FLOORStream, LNBatchRunner, LNConfig,
    LNStream, LOG10BatchRunner, LOG10Config, LOG10Stream, SINBatchRunner, SINConfig,
    SINHBatchRunner, SINHConfig, SINHStream, SINStream, SQRTBatchRunner, SQRTConfig, SQRTStream,
    TANBatchRunner, TANConfig, TANHBatchRunner, TANHConfig, TANHStream, TANStream,
};
use fast_ta::momentum::{
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
use fast_ta::overlap::{
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
use fast_ta::pattern_recognition::{
    CDL2CROWSBatchRunner, CDL2CROWSConfig, CDL2CROWSStream, CDL3BLACKCROWSBatchRunner,
    CDL3BLACKCROWSConfig, CDL3BLACKCROWSStream, CDL3INSIDEBatchRunner, CDL3INSIDEConfig,
    CDL3INSIDEStream, CDL3LINESTRIKEBatchRunner, CDL3LINESTRIKEConfig, CDL3LINESTRIKEStream,
    CDL3OUTSIDEBatchRunner, CDL3OUTSIDEConfig, CDL3OUTSIDEStream, CDL3STARSINSOUTHBatchRunner,
    CDL3STARSINSOUTHConfig, CDL3STARSINSOUTHStream, CDL3WHITESOLDIERSBatchRunner,
    CDL3WHITESOLDIERSConfig, CDL3WHITESOLDIERSStream, CDLABANDONEDBABYBatchRunner,
    CDLABANDONEDBABYConfig, CDLABANDONEDBABYStream, CDLADVANCEBLOCKBatchRunner,
    CDLADVANCEBLOCKConfig, CDLADVANCEBLOCKStream, CDLBELTHOLDBatchRunner, CDLBELTHOLDConfig,
    CDLBELTHOLDStream, CDLBREAKAWAYBatchRunner, CDLBREAKAWAYConfig, CDLBREAKAWAYStream,
    CDLCLOSINGMARUBOZUBatchRunner, CDLCLOSINGMARUBOZUConfig, CDLCLOSINGMARUBOZUStream,
    CDLCONCEALBABYSWALLBatchRunner, CDLCONCEALBABYSWALLConfig, CDLCONCEALBABYSWALLStream,
    CDLCOUNTERATTACKBatchRunner, CDLCOUNTERATTACKConfig, CDLCOUNTERATTACKStream,
    CDLDARKCLOUDCOVERBatchRunner, CDLDARKCLOUDCOVERConfig, CDLDARKCLOUDCOVERStream,
    CDLDOJIBatchRunner, CDLDOJIConfig, CDLDOJISTARBatchRunner, CDLDOJISTARConfig,
    CDLDOJISTARStream, CDLDOJIStream, CDLDRAGONFLYDOJIBatchRunner, CDLDRAGONFLYDOJIConfig,
    CDLDRAGONFLYDOJIStream, CDLENGULFINGBatchRunner, CDLENGULFINGConfig, CDLENGULFINGStream,
    CDLEVENINGDOJISTARBatchRunner, CDLEVENINGDOJISTARConfig, CDLEVENINGDOJISTARStream,
    CDLEVENINGSTARBatchRunner, CDLEVENINGSTARConfig, CDLEVENINGSTARStream,
    CDLGAPSIDESIDEWHITEBatchRunner, CDLGAPSIDESIDEWHITEConfig, CDLGAPSIDESIDEWHITEStream,
    CDLGRAVESTONEDOJIBatchRunner, CDLGRAVESTONEDOJIConfig, CDLGRAVESTONEDOJIStream,
    CDLHAMMERBatchRunner, CDLHAMMERConfig, CDLHAMMERStream, CDLHANGINGMANBatchRunner,
    CDLHANGINGMANConfig, CDLHANGINGMANStream, CDLHARAMIBatchRunner, CDLHARAMICROSSBatchRunner,
    CDLHARAMICROSSConfig, CDLHARAMICROSSStream, CDLHARAMIConfig, CDLHARAMIStream,
    CDLHIGHWAVEBatchRunner, CDLHIGHWAVEConfig, CDLHIGHWAVEStream, CDLHIKKAKEBatchRunner,
    CDLHIKKAKEConfig, CDLHIKKAKEMODBatchRunner, CDLHIKKAKEMODConfig, CDLHIKKAKEMODStream,
    CDLHIKKAKEStream, CDLHOMINGPIGEONBatchRunner, CDLHOMINGPIGEONConfig, CDLHOMINGPIGEONStream,
    CDLIDENTICAL3CROWSBatchRunner, CDLIDENTICAL3CROWSConfig, CDLIDENTICAL3CROWSStream,
    CDLINNECKBatchRunner, CDLINNECKConfig, CDLINNECKStream, CDLINVERTEDHAMMERBatchRunner,
    CDLINVERTEDHAMMERConfig, CDLINVERTEDHAMMERStream, CDLKICKINGBYLENGTHBatchRunner,
    CDLKICKINGBYLENGTHConfig, CDLKICKINGBYLENGTHStream, CDLKICKINGBatchRunner, CDLKICKINGConfig,
    CDLKICKINGStream, CDLLADDERBOTTOMBatchRunner, CDLLADDERBOTTOMConfig, CDLLADDERBOTTOMStream,
    CDLLONGLEGGEDDOJIBatchRunner, CDLLONGLEGGEDDOJIConfig, CDLLONGLEGGEDDOJIStream,
    CDLLONGLINEBatchRunner, CDLLONGLINEConfig, CDLLONGLINEStream, CDLMARUBOZUBatchRunner,
    CDLMARUBOZUConfig, CDLMARUBOZUStream, CDLMATCHINGLOWBatchRunner, CDLMATCHINGLOWConfig,
    CDLMATCHINGLOWStream, CDLMATHOLDBatchRunner, CDLMATHOLDConfig, CDLMATHOLDStream,
    CDLMORNINGDOJISTARBatchRunner, CDLMORNINGDOJISTARConfig, CDLMORNINGDOJISTARStream,
    CDLMORNINGSTARBatchRunner, CDLMORNINGSTARConfig, CDLMORNINGSTARStream, CDLONNECKBatchRunner,
    CDLONNECKConfig, CDLONNECKStream, CDLPIERCINGBatchRunner, CDLPIERCINGConfig, CDLPIERCINGStream,
    CDLRICKSHAWMANBatchRunner, CDLRICKSHAWMANConfig, CDLRICKSHAWMANStream,
    CDLRISEFALL3METHODSBatchRunner, CDLRISEFALL3METHODSConfig, CDLRISEFALL3METHODSStream,
    CDLSEPARATINGLINESBatchRunner, CDLSEPARATINGLINESConfig, CDLSEPARATINGLINESStream,
    CDLSHOOTINGSTARBatchRunner, CDLSHOOTINGSTARConfig, CDLSHOOTINGSTARStream,
    CDLSHORTLINEBatchRunner, CDLSHORTLINEConfig, CDLSHORTLINEStream, CDLSPINNINGTOPBatchRunner,
    CDLSPINNINGTOPConfig, CDLSPINNINGTOPStream, CDLSTALLEDPATTERNBatchRunner,
    CDLSTALLEDPATTERNConfig, CDLSTALLEDPATTERNStream, CDLSTICKSANDWICHBatchRunner,
    CDLSTICKSANDWICHConfig, CDLSTICKSANDWICHStream, CDLTAKURIBatchRunner, CDLTAKURIConfig,
    CDLTAKURIStream, CDLTASUKIGAPBatchRunner, CDLTASUKIGAPConfig, CDLTASUKIGAPStream,
    CDLTHRUSTINGBatchRunner, CDLTHRUSTINGConfig, CDLTHRUSTINGStream, CDLTRISTARBatchRunner,
    CDLTRISTARConfig, CDLTRISTARStream, CDLUNIQUE3RIVERBatchRunner, CDLUNIQUE3RIVERConfig,
    CDLUNIQUE3RIVERStream, CDLUPSIDEGAP2CROWSBatchRunner, CDLUPSIDEGAP2CROWSConfig,
    CDLUPSIDEGAP2CROWSStream, CDLXSIDEGAP3METHODSBatchRunner, CDLXSIDEGAP3METHODSConfig,
    CDLXSIDEGAP3METHODSStream,
};
use fast_ta::price_transform::{
    AVGDEVBatchRunner, AVGDEVConfig, AVGDEVStream, AVGPRICEBatchRunner, AVGPRICEConfig,
    AVGPRICEStream, MEDPRICEBatchRunner, MEDPRICEConfig, MEDPRICEStream, TYPPRICEBatchRunner,
    TYPPRICEConfig, TYPPRICEStream, WCLPRICEBatchRunner, WCLPRICEConfig, WCLPRICEStream,
};
use fast_ta::statistic::{
    BETABatchRunner, BETAConfig, BETAStream, CORRELBatchRunner, CORRELConfig, CORRELStream,
    LINEARREGBatchRunner, LINEARREGConfig, LINEARREGStream, LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEConfig, LINEARREG_ANGLEStream, LINEARREG_INTERCEPTBatchRunner,
    LINEARREG_INTERCEPTConfig, LINEARREG_INTERCEPTStream, LINEARREG_SLOPEBatchRunner,
    LINEARREG_SLOPEConfig, LINEARREG_SLOPEStream, STDDEVBatchRunner, STDDEVConfig, STDDEVStream,
    TSFBatchRunner, TSFConfig, TSFStream, VARBatchRunner, VARConfig, VARStream,
};
use fast_ta::volatility::{
    ATRBatchRunner, ATRConfig, ATRStream, NATRBatchRunner, NATRConfig, NATRStream,
    TRANGEBatchRunner, TRANGEConfig, TRANGEStream,
};
use fast_ta::volume::{
    ADBatchRunner, ADConfig, ADOSCBatchRunner, ADOSCConfig, ADOSCStream, ADStream, OBVBatchRunner,
    OBVConfig, OBVStream,
};

use fast_ta::{IndicatorConfig, PreparedBatchRunner, StreamingComputation};

fn assert_execution_types<C, R, S>()
where
    C: 'static + IndicatorConfig<BatchRunner = R, Stream = S>,
    R: PreparedBatchRunner<C>,
    S: StreamingComputation<C>,
{
}

#[test]
fn catalogue_lookup_by_definition_name_is_stable() {
    let sma = INDICATOR_CATALOGUE
        .definition("SMA")
        .expect("SMA belongs to the Indicator Catalogue");

    assert_eq!(sma.name, "SMA");
    assert_eq!(sma.group, FunctionGroup::OverlapStudies);
    assert!(core::ptr::eq(
        sma,
        INDICATOR_CATALOGUE.definition("SMA").unwrap()
    ));
    assert_eq!(function("SMA"), Some(sma));
    assert!(INDICATOR_CATALOGUE.definition("sma").is_none());
    assert!(INDICATOR_CATALOGUE.definition("NOT_A_DEFINITION").is_none());
}

#[test]
fn catalogue_contains_the_official_161_definitions() {
    assert_eq!(FUNCTION_COUNT, 161);
    assert_eq!(INDICATOR_CATALOGUE.definitions().len(), FUNCTION_COUNT);
}

#[test]
fn implemented_projection_matches_published_catalogue_coverage() {
    let implemented = INDICATOR_CATALOGUE.implemented_definitions();

    assert_eq!(implemented.count(), IMPLEMENTED_FUNCTION_COUNT);
    assert!(INDICATOR_CATALOGUE
        .implemented_definitions()
        .all(|definition| definition.is_implemented()));
    assert_eq!(
        INDICATOR_CATALOGUE.implemented_definition("RSI"),
        INDICATOR_CATALOGUE.definition("RSI")
    );
    assert!(INDICATOR_CATALOGUE
        .implemented_definition("NOT_A_DEFINITION")
        .is_none());
}

#[test]
fn family_projection_matches_official_family_counts() {
    let projected_count = FunctionGroup::ALL
        .iter()
        .copied()
        .map(|family| {
            let count = INDICATOR_CATALOGUE.family_count(family);
            assert_eq!(count, family.expected_count(), "{family:?}");
            assert!(INDICATOR_CATALOGUE
                .family(family)
                .all(|definition| definition.group == family));
            count
        })
        .sum::<usize>();

    assert_eq!(projected_count, FUNCTION_COUNT);
}

#[test]
fn definitions_report_their_owning_rust_module() {
    for family in FunctionGroup::ALL.iter().copied() {
        assert!(INDICATOR_CATALOGUE
            .family(family)
            .all(|definition| definition.owner_module() == family.rust_module()));
    }

    assert_eq!(
        INDICATOR_CATALOGUE
            .definition("RSI")
            .unwrap()
            .owner_module(),
        "momentum"
    );
    assert_eq!(
        INDICATOR_CATALOGUE
            .definition("CDLDOJI")
            .unwrap()
            .owner_module(),
        "pattern_recognition"
    );
}

#[test]
fn definition_names_are_unique() {
    let definitions = INDICATOR_CATALOGUE.definitions();
    for (idx, definition) in definitions.iter().enumerate() {
        assert!(
            definitions[idx + 1..]
                .iter()
                .all(|other| other.name != definition.name),
            "duplicate definition name {}",
            definition.name
        );
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

    // Pattern Recognition (50 implemented indicators).
    assert_execution_types::<CDLDOJIConfig, CDLDOJIBatchRunner, CDLDOJIStream>();
    assert_execution_types::<CDLENGULFINGConfig, CDLENGULFINGBatchRunner, CDLENGULFINGStream>();
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
    assert_execution_types::<CDLRICKSHAWMANConfig, CDLRICKSHAWMANBatchRunner, CDLRICKSHAWMANStream>(
    );
    assert_execution_types::<CDLSHORTLINEConfig, CDLSHORTLINEBatchRunner, CDLSHORTLINEStream>();
    assert_execution_types::<CDLSPINNINGTOPConfig, CDLSPINNINGTOPBatchRunner, CDLSPINNINGTOPStream>(
    );
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
    assert_execution_types::<CDLHARAMICROSSConfig, CDLHARAMICROSSBatchRunner, CDLHARAMICROSSStream>(
    );
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
    assert_execution_types::<CDLMATCHINGLOWConfig, CDLMATCHINGLOWBatchRunner, CDLMATCHINGLOWStream>(
    );
    assert_execution_types::<CDLHAMMERConfig, CDLHAMMERBatchRunner, CDLHAMMERStream>();
    assert_execution_types::<CDLHANGINGMANConfig, CDLHANGINGMANBatchRunner, CDLHANGINGMANStream>();
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
    assert_execution_types::<CDLEVENINGSTARConfig, CDLEVENINGSTARBatchRunner, CDLEVENINGSTARStream>(
    );
    assert_execution_types::<
        CDLMORNINGDOJISTARConfig,
        CDLMORNINGDOJISTARBatchRunner,
        CDLMORNINGDOJISTARStream,
    >();
    assert_execution_types::<CDLMORNINGSTARConfig, CDLMORNINGSTARBatchRunner, CDLMORNINGSTARStream>(
    );
    assert_execution_types::<
        CDLUNIQUE3RIVERConfig,
        CDLUNIQUE3RIVERBatchRunner,
        CDLUNIQUE3RIVERStream,
    >();
    assert_execution_types::<CDL2CROWSConfig, CDL2CROWSBatchRunner, CDL2CROWSStream>();
    assert_execution_types::<CDL3LINESTRIKEConfig, CDL3LINESTRIKEBatchRunner, CDL3LINESTRIKEStream>(
    );
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
    assert_execution_types::<CDL3BLACKCROWSConfig, CDL3BLACKCROWSBatchRunner, CDL3BLACKCROWSStream>(
    );
    assert_execution_types::<
        CDL3STARSINSOUTHConfig,
        CDL3STARSINSOUTHBatchRunner,
        CDL3STARSINSOUTHStream,
    >();
    assert_execution_types::<
        CDL3WHITESOLDIERSConfig,
        CDL3WHITESOLDIERSBatchRunner,
        CDL3WHITESOLDIERSStream,
    >();
    assert_execution_types::<
        CDLADVANCEBLOCKConfig,
        CDLADVANCEBLOCKBatchRunner,
        CDLADVANCEBLOCKStream,
    >();
    assert_execution_types::<
        CDLCONCEALBABYSWALLConfig,
        CDLCONCEALBABYSWALLBatchRunner,
        CDLCONCEALBABYSWALLStream,
    >();
    assert_execution_types::<
        CDLIDENTICAL3CROWSConfig,
        CDLIDENTICAL3CROWSBatchRunner,
        CDLIDENTICAL3CROWSStream,
    >();
    assert_execution_types::<
        CDLSTALLEDPATTERNConfig,
        CDLSTALLEDPATTERNBatchRunner,
        CDLSTALLEDPATTERNStream,
    >();
    assert_execution_types::<CDLBREAKAWAYConfig, CDLBREAKAWAYBatchRunner, CDLBREAKAWAYStream>();
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
    assert_execution_types::<CDLHIKKAKEConfig, CDLHIKKAKEBatchRunner, CDLHIKKAKEStream>();
    assert_execution_types::<CDLHIKKAKEMODConfig, CDLHIKKAKEMODBatchRunner, CDLHIKKAKEMODStream>();
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
fn pattern_recognition_catalogue_has_no_remaining_planned_definitions() {
    let family = FunctionGroup::PatternRecognition;
    assert_eq!(
        INDICATOR_CATALOGUE.implemented_family_count(family),
        INDICATOR_CATALOGUE.family_count(family)
    );
    assert!(INDICATOR_CATALOGUE
        .implemented_family(family)
        .all(|definition| definition.status == ImplementationStatus::Implemented));
}

#[test]
fn names_outside_the_indicator_catalogue_are_not_found() {
    for name in ["WWMA", "HMA", "VWAP"] {
        assert!(
            INDICATOR_CATALOGUE.definition(name).is_none(),
            "{name} should not be in the Indicator Catalogue"
        );
    }
}

#[test]
fn momentum_change_family_is_implemented_through_the_public_execution_seam() {
    use fast_ta::momentum::{
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
    use fast_ta::momentum::{
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
    use fast_ta::momentum::{
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
    use fast_ta::momentum::{
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
    use fast_ta::overlap::{
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
    use fast_ta::momentum::{
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
