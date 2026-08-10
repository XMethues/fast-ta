//! Rust-first Pattern Recognition indicators and shared Candle domain values.
//!
//! Every named configuration supports owned Compact Output, initialized
//! caller-owned Batch Computation, a capacity-checked Prepared Batch Runner,
//! and an independent Streaming Computation through [`crate::IndicatorConfig`].

mod engine;
mod types;

macro_rules! impl_pattern_execution {
    ($config:ident, $runner:ident, $stream:ident) => {
        impl crate::traits::sealed::Sealed for $config {}

        impl crate::IndicatorConfig for $config {
            type Input<'a> = crate::pattern_recognition::CandleInput<'a>;
            type Output = ::alloc::vec::Vec<crate::pattern_recognition::PatternSignal>;
            type OutputMut<'a> = &'a mut [crate::pattern_recognition::PatternSignal];
            type BatchRunner = $runner;
            type Stream = $stream;

            fn lookback(&self) -> usize {
                crate::pattern_recognition::engine::PatternDefinition::lookback(self)
            }

            fn compute<'a>(
                &self,
                input: Self::Input<'a>,
            ) -> crate::Result<crate::CompactOutput<Self::Output>> {
                crate::pattern_recognition::engine::compute_owned(*self, input)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> crate::Result<crate::OutputRange> {
                crate::pattern_recognition::engine::compute_into(*self, input, output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> crate::Result<Self::BatchRunner> {
                Ok($runner {
                    engine: crate::pattern_recognition::engine::RecognitionEngine::new(*self),
                    max_input_len,
                })
            }

            fn stream(&self) -> crate::Result<Self::Stream> {
                Ok($stream {
                    engine: crate::pattern_recognition::engine::RecognitionEngine::new(*self),
                })
            }
        }

        /// Reusable capacity-checked Prepared Batch Runner.
        #[derive(Debug, Clone)]
        pub struct $runner {
            engine: crate::pattern_recognition::engine::RecognitionEngine<$config>,
            max_input_len: usize,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl crate::PreparedBatchRunner<$config> for $runner {
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            fn compute_into<'a>(
                &mut self,
                input: <$config as crate::IndicatorConfig>::Input<'a>,
                output: <$config as crate::IndicatorConfig>::OutputMut<'a>,
            ) -> crate::Result<crate::OutputRange>
            where
                $config: 'a,
            {
                crate::pattern_recognition::engine::prepared_compute_into(
                    &mut self.engine,
                    self.max_input_len,
                    input,
                    output,
                )
            }
        }

        /// Independent stateful Streaming Computation.
        #[derive(Debug, Clone)]
        pub struct $stream {
            engine: crate::pattern_recognition::engine::RecognitionEngine<$config>,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl crate::StreamingComputation<$config> for $stream {
            type Tick = crate::pattern_recognition::Candle;
            type TickOutput = crate::pattern_recognition::PatternSignal;

            fn next(&mut self, input: Self::Tick) -> crate::Result<Option<Self::TickOutput>> {
                self.engine.next(input)
            }

            fn reset(&mut self) {
                self.engine.reset();
            }
        }
    };
}

mod body_containment;
mod crows_soldiers;
mod doji;
mod engulfing;
mod gap_continuation;
mod hikkake;
mod long_formation;
mod position_shadow;
mod single_candle;
mod three_candle_reversal;

pub use body_containment::{
    CDLCOUNTERATTACKBatchRunner, CDLCOUNTERATTACKConfig, CDLCOUNTERATTACKStream,
    CDLDARKCLOUDCOVERBatchRunner, CDLDARKCLOUDCOVERConfig, CDLDARKCLOUDCOVERStream,
    CDLDOJISTARBatchRunner, CDLDOJISTARConfig, CDLDOJISTARStream, CDLHARAMIBatchRunner,
    CDLHARAMICROSSBatchRunner, CDLHARAMICROSSConfig, CDLHARAMICROSSStream, CDLHARAMIConfig,
    CDLHARAMIStream, CDLHOMINGPIGEONBatchRunner, CDLHOMINGPIGEONConfig, CDLHOMINGPIGEONStream,
    CDLKICKINGBYLENGTHBatchRunner, CDLKICKINGBYLENGTHConfig, CDLKICKINGBYLENGTHStream,
    CDLKICKINGBatchRunner, CDLKICKINGConfig, CDLKICKINGStream, CDLMATCHINGLOWBatchRunner,
    CDLMATCHINGLOWConfig, CDLMATCHINGLOWStream,
};
pub use crows_soldiers::{
    CDL3BLACKCROWSBatchRunner, CDL3BLACKCROWSConfig, CDL3BLACKCROWSStream,
    CDL3STARSINSOUTHBatchRunner, CDL3STARSINSOUTHConfig, CDL3STARSINSOUTHStream,
    CDL3WHITESOLDIERSBatchRunner, CDL3WHITESOLDIERSConfig, CDL3WHITESOLDIERSStream,
    CDLADVANCEBLOCKBatchRunner, CDLADVANCEBLOCKConfig, CDLADVANCEBLOCKStream,
    CDLCONCEALBABYSWALLBatchRunner, CDLCONCEALBABYSWALLConfig, CDLCONCEALBABYSWALLStream,
    CDLIDENTICAL3CROWSBatchRunner, CDLIDENTICAL3CROWSConfig, CDLIDENTICAL3CROWSStream,
    CDLSTALLEDPATTERNBatchRunner, CDLSTALLEDPATTERNConfig, CDLSTALLEDPATTERNStream,
};
pub use doji::{CDLDOJIBatchRunner, CDLDOJIConfig, CDLDOJIStream};
pub use engulfing::{CDLENGULFINGBatchRunner, CDLENGULFINGConfig, CDLENGULFINGStream};
pub use gap_continuation::{
    CDL2CROWSBatchRunner, CDL2CROWSConfig, CDL2CROWSStream, CDL3LINESTRIKEBatchRunner,
    CDL3LINESTRIKEConfig, CDL3LINESTRIKEStream, CDLGAPSIDESIDEWHITEBatchRunner,
    CDLGAPSIDESIDEWHITEConfig, CDLGAPSIDESIDEWHITEStream, CDLSTICKSANDWICHBatchRunner,
    CDLSTICKSANDWICHConfig, CDLSTICKSANDWICHStream, CDLTASUKIGAPBatchRunner, CDLTASUKIGAPConfig,
    CDLTASUKIGAPStream, CDLTRISTARBatchRunner, CDLTRISTARConfig, CDLTRISTARStream,
    CDLUPSIDEGAP2CROWSBatchRunner, CDLUPSIDEGAP2CROWSConfig, CDLUPSIDEGAP2CROWSStream,
    CDLXSIDEGAP3METHODSBatchRunner, CDLXSIDEGAP3METHODSConfig, CDLXSIDEGAP3METHODSStream,
};
pub use hikkake::{
    CDLHIKKAKEBatchRunner, CDLHIKKAKEConfig, CDLHIKKAKEMODBatchRunner, CDLHIKKAKEMODConfig,
    CDLHIKKAKEMODStream, CDLHIKKAKEStream,
};
pub use long_formation::{
    CDLBREAKAWAYBatchRunner, CDLBREAKAWAYConfig, CDLBREAKAWAYStream, CDLLADDERBOTTOMBatchRunner,
    CDLLADDERBOTTOMConfig, CDLLADDERBOTTOMStream, CDLMATHOLDBatchRunner, CDLMATHOLDConfig,
    CDLMATHOLDStream, CDLRISEFALL3METHODSBatchRunner, CDLRISEFALL3METHODSConfig,
    CDLRISEFALL3METHODSStream,
};
pub use position_shadow::{
    CDLHAMMERBatchRunner, CDLHAMMERConfig, CDLHAMMERStream, CDLHANGINGMANBatchRunner,
    CDLHANGINGMANConfig, CDLHANGINGMANStream, CDLINNECKBatchRunner, CDLINNECKConfig,
    CDLINNECKStream, CDLINVERTEDHAMMERBatchRunner, CDLINVERTEDHAMMERConfig,
    CDLINVERTEDHAMMERStream, CDLONNECKBatchRunner, CDLONNECKConfig, CDLONNECKStream,
    CDLPIERCINGBatchRunner, CDLPIERCINGConfig, CDLPIERCINGStream, CDLSEPARATINGLINESBatchRunner,
    CDLSEPARATINGLINESConfig, CDLSEPARATINGLINESStream, CDLSHOOTINGSTARBatchRunner,
    CDLSHOOTINGSTARConfig, CDLSHOOTINGSTARStream, CDLTHRUSTINGBatchRunner, CDLTHRUSTINGConfig,
    CDLTHRUSTINGStream,
};
pub use single_candle::{
    CDLBELTHOLDBatchRunner, CDLBELTHOLDConfig, CDLBELTHOLDStream, CDLCLOSINGMARUBOZUBatchRunner,
    CDLCLOSINGMARUBOZUConfig, CDLCLOSINGMARUBOZUStream, CDLDRAGONFLYDOJIBatchRunner,
    CDLDRAGONFLYDOJIConfig, CDLDRAGONFLYDOJIStream, CDLGRAVESTONEDOJIBatchRunner,
    CDLGRAVESTONEDOJIConfig, CDLGRAVESTONEDOJIStream, CDLHIGHWAVEBatchRunner, CDLHIGHWAVEConfig,
    CDLHIGHWAVEStream, CDLLONGLEGGEDDOJIBatchRunner, CDLLONGLEGGEDDOJIConfig,
    CDLLONGLEGGEDDOJIStream, CDLLONGLINEBatchRunner, CDLLONGLINEConfig, CDLLONGLINEStream,
    CDLMARUBOZUBatchRunner, CDLMARUBOZUConfig, CDLMARUBOZUStream, CDLRICKSHAWMANBatchRunner,
    CDLRICKSHAWMANConfig, CDLRICKSHAWMANStream, CDLSHORTLINEBatchRunner, CDLSHORTLINEConfig,
    CDLSHORTLINEStream, CDLSPINNINGTOPBatchRunner, CDLSPINNINGTOPConfig, CDLSPINNINGTOPStream,
    CDLTAKURIBatchRunner, CDLTAKURIConfig, CDLTAKURIStream,
};
pub use three_candle_reversal::{
    CDL3INSIDEBatchRunner, CDL3INSIDEConfig, CDL3INSIDEStream, CDL3OUTSIDEBatchRunner,
    CDL3OUTSIDEConfig, CDL3OUTSIDEStream, CDLABANDONEDBABYBatchRunner, CDLABANDONEDBABYConfig,
    CDLABANDONEDBABYStream, CDLEVENINGDOJISTARBatchRunner, CDLEVENINGDOJISTARConfig,
    CDLEVENINGDOJISTARStream, CDLEVENINGSTARBatchRunner, CDLEVENINGSTARConfig,
    CDLEVENINGSTARStream, CDLMORNINGDOJISTARBatchRunner, CDLMORNINGDOJISTARConfig,
    CDLMORNINGDOJISTARStream, CDLMORNINGSTARBatchRunner, CDLMORNINGSTARConfig,
    CDLMORNINGSTARStream, CDLUNIQUE3RIVERBatchRunner, CDLUNIQUE3RIVERConfig, CDLUNIQUE3RIVERStream,
};
pub use types::{
    Candle, CandleInput, CandleRangeKind, CandleSetting, CandleSettingType, CandleSettings,
    PatternDirection, PatternSignal, PatternStrength, Penetration,
};
