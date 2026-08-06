//! Statistic Functions.
//!
//! These functions calculate rolling variance, paired statistics, and linear
//! regression projections. Batch APIs write compact outputs; convenience
//! wrappers return input-length vectors padded with `Float::NAN`.

mod beta;
mod correl;
mod moments;
mod regression;
mod variance;

use crate::Float;

pub use beta::{BETABatchRunner, BETAConfig, BETAStream, BETA_vec, BETA};
pub use correl::{CORRELBatchRunner, CORRELConfig, CORRELStream, CORREL_vec, CORREL};
pub use regression::{
    LINEARREGBatchRunner, LINEARREGConfig, LINEARREGStream, LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEConfig, LINEARREG_ANGLEStream, LINEARREG_ANGLE_vec,
    LINEARREG_INTERCEPTBatchRunner, LINEARREG_INTERCEPTConfig, LINEARREG_INTERCEPTStream,
    LINEARREG_INTERCEPT_vec, LINEARREG_SLOPEBatchRunner, LINEARREG_SLOPEConfig,
    LINEARREG_SLOPEStream, LINEARREG_SLOPE_vec, LINEARREG_vec, TSFBatchRunner, TSFConfig,
    TSFStream, TSF_vec, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT, LINEARREG_SLOPE, TSF,
};
pub use variance::{
    STDDEVBatchRunner, STDDEVConfig, STDDEVStream, STDDEV_vec, STDDEV_vec_with_default_nbdev,
    STDDEV_with_default_nbdev, VARBatchRunner, VARConfig, VARStream, VAR_vec,
    VAR_vec_with_default_nbdev, VAR_with_default_nbdev, STDDEV, VAR,
};

/// Borrowed paired real-valued inputs for statistic batch computation.
#[derive(Debug, Clone, Copy)]
pub struct PairInput<'a> {
    /// First real-valued input series.
    pub real0: &'a [Float],
    /// Second real-valued input series.
    pub real1: &'a [Float],
}

/// One paired real-valued tick for statistic streaming computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairTick {
    /// First real-valued input.
    pub real0: Float,
    /// Second real-valued input.
    pub real1: Float,
}
