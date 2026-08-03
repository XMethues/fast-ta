//! Volatility Indicators.
//!
//! These functions measure price range and volatility from high/low/close inputs.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod atr;
mod natr;
mod trange;

pub use atr::{ATRBatchRunner, ATRConfig, ATRInput, ATRStream, ATRTick, ATR_vec, ATR};
pub use natr::{NATRBatchRunner, NATRConfig, NATRInput, NATRStream, NATRTick, NATR_vec, NATR};
pub use trange::{
    TRANGEBatchRunner, TRANGEConfig, TRANGEInput, TRANGEStream, TRANGETick, TRANGE_vec, TRANGE,
};
