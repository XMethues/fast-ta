//! Volume Indicators.
//!
//! These functions derive cumulative price-volume measures from market data.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod ad;
mod adosc;
mod obv;

pub use ad::{ADBatchRunner, ADConfig, ADInput, ADStream, ADTick, AD_vec, AD};
pub use adosc::{
    ADOSCBatchRunner, ADOSCConfig, ADOSCInput, ADOSCStream, ADOSCTick, ADOSC_vec, ADOSC,
};
pub use obv::{OBVBatchRunner, OBVConfig, OBVInput, OBVStream, OBVTick, OBV_vec, OBV};
