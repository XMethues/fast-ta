//! Volatility Indicators.
//!
//! These functions measure price range and volatility from high/low/close inputs.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod atr;
mod natr;
mod trange;
pub(crate) use trange::true_range as directional_true_range;

pub use atr::{ATRBatchRunner, ATRConfig, ATRInput, ATRStream, ATRTick, ATR};
pub use natr::{NATRBatchRunner, NATRConfig, NATRInput, NATRStream, NATRTick, NATR};
pub use trange::{TRANGEBatchRunner, TRANGEConfig, TRANGEInput, TRANGEStream, TRANGETick, TRANGE};
