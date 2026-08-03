//! Overlap Studies: moving averages and other price overlay indicators.

mod dema;
mod ema;
mod ma;
mod sma;
mod t3;
mod tema;
mod trima;
mod wma;

pub use dema::{DEMABatchRunner, DEMAConfig, DEMAStream, DEMA_vec, DEMA};
pub use ema::{EMABatchRunner, EMAConfig, EMAStream, EMA_vec, EMA};
pub use ma::{MABatchRunner, MAConfig, MAStream, MAType, MA_vec, MA};
pub use sma::{SMABatchRunner, SMAConfig, SMAStream, SMA_vec, SMA};
pub use t3::{
    T3BatchRunner, T3Config, T3Stream, T3_vec, T3_vec_with_default_vfactor,
    T3_with_default_vfactor, T3, T3_DEFAULT_VFACTOR,
};
pub use tema::{TEMABatchRunner, TEMAConfig, TEMAStream, TEMA_vec, TEMA};
pub use trima::{TRIMABatchRunner, TRIMAConfig, TRIMAStream, TRIMA_vec, TRIMA};
pub use wma::{WMABatchRunner, WMAConfig, WMAStream, WMA_vec, WMA};
