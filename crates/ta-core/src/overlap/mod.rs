//! Overlap Studies: moving averages and other price overlay indicators.

mod bands;
mod dema;
mod ema;
mod hilbert;
mod kama;
mod ma;
mod mavp;
mod midpoint;
mod sar;
mod sma;
mod t3;
mod tema;
mod trima;
mod wma;

pub use bands::{
    ACCBANDSBatchRunner, ACCBANDSConfig, ACCBANDSInput, ACCBANDSStream, ACCBANDSTick,
    ACCBANDSValue, ACCBANDSValues, ACCBANDSValuesMut, BBANDSBatchRunner, BBANDSConfig,
    BBANDSStream, BBANDSValue, BBANDSValues, BBANDSValuesMut, ACCBANDS, BBANDS,
    BBANDS_DEFAULT_NBDEV_DOWN, BBANDS_DEFAULT_NBDEV_UP,
};
pub use dema::{DEMABatchRunner, DEMAConfig, DEMAStream, DEMA};
pub(crate) use ema::{ema_multiplier, ema_seed, ema_step};
pub use ema::{EMABatchRunner, EMAConfig, EMAStream, EMA};
pub use hilbert::{
    HT_TRENDLINEBatchRunner, HT_TRENDLINEConfig, HT_TRENDLINEStream, MAMABatchRunner, MAMAConfig,
    MAMAStream, MAMAValue, MAMAValues, MAMAValuesMut, HT_TRENDLINE, HT_TRENDLINE_LOOKBACK, MAMA,
    MAMA_DEFAULT_FAST_LIMIT, MAMA_DEFAULT_SLOW_LIMIT, MAMA_LOOKBACK,
};
pub use kama::{KAMABatchRunner, KAMAConfig, KAMAStream, KAMA};
pub use ma::{MABatchRunner, MAConfig, MAStream, PeriodMAType, MA};
pub use mavp::{MAVPBatchRunner, MAVPConfig, MAVPInput, MAVPStream, MAVPTick, MAVP};
pub use midpoint::{
    MIDPOINTBatchRunner, MIDPOINTConfig, MIDPOINTStream, MIDPRICEBatchRunner, MIDPRICEConfig,
    MIDPRICEInput, MIDPRICEStream, MIDPRICETick, MIDPOINT, MIDPRICE,
};
pub use sar::{
    SARBatchRunner, SARConfig, SAREXTBatchRunner, SAREXTConfig, SAREXTStream, SARInput, SARStream,
    SARTick, SAR, SAREXT, SAREXT_DEFAULT_OFFSET_ON_REVERSE, SAREXT_DEFAULT_START_VALUE,
    SAR_DEFAULT_ACCELERATION, SAR_DEFAULT_MAXIMUM,
};
pub use sma::{SMABatchRunner, SMAConfig, SMAStream, SMA};
pub use t3::{T3BatchRunner, T3Config, T3Stream, T3_with_default_vfactor, T3, T3_DEFAULT_VFACTOR};
pub use tema::{TEMABatchRunner, TEMAConfig, TEMAStream, TEMA};
pub use trima::{TRIMABatchRunner, TRIMAConfig, TRIMAStream, TRIMA};
pub use wma::{WMABatchRunner, WMAConfig, WMAStream, WMA};
