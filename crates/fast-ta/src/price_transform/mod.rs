//! Price Transform functions.
//!
//! These functions transform price-series inputs into derived real-valued series.
//! Batch APIs use separate TA-Lib-style input slices and compact output buffers.

mod avgdev;
mod avgprice;
mod medprice;
mod typprice;
mod wclprice;

pub use avgdev::{AVGDEVBatchRunner, AVGDEVConfig, AVGDEVStream, AVGDEV};
pub use avgprice::{
    AVGPRICEBatchRunner, AVGPRICEConfig, AVGPRICEInput, AVGPRICEStream, AVGPRICETick, AVGPRICE,
};
pub use medprice::{
    MEDPRICEBatchRunner, MEDPRICEConfig, MEDPRICEInput, MEDPRICEStream, MEDPRICETick, MEDPRICE,
};
pub use typprice::{
    TYPPRICEBatchRunner, TYPPRICEConfig, TYPPRICEInput, TYPPRICEStream, TYPPRICETick, TYPPRICE,
};
pub use wclprice::{
    WCLPRICEBatchRunner, WCLPRICEConfig, WCLPRICEInput, WCLPRICEStream, WCLPRICETick, WCLPRICE,
};
