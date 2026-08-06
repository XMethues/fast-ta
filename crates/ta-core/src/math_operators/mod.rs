//! Math Operators functions.
//!
//! This module contains TA-Lib math operators over one or two real input series.
//! Rolling and multi-output functions use compact outputs and `OutputRange`.

mod arithmetic;
mod extrema;
mod rolling;

pub use arithmetic::{
    ADDBatchRunner, ADDConfig, ADDStream, BinaryInput, BinaryTick, DIVBatchRunner, DIVConfig,
    DIVStream, MULTBatchRunner, MULTConfig, MULTStream, SUBBatchRunner, SUBConfig, SUBStream, ADD,
    DIV, MULT, SUB,
};
pub use extrema::{
    MAXINDEXBatchRunner, MAXINDEXConfig, MAXINDEXStream, MININDEXBatchRunner, MININDEXConfig,
    MININDEXStream, MINMAXBatchRunner, MINMAXConfig, MINMAXINDEXBatchRunner, MINMAXINDEXConfig,
    MINMAXINDEXStream, MINMAXINDEXStreamValue, MINMAXINDEXValues, MINMAXINDEXValuesMut,
    MINMAXStream, MINMAXValue, MINMAXValues, MINMAXValuesMut, MAXINDEX, MININDEX, MINMAX,
    MINMAXINDEX,
};
pub use rolling::{
    MAXBatchRunner, MAXConfig, MAXStream, MINBatchRunner, MINConfig, MINStream, SUMBatchRunner,
    SUMConfig, SUMStream, MAX, MIN, SUM,
};
