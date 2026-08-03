//! Math Operators functions.
//!
//! This module contains TA-Lib math operators over one or two real input series.
//! Rolling and multi-output functions use compact outputs and `OutputRange`.

mod arithmetic;
mod extrema;
mod rolling;

pub use arithmetic::{
    ADD_vec, BinaryInput, BinaryTick, DIV_vec, MULT_vec, SUB_vec, ADD, DIV, MULT, SUB,
};
pub use extrema::{
    MAXINDEXBatchRunner, MAXINDEXConfig, MAXINDEXStream, MAXINDEX_vec, MININDEXBatchRunner,
    MININDEXConfig, MININDEXStream, MININDEX_vec, MINMAXBatchRunner, MINMAXConfig,
    MINMAXINDEXBatchRunner, MINMAXINDEXConfig, MINMAXINDEXOutput, MINMAXINDEXOutputMut,
    MINMAXINDEXStream, MINMAXINDEXStreamValue, MINMAXINDEXValue, MINMAXINDEXValues,
    MINMAXINDEXValuesMut, MINMAXINDEX_vec, MINMAXOutput, MINMAXOutputMut, MINMAXStream,
    MINMAXValue, MINMAXValues, MINMAXValuesMut, MINMAX_vec, MAXINDEX, MININDEX, MINMAX,
    MINMAXINDEX,
};
pub use rolling::{
    MAXBatchRunner, MAXConfig, MAXStream, MAX_vec, MINBatchRunner, MINConfig, MINStream, MIN_vec,
    SUM_vec, MAX, MIN, SUM,
};
