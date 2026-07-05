//! Math Operators functions.
//!
//! This module contains TA-Lib math operators over one or two real input series.
//! Rolling and multi-output functions use compact outputs and `OutputRange`.

mod arithmetic;
mod extrema;
mod rolling;

pub use arithmetic::{ADD_vec, DIV_vec, MULT_vec, SUB_vec, ADD, DIV, MULT, SUB};
pub use extrema::{
    MAXINDEX_vec, MININDEX_vec, MINMAXINDEXOutput, MINMAXINDEX_vec, MINMAXOutput, MINMAX_vec,
    MAXINDEX, MININDEX, MINMAX, MINMAXINDEX,
};
pub use rolling::{MAX_vec, MIN_vec, SUM_vec, MAX, MIN, SUM};
