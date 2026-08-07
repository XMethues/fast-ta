//! TA-Core: Core library for technical analysis indicators
//!
//! This crate provides the core implementation of technical analysis indicators
//! with `no_std` compatibility, SIMD optimizations, and conditional float precision.
//!
//! # Features
//!
//! - `f64` (default): Double-precision floating-point
//! - `f32`: Single-precision floating-point
//! - `std`: Enable standard library support (for I/O and additional error conversions)
//! - `core_error`: Enable core::error::Error trait (requires Rust 1.81+)
//!
//! # Modules
//!
//! - [`types`]: Floating-point type configuration
//! - [`error`]: Error types and handling
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use alloc::{format, string::String, vec::Vec};

#[cfg(feature = "std")]
#[allow(unused_imports)]
use std::{format, string::String, vec::Vec};

pub mod common;
/// Cycle Indicators.
pub mod cycle;
pub mod error;
/// Official TA-Lib function inventory and implementation status.
pub mod inventory;
/// Math Operators functions.
pub mod math_operators;
/// Math Transform functions.
pub mod math_transform;
/// Overlap studies: Moving averages and other price overlay indicators.
pub mod overlap;
/// Price Transform functions.
pub mod price_transform;
pub mod simd;
/// Statistic Functions.
pub mod statistic;
pub mod traits;
pub mod types;
/// Volatility Indicators functions.
pub mod volatility;
/// Volume Indicators functions.
pub mod volume;

pub use common::{
    output_count, period_lookback, validate_all_same_len, validate_finite_slice,
    validate_finite_slices, validate_input_len, validate_output_len, validate_period,
    validate_same_len, CompactOutput, OutputRange,
};

pub use error::{Result, TalibError};
pub use traits::{IndicatorConfig, PreparedBatchRunner, StreamingComputation};

pub use types::Float;
