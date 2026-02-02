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

#![no_std]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

extern crate alloc;

pub mod error;
/// Overlap studies: Moving averages and other price overlay indicators
pub mod overlap;
pub mod simd;
pub mod traits;
pub mod types;

pub use error::{Result, TalibError};
pub use traits::{Indicator, Resettable};
pub use types::Float;
