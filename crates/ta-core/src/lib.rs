//! TA-Core: Core library for technical analysis indicators
//!
//! This crate provides the core implementation of technical analysis indicators
//! with `no_std` compatibility, SIMD optimizations, and conditional float precision.
//!
//! # Features
//!
//! - `f64` (default): Double-precision floating-point
//! - `f32`: Single-precision floating-point
//!
//! # Modules
//!
//! - [`types`]: Floating-point type configuration

#![no_std]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

pub mod types;

pub use types::Float;
