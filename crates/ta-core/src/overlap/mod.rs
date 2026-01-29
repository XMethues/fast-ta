//! Overlap Studies indicators
//!
//! This module contains indicators that "overlap" with the price data, meaning
//! they are plotted directly on the same chart as the price bars. Common examples
//! include:
//! - Simple Moving Average (SMA)
//! - Exponential Moving Average (EMA)
//! - Weighted Moving Average (WMA)
//! - Bollinger Bands
//!
//! These indicators are typically used to identify trends, support/resistance levels,
//! and potential reversal points.

mod sma;

pub use sma::Sma;
