//! Floating-point type configuration for conditional compilation
//!
//! This module defines a type alias that can be switched between `f32` and `f64`
//! using cargo features. Default is `f64`.
//!
//! # Features
//!
//! - `f64` (default): Use double-precision floating-point
//! - `f32`: Use single-precision floating-point
//!
//! # Example
//!
//! ```rust,ignore
//! use ta_core::types::Float;
//!
//! let x: Float = 1.0;
//! let y: Float = 2.0;
//! assert_eq!(x + y, 3.0);
//! ```

/// Floating-point type used throughout the library
///
/// When `f32` feature is enabled, this is `f32` (single-precision).
/// Otherwise, defaults to `f64` (double-precision).
#[cfg(feature = "f32")]
pub type Float = f32;

/// Floating-point type used throughout the library
///
/// When `f32` feature is NOT enabled, this is `f64` (double-precision).
/// This is the default configuration.
#[cfg(not(feature = "f32"))]
pub type Float = f64;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_type_exists() {
        let _: Float = 1.0;
        let _: Float = 2.0;
    }

    #[test]
    fn test_float_arithmetic() {
        let x: Float = 1.5;
        let y: Float = 2.5;
        assert!((x + y - 4.0).abs() < 1e-10);
    }
}
