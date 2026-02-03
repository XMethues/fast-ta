//! # SIMD Accelerated Technical Analysis
//!
//! This module provides SIMD-accelerated implementations of technical analysis operations.
//! It automatically selects the best available SIMD instruction set based on the target platform
//! and provides a scalar fallback for unsupported platforms.
//!
//! ## Usage
//!
//! The module provides a unified API for all SIMD levels. Functions are automatically
//! dispatched to the best available implementation at runtime (x86_64) or compile-time.
//!
//! ```rust
//! use ta_core::simd;
//!
//! let data = vec![1.0_f64, 2.0, 3.0, 4.0];
//! let result = simd::sum(&data);  // Auto-dispatched to AVX2/AVX-512/NEON/SIMD128/Scalar
//! assert_eq!(result, 10.0);
//! ```
//!
//! ## Performance Considerations
//!
//! - SIMD implementations require aligned data for optimal performance
//! - For very small arrays, scalar operations may be faster due to SIMD overhead
//! - Use the `dispatch` module for runtime-dispatched operations (recommended)
//! - Direct platform-specific modules are available via `arch` submodule
use crate::Float;
use wide;
pub mod scalar;
// Include arch module for all platforms with std support
#[cfg(feature = "std")]
mod arch;
#[cfg(not(feature = "std"))]
use core::mem;
#[cfg(feature = "std")]
use std::mem;

pub mod dispatch;

#[cfg(feature = "f32")]
/// wide f32 Float
pub type FastFloat = wide::f32x16;
#[cfg(not(feature = "f32"))]
/// wide f64 Float
pub type FastFloat = wide::f64x8;
/// Number of lanes in a SIMD vector
pub const LANES: usize = mem::size_of::<FastFloat>() / mem::size_of::<Float>();
