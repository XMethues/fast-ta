//! # SIMD support and indicator dispatch
//!
//! This module exposes the existing standalone reduction APIs and contains the
//! architecture dispatch used by integrated Indicator batch kernels.
//!
//! [`crate::price_transform::TYPPRICEConfig`] is the first integrated Indicator
//! path. With `std`, AArch64 uses explicit NEON, x86_64 selects AVX-512F,
//! AVX2, or scalar at runtime, and wasm32 uses SIMD128 when the module is
//! compiled with `target_feature=+simd128`. Scalar WASM modules, `no_std`
//! builds, and unsupported CPUs use the scalar fallback. Streaming execution
//! remains scalar because it processes one tick at a time.
//!
//! ## Standalone API
//!
//! The existing [`sum`] and [`dot_product`] functions retain their unified
//! dispatch API:
//!
//! ```rust
//! use fast_ta::{simd, Float};
//!
//! let data: Vec<Float> = vec![1.0 as Float, 2.0 as Float, 3.0 as Float, 4.0 as Float];
//! let result = simd::sum(&data);
//! assert_eq!(result, 10.0 as Float);
//! ```
//!
//! Architecture-specific implementation modules remain private; callers use
//! Indicator configurations or these architecture-neutral standalone APIs.
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
pub use dispatch::{dot_product, sum};

#[cfg(feature = "f32")]
/// wide f32 Float
pub type FastFloat = wide::f32x16;
#[cfg(not(feature = "f32"))]
/// wide f64 Float
pub type FastFloat = wide::f64x8;
/// Number of lanes in a SIMD vector
pub const LANES: usize = mem::size_of::<FastFloat>() / mem::size_of::<Float>();
