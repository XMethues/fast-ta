//! WebAssembly SIMD implementation

#[cfg(all(target_arch = "wasm32", feature = "std"))]
pub mod simd128;
