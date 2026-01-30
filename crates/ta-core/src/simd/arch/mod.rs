#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod x86_64;

#[cfg(all(target_arch = "aarch64", feature = "std"))]
pub mod aarch64;

#[cfg(all(target_arch = "wasm32", feature = "std"))]
pub mod wasm32;
