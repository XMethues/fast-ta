#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "wasm32")]
pub mod wasm32;
