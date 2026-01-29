//! TA-Wasm: WebAssembly bindings for technical analysis indicators
//!
//! This crate provides WebAssembly bindings for the core technical analysis library
//! using wasm-bindgen.

use wasm_bindgen::prelude::*;

/// Example function to verify WASM bindings work
#[wasm_bindgen]
pub fn hello_world() -> String {
    "Hello from ta-wasm!".to_string()
}

/// Add two numbers in WASM
#[wasm_bindgen]
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world() {
        assert_eq!(hello_world(), "Hello from ta-wasm!");
    }

    #[test]
    fn test_add() {
        assert_eq!(add(1.0, 2.0), 3.0);
    }
}
