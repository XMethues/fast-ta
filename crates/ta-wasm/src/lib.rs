//! TA-Wasm: WebAssembly bindings for technical analysis indicators
//!
//! This crate provides WebAssembly bindings for the core technical analysis library
//! using wasm-bindgen.

use fast_ta::{
    price_transform::{TYPPRICEConfig, TYPPRICEInput},
    simd::dispatch::active_indicator_backend,
    IndicatorConfig,
};
use wasm_bindgen::prelude::*;

/// Public Typical Price result, including the core Output Range.
#[wasm_bindgen]
pub struct TyppriceResult {
    output_begin: u32,
    output_count: u32,
    values: Vec<f64>,
}

#[wasm_bindgen]
impl TyppriceResult {
    /// Inclusive start index of the compact output in the input series.
    #[wasm_bindgen(getter)]
    pub fn output_begin(&self) -> u32 {
        self.output_begin
    }

    /// Number of valid output values.
    #[wasm_bindgen(getter)]
    pub fn output_count(&self) -> u32 {
        self.output_count
    }

    /// Copies the compact output values across the WASM boundary.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.values.clone()
    }
}

/// Computes TYPPRICE through fast-ta's validated public Indicator path.
///
/// JavaScript receives an `Error` for unequal input lengths or non-finite
/// values, using the same validation messages as fast-ta.
#[wasm_bindgen]
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> Result<TyppriceResult, JsError> {
    let output = TYPPRICEConfig::new()
        .compute(TYPPRICEInput { high, low, close })
        .map_err(|error| JsError::new(&error.to_string()))?;
    let range = output.range();
    let output_begin = u32::try_from(range.beg_idx)
        .map_err(|_| JsError::new("TYPPRICE output_begin exceeds the WASM u32 range"))?;
    let output_count = u32::try_from(range.nb_element)
        .map_err(|_| JsError::new("TYPPRICE output_count exceeds the WASM u32 range"))?;
    let values = output.into_values();
    Ok(TyppriceResult {
        output_begin,
        output_count,
        values,
    })
}

/// Returns the active fast-ta Indicator backend as a stable identifier.
#[wasm_bindgen]
pub fn typprice_backend() -> String {
    active_indicator_backend().as_str().to_owned()
}

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

    #[test]
    fn test_typprice_public_result() {
        let result = typprice(&[3.0, 6.0], &[0.0, 3.0], &[1.5, 4.5]).unwrap();
        assert_eq!(result.output_begin(), 0);
        assert_eq!(result.output_count(), 2);
        assert_eq!(result.values(), vec![1.5, 4.5]);
    }
}
