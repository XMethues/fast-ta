//! TA-Py: Python bindings for technical analysis indicators
//!
//! This crate provides Python bindings for the core technical analysis library
//! using PyO3.
//!
//! Note: This crate requires a Python 3.x interpreter to build.

use pyo3::prelude::*;

/// Python module for technical analysis indicators
#[pymodule]
fn ta_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}

/// Example function to verify Python bindings work
#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello from ta-py!".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world() {
        assert_eq!(hello_world().unwrap(), "Hello from ta-py!");
    }
}

/// Example function to verify Python bindings work
#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello from ta-py!".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_world() {
        assert_eq!(hello_world().unwrap(), "Hello from ta-py!");
    }
}
