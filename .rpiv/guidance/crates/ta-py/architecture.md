# ta-py

## Responsibility
Python extension-module adapter for `ta-core`. It exposes Python-friendly functions/classes while keeping all indicator algorithms in `ta-core`.

## Dependencies
- **PyO3**: module registry, `#[pyfunction]`, `#[pyclass]`, `PyResult`, Python exceptions.
- **numpy**: intended 1-D array boundary for batch indicator APIs.
- **pyo3-build-config**: extension-module linker flags, especially for macOS.

## Consumers
- **Python/maturin**: imports the `ta_py` native extension.
- **Cargo workspace**: builds explicitly with `cargo build -p ta-py` or full workspace builds.

## Module Structure
```text
crates/ta-py/
├── Cargo.toml       # cdylib extension module, PyO3/NumPy deps, lib test disabled
├── build.rs         # PyO3 extension-module linker args
└── src/             # active PyO3 registry and Python-callable wrappers
```

## Extension Module Packaging

```toml
[lib]
name = "ta_py"
crate-type = ["cdylib"]
test = false

[dependencies]
pyo3 = { version = "0.29.0", features = ["extension-module"] }
numpy = "0.29.0"
```

```rust
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
```

## PyO3 Registry and Test-Friendly Functions

```rust
use pyo3::prelude::*;

#[cfg(not(test))]
#[pymodule]
fn ta_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sma_numpy, m)?)?;
    Ok(())
}

#[cfg_attr(not(test), pyfunction)]
fn hello_world() -> PyResult<String> {
    Ok("Hello from ta-py!".to_string())
}
```

## Core Error Mapping and NumPy Boundary

```rust
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use ta_core::{overlap::SMA, Float, Indicator, TalibError};

fn ta_error_to_py(err: TalibError) -> PyErr { PyValueError::new_err(err.to_string()) }

#[cfg_attr(not(test), pyfunction)]
fn sma_numpy<'py>(py: Python<'py>, prices: PyReadonlyArray1<'py, Float>, period: usize)
    -> PyResult<Bound<'py, PyArray1<Float>>>
{
    if period == 0 { return Err(PyValueError::new_err("period must be greater than zero")); }
    let prices = prices.as_slice().map_err(|_| PyValueError::new_err("prices must be contiguous"))?;

    let output = SMA::new(period).compute_to_vec(prices).map_err(ta_error_to_py)?;
    Ok(PyArray1::from_vec(py, output)) // preserve core NaN warm-up values
}
```

## Architectural Boundaries
- **NO algorithm logic**: call `ta-core`; do not reimplement indicators in Python bindings.
- **NO panic across Python FFI**: validate inputs before calling legacy assert-based core constructors.
- **NO orphan side modules**: only files under `src/` are active; the old root `types.rs` was deleted and should not be recreated here.
- **NO Python dependency in `ta-core`**: exception mapping stays local to this crate.

<important if="you are adding a new Python binding">
## Adding a New Python Binding
1. Add a `#[cfg_attr(not(test), pyfunction)]` or `#[pyclass]` wrapper in `src/`.
2. Validate Python-facing inputs and convert failures to `PyValueError` or a more specific exception.
3. Call public `ta-core` APIs and map `TalibError` with a local mapper.
4. Register the function/class in the `#[pymodule]` function.
5. Build with `cargo build -p ta-py`; use `maturin develop` for Python-side tests.
</important>
