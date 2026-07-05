# ta-core

## Responsibility
Core Rust library for technical-analysis algorithms, shared numeric types, errors, indicator traits, and SIMD kernels. Indicator logic belongs here, not in bindings or benchmarks.

## Dependencies
- **wide**: SIMD vector abstraction for `FastFloat` and lane kernels.
- **aligned-vec**: aligned buffers for streaming indicator state.
- **OnceLock / once_cell**: one-time SIMD dispatch initialization.

## Consumers
- **ta-py**: maps core APIs/errors to PyO3/NumPy.
- **ta-wasm**: maps core APIs/errors to wasm-bindgen/JS values.
- **ta-benchmarks**: measures public core APIs with Criterion.

## Module Structure
```text
src/
├── lib.rs, error.rs, traits.rs, types.rs   # facade: Result, TalibError, Indicator, Float
├── overlap/                                # indicator families; private files re-export public types
└── simd/                                   # scalar baseline, dispatch table, cfg-gated arch backends
```

## Public Facade and Precision Boundary

```rust
#![cfg_attr(not(feature = "std"), no_std)]
pub mod error; pub mod overlap; pub mod simd; pub mod traits; pub mod types;
pub use error::{Result, TalibError};
pub use traits::{Indicator, Resettable};
pub use types::Float;
#[cfg(feature = "f32")] pub type Float = f32;
#[cfg(not(feature = "f32"))] pub type Float = f64;
```

## Indicator Contract (Result Constructors, NaN Warm-Up)

```rust
use crate::{Float, Indicator, Result, TalibError};

pub struct ExampleIndicator { period: usize }
impl ExampleIndicator {
    // Canonical for new indicators: invalid user input is an error, not a panic.
    pub fn new(period: usize) -> Result<Self> {
        if period == 0 { return Err(TalibError::invalid_period(period, "period must be greater than zero")); }
        Ok(Self { period })
    }
}

impl Indicator for ExampleIndicator {
    type Input = Float; type Output = Float;
    fn lookback(&self) -> usize { self.period.saturating_sub(1) }
    fn compute_to_vec(&self, input: &[Float]) -> Result<Vec<Float>> {
        Ok(vec![Float::NAN; input.len()]) // preserve length; NaN = warm-up
    }
    fn next(&mut self, input: Float) -> Float { let _ = input; Float::NAN }
}
```

## SIMD Dispatch Boundary

```rust
pub type SumFn = fn(&[Float]) -> Float;
pub struct DispatchTable { pub sum: SumFn }
static DISPATCH: OnceLock<DispatchTable> = OnceLock::new();
fn init_dispatch() -> DispatchTable {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if std::is_x86_feature_detected!("avx2") {
        return DispatchTable { sum: |d| unsafe { arch::x86_64::avx2::sum(d) } };
    }
    DispatchTable { sum: scalar::sum }
}
pub fn sum(data: &[Float]) -> Float { (DISPATCH.get_or_init(init_dispatch).sum)(data) }
```

## Architectural Boundaries
- **NO Python/WASM imports**: adapters belong in `ta-py` and `ta-wasm`.
- **NO public arch backend calls**: platform SIMD stays behind `simd::dispatch`.
- **NO new assert-based constructors**: current `SMA::new -> Self` is legacy; new constructors return `Result<Self>`.
- **NO hard-coded `f64` in core APIs**: use `Float` and `as Float` literals.

<important if="you are adding a new indicator to ta-core">
## Adding a New Indicator
1. Add a file under the indicator family and re-export its public type.
2. Implement a batch kernel that leaves warm-up outputs as `Float::NAN`.
3. Use `pub fn new(...) -> Result<Self>` and `TalibError` for invalid parameters.
4. Implement `Indicator<Input = Float, Output = Float>` and `Resettable` when stateful.
5. Test invalid params, warm-up NaNs, batch/streaming consistency, reset, and `f32`.
</important>

<important if="you are adding a SIMD operation to ta-core">
## Adding a SIMD Operation
1. Add the scalar baseline first in `simd/scalar.rs`.
2. Add dispatch-table fields and safe wrappers in `simd/dispatch.rs`.
3. Re-export from `simd/mod.rs` only if public.
4. Implement cfg-gated backends under `simd/arch/*`; keep unsafe calls inside dispatch.
5. Test scalar and dispatch results against the same expected values.
</important>
