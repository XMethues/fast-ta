//! Runtime SIMD dispatch system.
//!
//! This module provides runtime CPU feature detection and function pointer dispatch
//! to select optimal SIMD implementation at startup time.
//!
//! The dispatch tables are initialized once using `LazyLock`, and subsequent
//! calls have minimal overhead (~5-10ns) through function pointers.

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::sync::LazyLock;

#[cfg(any(
    test,
    not(feature = "std"),
    target_arch = "x86_64",
    all(feature = "simd-qualification", target_arch = "aarch64"),
    all(target_arch = "wasm32", not(target_feature = "simd128")),
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))
))]
use super::scalar;
use crate::types::Float;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(unused_imports)]
use super::arch::x86_64;

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[allow(unused_imports)]
use super::arch::aarch64;

#[cfg(all(feature = "std", target_arch = "wasm32", target_feature = "simd128"))]
use super::arch::wasm32;

/// Function pointer type for sum operations.
///
/// This type alias represents a function that computes the sum of a slice of Float values.
pub type SumFn = fn(&[Float]) -> Float;

/// Function pointer type for dot product operations.
///
/// This type alias represents a function that computes the dot product of two Float slices.
pub type DotProductFn = fn(&[Float], &[Float]) -> Float;

type FirstNonFiniteFn = fn(&[Float]) -> Option<usize>;
type TypicalPriceFn = fn(&[Float], &[Float], &[Float], &mut [Float]);
/// SIMD backend selected for validated Indicator batch kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorBackend {
    /// Portable scalar implementation.
    Scalar,
    /// x86_64 AVX2 implementation.
    Avx2,
    /// x86_64 AVX-512F implementation.
    Avx512,
    /// AArch64 NEON implementation.
    Neon,
    /// WebAssembly SIMD128 implementation.
    Simd128,
}

impl IndicatorBackend {
    /// Stable machine-readable backend name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512f",
            Self::Neon => "neon",
            Self::Simd128 => "simd128",
        }
    }
}

/// Dispatch table containing function pointers for all SIMD operations.
///
/// This struct holds function pointers for each operation, initialized with the
/// best available implementation based on CPU feature detection.
#[derive(Debug, Clone, Copy)]
pub struct DispatchTable {
    /// Function pointer for sum operations
    pub sum: SumFn,
    /// Function pointer for dot product operations
    pub dot_product: DotProductFn,
}

impl DispatchTable {
    /// Create a new dispatch table with the given function pointers.
    ///
    /// `new` is a pure constructor (no CPU feature detection), so it stays
    /// available to `no_std` unit tests via the `test` gate while the runtime
    /// path remains precise about the architectures that support SIMD dispatch.
    #[cfg(any(
        test,
        all(
            feature = "std",
            any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                all(target_arch = "wasm32", target_feature = "simd128")
            )
        )
    ))]
    #[inline]
    const fn new(sum: SumFn, dot_product: DotProductFn) -> Self {
        Self { sum, dot_product }
    }

    /// Create a scalar dispatch table (no SIMD acceleration).
    #[cfg(any(
        test,
        not(feature = "std"),
        target_arch = "x86_64",
        all(target_arch = "wasm32", not(target_feature = "simd128")),
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))
    ))]
    #[inline]
    const fn scalar() -> Self {
        Self {
            sum: scalar::sum,
            dot_product: scalar::dot_product,
        }
    }
}

#[derive(Clone, Copy)]
struct IndicatorDispatchTable {
    first_non_finite: FirstNonFiniteFn,
    backend: IndicatorBackend,
    typical_price: TypicalPriceFn,
}

impl IndicatorDispatchTable {
    #[cfg(all(
        feature = "std",
        any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        )
    ))]
    #[inline]
    const fn new(
        backend: IndicatorBackend,
        first_non_finite: FirstNonFiniteFn,
        typical_price: TypicalPriceFn,
    ) -> Self {
        Self {
            backend,
            first_non_finite,
            typical_price,
        }
    }

    #[cfg(any(
        not(feature = "std"),
        target_arch = "x86_64",
        all(feature = "simd-qualification", target_arch = "aarch64"),
        all(target_arch = "wasm32", not(target_feature = "simd128")),
        not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))
    ))]
    #[inline]
    const fn scalar() -> Self {
        Self {
            backend: IndicatorBackend::Scalar,
            first_non_finite: scalar::first_non_finite,
            typical_price: scalar::typical_price,
        }
    }
}

/// Global dispatch table initialized once at startup.
///
/// `LazyLock` provides thread-safe one-time initialization of the dispatch
/// table. After initialization, access is a global load.
#[cfg(feature = "std")]
static DISPATCH: LazyLock<DispatchTable> = LazyLock::new(init_dispatch);
#[cfg(not(feature = "std"))]
static DISPATCH_SCALAR: DispatchTable = DispatchTable::scalar();

#[cfg(feature = "std")]
static INDICATOR_DISPATCH: LazyLock<IndicatorDispatchTable> =
    LazyLock::new(init_indicator_dispatch);
#[cfg(not(feature = "std"))]
static INDICATOR_DISPATCH_SCALAR: IndicatorDispatchTable = IndicatorDispatchTable::scalar();
#[cfg(all(
    feature = "simd-qualification",
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
std::thread_local! {
    static QUALIFICATION_INDICATOR_DISPATCH:
        core::cell::Cell<Option<IndicatorDispatchTable>> = const {
            core::cell::Cell::new(None)
        };
}

/// Initialize the dispatch table with the best available SIMD implementation.
///
/// This function performs CPU feature detection and selects optimal implementation.
/// It is called automatically on first access to the dispatch table.
///
/// The detection priority is:
/// - **x86_64**: AVX-512F → AVX2 → scalar
/// - **aarch64**: NEON → scalar (though NEON is always available on AArch64)
/// - **wasm32**: SIMD128 → scalar
/// - **others**: scalar fallback
///
/// # Returns
///
/// The initialized dispatch table with function pointers to the best implementation.
#[cfg(feature = "std")]
#[cold]
#[inline(always)]
fn init_dispatch() -> DispatchTable {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        // Runtime feature detection for AVX-512F
        let has_avx512 = { std::is_x86_feature_detected!("avx512f") };
        if has_avx512 {
            return DispatchTable::new(
                |data| unsafe { x86_64::avx512::sum(data) },
                |a, b| unsafe {
                    match x86_64::avx512::dot_product(a, b) {
                        Ok(result) => result,
                        Err(e) => panic!("dot_product error: {}", e),
                    }
                },
            );
        }
        // Runtime feature detection for AVX2
        let has_avx2 = { std::is_x86_feature_detected!("avx2") };
        if has_avx2 {
            return DispatchTable::new(
                |data| unsafe { x86_64::avx2::sum(data) },
                |a, b| unsafe {
                    match x86_64::avx2::dot_product(a, b) {
                        Ok(result) => result,
                        Err(e) => panic!("dot_product error: {}", e),
                    }
                },
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on AArch64
        DispatchTable::new(
            |data| unsafe { aarch64::neon::sum(data) },
            |a, b| unsafe {
                match aarch64::neon::dot_product(a, b) {
                    Ok(result) => result,
                    Err(e) => panic!("dot_product error: {}", e),
                }
            },
        )
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        // A SIMD128 module is selected at compile time; scalar modules never
        // reference the target-feature kernels.
        DispatchTable::new(
            |data| unsafe { wasm32::simd128::sum(data) },
            |a, b| unsafe {
                match wasm32::simd128::dot_product(a, b) {
                    Ok(result) => result,
                    Err(e) => panic!("dot_product error: {}", e),
                }
            },
        )
    }

    // Fall back to scalar implementation
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    DispatchTable::scalar()
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn x86_indicator_backend(has_avx2: bool, has_avx512: bool) -> IndicatorBackend {
    // Retained reference qualification found that the f64 AVX2 and AVX-512
    // kernels failed the 5% benefit gate at representative sizes. Keep them
    // available for qualification, but only dispatch f32 calls to them.
    if !cfg!(feature = "f32") {
        return IndicatorBackend::Scalar;
    }
    if has_avx512 {
        IndicatorBackend::Avx512
    } else if has_avx2 {
        IndicatorBackend::Avx2
    } else {
        IndicatorBackend::Scalar
    }
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn x86_indicator_dispatch(backend: IndicatorBackend) -> IndicatorDispatchTable {
    match backend {
        IndicatorBackend::Scalar => IndicatorDispatchTable::scalar(),
        IndicatorBackend::Avx2 => IndicatorDispatchTable::new(
            IndicatorBackend::Avx2,
            |values| unsafe { x86_64::avx2::first_non_finite(values) },
            |high, low, close, output| unsafe {
                x86_64::avx2::typical_price(high, low, close, output)
            },
        ),
        IndicatorBackend::Avx512 => IndicatorDispatchTable::new(
            IndicatorBackend::Avx512,
            |values| unsafe { x86_64::avx512::first_non_finite(values) },
            |high, low, close, output| unsafe {
                x86_64::avx512::typical_price(high, low, close, output)
            },
        ),
        IndicatorBackend::Neon | IndicatorBackend::Simd128 => unreachable!(),
    }
}

#[cfg(all(
    feature = "simd-qualification",
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
#[inline]
fn qualification_indicator_dispatch(backend: IndicatorBackend) -> IndicatorDispatchTable {
    #[cfg(target_arch = "x86_64")]
    {
        x86_indicator_dispatch(backend)
    }
    #[cfg(target_arch = "aarch64")]
    {
        match backend {
            IndicatorBackend::Scalar => IndicatorDispatchTable::scalar(),
            IndicatorBackend::Neon => IndicatorDispatchTable::new(
                IndicatorBackend::Neon,
                |values| unsafe { aarch64::neon::first_non_finite(values) },
                |high, low, close, output| unsafe {
                    aarch64::neon::typical_price(high, low, close, output)
                },
            ),
            IndicatorBackend::Avx2 | IndicatorBackend::Avx512 | IndicatorBackend::Simd128 => {
                unreachable!()
            }
        }
    }
}

#[cfg(feature = "std")]
#[cold]
fn init_indicator_dispatch() -> IndicatorDispatchTable {
    #[cfg(target_arch = "x86_64")]
    {
        let backend = x86_indicator_backend(
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512f"),
        );
        x86_indicator_dispatch(backend)
    }

    #[cfg(target_arch = "aarch64")]
    {
        IndicatorDispatchTable::new(
            IndicatorBackend::Neon,
            |values| unsafe { aarch64::neon::first_non_finite(values) },
            |high, low, close, output| unsafe {
                aarch64::neon::typical_price(high, low, close, output)
            },
        )
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        IndicatorDispatchTable::new(
            IndicatorBackend::Simd128,
            |values| unsafe { wasm32::simd128::first_non_finite(values) },
            |high, low, close, output| unsafe {
                wasm32::simd128::typical_price(high, low, close, output)
            },
        )
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    IndicatorDispatchTable::scalar()
}

#[inline]
fn get_indicator_dispatch() -> IndicatorDispatchTable {
    #[cfg(all(
        feature = "simd-qualification",
        feature = "std",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    if let Some(dispatch) = QUALIFICATION_INDICATOR_DISPATCH.get() {
        return dispatch;
    }

    #[cfg(feature = "std")]
    {
        *INDICATOR_DISPATCH
    }
    #[cfg(not(feature = "std"))]
    {
        INDICATOR_DISPATCH_SCALAR
    }
}

/// Get the global dispatch table, initializing it if necessary.
///
/// This function provides access to the global dispatch table. The first call
/// triggers CPU feature detection and initialization. Subsequent calls are
/// essentially a simple load from a global variable.
///
/// # Performance
///
/// - First call: ~100-500ns (includes CPU feature detection)
/// - Subsequent calls: ~5-10ns (single pointer dereference)
///
/// # Returns
///
/// A reference to the dispatch table.
#[inline]
pub fn get_dispatch() -> &'static DispatchTable {
    #[cfg(feature = "std")]
    {
        &DISPATCH
    }
    #[cfg(not(feature = "std"))]
    {
        &DISPATCH_SCALAR
    }
}
/// Returns the backend selected for validated Indicator batch kernels.
///
/// This is intended for runtime qualification and diagnostics. Indicator
/// callers do not need to branch on the result.
#[inline]
pub fn active_indicator_backend() -> IndicatorBackend {
    get_indicator_dispatch().backend
}

/// Returns the index of the first non-finite value, if any.
///
/// Accelerated `std` builds recover the exact first invalid lane on SIMD
/// failure. Unsupported CPUs, architectures, and `no_std` builds use the
/// portable scalar fallback.
#[inline]
pub(crate) fn first_non_finite(values: &[Float]) -> Option<usize> {
    (get_indicator_dispatch().first_non_finite)(values)
}

/// Computes Typical Price into a caller-provided output buffer.
///
/// The public Indicator seam validates all inputs before this dispatched
/// kernel runs. Unsupported CPUs, architectures, and `no_std` builds retain
/// the portable scalar implementation.
#[inline]
pub(crate) fn typical_price(high: &[Float], low: &[Float], close: &[Float], output: &mut [Float]) {
    (get_indicator_dispatch().typical_price)(high, low, close, output);
}

/// Explicit kernels and scoped public-boundary overrides used by the x86 and
/// AArch64 platform qualification harnesses.
///
/// The module is feature-gated so ordinary consumers cannot accidentally
/// bypass or override the validated Indicator execution seam.
#[cfg(all(
    feature = "simd-qualification",
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub mod qualification {
    use super::{
        qualification_indicator_dispatch, IndicatorBackend, IndicatorDispatchTable,
        QUALIFICATION_INDICATOR_DISPATCH,
    };
    use crate::Float;

    /// Reports whether the current CPU can execute a qualification backend.
    pub fn backend_available(backend: IndicatorBackend) -> bool {
        match backend {
            IndicatorBackend::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            IndicatorBackend::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            IndicatorBackend::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "x86_64")]
            IndicatorBackend::Neon | IndicatorBackend::Simd128 => false,
            #[cfg(target_arch = "aarch64")]
            IndicatorBackend::Neon => true,
            #[cfg(target_arch = "aarch64")]
            IndicatorBackend::Avx2 | IndicatorBackend::Avx512 | IndicatorBackend::Simd128 => false,
        }
    }

    struct OverrideGuard {
        previous: Option<IndicatorDispatchTable>,
    }

    impl Drop for OverrideGuard {
        fn drop(&mut self) {
            QUALIFICATION_INDICATOR_DISPATCH.set(self.previous);
        }
    }

    /// Runs a closure with the selected backend at the normal public Indicator
    /// validation, output, and dispatch boundary.
    ///
    /// The override is scoped to the current thread, restores nested overrides,
    /// and is reset during unwinding.
    ///
    /// # Panics
    ///
    /// Panics when the selected backend is unavailable on the current CPU.
    pub fn with_indicator_backend<R>(backend: IndicatorBackend, run: impl FnOnce() -> R) -> R {
        assert!(
            backend_available(backend),
            "backend {} is unavailable",
            backend.as_str()
        );
        let previous = QUALIFICATION_INDICATOR_DISPATCH
            .replace(Some(qualification_indicator_dispatch(backend)));
        let guard = OverrideGuard { previous };
        let result = run();
        drop(guard);
        result
    }

    /// Executes one explicitly selected kernel for differential qualification.
    ///
    /// # Panics
    ///
    /// Panics when the slices differ in length, the output is too short, or
    /// the selected backend is unavailable on the current CPU.
    pub fn typical_price(
        backend: IndicatorBackend,
        high: &[Float],
        low: &[Float],
        close: &[Float],
        output: &mut [Float],
    ) {
        assert_eq!(high.len(), low.len());
        assert_eq!(high.len(), close.len());
        assert!(output.len() >= high.len());
        assert!(
            backend_available(backend),
            "backend {} is unavailable",
            backend.as_str()
        );

        (qualification_indicator_dispatch(backend).typical_price)(high, low, close, output);
    }
}

/// Calculate the sum of all elements in a slice.
///
/// This function automatically dispatches to the best available SIMD implementation.
/// The first call will initialize the dispatch table (~100-500ns), subsequent calls
/// have minimal overhead (~5-10ns).
///
/// # Arguments
///
/// * `data` - A slice of floating-point values
///
/// # Returns
///
/// The sum of all elements in slice.
///
/// # Examples
///
/// ```rust
/// use ta_core::{simd::dispatch, Float};
///
/// let data: Vec<Float> = vec![1.0 as Float, 2.0 as Float, 3.0 as Float];
/// let result = dispatch::sum(&data);
/// assert_eq!(result, 6.0 as Float);
/// ```
#[inline]
pub fn sum(data: &[Float]) -> Float {
    let dispatch = get_dispatch();
    (dispatch.sum)(data)
}

/// Calculate the dot product of two vectors.
///
/// This function automatically dispatches to the best available SIMD implementation.
/// The first call will initialize the dispatch table (~100-500ns), subsequent calls
/// have minimal overhead (~5-10ns).
///
/// # Arguments
///
/// * `a` - First vector (slice of floating-point values)
/// * `b` - Second vector (slice of floating-point values)
///
/// # Returns
///
/// The dot product (element-wise multiplication sum) of the two vectors.
///
/// # Panics
///
/// Panics if the input vectors have different lengths.
///
/// # Examples
///
/// ```rust
/// use ta_core::{simd::dispatch, Float};
///
/// let a: Vec<Float> = vec![1.0 as Float, 2.0 as Float, 3.0 as Float];
/// let b: Vec<Float> = vec![4.0 as Float, 5.0 as Float, 6.0 as Float];
/// let result = dispatch::dot_product(&a, &b);
/// // (1*4) + (2*5) + (3*6) = 32
/// assert_eq!(result, 32.0 as Float);
/// ```
#[inline]
pub fn dot_product(a: &[Float], b: &[Float]) -> Float {
    let dispatch = get_dispatch();
    (dispatch.dot_product)(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn active_indicator_lane_width() -> usize {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return if cfg!(feature = "f32") { 16 } else { 8 };
            }
            if std::is_x86_feature_detected!("avx2") {
                return if cfg!(feature = "f32") { 8 } else { 4 };
            }
            return 1;
        }

        #[cfg(not(all(feature = "std", target_arch = "x86_64")))]
        if cfg!(feature = "f32") {
            4
        } else {
            2
        }
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn x86_indicator_selection_uses_only_qualified_precision_backends() {
        if cfg!(feature = "f32") {
            assert_eq!(
                x86_indicator_backend(false, false),
                IndicatorBackend::Scalar
            );
            assert_eq!(x86_indicator_backend(true, false), IndicatorBackend::Avx2);
            assert_eq!(x86_indicator_backend(true, true), IndicatorBackend::Avx512);
        } else {
            for (has_avx2, has_avx512) in [(false, false), (true, false), (true, true)] {
                assert_eq!(
                    x86_indicator_backend(has_avx2, has_avx512),
                    IndicatorBackend::Scalar
                );
            }
        }
    }

    #[cfg(all(
        feature = "simd-qualification",
        feature = "std",
        target_arch = "x86_64"
    ))]
    #[test]
    fn qualification_override_controls_the_public_indicator_boundary() {
        let production_backend = active_indicator_backend();
        let high = [3.0 as Float, 6.0 as Float];
        let low = [1.0 as Float, 2.0 as Float];
        let close = [2.0 as Float, 4.0 as Float];
        let mut scalar_output = [0.0 as Float; 2];

        qualification::with_indicator_backend(IndicatorBackend::Scalar, || {
            assert_eq!(active_indicator_backend(), IndicatorBackend::Scalar);
            crate::price_transform::TYPPRICE(&high, &low, &close, &mut scalar_output).unwrap();
        });

        for backend in [IndicatorBackend::Avx2, IndicatorBackend::Avx512]
            .into_iter()
            .filter(|backend| qualification::backend_available(*backend))
        {
            let mut output = [0.0 as Float; 2];
            qualification::with_indicator_backend(backend, || {
                assert_eq!(active_indicator_backend(), backend);
                crate::price_transform::TYPPRICE(&high, &low, &close, &mut output).unwrap();
            });
            assert_eq!(output, scalar_output);
        }

        assert_eq!(active_indicator_backend(), production_backend);
    }

    #[test]
    fn test_dispatch_initialization() {
        let dispatch1 = get_dispatch();
        let dispatch2 = get_dispatch();
        assert!(
            core::ptr::eq(dispatch1, dispatch2),
            "Dispatch table should be initialized only once"
        );
    }

    #[test]
    fn test_dispatch_same_function_pointers() {
        let dispatch = get_dispatch();

        // Call multiple times and verify function pointers are the same
        let fp1 = dispatch.sum as *const ();
        let fp2 = dispatch.sum as *const ();
        assert_eq!(fp1, fp2, "Function pointers should remain constant");

        let fp3 = dispatch.dot_product as *const ();
        let fp4 = dispatch.dot_product as *const ();
        assert_eq!(fp3, fp4, "Function pointers should remain constant");
    }

    #[test]
    fn test_sum_dispatch() {
        let data: Vec<Float> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum(&data);
        assert!((result - Float::from(15.0)).abs() < Float::from(1e-10));
    }

    #[test]
    fn test_sum_empty() {
        let data: Vec<Float> = vec![];
        let result = sum(&data);
        assert_eq!(result, Float::from(0.0));
    }

    #[test]
    fn test_sum_single() {
        let data: Vec<Float> = vec![Float::from(42.0)];
        let result = sum(&data);
        assert_eq!(result, Float::from(42.0));
    }

    #[test]
    fn test_sum_with_negatives() {
        let data: Vec<Float> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let result = sum(&data);
        assert!((result - Float::from(3.0)).abs() < Float::from(1e-10));
    }

    #[test]
    fn test_dot_product_dispatch() {
        let a: Vec<Float> = vec![1.0, 2.0, 3.0];
        let b: Vec<Float> = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        assert!((result - Float::from(32.0)).abs() < Float::from(1e-10));
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<Float> = vec![];
        let b: Vec<Float> = vec![];
        let result = dot_product(&a, &b);
        assert_eq!(result, Float::from(0.0));
    }

    #[test]
    fn test_dot_product_single() {
        let a: Vec<Float> = vec![Float::from(5.0)];
        let b: Vec<Float> = vec![Float::from(3.0)];
        let result = dot_product(&a, &b);
        assert_eq!(result, Float::from(15.0));
    }

    #[test]
    fn test_dot_product_with_negatives() {
        let a: Vec<Float> = vec![1.0, -2.0, 3.0];
        let b: Vec<Float> = vec![4.0, 5.0, -6.0];
        let result = dot_product(&a, &b);
        // (1*4) + (-2*5) + (3*-6) = 4 - 10 - 18 = -24
        assert!((result - Float::from(-24.0)).abs() < Float::from(1e-10));
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_dot_product_unequal_lengths() {
        let a: Vec<Float> = vec![Float::from(1.0), Float::from(2.0)];
        let b: Vec<Float> = vec![Float::from(3.0)];
        dot_product(&a, &b);
    }

    #[test]
    fn test_dispatch_table_scalar() {
        let table = DispatchTable::scalar();
        let sum_result = (table.sum)(&[1.0 as Float, 2.0 as Float, 3.0 as Float]);
        assert!((sum_result - 6.0 as Float).abs() < Float::from(1e-10));
        let dot_result =
            (table.dot_product)(&[1.0 as Float, 2.0 as Float], &[3.0 as Float, 4.0 as Float]);
        assert!((dot_result - 11.0 as Float).abs() < Float::from(1e-10));
    }

    #[test]
    fn test_dispatch_table_new() {
        let table = DispatchTable::new(
            |data: &[Float]| data.iter().copied().sum(),
            |a: &[Float], b: &[Float]| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        );
        let sum_result = (table.sum)(&[1.0 as Float, 2.0 as Float, 3.0 as Float]);
        assert!((sum_result - 6.0 as Float).abs() < Float::from(1e-10));
        let dot_result =
            (table.dot_product)(&[1.0 as Float, 2.0 as Float], &[3.0 as Float, 4.0 as Float]);
        assert!((dot_result - 11.0 as Float).abs() < Float::from(1e-10));
    }

    #[test]
    fn first_non_finite_dispatch_matches_scalar_for_short_vectors_tails_and_failures() {
        let lane_width = active_indicator_lane_width();
        for len in [
            0,
            1,
            lane_width - 1,
            lane_width,
            lane_width + 1,
            lane_width * 4 - 1,
            lane_width * 4,
            lane_width * 4 + 1,
            257,
        ] {
            let values = vec![1.0 as Float; len];
            assert_eq!(first_non_finite(&values), None, "all-finite length {len}");
            assert_eq!(
                first_non_finite(&values),
                scalar::first_non_finite(&values),
                "scalar parity length {len}"
            );
        }

        let len = lane_width * 4 + 3;
        for (invalid_index, invalid_value) in [
            (0, Float::NAN),
            (lane_width - 1, Float::INFINITY),
            (lane_width, Float::NEG_INFINITY),
            (lane_width * 4 - 1, Float::NAN),
            (lane_width * 4, Float::INFINITY),
            (len - 1, Float::NEG_INFINITY),
        ] {
            let mut values = vec![1.0 as Float; len];
            values[invalid_index] = invalid_value;
            assert_eq!(
                first_non_finite(&values),
                Some(invalid_index),
                "invalid index {invalid_index}"
            );
            assert_eq!(
                first_non_finite(&values),
                scalar::first_non_finite(&values),
                "scalar parity at invalid index {invalid_index}"
            );
        }

        let mut values = vec![1.0 as Float; len];
        values[lane_width * 4 + 1] = Float::INFINITY;
        values[1] = Float::NAN;
        assert_eq!(first_non_finite(&values), Some(1));
    }

    #[test]
    fn typical_price_dispatch_matches_scalar_fallback() {
        let lane_width = active_indicator_lane_width();
        for len in [0, 1, lane_width, lane_width + 1, lane_width * 2 + 1, 257] {
            let high: Vec<Float> = (0..len)
                .map(|index| index as Float + 3.0 as Float)
                .collect();
            let low: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.5 as Float - 2.0 as Float)
                .collect();
            let close: Vec<Float> = (0..len)
                .map(|index| index as Float * 0.25 as Float + 1.0 as Float)
                .collect();
            let mut expected = vec![0.0 as Float; len];
            let mut actual = vec![0.0 as Float; len];

            scalar::typical_price(&high, &low, &close, &mut expected);
            typical_price(&high, &low, &close, &mut actual);

            assert_eq!(actual, expected, "length {len}");
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod benchmarks {
    use super::*;
    #[test]
    fn benchmark_dispatch_overhead() {
        let _ = get_dispatch();

        let data: Vec<Float> = vec![Float::from(1.0); 1000];
        let _ = sum(&data);

        let iterations = 100_000;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = sum(&data);
        }
        let duration = start.elapsed();

        let avg_ns = duration.as_nanos() / iterations as u128;

        std::println!("Average dispatch+compute time per call: {} ns", avg_ns);

        // In release mode with AVX2, 1000 elements should take ~5000-8000ns
        // In debug mode, performance is much worse (~20000ns)
        // So we only enforce strict check in release mode
        #[cfg(not(debug_assertions))]
        assert!(
            avg_ns < 10000,
            "Dispatch overhead too high: {} ns per call",
            avg_ns
        );
    }

    #[test]
    fn benchmark_initialization_time() {
        let start = std::time::Instant::now();
        let dispatch = get_dispatch();
        let duration = start.elapsed();

        std::println!("Dispatch initialization time: {:?}", duration);

        assert!(
            duration.as_micros() < 500,
            "Initialization too slow: {:?}",
            duration
        );

        assert_eq!((dispatch.sum)(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn benchmark_multiple_dispatches() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let expected = scalar::sum(&data);

        let result1 = sum(&data);
        let result2 = sum(&data);
        let result3 = sum(&data);

        assert_eq!(result1, expected);
        assert_eq!(result2, expected);
        assert_eq!(result3, expected);
    }
}
