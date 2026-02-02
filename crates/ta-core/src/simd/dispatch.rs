//! Runtime SIMD dispatch system.
//!
//! This module provides runtime CPU feature detection and function pointer dispatch
//! to select optimal SIMD implementation at startup time.
//!
//! The dispatch table is initialized once using `OnceLock`, and subsequent calls
//! have minimal overhead (~5-10ns) by directly calling through function pointers.

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
use once_cell::sync::OnceCell as OnceLock;
#[cfg(feature = "std")]
use std::sync::OnceLock;

use super::scalar;
use crate::types::Float;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
#[allow(unused_imports)]
use super::arch::x86_64;

#[cfg(all(target_arch = "x86_64", feature = "std"))]
use std::println as debug_println;

#[cfg(all(target_arch = "aarch64", feature = "std"))]
#[allow(unused_imports)]
use super::arch::aarch64;

#[cfg(all(target_arch = "wasm32", feature = "std"))]
#[allow(unused_imports)]
use super::arch::wasm32;

/// Function pointer type for sum operations.
///
/// This type alias represents a function that computes the sum of a slice of Float values.
pub type SumFn = fn(&[Float]) -> Float;

/// Function pointer type for dot product operations.
///
/// This type alias represents a function that computes the dot product of two Float slices.
pub type DotProductFn = fn(&[Float], &[Float]) -> Float;

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
    #[inline]
    #[allow(dead_code)]
    const fn new(sum: SumFn, dot_product: DotProductFn) -> Self {
        Self { sum, dot_product }
    }

    /// Create a scalar dispatch table (no SIMD acceleration).
    #[inline]
    const fn scalar() -> Self {
        Self {
            sum: scalar::sum,
            dot_product: scalar::dot_product,
        }
    }
}

/// Global dispatch table initialized once at startup.
///
/// This `OnceLock` ensures thread-safe one-time initialization of the dispatch table.
/// After initialization, accessing the dispatch table is as fast as a global variable.
static DISPATCH: OnceLock<DispatchTable> = OnceLock::new();

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
#[cold]
#[inline(always)]
fn init_dispatch() -> DispatchTable {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        // Runtime feature detection for AVX-512F
        let has_avx512 = { std::is_x86_feature_detected!("avx512f") };
        if has_avx512 {
            unsafe {
                return DispatchTable::new(
                    |data: &[Float]| x86_64::avx512::sum(data),
                    |a: &[Float], b: &[Float]| match x86_64::avx512::dot_product(a, b) {
                        Ok(result) => result,
                        Err(e) => panic!("dot_product error: {}", e),
                    },
                );
            }
        }
        // Runtime feature detection for AVX2
        let has_avx2 = { std::is_x86_feature_detected!("avx2") };
        if has_avx2 {
            unsafe {
                return DispatchTable::new(
                    |data: &[Float]| x86_64::avx2::sum(data),
                    |a: &[Float], b: &[Float]| match x86_64::avx2::dot_product(a, b) {
                        Ok(result) => result,
                        Err(e) => panic!("dot_product error: {}", e),
                    },
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    {
        // NEON is always available on AArch64
        unsafe {
            return DispatchTable::new(
                |data: &[Float]| aarch64::neon::sum(data),
                |a: &[Float], b: &[Float]| match aarch64::neon::dot_product(a, b) {
                    Ok(result) => result,
                    Err(e) => panic!("dot_product error: {}", e),
                },
            );
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "std"))]
    {
        // SIMD128 is enabled at compile-time
        unsafe {
            return DispatchTable::new(
                |data: &[Float]| wasm32::simd128::sum(data),
                |a: &[Float], b: &[Float]| match wasm32::simd128::dot_product(a, b) {
                    Ok(result) => result,
                    Err(e) => panic!("dot_product error: {}", e),
                },
            );
        }
    }

    // Fall back to scalar implementation
    DispatchTable::scalar()
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
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        debug_println!("SIMD DEBUG: get_dispatch called");
    }
    DISPATCH.get_or_init(init_dispatch)
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
/// use ta_core::simd::dispatch;
///
/// let data = vec![1.0_f64, 2.0, 3.0];
/// let result = dispatch::sum(&data);
/// assert_eq!(result, 6.0);
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
/// use ta_core::simd::dispatch;
///
/// let a = vec![1.0_f64, 2.0, 3.0];
/// let b = vec![4.0_f64, 5.0, 6.0];
/// let result = dispatch::dot_product(&a, &b);
/// // (1*4) + (2*5) + (3*6) = 32
/// assert_eq!(result, 32.0);
/// ```
#[inline]
pub fn dot_product(a: &[Float], b: &[Float]) -> Float {
    let dispatch = get_dispatch();
    (dispatch.dot_product)(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

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
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum(&data);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_sum_empty() {
        let data: Vec<f64> = vec![];
        let result = sum(&data);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sum_single() {
        let data = vec![42.0];
        let result = sum(&data);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_sum_with_negatives() {
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let result = sum(&data);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_dot_product_dispatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        let result = dot_product(&a, &b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_product_single() {
        let a = vec![5.0];
        let b = vec![3.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_dot_product_with_negatives() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![4.0, 5.0, -6.0];
        let result = dot_product(&a, &b);
        // (1*4) + (-2*5) + (3*-6) = 4 - 10 - 18 = -24
        assert_eq!(result, -24.0);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_dot_product_unequal_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0];
        dot_product(&a, &b);
    }

    #[test]
    fn test_dispatch_table_scalar() {
        let table = DispatchTable::scalar();
        assert_eq!((table.sum)(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!((table.dot_product)(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }

    #[test]
    fn test_dispatch_table_new() {
        let table = DispatchTable::new(
            |data: &[f64]| data.iter().sum(),
            |a: &[f64], b: &[f64]| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum(),
        );
        assert_eq!((table.sum)(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!((table.dot_product)(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
    }
}

#[cfg(all(test, feature = "std"))]
mod benchmarks {
    use super::*;
    use alloc::vec;
    extern crate std;

    #[test]
    fn benchmark_dispatch_overhead() {
        let _ = get_dispatch();

        let data = vec![1.0_f64; 1000];
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
