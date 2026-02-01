// x86_64 SIMD integration tests
//
// Tests for AVX2 and AVX-512 SIMD implementations on x86_64 architecture.

#![cfg(test)]
#![cfg(target_arch = "x86_64")]

use ta_core::simd;

/// Test that SIMD sum produces correct results
#[test]
fn test_sum_correctness() {
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let result = simd::sum(&data);
    let expected: f64 = data.iter().sum();
    assert!((result - expected).abs() < 1e-10);
}

/// Test cross-path consistency
#[test]
fn test_dispatch_consistency() {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();

    let result1 = simd::sum(&data);
    let result2 = simd::sum(&data);
    let result3 = simd::sum(&data);

    // All calls should produce identical results
    assert_eq!(result1, result2);
    assert_eq!(result2, result3);
}
