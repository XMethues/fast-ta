#![cfg(test)]

use ta_core::simd;

#[test]
fn test_cross_path_consistency() {
    let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();

    let result_simd = simd::sum(&data, simd::SimdLevel::Scalar);
    let result_dispatch = simd::sum(&data, simd::SimdLevel::detect());

    assert!((result_simd - result_dispatch).abs() < 1e-9);
}

#[test]
fn test_dispatch_sum() {
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();

    let result = simd::sum(&data, simd::SimdLevel::detect());
    let expected: f64 = data.iter().sum();

    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn test_dispatch_dot_product() {
    let a: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    let b: Vec<f64> = (0..100).map(|i| (i as f64).cos()).collect();

    let result = simd::dot_product(&a, &b, simd::SimdLevel::detect());
    let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    assert!((result - expected).abs() < 1e-9);
}

#[test]
fn test_large_array_sum() {
    let data: Vec<f64> = (0..10000).map(|i| i as f64 * 0.1).collect();

    let result = simd::sum(&data, simd::SimdLevel::detect());
    let expected: f64 = data.iter().sum();

    assert!((result - expected).abs() < 1e-9);
}

#[test]
fn test_empty_array() {
    let data: Vec<f64> = vec![];

    let result = simd::sum(&data, simd::SimdLevel::detect());
    assert_eq!(result, 0.0);
}

#[test]
fn test_single_element() {
    let data: Vec<f64> = vec![42.0];

    let result = simd::sum(&data, simd::SimdLevel::detect());
    assert_eq!(result, 42.0);
}

#[test]
fn test_dot_product_empty() {
    let a: Vec<f64> = vec![];
    let b: Vec<f64> = vec![];

    let result = simd::dot_product(&a, &b, simd::SimdLevel::detect());
    assert_eq!(result, 0.0);
}

#[test]
fn test_dot_product_single() {
    let a: Vec<f64> = vec![5.0];
    let b: Vec<f64> = vec![3.0];

    let result = simd::dot_product(&a, &b, simd::SimdLevel::detect());
    assert_eq!(result, 15.0);
}
