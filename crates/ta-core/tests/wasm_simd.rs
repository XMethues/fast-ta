#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd_tests {
    use ta_core::simd::arch::wasm32::simd128;

    fn create_test_data(size: usize) -> Vec<ta_core::types::Float> {
        (0..size).map(|i| i as ta_core::types::Float).collect()
    }

    #[test]
    fn test_sum_small() {
        let data: Vec<ta_core::types::Float> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        unsafe {
            let result = simd128::sum(&data);
            assert_eq!(result, 15.0);
        }
    }

    #[test]
    fn test_sum_large() {
        let data = create_test_data(1000);
        unsafe {
            let result = simd128::sum(&data);
            let expected: ta_core::types::Float = data.iter().sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sum_with_remainder() {
        let data = create_test_data(17);
        unsafe {
            let result = simd128::sum(&data);
            let expected: ta_core::types::Float = data.iter().sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sum_empty() {
        let data: Vec<ta_core::types::Float> = vec![];
        unsafe {
            let result = simd128::sum(&data);
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn test_sum_single_element() {
        let data: Vec<ta_core::types::Float> = vec![42.0];
        unsafe {
            let result = simd128::sum(&data);
            assert_eq!(result, 42.0);
        }
    }

    #[test]
    fn test_dot_product_small() {
        let a: Vec<ta_core::types::Float> = vec![1.0, 2.0, 3.0];
        let b: Vec<ta_core::types::Float> = vec![4.0, 5.0, 6.0];
        unsafe {
            let result = simd128::dot_product(&a, &b).unwrap();
            assert_eq!(result, 32.0);
        }
    }

    #[test]
    fn test_dot_product_large() {
        let a = create_test_data(1000);
        let b: Vec<ta_core::types::Float> = (0..1000)
            .map(|i| (i * 2) as ta_core::types::Float)
            .collect();
        unsafe {
            let result = simd128::dot_product(&a, &b).unwrap();
            let expected: ta_core::types::Float =
                a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dot_product_with_remainder() {
        let a = create_test_data(17);
        let b: Vec<ta_core::types::Float> =
            (0..17).map(|i| (i + 1) as ta_core::types::Float).collect();
        unsafe {
            let result = simd128::dot_product(&a, &b).unwrap();
            let expected: ta_core::types::Float =
                a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            assert!((result - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let a: Vec<ta_core::types::Float> = vec![1.0, 2.0, 3.0];
        let b: Vec<ta_core::types::Float> = vec![1.0, 2.0];
        unsafe {
            assert!(simd128::dot_product(&a, &b).is_err());
        }
    }

    #[test]
    fn test_dot_product_empty() {
        let a: Vec<ta_core::types::Float> = vec![];
        let b: Vec<ta_core::types::Float> = vec![];
        unsafe {
            let result = simd128::dot_product(&a, &b).unwrap();
            assert_eq!(result, 0.0);
        }
    }

    #[test]
    fn test_precision_f64() {
        let a: Vec<ta_core::types::Float> = vec![0.1, 0.2, 0.3];
        let b: Vec<ta_core::types::Float> = vec![1.0, 2.0, 3.0];
        unsafe {
            let result = simd128::dot_product(&a, &b).unwrap();
            let expected = 0.1 * 1.0 + 0.2 * 2.0 + 0.3 * 3.0;
            assert!((result - expected).abs() < 1e-10);
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod wasm_simd_tests {
    #[test]
    fn test_simd128_not_available() {
        println!("SIMD128 tests are skipped - not on WASM32 or SIMD not enabled");
    }
}
