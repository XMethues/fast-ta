use ta_core::price_transform::{
    AVGDEVConfig, AVGPRICEConfig, AVGPRICEInput, MEDPRICEConfig, MEDPRICEInput, TYPPRICEConfig,
    TYPPRICEInput, WCLPRICEConfig, WCLPRICEInput, AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE,
};
use ta_core::{Float, IndicatorConfig, OutputRange, PreparedBatchRunner, TalibError};

fn assert_close(actual: Float, expected: Float) {
    assert!(
        (actual - expected).abs() <= 1e-10 as Float,
        "expected {expected}, got {actual}"
    );
}

fn typprice_simd_lane_width() -> usize {
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

#[test]
fn avgprice_medprice_typprice_wclprice_compute_expected_values() {
    let open = [1.0, 2.0, 3.0];
    let high = [2.0, 3.0, 4.0];
    let low = [0.0, 1.0, 2.0];
    let close = [1.5, 2.5, 3.5];
    let mut output = [0.0; 3];

    assert_eq!(
        AVGPRICE(&open, &high, &low, &close, &mut output).unwrap(),
        OutputRange::new(0, 3)
    );
    assert_close(output[0], 1.125);
    assert_close(output[2], 3.125);

    MEDPRICE(&high, &low, &mut output).unwrap();
    assert_close(output[0], 1.0);
    assert_close(output[2], 3.0);

    TYPPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 7.0 as Float / 6.0 as Float);
    assert_close(output[2], 19.0 as Float / 6.0 as Float);

    WCLPRICE(&high, &low, &close, &mut output).unwrap();
    assert_close(output[0], 1.25);
    assert_close(output[2], 3.25);
}

#[test]
fn avgdev_computes_compact_outputs() {
    let real = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0; 5];

    let range = AVGDEV(&real, 3, &mut output).unwrap();
    assert_eq!(range, OutputRange::new(2, 3));
    assert_close(output[0], 2.0 / 3.0);
    assert_close(output[2], 2.0 / 3.0);
}

#[test]
fn price_transform_rejects_bad_lengths_and_non_finite_inputs() {
    let mut output = [0.0; 3];
    assert!(MEDPRICE(&[1.0, 2.0], &[1.0], &mut output).is_err());
    assert!(TYPPRICE(&[1.0, Float::NAN], &[1.0, 2.0], &[1.0, 2.0], &mut output).is_err());
    assert!(AVGDEV(&[1.0, 2.0], 3, &mut output).is_err());
    assert!(AVGDEVConfig::new(0).is_err());
}

#[test]
fn prepared_capacity_precedes_price_transform_input_alignment() {
    let within = [1.0 as Float; 2];
    let oversized = [1.0 as Float; 3];
    let mut output = [];
    let capacity_error = TalibError::PreparedCapacityExceeded {
        max_input_len: within.len(),
        actual_input_len: oversized.len(),
    };

    let mut avgprice = AVGPRICEConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        avgprice
            .compute_into(
                AVGPRICEInput {
                    open: &within,
                    high: &oversized,
                    low: &within,
                    close: &within,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );

    let mut medprice = MEDPRICEConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        medprice
            .compute_into(
                MEDPRICEInput {
                    high: &within,
                    low: &oversized,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );

    let mut typprice = TYPPRICEConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        typprice
            .compute_into(
                TYPPRICEInput {
                    high: &within,
                    low: &within,
                    close: &oversized,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );

    let mut wclprice = WCLPRICEConfig::new().prepare_batch(within.len()).unwrap();
    assert_eq!(
        wclprice
            .compute_into(
                WCLPRICEInput {
                    high: &within,
                    low: &oversized,
                    close: &within,
                },
                &mut output,
            )
            .unwrap_err(),
        capacity_error
    );
}

#[test]
fn typprice_public_batch_matches_scalar_at_simd_boundaries() {
    let lane_width = typprice_simd_lane_width();
    let lengths = [
        0,
        1,
        lane_width - 1,
        lane_width,
        lane_width + 1,
        lane_width * 2 + 1,
        65_537,
    ];
    let config = TYPPRICEConfig::new();
    const SENTINEL: Float = -12_345.0 as Float;

    for len in lengths {
        let high: Vec<Float> = (0..len)
            .map(|index| index as Float * 0.5 as Float + 3.25 as Float)
            .collect();
        let low: Vec<Float> = (0..len)
            .map(|index| index as Float * 0.25 as Float - 1.5 as Float)
            .collect();
        let close: Vec<Float> = (0..len)
            .map(|index| index as Float * 0.125 as Float + 0.75 as Float)
            .collect();
        let expected: Vec<Float> = (0..len)
            .map(|index| (high[index] + low[index] + close[index]) / 3.0 as Float)
            .collect();
        let input = TYPPRICEInput {
            high: &high,
            low: &low,
            close: &close,
        };

        let owned = config.compute(input).unwrap();
        assert_eq!(owned.range(), OutputRange::new(0, len), "owned len {len}");
        assert_eq!(owned.values(), expected.as_slice(), "owned len {len}");

        let mut caller_owned = vec![SENTINEL; len + 1];
        assert_eq!(
            config.compute_into(input, &mut caller_owned).unwrap(),
            OutputRange::new(0, len),
            "caller-owned len {len}"
        );
        assert_eq!(
            &caller_owned[..len],
            expected.as_slice(),
            "caller-owned len {len}"
        );
        assert_eq!(caller_owned[len], SENTINEL, "caller-owned tail len {len}");

        let mut prepared = config.prepare_batch(len).unwrap();
        let mut prepared_output = vec![SENTINEL; len + 1];
        assert_eq!(
            prepared.compute_into(input, &mut prepared_output).unwrap(),
            OutputRange::new(0, len),
            "prepared len {len}"
        );
        assert_eq!(
            &prepared_output[..len],
            expected.as_slice(),
            "prepared len {len}"
        );
        assert_eq!(prepared_output[len], SENTINEL, "prepared tail len {len}");
    }
}

#[test]
fn typprice_validates_all_input_before_simd_output_mutation() {
    let lane_width = typprice_simd_lane_width();
    let len = lane_width * 2 + 1;
    let mut high = vec![3.0 as Float; len];
    let low = vec![1.0 as Float; len];
    let close = vec![2.0 as Float; len];
    high[lane_width + 1] = Float::NAN;
    let mut output = vec![-12_345.0 as Float; len];

    assert!(TYPPRICEConfig::new()
        .compute_into(
            TYPPRICEInput {
                high: &high,
                low: &low,
                close: &close,
            },
            &mut output,
        )
        .is_err());
    assert_eq!(output, vec![-12_345.0 as Float; len]);
}

#[test]
fn typprice_non_finite_errors_preserve_named_slice_order_index_value_and_output() {
    let lane_width = typprice_simd_lane_width();
    let len = lane_width * 4 + 3;
    const SENTINEL: Float = -12_345.0 as Float;

    for (name, index, value) in [
        ("high", 0, Float::NAN),
        ("low", lane_width * 4 - 1, Float::INFINITY),
        ("close", len - 1, Float::NEG_INFINITY),
    ] {
        let mut high = vec![3.0 as Float; len];
        let mut low = vec![1.0 as Float; len];
        let mut close = vec![2.0 as Float; len];
        match name {
            "high" => high[index] = value,
            "low" => low[index] = value,
            "close" => close[index] = value,
            _ => unreachable!(),
        }
        let mut output = vec![SENTINEL; len];

        assert_eq!(
            TYPPRICE(&high, &low, &close, &mut output).unwrap_err(),
            TalibError::invalid_input(format!("{name}[{index}] must be finite, got {value}"))
        );
        assert_eq!(output, vec![SENTINEL; len]);
    }

    let mut high = vec![3.0 as Float; len];
    let mut low = vec![1.0 as Float; len];
    let mut close = vec![2.0 as Float; len];
    high[len - 1] = Float::INFINITY;
    low[0] = Float::NAN;
    close[0] = Float::NEG_INFINITY;
    let mut output = vec![SENTINEL; len];

    assert_eq!(
        TYPPRICE(&high, &low, &close, &mut output).unwrap_err(),
        TalibError::invalid_input(format!(
            "high[{}] must be finite, got {}",
            len - 1,
            Float::INFINITY
        ))
    );
    assert_eq!(output, vec![SENTINEL; len]);
}
