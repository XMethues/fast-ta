//! Compile-time and run-time checks for the code samples in
//! `docs/agents/migration-guide.md`. Each section of the guide exposes a
//! representative workload pattern that the contract tests below exercise.

use ta_core::math_operators::{MINMAXConfig, MINMAXValuesMut};
use ta_core::overlap::SMAConfig;
use ta_core::price_transform::{AVGPRICEConfig, AVGPRICEInput};
use ta_core::statistic::{CORRELConfig, CORRELStream, PairTick};
use ta_core::volatility::{ATRConfig, ATRInput};
use ta_core::{
    CompactOutput, Float, IndicatorConfig, OutputRange, PreparedBatchRunner, StreamingComputation,
};

const FIXTURE: &[Float] = &[
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0,
];

#[test]
fn one_shot_owned_compact_walk_pattern_matches_guide() {
    // Section 1: Compact Output.
    let config = SMAConfig::new(14).unwrap();
    let result: CompactOutput<Vec<Float>> = config.compute(FIXTURE).unwrap();
    let source_len = result.source_len();
    let range: OutputRange = result.range();
    let values = result.values();

    assert_eq!(source_len, FIXTURE.len());
    assert_eq!(values.len(), range.nb_element);
    assert_eq!(range.beg_idx + range.nb_element, source_len);
}

#[test]
fn caller_owned_compute_into_matches_guide() {
    // Section 1: caller-owned output.
    let config = SMAConfig::new(14).unwrap();
    let count = FIXTURE.len() - config.lookback();
    let mut owned = vec![0.0 as Float; count];
    let range = config.compute_into(FIXTURE, &mut owned).unwrap();
    assert_eq!(owned.len(), range.nb_element);
}

#[test]
fn aligned_walk_pattern_matches_guide() {
    // Section 2: walking Compact Output into a source-length buffer.
    let config = SMAConfig::new(14).unwrap();
    let result = config.compute(FIXTURE).unwrap();
    let source_len = result.source_len();
    let range = result.range();
    let compact = result.values();

    let mut aligned: Vec<Option<Float>> = vec![None; source_len];
    for (i, value) in compact.iter().enumerate() {
        aligned[range.beg_idx + i] = Some(*value);
    }

    assert_eq!(aligned.len(), source_len);
    for &value in compact {
        assert!(aligned.contains(&Some(value)));
    }
}

#[test]
fn prepared_runner_capacity_pattern_matches_guide() {
    // Section 3: Prepared Batch Runner capacity.
    let config = SMAConfig::new(14).unwrap();
    let mut runner = config.prepare_batch(4096).unwrap();
    assert_eq!(runner.max_input_len(), 4096);

    let count = FIXTURE.len() - config.lookback();
    let mut owned = vec![0.0 as Float; count];
    let range = runner.compute_into(FIXTURE, &mut owned).unwrap();
    assert_eq!(range.nb_element, count);
}

#[test]
fn multi_instrument_stream_pattern_matches_guide() {
    // Section 4: independent stream state per instrument.
    let config = SMAConfig::new(14).unwrap();
    let mut aapl = config.stream().unwrap();
    let mut msft = config.stream().unwrap();

    // Warm-up returns `Ok(None)`; the two streams are independent.
    for tick in &FIXTURE[..13] {
        assert!(aapl.next(*tick).unwrap().is_none());
        assert!(msft.next(*tick).unwrap().is_none());
    }
    for tick in &FIXTURE[13..] {
        let aapl_value = aapl.next(*tick).unwrap().expect("post-warm-up tick");
        let msft_value = msft.next(*tick).unwrap().expect("post-warm-up tick");
        // Same input sequence on each stream produces the same output.
        assert!((aapl_value - msft_value).abs() < 1e-9);
    }

    aapl.reset();
    assert!(aapl.next(FIXTURE[0]).unwrap().is_none());
}

#[test]
fn minmax_multi_output_pattern_matches_guide() {
    // Section "Per-worker Prepared Batch Runner" uses MINMAX named columns.
    let config = MINMAXConfig::new(14).unwrap();
    let count = FIXTURE.len() - config.lookback();
    let mut min = vec![0.0 as Float; count];
    let mut max = vec![0.0 as Float; count];
    let out = MINMAXValuesMut {
        min: &mut min,
        max: &mut max,
    };
    let range = config.compute_into(FIXTURE, out).unwrap();
    assert_eq!(min.len(), range.nb_element);
    assert_eq!(max.len(), range.nb_element);
}

#[test]
fn paired_input_stream_pattern_matches_guide() {
    // Section 4 paired-input streams.
    let config = CORRELConfig::new(14).unwrap();
    let mut stream: CORRELStream = config.stream().unwrap();
    let mut seen_values = Vec::new();
    for &value in FIXTURE {
        let tick = PairTick {
            real0: value,
            real1: value,
        };
        if let Some(out) = stream.next(tick).unwrap() {
            seen_values.push(out);
        }
    }
    // Constant inputs must produce a correlation of exactly 1.0.
    assert!(!seen_values.is_empty());
    for value in seen_values {
        assert!((value - 1.0 as Float).abs() < 1e-9);
    }
}

#[test]
fn multi_input_compute_into_uses_named_struct() {
    // Sections 1 and 4 also cover multi-input indicators.
    let config = AVGPRICEConfig::new();
    let input = AVGPRICEInput {
        open: FIXTURE,
        high: FIXTURE,
        low: FIXTURE,
        close: FIXTURE,
    };
    let mut out = vec![0.0 as Float; FIXTURE.len()];
    let range = config.compute_into(input, &mut out).unwrap();
    assert_eq!(range.nb_element, FIXTURE.len());
}

#[test]
fn volatility_multi_input_compute_into_uses_named_struct() {
    let config = ATRConfig::new(14).unwrap();
    let input = ATRInput {
        high: FIXTURE,
        low: FIXTURE,
        close: FIXTURE,
    };
    let count = FIXTURE.len() - config.lookback();
    let mut out = vec![0.0 as Float; count];
    let range = config.compute_into(input, &mut out).unwrap();
    assert_eq!(range.nb_element, count);
}

#[test]
fn parameter_sweep_pattern_matches_guide() {
    let prices: &[Float] = FIXTURE;
    let periods = [5usize, 8, 12];
    let configs: Vec<SMAConfig> = periods
        .iter()
        .map(|&p| SMAConfig::new(p).unwrap())
        .collect();
    // Each config owns a buffer sized to its own lookback so the output
    // capacity check passes for every sweep slot.
    let mut buffers: Vec<Vec<Float>> = configs
        .iter()
        .map(|cfg| vec![0.0 as Float; prices.len() - cfg.lookback()])
        .collect();
    for (cfg, buf) in configs.iter().zip(buffers.iter_mut()) {
        let _ = cfg.compute_into(prices, buf).unwrap();
    }
    for (cfg, buf) in configs.iter().zip(buffers.iter()) {
        assert_eq!(buf.len(), prices.len() - cfg.lookback());
    }
}
