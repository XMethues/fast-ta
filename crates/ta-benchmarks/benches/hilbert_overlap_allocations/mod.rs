use super::support::{series_fixture, REPEATED_SERIES_LEN as PROFILE_SIZE};
use super::{assert_profile, assert_zero_allocations, print_profile};
use std::hint::black_box;
use ta_core::overlap::{HT_TRENDLINEConfig, MAMAConfig, MAMAValuesMut};
use ta_core::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};

pub(super) fn profile_hilbert_overlap_execution() {
    let input = series_fixture(PROFILE_SIZE, 29);
    let oversized = series_fixture(PROFILE_SIZE + 1, 30);
    let float_bytes = core::mem::size_of::<Float>();

    let profile = print_profile("setup/MAMAConfig", MAMAConfig::default);
    assert_zero_allocations("setup/MAMAConfig", profile);
    let mama_config = MAMAConfig::default();
    let mama_count = PROFILE_SIZE - mama_config.lookback();
    let mama_column_bytes = mama_count * float_bytes;
    let mama_total_bytes = 2 * mama_column_bytes;
    let mut mama = vec![0.0 as Float; mama_count];
    let mut fama = vec![0.0 as Float; mama_count];
    let profile = print_profile("one_shot/MAMAConfig/caller_compact", || {
        mama_config
            .compute_into(
                input.as_slice(),
                MAMAValuesMut {
                    mama: mama.as_mut_slice(),
                    fama: fama.as_mut_slice(),
                },
            )
            .unwrap()
    });
    assert_zero_allocations("one_shot/MAMAConfig/caller_compact", profile);
    let profile = print_profile("one_shot/MAMAConfig/owned_compact", || {
        mama_config.compute(input.as_slice()).unwrap()
    });
    assert_profile(
        "one_shot/MAMAConfig/owned_compact",
        profile,
        2,
        mama_total_bytes,
        mama_total_bytes,
        mama_total_bytes,
    );
    let profile = print_profile("one_shot/MAMAConfig/owned_compact/count_0", || {
        mama_config.compute(&[]).unwrap()
    });
    assert_zero_allocations("one_shot/MAMAConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/MAMABatchRunner", || {
        mama_config.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/MAMABatchRunner", profile);
    let mut mama_runner = mama_config.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/MAMAConfig/{pass}");
        let profile = print_profile(&scenario, || {
            mama_runner
                .compute_into(
                    input.as_slice(),
                    MAMAValuesMut {
                        mama: mama.as_mut_slice(),
                        fama: fama.as_mut_slice(),
                    },
                )
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/MAMAConfig/oversize_rejection", || {
        mama_runner
            .compute_into(
                oversized.as_slice(),
                MAMAValuesMut {
                    mama: mama.as_mut_slice(),
                    fama: fama.as_mut_slice(),
                },
            )
            .unwrap_err()
    });
    assert_zero_allocations("prepared/MAMAConfig/oversize_rejection", profile);
    let profile = print_profile("setup/MAMAConfig/stream", || mama_config.stream().unwrap());
    assert_zero_allocations("setup/MAMAConfig/stream", profile);
    let mut mama_stream = mama_config.stream().unwrap();
    let profile = print_profile("streaming/MAMAConfig/ticks", || {
        for &tick in &input {
            black_box(mama_stream.next(black_box(tick)).unwrap());
        }
    });
    assert_zero_allocations("streaming/MAMAConfig/ticks", profile);

    let profile = print_profile("setup/HT_TRENDLINEConfig", HT_TRENDLINEConfig::new);
    assert_zero_allocations("setup/HT_TRENDLINEConfig", profile);
    let trendline_config = HT_TRENDLINEConfig::new();
    let trendline_count = PROFILE_SIZE - trendline_config.lookback();
    let trendline_bytes = trendline_count * float_bytes;
    let mut trendline = vec![0.0 as Float; trendline_count];
    let profile = print_profile("one_shot/HT_TRENDLINEConfig/caller_compact", || {
        trendline_config
            .compute_into(input.as_slice(), trendline.as_mut_slice())
            .unwrap()
    });
    assert_zero_allocations("one_shot/HT_TRENDLINEConfig/caller_compact", profile);
    let profile = print_profile("one_shot/HT_TRENDLINEConfig/owned_compact", || {
        trendline_config.compute(input.as_slice()).unwrap()
    });
    assert_profile(
        "one_shot/HT_TRENDLINEConfig/owned_compact",
        profile,
        1,
        trendline_bytes,
        trendline_bytes,
        trendline_bytes,
    );
    let profile = print_profile("one_shot/HT_TRENDLINEConfig/owned_compact/count_0", || {
        trendline_config.compute(&[]).unwrap()
    });
    assert_zero_allocations("one_shot/HT_TRENDLINEConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/HT_TRENDLINEBatchRunner", || {
        trendline_config.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/HT_TRENDLINEBatchRunner", profile);
    let mut trendline_runner = trendline_config.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/HT_TRENDLINEConfig/{pass}");
        let profile = print_profile(&scenario, || {
            trendline_runner
                .compute_into(input.as_slice(), trendline.as_mut_slice())
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/HT_TRENDLINEConfig/oversize_rejection", || {
        trendline_runner
            .compute_into(oversized.as_slice(), trendline.as_mut_slice())
            .unwrap_err()
    });
    assert_zero_allocations("prepared/HT_TRENDLINEConfig/oversize_rejection", profile);
    let profile = print_profile("setup/HT_TRENDLINEConfig/stream", || {
        trendline_config.stream().unwrap()
    });
    assert_zero_allocations("setup/HT_TRENDLINEConfig/stream", profile);
    let mut trendline_stream = trendline_config.stream().unwrap();
    let profile = print_profile("streaming/HT_TRENDLINEConfig/ticks", || {
        for &tick in &input {
            black_box(trendline_stream.next(black_box(tick)).unwrap());
        }
    });
    assert_zero_allocations("streaming/HT_TRENDLINEConfig/ticks", profile);
}
