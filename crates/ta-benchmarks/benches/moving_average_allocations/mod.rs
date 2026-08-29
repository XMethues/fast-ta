use super::support::{series_fixture, REPEATED_SERIES_LEN as PROFILE_SIZE};
use super::{assert_profile, assert_zero_allocations, print_profile};
use fast_ta::momentum::{
    APOConfig, MACDConfig, MACDEXTConfig, MACDFIXConfig, MACDValuesMut, PPOConfig, TRIXConfig,
};
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};
use std::hint::black_box;

pub(super) fn profile_moving_average_momentum_execution() {
    let real = series_fixture(PROFILE_SIZE, 31);
    let float_bytes = core::mem::size_of::<Float>();

    let apo = APOConfig::default();
    let pair_count = PROFILE_SIZE - apo.lookback();
    let pair_bytes = pair_count * float_bytes;
    let mut pair_output = vec![0.0 as Float; pair_count];
    let profile = print_profile("one_shot/APOConfig/caller_compact", || {
        apo.compute_into(&real, &mut pair_output).unwrap()
    });
    assert_zero_allocations("one_shot/APOConfig/caller_compact", profile);
    let profile = print_profile("one_shot/APOConfig/owned_compact", || {
        apo.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/APOConfig/owned_compact",
        profile,
        1,
        pair_bytes,
        pair_bytes,
        pair_bytes,
    );

    let ppo = PPOConfig::default();
    let profile = print_profile("one_shot/PPOConfig/caller_compact", || {
        ppo.compute_into(&real, &mut pair_output).unwrap()
    });
    assert_zero_allocations("one_shot/PPOConfig/caller_compact", profile);
    let profile = print_profile("one_shot/PPOConfig/owned_compact", || {
        ppo.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/PPOConfig/owned_compact",
        profile,
        1,
        pair_bytes,
        pair_bytes,
        pair_bytes,
    );

    let macd = MACDConfig::default();
    let macd_count = PROFILE_SIZE - macd.lookback();
    let column_bytes = macd_count * float_bytes;
    let mut macd_output = vec![0.0 as Float; macd_count];
    let mut signal_output = vec![0.0 as Float; macd_count];
    let mut histogram_output = vec![0.0 as Float; macd_count];
    let profile = print_profile("one_shot/MACDConfig/caller_compact", || {
        macd.compute_into(
            &real,
            MACDValuesMut {
                macd: &mut macd_output,
                signal: &mut signal_output,
                histogram: &mut histogram_output,
            },
        )
        .unwrap()
    });
    assert_zero_allocations("one_shot/MACDConfig/caller_compact", profile);
    let profile = print_profile("one_shot/MACDConfig/owned_compact", || {
        macd.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/MACDConfig/owned_compact",
        profile,
        3,
        3 * column_bytes,
        3 * column_bytes,
        3 * column_bytes,
    );

    let macdext = MACDEXTConfig::default();
    let profile = print_profile("one_shot/MACDEXTConfig/caller_compact", || {
        macdext
            .compute_into(
                &real,
                MACDValuesMut {
                    macd: &mut macd_output,
                    signal: &mut signal_output,
                    histogram: &mut histogram_output,
                },
            )
            .unwrap()
    });
    assert_zero_allocations("one_shot/MACDEXTConfig/caller_compact", profile);
    let profile = print_profile("one_shot/MACDEXTConfig/owned_compact", || {
        macdext.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/MACDEXTConfig/owned_compact",
        profile,
        3,
        3 * column_bytes,
        3 * column_bytes,
        3 * column_bytes,
    );

    let macdfix = MACDFIXConfig::default();
    let profile = print_profile("one_shot/MACDFIXConfig/caller_compact", || {
        macdfix
            .compute_into(
                &real,
                MACDValuesMut {
                    macd: &mut macd_output,
                    signal: &mut signal_output,
                    histogram: &mut histogram_output,
                },
            )
            .unwrap()
    });
    assert_zero_allocations("one_shot/MACDFIXConfig/caller_compact", profile);
    let profile = print_profile("one_shot/MACDFIXConfig/owned_compact", || {
        macdfix.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/MACDFIXConfig/owned_compact",
        profile,
        3,
        3 * column_bytes,
        3 * column_bytes,
        3 * column_bytes,
    );

    let trix = TRIXConfig::default();
    let trix_count = PROFILE_SIZE - trix.lookback();
    let trix_bytes = trix_count * float_bytes;
    let mut trix_output = vec![0.0 as Float; trix_count];
    let profile = print_profile("one_shot/TRIXConfig/caller_compact", || {
        trix.compute_into(&real, &mut trix_output).unwrap()
    });
    assert_zero_allocations("one_shot/TRIXConfig/caller_compact", profile);
    let profile = print_profile("one_shot/TRIXConfig/owned_compact", || {
        trix.compute(&real).unwrap()
    });
    assert_profile(
        "one_shot/TRIXConfig/owned_compact",
        profile,
        1,
        trix_bytes,
        trix_bytes,
        trix_bytes,
    );

    let profile = print_profile("setup/MACDEXTBatchRunner", || {
        macdext.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/MACDEXTBatchRunner", profile);
    let mut runner = macdext.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/MACDEXTConfig/{pass}");
        let profile = print_profile(&scenario, || {
            runner
                .compute_into(
                    &real,
                    MACDValuesMut {
                        macd: &mut macd_output,
                        signal: &mut signal_output,
                        histogram: &mut histogram_output,
                    },
                )
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }

    let profile = print_profile("setup/APOConfig/stream", || apo.stream().unwrap());
    assert_zero_allocations("setup/APOConfig/stream", profile);
    let mut apo_stream = apo.stream().unwrap();
    let profile = print_profile("streaming/APOConfig/ticks", || {
        for value in real.iter().copied() {
            black_box(apo_stream.next(value).unwrap());
        }
    });
    assert_zero_allocations("streaming/APOConfig/ticks", profile);

    let profile = print_profile("setup/MACDConfig/stream", || macd.stream().unwrap());
    assert_zero_allocations("setup/MACDConfig/stream", profile);
    let mut macd_stream = macd.stream().unwrap();
    let profile = print_profile("streaming/MACDConfig/ticks", || {
        for value in real.iter().copied() {
            black_box(macd_stream.next(value).unwrap());
        }
    });
    assert_zero_allocations("streaming/MACDConfig/ticks", profile);

    let profile = print_profile("setup/TRIXConfig/stream", || trix.stream().unwrap());
    assert_zero_allocations("setup/TRIXConfig/stream", profile);
    let mut trix_stream = trix.stream().unwrap();
    let profile = print_profile("streaming/TRIXConfig/ticks", || {
        for value in real.iter().copied() {
            black_box(trix_stream.next(value).unwrap());
        }
    });
    assert_zero_allocations("streaming/TRIXConfig/ticks", profile);
}
