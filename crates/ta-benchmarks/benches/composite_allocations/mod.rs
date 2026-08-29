use super::support::{ohlc_fixture, OhlcFixture, REPEATED_SERIES_LEN as PROFILE_SIZE};
use super::{assert_profile, assert_zero_allocations, print_profile};
use fast_ta::momentum::{
    BOPConfig, BOPInput, BOPTick, CCIConfig, CCIInput, CCITick, MFIConfig, MFIInput, MFITick,
    ULTOSCConfig, ULTOSCInput, ULTOSCTick,
};
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};
use std::hint::black_box;

#[inline]
fn bop_input(fixture: &OhlcFixture) -> BOPInput<'_> {
    BOPInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

#[inline]
fn cci_input(fixture: &OhlcFixture) -> CCIInput<'_> {
    CCIInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

#[inline]
fn mfi_input(fixture: &OhlcFixture) -> MFIInput<'_> {
    MFIInput {
        open: &fixture.open,
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
        volume: &fixture.volume,
    }
}

#[inline]
fn ultosc_input(fixture: &OhlcFixture) -> ULTOSCInput<'_> {
    ULTOSCInput {
        high: &fixture.high,
        low: &fixture.low,
        close: &fixture.close,
    }
}

pub(super) fn profile_composite_momentum_execution() {
    let fixture = ohlc_fixture(PROFILE_SIZE);
    let oversized = ohlc_fixture(PROFILE_SIZE + 1);
    let float_bytes = core::mem::size_of::<Float>();

    let profile = print_profile("setup/BOPConfig", BOPConfig::new);
    assert_zero_allocations("setup/BOPConfig", profile);
    let bop = BOPConfig::new();
    let bop_count = PROFILE_SIZE;
    let bop_bytes = bop_count * float_bytes;
    let mut bop_output = vec![0.0 as Float; bop_count];
    let profile = print_profile("one_shot/BOPConfig/caller_compact", || {
        bop.compute_into(bop_input(&fixture), bop_output.as_mut_slice())
            .unwrap()
    });
    assert_zero_allocations("one_shot/BOPConfig/caller_compact", profile);
    let profile = print_profile("one_shot/BOPConfig/owned_compact", || {
        bop.compute(bop_input(&fixture)).unwrap()
    });
    assert_profile(
        "one_shot/BOPConfig/owned_compact",
        profile,
        1,
        bop_bytes,
        bop_bytes,
        bop_bytes,
    );
    let profile = print_profile("one_shot/BOPConfig/owned_compact/count_0", || {
        bop.compute(BOPInput {
            open: &[],
            high: &[],
            low: &[],
            close: &[],
        })
        .unwrap()
    });
    assert_zero_allocations("one_shot/BOPConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/BOPBatchRunner", || {
        bop.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/BOPBatchRunner", profile);
    let mut bop_runner = bop.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/BOPConfig/{pass}");
        let profile = print_profile(&scenario, || {
            bop_runner
                .compute_into(bop_input(&fixture), bop_output.as_mut_slice())
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/BOPConfig/oversize_rejection", || {
        bop_runner
            .compute_into(bop_input(&oversized), bop_output.as_mut_slice())
            .unwrap_err()
    });
    assert_zero_allocations("prepared/BOPConfig/oversize_rejection", profile);
    let profile = print_profile("setup/BOPConfig/stream", || bop.stream().unwrap());
    assert_zero_allocations("setup/BOPConfig/stream", profile);
    let mut bop_stream = bop.stream().unwrap();
    let profile = print_profile("streaming/BOPConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                bop_stream
                    .next(BOPTick {
                        open: fixture.open[index],
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/BOPConfig/ticks", profile);

    let profile = print_profile("setup/CCIConfig", || CCIConfig::new(14).unwrap());
    assert_zero_allocations("setup/CCIConfig", profile);
    let cci = CCIConfig::new(14).unwrap();
    let cci_count = PROFILE_SIZE - cci.lookback();
    let cci_bytes = cci_count * float_bytes;
    let mut cci_output = vec![0.0 as Float; cci_count];
    let profile = print_profile("one_shot/CCIConfig/caller_compact", || {
        cci.compute_into(cci_input(&fixture), cci_output.as_mut_slice())
            .unwrap()
    });
    assert_zero_allocations("one_shot/CCIConfig/caller_compact", profile);
    let profile = print_profile("one_shot/CCIConfig/owned_compact", || {
        cci.compute(cci_input(&fixture)).unwrap()
    });
    assert_profile(
        "one_shot/CCIConfig/owned_compact",
        profile,
        1,
        cci_bytes,
        cci_bytes,
        cci_bytes,
    );
    let profile = print_profile("one_shot/CCIConfig/owned_compact/count_0", || {
        cci.compute(CCIInput {
            open: &[],
            high: &[],
            low: &[],
            close: &[],
        })
        .unwrap()
    });
    assert_zero_allocations("one_shot/CCIConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/CCIBatchRunner", || {
        cci.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/CCIBatchRunner", profile);
    let mut cci_runner = cci.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/CCIConfig/{pass}");
        let profile = print_profile(&scenario, || {
            cci_runner
                .compute_into(cci_input(&fixture), cci_output.as_mut_slice())
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/CCIConfig/oversize_rejection", || {
        cci_runner
            .compute_into(cci_input(&oversized), cci_output.as_mut_slice())
            .unwrap_err()
    });
    assert_zero_allocations("prepared/CCIConfig/oversize_rejection", profile);
    let cci_stream_bytes = 14 * float_bytes;
    let profile = print_profile("setup/CCIConfig/stream", || cci.stream().unwrap());
    assert_profile(
        "setup/CCIConfig/stream",
        profile,
        1,
        cci_stream_bytes,
        cci_stream_bytes,
        cci_stream_bytes,
    );
    let mut cci_stream = cci.stream().unwrap();
    let profile = print_profile("streaming/CCIConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                cci_stream
                    .next(CCITick {
                        open: fixture.open[index],
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/CCIConfig/ticks", profile);

    let profile = print_profile("setup/MFIConfig", || MFIConfig::new(14).unwrap());
    assert_zero_allocations("setup/MFIConfig", profile);
    let mfi = MFIConfig::new(14).unwrap();
    let mfi_count = PROFILE_SIZE - mfi.lookback();
    let mfi_bytes = mfi_count * float_bytes;
    let mut mfi_output = vec![0.0 as Float; mfi_count];
    let profile = print_profile("one_shot/MFIConfig/caller_compact", || {
        mfi.compute_into(mfi_input(&fixture), mfi_output.as_mut_slice())
            .unwrap()
    });
    assert_zero_allocations("one_shot/MFIConfig/caller_compact", profile);
    let profile = print_profile("one_shot/MFIConfig/owned_compact", || {
        mfi.compute(mfi_input(&fixture)).unwrap()
    });
    assert_profile(
        "one_shot/MFIConfig/owned_compact",
        profile,
        1,
        mfi_bytes,
        mfi_bytes,
        mfi_bytes,
    );
    let profile = print_profile("one_shot/MFIConfig/owned_compact/count_0", || {
        mfi.compute(MFIInput {
            open: &[],
            high: &[],
            low: &[],
            close: &[],
            volume: &[],
        })
        .unwrap()
    });
    assert_zero_allocations("one_shot/MFIConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/MFIBatchRunner", || {
        mfi.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/MFIBatchRunner", profile);
    let mut mfi_runner = mfi.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/MFIConfig/{pass}");
        let profile = print_profile(&scenario, || {
            mfi_runner
                .compute_into(mfi_input(&fixture), mfi_output.as_mut_slice())
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/MFIConfig/oversize_rejection", || {
        mfi_runner
            .compute_into(mfi_input(&oversized), mfi_output.as_mut_slice())
            .unwrap_err()
    });
    assert_zero_allocations("prepared/MFIConfig/oversize_rejection", profile);
    let mfi_stream_bytes = 2 * 14 * float_bytes;
    let profile = print_profile("setup/MFIConfig/stream", || mfi.stream().unwrap());
    assert_profile(
        "setup/MFIConfig/stream",
        profile,
        1,
        mfi_stream_bytes,
        mfi_stream_bytes,
        mfi_stream_bytes,
    );
    let mut mfi_stream = mfi.stream().unwrap();
    let profile = print_profile("streaming/MFIConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                mfi_stream
                    .next(MFITick {
                        open: fixture.open[index],
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                        volume: fixture.volume[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/MFIConfig/ticks", profile);

    let profile = print_profile("setup/ULTOSCConfig", || {
        ULTOSCConfig::new(7, 14, 28).unwrap()
    });
    assert_zero_allocations("setup/ULTOSCConfig", profile);
    let ultosc = ULTOSCConfig::new(7, 14, 28).unwrap();
    let ultosc_count = PROFILE_SIZE - ultosc.lookback();
    let ultosc_bytes = ultosc_count * float_bytes;
    let mut ultosc_output = vec![0.0 as Float; ultosc_count];
    let profile = print_profile("one_shot/ULTOSCConfig/caller_compact", || {
        ultosc
            .compute_into(ultosc_input(&fixture), ultosc_output.as_mut_slice())
            .unwrap()
    });
    assert_zero_allocations("one_shot/ULTOSCConfig/caller_compact", profile);
    let profile = print_profile("one_shot/ULTOSCConfig/owned_compact", || {
        ultosc.compute(ultosc_input(&fixture)).unwrap()
    });
    assert_profile(
        "one_shot/ULTOSCConfig/owned_compact",
        profile,
        1,
        ultosc_bytes,
        ultosc_bytes,
        ultosc_bytes,
    );
    let profile = print_profile("one_shot/ULTOSCConfig/owned_compact/count_0", || {
        ultosc
            .compute(ULTOSCInput {
                high: &[],
                low: &[],
                close: &[],
            })
            .unwrap()
    });
    assert_zero_allocations("one_shot/ULTOSCConfig/owned_compact/count_0", profile);
    let profile = print_profile("setup/ULTOSCBatchRunner", || {
        ultosc.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_zero_allocations("setup/ULTOSCBatchRunner", profile);
    let mut ultosc_runner = ultosc.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/ULTOSCConfig/{pass}");
        let profile = print_profile(&scenario, || {
            ultosc_runner
                .compute_into(ultosc_input(&fixture), ultosc_output.as_mut_slice())
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }
    let profile = print_profile("prepared/ULTOSCConfig/oversize_rejection", || {
        ultosc_runner
            .compute_into(ultosc_input(&oversized), ultosc_output.as_mut_slice())
            .unwrap_err()
    });
    assert_zero_allocations("prepared/ULTOSCConfig/oversize_rejection", profile);
    let ultosc_stream_bytes = 2 * 28 * float_bytes;
    let profile = print_profile("setup/ULTOSCConfig/stream", || ultosc.stream().unwrap());
    assert_profile(
        "setup/ULTOSCConfig/stream",
        profile,
        1,
        ultosc_stream_bytes,
        ultosc_stream_bytes,
        ultosc_stream_bytes,
    );
    let mut ultosc_stream = ultosc.stream().unwrap();
    let profile = print_profile("streaming/ULTOSCConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                ultosc_stream
                    .next(ULTOSCTick {
                        high: fixture.high[index],
                        low: fixture.low[index],
                        close: fixture.close[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/ULTOSCConfig/ticks", profile);
}
