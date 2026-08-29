use super::support::{ohlc_fixture, REPEATED_SERIES_LEN as PROFILE_SIZE};
use super::{assert_profile, assert_zero_allocations, print_profile};
use fast_ta::momentum::{
    AROONConfig, AROONOSCConfig, AROONValuesMut, AroonInput, STOCHConfig, STOCHFConfig,
    STOCHFValuesMut, STOCHRSIConfig, STOCHRSIValuesMut, STOCHValuesMut, StochasticInput,
    WILLRConfig,
};
use fast_ta::overlap::PeriodMAType;
use fast_ta::{Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation};
use std::hint::black_box;
const AROON_PERIOD: usize = 14;
const FAST_K_PERIOD: usize = 14;
const SLOW_PERIOD: usize = 3;
const RSI_PERIOD: usize = 14;

pub(super) fn profile_range_position_execution() {
    let ohlc = ohlc_fixture(PROFILE_SIZE);
    let float_bytes = core::mem::size_of::<Float>();

    // ---- AROON: two equal-length named columns. Caller-owned incurs the
    // RangeExtremaScratch (2 Vec<usize> monotonic deques). ----
    let aroon = AROONConfig::new(AROON_PERIOD).unwrap();
    let aroon_count = PROFILE_SIZE - aroon.lookback();
    let aroon_bytes = aroon_count * float_bytes;
    let scratch_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let mut down = vec![0.0 as Float; aroon_count];
    let mut up = vec![0.0 as Float; aroon_count];
    let profile = print_profile("one_shot/AROONConfig/caller_compact", || {
        aroon
            .compute_into(
                AroonInput {
                    high: ohlc.high.as_slice(),
                    low: ohlc.low.as_slice(),
                },
                AROONValuesMut {
                    down: &mut down,
                    up: &mut up,
                },
            )
            .unwrap()
    });
    assert_profile(
        "one_shot/AROONConfig/caller_compact",
        profile,
        2,
        scratch_bytes,
        scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/AROONConfig/owned_compact", || {
        aroon
            .compute(AroonInput {
                high: ohlc.high.as_slice(),
                low: ohlc.low.as_slice(),
            })
            .unwrap()
    });
    assert_profile(
        "one_shot/AROONConfig/owned_compact",
        profile,
        4,
        2 * aroon_bytes + scratch_bytes,
        2 * aroon_bytes + scratch_bytes,
        2 * aroon_bytes,
    );
    // ---- AROONOSC: single column. Caller-owned incurs the same
    // RangeExtremaScratch overhead as AROON. ----
    let aroon_osc = AROONOSCConfig::new(AROON_PERIOD).unwrap();
    let osc_count = PROFILE_SIZE - aroon_osc.lookback();
    let osc_bytes = osc_count * float_bytes;
    let osc_scratch_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let mut osc = vec![0.0 as Float; osc_count];
    let profile = print_profile("one_shot/AROONOSCConfig/caller_compact", || {
        aroon_osc
            .compute_into(
                AroonInput {
                    high: ohlc.high.as_slice(),
                    low: ohlc.low.as_slice(),
                },
                &mut osc,
            )
            .unwrap()
    });
    assert_profile(
        "one_shot/AROONOSCConfig/caller_compact",
        profile,
        2,
        osc_scratch_bytes,
        osc_scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/AROONOSCConfig/owned_compact", || {
        aroon_osc
            .compute(AroonInput {
                high: ohlc.high.as_slice(),
                low: ohlc.low.as_slice(),
            })
            .unwrap()
    });
    assert_profile(
        "one_shot/AROONOSCConfig/owned_compact",
        profile,
        3,
        osc_bytes + osc_scratch_bytes,
        osc_bytes + osc_scratch_bytes,
        osc_bytes,
    );

    // Helper to build a stochastic input view borrowing the fixture vectors.
    let stochastic = StochasticInput {
        high: ohlc.high.as_slice(),
        low: ohlc.low.as_slice(),
        close: ohlc.close.as_slice(),
    };

    // ---- STOCH: two equal-length named columns. Caller-owned incurs the
    // StochasticScratch (RangeExtremaScratch + raw_k + smoothed_k Vec<Float>)
    // overhead per the documented scratch contract. ----
    let stoch = STOCHConfig::new(
        FAST_K_PERIOD,
        SLOW_PERIOD,
        PeriodMAType::SMA,
        SLOW_PERIOD,
        PeriodMAType::SMA,
    )
    .unwrap();
    let stoch_count = PROFILE_SIZE - stoch.lookback();
    let stoch_bytes = stoch_count * float_bytes;
    let stoch_extrema_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let stoch_scratch_bytes = stoch_extrema_bytes + 2 * PROFILE_SIZE * float_bytes;
    let mut slow_k = vec![0.0 as Float; stoch_count];
    let mut slow_d = vec![0.0 as Float; stoch_count];
    let profile = print_profile("one_shot/STOCHConfig/caller_compact", || {
        stoch.compute_into(
            stochastic,
            STOCHValuesMut {
                slow_k: &mut slow_k,
                slow_d: &mut slow_d,
            },
        )
    });
    assert_profile(
        "one_shot/STOCHConfig/caller_compact",
        profile,
        4,
        stoch_scratch_bytes,
        stoch_scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/STOCHConfig/owned_compact", || {
        stoch.compute(stochastic).unwrap()
    });
    assert_profile(
        "one_shot/STOCHConfig/owned_compact",
        profile,
        6,
        2 * stoch_bytes + stoch_scratch_bytes,
        2 * stoch_bytes + stoch_scratch_bytes,
        2 * stoch_bytes,
    );
    let stochf = STOCHFConfig::new(FAST_K_PERIOD, SLOW_PERIOD, PeriodMAType::SMA).unwrap();
    let stochf_count = PROFILE_SIZE - stochf.lookback();
    let stochf_bytes = stochf_count * float_bytes;
    let stochf_extrema_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let stochf_scratch_bytes = stochf_extrema_bytes + PROFILE_SIZE * float_bytes;
    let mut fast_k = vec![0.0 as Float; stochf_count];
    let mut fast_d = vec![0.0 as Float; stochf_count];
    let profile = print_profile("one_shot/STOCHFConfig/caller_compact", || {
        stochf
            .compute_into(
                stochastic,
                STOCHFValuesMut {
                    fast_k: &mut fast_k,
                    fast_d: &mut fast_d,
                },
            )
            .unwrap()
    });
    assert_profile(
        "one_shot/STOCHFConfig/caller_compact",
        profile,
        3,
        stochf_scratch_bytes,
        stochf_scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/STOCHFConfig/owned_compact", || {
        stochf.compute(stochastic).unwrap()
    });
    assert_profile(
        "one_shot/STOCHFConfig/owned_compact",
        profile,
        5,
        2 * stochf_bytes + stochf_scratch_bytes,
        2 * stochf_bytes + stochf_scratch_bytes,
        2 * stochf_bytes,
    );
    let stochrsi =
        STOCHRSIConfig::new(RSI_PERIOD, FAST_K_PERIOD, SLOW_PERIOD, PeriodMAType::SMA).unwrap();
    let rsi_count = PROFILE_SIZE - stochrsi.lookback();
    let rsi_bytes = rsi_count * float_bytes;
    // STOCHRSIScratch owns 1 rsi Vec<Float> + StochasticScratch
    // (RangeExtremaScratch + 1 raw_k Vec<Float>).
    let stochrsi_extrema_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let stochrsi_scratch_bytes =
        PROFILE_SIZE * float_bytes + stochrsi_extrema_bytes + PROFILE_SIZE * float_bytes;
    let mut rsi_k = vec![0.0 as Float; rsi_count];
    let mut rsi_d = vec![0.0 as Float; rsi_count];
    let profile = print_profile("one_shot/STOCHRSIConfig/caller_compact", || {
        stochrsi
            .compute_into(
                ohlc.close.as_slice(),
                STOCHRSIValuesMut {
                    fast_k: &mut rsi_k,
                    fast_d: &mut rsi_d,
                },
            )
            .unwrap()
    });
    assert_profile(
        "one_shot/STOCHRSIConfig/caller_compact",
        profile,
        4,
        stochrsi_scratch_bytes,
        stochrsi_scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/STOCHRSIConfig/owned_compact", || {
        stochrsi.compute(ohlc.close.as_slice()).unwrap()
    });
    assert_profile(
        "one_shot/STOCHRSIConfig/owned_compact",
        profile,
        6,
        2 * rsi_bytes + stochrsi_scratch_bytes,
        2 * rsi_bytes + stochrsi_scratch_bytes,
        2 * rsi_bytes,
    );

    // ---- WILLR: single column ----
    let willr = WILLRConfig::new(AROON_PERIOD).unwrap();
    let willr_count = PROFILE_SIZE - willr.lookback();
    let willr_bytes = willr_count * float_bytes;
    let willr_scratch_bytes = PROFILE_SIZE * core::mem::size_of::<usize>() * 2;
    let mut willr_out = vec![0.0 as Float; willr_count];
    let profile = print_profile("one_shot/WILLRConfig/caller_compact", || {
        willr.compute_into(stochastic, &mut willr_out).unwrap()
    });
    assert_profile(
        "one_shot/WILLRConfig/caller_compact",
        profile,
        2,
        willr_scratch_bytes,
        willr_scratch_bytes,
        0,
    );
    let profile = print_profile("one_shot/WILLRConfig/owned_compact", || {
        willr.compute(stochastic).unwrap()
    });
    assert_profile(
        "one_shot/WILLRConfig/owned_compact",
        profile,
        3,
        willr_bytes + willr_scratch_bytes,
        willr_bytes + willr_scratch_bytes,
        willr_bytes,
    );

    // ---- Prepared Batch Runner reuse ----
    // STOCHBatchRunner reserves StochasticScratch (RangeExtremaScratch +
    // raw_k + smoothed_k) at preparation; the MABatchRunner sub-states are
    // zero-allocation storage.
    let stoch_setup_bytes =
        2 * PROFILE_SIZE * core::mem::size_of::<usize>() + 2 * PROFILE_SIZE * float_bytes;
    let profile = print_profile("setup/STOCHBatchRunner", || {
        stoch.prepare_batch(PROFILE_SIZE).unwrap()
    });
    assert_profile(
        "setup/STOCHBatchRunner",
        profile,
        4,
        stoch_setup_bytes,
        stoch_setup_bytes,
        stoch_setup_bytes,
    );
    let mut runner = stoch.prepare_batch(PROFILE_SIZE).unwrap();
    for pass in ["first", "repeated"] {
        let scenario = format!("prepared/STOCHConfig/{pass}");
        let profile = print_profile(&scenario, || {
            runner
                .compute_into(
                    stochastic,
                    STOCHValuesMut {
                        slow_k: &mut slow_k,
                        slow_d: &mut slow_d,
                    },
                )
                .unwrap()
        });
        assert_zero_allocations(&scenario, profile);
    }

    // ---- Streaming ----
    // AROON/STOCHRSI/WILLR streams each reserve a small range-extrema
    // scratch Vec<usize> at creation; the streaming tick loop is allocation
    // free because the deque is reused in-place.
    let range_observation_bytes =
        AROON_PERIOD * (2 * core::mem::size_of::<Float>() + core::mem::size_of::<usize>());
    // Vec header for the in-place deque (24 bytes on 64-bit).
    let range_stream_bytes = range_observation_bytes + 3 * core::mem::size_of::<usize>();
    let profile = print_profile("setup/AROONConfig/stream", || aroon.stream().unwrap());
    assert_profile(
        "setup/AROONConfig/stream",
        profile,
        1,
        range_stream_bytes,
        range_stream_bytes,
        range_stream_bytes,
    );
    let mut aroon_stream = aroon.stream().unwrap();
    let profile = print_profile("streaming/AROONConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                aroon_stream
                    .next(fast_ta::momentum::AroonTick {
                        high: ohlc.high[index],
                        low: ohlc.low[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/AROONConfig/ticks", profile);
    // STOCHRSI stream composes RangeStream (one Vec<RangeObservation> with
    // period+1 observations) and the fast-d MAStream scratch Vec. The
    // RSIStream and WilderGainLoss smoother are zero-allocation. Streaming
    // tick reuse keeps this fixed.
    let profile = print_profile("setup/STOCHRSIConfig/stream", || stochrsi.stream().unwrap());
    let stochrsi_setup_ops = 2;
    let stochrsi_setup_bytes = 360 + 8;
    assert_profile(
        "setup/STOCHRSIConfig/stream",
        profile,
        stochrsi_setup_ops,
        stochrsi_setup_bytes,
        stochrsi_setup_bytes,
        stochrsi_setup_bytes,
    );
    let mut stochrsi_stream = stochrsi.stream().unwrap();
    let profile = print_profile("streaming/STOCHRSIConfig/ticks", || {
        for value in ohlc.close.iter().copied() {
            black_box(stochrsi_stream.next(value).unwrap());
        }
    });
    assert_zero_allocations("streaming/STOCHRSIConfig/ticks", profile);

    let profile = print_profile("setup/WILLRConfig/stream", || willr.stream().unwrap());
    assert_profile(
        "setup/WILLRConfig/stream",
        profile,
        1,
        AROON_PERIOD * (2 * core::mem::size_of::<Float>() + core::mem::size_of::<usize>()),
        AROON_PERIOD * (2 * core::mem::size_of::<Float>() + core::mem::size_of::<usize>()),
        AROON_PERIOD * (2 * core::mem::size_of::<Float>() + core::mem::size_of::<usize>()),
    );
    let mut willr_stream = willr.stream().unwrap();
    let profile = print_profile("streaming/WILLRConfig/ticks", || {
        for index in 0..PROFILE_SIZE {
            black_box(
                willr_stream
                    .next(fast_ta::momentum::StochasticTick {
                        high: ohlc.high[index],
                        low: ohlc.low[index],
                        close: ohlc.close[index],
                    })
                    .unwrap(),
            );
        }
    });
    assert_zero_allocations("streaming/WILLRConfig/ticks", profile);
}
