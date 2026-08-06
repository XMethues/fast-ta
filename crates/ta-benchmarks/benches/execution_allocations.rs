//! Isolated allocation baselines for Rust-first Indicator execution.
//!
//! Run with `cargo bench -p ta-benchmarks --bench execution_allocations`.
//! Output is TSV so it can be copied into `EXECUTION_BASELINES.md`. Peak bytes
//! are the incremental requested-heap high-water mark during each measured
//! operation; they exclude stack, allocator metadata, RSS, and fixtures or
//! caller buffers allocated before measurement.

mod support;

use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering},
};
use support::{
    ohlc_fixture, output_len, series_fixture, PERIOD, REPEATED_SERIES_LEN as PROFILE_SIZE,
    STREAM_INSTRUMENTS, SWEEP_PERIODS, UNIVERSE_INSTRUMENTS, WORKERS,
};
use ta_core::{
    math_operators::{
        ADDConfig, BinaryInput, BinaryTick, DIVConfig, MAXConfig, MAXINDEXConfig, MINConfig,
        MININDEXConfig, MINMAXConfig, MINMAXINDEXConfig, MINMAXINDEXValuesMut, MINMAXValuesMut,
        MULTConfig, SUBConfig, SUMConfig, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX,
        MINMAXINDEX, MULT, SUB, SUM,
    },
    math_transform::{
        ACOSConfig, ASINConfig, ATANConfig, CEILConfig, COSConfig, COSHConfig, EXPConfig,
        FLOORConfig, LNConfig, LOG10Config, SINConfig, SINHConfig, SQRTConfig, TANConfig,
        TANHConfig, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN,
        TANH,
    },
    overlap::{
        DEMAConfig, EMAConfig, MAConfig, MAType, SMAConfig, T3Config, TEMAConfig, TRIMAConfig,
        WMAConfig, DEMA, EMA, MA, SMA, T3, TEMA, TRIMA, WMA,
    },
    price_transform::{
        AVGDEVConfig, AVGPRICEConfig, AVGPRICEInput, AVGPRICETick, MEDPRICEConfig, MEDPRICEInput,
        MEDPRICETick, TYPPRICEConfig, TYPPRICEInput, TYPPRICETick, WCLPRICEConfig, WCLPRICEInput,
        WCLPRICETick, AVGDEV, AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE,
    },
    statistic::{
        BETAConfig, CORRELConfig, LINEARREGConfig, LINEARREG_ANGLEConfig,
        LINEARREG_INTERCEPTConfig, LINEARREG_SLOPEConfig, PairInput, PairTick, STDDEVConfig,
        TSFConfig, VARConfig, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, LINEARREG_INTERCEPT,
        LINEARREG_SLOPE, STDDEV, TSF, VAR,
    },
    volatility::{
        ATRConfig, ATRInput, ATRTick, NATRConfig, NATRInput, NATRTick, TRANGEConfig, TRANGEInput,
        TRANGETick, ATR, NATR, TRANGE,
    },
    volume::{
        ADConfig, ADInput, ADOSCConfig, ADOSCInput, ADOSCTick, ADTick, OBVConfig, OBVInput,
        OBVTick, AD, ADOSC, OBV,
    },
    Float, IndicatorConfig, PreparedBatchRunner, StreamingComputation,
};

const REPEATED_MINMAX_CALLS: usize = 8;

struct TrackingAllocator;

static TRACKING: AtomicBool = AtomicBool::new(false);
static MEASUREMENT_ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATION_OPERATIONS: AtomicUsize = AtomicUsize::new(0);
static GROSS_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn record_live_delta(delta: isize) {
    let live = LIVE_BYTES.fetch_add(delta, Ordering::SeqCst) + delta;
    let mut peak = PEAK_LIVE_BYTES.load(Ordering::SeqCst);
    while live > peak {
        match PEAK_LIVE_BYTES.compare_exchange_weak(peak, live, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => break,
            Err(current) => peak = current,
        }
    }
}

fn record_allocation(size: usize) {
    ALLOCATION_OPERATIONS.fetch_add(1, Ordering::SeqCst);
    GROSS_ALLOCATED_BYTES.fetch_add(size, Ordering::SeqCst);
    record_live_delta(size as isize);
}

// SAFETY: every operation delegates to `System` with the exact pointer and
// layout supplied by the caller. Tracking changes only independent atomics.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACKING.load(Ordering::SeqCst) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && TRACKING.load(Ordering::SeqCst) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if TRACKING.load(Ordering::SeqCst) {
            record_live_delta(-(layout.size() as isize));
        }
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && TRACKING.load(Ordering::SeqCst) {
            ALLOCATION_OPERATIONS.fetch_add(1, Ordering::SeqCst);
            GROSS_ALLOCATED_BYTES.fetch_add(new_size, Ordering::SeqCst);
            record_live_delta(new_size as isize - layout.size() as isize);
        }
        new_pointer
    }
}

#[derive(Debug, Clone, Copy)]
struct AllocationProfile {
    operations: usize,
    gross_bytes: usize,
    peak_incremental_bytes: usize,
    retained_bytes: usize,
}

struct MeasurementGuard;

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        TRACKING.store(false, Ordering::SeqCst);
        MEASUREMENT_ACTIVE.store(false, Ordering::SeqCst);
    }
}

fn measure_allocations<T>(call: impl FnOnce() -> T) -> (T, AllocationProfile) {
    assert!(
        MEASUREMENT_ACTIVE
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok(),
        "allocation measurements must not overlap"
    );
    let guard = MeasurementGuard;

    ALLOCATION_OPERATIONS.store(0, Ordering::SeqCst);
    GROSS_ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    LIVE_BYTES.store(0, Ordering::SeqCst);
    PEAK_LIVE_BYTES.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);

    let value = call();

    TRACKING.store(false, Ordering::SeqCst);
    let profile = AllocationProfile {
        operations: ALLOCATION_OPERATIONS.load(Ordering::SeqCst),
        gross_bytes: GROSS_ALLOCATED_BYTES.load(Ordering::SeqCst),
        peak_incremental_bytes: PEAK_LIVE_BYTES.load(Ordering::SeqCst).max(0) as usize,
        retained_bytes: LIVE_BYTES.load(Ordering::SeqCst).max(0) as usize,
    };
    drop(guard);
    (value, profile)
}

fn print_profile<T>(scenario: &str, call: impl FnOnce() -> T) -> AllocationProfile {
    let (value, profile) = measure_allocations(call);
    black_box(&value);
    println!(
        "{scenario}\t{}\t{}\t{}\t{}",
        profile.operations,
        profile.gross_bytes,
        profile.peak_incremental_bytes,
        profile.retained_bytes
    );
    drop(value);
    profile
}

fn assert_profile(
    scenario: &str,
    profile: AllocationProfile,
    operations: usize,
    gross_bytes: usize,
    peak_bytes: usize,
    retained_bytes: usize,
) {
    assert_eq!(profile.operations, operations, "{scenario}: operations");
    assert_eq!(profile.gross_bytes, gross_bytes, "{scenario}: gross bytes");
    assert_eq!(
        profile.peak_incremental_bytes, peak_bytes,
        "{scenario}: peak bytes"
    );
    assert_eq!(
        profile.retained_bytes, retained_bytes,
        "{scenario}: retained bytes"
    );
}

fn assert_zero_allocations(scenario: &str, profile: AllocationProfile) {
    assert_profile(scenario, profile, 0, 0, 0, 0);
}

fn profile_setup() {
    let scenario = format!("setup/SMAConfig/period_{PERIOD}");
    let profile = print_profile(&scenario, || SMAConfig::new(PERIOD).expect("valid period"));
    assert_zero_allocations(&scenario, profile);

    print_profile(&format!("setup/SMAConfig/per_worker/{WORKERS}"), || {
        (0..WORKERS)
            .map(|_| SMAConfig::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>()
    });

    print_profile(
        &format!("setup/SMAConfig/parameter_sweep/{}", SWEEP_PERIODS.len()),
        || {
            SWEEP_PERIODS
                .iter()
                .map(|&period| SMAConfig::new(period).expect("valid period"))
                .collect::<Vec<_>>()
        },
    );

    print_profile(
        &format!("setup/SMAConfig/streaming/{STREAM_INSTRUMENTS}"),
        || {
            let config = SMAConfig::new(PERIOD).expect("valid period");
            (0..STREAM_INSTRUMENTS)
                .map(|_| IndicatorConfig::stream(&config).expect("valid period"))
                .collect::<Vec<_>>()
        },
    );
}

fn profile_one_shot() {
    let input = series_fixture(PROFILE_SIZE, 0);
    let ohlc = ohlc_fixture(PROFILE_SIZE);

    let sma_config = SMAConfig::new(PERIOD).expect("valid period");
    let mut config_output = vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)];
    let scenario = format!("one_shot/SMAConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(&sma_config, input.as_slice(), config_output.as_mut_slice())
            .expect("valid SMA fixture")
    });
    assert_zero_allocations(&scenario, profile);

    let compact_bytes = output_len(PROFILE_SIZE, PERIOD) * core::mem::size_of::<Float>();
    let scenario = format!("one_shot/SMAConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&sma_config, input.as_slice()).expect("valid SMA fixture")
    });
    assert_profile(
        &scenario,
        profile,
        1,
        compact_bytes,
        compact_bytes,
        compact_bytes,
    );

    let avgprice_config = AVGPRICEConfig::new();
    let avgprice_input = AVGPRICEInput {
        open: ohlc.open.as_slice(),
        high: ohlc.high.as_slice(),
        low: ohlc.low.as_slice(),
        close: ohlc.close.as_slice(),
    };
    let mut avgprice_output = vec![0.0 as Float; PROFILE_SIZE];
    print_profile(
        &format!("one_shot/AVGPRICEConfig/caller_compact/{PROFILE_SIZE}"),
        || {
            IndicatorConfig::compute_into(
                &avgprice_config,
                avgprice_input,
                avgprice_output.as_mut_slice(),
            )
            .expect("valid AVGPRICE fixture")
        },
    );

    let minmax_config = MINMAXConfig::new(PERIOD).expect("valid period");
    let output_len = output_len(PROFILE_SIZE, PERIOD);
    let mut min = vec![0.0 as Float; output_len];
    let mut max = vec![0.0 as Float; output_len];
    print_profile(
        &format!("one_shot/MINMAXConfig/caller_compact/{PROFILE_SIZE}"),
        || {
            IndicatorConfig::compute_into(
                &minmax_config,
                input.as_slice(),
                MINMAXValuesMut {
                    min: min.as_mut_slice(),
                    max: max.as_mut_slice(),
                },
            )
            .expect("valid MINMAX fixture")
        },
    );

    let minmaxindex_config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
    let mut min_idx = vec![0_usize; output_len];
    let mut max_idx = vec![0_usize; output_len];
    print_profile(
        &format!("one_shot/MINMAXINDEXConfig/caller_compact/{PROFILE_SIZE}"),
        || {
            IndicatorConfig::compute_into(
                &minmaxindex_config,
                input.as_slice(),
                MINMAXINDEXValuesMut {
                    min_idx: min_idx.as_mut_slice(),
                    max_idx: max_idx.as_mut_slice(),
                },
            )
            .expect("valid MINMAXINDEX fixture")
        },
    );
}

fn profile_extrema_execution() {
    let input = series_fixture(PROFILE_SIZE, 0);
    let count = output_len(PROFILE_SIZE, PERIOD);
    let value_config_scenario = format!("setup/MINMAXConfig/period_{PERIOD}");
    let profile = print_profile(&value_config_scenario, || {
        MINMAXConfig::new(PERIOD).expect("valid period")
    });
    assert_zero_allocations(&value_config_scenario, profile);

    let index_config_scenario = format!("setup/MINMAXINDEXConfig/period_{PERIOD}");
    let profile = print_profile(&index_config_scenario, || {
        MINMAXINDEXConfig::new(PERIOD).expect("valid period")
    });
    assert_zero_allocations(&index_config_scenario, profile);

    let value_config = MINMAXConfig::new(PERIOD).expect("valid period");
    let index_config = MINMAXINDEXConfig::new(PERIOD).expect("valid period");
    let mut min = vec![0.0 as Float; count];
    let mut max = vec![0.0 as Float; count];
    let mut min_idx = vec![0_usize; count];
    let mut max_idx = vec![0_usize; count];
    let scratch_bytes = 2 * PROFILE_SIZE * core::mem::size_of::<usize>();

    let scenario = format!("one_shot/MINMAXConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(
            &value_config,
            input.as_slice(),
            MINMAXValuesMut {
                min: min.as_mut_slice(),
                max: max.as_mut_slice(),
            },
        )
        .expect("valid MINMAX fixture")
    });
    assert_profile(&scenario, profile, 2, scratch_bytes, scratch_bytes, 0);

    let scenario = format!("one_shot/MINMAXINDEXConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(
            &index_config,
            input.as_slice(),
            MINMAXINDEXValuesMut {
                min_idx: min_idx.as_mut_slice(),
                max_idx: max_idx.as_mut_slice(),
            },
        )
        .expect("valid MINMAXINDEX fixture")
    });
    assert_profile(&scenario, profile, 2, scratch_bytes, scratch_bytes, 0);

    let value_output_bytes = 2 * count * core::mem::size_of::<Float>();
    let scenario = format!("one_shot/MINMAXConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&value_config, input.as_slice()).expect("valid MINMAX fixture")
    });
    assert_profile(
        &scenario,
        profile,
        4,
        scratch_bytes + value_output_bytes,
        scratch_bytes + value_output_bytes,
        value_output_bytes,
    );

    let index_output_bytes = 2 * count * core::mem::size_of::<usize>();
    let scenario = format!("one_shot/MINMAXINDEXConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&index_config, input.as_slice())
            .expect("valid MINMAXINDEX fixture")
    });
    assert_profile(
        &scenario,
        profile,
        4,
        scratch_bytes + index_output_bytes,
        scratch_bytes + index_output_bytes,
        index_output_bytes,
    );

    for (name, profile) in [
        (
            "one_shot/MINMAXConfig/owned_compact/empty",
            print_profile("one_shot/MINMAXConfig/owned_compact/empty", || {
                IndicatorConfig::compute(&value_config, &[]).expect("valid empty MINMAX fixture")
            }),
        ),
        (
            "one_shot/MINMAXINDEXConfig/owned_compact/empty",
            print_profile("one_shot/MINMAXINDEXConfig/owned_compact/empty", || {
                IndicatorConfig::compute(&index_config, &[])
                    .expect("valid empty MINMAXINDEX fixture")
            }),
        ),
    ] {
        assert_zero_allocations(name, profile);
    }

    let scenario = format!("setup/MINMAXBatchRunner/capacity_{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::prepare_batch(&value_config, PROFILE_SIZE)
            .expect("valid prepared capacity")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes,
        scratch_bytes,
        scratch_bytes,
    );

    let scenario = format!("setup/MINMAXINDEXBatchRunner/capacity_{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::prepare_batch(&index_config, PROFILE_SIZE)
            .expect("valid prepared capacity")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes,
        scratch_bytes,
        scratch_bytes,
    );

    let mut value_runner = IndicatorConfig::prepare_batch(&value_config, PROFILE_SIZE)
        .expect("valid prepared capacity");
    let mut index_runner = IndicatorConfig::prepare_batch(&index_config, PROFILE_SIZE)
        .expect("valid prepared capacity");
    for pass in ["first", "repeated"] {
        let scenario = format!("repeated/prepared_MINMAX/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MINMAXConfig>::compute_into(
                &mut value_runner,
                input.as_slice(),
                MINMAXValuesMut {
                    min: min.as_mut_slice(),
                    max: max.as_mut_slice(),
                },
            )
            .expect("valid prepared MINMAX fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("repeated/prepared_MINMAXINDEX/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
                &mut index_runner,
                input.as_slice(),
                MINMAXINDEXValuesMut {
                    min_idx: min_idx.as_mut_slice(),
                    max_idx: max_idx.as_mut_slice(),
                },
            )
            .expect("valid prepared MINMAXINDEX fixture")
        });
        assert_zero_allocations(&scenario, profile);
    }

    let mut oversized = series_fixture(PROFILE_SIZE + 1, 0);
    oversized[0] = Float::NAN;
    let mut insufficient_min = [0.0 as Float; 1];
    let mut insufficient_max = [0.0 as Float; 1];
    let mut insufficient_min_idx = [0_usize; 1];
    let mut insufficient_max_idx = [0_usize; 1];
    let scenario = format!(
        "repeated/prepared_MINMAX/oversize_rejection/{}",
        PROFILE_SIZE + 1
    );
    let profile = print_profile(&scenario, || {
        PreparedBatchRunner::<MINMAXConfig>::compute_into(
            &mut value_runner,
            oversized.as_slice(),
            MINMAXValuesMut {
                min: &mut insufficient_min,
                max: &mut insufficient_max,
            },
        )
        .expect_err("oversize must be rejected")
    });
    assert_zero_allocations(&scenario, profile);

    let scenario = format!(
        "repeated/prepared_MINMAXINDEX/oversize_rejection/{}",
        PROFILE_SIZE + 1
    );
    let profile = print_profile(&scenario, || {
        PreparedBatchRunner::<MINMAXINDEXConfig>::compute_into(
            &mut index_runner,
            oversized.as_slice(),
            MINMAXINDEXValuesMut {
                min_idx: &mut insufficient_min_idx,
                max_idx: &mut insufficient_max_idx,
            },
        )
        .expect_err("oversize must be rejected")
    });
    assert_zero_allocations(&scenario, profile);

    let mut value_stream = IndicatorConfig::stream(&value_config).expect("valid stream");
    let scenario = format!("streaming/MINMAXConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MINMAXConfig>::next(&mut value_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);

    let mut index_stream = IndicatorConfig::stream(&index_config).expect("valid stream");
    let scenario = format!("streaming/MINMAXINDEXConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MINMAXINDEXConfig>::next(&mut index_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);
}

fn profile_single_extrema_execution() {
    let input = series_fixture(PROFILE_SIZE, 0);
    let count = output_len(PROFILE_SIZE, PERIOD);
    let scratch_bytes = PROFILE_SIZE * core::mem::size_of::<usize>();
    let value_output_bytes = count * core::mem::size_of::<Float>();
    let index_output_bytes = count * core::mem::size_of::<usize>();

    let min_config = MINConfig::new(PERIOD).expect("valid period");
    let max_config = MAXConfig::new(PERIOD).expect("valid period");
    let min_index_config = MININDEXConfig::new(PERIOD).expect("valid period");
    let max_index_config = MAXINDEXConfig::new(PERIOD).expect("valid period");
    for (scenario, profile) in [
        (
            format!("setup/MINConfig/period_{PERIOD}"),
            print_profile(&format!("setup/MINConfig/period_{PERIOD}"), || {
                MINConfig::new(PERIOD).expect("valid period")
            }),
        ),
        (
            format!("setup/MAXConfig/period_{PERIOD}"),
            print_profile(&format!("setup/MAXConfig/period_{PERIOD}"), || {
                MAXConfig::new(PERIOD).expect("valid period")
            }),
        ),
        (
            format!("setup/MININDEXConfig/period_{PERIOD}"),
            print_profile(&format!("setup/MININDEXConfig/period_{PERIOD}"), || {
                MININDEXConfig::new(PERIOD).expect("valid period")
            }),
        ),
        (
            format!("setup/MAXINDEXConfig/period_{PERIOD}"),
            print_profile(&format!("setup/MAXINDEXConfig/period_{PERIOD}"), || {
                MAXINDEXConfig::new(PERIOD).expect("valid period")
            }),
        ),
    ] {
        assert_zero_allocations(&scenario, profile);
    }

    let mut values = vec![0.0 as Float; count];
    let mut indexes = vec![0_usize; count];

    let scenario = format!("one_shot/MINConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(&min_config, input.as_slice(), values.as_mut_slice())
            .expect("valid MIN fixture")
    });
    assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

    let scenario = format!("one_shot/MAXConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(&max_config, input.as_slice(), values.as_mut_slice())
            .expect("valid MAX fixture")
    });
    assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

    let scenario = format!("one_shot/MININDEXConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(&min_index_config, input.as_slice(), indexes.as_mut_slice())
            .expect("valid MININDEX fixture")
    });
    assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

    let scenario = format!("one_shot/MAXINDEXConfig/caller_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute_into(&max_index_config, input.as_slice(), indexes.as_mut_slice())
            .expect("valid MAXINDEX fixture")
    });
    assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

    let scenario = format!("one_shot/MINConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&min_config, input.as_slice()).expect("valid MIN fixture")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes + value_output_bytes,
        scratch_bytes + value_output_bytes,
        value_output_bytes,
    );

    let scenario = format!("one_shot/MAXConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&max_config, input.as_slice()).expect("valid MAX fixture")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes + value_output_bytes,
        scratch_bytes + value_output_bytes,
        value_output_bytes,
    );

    let scenario = format!("one_shot/MININDEXConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&min_index_config, input.as_slice())
            .expect("valid MININDEX fixture")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes + index_output_bytes,
        scratch_bytes + index_output_bytes,
        index_output_bytes,
    );

    let scenario = format!("one_shot/MAXINDEXConfig/owned_compact/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::compute(&max_index_config, input.as_slice())
            .expect("valid MAXINDEX fixture")
    });
    assert_profile(
        &scenario,
        profile,
        2,
        scratch_bytes + index_output_bytes,
        scratch_bytes + index_output_bytes,
        index_output_bytes,
    );

    for (scenario, profile) in [
        (
            format!("setup/MINBatchRunner/capacity_{PROFILE_SIZE}"),
            print_profile(
                &format!("setup/MINBatchRunner/capacity_{PROFILE_SIZE}"),
                || {
                    IndicatorConfig::prepare_batch(&min_config, PROFILE_SIZE)
                        .expect("valid preparation")
                },
            ),
        ),
        (
            format!("setup/MAXBatchRunner/capacity_{PROFILE_SIZE}"),
            print_profile(
                &format!("setup/MAXBatchRunner/capacity_{PROFILE_SIZE}"),
                || {
                    IndicatorConfig::prepare_batch(&max_config, PROFILE_SIZE)
                        .expect("valid preparation")
                },
            ),
        ),
        (
            format!("setup/MININDEXBatchRunner/capacity_{PROFILE_SIZE}"),
            print_profile(
                &format!("setup/MININDEXBatchRunner/capacity_{PROFILE_SIZE}"),
                || {
                    IndicatorConfig::prepare_batch(&min_index_config, PROFILE_SIZE)
                        .expect("valid preparation")
                },
            ),
        ),
        (
            format!("setup/MAXINDEXBatchRunner/capacity_{PROFILE_SIZE}"),
            print_profile(
                &format!("setup/MAXINDEXBatchRunner/capacity_{PROFILE_SIZE}"),
                || {
                    IndicatorConfig::prepare_batch(&max_index_config, PROFILE_SIZE)
                        .expect("valid preparation")
                },
            ),
        ),
    ] {
        assert_profile(
            &scenario,
            profile,
            1,
            scratch_bytes,
            scratch_bytes,
            scratch_bytes,
        );
    }

    let mut min_runner =
        IndicatorConfig::prepare_batch(&min_config, PROFILE_SIZE).expect("valid preparation");
    let mut max_runner =
        IndicatorConfig::prepare_batch(&max_config, PROFILE_SIZE).expect("valid preparation");
    let mut min_index_runner =
        IndicatorConfig::prepare_batch(&min_index_config, PROFILE_SIZE).expect("valid preparation");
    let mut max_index_runner =
        IndicatorConfig::prepare_batch(&max_index_config, PROFILE_SIZE).expect("valid preparation");
    for pass in ["first", "repeated"] {
        let scenario = format!("repeated/prepared_MIN/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MINConfig>::compute_into(
                &mut min_runner,
                input.as_slice(),
                values.as_mut_slice(),
            )
            .expect("valid prepared MIN fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("repeated/prepared_MAX/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MAXConfig>::compute_into(
                &mut max_runner,
                input.as_slice(),
                values.as_mut_slice(),
            )
            .expect("valid prepared MAX fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("repeated/prepared_MININDEX/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MININDEXConfig>::compute_into(
                &mut min_index_runner,
                input.as_slice(),
                indexes.as_mut_slice(),
            )
            .expect("valid prepared MININDEX fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("repeated/prepared_MAXINDEX/{pass}/{PROFILE_SIZE}");
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<MAXINDEXConfig>::compute_into(
                &mut max_index_runner,
                input.as_slice(),
                indexes.as_mut_slice(),
            )
            .expect("valid prepared MAXINDEX fixture")
        });
        assert_zero_allocations(&scenario, profile);
    }

    let mut min_stream = IndicatorConfig::stream(&min_config).expect("valid stream");
    let scenario = format!("streaming/MINConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MINConfig>::next(&mut min_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);

    let mut max_stream = IndicatorConfig::stream(&max_config).expect("valid stream");
    let scenario = format!("streaming/MAXConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MAXConfig>::next(&mut max_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);

    let mut min_index_stream = IndicatorConfig::stream(&min_index_config).expect("valid stream");
    let scenario = format!("streaming/MININDEXConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MININDEXConfig>::next(&mut min_index_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);

    let mut max_index_stream = IndicatorConfig::stream(&max_index_config).expect("valid stream");
    let scenario = format!("streaming/MAXINDEXConfig/ticks/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last = None;
        for &tick in &input {
            last = StreamingComputation::<MAXINDEXConfig>::next(&mut max_index_stream, tick)
                .expect("valid stream tick");
            black_box(last);
        }
        last
    });
    assert_zero_allocations(&scenario, profile);
}

macro_rules! profile_single_output_indicator {
    (
        $label:literal,
        $config:ty,
        $new_config:expr,
        $stream_operations:expr,
        $stream_bytes:expr
    ) => {{
        let input = series_fixture(PROFILE_SIZE, 0);
        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, || $new_config);
        assert_zero_allocations(&scenario, profile);

        let config: $config = $new_config;
        let count = PROFILE_SIZE.saturating_sub(IndicatorConfig::lookback(&config));
        let output_bytes = count * core::mem::size_of::<Float>();
        let mut output = vec![0.0 as Float; count];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(&config, input.as_slice(), output.as_mut_slice())
                .expect("valid caller-owned fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input.as_slice()).expect("valid owned fixture")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            output_bytes,
            output_bytes,
            output_bytes,
        );

        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, &[]).expect("valid empty fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity")
        });
        assert_zero_allocations(&scenario, profile);

        let mut runner =
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    input.as_slice(),
                    output.as_mut_slice(),
                )
                .expect("valid prepared fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized_input = series_fixture(PROFILE_SIZE + 1, 1);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                oversized_input.as_slice(),
                output.as_mut_slice(),
            )
            .expect_err("oversized input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid stream")
        });
        assert_profile(
            &scenario,
            profile,
            $stream_operations,
            $stream_bytes,
            $stream_bytes,
            $stream_bytes,
        );

        let mut stream = IndicatorConfig::stream(&config).expect("valid stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for &tick in &input {
                last = StreamingComputation::<$config>::next(&mut stream, tick)
                    .expect("valid stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

fn profile_single_output_execution() {
    const STREAM_BYTES: usize = PERIOD * core::mem::size_of::<Float>();
    profile_single_output_indicator!(
        "WMA",
        WMAConfig,
        WMAConfig::new(PERIOD).expect("valid period"),
        1,
        STREAM_BYTES
    );
    profile_single_output_indicator!(
        "TRIMA",
        TRIMAConfig,
        TRIMAConfig::new(PERIOD).expect("valid period"),
        1,
        STREAM_BYTES
    );
    profile_single_output_indicator!(
        "EMA",
        EMAConfig,
        EMAConfig::new(PERIOD).expect("valid period"),
        0,
        0
    );
    profile_single_output_indicator!(
        "DEMA",
        DEMAConfig,
        DEMAConfig::new(PERIOD).expect("valid period"),
        0,
        0
    );
    profile_single_output_indicator!(
        "TEMA",
        TEMAConfig,
        TEMAConfig::new(PERIOD).expect("valid period"),
        0,
        0
    );
    profile_single_output_indicator!(
        "T3",
        T3Config,
        T3Config::with_default_vfactor(PERIOD).expect("valid parameters"),
        0,
        0
    );
    profile_single_output_indicator!(
        "MA_EMA",
        MAConfig,
        MAConfig::new(PERIOD, MAType::EMA).expect("valid parameters"),
        0,
        0
    );
    profile_single_output_indicator!(
        "AVGDEV",
        AVGDEVConfig,
        AVGDEVConfig::new(PERIOD).expect("valid period"),
        1,
        STREAM_BYTES
    );
    profile_single_output_indicator!(
        "SUM",
        SUMConfig,
        SUMConfig::new(PERIOD).expect("valid period"),
        1,
        STREAM_BYTES
    );
}

fn profile_math_transform_execution() {
    profile_single_output_indicator!("ACOS", ACOSConfig, ACOSConfig::new(), 0, 0);
    profile_single_output_indicator!("ASIN", ASINConfig, ASINConfig::new(), 0, 0);
    profile_single_output_indicator!("ATAN", ATANConfig, ATANConfig::new(), 0, 0);
    profile_single_output_indicator!("CEIL", CEILConfig, CEILConfig::new(), 0, 0);
    profile_single_output_indicator!("COS", COSConfig, COSConfig::new(), 0, 0);
    profile_single_output_indicator!("COSH", COSHConfig, COSHConfig::new(), 0, 0);
    profile_single_output_indicator!("EXP", EXPConfig, EXPConfig::new(), 0, 0);
    profile_single_output_indicator!("FLOOR", FLOORConfig, FLOORConfig::new(), 0, 0);
    profile_single_output_indicator!("LN", LNConfig, LNConfig::new(), 0, 0);
    profile_single_output_indicator!("LOG10", LOG10Config, LOG10Config::new(), 0, 0);
    profile_single_output_indicator!("SIN", SINConfig, SINConfig::new(), 0, 0);
    profile_single_output_indicator!("SINH", SINHConfig, SINHConfig::new(), 0, 0);
    profile_single_output_indicator!("SQRT", SQRTConfig, SQRTConfig::new(), 0, 0);
    profile_single_output_indicator!("TAN", TANConfig, TANConfig::new(), 0, 0);
    profile_single_output_indicator!("TANH", TANHConfig, TANHConfig::new(), 0, 0);
}

macro_rules! profile_binary_operator {
    ($label:literal, $config:ty) => {{
        let real0 = series_fixture(PROFILE_SIZE, 0);
        let real1 = series_fixture(PROFILE_SIZE, 1);
        let input = BinaryInput {
            real0: real0.as_slice(),
            real1: real1.as_slice(),
        };

        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, <$config>::new);
        assert_zero_allocations(&scenario, profile);

        let config = <$config>::new();
        let mut output = vec![0.0 as Float; PROFILE_SIZE];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(&config, input, output.as_mut_slice())
                .expect("valid binary caller-owned fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let output_bytes = PROFILE_SIZE * core::mem::size_of::<Float>();
        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input).expect("valid binary owned fixture")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            output_bytes,
            output_bytes,
            output_bytes,
        );

        let empty = [];
        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(
                &config,
                BinaryInput {
                    real0: &empty,
                    real1: &empty,
                },
            )
            .expect("valid empty binary fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE)
                .expect("valid binary prepared capacity")
        });
        assert_zero_allocations(&scenario, profile);

        let mut runner = IndicatorConfig::prepare_batch(&config, PROFILE_SIZE)
            .expect("valid binary prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    input,
                    output.as_mut_slice(),
                )
                .expect("valid prepared binary fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized_real0 = series_fixture(PROFILE_SIZE + 1, 2);
        let oversized_real1 = series_fixture(PROFILE_SIZE + 1, 3);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                BinaryInput {
                    real0: oversized_real0.as_slice(),
                    real1: oversized_real1.as_slice(),
                },
                output.as_mut_slice(),
            )
            .expect_err("oversized binary input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid binary stream")
        });
        assert_zero_allocations(&scenario, profile);

        let mut stream = IndicatorConfig::stream(&config).expect("valid binary stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for idx in 0..PROFILE_SIZE {
                last = StreamingComputation::<$config>::next(
                    &mut stream,
                    BinaryTick {
                        real0: real0[idx],
                        real1: real1[idx],
                    },
                )
                .expect("valid binary stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

fn profile_binary_operator_execution() {
    profile_binary_operator!("ADD", ADDConfig);
    profile_binary_operator!("SUB", SUBConfig);
    profile_binary_operator!("MULT", MULTConfig);
    profile_binary_operator!("DIV", DIVConfig);
}

macro_rules! profile_single_statistic {
    ($label:literal, $config:ty, $new_config:expr, $prepared_scratch:expr) => {{
        let input = series_fixture(PROFILE_SIZE, 0);
        let scratch_bytes = (PERIOD - 1) * core::mem::size_of::<Float>();
        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, || $new_config);
        assert_zero_allocations(&scenario, profile);

        let config: $config = $new_config;
        let count = PROFILE_SIZE - IndicatorConfig::lookback(&config);
        let output_bytes = count * core::mem::size_of::<Float>();
        let mut output = vec![0.0 as Float; count];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(&config, input.as_slice(), output.as_mut_slice())
                .expect("valid statistic fixture")
        });
        assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input.as_slice()).expect("valid statistic fixture")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            output_bytes,
            output_bytes,
            output_bytes,
        );

        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, &[]).expect("valid empty statistic fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity")
        });
        if $prepared_scratch {
            assert_profile(
                &scenario,
                profile,
                1,
                scratch_bytes,
                scratch_bytes,
                scratch_bytes,
            );
        } else {
            assert_zero_allocations(&scenario, profile);
        }

        let mut runner =
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    input.as_slice(),
                    output.as_mut_slice(),
                )
                .expect("valid prepared statistic fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized = series_fixture(PROFILE_SIZE + 1, 1);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                oversized.as_slice(),
                output.as_mut_slice(),
            )
            .expect_err("oversized statistic input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid statistic stream")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            scratch_bytes,
            scratch_bytes,
            scratch_bytes,
        );

        let mut stream = IndicatorConfig::stream(&config).expect("valid statistic stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for &tick in &input {
                last = StreamingComputation::<$config>::next(&mut stream, tick)
                    .expect("valid statistic stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

macro_rules! profile_paired_statistic {
    ($label:literal, $config:ty, $new_config:expr) => {{
        let real0 = series_fixture(PROFILE_SIZE, 0);
        let real1 = series_fixture(PROFILE_SIZE, 1);
        let scratch_bytes = 2 * (PERIOD - 1) * core::mem::size_of::<Float>();
        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, || $new_config);
        assert_zero_allocations(&scenario, profile);

        let config: $config = $new_config;
        let count = PROFILE_SIZE - IndicatorConfig::lookback(&config);
        let output_bytes = count * core::mem::size_of::<Float>();
        let input = PairInput {
            real0: real0.as_slice(),
            real1: real1.as_slice(),
        };
        let mut output = vec![0.0 as Float; count];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(&config, input, output.as_mut_slice())
                .expect("valid paired statistic fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input).expect("valid paired statistic fixture")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            output_bytes,
            output_bytes,
            output_bytes,
        );

        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(
                &config,
                PairInput {
                    real0: &[],
                    real1: &[],
                },
            )
            .expect("valid empty paired statistic fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity")
        });
        assert_zero_allocations(&scenario, profile);

        let mut runner =
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    input,
                    output.as_mut_slice(),
                )
                .expect("valid prepared paired statistic fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized0 = series_fixture(PROFILE_SIZE + 1, 2);
        let oversized1 = series_fixture(PROFILE_SIZE + 1, 3);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                PairInput {
                    real0: oversized0.as_slice(),
                    real1: oversized1.as_slice(),
                },
                output.as_mut_slice(),
            )
            .expect_err("oversized paired statistic input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid paired statistic stream")
        });
        assert_profile(
            &scenario,
            profile,
            2,
            scratch_bytes,
            scratch_bytes,
            scratch_bytes,
        );

        let mut stream = IndicatorConfig::stream(&config).expect("valid paired statistic stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for (&real0, &real1) in real0.iter().zip(&real1) {
                last =
                    StreamingComputation::<$config>::next(&mut stream, PairTick { real0, real1 })
                        .expect("valid paired statistic stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

macro_rules! profile_regression_statistic {
    ($label:literal, $config:ty, $new_config:expr) => {{
        let input = series_fixture(PROFILE_SIZE, 0);
        let scratch_bytes = PERIOD * core::mem::size_of::<Float>();
        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, || $new_config);
        assert_zero_allocations(&scenario, profile);

        let config: $config = $new_config;
        let count = PROFILE_SIZE - IndicatorConfig::lookback(&config);
        let output_bytes = count * core::mem::size_of::<Float>();
        let mut output = vec![0.0 as Float; count];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(&config, input.as_slice(), output.as_mut_slice())
                .expect("valid regression fixture")
        });
        assert_profile(&scenario, profile, 1, scratch_bytes, scratch_bytes, 0);

        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input.as_slice()).expect("valid regression fixture")
        });
        assert_profile(
            &scenario,
            profile,
            2,
            scratch_bytes + output_bytes,
            scratch_bytes + output_bytes,
            output_bytes,
        );

        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, &[]).expect("valid empty regression fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            scratch_bytes,
            scratch_bytes,
            scratch_bytes,
        );

        let mut runner =
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    input.as_slice(),
                    output.as_mut_slice(),
                )
                .expect("valid prepared regression fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized = series_fixture(PROFILE_SIZE + 1, 1);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                oversized.as_slice(),
                output.as_mut_slice(),
            )
            .expect_err("oversized regression input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid regression stream")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            scratch_bytes,
            scratch_bytes,
            scratch_bytes,
        );

        let mut stream = IndicatorConfig::stream(&config).expect("valid regression stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for &tick in &input {
                last = StreamingComputation::<$config>::next(&mut stream, tick)
                    .expect("valid regression stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

fn profile_statistic_execution() {
    profile_single_statistic!(
        "VAR",
        VARConfig,
        VARConfig::with_default_nbdev(PERIOD).expect("valid parameters"),
        false
    );
    profile_single_statistic!(
        "STDDEV",
        STDDEVConfig,
        STDDEVConfig::with_default_nbdev(PERIOD).expect("valid parameters"),
        true
    );
    profile_paired_statistic!(
        "CORREL",
        CORRELConfig,
        CORRELConfig::new(PERIOD).expect("valid period")
    );
    profile_paired_statistic!(
        "BETA",
        BETAConfig,
        BETAConfig::new(PERIOD).expect("valid period")
    );
    profile_regression_statistic!(
        "LINEARREG",
        LINEARREGConfig,
        LINEARREGConfig::new(PERIOD).expect("valid period")
    );
    profile_regression_statistic!(
        "LINEARREG_SLOPE",
        LINEARREG_SLOPEConfig,
        LINEARREG_SLOPEConfig::new(PERIOD).expect("valid period")
    );
    profile_regression_statistic!(
        "LINEARREG_INTERCEPT",
        LINEARREG_INTERCEPTConfig,
        LINEARREG_INTERCEPTConfig::new(PERIOD).expect("valid period")
    );
    profile_regression_statistic!(
        "LINEARREG_ANGLE",
        LINEARREG_ANGLEConfig,
        LINEARREG_ANGLEConfig::new(PERIOD).expect("valid period")
    );
    profile_regression_statistic!(
        "TSF",
        TSFConfig,
        TSFConfig::new(PERIOD).expect("valid period")
    );
}

macro_rules! profile_named_input_indicator {
    (
        $label:literal,
        $config:ty,
        $new_config:expr,
        $input:ident,
        $tick:ident,
        [$($field:ident),+ $(,)?],
        $lookback:expr
    ) => {{
        let ohlc = ohlc_fixture(PROFILE_SIZE);
        let scenario = format!("setup/{}Config/parameters", $label);
        let profile = print_profile(&scenario, || $new_config);
        assert_zero_allocations(&scenario, profile);

        let config: $config = $new_config;
        let output_bytes = (PROFILE_SIZE - $lookback) * core::mem::size_of::<Float>();
        let mut output = vec![0.0 as Float; PROFILE_SIZE];
        let scenario = format!("one_shot/{}Config/caller_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute_into(
                &config,
                $input {
                    $($field: ohlc.$field.as_slice()),+
                },
                output.as_mut_slice(),
            )
            .expect("valid caller-owned multi-series fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("one_shot/{}Config/owned_compact/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(
                &config,
                $input {
                    $($field: ohlc.$field.as_slice()),+
                },
            )
            .expect("valid owned multi-series fixture")
        });
        assert_profile(
            &scenario,
            profile,
            1,
            output_bytes,
            output_bytes,
            output_bytes,
        );

        let empty = ohlc_fixture(0);
        let scenario = format!("one_shot/{}Config/owned_compact/count_0", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(
                &config,
                $input {
                    $($field: empty.$field.as_slice()),+
                },
            )
            .expect("valid empty multi-series fixture")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}BatchRunner/capacity_{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity")
        });
        assert_zero_allocations(&scenario, profile);

        let mut runner =
            IndicatorConfig::prepare_batch(&config, PROFILE_SIZE).expect("valid prepared capacity");
        for pass in ["first", "repeated"] {
            let scenario = format!("repeated/prepared_{}/{pass}/{PROFILE_SIZE}", $label);
            let profile = print_profile(&scenario, || {
                PreparedBatchRunner::<$config>::compute_into(
                    &mut runner,
                    $input {
                        $($field: ohlc.$field.as_slice()),+
                    },
                    output.as_mut_slice(),
                )
                .expect("valid prepared multi-series fixture")
            });
            assert_zero_allocations(&scenario, profile);
        }

        let oversized = ohlc_fixture(PROFILE_SIZE + 1);
        let scenario = format!(
            "repeated/prepared_{}/oversize_rejection/{}",
            $label,
            PROFILE_SIZE + 1
        );
        let profile = print_profile(&scenario, || {
            PreparedBatchRunner::<$config>::compute_into(
                &mut runner,
                $input {
                    $($field: oversized.$field.as_slice()),+
                },
                output.as_mut_slice(),
            )
            .expect_err("oversized multi-series input must be rejected")
        });
        assert_zero_allocations(&scenario, profile);

        let scenario = format!("setup/{}Config/stream", $label);
        let profile = print_profile(&scenario, || {
            IndicatorConfig::stream(&config).expect("valid multi-series stream")
        });
        assert_zero_allocations(&scenario, profile);

        let mut stream = IndicatorConfig::stream(&config).expect("valid multi-series stream");
        let scenario = format!("streaming/{}Config/ticks/{PROFILE_SIZE}", $label);
        let profile = print_profile(&scenario, || {
            let mut last = None;
            for idx in 0..PROFILE_SIZE {
                last = StreamingComputation::<$config>::next(
                    &mut stream,
                    $tick {
                        $($field: ohlc.$field[idx]),+
                    },
                )
                .expect("valid multi-series stream tick");
                black_box(last);
            }
            last
        });
        assert_zero_allocations(&scenario, profile);
    }};
}

fn profile_named_input_execution() {
    profile_named_input_indicator!(
        "AVGPRICE",
        AVGPRICEConfig,
        AVGPRICEConfig::new(),
        AVGPRICEInput,
        AVGPRICETick,
        [open, high, low, close],
        0
    );
    profile_named_input_indicator!(
        "MEDPRICE",
        MEDPRICEConfig,
        MEDPRICEConfig::new(),
        MEDPRICEInput,
        MEDPRICETick,
        [high, low],
        0
    );
    profile_named_input_indicator!(
        "TYPPRICE",
        TYPPRICEConfig,
        TYPPRICEConfig::new(),
        TYPPRICEInput,
        TYPPRICETick,
        [high, low, close],
        0
    );
    profile_named_input_indicator!(
        "WCLPRICE",
        WCLPRICEConfig,
        WCLPRICEConfig::new(),
        WCLPRICEInput,
        WCLPRICETick,
        [high, low, close],
        0
    );
    profile_named_input_indicator!(
        "AD",
        ADConfig,
        ADConfig::new(),
        ADInput,
        ADTick,
        [high, low, close, volume],
        0
    );
    profile_named_input_indicator!(
        "ADOSC",
        ADOSCConfig,
        ADOSCConfig::new(PERIOD / 2, PERIOD).expect("valid ADOSC parameters"),
        ADOSCInput,
        ADOSCTick,
        [high, low, close, volume],
        PERIOD - 1
    );
    profile_named_input_indicator!(
        "OBV",
        OBVConfig,
        OBVConfig::new(),
        OBVInput,
        OBVTick,
        [close, volume],
        1
    );
    profile_named_input_indicator!(
        "TRANGE",
        TRANGEConfig,
        TRANGEConfig::new(),
        TRANGEInput,
        TRANGETick,
        [high, low, close],
        1
    );
    profile_named_input_indicator!(
        "ATR",
        ATRConfig,
        ATRConfig::new(PERIOD).expect("valid ATR period"),
        ATRInput,
        ATRTick,
        [high, low, close],
        PERIOD
    );
    profile_named_input_indicator!(
        "NATR",
        NATRConfig,
        NATRConfig::new(PERIOD).expect("valid NATR period"),
        NATRInput,
        NATRTick,
        [high, low, close],
        PERIOD
    );
}

fn profile_small_owned_compact_counts() {
    let config = SMAConfig::new(3).expect("valid period");
    let count_0 = [];
    let count_1 = [1.0 as Float, 2.0, 3.0];
    let count_2 = [1.0 as Float, 2.0, 3.0, 4.0];
    let count_3 = [1.0 as Float, 2.0, 3.0, 4.0, 5.0];
    let inputs = [
        (0, count_0.as_slice()),
        (1, count_1.as_slice()),
        (2, count_2.as_slice()),
        (3, count_3.as_slice()),
    ];

    for (count, input) in inputs {
        let scenario = format!("one_shot/SMAConfig/owned_compact/count_{count}");
        let profile = print_profile(&scenario, || {
            IndicatorConfig::compute(&config, input).expect("valid small SMA fixture")
        });
        let payload_bytes = count * core::mem::size_of::<Float>();
        if count == 0 {
            assert_zero_allocations(&scenario, profile);
        } else {
            assert_profile(
                &scenario,
                profile,
                1,
                payload_bytes,
                payload_bytes,
                payload_bytes,
            );
        }
    }
}

fn profile_repeated_workloads() {
    let prepared_config = SMAConfig::new(PERIOD).expect("valid period");
    let scenario = format!("setup/SMABatchRunner/capacity_{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        IndicatorConfig::prepare_batch(&prepared_config, PROFILE_SIZE)
            .expect("valid prepared capacity")
    });
    assert_zero_allocations(&scenario, profile);

    let mut prepared = IndicatorConfig::prepare_batch(&prepared_config, PROFILE_SIZE)
        .expect("valid prepared capacity");
    let prepared_input = series_fixture(PROFILE_SIZE, 0);
    let mut prepared_output = vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)];
    let scenario = format!("repeated/prepared_SMA/first/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        PreparedBatchRunner::<SMAConfig>::compute_into(
            &mut prepared,
            prepared_input.as_slice(),
            prepared_output.as_mut_slice(),
        )
        .expect("valid prepared fixture")
    });
    assert_zero_allocations(&scenario, profile);

    let scenario = format!("repeated/prepared_SMA/repeated/{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        PreparedBatchRunner::<SMAConfig>::compute_into(
            &mut prepared,
            prepared_input.as_slice(),
            prepared_output.as_mut_slice(),
        )
        .expect("valid prepared fixture")
    });
    assert_zero_allocations(&scenario, profile);

    let mut oversized_input = series_fixture(PROFILE_SIZE + 1, 0);
    oversized_input[0] = Float::NAN;
    let mut insufficient_output = [0.0 as Float; 1];
    let scenario = format!(
        "repeated/prepared_SMA/oversize_rejection/{}",
        PROFILE_SIZE + 1
    );
    let profile = print_profile(&scenario, || {
        PreparedBatchRunner::<SMAConfig>::compute_into(
            &mut prepared,
            oversized_input.as_slice(),
            &mut insufficient_output,
        )
        .expect_err("oversized input must be rejected")
    });
    assert_zero_allocations(&scenario, profile);

    let universe = (0..UNIVERSE_INSTRUMENTS)
        .map(|seed| series_fixture(PROFILE_SIZE, seed))
        .collect::<Vec<_>>();
    let sma_config = SMAConfig::new(PERIOD).expect("valid period");
    let mut universe_output = vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)];
    print_profile(
        &format!("repeated/universe_SMA/caller_compact/{UNIVERSE_INSTRUMENTS}x{PROFILE_SIZE}"),
        || {
            let mut last_range = None;
            for series in &universe {
                last_range = Some(
                    IndicatorConfig::compute_into(
                        &sma_config,
                        series.as_slice(),
                        universe_output.as_mut_slice(),
                    )
                    .expect("valid Universe fixture"),
                );
                black_box(universe_output.as_slice());
            }
            last_range
        },
    );

    let sweep_input = series_fixture(PROFILE_SIZE, 0);
    let sweep = SWEEP_PERIODS
        .iter()
        .map(|&period| SMAConfig::new(period).expect("valid period"))
        .collect::<Vec<_>>();
    let mut sweep_output = vec![0.0 as Float; PROFILE_SIZE];
    print_profile(
        &format!(
            "repeated/parameter_sweep_SMA/caller_compact/{}x{PROFILE_SIZE}",
            SWEEP_PERIODS.len()
        ),
        || {
            let mut last_range = None;
            for config in &sweep {
                last_range = Some(
                    IndicatorConfig::compute_into(
                        config,
                        sweep_input.as_slice(),
                        sweep_output.as_mut_slice(),
                    )
                    .expect("valid parameter-sweep fixture"),
                );
                black_box(sweep_output.as_slice());
            }
            last_range
        },
    );

    let worker_configs = (0..WORKERS)
        .map(|_| SMAConfig::new(PERIOD).expect("valid period"))
        .collect::<Vec<_>>();
    let worker_inputs = (0..WORKERS)
        .map(|seed| series_fixture(PROFILE_SIZE, seed))
        .collect::<Vec<_>>();
    let mut worker_outputs = (0..WORKERS)
        .map(|_| vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)])
        .collect::<Vec<_>>();
    print_profile(
        &format!("repeated/no_prepared_runner/per_worker_instances/{WORKERS}x{PROFILE_SIZE}"),
        || {
            let mut last_range = None;
            for ((config, input), output) in worker_configs
                .iter()
                .zip(worker_inputs.iter())
                .zip(worker_outputs.iter_mut())
            {
                last_range = Some(
                    IndicatorConfig::compute_into(config, input.as_slice(), output.as_mut_slice())
                        .expect("valid per-worker fixture"),
                );
                black_box(output.as_slice());
            }
            last_range
        },
    );

    let minmax_config = MINMAXConfig::new(PERIOD).expect("valid period");
    let minmax_input = series_fixture(PROFILE_SIZE, 0);
    let output_len = output_len(PROFILE_SIZE, PERIOD);
    let mut min = vec![0.0 as Float; output_len];
    let mut max = vec![0.0 as Float; output_len];
    print_profile(
        &format!("repeated/MINMAX/caller_compact/{REPEATED_MINMAX_CALLS}x{PROFILE_SIZE}"),
        || {
            let mut last_range = None;
            for _ in 0..REPEATED_MINMAX_CALLS {
                last_range = Some(
                    IndicatorConfig::compute_into(
                        &minmax_config,
                        minmax_input.as_slice(),
                        MINMAXValuesMut {
                            min: min.as_mut_slice(),
                            max: max.as_mut_slice(),
                        },
                    )
                    .expect("valid repeated MINMAX fixture"),
                );
                black_box((min.as_slice(), max.as_slice()));
            }
            last_range
        },
    );

    let instrument_inputs = (0..STREAM_INSTRUMENTS)
        .map(|seed| series_fixture(PROFILE_SIZE, seed))
        .collect::<Vec<_>>();
    let stream_inputs = (0..PROFILE_SIZE)
        .map(|tick_index| {
            instrument_inputs
                .iter()
                .map(|series| series[tick_index])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut config_streams = (0..STREAM_INSTRUMENTS)
        .map(|_| IndicatorConfig::stream(&prepared_config).expect("valid period"))
        .collect::<Vec<_>>();
    let scenario =
        format!("streaming/SMAConfig/independent_streams/{STREAM_INSTRUMENTS}x{PROFILE_SIZE}");
    let profile = print_profile(&scenario, || {
        let mut last_output = None;
        for tick in &stream_inputs {
            for (stream, &input) in config_streams.iter_mut().zip(tick.iter()) {
                last_output = StreamingComputation::<SMAConfig>::next(stream, input)
                    .expect("valid streaming fixture");
                black_box(last_output);
            }
        }
        last_output
    });
    assert_zero_allocations(&scenario, profile);
}

fn main() {
    println!("scenario\tallocation_operations\tgross_allocated_bytes\tpeak_incremental_bytes\tretained_bytes");
    profile_setup();
    profile_one_shot();
    profile_extrema_execution();
    profile_single_extrema_execution();
    profile_single_output_execution();
    profile_math_transform_execution();
    profile_binary_operator_execution();
    profile_named_input_execution();
    profile_statistic_execution();
    profile_small_owned_compact_counts();
    profile_repeated_workloads();
}
