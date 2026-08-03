//! Isolated allocation baselines for the current Indicator execution seam.
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
    math_operators::{MINMAXINDEXOutputMut, MINMAXOutputMut, MINMAX, MINMAXINDEX},
    overlap::SMA,
    price_transform::{AVGPRICEInput, AVGPRICE},
    Float, Indicator, StreamingIndicator,
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

fn print_profile<T>(scenario: &str, call: impl FnOnce() -> T) {
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
}

fn profile_setup() {
    print_profile(&format!("setup/SMA_instance/period_{PERIOD}"), || {
        SMA::new(PERIOD).expect("valid period")
    });

    print_profile(&format!("setup/per_worker_SMA_instances/{WORKERS}"), || {
        (0..WORKERS)
            .map(|_| SMA::new(PERIOD).expect("valid period"))
            .collect::<Vec<_>>()
    });

    print_profile(
        &format!(
            "setup/parameter_sweep_SMA_instances/{}",
            SWEEP_PERIODS.len()
        ),
        || {
            SWEEP_PERIODS
                .iter()
                .map(|&period| SMA::new(period).expect("valid period"))
                .collect::<Vec<_>>()
        },
    );

    print_profile(
        &format!("setup/streaming_SMA_instances/{STREAM_INSTRUMENTS}"),
        || {
            (0..STREAM_INSTRUMENTS)
                .map(|_| SMA::new(PERIOD).expect("valid period"))
                .collect::<Vec<_>>()
        },
    );
}

fn profile_one_shot() {
    let input = series_fixture(PROFILE_SIZE, 0);
    let ohlc = ohlc_fixture(PROFILE_SIZE);

    let sma = SMA::new(PERIOD).expect("valid period");
    let mut sma_output = vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)];
    print_profile(
        &format!("one_shot/SMA/caller_compact/{PROFILE_SIZE}"),
        || {
            Indicator::compute(&sma, input.as_slice(), sma_output.as_mut_slice())
                .expect("valid SMA fixture")
        },
    );
    print_profile(
        &format!("one_shot/SMA/owned_legacy_aligned/{PROFILE_SIZE}"),
        || Indicator::compute_to_vec(&sma, input.as_slice()).expect("valid SMA fixture"),
    );

    let avgprice = AVGPRICE::new().expect("valid AVGPRICE configuration");
    let avgprice_input = AVGPRICEInput {
        open: ohlc.open.as_slice(),
        high: ohlc.high.as_slice(),
        low: ohlc.low.as_slice(),
        close: ohlc.close.as_slice(),
    };
    let mut avgprice_output = vec![0.0 as Float; PROFILE_SIZE];
    print_profile(
        &format!("one_shot/AVGPRICE/caller_compact/{PROFILE_SIZE}"),
        || {
            Indicator::compute(&avgprice, avgprice_input, avgprice_output.as_mut_slice())
                .expect("valid AVGPRICE fixture")
        },
    );
    print_profile(
        &format!("one_shot/AVGPRICE/owned_legacy_aligned/{PROFILE_SIZE}"),
        || Indicator::compute_to_vec(&avgprice, avgprice_input).expect("valid AVGPRICE fixture"),
    );

    let minmax = MINMAX::new(PERIOD).expect("valid period");
    let output_len = output_len(PROFILE_SIZE, PERIOD);
    let mut min = vec![0.0 as Float; output_len];
    let mut max = vec![0.0 as Float; output_len];
    print_profile(
        &format!("one_shot/MINMAX/caller_compact/{PROFILE_SIZE}"),
        || {
            Indicator::compute(
                &minmax,
                input.as_slice(),
                MINMAXOutputMut {
                    min: min.as_mut_slice(),
                    max: max.as_mut_slice(),
                },
            )
            .expect("valid MINMAX fixture")
        },
    );
    print_profile(
        &format!("one_shot/MINMAX/owned_legacy_aligned/{PROFILE_SIZE}"),
        || Indicator::compute_to_vec(&minmax, input.as_slice()).expect("valid MINMAX fixture"),
    );

    let minmaxindex = MINMAXINDEX::new(PERIOD).expect("valid period");
    let mut min_idx = vec![0_i32; output_len];
    let mut max_idx = vec![0_i32; output_len];
    print_profile(
        &format!("one_shot/MINMAXINDEX/caller_compact/{PROFILE_SIZE}"),
        || {
            Indicator::compute(
                &minmaxindex,
                input.as_slice(),
                MINMAXINDEXOutputMut {
                    min_idx: min_idx.as_mut_slice(),
                    max_idx: max_idx.as_mut_slice(),
                },
            )
            .expect("valid MINMAXINDEX fixture")
        },
    );
    print_profile(
        &format!("one_shot/MINMAXINDEX/owned_legacy_aligned/{PROFILE_SIZE}"),
        || {
            Indicator::compute_to_vec(&minmaxindex, input.as_slice())
                .expect("valid MINMAXINDEX fixture")
        },
    );
}

fn profile_repeated_workloads() {
    let universe = (0..UNIVERSE_INSTRUMENTS)
        .map(|seed| series_fixture(PROFILE_SIZE, seed))
        .collect::<Vec<_>>();
    let sma = SMA::new(PERIOD).expect("valid period");
    let mut universe_output = vec![0.0 as Float; output_len(PROFILE_SIZE, PERIOD)];
    print_profile(
        &format!("repeated/universe_SMA/caller_compact/{UNIVERSE_INSTRUMENTS}x{PROFILE_SIZE}"),
        || {
            let mut last_range = None;
            for series in &universe {
                last_range = Some(
                    Indicator::compute(&sma, series.as_slice(), universe_output.as_mut_slice())
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
        .map(|&period| SMA::new(period).expect("valid period"))
        .collect::<Vec<_>>();
    let mut sweep_output = vec![0.0 as Float; PROFILE_SIZE];
    print_profile(
        &format!(
            "repeated/parameter_sweep_SMA/caller_compact/{}x{PROFILE_SIZE}",
            SWEEP_PERIODS.len()
        ),
        || {
            let mut last_range = None;
            for indicator in &sweep {
                last_range = Some(
                    Indicator::compute(
                        indicator,
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

    let workers = (0..WORKERS)
        .map(|_| SMA::new(PERIOD).expect("valid period"))
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
            for ((indicator, input), output) in workers
                .iter()
                .zip(worker_inputs.iter())
                .zip(worker_outputs.iter_mut())
            {
                last_range = Some(
                    Indicator::compute(indicator, input.as_slice(), output.as_mut_slice())
                        .expect("valid per-worker fixture"),
                );
                black_box(output.as_slice());
            }
            last_range
        },
    );

    let minmax = MINMAX::new(PERIOD).expect("valid period");
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
                    Indicator::compute(
                        &minmax,
                        minmax_input.as_slice(),
                        MINMAXOutputMut {
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
    let mut streams = (0..STREAM_INSTRUMENTS)
        .map(|_| SMA::new(PERIOD).expect("valid period"))
        .collect::<Vec<_>>();
    print_profile(
        &format!("streaming/SMA/independent_instances/{STREAM_INSTRUMENTS}x{PROFILE_SIZE}"),
        || {
            let mut last_output = None;
            for tick in &stream_inputs {
                for (stream, &input) in streams.iter_mut().zip(tick.iter()) {
                    last_output =
                        StreamingIndicator::next(stream, input).expect("valid streaming fixture");
                    black_box(last_output);
                }
            }
            last_output
        },
    );
}

fn main() {
    println!("scenario\tallocation_operations\tgross_allocated_bytes\tpeak_incremental_bytes\tretained_bytes");
    profile_setup();
    profile_one_shot();
    profile_repeated_workloads();
}
