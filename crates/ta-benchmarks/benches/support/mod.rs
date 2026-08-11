use ta_core::Float;

pub(crate) const PERIOD: usize = 14;
pub(crate) const REPEATED_SERIES_LEN: usize = 4_096;
pub(crate) const UNIVERSE_INSTRUMENTS: usize = 128;
pub(crate) const SWEEP_PERIODS: &[usize] = &[5, 14, 50, 200];
pub(crate) const WORKERS: usize = 4;
pub(crate) const STREAM_INSTRUMENTS: usize = 16;

pub(crate) struct OhlcFixture {
    pub(crate) open: Vec<Float>,
    pub(crate) high: Vec<Float>,
    pub(crate) low: Vec<Float>,
    pub(crate) close: Vec<Float>,
    pub(crate) volume: Vec<Float>,
}

pub(crate) use ta_benchmarks::fixture::series_fixture;

pub(crate) fn ohlc_fixture(size: usize) -> OhlcFixture {
    let close = series_fixture(size, 0);
    let volume = series_fixture(size, 1);
    let open = close
        .iter()
        .enumerate()
        .map(|(index, value)| *value + (index % 5) as Float * 0.01 as Float)
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::max(*open, *close) + 1.0 as Float)
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(close.iter())
        .map(|(open, close)| Float::min(*open, *close) - 1.0 as Float)
        .collect::<Vec<_>>();
    OhlcFixture {
        open,
        high,
        low,
        close,
        volume,
    }
}

pub(crate) fn output_len(input_len: usize, period: usize) -> usize {
    input_len.saturating_sub(period.saturating_sub(1))
}
