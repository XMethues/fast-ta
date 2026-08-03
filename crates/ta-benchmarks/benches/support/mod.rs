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
}

pub(crate) fn series_fixture(size: usize, seed: usize) -> Vec<Float> {
    (0..size)
        .map(|index| {
            let trend = index as Float * 0.001 as Float;
            let cycle = ((index * 37 + seed * 17) % 101) as Float;
            trend + cycle + 1.0 as Float
        })
        .collect()
}

pub(crate) fn ohlc_fixture(size: usize) -> OhlcFixture {
    let close = series_fixture(size, 0);
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
    }
}

pub(crate) fn output_len(input_len: usize, period: usize) -> usize {
    input_len.saturating_sub(period.saturating_sub(1))
}
