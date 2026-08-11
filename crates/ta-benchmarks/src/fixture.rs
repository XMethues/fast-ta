//! Small deterministic fixtures shared by the default Criterion benchmarks.

pub fn series_fixture(size: usize, seed: usize) -> Vec<f64> {
    (0..size)
        .map(|index| {
            let trend = index as f64 * 0.001;
            let cycle = ((index * 37 + seed * 17) % 101) as f64;
            trend + cycle + 1.0
        })
        .collect()
}
