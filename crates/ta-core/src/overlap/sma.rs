//! Implementation of the Simple Moving Average (SMA) indicator.

use crate::{
    simd::{FastFloat, LANES},
    Float, Indicator,
};
use aligned_vec::AVec;

pub fn compute_sma(inputs: &[Float], period: usize, outputs: &mut [Float]) {
    let n = inputs.len();
    let window_size = period;
    let mut window_sum = 0.0;
    let mut i = 0;
    while i + LANES <= window_size {
        let slice = &inputs[i..i + LANES];
        let chunk = FastFloat::from(slice);
        window_sum += chunk.reduce_add();
        i += LANES;
    }
    while i < window_size {
        window_sum += inputs[i];
        i += 1;
    }
    // First window result
    outputs[window_size - 1] = window_sum / window_size as Float;
    // Use sliding window technique: subtract old element, add new element
    for i in window_size..n {
        window_sum = window_sum - inputs[i - window_size] + inputs[i];
        outputs[i] = window_sum / window_size as Float;
    }
}
/// SMA indicator
pub struct SMA {
    period: usize,
    inv_period: Float,
    // 计算用的原始数据环形缓冲区
    buffer: AVec<Float>,
    buffer_index: usize,

    // 固定的结果历史环形缓冲区
    history: AVec<Float>,
    history_index: usize,
    history_cap: usize,

    count: usize,
    current_sum: Float,

    // 性能优化标记
    mask: usize,
    is_power_of_two: bool,
}

impl SMA {
    /// Create a new SMA indicator with the given period.
    pub fn new(period: usize, data: &[Float]) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        let history_limit = period.max(data.len());
        let is_power_of_two = period > 0 && (period & (period - 1)) == 0;
        let inv_period = 1.0 / period as Float;
        let buffer = AVec::with_capacity(64, period);
        let history = AVec::with_capacity(64, history_limit);

        SMA {
            period,
            inv_period,
            buffer,
            buffer_index: 0,
            history,
            history_index: 0,
            history_cap: history_limit,
            count: 0,
            current_sum: 0.0,
            mask: if period > 0 { period - 1 } else { 0 },
            is_power_of_two,
        }
    }
}

impl Indicator for SMA {
    type Input = Float;

    type Output = Float;

    fn lookback(&self) -> usize {
        self.period.saturating_sub(1)
    }

    fn compute_to_vec(&self, inputs: &[Self::Input]) -> crate::Result<Vec<Self::Output>> {
        let mut result = vec![Float::NAN; inputs.len()];
        compute_sma(inputs, self.period, &mut result);
        Ok(result)
    }

    #[inline(always)]
    fn next(&mut self, input: Float) -> Float {
        // 1. 提取旧值并更新累加和 (Rolling Sum)
        let old_val = self.buffer[self.buffer_index];
        self.current_sum = self.current_sum - old_val + input;

        // 2. 更新原始数据缓冲区
        self.buffer[self.buffer_index] = input;

        // 3. 移动计算缓冲区指针 (位运算优化)
        if self.is_power_of_two {
            self.buffer_index = (self.buffer_index + 1) & self.mask;
        } else {
            self.buffer_index = (self.buffer_index + 1) % self.period;
        }

        // 4. 更新有效数据计数
        if self.count < self.period {
            self.count += 1;
        }

        // 5. 计算 SMA 结果
        let result = if self.count >= self.period {
            self.current_sum * self.inv_period
        } else {
            Float::NAN
        };

        // 6. 存入结果历史缓冲区
        if self.history_cap > 0 {
            self.history[self.history_index] = result;
            self.history_index = (self.history_index + 1) % self.history_cap;
        }

        result
    }

    fn stream(&mut self, _inputs: &[Self::Input]) -> Vec<Option<Self::Output>> {
        // For streaming computation, we would need to maintain internal state
        // This is a simplified implementation that returns empty vector
        Vec::new()
    }
}
