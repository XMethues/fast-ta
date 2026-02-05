//! Implementation of the Simple Moving Average (SMA) indicator.

use crate::{
    simd::{FastFloat, LANES},
    Float, Indicator,
};
use aligned_vec::AVec;

#[inline]
pub fn compute_sma(inputs: &[Float], period: usize, outputs: &mut [Float]) {
    let n = inputs.len();
    let window_size = period;
    let inv_period = 1.0 / period as Float;
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
    outputs[window_size - 1] = window_sum * inv_period;
    // Use sliding window technique: subtract old element, add new element
    for i in window_size..n {
        window_sum = window_sum - inputs[i - window_size] + inputs[i];
        outputs[i] = window_sum * inv_period;
    }
}
/// SMA indicator
pub struct SMA {
    period: usize,
    inv_period: Float,
    // 只保留计算必须的原始数据缓冲区
    buffer: AVec<Float>,
    index: usize,
    is_full: bool,
    current_sum: Float,

    // For performance
    mask: usize,
    is_power_of_two: bool,
}

impl SMA {
    /// Create a new SMA indicator with the given period.
    pub fn new(period: usize) -> Self {
        assert!(period > 0, "Period must be greater than 0");
        let is_power_of_two = period > 0 && (period & (period - 1)) == 0;
        let inv_period = 1.0 / period as Float;
        let buffer = AVec::with_capacity(64, period);

        SMA {
            period,
            inv_period,
            buffer,
            index: 0,
            is_full: false,
            current_sum: 0.0,
            mask: if period > 0 { period - 1 } else { 0 },
            is_power_of_two,
        }
    }
    /// warm up sma state
    pub fn from_data(period: usize, data: &[Float]) -> Self {
        let mut sma = Self::new(period);
        // 我们只需要最近的 period 个价格来填充状态
        let start = data.len().saturating_sub(period);
        let relevant_prices = &data[start..];
        for &p in relevant_prices {
            // 更新 buffer 和 sum
            sma.buffer[sma.index] = p;
            sma.current_sum += p;

            // 检查是否填满
            if !sma.is_full && sma.index == sma.period - 1 {
                sma.is_full = true;
            }

            // 移动指针
            if sma.is_power_of_two {
                sma.index = (sma.index + 1) & sma.mask;
            } else {
                sma.index = (sma.index + 1) % sma.period;
            }
        }
        sma
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
        // 1. 获取即将被替换的旧值 (O(1) 访问)
        let old_val = self.buffer[self.index];

        // 2. 更新累加和：加新减旧 (无循环)
        self.current_sum = self.current_sum - old_val + input;

        // 3. 将新值存入缓冲区
        self.buffer[self.index] = input;

        // 4. 检查是否刚填满缓冲区（关键优化：用 bool 替代 usize 计数器）
        if !self.is_full && self.index == self.period - 1 {
            self.is_full = true;
        }

        // 5. 指针跳转逻辑 (性能关键点)
        if self.is_power_of_two {
            // 如果 period 是 2 的幂，使用按位与 (&) 代替取模 (%)
            // 耗时从 ~20 纳秒降低到 <1 纳秒
            self.index = (self.index + 1) & self.mask;
        } else {
            // 普通取模运算
            self.index = (self.index + 1) % self.period;
        }

        // 6. 返回结果：使用预计算的倒数进行乘法 (比除法快 10 倍以上)
        if self.is_full {
            self.current_sum * self.inv_period
        } else {
            Float::NAN
        }
    }
}
