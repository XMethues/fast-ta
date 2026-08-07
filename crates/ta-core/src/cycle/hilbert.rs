// TA-Lib evaluates both input precisions in double precision. Keeping the
// recurrence in f64 also prevents branch instability on low-information series.
type Float = f64;

const HILBERT_A: Float = 0.0962 as Float;
const HILBERT_B: Float = 0.5769 as Float;
const RAD_TO_DEG: Float = 45.0 as Float / core::f64::consts::FRAC_PI_4 as Float;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct HilbertTransform {
    odd: [Float; 3],
    even: [Float; 3],
    previous_odd: Float,
    previous_even: Float,
    previous_input_odd: Float,
    previous_input_even: Float,
}

impl HilbertTransform {
    #[inline(always)]
    fn next(
        &mut self,
        input: Float,
        even: bool,
        hilbert_index: usize,
        adjusted_period: Float,
    ) -> Float {
        let transformed = HILBERT_A * input;
        let (history, previous, previous_input) = if even {
            (
                &mut self.even,
                &mut self.previous_even,
                &mut self.previous_input_even,
            )
        } else {
            (
                &mut self.odd,
                &mut self.previous_odd,
                &mut self.previous_input_odd,
            )
        };

        let mut output = -history[hilbert_index];
        history[hilbert_index] = transformed;
        output += transformed;
        output -= *previous;
        *previous = HILBERT_B * *previous_input;
        output += *previous;
        *previous_input = input;
        output * adjusted_period
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct HilbertState {
    observations: usize,
    price_history: [Float; 3],
    period_wma_sum: Float,
    period_wma_sub: Float,
    trailing_wma_value: Float,
    hilbert_index: usize,
    detrender: HilbertTransform,
    q1_transform: HilbertTransform,
    ji_transform: HilbertTransform,
    jq_transform: HilbertTransform,
    period: Float,
    smooth_period: Float,
    previous_q2: Float,
    previous_i2: Float,
    real_part: Float,
    imaginary_part: Float,
    i1_odd_previous_2: Float,
    i1_odd_previous_3: Float,
    i1_even_previous_2: Float,
    i1_even_previous_3: Float,
}

impl HilbertState {
    #[inline]
    pub(super) const fn observations(&self) -> usize {
        self.observations
    }

    #[inline]
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(super) fn next_dc_period(&mut self, input: Float) -> Option<Float> {
        let today = self.observations;
        self.observations += 1;

        match today {
            0 => {
                self.period_wma_sub = input;
                self.period_wma_sum = input;
                self.price_history[0] = input;
                return None;
            }
            1 => {
                self.period_wma_sub += input;
                self.period_wma_sum += input * 2.0 as Float;
                self.price_history[1] = input;
                return None;
            }
            2 => {
                self.period_wma_sub += input;
                self.period_wma_sum += input * 3.0 as Float;
                self.price_history[2] = input;
                return None;
            }
            _ => {}
        }

        let smoothed_value = self.next_price_wma(today, input);
        if today < 12 {
            return None;
        }

        let adjusted_period = 0.075 as Float * self.period + 0.54 as Float;
        let even = today.is_multiple_of(2);
        let detrender =
            self.detrender
                .next(smoothed_value, even, self.hilbert_index, adjusted_period);
        let q1 = self
            .q1_transform
            .next(detrender, even, self.hilbert_index, adjusted_period);

        let delayed_i1 = if even {
            self.i1_even_previous_3
        } else {
            self.i1_odd_previous_3
        };
        let ji = self
            .ji_transform
            .next(delayed_i1, even, self.hilbert_index, adjusted_period);
        let jq = self
            .jq_transform
            .next(q1, even, self.hilbert_index, adjusted_period);

        if even {
            self.hilbert_index = (self.hilbert_index + 1) % 3;
        }

        let q2 = 0.2 as Float * (q1 + ji) + 0.8 as Float * self.previous_q2;
        let i2 = 0.2 as Float * (delayed_i1 - jq) + 0.8 as Float * self.previous_i2;

        if even {
            self.i1_odd_previous_3 = self.i1_odd_previous_2;
            self.i1_odd_previous_2 = detrender;
        } else {
            self.i1_even_previous_3 = self.i1_even_previous_2;
            self.i1_even_previous_2 = detrender;
        }

        self.real_part = 0.2 as Float * (i2 * self.previous_i2 + q2 * self.previous_q2)
            + 0.8 as Float * self.real_part;
        self.imaginary_part = 0.2 as Float * (i2 * self.previous_q2 - q2 * self.previous_i2)
            + 0.8 as Float * self.imaginary_part;
        self.previous_q2 = q2;
        self.previous_i2 = i2;

        let previous_period = self.period;
        if self.imaginary_part != 0.0 as Float && self.real_part != 0.0 as Float {
            self.period =
                360.0 as Float / ((self.imaginary_part / self.real_part).atan() * RAD_TO_DEG);
        }
        self.period = self.period.min(1.5 as Float * previous_period);
        self.period = self.period.max(0.67 as Float * previous_period);
        self.period = self.period.clamp(6.0 as Float, 50.0 as Float);
        self.period = 0.2 as Float * self.period + 0.8 as Float * previous_period;
        self.smooth_period = 0.33 as Float * self.period + 0.67 as Float * self.smooth_period;

        (today >= 32).then_some(self.smooth_period)
    }

    #[inline(always)]
    fn next_price_wma(&mut self, today: usize, input: Float) -> Float {
        self.period_wma_sub += input;
        self.period_wma_sub -= self.trailing_wma_value;
        self.period_wma_sum += input * 4.0 as Float;
        let smoothed_value = self.period_wma_sum * 0.1 as Float;
        self.period_wma_sum -= self.period_wma_sub;
        self.trailing_wma_value = self.price_history[today % self.price_history.len()];
        self.price_history[today % self.price_history.len()] = input;
        smoothed_value
    }
}
