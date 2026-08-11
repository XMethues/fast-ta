// TA-Lib evaluates both input precisions in double precision. Keeping the
// recurrence in f64 also prevents branch instability on low-information series.
type Float = f64;

const HILBERT_A: Float = 0.0962 as Float;
const HILBERT_B: Float = 0.5769 as Float;
const RAD_TO_DEG: Float = 45.0 as Float / core::f64::consts::FRAC_PI_4 as Float;
const PHASE_RECURRENCE_START: usize = 37;
const PERIOD_RECURRENCE_START: usize = 12;
const SMOOTH_PRICE_SIZE: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HilbertTransition {
    pub(crate) in_phase: Float,
    pub(crate) quadrature: Float,
    pub(crate) smooth_period: Float,
    smoothed_value: Float,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct HilbertPhaseTransition {
    pub(super) today: usize,
    pub(super) phase: Float,
    pub(super) smooth_period: Float,
    pub(super) smoothed_value: Float,
}

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
pub(crate) struct HilbertState {
    observations: usize,
    price_history: [Float; 3],
    price_history_index: usize,
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
    pub(crate) const fn observations(&self) -> usize {
        self.observations
    }

    #[inline]
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(super) fn next_dc_period(&mut self, input: Float) -> Option<Float> {
        let today = self.observations;
        self.next_transition(input, PERIOD_RECURRENCE_START)
            .and_then(|transition| (today >= 32).then_some(transition.smooth_period))
    }

    #[inline]
    pub(super) fn next_phasor(&mut self, input: Float) -> Option<HilbertTransition> {
        let today = self.observations;
        self.next_transition(input, PERIOD_RECURRENCE_START)
            .filter(|_| today >= 32)
    }

    #[inline]
    pub(crate) fn next_mama_transition(&mut self, input: Float) -> Option<HilbertTransition> {
        self.next_transition(input, PERIOD_RECURRENCE_START)
    }

    #[inline]
    pub(crate) fn next_trendline_transition(&mut self, input: Float) -> Option<HilbertTransition> {
        self.next_transition(input, PHASE_RECURRENCE_START)
    }

    #[inline]
    fn next_transition(
        &mut self,
        input: Float,
        recurrence_start: usize,
    ) -> Option<HilbertTransition> {
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

        let smoothed_value = self.next_price_wma(input);
        if today < recurrence_start {
            return None;
        }

        let adjusted_period = 0.075 as Float * self.period + 0.54 as Float;
        let even = today.is_multiple_of(2);
        let detrender =
            self.detrender
                .next(smoothed_value, even, self.hilbert_index, adjusted_period);
        let quadrature =
            self.q1_transform
                .next(detrender, even, self.hilbert_index, adjusted_period);

        let in_phase = if even {
            self.i1_even_previous_3
        } else {
            self.i1_odd_previous_3
        };
        let ji = self
            .ji_transform
            .next(in_phase, even, self.hilbert_index, adjusted_period);
        let jq = self
            .jq_transform
            .next(quadrature, even, self.hilbert_index, adjusted_period);

        if even {
            self.hilbert_index += 1;
            if self.hilbert_index == 3 {
                self.hilbert_index = 0;
            }
        }

        let q2 = 0.2 as Float * (quadrature + ji) + 0.8 as Float * self.previous_q2;
        let i2 = 0.2 as Float * (in_phase - jq) + 0.8 as Float * self.previous_i2;

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

        Some(HilbertTransition {
            in_phase,
            quadrature,
            smooth_period: self.smooth_period,
            smoothed_value,
        })
    }

    #[inline(always)]
    fn next_price_wma(&mut self, input: Float) -> Float {
        self.period_wma_sub += input;
        self.period_wma_sub -= self.trailing_wma_value;
        self.period_wma_sum += input * 4.0 as Float;
        let smoothed_value = self.period_wma_sum * 0.1 as Float;
        self.period_wma_sum -= self.period_wma_sub;
        self.trailing_wma_value = self.price_history[self.price_history_index];
        self.price_history[self.price_history_index] = input;
        self.price_history_index += 1;
        if self.price_history_index == self.price_history.len() {
            self.price_history_index = 0;
        }
        smoothed_value
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct HilbertPhaseState {
    hilbert: HilbertState,
    smooth_price: [Float; SMOOTH_PRICE_SIZE],
    smooth_price_index: usize,
    dominant_cycle_phase: Float,
}

impl Default for HilbertPhaseState {
    fn default() -> Self {
        Self {
            hilbert: HilbertState::default(),
            smooth_price: [0.0 as Float; SMOOTH_PRICE_SIZE],
            smooth_price_index: 0,
            dominant_cycle_phase: 0.0 as Float,
        }
    }
}

impl HilbertPhaseState {
    #[inline]
    pub(super) const fn observations(&self) -> usize {
        self.hilbert.observations()
    }

    #[inline]
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    #[inline]
    pub(super) fn next_phase_transition(&mut self, input: Float) -> Option<HilbertPhaseTransition> {
        let today = self.hilbert.observations();
        let transition = self
            .hilbert
            .next_transition(input, PHASE_RECURRENCE_START)?;
        self.smooth_price[self.smooth_price_index] = transition.smoothed_value;
        self.smooth_price_index += 1;
        if self.smooth_price_index == SMOOTH_PRICE_SIZE {
            self.smooth_price_index = 0;
        }
        let phase = self.calculate_dominant_cycle_phase(transition.smooth_period);
        Some(HilbertPhaseTransition {
            today,
            phase,
            smooth_period: transition.smooth_period,
            smoothed_value: transition.smoothed_value,
        })
    }

    #[inline]
    pub(super) fn next_dc_phase(&mut self, input: Float) -> Option<Float> {
        self.next_phase_transition(input)
            .and_then(|transition| (transition.today >= 63).then_some(transition.phase))
    }

    #[inline]
    fn calculate_dominant_cycle_phase(&mut self, smooth_period: Float) -> Float {
        let period = (smooth_period + 0.5 as Float) as usize;
        let mut real_part = 0.0 as Float;
        let mut imaginary_part = 0.0 as Float;
        let mut index = if self.smooth_price_index == 0 {
            SMOOTH_PRICE_SIZE - 1
        } else {
            self.smooth_price_index - 1
        };

        for offset in 0..period {
            let angle =
                offset as Float * 2.0 as Float * core::f64::consts::PI as Float / period as Float;
            let value = self.smooth_price[index];
            real_part += angle.sin() * value;
            imaginary_part += angle.cos() * value;
            index = if index == 0 {
                SMOOTH_PRICE_SIZE - 1
            } else {
                index - 1
            };
        }

        if imaginary_part.abs() > 0.0 as Float {
            self.dominant_cycle_phase = (real_part / imaginary_part).atan() * RAD_TO_DEG;
        } else if real_part < 0.0 as Float {
            self.dominant_cycle_phase -= 90.0 as Float;
        } else if real_part > 0.0 as Float {
            self.dominant_cycle_phase += 90.0 as Float;
        }
        self.dominant_cycle_phase += 90.0 as Float;
        self.dominant_cycle_phase += 360.0 as Float / smooth_period;
        if imaginary_part < 0.0 as Float {
            self.dominant_cycle_phase += 180.0 as Float;
        }
        if self.dominant_cycle_phase > 315.0 as Float {
            self.dominant_cycle_phase -= 360.0 as Float;
        }
        self.dominant_cycle_phase
    }
}
