//! Variance (VAR) and Standard Deviation (STDDEV).

use super::moments::{
    statistic_lookback, validate_nbdev, RollingMoments, DEFAULT_NBDEV, TA_EPSILON,
};
use crate::{
    compact_buffer, padded_from_compact, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

#[derive(Debug, Clone, Copy)]
enum VarianceProjection {
    Variance,
    StandardDeviation,
}

#[inline]
fn project(variance: Float, nbdev: Float, projection: VarianceProjection) -> Float {
    match projection {
        VarianceProjection::Variance => variance,
        VarianceProjection::StandardDeviation if variance <= TA_EPSILON => 0.0 as Float,
        VarianceProjection::StandardDeviation => variance.sqrt() * nbdev,
    }
}

fn variance_batch(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    nbdev: Float,
    minimum_period: usize,
    projection: VarianceProjection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, minimum_period, 0)?;
    validate_nbdev(nbdev)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut moments = RollingMoments::new(timeperiod);
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(variance) = moments.push(value) {
            out_real[output_idx] = project(variance, nbdev, projection);
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

macro_rules! define_variance_indicator {
    (
        $name:ident,
        $vec_name:ident,
        $default_name:ident,
        $default_vec_name:ident,
        $minimum_period:expr,
        $projection:expr,
        $description:literal
    ) => {
        #[doc = concat!("TA-Lib-style ", $description, " batch function.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real: &[Float],
            timeperiod: usize,
            nbdev: Float,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            variance_batch(
                stringify!($name),
                real,
                timeperiod,
                nbdev,
                $minimum_period,
                $projection,
                out_real,
            )
        }

        #[doc = concat!("TA-Lib-style ", $description, " with default nbdev 1.0.")]
        #[allow(non_snake_case)]
        pub fn $default_name(
            real: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            $name(real, timeperiod, DEFAULT_NBDEV, out_real)
        }

        #[doc = concat!("Computes ", $description, " into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real: &[Float], timeperiod: usize, nbdev: Float) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real.len());
            let range = $name(real, timeperiod, nbdev, &mut compact)?;
            Ok(padded_from_compact(
                real.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!("Computes ", $description, " with default nbdev 1.0 into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $default_vec_name(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
            $vec_name(real, timeperiod, DEFAULT_NBDEV)
        }

        #[doc = concat!($description, " indicator.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            nbdev: Float,
            moments: RollingMoments,
        }

        impl $name {
            #[doc = concat!("Creates a new ", $description, " indicator.")]
            pub fn new(timeperiod: usize, nbdev: Float) -> Result<Self> {
                statistic_lookback(timeperiod, $minimum_period, 0)?;
                validate_nbdev(nbdev)?;
                Ok(Self {
                    period: timeperiod,
                    nbdev,
                    moments: RollingMoments::new(timeperiod),
                })
            }

            #[doc = concat!("Creates a new ", $description, " indicator with nbdev 1.0.")]
            pub fn with_default_nbdev(timeperiod: usize) -> Result<Self> {
                Self::new(timeperiod, DEFAULT_NBDEV)
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Returns the configured deviation multiplier.
            pub const fn nbdev(&self) -> Float {
                self.nbdev
            }

            /// Computes compact outputs using this indicator's configuration.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, self.nbdev, out_real)
            }

            /// Computes full-length padded outputs using this indicator's configuration.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period, self.nbdev)
            }

            /// Checked streaming update that returns `Float::NAN` during warm-up.
            pub fn next_checked(&mut self, input: Float) -> Result<Float> {
                Ok(self.next(input)?.unwrap_or(Float::NAN))
            }
        }

        impl Indicator for $name {
            type Input<'a> = &'a [Float];
            type OutputMut<'a> = &'a mut [Float];
            type OutputOwned = Vec<Float>;

            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input, self.period, self.nbdev, output)
            }

            fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(input, self.period, self.nbdev)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                Ok(self
                    .moments
                    .push(input)
                    .map(|variance| project(variance, self.nbdev, $projection)))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                self.moments.reset();
            }
        }
    };
}

define_variance_indicator!(
    VAR,
    VAR_vec,
    VAR_with_default_nbdev,
    VAR_vec_with_default_nbdev,
    1,
    VarianceProjection::Variance,
    "Variance"
);
define_variance_indicator!(
    STDDEV,
    STDDEV_vec,
    STDDEV_with_default_nbdev,
    STDDEV_vec_with_default_nbdev,
    2,
    VarianceProjection::StandardDeviation,
    "Standard Deviation"
);

#[cfg(test)]
mod tests {
    use super::{project, VarianceProjection, TA_EPSILON};
    use crate::Float;

    #[test]
    fn stddev_treats_epsilon_as_zero() {
        assert_eq!(
            project(
                TA_EPSILON,
                1.0 as Float,
                VarianceProjection::StandardDeviation,
            ),
            0.0 as Float
        );
    }
}
