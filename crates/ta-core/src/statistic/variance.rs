//! Variance (VAR) and Standard Deviation (STDDEV).

use super::moments::{
    statistic_lookback, validate_nbdev, RollingMoments, DEFAULT_NBDEV, TA_EPSILON,
};
use crate::common::validate_finite_value;
use crate::{
    compact_buffer, padded_from_compact, validate_finite_slice, validate_input_len,
    validate_output_len, CompactOutput, Float, Indicator, IndicatorConfig, OutputRange,
    PreparedBatchRunner, Resettable, Result, StreamingComputation, StreamingIndicator, TalibError,
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

fn validate_variance_input(
    real: &[Float],
    timeperiod: usize,
    nbdev: Float,
    minimum_period: usize,
) -> Result<(usize, usize)> {
    let lookback = statistic_lookback(timeperiod, minimum_period, 0)?;
    validate_nbdev(nbdev)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

#[inline]
fn variance_kernel<F>(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    mut project_value: F,
    out_real: &mut [Float],
) -> OutputRange
where
    F: FnMut(Float) -> Float,
{
    if count == 0 {
        return OutputRange::empty();
    }

    let mut sum = 0.0 as Float;
    let mut sum_sq = 0.0 as Float;
    for &value in &real[..timeperiod] {
        sum += value;
        sum_sq += value * value;
    }

    let period = timeperiod as Float;
    for output_idx in 0..count {
        let mean = sum / period;
        out_real[output_idx] = project_value(sum_sq / period - mean * mean);
        if output_idx + 1 < count {
            let old = real[output_idx];
            let new = real[output_idx + timeperiod];
            sum -= old;
            sum_sq -= old * old;
            sum += new;
            sum_sq += new * new;
        }
    }

    OutputRange::new(lookback, count)
}
fn variance_moments_kernel(
    real: &[Float],
    lookback: usize,
    count: usize,
    nbdev: Float,
    projection: VarianceProjection,
    moments: &mut RollingMoments,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut output_idx = 0usize;
    for &value in real {
        if let Some(variance) = moments.push(value) {
            out_real[output_idx] = project(variance, nbdev, projection);
            output_idx += 1;
        }
    }
    OutputRange::new(lookback, count)
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
    let (lookback, count) = validate_variance_input(real, timeperiod, nbdev, minimum_period)?;
    validate_output_len(name, out_real.len(), count)?;
    let mut moments = RollingMoments::new(timeperiod);
    Ok(variance_moments_kernel(
        real,
        lookback,
        count,
        nbdev,
        projection,
        &mut moments,
        out_real,
    ))
}

macro_rules! define_variance_indicator {
    (
        $name:ident,
        $vec_name:ident,
        $default_name:ident,
        $default_vec_name:ident,
        $config:ident,
        $runner:ident,
        $stream:ident,
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

        #[doc = concat!("Immutable ", $description, " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $config {
            period: usize,
            nbdev: Float,
        }

        impl $config {
            #[doc = concat!("Creates a ", $description, " configuration.")]
            pub fn new(timeperiod: usize, nbdev: Float) -> Result<Self> {
                statistic_lookback(timeperiod, $minimum_period, 0)?;
                validate_nbdev(nbdev)?;
                Ok(Self {
                    period: timeperiod,
                    nbdev,
                })
            }

            #[doc = concat!("Creates a ", $description, " configuration with nbdev 1.0.")]
            pub fn with_default_nbdev(timeperiod: usize) -> Result<Self> {
                Self::new(timeperiod, DEFAULT_NBDEV)
            }

            /// Returns the configured Period.
            #[inline]
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Returns the configured deviation multiplier.
            #[inline]
            pub const fn nbdev(&self) -> Float {
                self.nbdev
            }
        }

        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = &'a [Float];
            type Output = Vec<Float>;
            type OutputMut<'a> = &'a mut [Float];
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                self.period - 1
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let (lookback, count) =
                    validate_variance_input(input, self.period, self.nbdev, $minimum_period)?;
                let mut values = Vec::with_capacity(count);
                values.resize(count, 0.0 as Float);
                let range = variance_kernel(
                    input,
                    self.period,
                    lookback,
                    count,
                    |variance| project(variance, self.nbdev, $projection),
                    &mut values,
                );
                CompactOutput::new(input.len(), range, values)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input, self.period, self.nbdev, output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                let moments = match $projection {
                    VarianceProjection::Variance => None,
                    VarianceProjection::StandardDeviation => {
                        Some(RollingMoments::new(self.period))
                    }
                };
                Ok($runner {
                    config: *self,
                    max_input_len,
                    moments,
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    nbdev: self.nbdev,
                    moments: RollingMoments::new(self.period),
                })
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for ", $description, ".")]
        #[derive(Debug, Clone)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
            moments: Option<RollingMoments>,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            fn compute_into<'a>(
                &mut self,
                input: <$config as IndicatorConfig>::Input<'a>,
                output: <$config as IndicatorConfig>::OutputMut<'a>,
            ) -> Result<OutputRange>
            where
                $config: 'a,
            {
                if input.len() > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input.len(),
                    ));
                }
                let (lookback, count) = validate_variance_input(
                    input,
                    self.config.period,
                    self.config.nbdev,
                    $minimum_period,
                )?;
                validate_output_len(stringify!($name), output.len(), count)?;
                match $projection {
                    VarianceProjection::Variance => Ok(variance_kernel(
                        input,
                        self.config.period,
                        lookback,
                        count,
                        |variance| project(variance, self.config.nbdev, $projection),
                        output,
                    )),
                    VarianceProjection::StandardDeviation => {
                        let moments = self
                            .moments
                            .as_mut()
                            .expect("standard-deviation runner has rolling moments");
                        moments.reset();
                        Ok(variance_moments_kernel(
                            input,
                            lookback,
                            count,
                            self.config.nbdev,
                            $projection,
                            moments,
                            output,
                        ))
                    }
                }
            }
        }

        #[doc = concat!("Independent Streaming Computation for ", $description, ".")]
        #[derive(Debug, Clone)]
        pub struct $stream {
            nbdev: Float,
            moments: RollingMoments,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_value("input", 0, input)?;
                Ok(self
                    .moments
                    .push(input)
                    .map(|variance| project(variance, self.nbdev, $projection)))
            }

            fn reset(&mut self) {
                self.moments.reset();
            }
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

            /// Returns the configured Period.
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
    VARConfig,
    VARBatchRunner,
    VARStream,
    1,
    VarianceProjection::Variance,
    "Variance"
);
define_variance_indicator!(
    STDDEV,
    STDDEV_vec,
    STDDEV_with_default_nbdev,
    STDDEV_vec_with_default_nbdev,
    STDDEVConfig,
    STDDEVBatchRunner,
    STDDEVStream,
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
