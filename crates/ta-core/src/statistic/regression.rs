//! Linear Regression family and Time Series Forecast.

use super::moments::{statistic_lookback, RegressionFit, RollingRegression};
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

#[cfg(feature = "f32")]
const RAD_TO_DEG: Float = 180.0 as Float / core::f32::consts::PI;
#[cfg(not(feature = "f32"))]
const RAD_TO_DEG: Float = 180.0 as Float / core::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
enum RegressionProjection {
    Endpoint,
    Slope,
    Intercept,
    Angle,
    Forecast,
}

#[inline]
fn project(fit: RegressionFit, period: usize, projection: RegressionProjection) -> Float {
    match projection {
        RegressionProjection::Endpoint => fit.slope.mul_add((period - 1) as Float, fit.intercept),
        RegressionProjection::Slope => fit.slope,
        RegressionProjection::Intercept => fit.intercept,
        RegressionProjection::Angle => fit.slope.atan() * RAD_TO_DEG,
        RegressionProjection::Forecast => fit.slope.mul_add(period as Float, fit.intercept),
    }
}

fn validate_regression_input(real: &[Float], timeperiod: usize) -> Result<(usize, usize)> {
    let lookback = statistic_lookback(timeperiod, 2, 0)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    Ok((lookback, count))
}

fn regression_kernel(
    real: &[Float],
    timeperiod: usize,
    lookback: usize,
    count: usize,
    projection: RegressionProjection,
    regression: &mut RollingRegression,
    out_real: &mut [Float],
) -> OutputRange {
    if count == 0 {
        return OutputRange::empty();
    }

    let mut output_idx = 0usize;
    for &value in real {
        if let Some(fit) = regression.push(value) {
            out_real[output_idx] = project(fit, timeperiod, projection);
            output_idx += 1;
        }
    }

    OutputRange::new(lookback, count)
}

fn regression_batch(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    projection: RegressionProjection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let (lookback, count) = validate_regression_input(real, timeperiod)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut regression = RollingRegression::new(timeperiod);
    Ok(regression_kernel(
        real,
        timeperiod,
        lookback,
        count,
        projection,
        &mut regression,
        out_real,
    ))
}

macro_rules! define_regression_indicator {
    ($name:ident, $vec_name:ident, $config:ident, $runner:ident, $stream:ident, $projection:expr, $description:literal) => {
        #[doc = concat!("TA-Lib-style ", $description, " batch function.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real: &[Float],
            timeperiod: usize,
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            regression_batch(stringify!($name), real, timeperiod, $projection, out_real)
        }

        #[doc = concat!("Computes ", $description, " into a full-length padded vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real: &[Float], timeperiod: usize) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real.len());
            let range = $name(real, timeperiod, &mut compact)?;
            Ok(padded_from_compact(
                real.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!($description, " indicator.")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $name {
            period: usize,
            regression: RollingRegression,
        }

        impl $name {
            #[doc = concat!("Creates a new ", $description, " indicator.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                statistic_lookback(timeperiod, 2, 0)?;
                Ok(Self {
                    period: timeperiod,
                    regression: RollingRegression::new(timeperiod),
                })
            }

            /// Returns the configured period.
            pub const fn period(&self) -> usize {
                self.period
            }

            /// Computes compact outputs using this indicator's period.
            pub fn compute(&self, real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                $name(real, self.period, out_real)
            }

            /// Computes full-length padded outputs using this indicator's period.
            pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real, self.period)
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
                $name(input, self.period, output)
            }

            fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(input, self.period)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_slice("input", &[input])?;
                Ok(self
                    .regression
                    .push(input)
                    .map(|fit| project(fit, self.period, $projection)))
            }
        }

        impl Resettable for $name {
            fn reset(&mut self) {
                self.regression.reset();
            }
        }

        #[doc = concat!("Immutable ", $description, " Indicator Configuration.")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $config {
            period: usize,
        }

        impl $config {
            #[doc = concat!("Creates a ", $description, " configuration.")]
            pub fn new(timeperiod: usize) -> Result<Self> {
                statistic_lookback(timeperiod, 2, 0)?;
                Ok(Self { period: timeperiod })
            }

            /// Returns the configured Period.
            #[inline]
            pub const fn period(&self) -> usize {
                self.period
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
                let (lookback, count) = validate_regression_input(input, self.period)?;
                let mut values = Vec::with_capacity(count);
                values.resize(count, 0.0 as Float);
                let range = if count == 0 {
                    OutputRange::empty()
                } else {
                    let mut regression = RollingRegression::new(self.period);
                    regression_kernel(
                        input,
                        self.period,
                        lookback,
                        count,
                        $projection,
                        &mut regression,
                        &mut values,
                    )
                };
                CompactOutput::new(input.len(), range, values)
            }

            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input, self.period, output)
            }

            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner {
                    config: *self,
                    max_input_len,
                    regression: RollingRegression::new(self.period),
                })
            }

            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream {
                    period: self.period,
                    regression: RollingRegression::new(self.period),
                })
            }
        }

        #[doc = concat!("Reusable Prepared Batch Runner for ", $description, ".")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $runner {
            config: $config,
            max_input_len: usize,
            regression: RollingRegression,
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
                let (lookback, count) = validate_regression_input(input, self.config.period)?;
                validate_output_len(stringify!($name), output.len(), count)?;
                self.regression.reset();
                Ok(regression_kernel(
                    input,
                    self.config.period,
                    lookback,
                    count,
                    $projection,
                    &mut self.regression,
                    output,
                ))
            }
        }

        #[doc = concat!("Independent Streaming Computation for ", $description, ".")]
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone)]
        pub struct $stream {
            period: usize,
            regression: RollingRegression,
        }

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = Float;
            type TickOutput = Float;

            fn next(&mut self, input: Float) -> Result<Option<Float>> {
                validate_finite_value("input", 0, input)?;
                Ok(self
                    .regression
                    .push(input)
                    .map(|fit| project(fit, self.period, $projection)))
            }

            fn reset(&mut self) {
                self.regression.reset();
            }
        }
    };
}

define_regression_indicator!(
    LINEARREG,
    LINEARREG_vec,
    LINEARREGConfig,
    LINEARREGBatchRunner,
    LINEARREGStream,
    RegressionProjection::Endpoint,
    "Linear Regression"
);
define_regression_indicator!(
    LINEARREG_SLOPE,
    LINEARREG_SLOPE_vec,
    LINEARREG_SLOPEConfig,
    LINEARREG_SLOPEBatchRunner,
    LINEARREG_SLOPEStream,
    RegressionProjection::Slope,
    "Linear Regression Slope"
);
define_regression_indicator!(
    LINEARREG_INTERCEPT,
    LINEARREG_INTERCEPT_vec,
    LINEARREG_INTERCEPTConfig,
    LINEARREG_INTERCEPTBatchRunner,
    LINEARREG_INTERCEPTStream,
    RegressionProjection::Intercept,
    "Linear Regression Intercept"
);
define_regression_indicator!(
    LINEARREG_ANGLE,
    LINEARREG_ANGLE_vec,
    LINEARREG_ANGLEConfig,
    LINEARREG_ANGLEBatchRunner,
    LINEARREG_ANGLEStream,
    RegressionProjection::Angle,
    "Linear Regression Angle"
);
define_regression_indicator!(
    TSF,
    TSF_vec,
    TSFConfig,
    TSFBatchRunner,
    TSFStream,
    RegressionProjection::Forecast,
    "Time Series Forecast"
);
