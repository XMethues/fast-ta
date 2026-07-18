//! Linear Regression family and Time Series Forecast.

use super::moments::{statistic_lookback, RegressionFit, RollingRegression};
use crate::{
    compact_buffer, padded_from_compact, validate_finite_slice, validate_input_len,
    validate_output_len, Float, Indicator, OutputRange, Resettable, Result, StreamingIndicator,
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

fn regression_batch(
    name: &str,
    real: &[Float],
    timeperiod: usize,
    projection: RegressionProjection,
    out_real: &mut [Float],
) -> Result<OutputRange> {
    let lookback = statistic_lookback(timeperiod, 2, 0)?;
    validate_finite_slice("real", real)?;
    let count = validate_input_len(real.len(), lookback)?;
    validate_output_len(name, out_real.len(), count)?;

    if count == 0 {
        return Ok(OutputRange::empty());
    }

    let mut regression = RollingRegression::new(timeperiod);
    let mut output_idx = 0usize;
    for &value in real {
        if let Some(fit) = regression.push(value) {
            out_real[output_idx] = project(fit, timeperiod, projection);
            output_idx += 1;
        }
    }

    Ok(OutputRange::new(lookback, count))
}

macro_rules! define_regression_indicator {
    ($name:ident, $vec_name:ident, $projection:expr, $description:literal) => {
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
    };
}

define_regression_indicator!(
    LINEARREG,
    LINEARREG_vec,
    RegressionProjection::Endpoint,
    "Linear Regression"
);
define_regression_indicator!(
    LINEARREG_SLOPE,
    LINEARREG_SLOPE_vec,
    RegressionProjection::Slope,
    "Linear Regression Slope"
);
define_regression_indicator!(
    LINEARREG_INTERCEPT,
    LINEARREG_INTERCEPT_vec,
    RegressionProjection::Intercept,
    "Linear Regression Intercept"
);
define_regression_indicator!(
    LINEARREG_ANGLE,
    LINEARREG_ANGLE_vec,
    RegressionProjection::Angle,
    "Linear Regression Angle"
);
define_regression_indicator!(
    TSF,
    TSF_vec,
    RegressionProjection::Forecast,
    "Time Series Forecast"
);
