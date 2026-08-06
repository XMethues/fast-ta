//! Arithmetic Math Operators.
//!
//! Inputs must be equal-length finite Observation Series. Per-element arithmetic
//! follows IEEE-754: finite inputs may produce non-finite results, including
//! positive or negative infinity for nonzero division by zero, `NaN` for zero
//! divided by zero, and infinity on overflow.

use crate::common::validate_finite_value;
use crate::{
    validate_all_same_len, validate_finite_slices, validate_output_len, CompactOutput, Float,
    IndicatorConfig, OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Borrowed SoA inputs for binary math operators.
#[derive(Debug, Clone, Copy)]
pub struct BinaryInput<'a> {
    /// First real-valued input series.
    pub real0: &'a [Float],
    /// Second real-valued input series.
    pub real1: &'a [Float],
}

/// One streaming tick for binary math operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryTick {
    /// First real-valued input.
    pub real0: Float,
    /// Second real-valued input.
    pub real1: Float,
}

macro_rules! define_binary_operator {
    (
        $name:ident,
        $config:ident,
        $runner:ident,
        $stream:ident,
        $operation:expr
    ) => {
        #[doc = concat!("Immutable ", stringify!($name), " Indicator Configuration.")]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $config;

        impl $config {
            #[doc = concat!("Creates the parameter-free ", stringify!($name), " configuration.")]
            #[inline]
            pub const fn new() -> Self {
                Self
            }

            #[inline]
            fn validate_input(input: BinaryInput<'_>) -> Result<usize> {
                let len = validate_all_same_len(&[
                    ("real0", input.real0.len()),
                    ("real1", input.real1.len()),
                ])?;
                validate_finite_slices(&[("real0", input.real0), ("real1", input.real1)])?;
                Ok(len)
            }

            #[inline(always)]
            fn compute_validated(input: BinaryInput<'_>, output: &mut [Float]) -> OutputRange {
                let operation = $operation;
                for idx in 0..input.real0.len() {
                    output[idx] = operation(input.real0[idx], input.real1[idx]);
                }
                OutputRange::new(0, input.real0.len())
            }

            #[inline(always)]
            fn compute_tick(input: BinaryTick) -> Result<Float> {
                validate_finite_value("real0", 0, input.real0)?;
                validate_finite_value("real1", 0, input.real1)?;
                let operation = $operation;
                Ok(operation(input.real0, input.real1))
            }
        }

        #[doc = concat!("TA-Lib-style ", stringify!($name), " binary math operator.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real0: &[Float],
            real1: &[Float],
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            let input = BinaryInput { real0, real1 };
            let len = $config::validate_input(input)?;
            validate_output_len(stringify!($name), out_real.len(), len)?;
            Ok($config::compute_validated(input, out_real))
        }

        impl crate::traits::sealed::Sealed for $config {}

        impl IndicatorConfig for $config {
            type Input<'a> = BinaryInput<'a>;
            type Output = Vec<Float>;
            type OutputMut<'a> = &'a mut [Float];
            type BatchRunner = $runner;
            type Stream = $stream;

            #[inline]
            fn lookback(&self) -> usize {
                0
            }

            fn compute<'a>(&self, input: Self::Input<'a>) -> Result<CompactOutput<Self::Output>> {
                let len = Self::validate_input(input)?;
                let mut values = Vec::with_capacity(len);
                values.resize(len, 0.0 as Float);
                let range = Self::compute_validated(input, &mut values);
                CompactOutput::new(len, range, values)
            }

            #[inline(always)]
            fn compute_into<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                let len = Self::validate_input(input)?;
                validate_output_len(stringify!($name), output.len(), len)?;
                Ok(Self::compute_validated(input, output))
            }

            #[inline]
            fn prepare_batch(&self, max_input_len: usize) -> Result<Self::BatchRunner> {
                Ok($runner { max_input_len })
            }

            #[inline]
            fn stream(&self) -> Result<Self::Stream> {
                Ok($stream)
            }
        }

        #[doc = concat!("Prepared Batch Runner for ", stringify!($name), ".")]
        #[derive(Debug, Clone)]
        pub struct $runner {
            max_input_len: usize,
        }

        impl crate::traits::sealed::Sealed for $runner {}

        impl PreparedBatchRunner<$config> for $runner {
            #[inline]
            fn max_input_len(&self) -> usize {
                self.max_input_len
            }

            #[inline(always)]
            fn compute_into<'a>(
                &mut self,
                input: <$config as IndicatorConfig>::Input<'a>,
                output: <$config as IndicatorConfig>::OutputMut<'a>,
            ) -> Result<OutputRange>
            where
                $config: 'a,
            {
                let input_len = input.real0.len().max(input.real1.len());
                if input_len > self.max_input_len {
                    return Err(TalibError::prepared_capacity_exceeded(
                        self.max_input_len,
                        input_len,
                    ));
                }
                $name(input.real0, input.real1, output)
            }
        }

        #[doc = concat!("Independent Streaming Computation for ", stringify!($name), ".")]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
        pub struct $stream;

        impl crate::traits::sealed::Sealed for $stream {}

        impl StreamingComputation<$config> for $stream {
            type Tick = BinaryTick;
            type TickOutput = Float;

            #[inline(always)]
            fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
                Ok(Some($config::compute_tick(input)?))
            }

            #[inline]
            fn reset(&mut self) {}
        }
    };
}

define_binary_operator!(
    ADD,
    ADDConfig,
    ADDBatchRunner,
    ADDStream,
    |left: Float, right: Float| left + right
);
define_binary_operator!(
    SUB,
    SUBConfig,
    SUBBatchRunner,
    SUBStream,
    |left: Float, right: Float| left - right
);
define_binary_operator!(
    MULT,
    MULTConfig,
    MULTBatchRunner,
    MULTStream,
    |left: Float, right: Float| left * right
);
define_binary_operator!(
    DIV,
    DIVConfig,
    DIVBatchRunner,
    DIVStream,
    |left: Float, right: Float| left / right
);
