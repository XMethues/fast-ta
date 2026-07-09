//! Arithmetic Math Operators.

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, Indicator, OutputRange, Result, StreamingIndicator,
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
    ($name:ident, $vec_name:ident, $operation:expr) => {
        #[doc = concat!("TA-Lib-style ", stringify!($name), " binary math operator.")]
        #[allow(non_snake_case)]
        pub fn $name(
            real0: &[Float],
            real1: &[Float],
            out_real: &mut [Float],
        ) -> Result<OutputRange> {
            let len = validate_all_same_len(&[("real0", real0.len()), ("real1", real1.len())])?;
            validate_finite_slices(&[("real0", real0), ("real1", real1)])?;
            validate_output_len(stringify!($name), out_real.len(), len)?;
            let operation = $operation;
            for idx in 0..len {
                out_real[idx] = operation(real0[idx], real1[idx]);
            }
            Ok(OutputRange::new(0, len))
        }

        #[doc = concat!("Computes ", stringify!($name), " into a full-length vector.")]
        #[allow(non_snake_case)]
        pub fn $vec_name(real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
            let mut compact = compact_buffer::<Float>(real0.len());
            let range = $name(real0, real1, &mut compact)?;
            Ok(padded_from_compact(
                real0.len(),
                range,
                &compact[..range.nb_element],
            ))
        }

        #[doc = concat!(stringify!($name), " struct surface.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $name {
            _private: (),
        }

        impl $name {
            #[doc = concat!("Creates a ", stringify!($name), " calculator.")]
            pub fn new() -> Result<Self> {
                Ok(Self { _private: () })
            }

            /// Computes compact outputs.
            pub fn compute(
                &self,
                real0: &[Float],
                real1: &[Float],
                out_real: &mut [Float],
            ) -> Result<OutputRange> {
                $name(real0, real1, out_real)
            }

            /// Computes full-length outputs.
            pub fn compute_to_vec(&self, real0: &[Float], real1: &[Float]) -> Result<Vec<Float>> {
                $vec_name(real0, real1)
            }
        }

        impl Indicator for $name {
            type Input<'a> = BinaryInput<'a>;
            type OutputMut<'a> = &'a mut [Float];
            type OutputOwned = Vec<Float>;

            fn lookback(&self) -> usize {
                0
            }

            fn compute<'a>(
                &self,
                input: Self::Input<'a>,
                output: Self::OutputMut<'a>,
            ) -> Result<OutputRange> {
                $name(input.real0, input.real1, output)
            }

            fn compute_to_vec<'a>(&self, input: Self::Input<'a>) -> Result<Self::OutputOwned> {
                $vec_name(input.real0, input.real1)
            }
        }

        impl StreamingIndicator for $name {
            type Tick = BinaryTick;
            type TickOutput = Float;

            fn next(&mut self, input: BinaryTick) -> Result<Option<Float>> {
                validate_finite_slices(&[("real0", &[input.real0]), ("real1", &[input.real1])])?;
                let operation = $operation;
                Ok(Some(operation(input.real0, input.real1)))
            }
        }
    };
}

define_binary_operator!(ADD, ADD_vec, |left: Float, right: Float| left + right);
define_binary_operator!(SUB, SUB_vec, |left: Float, right: Float| left - right);
define_binary_operator!(MULT, MULT_vec, |left: Float, right: Float| left * right);
define_binary_operator!(DIV, DIV_vec, |left: Float, right: Float| left / right);
