//! Arithmetic Math Operators.

use crate::{
    compact_buffer, padded_from_compact, validate_all_same_len, validate_finite_slices,
    validate_output_len, Float, OutputRange, Result,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

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
    };
}

define_binary_operator!(ADD, ADD_vec, |left: Float, right: Float| left + right);
define_binary_operator!(SUB, SUB_vec, |left: Float, right: Float| left - right);
define_binary_operator!(MULT, MULT_vec, |left: Float, right: Float| left * right);
define_binary_operator!(DIV, DIV_vec, |left: Float, right: Float| left / right);
