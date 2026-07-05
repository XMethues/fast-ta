//! Math Transform functions.
//!
//! These are unary element-wise transforms over one real input series. They use
//! strict finite-input validation, compact zero-copy output buffers, and padded
//! convenience wrappers matching the first-tranche core contract.

mod functions {
    use crate::{
        compact_buffer, padded_from_compact, validate_finite_slice, validate_output_len, Float,
        Indicator, OutputRange, Result,
    };

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    macro_rules! define_transform {
        ($name:ident, $vec_name:ident, $operation:expr) => {
            #[doc = concat!("TA-Lib-style ", stringify!($name), " unary transform.")]
            #[allow(non_snake_case)]
            pub fn $name(real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                validate_finite_slice("real", real)?;
                validate_output_len(stringify!($name), out_real.len(), real.len())?;
                let operation = $operation;
                for (idx, value) in real.iter().copied().enumerate() {
                    out_real[idx] = operation(value);
                }
                Ok(OutputRange::new(0, real.len()))
            }

            #[doc = concat!("Computes ", stringify!($name), " into a full-length vector.")]
            #[allow(non_snake_case)]
            pub fn $vec_name(real: &[Float]) -> Result<Vec<Float>> {
                let mut compact = compact_buffer::<Float>(real.len());
                let range = $name(real, &mut compact)?;
                Ok(padded_from_compact(
                    real.len(),
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
                    real: &[Float],
                    out_real: &mut [Float],
                ) -> Result<OutputRange> {
                    $name(real, out_real)
                }

                /// Computes full-length outputs.
                pub fn compute_to_vec(&self, real: &[Float]) -> Result<Vec<Float>> {
                    $vec_name(real)
                }
            }

            impl Indicator for $name {
                type Input = Float;
                type Output = Float;

                fn lookback(&self) -> usize {
                    0
                }

                fn compute(
                    &self,
                    inputs: &[Self::Input],
                    outputs: &mut [Self::Output],
                ) -> Result<OutputRange> {
                    $name(inputs, outputs)
                }

                fn compute_to_vec(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
                    $vec_name(inputs)
                }

                fn next(&mut self, input: Float) -> Float {
                    let operation = $operation;
                    operation(input)
                }
            }
        };
    }

    define_transform!(ACOS, ACOS_vec, |value: Float| value.acos());
    define_transform!(ASIN, ASIN_vec, |value: Float| value.asin());
    define_transform!(ATAN, ATAN_vec, |value: Float| value.atan());
    define_transform!(CEIL, CEIL_vec, |value: Float| value.ceil());
    define_transform!(COS, COS_vec, |value: Float| value.cos());
    define_transform!(COSH, COSH_vec, |value: Float| value.cosh());
    define_transform!(EXP, EXP_vec, |value: Float| value.exp());
    define_transform!(FLOOR, FLOOR_vec, |value: Float| value.floor());
    define_transform!(LN, LN_vec, |value: Float| value.ln());
    define_transform!(LOG10, LOG10_vec, |value: Float| value.log10());
    define_transform!(SIN, SIN_vec, |value: Float| value.sin());
    define_transform!(SINH, SINH_vec, |value: Float| value.sinh());
    define_transform!(SQRT, SQRT_vec, |value: Float| value.sqrt());
    define_transform!(TAN, TAN_vec, |value: Float| value.tan());
    define_transform!(TANH, TANH_vec, |value: Float| value.tanh());
}

pub use functions::{
    ACOS_vec, ASIN_vec, ATAN_vec, CEIL_vec, COSH_vec, COS_vec, EXP_vec, FLOOR_vec, LN_vec,
    LOG10_vec, SINH_vec, SIN_vec, SQRT_vec, TANH_vec, TAN_vec, ACOS, ASIN, ATAN, CEIL, COS, COSH,
    EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
};
