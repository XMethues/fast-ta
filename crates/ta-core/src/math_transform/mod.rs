//! Math Transform functions.
//!
//! These are unary element-wise transforms over one real input series. They use
//! strict finite-input validation, compact zero-copy output buffers, padded
//! convenience wrappers, immutable configurations, reusable batch runners,
//! and independent per-tick streams. Finite values outside an operation's
//! mathematical domain remain valid inputs: the underlying IEEE-754 operation
//! returns its defined `NaN` or infinity result without changing the output
//! range.

mod functions {
    use crate::common::validate_finite_value;
    use crate::{
        compact_buffer, padded_from_compact, validate_finite_slice, validate_output_len,
        CompactOutput, Float, Indicator, IndicatorConfig, OutputRange, PreparedBatchRunner, Result,
        StreamingComputation, StreamingIndicator, TalibError,
    };

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    macro_rules! define_transform {
        (
            $name:ident,
            $vec_name:ident,
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
                fn validate_input(real: &[Float]) -> Result<usize> {
                    validate_finite_slice("real", real)?;
                    Ok(real.len())
                }

                #[inline(always)]
                fn compute_validated(real: &[Float], out_real: &mut [Float]) -> OutputRange {
                    let operation = $operation;
                    for (idx, value) in real.iter().copied().enumerate() {
                        out_real[idx] = operation(value);
                    }
                    OutputRange::new(0, real.len())
                }
            }

            #[doc = concat!("TA-Lib-style ", stringify!($name), " unary transform.")]
            #[allow(non_snake_case)]
            pub fn $name(real: &[Float], out_real: &mut [Float]) -> Result<OutputRange> {
                let len = $config::validate_input(real)?;
                validate_output_len(stringify!($name), out_real.len(), len)?;
                Ok($config::compute_validated(real, out_real))
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

            impl crate::traits::sealed::Sealed for $config {}

            impl IndicatorConfig for $config {
                type Input<'a> = &'a [Float];
                type Output = Vec<Float>;
                type OutputMut<'a> = &'a mut [Float];
                type BatchRunner = $runner;
                type Stream = $stream;

                #[inline]
                fn lookback(&self) -> usize {
                    0
                }

                fn compute<'a>(
                    &self,
                    input: Self::Input<'a>,
                ) -> Result<CompactOutput<Self::Output>> {
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
                    if input.len() > self.max_input_len {
                        return Err(TalibError::prepared_capacity_exceeded(
                            self.max_input_len,
                            input.len(),
                        ));
                    }
                    $name(input, output)
                }
            }

            #[doc = concat!("Independent Streaming Computation for ", stringify!($name), ".")]
            #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
            pub struct $stream;

            impl crate::traits::sealed::Sealed for $stream {}

            impl StreamingComputation<$config> for $stream {
                type Tick = Float;
                type TickOutput = Float;

                #[inline]
                fn next(&mut self, input: Self::Tick) -> Result<Option<Self::TickOutput>> {
                    validate_finite_value("input", 0, input)?;
                    let operation = $operation;
                    Ok(Some(operation(input)))
                }

                #[inline]
                fn reset(&mut self) {}
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
                type Input<'a> = &'a [Float];
                type OutputMut<'a> = &'a mut [Float];
                type OutputOwned = Vec<Float>;

                fn lookback(&self) -> usize {
                    0
                }

                fn compute<'a>(
                    &self,
                    inputs: Self::Input<'a>,
                    outputs: Self::OutputMut<'a>,
                ) -> Result<OutputRange> {
                    $name(inputs, outputs)
                }

                fn compute_to_vec<'a>(
                    &self,
                    inputs: Self::Input<'a>,
                ) -> Result<Self::OutputOwned> {
                    $vec_name(inputs)
                }
            }

            impl StreamingIndicator for $name {
                type Tick = Float;
                type TickOutput = Float;

                #[inline]
                fn next(&mut self, input: Float) -> Result<Option<Float>> {
                    validate_finite_value("input", 0, input)?;
                    let operation = $operation;
                    Ok(Some(operation(input)))
                }
            }
        };
    }

    define_transform!(
        ACOS,
        ACOS_vec,
        ACOSConfig,
        ACOSBatchRunner,
        ACOSStream,
        |value: Float| value.acos()
    );
    define_transform!(
        ASIN,
        ASIN_vec,
        ASINConfig,
        ASINBatchRunner,
        ASINStream,
        |value: Float| value.asin()
    );
    define_transform!(
        ATAN,
        ATAN_vec,
        ATANConfig,
        ATANBatchRunner,
        ATANStream,
        |value: Float| value.atan()
    );
    define_transform!(
        CEIL,
        CEIL_vec,
        CEILConfig,
        CEILBatchRunner,
        CEILStream,
        |value: Float| value.ceil()
    );
    define_transform!(
        COS,
        COS_vec,
        COSConfig,
        COSBatchRunner,
        COSStream,
        |value: Float| value.cos()
    );
    define_transform!(
        COSH,
        COSH_vec,
        COSHConfig,
        COSHBatchRunner,
        COSHStream,
        |value: Float| value.cosh()
    );
    define_transform!(
        EXP,
        EXP_vec,
        EXPConfig,
        EXPBatchRunner,
        EXPStream,
        |value: Float| value.exp()
    );
    define_transform!(
        FLOOR,
        FLOOR_vec,
        FLOORConfig,
        FLOORBatchRunner,
        FLOORStream,
        |value: Float| value.floor()
    );
    define_transform!(
        LN,
        LN_vec,
        LNConfig,
        LNBatchRunner,
        LNStream,
        |value: Float| value.ln()
    );
    define_transform!(
        LOG10,
        LOG10_vec,
        LOG10Config,
        LOG10BatchRunner,
        LOG10Stream,
        |value: Float| value.log10()
    );
    define_transform!(
        SIN,
        SIN_vec,
        SINConfig,
        SINBatchRunner,
        SINStream,
        |value: Float| value.sin()
    );
    define_transform!(
        SINH,
        SINH_vec,
        SINHConfig,
        SINHBatchRunner,
        SINHStream,
        |value: Float| value.sinh()
    );
    define_transform!(
        SQRT,
        SQRT_vec,
        SQRTConfig,
        SQRTBatchRunner,
        SQRTStream,
        |value: Float| value.sqrt()
    );
    define_transform!(
        TAN,
        TAN_vec,
        TANConfig,
        TANBatchRunner,
        TANStream,
        |value: Float| value.tan()
    );
    define_transform!(
        TANH,
        TANH_vec,
        TANHConfig,
        TANHBatchRunner,
        TANHStream,
        |value: Float| value.tanh()
    );
}

pub use functions::{
    ACOSBatchRunner, ACOSConfig, ACOSStream, ACOS_vec, ASINBatchRunner, ASINConfig, ASINStream,
    ASIN_vec, ATANBatchRunner, ATANConfig, ATANStream, ATAN_vec, CEILBatchRunner, CEILConfig,
    CEILStream, CEIL_vec, COSBatchRunner, COSConfig, COSHBatchRunner, COSHConfig, COSHStream,
    COSH_vec, COSStream, COS_vec, EXPBatchRunner, EXPConfig, EXPStream, EXP_vec, FLOORBatchRunner,
    FLOORConfig, FLOORStream, FLOOR_vec, LNBatchRunner, LNConfig, LNStream, LN_vec,
    LOG10BatchRunner, LOG10Config, LOG10Stream, LOG10_vec, SINBatchRunner, SINConfig,
    SINHBatchRunner, SINHConfig, SINHStream, SINH_vec, SINStream, SIN_vec, SQRTBatchRunner,
    SQRTConfig, SQRTStream, SQRT_vec, TANBatchRunner, TANConfig, TANHBatchRunner, TANHConfig,
    TANHStream, TANH_vec, TANStream, TAN_vec, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN,
    LOG10, SIN, SINH, SQRT, TAN, TANH,
};
