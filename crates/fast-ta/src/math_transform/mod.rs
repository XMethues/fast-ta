//! Math Transform functions.
//!
//! These are unary element-wise transforms over one real input series. They use
//! strict finite-input validation, compact zero-copy output buffers, immutable
//! configurations, reusable batch runners, and independent per-tick streams.
//! Finite values outside an operation's mathematical domain remain valid
//! inputs: the underlying IEEE-754 operation returns its defined `NaN` or
//! infinity result without changing the output range.

mod functions {
    use crate::common::validate_finite_value;
    use crate::{
        validate_finite_slice, validate_output_len, CompactOutput, Float, IndicatorConfig,
        OutputRange, PreparedBatchRunner, Result, StreamingComputation, TalibError,
    };

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    macro_rules! define_transform {
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
        };
    }

    define_transform!(
        ACOS,
        ACOSConfig,
        ACOSBatchRunner,
        ACOSStream,
        |value: Float| value.acos()
    );
    define_transform!(
        ASIN,
        ASINConfig,
        ASINBatchRunner,
        ASINStream,
        |value: Float| value.asin()
    );
    define_transform!(
        ATAN,
        ATANConfig,
        ATANBatchRunner,
        ATANStream,
        |value: Float| value.atan()
    );
    define_transform!(
        CEIL,
        CEILConfig,
        CEILBatchRunner,
        CEILStream,
        |value: Float| value.ceil()
    );
    define_transform!(COS, COSConfig, COSBatchRunner, COSStream, |value: Float| {
        value.cos()
    });
    define_transform!(
        COSH,
        COSHConfig,
        COSHBatchRunner,
        COSHStream,
        |value: Float| value.cosh()
    );
    define_transform!(EXP, EXPConfig, EXPBatchRunner, EXPStream, |value: Float| {
        value.exp()
    });
    define_transform!(
        FLOOR,
        FLOORConfig,
        FLOORBatchRunner,
        FLOORStream,
        |value: Float| value.floor()
    );
    define_transform!(LN, LNConfig, LNBatchRunner, LNStream, |value: Float| value
        .ln());
    define_transform!(
        LOG10,
        LOG10Config,
        LOG10BatchRunner,
        LOG10Stream,
        |value: Float| value.log10()
    );
    define_transform!(SIN, SINConfig, SINBatchRunner, SINStream, |value: Float| {
        value.sin()
    });
    define_transform!(
        SINH,
        SINHConfig,
        SINHBatchRunner,
        SINHStream,
        |value: Float| value.sinh()
    );
    define_transform!(
        SQRT,
        SQRTConfig,
        SQRTBatchRunner,
        SQRTStream,
        |value: Float| value.sqrt()
    );
    define_transform!(TAN, TANConfig, TANBatchRunner, TANStream, |value: Float| {
        value.tan()
    });
    define_transform!(
        TANH,
        TANHConfig,
        TANHBatchRunner,
        TANHStream,
        |value: Float| value.tanh()
    );
}

pub use functions::{
    ACOSBatchRunner, ACOSConfig, ACOSStream, ASINBatchRunner, ASINConfig, ASINStream,
    ATANBatchRunner, ATANConfig, ATANStream, CEILBatchRunner, CEILConfig, CEILStream,
    COSBatchRunner, COSConfig, COSHBatchRunner, COSHConfig, COSHStream, COSStream, EXPBatchRunner,
    EXPConfig, EXPStream, FLOORBatchRunner, FLOORConfig, FLOORStream, LNBatchRunner, LNConfig,
    LNStream, LOG10BatchRunner, LOG10Config, LOG10Stream, SINBatchRunner, SINConfig,
    SINHBatchRunner, SINHConfig, SINHStream, SINStream, SQRTBatchRunner, SQRTConfig, SQRTStream,
    TANBatchRunner, TANConfig, TANHBatchRunner, TANHConfig, TANHStream, TANStream, ACOS, ASIN,
    ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH,
};
