//! Error types and handling for TA library
//!
//! This module defines the `TalibError` enum which represents all possible errors
//! that can occur in the TA library operations. The error type implements the
//! standard `Error` trait for proper error handling and propagation.

/// Error type for TA library operations
///
/// All operations in the TA library that can fail will return a `Result<T, TalibError>`.
/// This enum covers all possible error scenarios that might occur during indicator
/// computation, input validation, and data processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TalibError {
    /// Invalid input data (e.g., NaN values, infinite values)
    InvalidInput {
        /// Description of the invalid input
        message: String,
    },

    /// Invalid period parameter (e.g., zero, negative, too large)
    InvalidPeriod {
        /// The invalid period value
        period: usize,
        /// Description of why it's invalid
        reason: String,
    },

    /// Insufficient data for computation (not enough data points)
    InsufficientData {
        /// Required minimum data length
        required: usize,
        /// Actual data length provided
        actual: usize,
    },

    /// Invalid parameter value
    InvalidParameter {
        /// Parameter name
        name: String,
        /// Invalid value (as string for flexible representation)
        value: String,
        /// Expected value description
        expected: String,
    },

    /// Computation error (e.g., numerical issues, overflow)
    ComputationError {
        /// Description of the computation failure
        message: String,
    },

    /// Feature not implemented
    NotImplemented {
        /// Feature name or description
        feature: String,
    },
}

impl TalibError {
    /// Creates an InvalidInput error
    ///
    /// # Arguments
    ///
    /// * `message` - Description of the invalid input
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::invalid_input("Input contains NaN values");
    /// ```
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        TalibError::InvalidInput {
            message: message.into(),
        }
    }

    /// Creates an InvalidPeriod error
    ///
    /// # Arguments
    ///
    /// * `period` - The invalid period value
    /// * `reason` - Description of why it's invalid
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::invalid_period(0, "period must be greater than zero");
    /// ```
    pub fn invalid_period<S: Into<String>>(period: usize, reason: S) -> Self {
        TalibError::InvalidPeriod {
            period,
            reason: reason.into(),
        }
    }

    /// Creates an InsufficientData error
    ///
    /// # Arguments
    ///
    /// * `required` - Minimum data length required
    /// * `actual` - Actual data length provided
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::insufficient_data(20, 10);
    /// ```
    pub fn insufficient_data(required: usize, actual: usize) -> Self {
        TalibError::InsufficientData { required, actual }
    }

    /// Creates an InvalidParameter error
    ///
    /// # Arguments
    ///
    /// * `name` - Parameter name
    /// * `value` - Invalid value (as string)
    /// * `expected` - Expected value description
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::invalid_parameter("alpha", "1.5", "value in [0.0, 1.0]");
    /// ```
    pub fn invalid_parameter<S: Into<String>>(name: S, value: S, expected: S) -> Self {
        TalibError::InvalidParameter {
            name: name.into(),
            value: value.into(),
            expected: expected.into(),
        }
    }

    /// Creates a ComputationError
    ///
    /// # Arguments
    ///
    /// * `message` - Description of the computation failure
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::computation_error("Numerical overflow in calculation");
    /// ```
    pub fn computation_error<S: Into<String>>(message: S) -> Self {
        TalibError::ComputationError {
            message: message.into(),
        }
    }

    /// Creates a NotImplemented error
    ///
    /// # Arguments
    ///
    /// * `feature` - Feature name or description
    ///
    /// # Example
    ///
    /// ```rust
    /// use ta_core::error::TalibError;
    ///
    /// let err = TalibError::not_implemented("Hull Moving Average with period > 100");
    /// ```
    pub fn not_implemented<S: Into<String>>(feature: S) -> Self {
        TalibError::NotImplemented {
            feature: feature.into(),
        }
    }
}

impl core::fmt::Display for TalibError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TalibError::InvalidInput { message } => {
                write!(f, "Invalid input: {}", message)
            }
            TalibError::InvalidPeriod { period, reason } => {
                write!(f, "Invalid period {}: {}", period, reason)
            }
            TalibError::InsufficientData { required, actual } => {
                write!(
                    f,
                    "Insufficient data: required {} data points, got {}",
                    required, actual
                )
            }
            TalibError::InvalidParameter {
                name,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid parameter '{}': got '{}', expected {}",
                    name, value, expected
                )
            }
            TalibError::ComputationError { message } => {
                write!(f, "Computation error: {}", message)
            }
            TalibError::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {}", feature)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TalibError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

// When std is not available, we need to provide a no_std compatible Error implementation
// In Rust 1.81+, core::error::Error is available in core
#[cfg(all(not(feature = "std"), feature = "core_error"))]
impl core::error::Error for TalibError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }
}

// Implementations for std error types (only available when std is enabled)
#[cfg(feature = "std")]
impl From<std::io::Error> for TalibError {
    fn from(err: std::io::Error) -> Self {
        TalibError::ComputationError {
            message: format!("I/O error: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::num::ParseFloatError> for TalibError {
    fn from(err: std::num::ParseFloatError) -> Self {
        TalibError::InvalidInput {
            message: format!("Failed to parse float: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::num::ParseIntError> for TalibError {
    fn from(err: std::num::ParseIntError) -> Self {
        TalibError::InvalidInput {
            message: format!("Failed to parse integer: {}", err),
        }
    }
}

/// Result type alias for TA library operations
///
/// This is a convenience alias that uses `TalibError` as the error type.
///
/// # Example
///
/// ```rust
/// use ta_core::error::{Result, TalibError};
///
/// fn calculate_something() -> Result<f64> {
///     // Returns Result<f64, TalibError>
///     Ok(42.0)
/// }
/// ```
pub type Result<T> = core::result::Result<T, TalibError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_input_creation() {
        let err = TalibError::invalid_input("Test message");
        assert_eq!(err.to_string(), "Invalid input: Test message");
    }

    #[test]
    fn test_invalid_period_creation() {
        let err = TalibError::invalid_period(0, "must be positive");
        assert_eq!(err.to_string(), "Invalid period 0: must be positive");
    }

    #[test]
    fn test_insufficient_data_creation() {
        let err = TalibError::insufficient_data(20, 10);
        assert_eq!(
            err.to_string(),
            "Insufficient data: required 20 data points, got 10"
        );
    }

    #[test]
    fn test_invalid_parameter_creation() {
        let err = TalibError::invalid_parameter("alpha", "1.5", "value in [0.0, 1.0]");
        assert_eq!(
            err.to_string(),
            "Invalid parameter 'alpha': got '1.5', expected value in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_computation_error_creation() {
        let err = TalibError::computation_error("Numerical overflow");
        assert_eq!(err.to_string(), "Computation error: Numerical overflow");
    }

    #[test]
    fn test_not_implemented_creation() {
        let err = TalibError::not_implemented("Feature X");
        assert_eq!(err.to_string(), "Feature not implemented: Feature X");
    }

    #[test]
    fn test_error_variants_are_equality_comparable() {
        let err1 = TalibError::invalid_input("Test");
        let err2 = TalibError::invalid_input("Test");
        let err3 = TalibError::invalid_input("Different");

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_is_debug() {
        let err = TalibError::insufficient_data(100, 50);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InsufficientData"));
    }

    #[test]
    fn test_error_is_clone() {
        let err1 = TalibError::invalid_input("Test");
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_insufficient_data_message_format() {
        let err = TalibError::InsufficientData {
            required: 100,
            actual: 42,
        };
        assert_eq!(
            err.to_string(),
            "Insufficient data: required 100 data points, got 42"
        );
    }

    #[test]
    fn test_invalid_period_message_format() {
        let err = TalibError::InvalidPeriod {
            period: 0,
            reason: "period cannot be zero".to_string(),
        };
        assert!(err.to_string().contains("Invalid period"));
        assert!(err.to_string().contains("cannot be zero"));
    }

    #[test]
    fn test_result_type_alias() {
        let ok_result: Result<f64> = Ok(42.0);
        let err_result: Result<f64> = Err(TalibError::invalid_input("Test"));

        assert!(ok_result.is_ok());
        assert!(err_result.is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let ta_err = TalibError::from(io_err);

        match ta_err {
            TalibError::ComputationError { message } => {
                assert!(message.contains("I/O error"));
                assert!(message.contains("file not found"));
            }
            _ => panic!("Expected ComputationError variant"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_parse_float_error() {
        let parse_err = "not_a_float".parse::<f64>().unwrap_err();
        let ta_err = TalibError::from(parse_err);

        match ta_err {
            TalibError::InvalidInput { message } => {
                assert!(message.contains("Failed to parse float"));
            }
            _ => panic!("Expected InvalidInput variant"),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_from_parse_int_error() {
        let parse_err = "not_an_int".parse::<i32>().unwrap_err();
        let ta_err = TalibError::from(parse_err);

        match ta_err {
            TalibError::InvalidInput { message } => {
                assert!(message.contains("Failed to parse integer"));
            }
            _ => panic!("Expected InvalidInput variant"),
        }
    }

    #[test]
    fn test_convenience_methods_create_correct_variants() {
        let err1 = TalibError::invalid_input("NaN value");
        assert!(matches!(err1, TalibError::InvalidInput { .. }));

        let err2 = TalibError::invalid_period(5, "must be even");
        assert!(matches!(err2, TalibError::InvalidPeriod { .. }));

        let err3 = TalibError::insufficient_data(10, 5);
        assert!(matches!(err3, TalibError::InsufficientData { .. }));

        let err4 = TalibError::invalid_parameter("x", "-1", "positive number");
        assert!(matches!(err4, TalibError::InvalidParameter { .. }));

        let err5 = TalibError::computation_error("division by zero");
        assert!(matches!(err5, TalibError::ComputationError { .. }));

        let err6 = TalibError::not_implemented("advanced feature");
        assert!(matches!(err6, TalibError::NotImplemented { .. }));
    }

    #[test]
    fn test_error_messages_are_helpful() {
        // Ensure error messages provide useful information
        let errors = vec![
            TalibError::InvalidInput {
                message: "Input contains NaN".to_string(),
            },
            TalibError::InvalidPeriod {
                period: 0,
                reason: "Period must be > 0".to_string(),
            },
            TalibError::InsufficientData {
                required: 50,
                actual: 10,
            },
            TalibError::InvalidParameter {
                name: "alpha".to_string(),
                value: "1.5".to_string(),
                expected: "0.0 to 1.0".to_string(),
            },
            TalibError::ComputationError {
                message: "Numerical overflow".to_string(),
            },
            TalibError::NotImplemented {
                feature: "Advanced indicator".to_string(),
            },
        ];

        for error in errors {
            let msg: String = error.to_string();
            // All error messages should be non-empty
            assert!(!msg.is_empty());
            // All error messages should be reasonably descriptive
            assert!(msg.len() > 10);
        }
    }

    #[test]
    fn test_error_chain_compatibility() {
        // Test that errors can be used in typical error propagation patterns
        fn inner_function() -> Result<()> {
            Err(TalibError::invalid_input("test"))
        }

        fn outer_function() -> Result<()> {
            inner_function()?;
            Ok(())
        }

        assert!(outer_function().is_err());
    }

    #[test]
    fn test_question_mark_operator() {
        fn validate(value: f64) -> Result<f64> {
            if value.is_nan() {
                return Err(TalibError::invalid_input("Value is NaN"));
            }
            Ok(value)
        }

        fn process(values: &[f64]) -> Result<Vec<f64>> {
            values.iter().map(|&v| validate(v)).collect()
        }

        // Valid input
        let result = process(&[1.0, 2.0, 3.0]);
        assert!(result.is_ok());

        // Invalid input with NaN
        let result = process(&[1.0, f64::NAN, 3.0]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Invalid input: Value is NaN"
        );
    }
}
