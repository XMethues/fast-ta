//! Shared TA-Lib core helpers.
//!
//! This module contains small, allocation-aware primitives shared by indicator
//! implementations: compact output ranges, padded-output conversion, and common
//! validation routines. Batch indicator kernels write compact TA-Lib-style output
//! buffers and return [`OutputRange`]; convenience wrappers use these helpers to
//! create full-length padded vectors.

use crate::{Float, Result, TalibError};

#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};
#[cfg(feature = "std")]
use std::{format, vec::Vec};

/// Location and length of valid compact output values within the original input.
///
/// TA-Lib C reports this as `outBegIdx` and `outNBElement`. Implementations in
/// this crate write valid values compactly starting at output index `0`, then
/// return the original beginning index and valid count in this type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputRange {
    /// Index in the original input where the first output value belongs.
    pub beg_idx: usize,
    /// Number of valid output elements written compactly from output index `0`.
    pub nb_element: usize,
}

impl OutputRange {
    /// Creates a new output range.
    #[inline]
    pub const fn new(beg_idx: usize, nb_element: usize) -> Self {
        Self {
            beg_idx,
            nb_element,
        }
    }

    /// Returns an empty range at the start of the input.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            beg_idx: 0,
            nb_element: 0,
        }
    }

    /// Returns the exclusive end index in the original input.
    #[inline]
    pub const fn end_idx(&self) -> usize {
        self.beg_idx + self.nb_element
    }

    /// Returns true when no valid output elements were produced.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.nb_element == 0
    }
}

/// Padding value used by full-length convenience output vectors.
pub trait PadValue: Copy {
    /// Returns the value used for positions outside [`OutputRange`].
    fn pad_value() -> Self;
}

impl PadValue for Float {
    #[inline]
    fn pad_value() -> Self {
        Float::NAN
    }
}

impl PadValue for i32 {
    #[inline]
    fn pad_value() -> Self {
        0
    }
}

/// Returns the number of compact outputs for an input length and lookback.
#[inline]
pub fn output_count(input_len: usize, lookback: usize) -> usize {
    input_len.saturating_sub(lookback)
}

/// Validates that a period parameter is greater than zero.
#[inline]
pub fn validate_period(name: &str, period: usize) -> Result<()> {
    if period == 0 {
        return Err(TalibError::invalid_period(
            period,
            format!("{} must be greater than zero", name),
        ));
    }
    Ok(())
}

/// Validates a period parameter and returns its standard TA-Lib lookback.
#[inline]
pub fn period_lookback(name: &str, period: usize) -> Result<usize> {
    validate_period(name, period)?;
    Ok(period - 1)
}

/// Validates that an input slice is long enough for a lookback.
#[inline]
pub fn validate_input_len(input_len: usize, lookback: usize) -> Result<usize> {
    let count = output_count(input_len, lookback);
    if count == 0 && input_len > 0 {
        return Err(TalibError::insufficient_data(lookback + 1, input_len));
    }
    Ok(count)
}

/// Validates that a compact output buffer can hold `required` values.
#[inline]
pub fn validate_output_len(name: &str, output_len: usize, required: usize) -> Result<()> {
    if output_len < required {
        return Err(TalibError::invalid_input(format!(
            "{} output buffer too small: need {}, got {}",
            name, required, output_len
        )));
    }
    Ok(())
}

/// Validates that one slice length matches another.
#[inline]
pub fn validate_same_len(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> Result<()> {
    if left_len != right_len {
        return Err(TalibError::invalid_input(format!(
            "{} and {} must have the same length: got {} and {}",
            left_name, right_name, left_len, right_len
        )));
    }
    Ok(())
}

/// Validates that every named slice length matches the first entry.
pub fn validate_all_same_len(lengths: &[(&str, usize)]) -> Result<usize> {
    let Some((first_name, first_len)) = lengths.first().copied() else {
        return Ok(0);
    };

    for &(name, len) in &lengths[1..] {
        validate_same_len(first_name, first_len, name, len)?;
    }

    Ok(first_len)
}

/// Validates that every input value is finite.
pub fn validate_finite_slice(name: &str, values: &[Float]) -> Result<()> {
    for (idx, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(TalibError::invalid_input(format!(
                "{}[{}] must be finite, got {}",
                name, idx, value
            )));
        }
    }
    Ok(())
}

/// Validates that all named input slices contain only finite values.
pub fn validate_finite_slices(slices: &[(&str, &[Float])]) -> Result<()> {
    for &(name, values) in slices {
        validate_finite_slice(name, values)?;
    }
    Ok(())
}

/// Builds a full-length padded vector from compact outputs and their range.
pub fn padded_from_compact<T>(input_len: usize, range: OutputRange, compact: &[T]) -> Vec<T>
where
    T: PadValue,
{
    let mut output = Vec::new();
    output.resize(input_len, T::pad_value());
    copy_compact_to_padded(range, compact, &mut output);
    output
}

/// Copies compact outputs into their full-length padded positions.
pub fn copy_compact_to_padded<T>(range: OutputRange, compact: &[T], padded: &mut [T])
where
    T: PadValue,
{
    let count = range.nb_element.min(compact.len());
    let end = range.beg_idx + count;
    if end <= padded.len() {
        padded[range.beg_idx..end].copy_from_slice(&compact[..count]);
    }
}

/// Allocates a compact output buffer large enough for any first-tranche function.
pub fn compact_buffer<T>(input_len: usize) -> Vec<T>
where
    T: PadValue,
{
    let mut output = Vec::new();
    output.resize(input_len, T::pad_value());
    output
}
