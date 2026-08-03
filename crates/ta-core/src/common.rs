//! Shared TA-Lib core helpers.
//!
//! This module contains small, allocation-aware primitives shared by indicator
//! implementations: compact outputs, output ranges, padded-output conversion,
//! and common validation routines. Batch indicator kernels write compact values
//! and return [`OutputRange`]; convenience wrappers use these helpers to create
//! full-length padded vectors.

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

/// Owned valid values together with their location in the source Observation Series.
///
/// The payload contains exactly [`OutputRange::nb_element`] values per output
/// column. Construction is private to the crate so source bounds and payload
/// lengths cannot be bypassed.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactOutput<O> {
    source_len: usize,
    range: OutputRange,
    values: O,
}

pub(crate) trait CompactPayloadLen {
    fn compact_payload_len(&self) -> Result<usize>;
}

impl<T> CompactPayloadLen for Vec<T> {
    #[inline]
    fn compact_payload_len(&self) -> Result<usize> {
        Ok(self.len())
    }
}

impl<O> CompactOutput<O> {
    pub(crate) fn new(source_len: usize, range: OutputRange, values: O) -> Result<Self>
    where
        O: CompactPayloadLen,
    {
        let end = range
            .beg_idx
            .checked_add(range.nb_element)
            .ok_or_else(|| TalibError::invalid_input("Compact Output range overflow"))?;
        if end > source_len {
            return Err(TalibError::invalid_input(format!(
                "Compact Output range {}..{} exceeds source length {}",
                range.beg_idx, end, source_len
            )));
        }

        let payload_len = values.compact_payload_len()?;
        if payload_len != range.nb_element {
            return Err(TalibError::invalid_input(format!(
                "Compact Output payload length mismatch: range has {}, payload has {}",
                range.nb_element, payload_len
            )));
        }

        Ok(Self {
            source_len,
            range,
            values,
        })
    }

    /// Returns the length of the source Observation Series.
    #[inline]
    pub const fn source_len(&self) -> usize {
        self.source_len
    }

    /// Returns the source Output Range represented by the payload.
    #[inline]
    pub const fn range(&self) -> OutputRange {
        self.range
    }

    /// Borrows the owned compact payload.
    #[inline]
    pub const fn values(&self) -> &O {
        &self.values
    }

    /// Consumes the result and returns its owned compact payload.
    #[inline]
    pub fn into_values(self) -> O {
        self.values
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

#[cold]
#[inline(never)]
fn non_finite_value_error(name: &str, idx: usize, value: Float) -> TalibError {
    TalibError::invalid_input(format!("{name}[{idx}] must be finite, got {value}"))
}

/// Validates one input value without putting error formatting on the hot path.
#[inline(always)]
pub(crate) fn validate_finite_value(name: &str, idx: usize, value: Float) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(non_finite_value_error(name, idx, value))
    }
}

/// Validates that every input value is finite.
pub fn validate_finite_slice(name: &str, values: &[Float]) -> Result<()> {
    for (idx, &value) in values.iter().enumerate() {
        validate_finite_value(name, idx, value)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct NamedColumns {
        left: Vec<Float>,
        right: Vec<Float>,
    }

    impl CompactPayloadLen for NamedColumns {
        fn compact_payload_len(&self) -> Result<usize> {
            if self.left.len() != self.right.len() {
                return Err(TalibError::invalid_input(
                    "Compact Output columns must have equal lengths",
                ));
            }
            Ok(self.left.len())
        }
    }

    #[test]
    fn compact_output_accepts_valid_source_range_and_payload() {
        let output = CompactOutput::new(5, OutputRange::new(2, 3), vec![2_i32, 3, 4]).unwrap();

        assert_eq!(output.source_len(), 5);
        assert_eq!(output.range(), OutputRange::new(2, 3));
        assert_eq!(output.values(), &vec![2, 3, 4]);
    }

    #[test]
    fn compact_output_rejects_range_overflow() {
        let error = CompactOutput::new(usize::MAX, OutputRange::new(usize::MAX, 1), vec![1_i32])
            .unwrap_err();

        assert!(matches!(error, TalibError::InvalidInput { .. }));
    }

    #[test]
    fn compact_output_rejects_out_of_source_range() {
        let error = CompactOutput::new(4, OutputRange::new(2, 3), vec![1_i32, 2, 3]).unwrap_err();

        assert!(matches!(error, TalibError::InvalidInput { .. }));
    }

    #[test]
    fn compact_output_rejects_payload_length_mismatch() {
        let error = CompactOutput::new(5, OutputRange::new(2, 3), vec![1_i32, 2]).unwrap_err();

        assert!(matches!(error, TalibError::InvalidInput { .. }));
    }

    #[test]
    fn compact_output_payload_length_machinery_supports_named_columns() {
        let error = CompactOutput::new(
            3,
            OutputRange::new(1, 2),
            NamedColumns {
                left: vec![1.0 as Float, 2.0],
                right: vec![3.0 as Float],
            },
        )
        .unwrap_err();

        assert!(matches!(error, TalibError::InvalidInput { .. }));
    }
}
