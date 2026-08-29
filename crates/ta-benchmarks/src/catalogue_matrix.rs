//! Deterministic fixtures and semantic gates for Catalogue measurements.

pub const ABS_TOLERANCE: f64 = 1.0e-9;
pub const REL_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct Fixture {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub auxiliary: Vec<f64>,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    pub fn validate(&self) -> Result<(), String> {
        let length = self.close.len();
        for (name, values) in [
            ("open", &self.open),
            ("high", &self.high),
            ("low", &self.low),
            ("volume", &self.volume),
            ("auxiliary", &self.auxiliary),
        ] {
            if values.len() != length {
                return Err(format!(
                    "fixture {name} length {} differs from close length {length}",
                    values.len()
                ));
            }
        }
        for index in 0..length {
            let values = [
                self.open[index],
                self.high[index],
                self.low[index],
                self.close[index],
                self.volume[index],
                self.auxiliary[index],
            ];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "fixture contains a non-finite value at index {index}"
                ));
            }
            if self.high[index] < self.open[index].max(self.close[index]) {
                return Err(format!(
                    "fixture high violates OHLC invariants at index {index}"
                ));
            }
            if self.low[index] > self.open[index].min(self.close[index]) {
                return Err(format!(
                    "fixture low violates OHLC invariants at index {index}"
                ));
            }
            if self.volume[index] < 0.0 {
                return Err(format!("fixture volume is negative at index {index}"));
            }
        }
        Ok(())
    }
}

pub use crate::fixture::series_fixture;

pub fn catalogue_fixture(size: usize) -> Fixture {
    let close = series_fixture(size, 0);
    let auxiliary = series_fixture(size, 2)
        .into_iter()
        .enumerate()
        .map(|(index, value)| value * 0.75 + (index % 13) as f64 * 0.02)
        .collect::<Vec<_>>();
    let open = close
        .iter()
        .enumerate()
        .map(|(index, value)| value + ((index % 9) as f64 - 4.0) * 0.035)
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| open.max(*close) + 0.5 + (index % 11) as f64 * 0.03)
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(&close)
        .enumerate()
        .map(|(index, (open, close))| open.min(*close) - 0.5 - (index % 7) as f64 * 0.025)
        .collect::<Vec<_>>();
    let volume = series_fixture(size, 1)
        .into_iter()
        .map(|value| 10_000.0 + value * 100.0)
        .collect::<Vec<_>>();
    Fixture {
        open,
        high,
        low,
        close,
        volume,
        auxiliary,
    }
}

pub fn input_checksum(values: &[f64]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    hash_f64s(&mut hash, values);
    format!("fnv1a64:{hash:016x}")
}

pub fn fixture_checksum(fixture: &Fixture) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for (name, values) in [
        ("open", fixture.open.as_slice()),
        ("high", fixture.high.as_slice()),
        ("low", fixture.low.as_slice()),
        ("close", fixture.close.as_slice()),
        ("volume", fixture.volume.as_slice()),
        ("auxiliary", fixture.auxiliary.as_slice()),
    ] {
        hash_bytes(&mut hash, name.as_bytes());
        hash_bytes(&mut hash, &(values.len() as u64).to_le_bytes());
        hash_f64s(&mut hash, values);
    }
    format!("fnv1a64:{hash:016x}")
}

fn hash_f64s(hash: &mut u64, values: &[f64]) {
    for value in values {
        hash_bytes(hash, &value.to_le_bytes());
    }
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum OutputValues {
    Float(Vec<Vec<f64>>),
    Integer(Vec<Vec<i32>>),
}

impl OutputValues {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Float(_) => "float",
            Self::Integer(_) => "integer",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            Self::Float(columns) => columns.len(),
            Self::Integer(columns) => columns.len(),
        }
    }

    pub fn column_len(&self) -> Result<usize, String> {
        let lengths = match self {
            Self::Float(columns) => columns.iter().map(Vec::len).collect::<Vec<_>>(),
            Self::Integer(columns) => columns.iter().map(Vec::len).collect::<Vec<_>>(),
        };
        let Some(&first) = lengths.first() else {
            return Err("output has no columns".to_owned());
        };
        if lengths.iter().any(|length| *length != first) {
            return Err(format!("output columns have unequal lengths: {lengths:?}"));
        }
        Ok(first)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedOutput {
    pub begin: usize,
    pub count: usize,
    pub values: OutputValues,
}

impl VerifiedOutput {
    pub fn validate_shape(&self) -> Result<(), String> {
        let actual = self.values.column_len()?;
        if actual != self.count {
            return Err(format!(
                "declared output count {} differs from column length {actual}",
                self.count
            ));
        }
        Ok(())
    }

    pub fn checksum(&self) -> String {
        let mut hash = 0xcbf29ce484222325_u64;
        hash_bytes(&mut hash, self.values.kind().as_bytes());
        hash_bytes(&mut hash, &(self.begin as u64).to_le_bytes());
        hash_bytes(&mut hash, &(self.count as u64).to_le_bytes());
        match &self.values {
            OutputValues::Float(columns) => {
                for column in columns {
                    hash_f64s(&mut hash, column);
                }
            }
            OutputValues::Integer(columns) => {
                for column in columns {
                    for value in column {
                        hash_bytes(&mut hash, &value.to_le_bytes());
                    }
                }
            }
        }
        format!("fnv1a64:{hash:016x}")
    }
}

pub fn validate_outputs(
    expected: &VerifiedOutput,
    implementation: &str,
    actual: &VerifiedOutput,
) -> Result<(), String> {
    expected
        .validate_shape()
        .map_err(|error| format!("reference output shape: {error}"))?;
    actual
        .validate_shape()
        .map_err(|error| format!("{implementation} output shape: {error}"))?;
    if (actual.begin, actual.count) != (expected.begin, expected.count) {
        return Err(format!(
            "{implementation} OutputRange mismatch: expected begin {} count {}, got begin {} count {}",
            expected.begin, expected.count, actual.begin, actual.count
        ));
    }
    if actual.values.kind() != expected.values.kind() {
        return Err(format!(
            "{implementation} output kind mismatch: expected {}, got {}",
            expected.values.kind(),
            actual.values.kind()
        ));
    }
    if actual.values.arity() != expected.values.arity() {
        return Err(format!(
            "{implementation} output arity mismatch: expected {}, got {}",
            expected.values.arity(),
            actual.values.arity()
        ));
    }
    match (&expected.values, &actual.values) {
        (OutputValues::Float(expected_columns), OutputValues::Float(actual_columns)) => {
            for (column, (expected_values, actual_values)) in
                expected_columns.iter().zip(actual_columns).enumerate()
            {
                for (index, (&expected_value, &actual_value)) in
                    expected_values.iter().zip(actual_values).enumerate()
                {
                    if !expected_value.is_finite() || !actual_value.is_finite() {
                        if expected_value.to_bits() != actual_value.to_bits() {
                            return Err(format!("{implementation} non-finite placement mismatch at column {column} compact index {index}: expected {expected_value:?}, got {actual_value:?}"));
                        }
                        continue;
                    }
                    let difference = (actual_value - expected_value).abs();
                    let tolerance = ABS_TOLERANCE.max(REL_TOLERANCE * expected_value.abs());
                    if difference > tolerance {
                        return Err(format!("{implementation} value mismatch at column {column} compact index {index}: expected {expected_value:.17e}, got {actual_value:.17e}, difference {difference:.3e}, tolerance {tolerance:.3e}"));
                    }
                }
            }
        }
        (OutputValues::Integer(expected_columns), OutputValues::Integer(actual_columns)) => {
            for (column, (expected_values, actual_values)) in
                expected_columns.iter().zip(actual_columns).enumerate()
            {
                for (index, (&expected_value, &actual_value)) in
                    expected_values.iter().zip(actual_values).enumerate()
                {
                    if actual_value != expected_value {
                        return Err(format!("{implementation} exact integer mismatch at column {column} compact index {index}: expected {expected_value}, got {actual_value}"));
                    }
                }
            }
        }
        _ => unreachable!("output kinds were checked above"),
    }
    Ok(())
}
