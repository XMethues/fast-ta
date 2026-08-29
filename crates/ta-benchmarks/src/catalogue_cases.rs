//! Canonical representative cases and executable Indicator Catalogue measurement coverage.

use fast_ta::inventory::{FunctionGroup, INDICATOR_CATALOGUE};
use std::collections::BTreeSet;

/// The typed Rust/C/Python adapter branch for one representative case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseKind {
    Sma,
    Bbands,
    Rsi,
    Macd,
    Atr,
    Adx,
    HtDcPhase,
    CdlDoji,
    CdlEngulfing,
    Cdl3WhiteSoldiers,
    LinearReg,
    TypPrice,
    Obv,
    Sin,
    Add,
}

/// Metadata shared by every implementation adapter for one measured case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseSpec {
    pub kind: CaseKind,
    pub id: &'static str,
    pub family: &'static str,
    pub definition: &'static str,
    pub parameters: &'static str,
    pub output_kind: &'static str,
    pub output_arity: usize,
}

include!(concat!(env!("OUT_DIR"), "/catalogue_cases.rs"));

/// Measurement coverage for one official Indicator Definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefinitionCoverage {
    pub name: &'static str,
    pub family: &'static str,
    pub measured: bool,
}

/// Executable relationship between implemented and measured Catalogue Coverage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasurementCoverage {
    pub implemented_count: usize,
    pub measured_count: usize,
    pub unmeasured_count: usize,
    pub definitions: Vec<DefinitionCoverage>,
}

impl MeasurementCoverage {
    /// Returns measured coverage as a percentage of implemented definitions.
    pub fn measured_percent(&self) -> f64 {
        if self.implemented_count == 0 {
            0.0
        } else {
            self.measured_count as f64 * 100.0 / self.implemented_count as f64
        }
    }

    /// Returns measured counts in official family order.
    pub fn measured_by_family(&self) -> Vec<(&'static str, usize, usize)> {
        FunctionGroup::ALL
            .iter()
            .map(|group| {
                let family = group.as_str();
                let implemented = self
                    .definitions
                    .iter()
                    .filter(|definition| definition.family == family)
                    .count();
                let measured = self
                    .definitions
                    .iter()
                    .filter(|definition| definition.family == family && definition.measured)
                    .count();
                (family, measured, implemented)
            })
            .collect()
    }
}

/// Validates and returns the canonical measurement coverage model.
pub fn measurement_coverage() -> Result<MeasurementCoverage, String> {
    let mut measured_names = BTreeSet::new();
    for case in MATRIX {
        if !measured_names.insert(case.id) {
            return Err(format!("duplicate representative case {:?}", case.id));
        }
        let definition = INDICATOR_CATALOGUE.definition(case.id).ok_or_else(|| {
            format!(
                "representative case {:?} is not in the Indicator Catalogue",
                case.id
            )
        })?;
        if !definition.is_implemented() {
            return Err(format!(
                "representative case {:?} is not implemented",
                case.id
            ));
        }
        if definition.group.as_str() != case.family {
            return Err(format!(
                "representative case {:?} has family {:?}, expected {:?}",
                case.id,
                case.family,
                definition.group.as_str()
            ));
        }
    }

    let definitions = INDICATOR_CATALOGUE
        .implemented_definitions()
        .map(|definition| DefinitionCoverage {
            name: definition.name,
            family: definition.group.as_str(),
            measured: measured_names.contains(definition.name),
        })
        .collect::<Vec<_>>();
    let measured_count = definitions
        .iter()
        .filter(|definition| definition.measured)
        .count();
    let implemented_count = definitions.len();
    Ok(MeasurementCoverage {
        implemented_count,
        measured_count,
        unmeasured_count: implemented_count - measured_count,
        definitions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representative_cases_are_an_executable_subset_of_the_catalogue() {
        let coverage = measurement_coverage().unwrap();
        assert_eq!(coverage.implemented_count, 161);
        assert_eq!(coverage.measured_count, MATRIX.len());
        assert_eq!(coverage.unmeasured_count, 161 - MATRIX.len());
        assert!(coverage
            .measured_by_family()
            .iter()
            .all(|(_, measured, implemented)| *measured > 0 && measured <= implemented));
    }
}
