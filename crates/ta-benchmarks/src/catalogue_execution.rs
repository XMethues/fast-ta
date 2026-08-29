//! Static dispatch for Catalogue Matrix case execution.
//!
//! The benchmark selects a case once, before constructing a measured operation. Implementors
//! remain concrete, so the compiler can monomorphize the selected path and timed closures need
//! neither dynamic dispatch nor an extra allocation.

use crate::catalogue_cases::{CaseKind, CaseSpec};

/// A Catalogue Matrix case selected independently of an execution backend or mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CaseAdapter {
    kind: CaseKind,
}

/// Metadata implemented by the concrete Rust, C, and Python case adapters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdapterContract {
    pub parameters: &'static str,
    pub output_kind: &'static str,
    pub output_arity: usize,
}

impl CaseAdapter {
    #[must_use]
    pub const fn new(kind: CaseKind) -> Self {
        Self { kind }
    }

    #[must_use]
    pub const fn kind(self) -> CaseKind {
        self.kind
    }

    /// Returns the metadata contract implemented by this case's concrete adapters.
    #[must_use]
    pub const fn contract(self) -> AdapterContract {
        match self.kind {
            CaseKind::Sma => AdapterContract {
                parameters: "timeperiod=14",
                output_kind: "float",
                output_arity: 1,
            },
            CaseKind::Bbands => AdapterContract {
                parameters: "timeperiod=20;nbdevup=2;nbdevdn=2;matype=SMA",
                output_kind: "float",
                output_arity: 3,
            },
            CaseKind::Rsi => AdapterContract {
                parameters: "timeperiod=14",
                output_kind: "float",
                output_arity: 1,
            },
            CaseKind::Macd => AdapterContract {
                parameters: "fastperiod=12;slowperiod=26;signalperiod=9",
                output_kind: "float",
                output_arity: 3,
            },
            CaseKind::Atr | CaseKind::Adx | CaseKind::LinearReg => AdapterContract {
                parameters: "timeperiod=14",
                output_kind: "float",
                output_arity: 1,
            },
            CaseKind::HtDcPhase
            | CaseKind::TypPrice
            | CaseKind::Obv
            | CaseKind::Sin
            | CaseKind::Add => AdapterContract {
                parameters: "none",
                output_kind: "float",
                output_arity: 1,
            },
            CaseKind::CdlDoji | CaseKind::CdlEngulfing | CaseKind::Cdl3WhiteSoldiers => {
                AdapterContract {
                    parameters: "candle_settings=TA-Lib defaults",
                    output_kind: "integer",
                    output_arity: 1,
                }
            }
        }
    }

    /// Rejects manifest metadata that no longer describes the executable adapters.
    pub fn validate_spec(self, spec: &CaseSpec) -> Result<(), String> {
        let contract = self.contract();
        if (spec.parameters, spec.output_kind, spec.output_arity)
            != (
                contract.parameters,
                contract.output_kind,
                contract.output_arity,
            )
        {
            return Err(format!(
                "case {} manifest metadata differs from executable adapter contract",
                spec.id
            ));
        }
        Ok(())
    }

    /// Dispatches to a concrete backend operation.
    ///
    /// `O` is statically known at each call site. In particular, this method does not erase a
    /// timed closure behind another trait object.
    pub fn execute<C, O>(self, operations: &mut O, context: C) -> O::Output
    where
        O: CaseOperations<C>,
    {
        match self.kind {
            CaseKind::Sma => operations.sma(context),
            CaseKind::Bbands => operations.bbands(context),
            CaseKind::Rsi => operations.rsi(context),
            CaseKind::Macd => operations.macd(context),
            CaseKind::Atr => operations.atr(context),
            CaseKind::Adx => operations.adx(context),
            CaseKind::HtDcPhase => operations.ht_dc_phase(context),
            CaseKind::CdlDoji => operations.cdl_doji(context),
            CaseKind::CdlEngulfing => operations.cdl_engulfing(context),
            CaseKind::Cdl3WhiteSoldiers => operations.cdl_3_white_soldiers(context),
            CaseKind::LinearReg => operations.linear_reg(context),
            CaseKind::TypPrice => operations.typ_price(context),
            CaseKind::Obv => operations.obv(context),
            CaseKind::Sin => operations.sin(context),
            CaseKind::Add => operations.add(context),
        }
    }
}

/// Concrete operations supported by every Catalogue execution backend.
///
/// Backends use `Context` to carry the fixture, execution mode, or preallocated C buffers. Keeping
/// it generic lets verification and timed-operation construction share the same case adapter while
/// returning different concrete result types.
pub trait CaseOperations<Context> {
    type Output;

    fn sma(&mut self, context: Context) -> Self::Output;
    fn bbands(&mut self, context: Context) -> Self::Output;
    fn rsi(&mut self, context: Context) -> Self::Output;
    fn macd(&mut self, context: Context) -> Self::Output;
    fn atr(&mut self, context: Context) -> Self::Output;
    fn adx(&mut self, context: Context) -> Self::Output;
    fn ht_dc_phase(&mut self, context: Context) -> Self::Output;
    fn cdl_doji(&mut self, context: Context) -> Self::Output;
    fn cdl_engulfing(&mut self, context: Context) -> Self::Output;
    fn cdl_3_white_soldiers(&mut self, context: Context) -> Self::Output;
    fn linear_reg(&mut self, context: Context) -> Self::Output;
    fn typ_price(&mut self, context: Context) -> Self::Output;
    fn obv(&mut self, context: Context) -> Self::Output;
    fn sin(&mut self, context: Context) -> Self::Output;
    fn add(&mut self, context: Context) -> Self::Output;
}
