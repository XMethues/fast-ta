//! Official TA-Lib function inventory and implementation ledger.
//!
//! This source-level ledger records the 161-function TA-Lib surface that
//! `ta-core` intends to implement in Rust. It deliberately follows official
//! TA-Lib groups rather than the older local planning documents, so future work
//! can advance group-by-group without rediscovering scope.

/// Total official TA-Lib function count tracked by this ledger.
pub const FUNCTION_COUNT: usize = 161;

/// Number of functions currently implemented in Rust `ta-core`.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 39;

/// Official TA-Lib function group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionGroup {
    /// Overlap Studies.
    OverlapStudies,
    /// Momentum Indicators.
    MomentumIndicators,
    /// Volume Indicators.
    VolumeIndicators,
    /// Volatility Indicators.
    VolatilityIndicators,
    /// Price Transform.
    PriceTransform,
    /// Cycle Indicators.
    CycleIndicators,
    /// Pattern Recognition.
    PatternRecognition,
    /// Statistic Functions.
    StatisticFunctions,
    /// Math Transform.
    MathTransform,
    /// Math Operators.
    MathOperators,
}

impl FunctionGroup {
    /// All official groups in TA-Lib display order.
    pub const ALL: &'static [Self] = &[
        Self::OverlapStudies,
        Self::MomentumIndicators,
        Self::VolumeIndicators,
        Self::VolatilityIndicators,
        Self::PriceTransform,
        Self::CycleIndicators,
        Self::PatternRecognition,
        Self::StatisticFunctions,
        Self::MathTransform,
        Self::MathOperators,
    ];

    /// Human-readable official group label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OverlapStudies => "Overlap Studies",
            Self::MomentumIndicators => "Momentum Indicators",
            Self::VolumeIndicators => "Volume Indicators",
            Self::VolatilityIndicators => "Volatility Indicators",
            Self::PriceTransform => "Price Transform",
            Self::CycleIndicators => "Cycle Indicators",
            Self::PatternRecognition => "Pattern Recognition",
            Self::StatisticFunctions => "Statistic Functions",
            Self::MathTransform => "Math Transform",
            Self::MathOperators => "Math Operators",
        }
    }

    /// Expected official function count for this group.
    pub const fn expected_count(self) -> usize {
        match self {
            Self::OverlapStudies => 18,
            Self::MomentumIndicators => 31,
            Self::VolumeIndicators => 3,
            Self::VolatilityIndicators => 3,
            Self::PriceTransform => 5,
            Self::CycleIndicators => 5,
            Self::PatternRecognition => 61,
            Self::StatisticFunctions => 9,
            Self::MathTransform => 15,
            Self::MathOperators => 11,
        }
    }

    /// Rust module planned for this group.
    pub const fn rust_module(self) -> &'static str {
        match self {
            Self::OverlapStudies => "overlap",
            Self::MomentumIndicators => "momentum",
            Self::VolumeIndicators => "volume",
            Self::VolatilityIndicators => "volatility",
            Self::PriceTransform => "price_transform",
            Self::CycleIndicators => "cycle",
            Self::PatternRecognition => "pattern_recognition",
            Self::StatisticFunctions => "statistic",
            Self::MathTransform => "math_transform",
            Self::MathOperators => "math_operators",
        }
    }
}

/// Implementation state for a TA-Lib function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationStatus {
    /// Implemented in Rust `ta-core`.
    Implemented,
    /// Official TA-Lib function recorded for future Rust implementation.
    Planned,
}

/// One official TA-Lib function inventory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionInfo {
    /// Uppercase TA-Lib function name.
    pub name: &'static str,
    /// Official TA-Lib group.
    pub group: FunctionGroup,
    /// Current Rust implementation status.
    pub status: ImplementationStatus,
}

impl FunctionInfo {
    /// Returns true when this function is implemented in Rust `ta-core`.
    pub const fn is_implemented(self) -> bool {
        match self.status {
            ImplementationStatus::Implemented => true,
            ImplementationStatus::Planned => false,
        }
    }

    /// Rust module for this function's official group.
    pub const fn rust_module(self) -> &'static str {
        self.group.rust_module()
    }
}

macro_rules! function {
    ($name:literal, $group:ident, $status:ident) => {
        FunctionInfo {
            name: $name,
            group: FunctionGroup::$group,
            status: ImplementationStatus::$status,
        }
    };
}

/// Official TA-Lib function inventory in group order.
pub const TALIB_FUNCTIONS: &[FunctionInfo] = &[
    // Overlap Studies — 18 functions.
    function!("ACCBANDS", OverlapStudies, Planned),
    function!("BBANDS", OverlapStudies, Planned),
    function!("DEMA", OverlapStudies, Implemented),
    function!("EMA", OverlapStudies, Implemented),
    function!("HT_TRENDLINE", OverlapStudies, Planned),
    function!("KAMA", OverlapStudies, Planned),
    function!("MA", OverlapStudies, Implemented),
    function!("MAMA", OverlapStudies, Planned),
    function!("MAVP", OverlapStudies, Planned),
    function!("MIDPOINT", OverlapStudies, Planned),
    function!("MIDPRICE", OverlapStudies, Planned),
    function!("SAR", OverlapStudies, Planned),
    function!("SAREXT", OverlapStudies, Planned),
    function!("SMA", OverlapStudies, Implemented),
    function!("T3", OverlapStudies, Implemented),
    function!("TEMA", OverlapStudies, Implemented),
    function!("TRIMA", OverlapStudies, Implemented),
    function!("WMA", OverlapStudies, Implemented),
    // Momentum Indicators — 31 functions.
    function!("ADX", MomentumIndicators, Planned),
    function!("ADXR", MomentumIndicators, Planned),
    function!("APO", MomentumIndicators, Planned),
    function!("AROON", MomentumIndicators, Planned),
    function!("AROONOSC", MomentumIndicators, Planned),
    function!("BOP", MomentumIndicators, Planned),
    function!("CCI", MomentumIndicators, Planned),
    function!("CMO", MomentumIndicators, Planned),
    function!("DX", MomentumIndicators, Planned),
    function!("IMI", MomentumIndicators, Planned),
    function!("MACD", MomentumIndicators, Planned),
    function!("MACDEXT", MomentumIndicators, Planned),
    function!("MACDFIX", MomentumIndicators, Planned),
    function!("MFI", MomentumIndicators, Planned),
    function!("MINUS_DI", MomentumIndicators, Planned),
    function!("MINUS_DM", MomentumIndicators, Planned),
    function!("MOM", MomentumIndicators, Planned),
    function!("PLUS_DI", MomentumIndicators, Planned),
    function!("PLUS_DM", MomentumIndicators, Planned),
    function!("PPO", MomentumIndicators, Planned),
    function!("ROC", MomentumIndicators, Planned),
    function!("ROCP", MomentumIndicators, Planned),
    function!("ROCR", MomentumIndicators, Planned),
    function!("ROCR100", MomentumIndicators, Planned),
    function!("RSI", MomentumIndicators, Planned),
    function!("STOCH", MomentumIndicators, Planned),
    function!("STOCHF", MomentumIndicators, Planned),
    function!("STOCHRSI", MomentumIndicators, Planned),
    function!("TRIX", MomentumIndicators, Planned),
    function!("ULTOSC", MomentumIndicators, Planned),
    function!("WILLR", MomentumIndicators, Planned),
    // Volume Indicators — 3 functions.
    function!("AD", VolumeIndicators, Planned),
    function!("ADOSC", VolumeIndicators, Planned),
    function!("OBV", VolumeIndicators, Planned),
    // Volatility Indicators — 3 functions.
    function!("ATR", VolatilityIndicators, Planned),
    function!("NATR", VolatilityIndicators, Planned),
    function!("TRANGE", VolatilityIndicators, Planned),
    // Price Transform — 5 functions.
    function!("AVGDEV", PriceTransform, Implemented),
    function!("AVGPRICE", PriceTransform, Implemented),
    function!("MEDPRICE", PriceTransform, Implemented),
    function!("TYPPRICE", PriceTransform, Implemented),
    function!("WCLPRICE", PriceTransform, Implemented),
    // Cycle Indicators — 5 functions.
    function!("HT_DCPERIOD", CycleIndicators, Planned),
    function!("HT_DCPHASE", CycleIndicators, Planned),
    function!("HT_PHASOR", CycleIndicators, Planned),
    function!("HT_SINE", CycleIndicators, Planned),
    function!("HT_TRENDMODE", CycleIndicators, Planned),
    // Pattern Recognition — 61 functions.
    function!("CDL2CROWS", PatternRecognition, Planned),
    function!("CDL3BLACKCROWS", PatternRecognition, Planned),
    function!("CDL3INSIDE", PatternRecognition, Planned),
    function!("CDL3LINESTRIKE", PatternRecognition, Planned),
    function!("CDL3OUTSIDE", PatternRecognition, Planned),
    function!("CDL3STARSINSOUTH", PatternRecognition, Planned),
    function!("CDL3WHITESOLDIERS", PatternRecognition, Planned),
    function!("CDLABANDONEDBABY", PatternRecognition, Planned),
    function!("CDLADVANCEBLOCK", PatternRecognition, Planned),
    function!("CDLBELTHOLD", PatternRecognition, Planned),
    function!("CDLBREAKAWAY", PatternRecognition, Planned),
    function!("CDLCLOSINGMARUBOZU", PatternRecognition, Planned),
    function!("CDLCONCEALBABYSWALL", PatternRecognition, Planned),
    function!("CDLCOUNTERATTACK", PatternRecognition, Planned),
    function!("CDLDARKCLOUDCOVER", PatternRecognition, Planned),
    function!("CDLDOJI", PatternRecognition, Planned),
    function!("CDLDOJISTAR", PatternRecognition, Planned),
    function!("CDLDRAGONFLYDOJI", PatternRecognition, Planned),
    function!("CDLENGULFING", PatternRecognition, Planned),
    function!("CDLEVENINGDOJISTAR", PatternRecognition, Planned),
    function!("CDLEVENINGSTAR", PatternRecognition, Planned),
    function!("CDLGAPSIDESIDEWHITE", PatternRecognition, Planned),
    function!("CDLGRAVESTONEDOJI", PatternRecognition, Planned),
    function!("CDLHAMMER", PatternRecognition, Planned),
    function!("CDLHANGINGMAN", PatternRecognition, Planned),
    function!("CDLHARAMI", PatternRecognition, Planned),
    function!("CDLHARAMICROSS", PatternRecognition, Planned),
    function!("CDLHIGHWAVE", PatternRecognition, Planned),
    function!("CDLHIKKAKE", PatternRecognition, Planned),
    function!("CDLHIKKAKEMOD", PatternRecognition, Planned),
    function!("CDLHOMINGPIGEON", PatternRecognition, Planned),
    function!("CDLIDENTICAL3CROWS", PatternRecognition, Planned),
    function!("CDLINNECK", PatternRecognition, Planned),
    function!("CDLINVERTEDHAMMER", PatternRecognition, Planned),
    function!("CDLKICKING", PatternRecognition, Planned),
    function!("CDLKICKINGBYLENGTH", PatternRecognition, Planned),
    function!("CDLLADDERBOTTOM", PatternRecognition, Planned),
    function!("CDLLONGLEGGEDDOJI", PatternRecognition, Planned),
    function!("CDLLONGLINE", PatternRecognition, Planned),
    function!("CDLMARUBOZU", PatternRecognition, Planned),
    function!("CDLMATCHINGLOW", PatternRecognition, Planned),
    function!("CDLMATHOLD", PatternRecognition, Planned),
    function!("CDLMORNINGDOJISTAR", PatternRecognition, Planned),
    function!("CDLMORNINGSTAR", PatternRecognition, Planned),
    function!("CDLONNECK", PatternRecognition, Planned),
    function!("CDLPIERCING", PatternRecognition, Planned),
    function!("CDLRICKSHAWMAN", PatternRecognition, Planned),
    function!("CDLRISEFALL3METHODS", PatternRecognition, Planned),
    function!("CDLSEPARATINGLINES", PatternRecognition, Planned),
    function!("CDLSHOOTINGSTAR", PatternRecognition, Planned),
    function!("CDLSHORTLINE", PatternRecognition, Planned),
    function!("CDLSPINNINGTOP", PatternRecognition, Planned),
    function!("CDLSTALLEDPATTERN", PatternRecognition, Planned),
    function!("CDLSTICKSANDWICH", PatternRecognition, Planned),
    function!("CDLTAKURI", PatternRecognition, Planned),
    function!("CDLTASUKIGAP", PatternRecognition, Planned),
    function!("CDLTHRUSTING", PatternRecognition, Planned),
    function!("CDLTRISTAR", PatternRecognition, Planned),
    function!("CDLUNIQUE3RIVER", PatternRecognition, Planned),
    function!("CDLUPSIDEGAP2CROWS", PatternRecognition, Planned),
    function!("CDLXSIDEGAP3METHODS", PatternRecognition, Planned),
    // Statistic Functions — 9 functions.
    function!("BETA", StatisticFunctions, Planned),
    function!("CORREL", StatisticFunctions, Planned),
    function!("LINEARREG", StatisticFunctions, Planned),
    function!("LINEARREG_ANGLE", StatisticFunctions, Planned),
    function!("LINEARREG_INTERCEPT", StatisticFunctions, Planned),
    function!("LINEARREG_SLOPE", StatisticFunctions, Planned),
    function!("STDDEV", StatisticFunctions, Planned),
    function!("TSF", StatisticFunctions, Planned),
    function!("VAR", StatisticFunctions, Planned),
    // Math Transform — 15 functions.
    function!("ACOS", MathTransform, Implemented),
    function!("ASIN", MathTransform, Implemented),
    function!("ATAN", MathTransform, Implemented),
    function!("CEIL", MathTransform, Implemented),
    function!("COS", MathTransform, Implemented),
    function!("COSH", MathTransform, Implemented),
    function!("EXP", MathTransform, Implemented),
    function!("FLOOR", MathTransform, Implemented),
    function!("LN", MathTransform, Implemented),
    function!("LOG10", MathTransform, Implemented),
    function!("SIN", MathTransform, Implemented),
    function!("SINH", MathTransform, Implemented),
    function!("SQRT", MathTransform, Implemented),
    function!("TAN", MathTransform, Implemented),
    function!("TANH", MathTransform, Implemented),
    // Math Operators — 11 functions.
    function!("ADD", MathOperators, Implemented),
    function!("DIV", MathOperators, Implemented),
    function!("MAX", MathOperators, Implemented),
    function!("MAXINDEX", MathOperators, Implemented),
    function!("MIN", MathOperators, Implemented),
    function!("MININDEX", MathOperators, Implemented),
    function!("MINMAX", MathOperators, Implemented),
    function!("MINMAXINDEX", MathOperators, Implemented),
    function!("MULT", MathOperators, Implemented),
    function!("SUB", MathOperators, Implemented),
    function!("SUM", MathOperators, Implemented),
];

/// Finds a TA-Lib function by uppercase name.
pub fn function(name: &str) -> Option<&'static FunctionInfo> {
    TALIB_FUNCTIONS.iter().find(|info| info.name == name)
}
