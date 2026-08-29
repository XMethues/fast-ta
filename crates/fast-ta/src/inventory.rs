//! Indicator Catalogue and Catalogue Coverage queries.
//!
//! The catalogue records the 161 official TA-Lib-named Indicator Definitions
//! and projects them by implementation status, family, and owning Rust module.
//! Queries borrow the source-level table directly, so they remain allocation-free
//! and available in `no_std` builds.

/// Number of Indicator Definitions in the official catalogue.
pub const FUNCTION_COUNT: usize = 161;

/// Number of Indicator Definitions in the current Catalogue Coverage.
pub const IMPLEMENTED_FUNCTION_COUNT: usize = 161;

/// Official Indicator Definition family.
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
    /// All official families in TA-Lib display order.
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

    /// Human-readable official family label.
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

    /// Expected official Indicator Definition count for this family.
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

    /// Rust module that owns this family's implementations.
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

/// Catalogue Coverage state for an Indicator Definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationStatus {
    /// Available for evaluation in `fast-ta`.
    Implemented,
    /// Recorded in the Indicator Catalogue for future implementation.
    Planned,
}

/// One Indicator Definition in the official catalogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunctionInfo {
    /// Uppercase TA-Lib definition name.
    pub name: &'static str,
    /// Official definition family.
    pub group: FunctionGroup,
    /// Current Catalogue Coverage state.
    pub status: ImplementationStatus,
}

impl FunctionInfo {
    /// Returns true when this definition is in the current Catalogue Coverage.
    pub const fn is_implemented(self) -> bool {
        match self.status {
            ImplementationStatus::Implemented => true,
            ImplementationStatus::Planned => false,
        }
    }

    /// Rust module that owns this definition's implementation.
    pub const fn owner_module(self) -> &'static str {
        self.group.rust_module()
    }

    /// Rust module for this definition's official family.
    pub const fn rust_module(self) -> &'static str {
        self.owner_module()
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

/// Official Indicator Catalogue source table in family order.
pub const TALIB_FUNCTIONS: &[FunctionInfo] = &[
    // Overlap Studies — 18 functions.
    function!("ACCBANDS", OverlapStudies, Implemented),
    function!("BBANDS", OverlapStudies, Implemented),
    function!("DEMA", OverlapStudies, Implemented),
    function!("EMA", OverlapStudies, Implemented),
    function!("HT_TRENDLINE", OverlapStudies, Implemented),
    function!("KAMA", OverlapStudies, Implemented),
    function!("MA", OverlapStudies, Implemented),
    function!("MAMA", OverlapStudies, Implemented),
    function!("MAVP", OverlapStudies, Implemented),
    function!("MIDPOINT", OverlapStudies, Implemented),
    function!("MIDPRICE", OverlapStudies, Implemented),
    function!("SAR", OverlapStudies, Implemented),
    function!("SAREXT", OverlapStudies, Implemented),
    function!("SMA", OverlapStudies, Implemented),
    function!("T3", OverlapStudies, Implemented),
    function!("TEMA", OverlapStudies, Implemented),
    function!("TRIMA", OverlapStudies, Implemented),
    function!("WMA", OverlapStudies, Implemented),
    // Momentum Indicators — 31 functions.
    function!("ADX", MomentumIndicators, Implemented),
    function!("ADXR", MomentumIndicators, Implemented),
    function!("APO", MomentumIndicators, Implemented),
    function!("AROON", MomentumIndicators, Implemented),
    function!("AROONOSC", MomentumIndicators, Implemented),
    function!("BOP", MomentumIndicators, Implemented),
    function!("CCI", MomentumIndicators, Implemented),
    function!("CMO", MomentumIndicators, Implemented),
    function!("DX", MomentumIndicators, Implemented),
    function!("IMI", MomentumIndicators, Implemented),
    function!("MACD", MomentumIndicators, Implemented),
    function!("MACDEXT", MomentumIndicators, Implemented),
    function!("MACDFIX", MomentumIndicators, Implemented),
    function!("MFI", MomentumIndicators, Implemented),
    function!("MINUS_DI", MomentumIndicators, Implemented),
    function!("MINUS_DM", MomentumIndicators, Implemented),
    function!("MOM", MomentumIndicators, Implemented),
    function!("PLUS_DI", MomentumIndicators, Implemented),
    function!("PLUS_DM", MomentumIndicators, Implemented),
    function!("PPO", MomentumIndicators, Implemented),
    function!("ROC", MomentumIndicators, Implemented),
    function!("ROCP", MomentumIndicators, Implemented),
    function!("ROCR", MomentumIndicators, Implemented),
    function!("ROCR100", MomentumIndicators, Implemented),
    function!("RSI", MomentumIndicators, Implemented),
    function!("STOCH", MomentumIndicators, Implemented),
    function!("STOCHF", MomentumIndicators, Implemented),
    function!("STOCHRSI", MomentumIndicators, Implemented),
    function!("TRIX", MomentumIndicators, Implemented),
    function!("ULTOSC", MomentumIndicators, Implemented),
    function!("WILLR", MomentumIndicators, Implemented),
    // Volume Indicators — 3 functions.
    function!("AD", VolumeIndicators, Implemented),
    function!("ADOSC", VolumeIndicators, Implemented),
    function!("OBV", VolumeIndicators, Implemented),
    // Volatility Indicators — 3 functions.
    function!("ATR", VolatilityIndicators, Implemented),
    function!("NATR", VolatilityIndicators, Implemented),
    function!("TRANGE", VolatilityIndicators, Implemented),
    // Price Transform — 5 functions.
    function!("AVGDEV", PriceTransform, Implemented),
    function!("AVGPRICE", PriceTransform, Implemented),
    function!("MEDPRICE", PriceTransform, Implemented),
    function!("TYPPRICE", PriceTransform, Implemented),
    function!("WCLPRICE", PriceTransform, Implemented),
    // Cycle Indicators — 5 functions.
    function!("HT_DCPERIOD", CycleIndicators, Implemented),
    function!("HT_DCPHASE", CycleIndicators, Implemented),
    function!("HT_PHASOR", CycleIndicators, Implemented),
    function!("HT_SINE", CycleIndicators, Implemented),
    function!("HT_TRENDMODE", CycleIndicators, Implemented),
    // Pattern Recognition — 61 functions.
    function!("CDL2CROWS", PatternRecognition, Implemented),
    function!("CDL3BLACKCROWS", PatternRecognition, Implemented),
    function!("CDL3INSIDE", PatternRecognition, Implemented),
    function!("CDL3LINESTRIKE", PatternRecognition, Implemented),
    function!("CDL3OUTSIDE", PatternRecognition, Implemented),
    function!("CDL3STARSINSOUTH", PatternRecognition, Implemented),
    function!("CDL3WHITESOLDIERS", PatternRecognition, Implemented),
    function!("CDLABANDONEDBABY", PatternRecognition, Implemented),
    function!("CDLADVANCEBLOCK", PatternRecognition, Implemented),
    function!("CDLBELTHOLD", PatternRecognition, Implemented),
    function!("CDLBREAKAWAY", PatternRecognition, Implemented),
    function!("CDLCLOSINGMARUBOZU", PatternRecognition, Implemented),
    function!("CDLCONCEALBABYSWALL", PatternRecognition, Implemented),
    function!("CDLCOUNTERATTACK", PatternRecognition, Implemented),
    function!("CDLDARKCLOUDCOVER", PatternRecognition, Implemented),
    function!("CDLDOJI", PatternRecognition, Implemented),
    function!("CDLDOJISTAR", PatternRecognition, Implemented),
    function!("CDLDRAGONFLYDOJI", PatternRecognition, Implemented),
    function!("CDLENGULFING", PatternRecognition, Implemented),
    function!("CDLEVENINGDOJISTAR", PatternRecognition, Implemented),
    function!("CDLEVENINGSTAR", PatternRecognition, Implemented),
    function!("CDLGAPSIDESIDEWHITE", PatternRecognition, Implemented),
    function!("CDLGRAVESTONEDOJI", PatternRecognition, Implemented),
    function!("CDLHAMMER", PatternRecognition, Implemented),
    function!("CDLHANGINGMAN", PatternRecognition, Implemented),
    function!("CDLHARAMI", PatternRecognition, Implemented),
    function!("CDLHARAMICROSS", PatternRecognition, Implemented),
    function!("CDLHIGHWAVE", PatternRecognition, Implemented),
    function!("CDLHIKKAKE", PatternRecognition, Implemented),
    function!("CDLHIKKAKEMOD", PatternRecognition, Implemented),
    function!("CDLHOMINGPIGEON", PatternRecognition, Implemented),
    function!("CDLIDENTICAL3CROWS", PatternRecognition, Implemented),
    function!("CDLINNECK", PatternRecognition, Implemented),
    function!("CDLINVERTEDHAMMER", PatternRecognition, Implemented),
    function!("CDLKICKING", PatternRecognition, Implemented),
    function!("CDLKICKINGBYLENGTH", PatternRecognition, Implemented),
    function!("CDLLADDERBOTTOM", PatternRecognition, Implemented),
    function!("CDLLONGLEGGEDDOJI", PatternRecognition, Implemented),
    function!("CDLLONGLINE", PatternRecognition, Implemented),
    function!("CDLMARUBOZU", PatternRecognition, Implemented),
    function!("CDLMATCHINGLOW", PatternRecognition, Implemented),
    function!("CDLMATHOLD", PatternRecognition, Implemented),
    function!("CDLMORNINGDOJISTAR", PatternRecognition, Implemented),
    function!("CDLMORNINGSTAR", PatternRecognition, Implemented),
    function!("CDLONNECK", PatternRecognition, Implemented),
    function!("CDLPIERCING", PatternRecognition, Implemented),
    function!("CDLRICKSHAWMAN", PatternRecognition, Implemented),
    function!("CDLRISEFALL3METHODS", PatternRecognition, Implemented),
    function!("CDLSEPARATINGLINES", PatternRecognition, Implemented),
    function!("CDLSHOOTINGSTAR", PatternRecognition, Implemented),
    function!("CDLSHORTLINE", PatternRecognition, Implemented),
    function!("CDLSPINNINGTOP", PatternRecognition, Implemented),
    function!("CDLSTALLEDPATTERN", PatternRecognition, Implemented),
    function!("CDLSTICKSANDWICH", PatternRecognition, Implemented),
    function!("CDLTAKURI", PatternRecognition, Implemented),
    function!("CDLTASUKIGAP", PatternRecognition, Implemented),
    function!("CDLTHRUSTING", PatternRecognition, Implemented),
    function!("CDLTRISTAR", PatternRecognition, Implemented),
    function!("CDLUNIQUE3RIVER", PatternRecognition, Implemented),
    function!("CDLUPSIDEGAP2CROWS", PatternRecognition, Implemented),
    function!("CDLXSIDEGAP3METHODS", PatternRecognition, Implemented),
    // Statistic Functions — 9 functions.
    function!("BETA", StatisticFunctions, Implemented),
    function!("CORREL", StatisticFunctions, Implemented),
    function!("LINEARREG", StatisticFunctions, Implemented),
    function!("LINEARREG_ANGLE", StatisticFunctions, Implemented),
    function!("LINEARREG_INTERCEPT", StatisticFunctions, Implemented),
    function!("LINEARREG_SLOPE", StatisticFunctions, Implemented),
    function!("STDDEV", StatisticFunctions, Implemented),
    function!("TSF", StatisticFunctions, Implemented),
    function!("VAR", StatisticFunctions, Implemented),
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

/// Allocation-free query model for the Indicator Catalogue.
///
/// The model borrows the single source-level definition table. Its iterators
/// preserve official catalogue order.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndicatorCatalogue;

/// Queryable official Indicator Catalogue.
pub const INDICATOR_CATALOGUE: IndicatorCatalogue = IndicatorCatalogue;

impl IndicatorCatalogue {
    /// Returns every Indicator Definition in official catalogue order.
    pub const fn definitions(self) -> &'static [FunctionInfo] {
        TALIB_FUNCTIONS
    }

    /// Looks up an Indicator Definition by its uppercase TA-Lib name.
    pub fn definition(self, name: &str) -> Option<&'static FunctionInfo> {
        TALIB_FUNCTIONS
            .iter()
            .find(|definition| definition.name == name)
    }

    /// Iterates over the current Catalogue Coverage in official catalogue order.
    pub fn implemented_definitions(self) -> impl Iterator<Item = &'static FunctionInfo> + Clone {
        TALIB_FUNCTIONS
            .iter()
            .filter(|definition| definition.is_implemented())
    }

    /// Looks up an implemented Indicator Definition by uppercase TA-Lib name.
    pub fn implemented_definition(self, name: &str) -> Option<&'static FunctionInfo> {
        self.definition(name)
            .filter(|definition| definition.is_implemented())
    }

    /// Projects the Indicator Definitions belonging to one official family.
    pub fn family(
        self,
        family: FunctionGroup,
    ) -> impl Iterator<Item = &'static FunctionInfo> + Clone {
        TALIB_FUNCTIONS
            .iter()
            .filter(move |definition| definition.group == family)
    }

    /// Counts the Indicator Definitions belonging to one official family.
    pub fn family_count(self, family: FunctionGroup) -> usize {
        self.family(family).count()
    }

    /// Projects implemented Indicator Definitions belonging to one family.
    pub fn implemented_family(
        self,
        family: FunctionGroup,
    ) -> impl Iterator<Item = &'static FunctionInfo> + Clone {
        self.family(family)
            .filter(|definition| definition.is_implemented())
    }

    /// Counts implemented Indicator Definitions belonging to one family.
    pub fn implemented_family_count(self, family: FunctionGroup) -> usize {
        self.implemented_family(family).count()
    }
}

/// Finds an Indicator Definition by uppercase TA-Lib name.
pub fn function(name: &str) -> Option<&'static FunctionInfo> {
    INDICATOR_CATALOGUE.definition(name)
}
