//! Representative Pattern Recognition execution-shape metadata.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatternShapeSpec {
    pub case_id: &'static str,
    pub execution_shape: &'static str,
    pub rationale: &'static str,
}

pub const PATTERN_SHAPES: [PatternShapeSpec; 3] = [
    PatternShapeSpec {
        case_id: "CDLDOJI",
        execution_shape: "single-setting stateful rolling average",
        rationale: "one immutable-default BodyDoji rolling history",
    },
    PatternShapeSpec {
        case_id: "CDL3WHITESOLDIERS",
        execution_shape: "multi-setting stateful rolling averages",
        rationale: "four independently aligned immutable-default rolling histories: ShadowVeryShort, BodyShort, Far, and Near",
    },
    PatternShapeSpec {
        case_id: "CDLENGULFING",
        execution_shape: "setting-free cross-candle predicate",
        rationale: "setting-free predicate over current and previous candles; Streaming retains cross-candle state",
    },
];
