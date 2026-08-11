# Technical Analysis Indicators

This context covers mathematical technical-analysis indicators evaluated over dense, ordered observation series in batch and streaming modes. Its intended coverage is the full TA-Lib indicator catalogue.

## Indicator language

**Indicator Catalogue**:
The complete set of TA-Lib-named indicators that defines this project's implementation scope; it does not define a compatible public interface.
_Avoid_: TA-Lib API surface, compatibility checklist

**Catalogue Coverage**:
The subset of the Indicator Catalogue currently available for evaluation. Increasing coverage does not add functions to the fixed catalogue.
_Avoid_: Catalogue expansion

**Indicator Definition**:
The mathematical meaning, parameters, edge behavior, and expected numerical result of an indicator. TA-Lib is the default reference, while a numerically more accurate implementation is preferred when it preserves that meaning.
_Avoid_: TA-Lib call semantics

**Pattern Recognition Indicator**:
An Indicator Definition that independently evaluates an OHLC Observation Series for one TA-Lib-named candlestick pattern. It does not aggregate or arbitrate results across patterns.
_Avoid_: Combined candlestick classifier, pattern scanner

**Indicator Configuration**:
The immutable parameter set that selects an indicator definition, such as a period or moving-average kind. It does not include observations accumulated while executing the indicator.
_Avoid_: Indicator state, calculator state

**Candle Settings**:
The immutable eleven-setting threshold collection carried by a Pattern Recognition Indicator Configuration. Its default is the TA-Lib v0.7.1 collection; only settings referenced by the Indicator Definition affect its result, Lookback, and Warm-up.
_Avoid_: Global candle settings, process-wide candle configuration

**Candle Setting**:
One candlestick threshold definition comprising a Candle Range Kind, an Average Period from zero through 100,000, and a finite nonnegative Factor. Period zero uses the current Observation range; a positive Period averages prior Observation ranges.
_Avoid_: Unvalidated threshold triple, mutable candle setting

**Candle Range Kind**:
The measurement selected by a Candle Setting: Real Body, High-Low Range, or Shadows.
_Avoid_: Arbitrary range function

**Real Body**:
The absolute difference between a Candle Observation's Close and Open.
_Avoid_: Signed body, normalized body

**High-Low Range**:
The difference between a Candle Observation's High and Low.
_Avoid_: True range, repaired range

**Upper Shadow**:
The difference between a Candle Observation's High and the greater of its Open and Close.
_Avoid_: Clamped upper wick

**Lower Shadow**:
The difference between the lesser of a Candle Observation's Open and Close and its Low.
_Avoid_: Clamped lower wick

**Shadows**:
The sum of a Candle Observation's Upper Shadow and Lower Shadow, equivalently its High-Low Range minus its Real Body.
_Avoid_: One individual shadow, clamped shadows

**Candle Color**:
The binary direction used by Pattern Recognition Indicators: white or bullish when Close is greater than or equal to Open, and black or bearish otherwise. A zero-body candle is white when an Indicator Definition consults color; doji is a threshold classification, not a third color.
_Avoid_: Three-state candle color, epsilon-based color

**Candle Average**:
The threshold derived from a Candle Setting at one source position. A positive Average Period averages the selected Candle Range Kind over immediately preceding Candle Observations, excluding the classified observation; period zero uses the current observation. The average is multiplied by the Factor and, only for Shadows, divided by two.
_Avoid_: Average including the classified observation, moving average output

**Penetration**:
A finite nonnegative ratio configured by the seven Pattern Recognition Indicator Definitions that expose it. Values above one are valid; the fixed fifty-percent rule in CDLPIERCING is not a Penetration configuration.
_Avoid_: Percentage restricted to zero through one

**Period**:
The configured observation count used by a rolling or recursive indicator.
_Avoid_: Window size when the indicator is recursive rather than windowed

**Period-based Moving Average**:
A single-output moving-average definition whose configured Period controls its averaging horizon. MESA Adaptive Moving Average is not period-based because its configuration and paired outputs have different meaning.
_Avoid_: Generic moving-average type that includes MAMA

## Series language

**Observation**:
One ordered input position containing the fields required by an indicator, such as a close value or an OHLCV tuple.
_Avoid_: Row, record

**Candle Observation**:
One finite Observation containing Open, High, Low, and Close. It carries no timestamp, volume, instrument identity, or gap metadata; ordering and OHLC consistency remain Observation Series responsibilities.
_Avoid_: Candlestick event, timestamped candle

**Observation Series**:
A dense, oldest-to-newest sequence of finite observations. Time gaps, resampling, missing-value repair, sign constraints, and OHLC consistency are responsibilities of the caller.
_Avoid_: Sparse series, timestamp-aware series

**Period Selection Series**:
A dense sequence of whole-number observation counts aligned with an Observation Series and consumed by a variable-period Indicator Definition. Each selection is constrained by the minimum and maximum Periods in the Indicator Configuration.
_Avoid_: Floating-point period series, variable Indicator Configuration

**Universe**:
A collection of instrument observation series processed under a common indicator definition or configuration.
_Avoid_: Dataset, batch

**Tick**:
One observation submitted to a streaming computation.
_Avoid_: Event, row

## Execution language

**Batch Computation**:
Evaluation of an indicator over a finite observation series.
_Avoid_: Bulk mode, offline mode

**Prepared Batch Runner**:
A reusable batch execution mode that applies one indicator configuration to many observation series without rebuilding execution state for each series. Concurrent workers use independent runners.
_Avoid_: Shared runner, streaming state

**Streaming Computation**:
Incremental evaluation that consumes ticks in order and retains state between them.
_Avoid_: Batch loop, prepared batch

**Lookback**:
The number of leading source positions before the first valid batch result; the first valid result is aligned to the zero-based source position equal to Lookback.
_Avoid_: Padding length

**Warm-up**:
The streaming phase spanning the first Lookback ticks. The following tick, at the source position equal to Lookback, produces the first valid result.
_Avoid_: Error, missing result

**Stabilization**:
A fixed initial observation span required by an Indicator Definition before its results are valid, beyond merely having enough inputs to evaluate its formula. It contributes to Lookback and Warm-up and is identical for every execution of that definition.
_Avoid_: Unstable period, global compatibility setting

## Result language

**Pattern Signal**:
The source-aligned result of one Pattern Recognition Indicator: No Match, or a Match comprising a Pattern Direction and Pattern Strength. No Match is a valid result and is distinct from Warm-up, when no result exists yet.
_Avoid_: Boolean pattern match, signed integer output

**Pattern Direction**:
The bullish or bearish direction of a matched Pattern Signal.
_Avoid_: Integer sign, market trend

**Pattern Strength**:
The categorical strength of a matched Pattern Signal: Partial for the magnitude-80 Engulfing or Harami boundary result, Standard for an ordinary magnitude-100 formation, or Confirmed for a magnitude-200 Hikkake confirmation. Strength is neither probability nor generic confidence, and each Indicator Definition determines which strengths it can emit.
_Avoid_: Match score, confidence percentage

**Pending Confirmation**:
The single unconfirmed Standard Hikkake formation retained for at most the next three source positions. A qualifying close produces a Confirmed Pattern Signal in the formation direction; a newer formation replaces it, including when that source position could also confirm it.
_Avoid_: Confirmation queue, indefinite pending pattern

**Compact Output**:
A result containing only valid indicator values together with the source range to which they belong.
_Avoid_: Padded output, full-length output

**Output Range**:
The contiguous range of source positions represented by a compact output. Compact element zero corresponds to the first position in this range.
_Avoid_: Output offset, padding count

**Aligned Output**:
A derived, source-length representation that places valid values at their source positions and explicitly represents unavailable positions.
_Avoid_: Implicitly padded output, sentinel-filled output

**Dominant Cycle Phase**:
The angular position of the dominant cycle at a valid source position, expressed in degrees using the canonical wrap of its Indicator Definition.
_Avoid_: Radians, unitless phase

**Trend Mode**:
A discrete Cycle Indicator result that classifies a valid source position as either Cycle or Trend. It does not express trend direction, strength, or probability.
_Avoid_: Boolean trend flag, integer 0/1 code, market regime
