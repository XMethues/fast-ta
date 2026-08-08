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

**Indicator Configuration**:
The immutable parameter set that selects an indicator definition, such as a period or moving-average kind. It does not include observations accumulated while executing the indicator.
_Avoid_: Indicator state, calculator state

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

**Observation Series**:
A dense, oldest-to-newest sequence of finite observations. Time gaps, resampling, missing-value repair, sign constraints, and OHLC consistency are responsibilities of the caller.
_Avoid_: Sparse series, timestamp-aware series

**Period Selection Series**:
A dense sequence of `usize` observation counts aligned with an Observation Series and consumed by a variable-period Indicator Definition. Each selection is constrained by the minimum and maximum Periods in the Indicator Configuration.
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
The number of leading source positions before the first valid batch result.
_Avoid_: Padding length

**Warm-up**:
The streaming phase before enough ticks have arrived to produce the first valid result.
_Avoid_: Error, missing result

**Stabilization**:
A fixed initial observation span required by an Indicator Definition before its results are valid, beyond merely having enough inputs to evaluate its formula. It contributes to Lookback and Warm-up and is identical for every execution of that definition.
_Avoid_: Unstable period, global compatibility setting

## Result language

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
The angular position of the dominant cycle at a valid source position, expressed as a `Float` number of degrees using the canonical wrap of its Indicator Definition.
_Avoid_: Radians, unitless phase

**Trend Mode**:
A discrete Cycle Indicator result that classifies a valid source position as either Cycle or Trend. It does not express trend direction, strength, or probability.
_Avoid_: Boolean trend flag, integer 0/1 code, market regime
