# Official TA-Lib Pattern Recognition reference model

## Resolution and authority

Use official TA-Lib **v0.7.1**, commit [`2247d599bddf37ed37e3a709371517e46efc66f6`](https://github.com/TA-Lib/ta-lib/commit/2247d599bddf37ed37e3a709371517e46efc66f6), as the immutable Pattern Recognition compatibility reference. The pinned source declares 0.7.1 ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/CMakeLists.txt#L8-L12)); it is the official [v0.7.1 release](https://github.com/TA-Lib/ta-lib/releases/tag/v0.7.1), not a moving branch.

**Resolution gist:** 61 OHLC-to-integer detectors whose observable model includes 11 mutable Candle Settings, dynamic lookbacks, seven penetration options, exact comparison boundaries, and exceptional `±80` and `±200` signals—not merely 61 boolean rules.

This records upstream facts; it does not design a Rust API, choose project policy, or implement indicators.

The first-party function index lists one contiguous 61-name Pattern Recognition group ([docs](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/docs/functions/index.md#L105-L167)); the abstraction table registers the same sequence ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_abstract/tables/table_c.c#L1677-L1737)). It exactly matches fast-ta's 61 `Planned` names ([local inventory](../../crates/ta-core/src/inventory.rs#L221-L282)): no omission, addition, or spelling difference.

Shipped pattern C files identify `ta_codegen/input/<name>/` as source of truth ([notice](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_func/ta_CDLHIKKAKE.c#L34-L36)); per-function links below point to the pinned template containing lookback and predicate.

## Common signature and output range

All 61 metadata records accept OHLC and return one integer output ([representative metadata](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl2crows/cdl2crows.yaml#L1-L15)). Calls use common `startIdx`, `endIdx`, inputs, optional inputs, `outBegIdx`, `outNBElement`, and output array ([API guide](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/docs/api.md#L69-L102)). Output starts at `max(startIdx, lookback)`, is dense thereafter, and insufficient input yields zero elements ([contract](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/docs/api.md#L126-L177)). Every detector writes `0` on no match.

### Seven penetration signature exceptions

Metadata calls penetration a percentage but permits `0..TA_REAL_MAX`, including values above 1.0:

| Function | Default | Pinned metadata |
|---|---:|---|
| `CDLABANDONEDBABY` | 0.3 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlabandonedbaby/cdlabandonedbaby.yaml#L10-L17) |
| `CDLDARKCLOUDCOVER` | 0.5 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdldarkcloudcover/cdldarkcloudcover.yaml#L10-L17) |
| `CDLEVENINGDOJISTAR` | 0.3 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdleveningdojistar/cdleveningdojistar.yaml#L10-L17) |
| `CDLEVENINGSTAR` | 0.3 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdleveningstar/cdleveningstar.yaml#L10-L17) |
| `CDLMATHOLD` | 0.5 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmathold/cdlmathold.yaml#L10-L17) |
| `CDLMORNINGDOJISTAR` | 0.3 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmorningdojistar/cdlmorningdojistar.yaml#L10-L17) |
| `CDLMORNINGSTAR` | 0.3 | [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmorningstar/cdlmorningstar.yaml#L10-L17) |

Generated entry points substitute defaults for `TA_REAL_DEFAULT`, reject negatives, otherwise allow through `TA_REAL_MAX`, and return `-1` from lookback on invalid input ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_func/ta_CDLDARKCLOUDCOVER.c#L59-L68)). `CDLPIERCING` instead hard-codes strict `>50%` ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlpiercing/cdlpiercing.c#L61-L87)).

## Candle Settings reference model

TA-Lib defines ranges `RealBody`, `HighLow`, `Shadows` and settings `BodyLong`, `BodyVeryLong`, `BodyShort`, `BodyDoji`, `ShadowLong`, `ShadowVeryLong`, `ShadowShort`, `ShadowVeryShort`, `Near`, `Far`, `Equal` ([enums](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/include/ta_defs.h#L300-L327)).

| Setting | Range | Period | Factor | Effective threshold |
|---|---|---:|---:|---|
| BodyLong | RealBody | 10 | 1.0 | average prior real body |
| BodyVeryLong | RealBody | 10 | 3.0 | 3 × average prior real body |
| BodyShort | RealBody | 10 | 1.0 | average prior real body |
| BodyDoji | HighLow | 10 | 0.1 | 10% × average prior high-low |
| ShadowLong | RealBody | 0 | 1.0 | current real body |
| ShadowVeryLong | RealBody | 0 | 2.0 | 2 × current real body |
| ShadowShort | Shadows | 10 | 1.0 | half average prior shadow sum |
| ShadowVeryShort | HighLow | 10 | 0.1 | 10% × average prior high-low |
| Near | HighLow | 5 | 0.2 | 20% × average prior high-low |
| Far | HighLow | 5 | 0.6 | 60% × average prior high-low |
| Equal | HighLow | 5 | 0.05 | 5% × average prior high-low |

The triples and meanings are literal defaults ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_common/ta_global.c#L134-L161)); initialization restores all and restore resets one/all ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_common/ta_global.c#L90-L103), [source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_common/ta_global.c#L164-L172)).

For setting `S`, sum `R`, candle `i`: `CandleAverage = factor × (avgPeriod != 0 ? R/avgPeriod : CandleRange(S,i))`, divided by 2 only for `Shadows`. `RealBody=abs(close-open)`, `HighLow=high-low`, `Shadows=high-low-abs(close-open)` ([source helper](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/helpers/candlestick.c#L1-L53), [shipped macro](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_func/ta_utility.h#L335-L368)). Nonzero-period sums update after the predicate, excluding the evaluated pattern candle ([example](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlharami/cdlharami.c#L106-L113)).

Settings are process-global mutable state, valid until restored ([header](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/include/ta_func.h#L5316-L5335)). The setter validates only `settingType`, then stores range, period, factor unchecked ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_common/ta_global.c#L117-L131)). Custom periods change thresholds and lookbacks.

## Complete 61-function matrix

“Rule bars” counts adjacent OHLC bars read by the direct predicate, excluding setting history. Formula names mean current `avgPeriod`; `→` gives default lookback. Each name links exact pinned lookback/rule. `T` means source explicitly leaves trend/context significance to caller. All also emit `0`.

| Function/source | Rule bars | Settings | Lookback → default | Nonzero output / exception |
|---|---:|---|---|---|
| [`CDL2CROWS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl2crows/cdl2crows.c#L15-L84) | 3 | BodyLong | `BodyLong+2` → 12 | −100; T |
| [`CDL3BLACKCROWS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3blackcrows/cdl3blackcrows.c#L15-L96) | 4 | ShadowVeryShort | `ShadowVeryShort+3` → 13 | −100; T |
| [`CDL3INSIDE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3inside/cdl3inside.c#L16-L92) | 3 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+2` → 12 | opposite first ×100; T |
| [`CDL3LINESTRIKE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3linestrike/cdl3linestrike.c#L16-L104) | 4 | Near | `Near+3` → 8 | third (i−1) color ×100; T |
| [`CDL3OUTSIDE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3outside/cdl3outside.c#L16-L80) | 3 | — | `3` → 3 | engulfing (i−1) color ×100; T |
| [`CDL3STARSINSOUTH`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3starsinsouth/cdl3starsinsouth.c#L16-L123) | 3 | BodyLong, BodyShort, ShadowLong, ShadowVeryShort | `max(max(ShadowVeryShort,ShadowLong),max(BodyLong,BodyShort))+2` → 12 | +100; T |
| [`CDL3WHITESOLDIERS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3whitesoldiers/cdl3whitesoldiers.c#L16-L135) | 3 | BodyShort, Far, Near, ShadowVeryShort | `max(max(ShadowVeryShort,BodyShort),max(Far,Near))+2` → 12 | +100; T |
| [`CDLABANDONEDBABY`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlabandonedbaby/cdlabandonedbaby.c#L16-L121) | 3 | BodyDoji, BodyLong, BodyShort | `max(max(BodyDoji,BodyLong),BodyShort)+2` → 12 | final color ×100; P=.3; T |
| [`CDLADVANCEBLOCK`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdladvanceblock/cdladvanceblock.c#L16-L167) | 3 | BodyLong, Far, Near, ShadowLong, ShadowShort | `max(max(max(ShadowLong,ShadowShort),max(Far,Near)),BodyLong)+2` → 12 | −100; T |
| [`CDLBELTHOLD`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlbelthold/cdlbelthold.c#L16-L90) | 1 | BodyLong, ShadowVeryShort | `max(BodyLong,ShadowVeryShort)` → 10 | current color ×100 |
| [`CDLBREAKAWAY`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlbreakaway/cdlbreakaway.c#L16-L98) | 5 | BodyLong | `BodyLong+4` → 14 | final color ×100; T |
| [`CDLCLOSINGMARUBOZU`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlclosingmarubozu/cdlclosingmarubozu.c#L16-L91) | 1 | BodyLong, ShadowVeryShort | `max(BodyLong,ShadowVeryShort)` → 10 | current color ×100 |
| [`CDLCONCEALBABYSWALL`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlconcealbabyswall/cdlconcealbabyswall.c#L16-L99) | 4 | ShadowVeryShort | `ShadowVeryShort+3` → 13 | +100; T |
| [`CDLCOUNTERATTACK`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlcounterattack/cdlcounterattack.c#L16-L92) | 2 | BodyLong, Equal | `max(Equal,BodyLong)+1` → 11 | second color ×100; T |
| [`CDLDARKCLOUDCOVER`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdldarkcloudcover/cdldarkcloudcover.c#L16-L86) | 2 | BodyLong | `BodyLong+1` → 11 | −100; P=.5; T |
| [`CDLDOJI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdldoji/cdldoji.c#L16-L74) | 1 | BodyDoji | `BodyDoji` → 10 | +100 |
| [`CDLDOJISTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdldojistar/cdldojistar.c#L16-L90) | 2 | BodyDoji, BodyLong | `max(BodyDoji,BodyLong)+1` → 11 | opposite first ×100 |
| [`CDLDRAGONFLYDOJI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdldragonflydoji/cdldragonflydoji.c#L16-L87) | 1 | BodyDoji, ShadowVeryShort | `max(BodyDoji,ShadowVeryShort)` → 10 | +100 |
| [`CDLENGULFING`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlengulfing/cdlengulfing.c#L18-L90) | 2 | — | `2` → 2 | current color ×80/100; T |
| [`CDLEVENINGDOJISTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdleveningdojistar/cdleveningdojistar.c#L16-L106) | 3 | BodyDoji, BodyLong, BodyShort | `max(max(BodyDoji,BodyLong),BodyShort)+2` → 12 | −100; P=.3; T |
| [`CDLEVENINGSTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdleveningstar/cdleveningstar.c#L16-L99) | 3 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+2` → 12 | −100; P=.3; T |
| [`CDLGAPSIDESIDEWHITE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlgapsidesidewhite/cdlgapsidesidewhite.c#L16-L98) | 3 | Equal, Near | `max(Near,Equal)+2` → 7 | gap up +100 / down −100; T |
| [`CDLGRAVESTONEDOJI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlgravestonedoji/cdlgravestonedoji.c#L16-L86) | 1 | BodyDoji, ShadowVeryShort | `max(BodyDoji,ShadowVeryShort)` → 10 | +100 |
| [`CDLHAMMER`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhammer/cdlhammer.c#L16-L105) | 2 | BodyShort, Near, ShadowLong, ShadowVeryShort | `max(max(max(BodyShort,ShadowLong),ShadowVeryShort),Near)+1` → 11 | +100; T |
| [`CDLHANGINGMAN`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhangingman/cdlhangingman.c#L16-L105) | 2 | BodyShort, Near, ShadowLong, ShadowVeryShort | `max(max(max(BodyShort,ShadowLong),ShadowVeryShort),Near)+1` → 11 | −100; T |
| [`CDLHARAMI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlharami/cdlharami.c#L18-L105) | 2 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+1` → 11 | opposite first ×80/100; T |
| [`CDLHARAMICROSS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlharamicross/cdlharamicross.c#L18-L104) | 2 | BodyDoji, BodyLong | `max(BodyDoji,BodyLong)+1` → 11 | opposite first ×80/100; T |
| [`CDLHIGHWAVE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhighwave/cdlhighwave.c#L16-L83) | 1 | BodyShort, ShadowVeryLong | `max(BodyShort,ShadowVeryLong)` → 10 | current color ×100 |
| [`CDLHIKKAKE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkake/cdlhikkake.c#L16-L115) | 3 + confirm≤3 | — | `5` → 5 | formation ±100; confirmation ±200 |
| [`CDLHIKKAKEMOD`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkakemod/cdlhikkakemod.c#L16-L147) | 4 + confirm≤3 | Near | `max(1,Near)+5` → 10 | formation ±100; confirmation ±200 |
| [`CDLHOMINGPIGEON`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhomingpigeon/cdlhomingpigeon.c#L15-L89) | 2 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+1` → 11 | +100; T |
| [`CDLIDENTICAL3CROWS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlidentical3crows/cdlidentical3crows.c#L16-L112) | 3 | Equal, ShadowVeryShort | `max(ShadowVeryShort,Equal)+2` → 12 | −100; T |
| [`CDLINNECK`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlinneck/cdlinneck.c#L16-L91) | 2 | BodyLong, Equal | `max(Equal,BodyLong)+1` → 11 | −100; T |
| [`CDLINVERTEDHAMMER`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlinvertedhammer/cdlinvertedhammer.c#L16-L96) | 2 | BodyShort, ShadowLong, ShadowVeryShort | `max(max(BodyShort,ShadowLong),ShadowVeryShort)+1` → 11 | +100; T |
| [`CDLKICKING`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlkicking/cdlkicking.c#L16-L104) | 2 | BodyLong, ShadowVeryShort | `max(ShadowVeryShort,BodyLong)+1` → 11 | final color ×100 |
| [`CDLKICKINGBYLENGTH`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlkickingbylength/cdlkickingbylength.c#L16-L105) | 2 | BodyLong, ShadowVeryShort | `max(ShadowVeryShort,BodyLong)+1` → 11 | longer body color ×100; tie→first |
| [`CDLLADDERBOTTOM`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlladderbottom/cdlladderbottom.c#L16-L87) | 5 | ShadowVeryShort | `ShadowVeryShort+4` → 14 | +100; T |
| [`CDLLONGLEGGEDDOJI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdllongleggeddoji/cdllongleggeddoji.c#L16-L86) | 1 | BodyDoji, ShadowLong | `max(BodyDoji,ShadowLong)` → 10 | +100 |
| [`CDLLONGLINE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdllongline/cdllongline.c#L16-L83) | 1 | BodyLong, ShadowShort | `max(BodyLong,ShadowShort)` → 10 | current color ×100 |
| [`CDLMARUBOZU`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmarubozu/cdlmarubozu.c#L16-L82) | 1 | BodyLong, ShadowVeryShort | `max(BodyLong,ShadowVeryShort)` → 10 | current color ×100 |
| [`CDLMATCHINGLOW`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmatchinglow/cdlmatchinglow.c#L16-L78) | 2 | Equal | `Equal+1` → 6 | +100 |
| [`CDLMATHOLD`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmathold/cdlmathold.c#L16-L120) | 5 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+4` → 14 | +100; P=.5 |
| [`CDLMORNINGDOJISTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmorningdojistar/cdlmorningdojistar.c#L16-L104) | 3 | BodyDoji, BodyLong, BodyShort | `max(max(BodyDoji,BodyLong),BodyShort)+2` → 12 | +100; P=.3; T |
| [`CDLMORNINGSTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlmorningstar/cdlmorningstar.c#L16-L97) | 3 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+2` → 12 | +100; P=.3; T |
| [`CDLONNECK`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlonneck/cdlonneck.c#L16-L90) | 2 | BodyLong, Equal | `max(Equal,BodyLong)+1` → 11 | −100; T |
| [`CDLPIERCING`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlpiercing/cdlpiercing.c#L16-L87) | 2 | BodyLong | `BodyLong+1` → 11 | +100; fixed >50%; T |
| [`CDLRICKSHAWMAN`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlrickshawman/cdlrickshawman.c#L16-L102) | 1 | BodyDoji, Near, ShadowLong | `max(max(BodyDoji,ShadowLong),Near)` → 10 | +100 |
| [`CDLRISEFALL3METHODS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlrisefall3methods/cdlrisefall3methods.c#L16-L114) | 5 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+4` → 14 | first color ×100 |
| [`CDLSEPARATINGLINES`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlseparatinglines/cdlseparatinglines.c#L16-L105) | 2 | BodyLong, Equal, ShadowVeryShort | `max(max(ShadowVeryShort,BodyLong),Equal)+1` → 11 | second color ×100; T |
| [`CDLSHOOTINGSTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlshootingstar/cdlshootingstar.c#L16-L96) | 2 | BodyShort, ShadowLong, ShadowVeryShort | `max(max(BodyShort,ShadowLong),ShadowVeryShort)+1` → 11 | −100; T |
| [`CDLSHORTLINE`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlshortline/cdlshortline.c#L16-L83) | 1 | BodyShort, ShadowShort | `max(BodyShort,ShadowShort)` → 10 | current color ×100 |
| [`CDLSPINNINGTOP`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlspinningtop/cdlspinningtop.c#L16-L77) | 1 | BodyShort | `BodyShort` → 10 | current color ×100 |
| [`CDLSTALLEDPATTERN`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlstalledpattern/cdlstalledpattern.c#L16-L126) | 3 | BodyLong, BodyShort, Near, ShadowVeryShort | `max(max(BodyLong,BodyShort),max(ShadowVeryShort,Near))+2` → 12 | −100; T |
| [`CDLSTICKSANDWICH`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlsticksandwich/cdlsticksandwich.c#L16-L83) | 3 | Equal | `Equal+2` → 7 | +100; T |
| [`CDLTAKURI`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdltakuri/cdltakuri.c#L16-L96) | 1 | BodyDoji, ShadowVeryLong, ShadowVeryShort | `max(max(BodyDoji,ShadowVeryShort),ShadowVeryLong)` → 10 | +100 |
| [`CDLTASUKIGAP`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdltasukigap/cdltasukigap.c#L16-L100) | 3 | Near | `Near+2` → 7 | second (i−1) color ×100 |
| [`CDLTHRUSTING`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlthrusting/cdlthrusting.c#L16-L92) | 2 | BodyLong, Equal | `max(Equal,BodyLong)+1` → 11 | −100; T |
| [`CDLTRISTAR`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdltristar/cdltristar.c#L16-L89) | 3 | BodyDoji | `BodyDoji+2` → 12 | gap up −100 / down +100 |
| [`CDLUNIQUE3RIVER`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlunique3river/cdlunique3river.c#L16-L92) | 3 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+2` → 12 | +100; T |
| [`CDLUPSIDEGAP2CROWS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlupsidegap2crows/cdlupsidegap2crows.c#L16-L94) | 3 | BodyLong, BodyShort | `max(BodyShort,BodyLong)+2` → 12 | −100; T |
| [`CDLXSIDEGAP3METHODS`](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlxsidegap3methods/cdlxsidegap3methods.c#L16-L83) | 3 | — | `2` → 2 | first (i−2) color ×100; T |

## Compatibility-critical exception groups

- **Magnitude:** full nonzero domain is `{-200,-100,-80,80,100,200}`. Engulfing, Harami, Harami Cross use 80 when bodies meet at one end and 100 for strict containment ([Engulfing](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlengulfing/cdlengulfing.c#L52-L87), [Harami](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlharami/cdlharami.c#L68-L104), [Cross](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlharamicross/cdlharamicross.c#L68-L103)). Hikkake variants use ±100 formation and ±200 confirmation ([Hikkake](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkake/cdlhikkake.c#L77-L112), [modified](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkakemod/cdlhikkakemod.c#L95-L142)).
- **State:** only Hikkake variants retain pending formation, reconstruct it before a requested range, confirm within three later bars, and give a new same-bar pattern precedence over old confirmation ([Hikkake](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkake/cdlhikkake.c#L46-L86), [modified](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkakemod/cdlhikkakemod.c#L47-L109)). Batch/range and streaming must preserve/reconstruct state.
- **Color/gaps:** `close >= open` is +1, so exact doji is white when color is consulted. Body/candle gaps use strict `>`/`<`; touching is not a gap ([helpers](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/helpers/candlestick.c#L1-L35)).
- **Literal boundaries:** long generally uses strict `>`, while short/doji/equality use pattern-specific `<`, `<=`, `>=`, or exact equality. In-Neck, for example, includes second closes from prior close through prior close plus `Equal` ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlinneck/cdlinneck.c#L67-L90)). Source predicates own boundaries.
- **Tristar:** all three bodies use one rolling `BodyDoji` sum and `i-2` threshold arguments, advanced using `i-2`; this is shipped observable behavior ([template](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdltristar/cdltristar.c#L49-L95), [generated](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/src/ta_func/ta_CDLTRISTAR.c#L123-L168)).
- **Setting-free:** 3 Outside, Engulfing, Hikkake, X-Side Gap use no settings and fixed lookbacks 3, 2, 5, 2 ([3 Outside](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3outside/cdl3outside.c#L16-L80), [Engulfing](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlengulfing/cdlengulfing.c#L18-L90), [Hikkake](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlhikkake/cdlhikkake.c#L16-L115), [X-Side](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlxsidegap3methods/cdlxsidegap3methods.c#L16-L83)).
- **Span/sign surprises:** 3 Black Crows reads a fourth preceding white candle ([source](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdl3blackcrows/cdl3blackcrows.c#L62-L92)). Sign is not uniformly current-bar color: Gap Side-Side White and Tristar use gap direction; Kicking by Length uses longer body, equal length selects first ([sources](https://github.com/TA-Lib/ta-lib/blob/2247d599bddf37ed37e3a709371517e46efc66f6/ta_codegen/input/cdlkickingbylength/cdlkickingbylength.c#L72-L104)). `T` rows do not add their stated trend qualification.

## Upstream facts versus later decisions

| Upstream fact | Still a fast-ta decision |
|---|---|
| Pin v0.7.1 `2247d599...`. | Whether/when a later release replaces it. |
| 11 global settings; period changes lookback. | Fixed vs custom settings and ownership/concurrency. |
| Setter leaves range/period/factor unchecked. | Supported custom domain and validation. |
| Seven penetrations through `TA_REAL_MAX`, defaults .3/.5. | Whether a high-level surface narrows to a fraction. |
| Outputs include ±80/±200 and pattern-specific sign. | Raw integer vs another lossless representation. |
| Hikkake needs pre-roll/pending state. | Batch/streaming state representation. |
| Exact comparisons, fixed lookbacks, Tristar are observable. | Bit-for-bit compatibility vs explicit divergence. |
| `T` detectors omit trend qualification. | Whether a separately named context layer exists. |

## Newly precise wayfinder tickets and fog patches

1. **Candle Settings compatibility/scope:** configurable triples, validation, ownership, concurrency, dynamic lookback.
2. **Lossless signal:** zero, ±80, ±100, ±200, pattern-specific sign source.
3. **Seven penetration contracts:** per-function defaults, >1.0 policy, Piercing separate.
4. **Hikkake batch/streaming equivalence:** pre-roll, cross-chunk pending state, expiry, same-bar precedence.
5. **Exact compatibility vs divergence:** close-equals-open, strict gaps, ±80 ties, Kicking tie, fixed lookbacks, Tristar threshold.
6. **Detector vs trend qualification:** raw 61 unchanged or separately named context layer.
7. **Fog—unsupported settings:** upstream stores negative periods/factors and unknown ranges without useful promised semantics; name supported domain.
8. **Fog—future revisions:** changing the pin requires source-diff decisions for predicates, lookbacks, and state.
