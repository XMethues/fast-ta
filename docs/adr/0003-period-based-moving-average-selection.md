# Restrict generic moving-average selection to Period-based definitions

The former `MAType` mirrored TA-Lib selector ids, including unimplemented `KAMA` and non-Period-based `MAMA`, so callers could construct configurations that failed only at execution and a future `MAMA` branch could not use the supplied Period or single-output shape truthfully. We replace it without an alias with `PeriodMAType`, closed over implemented, single-output Period-based Moving Average definitions, so every accepted `MAConfig` uses its Period meaningfully.

## Status

Accepted.

## Considered options

- Retain `MAType` and reject unsupported variants at runtime. Rejected because it represents configurations that cannot execute and leaves the Interface semantically incoherent.
- Expand one generic selector into variant-specific configuration and output sum types. Rejected because every moving-average consumer would inherit configuration and output complexity it does not use.
- Restrict and rename the selector. Accepted because invalid configurations become unrepresentable while the existing generic execution seam stays small.

## Consequences

- `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, default-vfactor `T3`, and the now-qualified `KAMA` provide owned, caller-owned, prepared, and streaming execution through `MAConfig`.
- `MAMA` never joins `PeriodMAType`; it requires its own configuration and paired MAMA/FAMA Compact Output.
- `MAVP` and later moving-average consumers can accept `PeriodMAType` knowing every selected definition is implemented, single-output, and governed by each supplied per-observation Period.
- Existing `MAType` callers must migrate to `PeriodMAType`; no compatibility alias or deprecated path is retained.
