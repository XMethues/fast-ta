#!/usr/bin/env python3
"""Regenerate issue #26 reference vectors with the official TA-Lib Python binding.

Requires `numpy` and `TA-Lib`. The source definitions are audited against:
https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ACCBANDS.c
https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_BBANDS.c
https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPOINT.c
https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_MIDPRICE.c
"""

import numpy as np
import talib

PERIOD = 3
REAL = np.array([10.0, 11.0, 9.0, 12.0, 13.0, 8.0, 14.0, 15.0, 11.0, 16.0])
HIGH = np.array([11.0, 12.0, 10.0, 13.0, 14.0, 9.0, 15.0, 16.0, 12.0, 17.0])
LOW = np.array([9.0, 10.0, 8.0, 11.0, 12.0, 7.0, 13.0, 14.0, 10.0, 15.0])


def compact(values: np.ndarray) -> list[float]:
    return values[~np.isnan(values)].tolist()


acc_upper, acc_middle, acc_lower = talib.ACCBANDS(HIGH, LOW, REAL, timeperiod=PERIOD)
bb_upper, bb_middle, bb_lower = talib.BBANDS(
    REAL,
    timeperiod=PERIOD,
    nbdevup=2.0,
    nbdevdn=2.0,
    matype=talib.MA_Type.SMA,
)

for name, values in (
    ("ACCBANDS_UPPER", acc_upper),
    ("ACCBANDS_MIDDLE", acc_middle),
    ("ACCBANDS_LOWER", acc_lower),
    ("BBANDS_UPPER", bb_upper),
    ("BBANDS_MIDDLE", bb_middle),
    ("BBANDS_LOWER", bb_lower),
    ("MIDPOINT", talib.MIDPOINT(REAL, timeperiod=PERIOD)),
    ("MIDPRICE", talib.MIDPRICE(HIGH, LOW, timeperiod=PERIOD)),
):
    print(f"{name} = {compact(values)!r}")
