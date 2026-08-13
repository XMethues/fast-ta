#!/usr/bin/env python3
"""Regenerate the independent Decimal KAMA/MAVP reference values."""

from decimal import Decimal, getcontext

getcontext().prec = 50
INPUT = [
    Decimal(value)
    for value in (
        "10", "10.5", "10.25", "11", "10.75", "12", "11.5", "13", "12.75", "14",
        "13", "15", "14.5", "16", "15.25", "17", "16.5", "18", "17.25", "19",
    )
]
SELECTIONS = [1, 2, 3, 4, 5, 0, 2, 9, 4, 3, 5, 2, 4, 8, 1, 3, 5, 4, 2, 0]


def kama(values, period):
    volatility = sum(abs(values[i] - values[i - 1]) for i in range(1, period + 1))
    previous = values[period - 1]
    slow = Decimal(2) / Decimal(31)
    difference = Decimal(2) / Decimal(3) - slow
    result = []
    for index in range(period, len(values)):
        if index > period:
            volatility -= abs(values[index - period] - values[index - period - 1])
            volatility += abs(values[index] - values[index - 1])
        change = values[index] - values[index - period]
        efficiency = Decimal(1) if volatility == 0 or abs(change) >= volatility else abs(change / volatility)
        smoothing = (efficiency * difference + slow) ** 2
        previous += (values[index] - previous) * smoothing
        result.append(previous)
    return result


def ema_from_global_lookback(values, period, global_lookback):
    start = global_lookback - (period - 1)
    source = values[start:]
    alpha = Decimal(2) / Decimal(period + 1)
    value = sum(source[:period]) / Decimal(period)
    result = {global_lookback: value}
    for index in range(global_lookback + 1, len(values)):
        value += (values[index] - value) * alpha
        result[index] = value
    return result


GLOBAL_LOOKBACK = 4
emas = {
    period: ema_from_global_lookback(INPUT, period, GLOBAL_LOOKBACK)
    for period in range(2, 6)
}
mavp = [
    emas[max(2, min(5, SELECTIONS[index]))][index]
    for index in range(GLOBAL_LOOKBACK, len(INPUT))
]
for name, values in (("KAMA_EXPECTED", kama(INPUT, 5)), ("MAVP_EMA_EXPECTED", mavp)):
    print(f"pub const {name}: &[f64] = &[")
    for value in values:
        print(f"    {float(value):.15f},")
    print("];\n")
