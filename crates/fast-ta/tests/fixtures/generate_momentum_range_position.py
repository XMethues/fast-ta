#!/usr/bin/env python3
"""Regenerate range-position Momentum decimal reference vectors.

The formulas and operation ordering follow TA-Lib 0.8.1 at commit
`e64d2ac896c595f38d65e44c812efbfdac8a64cf`: `ta_AROON.c`,
`ta_AROONOSC.c`, `ta_STOCH.c`, `ta_STOCHF.c`, `ta_STOCHRSI.c`, and
`ta_WILLR.c`. Decimal arithmetic keeps the generator independent of Rust and
makes the checked-in ordinary-market vectors deterministic.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 50
TALIB_VERSION = "0.8.1"
TALIB_GIT_REVISION = "e64d2ac896c595f38d65e44c812efbfdac8a64cf"
PERIOD = 5
SMOOTHING_PERIOD = 3
CLOSE_TEXT = (
    "44", "44.34", "44.09", "44.15", "43.61", "44.33", "44.83", "45.10",
    "45.42", "45.84", "46.08", "45.89", "46.03", "45.61", "46.28", "46.50",
    "46.31", "46.78", "47.02", "46.71", "47.25", "47.58", "47.40", "47.91",
    "48.15", "47.88", "48.36", "48.72", "48.51", "49.05",
)
HIGH_OFFSETS = ("0.5", "0.35", "0.65", "0.4", "0.55")
LOW_OFFSETS = ("0.4", "0.6", "0.3", "0.5", "0.45")
CLOSE = tuple(Decimal(value) for value in CLOSE_TEXT)
HIGH = tuple(
    value + Decimal(HIGH_OFFSETS[index % len(HIGH_OFFSETS)])
    for index, value in enumerate(CLOSE)
)
LOW = tuple(
    value - Decimal(LOW_OFFSETS[index % len(LOW_OFFSETS)])
    for index, value in enumerate(CLOSE)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("momentum_range_position_reference.rs"),
        help="Rust fixture file to create",
    )
    return parser.parse_args()


def aroon() -> tuple[list[Decimal], list[Decimal]]:
    down: list[Decimal] = []
    up: list[Decimal] = []
    for end in range(PERIOD, len(CLOSE)):
        start = end - PERIOD
        window_low = LOW[start:end + 1]
        window_high = HIGH[start:end + 1]
        lowest = min(window_low)
        highest = max(window_high)
        # AROON refreshes equal extrema, so the greatest matching source index wins.
        lowest_index = max(index for index in range(start, end + 1) if LOW[index] == lowest)
        highest_index = max(index for index in range(start, end + 1) if HIGH[index] == highest)
        down.append(Decimal(100) * (PERIOD - (end - lowest_index)) / PERIOD)
        up.append(Decimal(100) * (PERIOD - (end - highest_index)) / PERIOD)
    return down, up


def stochastic(high: tuple[Decimal, ...], low: tuple[Decimal, ...], close: tuple[Decimal, ...], period: int) -> list[Decimal]:
    values: list[Decimal] = []
    for end in range(period - 1, len(close)):
        lowest = min(low[end - period + 1:end + 1])
        highest = max(high[end - period + 1:end + 1])
        denominator = highest - lowest
        values.append(
            Decimal(0)
            if denominator == 0
            else Decimal(100) * (close[end] - lowest) / denominator
        )
    return values


def sma(values: list[Decimal], period: int) -> list[Decimal]:
    return [
        sum(values[end - period + 1:end + 1]) / period
        for end in range(period - 1, len(values))
    ]


def rsi(values: tuple[Decimal, ...], period: int) -> list[Decimal]:
    movements: list[tuple[Decimal, Decimal]] = []
    for previous, current in zip(values, values[1:]):
        movement = current - previous
        movements.append(
            (movement, Decimal(0)) if movement >= 0 else (Decimal(0), -movement)
        )
    gain = sum(value[0] for value in movements[:period]) / period
    loss = sum(value[1] for value in movements[:period]) / period
    output: list[Decimal] = []
    for index in range(period, len(movements) + 1):
        denominator = gain + loss
        output.append(Decimal(0) if denominator == 0 else Decimal(100) * gain / denominator)
        if index < len(movements):
            incoming_gain, incoming_loss = movements[index]
            gain = (gain * (period - 1) + incoming_gain) / period
            loss = (loss * (period - 1) + incoming_loss) / period
    return output


def willr() -> list[Decimal]:
    output: list[Decimal] = []
    for end in range(PERIOD - 1, len(CLOSE)):
        lowest = min(LOW[end - PERIOD + 1:end + 1])
        highest = max(HIGH[end - PERIOD + 1:end + 1])
        denominator = highest - lowest
        output.append(
            Decimal(0)
            if denominator == 0
            else -Decimal(100) * (highest - CLOSE[end]) / denominator
        )
    return output


def rust_decimal(value: Decimal) -> str:
    rendered = format(value, ".17g")
    return rendered if "." in rendered or "e" in rendered.lower() else f"{rendered}.0"


def rust_slice(name: str, values: tuple[Decimal, ...] | list[Decimal]) -> str:
    body = "\n".join(f"    {rust_decimal(value)}," for value in values)
    return f"pub const {name}: &[f64] = &[\n{body}\n];"


def render_fixture() -> str:
    down, up = aroon()
    raw_k = stochastic(HIGH, LOW, CLOSE, PERIOD)
    fast_d = sma(raw_k, SMOOTHING_PERIOD)
    fast_k = raw_k[SMOOTHING_PERIOD - 1:]
    smoothed_k = sma(raw_k, SMOOTHING_PERIOD)
    slow_d = sma(smoothed_k, SMOOTHING_PERIOD)
    slow_k = smoothed_k[SMOOTHING_PERIOD - 1:]
    rsi_values = rsi(CLOSE, PERIOD)
    stoch_rsi_raw = stochastic(tuple(rsi_values), tuple(rsi_values), tuple(rsi_values), PERIOD)
    stoch_rsi_d = sma(stoch_rsi_raw, SMOOTHING_PERIOD)
    stoch_rsi_k = stoch_rsi_raw[SMOOTHING_PERIOD - 1:]
    parts = [
        "// Generated by tests/fixtures/generate_momentum_range_position.py. Do not edit by hand.",
        f"// Reference source: TA-Lib {TALIB_VERSION}, commit {TALIB_GIT_REVISION}.",
        "// Definitions: ta_AROON.c, ta_AROONOSC.c, ta_STOCH.c, ta_STOCHF.c, ta_STOCHRSI.c, ta_WILLR.c.",
        "",
        f'pub const TALIB_VERSION: &str = "{TALIB_VERSION}";',
        f'pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";',
        f"pub const PERIOD: usize = {PERIOD};",
        f"pub const SMOOTHING_PERIOD: usize = {SMOOTHING_PERIOD};",
        "",
        rust_slice("HIGH", HIGH),
        rust_slice("LOW", LOW),
        rust_slice("CLOSE", CLOSE),
        rust_slice("AROON_DOWN_EXPECTED", down),
        rust_slice("AROON_UP_EXPECTED", up),
        rust_slice("AROONOSC_EXPECTED", [up_value - down_value for down_value, up_value in zip(down, up)]),
        rust_slice("STOCH_SLOW_K_EXPECTED", slow_k),
        rust_slice("STOCH_SLOW_D_EXPECTED", slow_d),
        rust_slice("STOCHF_FAST_K_EXPECTED", fast_k),
        rust_slice("STOCHF_FAST_D_EXPECTED", fast_d),
        rust_slice("STOCHRSI_FAST_K_EXPECTED", stoch_rsi_k),
        rust_slice("STOCHRSI_FAST_D_EXPECTED", stoch_rsi_d),
        rust_slice("WILLR_EXPECTED", willr()),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_fixture(), encoding="utf-8")


if __name__ == "__main__":
    main()
