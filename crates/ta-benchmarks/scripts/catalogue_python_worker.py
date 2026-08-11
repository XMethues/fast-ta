#!/usr/bin/env python3
"""Semantic and timed official Python TA-Lib calls for the catalogue matrix."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import timeit

import numpy as np
import talib
from talib import abstract


EXPRESSIONS = {
    "SMA": "talib.SMA(close, timeperiod=14)",
    "BBANDS": "talib.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)",
    "RSI": "talib.RSI(close, timeperiod=14)",
    "MACD": "talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)",
    "ATR": "talib.ATR(high, low, close, timeperiod=14)",
    "ADX": "talib.ADX(high, low, close, timeperiod=14)",
    "HT_DCPHASE": "talib.HT_DCPHASE(close)",
    "CDLDOJI": "talib.CDLDOJI(open_, high, low, close)",
    "CDLENGULFING": "talib.CDLENGULFING(open_, high, low, close)",
    "CDL3WHITESOLDIERS": "talib.CDL3WHITESOLDIERS(open_, high, low, close)",
    "LINEARREG": "talib.LINEARREG(close, timeperiod=14)",
    "TYPPRICE": "talib.TYPPRICE(high, low, close)",
    "OBV": "talib.OBV(close, volume)",
    "SIN": "talib.SIN(close)",
    "ADD": "talib.ADD(close, auxiliary)",
}

PARAMETERS = {
    "SMA": {"timeperiod": 14},
    "BBANDS": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0, "matype": 0},
    "RSI": {"timeperiod": 14},
    "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "ATR": {"timeperiod": 14},
    "ADX": {"timeperiod": 14},
    "LINEARREG": {"timeperiod": 14},
}

INTEGER_CASES = {"CDLDOJI", "CDLENGULFING", "CDL3WHITESOLDIERS"}


def version_text(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return str(value).split()[0]


def load_column(directory: Path, name: str) -> np.ndarray:
    values = np.fromfile(directory / f"{name}.f64le.bin", dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"shared {name} input must be a finite one-dimensional f64 array")
    return values


def load_fixture(directory: Path) -> dict[str, object]:
    fixture: dict[str, object] = {
        "talib": talib,
        "open_": load_column(directory, "open"),
        "high": load_column(directory, "high"),
        "low": load_column(directory, "low"),
        "close": load_column(directory, "close"),
        "volume": load_column(directory, "volume"),
        "auxiliary": load_column(directory, "auxiliary"),
    }
    lengths = {len(value) for key, value in fixture.items() if key != "talib"}
    if len(lengths) != 1:
        raise RuntimeError(f"shared fixture columns have unequal lengths: {sorted(lengths)}")
    open_ = fixture["open_"]
    high = fixture["high"]
    low = fixture["low"]
    close = fixture["close"]
    volume = fixture["volume"]
    assert isinstance(open_, np.ndarray)
    assert isinstance(high, np.ndarray)
    assert isinstance(low, np.ndarray)
    assert isinstance(close, np.ndarray)
    assert isinstance(volume, np.ndarray)
    if not ((high >= np.maximum(open_, close)).all() and (low <= np.minimum(open_, close)).all()):
        raise RuntimeError("shared fixture violates OHLC invariants")
    if (volume < 0.0).any():
        raise RuntimeError("shared fixture contains negative volume")
    return fixture


def emit_common() -> None:
    print(f"python_version={sys.version.split()[0]}")
    print(f"numpy_version={np.__version__}")
    print(f"python_binding_version={talib.__version__}")
    print(f"python_ta_lib_version={version_text(talib.__ta_version__)}")


def lookback(case_id: str) -> int:
    function = abstract.Function(case_id)
    parameters = PARAMETERS.get(case_id)
    if parameters:
        function.parameters = parameters
    return int(function.lookback)


def evaluate(case_id: str, fixture: dict[str, object]) -> object:
    if case_id not in EXPRESSIONS:
        raise RuntimeError(f"unknown matrix case {case_id!r}")
    return eval(EXPRESSIONS[case_id], {"__builtins__": {}}, fixture)


def semantic(fixture_dir: Path, output_dir: Path, case_id: str) -> None:
    fixture = load_fixture(fixture_dir)
    output = evaluate(case_id, fixture)
    columns = output if isinstance(output, tuple) else (output,)
    expected_arity = 3 if case_id in {"BBANDS", "MACD"} else 1
    if len(columns) != expected_arity:
        raise RuntimeError(f"{case_id} returned {len(columns)} output columns, expected {expected_arity}")
    begin = lookback(case_id)
    source_length = len(fixture["close"])
    count = max(0, source_length - begin)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, raw_column in enumerate(columns):
        column = np.asarray(raw_column)
        if column.ndim != 1 or column.size != source_length:
            raise RuntimeError(f"{case_id} output column {index} has unexpected shape {column.shape}")
        if case_id in INTEGER_CASES:
            if not np.issubdtype(column.dtype, np.integer):
                raise RuntimeError(f"{case_id} output column {index} is not integer-valued")
            if begin and (column[:begin] != 0).any():
                raise RuntimeError(f"{case_id} output column {index} has nonzero unavailable-prefix signals")
            compact = column[begin:].astype("<i4", copy=False)
            compact.tofile(output_dir / f"column-{index}.i32le.bin")
        else:
            float_column = column.astype(np.float64, copy=False)
            if begin and not np.isnan(float_column[:begin]).all():
                raise RuntimeError(f"{case_id} output column {index} has unexpected unavailable-value placement")
            if not np.isfinite(float_column[begin:]).all():
                raise RuntimeError(f"{case_id} output column {index} contains a non-finite compact value")
            float_column[begin:].astype("<f8", copy=False).tofile(
                output_dir / f"column-{index}.f64le.bin"
            )
    emit_common()
    print(f"output_kind={'integer' if case_id in INTEGER_CASES else 'float'}")
    print(f"output_arity={len(columns)}")
    print(f"output_begin={begin if count else 0}")
    print(f"output_count={count}")


def timing(
    fixture_dir: Path,
    case_id: str,
    sample_count: int,
    warmup_ms: int,
    sample_ms: int,
) -> None:
    if sample_count < 2 or warmup_ms <= 0 or sample_ms <= 0:
        raise RuntimeError("samples must be at least 2 and timing durations must be positive")
    fixture = load_fixture(fixture_dir)
    if case_id not in EXPRESSIONS:
        raise RuntimeError(f"unknown matrix case {case_id!r}")
    timer = timeit.Timer(EXPRESSIONS[case_id], globals=fixture)

    warmup_deadline = time.perf_counter_ns() + warmup_ms * 1_000_000
    warmup_iterations = 0
    while time.perf_counter_ns() < warmup_deadline:
        timer.timeit(number=1)
        warmup_iterations += 1

    calibration_ns = max(1, math.ceil(timer.timeit(number=1) * 1_000_000_000))
    iterations_per_sample = min(
        1_000_000,
        max(1, math.ceil(sample_ms * 1_000_000 / calibration_ns)),
    )
    samples_ns = [
        timer.timeit(number=iterations_per_sample) * 1_000_000_000 / iterations_per_sample
        for _ in range(sample_count)
    ]
    print(f"warmup_iterations={warmup_iterations}")
    print(f"iterations_per_sample={iterations_per_sample}")
    print("samples_ns=" + ",".join(f"{sample:.9f}" for sample in samples_ns))


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("expected semantic or timing command")
    command = sys.argv[1]
    if command == "semantic" and len(sys.argv) == 5:
        semantic(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
    elif command == "timing" and len(sys.argv) == 7:
        timing(
            Path(sys.argv[2]),
            sys.argv[3],
            int(sys.argv[4]),
            int(sys.argv[5]),
            int(sys.argv[6]),
        )
    else:
        raise RuntimeError("invalid worker arguments")


if __name__ == "__main__":
    main()
