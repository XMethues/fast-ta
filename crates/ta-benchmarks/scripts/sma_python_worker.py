#!/usr/bin/env python3
"""Semantic and timed official Python TA-Lib SMA calls for the Rust tracer."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import talib


def version_text(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return str(value).split()[0]


def load_input(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f8")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("shared input must be a finite one-dimensional f64 array")
    return values


def emit_common() -> None:
    print(f"python_version={sys.version.split()[0]}")
    print(f"numpy_version={np.__version__}")
    print(f"python_binding_version={talib.__version__}")
    print(f"python_ta_lib_version={version_text(talib.__ta_version__)}")


def semantic(input_path: Path, output_path: Path, period: int) -> None:
    values = load_input(input_path)
    output = talib.SMA(values, timeperiod=period)
    finite = np.flatnonzero(np.isfinite(output))
    if finite.size == 0:
        output_begin = 0
        compact = output[:0]
    else:
        output_begin = int(finite[0])
        if np.isfinite(output[:output_begin]).any() or not np.isfinite(output[output_begin:]).all():
            raise RuntimeError("Python SMA returned unexpected unavailable-value placement")
        compact = output[output_begin:]
    compact.astype("<f8", copy=False).tofile(output_path)
    emit_common()
    print(f"output_begin={output_begin}")
    print(f"output_count={compact.size}")


def timing(input_path: Path, period: int, sample_count: int, warmup_ms: int, sample_ms: int) -> None:
    if sample_count < 2 or warmup_ms <= 0 or sample_ms <= 0:
        raise RuntimeError("samples must be at least 2 and timing durations must be positive")
    values = load_input(input_path)

    warmup_deadline = time.perf_counter_ns() + warmup_ms * 1_000_000
    warmup_iterations = 0
    sink = None
    while time.perf_counter_ns() < warmup_deadline:
        sink = talib.SMA(values, timeperiod=period)
        warmup_iterations += 1

    calibration_start = time.perf_counter_ns()
    sink = talib.SMA(values, timeperiod=period)
    calibration_ns = max(1, time.perf_counter_ns() - calibration_start)
    iterations_per_sample = min(1_000_000, max(1, math.ceil(sample_ms * 1_000_000 / calibration_ns)))

    samples_ns: list[float] = []
    for _ in range(sample_count):
        started = time.perf_counter_ns()
        for _ in range(iterations_per_sample):
            sink = talib.SMA(values, timeperiod=period)
        samples_ns.append((time.perf_counter_ns() - started) / iterations_per_sample)

    if sink is None or sink.size != values.size:
        raise RuntimeError("Python timing call returned an unexpected output")
    print(f"warmup_iterations={warmup_iterations}")
    print(f"iterations_per_sample={iterations_per_sample}")
    print("samples_ns=" + ",".join(f"{sample:.9f}" for sample in samples_ns))


def main() -> None:
    if len(sys.argv) < 2:
        raise RuntimeError("expected semantic or timing command")
    command = sys.argv[1]
    if command == "semantic" and len(sys.argv) == 5:
        semantic(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
    elif command == "timing" and len(sys.argv) == 7:
        timing(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        raise RuntimeError("invalid worker arguments")


if __name__ == "__main__":
    main()
