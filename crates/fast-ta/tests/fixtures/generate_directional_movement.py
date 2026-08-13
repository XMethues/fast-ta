#!/usr/bin/env python3
"""Generate Directional Movement reference vectors through pinned TA-Lib ctypes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Iterable, Sequence

TALIB_GIT_REVISION = "43f9d5042ecc4bd367941846494ad907bf20ea50"
TALIB_VERSION = "0.6.4"
TALIB_SOURCE_ARCHIVE_SHA256 = (
    "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
)
PERIOD = 5
SOURCE_LENGTH = 40
FUNCTIONS = ("PLUS_DM", "MINUS_DM", "PLUS_DI", "MINUS_DI", "DX", "ADX", "ADXR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def observations() -> tuple[list[float], list[float], list[float]]:
    close = [
        100.0 + (
            index * 0.8
            if index < 12
            else 9.6 - (index - 12) * 1.1
            if index < 24
            else -3.6 + (index - 24) * 0.45 + 2.0 * math.sin((index - 24) * 0.7)
        )
        for index in range(SOURCE_LENGTH)
    ]
    high = [value + 1.0 + (index % 3) * 0.2 for index, value in enumerate(close)]
    low = [value - 0.8 - ((index + 1) % 4) * 0.15 for index, value in enumerate(close)]
    return high, low, close


def build_pinned_talib(source_archive: Path, workspace: Path) -> Path:
    digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if digest != TALIB_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"TA-Lib archive checksum mismatch: expected {TALIB_SOURCE_ARCHIVE_SHA256}, got {digest}"
        )
    with tarfile.open(source_archive, "r:gz") as archive:
        archive.extractall(workspace)
    source = workspace / f"ta-lib-{TALIB_VERSION}"
    install = workspace / "install"
    subprocess.run(["./configure", f"--prefix={install}"], cwd=source, check=True)
    subprocess.run(["make", f"-j{max(1, os.cpu_count() or 1)}"], cwd=source, check=True)
    subprocess.run(["make", "install"], cwd=source, check=True)
    for name in ("libta-lib.dylib", "libta-lib.so"):
        library = install / "lib" / name
        if library.exists():
            return library
    raise RuntimeError("TA-Lib build did not produce a shared library")


class TalibReference:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.TA_Initialize.restype = ctypes.c_int
        self.library.TA_Shutdown.restype = ctypes.c_int
        self.library.TA_GetVersionString.restype = ctypes.c_char_p
        self._check(self.library.TA_Initialize(), "TA_Initialize")
        reported = self.library.TA_GetVersionString().decode("ascii").split()[0]
        if reported != TALIB_VERSION:
            raise RuntimeError(f"expected TA-Lib {TALIB_VERSION}, got {reported}")
        hlc_arguments = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        hl_arguments = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        for name in FUNCTIONS:
            function = getattr(self.library, f"TA_{name}")
            function.argtypes = hl_arguments if name in ("PLUS_DM", "MINUS_DM") else hlc_arguments
            function.restype = ctypes.c_int
            lookback = getattr(self.library, f"TA_{name}_Lookback")
            lookback.argtypes = [ctypes.c_int]
            lookback.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with TA-Lib code {code}")

    def compute(
        self, name: str, high: Sequence[float], low: Sequence[float], close: Sequence[float]
    ) -> tuple[int, list[float]]:
        high_array = (ctypes.c_double * len(high))(*high)
        low_array = (ctypes.c_double * len(low))(*low)
        close_array = (ctypes.c_double * len(close))(*close)
        output = (ctypes.c_double * len(close))()
        begin = ctypes.c_int()
        count = ctypes.c_int()
        function = getattr(self.library, f"TA_{name}")
        if name in ("PLUS_DM", "MINUS_DM"):
            code = function(
                0,
                len(close) - 1,
                high_array,
                low_array,
                PERIOD,
                ctypes.byref(begin),
                ctypes.byref(count),
                output,
            )
        else:
            code = function(
                0,
                len(close) - 1,
                high_array,
                low_array,
                close_array,
                PERIOD,
                ctypes.byref(begin),
                ctypes.byref(count),
                output,
            )
        self._check(code, f"TA_{name}")
        expected_begin = getattr(self.library, f"TA_{name}_Lookback")(PERIOD)
        if begin.value != expected_begin or count.value != len(close) - expected_begin:
            raise RuntimeError(f"{name}: unexpected output range {begin.value}+{count.value}")
        return begin.value, list(output[: count.value])

    def close(self) -> None:
        self._check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_f64(value: float) -> str:
    rendered = f"{value:.12f}".rstrip("0").rstrip(".")
    return rendered + ".0" if "." not in rendered else rendered


def rust_slice(values: Iterable[float]) -> str:
    return "\n".join(f"    {rust_f64(value)}," for value in values)


def render(reference: TalibReference) -> str:
    high, low, close = observations()
    outputs = {name: reference.compute(name, high, low, close)[1] for name in FUNCTIONS}
    constants = "\n".join(
        f"pub const {name}: &[f64] = &[\n{rust_slice(values)}\n];"
        for name, values in outputs.items()
    )
    return f'''// Generated by tests/fixtures/generate_directional_movement.py. Do not edit by hand.
// Reference source: TA-Lib v{TALIB_VERSION}, commit {TALIB_GIT_REVISION}.
// Source archive SHA-256: {TALIB_SOURCE_ARCHIVE_SHA256}.

pub const PERIOD: usize = {PERIOD};
pub const TALIB_VERSION: &str = "{TALIB_VERSION}";
pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";
pub const TALIB_SOURCE_ARCHIVE_SHA256: &str = "{TALIB_SOURCE_ARCHIVE_SHA256}";

pub const CLOSE: &[f64] = &[
{rust_slice(close)}
];
pub const HIGH: &[f64] = &[
{rust_slice(high)}
];
pub const LOW: &[f64] = &[
{rust_slice(low)}
];

{constants}
'''


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fast-ta-directional-") as directory:
        library = build_pinned_talib(args.source_archive, Path(directory))
        reference = TalibReference(library)
        try:
            content = render(reference)
        finally:
            reference.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
