#!/usr/bin/env python3
"""Generate APO/PPO/MACD/MACDEXT/MACDFIX/TRIX references from pinned TA-Lib."""

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

TALIB_GIT_REVISION = "43f9d5042ecc4bd367941846494ad907bf20ea50"
TALIB_VERSION = "0.6.4"
TALIB_SOURCE_ARCHIVE_SHA256 = (
    "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
)
FAST_PERIOD = 5
SLOW_PERIOD = 11
SIGNAL_PERIOD = 4
TRIX_PERIOD = 5
EMA = 1
PREFIX_LENGTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def observations() -> list[float]:
    return [
        100.0
        + 0.37 * index
        + 2.4 * math.sin(index * 0.41)
        - 1.1 * math.cos(index * 0.17)
        + ((index % 7) - 3) * 0.08
        for index in range(40)
    ]


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
    for library in (install / "lib").glob("libta-lib.*"):
        if library.suffix in (".dylib", ".so") or ".so." in library.name:
            return library
    raise RuntimeError("TA-Lib build did not produce a shared library")


class Reference:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.TA_Initialize.restype = ctypes.c_int
        self.library.TA_Shutdown.restype = ctypes.c_int
        self.library.TA_GetVersionString.restype = ctypes.c_char_p
        self.check(self.library.TA_Initialize(), "TA_Initialize")
        reported = self.library.TA_GetVersionString().decode("ascii").split()[0]
        if reported != TALIB_VERSION:
            raise RuntimeError(f"expected TA-Lib {TALIB_VERSION}, got {reported}")

    @staticmethod
    def check(code: int, name: str) -> None:
        if code != 0:
            raise RuntimeError(f"{name} returned TA_RetCode {code}")

    def one(self, name: str, values: list[float], arguments: list[int]) -> list[float]:
        length = len(values)
        source = (ctypes.c_double * length)(*values)
        output = (ctypes.c_double * length)()
        begin = ctypes.c_int()
        count = ctypes.c_int()
        function = getattr(self.library, f"TA_{name}")
        code = function(
            ctypes.c_int(0),
            ctypes.c_int(length - 1),
            source,
            *(ctypes.c_int(argument) for argument in arguments),
            ctypes.byref(begin),
            ctypes.byref(count),
            output,
        )
        self.check(code, name)
        expected_lookback = max(FAST_PERIOD, SLOW_PERIOD) - 1 if name in ("APO", "PPO") else 3 * (TRIX_PERIOD - 1) + 1
        if begin.value != expected_lookback:
            raise RuntimeError(f"{name} began at {begin.value}, expected {expected_lookback}")
        return [output[index] for index in range(count.value)]

    def three(self, name: str, values: list[float], arguments: list[int]) -> tuple[list[float], ...]:
        length = len(values)
        source = (ctypes.c_double * length)(*values)
        outputs = [(ctypes.c_double * length)() for _ in range(3)]
        begin = ctypes.c_int()
        count = ctypes.c_int()
        function = getattr(self.library, f"TA_{name}")
        code = function(
            ctypes.c_int(0),
            ctypes.c_int(length - 1),
            source,
            *(ctypes.c_int(argument) for argument in arguments),
            ctypes.byref(begin),
            ctypes.byref(count),
            *outputs,
        )
        self.check(code, name)
        expected_lookback = SLOW_PERIOD - 1 + SIGNAL_PERIOD - 1
        if begin.value != expected_lookback:
            raise RuntimeError(f"{name} began at {begin.value}, expected {expected_lookback}")
        return tuple(
            [output[index] for index in range(count.value)] for output in outputs
        )

    def close(self) -> None:
        self.check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_array(name: str, values: list[float]) -> str:
    body = "\n".join(f"    {value:.17g}," for value in values)
    return f"pub const {name}: &[f64] = &[\n{body}\n];"


def render(reference: Reference) -> str:
    real = observations()
    apo = reference.one("APO", real, [FAST_PERIOD, SLOW_PERIOD, EMA])
    ppo = reference.one("PPO", real, [FAST_PERIOD, SLOW_PERIOD, EMA])
    macd = reference.three("MACD", real, [FAST_PERIOD, SLOW_PERIOD, SIGNAL_PERIOD])
    macdext = reference.three(
        "MACDEXT",
        real,
        [FAST_PERIOD, EMA, SLOW_PERIOD, EMA, SIGNAL_PERIOD, EMA],
    )
    if any(left != right for left, right in zip(macd, macdext)):
        raise RuntimeError("all-EMA MACDEXT did not match MACD")
    # The Rust-first MACDFIX definition is the fixed-period standard MACD, so
    # its pinned values come from the equivalent explicit 12/26 MACD call.
    macdfix = reference.three("MACD", real, [12, 26, SIGNAL_PERIOD])
    # Execute TA-Lib's separately named function as generator coverage even
    # though its legacy alternate smoothing constants are not this definition.
    reference.three("MACDFIX", real, [SIGNAL_PERIOD])
    trix = reference.one("TRIX", real, [TRIX_PERIOD])
    arrays = [
        rust_array("REAL", real),
        rust_array("APO_EMA_PREFIX", apo[:PREFIX_LENGTH]),
        rust_array("PPO_EMA_PREFIX", ppo[:PREFIX_LENGTH]),
        rust_array("MACD_PREFIX", macd[0][:PREFIX_LENGTH]),
        rust_array("MACD_SIGNAL_PREFIX", macd[1][:PREFIX_LENGTH]),
        rust_array("MACD_HISTOGRAM_PREFIX", macd[2][:PREFIX_LENGTH]),
        rust_array("MACDEXT_MACD_PREFIX", macdext[0][:PREFIX_LENGTH]),
        rust_array("MACDEXT_SIGNAL_PREFIX", macdext[1][:PREFIX_LENGTH]),
        rust_array("MACDEXT_HISTOGRAM_PREFIX", macdext[2][:PREFIX_LENGTH]),
        rust_array("MACDFIX_MACD_PREFIX", macdfix[0][:PREFIX_LENGTH]),
        rust_array("MACDFIX_SIGNAL_PREFIX", macdfix[1][:PREFIX_LENGTH]),
        rust_array("MACDFIX_HISTOGRAM_PREFIX", macdfix[2][:PREFIX_LENGTH]),
        rust_array("TRIX_PREFIX", trix[:PREFIX_LENGTH]),
    ]
    array_text = "\n\n".join(arrays)
    return f'''//! Pinned moving-average Momentum reference vectors.
//!
//! Generated from TA-Lib {TALIB_VERSION} at revision
//! `{TALIB_GIT_REVISION}`; source archive SHA-256
//! `{TALIB_SOURCE_ARCHIVE_SHA256}`.

pub const TALIB_VERSION: &str = "{TALIB_VERSION}";
pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";
pub const TALIB_SOURCE_ARCHIVE_SHA256: &str =
    "{TALIB_SOURCE_ARCHIVE_SHA256}";

pub const FAST_PERIOD: usize = {FAST_PERIOD};
pub const SLOW_PERIOD: usize = {SLOW_PERIOD};
pub const SIGNAL_PERIOD: usize = {SIGNAL_PERIOD};
pub const TRIX_PERIOD: usize = {TRIX_PERIOD};

{array_text}
'''


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="talib-moving-average-momentum-") as workspace:
        reference = Reference(build_pinned_talib(args.source_archive, Path(workspace)))
        try:
            rendered = render(reference)
        finally:
            reference.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
