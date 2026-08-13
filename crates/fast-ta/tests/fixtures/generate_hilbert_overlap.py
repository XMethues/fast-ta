#!/usr/bin/env python3
"""Generate deterministic MAMA and HT_TRENDLINE fixtures through pinned TA-Lib ctypes."""

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
SOURCE_LENGTH = 256
NOISE_SEED = 0x5EEDC0DE
FAST_LIMIT = 0.5
SLOW_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
        help="path to the official ta-lib-0.6.4-src.tar.gz release asset",
    )
    parser.add_argument("--output", required=True, type=Path, help="Rust fixture file to create")
    return parser.parse_args()


def deterministic_noise(length: int, seed: int) -> list[float]:
    state = seed
    output = []
    for _ in range(length):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        unit = state / 0xFFFF_FFFF
        output.append(100.0 + 8.0 * (2.0 * unit - 1.0))
    return output


def observation_cases() -> list[tuple[str, list[float]]]:
    indexes = range(SOURCE_LENGTH)
    return [
        ("constant", [100.0 for _ in indexes]),
        ("trend", [100.0 + 0.25 * i for i in indexes]),
        (
            "sine",
            [100.0 + 5.0 * math.sin(2.0 * math.pi * i / 20.0) for i in indexes],
        ),
        (
            "chirp",
            [
                100.0
                + 5.0 * math.sin(2.0 * math.pi * (i / 40.0 + i * i / 10_240.0))
                for i in indexes
            ],
        ),
        ("seeded_noise", deterministic_noise(SOURCE_LENGTH, NOISE_SEED)),
    ]


def build_pinned_talib(source_archive: Path, workspace: Path) -> Path:
    digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if digest != TALIB_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            "TA-Lib source archive checksum mismatch: "
            f"expected {TALIB_SOURCE_ARCHIVE_SHA256}, got {digest}"
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
    raise RuntimeError(f"TA-Lib build did not produce a shared library under {install}")


class TalibReference:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.library.TA_Initialize.argtypes = []
        self.library.TA_Initialize.restype = ctypes.c_int
        self.library.TA_Shutdown.argtypes = []
        self.library.TA_Shutdown.restype = ctypes.c_int
        self.library.TA_GetVersionString.argtypes = []
        self.library.TA_GetVersionString.restype = ctypes.c_char_p
        self.library.TA_MAMA_Lookback.argtypes = [ctypes.c_double, ctypes.c_double]
        self.library.TA_MAMA_Lookback.restype = ctypes.c_int
        self.library.TA_HT_TRENDLINE_Lookback.argtypes = []
        self.library.TA_HT_TRENDLINE_Lookback.restype = ctypes.c_int
        self.library.TA_MAMA.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.library.TA_MAMA.restype = ctypes.c_int
        self.library.TA_HT_TRENDLINE.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.library.TA_HT_TRENDLINE.restype = ctypes.c_int
        self._check(self.library.TA_Initialize(), "TA_Initialize")
        reported = self.library.TA_GetVersionString().decode("ascii").split()[0]
        if reported != TALIB_VERSION:
            raise RuntimeError(f"expected TA-Lib {TALIB_VERSION}, library reports {reported}")
        if self.library.TA_MAMA_Lookback(FAST_LIMIT, SLOW_LIMIT) != 32:
            raise RuntimeError("unexpected MAMA lookback")
        if self.library.TA_HT_TRENDLINE_Lookback() != 63:
            raise RuntimeError("unexpected HT_TRENDLINE lookback")

    @staticmethod
    def _check(return_code: int, operation: str) -> None:
        if return_code != 0:
            raise RuntimeError(f"{operation} failed with TA-Lib code {return_code}")

    @staticmethod
    def _arrays(values: Sequence[float], output_columns: int):
        inputs = (ctypes.c_double * len(values))(*values)
        outputs = [(ctypes.c_double * len(values))() for _ in range(output_columns)]
        begin = ctypes.c_int()
        count = ctypes.c_int()
        return inputs, outputs, begin, count

    def mama(self, values: Sequence[float]) -> tuple[int, list[float], list[float]]:
        inputs, outputs, begin, count = self._arrays(values, 2)
        self._check(
            self.library.TA_MAMA(
                0,
                len(values) - 1,
                inputs,
                FAST_LIMIT,
                SLOW_LIMIT,
                ctypes.byref(begin),
                ctypes.byref(count),
                outputs[0],
                outputs[1],
            ),
            "TA_MAMA",
        )
        return begin.value, list(outputs[0][: count.value]), list(outputs[1][: count.value])

    def trendline(self, values: Sequence[float]) -> tuple[int, list[float]]:
        inputs, outputs, begin, count = self._arrays(values, 1)
        self._check(
            self.library.TA_HT_TRENDLINE(
                0,
                len(values) - 1,
                inputs,
                ctypes.byref(begin),
                ctypes.byref(count),
                outputs[0],
            ),
            "TA_HT_TRENDLINE",
        )
        return begin.value, list(outputs[0][: count.value])

    def close(self) -> None:
        self._check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_slice(values: Iterable[float]) -> str:
    return "\n".join(f"            {value!r}," for value in values)


def render_fixture(reference: TalibReference) -> str:
    rendered_cases = []
    for name, input_values in observation_cases():
        mama_begin, mama, fama = reference.mama(input_values)
        trendline_begin, trendline = reference.trendline(input_values)
        if mama_begin != 32 or len(mama) != SOURCE_LENGTH - mama_begin:
            raise RuntimeError(f"{name}: unexpected MAMA Output Range")
        if trendline_begin != 63 or len(trendline) != SOURCE_LENGTH - trendline_begin:
            raise RuntimeError(f"{name}: unexpected HT_TRENDLINE Output Range")
        rendered_cases.append(
            "    ReferenceCase {\n"
            f"        name: \"{name}\",\n"
            "        mama: &[\n"
            f"{rust_slice(mama)}\n"
            "        ],\n"
            "        fama: &[\n"
            f"{rust_slice(fama)}\n"
            "        ],\n"
            "        trendline: &[\n"
            f"{rust_slice(trendline)}\n"
            "        ],\n"
            "    },"
        )

    return f'''// Generated by tests/fixtures/generate_hilbert_overlap.py. Do not edit by hand.
// Reference source: TA-Lib v{TALIB_VERSION}, commit {TALIB_GIT_REVISION}.
// Source archive SHA-256: {TALIB_SOURCE_ARCHIVE_SHA256}.
// Inputs are the identically named pinned vectors in ht_dcperiod_reference.rs.
// Run `cargo fmt --all` after regeneration.

#[derive(Debug, Clone, Copy)]
pub struct ReferenceCase {{
    pub name: &'static str,
    pub mama: &'static [f64],
    pub fama: &'static [f64],
    pub trendline: &'static [f64],
}}

pub const TALIB_VERSION: &str = "{TALIB_VERSION}";
pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";
pub const TALIB_SOURCE_ARCHIVE_SHA256: &str = "{TALIB_SOURCE_ARCHIVE_SHA256}";
pub const SOURCE_LENGTH: usize = {SOURCE_LENGTH};
pub const MAMA_OUTPUT_BEGIN: usize = 32;
pub const HT_TRENDLINE_OUTPUT_BEGIN: usize = 63;
pub const FAST_LIMIT: f64 = {FAST_LIMIT};
pub const SLOW_LIMIT: f64 = {SLOW_LIMIT};

pub const CASES: &[ReferenceCase] = &[
{chr(10).join(rendered_cases)}
];
'''


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fast-ta-hilbert-overlap-") as directory:
        library = build_pinned_talib(args.source_archive, Path(directory))
        reference = TalibReference(library)
        try:
            content = render_fixture(reference)
        finally:
            reference.close()
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
