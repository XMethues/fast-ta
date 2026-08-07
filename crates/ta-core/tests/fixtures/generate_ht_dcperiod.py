#!/usr/bin/env python3
"""Generate deterministic HT_DCPERIOD Rust fixtures through TA-Lib ctypes."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
        help="path to the official ta-lib-0.6.4-src.tar.gz release asset",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Rust fixture file to create",
    )
    return parser.parse_args()


def deterministic_noise(length: int, seed: int) -> list[float]:
    state = seed
    output = []
    for _ in range(length):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFF_FFFF
        unit = state / 0xFFFF_FFFF
        output.append(100.0 + 8.0 * (2.0 * unit - 1.0))
    return output


def observation_cases() -> list[tuple[str, str, list[float]]]:
    indexes = range(SOURCE_LENGTH)
    return [
        (
            "constant",
            "constant 100.0",
            [100.0 for _ in indexes],
        ),
        (
            "trend",
            "linear trend 100.0 + 0.25*i",
            [100.0 + 0.25 * i for i in indexes],
        ),
        (
            "sine",
            "20-observation sine 100.0 + 5.0*sin(2*pi*i/20)",
            [100.0 + 5.0 * math.sin(2.0 * math.pi * i / 20.0) for i in indexes],
        ),
        (
            "chirp",
            "chirp 100.0 + 5.0*sin(2*pi*(i/40 + i^2/10240))",
            [
                100.0
                + 5.0
                * math.sin(2.0 * math.pi * (i / 40.0 + i * i / 10_240.0))
                for i in indexes
            ],
        ),
        (
            "seeded_noise",
            f"LCG uniform noise seed 0x{NOISE_SEED:08X}",
            deterministic_noise(SOURCE_LENGTH, NOISE_SEED),
        ),
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
    subprocess.run(
        ["./configure", f"--prefix={install}"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["make", f"-j{max(1, os.cpu_count() or 1)}"],
        cwd=source,
        check=True,
    )
    subprocess.run(["make", "install"], cwd=source, check=True)

    for name in ["libta-lib.dylib", "libta-lib.so"]:
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
        self.library.TA_HT_DCPERIOD_Lookback.argtypes = []
        self.library.TA_HT_DCPERIOD_Lookback.restype = ctypes.c_int
        self.library.TA_HT_DCPERIOD.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.library.TA_HT_DCPERIOD.restype = ctypes.c_int

        self._check(self.library.TA_Initialize(), "TA_Initialize")
        reported = self.library.TA_GetVersionString().decode("ascii").split()[0]
        if reported != TALIB_VERSION:
            raise RuntimeError(
                f"expected TA-Lib {TALIB_VERSION}, library reports {reported}"
            )
        lookback = self.library.TA_HT_DCPERIOD_Lookback()
        if lookback != 32:
            raise RuntimeError(f"expected HT_DCPERIOD lookback 32, got {lookback}")

    @staticmethod
    def _check(return_code: int, operation: str) -> None:
        if return_code != 0:
            raise RuntimeError(f"{operation} failed with TA-Lib code {return_code}")

    def dcperiod(self, values: Sequence[float]) -> tuple[int, list[float]]:
        input_array = (ctypes.c_double * len(values))(*values)
        output_array = (ctypes.c_double * len(values))()
        begin = ctypes.c_int()
        count = ctypes.c_int()
        self._check(
            self.library.TA_HT_DCPERIOD(
                0,
                len(values) - 1,
                input_array,
                ctypes.byref(begin),
                ctypes.byref(count),
                output_array,
            ),
            "TA_HT_DCPERIOD",
        )
        return begin.value, list(output_array[: count.value])

    def close(self) -> None:
        self._check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_f64(value: float) -> str:
    return repr(value)


def rust_slice(values: Iterable[float], indent: str = "            ") -> str:
    return "\n".join(f"{indent}{rust_f64(value)}," for value in values)


def render_fixture(reference: TalibReference) -> str:
    rendered_cases = []
    for name, definition, input_values in observation_cases():
        begin, expected = reference.dcperiod(input_values)
        if begin != 32 or len(expected) != SOURCE_LENGTH - begin:
            raise RuntimeError(
                f"{name}: unexpected Output Range {begin}..{begin + len(expected)}"
            )
        rendered_cases.append(
            "    ReferenceCase {\n"
            f"        name: \"{name}\",\n"
            f"        definition: \"{definition}\",\n"
            "        input: &[\n"
            f"{rust_slice(input_values)}\n"
            "        ],\n"
            "        expected: &[\n"
            f"{rust_slice(expected)}\n"
            "        ],\n"
            "    },"
        )

    return f'''// Generated by tests/fixtures/generate_ht_dcperiod.py. Do not edit by hand.
// Reference source: TA-Lib v{TALIB_VERSION}, commit {TALIB_GIT_REVISION}.
// Source archive SHA-256: {TALIB_SOURCE_ARCHIVE_SHA256}.
// Run `cargo fmt --all` after regeneration.

#[derive(Debug, Clone, Copy)]
pub struct ReferenceCase {{
    pub name: &'static str,
    pub definition: &'static str,
    pub input: &'static [f64],
    pub expected: &'static [f64],
}}

pub const TALIB_VERSION: &str = "{TALIB_VERSION}";
pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";
pub const TALIB_SOURCE_ARCHIVE_SHA256: &str = "{TALIB_SOURCE_ARCHIVE_SHA256}";
pub const OUTPUT_BEGIN: usize = 32;
pub const SOURCE_LENGTH: usize = {SOURCE_LENGTH};
pub const NOISE_SEED: u32 = 0x{NOISE_SEED:08X};

pub const CASES: &[ReferenceCase] = &[
{chr(10).join(rendered_cases)}
];
'''


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fast-ta-ht-dcperiod-") as directory:
        library = build_pinned_talib(args.source_archive, Path(directory))
        reference = TalibReference(library)
        try:
            content = render_fixture(reference)
        finally:
            reference.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
