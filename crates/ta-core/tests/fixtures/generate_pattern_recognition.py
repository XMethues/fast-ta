#!/usr/bin/env python3
"""Generate CDLDOJI/CDLENGULFING fixtures through pinned TA-Lib C."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Iterable, Sequence

TALIB_VERSION = "0.7.1"
TALIB_GIT_REVISION = "2247d599bddf37ed37e3a709371517e46efc66f6"
TALIB_SOURCE_ARCHIVE_SHA256 = (
    "c9cce2810eab127722c9a0f6c721fc2e955ac86e6e999563ed1176d29e22fc2d"
)

DOJI_DEFAULT = (
    [10.0] * 16,
    [15.0] * 16,
    [5.0] * 16,
    [12.0] * 10 + [11.0, 11.1, 10.0, 9.5, 12.0, 10.0],
)
DOJI_CUSTOM = (
    [10.0] * 8,
    [20.0] * 8,
    [0.0] * 8,
    [12.0, 14.0, 16.0, 14.0, 11.0, 14.0, 11.5, 11.1],
)
ENGULFING = (
    [9.0, 10.0, 7.0, 12.0, 5.0, 12.0, 10.0],
    [10.0, 11.0, 12.0, 13.0, 13.0, 13.0, 11.0],
    [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 7.0],
    [9.0, 8.0, 11.0, 6.0, 12.0, 4.0, 8.0],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def build_pinned_talib(source_archive: Path, workspace: Path) -> Path:
    digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if digest != TALIB_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"TA-Lib archive checksum mismatch: expected "
            f"{TALIB_SOURCE_ARCHIVE_SHA256}, got {digest}"
        )
    with tarfile.open(source_archive, "r:gz") as archive:
        archive.extractall(workspace)
    source = workspace / f"ta-lib-{TALIB_GIT_REVISION}"
    build = workspace / "build"
    install = workspace / "install"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install}",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", str(max(1, os.cpu_count() or 1))],
        check=True,
    )
    subprocess.run(["cmake", "--install", str(build)], check=True)
    for directory in (install / "lib", install / "lib64"):
        for name in ("libta-lib.dylib", "libta-lib.so"):
            library = directory / name
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

        arguments = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        for name in ("CDLDOJI", "CDLENGULFING"):
            function = getattr(self.library, f"TA_{name}")
            function.argtypes = arguments
            function.restype = ctypes.c_int
            lookback = getattr(self.library, f"TA_{name}_Lookback")
            lookback.argtypes = []
            lookback.restype = ctypes.c_int

        self.library.TA_SetCandleSettings.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        ]
        self.library.TA_SetCandleSettings.restype = ctypes.c_int
        self.library.TA_RestoreCandleDefaultSettings.argtypes = [ctypes.c_int]
        self.library.TA_RestoreCandleDefaultSettings.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with TA-Lib code {code}")

    def restore_defaults(self) -> None:
        self._check(self.library.TA_RestoreCandleDefaultSettings(11), "restore defaults")

    def set_custom_doji(self) -> None:
        # TA_BodyDoji=3, TA_RangeType_RealBody=0.
        self._check(
            self.library.TA_SetCandleSettings(3, 0, 3, ctypes.c_double(0.5)),
            "set BodyDoji",
        )

    def compute(
        self,
        name: str,
        observations: tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
        quantize_f32: bool,
    ) -> tuple[int, list[int]]:
        columns = [
            [float(ctypes.c_float(value).value) for value in column]
            if quantize_f32
            else list(column)
            for column in observations
        ]
        open_values, high_values, low_values, close_values = columns
        length = len(open_values)
        arrays = [
            (ctypes.c_double * length)(*column)
            for column in (open_values, high_values, low_values, close_values)
        ]
        output = (ctypes.c_int * length)()
        begin = ctypes.c_int()
        count = ctypes.c_int()
        function = getattr(self.library, f"TA_{name}")
        self._check(
            function(
                0,
                length - 1,
                *arrays,
                ctypes.byref(begin),
                ctypes.byref(count),
                output,
            ),
            f"TA_{name}",
        )
        expected_lookback = getattr(self.library, f"TA_{name}_Lookback")()
        if begin.value != expected_lookback or count.value != length - expected_lookback:
            raise RuntimeError(f"{name}: unexpected output range {begin.value}+{count.value}")
        return begin.value, list(output[: count.value])

    def close(self) -> None:
        self._check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_float(value: float) -> str:
    rendered = repr(value)
    return rendered if "." in rendered else rendered + ".0"


def rust_slice(values: Iterable[float]) -> str:
    return ", ".join(rust_float(value) for value in values)


def render(reference: TalibReference) -> str:
    reference.restore_defaults()
    default64 = reference.compute("CDLDOJI", DOJI_DEFAULT, False)
    default32 = reference.compute("CDLDOJI", DOJI_DEFAULT, True)
    reference.set_custom_doji()
    custom64 = reference.compute("CDLDOJI", DOJI_CUSTOM, False)
    custom32 = reference.compute("CDLDOJI", DOJI_CUSTOM, True)
    reference.restore_defaults()
    engulf64 = reference.compute("CDLENGULFING", ENGULFING, False)
    engulf32 = reference.compute("CDLENGULFING", ENGULFING, True)

    lines = [
        "// Generated by tests/fixtures/generate_pattern_recognition.py. Do not edit by hand.",
        f"// Reference source: TA-Lib v{TALIB_VERSION}, commit {TALIB_GIT_REVISION}.",
        f"// Pinned source archive SHA-256: {TALIB_SOURCE_ARCHIVE_SHA256}.",
        "",
        f'pub const TALIB_VERSION: &str = "{TALIB_VERSION}";',
        f'pub const TALIB_GIT_REVISION: &str = "{TALIB_GIT_REVISION}";',
        f'pub const TALIB_SOURCE_ARCHIVE_SHA256: &str = "{TALIB_SOURCE_ARCHIVE_SHA256}";',
        "",
    ]
    fixtures = [
        ("DOJI_DEFAULT", DOJI_DEFAULT, default64, default32),
        ("DOJI_CUSTOM", DOJI_CUSTOM, custom64, custom32),
        ("ENGULFING", ENGULFING, engulf64, engulf32),
    ]
    for prefix, columns, f64_result, f32_result in fixtures:
        for suffix, values in zip(("OPEN", "HIGH", "LOW", "CLOSE"), columns):
            lines.append(f"pub const {prefix}_{suffix}: &[f64] = &[{rust_slice(values)}];")
        lines.append(f"pub const {prefix}_LOOKBACK: usize = {f64_result[0]};")
        lines.append(f"pub const {prefix}_F64_CODES: &[i32] = &{f64_result[1]!r};")
        lines.append(f"pub const {prefix}_F32_CODES: &[i32] = &{f32_result[1]!r};")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="fast-ta-pattern-recognition-") as directory:
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
