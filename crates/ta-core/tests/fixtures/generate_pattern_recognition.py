#!/usr/bin/env python3
"""Generate qualified Pattern Recognition fixtures through pinned TA-Lib C."""

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
SINGLE_CANDLE = (
    [10.0] * 10 + [10.0, 14.0, 13.0, 7.0, 9.5, 10.5, 10.0, 11.0, 9.75, 10.25],
    [15.0] * 10 + [14.25, 14.25, 13.25, 15.0, 14.0, 14.0, 11.5, 11.5, 12.0, 12.0],
    [5.0] * 10 + [9.75, 9.75, 5.0, 6.75, 6.0, 6.0, 9.5, 9.5, 8.0, 8.0],
    [12.0] * 10 + [14.0, 10.0, 13.0, 7.0, 10.5, 9.5, 11.0, 10.0, 10.25, 9.75],
)
SINGLE_CANDLE_NAMES = (
    "CDLBELTHOLD",
    "CDLCLOSINGMARUBOZU",
    "CDLDRAGONFLYDOJI",
    "CDLGRAVESTONEDOJI",
    "CDLHIGHWAVE",
    "CDLLONGLEGGEDDOJI",
    "CDLLONGLINE",
    "CDLMARUBOZU",
    "CDLRICKSHAWMAN",
    "CDLSHORTLINE",
    "CDLSPINNINGTOP",
    "CDLTAKURI",
)

TWO_CANDLE_FIXTURES = {
    "CDLCOUNTERATTACK": (
        [10.0] * 10 + [20.0, 0.0, 10.0],
        [12.0] * 10 + [21.0, 11.0, 12.0],
        [9.0] * 10 + [9.0, -1.0, 9.0],
        [11.0] * 10 + [10.0, 10.0, 11.0],
    ),
    "CDLDARKCLOUDCOVER": (
        [10.0] * 10 + [10.0, 22.0, 10.0],
        [12.0] * 10 + [21.0, 23.0, 12.0],
        [9.0] * 10 + [9.0, 13.0, 9.0],
        [11.0] * 10 + [20.0, 14.0, 11.0],
    ),
    "CDLDOJISTAR": (
        [10.0] * 10 + [10.0, 21.0, 10.0],
        [12.0] * 10 + [21.0, 22.0, 12.0],
        [9.0] * 10 + [9.0, 20.0, 9.0],
        [11.0] * 10 + [20.0, 21.0, 11.0],
    ),
    "CDLHARAMI": (
        [10.0] * 10 + [10.0, 15.5, 10.0],
        [12.0] * 10 + [21.0, 16.0, 12.0],
        [9.0] * 10 + [9.0, 14.0, 9.0],
        [11.0] * 10 + [20.0, 14.5, 11.0],
    ),
    "CDLHARAMICROSS": (
        [10.0] * 10 + [10.0, 15.0, 10.0],
        [12.0] * 10 + [21.0, 16.0, 12.0],
        [9.0] * 10 + [9.0, 14.0, 9.0],
        [11.0] * 10 + [20.0, 15.0, 11.0],
    ),
    "CDLHOMINGPIGEON": (
        [10.0] * 10 + [20.0, 17.0, 10.0],
        [12.0] * 10 + [21.0, 18.0, 12.0],
        [9.0] * 10 + [9.0, 15.0, 9.0],
        [11.0] * 10 + [10.0, 16.0, 11.0],
    ),
    "CDLKICKING": (
        [10.0] * 10 + [20.0, 22.0, 10.0],
        [12.0] * 10 + [20.0, 32.0, 12.0],
        [9.0] * 10 + [10.0, 22.0, 9.0],
        [11.0] * 10 + [10.0, 32.0, 11.0],
    ),
    "CDLKICKINGBYLENGTH": (
        [10.0] * 10 + [20.0, 22.0, 10.0],
        [12.0] * 10 + [20.0, 32.0, 12.0],
        [9.0] * 10 + [10.0, 22.0, 9.0],
        [11.0] * 10 + [10.0, 32.0, 11.0],
    ),
    "CDLMATCHINGLOW": (
        [10.0] * 10 + [20.0, 18.0, 10.0],
        [12.0] * 10 + [21.0, 19.0, 12.0],
        [9.0] * 10 + [9.0, 10.0, 9.0],
        [11.0] * 10 + [10.0, 10.0, 11.0],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def build_pinned_talib(source_archive: Path, workspace: Path) -> Path:
    digest = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    if digest != TALIB_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            f"TA-Lib archive checksum mismatch: expected {TALIB_SOURCE_ARCHIVE_SHA256}, got {digest}"
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
        standard_names = (
            "CDLDOJI",
            "CDLENGULFING",
            *SINGLE_CANDLE_NAMES,
            *(name for name in TWO_CANDLE_FIXTURES if name != "CDLDARKCLOUDCOVER"),
        )
        for name in standard_names:
            function = getattr(self.library, f"TA_{name}")
            function.argtypes = arguments
            function.restype = ctypes.c_int
            lookback = getattr(self.library, f"TA_{name}_Lookback")
            lookback.argtypes = []
            lookback.restype = ctypes.c_int
        dark_cloud = self.library.TA_CDLDARKCLOUDCOVER
        dark_cloud.argtypes = [*arguments[:6], ctypes.c_double, *arguments[6:]]
        dark_cloud.restype = ctypes.c_int
        dark_cloud_lookback = self.library.TA_CDLDARKCLOUDCOVER_Lookback
        dark_cloud_lookback.argtypes = [ctypes.c_double]
        dark_cloud_lookback.restype = ctypes.c_int

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

    def set_setting(self, setting_type: int, range_type: int, period: int, factor: float) -> None:
        self._check(
            self.library.TA_SetCandleSettings(setting_type, range_type, period, factor),
            f"set Candle Setting {setting_type}",
        )

    def set_custom_doji(self) -> None:
        self.set_setting(3, 0, 3, 0.5)

    def set_custom_single_candle(self) -> None:
        # The eight settings referenced by this wave. Enum/range numbers are pinned C values.
        for setting in (
            (0, 0, 3, 1.5),
            (2, 0, 3, 2.0),
            (3, 1, 3, 0.125),
            (4, 0, 0, 1.25),
            (5, 0, 0, 3.0),
            (6, 2, 3, 0.5),
            (7, 1, 3, 0.0625),
            (8, 1, 3, 0.125),
        ):
            self.set_setting(*setting)

    def set_custom_two_candle(self) -> None:
        # Referenced settings with shorter periods; enum/range numbers are pinned C values.
        for setting in (
            (0, 0, 3, 1.0),
            (2, 0, 3, 1.0),
            (3, 1, 3, 0.1),
            (7, 1, 3, 0.1),
            (10, 1, 3, 0.05),
        ):
            self.set_setting(*setting)

    def compute(
        self,
        name: str,
        observations: tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]],
        quantize_f32: bool,
        penetration: float | None = None,
    ) -> tuple[int, list[int]]:
        columns = [
            [float(ctypes.c_float(value).value) for value in column]
            if quantize_f32
            else list(column)
            for column in observations
        ]
        length = len(columns[0])
        arrays = [(ctypes.c_double * length)(*column) for column in columns]
        output = (ctypes.c_int * length)()
        begin = ctypes.c_int()
        count = ctypes.c_int()
        call_arguments = [0, length - 1, *arrays]
        if penetration is not None:
            call_arguments.append(penetration)
        call_arguments.extend((ctypes.byref(begin), ctypes.byref(count), output))
        self._check(
            getattr(self.library, f"TA_{name}")(*call_arguments),
            f"TA_{name}",
        )
        lookback_function = getattr(self.library, f"TA_{name}_Lookback")
        lookback = (
            lookback_function(penetration)
            if penetration is not None
            else lookback_function()
        )
        if begin.value != lookback or count.value != length - lookback:
            raise RuntimeError(f"{name}: unexpected output range {begin.value}+{count.value}")
        return begin.value, list(output[: count.value])

    def close(self) -> None:
        self._check(self.library.TA_Shutdown(), "TA_Shutdown")


def rust_float(value: float) -> str:
    rendered = repr(value)
    return rendered if "." in rendered else rendered + ".0"


def rust_slice(values: Iterable[float]) -> str:
    return ", ".join(rust_float(value) for value in values)


def append_columns(lines: list[str], prefix: str, columns: Sequence[Sequence[float]]) -> None:
    for suffix, values in zip(("OPEN", "HIGH", "LOW", "CLOSE"), columns):
        lines.append(f"pub const {prefix}_{suffix}: &[f64] = &[{rust_slice(values)}];")


def append_result(
    lines: list[str],
    prefix: str,
    f64_result: tuple[int, list[int]],
    f32_result: tuple[int, list[int]],
) -> None:
    lines.append(f"pub const {prefix}_LOOKBACK: usize = {f64_result[0]};")
    lines.append(f"pub const {prefix}_F64_CODES: &[i32] = &{f64_result[1]!r};")
    lines.append(f"pub const {prefix}_F32_CODES: &[i32] = &{f32_result[1]!r};")


def render(reference: TalibReference) -> str:
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

    reference.restore_defaults()
    append_columns(lines, "DOJI_DEFAULT", DOJI_DEFAULT)
    append_result(
        lines,
        "DOJI_DEFAULT",
        reference.compute("CDLDOJI", DOJI_DEFAULT, False),
        reference.compute("CDLDOJI", DOJI_DEFAULT, True),
    )
    lines.append("")

    reference.set_custom_doji()
    append_columns(lines, "DOJI_CUSTOM", DOJI_CUSTOM)
    append_result(
        lines,
        "DOJI_CUSTOM",
        reference.compute("CDLDOJI", DOJI_CUSTOM, False),
        reference.compute("CDLDOJI", DOJI_CUSTOM, True),
    )
    lines.append("")

    reference.restore_defaults()
    append_columns(lines, "ENGULFING", ENGULFING)
    append_result(
        lines,
        "ENGULFING",
        reference.compute("CDLENGULFING", ENGULFING, False),
        reference.compute("CDLENGULFING", ENGULFING, True),
    )
    lines.append("")

    append_columns(lines, "SINGLE_CANDLE", SINGLE_CANDLE)
    for name in SINGLE_CANDLE_NAMES:
        prefix = name.removeprefix("CDL") + "_DEFAULT"
        append_result(
            lines,
            prefix,
            reference.compute(name, SINGLE_CANDLE, False),
            reference.compute(name, SINGLE_CANDLE, True),
        )
    lines.append("")

    reference.set_custom_single_candle()
    for name in SINGLE_CANDLE_NAMES:
        prefix = name.removeprefix("CDL") + "_CUSTOM"
        append_result(
            lines,
            prefix,
            reference.compute(name, SINGLE_CANDLE, False),
            reference.compute(name, SINGLE_CANDLE, True),
        )
    lines.append("")

    reference.restore_defaults()
    for name, observations in TWO_CANDLE_FIXTURES.items():
        prefix = name.removeprefix("CDL")
        append_columns(lines, prefix, observations)
        penetration = 0.5 if name == "CDLDARKCLOUDCOVER" else None
        append_result(
            lines,
            prefix + "_DEFAULT",
            reference.compute(name, observations, False, penetration),
            reference.compute(name, observations, True, penetration),
        )
    lines.append("")

    reference.set_custom_two_candle()
    for name, observations in TWO_CANDLE_FIXTURES.items():
        prefix = name.removeprefix("CDL") + "_CUSTOM"
        penetration = 0.25 if name == "CDLDARKCLOUDCOVER" else None
        append_result(
            lines,
            prefix,
            reference.compute(name, observations, False, penetration),
            reference.compute(name, observations, True, penetration),
        )
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
