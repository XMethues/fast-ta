#!/usr/bin/env python3
"""Prepare pinned dependencies and run the opt-in Indicator Catalogue matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
import stat
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

TALIB_VERSION = "0.6.4"
TALIB_REVISION = "43f9d5042ecc4bd367941846494ad907bf20ea50"
TALIB_ARCHIVE_SHA256 = "aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86"
TALIB_ARCHIVE_URL = (
    "https://github.com/TA-Lib/ta-lib/releases/download/"
    f"v{TALIB_VERSION}/ta-lib-{TALIB_VERSION}-src.tar.gz"
)
PYTHON_BINDING_VERSION = "0.6.4"
NUMPY_VERSION = "2.2.3"
PIP_VERSION = "25.0.1"
SETUPTOOLS_VERSION = "75.8.2"
WHEEL_VERSION = "0.45.1"
# The 0.6.4 sdist's talib/_ta_lib.c header records Cython 3.0.11, while the
# sdist omits talib/_ta_lib.pxd and upstream setup.py selects the bundled C
# whenever Cython is absent. Keep Cython out so the published C is compiled.
BUNDLED_CYTHON_VERSION = "3.0.11"
MINIMUM_PYTHON = (3, 10)

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPOSITORY / "target" / "catalogue-matrix"
BENCHMARK_CRATE = REPOSITORY / "crates" / "ta-benchmarks"
DEFAULT_BASELINE = BENCHMARK_CRATE / "baselines" / "catalogue_matrix_post_scalar_diagnostic.tsv"
DEFAULT_OPTIMIZATION_EVIDENCE = (
    BENCHMARK_CRATE / "baselines" / "catalogue_matrix_optimization_evidence.tsv"
)
PUBLISHED_RAW = BENCHMARK_CRATE / "baselines" / "catalogue_matrix_optimized.tsv"
PUBLISHED_REPORT = BENCHMARK_CRATE / "CATALOGUE_MATRIX_REPORT.txt"

CANONICAL_SAMPLES = 50
CANONICAL_WARMUP_MS = 250
CANONICAL_SAMPLE_MS = 10
CASE_IDS = (
    "SMA",
    "BBANDS",
    "RSI",
    "MACD",
    "ATR",
    "ADX",
    "HT_DCPHASE",
    "CDLDOJI",
    "CDLENGULFING",
    "CDL3WHITESOLDIERS",
    "LINEARREG",
    "TYPPRICE",
    "OBV",
    "SIN",
    "ADD",
)
INPUT_LENGTHS = (256, 4_096, 65_536)
INPUT_CHECKSUMS = {
    256: "fnv1a64:73fedfe0ae0a803f",
    4_096: "fnv1a64:06be03be64d63c6c",
    65_536: "fnv1a64:a4171d0a7611733a",
}
VARIANTS = (
    ("fast-ta", "Owned Compact Output"),
    ("fast-ta", "caller-owned Batch Computation"),
    ("fast-ta", "Prepared Batch Runner"),
    ("fast-ta", "Streaming Computation"),
    ("TA-Lib C", "direct C caller-owned"),
    ("TA-Lib Python", "official Python NumPy API"),
)
FIXED_PUBLICATION_FIELDS = {
    "semantic_status": "verified",
    "timing_status": "measured",
    "sample_count": str(CANONICAL_SAMPLES),
    "dirty": "false",
    "fixture": "catalogue_fixture_v1:f64le",
    "ta_lib_version": TALIB_VERSION,
    "ta_lib_revision": TALIB_REVISION,
    "python_binding_version": PYTHON_BINDING_VERSION,
    "python_ta_lib_version": TALIB_VERSION,
    "numpy_version": NUMPY_VERSION,
    "float_width": "64",
    "features": "ta-core=default(f64,std); ta-benchmarks=catalogue-matrix",
}
UNIFORM_PROVENANCE_FIELDS = (
    "python_version",
    "rustc",
    "cpu",
    "os",
    "arch",
    "commit",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, help="use an already downloaded pinned source archive")
    parser.add_argument("--deps-dir", type=Path, default=DEFAULT_ROOT / "deps")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Python used to create the isolated environment")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--optimization-evidence",
        type=Path,
        default=DEFAULT_OPTIMIZATION_EVIDENCE,
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish a successful clean full run to the stable repository paths",
    )
    parser.add_argument(
        "--publish-existing",
        action="store_true",
        help="regenerate and publish an existing full raw result without rerunning timings",
    )
    parser.add_argument("--case", help="run one matrix case ID, for example ADX")
    parser.add_argument(
        "--input-length",
        type=int,
        choices=(256, 4096, 65536),
        help="run one representative input length",
    )
    parser.add_argument("--samples", type=int, help="override timed sample count")
    parser.add_argument("--warmup-ms", type=int, help="override warm-up duration per variant")
    parser.add_argument("--sample-ms", type=int, help="override target duration per timed sample")
    return parser.parse_args()


def run(arguments: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(arguments, cwd=cwd, env=env, check=True)


def inspect_python(python: str) -> tuple[str, tuple[int, int, int]]:
    probe = (
        "import json, sys; "
        "print(json.dumps([sys.implementation.name, list(sys.version_info[:3])]))"
    )
    try:
        completed = subprocess.run(
            [python, "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"cannot run requested Python interpreter {python!r}: {error}") from error

    try:
        implementation, raw_version = json.loads(completed.stdout)
        version = tuple(int(component) for component in raw_version)
        if not isinstance(implementation, str) or len(version) != 3:
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"requested Python interpreter {python!r} returned an invalid identity"
        ) from error

    if version < MINIMUM_PYTHON:
        required = ".".join(str(component) for component in MINIMUM_PYTHON)
        actual = ".".join(str(component) for component in version)
        raise SystemExit(
            f"unsupported Python {actual} ({implementation}) from {python!r}: "
            f"TA-Lib {PYTHON_BINDING_VERSION} with NumPy {NUMPY_VERSION} "
            f"requires Python {required} or newer; choose --python accordingly"
        )
    return implementation, version




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_archive(source_archive: Path | None, deps_dir: Path) -> Path:
    archive = source_archive or deps_dir / f"ta-lib-{TALIB_VERSION}-src.tar.gz"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(archive.suffix + ".download")
        print(f"Downloading {TALIB_ARCHIVE_URL}")
        urllib.request.urlretrieve(TALIB_ARCHIVE_URL, temporary)
        temporary.replace(archive)
    digest = sha256_file(archive)
    if digest != TALIB_ARCHIVE_SHA256:
        raise RuntimeError(
            f"TA-Lib archive checksum mismatch: expected {TALIB_ARCHIVE_SHA256}, got {digest}"
        )
    return archive


def remove_staging_tree(staging: Path) -> None:
    def make_directories_removable(directory: Path) -> None:
        mode = stat.S_IMODE(directory.stat().st_mode)
        directory.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        with os.scandir(directory) as entries:
            children = [
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
            ]
        for child in children:
            make_directories_removable(child)

    make_directories_removable(staging)
    shutil.rmtree(staging)


def extract_archive(archive: Path, destination: Path) -> Path:
    archive_digest = sha256_file(archive)
    if archive_digest != TALIB_ARCHIVE_SHA256:
        raise RuntimeError(
            f"TA-Lib archive checksum mismatch: expected {TALIB_ARCHIVE_SHA256}, "
            f"got {archive_digest}"
        )

    source = destination / f"ta-lib-{TALIB_VERSION}"
    marker = source / ".fast-ta-source-pin"
    expected_marker = (
        f"version={TALIB_VERSION}\n"
        f"revision={TALIB_REVISION}\n"
        f"archive_sha256={archive_digest}\n"
    )
    try:
        if marker.is_file() and marker.read_text(encoding="utf-8") == expected_marker:
            return source
    except (OSError, UnicodeError):
        pass

    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{source.name}.extract-", dir=destination)
    )
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            root = staging.resolve()
            for member in bundle.getmembers():
                member_path = (staging / member.name).resolve()
                if root not in member_path.parents and member_path != root:
                    raise RuntimeError(f"unsafe TA-Lib archive member {member.name!r}")
                if not (member.isdir() or member.isfile()):
                    raise RuntimeError(
                        f"unsupported TA-Lib archive member {member.name!r}"
                    )
            bundle.extractall(staging)
        extracted = staging / source.name
        if not extracted.is_dir():
            raise RuntimeError(f"pinned archive did not contain {source.name}")
        extracted_mode = stat.S_IMODE(extracted.stat().st_mode)
        extracted.chmod(extracted_mode | stat.S_IWUSR)
        (extracted / marker.name).write_text(expected_marker, encoding="utf-8")
        if source.exists() or source.is_symlink():
            if source.is_dir() and not source.is_symlink():
                shutil.rmtree(source)
            else:
                source.unlink()
        extracted.replace(source)
    finally:
        remove_staging_tree(staging)
    return source


def build_talib(archive: Path, deps_dir: Path) -> Path:
    install = deps_dir / f"ta-lib-{TALIB_VERSION}-install"
    marker = install / ".fast-ta-pin"
    expected_marker = f"version={TALIB_VERSION}\nrevision={TALIB_REVISION}\nsha256={TALIB_ARCHIVE_SHA256}\n"
    libraries = [install / "lib" / "libta-lib.dylib", install / "lib" / "libta-lib.so"]
    if marker.exists() and marker.read_text() == expected_marker and any(path.exists() for path in libraries):
        return install

    source_root = deps_dir / "source"
    source = extract_archive(archive, source_root)
    if install.exists():
        shutil.rmtree(install)
    run(["./configure", f"--prefix={install}"], cwd=source)
    run(["make", f"-j{max(1, os.cpu_count() or 1)}"], cwd=source)
    run(["make", "install"], cwd=source)
    marker.write_text(expected_marker)
    if not any(path.exists() for path in libraries):
        raise RuntimeError("TA-Lib build did not produce libta-lib.dylib or libta-lib.so")
    return install


def prepare_python(
    python: str,
    python_identity: tuple[str, tuple[int, int, int]],
    deps_dir: Path,
    talib_install: Path,
) -> Path:
    implementation, version = python_identity
    version_text = ".".join(str(component) for component in version)
    virtualenv = deps_dir / (
        f"venv-{implementation}-{version_text}-talib-{PYTHON_BINDING_VERSION}"
        f"-numpy-{NUMPY_VERSION}"
    )
    python_executable = virtualenv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = virtualenv / ".fast-ta-pin"
    expected_marker = (
        f"python={implementation}-{version_text}\n"
        f"binding={PYTHON_BINDING_VERSION}\nnumpy={NUMPY_VERSION}\n"
        f"bundled_cython={BUNDLED_CYTHON_VERSION}\n"
        f"talib_revision={TALIB_REVISION}\n"
    )
    if marker.exists() and marker.read_text() == expected_marker and python_executable.exists():
        return python_executable

    if virtualenv.exists():
        shutil.rmtree(virtualenv)
    run([python, "-m", "venv", str(virtualenv)])
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            f"pip=={PIP_VERSION}",
            f"setuptools=={SETUPTOOLS_VERSION}",
            f"wheel=={WHEEL_VERSION}",
            f"numpy=={NUMPY_VERSION}",
        ]
    )
    environment = os.environ.copy()
    environment["TA_INCLUDE_PATH"] = str(talib_install / "include")
    environment["TA_LIBRARY_PATH"] = str(talib_install / "lib")
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-binary=TA-Lib",
            f"TA-Lib=={PYTHON_BINDING_VERSION}",
        ],
        env=environment,
    )
    marker.write_text(expected_marker)
    return python_executable


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    if args.case is not None or args.input_length is not None:
        focus_name = f"{args.case or 'all'}-{args.input_length or 'all'}".lower()
        return (DEFAULT_ROOT / "focused" / focus_name).resolve()
    return (DEFAULT_ROOT / "results").resolve()


def generate_report(args: argparse.Namespace, output_dir: Path) -> None:
    arguments = [
        "cargo",
        "run",
        "--release",
        "-p",
        "ta-benchmarks",
        "--features",
        "catalogue-matrix",
        "--bin",
        "catalogue-report",
        "--",
        "--raw",
        str(output_dir / "catalogue-matrix-raw.tsv"),
        "--report",
        str(output_dir / "catalogue-matrix-report.txt"),
    ]
    if args.baseline is not None:
        arguments.extend(("--baseline", str(args.baseline.resolve())))
    if args.optimization_evidence is not None:
        arguments.extend(
            ("--optimization-evidence", str(args.optimization_evidence.resolve()))
        )
    run(arguments, cwd=REPOSITORY)


def validate_publishable(raw_path: Path) -> None:
    with raw_path.open(newline="", encoding="utf-8") as raw:
        reader = csv.DictReader(raw, delimiter="\t")
        rows = list(reader)
        fields = set(reader.fieldnames or ())

    required_fields = {
        "case_id",
        "input_length",
        "implementation",
        "mode",
        "indicator_family",
        "indicator_definition",
        "parameters",
        "output_kind",
        "output_arity",
        "input_checksum",
        *FIXED_PUBLICATION_FIELDS,
        *UNIFORM_PROVENANCE_FIELDS,
    }
    missing_fields = sorted(required_fields - fields)
    if missing_fields:
        raise RuntimeError(
            "cannot publish raw rows missing fields: " + ", ".join(missing_fields)
        )

    expected = Counter(
        (case_id, str(input_length), implementation, mode)
        for case_id in CASE_IDS
        for input_length in INPUT_LENGTHS
        for implementation, mode in VARIANTS
    )
    actual = Counter(
        (
            row["case_id"],
            row["input_length"],
            row["implementation"],
            row["mode"],
        )
        for row in rows
    )
    if actual != expected:
        missing = list((expected - actual).elements())
        unexpected = list((actual - expected).elements())
        raise RuntimeError(
            "cannot publish incomplete case/input/mode matrix: "
            f"missing={missing[:5]!r}; unexpected_or_duplicate={unexpected[:5]!r}"
        )

    for field, expected_value in FIXED_PUBLICATION_FIELDS.items():
        invalid = sum(row[field] != expected_value for row in rows)
        if invalid:
            raise RuntimeError(
                f"cannot publish: {invalid} rows have {field} other than "
                f"{expected_value!r}"
            )

    for field in UNIFORM_PROVENANCE_FIELDS:
        values = {row[field] for row in rows}
        if len(values) != 1 or not next(iter(values), "") or "unavailable" in values:
            raise RuntimeError(
                f"cannot publish inconsistent {field} provenance: {values}"
            )

    for input_length, expected_checksum in INPUT_CHECKSUMS.items():
        invalid = sum(
            row["input_checksum"] != expected_checksum
            for row in rows
            if row["input_length"] == str(input_length)
        )
        if invalid:
            raise RuntimeError(
                f"cannot publish: {invalid} rows have noncanonical input checksum "
                f"provenance for {input_length}"
            )

    definition_fields = (
        "indicator_family",
        "indicator_definition",
        "parameters",
        "output_kind",
        "output_arity",
    )
    for case_id in CASE_IDS:
        definitions = {
            tuple(row[field] for field in definition_fields)
            for row in rows
            if row["case_id"] == case_id
        }
        if len(definitions) != 1:
            raise RuntimeError(
                f"cannot publish inconsistent case provenance for {case_id}: "
                f"{definitions}"
            )


def publish(output_dir: Path) -> None:
    raw_path = output_dir / "catalogue-matrix-raw.tsv"
    report_path = output_dir / "catalogue-matrix-report.txt"
    validate_publishable(raw_path)
    if not report_path.is_file():
        raise RuntimeError(f"cannot publish missing generated report {report_path}")
    shutil.copy2(raw_path, PUBLISHED_RAW)
    shutil.copy2(report_path, PUBLISHED_REPORT)
    print(f"published raw rows: {PUBLISHED_RAW}")
    print(f"published human report: {PUBLISHED_REPORT}")


def validate_publication_args(args: argparse.Namespace) -> None:
    if args.publish and args.publish_existing:
        raise SystemExit("--publish and --publish-existing are mutually exclusive")
    if not (args.publish or args.publish_existing):
        return

    canonical_inputs = {
        "--baseline": (args.baseline, DEFAULT_BASELINE),
        "--optimization-evidence": (
            args.optimization_evidence,
            DEFAULT_OPTIMIZATION_EVIDENCE,
        ),
    }
    invalid_inputs = [
        flag
        for flag, (actual, expected) in canonical_inputs.items()
        if actual is None or actual.resolve() != expected.resolve()
    ]
    if invalid_inputs:
        raise SystemExit(
            "publication requires canonical comparison inputs: "
            + ", ".join(invalid_inputs)
        )

    if args.publish:
        if args.case is not None or args.input_length is not None:
            raise SystemExit("--publish requires the complete matrix")
        canonical_timing = {
            "--samples": (args.samples, CANONICAL_SAMPLES),
            "--warmup-ms": (args.warmup_ms, CANONICAL_WARMUP_MS),
            "--sample-ms": (args.sample_ms, CANONICAL_SAMPLE_MS),
        }
        invalid_timing = [
            f"{flag}={actual} (canonical {expected})"
            for flag, (actual, expected) in canonical_timing.items()
            if actual is not None and actual != expected
        ]
        if invalid_timing:
            raise SystemExit(
                "publication rejects noncanonical timing overrides: "
                + ", ".join(invalid_timing)
            )

def validate_publish_existing_args(args: argparse.Namespace) -> None:
    incompatible = {
        "--source-archive": args.source_archive,
        "--case": args.case,
        "--input-length": args.input_length,
        "--samples": args.samples,
        "--warmup-ms": args.warmup_ms,
        "--sample-ms": args.sample_ms,
    }
    supplied = [flag for flag, value in incompatible.items() if value is not None]
    if supplied:
        raise SystemExit(
            "--publish-existing cannot be combined with " + ", ".join(supplied)
        )


def main() -> None:
    args = parse_args()
    validate_publication_args(args)
    if args.publish_existing:
        validate_publish_existing_args(args)
    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.publish_existing:
        generate_report(args, output_dir)
        publish(output_dir)
        return

    python_identity = inspect_python(args.python)
    deps_dir = args.deps_dir.resolve()
    deps_dir.mkdir(parents=True, exist_ok=True)
    archive = checked_archive(args.source_archive, deps_dir)
    talib_install = build_talib(archive, deps_dir)
    python = prepare_python(args.python, python_identity, deps_dir, talib_install)

    environment = os.environ.copy()
    library_dir = str(talib_install / "lib")
    environment["CATALOGUE_TALIB_LIB_DIR"] = library_dir
    environment["TA_INCLUDE_PATH"] = str(talib_install / "include")
    environment["TA_LIBRARY_PATH"] = library_dir
    for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = environment.get(variable)
        environment[variable] = library_dir if not existing else library_dir + os.pathsep + existing
    benchmark_arguments = [
        "cargo",
        "run",
        "--release",
        "-p",
        "ta-benchmarks",
        "--features",
        "catalogue-matrix",
        "--bin",
        "catalogue-matrix",
        "--",
        "--python",
        str(python),
        "--output-dir",
        str(output_dir),
    ]
    for flag, value in (
        ("--case", args.case),
        ("--input-length", args.input_length),
        ("--samples", args.samples),
        ("--warmup-ms", args.warmup_ms),
        ("--sample-ms", args.sample_ms),
    ):
        if value is not None:
            benchmark_arguments.extend((flag, str(value)))
    run(
        benchmark_arguments,
        cwd=REPOSITORY,
        env=environment,
    )
    generate_report(args, output_dir)
    if args.publish:
        publish(output_dir)


if __name__ == "__main__":
    main()
