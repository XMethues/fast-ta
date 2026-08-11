#!/usr/bin/env python3
"""Capture same-session Criterion evidence for optimization issues 57 through 62."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
from typing import Any

PRE_COMMIT = "b156ac1ec5c781af0a386c4deab4769a4c5479c1"
DEFAULT_SAMPLES = 31
DEFAULT_WARMUP_SECONDS = 0.5
DEFAULT_MEASUREMENT_SECONDS = 1.0


@dataclass(frozen=True)
class BenchmarkSpec:
    ticket: int
    case_id: str
    role: str
    source_scope: str
    bench: str
    criterion_id: str


EXECUTION = "execution_baselines"
PATTERN = "pattern_recognition_baselines"
BENCHMARKS = (
    BenchmarkSpec(57, "ADX", "target", "specialized ADX batch state", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/ADX/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "PLUS_DM", "neighbor", "Directional Movement control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/PLUS_DM/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "MINUS_DM", "neighbor", "Directional Movement control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/MINUS_DM/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "PLUS_DI", "neighbor", "Directional Movement consumer control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/PLUS_DI/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "MINUS_DI", "neighbor", "Directional Movement consumer control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/MINUS_DI/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "DX", "neighbor", "Directional Movement consumer control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/DX/one_shot/caller_compact/65536"),
    BenchmarkSpec(57, "ADXR", "neighbor", "ADX-family consumer control", EXECUTION, "indicator_execution/expanded/momentum/directional_movement/ADXR/one_shot/caller_compact/65536"),
    BenchmarkSpec(58, "MACD", "target", "fused all-EMA MACD batch path", EXECUTION, "indicator_execution/expanded/moving_average_momentum/MACD/caller_owned/65536"),
    BenchmarkSpec(58, "EMA", "neighbor", "shared EMA control", EXECUTION, "indicator_execution/expanded/recursive_overlap/EMA/current_caller_compact/n=65536/period=14"),
    BenchmarkSpec(58, "MACDEXT", "neighbor", "default all-EMA MACDEXT prepared workload relative", EXECUTION, "indicator_execution/expanded/moving_average_momentum_workloads/MACDEXT/per_worker/prepared_runners"),
    BenchmarkSpec(59, "CDLDOJI/default", "target", "single-setting default shape", PATTERN, "pattern_recognition/throughput/CDLDOJI/default/caller_owned/65536"),
    BenchmarkSpec(59, "CDLDOJI/period_200", "neighbor", "single-setting custom-period shape", PATTERN, "pattern_recognition/throughput/CDLDOJI/period_200/caller_owned/65536"),
    BenchmarkSpec(59, "CDLENGULFING/default", "target", "setting-free cross-candle shape", PATTERN, "pattern_recognition/throughput/CDLENGULFING/default/caller_owned/65536"),
    BenchmarkSpec(59, "CDL3WHITESOLDIERS/default", "target", "multi-setting default shape", PATTERN, "pattern_recognition/throughput/CDL3WHITESOLDIERS/default/caller_owned/65536"),
    BenchmarkSpec(59, "CDL3WHITESOLDIERS/period_200", "neighbor", "multi-setting custom-period shape", PATTERN, "pattern_recognition/throughput/CDL3WHITESOLDIERS/period_200/caller_owned/65536"),
    BenchmarkSpec(60, "ATR", "target", "aligned True Range iterator and Wilder recurrence", EXECUTION, "indicator_execution/expanded/volatility/ATR/current_caller_compact/65536"),
    BenchmarkSpec(60, "TRANGE", "neighbor", "shared aligned True Range seam", EXECUTION, "indicator_execution/expanded/volatility/TRANGE/current_caller_compact/65536"),
    BenchmarkSpec(60, "NATR", "neighbor", "related normalized volatility control", EXECUTION, "indicator_execution/expanded/volatility/NATR/current_caller_compact/65536"),
    BenchmarkSpec(61, "HT_DCPHASE", "target", "Hilbert cursor-wrap state", EXECUTION, "indicator_execution/expanded/cycle/HT_DCPHASE/one_shot/caller_compact/65536"),
    BenchmarkSpec(61, "HT_DCPERIOD", "neighbor", "shared Hilbert transition", EXECUTION, "indicator_execution/expanded/cycle/HT_DCPERIOD/one_shot/caller_compact/65536"),
    BenchmarkSpec(61, "HT_PHASOR", "neighbor", "shared Hilbert transition", EXECUTION, "indicator_execution/expanded/cycle/HT_PHASOR/one_shot/caller_compact/65536"),
    BenchmarkSpec(61, "HT_SINE", "neighbor", "shared Hilbert phase transition", EXECUTION, "indicator_execution/expanded/cycle/HT_SINE/one_shot/caller_compact/65536"),
    BenchmarkSpec(61, "HT_TRENDMODE", "neighbor", "shared Hilbert transition", EXECUTION, "indicator_execution/expanded/cycle/HT_TRENDMODE/one_shot/caller_compact/65536"),
    BenchmarkSpec(62, "TYPPRICE/4096", "target", "public AArch64 dispatch path", EXECUTION, "indicator_execution/expanded/price_transform/TYPPRICE/current_caller_compact/4096"),
    BenchmarkSpec(62, "TYPPRICE/65536", "target", "public AArch64 dispatch path", EXECUTION, "indicator_execution/expanded/price_transform/TYPPRICE/current_caller_compact/65536"),
    BenchmarkSpec(62, "AVGPRICE", "neighbor", "Price Transform control", EXECUTION, "indicator_execution/expanded/price_transform/AVGPRICE/current_caller_compact/65536"),
    BenchmarkSpec(62, "MEDPRICE", "neighbor", "Price Transform control", EXECUTION, "indicator_execution/expanded/price_transform/MEDPRICE/current_caller_compact/65536"),
    BenchmarkSpec(62, "WCLPRICE", "neighbor", "Price Transform control", EXECUTION, "indicator_execution/expanded/price_transform/WCLPRICE/current_caller_compact/65536"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-worktree", type=Path, required=True)
    parser.add_argument("--final-worktree", type=Path, required=True)
    parser.add_argument("--pre-target-dir", type=Path, required=True)
    parser.add_argument("--final-target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    parser.add_argument("--measurement-seconds", type=float, default=DEFAULT_MEASUREMENT_SECONDS)
    parser.add_argument("--ticket", type=int, choices=range(57, 63), action="append")
    args = parser.parse_args()
    if args.samples < 31:
        parser.error("--samples must be at least 31")
    if args.warmup_seconds <= 0 or args.measurement_seconds <= 0:
        parser.error("timing durations must be positive")
    return args


def checked_output(arguments: list[str], cwd: Path) -> str:
    return subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def revision_provenance(worktree: Path) -> dict[str, Any]:
    commit = checked_output(["git", "rev-parse", "HEAD"], worktree)
    dirty_lines = checked_output(["git", "status", "--porcelain"], worktree).splitlines()
    return {
        "worktree": str(worktree.resolve()),
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_paths": dirty_lines,
    }


def environment_provenance() -> dict[str, Any]:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    return {
        "rustc": subprocess.run(["rustc", "--version", "--verbose"], check=True, capture_output=True, text=True).stdout.strip(),
        "host": subprocess.run(["uname", "-a"], check=True, capture_output=True, text=True).stdout.strip(),
        "cpu": cpu,
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "criterion": "0.8.2",
        "profile": "bench",
        "ta_core_features": "default(f64,std)",
        "float_bits": 64,
    }


def find_criterion_result(target_dir: Path, full_id: str) -> Path:
    candidates = []
    for metadata_path in (target_dir / "criterion").rglob("new/benchmark.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("full_id") == full_id:
            candidates.append(metadata_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Criterion result for {full_id!r}, found {len(candidates)}")
    return candidates[0]


def estimate_record(estimate: dict[str, Any]) -> dict[str, Any]:
    interval = estimate["confidence_interval"]
    return {
        "point_estimate_ns": estimate["point_estimate"],
        "ci95_lower_ns": interval["lower_bound"],
        "ci95_upper_ns": interval["upper_bound"],
        "confidence_level": interval["confidence_level"],
        "standard_error_ns": estimate["standard_error"],
    }


def run_benchmark(
    spec: BenchmarkSpec,
    revision: str,
    worktree: Path,
    target_dir: Path,
    samples: int,
    warmup_seconds: float,
    measurement_seconds: float,
) -> dict[str, Any]:
    arguments = [
        "cargo", "bench", "-p", "ta-benchmarks", "--bench", spec.bench, "--",
        spec.criterion_id,
        "--sample-size", str(samples),
        "--warm-up-time", str(warmup_seconds),
        "--measurement-time", str(measurement_seconds),
        "--noplot",
    ]
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_dir.resolve())
    completed = subprocess.run(
        arguments,
        cwd=worktree,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result_dir = find_criterion_result(target_dir, spec.criterion_id)
    benchmark = json.loads((result_dir / "benchmark.json").read_text(encoding="utf-8"))
    estimates = json.loads((result_dir / "estimates.json").read_text(encoding="utf-8"))
    sample = json.loads((result_dir / "sample.json").read_text(encoding="utf-8"))
    throughput = benchmark["throughput"]
    observations = throughput.get("Elements") if isinstance(throughput, dict) else None
    median = estimate_record(estimates["median"])
    return {
        "revision": revision,
        "ticket": spec.ticket,
        "case_id": spec.case_id,
        "role": spec.role,
        "source_scope": spec.source_scope,
        "benchmark_id": spec.criterion_id,
        "timed_boundary": "public caller-owned or prepared path named by benchmark_id; fixture and output allocation excluded by existing Criterion harness",
        "command": shlex.join(arguments),
        "cwd": str(worktree.resolve()),
        "environment_overrides": {"CARGO_TARGET_DIR": str(target_dir.resolve())},
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "sample_count": len(sample["times"]),
        "sampling_mode": sample["sampling_mode"],
        "iterations": sample["iters"],
        "elapsed_ns": sample["times"],
        "median": median,
        "slope": None if estimates["slope"] is None else estimate_record(estimates["slope"]),
        "observations_per_iteration": observations,
        "throughput_observations_per_second": None if observations is None else observations * 1.0e9 / median["point_estimate_ns"],
    }


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return max(left["ci95_lower_ns"], right["ci95_lower_ns"]) <= min(left["ci95_upper_ns"], right["ci95_upper_ns"])


def disposition(spec: BenchmarkSpec, change_percent: float, overlap: bool) -> str:
    if abs(change_percent) <= 5.0:
        return "within_5_percent_noise_gate"
    direction = "improvement" if change_percent < 0 else "regression"
    if overlap:
        return f">5_percent_{direction}_point_estimate_with_overlapping_95_percent_CIs; timing_variation_not_practical_change"
    if spec.case_id.startswith("CDLDOJI") and direction == "regression":
        return ">5_percent_confirmed_regression; source_correlated_with_shared_single_setting_helper_migration"
    if direction == "improvement":
        return f">5_percent_confirmed_improvement; source_scope={spec.source_scope}"
    return f">5_percent_confirmed_regression; source_scope={spec.source_scope}; requires_explicit_followup"


def compare(spec: BenchmarkSpec, pre: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    pre_median = pre["median"]["point_estimate_ns"]
    final_median = final["median"]["point_estimate_ns"]
    change_percent = (final_median / pre_median - 1.0) * 100.0
    overlap = intervals_overlap(pre["median"], final["median"])
    return {
        "ticket": spec.ticket,
        "case_id": spec.case_id,
        "role": spec.role,
        "pre_median_ns": pre_median,
        "final_median_ns": final_median,
        "final_over_pre": final_median / pre_median,
        "speedup": pre_median / final_median,
        "change_percent": change_percent,
        "ci95_overlap": overlap,
        "noise_gate_percent": 5.0,
        "disposition": disposition(spec, change_percent, overlap),
    }


def main() -> None:
    args = parse_args()
    pre = revision_provenance(args.pre_worktree)
    final = revision_provenance(args.final_worktree)
    if pre["commit"] != PRE_COMMIT:
        raise SystemExit(f"pre worktree must be {PRE_COMMIT}, found {pre['commit']}")
    tickets = set(args.ticket or range(57, 63))
    specs = [spec for spec in BENCHMARKS if spec.ticket in tickets]
    measurements = []
    comparisons = []
    for spec in specs:
        pre_measurement = run_benchmark(spec, "pre", args.pre_worktree, args.pre_target_dir, args.samples, args.warmup_seconds, args.measurement_seconds)
        final_measurement = run_benchmark(spec, "final", args.final_worktree, args.final_target_dir, args.samples, args.warmup_seconds, args.measurement_seconds)
        measurements.extend((pre_measurement, final_measurement))
        comparisons.append(compare(spec, pre_measurement, final_measurement))
    record = {
        "schema": "fast-ta.issue-57-62.criterion-diagnostics.v1",
        "pre_revision": pre,
        "final_revision": final,
        "environment": environment_provenance(),
        "configuration": {
            "samples": args.samples,
            "warmup_seconds": args.warmup_seconds,
            "measurement_seconds": args.measurement_seconds,
            "confidence_interval": "Criterion deterministic bootstrap 95% interval for median",
            "noise_gate_percent": 5.0,
            "interleaving": "one pre measurement followed immediately by one final measurement for each benchmark",
        },
        "benchmark_specs": [asdict(spec) for spec in specs],
        "measurements": measurements,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
