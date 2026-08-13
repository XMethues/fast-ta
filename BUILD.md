# Build Instructions

## Quick Start

To build the default workspace members (fast-ta, ta-wasm, ta-benchmarks):

```bash
cargo build
```

## Build Individual Crates

### fast-ta (Core Library)

Default (f64):
```bash
cargo build -p fast-ta
```

With f32 precision:
```bash
cargo build -p fast-ta --no-default-features --features f32,std
```

Run tests:
```bash
cargo test -p fast-ta
```

CI enforces `f32` correctness on the supported stable x86_64 Linux host with:

```bash
cargo test -p fast-ta --no-default-features --features f32,std
```

Performance evidence is intentionally host-local and uses the default `f64`
precision unless a benchmark record explicitly says otherwise; CI correctness
coverage is not a portable performance claim.

### ta-py (Python Bindings)

Requires Python 3.x interpreter to build:

```bash
# Only build ta-py crate
cargo build -p ta-py

# Build entire workspace including ta-py (requires Python installed)
cargo build --workspace
```

**Note**: ta-py is excluded from default build (no `--workspace`) because it requires Python.

### ta-wasm (WebAssembly Bindings)

```bash
cargo build -p ta-wasm
```

Run tests:
```bash
cargo test -p ta-wasm
```

### ta-benchmarks (Performance Benchmarks)

```bash
cargo build -p ta-benchmarks
```

Run benchmarks:
```bash
cargo bench -p ta-benchmarks
```

#### Pinned representative Indicator Catalogue matrix (opt-in)

The permanent cross-language tracer is deliberately outside default builds,
ordinary CI, and `cargo bench`; existing Criterion IDs and baselines are
unchanged. From the repository root, one unattended command downloads and
checksum-verifies TA-Lib C 0.6.4, builds revision
`43f9d5042ecc4bd367941846494ad907bf20ea50`, creates an interpreter-keyed isolated
environment for the official `TA-Lib==0.6.4` Python binding and NumPy 2.2.3, then
semantically qualifies every execution path before timing. Select Python 3.10
or newer explicitly when the system `python3` is older. Add `--publish` only
for a clean complete canonical run:

```bash
python3 crates/ta-benchmarks/scripts/run_catalogue_matrix.py \
  --python /path/to/python3.12 \
  --publish
```

The canonical matrix uses 50 timed samples, a 250 ms warm-up, and a 10 ms
target per sample for every variant at 256, 4,096, and 65,536 observations. It
contains `SMA(14)`, `BBANDS(20, 2, 2, SMA)`, `RSI(14)`,
`MACD(12, 26, 9)`, `ATR(14)`, `ADX(14)`, `HT_DCPHASE`, `CDLDOJI`,
`CDLENGULFING`, `CDL3WHITESOLDIERS`, `LINEARREG(14)`, `TYPPRICE`, `OBV`,
`SIN`, and `ADD`. The three Pattern Recognition cases use immutable TA-Lib
default Candle Settings. Each case reports fast-ta Owned Compact Output,
caller-owned
Batch Computation, Prepared Batch Runner, and Streaming Computation separately,
alongside direct caller-owned TA-Lib C and the official Python NumPy API. Only
caller-owned Rust/C pairs are marked comparable and enter same-size geometric
summaries; the report presents the other paths as separate cost indices.

The matrix and report support is feature-gated: default
`cargo check -p ta-benchmarks` builds only the small fixture helper used by
Criterion. Direct report generation must opt in with
`--features catalogue-matrix`; the Python runner supplies that feature for both
executables.

Prerequisites are Python 3.10 or newer with `venv`, a C compiler, `make`, a
POSIX shell, and the usual TA-Lib `configure` build prerequisites. Network
access is needed only for the first dependency preparation. An already
downloaded source archive can be selected with `--source-archive PATH`; it must
match the recorded SHA-256. Source extraction is staged outside the cache and
renamed into place only after it is complete. The extracted tree is reused only
when its `.fast-ta-source-pin` marker records that archive digest; an unmarked
or mismarked tree is discarded rather than treated as a usable source cache.
Dependencies remain under `target/catalogue-matrix/deps`. Every run writes
`target/catalogue-matrix/results/catalogue-matrix-raw.tsv`; its human report is
generated only after rereading those rows. `--publish` rejects noncanonical
sample, warm-up, sample-duration, and baseline inputs, malformed or non-positive
timing evidence, confidence intervals that do not contain their median,
incoherent throughput/outlier counts, and zero iteration counts.
It then requires every expected case × input length × implementation/mode cell
with one consistent pinned provenance before copying the stable review paths:

```text
crates/ta-benchmarks/baselines/catalogue_matrix_optimized.tsv
crates/ta-benchmarks/CATALOGUE_MATRIX_REPORT.txt
```

The canonical comparison input is
`baselines/catalogue_matrix_pre_optimization.tsv`, a clean complete 270-row
pre-optimization matrix. The historical
`baselines/catalogue_matrix_post_scalar_diagnostic.tsv` is retained only as a
dirty historical diagnostic and is not read by the canonical report generator.
Durable ticket hypotheses, pre-change sampled/objdump evidence, exact commands,
Criterion before/after measurements, semantic Rust/C pairs, and neighboring
dispositions come from `baselines/issue_57_62_diagnostic_evidence.json` and
`baselines/issue_57_62_criterion_diagnostics.json`. The issue 61 clean-revert
control comes from `baselines/issue61_cycle_regression.jsonl`. The renderer
derives every numeric table from these artifacts or the two clean raw matrices;
the older `catalogue_matrix_optimization_evidence.tsv` is not a report input.

To regenerate the committed report only from committed evidence artifacts:

```bash
cargo run --release -p ta-benchmarks --features catalogue-matrix \
  --bin catalogue-report -- \
  --raw crates/ta-benchmarks/baselines/catalogue_matrix_optimized.tsv \
  --baseline crates/ta-benchmarks/baselines/catalogue_matrix_pre_optimization.tsv \
  --diagnostic-evidence crates/ta-benchmarks/baselines/issue_57_62_diagnostic_evidence.json \
  --criterion-diagnostics crates/ta-benchmarks/baselines/issue_57_62_criterion_diagnostics.json \
  --cycle-regression crates/ta-benchmarks/baselines/issue61_cycle_regression.jsonl \
  --platform-qualification crates/ta-benchmarks/baselines/typprice_x86_f64_qualification.jsonl \
  --platform-qualification crates/ta-benchmarks/baselines/typprice_x86_f32_qualification.jsonl \
  --platform-qualification crates/ta-benchmarks/baselines/typprice_aarch64_f64_qualification.jsonl \
  --platform-qualification crates/ta-benchmarks/baselines/typprice_aarch64_f32_qualification.jsonl \
  --platform-qualification crates/ta-benchmarks/baselines/typprice_wasm_qualification.jsonl \
  --report crates/ta-benchmarks/CATALOGUE_MATRIX_REPORT.txt
```

The platform JSONL inputs are runtime evidence whose metadata records are the
source of truth for the current workflow run/job, source commit, precision,
release profile, feature set, OS, CPU, runtime, and active backend. The report
parser derives provenance from those current artifacts rather than relying on a
run ID or commit copied into this document. Every artifact must also contain
successful validation records before its scalar-equivalence measurements can
be reported. The renderer preserves per-size medians, confidence intervals,
throughput, speedup, and timed boundaries. Public validated timings are only
compared with matching scalar rows at the same boundary and precision; WASM
rows cover the public `wasm-bindgen` TYPPRICE boundary.
Output Range, per-column count, float values, and exact Pattern Signal codes are
checked before any catalogue row is timed. Mismatches suppress that case's
timings and fail the command.

For focused diagnosis, run one case at a time. Each command below exercises all
three canonical input lengths while reusing the pinned dependency cache:

```bash
RUNNER=crates/ta-benchmarks/scripts/run_catalogue_matrix.py
python3 "$RUNNER" --python /path/to/python3.12 --case ADX
python3 "$RUNNER" --python /path/to/python3.12 --case MACD
python3 "$RUNNER" --python /path/to/python3.12 --case CDLDOJI
python3 "$RUNNER" --python /path/to/python3.12 --case CDLENGULFING
python3 "$RUNNER" --python /path/to/python3.12 --case CDL3WHITESOLDIERS
python3 "$RUNNER" --python /path/to/python3.12 --case ATR
python3 "$RUNNER" --python /path/to/python3.12 --case HT_DCPHASE
python3 "$RUNNER" --python /path/to/python3.12 --case TYPPRICE
```

Each input length prints an independent `FOCUSED VERDICT`. `PASS` means all six
execution paths passed the semantic gate, both comparable timings exist, and
the same-run caller-owned fast-ta median divided by the direct caller-owned
TA-Lib C median is at most `1.050x` (the boundary passes). `FAIL` means semantic
verification failed, either comparable timing is unavailable, or that ratio is
greater than `1.050x`. Owned, Prepared, Streaming, and Python timings are
reported but do not enter this verdict. Float cases use the matrix's absolute
`1e-9` and relative `1e-12` tolerances; Pattern cases require exact integer
signals. A performance `FAIL` is a diagnosis printed by a successfully measured
command, not a nonzero process status; inspect every verdict line. Semantic or
measurement failures do return nonzero after recording their reasons. Focused
runs are diagnostic only and cannot be combined with `--publish`.

## All Workspace Members

To build all 4 crates (requires Python installed for ta-py):

```bash
cargo build --workspace
```

## Workspace Structure

```
rs-indicators/
├── Cargo.toml (workspace root)
└── crates/
    ├── ta-core/         # Core library (no_std)
    ├── ta-py/         # Python bindings (PyO3)
    ├── ta-wasm/       # WebAssembly bindings (wasm-bindgen)
    └── ta-benchmarks/ # Performance benchmarks (Criterion)
```

## Default Members

The following crates are built by default (without `--workspace`):
- fast-ta
- ta-wasm
- ta-benchmarks

ta-py is excluded from default build because it requires Python 3.x interpreter.
