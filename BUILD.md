# Build Instructions

## Quick Start

To build the default workspace members (ta-core, ta-wasm, ta-benchmarks):

```bash
cargo build
```

## Build Individual Crates

### ta-core (Core Library)

Default (f64):
```bash
cargo build -p ta-core
```

With f32 precision:
```bash
cargo build -p ta-core --features f32
```

Run tests:
```bash
cargo test -p ta-core
```

CI enforces `f32` correctness on the supported stable x86_64 Linux host with:

```bash
cargo test -p ta-core --no-default-features --features f32,std
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
semantically qualifies every execution path before timing:

```bash
python3 crates/ta-benchmarks/scripts/run_catalogue_matrix.py
```

The matrix runs at 256, 4,096, and 65,536 observations. It contains `SMA(14)`,
`BBANDS(20, 2, 2, SMA)`, `RSI(14)`, `MACD(12, 26, 9)`, `ATR(14)`, `ADX(14)`,
`HT_DCPHASE`, `CDLDOJI`, `CDLENGULFING`, `CDL3BLACKCROWS`, `LINEARREG(14)`,
`TYPPRICE`, `OBV`, `SIN`, and `ADD`. The three Pattern Recognition cases use
immutable TA-Lib-default Candle Settings. Each case reports fast-ta Owned Compact
Output, caller-owned Batch Computation, Prepared Batch Runner, and Streaming
Computation separately, alongside direct caller-owned TA-Lib C and the official
Python NumPy API. Only caller-owned Rust/C pairs are marked comparable and enter
same-size geometric summaries; other execution costs are explicitly unavailable
for that aggregate.

Prerequisites are Python 3.10 or newer with `venv`, a C compiler, `make`, a
POSIX shell, and the usual TA-Lib `configure` build prerequisites. Network access
is needed only for the first dependency preparation. An already downloaded
source archive can be selected with `--source-archive PATH`; it must match the
recorded SHA-256. Dependencies remain under `target/catalogue-matrix/deps`.
Successful and semantic-failure runs write stable machine-readable rows to
`target/catalogue-matrix/results/catalogue-matrix-raw.tsv`; the human report at
`catalogue-matrix-report.txt` is generated only after rereading those rows.
Output Range, per-column count, float values, and exact Pattern Signal codes are
checked before any row for a case is timed. Mismatches suppress that case's
timings, retain a precise reason in both artifacts, and make the command fail.
A dirty run is labeled diagnostic; only a clean successful run is a canonical
baseline candidate.

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
- ta-core
- ta-wasm
- ta-benchmarks

ta-py is excluded from default build because it requires Python 3.x interpreter.
