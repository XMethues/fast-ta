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

#### Pinned SMA Rust/C/Python comparison (opt-in)

The permanent three-way tracer is deliberately outside default builds, ordinary
CI, and `cargo bench`. From the repository root, one unattended command
downloads and checksum-verifies TA-Lib C 0.6.4, builds revision
`43f9d5042ecc4bd367941846494ad907bf20ea50`, creates an isolated environment for
the official `TA-Lib==0.6.4` Python binding and pinned NumPy, then validates and
times all three implementations:

```bash
python3 crates/ta-benchmarks/scripts/run_sma_three_way.py
```
Prerequisites are Python 3.10 or newer with `venv`, a C compiler, `make`, a
POSIX shell, and the usual TA-Lib `configure` build prerequisites. Network access
is needed only
for the first dependency preparation. An already downloaded source archive can
be selected with `--source-archive PATH`; it must match the recorded SHA-256.
Dependencies remain under `target/sma-three-way/deps`. Successful runs write
machine-readable raw rows and the report generated from those rows under
`target/sma-three-way/results`. A semantic Output Range, count, version, or value
mismatch exits before timing and removes any stale raw/report files.

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
