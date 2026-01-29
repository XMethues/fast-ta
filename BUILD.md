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
