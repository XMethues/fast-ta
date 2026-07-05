# Project Overview

`fast-ta` is a Rust Cargo workspace for technical-analysis indicators. `ta-core` owns pure indicator/SIMD logic; `ta-py`, `ta-wasm`, and `ta-benchmarks` are adapter/tooling crates around that core.

# Architecture

```text
fast-ta/
├── Cargo.toml                 # workspace, shared deps, default members
└── crates/
    ├── ta-core/               # algorithms, Float, Indicator, TalibError, SIMD
    ├── ta-py/                 # PyO3 extension module adapter
    ├── ta-wasm/               # wasm-bindgen adapter
    └── ta-benchmarks/         # Criterion benchmark crate
```

Dependency flow:

```text
ta-py ───────┐
ta-wasm ─────┼──> ta-core
ta-benchmarks┘
```

`ta-core` must remain independent of Python, WASM, and benchmarking concerns. Bindings convert boundary inputs/errors into their host runtime forms.

# Commands

| Command | What it does |
|---|---|
| `cargo check --workspace` | Check all workspace crates, including `ta-py` |
| `cargo build` | Build default members (`ta-core`, `ta-wasm`, `ta-benchmarks`) |
| `cargo build --workspace` | Build all crates; requires Python for `ta-py` |
| `cargo test --workspace` | Run workspace tests; `ta-py` lib harness is disabled |
| `cargo test -p ta-core --features f32` | Verify feature-gated `Float = f32` behavior |
| `cargo build -p ta-py` | Build the Python extension crate |
| `cargo build -p ta-wasm` | Build the WASM binding crate |
| `cargo bench -p ta-benchmarks` | Run Criterion benchmarks |
| `cargo fmt --all -- --check` | Verify Rust formatting |

# Business Context

The project is building a TA-Lib-style indicator core with native Rust APIs plus Python and WebAssembly distribution paths.

<important if="you are exposing a new indicator end-to-end">
1. Implement the indicator in `ta-core` first; see `.rpiv/guidance/crates/ta-core/architecture.md`.
2. Add Python bindings only as adapter code; see `.rpiv/guidance/crates/ta-py/architecture.md`.
3. Add WASM exports only as adapter code; see `.rpiv/guidance/crates/ta-wasm/architecture.md`.
4. Add performance coverage in Criterion; see `.rpiv/guidance/crates/ta-benchmarks/architecture.md`.
5. Run `cargo check --workspace`, `cargo test --workspace`, and any binding-specific build/test command.
</important>

<important if="you are changing workspace membership or build defaults">
- Keep `ta-core`, `ta-wasm`, and `ta-benchmarks` as default members for normal `cargo build`.
- `ta-py` may stay outside `default-members` because it requires Python/PyO3 extension linking.
- If adding a crate under `crates/`, add both its manifest and its relationship to the dependency-flow rule above.
</important>
