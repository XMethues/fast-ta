#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output="${1:-${repo_root}/target/qualification/wasm_typprice.jsonl}"
build_root="${repo_root}/target/wasm-typprice-qualification"
scalar_target="${build_root}/scalar"
simd_target="${build_root}/simd128"
scalar_bindings="${build_root}/bindings-scalar"
simd_bindings="${build_root}/bindings-simd128"

if ! command -v node >/dev/null 2>&1; then
  echo "node is required for the WASM runtime qualification" >&2
  exit 1
fi
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen-cli 0.2.126 is required; install it with: cargo install wasm-bindgen-cli --version 0.2.126 --locked" >&2
  exit 1
fi
if [[ "$(wasm-bindgen --version)" != "wasm-bindgen 0.2.126" ]]; then
  echo "wasm-bindgen-cli 0.2.126 is required to match Cargo.lock; found $(wasm-bindgen --version)" >&2
  exit 1
fi

mkdir -p "$(dirname "${output}")" "${scalar_bindings}" "${simd_bindings}"
cd "${repo_root}"

CARGO_TARGET_DIR="${scalar_target}" \
  RUSTFLAGS="-C target-feature=-simd128" \
  cargo build --locked --release --target wasm32-unknown-unknown -p ta-wasm
wasm-bindgen \
  --target nodejs \
  --out-dir "${scalar_bindings}" \
  "${scalar_target}/wasm32-unknown-unknown/release/ta_wasm.wasm"

CARGO_TARGET_DIR="${simd_target}" \
  RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --locked --release --target wasm32-unknown-unknown -p ta-wasm
wasm-bindgen \
  --target nodejs \
  --out-dir "${simd_bindings}" \
  "${simd_target}/wasm32-unknown-unknown/release/ta_wasm.wasm"

if [[ -z "${QUALIFICATION_COMMIT:-}" ]] && command -v git >/dev/null 2>&1; then
  QUALIFICATION_COMMIT="$(git rev-parse HEAD)"
  export QUALIFICATION_COMMIT
fi
node \
  "${repo_root}/crates/ta-wasm/tests/typprice_runtime.mjs" \
  "${scalar_bindings}/ta_wasm.js" \
  "${simd_bindings}/ta_wasm.js" \
  "${output}"
