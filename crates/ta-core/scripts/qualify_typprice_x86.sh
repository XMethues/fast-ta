#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_dir="${1:-${repo_root}/target/qualification}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "x86 TYPPRICE qualification requires an x86_64 host" >&2
  exit 1
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
cd "${repo_root}"
export QUALIFICATION_COMMIT="${QUALIFICATION_COMMIT:-$(git rev-parse HEAD)}"
export QUALIFICATION_RUNTIME="${QUALIFICATION_RUNTIME:-$(rustc --version)}"
export QUALIFICATION_OS="${QUALIFICATION_OS:-$(uname -s) $(uname -r)}"
export QUALIFICATION_ARCHITECTURE="${QUALIFICATION_ARCHITECTURE:-$(uname -m)}"
export QUALIFICATION_RUST_PROFILE="release"
export QUALIFICATION_TARGET_FEATURES="${QUALIFICATION_TARGET_FEATURES:-runtime-detected:+avx2,+avx512f;per-function:+avx2,+avx512f;RUSTFLAGS=${RUSTFLAGS:-<unset>}}"
export QUALIFICATION_WORKFLOW_RUN_ID="${GITHUB_RUN_ID:-local}"
export QUALIFICATION_WORKFLOW_JOB="${GITHUB_JOB:-local-x86-typprice}"
if [[ -n "${GITHUB_RUN_ID:-}" && -n "${GITHUB_SERVER_URL:-}" && -n "${GITHUB_REPOSITORY:-}" ]]; then
  export QUALIFICATION_WORKFLOW_RUN_URL="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
else
  export QUALIFICATION_WORKFLOW_RUN_URL="local://crates/ta-core/scripts/qualify_typprice_x86.sh"
fi
if command -v lscpu >/dev/null 2>&1; then
  export QUALIFICATION_CPU="${QUALIFICATION_CPU:-$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | sed -n '1p')}"
else
  export QUALIFICATION_CPU="${QUALIFICATION_CPU:-unknown x86_64 CPU}"
fi

export QUALIFICATION_OUTPUT="${output_dir}/x86_typprice_f64.jsonl"
export QUALIFICATION_CARGO_FEATURES="f64,std,simd-qualification"
export QUALIFICATION_COMMAND="cargo test --locked -p ta-core --release --no-default-features --features f64,std,simd-qualification --test x86_typprice_qualification -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture"
cargo test --locked -p ta-core \
  --release \
  --no-default-features \
  --features f64,std,simd-qualification \
  --test x86_typprice_qualification \
  -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture

export QUALIFICATION_OUTPUT="${output_dir}/x86_typprice_f32.jsonl"
export QUALIFICATION_CARGO_FEATURES="f32,std,simd-qualification"
export QUALIFICATION_COMMAND="cargo test --locked -p ta-core --release --no-default-features --features f32,std,simd-qualification --test x86_typprice_qualification -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture"
cargo test --locked -p ta-core \
  --release \
  --no-default-features \
  --features f32,std,simd-qualification \
  --test x86_typprice_qualification \
  -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture
