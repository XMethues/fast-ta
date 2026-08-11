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
if command -v lscpu >/dev/null 2>&1; then
  export QUALIFICATION_CPU="${QUALIFICATION_CPU:-$(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | sed -n '1p')}"
else
  export QUALIFICATION_CPU="${QUALIFICATION_CPU:-unknown x86_64 CPU}"
fi

export QUALIFICATION_OUTPUT="${output_dir}/x86_typprice_f64.jsonl"
cargo test --locked -p ta-core \
  --release \
  --features simd-qualification \
  --test x86_typprice_qualification \
  -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture

export QUALIFICATION_OUTPUT="${output_dir}/x86_typprice_f32.jsonl"
cargo test --locked -p ta-core \
  --release \
  --features f32,std,simd-qualification \
  --test x86_typprice_qualification \
  -- qualify_public_typprice_on_x86_simd --exact --ignored --nocapture
