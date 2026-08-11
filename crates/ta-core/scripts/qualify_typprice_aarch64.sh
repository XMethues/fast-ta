#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
output_dir="${1:-${repo_root}/target/qualification/aarch64}"
deps_dir="${QUALIFICATION_DEPS_DIR:-${repo_root}/target/catalogue-matrix/deps}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "AArch64 TYPPRICE qualification requires a native macOS arm64 host" >&2
  exit 1
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
mkdir -p "${deps_dir}"
deps_dir="$(cd "${deps_dir}" && pwd)"
cd "${repo_root}"

export QUALIFICATION_COMMIT="${QUALIFICATION_COMMIT:-$(git rev-parse HEAD)}"
export QUALIFICATION_RUNTIME="${QUALIFICATION_RUNTIME:-$(rustc --version --verbose | tr '\n' ';')}"
export QUALIFICATION_OS="${QUALIFICATION_OS:-$(sw_vers | tr '\n' ';')$(uname -srv)}"
export QUALIFICATION_CPU="${QUALIFICATION_CPU:-$(sysctl -n machdep.cpu.brand_string)}"
export QUALIFICATION_CPU_FEATURES="${QUALIFICATION_CPU_FEATURES:-$(sysctl -a 2>/dev/null | sed -n '/^hw\.optional\./p' | sort | tr '\n' ';')}"

# Reuse the catalogue runner's checksum-verified TA-Lib 0.6.4 builder rather
# than introducing a second native reference pin or installation convention.
talib_path_file="$(mktemp "${TMPDIR:-/tmp}/fast-ta-talib-path.XXXXXX")"
trap 'rm -f "${talib_path_file}"' EXIT
python3 - "${repo_root}" "${deps_dir}" "${talib_path_file}" <<'PY'
from pathlib import Path
import sys

repository = Path(sys.argv[1])
deps = Path(sys.argv[2])
destination = Path(sys.argv[3])
sys.path.insert(0, str(repository / "crates" / "ta-benchmarks" / "scripts"))
import run_catalogue_matrix as catalogue

archive = catalogue.checked_archive(None, deps)
install = catalogue.build_talib(archive, deps)
library = install / "lib" / "libta-lib.dylib"
if not library.is_file():
    raise SystemExit(f"pinned TA-Lib build did not produce {library}")
destination.write_text(f"{library.resolve()}\n", encoding="utf-8")
PY
talib_library="$(cat "${talib_path_file}")"
rm -f "${talib_path_file}"
trap - EXIT

export QUALIFICATION_TALIB_LIBRARY="${talib_library}"
export QUALIFICATION_OUTPUT="${output_dir}/aarch64_typprice_f64.jsonl"
export QUALIFICATION_FEATURES="ta-core=f64,std,simd-qualification"
export QUALIFICATION_COMMAND="cargo test --locked -p ta-core --release --no-default-features --features f64,std,simd-qualification --test aarch64_typprice_qualification -- qualify_public_typprice_on_aarch64 --exact --ignored --nocapture"
cargo test --locked -p ta-core \
  --release \
  --no-default-features \
  --features f64,std,simd-qualification \
  --test aarch64_typprice_qualification \
  -- qualify_public_typprice_on_aarch64 --exact --ignored --nocapture

unset QUALIFICATION_TALIB_LIBRARY
export QUALIFICATION_OUTPUT="${output_dir}/aarch64_typprice_f32.jsonl"
export QUALIFICATION_FEATURES="ta-core=f32,std,simd-qualification"
export QUALIFICATION_COMMAND="cargo test --locked -p ta-core --release --no-default-features --features f32,std,simd-qualification --test aarch64_typprice_qualification -- qualify_public_typprice_on_aarch64 --exact --ignored --nocapture"
cargo test --locked -p ta-core \
  --release \
  --no-default-features \
  --features f32,std,simd-qualification \
  --test aarch64_typprice_qualification \
  -- qualify_public_typprice_on_aarch64 --exact --ignored --nocapture
