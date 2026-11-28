#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cmake -S "$root/tests/cmake" -B "$work/vendored" -DCUTSTREAM_SOURCE_DIR="$root"
cmake --build "$work/vendored"
ctest --test-dir "$work/vendored" --output-on-failure

cmake -S "$root" -B "$work/package" -DCMAKE_INSTALL_PREFIX="$work/stage"
cmake --install "$work/package"
mv "$work/stage" "$work/relocated"
cmake -S "$root/tests/cmake" -B "$work/installed" -DCMAKE_PREFIX_PATH="$work/relocated"
cmake --build "$work/installed"
ctest --test-dir "$work/installed" --output-on-failure
