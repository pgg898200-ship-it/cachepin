#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Zero-dependency check (Cargo.toml has no external crates) ==="
if grep -qE '^\[dependencies\]' Cargo.toml; then
    echo "FAIL: unexpected [dependencies] section in Cargo.toml"
    exit 1
fi
echo "OK: pure zero-dependency Rust"

echo "=== 2. Rust release build & unit/integration tests (-D warnings) ==="
RUSTFLAGS="-D warnings" cargo test --release
RUSTFLAGS="-D warnings" cargo build --release

echo "=== 3. Binary cold-start smoke & CLI surface ==="
./target/release/cachepin --version
./target/release/cachepin models >/dev/null
./target/release/cachepin diff --help >/dev/null

echo "=== 4. Adversarial red-team & Python SDK suite ==="
python3 tests/adversarial_suite.py

echo "=== 5. Cross-language matched-control benchmarks ==="
python3 benchmarks/compare_controls.py >/dev/null
test -s benchmarks/results.json
echo "OK: benchmarks/results.json generated"

echo "=== 6. Local path & secret hygiene check ==="
if grep -rnEI --exclude-dir=__pycache__ '(/home/[a-zA-Z0-9_-]+|/Users/[a-zA-Z0-9_-]+|sk-ant-api[0-9a-zA-Z_-]{10,}|sk-proj-[0-9a-zA-Z_-]{10,})' src/ python/ benchmarks/ tests/ docs/ examples/ site/ mcpb/ Formula/ Dockerfile Dockerfile.release flake.nix server.json README.md SKILL.md llms.txt Cargo.toml pyproject.toml 2>/dev/null; then
    echo "FAIL: local path or API key leaked into standalone repo"
    exit 1
fi
echo "OK: clean standalone repository (zero local paths or secrets)"
echo "=== ALL CHECKS PASSED ==="
