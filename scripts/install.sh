#!/usr/bin/env bash
set -euo pipefail

# Build and install `cachepin` locally with zero external dependencies.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${CACHEPIN_PREFIX:-$HOME/.local/bin}"

echo "[cachepin] Building release binary and C-ABI shared library (zero external crates)..."
cargo build --release --manifest-path "$ROOT/Cargo.toml"

mkdir -p "$PREFIX"
cp "$ROOT/target/release/cachepin" "$PREFIX/cachepin"
chmod +x "$PREFIX/cachepin"

echo "[cachepin] Installed binary to $PREFIX/cachepin"
"$PREFIX/cachepin" --version
echo ""
echo "Quick start:"
echo "  cachepin audit ./trace.jsonl"
echo "  cachepin proxy --upstream http://127.0.0.1:8000 --port 47820"
echo "  cachepin mcp"
