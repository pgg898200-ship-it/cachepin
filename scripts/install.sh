#!/usr/bin/env bash
set -euo pipefail

# Install `cachepin` from a GitHub Release tarball (with SHA-256 verification)
# or build locally from source with zero external Rust crates.
#
# Examples:
#   ./scripts/install.sh
#   CACHEPIN_REPO=cachepin-dev/cachepin CACHEPIN_VERSION=v0.1.0 curl -fsSL ... | sh

PREFIX="${CACHEPIN_PREFIX:-$HOME/.local/bin}"
REPO="${CACHEPIN_REPO:-}"
VERSION="${CACHEPIN_VERSION:-latest}"

detect_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}:${arch}" in
        Linux:x86_64)        echo "x86_64-unknown-linux-gnu" ;;
        Linux:aarch64|Linux:arm64) echo "aarch64-unknown-linux-gnu" ;;
        Darwin:arm64|Darwin:aarch64) echo "aarch64-apple-darwin" ;;
        Darwin:x86_64)       echo "x86_64-apple-darwin" ;;
        *)                   echo "" ;;
    esac
}

install_from_release() {
    local target="$1"
    local tag="$VERSION"
    if [ "$tag" = "latest" ]; then
        tag="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
            | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    fi
    if [ -z "$tag" ]; then
        echo "[cachepin] Could not resolve latest release tag for ${REPO}." >&2
        return 1
    fi
    local tarball="cachepin-${tag}-${target}.tar.gz"
    local base="https://github.com/${REPO}/releases/download/${tag}"
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    echo "[cachepin] Downloading ${tarball}..."
    curl -fsSL "${base}/${tarball}" -o "${tmpdir}/${tarball}"
    if curl -fsSL "${base}/SHA256SUMS" -o "${tmpdir}/SHA256SUMS" 2>/dev/null; then
        if command -v sha256sum >/dev/null 2>&1; then
            (cd "$tmpdir" && grep " ${tarball}\$" SHA256SUMS | sha256sum -c -)
        elif command -v shasum >/dev/null 2>&1; then
            (cd "$tmpdir" && grep " ${tarball}\$" SHA256SUMS | shasum -a 256 -c -)
        fi
    fi

    tar -xzf "${tmpdir}/${tarball}" -C "$tmpdir"
    mkdir -p "$PREFIX"
    cp "${tmpdir}/cachepin" "${PREFIX}/cachepin"
    chmod +x "${PREFIX}/cachepin"
}

if [ -n "$REPO" ]; then
    TARGET="$(detect_target)"
    if [ -n "$TARGET" ] && install_from_release "$TARGET"; then
        echo "[cachepin] Installed binary to ${PREFIX}/cachepin"
        "${PREFIX}/cachepin" --version
        exit 0
    fi
    echo "[cachepin] Falling back to local zero-dependency source build..."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ ! -f "${ROOT}/Cargo.toml" ]; then
    echo "[cachepin] Cargo.toml not found. Set CACHEPIN_REPO=<owner>/cachepin or run inside a checkout." >&2
    exit 1
fi

echo "[cachepin] Building release binary and C-ABI shared library (0 external crates)..."
cargo build --release --manifest-path "${ROOT}/Cargo.toml"
mkdir -p "$PREFIX"
cp "${ROOT}/target/release/cachepin" "${PREFIX}/cachepin"
chmod +x "${PREFIX}/cachepin"

echo "[cachepin] Installed binary to ${PREFIX}/cachepin"
"${PREFIX}/cachepin" --version
echo ""
echo "Quick start:"
echo "  cachepin audit --claude-code"
echo "  cachepin proxy --upstream https://api.anthropic.com --port 47820"
echo "  cachepin mcp"
