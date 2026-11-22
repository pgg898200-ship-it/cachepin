#!/bin/sh
set -eu

# Install `cachepin` from a GitHub Release tarball (with SHA-256 verification)
# or build locally from source with zero external Rust crates.
#
# Examples:
#   curl -fsSL https://raw.githubusercontent.com/pgg898200-ship-it/cachepin/main/scripts/install.sh | sh
#   ./scripts/install.sh

PREFIX="${CACHEPIN_PREFIX:-$HOME/.local/bin}"
REPO="${CACHEPIN_REPO:-pgg898200-ship-it/cachepin}"
VERSION="${CACHEPIN_VERSION:-latest}"

detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"
    case "${os}:${arch}" in
        Linux:x86_64)                echo "x86_64-unknown-linux-gnu" ;;
        Linux:aarch64|Linux:arm64)   echo "aarch64-unknown-linux-gnu" ;;
        Darwin:arm64|Darwin:aarch64) echo "aarch64-apple-darwin" ;;
        Darwin:x86_64)               echo "x86_64-apple-darwin" ;;
        *)                           echo "" ;;
    esac
}

install_from_release() {
    target="$1"
    tag="$VERSION"
    if [ "$tag" = "latest" ]; then
        tag="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
            | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    fi
    if [ -z "$tag" ]; then
        echo "[cachepin] Could not resolve latest release tag for ${REPO}." >&2
        return 1
    fi
    tarball="cachepin-${tag}-${target}.tar.gz"
    base="https://github.com/${REPO}/releases/download/${tag}"
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
        echo ""
        echo "Quick start:"
        echo "  cachepin audit --claude-code"
        echo "  cachepin proxy --upstream https://api.anthropic.com --port 47820"
        echo "  cachepin mcp"
        exit 0
    fi
    echo "[cachepin] Prebuilt release unavailable; falling back to local source build..."
fi

ROOT="."
if [ -f "$0" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [ -f "${SCRIPT_DIR}/../Cargo.toml" ]; then
        ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

if [ ! -f "${ROOT}/Cargo.toml" ]; then
    echo "[cachepin] Cargo.toml not found. Clone https://github.com/${REPO} and run: cargo build --release" >&2
    exit 1
fi

echo "[cachepin] Building release binary and C-ABI shared library (0 external crates)..."
cargo build --release --manifest-path "${ROOT}/Cargo.toml"
mkdir -p "$PREFIX"
cp "${ROOT}/target/release/cachepin" "${PREFIX}/cachepin"
chmod +x "${PREFIX}/cachepin"

echo "[cachepin] Installed binary to ${PREFIX}/cachepin"
"${PREFIX}/cachepin" --version
