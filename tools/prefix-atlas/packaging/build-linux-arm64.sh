#!/bin/sh
# Build the ARM64 wheel from the published sdist. Run in a Python environment
# with setuptools==84.2.1, wheel==0.48.0 and ziglang==0.14.1.
set -eu
dist=$(cd "$(dirname "$0")/../../../site/prefix-atlas/dist" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
tar -xzf "$dist/prefix_atlas-0.1.0.tar.gz"
cd prefix_atlas-0.1.0
zigdir=$(python -c 'from pathlib import Path; import ziglang; print(Path(ziglang.__file__).parent)')
cp "$zigdir/LICENSE" zig-LICENSE.txt
cp "$zigdir/lib/libc/musl/COPYRIGHT" musl-LICENSE.txt
for runtime in libcxx libcxxabi libunwind; do
    cp "$zigdir/lib/$runtime/LICENSE.TXT" "$runtime-LICENSE.txt"
done
python - <<'PY'
from pathlib import Path
project = Path("pyproject.toml")
project.write_text(project.read_text().replace('license-files = ["LICENSE"]',
                                              'license-files = ["LICENSE", "*-LICENSE.txt"]'))
PY
# Put the strip flags after any debug flags inherited from the host Python.
export ZIG_BIN="$zigdir/zig"
cat > "$work/cxx" <<'WRAPPER'
#!/bin/sh
exec "$ZIG_BIN" c++ -target aarch64-linux-musl -mcpu=baseline "$@" -g0 -static -s
WRAPPER
chmod +x "$work/cxx"
export CXX="$work/cxx"
export SOURCE_DATE_EPOCH=1795305600
python setup.py bdist_wheel --plat-name linux_aarch64 --dist-dir "$dist"
