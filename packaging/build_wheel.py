#!/usr/bin/env python3
"""Build in a temporary staging directory using installed setuptools and wheel."""
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    with TemporaryDirectory() as tmp:
        stage = Path(tmp)
        package = stage / "racecurve_tools"
        package.mkdir()
        (package / "__init__.py").write_text('"""RaceCurve paired-trace analysis tools."""\n')
        for source in ("bounds.py", "integrations/otlp.py", "packaging/demo.py", "examples/loopback.csv"):
            copyfile(ROOT / source, package / Path(source).name)
        for source in ("packaging/pyproject.toml", "packaging/README.md", "LICENSE"):
            copyfile(ROOT / source, stage / Path(source).name)
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(out), str(stage)], check=True)


if __name__ == "__main__":
    main()
