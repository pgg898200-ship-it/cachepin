"""Build the static explorer and a portable source archive outside the source tree."""
import argparse
import hashlib
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".cpp", ".hpp", ".py", ".csv", ".json", ".jsonl", ".md", ".toml", ".js", ".cjs", ".html", ".css"}
SPECIAL = {"Makefile", "LICENSE", ".gitignore"}

def source_files():
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if any(part in {".git", "__pycache__", "dist"} for part in parts):
            continue
        if path.is_file() and (path.suffix in SUFFIXES or path.name in SPECIAL):
            yield path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    target = parser.parse_args().output.resolve()
    if target == ROOT or ROOT in target.parents:
        parser.error("choose an output directory outside the source tree")
    target.mkdir(parents=True, exist_ok=True)
    for src in (ROOT/"web").iterdir():
        shutil.copyfile(src, target/src.name)
    shutil.copyfile(ROOT/"examples"/"loopback.csv", target/"loopback.csv")
    archive = target/"source.zip"
    with ZipFile(archive, "w") as package:
        for path in source_files():
            info = ZipInfo("racecurve/"+path.relative_to(ROOT).as_posix(), (2026,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            package.writestr(info, path.read_bytes())
    wheel_dir = ROOT/"dist"
    for wheel in sorted(wheel_dir.glob("*.whl")):
        shutil.copyfile(wheel,target/wheel.name)
    (target/"SHA256SUMS").write_text("".join(
        hashlib.sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n"
        for path in [archive,*sorted(target.glob("*.whl"))]))
if __name__ == "__main__":
    main()
