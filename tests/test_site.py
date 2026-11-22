"""The analyzer's source download must rebuild offline from its own files."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def build(source, target):
    subprocess.run([sys.executable, source / "packaging/build_site.py", target], check=True)
    return target / "source.zip"


class SourceDownloadTest(unittest.TestCase):
    def test_rebuild_with_local_development_files(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            original = build(ROOT, work / "site")
            with ZipFile(original) as archive:
                archive.extractall(work)

            source = work / "racecurve"
            for extra in ("scratch.csv", ".venv/local_module.py", "web/notes.md"):
                path = source / extra
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("local development file")

            rebuilt = build(source, work / "rebuilt")
            self.assertEqual(original.read_bytes(), rebuilt.read_bytes())
            self.assertFalse((rebuilt.parent / "notes.md").exists())


if __name__ == "__main__":
    unittest.main()
