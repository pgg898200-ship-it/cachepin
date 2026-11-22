from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from prefix_atlas.writer import TraceWriter


class WriterTests(unittest.TestCase):
    def test_reuse_forks_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.tree"
            with TraceWriter(path) as trace:
                root = trace.extend(0, 4)
                first = trace.extend(root, 2)
                trace.request(first)
                trace.request(trace.extend(root, 1))
                trace.request(root)
                trace.request(0)
                trace.request(first)
                self.assertFalse(path.exists())
            self.assertEqual(path.read_text(), "3 5\n0 4\n1 2\n1 1\n2\n3\n1\n0\n2\n")
            curve = Path(__file__).resolve().with_name("curve")
            with path.open() as source:
                result = subprocess.check_output([curve, "--prefix-capacities", "1,4,7"], stdin=source, text=True)
            self.assertEqual(result.splitlines()[1:], ["1,3,21", "4,12,21", "7,14,21"])
            with TraceWriter(path):
                pass
            self.assertEqual(path.read_text(), "0 0\n")

    def test_validation_leaves_counts_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.tree"
            writer = TraceWriter(path)
            with self.assertRaises(RuntimeError):
                writer.extend(0, 1)
            with writer as trace:
                for handle in (1, -1, True, "0"):
                    with self.assertRaises(ValueError):
                        trace.request(handle)
                for pages in (0, -1, False, 1.5, 2**64):
                    with self.assertRaises(ValueError):
                        trace.extend(0, pages)
                self.assertEqual(trace.extend(0, 5), 1)
            self.assertEqual(path.read_text(), "1 0\n0 5\n")
            with self.assertRaises(RuntimeError):
                writer.__enter__()

    def test_failure_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.tree"
            path.write_text("older")
            with self.assertRaises(KeyError):
                with TraceWriter(path) as trace:
                    trace.request(0)
                    raise KeyError("producer stopped")
            with patch("prefix_atlas.writer.os.replace", side_effect=OSError("publish failed")):
                with self.assertRaises(OSError):
                    with TraceWriter(path) as trace:
                        trace.request(0)
            self.assertEqual(path.read_text(), "older")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_caught_write_failure_aborts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.tree"
            with self.assertRaises(RuntimeError):
                with TraceWriter(path) as trace:
                    with patch.object(trace._tips, "write", side_effect=OSError("disk error")):
                        with self.assertRaises(OSError):
                            trace.request(0)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
