from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from replay import export, INT32_MAX, UINT64_MAX


def row(timestamp, path, n=None):
    return dict(timestamp=timestamp, hash_ids=path, input_length=len(path) * 16 if n is None else n,
                output_length=3)


def convert(rows):
    result = StringIO()
    export(StringIO("".join(json.dumps(r) + "\n" for r in rows)), result, 16, "ms")
    return list(map(json.loads, result.getvalue().splitlines()))


class Replay(unittest.TestCase):
    def test_timestamp_and_block_identity(self):
        start = 1778977459236
        rows = convert([row(start, [1, UINT64_MAX]), row(start + 1020, [1, UINT64_MAX]),
                        row(start + 1030, [2, UINT64_MAX]), row(start + 1030, [1], 16)])
        self.assertEqual([r["timestamp"] for r in rows], [0, 1.02, 1.03, 1.03])
        self.assertEqual(rows[0]["hash_ids"], rows[1]["hash_ids"])
        self.assertNotEqual(rows[0]["hash_ids"][1], rows[2]["hash_ids"][1])
        for r in rows:
            self.assertTrue(all(0 <= i <= INT32_MAX for i in r["hash_ids"]))
        self.assertEqual(rows[-1]["hash_ids"], [1])

    def test_partial_and_local_labels(self):
        rows = convert([row(0, [0, 2], 27), row(1, [3, 2], 27)])
        self.assertEqual(rows[0]["hash_ids"], [0, 2])
        self.assertEqual(rows[1]["input_length"], 27)
        self.assertNotEqual(rows[1]["hash_ids"][-1], 2)

    def test_reject_ambiguous_width_and_invalid_shape(self):
        for rows in ([row(2, [1]), row(1, [1])], [row(0, [1], 17)], [row(0, [-1])],
                     [row(0, [True])], [row(0, [1], 7), row(0, [1], 9)], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                convert(rows)

    def test_file_output_is_atomic_on_bad_input(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "replay.jsonl"
            dest.write_text("old\n")
            proc = subprocess.run([sys.executable, Path(__file__).with_name("replay.py"),
                                   "--block-size", "16", "-", dest],
                                  input=json.dumps(row(0, [1])) + "\n{}\n", text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("line 2", proc.stderr)
            self.assertEqual(dest.read_text(), "old\n")
