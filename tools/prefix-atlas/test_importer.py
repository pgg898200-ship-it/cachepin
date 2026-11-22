"""Differential coverage for the compact and expanded JSONL trace importers."""
import json
import random
from pathlib import Path
import runpy
import unittest

from mooncake import convert

reference = runpy.run_path(Path(__file__).parent / "bench" / "expanded_import.py")["convert"]


class TraceImporter(unittest.TestCase):
    def check(self, rows, block_size=1):
        lines = [json.dumps(row) for row in rows]
        self.assertEqual(convert(lines, block_size), reference(lines, block_size))

    def test_random_branching_histories_preserve_packed_ordinals(self):
        for seed in range(128):
            rng = random.Random(seed)
            paths = [[]]
            rows = []
            block_size = rng.randrange(2, 12)
            for _ in range(100):
                prev = rng.choice(paths)
                keep = rng.randrange(len(prev) + 1)
                path = prev[:keep] + rng.choices(range(-4, 5), k=rng.randrange(9))
                paths.append(path)
                remainder = rng.randrange(block_size)
                tail = [rng.randint(-100, 100)] if remainder and rng.randrange(2) else []
                rows.append({
                    "timestamp": rng.randrange(-5, 6),
                    "input_length": len(path) * block_size + remainder,
                    "hash_ids": path + tail,
                })
            with self.subTest(seed=seed):
                self.check(rows, block_size)

    def test_long_spans_split_at_the_tip_and_middle(self):
        path = list(range(65_536))
        variants = [path, [], [0], path[:-1], path[:-5] + [-1],
                    path[:19] + [2**90, -10], path[:19], path[:-5], path]
        self.check([dict(timestamp=0, input_length=len(ids), hash_ids=ids) for ids in variants])

    def test_validation_includes_unused_partial_id(self):
        for hashes in [[1, "tail"], [1, True], [1, 0.5]]:
            row = json.dumps(dict(timestamp=0, input_length=3, hash_ids=hashes))
            with self.assertRaisesRegex(ValueError, "record 1"):
                convert([row], 2)


if __name__ == "__main__":
    unittest.main()
