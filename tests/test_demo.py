"""Check the source quickstart and the integer-delay policy selector."""
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

DEMO = Path(__file__).resolve().parents[1] / "packaging" / "demo.py"
choose = runpy.run_path(str(DEMO))["choose"]


class DemoTest(unittest.TestCase):
    def test_bundled_trace_from_checkout(self):
        with TemporaryDirectory() as workdir:
            output = subprocess.check_output([sys.executable, "-I", DEMO], cwd=workdir)
        result = json.loads(output)
        self.assertEqual(result["delay_us"], 1177)
        self.assertEqual(result["holdout"]["requests"], 300)
        self.assertEqual(result["holdout"]["slo_misses"], 0)
        self.assertEqual(result["holdout"]["backup_launches"], 42)
        self.assertEqual(result["holdout_primary_only"]["slo_misses"], 27)

    def test_selector_matches_every_delay_replay(self):
        rng = random.Random(124)
        for _ in range(300):
            slo = rng.randrange(12)
            pairs = [(rng.randrange(15), rng.randrange(15))
                     for _ in range(rng.randint(1, 12))]
            budget = rng.choice([0, 25, 50, 100])
            primary_misses = sum(primary > slo for primary, _ in pairs)
            scores = [(primary_misses, 0, slo + 1)]
            for delay in range(slo + 1):
                launches = sum(primary > delay for primary, _ in pairs)
                if 100 * launches <= budget * len(pairs):
                    misses = sum(min(primary, delay + backup) > slo
                                 for primary, backup in pairs)
                    scores.append((misses, launches, delay))
            best_delay = min(scores)[2]
            self.assertEqual(choose(pairs, slo, budget),
                             best_delay if best_delay <= slo else None)


if __name__ == "__main__":
    unittest.main()
