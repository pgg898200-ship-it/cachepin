"""Compare the Python delay selector with replay at every integer delay."""
from itertools import combinations_with_replacement, product
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
from demo import choose


def replay_best(pairs, deadline, budget):
    primary_misses = sum(primary > deadline for primary, _ in pairs)
    scores = [(primary_misses, 0, float("inf"))]
    for delay in range(deadline + 1):
        launches = sum(primary > delay for primary, _ in pairs)
        if 100 * launches <= budget * len(pairs):
            misses = sum(min(primary, delay + backup) > deadline for primary, backup in pairs)
            scores.append((misses, launches, delay))
    delay = min(scores)[2]
    return None if delay == float("inf") else delay


class SelectorTests(unittest.TestCase):
    def test_all_small_paired_traces(self):
        durations = list(product(range(3), repeat=2))
        for length in (1, 2, 3):
            for pairs in combinations_with_replacement(durations, length):
                for deadline in range(3):
                    for budget in (0, 33, 34, 50, 66, 67, 100):
                        self.assertEqual(choose(pairs, deadline, budget),
                                         replay_best(pairs, deadline, budget),
                                         (pairs, deadline, budget))


if __name__ == "__main__":
    unittest.main()
