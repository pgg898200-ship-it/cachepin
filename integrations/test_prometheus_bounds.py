import itertools
import math
import unittest

from prometheus_bounds import Histogram, query, read_histograms


def hist(values):
    return Histogram({e: sum(x <= e for x in values) for e in (0, 2, math.inf)})


class HistogramTests(unittest.TestCase):
    def test_all_bucket_consistent_pairings(self):
        # For every grouped observation, enumerate the possible true outcomes.
        # Four is beyond all thresholds in this grid and represents the open tail.
        durations = list(itertools.product(range(5), repeat=2))
        for deadline, delay in itertools.product(range(4), repeat=2):
            groups = {}
            for p, b in itertools.product(durations, repeat=2):
                hp, hb = hist(p), hist(b)
                key = tuple(hp.counts + hb.counts)
                launches, misses = groups.setdefault(key, ([], []))
                launches.append(sum(x > delay for x in p))
                misses.append(sum(min(x, delay + y) > deadline for x, y in zip(p, b)))
            for key, (launches, misses) in groups.items():
                edges = [0, 2, math.inf]
                actual = query(Histogram(dict(zip(edges, key[:3]))),
                               Histogram(dict(zip(edges, key[3:]))), deadline, delay)
                self.assertEqual(list(actual["backup_launches"]), [min(launches), max(launches)])
                self.assertEqual(actual["slo_misses"], [min(misses), max(misses)])

    def test_rejects_selectively_launched_backups(self):
        series = [{"metric": {"arm": arm, "le": "+Inf"}, "value": [0, n]}
                  for arm, n in [("primary", "10"), ("backup", "6")]]
        with self.assertRaisesRegex(ValueError, "cohort"):
            read_histograms({"status": "success", "data": {"resultType": "vector", "result": series}})
