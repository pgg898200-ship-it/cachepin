import io
import itertools
import unittest

from bounds import Bounds, Duration, from_race, observed_duration, read_pairs


class TestBounds(unittest.TestCase):
    def test_reconstructs_all_small_race_logs_sharply(self):
        # Enumerate every completion pair, group by the visible log, then enumerate
        # counterfactuals. 20 exceeds every candidate delay+observed backup and SLO.
        for logged_delay in range(5):
            groups = {}
            for p, b in itertools.product(range(21), repeat=2):
                e = min(p, logged_delay + b)
                if e > 8:
                    continue
                winner = "primary" if p <= logged_delay + b else "backup"
                groups.setdefault((winner, e), []).append((p, b))
            for (winner, e), latent in groups.items():
                pair = from_race(winner, e, logged_delay)
                index = Bounds([pair], 6)
                for delay in range(9):
                    possible = [min(p, delay + b) for p, b in latent]
                    result = index.query(delay)
                    self.assertEqual(result["sum_response_us"], [min(possible), max(possible)])
                    outcomes = [int(r > 6) for r in possible]
                    self.assertEqual(result["slo_misses"], [min(outcomes), max(outcomes)])

    def test_aggregate_index_matches_direct_endpoints(self):
        choices = (Duration(0, 0), Duration(1, 1), Duration(0, 7), Duration(4, 7),
                   Duration(9, None))
        pairs = list(itertools.product(choices, repeat=2))
        for dataset in [pairs, pairs[:-1]]:
            for slo in [0, 1, 3, 7, 40, 10**60]:
                index = Bounds(dataset, slo)
                for delay in [0, 1, 2, 7, 9, 45, 10**61]:
                    low = [min(a.low, delay + b.low) for a, b in dataset]
                    high = []
                    for a, b in dataset:
                        choices = [a.high, None if b.high is None else delay + b.high]
                        high.append(min((c for c in choices if c is not None), default=None))
                    result = index.query(delay)
                    high_total = None if None in high else sum(high)
                    self.assertEqual(result["sum_response_us"], [sum(low), high_total])
                    self.assertEqual(result["slo_misses"],
                                     [sum(v > slo for v in low), sum(v is None or v > slo for v in high)])

    def test_csv_and_boundaries(self):
        synthetic = "winner,elapsed_us,old_delay_us\nprimary,2,5\nbackup,8,5\nprimary,8,5\n"
        pairs = list(read_pairs(io.StringIO(synthetic)))
        self.assertEqual(pairs, [(Duration(2, 2), Duration(0, None)),
                                 (Duration(9, None), Duration(3, 3)),
                                 (Duration(8, 8), Duration(3, None))])
        result = Bounds(pairs, 8).query(5)
        self.assertEqual(result["sum_response_us"], [18, 18])
        self.assertEqual(result["slo_misses"], [0, 0])
        raw = "a_status,a_us,b_status,b_us\ncensored,5,unstarted,\nat_least,0,exact,3\n"
        self.assertEqual(list(read_pairs(io.StringIO(raw))),
                         [(Duration(6, None), Duration(0, None)), (Duration(0, None), Duration(3, 3))])

    def test_invalid_csv(self):
        for data in ["", "winner,winner,elapsed_us,old_delay_us\n",
                     "winner,elapsed_us,old_delay_us\nbackup,3,5\n",
                     "winner,elapsed_us,old_delay_us\nbackup,5\n",
                     "winner,elapsed_us,old_delay_us\nmystery,1,0\n",
                     "winner,elapsed_us,old_delay_us\nprimary,1,0,junk\n"]:
            with self.assertRaises(ValueError):
                list(read_pairs(io.StringIO(data)))
        for args in [("unstarted", "0"), ("exact", "-1"), ("exact", "0.5"), ("censored", ""), ("interval", "0"), ("interval", "4..2"), ("bad", "0")]:
            with self.assertRaises(ValueError):
                observed_duration(*args)
        with self.assertRaises(ValueError):
            Bounds([], 100)


if __name__ == "__main__":
    unittest.main()
