import copy
import io
import json
from pathlib import Path
import unittest
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounds import Bounds, observed_duration

try:
    from .otlp import observations
except ImportError:
    from otlp import observations


class TestOTLP(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).with_name("race.otlp.jsonl").read_text()
        self.batches = [json.loads(line) for line in self.fixture.splitlines()]
        self.a = self.batches[0]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        self.b, self.race = self.batches[1]["resourceSpans"][0]["scopeSpans"][0]["spans"]

    def read(self):
        return observations(io.StringIO("\n".join(map(json.dumps, self.batches))))

    def test_reordered_batches_cancellation_and_retry(self):
        self.batches = list(reversed(self.batches)) + self.batches
        self.assertEqual(self.read(), [["at_least", 8, "exact", 3]])

    def test_trace_id_and_parent_checks(self):
        self.a["traceId"] = "de" * 16
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.read()
        self.a["traceId"] = self.b["traceId"]
        self.a["parentSpanId"] = self.b["spanId"]
        with self.assertRaisesRegex(ValueError, "orphan"):
            self.read()

    def test_missing_and_declared_unstarted(self):
        del self.batches[1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        with self.assertRaisesRegex(ValueError, "missing or unexpected"):
            self.read()
        self.race["attributes"][0]["value"]["stringValue"] = "primary"
        self.race["attributes"][1]["value"]["boolValue"] = False
        self.a["attributes"][1]["value"]["stringValue"] = "complete"
        self.a["endTimeUnixNano"] = self.race["endTimeUnixNano"]
        self.assertEqual(self.read(), [["exact", 8, "unstarted", ""]])

    def test_quantization_and_late_shadow(self):
        self.a["attributes"][1]["value"]["stringValue"] = "complete"
        self.b["endTimeUnixNano"] = str(int(self.b["endTimeUnixNano"]) - 999)
        self.assertEqual(self.read(), [["exact", 105, "interval", "2..3"]])

    def test_conflicting_retry_and_bad_timing(self):
        self.batches.append(copy.deepcopy(self.batches[0]))
        self.a["endTimeUnixNano"] = self.a["startTimeUnixNano"]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.read()
        self.batches.pop()
        with self.assertRaisesRegex(ValueError, "timing conflicts"):
            self.read()

    def test_resource_identity_mismatch(self):
        resource = self.batches[0]["resourceSpans"][0]["resource"]
        resource["attributes"][0]["value"]["stringValue"] = "remote_backend"
        with self.assertRaisesRegex(ValueError, "cross coordinator"):
            self.read()

    def test_jittered_dispatch_contains_latent_outcomes(self):
        # Create real nanosecond service durations and a jittered historical timer.
        # Export only the winning completion. Every counterfactual at a whole-µs
        # launch delay must still bracket its latent, continuous-time outcome.
        rng = random.Random(8191)
        base = 1_750_000_000_000_000_371
        for _ in range(500):
            p_ns, b_ns = rng.randrange(12_000), rng.randrange(12_000)
            launch = 2_000 + rng.randrange(6_000)
            started = p_ns > launch
            backup_wins = started and launch + b_ns < p_ns
            winner = "backup" if backup_wins else "primary"
            decision = min(p_ns, launch + b_ns)
            a, b, parent = copy.deepcopy([self.a, self.b, self.race])
            parent["startTimeUnixNano"] = str(base - 500)
            parent["endTimeUnixNano"] = str(base + decision + 70_000)
            parent["attributes"][0]["value"]["stringValue"] = winner
            parent["attributes"][1]["value"]["boolValue"] = started
            exported = [parent]
            for role, child, t in [("primary", a, 0), ("backup", b, launch)]:
                if role == "backup" and not started:
                    continue
                child["attributes"][1]["value"]["stringValue"] = "complete" if role == winner else "pending"
                child["startTimeUnixNano"] = str(base + t)
                end = decision if role == winner else decision + 100_000
                child["endTimeUnixNano"] = str(base + end)
                exported.append(child)
            batch = {"resourceSpans": [{"scopeSpans": [{"spans": exported}]}]}
            (sa, va, sb, vb), = observations(io.StringIO(json.dumps(batch)))
            pair = (observed_duration(sa, str(va)), observed_duration(sb, str(vb)))
            for delay in range(11):
                slo = 6
                answer = Bounds([pair], slo).query(delay)
                latent = min(p_ns, 1000 * delay + b_ns)
                low, high = answer["sum_response_us"]
                self.assertLessEqual(1000 * low, latent)
                self.assertLessEqual(latent, 1000 * high)
                mlo, mhi = answer["slo_misses"]
                self.assertLessEqual(mlo, int(latent > 1000 * slo))
                self.assertLessEqual(int(latent > 1000 * slo), mhi)

    def test_pretty_json(self):
        pretty = json.dumps({"resourceSpans": [resource for b in self.batches
                                               for resource in b["resourceSpans"]]}, indent=2)
        self.assertEqual(observations(io.StringIO(pretty)), [["at_least", 8, "exact", 3]])


if __name__ == "__main__":
    unittest.main()
