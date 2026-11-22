import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss import pairs_from, wall_us


class SourceTests(unittest.TestCase):
    def record(self, round_id=1):
        return {"kind": "round", "stage": "moss", "round_id": round_id, "providers": {
            "runpod": {"wall_s": 121.70, "warm_wall_s": 0.01, "state": "COMPLETED", "valid": True},
            "modal": {"wall_s": 38.32, "state": "COMPLETED", "valid": True}}}

    def test_extracts_only_concurrent_cold_arm_wall_times(self):
        self.assertEqual(pairs_from(json.dumps(self.record())), [(121700000, 38320000)])

    def test_detects_missing_or_invalid_outcomes(self):
        for state, valid in [("SUBMIT_FAILED", False), ("COMPLETED", False), ("CANCELLED", True)]:
            with self.assertRaises(ValueError):
                wall_us({"state": state, "valid": valid, "wall_s": 1})

    def test_detects_round_gap(self):
        with self.assertRaises(ValueError):
            pairs_from(json.dumps(self.record(2)))

if __name__ == "__main__":
    unittest.main()
