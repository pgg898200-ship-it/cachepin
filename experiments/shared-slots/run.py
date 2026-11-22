#!/usr/bin/env python3
"""Compile and run the fixed prospective Racecurve load experiment with a C++17 compiler."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
DELAY_US = 1177
DEADLINE_NS = 6_000_000
SEEDS = (127995, 127996)
RATES = (400, 1600, 2600)
SECONDS = 2


def draw(seed, rid, channel):
    key = f"{seed}/{rid}/{channel}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big")


def workload(rate, seed):
    lines = []
    for rid in range(1, rate * SECONDS + 1):
        a = draw(seed, rid, "primary")
        b = draw(seed, rid, "backup")
        common = 4000 if draw(seed, rid, "shared") % 50 == 0 else 0
        p = 100 + a % 401 + common + 6000 * (a // 401 % 12 == 0)
        h = 100 + b % 401 + common + 4000 * (b // 401 % 20 == 0)
        lines.append(f"{p} {h}\n")
    return "".join(lines)


def percentiles_ns(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {f"p{pct}_us": round(ordered[math.ceil(len(ordered) * pct / 100) - 1] / 1000, 1)
            for pct in (50, 95, 99)}


def summary(raw, rate, hedged):
    rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(io.StringIO(raw))]
    assert len(rows) == rate * SECONDS
    selected = [r for r in rows if r["backup_submit_ns"] >= 0]
    assert hedged or not selected
    for idx, r in enumerate(rows):
        assert r["id"] == idx + 1 and r["planned_ns"] == idx * 1_000_000_000 // rate
        assert r["planned_ns"] <= r["primary_submit_ns"] <= r["primary_dequeued_ns"] <= r["primary_start_ns"] <= r["primary_end_ns"]
        ends = [r["primary_end_ns"]]
        if r["backup_submit_ns"] >= 0:
            assert r["primary_submit_ns"] + DELAY_US * 1000 <= r["backup_submit_ns"] <= r["backup_dequeued_ns"] <= r["backup_start_ns"] <= r["backup_end_ns"]
            ends.append(r["backup_end_ns"])
        assert r["first_ns"] == min(ends)
    horizon = SECONDS * 1e9
    latency = [r["first_ns"] - r["planned_ns"] for r in rows]
    finished_by_horizon = sum(r["first_ns"] <= horizon for r in rows)
    final_end = max(max(r["primary_end_ns"], r["backup_end_ns"]) for r in rows)
    busy = sum(r["primary_end_ns"] - r["primary_start_ns"] for r in rows)
    busy += sum(r["backup_end_ns"] - r["backup_start_ns"] for r in selected)
    arm = {
        "requests": len(rows),
        "planned_rate_rps": rate,
        "admission_rate_rps": round((len(rows)-1)*1e9/(rows[-1]["primary_submit_ns"]-rows[0]["primary_submit_ns"]), 1),
        "backup_launches": len(selected),
        "backup_fraction": round(len(selected)/len(rows), 5),
        "backup_wins": sum(r["backup_end_ns"] < r["primary_end_ns"] for r in selected),
        "misses": sum(t > DEADLINE_NS for t in latency),
        "miss_pct": round(100*sum(t > DEADLINE_NS for t in latency)/len(rows), 2),
        "planned_arrival_latency": percentiles_ns(latency),
        "request_dispatch_lag": percentiles_ns([r["primary_submit_ns"]-r["planned_ns"] for r in rows]),
        "max_dispatch_lag_us": round(max(r["primary_submit_ns"]-r["planned_ns"] for r in rows)/1000, 1),
        "primary_total_queue_wait": percentiles_ns([r["primary_start_ns"]-r["primary_submit_ns"] for r in rows]),
        "backup_total_queue_wait": percentiles_ns([r["backup_start_ns"]-r["backup_submit_ns"] for r in selected]),
        "primary_shared_slot_wait": percentiles_ns([r["primary_start_ns"]-r["primary_dequeued_ns"] for r in rows]),
        "backup_shared_slot_wait": percentiles_ns([r["backup_start_ns"]-r["backup_dequeued_ns"] for r in selected]),
        "arrivals_dispatched_by_horizon": sum(r["primary_submit_ns"] <= horizon for r in rows),
        "first_completions_by_horizon": finished_by_horizon,
        "completed_rps_within_offered_window": round(finished_by_horizon / SECONDS, 1),
        "logical_rps_including_full_loser_drain": round(len(rows)*1e9/final_end, 1),
        "drain_after_window_ms": round(max(0, final_end-horizon)/1e6, 1),
        "two_slot_utilization_through_drain_pct": round(100*busy/(2*final_end), 2),
    }
    return arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    compiler = os.environ.get("CXX", "c++")
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
              "delay_us": DELAY_US, "deadline_us": DEADLINE_NS//1000, "arrival_window_s": SECONDS,
              "workers_per_replica": 2, "shared_execution_slots": 2, "runs": []}
    with tempfile.TemporaryDirectory() as tmp:
        binary = str(Path(tmp) / "load")
        subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(ROOT / "load.cpp"), "-o", binary], check=True)
        for rate in RATES:
            for rep, seed in enumerate(SEEDS):
                work = workload(rate, seed)
                for hedged in ((False, True) if rep == 0 else (True, False)):
                    label = "hedge" if hedged else "control"
                    out = subprocess.run([binary, str(rate), str(DELAY_US if hedged else -1)],
                                         input=work, text=True, capture_output=True, check=True)
                    name = f"r{rate}-s{seed}-{label}.csv.gz"
                    data = gzip.compress(out.stdout.encode(), compresslevel=6, mtime=0)
                    (args.output / name).write_bytes(data)
                    arm = summary(out.stdout, rate, hedged)
                    arm.update({"arm": label, "seed": seed, "raw": name,
                                "raw_sha256": hashlib.sha256(data).hexdigest()})
                    result["runs"].append(arm)
                    print(f"{rate} {seed} {label}: {arm['miss_pct']}% missed, {arm['backup_fraction']} launched", flush=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
