#!/usr/bin/env python3
"""Replay GPUHedge's public paired MOSS-TTS benchmark (RunPod to Modal)."""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


COMMIT = "669c8c9327c95164cfd317a4cc4c7c123f4afcc4"
URL = f"https://raw.githubusercontent.com/mireklzicar/gpuhedge/{COMMIT}/traces/moss_rounds.jsonl"
SHA256 = "7b2032386f7ee6183dc260298785cec03f83d22cac464bf0fb695067c79af960"
SLO = 60_000_000
STEP = 100_000
POINTS = 3001


def wall_us(arm):
    if arm["state"] != "COMPLETED" or arm["valid"] is not True:
        raise ValueError("selected arm is missing a complete valid response")
    micros = Decimal(arm["wall_s"]) * 1_000_000
    if not micros.is_finite() or not 0 <= micros <= 300_000_000 or micros != int(micros):
        raise ValueError("unexpected wall time")
    return int(micros)


def pairs_from(data):
    pairs = []
    for line in data.splitlines():
        record = json.loads(line, parse_float=Decimal)
        if record["kind"] != "round" or record["stage"] != "moss":
            raise ValueError("unexpected record type")
        if record["round_id"] != len(pairs) + 1:
            raise ValueError("rounds must be complete and chronological")
        arms = record["providers"]
        pairs.append((wall_us(arms["runpod"]), wall_us(arms["modal"])))
    return pairs


def paired_csv(pairs):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("primary_us", "backup_us"))
    writer.writerows(pairs)
    return stream.getvalue()


def curve(binary, pairs):
    out = subprocess.run(
        [str(binary.resolve()), str(SLO), str(STEP), str(POINTS)],
        input=paired_csv(pairs), text=True, capture_output=True, check=True,
    )
    values = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(io.StringIO(out.stdout))]
    if len(values) != POINTS:
        raise ValueError("incomplete Racecurve output")
    # Cross-check every grid point against direct per-request replay.
    for index, row in enumerate(values):
        delay = index * STEP
        times = [min(a, delay + b) for a, b in pairs]
        if (row["delay_us"], row["launches"], row["leaf_misses"], row["elapsed_us"]) != (
            delay, sum(a > delay for a, b in pairs), sum(t > SLO for t in times), sum(times)
        ):
            raise ValueError(f"native and direct replay disagree at delay {delay}")
    return values


def summarize(pairs, row):
    delay = row["delay_us"]
    times = sorted(min(a, delay + b) for a, b in pairs)
    return {"delay_us": delay, "requests": len(times), "misses": row["leaf_misses"],
            "launches": row["launches"], "p95_nearest_rank_us": times[math.ceil(0.95 * len(times))-1],
            "mean_us": round(row["elapsed_us"] / len(times), 2)}


def evaluate(pairs, binary):
    if len(pairs) != 54:
        raise ValueError("expected 54 MOSS-TTS rounds")
    calibration, evaluation = pairs[:18], pairs[18:]
    cal = curve(binary, calibration)
    feasible = [r for r in cal if r["launches"] * 4 <= len(calibration)]
    selected = min(feasible, key=lambda r: (r["leaf_misses"], r["launches"], r["delay_us"]))
    later = curve(binary, evaluation)
    result = {"source": URL, "sha256": SHA256, "primary": "runpod", "backup": "modal",
              "slo_us": SLO, "step_us": STEP, "points": POINTS, "calibration_launch_cap": 0.25,
              "selected_delay_us": selected["delay_us"]}
    for name, subset, table in [("rounds_1_18", calibration, cal),
                                ("rounds_19_54", evaluation, later)]:
        # Both arms finish within the upstream 300 s observation cap.
        result[name] = {"primary_only": summarize(subset, table[-1]),
                        "upstream_fixed_10s": summarize(subset, table[10_000_000 // STEP]),
                        "calibrated": summarize(subset, table[selected["delay_us"] // STEP])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="use a local copy of the pinned trace")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "racecurve")
    parser.add_argument("--csv", type=Path, help="also write converted pairs to this local CSV")
    args = parser.parse_args()
    if args.source:
        data = args.source.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        parser.error("trace SHA-256 mismatch")
    pairs = pairs_from(data)
    report = evaluate(pairs, args.binary)
    if args.csv:
        args.csv.write_text(paired_csv(pairs), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
