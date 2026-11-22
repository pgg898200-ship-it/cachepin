"""Choose a hedge delay on a calibration split, then report held-out results."""
import argparse
from bisect import bisect_right
import csv
from importlib import resources
import json
from pathlib import Path
import sys


def read_trace(stream):
    reader = csv.reader(stream, strict=True)
    if next(reader, None) != ["primary_us", "backup_us"]:
        raise ValueError("expected primary_us,backup_us header")
    pairs = []
    for row in reader:
        if len(row) != 2 or any(not value.isascii() or not value.isdigit() for value in row):
            raise ValueError(f"line {reader.line_num}: expected two nonnegative integer durations")
        pairs.append(tuple(map(int, row)))
    if len(pairs) < 2:
        raise ValueError("need at least two requests for calibration and holdout")
    return pairs


def choose(pairs, slo, budget):
    # For integer delay d, a backup fires iff P > d. A request that misses
    # primary-only is rescued iff d <= slo-B. Objective changes only at a P
    # or at a rescue cutoff + 1; selected ties prefer fewer launches then d.
    primary = sorted(p for p, b in pairs)
    cutoffs = sorted(slo - b + 1 for p, b in pairs if p > slo)
    primary_misses = len(cutoffs)
    optimum = (primary_misses, 0, slo + 1)
    for delay in sorted({0, *(p for p in primary if p <= slo),
                         *(c for c in cutoffs if 0 < c <= slo)}):
        launches = len(pairs) - bisect_right(primary, delay)
        if 100 * launches <= budget * len(pairs):
            optimum = min(optimum, (bisect_right(cutoffs, delay), launches, delay))
    return None if optimum[2] > slo else optimum[2]


def counts(pairs, slo, delay):
    elapsed = sorted(p if delay is None else min(p, delay + b) for p, b in pairs)
    return {"requests": len(pairs), "slo_misses": len(pairs) - bisect_right(elapsed, slo),
            "backup_launches": 0 if delay is None else sum(p > delay for p, b in pairs),
            "p99_us": elapsed[(99 * len(elapsed) - 1) // 100]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="complete paired CSV; defaults to included loopback trace")
    parser.add_argument("--slo-us", type=int, default=6000)
    parser.add_argument("--budget-pct", type=int, default=25)
    args = parser.parse_args(argv)
    if args.slo_us < 0 or not 0 <= args.budget_pct <= 100:
        parser.error("require a nonnegative SLO and budget percentage from 0 to 100")
    path = args.trace or resources.files("racecurve_tools").joinpath("loopback.csv")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            pairs = read_trace(stream)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"racecurve-demo: {exc}\n")
    n = max(1, 7 * len(pairs) // 10)
    train, hold = pairs[:n], pairs[n:]
    delay = choose(train, args.slo_us, args.budget_pct)
    print(json.dumps({"trace": str(args.trace) if args.trace else "bundled constructed loopback sample",
                      "slo_us": args.slo_us, "calibration_launch_budget_pct": args.budget_pct,
                      "policy": "primary_only" if delay is None else "hedged", "delay_us": delay,
                      "calibration": counts(train, args.slo_us, delay),
                      "holdout": counts(hold, args.slo_us, delay),
                      "holdout_primary_only": counts(hold, args.slo_us, None)}, indent=2))


if __name__ == "__main__":
    main()
