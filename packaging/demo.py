"""Choose a hedge delay on a calibration split, then report held-out results."""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
from importlib import resources
import json
from pathlib import Path


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
    """Minimize misses, then launches, then delay under the launch budget."""
    # Between primary completions the launch count stays fixed, and waiting
    # longer can only lose rescues. The earliest delay in each interval wins.
    completions = Counter({0: 0})
    miss_starts = []
    for primary, backup in pairs:
        if primary <= slo:
            completions[primary] += 1
        else:
            # The primary already misses. Its backup also misses from this delay on.
            miss_starts.append(slo - backup + 1)

    miss_starts.sort()
    best = (len(miss_starts), 0, slo + 1)  # Backups off.
    launches = len(pairs)
    for delay, finished in sorted(completions.items()):
        launches -= finished
        if 100 * launches <= budget * len(pairs):
            misses = bisect_right(miss_starts, delay)
            best = min(best, (misses, launches, delay))
    return None if best[2] > slo else best[2]


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
