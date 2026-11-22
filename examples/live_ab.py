#!/usr/bin/env python3
"""Compare primary-only and delayed replica requests on two loopback servers.

    python3 examples/live_ab.py --output results/prospective.json

Defaults freeze the independently calibrated delay. A workload seed distinct
from calibration and randomized presentation order produce a prospective,
matched A/B comparison. Every response drains before the next trial.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect import replica_connection, request


def micros(nanoseconds):
    return (nanoseconds + 999) // 1000


def run_arm(pool, primary, backup, request_id, delay_us):
    start = perf_counter_ns()
    first = pool.submit(request, primary, request_id)
    launched_at = None
    if delay_us is not None:
        remaining_ns = max(0, start + delay_us * 1000 - perf_counter_ns())
        wait([first], timeout=remaining_ns / 1e9)
        if not first.done():
            launched_at = micros(perf_counter_ns() - start)
            second = pool.submit(request, backup, request_id)
            finished, _ = wait([first, second], return_when=FIRST_COMPLETED)
            for response in finished:
                response.result()
            elapsed_us = micros(perf_counter_ns() - start)
            # Drain both sockets so the next trial starts with idle replicas.
            first.result()
            second.result()
            return elapsed_us, launched_at
    first.result()
    return micros(perf_counter_ns() - start), launched_at


def summarize(times, deadline):
    ordered = sorted(times)

    def percentile(fraction):
        return ordered[math.ceil(len(ordered) * fraction) - 1]

    return {
        "misses": sum(t > deadline for t in times),
        "mean_us": round(sum(times) / len(times), 2),
        "p50_us": percentile(0.5),
        "p95_us": percentile(0.95),
        "p99_us": percentile(0.99),
        "max_us": max(times),
    }


def experiment(args):
    rng = random.Random(args.order_seed)
    rows = []
    with ExitStack() as stack:
        primary = stack.enter_context(replica_connection("primary", args.seed))
        backup = stack.enter_context(replica_connection("backup", args.seed))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        warmups = [pool.submit(request, conn, 0) for conn in [primary, backup]]
        for warmup in warmups:
            warmup.result()
        start = perf_counter_ns()
        for request_id in range(1, args.pairs + 1):
            hedge_first = bool(rng.getrandbits(1))
            arms = [True, False] if hedge_first else [False, True]
            measured = {}
            for hedged in arms:
                delay = args.delay_us if hedged else None
                measured[hedged] = run_arm(pool, primary, backup, request_id, delay)
            control, _ = measured[False]
            treated, launch = measured[True]
            rows.append([request_id, int(hedge_first), control, treated, launch])
        wall_seconds = (perf_counter_ns() - start) / 1e9
    control = [r[2] for r in rows]
    treated = [r[3] for r in rows]
    launches = [r[4] for r in rows if r[4] is not None]
    cutoff = args.deadline_us
    return {
        "workload": "Two separate loopback HTTP processes, seeded sleeps, same work ID in both arms, one arm active, losers drained",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": args.seed,
        "order_seed": args.order_seed,
        "request_ids": args.pairs,
        "deadline_us": cutoff,
        "configured_hedge_delay_us": args.delay_us,
        "primary_only": summarize(control, cutoff),
        "hedged": summarize(treated, cutoff),
        "backup_launches": len(launches),
        "actual_backup_submit_us_min_median_max": [
            min(launches), sorted(launches)[(len(launches)-1)//2], max(launches)
        ] if launches else None,
        "primary_miss_hedge_meet": sum(a > cutoff >= b for a, b in zip(control, treated)),
        "primary_meet_hedge_miss": sum(a <= cutoff < b for a, b in zip(control, treated)),
        "hedged_arm_slower_pairs": sum(a < b for a, b in zip(control, treated)),
        "experiment_wall_seconds_including_drain": round(wall_seconds, 6),
        "trial_arms_per_wall_second": round(2 * args.pairs / wall_seconds, 2),
        "row_columns": ["request_id", "hedge_first", "primary_only_observed_us", "hedged_observed_us", "backup_submit_us_or_null"],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=127994)
    parser.add_argument("--order-seed", type=int, default=30513)
    parser.add_argument("--delay-us", type=int, default=1177)
    parser.add_argument("--deadline-us", type=int, default=6000)
    parser.add_argument("--output", type=Path, help="output JSON with summary and raw observations")
    args = parser.parse_args()
    if args.pairs < 1 or min(args.delay_us, args.deadline_us) < 0:
        parser.error("pairs must be positive and times nonnegative")
    payload = json.dumps(experiment(args), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
