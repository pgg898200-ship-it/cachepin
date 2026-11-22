"""Measure wall time for overlapping one native PrefixSim process per cache size (Linux)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys
import subprocess
import tempfile
import time

from compare_prefixsim import prefixsim


def available_memory_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def profile(binary, path, cap):
    # wait4 returns this child's peak RSS rather than a cumulative maximum.
    with tempfile.TemporaryFile() as output:
        proc = subprocess.Popen([binary, "--trace", str(path), "--cache-size", str(cap),
                                 "--algo", "lru", "--no-verify", "--replay-order", "deepest-first"],
                                stdout=output, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
        if proc.returncode:
            output.seek(0)
            raise RuntimeError(output.read().decode(errors="replace"))
    return usage.ru_maxrss


def sweep(binary, path, workers, caps, verified):
    start = time.perf_counter_ns()
    def measure(cap):
        _, result = prefixsim(binary, path, cap)
        hits, refs = verified[str(cap)]
        if (int(result["n_req_skipped"]) or int(result["n_blocks"]) != refs or
                abs(hits/refs - float(result["block_hit_ratio"])) > 0.50001e-6):
            raise RuntimeError(f"result differs at capacity {cap}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(measure, caps))
    return (time.perf_counter_ns()-start) / 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path, help="serial benchmark report with known capacity hit counts")
    ap.add_argument("--lcp", required=True, type=Path)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--prefixsim", default="prefixsim")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reserve-mib", type=int, default=256)
    ap.add_argument("--output", type=Path, help="save progress after every sweep")
    args = ap.parse_args()
    if args.repeats < 1 or args.reserve_mib < 0:
        ap.error("expected positive repeats and nonnegative reserve")
    ref = json.loads(args.reference.read_text())
    caps = ref["capacities"]
    verified = ref["trials"][0]["curve_counts"]
    reports = {}
    result = dict(cpus_in_affinity=len(os.sched_getaffinity(0)), capacities=caps,
                  trials=args.repeats, reserve_mib=args.reserve_mib, formats=reports)
    def checkpoint():
        if args.output:
            args.output.write_text(json.dumps(result, indent=2)+"\n")
    for fmt, path in [("lcp", args.lcp), ("jsonl", args.jsonl)]:
        peaks = {cap: profile(args.prefixsim, path, cap) for cap in caps}
        report = dict(max_rss_kib_by_capacity=peaks, measurements=[], median_ms={}, fastest_ms={})
        reports[fmt] = report
        checkpoint()
        for trial in range(args.repeats):
            for workers in ((2, 7) if trial % 2 == 0 else (7, 2)):
                headroom = available_memory_kib()
                estimate = sum(sorted(peaks.values(), reverse=True)[:workers])
                if headroom < 1.15 * estimate + args.reserve_mib * 1024:
                    report.setdefault("skipped", []).append(dict(trial=trial, workers=workers,
                                                                 available_kib=headroom, estimated_kib=estimate))
                    checkpoint()
                    continue
                ms = sweep(args.prefixsim, path, workers, caps, verified)
                report["measurements"].append(dict(workers=workers, elapsed_ms=ms, available_kib=headroom))
                checkpoint()
        for w in (2, 7):
            values = [r["elapsed_ms"] for r in report["measurements"] if r["workers"] == w]
            if not values:
                continue
            report["median_ms"][w] = statistics.median(values)
            report["fastest_ms"][w] = min(values)
        checkpoint()
        print(fmt, report["median_ms"], file=sys.stderr, flush=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
