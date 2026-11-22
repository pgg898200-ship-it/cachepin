"""Time unmodified PrefixSim and the weighted curve engine from the same full-page JSONL."""
import argparse
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent.parent


def timed(command, **kwargs):
    start = time.perf_counter_ns()
    result = subprocess.run(command, check=True, stderr=subprocess.PIPE, **kwargs)
    return (time.perf_counter_ns() - start) / 1e6, result


def prefixsim(binary, trace, cap):
    args = [binary, "--trace", trace, "--algo", "lru", "--no-verify", "--replay-order",
            "deepest-first", "--cache-size", str(cap)]
    ms, proc = timed(args, stdout=subprocess.PIPE, text=True)
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
    return ms, dict(field.split("=", 1) for field in line.split()[1:])


def curve(binary, trace, caps):
    with trace.open("rb") as source:
        ms, proc = timed([binary, "--prefix-capacities", ",".join(map(str, caps))],
                         stdin=source, stdout=subprocess.PIPE, text=True)
    return ms, {int(row["capacity_pages"]): (int(row["hits"]), int(row["page_references"]))
                for row in csv.DictReader(io.StringIO(proc.stdout))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path, help="time-ordered source with complete hash blocks only")
    p.add_argument("--curve", default=str(HERE / "curve"))
    p.add_argument("--prefixsim", default="prefixsim")
    p.add_argument("--converter", default="traceConvLLM")
    p.add_argument("--block-size", required=True, type=int)
    p.add_argument("--capacities", required=True, help="comma-separated, all must fit longest request")
    p.add_argument("--one", required=True, type=int, help="single capacity also in the sweep")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--direct-json", action="store_true", help="also time native PrefixSim direct from JSONL")
    args = p.parse_args()
    caps = sorted(set(map(int, args.capacities.split(","))))
    if args.one not in caps or args.repeats < 1:
        p.error("one must be in capacities and repeats must be positive")
    trials = []
    digest = hashlib.sha256(args.jsonl.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        tree, lcp = work / "trace.tree", work / "trace.lcp.zst"
        for i in range(args.repeats):
            row = {}
            def convert_tree():
                with tree.open("wb") as out:
                    row["tree_import_ms"], _ = timed([sys.executable, HERE / "mooncake.py",
                        "--block-size", str(args.block_size), args.jsonl], stdout=out)
            def convert_lcp():
                row["lcp_import_ms"], _ = timed([args.converter, "--input", args.jsonl,
                    "--output", lcp, "--block-size-tokens", str(args.block_size)],
                    stdout=subprocess.DEVNULL)
            order = [convert_tree, convert_lcp]
            for fn in (order if i % 2 == 0 else reversed(order)):
                fn()
            def run_weighted():
                row["curve_one_ms"], one_result = curve(args.curve, tree, [args.one])
                row["curve_sweep_ms"], row["curve_counts"] = curve(args.curve, tree, caps)
                assert row["curve_counts"][args.one] == one_result[args.one]
            def run_simulator():
                row["sim_ms_by_capacity"] = {}
                row["sim_ratios"] = {}
                row["sim_refs"] = {}
                for cap in caps:
                    ms, result = prefixsim(args.prefixsim, lcp, cap)
                    assert int(result["n_req_skipped"]) == 0
                    row["sim_ms_by_capacity"][cap] = ms
                    row["sim_ratios"][cap] = float(result["block_hit_ratio"])
                    row["sim_refs"][cap] = int(result["n_blocks"])
                row["sim_one_ms"] = row["sim_ms_by_capacity"][args.one]
                row["sim_sweep_ms"] = sum(row["sim_ms_by_capacity"].values())
            order = [run_weighted, run_simulator]
            for fn in (order if i % 2 == 0 else reversed(order)):
                fn()
            for cap, (hits, refs) in row["curve_counts"].items():
                assert refs == row["sim_refs"][cap]
                assert abs(hits / refs - row["sim_ratios"][cap]) <= 0.50001e-6
            if args.direct_json:
                times = {}
                for cap, (hits, refs) in row["curve_counts"].items():
                    ms, reported = prefixsim(args.prefixsim, args.jsonl, cap)
                    if (int(reported["n_req_skipped"]) or int(reported["n_blocks"]) != refs
                            or abs(hits / refs - float(reported["block_hit_ratio"])) > 0.50001e-6):
                        raise ValueError(f"direct JSONL result differs at capacity {cap}")
                    times[cap] = ms
                row["sim_json_ms_by_capacity"] = times
                row["sim_json_one_ms"] = times[args.one]
                row["sim_json_sweep_ms"] = sum(times.values())
            trials.append(row)
            print(f"completed {i+1}/{args.repeats} for {args.jsonl.name}", file=sys.stderr, flush=True)
    metrics = ("tree_import_ms", "lcp_import_ms", "sim_one_ms", "sim_sweep_ms",
               "curve_one_ms", "curve_sweep_ms")
    med = {key: statistics.median(row[key] for row in trials) for key in metrics}
    for key in ("one", "sweep"):
        if args.direct_json:
            med[f"sim_json_{key}_ms"] = statistics.median(row[f"sim_json_{key}_ms"] for row in trials)
        med[f"{key}_speedup"] = med[f"sim_{key}_ms"] / med[f"curve_{key}_ms"]
        weighted = statistics.median(row["tree_import_ms"] + row[f"curve_{key}_ms"] for row in trials)
        sim = statistics.median(row["lcp_import_ms"] + row[f"sim_{key}_ms"] for row in trials)
        med[f"{key}_with_import_ms"] = {"weighted": weighted, "prefixsim": sim}
    print(json.dumps(dict(source_sha256=digest, platform=platform.platform(), one=args.one,
                          capacities=caps, trials=trials, median=med), indent=2))


if __name__ == "__main__":
    main()
