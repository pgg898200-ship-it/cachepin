"""Synthetic growing history. Compare direct producer output with expanded JSONL plus import."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prefix_atlas import TraceWriter


def direct(path, turns):
    current = 0
    with TraceWriter(path) as trace:
        for _ in range(turns):
            current = trace.extend(current, 1)
            trace.request(current)


def flattened(path, turns):
    # Each turn appends exactly one brand-new complete page. Its prefix ID
    # sequence is known in both arms. Emit the same history to Mooncake JSONL.
    ids = []
    with path.open("w") as stream:
        for turn in range(turns):
            ids.append(turn+1)
            row = {"timestamp": turn, "input_length": len(ids), "hash_ids": ids}
            stream.write(json.dumps(row, separators=(",", ":"))+"\n")


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=3000)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if min(args.turns, args.trials) < 1:
        parser.error("turns and trials must be positive")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        compact, flat, imported = (root/item for item in ("direct.tree", "expanded.jsonl", "import.tree"))
        direct_times, flat_times, import_times = [], [], []
        for trial in range(args.trials):
            if trial % 2 == 0:
                direct_times.append(timed(direct, compact, args.turns))
            flat_times.append(timed(flattened, flat, args.turns))
            with imported.open("wb") as output:
                import_times.append(timed(subprocess.run, [sys.executable, str(ROOT/"mooncake.py"),
                                                            "--block-size", "1", str(flat)], stdout=output,
                                          check=True))
            if trial % 2 == 1:
                direct_times.append(timed(direct, compact, args.turns))
            if compact.read_bytes() != imported.read_bytes():
                raise RuntimeError("producer and imported trees differ")
        print(json.dumps(dict(turns=args.turns, trials=args.trials, expanded_page_ids=args.turns*(args.turns+1)//2,
                              tree_bytes=compact.stat().st_size, jsonl_bytes=flat.stat().st_size,
                              tree_sha256=hashlib.sha256(compact.read_bytes()).hexdigest(),
                              median_direct_seconds=statistics.median(direct_times),
                              median_emit_jsonl_seconds=statistics.median(flat_times),
                              median_import_seconds=statistics.median(import_times),
                              median_emit_and_import_seconds=statistics.median(a+b for a,b in zip(flat_times,import_times))),
                         indent=2))


if __name__ == "__main__":
    main()
