"""Time both JSONL importers on a Mooncake and a 16-token FreeInference fixture."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
IMPLEMENTATIONS = {"expanded": HERE / "expanded_import.py", "radix": HERE.parent / "mooncake.py"}


def measure(trace, block_size, repeats, work):
    samples = {name: [] for name in IMPLEMENTATIONS}
    tree_hash = None
    for i in range(repeats + 1):
        trials = list(IMPLEMENTATIONS.items())
        if i % 2:
            trials.reverse()
        for name, script in trials:
            dest = work / "output.tree"
            with dest.open("wb") as stream:
                start = time.perf_counter_ns()
                proc = subprocess.Popen([sys.executable, script, "--block-size", str(block_size), trace],
                                        stdout=stream)
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if proc.returncode:
                    raise RuntimeError(f"{name} exited with {proc.returncode}")
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if tree_hash is None:
                tree_hash = digest
            if digest != tree_hash:
                raise RuntimeError(f"{name} produced a different tree")
            if i:
                rss = usage.ru_maxrss / 1024 if sys.platform == "darwin" else usage.ru_maxrss
                samples[name].append({"elapsed_ms": round(elapsed, 3), "peak_rss_kib": int(rss)})
    return {
        "input": trace.name, "input_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "block_size": block_size, "output_sha256": tree_hash, "samples": samples,
        "summary": {
            name: {
                "median_elapsed_ms": statistics.median(row["elapsed_ms"] for row in group),
                "median_peak_rss_kib": statistics.median(row["peak_rss_kib"] for row in group),
            } for name, group in samples.items()
        },
    }


def pipelines(trace, block_size, capacities, repeats, executables):
    samples = {name: [] for name in executables}
    checksum = None
    for i in range(repeats + 1):
        trials = list(executables.items())
        if i % 2:
            trials.reverse()
        for name, binary in trials:
            start = time.perf_counter_ns()
            with subprocess.Popen([sys.executable, IMPLEMENTATIONS["radix"], "--block-size",
                                   str(block_size), trace], stdout=subprocess.PIPE) as importer:
                result = subprocess.run([binary, "--prefix-capacities", capacities],
                                        stdin=importer.stdout, stdout=subprocess.PIPE)
                importer.stdout.close()
                if importer.wait() or result.returncode:
                    raise RuntimeError(f"pipeline failed: {name}")
            elapsed = (time.perf_counter_ns() - start) / 1e6
            digest = hashlib.sha256(result.stdout).hexdigest()
            if checksum is None:
                checksum = digest
            if digest != checksum:
                raise RuntimeError("different capacity curves on identical input")
            if i:
                samples[name].append(round(elapsed, 3))
    return {
        "input": trace.name, "block_size": block_size, "capacities": capacities,
        "output_sha256": checksum, "samples_ms": samples,
        "median_elapsed_ms": {name: statistics.median(times) for name, times in samples.items()},
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mooncake", type=Path)
    parser.add_argument("harvard", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefix-capacities", help="time raw JSONL through both analyzers at these capacities")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fixtures = [(args.mooncake, 512), (args.harvard, 16)]
        if args.prefix_capacities:
            flags = ["-O3", "-DNDEBUG", "-std=c++17", "-march=native"]
            cxx = os.environ.get("CXX", "c++")
            binaries = {}
            for name, source in [("weighted", HERE.parent / "curve.cpp"),
                                 ("expanded", HERE / "expanded_curve.cpp")]:
                binary = work / name
                subprocess.run([cxx, *flags, source, "-o", binary], check=True)
                binaries[name] = binary
            results = [pipelines(trace, size, args.prefix_capacities, args.repeats, binaries)
                       for trace, size in fixtures]
        else:
            results = [measure(trace, size, args.repeats, work) for trace, size in fixtures]
    print(json.dumps({
        "python": sys.version, "machine": platform.machine(), "platform": platform.platform(),
        "warmups": 1, "trials": args.repeats, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
