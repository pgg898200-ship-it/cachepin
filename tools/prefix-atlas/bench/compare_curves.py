"""Compare exact histograms and time native analyzers on identical weighted trees."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent


def synthetic(work, name, edges, requests):
    path = work / f"{name}.tree"
    with path.open("w") as out:
        print(len(edges), len(requests), file=out)
        for parent, length in edges:
            print(parent, length, file=out)
        print(*requests, sep="\n", file=out)
    return path


def compare(path, executables, repeats, capacities):
    payload = path.read_bytes()
    values = iter(map(int, payload.split()))
    n, requests = next(values), next(values)
    depths = [0]
    for _ in range(n):
        parent, length = next(values), next(values)
        depths.append(depths[parent] + length)
    touches = sum(depths[next(values)] for _ in range(requests))
    assert next(values, None) is None
    expected = None
    samples = {name: [] for name in executables}
    for trial in range(repeats + 1):
        order = list(executables)
        if trial % 2:
            order.reverse()
        for name in order:
            with path.open("rb") as source:
                start = time.perf_counter_ns()
                argv = [executables[name]]
                if capacities is not None:
                    argv.extend(["--prefix-capacities", capacities])
                raw = subprocess.check_output(argv, stdin=source)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if expected is None:
                expected = raw
            if raw != expected:
                raise RuntimeError(f"histogram differs on {path} for {name}")
            if trial:
                samples[name].append(elapsed_ms)
    if capacities is None:
        counts = [int(line.split()[1]) for line in expected.splitlines()]
        if sum(counts) != touches:
            raise RuntimeError(f"page-access accounting mismatch on {path}")
    else:
        rows = [list(map(int, row.split(b","))) for row in expected.splitlines()[1:]]
        if any(total != touches or not 0 <= hits <= total for _, hits, total in rows):
            raise RuntimeError(f"invalid prefix hit accounting on {path}")
    medians = {key: statistics.median(vals) for key, vals in samples.items()}
    return dict(name=path.stem, nodes=n, requests=requests, page_accesses=touches,
                mode="prefix" if capacities is not None else "forward",
                capacities=capacities, input_sha256=hashlib.sha256(payload).hexdigest(),
                histogram_sha256=hashlib.sha256(expected).hexdigest(),
                milliseconds=samples, median_ms=medians,
                speedup=medians["expanded"]/medians["curve"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trees", type=Path, nargs="*")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--no-synthetic", action="store_true")
    parser.add_argument("--prefix-capacities", help="atomic prefix lookup capacities, comma-separated")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    flags = ["-O3", "-DNDEBUG", "-march=native", "-std=c++17"]
    cxx = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        exes = {}
        for name, source in [("curve", HERE.parent / "curve.cpp"),
                             ("expanded", HERE / "expanded_curve.cpp")]:
            exes[name] = work / name
            subprocess.run([cxx, *flags, source, "-o", exes[name]], check=True)
        trees = args.trees.copy()
        if not args.no_synthetic:
            # Wide recurring prefix: 1,024 pages shared across 128 branches.
            many = [(0, 1024)] + [(1, 4)] * 128
            trees.append(synthetic(work, "shared-1024", many,
                                   [2 + i % 128 for i in range(20000)]))
            # A full depth-one cycle models a stream with no shared prefix.
            trees.append(synthetic(work, "shallow-cycle", [(0, 1)] * 4096,
                                   [1 + i % 4096 for i in range(200000)]))
        results = [compare(path, exes, args.repeats, args.prefix_capacities) for path in trees]
        cpuinfo = Path("/proc/cpuinfo")
        cpu = platform.processor()
        if cpuinfo.exists():
            cpu = next((line.split(":", 1)[1].strip() for line in
                        cpuinfo.read_text().splitlines() if line.startswith("model name")), cpu)
        report = dict(platform=platform.platform(), arch=platform.machine(), cpu=cpu,
                      compiler=subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
                      flags=flags, warmups=1, samples=args.repeats, results=results)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
