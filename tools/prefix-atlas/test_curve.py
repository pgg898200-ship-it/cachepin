"""Run with python3 test_curve.py /path/to/curve."""
import bisect
import collections
import csv
import io
import json
import pathlib
import unittest
import random
import subprocess
import sys


def invoke(exe, tree, tips, *args):
    data = [f"{len(tree)} {len(tips)}", *(f"{p} {w}" for p, w in tree), *map(str, tips)]
    out = subprocess.run([exe, *args], input="\n".join(data), text=True, capture_output=True, check=True)
    return out.stdout


def run(exe, tree, tips):
    return {key: int(value) for key, value in (line.split() for line in invoke(exe, tree, tips).splitlines())}


def materialize(tree):
    paths = [()]
    for node, (parent, span) in enumerate(tree, 1):
        paths.append(paths[parent] + tuple((node, offset) for offset in range(span)))
    return paths


def enumerate_oracle(tree, tips):
    paths = materialize(tree)
    stack = []
    hist = collections.Counter(cold=0)
    for tip in tips:
        for page in paths[tip]:
            try:
                pos = stack.index(page)
            except ValueError:
                hist["cold"] += 1
            else:
                hist[str(pos + 1)] += 1
                stack.pop(pos)
            stack.insert(0, page)
    return dict(hist)


def prefix_oracle(tree, tips, capacity):
    paths = materialize(tree)
    resident = collections.OrderedDict()
    hits = 0
    total = 0
    for tip in tips:
        path = paths[tip]
        total += len(path)
        for page in path:
            if page not in resident:
                break
            hits += 1
        for page in reversed(path):
            resident[page] = True
            resident.move_to_end(page)
            if len(resident) > capacity:
                resident.popitem(last=False)
    return capacity, hits, total


def prefix_run(exe, tree, tips, capacities):
    raw = invoke(exe, tree, tips, "--prefix-capacities", ",".join(map(str, capacities)))
    reader = csv.DictReader(io.StringIO(raw))
    return [tuple(int(row[field]) for field in ("capacity_pages", "hits", "page_references")) for row in reader]


def check_breakpoints(exe, tree, tips, capacities, expected):
    raw = invoke(exe, tree, tips, "--prefix-curve")
    knots = [(int(r["capacity_pages"]), int(r["hits"]), int(r["hits_per_additional_page"]),
              int(r["page_references"])) for r in csv.DictReader(io.StringIO(raw))]
    points = [k[0] for k in knots]
    assert points[0] == 0 and points == sorted(set(points)) and knots[-1][2] == 0
    for cap, (_, hits, total) in zip(capacities, expected):
        point, value, slope, denominator = knots[bisect.bisect_right(points, cap)-1]
        assert value + (cap-point)*slope == hits and denominator == total


def test_import():
    from mooncake import convert
    records = [
        {"timestamp": 3, "input_length": 33, "hash_ids": [1, 2, 5, 6]},
        {"timestamp": 1, "input_length": 21, "hash_ids": [1, 2, 8]},
        {"timestamp": 1, "input_length": 30, "hash_ids": [1, 2, 3]},
        {"timestamp": 4, "input_length": 9, "hash_ids": [9]},
    ]
    assert convert(map(json.dumps, records), 10) == ([(0, 2), (1, 1), (1, 1)], [1, 3, 2, 0])
    try:
        convert([json.dumps(dict(records[0], input_length=778))], 10)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted inconsistent block count")


def verify(exe):
    test_import()
    for seed in range(400):
        rng = random.Random(seed)
        n = rng.randrange(45)
        tree = [(rng.randrange(i), rng.randint(1, 8)) for i in range(1, n + 1)]
        tips = [rng.randrange(n + 1) for _ in range(rng.randrange(70))]
        assert run(exe, tree, tips) == enumerate_oracle(tree, tips), seed
        caps = sorted({0, 1, 3, 7, 10, 50, 1000, rng.randrange(1, 100)})
        expected = [prefix_oracle(tree, tips, c) for c in caps]
        assert prefix_run(exe, tree, tips, caps) == expected, seed
        check_breakpoints(exe, tree, tips, caps, expected)
    trillion = 10**12
    assert run(exe, [(0, trillion), (1, 3), (1, 2)], [2, 3, 1]) == {
        "cold": trillion + 5, str(trillion + 3): trillion, str(trillion + 2): trillion,
    }
    caps = [0, 1, trillion-1, trillion, trillion+1, 2**64-1]
    assert prefix_run(exe, [(0, trillion), (1, 3), (1, 2)], [2, 3, 1], caps) == [
        (c, 2 * min(c, trillion), 3 * trillion + 5) for c in caps
    ]



class CurveTests(unittest.TestCase):
    def test_explicit_oracles(self):
        verify(str(pathlib.Path(__file__).resolve().with_name("curve")))


if __name__ == "__main__":
    verify(sys.argv[1])
    print("Both modes match independent oracles on 400 seeded trees; trillion-page and import cases passed")
