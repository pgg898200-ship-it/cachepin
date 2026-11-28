"""Check isolated session/model caches on a pinned Weka production trace sample."""
import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

SAMPLE_BYTES = 32 * 1024**2
SAMPLE_SHA256 = "0bf39132d3a36ff7efe90c221902ebd3aaaf391fb5650f4c7da732b4a30e66e3"
CAPACITIES = (512, 2048, 5836, 5837, 8192, 65536)


def direct_replay(paths, capacity):
    cache = OrderedDict()
    hits = 0
    for path in paths:
        for page in path:
            if page not in cache:
                break
            hits += 1
        for page in reversed(path):
            cache[page] = None
            cache.move_to_end(page)
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits


def measure(requests, cli):
    page_ids = {}
    paths = []
    records = []
    previous = []
    adjacent = [0] * len(CAPACITIES)
    for row in sorted(requests, key=lambda r: r["t"]):
        length = row["in"]
        hashes = row["hash_ids"]
        records.append(json.dumps(dict(timestamp=row["t"], input_length=length, hash_ids=hashes)))
        path = []
        parent = 0
        for block in hashes[:length // 64]:
            parent = page_ids.setdefault((parent, block), len(page_ids) + 1)
            path.append(parent)
        common = 0
        for left, right in zip(previous, path):
            if left != right:
                break
            common += 1
        for i, capacity in enumerate(CAPACITIES):
            adjacent[i] += min(common, capacity)
        previous = path
        paths.append(path)
    total = sum(map(len, paths))
    budgets = ",".join(map(str, CAPACITIES))
    output = subprocess.check_output(
        [cli, "-", "--block-size", "64", "--capacities", budgets],
        input="\n".join(records), text=True,
    )
    reported = list(csv.DictReader(io.StringIO(output)))
    results = []
    for i, capacity in enumerate(CAPACITIES):
        hits = direct_replay(paths, capacity)
        results.append((capacity, hits, adjacent[i]))
    expected = [(c, h, total) for c, h, _ in results]
    actual = [(int(r["capacity_pages"]), int(r["hits"]), int(r["page_references"])) for r in reported]
    if actual != expected:
        raise ValueError(f"CLI and direct replay disagree: {actual} != {expected}")
    return total, total - len(page_ids), results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", type=Path)
    p.add_argument("--cli", default="prefix-atlas")
    args = p.parse_args()
    with args.sample.open("rb") as source:
        data = source.read(SAMPLE_BYTES)
    if hashlib.sha256(data).hexdigest() != SAMPLE_SHA256:
        p.error("expected the pinned first 32 MiB of traces.jsonl")
    # Discard the incomplete next session at the byte boundary.
    sessions = data.split(b"\n")[:-1]
    totals = {capacity: [0, 0] for capacity in CAPACITIES}
    requests = references = streams = maximum_hits = 0
    for line in sessions:
        session = json.loads(line)
        if (session["hash_id_scope"], session["block_size"]) != ("local", 64):
            raise ValueError("unexpected hash scope or block size")
        models = defaultdict(list)
        for row in session["requests"]:
            models[row["model"]].append(row)
        for rows in models.values():
            count, ceiling, results = measure(rows, args.cli)
            maximum_hits += ceiling
            streams += 1
            requests += len(rows)
            references += count
            for capacity, hits, adjacent in results:
                totals[capacity][0] += hits
                totals[capacity][1] += adjacent
    print(f"{len(sessions)} sessions, {streams} session/model streams, {requests} requests")
    print(f"Unlimited-cache ceiling: {maximum_hits} hits")
    print("capacity_pages,lru_hits,previous_request_hits,page_references")
    for capacity, (hits, adjacent) in totals.items():
        print(capacity, hits, adjacent, references, sep=",")


if __name__ == "__main__":
    main()
