"""Compare the forward page-stream histogram with upstream KVSET token-ID replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Mooncake-style input JSONL")
    parser.add_argument("tree", type=Path)
    parser.add_argument("kvset", type=Path, help="KVSET checkout containing src/token_ids_replay.py")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--caps", default="256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--curve", default="./curve")
    args = parser.parse_args()
    with args.tree.open() as tree:
        hist = subprocess.check_output([args.curve], stdin=tree, text=True).splitlines()
    cold = int(hist[0].split()[1])
    histogram = [tuple(map(int, line.split())) for line in hist[1:]]
    with tempfile.TemporaryDirectory() as temp:
        trace = Path(temp) / "input_ids.jsonl"
        rows = []
        with args.trace.open() as raw:
            for line in raw:
                row = json.loads(line)
                n = row["input_length"] // args.block_size
                rows.append((row["timestamp"], row["hash_ids"][:n]))
        rows.sort(key=lambda row: row[0])
        with trace.open("w") as out:
            for _, ids in rows:
                out.write(json.dumps(ids, separators=(",", ":")) + "\n")
        data = subprocess.check_output([sys.executable, str(args.kvset/"src"/"token_ids_replay.py"), str(trace),
                                       "--kv-bytes-per-token", "1", "--page-size", "1", "--capacities", args.caps,
                                       "--warm-up-percent", "0", "--log-level", "ERROR", "--out-format", "json"])
    result = json.loads(data)
    total = cold + sum(n for _, n in histogram)
    if (total, cold) != (result["page_accesses"], result["unique_pages"]):
        raise RuntimeError("different reference count or cold count")
    expected = Counter(map(int, args.caps.split(",")))
    print("capacity_pages,agreed_forward_hits,page_references")
    for row in result["results"]:
        cap = row["capacity_pages"]
        if cap is None:
            continue
        if sum(n for threshold, n in histogram if threshold <= cap) != row["page_hits"]:
            raise RuntimeError(f"mismatch at {cap}")
        print(f'{cap},{row["page_hits"]},{total}')
        expected[cap] -= 1
    if any(expected.values()):
        raise RuntimeError("upstream result omitted a requested capacity")


if __name__ == "__main__":
    main()
