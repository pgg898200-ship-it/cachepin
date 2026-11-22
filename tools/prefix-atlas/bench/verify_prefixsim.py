"""Compare exact atomic prefix hits with native PrefixSim LRU on a Mooncake-style trace."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Mooncake-style JSONL")
    parser.add_argument("tree", type=Path, help="tree built from the same JSONL")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--caps", default="256,1024,8192")
    parser.add_argument("--prefixsim", default="prefixsim")
    parser.add_argument("--curve", default="./curve")
    parser.add_argument("--save-normalized", type=Path, help="retain the complete-page JSONL for LCP encoding")
    args = parser.parse_args()
    caps = sorted(set(map(int, args.caps.split(","))))
    with args.tree.open() as tree:
        output = subprocess.check_output([args.curve, "--prefix-capacities", args.caps], stdin=tree, text=True)
    predicted = {int(row["capacity_pages"]): (int(row["hits"]), int(row["page_references"]))
                 for row in csv.DictReader(io.StringIO(output))}
    with tempfile.TemporaryDirectory() as dirname:
        work = Path(dirname)
        trace = work / "full_pages.jsonl"
        rows = []
        with args.trace.open() as source:
            for line in source:
                row = json.loads(line)
                blocks = row["input_length"] // args.block_size
                rows.append((row["timestamp"], row["hash_ids"][:blocks]))
        rows.sort(key=lambda row: row[0])
        if max((len(ids) for _, ids in rows), default=0) > caps[0]:
            parser.error("PrefixSim skips oversized requests; select a capacity that fits the largest request")
        with trace.open("w") as output:
            for timestamp, ids in rows:
                output.write(json.dumps(dict(timestamp=timestamp, input_length=len(ids)*args.block_size,
                                             output_length=0, hash_ids=ids), separators=(",", ":"))+"\n")
        if args.save_normalized:
            shutil.copyfile(trace, args.save_normalized)
        print("capacity_pages,agreed_hits,page_references")
        for cap in caps:
            directory = work / str(cap)
            directory.mkdir()
            subprocess.run([args.prefixsim, "--trace", str(trace), "--cache-size", str(cap), "--algo", "lru",
                            "--replay-order", "deepest-first", "--dump-holes", str(directory)], check=True,
                           stdout=subprocess.DEVNULL)
            dump, = directory.iterdir()
            with dump.open() as holes:
                missing = 0
                dumped = 0
                for line in holes:
                    dumped += 1
                    missing += sum(map(int, re.findall(r"len: (\d+)\)", line)))
            hits, total = predicted[cap]
            if dumped != len(rows) or total != sum(len(ids) for _, ids in rows) or total-missing != hits:
                raise RuntimeError(f"mismatch at {cap}: dump rows={dumped}, misses={missing}, ours={hits}/{total}")
            print(f"{cap},{hits},{total}")


if __name__ == "__main__":
    main()
