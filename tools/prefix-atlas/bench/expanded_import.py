"""Convert Mooncake JSONL request prefixes to compact weighted tree rows."""
import argparse
import json
import math
import sys


def convert(records, block_size=512):
    if block_size <= 0:
        raise ValueError("block size must be positive")
    transitions = {}
    parents = [0]
    children = [0]
    is_tip = [True]
    requests = []
    for row_number, line in enumerate(records, 1):
        try:
            row = json.loads(line)
            length = row["input_length"]
            hashes = row["hash_ids"]
            if type(length) is not int or length < 0 or not isinstance(hashes, list):
                raise ValueError("expected a nonnegative integer input_length and a hash_ids array")
            complete, remainder = divmod(length, block_size)
            if len(hashes) not in (complete, complete + bool(remainder)):
                raise ValueError("number of hash IDs disagrees with input length and block size")
            for value in hashes:
                if type(value) is not int:
                    raise ValueError("expected integer hash IDs")
            stamp = row["timestamp"]
            if type(stamp) not in (int, float) or not math.isfinite(stamp):
                raise ValueError("expected a finite numeric timestamp")
            node = 0
            for block in hashes[:complete]:
                key = (node, block)
                found = transitions.get(key)
                if found is None:
                    found = len(parents)
                    transitions[key] = found
                    parents.append(node)
                    children[node] += 1
                    children.append(0)
                    is_tip.append(False)
                node = found
            is_tip[node] = True
            requests.append((stamp, node))
        except (KeyError, TypeError, ValueError, OverflowError) as err:
            raise ValueError(f"record {row_number}: {err}") from err

    # Contract unary chains that never terminate a request into a single edge.
    packed_id = [0] * len(parents)
    span = [0] * len(parents)
    edges = []
    for node in range(1, len(parents)):
        parent = parents[node]
        weight = span[parent] + 1
        if is_tip[node] or children[node] != 1:
            edges.append((packed_id[parent], weight))
            packed_id[node] = len(edges)
        else:
            packed_id[node] = packed_id[parent]
            span[node] = weight
    requests.sort(key=lambda request: request[0])
    return edges, [packed_id[node] for _, node in requests]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", help="read JSONL from this file (default stdin)")
    parser.add_argument("--block-size", type=int, default=512, help="tokens per hash block")
    args = parser.parse_args()
    try:
        if args.trace:
            with open(args.trace, encoding="utf-8") as records:
                edges, tips = convert(records, args.block_size)
        else:
            edges, tips = convert(sys.stdin, args.block_size)
        print(len(edges), len(tips))
        for parent, weight in edges:
            print(parent, weight)
        for tip in tips:
            print(tip)
    except (ValueError, OSError) as err:
        parser.exit(1, f"mooncake: {err}\n")


if __name__ == "__main__":
    main()
