"""Convert Mooncake JSONL request prefixes to compact weighted tree rows."""
import argparse
import json
import math
import sys


class Edge:
    """A token span with its end node's original first-insertion ordinal."""

    __slots__ = ("parent_ordinal", "tokens", "start", "stop", "ordinal", "children")

    def __init__(self, parent_ordinal, tokens, start, stop, ordinal):
        self.parent_ordinal = parent_ordinal
        self.tokens = tokens
        self.start = start
        self.stop = stop
        self.ordinal = ordinal
        self.children = {}


class PrefixTree:
    def __init__(self):
        self.root = Edge(0, [], 0, 0, 0)
        self.edges = []
        self.last_ordinal = 0

    def insert(self, hashes, stop):
        parent = self.root
        pos = 0
        while pos < stop:
            edge = parent.children.get(hashes[pos])
            if edge is None:
                break

            # The first ID identifies the edge. Long matches use native list
            # comparison instead of hashing every (parent, block) pair in Python.
            n = min(edge.stop - edge.start, stop - pos)
            common = 1
            if n > 1:
                if edge.tokens[edge.start+1:edge.start+n] == hashes[pos+1:pos+n]:
                    common = n
                else:
                    while common < n and edge.tokens[edge.start+common] == hashes[pos+common]:
                        common += 1
            pos += common
            if edge.start + common == edge.stop:
                parent = edge
                continue

            # Split at the new branch or shorter request tip. Both sides retain
            # the same backing label and the ordinals from its first insertion.
            split = edge.start + common
            junction = Edge(parent.ordinal, edge.tokens, edge.start, split,
                            edge.ordinal - (edge.stop - split))
            parent.children[edge.tokens[edge.start]] = junction
            junction.children[edge.tokens[split]] = edge
            edge.start = split
            edge.parent_ordinal = junction.ordinal
            self.edges.append(junction)
            parent = junction
            break

        if pos < stop:
            suffix = hashes[pos:stop]
            self.last_ordinal += len(suffix)
            edge = Edge(parent.ordinal, suffix, 0, len(suffix), self.last_ordinal)
            parent.children[suffix[0]] = edge
            self.edges.append(edge)
            parent = edge
        return parent.ordinal

    def pack(self, requests):
        # The historical importer numbered one node per block as first observed,
        # then removed unary nontips. These ordinals preserve identical output.
        edges = sorted(self.edges, key=lambda edge: edge.ordinal)
        number = {edge.ordinal: i for i, edge in enumerate(edges, 1)}
        number[0] = 0
        weighted = [(number[edge.parent_ordinal], edge.stop - edge.start) for edge in edges]
        requests.sort(key=lambda request: request[0])
        return weighted, [number[node] for _, node in requests]


def convert(records, block_size=512):
    if block_size <= 0:
        raise ValueError("block size must be positive")
    tree = PrefixTree()
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
            node = tree.insert(hashes, complete)
            requests.append((stamp, node))
        except (KeyError, TypeError, ValueError, OverflowError) as err:
            raise ValueError(f"record {row_number}: {err}") from err

    return tree.pack(requests)


def write_tree(edges, tips, stream):
    print(len(edges), len(tips), file=stream)
    for parent, weight in edges:
        print(parent, weight, file=stream)
    for tip in tips:
        print(tip, file=stream)


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
        write_tree(edges, tips, sys.stdout)
    except (ValueError, OSError) as err:
        parser.exit(1, f"mooncake: {err}\n")


if __name__ == "__main__":
    main()
