"""Count compulsory prefix-page misses independently of the Prefix Atlas tree."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Mooncake-style JSONL with integer hash_ids")
    parser.add_argument("--block-size", type=int, required=True)
    args = parser.parse_args()
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    paths = []
    block_ids = set()
    references = 0
    with open(args.trace, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            row = json.loads(line)
            complete, tail = divmod(row["input_length"], args.block_size)
            hashes = row["hash_ids"]
            if complete < 0 or len(hashes) not in (complete, complete + bool(tail)):
                parser.error(f"record {number} has inconsistent length and block count")
            path = hashes[:complete]
            paths.append(path)
            block_ids.update(path)
            references += complete

    # In lexical order, every previously seen prefix is shared with the predecessor.
    paths.sort()
    distinct = 0
    previous = []
    for path in paths:
        common = 0
        for left, right in zip(previous, path):
            if left != right:
                break
            common += 1
        distinct += len(path) - common
        previous = path
    print("requests,page_references,distinct_block_ids,distinct_prefix_pages,max_hits")
    print(len(paths), references, len(block_ids), distinct, references - distinct, sep=",")


if __name__ == "__main__":
    main()
