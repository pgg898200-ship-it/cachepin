"""Find the smallest cache meeting a page-hit target on a Prefix Atlas curve."""
import argparse
import csv
from fractions import Fraction
import json
from math import ceil
import sys


COLUMNS = ["capacity_pages", "hits", "hits_per_additional_page", "page_references"]


def solve(source, target):
    reader = csv.reader(source)
    if next(reader, None) != COLUMNS:
        raise ValueError("expected CSV from prefix-atlas --curve")
    previous = None
    answer = None
    for line, row in enumerate(reader, start=2):
        cap, hits, slope, total = map(int, row)
        if not (0 <= cap <= total and 0 <= hits <= total and 0 <= slope <= total):
            raise ValueError(f"out-of-range value on line {line}")
        if previous is None:
            if cap != 0 or hits != 0:
                raise ValueError("the curve must begin at zero")
            required = ceil(total * target / 100)
            if total and required == 0:
                answer = (0, 0)
        else:
            old_cap, old_hits, old_slope, old_total = previous
            if cap <= old_cap or total != old_total or hits != old_hits + (cap - old_cap) * old_slope:
                raise ValueError(f"discontinuous curve on line {line}")
            if answer is None and old_hits < required <= hits:
                extra = (required - old_hits + old_slope - 1) // old_slope
                answer = (old_cap + extra, old_hits + extra * old_slope)
        previous = cap, hits, slope, total
    if previous is None or previous[2] != 0:
        raise ValueError("expected a final breakpoint with zero slope")
    if total == 0:
        raise ValueError("the trace contains no complete page references")
    return {"required_hits": required, "page_references": total, "maximum_hits": hits,
            "minimum_capacity_pages": answer[0] if answer else None,
            "hits_at_minimum": answer[1] if answer else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_percent", type=Fraction, help="percentage, for example 80 or 99.5")
    parser.add_argument("curve", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args()
    if not 0 <= args.target_percent <= 100:
        parser.error("target must be between 0 and 100 percent")
    try:
        print(json.dumps(solve(args.curve, args.target_percent)))
    except (ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
