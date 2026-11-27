"""Bound single-request hedge outcomes from two complete classic latency histograms."""
import argparse
from bisect import bisect_left
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
from pathlib import Path


def nonnegative_seconds(value):
    try:
        seconds = Decimal(value)
        if seconds.is_finite() and seconds >= 0:
            return Fraction(seconds)
    except InvalidOperation:
        pass
    raise ValueError(f"expected nonnegative finite seconds, got {value!r}")


class Histogram:
    def __init__(self, buckets):
        self.edges = sorted(buckets)
        self.counts = [buckets[edge] for edge in self.edges]
        if not self.edges or self.edges[-1] != math.inf:
            raise ValueError("each arm needs a +Inf bucket")
        if self.counts != sorted(self.counts) or self.counts[-1] == 0:
            raise ValueError("expected nondecreasing cumulative counts and a positive total")
        self.total = self.counts[-1]

    def exceeding(self, threshold):
        if threshold < 0:
            return self.total, self.total
        i = bisect_left(self.edges, threshold)
        minimum = self.total - self.counts[i]
        if self.edges[i] == threshold:
            return minimum, minimum
        previous = self.counts[i - 1] if i else 0
        return minimum, self.total - previous


def read_histograms(response):
    if response["status"] != "success" or response.get("warnings") or response["data"]["resultType"] != "vector":
        raise ValueError("expected a successful, warning-free instant vector")
    arms = {"primary": {}, "backup": {}}
    timestamps = set()
    for series in response["data"]["result"]:
        metric = series["metric"]
        if set(metric) != {"arm", "le"} or metric["arm"] not in arms:
            raise ValueError("aggregate by (arm, le) with primary and backup labels")
        timestamp, raw_count = series["value"]
        timestamps.add(timestamp)
        count = Decimal(raw_count)
        if not count.is_finite() or not 0 <= count < 2**53 or count != count.to_integral_value():
            raise ValueError("expected exactly represented nonnegative integer bucket counts")
        edge = math.inf if metric["le"] == "+Inf" else nonnegative_seconds(metric["le"])
        buckets = arms[metric["arm"]]
        if edge in buckets:
            raise ValueError("duplicate bucket boundary")
        buckets[edge] = int(count)
    if len(timestamps) != 1:
        raise ValueError("expected one common evaluation timestamp")
    primary, backup = (Histogram(arms[name]) for name in ("primary", "backup"))
    if primary.total != backup.total:
        raise ValueError("arms must cover the same complete request cohort")
    return primary, backup


def query(primary, backup, deadline, delay):
    p_low, p_high = primary.exceeding(deadline)
    b_low, b_high = backup.exceeding(deadline - delay)
    return {"requests": primary.total, "backup_launches": primary.exceeding(delay),
            "slo_misses": [max(0, p_low + b_low - primary.total), min(p_high, b_high)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vector", type=Path, help="Prometheus /api/v1/query JSON response")
    parser.add_argument("--deadline-s", required=True, type=nonnegative_seconds)
    parser.add_argument("--delay-s", nargs="+", required=True, metavar="SECONDS")
    args = parser.parse_args()
    try:
        primary, backup = read_histograms(json.loads(args.vector.read_text()))
        rows = [dict(delay_s=d, **query(primary, backup, args.deadline_s, nonnegative_seconds(d)))
                for d in args.delay_s]
    except (OSError, ValueError, KeyError, TypeError, InvalidOperation) as exc:
        parser.exit(2, f"invalid histogram input: {exc}\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
