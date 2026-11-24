#!/usr/bin/env python3
"""Sharp empirical response bounds for delayed races with missing service times."""
import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Duration:
    """Integer microseconds. None means arbitrarily large finite service time."""
    low: int
    high: int | None

    def __post_init__(self):
        if type(self.low) is not int or self.low < 0:
            raise ValueError("invalid lower duration")
        if self.high is not None and (type(self.high) is not int or self.high < self.low):
            raise ValueError("invalid upper duration")


def nonnegative(value):
    token = str(value).strip()
    if not token.isascii() or not token.isdigit():
        raise ValueError(f"expected nonnegative integer microseconds, got {value!r}")
    return int(token)


def observed_duration(status, value):
    """censored means unfinished through t, at_least allows a completion tied at t."""
    if status == "unstarted":
        if value != "":
            raise ValueError("unstarted branch must have a blank duration")
        return Duration(0, None)
    if status == "interval":
        edges = value.split("..")
        if len(edges) != 2:
            raise ValueError("interval requires lower..upper inclusive integer endpoints")
        return Duration(nonnegative(edges[0]), nonnegative(edges[1]))
    t = nonnegative(value)
    if status == "exact":
        return Duration(t, t)
    if status == "censored":
        return Duration(t + 1, None)
    if status == "at_least":
        return Duration(t, None)
    raise ValueError(f"unknown observation status {status!r}")


def from_race(winner, elapsed, old_delay):
    """Instant cancellation, backup launches iff primary exceeds delay; primary wins ties."""
    elapsed = nonnegative(elapsed)
    old_delay = nonnegative(old_delay)
    if winner == "primary":
        backup = Duration(0 if elapsed <= old_delay else elapsed - old_delay, None)
        return Duration(elapsed, elapsed), backup
    if winner == "backup":
        if elapsed < old_delay:
            raise ValueError("backup winner completed before its launch")
        service = elapsed - old_delay
        return Duration(elapsed + 1, None), Duration(service, service)
    raise ValueError(f"unknown winner {winner!r}")


def read_pairs(stream):
    reader = csv.DictReader(stream, strict=True)
    schema = set(reader.fieldnames or [])
    status_schema = {"a_status", "a_us", "b_status", "b_us"}
    race_schema = {"winner", "elapsed_us", "old_delay_us"}
    if schema not in (status_schema, race_schema) or len(reader.fieldnames) != len(schema):
        raise ValueError("expected a_status,a_us,b_status,b_us or winner,elapsed_us,old_delay_us header")
    for row in reader:
        try:
            if None in row or None in row.values():
                raise ValueError("wrong column count")
            if schema == race_schema:
                yield from_race(row["winner"], row["elapsed_us"], row["old_delay_us"])
            else:
                yield (observed_duration(row["a_status"], row["a_us"]),
                       observed_duration(row["b_status"], row["b_us"]))
        except ValueError as exc:
            raise ValueError(f"line {reader.line_num}: {exc}") from exc


class _Endpoint:
    """Index sum(min(a,d+b)) and count(min(a,d+b)>s) over one endpoint per branch."""

    def __init__(self, slo):
        self.slo = slo
        self.base = 0
        self.never_plateau = 0
        self.unbounded = False
        self.breaks = []
        self.miss_starts = []

    def add(self, a, b):
        if b is None:
            if a is None:
                self.unbounded = True
            else:
                self.base += a
        elif a is None:
            self.base += b
            self.never_plateau += 1
        else:
            self.base += min(a, b)
            if a > b:
                self.breaks.append(a - b)
        if a is None or a > self.slo:
            start = 0 if b is None else max(0, self.slo + 1 - b)
            self.miss_starts.append(start)

    def finish(self):
        self.breaks.sort()
        self.miss_starts.sort()
        self.prefix = [0]
        for offset in self.breaks:
            self.prefix.append(self.prefix[-1] + offset)

    def query(self, delay):
        misses = bisect_right(self.miss_starts, delay)
        if self.unbounded:
            return None, misses
        cut = bisect_left(self.breaks, delay)
        total = (self.base + self.prefix[cut]
                 + delay * (self.never_plateau + len(self.breaks) - cut))
        return total, misses


class Bounds:
    """Index reusable across delay queries. One fixed SLO, integer microseconds."""
    def __init__(self, pairs, slo_us):
        self.slo = nonnegative(slo_us)
        self.lower = _Endpoint(self.slo)
        self.upper = _Endpoint(self.slo)
        self.primary_lows = []
        self.primary_finite_highs = []
        self.count = 0
        for a, b in pairs:
            self.primary_lows.append(a.low)
            if a.high is not None:
                self.primary_finite_highs.append(a.high)
            self.lower.add(a.low, b.low)
            self.upper.add(a.high, b.high)
            self.count += 1
        if not self.count:
            raise ValueError("trace has no observations")
        self.primary_lows.sort()
        self.primary_finite_highs.sort()
        self.lower.finish()
        self.upper.finish()

    def query(self, delay_us):
        delay = nonnegative(delay_us)
        lo_sum, lo_miss = self.lower.query(delay)
        hi_sum, hi_miss = self.upper.query(delay)
        # An unknown upper endpoint can exceed every finite launch delay.
        launches = [self.count - bisect_right(self.primary_lows, delay),
                    self.count - bisect_right(self.primary_finite_highs, delay)]
        return {"backup_launches": launches, "delay_us": delay, "sum_response_us": [lo_sum, hi_sum],
                "slo_misses": [lo_miss, hi_miss], "requests": self.count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="CSV file or '-' for stdin")
    parser.add_argument("--slo-us", type=nonnegative, required=True)
    parser.add_argument("--delay-us", type=nonnegative, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        if args.trace == "-":
            bounds = Bounds(read_pairs(sys.stdin), args.slo_us)
        else:
            with Path(args.trace).open(encoding="utf-8", newline="") as stream:
                bounds = Bounds(read_pairs(stream), args.slo_us)
        result = {"slo_us": bounds.slo, "queries": [bounds.query(d) for d in args.delay_us]}
        print(json.dumps(result, separators=(",", ":"), allow_nan=False))
    except (ValueError, csv.Error, OSError) as exc:
        parser.exit(2, f"bounds: {exc}\n")


if __name__ == "__main__":
    main()
