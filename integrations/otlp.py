#!/usr/bin/env python3
"""Convert caller-side OTLP JSON race spans into RaceCurve observation CSV."""
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Span:
    parent: str | None
    start: int
    end: int
    tags: dict
    origin: tuple


def identifier(raw, length):
    if not isinstance(raw, str) or not re.fullmatch(f"[a-fA-F0-9]{{{length}}}", raw) or int(raw, 16) == 0:
        raise ValueError(f"expected nonzero {length}-digit OTLP hex ID")
    return raw.lower()


def timestamp(value):
    if type(value) not in (int, str) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError("expected integer epoch nanoseconds")
    result = int(value)
    if result >= 2**64:
        raise ValueError("OTLP timestamp exceeds uint64")
    return result


def batches(stream):
    first = next((line for line in stream if line.strip()), "")
    if not first:
        raise ValueError("empty OTLP input")
    try:
        batch = json.loads(first)
    except json.JSONDecodeError:
        # A single pretty-printed ExportTraceServiceRequest.
        yield json.loads(first + stream.read())
        return
    yield batch
    for line in stream:
        if line.strip():
            yield json.loads(line)


def tagged_spans(stream):
    spans = {}
    for batch in batches(stream):
        for resource in batch["resourceSpans"]:
            ids = {a["key"]: a["value"].get("stringValue")
                   for a in resource.get("resource", {}).get("attributes", [])
                   if a["key"] in ("service.name", "service.instance.id")}
            origin = (ids.get("service.name"), ids.get("service.instance.id"))
            for scope in resource.get("scopeSpans", []):
                for raw in scope.get("spans", []):
                    tags = {}
                    for attr in raw.get("attributes", []):
                        name = attr["key"]
                        if name in ("racecurve.winner", "racecurve.backup_started",
                                    "racecurve.role", "racecurve.outcome"):
                            if name in tags:
                                raise ValueError(f"duplicate attribute {name}")
                            kind = "boolValue" if name == "racecurve.backup_started" else "stringValue"
                            value = attr["value"][kind]
                            if type(value) is not (bool if kind == "boolValue" else str):
                                raise ValueError(f"incorrect OTLP value for {name}")
                            tags[name] = value
                    if not tags:
                        continue
                    trace = identifier(raw["traceId"], 32)
                    span_id = identifier(raw["spanId"], 16)
                    parent = raw.get("parentSpanId")
                    if parent and parent != "0" * 16:
                        parent = identifier(parent, 16)
                    span = Span(parent, timestamp(raw["startTimeUnixNano"]),
                                timestamp(raw["endTimeUnixNano"]), tags, origin)
                    if span.end < span.start:
                        raise ValueError("span ends before it starts")
                    key = (trace, span_id)
                    if key in spans and spans[key] != span:
                        raise ValueError(f"conflicting exports for span {span_id}")
                    spans[key] = span
    return spans


def observations(stream):
    spans = tagged_spans(stream)
    races = {}
    attempts = {}
    for key, span in spans.items():
        tags = span.tags
        if set(tags) == {"racecurve.winner", "racecurve.backup_started"}:
            if tags["racecurve.winner"] not in ("primary", "backup"):
                raise ValueError("unknown winner")
            races[key] = span
        elif set(tags) == {"racecurve.role", "racecurve.outcome"}:
            role = tags["racecurve.role"]
            if role not in ("primary", "backup") or tags["racecurve.outcome"] not in ("complete", "pending"):
                raise ValueError("unknown attempt role or outcome")
            children = attempts.setdefault((key[0], span.parent), {})
            if role in children:
                raise ValueError(f"duplicate {role} attempt")
            children[role] = span
        else:
            raise ValueError(f"incomplete or mixed race attributes on span {key[1]}")
    if attempts.keys() - races.keys():
        raise ValueError("orphan attempt; join requires matching traceId and direct parentSpanId")
    if not races:
        raise ValueError("no tagged race parents")
    rows = []
    for key, race in races.items():
        children = attempts.get(key, {})
        expected = {"primary", "backup"} if race.tags["racecurve.backup_started"] else {"primary"}
        if children.keys() != expected:
            raise ValueError(f"missing or unexpected attempt in race {key[1]}")
        winner = race.tags["racecurve.winner"]
        if winner not in children or children[winner].tags["racecurve.outcome"] != "complete":
            raise ValueError("winner requires a completed attempt span")
        for field in range(2):
            identities = {span.origin[field] for span in [race, *children.values()]
                          if span.origin[field] is not None}
            if len(identities) > 1:
                raise ValueError("tagged spans cross coordinator resource identities")
        decision = children[winner].end
        first = children["primary"].start
        if race.start > first or race.end < decision:
            raise ValueError("race does not contain dispatch and decision")
        row = []
        for role in ("primary", "backup"):
            if role not in children:
                row.extend(("unstarted", ""))
                continue
            child = children[role]
            if not first <= child.start <= decision <= child.end:
                raise ValueError("attempt timing conflicts with dispatch or declared winner")
            if child.tags["racecurve.outcome"] == "complete":
                duration = child.end - child.start
                low, rem = divmod(duration, 1000)
                if rem:
                    row.extend(("interval", f"{low}..{low + 1}"))
                else:
                    row.extend(("exact", low))
            else:
                # We only observed it outstanding at decision; teardown may be much later.
                duration = decision - child.start
                row.extend(("at_least", duration // 1000))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="OTLP JSON file, JSON Lines file or '-' for stdin")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            rows = observations(sys.stdin)
        else:
            with Path(args.input).open(encoding="utf-8") as stream:
                rows = observations(stream)
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(("a_status", "a_us", "b_status", "b_us"))
        writer.writerows(rows)
    except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
        parser.exit(2, f"otlp: {exc}\n")


if __name__ == "__main__":
    main()
