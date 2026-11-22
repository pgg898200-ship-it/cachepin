"""Convert eight preselected measured requests to Chrome Trace Event JSON."""
import argparse
import csv
import gzip
import json
from pathlib import Path


def read_request(path, request_id):
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["id"]) == request_id:
                return {key: int(value) for key, value in row.items()}
    raise ValueError(f"Missing request {request_id} in {path}")


def trace_events(measured):
    pid = 0
    for rate in (400, 1600):
        for seed in (127995, 127996):
            for arm in ("control", "hedge"):
                pid += 1
                row = read_request(measured / f"r{rate}-s{seed}-{arm}.csv.gz", rate)
                start = row["planned_ns"]
                group = f"{rate}/s · {arm} · seed {seed}"
                yield dict(ph="M", pid=pid, tid=0, name="process_name", args={"name": group})
                info = {"offered_rps": rate, "seed": seed, "arm": arm, "request_id": row["id"]}
                for tid, name in enumerate(("Response", "Primary", "Backup"), 1):
                    branch = name.lower()
                    if tid > 1 and row[f"{branch}_submit_ns"] < 0:
                        continue
                    yield dict(ph="M", pid=pid, tid=tid, name="thread_name", args={"name": name})
                    if tid == 1:
                        spans = [("Time to first result", start, row["first_ns"])]
                        yield dict(ph="i", s="t", name="6 ms deadline", pid=pid, tid=tid, ts=6000, args=info)
                    else:
                        stamps = [row[f"{branch}_{field}_ns"] for field in ("submit", "dequeued", "start", "end")]
                        spans = zip(("Replica queue", "Wait for shared slot", "In service"), stamps, stamps[1:])
                    for label, first, last in spans:
                        if first < start or last < first:
                            raise ValueError(f"Invalid interval for {group}, request {row['id']}")
                        yield dict(ph="X", cat="request", name=label, pid=pid, tid=tid,
                                   ts=(first-start)/1000, dur=(last-first)/1000, args=info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measured", type=Path, help="Directory with Racecurve's per-run .csv.gz files")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lines = [json.dumps(event, ensure_ascii=False) for event in trace_events(args.measured)]
    args.output.write_text('{"traceEvents":[\n' + ",\n".join(lines) + '\n]}\n')
