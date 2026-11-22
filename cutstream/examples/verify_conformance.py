"""Verify a Cutstream v1 corpus against the Python API and an offline oracle.

Each JSONL line is one independent stream. All byte strings use hexadecimal.
Compile markers in listed order, feed chunks in listed order, and record the
newly emitted bytes and number consumed from that chunk (including a match).
Match is the zero-based winner ID or null. Pending is the observed suffix
still held back. Finally call finish and check the EOF object. A second finish
emits zero bytes. Feed calls after a terminal match are outside this corpus.

Run python examples/verify_conformance.py [path/to/corpus.jsonl].
"""

import argparse
import json
from pathlib import Path

from cutstream import Plan


def winner(data, markers):
    """Tuples order by earliest end, then longest marker, then lowest index."""
    matches = []
    for index, marker in enumerate(markers):
        pos = data.find(marker)
        if pos >= 0:
            matches.append((pos + len(marker), -len(marker), index))
    return min(matches, default=None)


def oracle(markers, inputs):
    seen = b""
    emitted = 0
    calls = []
    match = None
    for chunk in inputs:
        if match is not None:
            raise ValueError("Unexpected feed after a terminal match")
        previous = len(seen)
        seen += chunk
        hit = winner(seen, markers)
        pending = b""
        if hit:
            end, neg_length, match = hit
            safe = end + neg_length
            consumed = end - previous
        else:
            consumed = len(chunk)
            safe = len(seen)
            # For a future stop to suppress any observed byte, its marker
            # must cross the boundary. Complete every such candidate and
            # apply the full offline competition to each possible future.
            for marker in markers:
                for overlap in range(1, min(len(marker) - 1, len(seen)) + 1):
                    if seen.endswith(marker[:overlap]):
                        end, neg_length, _ = winner(seen + marker[overlap:], markers)
                        safe = min(safe, end + neg_length)
            pending = seen[safe:]
        calls.append({"chunk": chunk.hex(), "emit": seen[emitted:safe].hex(),
                      "consumed": consumed, "match": match, "pending": pending.hex()})
        emitted = safe
    return calls, {"emit": seen[emitted:].hex() if match is None else "",
                   "match": match, "pending": "", "done": True}


def expect(case_id, phase, actual, expected):
    if actual != expected:
        raise ValueError(f"{case_id} {phase}: observed {actual!r}, expected {expected!r}")


def verify(case):
    if case["version"] != 1:
        raise ValueError(f"Unsupported version {case['version']}")
    case_id = case["id"]
    markers = [bytes.fromhex(value) for value in case["markers"]]
    chunks = [bytes.fromhex(call["chunk"]) for call in case["calls"]]
    expect(case_id, "offline oracle", oracle(markers, chunks), (case["calls"], case["eof"]))
    stream = Plan(markers).stream()
    for i, (chunk, call) in enumerate(zip(chunks, case["calls"])):
        emitted = stream.feed(chunk)
        actual = {"chunk": chunk.hex(), "emit": emitted.hex(), "consumed": stream.consumed,
                  "match": stream.match, "pending": stream.pending.hex()}
        expect(case_id, f"feed {i}", actual, call)
        expect(case_id, f"feed {i} done", stream.done, call["match"] is not None)
    actual_eof = {"emit": stream.finish().hex(), "match": stream.match,
                  "pending": stream.pending.hex(), "done": stream.done}
    expect(case_id, "EOF", actual_eof, case["eof"])
    expect(case_id, "second EOF", stream.finish(), b"")


def main():
    default = Path(__file__).resolve().parents[1] / "evidence" / "conformance.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default=default, type=Path)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("Empty corpus")
    for case in cases:
        verify(case)
    print(f"{len(cases)} v1 cases passed, {sum(len(c['calls']) for c in cases)} feed calls")


if __name__ == "__main__":
    main()
