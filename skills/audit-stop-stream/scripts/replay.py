"""Replay one choice from a JSONL capture of OpenAI-compatible chat completion chunks."""
import argparse
import json
import sys

from cutstream import Plan


def replay(source, markers, choice_index):
    stream = Plan(marker.encode("utf-8") for marker in markers).stream()
    total = 0
    deltas = 0
    for line_no, line in enumerate(source, start=1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
            choices = chunk["choices"]
            if not isinstance(choices, list):
                raise ValueError("choices must be a list")
            for choice in choices:
                if choice["index"] != choice_index:
                    continue
                delta = choice["delta"].get("content")
                if delta is None or delta == "":
                    continue
                if not isinstance(delta, str):
                    raise ValueError("expected string delta.content")
                data = delta.encode("utf-8")
                emitted = stream.feed(data)
                deltas += 1
                total += stream.consumed
                yield {"line": line_no, "emitted_hex": emitted.hex(), "pending_hex": stream.pending.hex(),
                       "consumed_bytes": stream.consumed}
                if stream.done:
                    yield {"terminal": "stop", "marker": markers[stream.match], "line": line_no,
                           "consumed_bytes_total": total, "ignored_bytes_in_delta": len(data) - stream.consumed}
                    return
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"line {line_no}: {error}") from error
    yield {"terminal": "eof", "deltas": deltas, "consumed_bytes_total": total,
           "emitted_hex": stream.finish().hex()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", required=True, action="append", metavar="TEXT")
    parser.add_argument("--choice", type=int, default=0, help="completion choice index")
    parser.add_argument("capture", nargs="?", type=argparse.FileType("r", encoding="utf-8"), default=sys.stdin)
    args = parser.parse_args()
    if args.choice < 0 or any(not marker for marker in args.stop):
        parser.error("choose a nonnegative choice index and nonempty stop strings")
    try:
        for record in replay(args.capture, args.stop, args.choice):
            print(json.dumps(record, ensure_ascii=True))
    except ValueError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
