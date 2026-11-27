import argparse
import json
import sys

from . import Plan


def ollama_generate(lines):
    """Decode append-only response text from Ollama /api/generate NDJSON."""
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict):
            raise ValueError("expected a JSON object")
        if "error" in event:
            raise ValueError(f"Ollama error: {event['error']}")
        done = event.get("done", False)
        if not isinstance(done, bool):
            raise ValueError("expected a boolean done field")
        text = event.get("response")
        if "response" in event:
            if not isinstance(text, str):
                raise ValueError("expected a string response field")
            if text:
                yield text.encode("utf-8")
        elif not done and "thinking" not in event:
            raise ValueError("expected a generate response field")
        if done:
            return
    raise ValueError("stream ended before an Ollama done record")


def main():
    parser = argparse.ArgumentParser(description="Stream stdin through a set of literal UTF-8 stop strings.")
    parser.add_argument("--stop", action="append", required=True, metavar="TEXT")
    parser.add_argument("--input-format", choices=("raw", "ollama-generate"), default="raw",
                        help="stdin format (default: raw bytes)")
    args = parser.parse_args()
    if "" in args.stop:
        parser.error("stop strings must be nonempty")
    plan = Plan(stop.encode("utf-8") for stop in args.stop)
    if args.input_format == "ollama-generate":
        chunks = ollama_generate(sys.stdin.buffer)
    else:
        chunks = iter(lambda: sys.stdin.buffer.read1(16384), b"")
    try:
        for chunk in plan.filter(chunks):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    except ValueError as exc:
        parser.exit(1, f"cutstream: {exc}\\n")


if __name__ == "__main__":
    main()
