"""Replay pinned, public Vercel AI SDK fixture boundaries. No model or key needed."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from cutstream import Plan

REV = "d55f3cfc456d1b7bc7db3a4b70fbf602ddc8647d"
BASE = f"https://raw.githubusercontent.com/vercel/ai/{REV}/packages/openai/src/chat/__fixtures__/"
FIXTURES = ("openai-text", "azure-model-router.1", "openai-refusal")
STOPS = [b"<|end_of_message|>", b"</tool_response>", b"<|im_end|>", b"</answer>"]


def replay(name, raw):
    chunks = []
    for line in raw.splitlines():
        event = json.loads(line)
        for choice in event.get("choices", ()):
            delta = choice.get("delta", {})
            value = delta.get("content") or delta.get("refusal")
            if value:
                chunks.append(value.encode())
    data = b"".join(chunks)
    if not data or any(stop in data for stop in STOPS):
        raise ValueError("fixture no longer matches the no-hit replay contract")
    stream = Plan(STOPS).stream()
    exact_total = 0
    first_exact = first_fixed = None
    seen = 0
    fixed_window = max(map(len, STOPS)) - 1
    fixed_backlog = []
    exact_backlog = []
    for idx, chunk in enumerate(chunks, 1):
        seen += len(chunk)
        exact_total += len(stream.feed(chunk))
        fixed_visible = max(0, seen - fixed_window)
        if first_fixed is None and fixed_visible:
            first_fixed = idx
        if first_exact is None and exact_total:
            first_exact = idx
        fixed_backlog.append(seen - fixed_visible)
        exact_backlog.append(seen - exact_total)
    assert exact_total + len(stream.finish()) == len(data)
    return {
        "fixture": name, "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_length": len(data), "content_deltas": len(chunks),
        "first_visible_delta": {"cutstream": first_exact, "fixed_window": first_fixed},
        "peak_withheld_bytes": {"cutstream": max(exact_backlog), "fixed_window": max(fixed_backlog)}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, help="optional local upstream fixture directory")
    args = parser.parse_args()
    results = []
    for name in FIXTURES:
        filename = name + ".chunks.txt"
        if args.fixtures:
            raw = (args.fixtures / filename).read_bytes()
        else:
            with urlopen(BASE + filename, timeout=20) as response:
                raw = response.read()
        results.append(replay(name, raw))
    print(json.dumps({"revision": REV, "stops": [s.decode() for s in STOPS],
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
