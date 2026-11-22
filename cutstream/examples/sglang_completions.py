"""Apply a client-owned stop string to SGLang's /v1/completions output.

    python examples/sglang_completions.py
    python examples/sglang_completions.py --endpoint http://localhost:30000/v1/completions

The default replays constructed completion deltas using the text and marker
reported in https://github.com/sgl-project/sglang/issues/50735. Supplying
an endpoint sends a real request. This example requires only Cutstream and
the Python standard library.
"""

import argparse
import json
import sys
from urllib.request import Request, urlopen

from cutstream import Plan


def completion_bytes(lines):
    """SGLang emits one JSON object per SSE data line for a single choice."""
    for line in lines:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(event["error"])
        for choice in event["choices"]:  # A usage-only event can have no choices.
            if choice["index"] != 0:
                raise ValueError("This example supports one completion choice")
            if choice["text"]:
                yield choice["text"].encode("utf-8")
    raise EOFError("Completion connection closed before [DONE]")


def remote(args, plan):
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "max_tokens": 96,
        "stream": True,
        "n": 1,
    }
    # Supply the marker only to the local plan. The gateway needs the full text
    # to decide whether a stop prefix is literal at EOF.
    request = Request(args.endpoint, data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    with urlopen(request, timeout=60) as response:
        for part in plan.filter(completion_bytes(response)):
            sys.stdout.buffer.write(part)
            sys.stdout.buffer.flush()
    # Closing here also ends the HTTP stream when a local stop matches.


def replay():
    def fixture():
        for text in ["All good. ", "Done ", "✅ bye"]:
            event = {"choices": [{"index": 0, "text": text}]}
            yield b"data: " + json.dumps(event).encode() + b"\n"
        raise AssertionError("read past the matching event")

    output = list(Plan(["Done ✅".encode()]).filter(completion_bytes(fixture())))
    assert output == [b"All good. "]
    print("Published before the marker arrived:", repr(output[0].decode()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="full URL of a running SGLang completions endpoint")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--stop", default="Done ✅")
    parser.add_argument("--prompt", default="Please repeat exactly: All good. Done ✅ bye\n")
    args = parser.parse_args()
    if args.endpoint:
        remote(args, Plan([args.stop.encode()]))
    else:
        replay()


if __name__ == "__main__":
    main()
