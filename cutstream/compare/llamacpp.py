"""Benchmark pinned llama.cpp stop-search routines and Cutstream as owned-byte APIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REV = "5bacb7c868a2bc98682de15f447fc1ce17c9c49c"
BASE = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{REV}/"
SDK = "d55f3cfc456d1b7bc7db3a4b70fbf602ddc8647d"
SDK_URL = f"https://raw.githubusercontent.com/vercel/ai/{SDK}/packages/openai/src/chat/__fixtures__/openai-text.chunks.txt"


def fetch(url):
    with urlopen(url, timeout=30) as r:
        return r.read().decode()


def function(src, signature):
    start = src.index(signature)
    depth = 1
    pos = src.index("{", start) + 1
    while depth:
        depth += (src[pos] == "{") - (src[pos] == "}")
        pos += 1
    return src[start:pos]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, help="checkout of the pinned llama.cpp revision")
    ap.add_argument("--fixture", type=Path, help="pinned Vercel openai-text.chunks.txt")
    args = ap.parse_args()
    def source(path):
        return (args.source / path).read_text() if args.source else fetch(BASE + path)

    common = source("common/common.h")
    context = source("tools/server/server-context.cpp")
    fixture = args.fixture.read_text() if args.fixture else fetch(SDK_URL)
    events = (json.loads(line) for line in fixture.splitlines())
    pieces = [e["choices"][0]["delta"].get("content") for e in events if e["choices"]]
    chunks = [s.encode() for s in pieces if s]
    assert len(chunks) == 300
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "common.inc").write_text(function(common, "inline bool string_ends_with(") +
                "\n" + function(common, "inline size_t string_find_partial_stop("))
        (work / "slot.inc").write_text(function(context, "size_t find_stopping_strings("))
        # Hex escapes are separate C++ string literals so follows cannot extend an escape.
        def literal(data):
            return '"' + "".join(f"\\x{b:02x}" for b in data) + '"'
        (work / "chunks.inc").write_text(",\n".join(map(literal, chunks)))
        cmd = [os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-DNDEBUG",
               "-I" + str(ROOT / "include"), "-I" + tmp, str(ROOT / "compare/llamacpp.cpp"),
               "-o", str(work / "compare")]
        subprocess.run(cmd, check=True)
        subprocess.run([str(work / "compare")], check=True)


if __name__ == "__main__":
    main()
