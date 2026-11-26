"""Fetch pinned public benchmark inputs using Python 3.9+ and the zstd CLI."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen

MOONCAKE = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/"
    "3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/toolagent_trace.jsonl",
    "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71",
)
CONVERSATION = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/"
    "3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/conversation_trace.jsonl",
    "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df",
)
QWEN_API = (
    "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/"
    "5f7439c51ec248a0c585f7d90a41a6f57773b912/qwen_traceB_blksz_16.jsonl?download=1",
    "68e3f98e2d601d60d0abf4b89bc8a3654372abab7b1cde6373a13d0054379d59",
)
HARVARD = (
    "https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/resolve/"
    "bb26cf72c9c20f2b05f765d88548156a74d3c253/release/2026-05-17_2026-05-23/"
    "traces/2026-05-17.jsonl.zst",
    "f768c3a44d6c7eb02340adcb7bceaf5a451e20dc568d7a96d69f78bfb70e09a8",
)


def download(source, target):
    url, digest = source
    with urlopen(url, timeout=90) as response, target.open("wb") as dest:
        shutil.copyfileobj(response, dest)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError(f"checksum mismatch for {url}")


def project_harvard(source, target):
    # Top-level requests only. Use encoded complete-block token counts so the
    # fixture is independent of the provider tokenizer and has no partial tail.
    rows = []
    with tempfile.TemporaryFile() as decoded:
        subprocess.run(["zstd", "-dc", str(source)], stdout=decoded, check=True)
        decoded.seek(0)
        for line in decoded:
            for req in json.loads(line)["requests"]:
                ids = req["block_ids"]
                rows.append(dict(timestamp=req["start_ms"], input_length=16 * len(ids),
                                 output_length=req["provider_tokens"]["completion"], hash_ids=ids))
    rows.sort(key=lambda row: row["timestamp"])
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = {"mooncake": MOONCAKE, "conversation": CONVERSATION, "harvard": HARVARD,
               "qwen-api": QWEN_API}
    parser.add_argument("dataset", choices=sources)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.output.parent) as work:
        source = Path(work) / "source"
        result = source
        download(sources[args.dataset], source)
        if args.dataset == "harvard":
            result = Path(work) / "projected.jsonl"
            project_harvard(source, result)
        result.replace(args.output)


if __name__ == "__main__":
    main()
