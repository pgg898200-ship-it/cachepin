#!/usr/bin/env python3
"""Self-contained 30-second demo showing multi-agent prompt cache collapse and recovery.

Generates a 32-agent coding swarm JSONL trace with realistic cache-busters:
  - Non-deterministic MCP tool registration order
  - Permuted JSON schema keys
  - Volatile timestamps, UUIDs, and request IDs in the system prefix
  - Permuted All-Gather shared repository files across agents

Then runs `cachepin audit` and `cachepin` Python FFI optimization on it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from compare_controls import make_payload  # noqa: E402


def main() -> None:
    bin_path = REPO_ROOT / "target" / "release" / "cachepin"
    if not bin_path.is_file():
        print("Build first with: cargo build --release")
        raise SystemExit(1)

    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as f:
        trace_path = Path(f.name)
        for step in range(128):
            f.write(make_payload(step % 32, step // 32) + b"\n")

    try:
        print(f"Generated 32-agent x 4-turn swarm trace ({trace_path.stat().st_size / 1024:.1f} KiB)\n")
        subprocess.run([str(bin_path), "audit", str(trace_path)], check=True)
    finally:
        trace_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
