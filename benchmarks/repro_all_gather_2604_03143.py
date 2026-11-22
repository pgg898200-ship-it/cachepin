#!/usr/bin/env python3
"""Reproducible benchmark for arXiv:2604.03143 — All-Gather Collective Multi-Agent
Prefix Cache Collapse (raw permuted peer observations vs cachepin Pass 3 AllGatherAlign).

Run:
    python3 benchmarks/repro_all_gather_2604_03143.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cachepin"


def build_trace(n_agents: int = 16, n_rounds: int = 6, seed: int = 260403143) -> str:
    rng = random.Random(seed)
    shared_chunks = [
        f"ROUND_SHARED_OBSERVATION_{c:02d}: "
        + ("telemetry_vector=[0.42,-1.18,3.05,0.91] consensus_verified=true " * 28)
        for c in range(8)
    ]
    lines: list[str] = []
    for r in range(n_rounds):
        for a in range(n_agents):
            order = list(range(8))
            rng.shuffle(order)
            sys_blocks = [
                {"type": "text", "text": "Shared multi-agent research swarm coordinator protocol v1."}
            ] + [{"type": "text", "text": shared_chunks[c]} for c in order]
            payload = {
                "model": "claude-opus-5-5",
                "max_tokens": 512,
                "system": sys_blocks,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Round {r} Agent {a} private hypothesis: evaluate observation {(a + r) % 8}.",
                    }
                ],
            }
            lines.append(json.dumps(payload, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def main() -> None:
    if not BIN.exists():
        print(f"Missing binary {BIN}. Run `cargo build --release` first.", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tf:
        tf.write(build_trace())
        trace_path = Path(tf.name)

    try:
        proc = subprocess.run(
            [str(BIN), "audit", str(trace_path), "--mode", "swarm", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        rep = json.loads(proc.stdout)
        print("=== arXiv:2604.03143 All-Gather Multi-Agent Prefix Collapse [SIM-TRACE] ===")
        print(f"Requests analyzed       : {rep['requests_analyzed']} (16 agents x 6 rounds)")
        print(f"All-Gather repairs      : {rep['repairs']['all_gather_reordered']} requests aligned")
        print(f"RAW hit rate            : {rep['raw_hit_rate_pct']:.2f}%  (est. ${rep['raw_est_cost_usd']:.4f})")
        print(f"AFTER cachepin hit rate : {rep['opt_hit_rate_pct']:.2f}%  (est. ${rep['opt_est_cost_usd']:.4f})")
        print(f"Cost reduction          : -{rep['cost_reduction_pct']:.2f}% ({rep['cost_multiplier_saved']:.2f}x cheaper)")
    finally:
        trace_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
