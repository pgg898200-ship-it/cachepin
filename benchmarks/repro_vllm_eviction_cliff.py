#!/usr/bin/env python3
"""Reproducible 8-GPU vLLM/SGLang Eviction-Regime Shadow-KV Routing Benchmark.

Compares Round-Robin, Least-Loaded, Consistent-Hash, SingleHome-2xDivert (Ranvier-style),
and CachePin Adaptive-Replication under tight KV eviction pressure (48 prefixes, Zipf-1.25).

Run:
    python3 benchmarks/repro_vllm_eviction_cliff.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cachepin"


def main() -> None:
    if not BIN.exists():
        print(f"Missing binary {BIN}. Run `cargo build --release` first.", file=sys.stderr)
        sys.exit(1)

    proc = subprocess.run([str(BIN), "bench", "--json"], check=True, capture_output=True, text=True)
    res = json.loads(proc.stdout)
    rows = res["eviction_routing"]

    print("=== 8-GPU vLLM Eviction-Regime Shadow-KV Routing [SIM-TRACE] ===")
    print(f"Live routing decision latency [LIVE-CPU]: P50 = {res['route_p50_us']:.2f} us | P99 = {res['route_p99_us']:.2f} us\n")
    print(f"{'Policy':<36} {'KV Hit %':>9} {'P50 ms':>9} {'P99 ms':>9} {'Mean ms':>9} {'Repl':>6}")
    print("-" * 84)
    for r in rows:
        print(
            f"{r['policy']:<36} {r['kv_hit_rate_pct']:>8.2f}% "
            f"{r['p50_ttft_ms']:>8.1f}  {r['p99_ttft_ms']:>8.1f}  "
            f"{r['mean_ttft_ms']:>8.1f}  {r['replications']:>5}"
        )


if __name__ == "__main__":
    main()
