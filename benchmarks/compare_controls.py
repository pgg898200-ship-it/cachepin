#!/usr/bin/env python3
"""Matched cross-language & end-to-end benchmark harness for `cachepin`.

Measures [LIVE-CPU] matched controls on identical 16 KB multi-agent payloads:
  1. Pure Python 3.12 stdlib canonicalizer + LCP diff
  2. Node.js 22 V8 canonicalizer + LCP diff (`benchmarks/node_control.mjs`)
  3. Native `cachepin` Rust/AVX2 binary + `libcachepin.so` Python FFI
  4. Cold-start CLI startup latency comparison (cachepin vs Python vs Node)
  5. Full `cachepin bench --json` suite (SIMD, DP, Router, Loopback Proxy, 32-Agent Herd)

Writes reproducible machine-readable results to `benchmarks/results.json`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

ISO_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ID_RE = re.compile(r"\b(?:req|trace|span|run|session|call|agent_run)_[0-9a-zA-Z]{8,}\b")


def make_payload(agent_id: int, turn: int) -> bytes:
    tools = []
    order = [(i * 7 + agent_id) % 12 for i in range(12)]
    for ti in order:
        if (agent_id + ti) % 2 == 0:
            tools.append(
                {
                    "name": f"tool_{ti:02d}_exec",
                    "description": f"Execute deterministic workspace inspection step {ti} with structured schema output.",
                    "input_schema": {
                        "type": "object",
                        "required": ["path", "mode"],
                        "properties": {
                            "mode": {"type": "string", "enum": ["fast", "exact"]},
                            "path": {"type": "string", "description": "Relative file path inside workspace"},
                        },
                    },
                }
            )
        else:
            tools.append(
                {
                    "input_schema": {
                        "properties": {
                            "path": {"description": "Relative file path inside workspace", "type": "string"},
                            "mode": {"enum": ["fast", "exact"], "type": "string"},
                        },
                        "required": ["path", "mode"],
                        "type": "object",
                    },
                    "description": f"Execute deterministic workspace inspection step {ti} with structured schema output.",
                    "name": f"tool_{ti:02d}_exec",
                }
            )

    repo_spec = "".join(
        f"Module {k:02d}: maintains strict lock-free invariants, zero-allocation hot path, deterministic ordering, and explicit error contracts across distributed worker nodes. "
        for k in range(48)
    )
    sys_text = (
        f"Current time: 2026-11-22T15:{turn % 60:02d}:{agent_id % 60:02d}Z | "
        f"Trace: req_{(agent_id * 10007 + turn):012x} | "
        f"Session: 550e8400-e29b-41d4-a716-{agent_id:06x}{turn:06x}\n"
        f"You are an autonomous systems engineer working on a shared repository.\n{repo_spec}"
    )
    doc_ids = [(d + agent_id) % 4 for d in range(4)]
    system = [{"type": "text", "text": sys_text}]
    for d in doc_ids:
        system.append(
            {
                "type": "text",
                "text": f"SHARED_REPO_FILE_{d:02d}: "
                + ("pub fn invariant_check(x: u64) -> bool { x.rotate_left(13) != 0 } " * 24),
            }
        )
    req = {
        "model": "claude-opus-5-5",
        "max_tokens": 1024,
        "tools": tools,
        "system": system,
        "messages": [
            {
                "role": "user",
                "content": f"Agent {agent_id} turn {turn}: inspect module {(agent_id + turn) % 48} and propose a patch.",
            }
        ],
    }
    return json.dumps(req, separators=(",", ":")).encode("utf-8")


def python_compile_control(raw: bytes) -> bytes:
    obj = json.loads(raw)
    if isinstance(obj.get("tools"), list):
        obj["tools"].sort(key=lambda t: t.get("name") or t.get("function", {}).get("name", ""))
    bindings: List[Dict[str, str]] = []

    def hoist(text: str) -> str:
        text = text.replace("⟦cp:", "⟦cp:esc:")

        def repl(kind: str):
            def _sub(m: re.Match[str]) -> str:
                anchor = f"⟦cp:v{len(bindings) + 1}⟧"
                bindings.append({"anchor": anchor, "value": m.group(0), "kind": kind})
                return anchor

            return _sub

        text = ISO_RE.sub(repl("iso_timestamp"), text)
        text = UUID_RE.sub(repl("uuid"), text)
        text = ID_RE.sub(repl("opaque_id"), text)
        return text

    if isinstance(obj.get("system"), str):
        obj["system"] = hoist(obj["system"])
    elif isinstance(obj.get("system"), list):
        for blk in obj["system"]:
            if isinstance(blk, dict) and isinstance(blk.get("text"), str):
                blk["text"] = hoist(blk["text"])
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def python_lcp_len(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def pct(vals: List[float], p: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def measure_cold_start(cmd: List[str], runs: int = 15) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(pct(times, 0.5), 2)


def main() -> None:
    bin_path = REPO_ROOT / "target" / "release" / "cachepin"
    if not bin_path.is_file():
        raise SystemExit(f"Binary not found at {bin_path}; run `cargo build --release` first.")

    payloads = [make_payload(i % 16, i // 16) for i in range(64)]

    # 1. Pure Python control
    for p in payloads[:16]:
        python_compile_control(p)
    py_us = []
    for i in range(256):
        t0 = time.perf_counter()
        python_compile_control(payloads[i % len(payloads)])
        py_us.append((time.perf_counter() - t0) * 1e6)

    buf_size = 256 * 1024
    ba = b"A" * buf_size
    bb = b"A" * (buf_size - 17) + b"B" + b"A" * 16
    t_lcp = time.perf_counter()
    for _ in range(10):
        python_lcp_len(ba, bb)
    py_lcp_gb_s = (10 * 2 * (buf_size - 17)) / max(1e-9, time.perf_counter() - t_lcp) / 1e9

    # 2. Node.js 22 V8 control
    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as tf:
        for p in payloads:
            tf.write(p + b"\n")
        tmp_path = tf.name
    try:
        node_out = subprocess.check_output(
            ["node", str(REPO_ROOT / "benchmarks" / "node_control.mjs"), tmp_path],
            text=True,
        )
        node_res = json.loads(node_out)
    finally:
        os.unlink(tmp_path)

    # 3. Native Rust binary bench (--json)
    rust_out = subprocess.check_output([str(bin_path), "bench", "--json"], text=True)
    rust_res = json.loads(rust_out)

    # 4. Python FFI binding (`libcachepin.so`) roundtrip latency
    from cachepin import CachePin, lcp_len

    ffi_us = []
    with CachePin(mode="swarm") as cp:
        for p in payloads[:16]:
            cp.compile_and_plan(p, tenant="ffi-bench", provider="anthropic")
        for i in range(256):
            t0 = time.perf_counter()
            cp.compile_and_plan(payloads[i % len(payloads)], tenant="ffi-bench", provider="anthropic")
            ffi_us.append((time.perf_counter() - t0) * 1e6)
    t_ffi_lcp = time.perf_counter()
    for _ in range(500):
        assert lcp_len(ba, bb) == buf_size - 17
    ffi_lcp_gb_s = (500 * 2 * (buf_size - 17)) / max(1e-9, time.perf_counter() - t_ffi_lcp) / 1e9

    # 5. Cold-start CLI latency comparison
    cold_rust_ms = measure_cold_start([str(bin_path), "--version"])
    cold_py_ms = measure_cold_start([sys.executable, "-c", "import json, re; print('ok')"])
    cold_node_ms = measure_cold_start(["node", "-e", "console.log('ok')"])

    py_p50 = round(pct(py_us, 0.5), 2)
    py_p99 = round(pct(py_us, 0.99), 2)
    ffi_p50 = round(pct(ffi_us, 0.5), 2)
    ffi_p99 = round(pct(ffi_us, 0.99), 2)
    rust_p50 = float(rust_res["compile_p50_us"])

    summary: Dict[str, Any] = {
        "evidence_note": "[LIVE-CPU] = measured live on this CPU; [SIM-TRACE] = deterministic workload/fleet replay.",
        "payload_bytes": len(payloads[0]),
        "cross_language_controls_live_cpu": {
            "cachepin_rust_native": {
                "compile_p50_us": round(rust_p50, 2),
                "compile_p99_us": round(float(rust_res["compile_p99_us"]), 2),
                "lcp_throughput_gb_s": round(float(rust_res["lcp_throughput_gb_s"]), 2),
                "cli_cold_start_ms": cold_rust_ms,
            },
            "cachepin_python_ffi": {
                "compile_p50_us": ffi_p50,
                "compile_p99_us": ffi_p99,
                "lcp_throughput_gb_s": round(ffi_lcp_gb_s, 2),
            },
            "node22_v8_control": {
                "compile_p50_us": node_res["compile_p50_us"],
                "compile_p99_us": node_res["compile_p99_us"],
                "lcp_throughput_gb_s": node_res["lcp_throughput_gb_s"],
                "cli_cold_start_ms": cold_node_ms,
            },
            "python312_stdlib_control": {
                "compile_p50_us": py_p50,
                "compile_p99_us": py_p99,
                "lcp_throughput_gb_s": round(py_lcp_gb_s, 4),
                "cli_cold_start_ms": cold_py_ms,
            },
            "speedups_cachepin_rust": {
                "compile_vs_python_x": round(py_p50 / max(rust_p50, 0.1), 2),
                "compile_vs_node22_x": round(node_res["compile_p50_us"] / max(rust_p50, 0.1), 2),
                "lcp_vs_python_x": round(float(rust_res["lcp_throughput_gb_s"]) / max(py_lcp_gb_s, 1e-6), 1),
                "lcp_vs_node22_x": round(float(rust_res["lcp_throughput_gb_s"]) / max(node_res["lcp_throughput_gb_s"], 1e-6), 1),
                "cold_start_vs_node22_x": round(cold_node_ms / max(cold_rust_ms, 0.1), 2),
                "cold_start_vs_python_x": round(cold_py_ms / max(cold_rust_ms, 0.1), 2),
            },
        },
        "cachepin_full_suite": rust_res,
    }

    out_path = REPO_ROOT / "benchmarks" / "results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
