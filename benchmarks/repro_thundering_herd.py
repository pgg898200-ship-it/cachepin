#!/usr/bin/env python3
"""Reproducible Live 32-Agent Cold-Start Thundering-Herd Coalescing Benchmark [LIVE-CPU].

Launches a real local mock LLM server tracking exact cold cache writes (25 ms simulated
prefill), spawns 32 concurrent agents firing the exact same cold prefix simultaneously:
  1. Through `cachepin proxy --no-coalesce` -> 32 redundant cold writes
  2. Through `cachepin proxy` (TTFT-triggered single-flight) -> 1 cold write (-96.88%)

Run:
    python3 benchmarks/repro_thundering_herd.py
"""

from __future__ import annotations

import hashlib
import http.server
import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target" / "release" / "cachepin"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class MockUpstreamState:
    def __init__(self, prefill_ms: float = 25.0) -> None:
        self.prefill_ms = prefill_ms
        self.lock = threading.Lock()
        self.warm: set[str] = set()
        self.cold_writes = 0
        self.warm_hits = 0


def start_mock_server(state: MockUpstreamState) -> tuple[socketserver.ThreadingTCPServer, int]:
    port = free_port()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            clen = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(clen) if clen else b""
            try:
                obj = json.loads(body)
                prefix_bytes = json.dumps(
                    [obj.get("model"), obj.get("tools"), obj.get("system")],
                    separators=(",", ":"),
                ).encode("utf-8")
            except Exception:
                prefix_bytes = body[:4096]
            key = hashlib.sha256(prefix_bytes).hexdigest()
            with state.lock:
                is_warm = key in state.warm
            if not is_warm:
                time.sleep(state.prefill_ms / 1000.0)
                with state.lock:
                    state.cold_writes += 1
                    state.warm.add(key)
            else:
                with state.lock:
                    state.warm_hits += 1
            resp = b'{"id":"msg_herd","usage":{"input_tokens":64,"cache_read_input_tokens":3200}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: object) -> None:
            return

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        request_queue_size = 128

    srv = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def run_burst(proxy_port: int, n_agents: int = 32) -> int:
    base_doc = "Shared repository specification for swarm cold-start test. " * 260
    barrier = threading.Barrier(n_agents)
    ok_count = [0]
    lock = threading.Lock()

    def worker(idx: int) -> None:
        payload = json.dumps(
            {
                "model": "claude-opus-5-5",
                "max_tokens": 256,
                "system": [{"type": "text", "text": f"Current time: 2026-11-22T12:00:{idx:02d}Z\n{base_doc}"}],
                "messages": [{"role": "user", "content": f"Worker {idx}: inspect cache."}],
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json", "X-Api-Key": "repro-herd"},
            method="POST",
        )
        barrier.wait()
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status == 200:
                with lock:
                    ok_count[0] += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_agents)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return ok_count[0]


def wait_health(port: int) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"proxy on port {port} failed healthcheck")


def run_case(coalesce: bool, n_agents: int = 32) -> tuple[int, int, int]:
    state = MockUpstreamState(prefill_ms=25.0)
    srv, up_port = start_mock_server(state)
    p_port = free_port()
    cmd = [
        str(BIN),
        "proxy",
        "--port",
        str(p_port),
        "--upstream",
        f"http://127.0.0.1:{up_port}",
        "--mode",
        "swarm",
    ]
    if not coalesce:
        cmd.append("--no-coalesce")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_health(p_port)
        ok = run_burst(p_port, n_agents=n_agents)
        return ok, state.cold_writes, state.warm_hits
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        srv.shutdown()


def main() -> None:
    if not BIN.exists():
        print(f"Missing binary {BIN}. Run `cargo build --release` first.", file=sys.stderr)
        sys.exit(1)

    ok_raw, writes_raw, hits_raw = run_case(coalesce=False)
    ok_opt, writes_opt, hits_opt = run_case(coalesce=True)
    red = ((writes_raw - writes_opt) / max(writes_raw, 1)) * 100.0
    print("=== Live 32-Agent Cold-Start Thundering Herd [LIVE-CPU] ===")
    print(f"Coalescing OFF : {ok_raw}/32 HTTP 200 | {writes_raw} cold writes | {hits_raw} warm hits")
    print(f"Coalescing ON  : {ok_opt}/32 HTTP 200 | {writes_opt} cold write  | {hits_opt} warm hits")
    print(f"Write reduction: {writes_raw} -> {writes_opt} (-{red:.2f}%)")


if __name__ == "__main__":
    main()
