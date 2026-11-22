"""Zero-dependency HTTPS/HTTP local proxy bridge powered by native `libcachepin.so`.

Enables 1-command local proxying to HTTPS cloud APIs (https://api.anthropic.com,
https://api.openai.com) using Python's stdlib TLS + Rust native canonicalization,
exact DP breakpoint injection, and TTFT-triggered single-flight coalescing.

Usage:
    python3 -m cachepin.https_proxy --upstream https://api.anthropic.com --port 47820
    ANTHROPIC_BASE_URL=http://127.0.0.1:47820 claude
"""

from __future__ import annotations

import argparse
import http.client
import json
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple
from urllib.parse import urlparse

from . import CachePin


class _SingleFlight:
    def __init__(self, max_wait_s: float = 1.5, warm_ttl_s: float = 280.0) -> None:
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._warm: Dict[str, float] = {}
        self._max_wait_s = max_wait_s
        self._warm_ttl_s = warm_ttl_s
        self.leaders = 0
        self.followers_coalesced = 0

    def acquire(self, key: str) -> Tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            if self._warm.get(key, 0.0) > now:
                return (False, 0.0)
            ev = self._inflight.get(key)
            if ev is None:
                self._inflight[key] = threading.Event()
                self.leaders += 1
                return (True, 0.0)
            self.followers_coalesced += 1
        t0 = time.monotonic()
        ev.wait(timeout=self._max_wait_s)
        return (False, (time.monotonic() - t0) * 1000.0)

    def release(self, key: str, warmed: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if warmed:
                self._warm[key] = now + self._warm_ttl_s
            ev = self._inflight.pop(key, None)
            if ev is not None:
                ev.set()


def run_https_proxy(
    host: str = "127.0.0.1",
    port: int = 47820,
    upstream: str = "https://api.anthropic.com",
    mode: str = "swarm",
    provider: str = "auto",
) -> None:
    parsed = urlparse(upstream)
    use_tls = parsed.scheme == "https"
    up_host = parsed.hostname or "api.anthropic.com"
    up_port = parsed.port or (443 if use_tls else 80)
    ssl_ctx = ssl.create_default_context() if use_tls else None

    engine_lock = threading.Lock()
    engine = CachePin(mode=mode)
    coalescer = _SingleFlight()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/health":
                body = b'{"status":"ok","engine":"cachepin-https-bridge"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._forward(b"")

        def do_POST(self) -> None:
            clen = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(clen) if clen > 0 else b""
            self._forward(raw)

        def _forward(self, raw_body: bytes) -> None:
            tenant = (
                self.headers.get("x-api-key")
                or self.headers.get("authorization")
                or "default"
            )
            forward_body = raw_body
            flight_key = None
            is_leader = False
            extra_headers = {}

            if raw_body and any(
                seg in self.path
                for seg in ("/messages", "/chat/completions", "/responses")
            ):
                with engine_lock:
                    res = engine.compile_and_plan(
                        raw_body, tenant=tenant, provider=provider
                    )
                if "error" not in res:
                    opt = res.get("optimized_json") or res.get("canonical_json")
                    if isinstance(opt, str):
                        forward_body = opt.encode("utf-8")
                    elif isinstance(opt, dict):
                        forward_body = json.dumps(opt, separators=(",", ":")).encode("utf-8")
                    diag = res.get("diagnostic", {})
                    extra_headers["X-CachePin-Status"] = str(diag.get("reason", "ok"))
                    extra_headers["X-CachePin-Compile-Us"] = f"{res.get('compile_ns', 0) / 1000.0:.2f}"
                    flight_key = f"{hash(tenant)}:{res.get('shared_prefix_hash', 0)}"
                    is_leader, waited_ms = coalescer.acquire(flight_key)
                    if waited_ms > 0:
                        extra_headers["X-CachePin-Coalesced-Ms"] = f"{waited_ms:.1f}"

            conn = (
                http.client.HTTPSConnection(up_host, up_port, context=ssl_ctx, timeout=120)
                if use_tls
                else http.client.HTTPConnection(up_host, up_port, timeout=120)
            )
            try:
                fwd_headers = {
                    k: v
                    for k, v in self.headers.items()
                    if k.lower() not in ("host", "content-length", "transfer-encoding", "connection")
                }
                fwd_headers["Host"] = up_host
                fwd_headers["Content-Length"] = str(len(forward_body))
                conn.request(self.command, self.path, body=forward_body, headers=fwd_headers)
                resp = conn.getresponse()
                # Release followers IMMEDIATELY on upstream TTFT headers!
                if is_leader and flight_key is not None:
                    coalescer.release(flight_key, warmed=(200 <= resp.status < 300))
                    is_leader = False

                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() == "transfer-encoding":
                        continue
                    self.send_header(k, v)
                for k, v in extra_headers.items():
                    self.send_header(k, v)
                body = resp.read()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                if is_leader and flight_key is not None:
                    coalescer.release(flight_key, warmed=False)
                conn.close()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[cachepin-https] listening on http://{host}:{port} -> {upstream}", flush=True)
    try:
        server.serve_forever()
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="cachepin HTTPS/HTTP local proxy bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47820)
    parser.add_argument("--upstream", default="https://api.anthropic.com")
    parser.add_argument("--mode", default="swarm")
    parser.add_argument("--provider", default="auto")
    args = parser.parse_args()
    run_https_proxy(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        mode=args.mode,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
