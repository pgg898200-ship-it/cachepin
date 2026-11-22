#!/usr/bin/env python3
"""Collect complete, paired HTTP response times on loopback with Python 3.

    python3 collect.py --smoke
    python3 collect.py --pairs 1000 --seed 42 --output paired.csv

Two separate local server processes handle identical request IDs. A seed fixes
the simulated service delays. Each replica adds 100-500 us of jitter; the
primary stalls for another 6 ms on roughly 1/12 of calls, the backup for 4 ms
on roughly 1/20. About 1/50 of request IDs add 4 ms on BOTH replicas.
The measured durations also include Python, OS scheduling and loopback HTTP,
so successive runs differ. Connections are warmed before collection.

One pair is outstanding at a time. Both requests run to completion. This is a
constructed, unloaded workload for evaluating paired-trace tooling, not a
production trace or a measurement of the effects of hedging under load.
"""

import argparse
import csv
import hashlib
import http.client
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def number(seed, request_id, channel):
    key = f"{seed}/{request_id}/{channel}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big")


def serve(replica, seed):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            request_id = self.path.removeprefix("/work/")
            if not request_id.isascii() or not request_id.isdecimal():
                self.send_error(400)
                return

            draw = number(seed, request_id, replica)
            jitter_us = 100 + draw % 401
            stall_draw = draw // 401
            if replica == "primary":
                stall_us = 6000 if stall_draw % 12 == 0 else 0
            else:
                stall_us = 4000 if stall_draw % 20 == 0 else 0
            common_us = 4000 if number(seed, request_id, "shared") % 50 == 0 else 0
            time.sleep((jitter_us + stall_us + common_us) / 1_000_000)
            self.send_response(204)
            self.send_header("X-Request-Id", request_id)
            self.end_headers()

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(server.server_port, flush=True)
        server.serve_forever()


@contextmanager
def replica_connection(replica, seed):
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--serve", replica, str(seed)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = proc.stdout.readline().strip()
        if not port.isdecimal():
            raise RuntimeError(f"{replica} failed to start")
        connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=10)
        try:
            yield connection
        finally:
            connection.close()
    finally:
        proc.stdout.close()
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def request(connection, request_id):
    path = f"/work/{request_id}"
    start = time.perf_counter_ns()
    connection.request("GET", path)
    response = connection.getresponse()
    response.read()
    duration_ns = time.perf_counter_ns() - start
    if response.status != 204 or response.getheader("X-Request-Id") != str(request_id):
        raise RuntimeError(f"unexpected response to {path}")
    return (duration_ns + 999) // 1000


def timed_request(connection, request_id, gate):
    gate.wait(timeout=10)
    return request(connection, request_id)


def collect(pairs, seed):
    with ExitStack() as stack:
        primary = stack.enter_context(replica_connection("primary", seed))
        backup = stack.enter_context(replica_connection("backup", seed))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2))
        gate = threading.Barrier(2)
        for request_id in range(pairs + 1):
            first = pool.submit(timed_request, primary, request_id, gate)
            second = pool.submit(timed_request, backup, request_id, gate)
            durations = first.result(), second.result()
            if request_id:  # Request zero opens both HTTP connections.
                yield durations


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--serve":
        serve(sys.argv[2], int(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--pairs", type=int, default=1000, help="number of samples (default 1000)")
    size.add_argument("--smoke", action="store_true", help="collect 12 pairs")
    parser.add_argument("--seed", type=int, default=42, help="service-delay seed (default 42)")
    parser.add_argument("--output", type=Path, help="write CSV to this path (default stdout)")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")

    rows = collect(12 if args.smoke else args.pairs, args.seed)
    output = args.output.open("w", newline="", encoding="utf-8") if args.output else nullcontext(sys.stdout)
    with output as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("primary_us", "backup_us"))
        writer.writerows(rows)


if __name__ == "__main__":
    main()
