"""Scrape GPUHedge holdout histograms with Prometheus and audit the resulting bounds."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from moss import SHA256, URL, pairs_from

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from prometheus_bounds import query, read_histograms

METRIC = "shadow_duration_seconds"
PROMQL = f"sum by (arm, le) ({METRIC}_bucket)"


def exposition(pairs):
    lines = [f"# TYPE {METRIC} histogram"]
    for i, arm in enumerate(("primary", "backup")):
        times = [row[i] for row in pairs]
        for edge in (0, 10, 30, 50, 60, 120, 300):
            count = sum(t <= edge * 1_000_000 for t in times)
            lines.append(f'{METRIC}_bucket{{arm="{arm}",le="{edge}"}} {count}')
        lines.extend([f'{METRIC}_bucket{{arm="{arm}",le="+Inf"}} {len(times)}',
                      f'{METRIC}_count{{arm="{arm}"}} {len(times)}',
                      f'{METRIC}_sum{{arm="{arm}"}} {sum(times) / 1_000_000:.6f}'])
    return ("\n".join(lines) + "\n").encode()


def scrape(binary, payload):
    class Metrics(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    with HTTPServer(("127.0.0.1", 0), Metrics) as server, tempfile.TemporaryDirectory() as tmp:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = Path(tmp) / "prom.yml"
        config.write_text("global:\n  scrape_interval: 1s\nscrape_configs:\n  - job_name: shadow\n"
                          f"    static_configs:\n      - targets: ['127.0.0.1:{server.server_port}']\n")
        command = [binary, f"--config.file={config}", f"--storage.tsdb.path={tmp}/data",
                   f"--web.listen-address=127.0.0.1:{port}"]
        api = f"http://127.0.0.1:{port}/api/v1/query?" + urlencode({"query": PROMQL})
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
            try:
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("Prometheus exited before the first scrape")
                    try:
                        with urlopen(api, timeout=2) as response:
                            vector = json.load(response)
                        if vector.get("data", {}).get("result"):
                            return vector
                    except URLError:
                        pass
                    time.sleep(0.25)
                raise RuntimeError("timed out waiting for histogram buckets")
            finally:
                process.terminate()
                server.shutdown()
                thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus", default="prometheus", help="Prometheus server executable")
    parser.add_argument("--source", type=Path, help="local copy of the pinned upstream JSONL")
    parser.add_argument("--vector-out", type=Path, help="save the actual API response")
    args = parser.parse_args()
    if args.source:
        data = args.source.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        parser.error("trace checksum mismatch")
    pairs = pairs_from(data)[18:]
    vector = scrape(args.prometheus, exposition(pairs))
    histograms = read_histograms(vector)
    if args.vector_out:
        args.vector_out.write_text(json.dumps(vector, indent=2) + "\n")
    print("delay_s,launch_min,launch_actual,launch_max,miss_min,miss_actual,miss_max")
    for delay in (0, 10, 20, 60, 120):
        result = query(*histograms, 60, delay)
        lmin, lmax = result["backup_launches"]
        mmin, mmax = result["slo_misses"]
        launches = sum(p > delay * 1_000_000 for p, _ in pairs)
        misses = sum(min(p, delay * 1_000_000 + b) > 60_000_000 for p, b in pairs)
        if result["requests"] != len(pairs) or not (lmin <= launches <= lmax and mmin <= misses <= mmax):
            raise ValueError("histogram bounds disagree with paired replay")
        print(delay, lmin, launches, lmax, mmin, misses, mmax, sep=",")


if __name__ == "__main__":
    main()
