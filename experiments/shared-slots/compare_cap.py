"""Compare primary-only, fixed-delay and prefix-budgeted hedging on the shared-slot workload."""
import argparse
import csv
import gzip
import io
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from run import DEADLINE_NS, DELAY_US, RATES, ROOT, SECONDS, summary, workload


def verify_cap(raw, cap):
    launches = 0
    for count, row in enumerate(csv.DictReader(io.StringIO(raw)), 1):
        launches += int(row["backup_submit_ns"]) >= 0
        if 100 * launches > cap * count:
            raise ValueError(f"launch cap exceeded at request {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory for raw timings and summary")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    compiler = os.environ.get("CXX", "c++")
    policies = [("off", -1, 0), ("fixed", DELAY_US, 100), ("capped", DELAY_US, 25)]
    report = {"compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
              "host": platform.platform(), "deadline_us": DEADLINE_NS // 1000,
              "arrival_window_s": SECONDS, "runs": []}
    with tempfile.TemporaryDirectory() as tmp:
        binary = Path(tmp) / "load"
        subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", ROOT / "load.cpp", "-o", binary], check=True)
        for rate in RATES:
            for rep, seed in enumerate((127995, 127996, 127997)):
                work = workload(rate, seed)
                for name, delay, cap in policies[rep:] + policies[:rep]:
                    result = subprocess.run([binary, str(rate), str(delay), str(cap)], input=work,
                                            text=True, capture_output=True, check=True)
                    raw = result.stdout
                    verify_cap(raw, cap)
                    stats = summary(raw, rate, delay >= 0)
                    filename = f"{rate}-{seed}-{name}.csv.gz"
                    (output / filename).write_bytes(gzip.compress(raw.encode(), mtime=0))
                    report["runs"].append(dict(policy=name, seed=seed, delay_us=delay, cap_pct=cap,
                                               raw=filename, **stats))
                    print(rate, seed, name, stats["misses"], stats["backup_launches"], flush=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
