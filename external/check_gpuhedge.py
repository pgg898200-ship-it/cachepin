"""Compare the two shared outputs with GPUHedge's own pinned replay implementation."""
import argparse
import hashlib
from pathlib import Path
import sys

from moss import SHA256, SLO, curve, pairs_from


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="GPUHedge source checkout at the documented commit")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "racecurve")
    parser.add_argument("--trace", type=Path, help="default uses the trace in the GPUHedge checkout")
    args = parser.parse_args()
    src = args.checkout.resolve() / "src"
    sys.path.insert(0, str(src))
    from gpuhedge.benchmark.replay import evaluate_hedge, load_rounds
    from gpuhedge.config import load_config

    source = args.trace or args.checkout / "traces" / "moss_rounds.jsonl"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        parser.error("trace SHA-256 mismatch")
    rows = load_rounds(source)
    pairs = pairs_from(raw)
    config = load_config(src / "gpuhedge" / "config" / "benchmark.yaml")
    checked = 0
    for partition in (slice(0, 18), slice(18, 54)):
        rounds = rows[partition]
        for point in curve(args.binary, pairs[partition]):
            delay = point["delay_us"] / 1_000_000
            deadline = SLO // 1_000_000
            reference = evaluate_hedge(config, rounds, "runpod", "modal", delay, [deadline])
            launches = round(reference.hedge_rate * len(rounds))
            if (reference.miss_counts[deadline], launches) != (point["leaf_misses"], point["launches"]):
                raise ValueError(f"GPUHedge disagreement at {delay}s")
            checked += 1
    print(f"GPUHedge and Racecurve agree on launches and misses at {checked} partition/delay combinations.")


if __name__ == "__main__":
    main()
