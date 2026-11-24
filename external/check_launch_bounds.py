"""Hide losing times in GPUHedge's published holdout and audit backup launch bounds."""
import argparse
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

from moss import SHA256, SLO, URL, pairs_from

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bounds import Bounds, from_race


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the pinned upstream JSONL")
    args = parser.parse_args()
    if args.source:
        data = args.source.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        parser.error("upstream trace checksum mismatch")
    pairs = pairs_from(data)
    if len(pairs) != 54:
        parser.error("expected 54 chronological pairs")
    holdout = pairs[18:]
    old_delay = 10_000_000
    observations = []
    for primary, backup in holdout:
        elapsed = min(primary, old_delay + backup)
        winner = "primary" if primary == elapsed else "backup"
        observations.append(from_race(winner, elapsed, old_delay))
    index = Bounds(observations, SLO)
    print("delay_us,minimum_launches,complete_pair_launches,maximum_launches")
    for delay in (0, old_delay, SLO, 116_100_000, 121_700_000):
        low, high = index.query(delay)["backup_launches"]
        exact = sum(primary > delay for primary, _ in holdout)
        if not low <= exact <= high or (delay == old_delay and low != high):
            raise ValueError(f"invalid launch bounds at {delay} microseconds")
        print(f"{delay},{low},{exact},{high}")


if __name__ == "__main__":
    main()
