import argparse
import sys

from . import Plan


def main():
    parser = argparse.ArgumentParser(description="Stream stdin through a set of literal UTF-8 stop strings.")
    parser.add_argument("--stop", action="append", required=True, metavar="TEXT")
    args = parser.parse_args()
    if "" in args.stop:
        parser.error("stop strings must be nonempty")
    plan = Plan(stop.encode("utf-8") for stop in args.stop)
    for chunk in plan.filter(iter(lambda: sys.stdin.buffer.read1(16384), b"")):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
