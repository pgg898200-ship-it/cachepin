"""Measure prefix-cache reuse for serialized requests using atomic prefix lookup."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys

from .mooncake import convert, write_tree

KERNEL = Path(__file__).with_name("_curve")


def capacities(value):
    for item in value.split(","):
        if not item.isascii() or not item.isdecimal() or len(item) > 20 or int(item) > 2**64 - 1:
            raise argparse.ArgumentTypeError("expected comma-separated unsigned 64-bit page counts")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", metavar="TRACE", help="Mooncake JSONL file, compact .tree file, or - for stdin")
    parser.add_argument("--format", choices=("mooncake", "tree"),
                        help="defaults to tree for .tree files and mooncake for other inputs")
    parser.add_argument("--block-size", type=int, default=512, metavar="TOKENS",
                        help="tokens per Mooncake hash block (default 512)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--capacities", "--prefix-capacities", type=capacities, default="8192,65536",
                      metavar="PAGES,...", help="report CSV for these cache budgets (default 8192,65536)")
    mode.add_argument("--curve", "--prefix-curve", action="store_true",
                      help="export exact piecewise-linear breakpoints for every capacity as CSV")
    args = parser.parse_args(argv)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    kind = args.format or ("tree" if args.trace.endswith(".tree") else "mooncake")
    command = [str(KERNEL), "--prefix-curve"] if args.curve else [
        str(KERNEL), "--prefix-capacities", args.capacities
    ]
    try:
        if kind == "tree":
            source = nullcontext(sys.stdin.buffer) if args.trace == "-" else open(args.trace, "rb")
            with source as tree:
                return subprocess.run(command, stdin=tree).returncode

        source = nullcontext(sys.stdin) if args.trace == "-" else open(args.trace, encoding="utf-8")
        with source as rows:
            edges, tips = convert(rows, args.block_size)
        with subprocess.Popen(command, stdin=subprocess.PIPE, text=True, encoding="ascii") as process:
            with process.stdin as tree:
                write_tree(edges, tips, tree)
            return process.wait()
    except (OSError, ValueError) as error:
        parser.exit(1, f"prefix-atlas: {error}\n")
