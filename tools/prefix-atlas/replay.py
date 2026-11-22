#!/usr/bin/env python3
"""Export one-request-per-row Mooncake JSONL for GuideLLM replay."""
import argparse
from contextlib import nullcontext
from decimal import Decimal, DecimalException
import json
import math
from pathlib import Path
import sys
import tempfile

INT32_MAX = 2**31 - 1
UINT64_MAX = 2**64 - 1
TIME_UNITS = {"s": 1, "ms": 1000, "us": 10**6, "ns": 10**9}


def integer(value, maximum, name):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return value


class BlockIds:
    """Assign unique signed-32-bit IDs to labeled trie edges across all rows."""

    def __init__(self):
        self.paths = {}
        self.widths = {}
        self.next_spare = INT32_MAX

    def intern(self, parent, raw, width):
        raw = integer(raw, UINT64_MAX, "hash_id")
        key = (parent, raw)
        if key in self.paths:
            block = self.paths[key]
            if self.widths[block] != width:
                raise ValueError("a repeated block path has inconsistent token counts")
            return block
        if raw <= INT32_MAX and raw not in self.widths:
            block = raw
        else:
            while self.next_spare in self.widths:
                self.next_spare -= 1
            if self.next_spare < 0:
                raise ValueError("more unique block paths than signed int32 can represent")
            block = self.next_spare
            self.next_spare -= 1
        self.paths[key] = block
        self.widths[block] = width
        return block


def export(source, dest, block_size, time_unit):
    """Stream records in arrival order, retaining one dictionary entry per block path."""
    if block_size <= 0:
        raise ValueError("block size must be positive")
    ids = BlockIds()
    origin = previous = None
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_float=Decimal)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            timestamp = row["timestamp"]
            if type(timestamp) not in (int, Decimal) or not Decimal(timestamp).is_finite():
                raise ValueError("timestamp must be a finite number")
            timestamp = Decimal(timestamp)
            if previous is not None and timestamp < previous:
                raise ValueError("timestamps must be in arrival order")
            if origin is None:
                origin = timestamp
            previous = timestamp
            relative = float((timestamp - origin) / TIME_UNITS[time_unit])
            if not math.isfinite(relative) or relative > 3.4028234663852886e38:
                raise ValueError("timestamp exceeds GuideLLM's float32 range")
            prompt = integer(row["input_length"], INT32_MAX, "input_length")
            output = integer(row["output_length"], INT32_MAX, "output_length")
            path = row["hash_ids"]
            if not isinstance(path, list) or len(path) != (prompt + block_size - 1) // block_size:
                raise ValueError("hash_ids length must equal ceil(input_length / block_size)")
            parent = None
            for i, raw in enumerate(path):
                parent = ids.intern(parent, raw, min(block_size, prompt - i * block_size))
                path[i] = parent
            result = dict(timestamp=relative, input_length=prompt, output_length=output, hash_ids=path)
            dest.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        except (ValueError, KeyError, DecimalException) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    if origin is None:
        raise ValueError("no request rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="source path or - for stdin")
    parser.add_argument("output", nargs="?", default="-", help="destination path or - for stdout")
    parser.add_argument("--block-size", type=int, default=512, help="source tokens per hash (default 512)")
    parser.add_argument("--time-unit", choices=TIME_UNITS, default="ms", help="source timestamp unit (default ms)")
    args = parser.parse_args()
    temporary = None
    try:
        source = nullcontext(sys.stdin) if args.input == "-" else open(args.input, encoding="utf-8")
        with source as stream:
            if args.output == "-":
                export(stream, sys.stdout, args.block_size, args.time_unit)
            else:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                                 dir=Path(args.output).parent) as dest:
                    temporary = Path(dest.name)
                    export(stream, dest, args.block_size, args.time_unit)
                temporary.replace(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"replay.py: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
