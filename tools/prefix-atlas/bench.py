#!/usr/bin/env python3
"""Benchmark prefixpack and native Zstandard readers. Requires Python 3, C++17 and libzstd."""
import argparse
from array import array
import csv
import ctypes
from ctypes.util import find_library
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent


class Zstd:
    def __init__(self, level):
        self.level = level
        self.lib = ctypes.CDLL(find_library("zstd") or "libzstd.so")
        size = ctypes.c_size_t
        self.lib.ZSTD_compressBound.argtypes = [size]
        self.lib.ZSTD_compressBound.restype = size
        self.lib.ZSTD_compress.argtypes = [
            ctypes.c_void_p, size, ctypes.c_void_p, size, ctypes.c_int,
        ]
        self.lib.ZSTD_compress.restype = size
        self.lib.ZSTD_isError.argtypes = [size]
        self.lib.ZSTD_isError.restype = ctypes.c_uint
        self.lib.ZSTD_getErrorName.argtypes = [size]
        self.lib.ZSTD_getErrorName.restype = ctypes.c_char_p
        self.lib.ZSTD_versionString.restype = ctypes.c_char_p

    def compress(self, source):
        cap = self.lib.ZSTD_compressBound(len(source))
        dest = ctypes.create_string_buffer(cap)
        length = self.lib.ZSTD_compress(dest, cap, source, len(source), self.level)
        if self.lib.ZSTD_isError(length):
            raise RuntimeError(self.lib.ZSTD_getErrorName(length).decode())
        return dest.raw[:length]


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def put_varint(buf, value):
    while value >= 128:
        buf.append(128 | (value & 127))
        value >>= 7
    buf.append(value)


def put_runs(buf, ids):
    # A segment's header holds its length and one bit for literal vs ascending run.
    # Two adjacent numbers alone form a literal block, reducing short-run markers.
    i, literal = 0, 0
    while i < len(ids):
        end = i + 1
        while end < len(ids) and ids[end] == ids[end - 1] + 1:
            end += 1
        if end - i >= 3:
            if literal < i:
                put_varint(buf, (i - literal) << 1)
                for n in ids[literal:i]:
                    put_varint(buf, n)
            put_varint(buf, (end - i) << 1 | 1)
            put_varint(buf, ids[i])
            literal = end
        i = end
    if literal < len(ids):
        put_varint(buf, (len(ids) - literal) << 1)
        for n in ids[literal:]:
            put_varint(buf, n)


def delta_frame(rows, runs):
    # Keep the longest common hash_ids prefix with the preceding row in this frame.
    # Store the rest as unsigned varints, plus compact UTF-8 for the other fields.
    result = bytearray()
    prev = []
    for row in rows:
        ids = row["hash_ids"]
        i = 0
        while i < min(len(prev), len(ids)) and prev[i] == ids[i]:
            i += 1
        header = compact({k: v for k, v in row.items() if k != "hash_ids"})[:-1]
        put_varint(result, len(header))
        result.extend(header)
        put_varint(result, i)
        put_varint(result, len(ids) - i)
        if runs:
            put_runs(result, ids[i:])
        else:
            for value in ids[i:]:
                put_varint(result, value)
        prev = ids
    return bytes(result)


def pack_frames(archive, rows, lines, nper, zstd, mode):
    count = (len(rows) + nper - 1) // nper
    with archive.open("wb") as out:
        out.write(struct.pack("<3Q", 1, len(rows), nper))
        out.seek(8 * (count + 1), 1)
        if mode == "sorted" or mode == "sorted_runs":
            order = sorted(range(len(rows)), key=lambda i: rows[i]["hash_ids"])
            permutation = array("I", [0]) * len(rows)
            if permutation.itemsize != 4:
                raise RuntimeError("expected 32-bit unsigned int")
            for physical_id, original_id in enumerate(order):
                permutation[original_id] = physical_id
            if sys.byteorder != "little":
                permutation.byteswap()
            out.write(permutation)
            lines = [lines[i] for i in order]
            rows = [rows[i] for i in order]
        offsets = []
        for i in range(0, len(rows), nper):
            offsets.append(out.tell())
            if mode in {"delta", "runs", "sorted_runs"}:
                data = delta_frame(rows[i:i+nper], runs=mode != "delta")
            else:
                data = b"".join(lines[i:i+nper])
            out.write(zstd.compress(data))
        offsets.append(out.tell())
        out.seek(24)
        out.write(struct.pack(f"<{len(offsets)}Q", *offsets))


def command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def check_lines(lines, expected, label):
    n = 0
    for line in lines:
        record = json.loads(line)
        if isinstance(record, dict) and isinstance(record.get("hash_ids"), list):
            record["hash_ids"] = array("Q", record["hash_ids"])
        if n >= len(expected) or record != expected[n]:
            raise ValueError(f"{label}: incorrect record at output index {n}")
        n += 1
    if n != len(expected):
        raise ValueError(f"{label}: expected {len(expected)} rows, received {n}")


def parse_ints(raw):
    vals = list(dict.fromkeys(int(item) for item in raw.split(",")))
    if not vals or min(vals) < 1:
        raise ValueError("expected comma-separated positive integers")
    return vals


def run(args, work):
    reader = work / "bench-reader"
    compiler = os.environ.get("CXX", "c++")
    command([compiler, "-O3", "-DNDEBUG", "-std=c++17", HERE / "bench_reader.cc",
             "-lzstd", "-o", reader])
    id_map = None
    if args.permute_ids is not None:
        # Preserve equality, path lengths and ordering while removing sequential IDs.
        vocabulary = set()
        with args.input.open("rb") as inp:
            for line in inp:
                if line.strip():
                    vocabulary.update(json.loads(line)["hash_ids"])
        ordered = sorted(vocabulary)
        permuted = ordered.copy()
        random.Random(args.permute_ids).shuffle(permuted)
        id_map = dict(zip(ordered, permuted))
        del vocabulary, ordered, permuted
    rows, lines = [], []
    source_hash, compact_hash = hashlib.sha256(), hashlib.sha256()
    compact_size = 0
    source = work / "input.jsonl"
    with args.input.open("rb") as inp, source.open("wb") as out:
        for line in inp:
            source_hash.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or not isinstance(row.get("hash_ids"), list)
                    or any(type(x) is not int or not 0 <= x < 2**64 for x in row["hash_ids"])):
                raise ValueError(f"expected unsigned 64-bit hash_ids at row {len(rows)}")
            if id_map is not None:
                row["hash_ids"] = [id_map[n] for n in row["hash_ids"]]
            encoded = compact(row) + b"\n"
            out.write(encoded)
            compact_size += len(encoded)
            compact_hash.update(encoded)
            lines.append(encoded)
            # Millions of boxed Python integers would dwarf the trace on deep sessions.
            row["hash_ids"] = array("Q", row["hash_ids"])
            rows.append(row)
    del id_map
    if not rows:
        raise ValueError("empty trace")
    zstd = Zstd(args.level)
    sizes = {}
    queries = {}
    scans = {}
    nrows = len(rows)
    rng = random.Random(args.seed)
    batches = {f"get_{n}": rng.choices(range(nrows), k=n) for n in args.batch_sizes}

    def add_reader(label, mode, file):
        sizes[label] = file.stat().st_size
        queries[label] = {key: [reader, mode, "get", file, *map(str, ids)]
                          for key, ids in batches.items()}
        scans[label] = [reader, mode, "scan", file]

    path = work / "whole.zst"
    path.write_bytes(zstd.compress(source.read_bytes()))
    add_reader("whole_zstd", "whole", path)
    for nper in args.frames:
        for mode in ["frame", "delta", "sorted", "runs", "sorted_runs"]:
            label = f"{mode}_{nper}_zstd"
            path = work / label
            pack_frames(path, rows, lines, nper, zstd, mode)
            add_reader(label, mode, path)
    product_hash = None
    if not args.baselines_only:
        archive = work / "archive.ppk"
        product = work / "prefixpack"
        shutil.copy2(args.binary, product)
        product_hash = hashlib.sha256(product.read_bytes()).hexdigest()
        command([product, "pack", source, archive], stdout=subprocess.DEVNULL)
        command([product, "verify", archive], stdout=subprocess.DEVNULL)
        label = "prefixpack"
        sizes[label] = archive.stat().st_size
        queries[label] = {key: [product, "get", archive, *map(str, ids)]
                          for key, ids in batches.items()}
        scans[label] = [product, "unpack", archive]

    del lines
    observations = {label: {key: [] for key in [*batches, "scan"]} for label in sizes}
    # The warm-up validates the outputs and primes each reader and its input. Measured
    # trials cover subprocess creation, index reads, decompression and JSONL delivery.
    for trial in range(args.repeats + 1):
        jobs = [(name, op) for name in sizes for op in observations[name]]
        rng.shuffle(jobs)
        for name, op in jobs:
            target = work / "unpacked.jsonl"
            if op == "scan":
                target.unlink(missing_ok=True)
                argv = [*scans[name], target]
                stdout = subprocess.DEVNULL
            else:
                argv = queries[name][op]
                stdout = subprocess.PIPE
            start = time.perf_counter_ns()
            result = command(argv, stdout=stdout)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if trial:
                observations[name][op].append(elapsed)
            elif op == "scan":
                with target.open("rb") as output:
                    check_lines(output, rows, name)
            else:
                check_lines(result.stdout.splitlines(), [rows[i] for i in batches[op]], name)

    results = []
    for name, size in sizes.items():
        entry = {"codec": name, "bytes": size, "ratio_to_compact_input": size / compact_size}
        for op, times in observations[name].items():
            entry[f"{op}_median_ms"] = round(statistics.median(times), 3)
        results.append(entry)
    return {
        "input": args.input.name, "source_sha256": source_hash.hexdigest(),
        "product_binary_sha256": product_hash, "id_permutation_seed": args.permute_ids,
        "compact_sha256": compact_hash.hexdigest(), "rows": nrows,
        "raw_bytes": args.input.stat().st_size, "compact_bytes": compact_size,
        "sample_seed": args.seed, "random_batches": {
            key: {"count": len(ids), "unique": len(set(ids)), "first_id": ids[0],
                  "ids_sha256": hashlib.sha256(compact(ids)).hexdigest()}
            for key, ids in batches.items()
        },
        "frames_in_rows": args.frames, "zstd_level": args.level,
        "zstd_version": zstd.lib.ZSTD_versionString().decode(), "repeats": args.repeats,
        "architecture": platform.machine(), "kernel": platform.release(),
        "cpu": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                     if l.startswith("model name")), "unknown") if Path("/proc/cpuinfo").exists() else platform.processor(),
        "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
        "method": ("Warm OS page cache. Native subprocess per operation. Random requests sampled "
                   "with replacement in fixed order; one batch per process. Median wall time includes "
                   "launch, file open, decompression and JSONL output. Scan writes a local file. "
                   "Warm-up outputs compared record-by-record with input. Includes all index bytes. "
                   "Each frame starts a fresh compression context. Sorted frames include a 4-byte-per-row inverse "
                   "permutation. No fsync or network."),
        "results": results, "samples_ms": observations,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--binary", type=Path, default=HERE / "prefixpack")
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--frames", type=parse_ints, default=[16, 64, 512])
    p.add_argument("--batch-sizes", type=parse_ints, default=[1, 32, 256])
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--seed", type=int, default=8128)
    p.add_argument("--permute-ids", type=int, metavar="SEED", help=(
        "measure a diagnostic trace with the same prefixes under a bijective shuffle of "
        "the input's observed hash IDs; disables dependence on consecutive numbering"
    ))
    p.add_argument("--format", choices=["json", "tsv"], default="json")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="prefixpack-bench-") as directory:
        results = run(args, Path(directory))
    if args.format == "tsv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(results["results"][0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(results["results"])
    else:
        json.dump(results, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
