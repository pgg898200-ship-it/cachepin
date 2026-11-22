"""Replay identical ASCII streams through the native Cutstream and TurboSpark matchers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
RELEASE = "5ede725e9a1e375fb66d135bc0ab6117bf2a5259"
SOURCE = f"https://raw.githubusercontent.com/whit3rabbit/turbospark/{RELEASE}/crates/tokenizer/src/stop_matcher.rs"
SHA256 = "e70844fdf4a52854f921061271b8b0687a53ee8c25b76ab3306bc567001ced98"
STOPS = ["</tool>", "[DONE]", "<|endofturn|>", "END_REPLY"]
LITERAL = "The agent reads an event, checks the cache and reports a delta. "


def write_case(path, stops, width):
    """32 round-robin streams. Half stop once, the others terminate at EOF."""
    path.mkdir()
    (path / "stops.txt").write_text("\n".join(stops) + "\n", encoding="ascii")
    chunk_lists = []
    for stream in range(32):
        # Include false alarms and exercise release of a previously retained prefix.
        stem = f"Stream {stream} reports </too? and [DO? before continuing. "
        desired = 1024 + stream * 11
        body = (stem + LITERAL * (desired // len(LITERAL) + 2))[:desired]
        stop = stops[(stream // 2) % len(stops)]
        if stream % 2 == 0:
            # One candidate match crossing the next chunk boundary.
            pad = (width - 1 - len(body)) % width
            text = body + "." * pad + stop
        else:
            text = body + stop[:2]
        chunk_lists.append([text[i:i + width] for i in range(0, len(text), width)])
    events = [(i, chunks[j]) for j in range(max(map(len, chunk_lists)))
              for i, chunks in enumerate(chunk_lists) if j < len(chunks)]
    (path / "events.tsv").write_text("".join(f"{i}\t{s}\n" for i, s in events), encoding="ascii")
    return events


def oracle(events, stops):
    # Outside the timed loop. The record boundary and stopped flag enter the
    # digest after every feed, including empty output, and after every flush.
    count = 0
    digest = 14695981039346656037
    def consume(text, done):
        nonlocal count, digest
        count += len(text)
        for x in [*text.encode("ascii"), 256 + int(done)]:
            digest = ((digest ^ x) * 1099511628211) & ((1 << 64) - 1)
    pending = [""] * (1 + max(i for i, _ in events))
    stopped = [False] * len(pending)
    for i, chunk in events:
        assert not stopped[i]
        data = pending[i] + chunk
        hits = [(data.index(s), s) for s in stops if s in data]
        assert len(hits) <= 1, "fixture admits competing stop semantics"
        if hits:
            start, stop = hits[0]
            assert start + len(stop) == len(data)
            consume(data[:start], True)
            stopped[i] = True
            pending[i] = ""
        else:
            length = max((n for s in stops for n in range(1, len(s))
                          if data.endswith(s[:n])), default=0)
            where = len(data) - length
            consume(data[:where], False)
            pending[i] = data[where:]
    for held, done in zip(pending, stopped):
        consume(held, done)
    return digest, count


def sample(exe, path, budget_ns, expected):
    output = subprocess.check_output([str(exe), str(path), str(budget_ns)], text=True)
    digest, count, elapsed_ns, passes = map(int, output.split())
    if (digest, count) != expected:
        raise RuntimeError(f"{exe.name} disagrees with oracle in {path.name}: {(digest, count)} != {expected}")
    return {"elapsed_ns": elapsed_ns, "passes": passes, "ns_per_pass": elapsed_ns / passes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, help="offline copy of the pinned stop_matcher.rs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-ms", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_ms < 1:
        parser.error("samples and sample-ms must be positive")
    cxx, rustc = os.environ.get("CXX", "c++"), os.environ.get("RUSTC", "rustc")
    for tool in (cxx, rustc):
        if not shutil.which(tool):
            parser.exit(1, f"Required compiler missing: {tool}\n")
    if args.source_file:
        source = args.source_file.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            source = response.read()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise SystemExit("Pinned TurboSpark source checksum mismatch")
    info = {"upstream": SOURCE, "source_sha256": SHA256, "platform": platform.platform(),
            "runner": os.environ.get("RUNNER_NAME"), "revision": os.environ.get("GITHUB_SHA"),
            "cxx": subprocess.check_output([cxx, "--version"], text=True).splitlines()[0],
            "rustc": subprocess.check_output([rustc, "--version"], text=True).strip()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        info["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                            if line.startswith("model name")), platform.machine())
    rows = []
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "stop_matcher.rs").write_bytes(source)
        shutil.copyfile(HERE / "native.rs", root / "native.rs")
        bins = {name: root / name for name in ("cutstream", "turbospark")}
        subprocess.run([cxx, "-std=c++17", "-O3", "-march=native", "-DNDEBUG", "-I",
                        str(HERE.parents[1] / "include"), str(HERE / "native.cpp"),
                        "-o", str(bins["cutstream"])], check=True)
        subprocess.run([rustc, "--edition=2021", "-C", "opt-level=3", "-C", "target-cpu=native",
                        str(root / "native.rs"), "-o", str(bins["turbospark"])], check=True)
        for count in (1, 4):
            for width in (4, 32):
                case = root / f"stops{count}-chunk{width}"
                events = write_case(case, STOPS[:count], width)
                expected = oracle(events, STOPS[:count])
                samples = {k: [] for k in bins}
                for exe in bins.values():
                    sample(exe, case, 0, expected)
                for n in range(args.samples):
                    order = list(bins) if n % 2 == 0 else list(reversed(bins))
                    for name in order:
                        samples[name].append(sample(bins[name], case, args.sample_ms * 1_000_000, expected))
                median = {k: statistics.median(s["ns_per_pass"] for s in samples[k]) for k in bins}
                size = sum(len(chunk) for _, chunk in events)
                rows.append({"stops": count, "chunk_bytes": width, "input_bytes": size,
                             "chunks": len(events), "output_bytes": expected[1],
                             "frame_digest": format(expected[0], "016x"),
                             "median_input_MiB_per_s": {k: size * 1e9 / (ns * (1<<20)) for k, ns in median.items()},
                             "median_time_ratio_turbospark_over_cutstream": median["turbospark"] / median["cutstream"],
                             "samples": samples})
    result = json.dumps({"environment": info, "results": rows}, indent=2) + "\n"
    print(result, end="")
    if args.output:
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
