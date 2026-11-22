"""Compare decoded ASCII deltas against two functions from a pinned vLLM checkout."""

import argparse
import ast
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from cutstream import Plan

REVISION = "0e3c8dbdae9d55149b0fba4e6c1014b4b08c9c73"
SOURCE = f"https://raw.githubusercontent.com/vllm-project/vllm/{REVISION}/vllm/v1/engine/detokenizer.py"
DIGEST = "20d14044b77e701532de1ef48b3f7a3917cd660618d59fc1135aa3899b7f0405"


def load_vllm(local_source):
    if local_source:
        source = local_source.read_bytes()
    else:
        with urlopen(SOURCE, timeout=20) as response:
            source = response.read()
    if hashlib.sha256(source).hexdigest() != DIGEST:
        raise ValueError("vLLM source differs from the pinned revision")
    tree = ast.parse(source)
    check = next(node for node in tree.body if node.name == "check_stop_strings"
                 ) if all(hasattr(node, "name") for node in tree.body) else next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "check_stop_strings"
    )
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "BaseIncrementalDetokenizer")
    delta = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name == "get_next_output_text")
    namespace = {"sys": sys}
    exec(compile(ast.Module([check, delta], type_ignores=[]), SOURCE, "exec"), namespace)
    return namespace["check_stop_strings"], namespace["get_next_output_text"]


class VllmStream:
    """Call the upstream check and emission functions on pre-decoded deltas."""

    def __init__(self, markers, check, delta, holdback):
        self.markers = markers
        self.check = check
        self.delta = delta
        self.stop_buffer_length = holdback
        self._last_output_text_offset = 0
        self.output_text = ""
        self.match = None

    def feed(self, chunk):
        self.output_text += chunk
        stop = self.check(self.output_text, len(chunk), self.markers, False)
        if stop is not None:
            self.match, truncate = stop
            self.output_text = self.output_text[:truncate]
        return self.delta(self, self.match is not None, True)


def run_stream(factory, chunks):
    stream = factory()
    visible = 0
    for chunk in chunks:
        visible += len(stream.feed(chunk))
    return visible


def observe(factory, chunks):
    stream = factory()
    data = []
    received = 0
    sent = 0
    lag = []
    first = None
    for idx, chunk in enumerate(chunks, 1):
        received += len(chunk)
        visible = stream.feed(chunk)
        sent += len(visible)
        if visible and first is None:
            first = idx
        data.append(visible)
        if idx < len(chunks):
            lag.append(received - sent)
    return data, first, statistics.mean(lag), stream.match


def measure(factory, chunks, loops):
    start = time.perf_counter_ns()
    total = sum(run_stream(factory, chunks) for _ in range(loops))
    elapsed = time.perf_counter_ns() - start
    if total == 0:
        raise AssertionError("benchmark produced no output")
    return elapsed / (loops * len(chunks))


def run_case(check, delta, count, width, repeats):
    # All stops have the same length and cannot tie on this ASCII fixture.
    markers = [f"<|stop_{i:03d}|>" for i in range(count)]
    sentence = "Tool returned 12 records. Next page starts after 0000496. <|stony|> "
    payload = (sentence * 128)[:8192]
    wire = payload + markers[-1]
    text_chunks = [wire[i:i + width] for i in range(0, len(wire), width)]
    byte_chunks = [chunk.encode("ascii") for chunk in text_chunks]
    plan = Plan([marker.encode("ascii") for marker in markers])
    holdback = max(map(len, markers)) - 1

    def vllm():
        return VllmStream(markers, check, delta, holdback)

    subjects = {
        "cutstream": (plan.stream, byte_chunks),
        "vllm": (vllm, text_chunks),
    }
    samples = {name: [] for name in subjects}
    row = {"markers": count, "chunk_width_bytes": width, "feeds": len(text_chunks)}
    for name, (factory, chunks) in subjects.items():
        emitted, first, mean_lag, match = observe(factory, chunks)
        joined = b"".join(emitted) if name == "cutstream" else "".join(emitted).encode("ascii")
        assert joined == payload.encode("ascii")
        assert match == (count - 1 if name == "cutstream" else markers[-1])
        row[name] = {"first_nonempty_feed": first, "mean_pending_bytes": round(mean_lag, 3)}
    loops = max(4, 7000 // len(text_chunks))
    for factory, chunks in subjects.values():
        run_stream(factory, chunks)
    order = list(subjects)
    for _ in range(repeats):
        for name in order:
            samples[name].append(measure(*subjects[name], loops))
        order.reverse()
    for name, values in samples.items():
        row[name]["ns_per_feed_trials"] = [round(v, 2) for v in values]
        row[name]["median_ns_per_feed"] = round(statistics.median(values), 2)
    row["median_ratio_vllm_over_cutstream"] = round(
        statistics.median(samples["vllm"]) / statistics.median(samples["cutstream"]), 3
    )
    return row


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--vllm-source", type=Path, help="use a local copy of the pinned detokenizer.py")
    args.add_argument("--repeats", type=int, default=7)
    options = args.parse_args()
    if options.repeats < 1:
        args.error("--repeats must be positive")
    check, delta = load_vllm(options.vllm_source)
    cases = [(n, width) for n in [1, 4, 32] for width in [4, 32, 1024]]
    random.Random(73051).shuffle(cases)
    rows = [run_case(check, delta, n, width, options.repeats) for n, width in cases]
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    print(json.dumps({
        "source": SOURCE, "sha256": DIGEST, "python": platform.python_version(),
        "platform": platform.platform(), "cpu": cpu, "synthetic_visible_bytes": 8192,
        "samples": sorted(rows, key=lambda row: (row["markers"], row["chunk_width_bytes"])),
    }, indent=2))


if __name__ == "__main__":
    main()
