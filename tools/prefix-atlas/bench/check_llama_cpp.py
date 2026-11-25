"""Compare serial LRU with prompt reuse in llama-cpp-python 0.3.16 on CPU."""
import argparse
from contextlib import closing
import csv
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import subprocess

from llama_cpp import Llama, LlamaRAMCache


BLOCK = 8
CAPACITY = 8
MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"


class CountedLlama(Llama):
    def eval(self, tokens):
        super().eval(tokens)
        self.evaluated += len(tokens)


def atlas_hits(prompts, cli):
    # Intern full token prefixes so the numeric IDs represent exact page identity.
    identities = {}
    rows = []
    cumulative = [0]
    for prompt in prompts:
        ids = []
        for end in range(BLOCK, len(prompt) + 1, BLOCK):
            prefix = tuple(prompt[:end])
            ids.append(identities.setdefault(prefix, len(identities)))
        rows.append(json.dumps(dict(timestamp=len(rows), input_length=len(prompt), hash_ids=ids)))
        output = subprocess.check_output(
            [cli, "-", "--block-size", str(BLOCK), "--capacities", str(CAPACITY)],
            input="\n".join(rows) + "\n", text=True,
        )
        row, = csv.DictReader(io.StringIO(output))
        assert int(row["page_references"]) == sum(len(p) // BLOCK for p in prompts[:len(rows)])
        cumulative.append(int(row["hits"]))
    return [b - a for a, b in zip(cumulative, cumulative[1:])]


def engine_hits(model, prompts, snapshots):
    reused_tokens = []
    with closing(CountedLlama(model_path=str(model), n_ctx=128, n_batch=64, n_threads=1,
                      n_gpu_layers=0, verbose=False)) as engine:
        if snapshots:
            engine.set_cache(LlamaRAMCache(capacity_bytes=64 * 1024**2))
        for prompt in prompts:
            engine.evaluated = 0
            completion = engine.create_completion(prompt=prompt, max_tokens=1, temperature=0)
            assert completion["usage"]["prompt_tokens"] == len(prompt)
            assert 0 < engine.evaluated <= len(prompt)
            reused_tokens.append(len(prompt) - engine.evaluated)
    return reused_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--cli", default="prefix-atlas")
    args = parser.parse_args()
    if version("llama-cpp-python") != "0.3.16":
        parser.error("this check requires llama-cpp-python 0.3.16")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("expected the pinned stories260K.gguf fixture")
    # Unique trailing tokens force a fresh final evaluation after every full page match.
    prompts = [[1] + [10] * 15 + [branch] * 16 + [40 + i]
               for i, branch in enumerate([20, 30, 20, 20])]
    atlas = atlas_hits(prompts, args.cli)
    active = engine_hits(args.model, prompts, False)
    saved = engine_hits(args.model, prompts, True)
    print("request,atlas_hit_pages,active_reused_tokens,snapshot_reused_tokens")
    for index, counts in enumerate(zip(atlas, active, saved), 1):
        print(index, *counts, sep=",")
    assert atlas == [0, 2, 4, 4]
    assert active == [0, 16, 16, 32]
    assert saved == [0, 16, 32, 32]


if __name__ == "__main__":
    main()
