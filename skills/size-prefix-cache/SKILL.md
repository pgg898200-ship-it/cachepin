---
name: size-prefix-cache
description: Find the smallest KV-cache page budget that meets a requested prefix hit percentage on a Mooncake-format trace, Prefix Atlas tree, or exported capacity curve. Report unattainable targets and cold-page limits.
---

# Size a prefix cache

Use the local trace to answer a concrete question such as “How many 16-token pages would have served 80% of our observed input pages?” Keep separate tenants and model namespaces in separate analyses. The calculation assumes serial requests and fully associative prefix LRU. The denominator is complete page references.

For an exported Prefix Atlas curve, run the bundled Python 3 script directly. Resolve `scripts/target.py` relative to this skill. The target is a percentage.

```sh
python scripts/target.py 80 curve.csv
```

For Mooncake JSONL or a compact `.tree` file, first run Prefix Atlas. With an active Python 3.9+ environment on glibc 2.17+ Linux x86-64, install the released wheel.

```sh
python -m pip install --no-index --find-links https://pgg898200-ship-it.github.io/cachepin/prefix-atlas/dist/ prefix-atlas==0.1.0
```

On other platforms build from source with Python 3.9+ and a C++17 compiler.

```sh
python -m pip install 'git+https://github.com/pgg898200-ship-it/cachepin.git@dfb0d5f90dadae3a6ca3a910923b9178724b9505#subdirectory=tools/prefix-atlas'
```

Read the trace producer's tokens-per-hash setting. Set `--block-size` to that value for JSONL. A compact tree already counts pages.

```sh
prefix-atlas requests.jsonl --block-size 16 --curve > curve.csv
python scripts/target.py 80 curve.csv
```

Report the minimum capacity in pages, achieved hits and total page references. A null capacity means the target exceeds the maximum observed reuse. Report `maximum_hits / page_references` in that case. To verify a positive minimum, pass the preceding capacity and the minimum to `prefix-atlas --capacities` on the same input. Check for concurrent requests, pinned pages, routing and decode memory before interpreting the result as a serving configuration.

[Trace format and model assumptions](https://github.com/pgg898200-ship-it/cachepin/blob/main/tools/prefix-atlas/README-curve.md)
