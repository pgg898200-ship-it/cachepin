# Prefix Atlas

**How many KV-cache pages would have served 80% of the prefix blocks in this agent trace?** On the first published day of Harvard FreeInference, the answer under serial prefix LRU is **8,822 pages**. At 8,192 pages the same workload reuses 4,471,892 of 5,684,383 complete page references. Change the budget or open your own curve in the [interactive calculator](https://pgg898200-ship-it.github.io/cachepin/prefix-atlas/).

Prefix Atlas computes an exact capacity curve from a tree of shared prompt histories. Long shared paths cost one operation per overlapping run. Developers can emit a fork as a parent, a number of new pages and a request tip, or import an existing Mooncake-format JSONL file. Run the tool offline with a C++17 compiler. The calculator works locally in a browser.

[Recompute the 80% cache threshold in JupyterLab](https://jupyter.org/try-jupyter/lab/index.html?fromURL=https%3A%2F%2Fraw.githubusercontent.com%2Fpgg898200-ship-it%2Fcachepin%2Fmain%2Ftools%2Fprefix-atlas%2Fexamples%2Fsize-prefix-cache.ipynb). Run all cells in your browser with the Python (Pyodide) kernel. [Read the executed notebook](https://nbviewer.org/github/pgg898200-ship-it/cachepin/blob/main/tools/prefix-atlas/examples/size-prefix-cache.ipynb).

## Install the CLI

On Linux ARM64 (AArch64) or x86-64, install the prebuilt CLI with Python 3.9+ and pip. The ARM64 analyzer is statically linked. The x86-64 wheel requires glibc 2.17 or newer.

```sh
python -m pip install --no-index --only-binary=:all: --find-links https://pgg898200-ship-it.github.io/cachepin/prefix-atlas/dist/ prefix-atlas==0.1.0
```

With Python 3.9+ and a Unix-like C++17 toolchain, pip can build the source archive without a Git checkout. Pip downloads setuptools for the isolated build. The installed CLI has no Python runtime dependencies.

```sh
python -m pip install https://pgg898200-ship-it.github.io/cachepin/prefix-atlas/dist/prefix_atlas-0.1.0.tar.gz
prefix-atlas requests.jsonl --block-size 16 --capacities 8192,65536
prefix-atlas requests.jsonl --block-size 16 --curve > capacity.csv
```

The [Hugging Face Space](https://huggingface.co/spaces/pgg-engineering/prefix-atlas) opens the two measured workloads and reads exported curves entirely in your browser.

Ask a coding agent for the smallest page budget that meets a hit target. Install the [size-prefix-cache skill](https://github.com/pgg898200-ship-it/cachepin/tree/main/skills/size-prefix-cache) in your project:

```sh
npx skills add pgg898200-ship-it/cachepin --skill size-prefix-cache
```

## Try a real workload

From this directory:

```sh
make curve
./curve --prefix-capacities 8192,8821,8822,65536 < examples/freeinference-may17.tree
./curve --prefix-curve < examples/freeinference-may17.tree > capacity.csv
```

Open the resulting CSV in the calculator. The bundled 16,604-byte tree contains the branching structure and request order of 1,375 top-level FreeInference requests on May 17, 2026. The tree retains all 5,684,383 full-block references needed for this calculation. It omits timestamps, labels and original prompts. [Dataset, attribution and transformations](trace_sources.md)

For your own requests, set the source block size in tokens:

```sh
python3 mooncake.py --block-size 16 requests.jsonl > requests.tree
./curve --prefix-capacities 1024,4096,8192,65536 < requests.tree
```

The importer accepts numeric timestamps, input lengths and integer hash lists, stable-sorts arrivals, and excludes partial blocks. A full curve records the exact hit count and slope at each breakpoint. It can answer new budgets without reprocessing the trace. Workload generators can use `from prefix_atlas import TraceWriter` to record each known fork or append once, as it happens. On a synthetic 3,000-turn append-only session the writer emitted a 33,793-byte tree in 2.80 ms. Emitting cumulative JSONL and importing it took 1.845 s and produced the identical tree. This requires exact shared-prefix identities from the producer. [Producer benchmark](bench/producer_results.json) [Input format and algorithm](README-curve.md)

## What was measured

On Alibaba’s two-hour Qwen Bailian API trace, a serial prefix cache needs **92,386 sixteen-token pages** to serve half of the 9,794,201 complete page references. Counting repeated block IDs alone overstates maximum reuse by 7.28 percentage points because a block can occur after different prefixes. [Qwen workload, exact counts and reproduction](trace_sources.md#qwen-bailian-api)

On Mooncake’s FAST’25 **conversation** trace, an 8,192-page cache served 54,387 of 276,491 complete page references (19.67%). Even an unlimited cache could serve only 38.19%, so an 80% target is unreachable on that workload. [Conversation trace, counts and reproduction commands](trace_sources.md#mooncake-conversation)

| Workload and operation | Prefix Atlas | Native page-by-page control | Ratio |
| --- | ---: | ---: | ---: |
| FreeInference, 7 serial prefix-cache budgets from a prepared tree | 3.59 ms | 120.53 ms | 33.55× |
| Mooncake tool/agent, same operation | 19.65 ms | 21.07 ms | 1.07× |
| FreeInference, raw JSONL through seven-budget answer | 615.70 ms | 657.02 ms | 1.07× |
| Mooncake tool/agent, raw JSONL through seven-budget answer | 273.67 ms | 255.40 ms | 0.93× |

A separate comparison used the unmodified upstream PrefixSim CLI. From normalized JSONL through **seven capacities**, Prefix Atlas took **644 ms versus 10.47 s** on FreeInference and **203 ms versus 740 ms** on Mooncake with the upstream jobs run sequentially. Both tools prepared their inputs once. With seven concurrent upstream processes, the fastest observed upstream route by median was 267 ms on Mooncake and an estimated 5.88 s on FreeInference. The FreeInference estimate adds separately measured conversion and execution medians. For **one** Mooncake capacity, PrefixSim's direct JSONL reader was faster at **131 ms versus 202 ms**. [Upstream comparison, commands and raw timings](bench/prefixsim_benchmark.md)

The two native implementations emitted identical hit counts. Trials used the same machine, input and compiler flags. The native control calculates all requested budgets in one pass. On a synthetic singleton cycle it finished faster. Seven additional capacities on each real fixture matched the upstream PrefixSim native LRU implementation where every request fit in the cache. [Kernel measurements](bench/curve_benchmark.md) · [Import and end-to-end timings](bench/import_benchmark.md) · [Reproduce PrefixSim parity](bench/verify_prefixsim.py)

This is a sizing model for serial, fully associative LRU. At arrival a request can reuse its longest contiguous resident prefix. On completion, the cache makes shallower blocks most recent. The denominator is the count of referenced full pages. Concurrency, in-flight pins, decode memory, sliding windows, routing and preemption require engine-specific measurements. To model separate tenants or model namespaces, analyze each isolated cache separately. PrefixSim supports additional cache policies. [Model details](README-curve.md) · [PrefixSim](https://github.com/cacheMon/PrefixSim)

On a four-request CPU inference check, returning to an earlier branch exposed a difference between the model and llama.cpp’s active context. Prefix Atlas predicted four reusable pages for that request. The llama-cpp-python completion API reused two with its active context alone and four with its RAM snapshot cache enabled. [Request-level results and reproduction](trace_sources.md#llamacpp-prompt-reuse)

## Extract individual requests for downstream replay

The optional indexed archive preserves original bytes and line endings. It stores decimal ID runs and compresses bounded groups of rows with Zstandard. It supports duplicate and out-of-order row selections. Install libzstd development headers to build the archive command.

```sh
make prefixpack
./prefixpack pack requests.jsonl requests.ppk
./prefixpack verify requests.ppk
./prefixpack get requests.ppk 0 4 9 > selected.jsonl
./prefixpack unpack requests.ppk restored.jsonl
python3 replay.py --block-size 16 --time-unit ms selected.jsonl replay.jsonl
```

The adapter exports time-normalized Mooncake data for GuideLLM. An offline integration check exercised the installed GuideLLM 0.8.1 reader and prompt generator. The archive benchmark includes indexed Zstandard, previous-row prefix deltas, sorted frames and a bijective ID relabeling. Dense consecutive IDs compress well; a specialized delta baseline produces a smaller archive on both published fixtures. [GuideLLM workflow](trace_sources.md#replay-selected-requests-in-guidellm) · [Archive measurements](bench/archive_benchmark.md)

## Verify

```sh
make test
python3 bench/fetch_traces.py mooncake /tmp/toolagent.jsonl
python3 mooncake.py --block-size 512 /tmp/toolagent.jsonl > /tmp/toolagent.tree
python3 bench/compare_curves.py --help
```

Report a mismatch or an independent reproduction with the compiler, trace revision, block size, access model, capacities and expected output in [CachePin issues](https://github.com/pgg898200-ship-it/cachepin/issues). The source is published under CachePin’s MIT license. The included FreeInference-derived example and its measurements include [source attribution](trace_sources.md).
