# Prefix Atlas

**How many KV-cache pages would have served 80% of the prefix blocks in this agent trace?** On the first published day of Harvard FreeInference, the answer under serial prefix LRU is **8,822 pages**. At 8,192 pages the same workload reuses 4,471,892 of 5,684,383 complete page references. Change the budget or open your own curve in the [interactive calculator](https://pgg898200-ship-it.github.io/cachepin/prefix-atlas/).

Prefix Atlas computes an exact capacity curve from a tree of shared prompt histories. Long shared paths cost one operation per overlapping run. Developers can emit a fork as a parent, a number of new pages and a request tip, or import an existing Mooncake-format JSONL file. Run the tool offline with a C++17 compiler. The calculator works locally in a browser.

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

The importer accepts numeric timestamps, input lengths and integer hash lists, stable-sorts arrivals, and excludes partial blocks. A full curve records the exact hit count and slope at each breakpoint. It can answer new budgets without reprocessing the trace. A producer that already knows its fork and append history can write a tree directly. [Input format and algorithm](README-curve.md)

## What was measured

| Workload and operation | Prefix Atlas | Native page-by-page control | Ratio |
| --- | ---: | ---: | ---: |
| FreeInference, 7 serial prefix-cache budgets from a prepared tree | 3.59 ms | 120.53 ms | 33.55× |
| Mooncake tool/agent, same operation | 19.65 ms | 21.07 ms | 1.07× |
| FreeInference, raw JSONL through seven-budget answer | 615.70 ms | 657.02 ms | 1.07× |
| Mooncake tool/agent, raw JSONL through seven-budget answer | 273.67 ms | 255.40 ms | 0.93× |

The two implementations emitted identical hit counts. Trials used the same machine, input and compiler flags. The native control calculates all requested budgets in one pass. On a synthetic singleton cycle it finished faster. Seven additional capacities on each real fixture matched the upstream PrefixSim native LRU implementation where every request fit in the cache. [Kernel measurements](bench/curve_benchmark.md) · [Import and end-to-end timings](bench/import_benchmark.md) · [Reproduce PrefixSim parity](bench/verify_prefixsim.py)

This is a sizing model for serial, fully associative LRU. At arrival a request can reuse its longest contiguous resident prefix. On completion, the cache makes shallower blocks most recent. Pages, not requests, are the hit-rate denominator. Concurrency, in-flight pins, decode memory, sliding windows, routing and preemption require engine-specific measurements. To model separate tenants or model namespaces, analyze each isolated cache separately. PrefixSim supports additional cache policies. [Model details](README-curve.md) · [PrefixSim](https://github.com/cacheMon/PrefixSim)

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

Report a mismatch or an independent reproduction with the compiler, trace revision, block size, access model, capacities and expected output in [CachePin issues](https://github.com/pgg898200-ship-it/cachepin/issues). Code follows the parent CachePin project's licenses. The included FreeInference-derived example and its measurements include [source attribution](trace_sources.md).
