# Prefix cache capacity curves

How much of an observed request stream can a cache of 8,192 pages reuse? The analyzer computes exact hit counts at any list of capacities from a tree of shared prompt histories. An edge can stand for billions of distinct consecutive pages. Neither mode allocates those pages or expands request paths into page references.

## Install the CLI

With Python 3.9+ and a C++17 compiler, install directly from the project repository:

```sh
python3 -m pip install 'git+https://github.com/pgg898200-ship-it/cachepin.git#subdirectory=tools/prefix-atlas'
prefix-atlas trace.jsonl --capacities 8192,65536
prefix-atlas trace.jsonl --block-size 16 --curve > curve.csv
prefix-atlas workload.tree --capacities 1024,8192
```

The source installation builds the analyzer once. Installing a matching prebuilt wheel needs only Python. The CLI accepts Mooncake JSONL by default and recognizes compact trees by their `.tree` extension. Use `--format tree -` to read a compact tree from stdin. The default block size for JSONL is 512 tokens. The exact curve CSV can be opened in the browser demo.

## Record a trace in an agent workload generator

A generator that already tracks exact page-aligned prompt forks can emit one edge per append. Here two requests share 1,024 immutable pages and add disjoint suffixes. A third request reuses the first tip.

```python
from prefix_atlas import TraceWriter

with TraceWriter("run.tree") as trace:
    shared = trace.extend(0, 1024)
    first = trace.extend(shared, 4)
    trace.request(first)
    trace.request(trace.extend(shared, 7))
    trace.request(first)
```

`extend(parent, pages)` returns a node number. Each call declares fresh page identities. Pass the same handle for actual reuse. If generated paths share only part of an append, construct the common edge first and attach separate suffixes to its tip. Prompt changes affecting tokenization, adapter, model weights or cache isolation require separate handles. Distinct children of root can represent isolated namespaces sharing one physical cache budget. Use separate traces when namespaces have independent budgets. For existing logs with uncertain sharing, use the JSONL importer.

Calls to one writer follow one producer's event order. The writer retains two buffered disk streams and counters, with constant Python memory across events. The caller holds handles it needs for future forks. During finalization it copies both streams into a completed tree, requiring up to about twice the new file size in temporary disk space. A successful context exit atomically replaces the destination; an exception leaves the previous file intact. Concurrent independent writers should use separate destinations; a later successful replacement wins otherwise. The writer validates handles and individual edge sizes. The analyzer also checks total path lengths and references for 64-bit overflow. Root 0 denotes an empty path.

In a synthetic append-only session of 3,000 turns, direct emission produced 33,793 bytes in 2.80 ms (median of five). Full prefix JSONL contained 4,501,500 page IDs and occupied 19,843,671 bytes. Emitting it and importing it took 1.845 s and produced the identical tree. This comparison assumes the generator already knows exact prefix identity and uses uncompressed JSONL. It measures local trace construction; it excludes inference, serving latency and analytics. Reproduce with `python3 bench/compare_producers.py --turns 3000 --trials 5`. [Raw measurements](bench/producer_results.json)

## Standalone build

```sh
c++ -std=c++17 -O3 curve.cpp -o curve
python3 bench/fetch_traces.py mooncake /tmp/mooncake.jsonl
python3 mooncake.py /tmp/mooncake.jsonl > /tmp/mooncake.tree
./curve --prefix-capacities 0,1024,8192,65536,1048576 < /tmp/mooncake.tree
python3 test_curve.py ./curve
```

On the FAST’25 Tool and Agent production trace the importer finds 386,041 complete 512-token page references across 23,608 requests. A capacity of 8,192 pages gives 183,552 hits (47.5%). The largest possible cache would give 226,058 hits (58.6%). For Harvard FreeInference May 17, pass `--block-size 16` to the importer. On its 5,684,383 page references, 8,192 pages give 4,471,892 hits (78.7%). See [trace provenance](trace_sources.md).

## Two access models

**Atomic prefix lookup.** With `--prefix-capacities C1,C2,...`, process one request to completion before the next. At arrival count the longest contiguous resident prefix. At completion touch its entire path from the deepest page back to the root and evict least recently touched pages to reach capacity. Each request leaves its shallowest pages most recent, maintaining prefix closure at every capacity between requests. For a request larger than the cache this policy retains up to its first C pages on completion. The CSV columns report capacity, total hit pages and total referenced pages. Capacities are distinct, sorted page counts.

To serialize the entire capacity response, use `./curve --prefix-curve < trace.tree`. Each row reports a breakpoint capacity, its hit count, the gain per additional page until the next breakpoint, and the total reference count. Between breakpoints, compute `hits + (wanted_capacity - capacity_pages) * hits_per_additional_page` for the last breakpoint at or below the requested capacity. The final slope is zero.

**Forward page stream.** With no argument, scan each path root to tip and immediately touch each page. Output starts with `cold N` for compulsory misses, followed by `capacity count` pairs. Each pair counts accesses that first become hits at that capacity. Sum counts with capacity at or below your budget to obtain hits. Scanning an over-capacity path repeatedly can have zero hits under this model.

Both models start empty. They treat every page as equally sized. Neither simulates overlapping requests, in-flight pins, decode allocations, sliding windows, scheduling, preemption, cross-engine eviction, or wall-clock latency. Real serving-engine hit rates require validation against that engine. For engine-level policy simulation see [PrefixSim](https://github.com/cacheMon/PrefixSim). For online page stack distance and KV working-set analysis see [KVSET](https://arxiv.org/abs/2609.27746).

## Upstream validation

We replayed both production fixtures through unmodified [PrefixSim](https://github.com/cacheMon/PrefixSim/tree/160d5f81724dd9ec28b10d6abb64615e6adbb332), using its native LRU policy with the default deepest-first replay order and pinning. Before replay both engines receive the same time-ordered complete pages. We summed missing-page lengths from the native per-request hole dump to recover exact hit counts. All 14 combinations agreed.

| Fixture | Tested capacities in pages | Agreed hits at those capacities |
| --- | --- | --- |
| Mooncake Tool and Agent | 256, 512, 1,024, 2,048, 4,096, 8,192, 16,384 | 136,442 · 138,521 · 139,727 · 142,859 · 155,765 · 183,552 · 207,503 |
| FreeInference May 17 | 8,192, 16,384, 32,768, 65,536, 131,072, 262,144, 524,288 | 4,471,892 · 4,919,957 · 5,002,217 · 5,015,976 · 5,017,697 · 5,023,431 · 5,029,227 |

Each tested capacity fits its largest request (246 or 7,575 complete pages respectively). PrefixSim skips an oversized request. Prefix Atlas processes it and retains up to its first C pages, so smaller capacities can produce different results. Other eviction policies and concurrent serving require their own comparisons. To repeat the checks after fetching and converting the fixtures:

```sh
python3 bench/verify_prefixsim.py /tmp/mooncake.jsonl /tmp/mooncake.tree --caps 256,512,1024,2048,4096,8192,16384
python3 bench/verify_prefixsim.py /tmp/harvard.jsonl /tmp/harvard.tree --block-size 16 --caps 8192,16384,32768,65536,131072,262144,524288
```

We also compared forward-stream page-hit counts with [KVSET’s unmodified token-ID replay](https://github.com/llc-kc/kv_cache_capacity_estimator/tree/5478962406c5dfbf4393c9d8656ca782968455e2). Each complete source block ID represents one token in a one-token page, with one byte per page and no warm-up. KVSET rehashes these IDs along each prefix. Both tools agreed on page references, unique pages and page hits at the same 14 capacities. At 8,192 pages, KVSET and the forward-stream histogram report 183,359 Mooncake hits, while the atomic model and PrefixSim each report 183,552. KVSET also reports per-request capacity requirements and can analyze online traffic. Reproduce the page-hit comparison with:

```sh
python3 bench/verify_kvset.py /tmp/mooncake.jsonl /tmp/mooncake.tree /path/to/kv_cache_capacity_estimator
python3 bench/verify_kvset.py /tmp/harvard.jsonl /tmp/harvard.tree /path/to/kv_cache_capacity_estimator --block-size 16 --caps 8192,16384,32768,65536,131072,262144,524288
```

## Input

The first line of the whitespace-delimited `.tree` format is `N T`. The next N rows define nodes numbered 1 through N, each as `parent edge_pages`, where 0 is the implicit empty root, the parent already exists, and the length is positive. The final T integers are the tips of requested root-to-node paths in arrival order. Tip 0 is an empty request. Pages on the same edge have a fixed order; pages from different edges have distinct identities. To fork midway along an edge, split it at the fork. Use 64-bit unsigned counts for pages and total references. The implementation accepts fewer than 2³¹−1 nodes or requests.

```text
3 3
0 4
1 2
1 1
2
3
1
```

Here the first request has six pages. The second shares the first four and appends a different last page. The third requests only the four shared pages.

The Mooncake importer accepts one JSON object per request with `hash_ids`, `input_length` and numeric `timestamp`. It stable-sorts by timestamp, retaining file order on ties. It uses just the first `floor(input_length / block_size)` hashes, so partially filled pages do not participate in cache reuse or the denominator. A source row may include its partial tail hash. The importer builds a radix tree from shared ID spans, splitting edges at branches and request endpoints. It reads every supplied hash and retains the distinct spans, their branch nodes, and request endpoints. On the May 17 fixture direct radix insertion reduced median conversion time from 1.84 s to 0.64 s and peak resident memory from 156 MiB to 35 MiB. [Importer measurements](bench/import_benchmark.md). The analysis can be run directly on producer-supplied trees to avoid that cost.

## How the calculation works

Give each request an epoch. Each page belongs to the most recent epoch that touched it. Later root-to-tip paths overwrite prefixes of earlier paths, so the pages still belonging to an old epoch form a single surviving suffix. The unbounded recency stack consists of these suffixes in descending epoch order. Their order within each epoch follows the direction used to touch pages.

For the next requested path, find the most recent old epoch passing through its first edge. The lowest common ancestor of the old and new tips gives the final edge of their shared run. Jump there and repeat. When an edge has no prior epoch, the rest of the path is cold. A subtree maximum over a DFS order answers the epoch lookup, binary lifting finds the shared ancestor, and a Fenwick tree counts surviving pages by epoch. Apply all epoch weight changes after computing the request's thresholds.

For atomic lookup, suppose a run shares L pages of an old epoch and K pages belong to newer epochs. The run starts at the shallow end of that epoch's surviving suffix. Its pages occupy ranks K+1 through K+L at arrival. A capacity C gets `min(L, max(0, C-K))` hits. Adding slope changes at K and K+L builds the total hit count function for all capacities together. For forward page touches, an old epoch is deepest-first in the stack. All pages in the covered run have the same rank when visited, equal to the number of pages in that epoch and all newer epochs. This gives one histogram update per run.

With N explicit nodes, T requests and R matched runs across the trace, indexing costs O(N log(N+1)). Each run costs O(log(N+1) + log(T+1) + log(R+1)), including histogram updates. Each request additionally costs O(log(N+1) + log(T+1)). Memory is O(N log(N+1) + T + R) in the worst case. Work depends on run fragmentation, and short unshared paths can favor an expanded analyzer. The brute-force tests compare sorted stack ranks for forward scans and explicitly size-limited prefix caches for atomic lookups at multiple capacities on 400 randomized weighted trees. They also exercise empty traces, partial blocks, stable arrival ordering and trillion-page edges.
