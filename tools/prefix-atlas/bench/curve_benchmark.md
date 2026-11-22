# Exact all-capacity LRU benchmark

Both C++ programs read the same compact tree and request order. They write the same count of cold misses and the same complete histogram of minimum capacities that turn reuses into hits. The independent control enumerates every page access. It tracks last positions in a dense array and computes stack distance with a bit-packed bitmap plus a word-level Fenwick tree. Both programs are single threaded.

| Workload | Requests | Page accesses | Weighted curve, ms | Expanded control, ms | Expanded / weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harvard FreeInference, May 17 | 1,375 | 5,684,383 | 6.85 | 235.24 | 34.34× |
| Mooncake FAST'25 tool/agent | 23,608 | 386,041 | 34.93 | 28.60 | 0.82× |
| Shared 1,024-page prefix, 128 four-page suffixes | 20,000 | 20,560,000 | 11.60 | 1,307.37 | 112.69× |
| Cycling through 4,096 singleton prefixes | 200,000 | 200,000 | 56.02 | 31.82 | 0.57× |

Each cell is the median of nine runs after one warm-up. Trials alternate process order. Times include process launch, reading and parsing the warm tree file, analysis, and emission of every histogram bin. They exclude converting source JSON to the tree. Both binaries use the same compiler options on the same AMD EPYC 9V74 host. Scheduling on a shared host adds timing variation. [Raw samples, tree and histogram hashes](curve_results.json)

The end-to-end advantage depends on the lengths of the compressed paths. The Harvard input averages 4,134 page accesses per request. The Mooncake input averages 16.4 and favors the expanded implementation. The simulator counts demand references under page-granular, fully associative LRU. Timing this calculation measures offline analysis speed.

## Atomic prefix lookup

When a server looks up a prompt before changing recency, the hit count can differ from forward page-by-page LRU. This mode counts the resident prefix at request arrival, then releases the pages in descending depth order so the shallowest page becomes the most recent. The expanded control independently reads every previous page rank at request start, then updates all page positions in reverse order. The benchmark requests 0, 1, 64, 1,024, 8,192, 65,536 and 1,048,576 pages of capacity in a single call.

| Workload | Weighted curve, ms | Expanded control, ms | Expanded / weighted |
| --- | ---: | ---: | ---: |
| Harvard FreeInference, May 17 | 3.59 | 120.53 | 33.55× |
| Mooncake FAST'25 tool/agent | 19.65 | 21.07 | 1.07× |
| Shared 1,024-page prefix | 10.19 | 544.39 | 53.41× |
| Shallow singleton cycle | 30.97 | 24.97 | 0.81× |

These runs agreed exactly across all seven capacities on each workload. For example, 8,192 pages yield 4,471,892 hits on Harvard, up from 973,381 hits with 1,024 pages, out of 5,684,383 complete-page references. The two benchmark modes ran at different times on the shared host. Ratios compare implementations within a given mode. [Atomic prefix raw samples](prefix_curve_results.json)

## Reproduce

From the project directory, Python 3, a C++17 compiler for x86-64, and the zstd command are sufficient. The public workloads are version-pinned and checked on download. [Trace sources](../trace_sources.md)

```sh
python3 bench/fetch_traces.py mooncake mooncake.jsonl
python3 bench/fetch_traces.py harvard harvard.jsonl
python3 mooncake.py --block-size 512 mooncake.jsonl > mooncake.tree
python3 mooncake.py --block-size 16 harvard.jsonl > harvard.tree
python3 bench/compare_curves.py mooncake.tree harvard.tree > results.json
python3 bench/compare_curves.py --prefix-capacities 0,1,64,1024,8192,65536,1048576 mooncake.tree harvard.tree > prefix-results.json
```

The harness builds both binaries and verifies byte-identical histograms on every trial. It checks that cold and reused accesses sum to the request path lengths. The two synthetic cases run by default, including with no tree arguments.
