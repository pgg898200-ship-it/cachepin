# Converting request histories to weighted trees

The importer inserts each request directly into a radix tree. A shared edge stores a span of block IDs. When a request diverges or ends partway through an edge, the importer splits that span and keeps its existing storage. Each node retains the ordinal it would have received during block-by-block insertion. This preserves the exact tree bytes, including edge numbering, request ordering for equal timestamps, and all existing tree fingerprints.

| Input | Previous importer | Radix importer | Wall-time speedup | Previous peak RSS | Radix peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harvard FreeInference, May 17 | 1,844 ms | 641 ms | 2.88× | 155.7 MiB | 35.3 MiB |
| Mooncake FAST’25 tool/agent | 374 ms | 281 ms | 1.33× | 44.2 MiB | 24.9 MiB |

Each measurement starts a fresh Python process, parses the same JSONL, constructs the tree, sorts requests by timestamp and writes the weighted tree to a local file. The harness compares the full output SHA-256 on every trial. Times and peak resident memory are medians of five measured runs after one warm-up with alternating run order. The previous importer built a dictionary entry for each distinct full prefix, then contracted unary chains. [Raw samples](import_results.json) · [Trace sources](../trace_sources.md)

```sh
python3 bench/compare_importers.py mooncake.jsonl harvard.jsonl
python3 -m unittest test_importer
```

## From raw JSONL to prefix-hit counts

Both pipelines launch the radix importer and pipe its output into a native analyzer. They cover seven capacities from zero to 1,048,576 pages. The expanded control enumerates page accesses. The weighted analyzer works on edge spans. Each measured invocation includes process startup, JSON parsing, tree construction, request ordering, the pipe, the calculation, and delivery of the result CSV. All paired CSV digests agree.

| Input | Weighted pipeline | Expanded pipeline | Expanded / weighted |
| --- | ---: | ---: | ---: |
| Harvard FreeInference, May 17 | 616 ms | 657 ms | 1.07× |
| Mooncake FAST’25 tool/agent | 274 ms | 255 ms | 0.93× |

The one-shot pipelines spend most of their time converting JSONL. A saved tree allows subsequent capacity studies to pay only the analyzer cost. [Pipeline samples](import_pipeline_results.json) · [Prepared-tree comparison](curve_benchmark.md)

```sh
python3 bench/compare_importers.py mooncake.jsonl harvard.jsonl --prefix-capacities 0,1,64,1024,8192,65536,1048576
```
