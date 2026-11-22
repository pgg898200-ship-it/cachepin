# LRU capacity planning against upstream PrefixSim

The experiment asks how long it takes to produce the number of prefix-cache hits at seven feasible cache sizes on a single CPU. Both tools use serialized arrivals, full prompt pages, empty initial state, uniform block cost, and deepest-first release. All 14 capacity/trace pairs agreed exactly with an independent reconstruction from the upstream simulator's recompute-hole dump. [Parity script](verify_prefixsim.py)

| Input | Prefix Atlas, seven sizes from prepared tree | PrefixSim, seven sizes from prepared LCP | Prefix Atlas, source to seven sizes | PrefixSim, source via LCP to seven sizes |
| --- | ---: | ---: | ---: | ---: |
| FreeInference May 17, 5,684,383 references | 2.78 ms | 8,863 ms | 643.59 ms | 10,473 ms |
| Mooncake Tool and Agent, 386,041 references | 11.44 ms | 354.24 ms | 202.57 ms | 739.98 ms |

Prefix Atlas computes the capacity sweep in one process. PrefixSim's documented CLI accepts one cache size per process. Its seven processes ran serially. The programs read their own prepared representations. The timing measures all work that each program does, including LCP decompression and PrefixSim's general-purpose per-request bookkeeping. The companion [native page-enumerating baseline](curve_benchmark.md) computes all capacities in one process from the same prepared tree.

Source-to-result measurements start from timestamp-ordered JSONL containing complete pages. Preparation runs once per seven-size sweep. Timing starts after download, projection and normalization. The prepared timings include process launch, input parsing, and result emission. Numbers are medians of three trials on an AMD EPYC 9V74 shared Linux host. Runs alternated engine order. Sub-millisecond differences between analyses lasting a few milliseconds can reflect scheduling. PrefixSim ran in release mode with LRU and its post-request verification disabled. The benchmark uses PrefixSim revision `160d5f81724dd9ec28b10d6abb64615e6adbb332`. PrefixSim supports additional policies, request skipping, and cost models beyond this comparison.

## Direct JSONL

The upstream CLI also reads JSONL without a separate conversion. Direct JSONL took 15,200 ms for the seven FreeInference capacities and 1,424 ms for seven Mooncake capacities. For a single 1,024-page Mooncake budget, direct upstream JSONL was faster, with 131 ms versus 202 ms for Prefix Atlas including tree construction. These direct JSONL runs were timed separately on the shared host. The weighted engine's use case is repeated or multi-budget sizing over highly shared prompts.

## Parallel capacity sweeps

A second experiment allowed two or seven independent PrefixSim processes to analyze different capacities simultaneously. The host exposed nine logical CPUs. Elapsed time includes launching all children and waiting for the slowest. All seven sizes read the same prepared LCP file or the same normalized JSONL file. Smaller capacities were scheduled first. The fastest single trials and medians are reported to make host scheduling variation visible.

| Trace | PrefixSim input | Two workers, median / best (ms) | Seven workers, median / best (ms) |
| --- | --- | ---: | ---: |
| FreeInference | LCP | 4,724.7 / 4,434.6 | 4,082.1 / 4,054.2 |
| FreeInference | JSONL | 9,139.7 / 8,260.6 | 6,561.2 / 6,114.5 |
| Mooncake | LCP | 261.7 / 254.3 | 101.3 / 76.0 |
| Mooncake | JSONL | 808.8 / 629.3 | 267.0 / 245.7 |

For FreeInference, the seven-worker prepared LCP median was 4,082 ms against 2.78 ms for the weighted curve. Adding the separately measured median LCP import time gives an estimated 5,878 ms from source via LCP, against 644 ms measured for the weighted pipeline. Direct JSONL took 6,561 ms with seven workers. On Mooncake, the fastest upstream source-to-result route by median was seven direct JSONL workers at 267 ms, versus 203 ms for the weighted pipeline. That narrows the Mooncake sweep advantage to 1.32×. Each median uses three completed trials. One Harvard JSONL trial was retried after available memory rose.

The source-to-result comparison for direct JSONL starts at the same input as the tree importer. Converting JSONL to LCP adds a separate preprocessing cost, measured above. Resident memory was measured in isolated runs per capacity. Before each parallel trial, the harness checked the sum of the largest per-process peaks for that worker count against available memory, allowing 15% plus 256 MiB of headroom. The raw files include the memory measurements and trial times. [FreeInference parallel samples](prefixsim_parallel_harvard.json) · [Mooncake parallel samples](prefixsim_parallel_moon.json)

## Reproduce

Build the project with `make` and put the upstream `prefixsim` and `traceConvLLM` binaries on PATH. Use the fixture fetcher and the exact-comparison script to trim and order Mooncake's complete pages. The Harvard projection from the fetcher already contains complete pages in time order.

```sh
python3 bench/fetch_traces.py mooncake mooncake.jsonl
python3 bench/fetch_traces.py harvard harvard.jsonl
python3 mooncake.py --block-size 512 mooncake.jsonl > mooncake.tree
python3 bench/verify_prefixsim.py mooncake.jsonl mooncake.tree --caps 256,512,1024,2048,4096,8192,16384 --save-normalized mooncake-full.jsonl
python3 bench/compare_prefixsim.py mooncake-full.jsonl --block-size 512 --capacities 256,512,1024,2048,4096,8192,16384 --one 1024 --direct-json
python3 bench/compare_prefixsim.py harvard.jsonl --block-size 16 --capacities 8192,16384,32768,65536,131072,262144,524288 --one 8192 --direct-json
```

To reproduce the parallel panel, reuse the same `harvard.jsonl` and `mooncake-full.jsonl` from above. The linked serial result files supply the verified expected capacities and hit counts.

```sh
traceConvLLM --input harvard.jsonl --output harvard.lcp.zst --block-size-tokens 16
traceConvLLM --input mooncake-full.jsonl --output mooncake.lcp.zst --block-size-tokens 512
python3 bench/parallel_prefixsim.py bench/prefixsim_harvard.json --lcp harvard.lcp.zst --jsonl harvard.jsonl
python3 bench/parallel_prefixsim.py bench/prefixsim_moon.json --lcp mooncake.lcp.zst --jsonl mooncake-full.jsonl
```

[Raw FreeInference results](prefixsim_harvard.json) · [Raw Mooncake results](prefixsim_moon.json) · [Trace sources and attribution](../trace_sources.md)
