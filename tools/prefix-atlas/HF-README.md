---
title: Prefix Atlas
emoji: 📈
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
short_description: Size an LLM prefix cache from a real agent trace
datasets:
  - harvardMadsys/freeinference_agentic_trace
tags:
  - llm-serving
  - kv-cache
  - performance
  - trace-replay
---

# Prefix Atlas

Calculate how many KV-cache pages a serial LRU cache needs to reach a target prefix hit rate. Use two measured agent workloads or import an exact capacity CSV locally. The calculator includes the complete capacity breakpoints and works without a backend.

On 1,375 top-level FreeInference requests from May 17, 2026, 8,822 pages are sufficient for an 80% hit rate under this model. The denominator is the number of complete page references. See the code for the cache policy, workloads where a simpler implementation is faster, and a reproducible upstream PrefixSim comparison. Serving latency, concurrency and in-flight cache pins require engine-specific testing.

[Source, CLI, and benchmarks](https://github.com/pgg898200-ship-it/cachepin/tree/main/tools/prefix-atlas) · [Harvard MaDSys FreeInference trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace) · [Trace attribution](https://github.com/pgg898200-ship-it/cachepin/blob/main/tools/prefix-atlas/trace_sources.md)
