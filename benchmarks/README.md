# Reproduce the native benchmark

Run from the repository root with GCC or clang and a C++17 standard library.

```sh
make tests/benchmark
./tests/benchmark 200000 5
```

[native.csv](native.csv) records medians in milliseconds from a Linux x86-64 container, AMD EPYC 9V74, g++ 14.2.0, flags `-O3 -std=c++17 -Wall -Wextra -Wpedantic`. One thread processed 200,000 paired integer service durations from the deterministic generator in [tests/reference.hpp](../tests/reference.hpp). Groups contain eight leaves. A correlated base delay, replica-local jitter and occasional long service times produce heterogeneous races. Test SLO is 7,500 μs, and grid spacing is 32 μs. Each method warms up once, then rotates its order over five repetitions. Times exclude trace generation, CSV parsing and output serialization. Every output field is compared for exact equality outside the timed interval.

| Method | Measured operation | Storage beyond input and output |
|---|---|---|
| Compiled | Build fixed-grid accumulator from input pairs and finalize all results | O(K) |
| Direct | Evaluate every leaf at every delay, aggregate leaves and fanout misses | O(1) |
| Sorted build + scan | Construct sorted raw breakpoint arrays and sweep once | O(N) |
| Sorted warm scan | Reuse those arrays with the same SLO and a new grid | O(N) retained index |

Here N counts leaves and K counts grid points. Direct wins for a single delay on this host. A sorted index can serve fresh grids quickly after the raw events have been sorted. The fixed-grid kernel is designed for streaming ingestion when the grid is known up front. A saved CSV also supports instant lookup for a previously evaluated delay. Benchmarks are synthetic, machine-specific, and establish no production latency improvement.

## Ten-million-leaf streaming stress test

```sh
make benchmarks/stream
./benchmarks/stream 10000000 1 5
./benchmarks/stream 10000000 4 5
```

This Linux stress harness generates each pair inside the timed ingest loop, discards it after accumulation, closes each group of eight, and evaluates 16,384 integer delays at a 7,500 μs deadline. Each worker owns a contiguous partition of complete groups. Ingestion includes grid initialization and thread startup/join. Merge and finalization are independently timed. A direct replay oracle regenerates all ten million rows outside the timed region and verifies all outcome fields at seven delays. The full-output hash also matches across worker counts and all trials. No CSV parsing, network traffic, or external service is involved.

| Workers | Median ingest | Leaves/s | Median merge | Median snapshot | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 117.338 ms | 85.2 million | — | 0.147 ms | 3,152 KiB |
| 4 | 85.995 ms | 116.3 million | 0.270 ms | 0.275 ms | 5,468 KiB |

Same host and compiler as above. Peak RSS is the Linux process high-water mark from getrusage over five trials. Timing and memory accounting under virtualization can vary. Raw results are in [stream-1.csv](stream-1.csv) and [stream-4.csv](stream-4.csv).

## Bounded-time histogram

```sh
make benchmarks/dense
./benchmarks/dense
```

[dense.csv](dense.csv) measures a second exact competitor on the same deterministic trace. It builds four 32-bit frequency counters at each integer microsecond through a known trace bound of 103,000 μs. Prefix counters and first moments recover outcomes when scanning new grids through this range at the same SLO. Each method warms up, then rotates timing order across nine trials, checking every field against the sorted event-sweep reference.

For 1,024 delays the full-domain histogram took 0.8944 ms to build and scan, compared with 1.0727 ms for Racecurve in this run. Its warm scan took 0.0547 ms. The histogram used 1,648,032 bytes of counter storage versus 49,152 for Racecurve's grid bins, excluding output buffers and common scalar fields. This benchmark favors the histogram by providing the bound in advance, and the histogram delivers flexibility by answering additional grids within that bound. An implementation tailored to a single query horizon could also truncate out-of-range events with overflow aggregates. Raw process RSS is reported only for the separate streaming test above.
