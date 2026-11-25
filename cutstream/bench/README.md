# Streaming stop benchmark

The benchmark sends 256 bytes per stream through a single calling thread. It advances each stream by one chunk per round in a fixed order. A quarter of the streams end in a complete stop, with the other three quarters ending at EOF. Patterns, payloads and cursors are constructed before timing. Cursor allocations triggered by a feed count toward its time. Results include feed, EOF flush and emitted byte-count accounting. They exclude tokenization, IO, output copying and hashing.

To reproduce the native benchmark from the project root on Linux, run

```sh
c++ -std=c++17 -O3 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/cutbench
/tmp/cutbench 9 > /tmp/cutstream-results.csv
```

The three implementations run in rotating order for one warm-up and nine measured samples per case. Each sample averages all calls in the case. The CSV includes the median of the samples, their individual values, and total emitted-byte FNV-1a checksums. CPU time measures process service demand for this single-threaded benchmark. Wall time includes time lost to host scheduling. Neither figure measures individual call tail latency.

| Engine | Search | Emission |
| --- | --- | --- |
| cutstream | Shared compiled plan. Each stream owns one 32-bit cursor. | Emits the longest literal prefix that is safe to release. |
| grouped-rescan | Native C++ candidate scanner grouped by first byte. Single-start sets use the library's byte find. All prefixes are compared directly. | Same release contract as CutStream on all fixtures. |
| fixed-find | Runs native string-view find once per marker on the shortest suffix that could contain a new full match. | Withholds up to the longest marker length minus one until the next call or EOF. |

The two rescanners use one reusable joining buffer per calling thread and one string per stream for pending bytes. All fixtures use reachable markers. CutStream also uses a shared compiled table whose size depends on the marker set. The four-byte cursor describes per-stream state only.

The ordinary cases sample bytes from an ASCII sentence alphabet with no marker starts. Each stream receives a different reproducible payload. These fixtures isolate the cost of checking ordinary bytes. The prefix cases cycle through two near misses, `<|agent_event_!|>` and `<|im_enX`, with a different phase per stream. Four and 32 marker cases use four distinct starting bytes unless the case name says shared-start or prefix. The single marker is `<|im_end|>`, and the remaining synthetic agent-event markers have numeric IDs.

The fixture validator compares match IDs and consumed lengths after every feed. It compares each emitted byte at each feed for exact release and verifies the delayed output is a prefix of the exact output for fixed buffering. It verifies byte-identical final output for all three. The grouped rescanner provides independent decision logic. Mean held bytes is the average pending count after each nonterminal feed, including zero counts.

The recorded run used a Linux x86-64 virtual machine reporting AMD EPYC 9V74, GCC 14.2.0 and glibc 2.41, with process affinity set to reported CPU 6. The machine was shared. The rescanners are implementations in this benchmark. Performance numbers are for matcher kernels and are not measurements of serving engines.

In the recorded run, the 32-marker near-miss fixture took 8.85 CPU ns per feed for CutStream and 132.43 for grouped-rescan. In the four-marker bulk case, each stream's entire 256-byte payload arrived in a single call, and fixed-find took 39.06 CPU ns per feed versus CutStream's 92.77. The full ten-case matrix and all nine samples per implementation are in [results.csv](results.csv).

## Single literal against PCRE2 JIT

A gateway already using PCRE2 can compile a literal once, search each chunk with hard partial matching, and retain the suffix identified by a partial result. The [comparator](pcre_literal.hpp) calls the 8-bit JIT entry point directly, with one shared match-data object and joining buffer per worker. Compilation and JIT setup occur before timing. Both engines return borrowed output. The verifier checks per-feed emission, match ID, consumed length and final bytes. The experiment covers the single literal `<|im_end|>`. Multiple regex alternatives can choose different winning stops. [PCRE2 partial matching](https://www.pcre.org/current/doc/html/pcre2partial.html)

From the Cutstream directory on Linux with PCRE2 development headers and JIT support:

```sh
pkg-config --modversion libpcre2-8
c++ -std=c++17 -O3 -DNDEBUG -Iinclude -DCUTSTREAM_BENCH_PCRE2 bench/bench.cpp -lpcre2-8 -o /tmp/cutbench-pcre2
/tmp/cutbench-pcre2 9 > pcre2.csv
/tmp/cutbench-pcre2 9 > pcre2-repeat.csv
```

Two publisher-run processes on Linux x86-64 (AMD EPYC 9V74, Debian 13, GCC 14.2.0, PCRE2 10.46) each rotated the four engines through one warm-up and nine measured rounds. The table gives median process CPU ns/feed for the first run and PCRE2 time divided by Cutstream time for both. All payloads were 256 bytes. One-byte chunks used 2,048 interleaved streams and the other cases used 4,096.

| Input | Chunk bytes | Cutstream | PCRE2 JIT | Ratio, first / repeat |
| --- | ---: | ---: | ---: | ---: |
| Ordinary text | 1 | 5.041 | 28.633 | 5.68× / 5.47× |
| Ordinary text | 4 | 5.222 | 27.885 | 5.34× / 5.02× |
| Ordinary text | 64 | 7.141 | 28.076 | 3.93× / 3.84× |
| Frequent incomplete prefixes | 4 | 6.889 | 30.735 | 4.46× / 3.93× |

The grouped literal rescanner was the closest comparator in each of these four cases. All four engines' CPU and wall samples, counts and output checksums are in the [first run](pcre2.csv) and [repeat](pcre2-repeat.csv). These measurements cover a reused plan and one worker on synthetic byte streams. They exclude JIT startup, output copying, network latency and model inference.
