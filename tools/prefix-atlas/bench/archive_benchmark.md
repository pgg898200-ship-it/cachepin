# Indexed archive measurements

Each tool receives identical compact Mooncake-style JSONL and returns the requested records in source order for a scan or query order for a gather. The Harvard input is a projection of one day of FreeInference traffic. The Mooncake input is the complete FAST’25 tool/agent trace. The shuffled control permutes the observed Mooncake ID vocabulary bijectively, preserving every prefix relationship and row order. [Input provenance](../trace_sources.md)

| Workload | Method and rows per frame | Archive bytes | Get 1, ms | Get 32, ms | Get 256, ms | Scan, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mooncake | Whole-file Zstd | 505,319 | 9.09 | 7.49 | 9.03 | 9.66 |
| Mooncake | Indexed raw, 64 | 654,315 | 2.02 | 2.84 | 6.29 | 10.50 |
| Mooncake | Prefix delta + runs, 64 | 375,338 | 2.16 | 2.65 | 5.02 | 18.15 |
| Mooncake | Prefix delta + runs, 512 | 326,583 | 1.96 | 3.13 | 4.54 | 15.63 |
| Mooncake | PrefixPack | 431,437 | 2.43 | 2.83 | 8.81 | 21.87 |
| Harvard | Whole-file Zstd | 405,567 | 11.58 | 14.37 | 26.54 | 27.08 |
| Harvard | Indexed raw, 64 | 520,866 | 4.36 | 15.64 | 23.15 | 33.60 |
| Harvard | Prefix delta + runs, 64 | 29,207 | 2.53 | 7.06 | 33.38 | 135.39 |
| Harvard | Prefix delta + runs, 512 | 26,596 | 2.71 | 7.19 | 33.81 | 111.36 |
| Harvard | PrefixPack | 31,804 | 2.33 | 7.71 | 61.89 | 101.91 |
| Mooncake shuffled IDs | Whole-file Zstd | 877,924 | 17.65 | 20.21 | 29.78 | 25.14 |
| Mooncake shuffled IDs | Indexed raw, 64 | 1,166,355 | 3.89 | 7.77 | 14.62 | 23.50 |
| Mooncake shuffled IDs | PrefixPack | 1,231,662 | 6.31 | 7.23 | 25.62 | 71.17 |

On the Harvard projection, whole-file Zstd occupies 12.75 times the space of PrefixPack. The 512-row prefix-delta/run baseline takes 26,596 bytes and fits both tested production traces in less space than PrefixPack. Shuffling the numeric labels on Mooncake reverses the whole-file size comparison. PrefixPack benefits from the consecutive IDs used in the original releases.

## Method

Zstandard 1.5.7 uses compression level 3. Indexed competitors store independent frames and an offset table. The delta codec retains the common prefix with the preceding row inside its frame and encodes the remaining IDs as unsigned varints. The run variant also encodes runs of three or more consecutive numbers. Sorted variants cluster rows by their complete ID vectors and include a four-byte permutation entry per row to restore the original order. All index bytes count toward size. The whole-file reader streams up to the last requested row and retains selected rows plus a partial line across decompression buffers.

The harness compares each scan and random gather with input records before timing. It checks parsed JSON equality. Each timed command runs as a fresh native process against a warm OS page cache. Timings include startup, file open, index reads, decompression and JSONL delivery. One fixed, pseudorandom batch of each size is sampled with replacement. The single-row measurement repeats one chosen row and does not estimate latency across the whole archive. Scans write an ordinary local file without fsync. The shared AMD EPYC 9V74 host introduces noise, especially in short operations. Values are medians of seven trials for the two public traces and three trials for the shuffled control. The raw results also include 16-row frames, sorted frames, delta-only frames, per-trial timings, query fingerprints and tested binary hashes. [Raw results](archive_results.json)

## Reproduce

From the repository root, use a C++17 compiler, libzstd, Python 3 and the zstd command for the Harvard download.

```sh
make prefixpack
python3 bench/fetch_traces.py mooncake mooncake.jsonl
python3 bench/fetch_traces.py harvard harvard.jsonl
python3 bench.py mooncake.jsonl > mooncake-results.json
python3 bench.py harvard.jsonl > harvard-results.json
python3 bench.py mooncake.jsonl --permute-ids 2026 --frames 64 --repeats 3 > shuffled-results.json
```
