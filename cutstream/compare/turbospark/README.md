# Native TurboSpark comparison

This harness builds the unmodified `StreamingStopMatcher` module from the [TurboSpark v0.7.1 source revision](https://github.com/whit3rabbit/turbospark/blob/5ede725e9a1e375fb66d135bc0ab6117bf2a5259/crates/tokenizer/src/stop_matcher.rs) with `rustc`, and Cutstream's header with a C++17 compiler. TurboSpark's module uses the Rust standard library. Python downloads the single upstream file and verifies its SHA-256 before compilation. Both binaries use optimization level 3 and the host CPU instruction set.

```sh
python3 compare/turbospark/run.py --output /tmp/cutstream-turbospark.json
```

Requires Python 3.8+, Rust 1.58+, and a C++17 compiler supporting `-march=native`. Use `CXX` or `RUSTC` for alternate compiler executables. For an offline run, pass `--source-file /path/to/stop_matcher.rs` with the same pinned bytes. The output includes compiler versions, machine metadata, raw samples, per-case throughput, and the ratio of median replay times.

Each fixture interleaves 32 streams of synthetic ASCII text. The matrix covers one or four stop strings and 4- or 32-byte chunks. Half the streams end at a unique stop split across chunks. The rest end with an incomplete stop. Several streams contain false stop prefixes. ASCII and a single possible complete match per stream keep TurboSpark's chunk-local earliest-start behavior and Cutstream's earliest-end policy equivalent on these inputs.

Each binary feeds the same pre-generated chunks, materializes one owned string per push, and hashes every emitted byte, frame boundary and terminal flag. A separate Python oracle checks the output count and digest on every pass. Setup, dictionary compilation, cursor allocation, input loading and process launch are excluded from the comparison. The measured loop includes matching, materializing output, checksumming, and flushing at EOF. Each implementation runs full passes for at least 100 ms per sample. Five samples alternate execution order. Ratios characterize this ASCII replay on the reported host.
