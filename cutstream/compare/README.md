# Stop matching on vLLM-style deltas

[The pinned vLLM implementation](https://github.com/vllm-project/vllm/blob/0e3c8dbdae9d55149b0fba4e6c1014b4b08c9c73/vllm/v1/engine/detokenizer.py) finds the first stop to complete in decoded text. When the caller excludes the stop, its output routine holds back the length of the longest stop minus one Unicode code point. Cutstream evaluates supplied byte deltas and emits the prefix that is already safe.

From the repository root, install Cutstream and run:

```sh
python -m pip install .
python compare/vllm.py > local-vllm-results.json
```

The script retrieves `detokenizer.py` at the pinned commit, verifies the complete file's SHA-256, and compiles the original `check_stop_strings` and `BaseIncrementalDetokenizer.get_next_output_text` function bodies. A local copy at that revision also works with `--vllm-source /path/to/detokenizer.py`. The harness performs the decoded-text append and stop truncation around those calls and passes actual Python strings to vLLM. The Cutstream side receives the same pre-encoded ASCII deltas through its public Python API. Upstream model execution, token ID bookkeeping, tokenizer, transport, and encoding are outside both measurements.

Each warm-plan synthetic response emits 8,192 bytes of repetitive tool output with occasional partial prefixes and ends with one of the configured markers. The main grid uses 1, 4 and 32 twelve-byte stops, shared and mixed starting bytes where applicable, and 4-, 32- and 1,024-byte feeds. Three additional cases cover fewer stops and coalesced chunks up to 4,096 bytes. Each side must emit exactly the same response and identify the expected marker. The stop ends in the final feed. One stream object is constructed per response. Compiling the shared Cutstream plan and creating the common marker lists happen before timing. Each of the seven trials times batches of complete responses using the benchmark thread's CPU clock. Method order alternates. A separate cold-plan grid recompiles 1, 4, or 32 stops on each 64-byte response at 4- and 128-byte feed widths. In those cases the vLLM wrapper also computes the required holdback length per response. This measures the cost of foregoing reuse, as when requests supply unique marker sets. The [raw results](vllm-results.json) include every trial, the processor and Python version.

Mean pending bytes are measured across nonfinal feeds as the number of received bytes still hidden from a downstream consumer. This measures visibility at feed boundaries. Scheduling and network latency determine when those bytes reach a real client. On this fixture a four-byte delta exposes output at feed one with Cutstream and feed three with vLLM. A client placed after vLLM with its own stop matching enabled cannot recover bytes that vLLM has already withheld.

The two implementations handle tie cases differently. Cutstream prefers the longer stop for a common end position. vLLM uses stop-list order. With stops `["b", "ab"]` and input `"ab"`, vLLM emits `"a"` and Cutstream emits an empty response. The benchmark markers are equal-length and unique. All fixture inputs use ASCII so byte boundaries coincide with Unicode character boundaries. The results describe host-side matching and emission on this fixture.

Other available implementations include [TurboSpark's incremental longest-safe-prefix matcher](https://docs.rs/turbospark-tokenizer/0.1.0/turbospark_tokenizer/struct.StreamingStopMatcher.html), [llm-tokenizer's shared stop automaton](https://docs.rs/llm-tokenizer/2.0.0/llm_tokenizer/stop/struct.StopRules.html), and [Rust aho-corasick's buffered stream iterator](https://docs.rs/aho-corasick/latest/aho_corasick/struct.AhoCorasick.html#method.stream_find_iter). None of their timings appear in the vLLM results.

## llama.cpp native stop-search comparison

Run `python3 compare/llamacpp.py` with Python and a C++17 compiler. The runner downloads the pinned upstream server and common header and the Vercel OpenAI fixture. It extracts and compiles the original `find_stopping_strings`, `string_find_partial_stop`, and `string_ends_with` functions. The surrounding adapter holds only unsent bytes in each slot and honors the server's full-match/partial-match send decisions on these complete UTF-8 text chunks. Both engines return an owned string and checksum its bytes. The recorded text has no complete marker. A second case adds synthetic near misses and one split delimiter. The runner checks all output against expected text before timing. It replays the same 300 (or 302) chunks across 512 slots, rotates two subjects, and reports the median of eleven single-thread process CPU samples. The measurement excludes UTF-8 validation, detokenization, server history, token metadata, sockets and model execution. `--source` and `--fixture` accept pinned local sources for offline use.

[Original server implementation](https://github.com/ggml-org/llama.cpp/blob/5bacb7c868a2bc98682de15f447fc1ce17c9c49c/tools/server/server-context.cpp) · [Raw CPU samples](../evidence/llamacpp.csv) · [TurboSpark native comparison](turbospark/README.md)
