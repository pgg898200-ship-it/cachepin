# Cutstream

**Send the safe bytes of a model response as soon as they arrive.** Compile your gateway's literal stop strings once. Each C++ stream needs one four-byte cursor, regardless of marker length. A feed returns up to two borrowed spans in emission order. It allocates no memory.

[Try the interactive replay](https://pgg-cutstream.tancurl3.chatgpt.site) · [Native implementation](include/cutstream.hpp) · [Python adapter for streamed chat content](examples/gateway.py) · [Reproduce the SDK replay](examples/replay_sdk.py)

With the stop string `</tool>`, the chunk `hello </to` releases `hello ` immediately. The cursor identifies the withheld `</to` in the shared dictionary. If the next chunk is `ol>tail`, the stream stops after its third byte. If the next chunk is `ast`, the stream releases `</toast`. No caller-owned copy of the preceding chunk needs to survive.

## Measured performance

A public GitHub Actions run compiled Cutstream and TurboSpark v0.7.1's original Rust matcher on an Intel Xeon 6975P-C. Both consumed the same chunks, materialized owned output strings and checked the bytes, frame boundaries and terminal flags against an independent oracle.

| Stops | Chunk size | Cutstream | TurboSpark | Throughput ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 bytes | 253.2 MiB/s | 17.6 MiB/s | 14.4× |
| 1 | 32 bytes | 688.7 MiB/s | 45.2 MiB/s | 15.2× |
| 4 | 4 bytes | 248.4 MiB/s | 4.9 MiB/s | 51.1× |
| 4 | 32 bytes | 542.1 MiB/s | 9.2 MiB/s | 58.6× |

These are medians of five alternating samples on 32 interleaved synthetic ASCII streams. Dictionary construction was excluded from both timings. TurboSpark also handles Unicode text, and competing matches can follow different policies. The workloads ensure the two policies agree. [Public run](https://github.com/pgg898200-ship-it/cachepin/actions/runs/45065919328) · [Raw samples](evidence/turbospark-ci.json) · [Reproduction](compare/turbospark/README.md)

**Where it pays off.** On a separate AMD EPYC 9V74, an owned-output replay of 300 recorded SDK deltas across 512 streams took 20.41 CPU ns/feed against 48.39 for llama.cpp's pinned stop-search functions in a bounded-buffer adapter. A native 32-marker near-miss workload with 4,096 interleaved streams took 8.85 CPU ns/feed against 132.43 for a grouped prefix rescanner with equivalent release timing. The results cover host-side matching and emission. [llama.cpp replay](compare/llamacpp.py) · [Raw samples](evidence/llamacpp.csv) · [Native benchmark](bench/README.md)

**Where it loses.** With four stops and 256-byte chunks, fixed-window search took 39.06 CPU ns/feed against Cutstream's 92.77, at the cost of later release. Compiling a fresh one-stop plan for a 64-byte response delivered in one chunk took 1.36 µs in Cutstream against 1.05 µs in vLLM's Python routines. For four-byte chunks with four reusable markers, the same vLLM comparison took 880 ns/feed against Cutstream's 110. [Native samples](bench/results.csv) · [Pinned vLLM routines, cold and warm results](compare/README.md)

## Install

For C++17, copy [cutstream.hpp](include/cutstream.hpp) into your include path. For Python 3.10+ with a C++17 compiler, install from the public source branch.

```sh
python -m pip install 'pgg-cutstream @ git+https://github.com/pgg898200-ship-it/cachepin.git@cutstream-v0.1#subdirectory=cutstream'
printf 'hello </tool>trailing' | cutstream --stop '</tool>'
```

```python
from cutstream import Plan

stops = Plan([b"</tool>", b"<|im_end|>"])  # one immutable plan for all requests
s = stops.stream()
assert s.feed(b"hello </to") == b"hello "
assert s.feed(b"ol>trailing") == b""
assert (s.match, s.consumed, s.done) == (0, 3, True)
```

`plan.filter(byte_chunks)` and `plan.afilter(async_byte_chunks)` yield nonempty visible output and stop reading at a match. In a gateway, consume **decoded append-only content deltas** inside the upstream SDK's stream context. This lets the caller close the producer on a match. Raw SSE frames, token IDs, cumulative snapshots, and tool-call argument events require their own decoding. The Python API returns bytes and includes Python object overhead. The four-byte and borrowed-output claims apply to the native cursor.

```cpp
#include "cutstream.hpp"

cutstream::Plan stops({"</tool>", "<|im_end|>"});
cutstream::Cursor cursor;
// For each transport delta:
auto out = stops.feed(cursor, "hello </to");
// Write out.prefix, then out.input. At EOF, write stops.finish(cursor).
```

The input span borrows the current chunk. The prefix span and EOF tail borrow the plan. Consume the views before the backing storage expires. Share a plan across threads with one cursor per ordered stream. Reset the cursor before switching plans.

## What it guarantees

The earliest **ending byte** wins. Among markers ending together, the longest wins, with original index breaking duplicates. Zero markers pass through every byte. An empty marker is rejected. A hit consumes through the delimiter, leaving trailing bytes available to a framing caller. EOF flushes an incomplete marker literally. These byte rules are independent of transport partitioning and preserve arbitrary binary input. A text sink should apply an incremental character decoder to the visible bytes.

The plan removes markers containing an earlier-ending stop. With stops `BC` and `ABCX`, the initial `A` can be emitted immediately because `ABCX` can never win. The remaining Aho–Corasick state identifies the longest suffix that is a prefix of a viable marker. Completing that marker is a witness that the withheld suffix could still be dropped. No longer suffix can be dropped, because it would have to begin a viable marker. This gives the largest irrevocably visible prefix at each feed boundary for the stated policy. The compiled byte-class table trades shared memory and construction time for constant-time transitions. The four-byte figure excludes that shared table.

## Evidence from public SDK fixtures

Three pinned Vercel AI SDK fixtures were replayed at their original content-delta boundaries with four configured ASCII stops. None contained a complete stop. A fixed 17-byte holdback and Cutstream returned identical final content.

| SDK fixture | Nonempty deltas | First visible delta, Cutstream | First visible delta, fixed window |
| --- | ---: | ---: | ---: |
| OpenAI text | 300 | 1 | 5 |
| Azure model router | 4 | 1 | 3 |
| OpenAI refusal | 11 | 1 | 5 |

These are delta positions. The fixtures have no inter-delta wall-clock timing. The byte-window comparator models a buffering policy. [Replay, upstream revision and fixture checksums](evidence/sdk-replay.json)

## Related implementations

vLLM's detokenizer has an output holdback and supports token-aware behaviors including minimum token counts. TurboSpark's StreamingStopMatcher also emits a safe prefix. llm-tokenizer already shares compiled stop rules among decoders. Cutstream targets decoded bytes in a gateway that can reuse its dictionary across many simultaneous streams. Compare stop-order and text-decoding rules before replacing an engine's existing matcher. [vLLM detokenizer](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/detokenizer/) · [TurboSpark](https://docs.rs/turbospark-tokenizer/latest/turbospark_tokenizer/struct.StreamingStopMatcher.html) · [llm-tokenizer](https://docs.rs/llm-tokenizer/latest/llm_tokenizer/stop/struct.StopRules.html)

## Reproduce

From this directory, run the native differential suite and the Python integration tests.

```sh
c++ -std=c++17 -O3 -Wall -Wextra -Iinclude tests/test.cpp -o /tmp/cutstream-test
/tmp/cutstream-test
python -m pip install .
python -m unittest discover -s tests -p 'test_python.py'
python examples/replay_sdk.py
```

For an OpenAI-compatible completions endpoint, run the [SGLang integration example](examples/sglang_completions.py). Its offline mode replays the split-marker case in [SGLang issue 50735](https://github.com/sgl-project/sglang/issues/50735). The issue author submitted an upstream fix; the example shows how to enforce caller-owned stops at the client boundary.

[Adoption evidence](evidence/adoption.md) · MIT licensed. [License](LICENSE)
