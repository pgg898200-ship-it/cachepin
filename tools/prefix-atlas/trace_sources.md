# Production workload fixtures

The benchmark uses published, anonymized request traces. The workloads contain real prefix block identities. Token-count-only traces cannot reproduce the same block sharing. Download and verify the inputs from the Prefix Atlas tool directory with Python 3 and the zstd CLI.

```sh
python3 bench/fetch_traces.py mooncake /tmp/mooncake.jsonl
python3 bench/fetch_traces.py harvard /tmp/harvard.jsonl
```

| Fixture | Requests | Block references | Input bytes | SHA-256 of benchmark JSONL |
| --- | ---: | ---: | ---: | --- |
| Mooncake FAST’25 Tool and Agent | 23,608 | 409,616 | 4,415,857 | `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71` |
| FreeInference, May 17 root requests | 1,375 | 5,684,383 | 39,813,265 | `b55fe0c2c6b5e069bc1551159fdec4078e45842e5e3b7a5b0dd221c925260956` |

**Mooncake.** Qin et al., *Mooncake: Trading More Storage for Less Computation*, FAST ’25. This fixture is the complete released hour of Tool and Agent requests, unchanged at revision `3cca71daccf2a7afb8fe3f0295358f70e3a69fdb`. Each source hash describes a prefix through one 512-token page. The codec retains partial last pages. The pinned repository includes Apache and Mulan license files. The download script fetches the trace directly from its publisher. [Paper](https://www.usenix.org/conference/fast25/presentation/qin) · [Trace description](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)

**FreeInference.** Harvard MaDSys, *From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads*. We selected the first released date, May 17, 2026, from dataset revision `bb26cf72c9c20f2b05f765d88548156a74d3c253`. The source is CC BY 4.0. The projection includes every top-level request in the day's 25 root sessions, stable-sorted by `start_ms`. It omits nested subagent requests and copies the ordered 16-token `block_ids` into `hash_ids`. The fixture sets `timestamp` to source start time in milliseconds, `output_length` to provider completion tokens, and `input_length` to 16 times the number of complete blocks under the trace's tokenizer. The projection deliberately retains four fields for comparable codec inputs. The provider's original prompt-token count can differ. [Dataset and attribution](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace) · [License](https://creativecommons.org/licenses/by/4.0/)

## Mooncake conversation

The FAST’25 Conversation workload contains 12,031 online requests sampled over one hour. The publisher released it separately from its Tool and Agent workload. The download helper verifies the original 3,029,533-byte JSONL against SHA-256 `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df`. We used every request, stable-sorted by arrival, and excluded partial trailing 512-token pages. [Pinned trace and workload description](https://github.com/kvcache-ai/Mooncake/tree/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release)

| Capacity (pages) | Hit pages | Hit rate |
| ---: | ---: | ---: |
| 0 | 0 | 0.00% |
| 1,024 | 13,034 | 4.71% |
| 8,192 | 54,387 | 19.67% |
| 65,536 | 103,786 | 37.54% |
| 147,880 | 105,591 | 38.19% |
| 147,881 | 105,592 | 38.19% |

Of 276,491 complete page references, 170,899 distinct pages each require a first access. The remaining 105,592 references set the maximum possible hit count, first reached at 147,881 pages. We checked all six counts by replaying the source hash IDs through a page-by-page LRU cache, looking up the resident prefix at arrival and touching pages deepest first on completion.

```sh
python3 bench/fetch_traces.py conversation /tmp/conversation.jsonl
python3 mooncake.py --block-size 512 /tmp/conversation.jsonl > /tmp/conversation.tree
prefix-atlas /tmp/conversation.tree --capacities 0,1024,8192,65536,147880,147881
python3 bench/compare_curves.py --no-synthetic --prefix-capacities 0,1024,8192,65536,147880,147881 /tmp/conversation.tree
```

On the prepared tree the two native analyzers agreed at every tested budget. Median runtime over nine trials was 6.09 ms for Prefix Atlas and 5.50 ms for the expanded-page control. These measurements cover serial cache analysis and exclude JSONL import. [Trial times and machine details](bench/conversation_results.json)

## PrefixSim comparison

PrefixSim's LCP format serves offline simulation. Its encoder interns each block path to a dense ID and stores future accesses for each block, so oracle eviction policies can replay in one pass. Conversion adds fixed simulation fields and changes source IDs on every row of both fixtures. The format stores one sequential Zstandard stream with variable-size records and no row offsets. An application can keep the decoded trace in memory for repeated lookups. [LCP specification](https://github.com/cacheMon/PrefixSim/blob/160d5f81724dd9ec28b10d6abb64615e6adbb332/doc/lcp_format.md)

We built the unmodified upstream `traceConvLLM` converter at revision `160d5f81724dd9ec28b10d6abb64615e6adbb332` with libCacheSim submodule `3fa480b6894ed1b9f75b8a9c737a2e34c2ca6016`. Default LCP compression uses Zstandard level 8 with long-distance matching. These commands reproduce the comparison with that executable on the PATH.

```sh
traceConvLLM --input /tmp/mooncake.jsonl --output /tmp/mooncake.lcp.zst --block-size-tokens 512
traceConvLLM --input /tmp/harvard.jsonl --output /tmp/harvard.lcp.zst --block-size-tokens 16
wc -c /tmp/mooncake.lcp.zst /tmp/harvard.lcp.zst
```

| Fixture | LCP Zstandard bytes | Uncompressed LCP bytes consumed through middle row, starting at byte zero | Distinct positional IDs |
| --- | ---: | ---: | ---: |
| Mooncake | 785,875 | 4,210,772 | 183,300 |
| FreeInference | 886,332 | 45,453,504 | 655,155 |

The scan count includes the 8,192-byte header and each row through zero-based index `rows // 2`. Each row occupies 68 bytes plus 16 bytes per block in LCP v1. These counts describe logical decoder work. Physical I/O depends on compression and buffering. An independent comparison with indexed, key-preserving Zstandard frames is available in the benchmark harness. Dates and time units are carried as numeric values in this storage experiment. Running a time-aware LCP simulation requires normalization to seconds.

## Replay selected requests in GuideLLM

GuideLLM's Mooncake loader turns shared block IDs into shared synthetic prompt tokens. Export a chronological selection from an archive, then run the benchmark against an OpenAI-compatible endpoint with a tokenizer available to GuideLLM.

```sh
./prefixpack get toolagent.ppk 0 1 2 | python3 replay.py --block-size 512 --time-unit ms - replay.jsonl
guidellm run --backend kind=openai_http,target=http://localhost:8000 \
  --profile kind=replay \
  --data kind=mooncake,source.kind=json_file,source.path=replay.jsonl,hash_id_block_size=512
```

For a full archive, run `./prefixpack unpack toolagent.ppk raw.jsonl`, then `python3 replay.py raw.jsonl replay.jsonl`. Use `--block-size 16` and `hash_id_block_size=16` for the Harvard projection. A slice starts with an empty synthetic prompt cache. Include the preceding requests when the original warmup matters.

The adapter accepts ordered JSONL from a file or standard input, writes four GuideLLM fields, and rebases the first request to zero before converting timestamps to seconds. Choose `--time-unit s|ms|us|ns` to match the input. Named output files are replaced after successful validation. Piped output can contain earlier rows if a later row fails validation.

GuideLLM requires signed-32-bit nonnegative hash IDs. The adapter preserves available source IDs in that range, assigns spare IDs when a source label is wider or appears below another parent, and shares the assignment across all rows. Memory scales with distinct block paths. It checks token counts, number of blocks, ordering, and consistent partial-tail size for each reused prefix. Additional fields, including session graphs and request durations, are omitted. GuideLLM schedules these rows as separate requests.

Both production fixtures passed full export. The optional check below exercises the installed GuideLLM 0.8.1 JSON-file deserializer and prompt generator using a local byte tokenizer. It compares three generated prompt lengths, repeated/shared blocks and exact scheduled offsets through the archive → indexed lookup → export → deserializer path. It runs offline after installation. Server-side latency and cache-hit measurements require a serving endpoint.

```sh
make
python -m pip install 'guidellm==0.8.1'
python bench/check_guidellm.py
```

[GuideLLM trace-replay guide](https://vllm-project.github.io/guidellm/latest/guides/trace_replay/) · [Mooncake reader source](https://github.com/vllm-project/guidellm/blob/92799ecb9ff4c14d80efada9791d2aa6a5580ec0/src/guidellm/data/deserializers/trace_mooncake.py)

## llama.cpp prompt reuse

On 2026-11-25 we ran four serial, in-process completions with llama-cpp-python 0.3.16 and the stories260K GGUF on CPU. Each 33-token prompt contains two shared eight-token pages, two branch-specific pages and a different final token. Branch order is A, B, A, A. A fresh model instance starts each of the two runs. The active context holds 128 tokens. The second run also enables a 64 MiB LlamaRAMCache for saved inference states. [Fixture](https://huggingface.co/ggml-org/models/blob/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf) · [Completion implementation](https://github.com/abetlen/llama-cpp-python/blob/v0.3.16/llama_cpp/llama.py)

| Request | Atlas hit pages at an eight-page budget | Active context, tokens reused | With snapshots, tokens reused |
| --- | ---: | ---: | ---: |
| A, first visit | 0 | 0 | 0 |
| B | 2 | 16 | 16 |
| A, return | 4 | 16 | 32 |
| A, consecutive | 4 | 32 | 32 |

The Atlas budget fits all six distinct complete pages. The engine check counts tokens passed to the real evaluation method during each one-token completion and subtracts that count from the prompt length. The unique tail always requires evaluation, so dividing reused tokens by eight gives complete hit pages. On the return to A, the active context retains the common prefix with B. Restoring a saved state also recovers A’s suffix. The capacities in pages and snapshot bytes describe different cache mechanisms. Agreement here applies to this controlled sequence; the check does not measure concurrent serving, latency or equal memory cost.

With the Prefix Atlas CLI on PATH and a Python environment with a C++ compiler, run:

```sh
python -m pip install llama-cpp-python==0.3.16
curl -fL https://huggingface.co/ggml-org/models/resolve/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories260K.gguf -o /tmp/stories260K.gguf
python bench/check_llama_cpp.py /tmp/stories260K.gguf
```

The check verifies the model SHA-256, interns exact token prefixes for the CLI, and compares its cumulative counts with the engine’s per-request counts. It uses one CPU inference thread and requires no endpoint or GPU.
