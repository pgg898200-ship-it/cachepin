---
name: cachepin
description: Diagnose LLM prompt-cache misses in Anthropic or OpenAI-style request JSON. Use CachePin to compare two prompts, audit local Claude Code or Codex traces, normalize tool order, plan cache breakpoints, or simulate prefix routing across inference backends.
---

# CachePin

Start with the smallest input that reproduces the problem. Work from two request payloads to explain a single miss, or from a local JSONL trace to examine repeated requests.

## Compare two requests

Run the read-only CLI comparison from a checkout:

```sh
cachepin diff examples/prefix-diff/a.json examples/prefix-diff/b.json --mode json --provider anthropic --json
```

Read `raw_lcp_bytes` for the literal shared byte count. Read `reason` and `divergence_path` to locate the likely cause. Compare `raw_matched_tokens` and `opt_matched_tokens` for estimated reuse. The bundled pair yields `tool_order_drift` at `tools[0]`, with 68 shared input bytes and 0 versus 1,099 estimated matched tokens. Provider billing and tokenizer counts can differ from the replay.

## Use MCP

Configure the local stdio command `cachepin` with arguments `["mcp"]`. Send `tools/list` to obtain the schema for the installed version. Parse the JSON in the text content of each tool result.

| Goal | Tool | Inputs |
|---|---|---|
| Compare request payloads | `diagnose_diff` | `prev_json` and `curr_json` hold the previous and current request. For byte offsets in the original JSON formatting, pass strings. |
| Normalize a request | `optimize_prompt` | Pass `payload`. Set `mode` to `json` for key and tool ordering. Set `inject_breakpoints` to `false` to inspect just the transformed payload. |
| Estimate caching breakpoints | `plan_breakpoints` | Pass `block_tokens` and `reuse_counts` arrays or supply the structured `blocks` accepted by the schema. |
| Audit a local trace | `audit_trace` | Pass the trace `path` on the machine running the server. |
| Compare simulated backend routing | `route_simulate` | Set `num_backends`, `num_requests`, and `kv_capacity_tokens` to match the scenario. |

## Use the CLI in a stack

- Run `cachepin audit trace.jsonl --json` on a workload, or select local sessions with `--claude-code` or `--codex`.
- Pipe request JSON to `cachepin optimize --mode json --no-breakpoints` to normalize keys and tool order. Inspect the output before passing it to a provider SDK.
- Use `hoist` or `swarm` when volatile values or the order of shared blocks cause misses. Those modes change how context appears to the model. Evaluate outputs on representative requests.
- Place the optional `cachepin proxy` between the agent and its upstream API. Use `cachepin proxy --help` for upstream, backend and coalescing flags.

[Install the CLI and download sample inputs](https://github.com/pgg898200-ship-it/cachepin#install) · [Integration guide](https://github.com/pgg898200-ship-it/cachepin/blob/main/docs/INTEGRATIONS.md) · [Benchmark data](https://raw.githubusercontent.com/pgg898200-ship-it/cachepin/main/benchmarks/results.json)
