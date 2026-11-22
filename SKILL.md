---
name: cachepin
description: Diagnose, canonicalize, coalesce, and optimize multi-agent LLM prompt/KV caches losslessly across Anthropic, OpenAI, vLLM, and SGLang. Use when investigating low prompt-cache hit rates, high multi-agent token spend, cold-start swarm bursts, mixed 1h/5m TTL cache_control placement, or multi-GPU vLLM/SGLang prefix routing.
---

# `cachepin` Agent Skill

`cachepin` is a zero-dependency Rust + AVX2 SIMD binary (`1.1 MB`), C-ABI library (`libcachepin.so`), Python SDK, HTTP/HTTPS/SSE reverse proxy, and native stdio MCP server that fixes multi-agent LLM prompt-cache collapse with **100% lossless** transformations.

## When to Use `cachepin`

- Multi-agent swarms experiencing `<80%` prompt-cache hit rates or unexpected 5×–20× input token bills.
- Non-deterministic MCP/SDK tool registration order, unsorted JSON schema keys, or early timestamps/UUIDs/`req_*` IDs invalidating prefixes.
- All-Gather multi-agent rounds where agents share identical repo files/documents in different orders (*arXiv:2604.03143*).
- Synchronized subagent bursts (`N` agents launching cold on the same prefix) causing `N` redundant cache writes.
- Mixed-TTL (`1h` vs `5m`) Anthropic `cache_control` breakpoint planning (`<= 4` breakpoints, strict `1h`-before-`5m` monotonicity, `270s` keepalive break-even).
- Multi-GPU vLLM / SGLang clusters suffering eviction-regime tail latency under single-home routing.

## When NOT to Use `cachepin`

- Requests shorter than provider minimum cache thresholds (`<512` tokens for Claude 5.5/5.1, `<1024` for Claude 4 / OpenAI).
- One-off zero-reuse prompts (no follow-up turn within 5–60 minutes).
- Workloads whose raw trace already exceeds `95%` cache hit rate (verify with `cachepin audit` first).

## Task-to-Command Quick Reference

| Goal | Command / API |
|---|---|
| Build offline (0 external crates) | `cargo build --release` |
| Audit local Claude Code sessions (`~/.claude/projects`) | `./target/release/cachepin audit --claude-code` |
| Audit a JSONL request trace or directory | `./target/release/cachepin audit <trace.jsonl\|dir> --json` |
| Optimize a single request JSON (stdin/file) | `cat req.json \| ./target/release/cachepin optimize --mode swarm` |
| Start cloud API proxy (Anthropic / OpenAI) | `./target/release/cachepin proxy --upstream https://api.anthropic.com --port 47820` |
| Start multi-GPU vLLM/SGLang adaptive router | `./target/release/cachepin proxy --backends http://g0:8000,http://g1:8000 --kv-tokens 131072 --policy adaptive` |
| Start native stdio MCP server | `./target/release/cachepin mcp` |
| List built-in late-2026 model pricing & cache geometry | `./target/release/cachepin models` |
| Run reproducible `[LIVE-CPU]` + `[SIM-TRACE]` benchmarks | `./target/release/cachepin bench --json` |
| Run full verification gate (53 unit + 10 red-team + controls) | `./scripts/verify.sh` |

## Native MCP Tools (`cachepin mcp`)

Configure in Claude Code / Codex / Cursor / Gemini CLI:

```json
{
  "mcpServers": {
    "cachepin": {
      "command": "cachepin",
      "args": ["mcp"]
    }
  }
}
```

1. `optimize_prompt(payload, provider?, mode?, tenant_id?)` — Canonicalize request + solve exact $O(M^2 K)$ breakpoints + return lossless optimized JSON.
2. `diagnose_diff(prev_payload, curr_payload, model?, provider?)` — AVX2 byte-level prefix divergence offset + root-cause classification + counterfactual recovery.
3. `plan_breakpoints(blocks, model?, provider?)` — Solve exact constrained mixed-TTL (`1h` before `5m`, `K <= 4`) breakpoints + 270s keepalive break-even.
4. `audit_trace(path, mode?, provider?)` — Stream-audit a JSONL trace file for raw vs counterfactual hit rate, USD waste, and top cost-bleeding misses.
5. `route_simulate(backends, requests, policy?)` — Compare `CachePinAdaptiveReplication` against Round-Robin, Least-Loaded, Consistent-Hash, and SingleHome-2xDivert.

## Zero-Dependency Python SDK

```python
from cachepin import CachePin, lcp_len

with CachePin(mode="swarm") as cp:
    report = cp.compile_and_plan(payload_dict, tenant="org-1", provider="anthropic")
    optimized_dict = cp.optimize_dict(payload_dict, tenant="org-1", provider="anthropic")
```

## Verified Performance Receipt (`benchmarks/results.json`)

- **Compile latency (`18.5 KiB` payload, `[LIVE-CPU]`):** `195.3 µs` P50 / `260.1 µs` P99 (`9.57×` faster than Python 3.12, `1.44×` faster P50 / `2.42×` tighter P99 than Node.js 22 V8).
- **AVX2 LCP diff (`[LIVE-CPU]`):** `44.55 GB/s` (`2,155×` faster than Python, `52.1×` faster than Node.js 22).
- **Exact DP solver (`[LIVE-CPU]`):** `7.81 µs` P50; exact match (`<1e-12`) against brute-force on 240 randomized trees.
- **Shadow-KV route decision (`[LIVE-CPU]`):** `1.15 µs` P50; cuts 8-GPU eviction-regime P99 TTFT by `37.0%` vs single-home 2×-divert (`260.0 ms` vs `413.0 ms`) `[SIM-TRACE]`.
- **32-agent live cold-start herd (`[LIVE-CPU]`):** `32 → 1` cold cache writes (`-96.88%`).
- **32-agent coding swarm (`[SIM-TRACE]`):** `2.25% → 98.43%` cache hit rate (`-93.49%` cost, `15.37×` cheaper).
