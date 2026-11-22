---
name: cachepin
description: Diagnose, canonicalize, coalesce, and optimize multi-agent LLM prompt/KV caches losslessly across Anthropic, OpenAI, vLLM, and SGLang. Use when investigating low prompt-cache hit rates, high multi-agent token spend, cold-start swarm bursts, mixed 1h/5m TTL cache_control placement, or multi-GPU vLLM/SGLang prefix routing.
---

# `cachepin` Agent Skill

`cachepin` is a zero-dependency Rust + AVX2/NEON SIMD binary (`~1.1 MB`), C-ABI library (`libcachepin.so`), Python SDK, HTTP/1.1 keep-alive reverse proxy, and native stdio MCP server that fixes multi-agent LLM prompt-cache collapse with **100% lossless** transformations.

## When to Use `cachepin`

- Multi-agent swarms experiencing `<80%` prompt-cache hit rates or unexpected 5×–40× input token bills.
- Non-deterministic MCP/SDK tool registration order, unsorted JSON schema keys, or early timestamps/UUIDs/`req_*` IDs invalidating prefixes.
- All-Gather multi-agent rounds where agents share identical repo files/documents in different orders (*arXiv:2604.03143*).
- Synchronized subagent bursts (`N` agents launching cold on the same prefix) causing `N` redundant cache writes.
- Mixed-TTL (`1h` vs `5m`) Anthropic `cache_control` breakpoint planning (`<= 4` breakpoints, strict `1h`-before-`5m` monotonicity, `270s` keepalive break-even).
- Multi-GPU vLLM / SGLang clusters suffering eviction-regime tail latency under single-home routing.

## When NOT to Use `cachepin`

- Requests shorter than provider minimum cache thresholds (`<512` tokens for Claude 5.5/5.1, `<1024` for Claude 4 / OpenAI).
- One-off zero-reuse prompts (no follow-up turn within 5–60 minutes).
- Workloads whose raw trace already exceeds `95%` cache hit rate (verify with `cachepin audit --auto` first).

## Task-to-Command Quick Reference

| Goal | Command / API |
|---|---|
| Install prebuilt binary (SHA-256 verified) | `curl -fsSL https://raw.githubusercontent.com/pgg898200-ship-it/cachepin/main/scripts/install.sh \| sh` |
| Build offline (0 external crates) | `cargo build --release` |
| Auto-audit local Claude Code & Codex logs | `cachepin audit --auto` |
| Audit Claude Code (`~/.claude/projects`) or Codex (`~/.codex/sessions`) | `cachepin audit --claude-code` / `cachepin audit --codex` |
| Instant SIMD byte-level diff between 2 JSON requests | `cachepin diff agent_a.json agent_b.json [--json]` |
| Audit any JSONL request trace or directory | `cachepin audit <trace.jsonl\|dir> --json` |
| Optimize a single request JSON (stdin/file) | `cat req.json \| cachepin optimize --mode swarm` |
| Start cloud API proxy (Anthropic / OpenAI) | `cachepin proxy --upstream https://api.anthropic.com --port 47820` |
| Start multi-GPU vLLM/SGLang adaptive router | `cachepin proxy --backends http://g0:8000,http://g1:8000 --kv-tokens 131072 --policy adaptive` |
| Start native stdio MCP server | `cachepin mcp` (or `docker run -i --rm ghcr.io/pgg898200-ship-it/cachepin:latest mcp`) |
| List built-in late-2026 model pricing & cache geometry | `cachepin models` |
| Run reproducible `[LIVE-CPU]` + `[SIM-TRACE]` benchmarks | `cachepin bench --json` |
| Run full verification gate (64 unit + 12 red-team + controls) | `./scripts/verify.sh` |

## Native MCP Tools (`cachepin mcp`)

Configure in Claude Code / Codex / Cursor / Gemini CLI (also published to the Official MCP Registry as `io.github.pgg898200-ship-it/cachepin`):

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
2. `diagnose_diff(prev_payload, curr_payload, model?, provider?)` — SIMD byte-level prefix divergence offset + root-cause classification + counterfactual recovery.
3. `plan_breakpoints(blocks, model?, provider?)` — Solve exact constrained mixed-TTL (`1h` before `5m`, `K <= 4`) breakpoints + 270s keepalive break-even.
4. `audit_trace(path, mode?, provider?)` — Stream-audit a JSONL trace file or Claude Code / Codex directory for raw vs counterfactual hit rate, USD waste, and top cost-bleeding misses.
5. `route_simulate(backends, requests, policy?)` — Compare `CachePinAdaptiveReplication` against Round-Robin, Least-Loaded, Consistent-Hash, and SingleHome-2xDivert.

## Zero-Dependency Python SDK

```python
from cachepin import CachePin, lcp_len

with CachePin(mode="swarm") as cp:
    report = cp.compile_and_plan(payload_dict, tenant="org-1", provider="anthropic")
    optimized_dict = cp.optimize_dict(payload_dict, tenant="org-1", provider="anthropic")
```

## Verified Performance Receipt (`benchmarks/results.json`, `v0.2.0`)

- **Compile latency (`18.5 KiB` payload, `[LIVE-CPU]`):** `84.2 µs` P50 / `138.4 µs` P99 (`22.3×` faster than Python 3.12, `3.45×` faster P50 / `8.91×` tighter P99 than Node.js 22 V8).
- **AVX2 LCP diff (`[LIVE-CPU]`):** `42.56 GB/s` (`2,108×` faster than Python, `50.3×` faster than Node.js 22); native ARM64 NEON (`simd: neon`) on Apple Silicon & AWS Graviton.
- **Exact DP solver (`[LIVE-CPU]`):** `8.04 µs` P50; exact match (`<1e-12`) against brute-force on 240 randomized trees.
- **Live HTTP/1.1 keep-alive proxy overhead (`[LIVE-CPU]`):** `219.4 µs` P50 (`0.22 ms`) / `712.8 µs` P99 with warm upstream connection pooling & live provider `usage` reconciliation.
- **Shadow-KV route decision (`[LIVE-CPU]`):** `1.16 µs` P50; cuts 8-GPU eviction-regime P99 TTFT by `37.0%` vs single-home 2×-divert (`260.0 ms` vs `413.0 ms`) `[SIM-TRACE]`.
- **32-agent live cold-start herd (`[LIVE-CPU]`):** `32 → 1` cold cache writes (`-96.88%`).
- **32-agent coding swarm (`[SIM-TRACE]`):** `2.24% → 98.44%` cache hit rate (`-93.50%` cost, `15.39×` cheaper).
