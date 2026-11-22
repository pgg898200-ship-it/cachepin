# `cachepin` Integration Guide

Drop `cachepin` in front of any multi-agent harness, coding agent, SDK, or self-hosted vLLM/SGLang cluster. Zero external dependencies, 100% lossless.

---

## 1. Self-Hosted vLLM / SGLang / Ollama Fleets (Pure Rust Proxy: `1.15 µs` Route / `0.59 ms` Loopback Overhead)

### Single Backend
```bash
cachepin proxy --upstream http://127.0.0.1:8000 --port 47820 --mode swarm
export OPENAI_BASE_URL=http://127.0.0.1:47820/v1
```

### Multi-GPU Fleet (Replication-Aware Shadow-KV Routing)
```bash
cachepin proxy \
  --backends http://gpu0:8000,http://gpu1:8000,http://gpu2:8000,http://gpu3:8000 \
  --kv-tokens 131072 \
  --policy adaptive \
  --port 47820
```
Open live telemetry at `http://127.0.0.1:47820/dashboard` or scrape Prometheus at `http://127.0.0.1:47820/metrics`.

---

## 2. Cloud APIs (Anthropic Claude / OpenAI / Claude Code / Codex / OpenClaw)

Run the single native binary directly (verified TLS via system OpenSSL, zero secrets in argv):

```bash
# Anthropic Messages API (Claude Code, custom multi-agent swarms)
cachepin proxy --upstream https://api.anthropic.com --port 47820 --mode swarm
ANTHROPIC_BASE_URL=http://127.0.0.1:47820 claude

# OpenAI Chat Completions & Responses API (Codex, OpenAI Agents SDK, LangGraph)
cachepin proxy --upstream https://api.openai.com --port 47821 --mode swarm
OPENAI_BASE_URL=http://127.0.0.1:47821/v1 my-agent-swarm
```

Or use the zero-dependency Python stdlib TLS bridge backed by `libcachepin.so`:

```bash
PYTHONPATH=python python3 -m cachepin.https_proxy \
  --upstream https://api.anthropic.com \
  --port 47820 --mode swarm
```

---

## 3. Native MCP Server (Claude Code, Codex, Cursor, Gemini CLI, OpenCode)

With the [release executable](https://github.com/pgg898200-ship-it/cachepin/releases/latest) on your path, connect it to Claude Code in the current project.

```sh
claude mcp add --transport stdio --scope local cachepin -- cachepin mcp
claude mcp get cachepin
```

The status should say `Connected`. To try it from a checkout, ask Claude to call `diagnose_diff` with the JSON objects from [a.json](../examples/prefix-diff/a.json) and [b.json](../examples/prefix-diff/b.json). In this example the first differing tool is `tools[0]`. The local estimate reports 1,099 recoverable prefix tokens.

For a client that reads `mcpServers`, use the same executable and argument.

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

Available MCP tools:
- `optimize_prompt` — canonicalize + exact-DP breakpoint plan on an API request JSON
- `diagnose_diff` — AVX2 byte-level divergence forensics between two consecutive prompts
- `plan_breakpoints` — solve exact $O(M^2 K)$ mixed-TTL (`1h` / `5m`) breakpoints + keepalive break-even
- `audit_trace` — scan a JSONL trace file for root-cause misses & dollar savings
- `route_simulate` — evaluate multi-backend Shadow-KV routing decisions

---

## 4. Direct Python SDK (Zero Dependencies, Native Rust FFI)

```python
from cachepin import CachePin

cp = CachePin(mode="swarm")

# Wrap any Anthropic or OpenAI request dict before sending
optimized = cp.optimize_dict(request_payload, tenant="org-42", provider="anthropic")

# Or inspect full forensics + exact DP plan
report = cp.compile_and_plan(request_payload, tenant="org-42", provider="anthropic")
print(report["diagnostic"]["reason"], report["plan"]["breakpoints"])
```

---

## 5. Offline Trace Forensics (Zero Proxy Required)

Measure whether your agent swarm has a prompt-cache problem **before** deploying a proxy:

```bash
cachepin audit ./agent_requests.jsonl
cachepin audit ./agent_requests.jsonl --json > forensics.json
```

---

## 6. Stacking with Compression Tools

`cachepin` is **100% lossless** and strictly orthogonal to lossy prompt compressors:
- **`cachepin` + `caveman`**: Keep shared prefixes byte-frozen with `cachepin` while `caveman` shortens output prose.
- **`cachepin` + `context-mode`**: Sandbox bulky tool outputs with `context-mode` while `cachepin` canonicalizes schemas, aligns All-Gather blocks, and coalesces subagent cold starts.
