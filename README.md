# `cachepin`

[![CI](https://github.com/pgg898200-ship-it/cachepin/actions/workflows/ci.yml/badge.svg)](https://github.com/pgg898200-ship-it/cachepin/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/pgg898200-ship-it/cachepin?color=10b981)](https://github.com/pgg898200-ship-it/cachepin/releases/latest)
[![MCP Registry](https://img.shields.io/badge/MCP%20Registry-io.github.pgg898200--ship--it%2Fcachepin-38bdf8)](https://registry.modelcontextprotocol.io/v0.1/servers?search=cachepin)
[![Zero Dependencies](https://img.shields.io/badge/crates.io%20deps-0-34d399)](#verification--safety-guarantees)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-64748b)](LICENSE)

**Pin your prefix. Stop burning 15×–40× on broken multi-agent LLM prompt caches.**

> Zero-dependency Rust + AVX2 / ARM64 NEON multi-agent prompt-cache compiler, exact $O(M^2 K)$ mixed-TTL breakpoint optimizer, TTFT-triggered single-flight thundering-herd coalescer, HTTP/1.1 keep-alive proxy, and replication-aware Shadow-KV radix router.
> **~1.1 MB binary · 0 external crates · 100% lossless · works with Anthropic, OpenAI, vLLM, SGLang, Claude Code, Codex, and MCP.**

<p align="center">
  <img src="docs/assets/benchmarks.svg" alt="cachepin verified benchmark comparison: 84.2 µs Rust AVX2 compile (3.45x vs Node 22 V8, 22.3x vs Python 3.12), 42.56 GB/s SIMD LCP diff, 32 to 1 herd writes, and 15.39x swarm cost reduction" width="100%" />
</p>

Interactive browser playground & swarm billing simulator: **[pgg898200-ship-it.github.io/cachepin](https://pgg898200-ship-it.github.io/cachepin/)**

---

## Why Multi-Agent Prompt Caches Collapse

In late-2026 LLM economics, a cached token read costs **0.05×** on Anthropic Opus/Sonnet 5.5 and OpenAI GPT-5.6+ (**20× cheaper**), and **0.025×** on Anthropic Fable 5.1 (**40× cheaper**). Yet multi-agent coding and research swarms routinely fall to **2%–20% cache hit rates** because a single differing byte at token 50 invalidates the entire 50,000-token KV prefix after it:

1. **Non-deterministic tool & schema key order** across MCP servers and language runtimes.
2. **Volatile tokens injected early** (ISO-8601 timestamps, UUIDs, `req_*` / `trace_*` / `session_*` IDs).
3. **All-Gather collective prefix shattering** (*arXiv:2604.03143*): $N$ agents ingest the same shared repo files or peer summaries in different orders.
4. **Cold-start thundering herds**: 32 subagents spawn simultaneously on the same cold prefix and pay 32 redundant cache writes.
5. **Eviction-regime routing cliffs** on self-hosted vLLM/SGLang clusters: single-home prefix routers pile bursts onto one GPU and regress P99 TTFT when working sets exceed fleet KV capacity.

**`cachepin` fixes all five losslessly — without dropping or rephrasing a single token.** It stacks orthogonally on top of lossy prompt compressors (`caveman`, `context-mode`).

---

## 30-Second Quickstart

```bash
# Option A: Install prebuilt release binary (Linux/macOS x86_64 & ARM64, SHA-256 verified)
curl -fsSL https://raw.githubusercontent.com/pgg898200-ship-it/cachepin/main/scripts/install.sh | sh

# Option B: Homebrew / Nix / Docker / Offline Cargo build (0 external crates)
brew install --HEAD pgg898200-ship-it/cachepin/cachepin   # Homebrew tap
nix run github:pgg898200-ship-it/cachepin -- --version    # Nix Flake
docker run -i --rm ghcr.io/pgg898200-ship-it/cachepin:latest mcp
cargo build --release                                     # 100% offline, 0 crates
```

### Run Immediately (Zero Config Needed)

```bash
# 1. Audit your real local Claude Code (~/.claude/projects) & Codex (~/.codex/sessions) logs
cachepin audit --auto
cachepin audit --claude-code
cachepin audit --codex

# 2. Pinpoint the exact byte where two prompts shatter KV caching + counterfactual recovery
cachepin diff agent_a.json agent_b.json

# 3. Drop-in HTTP/1.1 keep-alive reverse proxy for Cloud APIs or multi-GPU vLLM/SGLang fleets
cachepin proxy --upstream https://api.anthropic.com --port 47820
ANTHROPIC_BASE_URL=http://127.0.0.1:47820 claude
# Live dark-mode telemetry: http://127.0.0.1:47820/dashboard  |  Prometheus: /metrics

# 4. Native stdio MCP server, Claude Code Plugin, Universal .mcpb Extension, or Agent Skill
cachepin mcp
# Claude Code Plugin: /plugin marketplace add pgg898200-ship-it/cachepin && /plugin install cachepin
# Cross-agent skill : npx skills add pgg898200-ship-it/cachepin
```

Or embed directly in Python (zero dependencies via `ctypes` + `libcachepin.so`):

```python
from cachepin import CachePin

with CachePin(mode="swarm") as cp:
    optimized = cp.optimize_dict(request_payload, tenant="team-alpha", provider="anthropic")
```

---

## Architecture

```mermaid
flowchart LR
    A["Multi-Agent Swarm<br/>(N concurrent agents)"] --> B["Pass 1-4 Prefix Compiler<br/>AVX2/NEON JSON + Tool Sort + Bijective Hoist + All-Gather"]
    B --> C["TTFT Single-Flight Coalescer<br/>1 Leader -> N-1 Followers wake on 200 OK headers"]
    C --> D["Tenant-Isolated Radix Tree<br/>+ Exact O(M^2 K) Mixed-TTL DP (1h before 5m)"]
    D --> E{"Upstream Mode"}
    E -->|Cloud API| F["Anthropic / OpenAI<br/>(Verified TLS + zero-buffer SSE + usage reconciliation)"]
    E -->|Self-Hosted Fleet| G["Shadow-KV Adaptive-Replication Router<br/>(1.16 us/route + HTTP/1.1 keep-alive pool)"]
    G --> H["vLLM / SGLang GPU Fleet"]
```

---

## Reproducible Benchmarks

Every number below is generated directly on this machine (`Intel Xeon, AVX2, 2 vCPUs`, `rustc 1.100`, `Node.js v22.23`, `Python 3.12`) and saved verbatim in [`benchmarks/results.json`](benchmarks/results.json). Reproduce everything with `./scripts/verify.sh`, or run standalone paper reproductions (`benchmarks/repro_all_gather_2604_03143.py`, `benchmarks/repro_thundering_herd.py`, `benchmarks/repro_vllm_eviction_cliff.py`).

- **`[LIVE-CPU]`**: Direct wall-clock measurements on live CPU execution & loopback TCP sockets.
- **`[SIM-TRACE]`**: Deterministic discrete-event workload & 8-GPU cluster replays under late-2026 provider pricing.

### 1. Cross-Language Matched Controls (`18.5 KiB` Multi-Agent Request) `[LIVE-CPU]`

| Runtime | Compile P50 | Compile P99 | LCP Diff Throughput | CLI Cold Start |
|---|---:|---:|---:|---:|
| **`cachepin` (Rust + AVX2)** | **84.2 µs** | **138.4 µs** | **42.56 GB/s** | **2.53 ms** |
| `cachepin` Python FFI (`libcachepin.so`) | 392.9 µs | 828.1 µs | **39.13 GB/s** | — |
| Node.js 22 (V8 JIT control) | 290.3 µs | 1,233.9 µs | 0.85 GB/s | 56.54 ms |
| Python 3.12 (stdlib control) | 1,876.3 µs | 2,096.4 µs | 0.02 GB/s | 35.96 ms |
| **Speedup (`cachepin` Rust vs Python / Node)** | **22.3× / 3.45×** | **15.1× / 8.91×** | **2,108× / 50.3×** | **14.2× / 22.4×** |

> Note: `cachepin` Rust executes **strictly more work** than the Node/Python controls (adding All-Gather frequency-bucket alignment, SipHash-2-4 tenant KDF, 18-block prefix hashing, SIMD BPE token estimation, and config-trap guard), yet beats Node.js 22 V8 by **3.45× at P50** and **8.91× at P99**. On ARM64 (Apple Silicon / AWS Graviton), `cachepin` automatically dispatches native **ARM64 NEON** (`simd: neon`), verified natively in CI on `ubuntu-24.04-arm`.

### 2. Core Native Kernels & Live Reverse Proxy `[LIVE-CPU]`

| Kernel / Live Measurement | Measured Value | Evidence |
|---|---:|---|
| AVX2 Longest-Common-Prefix Diff (`256 KiB` buffers) | **42.56 GB/s** | `[LIVE-CPU]` |
| Exact $O(M^2 K)$ Mixed-TTL DP Breakpoint Solver (16 blocks) | **8.04 µs** P50 / **27.85 µs** P99 | `[LIVE-CPU]` |
| Shadow-KV Adaptive-Replication Route Decision (8 backends) | **1.16 µs** P50 / **2.35 µs** P99 | `[LIVE-CPU]` |
| Live Loopback HTTP/1.1 Keep-Alive Proxy Added Overhead | **0.22 ms** P50 (`219.4 µs`) / **0.71 ms** P99 | `[LIVE-CPU]` |
| 32-Agent Live Cold-Start Thundering Herd (Cache Writes) | **32 → 1 writes (-96.88%)** | `[LIVE-CPU]` |

### 3. Multi-Agent Prefix Recovery & Exact DP Savings `[SIM-TRACE]`

| Workload (Anthropic Opus 5.5: `$4/$5/$8/$0.20` per MTok) | Raw Baseline | After `cachepin` | Improvement |
|---|---:|---:|---:|
| 32-Agent Coding Swarm (Tool/Key Drift + Volatile IDs + Docs) | 2.24% hit | **98.44% hit** | **-93.50% cost (15.39× cheaper)** |
| 16-Agent All-Gather Collective Rounds (*arXiv:2604.03143*) | 19.61% hit | **98.61% hit** | **-93.43% cost** |
| 64-Agent Branching Tree (Mixed `1h`/`5m` Exact DP) | Uncached / Naive 1-BP | **Exact DP** | **-88.51% vs uncached / -88.59% vs naive 1-BP** |

### 4. 8-GPU vLLM Eviction-Regime Routing (`48 Prefixes`, Zipf-1.25 Bursts) `[SIM-TRACE]`

<p align="center">
  <img src="docs/assets/vllm_routing.svg" alt="8-GPU vLLM Eviction-Regime Shadow-KV Routing comparison showing CachePin cutting P99 TTFT by 37.0% and Mean TTFT by 2.42x vs SingleHome-2xDivert" width="100%" />
</p>

| Routing Policy | Fleet KV Hit % | P50 TTFT | P99 TTFT | Mean TTFT | Auto-Replications |
|---|---:|---:|---:|---:|---:|
| Round-Robin | 51.1% | 136.6 ms | 443.0 ms | 137.3 ms | 366 |
| Least-Loaded | 52.6% | 106.0 ms | 376.0 ms | 119.1 ms | 355 |
| Consistent-Hash | 96.1% | 16.8 ms | 351.6 ms | 72.9 ms | 0 |
| SingleHome-2xDivert (Ranvier-style) | 66.9% | 14.4 ms | 413.0 ms | 89.7 ms | 180 |
| **`cachepin` Adaptive-Replication** | **87.5%** | **12.8 ms** | **260.0 ms** | **37.1 ms** | **66** |

`cachepin` cuts **P99 TTFT by 37.0% vs single-home 2×-divert** and **30.9% vs Least-Loaded**, and cuts **Mean TTFT by 2.42× vs single-home 2×-divert** and **3.21× vs Least-Loaded**, while deciding routes in **1.16 µs P50**.

---

## Comparison Against Existing Tools

| Capability | `cachepin` | `caveman` / `context-mode` | `claude-code-cache-fix` | `Ranvier` |
|---|---|---|---|---|
| **100% lossless** (zero eval degradation) | **Yes** (proven bijection) | No (lossy prose/output trim) | Yes | Yes |
| **Multi-agent All-Gather prefix alignment** | **Yes** | No | No | No |
| **Exact $O(M^2 K)$ mixed `1h`/`5m` DP solver** | **Yes** (enforces A/B/C rule) | No | No | No |
| **TTFT single-flight thundering-herd coalescer** | **Yes** (`32 → 1` writes) | No | No | No |
| **Cloud APIs (Anthropic/OpenAI) + vLLM/SGLang** | **Both** | Cloud/client | Anthropic only | vLLM only |
| **Eviction-regime adaptive KV replication** | **Yes** (`1.16 µs` route) | No | No | No (+eviction cliff) |
| **Runtime footprint & SIMD** | **~1.1 MB binary, AVX2 + NEON, 0 deps** | Node / plugin | Node proxy | C++20 Seastar + Docker |
| **Native MCP + Python FFI + CLI Diff & Audit** | **All built-in** | MCP/Skill | Proxy only | Router only |

---

## When You Do NOT Need `cachepin`

- **Short prompts (`< 512` tokens)**: Below provider minimum cacheable prefix thresholds (`cachepin` automatically passes these through unchanged with `below_min_tokens`).
- **Single-turn, zero-reuse one-off queries**: If a system prompt or context is never reused within 5–60 minutes, caching adds a 1.25× write surcharge without future read hits (our DP solver detects `expected_hits == 0` and places 0 breakpoints).
- **Single-agent workloads already at `>95%` cache hit rate**: Run `cachepin audit --auto` first — if your raw hit rate is already `>95%`, you don't need another proxy in your stack.

---

## Verification & Safety Guarantees

Run the complete verification gate (zero-dependency check, **64 Rust unit/property tests** under `-D warnings`, **12-vector adversarial red-team & live integration suite**, and cross-language matched-control benchmarks):

```bash
./scripts/verify.sh
```

Verified guarantees (tested in `tests/adversarial_suite.py`, `cargo test --release`, and native ARM64 NEON CI):
- **Prompt-injection sentinel safety**: Literal `⟦cp:v1⟧` / `⟦cp:esc:` strings in untrusted inputs are escaped bijectively (`reconstruct_lossless(hoist(s)) == s` for all UTF-8 strings).
- **Strict cross-tenant isolation**: Every block hash, radix namespace, and single-flight coalescer key is derived via per-tenant SipHash-2-4 KDF — zero cross-tenant cache sharing or probing.
- **RFC 8259 DoS resistance**: Strict `MAX_DEPTH = 128` parser guard, full UTF-16 surrogate pair validation, zero `unwrap()` in production paths, panic-safe `catch_unwind` FFI boundary.
- **Anthropic mixed-TTL compliance**: Emitted breakpoints strictly satisfy `<= 4` and all `1h` markers preceding all `5m` markers.
- **Fail-open RAII coalescing**: Leader errors, disconnects, or panics immediately wake parked followers (`Drop for LeaderGuard`).
- **Zero-buffering keep-alive pooling & provider usage reconciliation**: Plain HTTP upstreams reuse warm TCP sockets cleanly across requests while reconciling provider-reported `cache_read_input_tokens` / `cache_creation_input_tokens` / `cached_tokens` after client flush.

---

## Documentation & Registry Links

- [Cutstream](cutstream/) is a standalone output-side companion for gateways that need streaming stop strings. It provides a C++17 header, Python bindings and a reproducible latency comparison.

- Interactive Playground & Explorer: **[pgg898200-ship-it.github.io/cachepin](https://pgg898200-ship-it.github.io/cachepin/)**
- Official MCP Registry identifier: `mcp-name: io.github.pgg898200-ship-it/cachepin`
- [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md) — Formal mathematical specification of all 5 mechanisms
- [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md) — Drop-in setup for Claude Code, Codex, Anthropic, OpenAI, vLLM/SGLang, MCP, and Python SDK
- [`SKILL.md`](SKILL.md) — Agent-first declarative skill manifest (`npx skills add pgg898200-ship-it/cachepin`)
- [`llms.txt`](llms.txt) — Concise machine-readable repository index for AI coding agents

## License

Dual-licensed under MIT OR Apache-2.0. See [`LICENSE`](LICENSE).
