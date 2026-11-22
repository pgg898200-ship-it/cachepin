# `cachepin` Algorithms & Formal Invariants

This document specifies the five mathematical and systems mechanisms inside `cachepin`. Every invariant below is verified by executable unit tests (`cargo test --release`) and the adversarial fuzz suite (`tests/adversarial_suite.py`).

---

## 1. Bijective Lossless Prefix Canonicalization & Volatile Hoisting

Prefix-based KV caches (Anthropic, OpenAI, Gemini, vLLM APC, SGLang RadixAttention) are **causally prefix-sensitive**: mutating a single byte at token $i$ invalidates KV tensors for all tokens $j \ge i$.

`cachepin` applies four deterministic, semantics-preserving rewrites:

1. **Deterministic Schema & Tool Canonicalization (`ToolCanonicalize`):**
   - Recursively sorts all JSON object keys lexicographically without reordering JSON arrays.
   - Deterministically sorts `tools[]` by canonical tool name (`name` or `function.name`), eliminating Python/JS/MCP registration-order drift.
2. **Bijective Volatile-Span Hoisting (`VolatileHoist`):**
   - Detects volatile tokens in early prefix blocks (`system`, `instructions`, prior turns): ISO-8601/RFC-3339 timestamps, UUIDv4/v7, and opaque run/request/trace IDs (`req_*`, `trace_*`, `span_*`, `session_*`, `run_*`, `call_*`, `agent_run_*`).
   - **Prompt-injection-safe escaping:** Any literal sentinel prefix `⟦cp:` occurring in untrusted input is first escaped to `⟦cp:esc:` before substituting volatile spans with indexed anchors `⟦cp:v1⟧`, `⟦cp:v2⟧`, ....
   - Appends a compact trailing binding block strictly after the final cache breakpoint (`breakpoint_eligible = false`).
   - **Exact Recovery Theorem:** For every UTF-8 string $s$,
     $$\text{reconstruct}(\text{hoist}(s)) = s$$
     byte-for-byte, including adversarial inputs containing literal `⟦cp:v1⟧` or nested `⟦cp:esc:` strings.
3. **All-Gather Collective Prefix Alignment (`SwarmAlign`):**
   - In multi-agent rounds where $N$ agents ingest overlapping shared context blocks (documents, repo files, peer summaries) in arbitrary orders, `cachepin` maintains a tenant-isolated exponentially-decayed frequency table over `(tenant_hmac, block_hash)`.
   - Reorderable sibling context blocks are stably sorted by `(descending swarm frequency bucket, canonical block_hash)`, converging all $N$ agents onto one shared byte-identical prefix while strictly preserving conversational `messages[]` turn order.
4. **Tenant Isolation (HMAC-Keyed SipHash-1-3):**
   - Every tenant's API key / tenant header derives an independent 128-bit key $K_{\text{tenant}}$ via SipHash-2-4 KDF.
   - All block hashes, radix paths, and single-flight coalescer keys are keyed by $K_{\text{tenant}}$, preventing cross-tenant cache hits, timing probes, and dictionary attacks.

---

## 2. Exact $O(M^2 K)$ Mixed-TTL Breakpoint Dynamic Program

Given $M$ prefix blocks with cumulative token counts $L_1 < L_2 < \dots < L_M$, estimated future reuse counts $(h_i^{(5\text{m})}, h_i^{(1\text{h})})$ from the live radix tree, model pricing $(c_{\text{in}}, c_{5\text{m}}, c_{1\text{h}}, c_{\text{read}})$, minimum cacheable prefix $T_{\min}$, and maximum breakpoints $K \le 4$:

### Provider Constraints Modeled Exactly
- **Eligibility:** Breakpoints may only be placed on blocks with `breakpoint_eligible == true` and $L_b \ge T_{\min}$.
- **Anthropic Mixed-TTL Monotonicity (A/B/C Rule):** All `1h` breakpoints MUST precede all `5m` breakpoints. Placing a `1h` marker after a `5m` marker triggers API rejection or full-prefix rewrite.
- **Warm-Prefix Awareness:** If prefix $[0..i]$ is already warm at TTL $\tau$, its incremental write cost is $0$; otherwise the write surcharge applies strictly to newly written tokens $\Delta L = L_i - L_{\text{prev}}$.

### Dynamic Program Recurrence
Let $DP[k][i][t]$ denote the minimum total expected USD cost covering blocks $0 \dots i$ using exactly $k \in \{1, \dots, K\}$ breakpoints, with the $k$-th breakpoint at block $i$ carrying TTL tier $t \in \{0=\text{1h},\, 1=\text{5m}\}$:

$$DP[k][i][t] = \min_{\substack{j < i \\ t' \le t}} \Big( DP[k-1][j][t'] + \text{SegmentCost}(j \to i, t) \Big)$$

The transition constraint $t' \le t$ mechanically enforces `1h`-before-`5m` monotonicity (`0→0`, `0→1`, `1→1` permitted; `1→0` forbidden). The optimal total cost adds the uncached tail cost over $(i, M)$:

$$C^* = \min\Big(C_{\text{uncached}},\, \min_{k, i, t} \big(DP[k][i][t] + \text{TailCost}(i \to M)\big)\Big)$$

### Keepalive Break-Even Rule
When inter-arrival gap $g \in (300\text{s}, 3600\text{s})$, `cachepin` compares the 1-hour write premium against periodic zero-output (`max_tokens: 0`) 5-minute keepalive pings at 270 s intervals:
$$N_{\text{pings}} = \left\lceil \frac{g}{270} \right\rceil, \qquad \text{PingCost} = N_{\text{pings}} \cdot L_{\text{warm}} \cdot c_{\text{read}}$$
recommending keepalive pings iff $\text{PingCost} < \Delta C_{1\text{h}-5\text{m}}$ and $\text{PingCost} < \text{RewritePenalty}$.

---

## 3. TTFT-Triggered Single-Flight Thundering-Herd Coalescing

When $N$ subagents launch simultaneously on an identical cold prefix $P$ ($L \ge T_{\min}$):
- Naive concurrency dispatches all $N$ requests cold, paying $N \times c_{\text{write}}$ and saturating upstream prefill queues.
- Because KV-cache entries commit at the **end of prefill** (when upstream response headers / first SSE chunk arrive — TTFT, ~0.3–1.0 s), not at the end of decode (~10–30 s), `cachepin` elects **1 Leader** and parks $N - 1$ **Followers** on a per-prefix `Condvar` bounded by `coalesce_max_wait_ms`.
- The instant the Leader receives HTTP `200 OK` headers, `LeaderGuard::mark_warm_and_release()` wakes all $N-1$ followers simultaneously.
- **Fail-open RAII guarantee:** If the Leader errors, disconnects, or panics, `Drop for LeaderGuard` immediately wakes all followers (`leader_warmed = false`) so swarms never deadlock.

---

## 4. Replication-Aware Shadow-KV Adaptive Radix Routing

Single-home prefix routers (e.g., hash/ART-pinned routing with fixed 2× load diversion) suffer a documented **eviction-regime tail cliff**: when the active working set exceeds fleet KV capacity or Zipf-hot prefixes saturate one replica, requests serialize behind a single holder's prefill queue.

`cachepin`'s `ShadowKvRouter` maintains per backend $b$:
1. A bounded **Shadow KV-Cache LRU** tracking resident prefix blocks up to exact token capacity $C_{\text{kv}}$.
2. Live in-flight prefill-token queue depth $Q_b(t)$ and EWMA service rate.

For each arriving request $r$, it computes predicted TTFT across every backend $b$:
$$\widehat{\text{TTFT}}(r, b) = Q_b(t) + \frac{L(r) - L_{\text{hit}}(r, b)}{\text{prefill\_tps}_b} + \frac{L_{\text{hit}}(r, b)}{\text{read\_tps}_b}$$

- **Cool prefixes** stay pinned to a single warm home to avoid duplicating cold KV blocks and wasting fleet KV capacity.
- **Hot prefixes** whose primary holder queue exceeds the amortized replication cost automatically replicate onto the lowest-pressure backend with maximum cold-KV headroom, eliminating single-holder tail explosions while keeping fleet KV hit rates high.

---

## 5. SIMD AVX2 Vectorized Prefix Forensics

`src/simd.rs` implements runtime-dispatched x86-64 AVX2 (`_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` + trailing-ones count) scanning 32 bytes per iteration with an 8-byte `u64` portable fallback for ARM64/non-AVX2 targets. Combined with block-level radix hashing, cache-miss byte-offset divergence detection executes at multi-GB/s memory bandwidth speeds.
