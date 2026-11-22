//! Algebraic Multi-Agent Prefix Compiler for `cachepin`.
//!
//! Executes four composable, lossless canonicalization passes over LLM API
//! payloads (Anthropic Messages, OpenAI Chat, OpenAI Responses, vLLM):
//!
//! 1. **Pass 1 (`ToolCanonicalize`)**: Deterministic tool sorting and recursive
//!    RFC-8259 JSON key canonicalization.
//! 2. **Pass 2 (`VolatileHoist`)**: Prompt-injection-safe, strictly bijective
//!    hoisting of volatile timestamps, UUIDs, prefixed opaque IDs, and inline
//!    clock fields to a non-cacheable trailing binding block.
//! 3. **Pass 3 (`AllGatherAlign`)**: Frequency-bucketed All-Gather shared-block
//!    alignment solving multi-agent collective prefix collapse without ever
//!    reordering sequential conversation turns or tool call/result pairs.
//! 4. **Pass 4 (`ConfigGuard`)**: Cache-invalidating configuration fingerprinting
//!    and static detection of silent prompt-cache traps.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;

use crate::json::{
    estimate_serialized_len, parse_json, serialize_canonical, serialize_canonical_skip_top_key,
    serialize_json, serialize_json_skip_top_key, write_escaped_key_colon, JsonValue,
};
use crate::simd::{
    chain_hash64, derive_tenant_key, estimate_tokens_fast, find_any3_byte, keyed_hash64,
    scan_any3_byte, Any3Visitor, KeyedHasher,
};
use crate::types::{
    BreakpointPlan, CacheTtl, CanonicalMode, CompileResult, HoistedBinding, PlannedBreakpoint,
    PrefixBlock, ProviderFormat,
};

/// Sentinel prefix reserved for hoisted volatile anchors (`⟦cp:v1⟧`, ...).
pub const SENTINEL_PREFIX: &str = "⟦cp:";
/// Escaped form of any pre-existing literal `⟦cp:` present in untrusted input.
pub const SENTINEL_ESCAPED: &str = "⟦cp:esc:";
/// Closing delimiter of a hoisted volatile anchor.
pub const SENTINEL_SUFFIX: &str = "⟧";

const MAX_SWARM_ENTRIES: usize = 65_536;
const SWARM_EVICT_TARGET: usize = 32_768;
const SWARM_DECAY_TAU_SECS: f64 = 300.0;

#[inline]
fn push_usize(out: &mut String, mut n: usize) {
    let mut buf = [0u8; 20];
    let mut idx = buf.len();
    loop {
        idx -= 1;
        buf[idx] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    // SAFETY: All written bytes are ASCII digits `'0'..='9'`.
    let s = unsafe { std::str::from_utf8_unchecked(&buf[idx..]) };
    out.push_str(s);
}

#[inline]
fn fmt_indexed(prefix: &str, idx: usize, suffix: &str) -> String {
    let mut s = String::with_capacity(prefix.len() + suffix.len() + 6);
    s.push_str(prefix);
    push_usize(&mut s, idx);
    s.push_str(suffix);
    s
}

#[inline]
fn fmt_msg_part(prefix: &str, msg_idx: usize, mid: &str, part_idx: usize) -> String {
    let mut s = String::with_capacity(prefix.len() + mid.len() + 12);
    s.push_str(prefix);
    push_usize(&mut s, msg_idx);
    s.push_str(mid);
    push_usize(&mut s, part_idx);
    s.push(']');
    s
}

/// SIMD-accelerated search for the 6-byte UTF-8 `SENTINEL_PREFIX` (`"⟦cp:"`) inside `bytes`.
#[inline]
fn find_sentinel_bytes(bytes: &[u8]) -> Option<usize> {
    const NEEDLE: &[u8] = SENTINEL_PREFIX.as_bytes();
    let len = bytes.len();
    let mut pos = 0usize;
    while pos + NEEDLE.len() <= len {
        let rel = find_any3_byte(&bytes[pos..], 0xE2, 0xE2, 0xE2);
        pos += rel;
        if pos + NEEDLE.len() > len {
            return None;
        }
        if &bytes[pos..pos + NEEDLE.len()] == NEEDLE {
            return Some(pos);
        }
        pos += 1;
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct SwarmEntry {
    freq: f64,
    last_seen: Instant,
    last_tick: u64,
}

#[derive(Debug)]
struct SwarmStatsInner {
    entries: HashMap<(u64, u64), SwarmEntry>,
    tick: u64,
}

/// Thread-safe exponentially-decayed `(tenant_hash, block_hash) -> frequency`
/// table powering All-Gather shared-block hoisting across concurrent agents.
///
/// Strictly bounded to 65,536 entries with half-decay + LRU eviction.
#[derive(Debug)]
pub struct SwarmBlockStats {
    inner: RwLock<SwarmStatsInner>,
}

impl Default for SwarmBlockStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmBlockStats {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(SwarmStatsInner {
                entries: HashMap::new(),
                tick: 0,
            }),
        }
    }

    /// Records an observation of `(tenant_hash, block_hash)` and returns its
    /// updated exponentially-decayed swarm frequency score.
    pub fn record_and_score(&self, tenant_hash: u64, block_hash: u64) -> f64 {
        let now = Instant::now();
        let mut state = self.inner.write().unwrap_or_else(|e| e.into_inner());
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        let key = (tenant_hash, block_hash);

        if let Some(entry) = state.entries.get_mut(&key) {
            let dt = now.saturating_duration_since(entry.last_seen).as_secs_f64();
            let decayed = entry.freq * (-dt / SWARM_DECAY_TAU_SECS).exp();
            entry.freq = decayed + 1.0;
            entry.last_seen = now;
            entry.last_tick = tick;
            return entry.freq;
        }

        if state.entries.len() >= MAX_SWARM_ENTRIES {
            Self::evict_half_decay(&mut state, now);
        }

        state.entries.insert(
            key,
            SwarmEntry {
                freq: 1.0,
                last_seen: now,
                last_tick: tick,
            },
        );
        1.0
    }

    /// Batched version of [`Self::record_and_score`] taking the write lock and
    /// reading the clock once for an entire All-Gather block slice.
    pub fn record_batch_and_score(&self, tenant_hash: u64, hashes: &[u64]) -> Vec<f64> {
        let now = Instant::now();
        let mut state = self.inner.write().unwrap_or_else(|e| e.into_inner());
        let mut scores = Vec::with_capacity(hashes.len());
        for &block_hash in hashes {
            state.tick = state.tick.wrapping_add(1);
            let tick = state.tick;
            let key = (tenant_hash, block_hash);
            if let Some(entry) = state.entries.get_mut(&key) {
                let dt = now.saturating_duration_since(entry.last_seen).as_secs_f64();
                let decayed = entry.freq * (-dt / SWARM_DECAY_TAU_SECS).exp();
                entry.freq = decayed + 1.0;
                entry.last_seen = now;
                entry.last_tick = tick;
                scores.push(entry.freq);
            } else {
                if state.entries.len() >= MAX_SWARM_ENTRIES {
                    Self::evict_half_decay(&mut state, now);
                }
                state.entries.insert(
                    key,
                    SwarmEntry {
                        freq: 1.0,
                        last_seen: now,
                        last_tick: tick,
                    },
                );
                scores.push(1.0);
            }
        }
        scores
    }

    /// Returns the current decayed swarm frequency score without mutating state.
    pub fn get_score(&self, tenant_hash: u64, block_hash: u64) -> f64 {
        let now = Instant::now();
        let state = self.inner.read().unwrap_or_else(|e| e.into_inner());
        match state.entries.get(&(tenant_hash, block_hash)) {
            Some(entry) => {
                let dt = now.saturating_duration_since(entry.last_seen).as_secs_f64();
                entry.freq * (-dt / SWARM_DECAY_TAU_SECS).exp()
            }
            None => 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap_or_else(|e| e.into_inner()).entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        let mut state = self.inner.write().unwrap_or_else(|e| e.into_inner());
        state.entries.clear();
        state.tick = 0;
    }

    fn evict_half_decay(state: &mut SwarmStatsInner, now: Instant) {
        for entry in state.entries.values_mut() {
            let dt = now.saturating_duration_since(entry.last_seen).as_secs_f64();
            entry.freq = entry.freq * (-dt / SWARM_DECAY_TAU_SECS).exp() * 0.5;
            entry.last_seen = now;
        }

        if state.entries.len() <= SWARM_EVICT_TARGET {
            return;
        }

        let mut ranked: Vec<((u64, u64), f64, u64)> = state
            .entries
            .iter()
            .map(|(&k, e)| (k, e.freq, e.last_tick))
            .collect();
        ranked.sort_by(|a, b| {
            a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2))
        });

        let to_remove = state.entries.len().saturating_sub(SWARM_EVICT_TARGET);
        for (k, _, _) in ranked.into_iter().take(to_remove) {
            state.entries.remove(&k);
        }
    }
}

#[inline]
fn swarm_freq_bucket(score: f64) -> u32 {
    if !score.is_finite() || score < 1.5 {
        0
    } else {
        let ratio = score / 1.5;
        (ratio.log2().floor().max(0.0) as u32).saturating_add(1)
    }
}

/// Configuration for [`PrefixCompiler`].
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub mode: CanonicalMode,
    pub tenant_secret: [u8; 16],
    pub max_volatile_per_block: usize,
    pub all_gather_delimiter: Option<String>,
    pub default_provider: ProviderFormat,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            mode: CanonicalMode::SwarmAlign,
            tenant_secret: *b"cachepin-dflt-v1",
            max_volatile_per_block: 64,
            all_gather_delimiter: None,
            default_provider: ProviderFormat::OpenAiChat,
        }
    }
}

/// Stateful, thread-safe algebraic prefix compiler.
pub struct PrefixCompiler {
    config: CompilerConfig,
    swarm_stats: SwarmBlockStats,
}

impl Default for PrefixCompiler {
    fn default() -> Self {
        Self::new(CompilerConfig::default())
    }
}

impl PrefixCompiler {
    pub fn new(config: CompilerConfig) -> Self {
        Self {
            config,
            swarm_stats: SwarmBlockStats::new(),
        }
    }

    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }

    pub fn swarm_stats(&self) -> &SwarmBlockStats {
        &self.swarm_stats
    }

    /// Compiles and canonicalizes a raw LLM API JSON payload for `tenant_id`.
    #[inline]
    pub fn compile_request(
        &self,
        tenant_id: &str,
        raw_json: &[u8],
        provider_hint: Option<ProviderFormat>,
    ) -> Result<CompileResult, String> {
        self.compile_request_with_ast(tenant_id, raw_json, provider_hint, true)
            .map(|(res, _)| res)
    }

    /// Compiles and canonicalizes a raw LLM API JSON payload for `tenant_id`,
    /// returning both the [`CompileResult`] and the canonicalized [`JsonValue`] AST.
    ///
    /// When `emit_canonical_json` is `false`, `CompileResult::canonical_json` is
    /// left empty so callers can inject breakpoints directly onto the returned
    /// `JsonValue` via [`apply_breakpoints_to_value`] and serialize once with
    /// zero re-parsing.
    pub fn compile_request_with_ast(
        &self,
        tenant_id: &str,
        raw_json: &[u8],
        provider_hint: Option<ProviderFormat>,
        emit_canonical_json: bool,
    ) -> Result<(CompileResult, JsonValue), String> {
        let start = Instant::now();
        let mut root = parse_json(raw_json)?;
        if root.as_object().is_none() {
            return Err("request payload root must be a JSON object".to_string());
        }

        let provider = provider_hint.unwrap_or_else(|| {
            detect_provider(&root, self.config.default_provider)
        });
        let model = root
            .get("model")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .to_string();

        let tenant_key = derive_tenant_key(&self.config.tenant_secret, tenant_id);
        let tenant_hash = keyed_hash64(&tenant_key, tenant_id.as_bytes());

        // Pass 4 diagnostics inspect config and existing cache_control markers
        // before any mutations mask client misconfigurations.
        let config_fingerprint = compute_config_fingerprint(&root, &tenant_key);
        let config_warnings = detect_config_warnings(&root, provider);

        if self.config.mode == CanonicalMode::ObserveOnly {
            let (blocks, _) = extract_blocks(&root, provider, &tenant_key, None, false, None);
            let (shared_prefix_hash, full_prefix_hash, total_est_tokens) =
                summarize_blocks(&blocks, raw_json);
            let canonical_json = if emit_canonical_json {
                raw_json.to_vec()
            } else {
                Vec::new()
            };
            let res = CompileResult {
                provider,
                model,
                canonical_json,
                blocks,
                bindings: Vec::new(),
                tools_reordered: false,
                json_keys_reordered: false,
                all_gather_reordered: false,
                volatile_spans_hoisted: 0,
                total_est_tokens,
                shared_prefix_hash,
                full_prefix_hash,
                config_fingerprint,
                config_warnings,
                compile_ns: elapsed_ns(start),
            };
            return Ok((res, root));
        }

        // Pass 1: JSON key canonicalization + deterministic tool sorting.
        let json_keys_reordered = root.canonicalize_in_place();
        let tools_reordered = canonicalize_tools(&mut root);

        // Pass 2 (Stage A): Lossless volatile hoisting with sequential temporary IDs so
        // Pass 3 can match semantically identical blocks even if their volatile
        // tokens appeared in different initial orders across swarm agents.
        let mut temp_bindings: Vec<(&'static str, String)> = Vec::new();
        if matches!(
            self.config.mode,
            CanonicalMode::HoistVolatile | CanonicalMode::SwarmAlign
        ) {
            hoist_volatile_pass(
                &mut root,
                provider,
                self.config.max_volatile_per_block,
                &mut temp_bindings,
            );
        }

        // Pass 3: All-Gather shared-block alignment across multi-agent swarms.
        let (all_gather_reordered, cached_system_blocks) =
            if self.config.mode == CanonicalMode::SwarmAlign {
                all_gather_align_pass(
                    &mut root,
                    provider,
                    &tenant_key,
                    tenant_hash,
                    &self.swarm_stats,
                    self.config.all_gather_delimiter.as_deref(),
                    !temp_bindings.is_empty(),
                )
            } else {
                (false, None)
            };

        // Pass 2 (Stage B): Renumber temporary anchors into canonical final
        // `⟦cp:v1⟧`, `⟦cp:v2⟧`, ... in post-alignment prefix order, then append
        // the compact trailing binding block strictly at the end of the request.
        let mut bindings = Vec::new();
        if !temp_bindings.is_empty() {
            bindings.reserve(temp_bindings.len());
            finalize_hoisted_anchors(&mut root, provider, &temp_bindings, &mut bindings);
        }
        let volatile_spans_hoisted = bindings.len();

        let volatile_tail_loc = if !bindings.is_empty() {
            let summary = format_binding_summary(&bindings);
            Some(append_volatile_tail(&mut root, provider, summary))
        } else {
            None
        };

        let (blocks, block_ranges) = extract_blocks(
            &root,
            provider,
            &tenant_key,
            volatile_tail_loc,
            true,
            cached_system_blocks,
        );

        // `root` is strictly sorted after Pass 1 + `set_field_sorted` mutations.
        // Emit canonical JSON by copying already-serialized block bytes for `tools`,
        // `system`, and `instructions` instead of re-serializing them from scratch.
        let canonical_json = if emit_canonical_json {
            let est = blocks
                .iter()
                .map(|b| b.canonical_bytes.len() + 2)
                .sum::<usize>()
                + 512;
            let mut out = Vec::with_capacity(est);
            serialize_root_with_block_cache(&root, &blocks, block_ranges, &mut out);
            out
        } else {
            Vec::new()
        };

        let fallback_slice = if emit_canonical_json {
            canonical_json.as_slice()
        } else {
            raw_json
        };
        let (shared_prefix_hash, full_prefix_hash, total_est_tokens) =
            summarize_blocks(&blocks, fallback_slice);

        let res = CompileResult {
            provider,
            model,
            canonical_json,
            blocks,
            bindings,
            tools_reordered,
            json_keys_reordered,
            all_gather_reordered,
            volatile_spans_hoisted,
            total_est_tokens,
            shared_prefix_hash,
            full_prefix_hash,
            config_fingerprint,
            config_warnings,
            compile_ns: elapsed_ns(start),
        };
        Ok((res, root))
    }
}

#[inline]
fn elapsed_ns(start: Instant) -> u64 {
    let ns = start.elapsed().as_nanos();
    u64::try_from(ns).unwrap_or(u64::MAX)
}

/// Auto-detects wire format when no `provider_hint` is provided.
fn detect_provider(root: &JsonValue, default_provider: ProviderFormat) -> ProviderFormat {
    if root.get("anthropic_version").is_some()
        || root.get("system").is_some()
        || root.get("thinking").is_some()
    {
        return ProviderFormat::AnthropicMessages;
    }

    if root.get("input").is_some()
        && (root.get("instructions").is_some() || root.get("messages").is_none())
    {
        return ProviderFormat::OpenAiResponses;
    }

    if let Some(tools) = root.get("tools").and_then(JsonValue::as_array) {
        if tools.iter().any(|t| t.get("input_schema").is_some()) {
            return ProviderFormat::AnthropicMessages;
        }
    }

    if let Some(messages) = root.get("messages").and_then(JsonValue::as_array) {
        let has_max_tokens = root.get("max_tokens").is_some();
        let model_is_claude = root
            .get("model")
            .and_then(JsonValue::as_str)
            .map(|m| m.to_ascii_lowercase().starts_with("claude"))
            .unwrap_or(false);

        let has_anthropic_blocks = messages.iter().any(|m| {
            if m.get("cache_control").is_some() {
                return true;
            }
            if let Some(parts) = m.get("content").and_then(JsonValue::as_array) {
                return parts.iter().any(|p| {
                    if p.get("cache_control").is_some() {
                        return true;
                    }
                    match p.get("type").and_then(JsonValue::as_str) {
                        Some("tool_use" | "tool_result" | "thinking" | "redacted_thinking" | "document") => true,
                        Some("text") => has_max_tokens && model_is_claude,
                        _ => false,
                    }
                });
            }
            false
        });

        if has_anthropic_blocks || (has_max_tokens && model_is_claude) {
            return ProviderFormat::AnthropicMessages;
        }
        return ProviderFormat::OpenAiChat;
    }

    default_provider
}

// ============================================================================
// Pass 1: ToolCanonicalize
// ============================================================================

fn extract_tool_name(tool: &JsonValue) -> &str {
    if let Some(name) = tool.get("name").and_then(JsonValue::as_str) {
        return name;
    }
    if let Some(name) = tool
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(JsonValue::as_str)
    {
        return name;
    }
    ""
}

fn canonicalize_tools(root: &mut JsonValue) -> bool {
    let tools = match root.get_mut("tools").and_then(JsonValue::as_array_mut) {
        Some(t) if t.len() >= 2 => t,
        _ => return false,
    };

    // Fast check: already strictly sorted by distinct tool name?
    let mut already_strictly_sorted = true;
    for w in tools.windows(2) {
        if extract_tool_name(&w[0]) >= extract_tool_name(&w[1]) {
            already_strictly_sorted = false;
            break;
        }
    }
    if already_strictly_sorted {
        return false;
    }

    // Collect indices + check if all names are distinct after sorting by name.
    let mut order: Vec<usize> = (0..tools.len()).collect();
    order.sort_by(|&a, &b| extract_tool_name(&tools[a]).cmp(extract_tool_name(&tools[b])));

    let has_dup_name = order
        .windows(2)
        .any(|w| extract_tool_name(&tools[w[0]]) == extract_tool_name(&tools[w[1]]));

    if has_dup_name {
        let mut keyed: Vec<(String, Vec<u8>, usize)> = Vec::with_capacity(tools.len());
        for (idx, tool) in tools.iter().enumerate() {
            let name = extract_tool_name(tool).to_string();
            let mut bytes = Vec::with_capacity(128);
            serialize_json(tool, &mut bytes);
            keyed.push((name, bytes, idx));
        }
        let already_sorted = keyed
            .windows(2)
            .all(|w| (&w[0].0, &w[0].1) <= (&w[1].0, &w[1].1));
        if already_sorted {
            return false;
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut old: Vec<Option<JsonValue>> = tools.drain(..).map(Some).collect();
        for (_, _, orig_idx) in keyed {
            if let Some(item) = old[orig_idx].take() {
                tools.push(item);
            }
        }
        return true;
    }

    if order.iter().enumerate().all(|(i, &orig)| i == orig) {
        return false;
    }
    let mut old: Vec<Option<JsonValue>> = tools.drain(..).map(Some).collect();
    for orig_idx in order {
        if let Some(item) = old[orig_idx].take() {
            tools.push(item);
        }
    }
    true
}

// ============================================================================
// Pass 2: VolatileHoist (Lossless & Prompt-Injection Safe)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VolatileTailLoc {
    Message(usize),
    MessagePart(usize, usize),
    InputItem(usize),
}

/// Restores EVERY byte of the original text from `canonical_text` and its
/// corresponding `bindings`, safely unescaping any pre-existing literal `⟦cp:`
/// sentinels without ever allowing spoofed anchors.
pub fn reconstruct_lossless(canonical_text: &str, bindings: &[HoistedBinding]) -> String {
    if !canonical_text.contains(SENTINEL_PREFIX) {
        return canonical_text.to_string();
    }

    let mut map: HashMap<&str, &str> = HashMap::with_capacity(bindings.len());
    for b in bindings {
        map.insert(b.anchor.as_str(), b.original_value.as_str());
    }

    let mut out = String::with_capacity(canonical_text.len());
    let mut rest = canonical_text;

    while let Some(pos) = rest.find(SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];

        if let Some(tail) = after.strip_prefix(SENTINEL_ESCAPED) {
            out.push_str(SENTINEL_PREFIX);
            rest = tail;
            continue;
        }

        if let Some(after_prefix) = after.strip_prefix(SENTINEL_PREFIX) {
            if let Some(after_v) = after_prefix.strip_prefix('v') {
                let digit_len = after_v
                    .bytes()
                    .take_while(|b| b.is_ascii_digit())
                    .count();
                if digit_len > 0 && after_v[digit_len..].starts_with(SENTINEL_SUFFIX) {
                    let full_len =
                        SENTINEL_PREFIX.len() + 1 + digit_len + SENTINEL_SUFFIX.len();
                    let anchor = &after[..full_len];
                    if let Some(&orig) = map.get(anchor) {
                        out.push_str(orig);
                        rest = &after[full_len..];
                        continue;
                    }
                }
            }
        }

        out.push_str(SENTINEL_PREFIX);
        rest = &after[SENTINEL_PREFIX.len()..];
    }

    out.push_str(rest);
    out
}

fn escape_sentinels_into(segment: &str, out: &mut String) {
    if find_sentinel_bytes(segment.as_bytes()).is_none() {
        out.push_str(segment);
        return;
    }
    let mut rest = segment;
    while let Some(pos) = find_sentinel_bytes(rest.as_bytes()) {
        out.push_str(&rest[..pos]);
        out.push_str(SENTINEL_ESCAPED);
        rest = &rest[pos + SENTINEL_PREFIX.len()..];
    }
    out.push_str(rest);
}

fn hoist_text_stage_a(
    text: &mut String,
    max_spans: usize,
    temp_bindings: &mut Vec<(&'static str, String)>,
) {
    // Every volatile pattern AND `SENTINEL_PREFIX` ("⟦cp:") contains '-', '_', or ':'.
    // If none exist, return immediately in O(len/32) with zero allocations.
    let first_hit = find_any3_byte(text.as_bytes(), b'-', b'_', b':');
    if first_hit == text.len() {
        return;
    }

    let (spans, has_sentinel) = find_volatile_spans_and_sentinel_from(text, max_spans, first_hit);
    if spans.is_empty() && !has_sentinel {
        return;
    }

    let need_escape = has_sentinel || spans.len() >= max_spans;
    let mut out = String::with_capacity(text.len() + spans.len() * 8 + 16);
    let mut cursor = 0usize;

    for (start, end, kind) in spans {
        if start < cursor
            || end > text.len()
            || !text.is_char_boundary(start)
            || !text.is_char_boundary(end)
        {
            continue;
        }
        if need_escape {
            escape_sentinels_into(&text[cursor..start], &mut out);
        } else {
            out.push_str(&text[cursor..start]);
        }
        temp_bindings.push((kind, text[start..end].to_string()));
        let tid = temp_bindings.len();
        out.push_str(SENTINEL_PREFIX);
        out.push('t');
        push_usize(&mut out, tid);
        out.push_str(SENTINEL_SUFFIX);
        cursor = end;
    }

    if need_escape {
        escape_sentinels_into(&text[cursor..], &mut out);
    } else {
        out.push_str(&text[cursor..]);
    }
    *text = out;
}

fn hoist_volatile_pass(
    root: &mut JsonValue,
    provider: ProviderFormat,
    max_per_block: usize,
    temp_bindings: &mut Vec<(&'static str, String)>,
) {
    // 1. Anthropic `system`
    if let Some(sys) = root.get_mut("system") {
        match sys {
            JsonValue::String(s) => {
                hoist_text_stage_a(s, max_per_block, temp_bindings);
            }
            JsonValue::Array(items) => {
                for item in items.iter_mut() {
                    if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                        if let Some(JsonValue::String(s)) = item.get_mut("text") {
                            hoist_text_stage_a(s, max_per_block, temp_bindings);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // 2. OpenAI Responses `instructions`
    if let Some(JsonValue::String(s)) = root.get_mut("instructions") {
        hoist_text_stage_a(s, max_per_block, temp_bindings);
    }

    // 3. Early `messages` (excluding the final user turn)
    if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
        let scan_limit = prefix_message_scan_limit(msgs);
        for msg in msgs.iter_mut().take(scan_limit) {
            if let Some(content) = msg.get_mut("content") {
                match content {
                    JsonValue::String(s) => {
                        hoist_text_stage_a(s, max_per_block, temp_bindings);
                    }
                    JsonValue::Array(parts) => {
                        for part in parts.iter_mut() {
                            let is_text = matches!(
                                part.get("type").and_then(JsonValue::as_str),
                                Some("text" | "input_text")
                            );
                            if is_text {
                                if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                    hoist_text_stage_a(s, max_per_block, temp_bindings);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 4. OpenAI Responses early `input` array items (excluding final user turn)
    if provider == ProviderFormat::OpenAiResponses {
        if let Some(items) = root.get_mut("input").and_then(JsonValue::as_array_mut) {
            let scan_limit = prefix_message_scan_limit(items);
            for item in items.iter_mut().take(scan_limit) {
                if let Some(content) = item.get_mut("content") {
                    match content {
                        JsonValue::String(s) => {
                            hoist_text_stage_a(s, max_per_block, temp_bindings);
                        }
                        JsonValue::Array(parts) => {
                            for part in parts.iter_mut() {
                                let is_text = matches!(
                                    part.get("type").and_then(JsonValue::as_str),
                                    Some("text" | "input_text")
                                );
                                if is_text {
                                    if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                        hoist_text_stage_a(s, max_per_block, temp_bindings);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn prefix_message_scan_limit(msgs: &[JsonValue]) -> usize {
    if let Some(last) = msgs.last() {
        if last.get("role").and_then(JsonValue::as_str) == Some("user") {
            return msgs.len().saturating_sub(1);
        }
    }
    msgs.len()
}

#[inline]
fn format_usize_buf(mut n: usize, buf: &mut [u8; 20]) -> &[u8] {
    let mut idx = buf.len();
    loop {
        idx -= 1;
        buf[idx] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    &buf[idx..]
}

#[inline]
fn parse_ascii_usize(digits: &[u8]) -> Option<usize> {
    let mut val = 0usize;
    for &b in digits {
        val = val.checked_mul(10)?.checked_add((b - b'0') as usize)?;
    }
    Some(val)
}

fn renumber_text_anchors(
    text: &mut String,
    source_path: &str,
    temp_bindings: &[(&'static str, String)],
    next_final_id: &mut usize,
    out_bindings: &mut Vec<HoistedBinding>,
) {
    let first_pos = match find_sentinel_bytes(text.as_bytes()) {
        Some(p) => p,
        None => return,
    };

    const SENT_PRE: &[u8] = SENTINEL_PREFIX.as_bytes();
    const SENT_ESC: &[u8] = SENTINEL_ESCAPED.as_bytes();
    const SENT_SUF: &[u8] = SENTINEL_SUFFIX.as_bytes();

    // Fast path: patch `⟦cp:t<tid>⟧` -> `⟦cp:v<fid>⟧` in place whenever digit counts match
    // (zero String allocation and zero memcpy for multi-KB prompts!).
    let mut fallback_pos: Option<usize> = None;
    {
        // SAFETY: Every write below strictly replaces ASCII bytes ('t' and ASCII digits)
        // with ASCII bytes ('v' and ASCII digits) of identical byte length, preserving UTF-8.
        let bytes = unsafe { text.as_bytes_mut() };
        let mut pos = first_pos;

        while pos < bytes.len() {
            let after = &bytes[pos..];
            if after.starts_with(SENT_ESC) {
                pos += SENT_ESC.len();
            } else if after.len() > SENT_PRE.len() && after[SENT_PRE.len()] == b't' {
                let d_start = pos + SENT_PRE.len() + 1;
                let mut d_end = d_start;
                while d_end < bytes.len() && bytes[d_end].is_ascii_digit() {
                    d_end += 1;
                }
                let digit_len = d_end - d_start;
                if digit_len > 0 && bytes[d_end..].starts_with(SENT_SUF) {
                    if let Some(tid) = parse_ascii_usize(&bytes[d_start..d_end]) {
                        if tid > 0 {
                            if let Some(&(kind, ref orig)) = temp_bindings.get(tid - 1) {
                                let fid = *next_final_id;
                                let mut num_buf = [0u8; 20];
                                let fid_bytes = format_usize_buf(fid, &mut num_buf);
                                if fid_bytes.len() == digit_len {
                                    *next_final_id += 1;
                                    bytes[pos + SENT_PRE.len()] = b'v';
                                    bytes[d_start..d_end].copy_from_slice(fid_bytes);
                                    let full_end = d_end + SENT_SUF.len();
                                    // SAFETY: `bytes[pos..full_end]` is valid UTF-8 ("⟦cp:v<digits>⟧").
                                    let anchor = unsafe {
                                        std::str::from_utf8_unchecked(&bytes[pos..full_end])
                                    }
                                    .to_owned();
                                    out_bindings.push(HoistedBinding {
                                        anchor,
                                        original_value: orig.clone(),
                                        kind,
                                        source_path: source_path.to_string(),
                                    });
                                    pos = full_end;
                                } else {
                                    fallback_pos = Some(pos);
                                    break;
                                }
                            } else {
                                pos += SENT_PRE.len();
                            }
                        } else {
                            pos += SENT_PRE.len();
                        }
                    } else {
                        pos += SENT_PRE.len();
                    }
                } else {
                    pos += SENT_PRE.len();
                }
            } else {
                pos += SENT_PRE.len();
            }

            match find_sentinel_bytes(&bytes[pos..]) {
                Some(rel) => pos += rel,
                None => break,
            }
        }
    }

    let start_fallback = match fallback_pos {
        Some(p) => p,
        None => return,
    };

    // Rare fallback when `tid` and `fid` have different decimal digit counts (e.g. t10 -> v1).
    let mut out = String::with_capacity(text.len() + 16);
    out.push_str(&text[..start_fallback]);
    let mut rest = &text[start_fallback..];
    let mut next_pos = Some(0usize);

    while let Some(pos) = next_pos {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];

        if after.starts_with(SENTINEL_ESCAPED) {
            out.push_str(SENTINEL_ESCAPED);
            rest = &after[SENTINEL_ESCAPED.len()..];
            next_pos = find_sentinel_bytes(rest.as_bytes());
            continue;
        }

        if let Some(after_prefix) = after.strip_prefix(SENTINEL_PREFIX) {
            if let Some(after_t) = after_prefix.strip_prefix('t') {
                let digit_len = after_t
                    .bytes()
                    .take_while(|b| b.is_ascii_digit())
                    .count();
                if digit_len > 0 && after_t[digit_len..].starts_with(SENTINEL_SUFFIX) {
                    if let Some(tid) = parse_ascii_usize(&after_t.as_bytes()[..digit_len]) {
                        if tid > 0 {
                            if let Some(&(kind, ref orig)) = temp_bindings.get(tid - 1) {
                                let fid = *next_final_id;
                                *next_final_id += 1;
                                let mut anchor = String::with_capacity(
                                    SENTINEL_PREFIX.len() + 1 + 6 + SENTINEL_SUFFIX.len(),
                                );
                                anchor.push_str(SENTINEL_PREFIX);
                                anchor.push('v');
                                push_usize(&mut anchor, fid);
                                anchor.push_str(SENTINEL_SUFFIX);
                                out.push_str(&anchor);
                                out_bindings.push(HoistedBinding {
                                    anchor,
                                    original_value: orig.clone(),
                                    kind,
                                    source_path: source_path.to_string(),
                                });
                                let full_len = SENTINEL_PREFIX.len()
                                    + 1
                                    + digit_len
                                    + SENTINEL_SUFFIX.len();
                                rest = &after[full_len..];
                                next_pos = find_sentinel_bytes(rest.as_bytes());
                                continue;
                            }
                        }
                    }
                }
            }
        }

        out.push_str(SENTINEL_PREFIX);
        rest = &after[SENTINEL_PREFIX.len()..];
        next_pos = find_sentinel_bytes(rest.as_bytes());
    }

    out.push_str(rest);
    *text = out;
}

fn finalize_hoisted_anchors(
    root: &mut JsonValue,
    provider: ProviderFormat,
    temp_bindings: &[(&'static str, String)],
    out_bindings: &mut Vec<HoistedBinding>,
) {
    let mut next_final_id = 1usize;

    if let Some(sys) = root.get_mut("system") {
        match sys {
            JsonValue::String(s) => {
                renumber_text_anchors(
                    s,
                    "system",
                    temp_bindings,
                    &mut next_final_id,
                    out_bindings,
                );
            }
            JsonValue::Array(items) => {
                for (i, item) in items.iter_mut().enumerate() {
                    if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                        if let Some(JsonValue::String(s)) = item.get_mut("text") {
                            if find_sentinel_bytes(s.as_bytes()).is_some() {
                                let path = fmt_indexed("system[", i, "]");
                                renumber_text_anchors(
                                    s,
                                    &path,
                                    temp_bindings,
                                    &mut next_final_id,
                                    out_bindings,
                                );
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(JsonValue::String(s)) = root.get_mut("instructions") {
        renumber_text_anchors(
            s,
            "instructions",
            temp_bindings,
            &mut next_final_id,
            out_bindings,
        );
    }

    if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
        let scan_limit = prefix_message_scan_limit(msgs);
        for (i, msg) in msgs.iter_mut().take(scan_limit).enumerate() {
            if let Some(content) = msg.get_mut("content") {
                match content {
                    JsonValue::String(s) => {
                        if find_sentinel_bytes(s.as_bytes()).is_some() {
                            let path = fmt_indexed("messages[", i, "]");
                            renumber_text_anchors(
                                s,
                                &path,
                                temp_bindings,
                                &mut next_final_id,
                                out_bindings,
                            );
                        }
                    }
                    JsonValue::Array(parts) => {
                        for (j, part) in parts.iter_mut().enumerate() {
                            if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                if find_sentinel_bytes(s.as_bytes()).is_some() {
                                    let path = fmt_msg_part("messages[", i, "].content[", j);
                                    renumber_text_anchors(
                                        s,
                                        &path,
                                        temp_bindings,
                                        &mut next_final_id,
                                        out_bindings,
                                    );
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if provider == ProviderFormat::OpenAiResponses {
        if let Some(items) = root.get_mut("input").and_then(JsonValue::as_array_mut) {
            let scan_limit = prefix_message_scan_limit(items);
            for (i, item) in items.iter_mut().take(scan_limit).enumerate() {
                if let Some(content) = item.get_mut("content") {
                    match content {
                        JsonValue::String(s) => {
                            if find_sentinel_bytes(s.as_bytes()).is_some() {
                                let path = fmt_indexed("input[", i, "]");
                                renumber_text_anchors(
                                    s,
                                    &path,
                                    temp_bindings,
                                    &mut next_final_id,
                                    out_bindings,
                                );
                            }
                        }
                        JsonValue::Array(parts) => {
                            for (j, part) in parts.iter_mut().enumerate() {
                                if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                    if find_sentinel_bytes(s.as_bytes()).is_some() {
                                        let path = fmt_msg_part("input[", i, "].content[", j);
                                        renumber_text_anchors(
                                            s,
                                            &path,
                                            temp_bindings,
                                            &mut next_final_id,
                                            out_bindings,
                                        );
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

fn format_binding_summary(bindings: &[HoistedBinding]) -> String {
    let est = 32
        + bindings
            .iter()
            .map(|b| b.anchor.len() + b.original_value.len() + 2)
            .sum::<usize>();
    let mut out = String::with_capacity(est);
    out.push_str("[cachepin volatile bindings]");
    for b in bindings {
        out.push('\n');
        out.push_str(&b.anchor);
        out.push('=');
        out.push_str(&b.original_value);
    }
    out
}

fn append_volatile_tail(
    root: &mut JsonValue,
    provider: ProviderFormat,
    summary: String,
) -> VolatileTailLoc {
    match provider {
        ProviderFormat::AnthropicMessages => {
            if root.get("messages").and_then(JsonValue::as_array).is_none() {
                root.set_field_sorted("messages", JsonValue::Array(Vec::new()));
            }
            if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
                let tail_part = JsonValue::Object(vec![
                    ("text".to_string(), JsonValue::String(summary)),
                    ("type".to_string(), JsonValue::String("text".to_string())),
                ]);
                let last_idx = msgs.len().saturating_sub(1);
                if let Some(last) = msgs.last_mut() {
                    if last.get("role").and_then(JsonValue::as_str) == Some("user") {
                        let msg_idx = last_idx;
                        match last.get_mut("content") {
                            Some(JsonValue::Array(parts)) => {
                                parts.push(tail_part);
                                return VolatileTailLoc::MessagePart(msg_idx, parts.len() - 1);
                            }
                            Some(JsonValue::String(orig)) => {
                                let orig_text = std::mem::take(orig);
                                let first_part = JsonValue::Object(vec![
                                    ("text".to_string(), JsonValue::String(orig_text)),
                                    ("type".to_string(), JsonValue::String("text".to_string())),
                                ]);
                                last.set_field_sorted(
                                    "content",
                                    JsonValue::Array(vec![first_part, tail_part]),
                                );
                                return VolatileTailLoc::MessagePart(msg_idx, 1);
                            }
                            _ => {
                                last.set_field_sorted("content", JsonValue::Array(vec![tail_part]));
                                return VolatileTailLoc::MessagePart(msg_idx, 0);
                            }
                        }
                    }
                }
                let msg_idx = msgs.len();
                msgs.push(JsonValue::Object(vec![
                    ("content".to_string(), JsonValue::Array(vec![tail_part])),
                    ("role".to_string(), JsonValue::String("user".to_string())),
                ]));
                VolatileTailLoc::MessagePart(msg_idx, 0)
            } else {
                VolatileTailLoc::Message(0)
            }
        }
        ProviderFormat::OpenAiResponses => {
            let tail_item = JsonValue::Object(vec![
                (
                    "content".to_string(),
                    JsonValue::Array(vec![JsonValue::Object(vec![
                        ("text".to_string(), JsonValue::String(summary)),
                        (
                            "type".to_string(),
                            JsonValue::String("input_text".to_string()),
                        ),
                    ])]),
                ),
                ("role".to_string(), JsonValue::String("user".to_string())),
            ]);
            match root.get_mut("input") {
                Some(JsonValue::Array(items)) => {
                    items.push(tail_item);
                    VolatileTailLoc::InputItem(items.len() - 1)
                }
                Some(JsonValue::String(orig)) => {
                    let orig_text = std::mem::take(orig);
                    let first_item = JsonValue::Object(vec![
                        (
                            "content".to_string(),
                            JsonValue::Array(vec![JsonValue::Object(vec![
                                ("text".to_string(), JsonValue::String(orig_text)),
                                (
                                    "type".to_string(),
                                    JsonValue::String("input_text".to_string()),
                                ),
                            ])]),
                        ),
                        ("role".to_string(), JsonValue::String("user".to_string())),
                    ]);
                    root.set_field_sorted("input", JsonValue::Array(vec![first_item, tail_item]));
                    VolatileTailLoc::InputItem(1)
                }
                _ => {
                    root.set_field_sorted("input", JsonValue::Array(vec![tail_item]));
                    VolatileTailLoc::InputItem(0)
                }
            }
        }
        ProviderFormat::OpenAiChat | ProviderFormat::VllmOpenAi | ProviderFormat::Generic => {
            if root.get("messages").and_then(JsonValue::as_array).is_none() {
                root.set_field_sorted("messages", JsonValue::Array(Vec::new()));
            }
            if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
                let idx = msgs.len();
                msgs.push(JsonValue::Object(vec![
                    ("content".to_string(), JsonValue::String(summary)),
                    ("role".to_string(), JsonValue::String("user".to_string())),
                ]));
                VolatileTailLoc::Message(idx)
            } else {
                VolatileTailLoc::Message(0)
            }
        }
    }
}

// --- SIMD-Accelerated Volatile Pattern Detectors ---

struct VolatileVisitor<'a> {
    text: &'a str,
    bytes: &'a [u8],
    start_offset: usize,
    max_spans: usize,
    cursor: usize,
    spans: Vec<(usize, usize, &'static str)>,
    has_sentinel: bool,
}

impl VolatileVisitor<'_> {
    #[cold]
    #[inline(never)]
    fn try_iso(&mut self, start: usize, rel: usize) -> usize {
        if let Some(end) = match_iso_timestamp(self.bytes, start) {
            self.spans.push((start, end, "timestamp"));
            self.cursor = end;
            return if self.spans.len() >= self.max_spans {
                usize::MAX
            } else {
                end - self.start_offset
            };
        }
        rel + 1
    }

    #[cold]
    #[inline(never)]
    fn try_uuid(&mut self, start: usize, rel: usize) -> usize {
        if let Some(end) = match_uuid(self.bytes, start) {
            self.spans.push((start, end, "uuid"));
            self.cursor = end;
            return if self.spans.len() >= self.max_spans {
                usize::MAX
            } else {
                end - self.start_offset
            };
        }
        rel + 1
    }

    #[cold]
    #[inline(never)]
    fn try_prefixed_id(&mut self, start: usize, rel: usize) -> usize {
        if let Some(end) = match_prefixed_id(self.bytes, start) {
            self.spans.push((start, end, "opaque_id"));
            self.cursor = end;
            return if self.spans.len() >= self.max_spans {
                usize::MAX
            } else {
                end - self.start_offset
            };
        }
        rel + 1
    }

    #[cold]
    #[inline(never)]
    fn try_colon_e(&mut self, hit: usize, rel: usize) -> usize {
        let bytes = self.bytes;
        let mut matched = None;
        if hit >= self.cursor + 16 && (bytes[hit - 16] | 0x20) == b'c' {
            matched = match_inline_header(self.text, bytes, hit - 16);
        }
        if matched.is_none() && hit >= self.cursor + 12 && (bytes[hit - 12] | 0x20) == b'c' {
            matched = match_inline_header(self.text, bytes, hit - 12);
        }
        if matched.is_none() && hit >= self.cursor + 4 && (bytes[hit - 4] | 0x20) == b'd' {
            matched = match_inline_header(self.text, bytes, hit - 4);
        }
        if let Some((val_start, val_end, kind)) = matched {
            self.spans.push((val_start, val_end, kind));
            self.cursor = val_end;
            return if self.spans.len() >= self.max_spans {
                usize::MAX
            } else {
                val_end - self.start_offset
            };
        }
        rel + 1
    }

    #[cold]
    #[inline(never)]
    fn try_colon_p(&mut self, hit: usize, rel: usize) -> usize {
        if let Some((val_start, val_end, kind)) =
            match_inline_header(self.text, self.bytes, hit - 9)
        {
            self.spans.push((val_start, val_end, kind));
            self.cursor = val_end;
            return if self.spans.len() >= self.max_spans {
                usize::MAX
            } else {
                val_end - self.start_offset
            };
        }
        rel + 1
    }
}

impl Any3Visitor for VolatileVisitor<'_> {
    #[inline(always)]
    fn on_hit(&mut self, rel: usize) -> usize {
        let bytes = self.bytes;
        let len = bytes.len();
        let hit = self.start_offset + rel;
        let b = bytes[hit];

        if b == b'-' {
            // ISO-8601 (`YYYY-MM-DD[T ]HH:MM:SS...`) has '-' at start + 4 and '-' at hit + 3.
            // UUID (`8-4-4-4-12`) has '-' at start + 8 and hex at hit + 3 (mutually exclusive).
            if hit >= self.cursor + 4
                && hit + 15 <= len
                && (bytes[hit - 4] == b'1' || bytes[hit - 4] == b'2')
                && bytes[hit + 3] == b'-'
                && bytes[hit + 9] == b':'
            {
                return self.try_iso(hit - 4, rel);
            }
            if hit >= self.cursor + 8
                && hit + 28 <= len
                && bytes[hit + 5] == b'-'
                && bytes[hit + 10] == b'-'
                && bytes[hit + 15] == b'-'
            {
                return self.try_uuid(hit - 8, rel);
            }
            return rel + 1;
        }

        if b == b'_' {
            // Prefixed opaque IDs (`req_`, `run_`, `span_`, `call_`, `trace_`, `session_`, `agent_run_`).
            if hit > self.cursor {
                let cand_start = match bytes[hit - 1] {
                    b'q' if hit >= self.cursor + 3
                        && hit + 9 <= len
                        && bytes[hit - 3] == b'r' =>
                    {
                        Some(hit - 3)
                    }
                    b'l' if hit >= self.cursor + 4
                        && hit + 9 <= len
                        && bytes[hit - 4] == b'c' =>
                    {
                        Some(hit - 4)
                    }
                    b'e' if hit >= self.cursor + 5
                        && hit + 9 <= len
                        && bytes[hit - 5] == b't' =>
                    {
                        Some(hit - 5)
                    }
                    b't' if hit >= self.cursor + 5
                        && hit + 13 <= len
                        && bytes[hit - 5] == b'a'
                        && bytes[hit + 4] == b'_' =>
                    {
                        Some(hit - 5)
                    }
                    b'n' if hit + 9 <= len => {
                        if hit >= self.cursor + 7
                            && bytes[hit - 7] == b's'
                            && bytes[hit - 2] == b'o'
                        {
                            Some(hit - 7)
                        } else if hit >= self.cursor + 4
                            && bytes[hit - 4] == b's'
                            && bytes[hit - 2] == b'a'
                        {
                            Some(hit - 4)
                        } else if hit >= self.cursor + 3
                            && bytes[hit - 3] == b'r'
                            && bytes[hit - 2] == b'u'
                        {
                            Some(hit - 3)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(start) = cand_start {
                    return self.try_prefixed_id(start, rel);
                }
            }
            return rel + 1;
        }

        // b == b':' — detect literal sentinel prefix ("⟦cp:") and inline headers.
        if hit > 0 {
            let prev = bytes[hit - 1];
            let prev_lower = prev | 0x20;
            if prev_lower == b'e' {
                if hit > self.cursor
                    && ((hit >= self.cursor + 16 && (bytes[hit - 16] | 0x20) == b'c')
                        || (hit >= self.cursor + 12 && (bytes[hit - 12] | 0x20) == b'c')
                        || (hit >= self.cursor + 4 && (bytes[hit - 4] | 0x20) == b'd'))
                {
                    return self.try_colon_e(hit, rel);
                }
            } else if prev_lower == b'p' {
                if !self.has_sentinel
                    && prev == b'p'
                    && hit >= 5
                    && bytes[hit - 2] == b'c'
                    && bytes[hit - 3] == 0xA6
                    && bytes[hit - 4] == 0x9F
                    && bytes[hit - 5] == 0xE2
                {
                    self.has_sentinel = true;
                }
                if hit >= self.cursor + 9 && (bytes[hit - 9] | 0x20) == b't' {
                    return self.try_colon_p(hit, rel);
                }
            }
        }
        rel + 1
    }
}

/// Finds volatile spans and simultaneously detects any literal `SENTINEL_PREFIX`
/// in a single SIMD pass via `scan_any3_byte(..., b'-', b'_', b':')`.
///
/// Correctness proof:
/// - Every volatile pattern contains an anchor delimiter `'-'`, `'_'`, or `':'` at
///   a fixed offset `<= 16` after its start byte with strictly ASCII, delimiter-free
///   prefix bytes before that anchor:
///   - ISO-8601 (`YYYY-MM-DD...`): `'-'` at `start + 4`
///   - UUID (`8-4-4-4-12`): `'-'` at `start + 8`
///   - Prefixed IDs (`req_`, `run_`, `span_`, `call_`, `trace_`, `agent_run_`, `session_`):
///     first `'_'` at `start + {3, 4, 5, 7}`
///   - Inline headers (`current utc time:`, `current time:`, `current date:`,
///     `timestamp:`, `date:`): `':'` at `start + {16, 12, 9, 4}`
/// - `SENTINEL_PREFIX` (`"⟦cp:"`, 6 UTF-8 bytes) ends with `':'`.
/// - UTF-8 is self-synchronizing (no multi-byte sequence contains ASCII bytes), and
///   every matched pattern prefix is strictly ASCII, so every matched `start` is
///   guaranteed to be a valid UTF-8 character boundary.
fn find_volatile_spans_and_sentinel_from(
    text: &str,
    max_spans: usize,
    start_offset: usize,
) -> (Vec<(usize, usize, &'static str)>, bool) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 || start_offset >= len {
        return (Vec::new(), false);
    }
    if max_spans == 0 {
        return (Vec::new(), find_sentinel_bytes(&bytes[start_offset..]).is_some());
    }

    let mut visitor = VolatileVisitor {
        text,
        bytes,
        start_offset,
        max_spans,
        cursor: 0,
        spans: Vec::new(),
        has_sentinel: false,
    };

    scan_any3_byte(&bytes[start_offset..], b'-', b'_', b':', &mut visitor);

    (visitor.spans, visitor.has_sentinel)
}

#[cfg(test)]
fn find_volatile_spans_and_sentinel(
    text: &str,
    max_spans: usize,
) -> (Vec<(usize, usize, &'static str)>, bool) {
    find_volatile_spans_and_sentinel_from(text, max_spans, 0)
}

#[cfg(test)]
fn find_volatile_spans(text: &str, max_spans: usize) -> Vec<(usize, usize, &'static str)> {
    find_volatile_spans_and_sentinel(text, max_spans).0
}

#[inline]
fn utf8_char_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

#[inline]
fn is_word_boundary_before(bytes: &[u8], i: usize) -> bool {
    if i == 0 {
        return true;
    }
    let prev = bytes[i - 1];
    !(prev.is_ascii_alphanumeric() || prev == b'_')
}

#[inline]
fn starts_with_ignore_ascii_case(bytes: &[u8], pos: usize, needle: &[u8]) -> bool {
    match bytes.get(pos..pos + needle.len()) {
        Some(slice) => slice.eq_ignore_ascii_case(needle),
        None => false,
    }
}

#[inline]
fn parse_2_digits(bytes: &[u8], pos: usize) -> Option<u32> {
    let a = *bytes.get(pos)?;
    let b = *bytes.get(pos + 1)?;
    if a.is_ascii_digit() && b.is_ascii_digit() {
        Some(((a - b'0') as u32) * 10 + ((b - b'0') as u32))
    } else {
        None
    }
}

#[inline]
fn parse_4_digits(bytes: &[u8], pos: usize) -> Option<u32> {
    let hi = parse_2_digits(bytes, pos)?;
    let lo = parse_2_digits(bytes, pos + 2)?;
    Some(hi * 100 + lo)
}

fn match_iso_date(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 10 > bytes.len() {
        return None;
    }
    let year = parse_4_digits(bytes, pos)?;
    if !(1900..=2199).contains(&year) {
        return None;
    }
    let sep = bytes[pos + 4];
    if sep != b'-' && sep != b'/' {
        return None;
    }
    let month = parse_2_digits(bytes, pos + 5)?;
    if !(1..=12).contains(&month) || bytes[pos + 7] != sep {
        return None;
    }
    let day = parse_2_digits(bytes, pos + 8)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    if bytes.get(pos + 10).map_or(false, |b| b.is_ascii_digit()) {
        return None;
    }
    Some(pos + 10)
}

fn match_iso_timestamp(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 19 > bytes.len() {
        return None;
    }
    if pos > 0 && bytes[pos - 1].is_ascii_alphanumeric() {
        return None;
    }

    let year = parse_4_digits(bytes, pos)?;
    if !(1900..=2199).contains(&year) || bytes[pos + 4] != b'-' {
        return None;
    }
    let month = parse_2_digits(bytes, pos + 5)?;
    if !(1..=12).contains(&month) || bytes[pos + 7] != b'-' {
        return None;
    }
    let day = parse_2_digits(bytes, pos + 8)?;
    if !(1..=31).contains(&day) {
        return None;
    }
    let sep = bytes[pos + 10];
    if sep != b'T' && sep != b't' && sep != b' ' {
        return None;
    }
    let hour = parse_2_digits(bytes, pos + 11)?;
    if hour > 23 || bytes[pos + 13] != b':' {
        return None;
    }
    let min = parse_2_digits(bytes, pos + 14)?;
    if min > 59 || bytes[pos + 16] != b':' {
        return None;
    }
    let sec = parse_2_digits(bytes, pos + 17)?;
    if sec > 60 {
        return None;
    }

    let mut end = pos + 19;

    // Optional fractional seconds `.123456789`
    if bytes.get(end) == Some(&b'.') {
        let frac_start = end + 1;
        let mut f = frac_start;
        while f < bytes.len() && bytes[f].is_ascii_digit() && f - frac_start < 9 {
            f += 1;
        }
        if f > frac_start {
            end = f;
        }
    }

    // Optional timezone suffix: `Z`, `+HH:MM`, `-HH:MM`, `+HHMM`, `-HHMM`
    match bytes.get(end) {
        Some(b'Z' | b'z') => {
            end += 1;
        }
        Some(b'+' | b'-') => {
            if let Some(tz_h) = parse_2_digits(bytes, end + 1) {
                if tz_h <= 23 {
                    if bytes.get(end + 3) == Some(&b':') {
                        if let Some(tz_m) = parse_2_digits(bytes, end + 4) {
                            if tz_m <= 59 {
                                end += 6;
                            }
                        }
                    } else if let Some(tz_m) = parse_2_digits(bytes, end + 3) {
                        if tz_m <= 59 {
                            end += 5;
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Reject if immediately followed by a digit (avoids partial number captures).
    if bytes.get(end).map_or(false, |b| b.is_ascii_digit()) {
        return None;
    }

    Some(end)
}

fn match_uuid(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 36 > bytes.len() {
        return None;
    }
    if pos > 0 && bytes[pos - 1].is_ascii_alphanumeric() {
        return None;
    }
    if bytes[pos + 8] != b'-'
        || bytes[pos + 13] != b'-'
        || bytes[pos + 18] != b'-'
        || bytes[pos + 23] != b'-'
    {
        return None;
    }
    for offset in 0..36 {
        if matches!(offset, 8 | 13 | 18 | 23) {
            continue;
        }
        if !bytes[pos + offset].is_ascii_hexdigit() {
            return None;
        }
    }
    if bytes
        .get(pos + 36)
        .map_or(false, |b| b.is_ascii_alphanumeric())
    {
        return None;
    }
    Some(pos + 36)
}

fn match_prefixed_id(bytes: &[u8], pos: usize) -> Option<usize> {
    const PREFIXES: &[&[u8]] = &[
        b"agent_run_",
        b"session_",
        b"trace_",
        b"span_",
        b"call_",
        b"req_",
        b"run_",
    ];

    if !is_word_boundary_before(bytes, pos) {
        return None;
    }

    let mut matched_prefix_len = 0usize;
    for &prefix in PREFIXES {
        if bytes.get(pos..pos + prefix.len()) == Some(prefix) {
            matched_prefix_len = prefix.len();
            break;
        }
    }
    if matched_prefix_len == 0 {
        return None;
    }

    let tail_start = pos + matched_prefix_len;
    let mut end = tail_start;
    let mut has_digit = false;
    let mut all_hex = true;

    while end < bytes.len() {
        let b = bytes[end];
        if b.is_ascii_alphanumeric() {
            if b.is_ascii_digit() {
                has_digit = true;
            }
            if !b.is_ascii_hexdigit() {
                all_hex = false;
            }
            end += 1;
        } else {
            break;
        }
    }

    let tail_len = end - tail_start;
    if tail_len < 8 {
        return None;
    }
    // Reject trailing underscore continuation (e.g. snake_case code identifiers).
    if bytes.get(end) == Some(&b'_') {
        return None;
    }
    // Require at least one digit OR >=12 pure hex characters to prevent false
    // positives on ordinary English identifiers like `run_benchmark`.
    if !has_digit && !(all_hex && tail_len >= 12) {
        return None;
    }

    Some(end)
}

fn match_inline_header(
    text: &str,
    bytes: &[u8],
    pos: usize,
) -> Option<(usize, usize, &'static str)> {
    const LABELS: &[(&[u8], bool, &'static str)] = &[
        (b"current utc time:", false, "inline_time"),
        (b"current time:", false, "inline_time"),
        (b"current date:", false, "inline_date"),
        (b"timestamp:", false, "inline_time"),
        (b"date:", true, "inline_date"),
    ];

    if !is_word_boundary_before(bytes, pos) {
        return None;
    }

    let mut label_len = 0usize;
    let mut require_date = false;
    let mut kind = "inline_time";

    for &(label, req_date, k) in LABELS {
        if starts_with_ignore_ascii_case(bytes, pos, label) {
            label_len = label.len();
            require_date = req_date;
            kind = k;
            break;
        }
    }
    if label_len == 0 {
        return None;
    }

    let mut val_start = pos + label_len;
    while val_start < bytes.len() && (bytes[val_start] == b' ' || bytes[val_start] == b'\t') {
        val_start += 1;
    }
    if val_start >= bytes.len() {
        return None;
    }

    // Fast path: value starts directly with an ISO-8601 timestamp.
    if let Some(mut ts_end) = match_iso_timestamp(bytes, val_start) {
        for tz in [b" UTC".as_slice(), b" GMT".as_slice()] {
            if bytes.get(ts_end..ts_end + tz.len()) == Some(tz)
                && !bytes
                    .get(ts_end + tz.len())
                    .map_or(false, |b| b.is_ascii_alphanumeric())
            {
                ts_end += tz.len();
                break;
            }
        }
        return Some((val_start, ts_end, kind));
    }

    if require_date {
        // For `Date:`, require explicit YYYY-MM-DD / YYYY/MM/DD to avoid false positives.
        let date_end = match_iso_date(bytes, val_start)?;
        return Some((val_start, date_end, kind));
    }

    // Scan bounded single-line value for human-formatted timestamps/epoch ints.
    let limit = bytes.len().min(val_start + 64);
    let mut j = val_start;
    while j < limit {
        let b = bytes[j];
        if matches!(
            b,
            b'\n' | b'\r' | b'"' | b'\'' | b']' | b')' | b'}' | b';' | b'|' | b'<' | b'>'
        ) {
            break;
        }
        if bytes[j..].starts_with(SENTINEL_PREFIX.as_bytes())
            || bytes[j..].starts_with(SENTINEL_SUFFIX.as_bytes())
        {
            break;
        }
        // Stop at sentence-ending period (`. ` or trailing `.`).
        if b == b'.'
            && (j + 1 >= bytes.len()
                || bytes[j + 1] == b' '
                || bytes[j + 1] == b'\n'
                || bytes[j + 1] == b'\r')
        {
            break;
        }
        // Stop if a prefixed opaque ID begins on the same line after whitespace.
        if j > val_start && bytes[j - 1] == b' ' && match_prefixed_id(bytes, j).is_some() {
            break;
        }
        j += utf8_char_len(b);
    }

    let mut val_end = j.min(bytes.len());
    while val_end > val_start && !text.is_char_boundary(val_end) {
        val_end -= 1;
    }
    while val_end > val_start
        && matches!(bytes[val_end - 1], b' ' | b'\t' | b',' | b'.')
    {
        val_end -= 1;
    }

    if val_end.saturating_sub(val_start) < 4 {
        return None;
    }
    if !bytes[val_start..val_end].iter().any(|b| b.is_ascii_digit()) {
        return None;
    }

    Some((val_start, val_end, kind))
}

// ============================================================================
// Pass 3: AllGatherAlign (Collective Multi-Agent Prefix Convergence)
// ============================================================================

type CachedBlock = Option<(Vec<u8>, u64)>;

/// Stream-hashes `bytes` with SipHash-1-3 while normalizing temporary volatile
/// anchors `⟦cp:t<digits>⟧` to `⟦cp:v0⟧` on the fly with zero heap allocation.
fn hash_normalized_temp_anchors(tenant_key: &[u8; 16], bytes: &[u8]) -> u64 {
    const SENT_PRE: &[u8] = SENTINEL_PREFIX.as_bytes();
    const SENT_ESC: &[u8] = SENTINEL_ESCAPED.as_bytes();
    const SENT_SUF: &[u8] = SENTINEL_SUFFIX.as_bytes();

    let mut hasher = KeyedHasher::new(tenant_key);
    let mut rest = bytes;

    while let Some(pos) = find_sentinel_bytes(rest) {
        hasher.write(&rest[..pos]);
        let after = &rest[pos..];
        if after.starts_with(SENT_ESC) {
            hasher.write(SENT_ESC);
            rest = &after[SENT_ESC.len()..];
            continue;
        }
        if let Some(after_prefix) = after.strip_prefix(SENT_PRE) {
            if let Some(after_t) = after_prefix.strip_prefix(b"t") {
                let dlen = after_t.iter().take_while(|b| b.is_ascii_digit()).count();
                if dlen > 0 && after_t[dlen..].starts_with(SENT_SUF) {
                    hasher.write("⟦cp:v0⟧".as_bytes());
                    let full_len = SENT_PRE.len() + 1 + dlen + SENT_SUF.len();
                    rest = &after[full_len..];
                    continue;
                }
            }
        }
        hasher.write(SENT_PRE);
        rest = &after[SENT_PRE.len()..];
    }

    hasher.write(rest);
    hasher.finish()
}

#[inline]
fn canonical_bytes_without_cache_control(val: &JsonValue, already_canonical: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(estimate_serialized_len(val) + 16);
    if already_canonical {
        serialize_json_skip_top_key(val, "cache_control", &mut out);
    } else {
        serialize_canonical_skip_top_key(val, "cache_control", &mut out);
    }
    out
}

fn is_system_alignable_block(item: &JsonValue) -> bool {
    matches!(
        item.get("type").and_then(JsonValue::as_str),
        Some("text" | "document")
    )
}

fn is_shared_context_block(part: &JsonValue) -> bool {
    match part.get("type").and_then(JsonValue::as_str) {
        Some("document") => return true,
        Some("text" | "input_text") => {}
        _ => return false,
    }

    if part.get("shared_context").and_then(JsonValue::as_bool) == Some(true) {
        return true;
    }
    if part.get("cache_scope").and_then(JsonValue::as_str) == Some("shared") {
        return true;
    }

    if let Some(txt) = part.get("text").and_then(JsonValue::as_str) {
        let t = txt.trim_start();
        let bytes = t.as_bytes();
        let first = match bytes.first() {
            Some(&b) => b,
            None => return false,
        };
        if first != b'<' && first != b'-' {
            return false;
        }
        return starts_with_ignore_ascii_case(bytes, 0, b"<doc")
            || starts_with_ignore_ascii_case(bytes, 0, b"<file")
            || starts_with_ignore_ascii_case(bytes, 0, b"<context")
            || starts_with_ignore_ascii_case(bytes, 0, b"<shared_context")
            || starts_with_ignore_ascii_case(bytes, 0, b"<peer_output")
            || starts_with_ignore_ascii_case(bytes, 0, b"--- file:");
    }
    false
}

fn align_block_slice(
    slice: &mut [JsonValue],
    tenant_key: &[u8; 16],
    tenant_hash: u64,
    swarm_stats: &SwarmBlockStats,
    has_temp_anchors: bool,
    capture_cache: bool,
) -> (bool, Option<Vec<CachedBlock>>) {
    if slice.len() < 2 {
        return (false, None);
    }

    let mut hashes = Vec::with_capacity(slice.len());
    let mut cached: Option<Vec<CachedBlock>> = if capture_cache {
        Some(Vec::with_capacity(slice.len()))
    } else {
        None
    };
    let mut scratch = if capture_cache {
        Vec::new()
    } else {
        Vec::with_capacity(1024)
    };

    for item in slice.iter() {
        if let Some(ref mut cache_vec) = cached {
            let mut buf = Vec::with_capacity(estimate_serialized_len(item) + 16);
            serialize_json_skip_top_key(item, "cache_control", &mut buf);
            if has_temp_anchors && find_sentinel_bytes(&buf).is_some() {
                let bh = hash_normalized_temp_anchors(tenant_key, &buf);
                hashes.push(bh);
                cache_vec.push(None);
            } else {
                let bh = keyed_hash64(tenant_key, &buf);
                hashes.push(bh);
                cache_vec.push(Some((buf, bh)));
            }
        } else {
            scratch.clear();
            serialize_json_skip_top_key(item, "cache_control", &mut scratch);
            let bh = if has_temp_anchors && find_sentinel_bytes(&scratch).is_some() {
                hash_normalized_temp_anchors(tenant_key, &scratch)
            } else {
                keyed_hash64(tenant_key, &scratch)
            };
            hashes.push(bh);
        }
    }

    let scores = swarm_stats.record_batch_and_score(tenant_hash, &hashes);
    let mut ranked: Vec<(u32, u64, usize)> = Vec::with_capacity(slice.len());
    for (idx, (&bh, &score)) in hashes.iter().zip(scores.iter()).enumerate() {
        let bucket = swarm_freq_bucket(score);
        ranked.push((bucket, bh, idx));
    }

    let already_sorted = ranked
        .windows(2)
        .all(|w| (Reverse(w[0].0), Reverse(w[0].1)) <= (Reverse(w[1].0), Reverse(w[1].1)));
    if already_sorted {
        return (false, cached);
    }

    // Stable sort descending by `(swarm_freq_bucket, canonical_block_hash)`.
    ranked.sort_by_key(|&(bucket, bh, _)| (Reverse(bucket), Reverse(bh)));

    let mut owned: Vec<Option<JsonValue>> = slice
        .iter_mut()
        .map(|v| Some(std::mem::replace(v, JsonValue::Null)))
        .collect();

    if let Some(ref mut old_cache) = cached {
        let mut new_cache = Vec::with_capacity(old_cache.len());
        for (dst, (_, _, src_idx)) in ranked.into_iter().enumerate() {
            if let Some(val) = owned[src_idx].take() {
                slice[dst] = val;
            }
            new_cache.push(old_cache[src_idx].take());
        }
        *old_cache = new_cache;
    } else {
        for (dst, (_, _, src_idx)) in ranked.into_iter().enumerate() {
            if let Some(val) = owned[src_idx].take() {
                slice[dst] = val;
            }
        }
    }
    (true, cached)
}

fn align_delimited_text(
    text: &mut String,
    delim: &str,
    tenant_key: &[u8; 16],
    tenant_hash: u64,
    swarm_stats: &SwarmBlockStats,
    has_temp_anchors: bool,
) -> bool {
    if delim.is_empty() || !text.contains(delim) {
        return false;
    }
    let parts: Vec<&str> = text.split(delim).collect();
    if parts.len() < 2 {
        return false;
    }

    let keep_leading_empty = parts[0].trim().is_empty() && parts.len() >= 3;
    let start_idx = usize::from(keep_leading_empty);
    let sortable = &parts[start_idx..];

    let mut ranked: Vec<(u32, u64, usize)> = Vec::with_capacity(sortable.len());
    for (idx, &chunk) in sortable.iter().enumerate() {
        let cb = chunk.as_bytes();
        let bh = if has_temp_anchors && find_sentinel_bytes(cb).is_some() {
            hash_normalized_temp_anchors(tenant_key, cb)
        } else {
            keyed_hash64(tenant_key, cb)
        };
        let score = swarm_stats.record_and_score(tenant_hash, bh);
        let bucket = swarm_freq_bucket(score);
        ranked.push((bucket, bh, idx));
    }

    let already_sorted = ranked
        .windows(2)
        .all(|w| (Reverse(w[0].0), Reverse(w[0].1)) <= (Reverse(w[1].0), Reverse(w[1].1)));
    if already_sorted {
        return false;
    }

    ranked.sort_by_key(|&(bucket, bh, _)| (Reverse(bucket), Reverse(bh)));

    let mut rebuilt = String::with_capacity(text.len());
    if keep_leading_empty {
        rebuilt.push_str(parts[0]);
        rebuilt.push_str(delim);
    }
    for (pos, (_, _, orig_idx)) in ranked.into_iter().enumerate() {
        if pos > 0 {
            rebuilt.push_str(delim);
        }
        rebuilt.push_str(sortable[orig_idx]);
    }
    *text = rebuilt;
    true
}

fn message_has_tool_blocks(msg: &JsonValue) -> bool {
    if msg.get("tool_calls").is_some() || msg.get("tool_call_id").is_some() {
        return true;
    }
    if let Some(parts) = msg.get("content").and_then(JsonValue::as_array) {
        return parts.iter().any(|p| {
            matches!(
                p.get("type").and_then(JsonValue::as_str),
                Some("tool_use" | "tool_result")
            )
        });
    }
    false
}

fn all_gather_align_pass(
    root: &mut JsonValue,
    provider: ProviderFormat,
    tenant_key: &[u8; 16],
    tenant_hash: u64,
    swarm_stats: &SwarmBlockStats,
    delimiter: Option<&str>,
    has_temp_anchors: bool,
) -> (bool, Option<Vec<CachedBlock>>) {
    let mut reordered = false;
    let mut cached_system_blocks = None;

    // 1. Delimiter-chunk alignment inside `system` / `instructions` / early messages.
    if let Some(delim) = delimiter {
        if let Some(sys) = root.get_mut("system") {
            match sys {
                JsonValue::String(s) => {
                    if align_delimited_text(
                        s,
                        delim,
                        tenant_key,
                        tenant_hash,
                        swarm_stats,
                        has_temp_anchors,
                    ) {
                        reordered = true;
                    }
                }
                JsonValue::Array(items) => {
                    for item in items.iter_mut() {
                        if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                            if let Some(JsonValue::String(s)) = item.get_mut("text") {
                                if align_delimited_text(
                                    s,
                                    delim,
                                    tenant_key,
                                    tenant_hash,
                                    swarm_stats,
                                    has_temp_anchors,
                                ) {
                                    reordered = true;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(JsonValue::String(s)) = root.get_mut("instructions") {
            if align_delimited_text(
                s,
                delim,
                tenant_key,
                tenant_hash,
                swarm_stats,
                has_temp_anchors,
            ) {
                reordered = true;
            }
        }
        if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
            for msg in msgs.iter_mut() {
                if message_has_tool_blocks(msg) {
                    continue;
                }
                if let Some(content) = msg.get_mut("content") {
                    match content {
                        JsonValue::String(s) => {
                            if align_delimited_text(
                                s,
                                delim,
                                tenant_key,
                                tenant_hash,
                                swarm_stats,
                                has_temp_anchors,
                            ) {
                                reordered = true;
                            }
                        }
                        JsonValue::Array(parts) => {
                            for part in parts.iter_mut() {
                                if matches!(
                                    part.get("type").and_then(JsonValue::as_str),
                                    Some("text" | "input_text")
                                ) {
                                    if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                        if align_delimited_text(
                                            s,
                                            delim,
                                            tenant_key,
                                            tenant_hash,
                                            swarm_stats,
                                            has_temp_anchors,
                                        ) {
                                            reordered = true;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // 2. Anthropic multi-block `system` content arrays.
    if let Some(sys_arr) = root.get_mut("system").and_then(JsonValue::as_array_mut) {
        if sys_arr.len() >= 2 && sys_arr.iter().all(is_system_alignable_block) {
            let (sys_reordered, sys_cache) = align_block_slice(
                sys_arr,
                tenant_key,
                tenant_hash,
                swarm_stats,
                has_temp_anchors,
                true,
            );
            if sys_reordered {
                reordered = true;
            }
            cached_system_blocks = sys_cache;
        }
    }

    // 3. Contiguous runs of independent `<doc>` / `shared_context` blocks inside
    //    a single user/system/developer message content array.
    //    NEVER reorders sequential `messages` turns or tool call/result pairs.
    let container_key = if provider == ProviderFormat::OpenAiResponses
        && root.get("input").and_then(JsonValue::as_array).is_some()
    {
        "input"
    } else {
        "messages"
    };

    if let Some(msgs) = root.get_mut(container_key).and_then(JsonValue::as_array_mut) {
        for msg in msgs.iter_mut() {
            if message_has_tool_blocks(msg) {
                continue;
            }
            let role = msg.get("role").and_then(JsonValue::as_str).unwrap_or("");
            if !matches!(role, "user" | "system" | "developer") {
                continue;
            }
            if let Some(parts) = msg.get_mut("content").and_then(JsonValue::as_array_mut) {
                let mut i = 0usize;
                while i < parts.len() {
                    if is_shared_context_block(&parts[i]) {
                        let start = i;
                        while i < parts.len() && is_shared_context_block(&parts[i]) {
                            i += 1;
                        }
                        if i - start >= 2
                            && align_block_slice(
                                &mut parts[start..i],
                                tenant_key,
                                tenant_hash,
                                swarm_stats,
                                has_temp_anchors,
                                false,
                            )
                            .0
                        {
                            reordered = true;
                        }
                    } else {
                        i += 1;
                    }
                }
            }
        }
    }

    (reordered, cached_system_blocks)
}

// ============================================================================
// Pass 4: ConfigGuard
// ============================================================================

fn compute_config_fingerprint(root: &JsonValue, tenant_key: &[u8; 16]) -> u64 {
    const FIELDS: &[&str] = &["model", "reasoning", "temperature", "thinking", "tool_choice"];
    let mut bytes = Vec::with_capacity(128);
    bytes.push(b'{');
    let mut emitted = false;
    for &field in FIELDS {
        if let Some(val) = root.get(field) {
            if emitted {
                bytes.push(b',');
            }
            bytes.push(b'"');
            bytes.extend_from_slice(field.as_bytes());
            bytes.extend_from_slice(b"\":");
            let _ = serialize_canonical(val, &mut bytes);
            emitted = true;
        }
    }
    bytes.push(b'}');
    keyed_hash64(tenant_key, &bytes)
}

fn detect_config_warnings(root: &JsonValue, provider: ProviderFormat) -> Vec<String> {
    let mut warnings = Vec::new();

    // Check `tool_choice` disabling tools trap.
    if let Some(tc) = root.get("tool_choice") {
        let is_none = tc.as_str() == Some("none")
            || tc.get("type").and_then(JsonValue::as_str) == Some("none");
        if is_none {
            warnings.push(
                "tool_choice is set to 'none'; toggling tool availability via removal or 'none' can invalidate the tool-prefix cache—prefer keeping tools constant and restricting via allowed_tools or prompt instructions"
                    .to_string(),
            );
        }
    }

    // Check if conversation history contains tool calls/results while `tools` is absent/empty.
    let tools_missing_or_empty = match root.get("tools") {
        None | Some(JsonValue::Null) => true,
        Some(JsonValue::Array(arr)) => arr.is_empty(),
        _ => false,
    };
    if tools_missing_or_empty {
        if let Some(msgs) = root.get("messages").and_then(JsonValue::as_array) {
            if msgs.iter().any(message_has_tool_blocks) {
                warnings.push(
                    "tools array was removed/emptied despite prior tool_use/tool_result turns in messages; removing tools busts the entire prefix cache—keep tools intact and restrict via allowed_tools"
                        .to_string(),
                );
            }
        }
    }

    // Check Anthropic `thinking` + incompatible `temperature` or forced `tool_choice`.
    if let Some(thinking) = root.get("thinking") {
        if thinking.get("type").and_then(JsonValue::as_str) == Some("enabled") {
            if let Some(temp) = root.get("temperature").and_then(JsonValue::as_f64) {
                if (temp - 1.0).abs() > 1e-6 {
                    warnings.push(format!(
                        "thinking is enabled with temperature={temp}; Anthropic requires temperature=1 when extended thinking is enabled"
                    ));
                }
            }
            if let Some(tc) = root.get("tool_choice") {
                if matches!(
                    tc.get("type").and_then(JsonValue::as_str),
                    Some("any" | "tool")
                ) {
                    warnings.push(
                        "thinking is enabled alongside forced tool_choice ('any'/'tool'), which is rejected or invalidates cached thinking prefixes"
                            .to_string(),
                    );
                }
            }
        }
    }

    // Check existing `cache_control` TTL ordering (1h must precede 5m) and max-4 cap.
    let mut ttls = Vec::new();
    collect_existing_cache_controls(root, &mut ttls);
    if ttls.len() > 4 && provider == ProviderFormat::AnthropicMessages {
        warnings.push(format!(
            "found {} existing cache_control breakpoints; Anthropic enforces a maximum of 4 breakpoints per request",
            ttls.len()
        ));
    }
    let mut seen_5m = false;
    for ttl in ttls {
        match ttl {
            CacheTtl::FiveMin => seen_5m = true,
            CacheTtl::OneHour => {
                if seen_5m {
                    warnings.push(
                        "invalid cache_control TTL order: a 1h breakpoint appears after a 5m breakpoint (longer TTLs must strictly precede shorter TTLs)"
                            .to_string(),
                    );
                    break;
                }
            }
        }
    }

    warnings
}

fn parse_cache_control_ttl(val: &JsonValue) -> Option<CacheTtl> {
    let cc = val.get("cache_control")?;
    if cc.as_object().is_none() {
        return None;
    }
    match cc.get("ttl").and_then(JsonValue::as_str) {
        Some("1h") => Some(CacheTtl::OneHour),
        _ => Some(CacheTtl::FiveMin),
    }
}

fn collect_existing_cache_controls(root: &JsonValue, out: &mut Vec<CacheTtl>) {
    if let Some(tools) = root.get("tools").and_then(JsonValue::as_array) {
        for tool in tools {
            if let Some(ttl) = parse_cache_control_ttl(tool) {
                out.push(ttl);
            }
        }
    }
    if let Some(sys) = root.get("system").and_then(JsonValue::as_array) {
        for item in sys {
            if let Some(ttl) = parse_cache_control_ttl(item) {
                out.push(ttl);
            }
        }
    }
    if let Some(msgs) = root.get("messages").and_then(JsonValue::as_array) {
        for msg in msgs {
            if let Some(ttl) = parse_cache_control_ttl(msg) {
                out.push(ttl);
            }
            if let Some(parts) = msg.get("content").and_then(JsonValue::as_array) {
                for part in parts {
                    if let Some(ttl) = parse_cache_control_ttl(part) {
                        out.push(ttl);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Block Extraction & Prefix Hashing
// ============================================================================

#[inline]
fn push_block_precomputed(
    blocks: &mut Vec<PrefixBlock>,
    tenant_key: &[u8; 16],
    running_prefix: &mut u64,
    path: String,
    kind: &str,
    canonical_bytes: Vec<u8>,
    block_hash: u64,
    breakpoint_eligible: bool,
) {
    *running_prefix = chain_hash64(tenant_key, *running_prefix, block_hash);
    let est_tokens = estimate_tokens_fast(&canonical_bytes);
    blocks.push(PrefixBlock {
        path,
        kind: kind.to_string(),
        canonical_bytes,
        block_hash,
        prefix_hash: *running_prefix,
        est_tokens,
        breakpoint_eligible,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_block(
    blocks: &mut Vec<PrefixBlock>,
    tenant_key: &[u8; 16],
    running_prefix: &mut u64,
    path: String,
    kind: &str,
    val: &JsonValue,
    breakpoint_eligible: bool,
    already_canonical: bool,
) {
    let canonical_bytes = canonical_bytes_without_cache_control(val, already_canonical);
    let block_hash = keyed_hash64(tenant_key, &canonical_bytes);
    push_block_precomputed(
        blocks,
        tenant_key,
        running_prefix,
        path,
        kind,
        canonical_bytes,
        block_hash,
        breakpoint_eligible,
    );
}

#[derive(Debug, Default, Clone, Copy)]
struct BlockRanges {
    tools_start: Option<usize>,
    system_start: Option<usize>,
    instructions_idx: Option<usize>,
}

/// Serializes an already-canonicalized root JSON object by copying pre-serialized
/// `canonical_bytes` from extracted `blocks` for `tools`, `system`, and `instructions`
/// (whenever an item has no top-level `cache_control` key), eliminating duplicate
/// serialization of 95%+ of the request payload.
fn serialize_root_with_block_cache(
    root: &JsonValue,
    blocks: &[PrefixBlock],
    mut ranges: BlockRanges,
    out: &mut Vec<u8>,
) {
    let entries = match root {
        JsonValue::Object(entries) => entries,
        other => {
            serialize_json(other, out);
            return;
        }
    };

    out.push(b'{');
    for (i, (k, v)) in entries.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        write_escaped_key_colon(k, out);
        match (k.as_str(), v) {
            ("tools", JsonValue::Array(tools)) if ranges.tools_start.is_some() => {
                let start = ranges.tools_start.take().unwrap_or(0);
                if start + tools.len() <= blocks.len() {
                    out.push(b'[');
                    for (ti, tool) in tools.iter().enumerate() {
                        if ti > 0 {
                            out.push(b',');
                        }
                        if tool.get("cache_control").is_none() {
                            out.extend_from_slice(&blocks[start + ti].canonical_bytes);
                        } else {
                            serialize_json(tool, out);
                        }
                    }
                    out.push(b']');
                } else {
                    serialize_json(v, out);
                }
            }
            ("system", JsonValue::Array(items)) if ranges.system_start.is_some() => {
                let start = ranges.system_start.take().unwrap_or(0);
                if start + items.len() <= blocks.len() {
                    out.push(b'[');
                    for (si, item) in items.iter().enumerate() {
                        if si > 0 {
                            out.push(b',');
                        }
                        if item.get("cache_control").is_none() {
                            out.extend_from_slice(&blocks[start + si].canonical_bytes);
                        } else {
                            serialize_json(item, out);
                        }
                    }
                    out.push(b']');
                } else {
                    serialize_json(v, out);
                }
            }
            ("system", other)
                if ranges.system_start.is_some()
                    && *other != JsonValue::Null
                    && other.get("cache_control").is_none() =>
            {
                let idx = ranges.system_start.take().unwrap_or(0);
                if let Some(b) = blocks.get(idx) {
                    out.extend_from_slice(&b.canonical_bytes);
                } else {
                    serialize_json(other, out);
                }
            }
            ("instructions", other)
                if ranges.instructions_idx.is_some()
                    && *other != JsonValue::Null
                    && other.get("cache_control").is_none() =>
            {
                let idx = ranges.instructions_idx.take().unwrap_or(0);
                if let Some(b) = blocks.get(idx) {
                    out.extend_from_slice(&b.canonical_bytes);
                } else {
                    serialize_json(other, out);
                }
            }
            _ => serialize_json(v, out),
        }
    }
    out.push(b'}');
}

fn extract_blocks(
    root: &JsonValue,
    provider: ProviderFormat,
    tenant_key: &[u8; 16],
    volatile_tail_loc: Option<VolatileTailLoc>,
    already_canonical: bool,
    mut cached_system_blocks: Option<Vec<CachedBlock>>,
) -> (Vec<PrefixBlock>, BlockRanges) {
    let est_cap = root
        .get("tools")
        .and_then(JsonValue::as_array)
        .map_or(0, |a| a.len())
        + root
            .get("system")
            .map_or(0, |s| s.as_array().map_or(1, |a| a.len()))
        + root
            .get("messages")
            .and_then(JsonValue::as_array)
            .map_or(0, |a| a.len() + 2);
    let mut blocks = Vec::with_capacity(est_cap.max(4));
    let mut ranges = BlockRanges::default();
    let mut running_prefix = 0u64;

    // 1. `tools`
    if let Some(tools) = root.get("tools").and_then(JsonValue::as_array) {
        ranges.tools_start = Some(blocks.len());
        for (i, tool) in tools.iter().enumerate() {
            push_block(
                &mut blocks,
                tenant_key,
                &mut running_prefix,
                fmt_indexed("tools[", i, "]"),
                "tools",
                tool,
                true,
                already_canonical,
            );
        }
    }

    // 2. `system` (Anthropic)
    if let Some(sys) = root.get("system") {
        match sys {
            JsonValue::Null => {}
            JsonValue::Array(items) => {
                ranges.system_start = Some(blocks.len());
                for (i, item) in items.iter().enumerate() {
                    let kind = if is_shared_context_block(item) {
                        "shared_doc"
                    } else {
                        "system"
                    };
                    let cached = cached_system_blocks
                        .as_mut()
                        .and_then(|v| v.get_mut(i))
                        .and_then(Option::take);
                    if let Some((canonical_bytes, block_hash)) = cached {
                        push_block_precomputed(
                            &mut blocks,
                            tenant_key,
                            &mut running_prefix,
                            fmt_indexed("system[", i, "]"),
                            kind,
                            canonical_bytes,
                            block_hash,
                            true,
                        );
                    } else {
                        push_block(
                            &mut blocks,
                            tenant_key,
                            &mut running_prefix,
                            fmt_indexed("system[", i, "]"),
                            kind,
                            item,
                            true,
                            already_canonical,
                        );
                    }
                }
            }
            other => {
                ranges.system_start = Some(blocks.len());
                push_block(
                    &mut blocks,
                    tenant_key,
                    &mut running_prefix,
                    "system".to_string(),
                    "system",
                    other,
                    true,
                    already_canonical,
                );
            }
        }
    }

    // 3. `instructions` (OpenAI Responses)
    if let Some(instr) = root.get("instructions") {
        if *instr != JsonValue::Null {
            ranges.instructions_idx = Some(blocks.len());
            push_block(
                &mut blocks,
                tenant_key,
                &mut running_prefix,
                "instructions".to_string(),
                "system",
                instr,
                true,
                already_canonical,
            );
        }
    }

    // 4. `messages`
    if let Some(msgs) = root.get("messages").and_then(JsonValue::as_array) {
        for (i, msg) in msgs.iter().enumerate() {
            let role = msg.get("role").and_then(JsonValue::as_str).unwrap_or("");
            let split_parts = match msg.get("content").and_then(JsonValue::as_array) {
                Some(parts) => {
                    let has_tail = matches!(
                        volatile_tail_loc,
                        Some(VolatileTailLoc::MessagePart(mi, _)) if mi == i
                    );
                    let has_shared = parts.iter().any(is_shared_context_block);
                    has_tail || has_shared || parts.len() > 1
                }
                None => false,
            };

            if split_parts {
                if let Some(parts) = msg.get("content").and_then(JsonValue::as_array) {
                    for (j, part) in parts.iter().enumerate() {
                        let is_tail =
                            volatile_tail_loc == Some(VolatileTailLoc::MessagePart(i, j));
                        let is_thinking = matches!(
                            part.get("type").and_then(JsonValue::as_str),
                            Some("thinking" | "redacted_thinking")
                        );
                        let (kind, eligible) = if is_tail {
                            ("volatile_tail", false)
                        } else if matches!(role, "system" | "developer") {
                            ("system", !is_thinking)
                        } else if is_shared_context_block(part) {
                            ("shared_doc", !is_thinking)
                        } else {
                            ("message", !is_thinking)
                        };
                        push_block(
                            &mut blocks,
                            tenant_key,
                            &mut running_prefix,
                            fmt_msg_part("messages[", i, "].content[", j),
                            kind,
                            part,
                            eligible,
                            already_canonical,
                        );
                    }
                }
            } else {
                let is_tail = volatile_tail_loc == Some(VolatileTailLoc::Message(i));
                let (kind, eligible) = if is_tail {
                    ("volatile_tail", false)
                } else if matches!(role, "system" | "developer") {
                    ("system", true)
                } else {
                    ("message", true)
                };
                push_block(
                    &mut blocks,
                    tenant_key,
                    &mut running_prefix,
                    fmt_indexed("messages[", i, "]"),
                    kind,
                    msg,
                    eligible,
                    already_canonical,
                );
            }
        }
    }

    // 5. `input` (OpenAI Responses)
    if provider == ProviderFormat::OpenAiResponses {
        if let Some(input) = root.get("input") {
            match input {
                JsonValue::Array(items) => {
                    for (i, item) in items.iter().enumerate() {
                        let role = item.get("role").and_then(JsonValue::as_str).unwrap_or("");
                        let is_tail = volatile_tail_loc == Some(VolatileTailLoc::InputItem(i));
                        let (kind, eligible) = if is_tail {
                            ("volatile_tail", false)
                        } else if matches!(role, "system" | "developer") {
                            ("system", true)
                        } else {
                            ("message", true)
                        };
                        push_block(
                            &mut blocks,
                            tenant_key,
                            &mut running_prefix,
                            fmt_indexed("input[", i, "]"),
                            kind,
                            item,
                            eligible,
                            already_canonical,
                        );
                    }
                }
                JsonValue::Null => {}
                other => {
                    push_block(
                        &mut blocks,
                        tenant_key,
                        &mut running_prefix,
                        "input".to_string(),
                        "message",
                        other,
                        true,
                        already_canonical,
                    );
                }
            }
        }
    }

    (blocks, ranges)
}

fn summarize_blocks(blocks: &[PrefixBlock], fallback_json: &[u8]) -> (u64, u64, u32) {
    if blocks.is_empty() {
        return (0, 0, estimate_tokens_fast(fallback_json));
    }

    let mut shared_hash = 0u64;
    let mut found_shared = false;
    for b in blocks {
        if matches!(b.kind.as_str(), "tools" | "system" | "shared_doc") {
            shared_hash = b.prefix_hash;
            found_shared = true;
        } else {
            break;
        }
    }
    if !found_shared {
        shared_hash = blocks[0].prefix_hash;
    }

    let full_hash = blocks.last().map(|b| b.prefix_hash).unwrap_or(0);
    let total_tokens = blocks
        .iter()
        .fold(0u32, |acc, b| acc.saturating_add(b.est_tokens));
    (shared_hash, full_hash, total_tokens)
}

// ============================================================================
// Breakpoint Application to Wire JSON
// ============================================================================

/// Mutates an already-canonicalized `root` [`JsonValue`] in place to apply a
/// [`BreakpointPlan`] (and optional OpenAI `prompt_cache_key`) while preserving
/// lexicographical key order via [`JsonValue::set_field_sorted`], then
/// serializes once into a pre-sized wire-format JSON buffer with zero re-parsing.
pub fn apply_breakpoints_to_value(
    root: &mut JsonValue,
    provider: ProviderFormat,
    plan: &BreakpointPlan,
    prompt_cache_key: Option<&str>,
) -> Vec<u8> {
    match provider {
        ProviderFormat::AnthropicMessages => {
            clear_all_cache_controls(root);

            let mut selected: Vec<&PlannedBreakpoint> = plan.breakpoints.iter().collect();
            selected.sort_by_key(|bp| bp.block_index);
            selected.dedup_by_key(|bp| bp.block_index);
            selected.truncate(4);

            // Enforce Anthropic mixed-TTL invariant: 1h breakpoints must precede 5m.
            let mut seen_5m = false;
            for bp in selected {
                let effective_ttl = match bp.ttl {
                    CacheTtl::OneHour if seen_5m => CacheTtl::FiveMin,
                    CacheTtl::OneHour => CacheTtl::OneHour,
                    CacheTtl::FiveMin => {
                        seen_5m = true;
                        CacheTtl::FiveMin
                    }
                };
                let cc = make_anthropic_cache_control(effective_ttl);
                inject_anthropic_breakpoint(root, &bp.path, cc);
            }
        }
        ProviderFormat::OpenAiChat
        | ProviderFormat::OpenAiResponses
        | ProviderFormat::VllmOpenAi
        | ProviderFormat::Generic => {
            if let Some(key) = prompt_cache_key {
                if !key.is_empty() {
                    root.set_field_sorted("prompt_cache_key", JsonValue::String(key.to_string()));
                }
            }
        }
    }

    let mut out = Vec::with_capacity(estimate_serialized_len(root) + 32);
    serialize_json(root, &mut out);
    out
}

/// Applies a [`BreakpointPlan`] (and optional OpenAI `prompt_cache_key`) onto
/// `canonical_json`, returning canonical wire-format JSON ready to forward upstream.
///
/// - For `ProviderFormat::AnthropicMessages`: strips stale `cache_control`
///   markers, enforces the 4-breakpoint limit and 1h-before-5m TTL monotonicity,
///   converts plain string `system` (or string message `content`) into content
///   block arrays when targeted, and injects `{"ttl": "1h"|"5m", "type": "ephemeral"}`.
/// - For OpenAI / vLLM / Generic providers: sets top-level `prompt_cache_key`
///   when `prompt_cache_key` is `Some` and non-empty.
pub fn apply_breakpoints_to_json(
    canonical_json: &[u8],
    provider: ProviderFormat,
    plan: &BreakpointPlan,
    prompt_cache_key: Option<&str>,
) -> Result<Vec<u8>, String> {
    let mut root = parse_json(canonical_json)?;
    if root.as_object().is_none() {
        return Err("canonical_json root must be a JSON object".to_string());
    }
    let _ = root.canonicalize_in_place();
    Ok(apply_breakpoints_to_value(
        &mut root,
        provider,
        plan,
        prompt_cache_key,
    ))
}

fn make_anthropic_cache_control(ttl: CacheTtl) -> JsonValue {
    JsonValue::Object(vec![
        (
            "ttl".to_string(),
            JsonValue::String(ttl.as_anthropic_str().to_string()),
        ),
        (
            "type".to_string(),
            JsonValue::String("ephemeral".to_string()),
        ),
    ])
}

fn clear_all_cache_controls(root: &mut JsonValue) {
    if let Some(tools) = root.get_mut("tools").and_then(JsonValue::as_array_mut) {
        for tool in tools.iter_mut() {
            tool.remove_field("cache_control");
        }
    }
    if let Some(sys) = root.get_mut("system").and_then(JsonValue::as_array_mut) {
        for item in sys.iter_mut() {
            item.remove_field("cache_control");
        }
    }
    if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
        for msg in msgs.iter_mut() {
            msg.remove_field("cache_control");
            if let Some(parts) = msg.get_mut("content").and_then(JsonValue::as_array_mut) {
                for part in parts.iter_mut() {
                    part.remove_field("cache_control");
                }
            }
        }
    }
}

fn parse_bracket_index(s: &str, prefix: &str) -> Option<usize> {
    let rest = s.strip_prefix(prefix)?;
    let num = rest.strip_suffix(']')?;
    num.parse::<usize>().ok()
}

fn parse_message_content_path(path: &str) -> Option<(usize, usize)> {
    let rest = path.strip_prefix("messages[")?;
    let sep = rest.find("].content[")?;
    let msg_idx = rest[..sep].parse::<usize>().ok()?;
    let part_str = rest[sep + "].content[".len()..].strip_suffix(']')?;
    let part_idx = part_str.parse::<usize>().ok()?;
    Some((msg_idx, part_idx))
}

fn inject_anthropic_breakpoint(root: &mut JsonValue, path: &str, cc: JsonValue) {
    if path == "system" {
        match root.get_mut("system") {
            Some(JsonValue::String(s)) => {
                let text = std::mem::take(s);
                let block = JsonValue::Object(vec![
                    ("cache_control".to_string(), cc),
                    ("text".to_string(), JsonValue::String(text)),
                    ("type".to_string(), JsonValue::String("text".to_string())),
                ]);
                root.set_field_sorted("system", JsonValue::Array(vec![block]));
            }
            Some(JsonValue::Array(arr)) => {
                if let Some(last) = arr.last_mut() {
                    last.set_field_sorted("cache_control", cc);
                }
            }
            _ => {}
        }
        return;
    }

    if let Some(idx) = parse_bracket_index(path, "system[") {
        match root.get_mut("system") {
            Some(JsonValue::Array(arr)) => {
                if let Some(item) = arr.get_mut(idx) {
                    item.set_field_sorted("cache_control", cc);
                }
            }
            Some(JsonValue::String(s)) if idx == 0 => {
                let text = std::mem::take(s);
                let block = JsonValue::Object(vec![
                    ("cache_control".to_string(), cc),
                    ("text".to_string(), JsonValue::String(text)),
                    ("type".to_string(), JsonValue::String("text".to_string())),
                ]);
                root.set_field_sorted("system", JsonValue::Array(vec![block]));
            }
            _ => {}
        }
        return;
    }

    if let Some(idx) = parse_bracket_index(path, "tools[") {
        if let Some(tools) = root.get_mut("tools").and_then(JsonValue::as_array_mut) {
            if let Some(tool) = tools.get_mut(idx) {
                tool.set_field_sorted("cache_control", cc);
            }
        }
        return;
    }

    if let Some((msg_idx, part_idx)) = parse_message_content_path(path) {
        if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
            if let Some(msg) = msgs.get_mut(msg_idx) {
                if let Some(parts) = msg.get_mut("content").and_then(JsonValue::as_array_mut) {
                    if let Some(part) = parts.get_mut(part_idx) {
                        part.set_field_sorted("cache_control", cc);
                    }
                }
            }
        }
        return;
    }

    if let Some(msg_idx) = parse_bracket_index(path, "messages[") {
        if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
            if let Some(msg) = msgs.get_mut(msg_idx) {
                match msg.get_mut("content") {
                    Some(JsonValue::String(s)) => {
                        let text = std::mem::take(s);
                        let block = JsonValue::Object(vec![
                            ("cache_control".to_string(), cc),
                            ("text".to_string(), JsonValue::String(text)),
                            ("type".to_string(), JsonValue::String("text".to_string())),
                        ]);
                        msg.set_field_sorted("content", JsonValue::Array(vec![block]));
                    }
                    Some(JsonValue::Array(parts)) => {
                        if let Some(target) = parts.iter_mut().rev().find(|p| {
                            !matches!(
                                p.get("type").and_then(JsonValue::as_str),
                                Some("thinking" | "redacted_thinking")
                            )
                        }) {
                            target.set_field_sorted("cache_control", cc);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_auto_detection() {
        let compiler = PrefixCompiler::default();

        let anth = br#"{"model":"claude-sonnet-4","system":"You are helpful.","messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(
            compiler.compile_request("t1", anth, None).unwrap().provider,
            ProviderFormat::AnthropicMessages
        );

        let resp = br#"{"model":"gpt-4.1","instructions":"Be concise.","input":"Explain SIMD."}"#;
        assert_eq!(
            compiler.compile_request("t1", resp, None).unwrap().provider,
            ProviderFormat::OpenAiResponses
        );

        let chat = br#"{"model":"gpt-4o","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"Hi"}]}"#;
        assert_eq!(
            compiler.compile_request("t1", chat, None).unwrap().provider,
            ProviderFormat::OpenAiChat
        );
    }

    #[test]
    fn test_pass1_tool_sorting_and_json_key_canonicalization() {
        let compiler = PrefixCompiler::new(CompilerConfig {
            mode: CanonicalMode::JsonCanonical,
            ..CompilerConfig::default()
        });

        // Two semantically identical Anthropic payloads with scrambled tools and keys.
        let req_a = br#"{
            "messages": [{"content": "Check weather", "role": "user"}],
            "tools": [
                {"name": "zebra_tool", "input_schema": {"type": "object", "properties": {"z": {"type": "string"}, "a": {"type": "number"}}}},
                {"name": "alpha_tool", "input_schema": {"properties": {"b": {"type": "boolean"}}, "type": "object"}}
            ],
            "model": "claude-sonnet-4",
            "system": "System instructions"
        }"#;

        let req_b = br#"{
            "model": "claude-sonnet-4",
            "system": "System instructions",
            "tools": [
                {"input_schema": {"type": "object", "properties": {"b": {"type": "boolean"}}}, "name": "alpha_tool"},
                {"input_schema": {"properties": {"a": {"type": "number"}, "z": {"type": "string"}}, "type": "object"}, "name": "zebra_tool"}
            ],
            "messages": [{"role": "user", "content": "Check weather"}]
        }"#;

        let res_a = compiler.compile_request("tenant-1", req_a, None).unwrap();
        let res_b = compiler.compile_request("tenant-1", req_b, None).unwrap();

        assert!(res_a.tools_reordered);
        assert!(res_a.json_keys_reordered);
        assert!(!res_b.tools_reordered);
        assert!(res_b.json_keys_reordered);
        assert_eq!(res_a.canonical_json, res_b.canonical_json);
        assert_eq!(res_a.shared_prefix_hash, res_b.shared_prefix_hash);
        assert_eq!(res_a.full_prefix_hash, res_b.full_prefix_hash);

        // OpenAI Chat nested `function.name` tool sorting.
        let oai_a = br#"{
            "model": "gpt-4o",
            "tools": [
                {"type": "function", "function": {"name": "search_docs", "parameters": {}}},
                {"type": "function", "function": {"name": "calc_math", "parameters": {}}}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let oai_res = compiler.compile_request("tenant-1", oai_a, None).unwrap();
        assert!(oai_res.tools_reordered);
        let parsed = parse_json(&oai_res.canonical_json).unwrap();
        let tools = parsed.get("tools").and_then(JsonValue::as_array).unwrap();
        assert_eq!(extract_tool_name(&tools[0]), "calc_math");
        assert_eq!(extract_tool_name(&tools[1]), "search_docs");
    }

    #[test]
    fn test_pass2_lossless_bijection_and_prompt_injection_safety() {
        let compiler = PrefixCompiler::new(CompilerConfig {
            mode: CanonicalMode::HoistVolatile,
            ..CompilerConfig::default()
        });

        // Adversarial system prompt containing:
        // - Literal `⟦cp:v1⟧` & `⟦cp:esc:v1⟧` & `⟦cp:t1⟧` spoofing attempts
        // - Sentinel immediately adjacent to an ISO timestamp
        // - UUID, prefixed opaque IDs, inline `Current time:`, `Date:`, `Timestamp:`
        // - Regular English snake_case identifiers that MUST NOT be hoisted!
        let original_sys = concat!(
            "Spoof ⟦cp:v1⟧ and ⟦cp:esc:v1⟧ and ⟦cp:t1⟧ and ⟦cp:2026-11-22T15:48:57Z. ",
            "Current time: Sun, 22 Nov 2026 15:48:57 GMT\n",
            "Date: 2026-11-22\n",
            "Timestamp: 1732290537\n",
            "UUID: 550e8400-e29b-41d4-a716-446655440000 ",
            "IDs: req_a1b2c3d4e5f6 trace_99887766aabb span_1234abcd run_01J9X8K2M4 ",
            "session_9f8e7d6c5b4a call_77889900aabb agent_run_00112233aabb. ",
            "Keep code identifiers: run_benchmark, session_manager, call_handler."
        );

        let payload = JsonValue::Object(vec![
            ("model".to_string(), JsonValue::String("claude-sonnet-4".to_string())),
            ("system".to_string(), JsonValue::String(original_sys.to_string())),
            (
                "messages".to_string(),
                JsonValue::Array(vec![JsonValue::Object(vec![
                    ("role".to_string(), JsonValue::String("user".to_string())),
                    (
                        "content".to_string(),
                        JsonValue::String("Final user turn 2026-11-22T23:59:59Z stays untouched".to_string()),
                    ),
                ])]),
            ),
        ]);
        let mut raw = Vec::new();
        serialize_canonical(&payload, &mut raw);

        let compiled = compiler.compile_request("tenant-sec", &raw, None).unwrap();
        assert_eq!(compiled.volatile_spans_hoisted, 12);
        assert_eq!(compiled.bindings.len(), 12);

        let parsed = parse_json(&compiled.canonical_json).unwrap();
        let canonical_sys = parsed.get("system").and_then(JsonValue::as_str).unwrap();

        // Untrusted literals were escaped, not confused with real anchors.
        assert!(canonical_sys.contains("⟦cp:esc:v1⟧"));
        assert!(canonical_sys.contains("⟦cp:esc:esc:v1⟧"));
        assert!(canonical_sys.contains("⟦cp:esc:t1⟧"));
        // Non-volatile English snake_case identifiers stayed intact.
        assert!(canonical_sys.contains("run_benchmark"));
        assert!(canonical_sys.contains("session_manager"));
        assert!(canonical_sys.contains("call_handler"));

        // Exact byte-for-byte lossless reconstruction!
        let restored = reconstruct_lossless(canonical_sys, &compiled.bindings);
        assert_eq!(restored, original_sys);

        // Final user turn timestamp was NOT hoisted; trailing volatile_tail block exists and is ineligible.
        let msgs = parsed.get("messages").and_then(JsonValue::as_array).unwrap();
        let user_parts = msgs[0].get("content").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            user_parts[0].get("text").and_then(JsonValue::as_str),
            Some("Final user turn 2026-11-22T23:59:59Z stays untouched")
        );
        let tail_block = compiled.blocks.last().unwrap();
        assert_eq!(tail_block.kind, "volatile_tail");
        assert!(!tail_block.breakpoint_eligible);
    }

    #[test]
    fn test_pass3_all_gather_multi_agent_convergence_and_turn_safety() {
        let compiler = PrefixCompiler::new(CompilerConfig {
            mode: CanonicalMode::SwarmAlign,
            all_gather_delimiter: Some("\n---CHUNK---\n".to_string()),
            ..CompilerConfig::default()
        });

        // 3 concurrent agents read the same 3 shared context docs in different
        // permutations AND with different per-agent timestamps/request IDs!
        let agent1 = br#"{
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": "<doc id='B'>Shared repo index req_111122223333</doc>"},
                {"type": "text", "text": "<doc id='A'>Shared architecture spec 2026-11-22T10:00:01Z</doc>"},
                {"type": "text", "text": "<doc id='C'>Shared style guide</doc>"}
            ],
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Ack 1"},
                {"role": "user", "content": "Agent 1 question"}
            ]
        }"#;

        let agent2 = br#"{
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": "<doc id='C'>Shared style guide</doc>"},
                {"type": "text", "text": "<doc id='B'>Shared repo index req_999988887777</doc>"},
                {"type": "text", "text": "<doc id='A'>Shared architecture spec 2026-11-22T10:05:59Z</doc>"}
            ],
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Ack 1"},
                {"role": "user", "content": "Agent 2 question"}
            ]
        }"#;

        let r1 = compiler.compile_request("swarm-t", agent1, None).unwrap();
        let r2 = compiler.compile_request("swarm-t", agent2, None).unwrap();

        // Both agents converge onto the EXACT same shared_prefix_hash!
        assert_eq!(r1.shared_prefix_hash, r2.shared_prefix_hash);
        assert_eq!(r1.blocks[0].canonical_bytes, r2.blocks[0].canonical_bytes);
        assert_eq!(r1.blocks[1].canonical_bytes, r2.blocks[1].canonical_bytes);
        assert_eq!(r1.blocks[2].canonical_bytes, r2.blocks[2].canonical_bytes);

        // Agent 3 carries 2 shared docs + 1 unique private doc: shared docs must
        // hoist strictly ahead of the private doc because their swarm freq bucket > 0!
        let agent3 = br#"{
            "model": "claude-sonnet-4",
            "system": [
                {"type": "text", "text": "<doc id='PRIVATE'>Agent 3 unique scratchpad</doc>"},
                {"type": "text", "text": "<doc id='A'>Shared architecture spec 2026-11-22T10:09:00Z</doc>"},
                {"type": "text", "text": "<doc id='C'>Shared style guide</doc>"}
            ],
            "messages": [
                {"role": "user", "content": "Turn 1"},
                {"role": "assistant", "content": "Ack 1"},
                {"role": "user", "content": "Agent 3 question"}
            ]
        }"#;
        let r3 = compiler.compile_request("swarm-t", agent3, None).unwrap();
        let p3 = parse_json(&r3.canonical_json).unwrap();
        let sys3 = p3.get("system").and_then(JsonValue::as_array).unwrap();
        let last_sys_text = sys3[2].get("text").and_then(JsonValue::as_str).unwrap();
        assert!(
            last_sys_text.contains("PRIVATE"),
            "private doc should sink after hot shared docs, got: {last_sys_text}"
        );

        // Sequential conversation turns were NEVER reordered.
        let msgs3 = p3.get("messages").and_then(JsonValue::as_array).unwrap();
        assert_eq!(msgs3[0].get("content").and_then(JsonValue::as_str), Some("Turn 1"));
        assert_eq!(msgs3[1].get("content").and_then(JsonValue::as_str), Some("Ack 1"));

        // Explicit delimiter-chunk alignment converges across agents.
        let d1 = br#"{"model":"gpt-4o","messages":[{"role":"system","content":"chunkB\n---CHUNK---\nchunkA\n---CHUNK---\nchunkC"},{"role":"user","content":"go"}]}"#;
        let d2 = br#"{"model":"gpt-4o","messages":[{"role":"system","content":"chunkC\n---CHUNK---\nchunkB\n---CHUNK---\nchunkA"},{"role":"user","content":"go"}]}"#;
        let rd1 = compiler.compile_request("swarm-d", d1, None).unwrap();
        let rd2 = compiler.compile_request("swarm-d", d2, None).unwrap();
        assert_eq!(rd1.canonical_json, rd2.canonical_json);
    }

    #[test]
    fn test_pass4_config_guard_fingerprint_and_warnings() {
        let compiler = PrefixCompiler::default();

        let bad_cfg = br#"{
            "model": "claude-sonnet-4",
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "temperature": 0.5,
            "tool_choice": {"type": "none"},
            "system": [
                {"type": "text", "text": "Part 1", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                {"type": "text", "text": "Part 2", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "ls", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}]}
            ]
        }"#;

        let r = compiler.compile_request("t1", bad_cfg, None).unwrap();
        assert!(
            r.config_warnings.len() >= 4,
            "expected warnings for tool_choice=none, missing tools, thinking+temp, and 5m->1h order; got {:?}",
            r.config_warnings
        );

        let clean_a = br#"{"model":"claude-sonnet-4","temperature":0.2,"system":"ok","messages":[{"role":"user","content":"hi"}]}"#;
        let clean_b = br#"{"model":"claude-sonnet-4","temperature":0.9,"system":"ok","messages":[{"role":"user","content":"hi"}]}"#;
        let ra = compiler.compile_request("t1", clean_a, None).unwrap();
        let rb = compiler.compile_request("t1", clean_b, None).unwrap();
        assert!(ra.config_warnings.is_empty());
        assert_ne!(ra.config_fingerprint, rb.config_fingerprint);
    }

    #[test]
    fn test_apply_breakpoints_anthropic_and_openai() {
        // Anthropic: plain string `system` must convert to content block array,
        // stale markers cleared, and mixed-TTL 1h-before-5m enforced.
        let anth = br#"{
            "messages": [
                {"content": "Hello earlier", "role": "user"},
                {"content": "Reply", "role": "assistant"},
                {"content": [{"cache_control": {"type": "ephemeral"}, "text": "Latest", "type": "text"}], "role": "user"}
            ],
            "model": "claude-sonnet-4",
            "system": "You are an expert systems engineer.",
            "tools": [{"input_schema": {"type": "object"}, "name": "read_file"}]
        }"#;

        let plan = BreakpointPlan {
            breakpoints: vec![
                PlannedBreakpoint {
                    block_index: 0,
                    path: "tools[0]".to_string(),
                    cumulative_tokens: 500,
                    ttl: CacheTtl::OneHour,
                },
                PlannedBreakpoint {
                    block_index: 1,
                    path: "system".to_string(),
                    cumulative_tokens: 1500,
                    ttl: CacheTtl::FiveMin,
                },
                PlannedBreakpoint {
                    block_index: 2,
                    path: "messages[0]".to_string(),
                    cumulative_tokens: 2200,
                    // Intentionally out-of-order 1h after 5m to verify defensive downgrade.
                    ttl: CacheTtl::OneHour,
                },
            ],
            expected_cost_usd: 0.01,
            uncached_cost_usd: 0.05,
            naive_single_bp_cost_usd: 0.02,
            expected_savings_vs_uncached_pct: 80.0,
            keepalive_recommended: false,
            keepalive_interval_secs: None,
        };

        let out = apply_breakpoints_to_json(
            anth,
            ProviderFormat::AnthropicMessages,
            &plan,
            Some("ignored-for-anthropic"),
        )
        .unwrap();
        let parsed = parse_json(&out).unwrap();

        // Must NOT inject OpenAI prompt_cache_key onto Anthropic payload.
        assert!(parsed.get("prompt_cache_key").is_none());

        // tools[0] got 1h cache_control.
        let tool0_cc = parsed
            .get("tools")
            .and_then(JsonValue::as_array)
            .unwrap()[0]
            .get("cache_control")
            .unwrap();
        assert_eq!(tool0_cc.get("ttl").and_then(JsonValue::as_str), Some("1h"));
        assert_eq!(
            tool0_cc.get("type").and_then(JsonValue::as_str),
            Some("ephemeral")
        );

        // Plain string `system` was converted to array with 5m cache_control.
        let sys_arr = parsed.get("system").and_then(JsonValue::as_array).unwrap();
        assert_eq!(sys_arr.len(), 1);
        assert_eq!(
            sys_arr[0].get("type").and_then(JsonValue::as_str),
            Some("text")
        );
        assert_eq!(
            sys_arr[0].get("text").and_then(JsonValue::as_str),
            Some("You are an expert systems engineer.")
        );
        assert_eq!(
            sys_arr[0]
                .get("cache_control")
                .and_then(|c| c.get("ttl"))
                .and_then(JsonValue::as_str),
            Some("5m")
        );

        // messages[0] converted to array and downgraded 1h->5m after prior 5m.
        let msgs = parsed.get("messages").and_then(JsonValue::as_array).unwrap();
        let m0_parts = msgs[0].get("content").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            m0_parts[0]
                .get("cache_control")
                .and_then(|c| c.get("ttl"))
                .and_then(JsonValue::as_str),
            Some("5m")
        );

        // Stale cache_control on messages[2].content[0] was cleared.
        let m2_parts = msgs[2].get("content").and_then(JsonValue::as_array).unwrap();
        assert!(m2_parts[0].get("cache_control").is_none());

        // OpenAI Chat: sets prompt_cache_key cleanly.
        let oai = br#"{"messages":[{"content":"hi","role":"user"}],"model":"gpt-4o"}"#;
        let oai_out = apply_breakpoints_to_json(
            oai,
            ProviderFormat::OpenAiChat,
            &plan,
            Some("cp-tenant1-9f8e7d6c"),
        )
        .unwrap();
        let oai_parsed = parse_json(&oai_out).unwrap();
        assert_eq!(
            oai_parsed.get("prompt_cache_key").and_then(JsonValue::as_str),
            Some("cp-tenant1-9f8e7d6c")
        );
    }

    #[test]
    fn test_swarm_stats_bounded_capacity_and_eviction() {
        let stats = SwarmBlockStats::new();
        // Seed hot entries seen multiple times.
        for _ in 0..5 {
            stats.record_and_score(1, 777);
            stats.record_and_score(1, 888);
        }
        // Push past MAX_SWARM_ENTRIES (65_536) to trigger half-decay + LRU eviction.
        for i in 10_000..(10_000 + MAX_SWARM_ENTRIES as u64 + 500) {
            let s = stats.record_and_score(1, i);
            assert!(s >= 1.0);
        }
        assert!(stats.len() <= MAX_SWARM_ENTRIES);
        // Hot keys survived eviction with positive decayed score.
        assert!(stats.get_score(1, 777) > 1.0);
        assert!(stats.get_score(1, 888) > 1.0);
        assert!(swarm_freq_bucket(stats.get_score(1, 777)) >= 1);

        stats.clear();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_user_doc_all_gather_preserves_question_and_tool_turns() {
        let compiler = PrefixCompiler::default();

        let req1 = br#"{
            "model": "claude-sonnet-4",
            "system": "Analyzer",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "fetch", "input": {"k": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result_1"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "<doc id='2'>Second shared file</doc>"},
                    {"type": "text", "text": "<doc id='1'>First shared file</doc>"},
                    {"type": "text", "text": "Now compare both files."}
                ]}
            ]
        }"#;

        let req2 = br#"{
            "model": "claude-sonnet-4",
            "system": "Analyzer",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "fetch", "input": {"k": 1}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result_1"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "<doc id='1'>First shared file</doc>"},
                    {"type": "text", "text": "<doc id='2'>Second shared file</doc>"},
                    {"type": "text", "text": "Now compare both files."}
                ]}
            ]
        }"#;

        let c1 = compiler.compile_request("t-doc", req1, None).unwrap();
        let c2 = compiler.compile_request("t-doc", req2, None).unwrap();
        assert_eq!(c1.canonical_json, c2.canonical_json);
        assert_eq!(c1.full_prefix_hash, c2.full_prefix_hash);

        let parsed = parse_json(&c1.canonical_json).unwrap();
        let msgs = parsed.get("messages").and_then(JsonValue::as_array).unwrap();
        // Tool use and tool result remain strictly in positions 0 and 1.
        assert_eq!(
            msgs[0].get("content").and_then(JsonValue::as_array).unwrap()[0]
                .get("type")
                .and_then(JsonValue::as_str),
            Some("tool_use")
        );
        assert_eq!(
            msgs[1].get("content").and_then(JsonValue::as_array).unwrap()[0]
                .get("type")
                .and_then(JsonValue::as_str),
            Some("tool_result")
        );
        // Trailing user prompt remains strictly last after the aligned `<doc>` blocks.
        let parts = msgs[2].get("content").and_then(JsonValue::as_array).unwrap();
        assert_eq!(
            parts[2].get("text").and_then(JsonValue::as_str),
            Some("Now compare both files.")
        );
    }

    #[test]
    fn test_observe_only_and_openai_responses_and_max4_breakpoints() {
        let obs_compiler = PrefixCompiler::new(CompilerConfig {
            mode: CanonicalMode::ObserveOnly,
            ..CompilerConfig::default()
        });
        let raw = br#"{ "z": 1, "model": "gpt-4o", "messages": [{"role":"user","content":"hi"}] }"#;
        let obs = obs_compiler.compile_request("t1", raw, None).unwrap();
        assert_eq!(obs.canonical_json, raw);
        assert!(!obs.json_keys_reordered);
        assert!(!obs.tools_reordered);
        assert!(!obs.all_gather_reordered);
        assert_eq!(obs.volatile_spans_hoisted, 0);

        // OpenAI Responses API hoisting + prompt_cache_key injection.
        let compiler = PrefixCompiler::default();
        let resp_req = br#"{
            "model": "gpt-4.1",
            "instructions": "Run id: run_01J9X8K2M4P5 at 2026-11-22T15:48:57Z",
            "input": "Summarize the logs."
        }"#;
        let compiled = compiler.compile_request("t1", resp_req, None).unwrap();
        assert_eq!(compiled.provider, ProviderFormat::OpenAiResponses);
        assert_eq!(compiled.volatile_spans_hoisted, 2);
        let parsed = parse_json(&compiled.canonical_json).unwrap();
        let instr = parsed.get("instructions").and_then(JsonValue::as_str).unwrap();
        assert_eq!(
            reconstruct_lossless(instr, &compiled.bindings),
            "Run id: run_01J9X8K2M4P5 at 2026-11-22T15:48:57Z"
        );
        assert_eq!(compiled.blocks.last().unwrap().kind, "volatile_tail");
        assert!(!compiled.blocks.last().unwrap().breakpoint_eligible);

        // Verify Anthropic breakpoint injection caps strictly at 4 even if plan has 5.
        let five_blocks = br#"{
            "model": "claude-sonnet-4",
            "system": [
                {"type":"text","text":"s0"},
                {"type":"text","text":"s1"},
                {"type":"text","text":"s2"},
                {"type":"text","text":"s3"},
                {"type":"text","text":"s4"}
            ],
            "messages": [{"role":"user","content":"go"}]
        }"#;
        let over_plan = BreakpointPlan {
            breakpoints: (0..5)
                .map(|i| PlannedBreakpoint {
                    block_index: i,
                    path: format!("system[{i}]"),
                    cumulative_tokens: ((i + 1) * 500) as u32,
                    ttl: CacheTtl::OneHour,
                })
                .collect(),
            expected_cost_usd: 0.01,
            uncached_cost_usd: 0.05,
            naive_single_bp_cost_usd: 0.02,
            expected_savings_vs_uncached_pct: 80.0,
            keepalive_recommended: false,
            keepalive_interval_secs: None,
        };
        let injected = apply_breakpoints_to_json(
            five_blocks,
            ProviderFormat::AnthropicMessages,
            &over_plan,
            None,
        )
        .unwrap();
        let inj_val = parse_json(&injected).unwrap();
        let sys = inj_val.get("system").and_then(JsonValue::as_array).unwrap();
        let cc_count = sys.iter().filter(|b| b.get("cache_control").is_some()).count();
        assert_eq!(cc_count, 4);
    }

    fn naive_find_volatile_spans_ref(
        text: &str,
        max_spans: usize,
    ) -> Vec<(usize, usize, &'static str)> {
        if max_spans == 0 || text.is_empty() {
            return Vec::new();
        }
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut spans = Vec::new();
        let mut i = 0usize;

        while i < len && spans.len() < max_spans {
            let b = bytes[i];
            if b >= 0x80 {
                i += utf8_char_len(b);
                continue;
            }
            if i + 5 < len {
                let next = bytes[i + 1] | 0x20;
                if matches!(
                    (b | 0x20, next),
                    (b'c', b'u') | (b't', b'i') | (b'd', b'a')
                ) {
                    if let Some((val_start, val_end, kind)) = match_inline_header(text, bytes, i) {
                        spans.push((val_start, val_end, kind));
                        i = val_end;
                        continue;
                    }
                }
            }
            if (b == b'1' || b == b'2')
                && i + 19 <= len
                && bytes[i + 4] == b'-'
                && bytes[i + 7] == b'-'
                && bytes[i + 13] == b':'
            {
                if let Some(end) = match_iso_timestamp(bytes, i) {
                    spans.push((i, end, "timestamp"));
                    i = end;
                    continue;
                }
            }
            if b.is_ascii_hexdigit()
                && i + 36 <= len
                && bytes[i + 8] == b'-'
                && bytes[i + 13] == b'-'
                && bytes[i + 18] == b'-'
                && bytes[i + 23] == b'-'
            {
                if let Some(end) = match_uuid(bytes, i) {
                    spans.push((i, end, "uuid"));
                    i = end;
                    continue;
                }
            }
            if i + 12 <= len {
                let maybe_prefix = match b {
                    b'r' => bytes[i + 3] == b'_',
                    b'c' => bytes[i + 4] == b'_',
                    b't' => bytes[i + 5] == b'_',
                    b's' => bytes[i + 4] == b'_' || (i + 7 < len && bytes[i + 7] == b'_'),
                    b'a' => i + 9 < len && bytes[i + 9] == b'_',
                    _ => false,
                };
                if maybe_prefix {
                    if let Some(end) = match_prefixed_id(bytes, i) {
                        spans.push((i, end, "opaque_id"));
                        i = end;
                        continue;
                    }
                }
            }
            i += 1;
        }
        spans
    }

    #[test]
    fn test_simd_find_volatile_spans_equivalence_vs_naive_reference() {
        let cases = [
            "",
            "Plain ASCII text with no delimiters at all and plenty of padding over sixty four bytes long.",
            "2026-11-22T15:48:57Z",
            "550e8400-e29b-41d4-a716-446655440000",
            "req_a1b2c3d4e5f6 run_01J9X8K2M4 span_1234abcd call_77889900aabb trace_99887766aabb session_9f8e7d6c5b4a agent_run_00112233aabb",
            "Current UTC time: 2026-11-22T15:48:57.123456Z UTC | Current time: Sun, 22 Nov 2026 15:48:57 GMT | Current date: 2026-11-22 | Timestamp: 1732290537 | Date: 2026-11-22",
            // Boundary & precedence tricks:
            "xCurrent date: 2026-11-22 xCurrent utc time: 2026-11-22T15:48:57Z foo:Date: 2026/11/22",
            "Date:2026-11-22Current time: 1732290537;req_112233445566-550e8400-e29b-41d4-a716-446655440000",
            "foo_agent_run_00112233aabb xagent_run_00112233aabb agent_run_short run_benchmark session_manager",
            // Multi-byte UTF-8 directly before every volatile pattern type:
            "🚀2026-11-22T15:48:57Z €550e8400-e29b-41d4-a716-446655440000 中req_a1b2c3d4e5f6 日本語Date: 2026-11-22 ⟦cp:2026-11-22T15:48:57Z",
            "日本語Current UTC time: 2026-11-22T12:00:00Z — こんにちはagent_run_1234567890ab 😀session_deadbeef1234",
            // Frequent false-positive delimiters (Rust code + prose):
            "SHARED_REPO_FILE_00: pub fn invariant_check(x: u64) -> bool { x.rotate_left(13) != 0 } lock-free zero-allocation",
            // Back-to-back timestamps, UUIDs, headers, sentinels:
            "2026-11-22T10:00:00Z,2026-11-22T10:00:01Z|550e8400-e29b-41d4-a716-446655440000:550e8400-e29b-41d4-a716-446655440001",
            "⟦cp:v1⟧ ⟦cp:esc:t1⟧ Current time: no_digits_here\n2026-11-22T15:00:00Z Date: nope 2026-11-22T16:00:00+02:00",
        ];

        for base in cases {
            // Test every alignment padding 0..=40 (crossing 16- and 32-byte SIMD boundaries).
            for pad in 0..=40 {
                let mut padded = " ".repeat(pad);
                padded.push_str(base);
                for cap in [0usize, 1, 2, 5, 64] {
                    let got = find_volatile_spans(&padded, cap);
                    let expected = naive_find_volatile_spans_ref(&padded, cap);
                    assert_eq!(
                        got, expected,
                        "SIMD vs naive mismatch (pad={pad}, cap={cap}) on {padded:?}"
                    );
                }
                // Also verify sentinel detection matches `contains(SENTINEL_PREFIX)` when no spans cap early.
                let (spans, has_sent) = find_volatile_spans_and_sentinel(&padded, 64);
                if spans.is_empty() {
                    assert_eq!(
                        has_sent,
                        padded.contains(SENTINEL_PREFIX),
                        "sentinel flag mismatch on {padded:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_compile_request_with_ast_and_apply_breakpoints_to_value_equivalence() {
        let compiler = PrefixCompiler::default();
        let anth_raw = crate::bench::make_realistic_agent_request(3, 1, 7777);

        let compiled_std = compiler
            .compile_request("t-eq", &anth_raw, Some(ProviderFormat::AnthropicMessages))
            .unwrap();
        let (compiled_no_emit, mut ast) = compiler
            .compile_request_with_ast(
                "t-eq",
                &anth_raw,
                Some(ProviderFormat::AnthropicMessages),
                false,
            )
            .unwrap();

        assert!(compiled_no_emit.canonical_json.is_empty());
        assert_eq!(compiled_no_emit.blocks, compiled_std.blocks);
        assert_eq!(compiled_no_emit.bindings, compiled_std.bindings);
        assert_eq!(compiled_no_emit.shared_prefix_hash, compiled_std.shared_prefix_hash);
        assert_eq!(compiled_no_emit.full_prefix_hash, compiled_std.full_prefix_hash);
        assert_eq!(compiled_no_emit.total_est_tokens, compiled_std.total_est_tokens);

        let plan = BreakpointPlan {
            breakpoints: vec![
                PlannedBreakpoint {
                    block_index: 0,
                    path: "tools[11]".to_string(),
                    cumulative_tokens: 1000,
                    ttl: CacheTtl::OneHour,
                },
                PlannedBreakpoint {
                    block_index: 1,
                    path: "system[4]".to_string(),
                    cumulative_tokens: 3000,
                    ttl: CacheTtl::FiveMin,
                },
                PlannedBreakpoint {
                    block_index: 2,
                    path: "messages[0].content[0]".to_string(),
                    cumulative_tokens: 3500,
                    ttl: CacheTtl::OneHour,
                },
            ],
            expected_cost_usd: 0.01,
            uncached_cost_usd: 0.05,
            naive_single_bp_cost_usd: 0.02,
            expected_savings_vs_uncached_pct: 80.0,
            keepalive_recommended: false,
            keepalive_interval_secs: None,
        };

        let via_json = apply_breakpoints_to_json(
            &compiled_std.canonical_json,
            ProviderFormat::AnthropicMessages,
            &plan,
            Some("ignored"),
        )
        .unwrap();
        let via_value = apply_breakpoints_to_value(
            &mut ast,
            ProviderFormat::AnthropicMessages,
            &plan,
            Some("ignored"),
        );
        assert_eq!(via_value, via_json);

        // OpenAI Chat equivalence (`prompt_cache_key` sorted insertion).
        let oai_raw = br#"{"temperature":0.2,"messages":[{"role":"user","content":"hello"}],"model":"gpt-4o"}"#;
        let oai_std = compiler
            .compile_request("t-eq", oai_raw, Some(ProviderFormat::OpenAiChat))
            .unwrap();
        let (_, mut oai_ast) = compiler
            .compile_request_with_ast("t-eq", oai_raw, Some(ProviderFormat::OpenAiChat), false)
            .unwrap();
        let oai_via_json = apply_breakpoints_to_json(
            &oai_std.canonical_json,
            ProviderFormat::OpenAiChat,
            &plan,
            Some("cp-0123456789abcdef"),
        )
        .unwrap();
        let oai_via_val = apply_breakpoints_to_value(
            &mut oai_ast,
            ProviderFormat::OpenAiChat,
            &plan,
            Some("cp-0123456789abcdef"),
        );
        assert_eq!(oai_via_val, oai_via_json);
    }

    #[test]
    fn test_renumber_digit_length_change_and_block_cached_serialization() {
        // Trigger `t10 -> v1` (2-digit temp ID renumbered to 1-digit final ID) via All-Gather reorder!
        let mut s = "Prefix ⟦cp:t10⟧ mid ⟦cp:esc:t1⟧ end ⟦cp:t1⟧".to_string();
        let temp_bindings: Vec<(&'static str, String)> = (1..=10)
            .map(|i| ("opaque_id", format!("req_{i:012}")))
            .collect();
        let mut next_fid = 1usize;
        let mut out_bindings = Vec::new();
        renumber_text_anchors(&mut s, "system[0]", &temp_bindings, &mut next_fid, &mut out_bindings);
        assert_eq!(s, "Prefix ⟦cp:v1⟧ mid ⟦cp:esc:t1⟧ end ⟦cp:v2⟧");
        assert_eq!(out_bindings.len(), 2);
        assert_eq!(out_bindings[0].anchor, "⟦cp:v1⟧");
        assert_eq!(out_bindings[0].original_value, "req_000000000010");
        assert_eq!(out_bindings[1].anchor, "⟦cp:v2⟧");
        assert_eq!(out_bindings[1].original_value, "req_000000000001");
        assert_eq!(
            reconstruct_lossless(&s, &out_bindings),
            "Prefix req_000000000010 mid ⟦cp:t1⟧ end req_000000000001"
        );

        // Verify block-cached serialization matches `serialize_canonical` byte-for-byte
        // both with and without pre-existing `cache_control` markers!
        let compiler = PrefixCompiler::default();
        let with_cc = br#"{
            "model": "claude-sonnet-4",
            "tools": [{"name": "z_tool", "cache_control": {"type": "ephemeral"}}, {"name": "a_tool"}],
            "system": [
                {"type": "text", "text": "Part B 2026-11-22T15:00:00Z", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"type": "text", "text": "Part A stable"}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let (compiled, ast) = compiler
            .compile_request_with_ast("t-cc", with_cc, Some(ProviderFormat::AnthropicMessages), true)
            .unwrap();
        let mut expected = Vec::new();
        assert!(!serialize_canonical(&ast, &mut expected));
        assert_eq!(compiled.canonical_json, expected);
    }
}
