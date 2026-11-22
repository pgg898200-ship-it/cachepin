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
    parse_json, serialize_canonical, serialize_canonical_skip_top_key, serialize_json,
    serialize_json_skip_top_key, JsonValue,
};
use crate::simd::{chain_hash64, derive_tenant_key, estimate_tokens_fast, keyed_hash64};
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
    pub fn compile_request(
        &self,
        tenant_id: &str,
        raw_json: &[u8],
        provider_hint: Option<ProviderFormat>,
    ) -> Result<CompileResult, String> {
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
            let blocks = extract_blocks(&root, provider, &tenant_key, None);
            let (shared_prefix_hash, full_prefix_hash, total_est_tokens) =
                summarize_blocks(&blocks, raw_json);
            return Ok(CompileResult {
                provider,
                model,
                canonical_json: raw_json.to_vec(),
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
            });
        }

        // Pass 1: JSON key canonicalization + deterministic tool sorting.
        let json_keys_reordered = root.canonicalize_in_place();
        let tools_reordered = canonicalize_tools(&mut root);

        // Pass 2 (Stage A): Lossless volatile hoisting with temporary IDs so
        // Pass 3 can match semantically identical blocks even if their volatile
        // tokens appeared in different initial orders across swarm agents.
        let mut temp_bindings: HashMap<usize, (&'static str, String)> = HashMap::new();
        let mut next_temp_id: usize = 1;
        if matches!(
            self.config.mode,
            CanonicalMode::HoistVolatile | CanonicalMode::SwarmAlign
        ) {
            hoist_volatile_pass(
                &mut root,
                provider,
                self.config.max_volatile_per_block,
                &mut next_temp_id,
                &mut temp_bindings,
            );
        }

        // Pass 3: All-Gather shared-block alignment across multi-agent swarms.
        let all_gather_reordered = if self.config.mode == CanonicalMode::SwarmAlign {
            all_gather_align_pass(
                &mut root,
                provider,
                &tenant_key,
                tenant_hash,
                &self.swarm_stats,
                self.config.all_gather_delimiter.as_deref(),
            )
        } else {
            false
        };

        // Pass 2 (Stage B): Renumber temporary anchors into canonical final
        // `⟦cp:v1⟧`, `⟦cp:v2⟧`, ... in post-alignment prefix order, then append
        // the compact trailing binding block strictly at the end of the request.
        let mut bindings = Vec::new();
        if !temp_bindings.is_empty() {
            finalize_hoisted_anchors(&mut root, provider, &temp_bindings, &mut bindings);
        }
        let volatile_spans_hoisted = bindings.len();

        let volatile_tail_loc = if !bindings.is_empty() {
            let summary = format_binding_summary(&bindings);
            Some(append_volatile_tail(&mut root, provider, summary))
        } else {
            None
        };

        let mut canonical_json = Vec::with_capacity(raw_json.len() + 64);
        let _ = serialize_canonical(&root, &mut canonical_json);

        let blocks = extract_blocks(&root, provider, &tenant_key, volatile_tail_loc);
        let (shared_prefix_hash, full_prefix_hash, total_est_tokens) =
            summarize_blocks(&blocks, &canonical_json);

        Ok(CompileResult {
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
        })
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
    if !segment.contains(SENTINEL_PREFIX) {
        out.push_str(segment);
        return;
    }
    let mut rest = segment;
    while let Some(pos) = rest.find(SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        out.push_str(SENTINEL_ESCAPED);
        rest = &rest[pos + SENTINEL_PREFIX.len()..];
    }
    out.push_str(rest);
}

fn hoist_text_stage_a(
    text: &mut String,
    max_spans: usize,
    next_temp_id: &mut usize,
    temp_bindings: &mut HashMap<usize, (&'static str, String)>,
) {
    let spans = find_volatile_spans(text, max_spans);
    let has_sentinel = text.contains(SENTINEL_PREFIX);
    if spans.is_empty() && !has_sentinel {
        return;
    }

    let mut out = String::with_capacity(text.len() + spans.len() * 8 + 16);
    let mut cursor = 0usize;

    for (start, end, kind) in spans {
        if start < cursor || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            continue;
        }
        escape_sentinels_into(&text[cursor..start], &mut out);
        let tid = *next_temp_id;
        *next_temp_id += 1;
        temp_bindings.insert(tid, (kind, text[start..end].to_string()));
        out.push_str(SENTINEL_PREFIX);
        out.push('t');
        out.push_str(&tid.to_string());
        out.push_str(SENTINEL_SUFFIX);
        cursor = end;
    }

    escape_sentinels_into(&text[cursor..], &mut out);
    *text = out;
}

fn hoist_volatile_pass(
    root: &mut JsonValue,
    provider: ProviderFormat,
    max_per_block: usize,
    next_temp_id: &mut usize,
    temp_bindings: &mut HashMap<usize, (&'static str, String)>,
) {
    // 1. Anthropic `system`
    if let Some(sys) = root.get_mut("system") {
        match sys {
            JsonValue::String(s) => {
                hoist_text_stage_a(s, max_per_block, next_temp_id, temp_bindings);
            }
            JsonValue::Array(items) => {
                for item in items.iter_mut() {
                    if item.get("type").and_then(JsonValue::as_str) == Some("text") {
                        if let Some(JsonValue::String(s)) = item.get_mut("text") {
                            hoist_text_stage_a(s, max_per_block, next_temp_id, temp_bindings);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // 2. OpenAI Responses `instructions`
    if let Some(JsonValue::String(s)) = root.get_mut("instructions") {
        hoist_text_stage_a(s, max_per_block, next_temp_id, temp_bindings);
    }

    // 3. Early `messages` (excluding the final user turn)
    if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
        let scan_limit = prefix_message_scan_limit(msgs);
        for msg in msgs.iter_mut().take(scan_limit) {
            if let Some(content) = msg.get_mut("content") {
                match content {
                    JsonValue::String(s) => {
                        hoist_text_stage_a(s, max_per_block, next_temp_id, temp_bindings);
                    }
                    JsonValue::Array(parts) => {
                        for part in parts.iter_mut() {
                            let is_text = matches!(
                                part.get("type").and_then(JsonValue::as_str),
                                Some("text" | "input_text")
                            );
                            if is_text {
                                if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                    hoist_text_stage_a(
                                        s,
                                        max_per_block,
                                        next_temp_id,
                                        temp_bindings,
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

    // 4. OpenAI Responses early `input` array items (excluding final user turn)
    if provider == ProviderFormat::OpenAiResponses {
        if let Some(items) = root.get_mut("input").and_then(JsonValue::as_array_mut) {
            let scan_limit = prefix_message_scan_limit(items);
            for item in items.iter_mut().take(scan_limit) {
                if let Some(content) = item.get_mut("content") {
                    match content {
                        JsonValue::String(s) => {
                            hoist_text_stage_a(s, max_per_block, next_temp_id, temp_bindings);
                        }
                        JsonValue::Array(parts) => {
                            for part in parts.iter_mut() {
                                let is_text = matches!(
                                    part.get("type").and_then(JsonValue::as_str),
                                    Some("text" | "input_text")
                                );
                                if is_text {
                                    if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                        hoist_text_stage_a(
                                            s,
                                            max_per_block,
                                            next_temp_id,
                                            temp_bindings,
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

fn prefix_message_scan_limit(msgs: &[JsonValue]) -> usize {
    if let Some(last) = msgs.last() {
        if last.get("role").and_then(JsonValue::as_str) == Some("user") {
            return msgs.len().saturating_sub(1);
        }
    }
    msgs.len()
}

fn renumber_text_anchors(
    text: &mut String,
    source_path: &str,
    temp_bindings: &HashMap<usize, (&'static str, String)>,
    next_final_id: &mut usize,
    out_bindings: &mut Vec<HoistedBinding>,
) {
    if !text.contains(SENTINEL_PREFIX) {
        return;
    }

    let mut out = String::with_capacity(text.len());
    let mut rest = text.as_str();

    while let Some(pos) = rest.find(SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];

        // Preserve escaped literal sentinels verbatim.
        if after.starts_with(SENTINEL_ESCAPED) {
            out.push_str(SENTINEL_ESCAPED);
            rest = &after[SENTINEL_ESCAPED.len()..];
            continue;
        }

        if let Some(after_prefix) = after.strip_prefix(SENTINEL_PREFIX) {
            if let Some(after_t) = after_prefix.strip_prefix('t') {
                let digit_len = after_t
                    .bytes()
                    .take_while(|b| b.is_ascii_digit())
                    .count();
                if digit_len > 0 && after_t[digit_len..].starts_with(SENTINEL_SUFFIX) {
                    if let Ok(tid) = after_t[..digit_len].parse::<usize>() {
                        if let Some((kind, orig)) = temp_bindings.get(&tid) {
                            let fid = *next_final_id;
                            *next_final_id += 1;
                            let anchor =
                                format!("{SENTINEL_PREFIX}v{fid}{SENTINEL_SUFFIX}");
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
                            continue;
                        }
                    }
                }
            }
        }

        out.push_str(SENTINEL_PREFIX);
        rest = &after[SENTINEL_PREFIX.len()..];
    }

    out.push_str(rest);
    *text = out;
}

fn finalize_hoisted_anchors(
    root: &mut JsonValue,
    provider: ProviderFormat,
    temp_bindings: &HashMap<usize, (&'static str, String)>,
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
                            let path = format!("system[{i}]");
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
                        let path = format!("messages[{i}]");
                        renumber_text_anchors(
                            s,
                            &path,
                            temp_bindings,
                            &mut next_final_id,
                            out_bindings,
                        );
                    }
                    JsonValue::Array(parts) => {
                        for (j, part) in parts.iter_mut().enumerate() {
                            if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                let path = format!("messages[{i}].content[{j}]");
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
                            let path = format!("input[{i}]");
                            renumber_text_anchors(
                                s,
                                &path,
                                temp_bindings,
                                &mut next_final_id,
                                out_bindings,
                            );
                        }
                        JsonValue::Array(parts) => {
                            for (j, part) in parts.iter_mut().enumerate() {
                                if let Some(JsonValue::String(s)) = part.get_mut("text") {
                                    let path = format!("input[{i}].content[{j}]");
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
                        _ => {}
                    }
                }
            }
        }
    }
}

fn format_binding_summary(bindings: &[HoistedBinding]) -> String {
    let mut out = String::from("[cachepin volatile bindings]");
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
                root.set_field("messages", JsonValue::Array(Vec::new()));
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
                                last.set_field(
                                    "content",
                                    JsonValue::Array(vec![first_part, tail_part]),
                                );
                                return VolatileTailLoc::MessagePart(msg_idx, 1);
                            }
                            _ => {
                                last.set_field("content", JsonValue::Array(vec![tail_part]));
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
                    root.set_field("input", JsonValue::Array(vec![first_item, tail_item]));
                    VolatileTailLoc::InputItem(1)
                }
                _ => {
                    root.set_field("input", JsonValue::Array(vec![tail_item]));
                    VolatileTailLoc::InputItem(0)
                }
            }
        }
        ProviderFormat::OpenAiChat | ProviderFormat::VllmOpenAi | ProviderFormat::Generic => {
            if root.get("messages").and_then(JsonValue::as_array).is_none() {
                root.set_field("messages", JsonValue::Array(Vec::new()));
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

// --- Pure-Rust Volatile Pattern Detectors ---

const MASK_ISO: u8 = 1 << 0;
const MASK_HEX: u8 = 1 << 1;
const MASK_PREFIX_ID: u8 = 1 << 2;
const MASK_HEADER: u8 = 1 << 3;

const fn build_volatile_start_mask() -> [u8; 256] {
    let mut t = [0u8; 256];
    // ISO-8601 years 1900..=2199 start with '1' or '2'
    t[b'1' as usize] |= MASK_ISO;
    t[b'2' as usize] |= MASK_ISO;
    // UUID hex digits
    let mut b = b'0';
    while b <= b'9' {
        t[b as usize] |= MASK_HEX;
        b += 1;
    }
    let mut b = b'a';
    while b <= b'f' {
        t[b as usize] |= MASK_HEX;
        b += 1;
    }
    let mut b = b'A';
    while b <= b'F' {
        t[b as usize] |= MASK_HEX;
        b += 1;
    }
    // Prefixed IDs: agent_run_, session_, trace_, span_, call_, req_, run_
    t[b'a' as usize] |= MASK_PREFIX_ID;
    t[b's' as usize] |= MASK_PREFIX_ID;
    t[b't' as usize] |= MASK_PREFIX_ID;
    t[b'c' as usize] |= MASK_PREFIX_ID;
    t[b'r' as usize] |= MASK_PREFIX_ID;
    // Inline headers: Current ..., Timestamp:, Date:
    t[b'c' as usize] |= MASK_HEADER;
    t[b'C' as usize] |= MASK_HEADER;
    t[b't' as usize] |= MASK_HEADER;
    t[b'T' as usize] |= MASK_HEADER;
    t[b'd' as usize] |= MASK_HEADER;
    t[b'D' as usize] |= MASK_HEADER;
    t
}

static VOLATILE_START_MASK: [u8; 256] = build_volatile_start_mask();

fn find_volatile_spans(text: &str, max_spans: usize) -> Vec<(usize, usize, &'static str)> {
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
        let mask = VOLATILE_START_MASK[b as usize];
        if mask == 0 {
            i += 1;
            continue;
        }

        // (d) Inline time/date headers (`Current time:`, `Timestamp:`, `Date:`)
        if (mask & MASK_HEADER) != 0 && i + 5 < len {
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

        // (a) ISO-8601 / RFC-3339 timestamp (`YYYY-MM-DD[T ]HH:MM:SS...`)
        if (mask & MASK_ISO) != 0
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

        // (b) UUID (`8-4-4-4-12` hex): check hyphen skeleton first in O(1)
        if (mask & MASK_HEX) != 0
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

        // (c) Prefixed opaque IDs (`req_`, `run_`, `span_`, `call_`, `trace_`, `session_`, `agent_run_`)
        if (mask & MASK_PREFIX_ID) != 0 && i + 12 <= len {
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

/// Normalizes temporary volatile anchors `⟦cp:t<digits>⟧` to a constant token
/// `⟦cp:v0⟧` exclusively for All-Gather block-hash equality comparison so that
/// two agents holding the same shared block in different initial orders compute
/// the exact same block hash.
fn normalize_temp_anchors_for_hash(s: &str) -> String {
    if !s.contains(SENTINEL_PREFIX) {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find(SENTINEL_PREFIX) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        if after.starts_with(SENTINEL_ESCAPED) {
            out.push_str(SENTINEL_ESCAPED);
            rest = &after[SENTINEL_ESCAPED.len()..];
            continue;
        }
        if let Some(after_prefix) = after.strip_prefix(SENTINEL_PREFIX) {
            if let Some(after_t) = after_prefix.strip_prefix('t') {
                let dlen = after_t.bytes().take_while(|b| b.is_ascii_digit()).count();
                if dlen > 0 && after_t[dlen..].starts_with(SENTINEL_SUFFIX) {
                    out.push_str("⟦cp:v0⟧");
                    let full_len =
                        SENTINEL_PREFIX.len() + 1 + dlen + SENTINEL_SUFFIX.len();
                    rest = &after[full_len..];
                    continue;
                }
            }
        }
        out.push_str(SENTINEL_PREFIX);
        rest = &after[SENTINEL_PREFIX.len()..];
    }
    out.push_str(rest);
    out
}

#[inline]
fn canonical_bytes_without_cache_control(val: &JsonValue) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    serialize_canonical_skip_top_key(val, "cache_control", &mut out);
    out
}

fn hash_for_all_gather_with_buf(
    val: &JsonValue,
    tenant_key: &[u8; 16],
    scratch: &mut Vec<u8>,
) -> u64 {
    scratch.clear();
    serialize_json_skip_top_key(val, "cache_control", scratch);
    if let Ok(s) = std::str::from_utf8(scratch) {
        if s.contains(SENTINEL_PREFIX) {
            let norm = normalize_temp_anchors_for_hash(s);
            return keyed_hash64(tenant_key, norm.as_bytes());
        }
    }
    keyed_hash64(tenant_key, scratch)
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
        let lower_prefix_len = t.len().min(20);
        // Safe char-boundary prefix for ASCII tag inspection.
        let mut end = lower_prefix_len;
        while end > 0 && !t.is_char_boundary(end) {
            end -= 1;
        }
        let head = t[..end].to_ascii_lowercase();
        return head.starts_with("<doc")
            || head.starts_with("<document")
            || head.starts_with("<file")
            || head.starts_with("<context")
            || head.starts_with("<shared_context")
            || head.starts_with("<peer_output")
            || head.starts_with("--- file:");
    }
    false
}

fn align_block_slice(
    slice: &mut [JsonValue],
    tenant_key: &[u8; 16],
    tenant_hash: u64,
    swarm_stats: &SwarmBlockStats,
) -> bool {
    if slice.len() < 2 {
        return false;
    }

    let mut hashes = Vec::with_capacity(slice.len());
    let mut scratch = Vec::with_capacity(1024);
    for item in slice.iter() {
        hashes.push(hash_for_all_gather_with_buf(item, tenant_key, &mut scratch));
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
        return false;
    }

    // Stable sort descending by `(swarm_freq_bucket, canonical_block_hash)`.
    ranked.sort_by_key(|&(bucket, bh, _)| (Reverse(bucket), Reverse(bh)));

    let mut owned: Vec<Option<JsonValue>> = slice
        .iter_mut()
        .map(|v| Some(std::mem::replace(v, JsonValue::Null)))
        .collect();

    for (dst, (_, _, src_idx)) in ranked.into_iter().enumerate() {
        if let Some(val) = owned[src_idx].take() {
            slice[dst] = val;
        }
    }
    true
}

fn align_delimited_text(
    text: &mut String,
    delim: &str,
    tenant_key: &[u8; 16],
    tenant_hash: u64,
    swarm_stats: &SwarmBlockStats,
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
        let norm = normalize_temp_anchors_for_hash(chunk);
        let bh = keyed_hash64(tenant_key, norm.as_bytes());
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
) -> bool {
    let mut reordered = false;

    // 1. Delimiter-chunk alignment inside `system` / `instructions` / early messages.
    if let Some(delim) = delimiter {
        if let Some(sys) = root.get_mut("system") {
            match sys {
                JsonValue::String(s) => {
                    if align_delimited_text(s, delim, tenant_key, tenant_hash, swarm_stats) {
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
            if align_delimited_text(s, delim, tenant_key, tenant_hash, swarm_stats) {
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
                            if align_delimited_text(s, delim, tenant_key, tenant_hash, swarm_stats)
                            {
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
            if align_block_slice(sys_arr, tenant_key, tenant_hash, swarm_stats) {
                reordered = true;
            }
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
                            )
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

    reordered
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

fn push_block(
    blocks: &mut Vec<PrefixBlock>,
    tenant_key: &[u8; 16],
    running_prefix: &mut u64,
    path: String,
    kind: &str,
    val: &JsonValue,
    breakpoint_eligible: bool,
) {
    let canonical_bytes = canonical_bytes_without_cache_control(val);
    let block_hash = keyed_hash64(tenant_key, &canonical_bytes);
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

fn extract_blocks(
    root: &JsonValue,
    provider: ProviderFormat,
    tenant_key: &[u8; 16],
    volatile_tail_loc: Option<VolatileTailLoc>,
) -> Vec<PrefixBlock> {
    let mut blocks = Vec::new();
    let mut running_prefix = 0u64;

    // 1. `tools`
    if let Some(tools) = root.get("tools").and_then(JsonValue::as_array) {
        for (i, tool) in tools.iter().enumerate() {
            push_block(
                &mut blocks,
                tenant_key,
                &mut running_prefix,
                format!("tools[{i}]"),
                "tools",
                tool,
                true,
            );
        }
    }

    // 2. `system` (Anthropic)
    if let Some(sys) = root.get("system") {
        match sys {
            JsonValue::Null => {}
            JsonValue::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    let kind = if is_shared_context_block(item) {
                        "shared_doc"
                    } else {
                        "system"
                    };
                    push_block(
                        &mut blocks,
                        tenant_key,
                        &mut running_prefix,
                        format!("system[{i}]"),
                        kind,
                        item,
                        true,
                    );
                }
            }
            other => {
                push_block(
                    &mut blocks,
                    tenant_key,
                    &mut running_prefix,
                    "system".to_string(),
                    "system",
                    other,
                    true,
                );
            }
        }
    }

    // 3. `instructions` (OpenAI Responses)
    if let Some(instr) = root.get("instructions") {
        if *instr != JsonValue::Null {
            push_block(
                &mut blocks,
                tenant_key,
                &mut running_prefix,
                "instructions".to_string(),
                "system",
                instr,
                true,
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
                            format!("messages[{i}].content[{j}]"),
                            kind,
                            part,
                            eligible,
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
                    format!("messages[{i}]"),
                    kind,
                    msg,
                    eligible,
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
                            format!("input[{i}]"),
                            kind,
                            item,
                            eligible,
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
                    );
                }
            }
        }
    }

    blocks
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

    match provider {
        ProviderFormat::AnthropicMessages => {
            clear_all_cache_controls(&mut root);

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
                inject_anthropic_breakpoint(&mut root, &bp.path, cc);
            }
        }
        ProviderFormat::OpenAiChat
        | ProviderFormat::OpenAiResponses
        | ProviderFormat::VllmOpenAi
        | ProviderFormat::Generic => {
            if let Some(key) = prompt_cache_key {
                if !key.is_empty() {
                    root.set_field("prompt_cache_key", JsonValue::String(key.to_string()));
                }
            }
        }
    }

    let mut out = Vec::with_capacity(canonical_json.len() + 128);
    let _ = serialize_canonical(&root, &mut out);
    Ok(out)
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
                root.set_field("system", JsonValue::Array(vec![block]));
            }
            Some(JsonValue::Array(arr)) => {
                if let Some(last) = arr.last_mut() {
                    last.set_field("cache_control", cc);
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
                    item.set_field("cache_control", cc);
                }
            }
            Some(JsonValue::String(s)) if idx == 0 => {
                let text = std::mem::take(s);
                let block = JsonValue::Object(vec![
                    ("cache_control".to_string(), cc),
                    ("text".to_string(), JsonValue::String(text)),
                    ("type".to_string(), JsonValue::String("text".to_string())),
                ]);
                root.set_field("system", JsonValue::Array(vec![block]));
            }
            _ => {}
        }
        return;
    }

    if let Some(idx) = parse_bracket_index(path, "tools[") {
        if let Some(tools) = root.get_mut("tools").and_then(JsonValue::as_array_mut) {
            if let Some(tool) = tools.get_mut(idx) {
                tool.set_field("cache_control", cc);
            }
        }
        return;
    }

    if let Some((msg_idx, part_idx)) = parse_message_content_path(path) {
        if let Some(msgs) = root.get_mut("messages").and_then(JsonValue::as_array_mut) {
            if let Some(msg) = msgs.get_mut(msg_idx) {
                if let Some(parts) = msg.get_mut("content").and_then(JsonValue::as_array_mut) {
                    if let Some(part) = parts.get_mut(part_idx) {
                        part.set_field("cache_control", cc);
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
                        msg.set_field("content", JsonValue::Array(vec![block]));
                    }
                    Some(JsonValue::Array(parts)) => {
                        if let Some(target) = parts.iter_mut().rev().find(|p| {
                            !matches!(
                                p.get("type").and_then(JsonValue::as_str),
                                Some("thinking" | "redacted_thinking")
                            )
                        }) {
                            target.set_field("cache_control", cc);
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
}
