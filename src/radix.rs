//! Tenant- and model-isolated compressed block-level Prefix Radix Tree and Miss Forensics Engine.
//!
//! Tracks prefix reuse statistics, EWMA inter-arrival times, live cache TTL state,
//! and bounded sample payloads (`O(1)` bounded RAM via oldest-timestamp eviction).
//! Pinpoints exact cache miss root causes and byte-level divergence offsets using
//! SIMD `longest_common_prefix_len`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::dp_planner::BlockReuseEstimate;
use crate::simd::longest_common_prefix_len;
use crate::types::{
    CacheDiagnostic, CacheTtl, MissReason, ModelPricing, PlannedBreakpoint, PrefixBlock,
};

/// Default maximum resident prefix nodes before oldest-timestamp eviction triggers.
pub const DEFAULT_MAX_NODES: usize = 65_536;

/// Maximum canonical bytes retained per prefix node for SIMD byte-level diffing.
const MAX_SAMPLE_BYTES: usize = 8_192;

/// Maximum recent sibling children indexed per parent node for O(1) forensics lookup.
const MAX_CHILDREN_PER_PARENT: usize = 16;

/// EWMA smoothing factor for inter-arrival time updates.
const INTERARRIVAL_EWMA_ALPHA: f64 = 0.35;

/// Cap on predicted hits per window to avoid runaway extrapolation on simultaneous bursts.
const MAX_PREDICTED_HITS: f64 = 500.0;

/// Compact key identifying a prefix node inside a `(tenant_key, model)` namespace.
type NodeKey = (u32, u64);

/// Per-node statistics stored in the compressed block-level prefix radix tree.
#[derive(Debug, Clone)]
pub struct PrefixNodeStats {
    pub prefix_hash: u64,
    pub block_hash: u64,
    pub depth: usize,
    pub cumulative_tokens: u32,
    pub hits: u64,
    pub last_seen_ms: u64,
    pub warm_until_ms: u64,
    pub active_ttl: Option<CacheTtl>,
    pub ewma_interarrival_secs: f64,
    pub sample_bytes: Vec<u8>,
    pub sample_path: String,
}

/// High-level telemetry summary of the prefix radix tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixSummary {
    pub total_nodes: usize,
    pub max_nodes: usize,
    pub warm_nodes: usize,
    pub namespaces: usize,
    pub total_requests_recorded: u64,
    pub total_evictions: u64,
}

/// Internal entry wrapping public stats with eviction and parent-link metadata.
#[derive(Debug, Clone)]
struct NodeEntry {
    stats: PrefixNodeStats,
    parent_prefix_hash: u64,
    last_seq: u64,
}

/// Tenant- and model-isolated compressed block-level prefix radix tree with
/// bounded memory (oldest-timestamp eviction) and SIMD miss forensics.
#[derive(Debug)]
pub struct PrefixRadixTree {
    max_nodes: usize,
    next_ns_id: u32,
    next_seq: u64,
    last_now_ms: u64,
    total_requests_recorded: u64,
    total_evictions: u64,
    /// Maps `(tenant_key, model_name)` -> compact `ns_id`.
    namespace_ids: HashMap<(u64, String), u32>,
    /// Per-tenant index of `(ns_id, model_name)` for cross-model flip forensics
    /// strictly within the same tenant (never crossing tenant boundaries).
    tenant_models: HashMap<u64, Vec<(u32, String)>>,
    /// Resident prefix nodes keyed by `(ns_id, prefix_hash)`.
    nodes: HashMap<NodeKey, NodeEntry>,
    /// Parent -> bounded list of child `prefix_hash`es (`parent == 0` is virtual root).
    children: HashMap<(u32, u64), Vec<u64>>,
    /// Lazy-deletion min-heap ordered by `(last_seen_ms, last_seq, key)` for oldest-timestamp eviction.
    eviction_heap: BinaryHeap<Reverse<(u64, u64, NodeKey)>>,
}

impl Default for PrefixRadixTree {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_NODES)
    }
}

impl PrefixRadixTree {
    /// Create a new bounded prefix radix tree holding at most `max_nodes` nodes.
    pub fn new(max_nodes: usize) -> Self {
        Self {
            max_nodes,
            next_ns_id: 1,
            next_seq: 0,
            last_now_ms: 0,
            total_requests_recorded: 0,
            total_evictions: 0,
            namespace_ids: HashMap::new(),
            tenant_models: HashMap::new(),
            nodes: HashMap::new(),
            children: HashMap::new(),
            eviction_heap: BinaryHeap::new(),
        }
    }

    /// Create a new prefix radix tree with the default `65_536`-node cap.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_MAX_NODES)
    }

    /// Read-only lookup of a specific prefix node for inspection/testing.
    pub fn get_node(
        &self,
        tenant_key: u64,
        model: &str,
        prefix_hash: u64,
    ) -> Option<&PrefixNodeStats> {
        let &ns_id = self.namespace_ids.get(&(tenant_key, model.to_string()))?;
        self.nodes.get(&(ns_id, prefix_hash)).map(|e| &e.stats)
    }

    /// Lookup a request's block prefix in `O(M)` hash operations and diagnose
    /// cache hit/miss root cause, exact SIMD byte divergence offset, and estimated USD waste.
    #[allow(clippy::too_many_arguments)]
    pub fn lookup_and_diagnose(
        &self,
        tenant_key: u64,
        model: &str,
        blocks: &[PrefixBlock],
        now_ms: u64,
        pricing: &ModelPricing,
        tools_reordered: bool,
        json_keys_reordered: bool,
        volatile_hoisted: usize,
        all_gather_reordered: bool,
    ) -> CacheDiagnostic {
        let total_blocks = blocks.len();
        let total_tokens: u32 = blocks
            .iter()
            .map(|b| b.est_tokens)
            .fold(0u32, u32::saturating_add);

        if total_blocks == 0 || total_tokens < pricing.min_cache_tokens {
            return CacheDiagnostic {
                reason: MissReason::BelowMinTokens,
                matched_blocks: 0,
                total_blocks,
                matched_tokens: 0,
                total_tokens,
                divergence_block: None,
                divergence_path: None,
                divergence_byte_offset: None,
                wasted_usd_est: 0.0,
                advice: format!(
                    "Prefix length ({total_tokens} tokens) is below model minimum cache threshold ({} tokens).",
                    pricing.min_cache_tokens
                ),
            };
        }

        let ns_id = self.lookup_namespace_id(tenant_key, model);

        // Walk the deepest contiguous warm prefix in O(M) hash lookups.
        let mut matched_blocks = 0usize;
        let mut matched_tokens = 0u32;
        let mut expired_at_divergence = false;

        if let Some(ns) = ns_id {
            for blk in blocks {
                if let Some(entry) = self.nodes.get(&(ns, blk.prefix_hash)) {
                    if entry.stats.warm_until_ms > now_ms && entry.stats.active_ttl.is_some() {
                        matched_blocks += 1;
                        matched_tokens = matched_tokens.saturating_add(blk.est_tokens);
                        continue;
                    }
                    if entry.stats.warm_until_ms > 0 && entry.stats.warm_until_ms <= now_ms {
                        expired_at_divergence = true;
                    }
                }
                break;
            }
        }

        // Full prefix hit!
        if matched_blocks == total_blocks {
            return CacheDiagnostic {
                reason: MissReason::CacheHit,
                matched_blocks,
                total_blocks,
                matched_tokens,
                total_tokens,
                divergence_block: None,
                divergence_path: None,
                divergence_byte_offset: None,
                wasted_usd_est: 0.0,
                advice: "Full prefix matched warm cache.".to_string(),
            };
        }

        let div_idx = matched_blocks;
        let div_block = &blocks[div_idx];
        let parent_hash = if div_idx == 0 {
            0
        } else {
            blocks[div_idx - 1].prefix_hash
        };

        // Check for a warm (or previously seen) sibling under the same parent at depth `div_idx`.
        let best_sibling = ns_id.and_then(|ns| {
            self.find_best_sibling(ns, parent_hash, &div_block.canonical_bytes, now_ms)
        });
        let has_warm_sibling = best_sibling
            .map(|(s, _)| s.warm_until_ms > now_ms && s.active_ttl.is_some())
            .unwrap_or(false);
        let any_drift_flag =
            tools_reordered || json_keys_reordered || volatile_hoisted > 0 || all_gather_reordered;

        // Warm prefix hit when all shared/cached prefix blocks matched (>= min_cache_tokens,
        // non-expired) and only uncached message/volatile_tail blocks remain — either the fresh
        // final user turn (+ optional hoisted volatile_tail), or a private per-agent message
        // history branching after the shared warm prefix without mutating any warm cached turn.
        let trailing_non_volatile = blocks[div_idx..]
            .iter()
            .filter(|b| b.kind != "volatile_tail")
            .count();
        let all_remaining_messages = blocks[div_idx..]
            .iter()
            .all(|b| b.kind == "message" || b.kind == "volatile_tail");
        let shared_prefix_boundary_hit = div_idx > 0
            && blocks[div_idx - 1].kind != "message"
            && (matched_tokens as u64) * 5 >= (total_tokens as u64) * 4;
        let only_fresh_tail_remaining = all_remaining_messages
            && (trailing_non_volatile <= 1 || !has_warm_sibling || shared_prefix_boundary_hit);

        if matched_blocks > 0
            && matched_tokens >= pricing.min_cache_tokens
            && !expired_at_divergence
            && only_fresh_tail_remaining
        {
            return CacheDiagnostic {
                reason: MissReason::CacheHit,
                matched_blocks,
                total_blocks,
                matched_tokens,
                total_tokens,
                divergence_block: None,
                divergence_path: None,
                divergence_byte_offset: None,
                wasted_usd_est: 0.0,
                advice: format!(
                    "Warm prefix hit ({matched_tokens}/{total_tokens} tokens); only fresh tail at {} was uncached.",
                    div_block.path
                ),
            };
        }

        // Compute SIMD byte-level divergence offset against the closest sibling if present.
        let (divergence_byte_offset, sibling_for_heuristic) = match best_sibling {
            Some((sib, lcp_len)) => (Some(lcp_len), Some(sib)),
            None => (None, None),
        };

        // Check whether the exact same prefix is currently warm under a DIFFERENT model
        // for the SAME tenant (`ModelOrConfigFlip`).
        let warm_on_other_model =
            self.find_warm_other_model(tenant_key, model, div_block.prefix_hash, now_ms);

        // Determine whether this namespace has any prior state at all (for ColdStart vs Mutation).
        let namespace_has_history = ns_id
            .and_then(|ns| self.children.get(&(ns, 0)))
            .map(|c| !c.is_empty())
            .unwrap_or(false);

        let (reason, advice) = if expired_at_divergence {
            (
                MissReason::TtlExpired,
                format!(
                    "Prefix at {} was previously cached but its TTL expired; consider 1h TTL or 270s keepalive pings.",
                    div_block.path
                ),
            )
        } else if let Some(other_model) = warm_on_other_model {
            (
                MissReason::ModelOrConfigFlip,
                format!(
                    "Exact prefix at {} is currently warm under model '{other_model}', not '{model}'; pin sessions to a single model to preserve cache locality.",
                    div_block.path
                ),
            )
        } else if matched_blocks == 0 && !namespace_has_history {
            let extra = if any_drift_flag {
                " Canonicalization drift was also detected and normalized."
            } else {
                ""
            };
            (
                MissReason::ColdStart,
                format!(
                    "First observation of this prefix for tenant/model '{model}'; cache warmed on this request.{extra}"
                ),
            )
        } else if tools_reordered {
            (
                MissReason::ToolOrderDrift,
                format!(
                    "Tool definitions changed order at {}; enable JsonCanonical or SwarmAlign mode to deterministically sort tools.",
                    div_block.path
                ),
            )
        } else if json_keys_reordered {
            (
                MissReason::JsonKeyDrift,
                format!(
                    "JSON object key order drifted at {}; enable JsonCanonical mode to sort object keys deterministically.",
                    div_block.path
                ),
            )
        } else if volatile_hoisted > 0 {
            (
                MissReason::VolatileTimestampOrId,
                format!(
                    "Detected {volatile_hoisted} volatile timestamp/UUID/ID span(s) causing prefix churn at {}; keep HoistVolatile enabled.",
                    div_block.path
                ),
            )
        } else if all_gather_reordered {
            (
                MissReason::AllGatherReorder,
                format!(
                    "Shared context documents arrived in a different order at {}; enable SwarmAlign to frequency-rank shared blocks.",
                    div_block.path
                ),
            )
        } else if let Some(sib) = sibling_for_heuristic {
            let off = divergence_byte_offset.unwrap_or(0);
            if looks_like_volatile_diff(&sib.sample_bytes, &div_block.canonical_bytes, off) {
                (
                    MissReason::VolatileTimestampOrId,
                    format!(
                        "Prefix diverged at {} byte {off} on a volatile timestamp/UUID/ID token; enable HoistVolatile mode.",
                        div_block.path
                    ),
                )
            } else if div_block.kind == "tools" {
                (
                    MissReason::ToolOrderDrift,
                    format!(
                        "Tool schema block diverged at {} byte {off} relative to warm sibling {}; sort tools deterministically.",
                        div_block.path, sib.sample_path
                    ),
                )
            } else {
                (
                    MissReason::PrefixMutation,
                    format!(
                        "Prefix mutated at {} byte {off} relative to warm sibling {}; move dynamic content after the cache breakpoint.",
                        div_block.path, sib.sample_path
                    ),
                )
            }
        } else {
            (
                MissReason::ColdStart,
                format!(
                    "Unseen branch starting at {}; cache breakpoint will warm this suffix for subsequent turns.",
                    div_block.path
                ),
            )
        };

        let missed_tokens = total_tokens.saturating_sub(matched_tokens);
        let wasted_usd_est = match reason {
            MissReason::CacheHit | MissReason::ColdStart | MissReason::BelowMinTokens => 0.0,
            _ => (pricing.cost_input(missed_tokens as u64)
                - pricing.cost_read(missed_tokens as u64))
            .max(0.0),
        };

        CacheDiagnostic {
            reason,
            matched_blocks,
            total_blocks,
            matched_tokens,
            total_tokens,
            divergence_block: Some(div_idx),
            divergence_path: Some(div_block.path.clone()),
            divergence_byte_offset,
            wasted_usd_est,
            advice,
        }
    }

    /// Record a request and its active breakpoints into the radix tree, updating
    /// reuse counts, EWMA inter-arrival times, warm deadlines, and evicting the
    /// oldest nodes if capacity `max_nodes` is reached.
    pub fn record_request(
        &mut self,
        tenant_key: u64,
        model: &str,
        blocks: &[PrefixBlock],
        breakpoints: &[PlannedBreakpoint],
        now_ms: u64,
    ) {
        self.total_requests_recorded = self.total_requests_recorded.saturating_add(1);
        if now_ms > self.last_now_ms {
            self.last_now_ms = now_ms;
        }
        if blocks.is_empty() || self.max_nodes == 0 {
            return;
        }

        let ns_id = self.get_or_create_namespace(tenant_key, model);
        let m = blocks.len();

        // Determine the longest TTL covering each block index `0..m`.
        // A breakpoint at `bp.block_index` warms every block `0..=bp.block_index`.
        let mut covering_ttl: Vec<Option<CacheTtl>> = vec![None; m];
        for bp in breakpoints {
            let end = bp.block_index.min(m - 1);
            for slot in covering_ttl.iter_mut().take(end + 1) {
                *slot = match (*slot, bp.ttl) {
                    (Some(CacheTtl::OneHour), _) | (_, CacheTtl::OneHour) => {
                        Some(CacheTtl::OneHour)
                    }
                    (_, CacheTtl::FiveMin) => Some(CacheTtl::FiveMin),
                };
            }
        }

        let mut cum_tokens: u32 = 0;
        for (i, blk) in blocks.iter().enumerate() {
            cum_tokens = cum_tokens.saturating_add(blk.est_tokens);
            let parent_hash = if i == 0 { 0 } else { blocks[i - 1].prefix_hash };
            let key = (ns_id, blk.prefix_hash);
            let ttl_opt = covering_ttl[i];
            let new_warm_until = ttl_opt
                .map(|t| now_ms.saturating_add((t.seconds() * 1000.0) as u64))
                .unwrap_or(0);

            self.next_seq = self.next_seq.wrapping_add(1);
            let seq = self.next_seq;

            if let Some(entry) = self.nodes.get_mut(&key) {
                let gap_secs = now_ms.saturating_sub(entry.stats.last_seen_ms) as f64 / 1000.0;
                let clamped_gap = gap_secs.max(0.05);
                if entry.stats.hits <= 1 || entry.stats.ewma_interarrival_secs <= 0.0 {
                    entry.stats.ewma_interarrival_secs = clamped_gap;
                } else {
                    entry.stats.ewma_interarrival_secs = INTERARRIVAL_EWMA_ALPHA * clamped_gap
                        + (1.0 - INTERARRIVAL_EWMA_ALPHA) * entry.stats.ewma_interarrival_secs;
                }

                entry.stats.hits = entry.stats.hits.saturating_add(1);
                entry.stats.last_seen_ms = now_ms;
                entry.stats.cumulative_tokens = cum_tokens;
                entry.last_seq = seq;

                if let Some(ttl) = ttl_opt {
                    if new_warm_until >= entry.stats.warm_until_ms {
                        entry.stats.warm_until_ms = new_warm_until;
                        entry.stats.active_ttl = Some(ttl);
                    }
                }

                self.eviction_heap.push(Reverse((now_ms, seq, key)));
            } else {
                // Enforce strict bounded memory via oldest-timestamp eviction.
                while self.nodes.len() >= self.max_nodes {
                    if !self.evict_oldest_node() {
                        break;
                    }
                }

                let sample_len = blk.canonical_bytes.len().min(MAX_SAMPLE_BYTES);
                let stats = PrefixNodeStats {
                    prefix_hash: blk.prefix_hash,
                    block_hash: blk.block_hash,
                    depth: i,
                    cumulative_tokens: cum_tokens,
                    hits: 1,
                    last_seen_ms: now_ms,
                    warm_until_ms: new_warm_until,
                    active_ttl: ttl_opt,
                    ewma_interarrival_secs: 0.0,
                    sample_bytes: blk.canonical_bytes[..sample_len].to_vec(),
                    sample_path: blk.path.clone(),
                };
                self.nodes.insert(
                    key,
                    NodeEntry {
                        stats,
                        parent_prefix_hash: parent_hash,
                        last_seq: seq,
                    },
                );
                self.register_child(ns_id, parent_hash, blk.prefix_hash);
                self.eviction_heap.push(Reverse((now_ms, seq, key)));
            }
        }

        // Compact stale lazy-deletion heap entries periodically for bounded RAM.
        let heap_limit = self.max_nodes.saturating_mul(4).max(1024);
        if self.eviction_heap.len() > heap_limit {
            self.rebuild_eviction_heap();
        }
    }

    /// Estimate per-block reuse rates and cache warmth for direct input into
    /// [`crate::dp_planner::solve_optimal_breakpoints`].
    pub fn estimate_reuse_profile(
        &self,
        tenant_key: u64,
        model: &str,
        blocks: &[PrefixBlock],
        now_ms: u64,
    ) -> Vec<BlockReuseEstimate> {
        let ns_id = self.lookup_namespace_id(tenant_key, model);
        let mut out = Vec::with_capacity(blocks.len());

        for blk in blocks {
            let node_opt = ns_id.and_then(|ns| self.nodes.get(&(ns, blk.prefix_hash)));
            let est = match node_opt {
                Some(entry) => {
                    let s = &entry.stats;
                    let already_warm = s.warm_until_ms > now_ms && s.active_ttl.is_some();
                    let currently_warm_ttl = if already_warm { s.active_ttl } else { None };
                    let elapsed_secs = now_ms.saturating_sub(s.last_seen_ms) as f64 / 1000.0;

                    if s.hits >= 2 && s.ewma_interarrival_secs > 0.0 {
                        let gap = s.ewma_interarrival_secs.max(0.05);
                        let eff_gap = gap.max(elapsed_secs * 0.5);
                        let confidence = ((s.hits - 1) as f64 / 3.0).clamp(0.4, 1.0);
                        let (h5, h1) = predict_hits_from_gap(eff_gap, confidence);
                        BlockReuseEstimate {
                            expected_hits_5m: h5,
                            expected_hits_1h: h1,
                            ewma_gap_secs: eff_gap,
                            currently_warm_ttl,
                            already_warm,
                        }
                    } else {
                        // Seen once before — this is the 2nd observation; elapsed time is first real interval.
                        let eff_gap = elapsed_secs.max(1.0);
                        let (h5, h1) = predict_hits_from_gap(eff_gap, 0.5);
                        BlockReuseEstimate {
                            expected_hits_5m: h5,
                            expected_hits_1h: h1,
                            ewma_gap_secs: eff_gap,
                            currently_warm_ttl,
                            already_warm,
                        }
                    }
                }
                None => cold_structural_prior(blk),
            };
            out.push(est);
        }

        // Enforce physical prefix-tree monotonicity: a deeper block cannot have
        // strictly more future prefix matches than its ancestor.
        for i in 1..out.len() {
            let prev_5m = out[i - 1].expected_hits_5m;
            let prev_1h = out[i - 1].expected_hits_1h;
            if out[i].expected_hits_5m > prev_5m {
                out[i].expected_hits_5m = prev_5m;
            }
            if out[i].expected_hits_1h > prev_1h {
                out[i].expected_hits_1h = prev_1h;
            }
            if out[i].expected_hits_1h < out[i].expected_hits_5m {
                out[i].expected_hits_1h = out[i].expected_hits_5m;
            }
        }

        out
    }

    /// Return aggregate occupancy and eviction metrics for observability.
    pub fn stats_summary(&self) -> RadixSummary {
        let warm_nodes = self
            .nodes
            .values()
            .filter(|e| e.stats.warm_until_ms > self.last_now_ms && e.stats.active_ttl.is_some())
            .count();
        RadixSummary {
            total_nodes: self.nodes.len(),
            max_nodes: self.max_nodes,
            warm_nodes,
            namespaces: self.namespace_ids.len(),
            total_requests_recorded: self.total_requests_recorded,
            total_evictions: self.total_evictions,
        }
    }

    fn lookup_namespace_id(&self, tenant_key: u64, model: &str) -> Option<u32> {
        self.namespace_ids
            .get(&(tenant_key, model.to_string()))
            .copied()
    }

    fn get_or_create_namespace(&mut self, tenant_key: u64, model: &str) -> u32 {
        let key = (tenant_key, model.to_string());
        if let Some(&id) = self.namespace_ids.get(&key) {
            return id;
        }
        let id = self.next_ns_id;
        self.next_ns_id = self.next_ns_id.wrapping_add(1).max(1);
        self.namespace_ids.insert(key, id);
        self.tenant_models
            .entry(tenant_key)
            .or_default()
            .push((id, model.to_string()));
        id
    }

    fn register_child(&mut self, ns_id: u32, parent_hash: u64, child_hash: u64) {
        let list = self.children.entry((ns_id, parent_hash)).or_default();
        if let Some(pos) = list.iter().position(|&h| h == child_hash) {
            list.remove(pos);
        } else if list.len() >= MAX_CHILDREN_PER_PARENT {
            list.remove(0);
        }
        list.push(child_hash);
    }

    fn find_best_sibling<'a>(
        &'a self,
        ns_id: u32,
        parent_hash: u64,
        candidate_bytes: &[u8],
        now_ms: u64,
    ) -> Option<(&'a PrefixNodeStats, usize)> {
        let siblings = self.children.get(&(ns_id, parent_hash))?;
        let mut best: Option<(&PrefixNodeStats, usize, bool, u64)> = None;

        for &sib_hash in siblings {
            let Some(entry) = self.nodes.get(&(ns_id, sib_hash)) else {
                continue;
            };
            let s = &entry.stats;
            let is_warm = s.warm_until_ms > now_ms && s.active_ttl.is_some();
            let lcp = longest_common_prefix_len(&s.sample_bytes, candidate_bytes);

            let replace = match best {
                None => true,
                Some((_, best_lcp, best_warm, best_seen)) => {
                    // Prefer warm sibling first, then deepest SIMD common prefix, then most recent.
                    (is_warm, lcp, s.last_seen_ms) > (best_warm, best_lcp, best_seen)
                }
            };
            if replace {
                best = Some((s, lcp, is_warm, s.last_seen_ms));
            }
        }

        best.map(|(s, lcp, _, _)| (s, lcp))
    }

    fn find_warm_other_model(
        &self,
        tenant_key: u64,
        current_model: &str,
        prefix_hash: u64,
        now_ms: u64,
    ) -> Option<&str> {
        let models = self.tenant_models.get(&tenant_key)?;
        for (other_ns, other_model) in models {
            if other_model == current_model {
                continue;
            }
            if let Some(entry) = self.nodes.get(&(*other_ns, prefix_hash)) {
                if entry.stats.warm_until_ms > now_ms && entry.stats.active_ttl.is_some() {
                    return Some(other_model.as_str());
                }
            }
        }
        None
    }

    fn evict_oldest_node(&mut self) -> bool {
        while let Some(Reverse((_, seq, key))) = self.eviction_heap.pop() {
            let is_live = self
                .nodes
                .get(&key)
                .map(|e| e.last_seq == seq)
                .unwrap_or(false);
            if !is_live {
                continue;
            }
            if let Some(removed) = self.nodes.remove(&key) {
                let (ns_id, prefix_hash) = key;
                if let Some(list) = self.children.get_mut(&(ns_id, removed.parent_prefix_hash)) {
                    list.retain(|&h| h != prefix_hash);
                    if list.is_empty() {
                        self.children.remove(&(ns_id, removed.parent_prefix_hash));
                    }
                }
                self.total_evictions = self.total_evictions.saturating_add(1);
                return true;
            }
        }

        // Fallback if heap was somehow out of sync: evict minimum `(last_seen_ms, last_seq)` directly.
        let oldest_key = self
            .nodes
            .iter()
            .min_by_key(|(_, e)| (e.stats.last_seen_ms, e.last_seq))
            .map(|(&k, _)| k);
        if let Some(key) = oldest_key {
            if let Some(removed) = self.nodes.remove(&key) {
                let (ns_id, prefix_hash) = key;
                if let Some(list) = self.children.get_mut(&(ns_id, removed.parent_prefix_hash)) {
                    list.retain(|&h| h != prefix_hash);
                    if list.is_empty() {
                        self.children.remove(&(ns_id, removed.parent_prefix_hash));
                    }
                }
                self.total_evictions = self.total_evictions.saturating_add(1);
                return true;
            }
        }
        false
    }

    fn rebuild_eviction_heap(&mut self) {
        self.eviction_heap.clear();
        self.eviction_heap.reserve(self.nodes.len());
        for (&key, entry) in &self.nodes {
            self.eviction_heap
                .push(Reverse((entry.stats.last_seen_ms, entry.last_seq, key)));
        }
    }
}

/// Convert an effective inter-arrival gap in seconds into `(expected_hits_5m, expected_hits_1h)`.
///
/// Models provider sliding-window TTL physics: when inter-arrival `gap <= 240s`,
/// each hit refreshes the 5-minute TTL for free so a 5m cache stays warm across
/// the full 1-hour horizon (`h5 == h1`). When `gap > 240s`, the 5m cache risks
/// expiring between turns while a 1h cache survives gaps up to 3600s.
fn predict_hits_from_gap(eff_gap_secs: f64, confidence: f64) -> (f64, f64) {
    let gap = eff_gap_secs.max(0.05);
    let raw_1h = (3600.0 / gap) * confidence;
    let h1 = if gap <= 3600.0 {
        raw_1h
    } else {
        raw_1h * (-(gap - 3600.0) / 3600.0).exp()
    }
    .min(MAX_PREDICTED_HITS);

    let h5 = if gap <= 240.0 {
        // Sliding 5m TTL refreshes on every hit; survives the full horizon.
        h1
    } else {
        let raw_5m = (300.0 / gap) * confidence;
        raw_5m * (-(gap - 240.0) / 180.0).exp()
    }
    .min(h1);

    (h5, h1)
}

/// Structural prior for never-before-seen blocks based on their role/kind.
fn cold_structural_prior(blk: &PrefixBlock) -> BlockReuseEstimate {
    if !blk.breakpoint_eligible || blk.kind == "volatile_tail" {
        return BlockReuseEstimate::cold();
    }
    match blk.kind.as_str() {
        "tools" | "system" | "shared_doc" => BlockReuseEstimate {
            expected_hits_5m: 3.0,
            expected_hits_1h: 3.0,
            ewma_gap_secs: 60.0,
            currently_warm_ttl: None,
            already_warm: false,
        },
        _ => BlockReuseEstimate {
            expected_hits_5m: 0.5,
            expected_hits_1h: 0.5,
            ewma_gap_secs: 120.0,
            currently_warm_ttl: None,
            already_warm: false,
        },
    }
}

/// Lightweight byte-window heuristic checking whether a prefix divergence around
/// `offset` falls inside a timestamp, UUID, or request/trace ID field.
///
/// Stops scanning at field delimiters (`,`, `{`, `}`, `\n`) so an unrelated
/// timestamp field elsewhere in the same block does not misclassify a semantic
/// [`MissReason::PrefixMutation`].
fn looks_like_volatile_diff(a: &[u8], b: &[u8], offset: usize) -> bool {
    let check_slice = |bytes: &[u8]| -> bool {
        if bytes.is_empty() {
            return false;
        }
        let pos = offset.min(bytes.len());
        let min_start = pos.saturating_sub(36);
        let mut start = pos;
        while start > min_start {
            let prev = bytes[start - 1];
            if matches!(prev, b',' | b'{' | b'}' | b'\n') {
                break;
            }
            start -= 1;
        }

        let max_end = pos.saturating_add(40).min(bytes.len());
        let mut end = pos;
        while end < max_end {
            let cur = bytes[end];
            if matches!(cur, b',' | b'{' | b'}' | b'\n') {
                break;
            }
            end += 1;
        }

        let window = &bytes[start..end];
        let lower: Vec<u8> = window.iter().map(|c| c.to_ascii_lowercase()).collect();

        // Check common volatile field names inside the diverging field.
        const MARKERS: &[&[u8]] = &[
            b"timestamp",
            b"created_at",
            b"updated_at",
            b"request_id",
            b"trace_id",
            b"session_id",
            b"nonce",
            b"uuid",
            b"date",
            b"time",
        ];
        if MARKERS
            .iter()
            .any(|m| lower.windows(m.len()).any(|w| w == *m))
        {
            return true;
        }

        // Check ISO-8601 date fragment like `2026-11-22`.
        if window.windows(10).any(|w| {
            w[0] == b'2'
                && w[1] == b'0'
                && w[2].is_ascii_digit()
                && w[3].is_ascii_digit()
                && w[4] == b'-'
                && w[5].is_ascii_digit()
                && w[6].is_ascii_digit()
                && w[7] == b'-'
                && w[8].is_ascii_digit()
                && w[9].is_ascii_digit()
        }) {
            return true;
        }

        // Check UUID-style hex-dash pattern `xxxx-xxxx-`.
        if window.windows(10).any(|w| {
            w[0..4].iter().all(|c| c.is_ascii_hexdigit())
                && w[4] == b'-'
                && w[5..9].iter().all(|c| c.is_ascii_hexdigit())
                && w[9] == b'-'
        }) {
            return true;
        }

        false
    };

    check_slice(a) || check_slice(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::lookup_pricing;
    use crate::types::ProviderFormat;

    fn hash_bytes(seed: u64, data: &[u8]) -> u64 {
        let mut h = seed ^ 0xcbf29ce484222325u64;
        for &b in data {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    fn make_chain(specs: &[(&str, &str, &[u8], u32, bool)]) -> Vec<PrefixBlock> {
        let mut out = Vec::with_capacity(specs.len());
        let mut prefix_h = 0x517cc1b727220a95u64;
        for &(path, kind, bytes, tokens, eligible) in specs {
            let bh = hash_bytes(0x1234, bytes);
            prefix_h = hash_bytes(prefix_h, &bh.to_le_bytes());
            out.push(PrefixBlock {
                path: path.to_string(),
                kind: kind.to_string(),
                canonical_bytes: bytes.to_vec(),
                block_hash: bh,
                prefix_hash: prefix_h,
                est_tokens: tokens,
                breakpoint_eligible: eligible,
            });
        }
        out
    }

    #[test]
    fn test_cold_start_warm_hit_and_ttl_expiry() {
        let pricing = lookup_pricing("claude-sonnet-5-5", ProviderFormat::AnthropicMessages);
        let mut tree = PrefixRadixTree::new(1024);
        let blocks = make_chain(&[
            ("system[0]", "system", b"You are a principal Rust engineer.", 1000, true),
            ("messages[0]", "message", b"Explain lock-free radix trees.", 400, true),
        ]);

        // 1. Cold start on first observation.
        let d1 = tree.lookup_and_diagnose(
            42,
            "claude-sonnet-5-5",
            &blocks,
            1_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d1.reason, MissReason::ColdStart);
        assert_eq!(d1.matched_blocks, 0);
        assert_eq!(d1.wasted_usd_est, 0.0);

        let bps = vec![PlannedBreakpoint {
            block_index: 1,
            path: "messages[0]".to_string(),
            cumulative_tokens: 1400,
            ttl: CacheTtl::FiveMin,
        }];
        tree.record_request(42, "claude-sonnet-5-5", &blocks, &bps, 1_000);

        // 2. Warm hit within 5-minute TTL.
        let d2 = tree.lookup_and_diagnose(
            42,
            "claude-sonnet-5-5",
            &blocks,
            60_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d2.reason, MissReason::CacheHit);
        assert_eq!(d2.matched_blocks, 2);
        assert_eq!(d2.matched_tokens, 1400);
        assert_eq!(d2.wasted_usd_est, 0.0);

        // 3. Tenant isolation: tenant 99 MUST NOT hit tenant 42's cache!
        let d_other_tenant = tree.lookup_and_diagnose(
            99,
            "claude-sonnet-5-5",
            &blocks,
            60_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d_other_tenant.reason, MissReason::ColdStart);
        assert_eq!(d_other_tenant.matched_blocks, 0);

        // 4. Model isolation + ModelOrConfigFlip detection for same tenant!
        let d_flip = tree.lookup_and_diagnose(
            42,
            "claude-opus-5-5",
            &blocks,
            60_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d_flip.reason, MissReason::ModelOrConfigFlip);
        assert_eq!(d_flip.matched_blocks, 0);
        assert!(d_flip.wasted_usd_est > 0.0);
        assert!(d_flip.advice.contains("claude-sonnet-5-5"));

        // 5. TTL expiration after 300s (warm_until_ms = 301_000).
        let d_exp = tree.lookup_and_diagnose(
            42,
            "claude-sonnet-5-5",
            &blocks,
            305_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d_exp.reason, MissReason::TtlExpired);
        assert_eq!(d_exp.matched_blocks, 0);
        assert!(d_exp.wasted_usd_est > 0.0);
    }

    #[test]
    fn test_simd_divergence_byte_offset_and_miss_reason_taxonomy() {
        let pricing = lookup_pricing("claude-opus-5-5", ProviderFormat::AnthropicMessages);
        let mut tree = PrefixRadixTree::new(1024);

        let warm_blocks = make_chain(&[
            ("system[0]", "system", b"Stable system prompt header.", 800, true),
            (
                "shared_doc[0]",
                "shared_doc",
                b"{\"repo\":\"cachepin\",\"timestamp\":\"2026-11-22T15:00:00Z\",\"body\":\"hello\"}",
                1200,
                true,
            ),
            ("messages[0]", "message", b"User question.", 200, true),
        ]);
        let bps = vec![PlannedBreakpoint {
            block_index: 1,
            path: "shared_doc[0]".to_string(),
            cumulative_tokens: 2000,
            ttl: CacheTtl::FiveMin,
        }];
        tree.record_request(7, "claude-opus-5-5", &warm_blocks, &bps, 10_000);

        // Diverge inside block 1 ONLY at the timestamp minute digit (exact byte offset 48!).
        let ts_mutated = make_chain(&[
            ("system[0]", "system", b"Stable system prompt header.", 800, true),
            (
                "shared_doc[0]",
                "shared_doc",
                b"{\"repo\":\"cachepin\",\"timestamp\":\"2026-11-22T15:05:00Z\",\"body\":\"hello\"}",
                1200,
                true,
            ),
            ("messages[0]", "message", b"User question.", 200, true),
        ]);
        let diag_ts = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &ts_mutated,
            20_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(diag_ts.reason, MissReason::VolatileTimestampOrId);
        assert_eq!(diag_ts.matched_blocks, 1);
        assert_eq!(diag_ts.matched_tokens, 800);
        assert_eq!(diag_ts.divergence_block, Some(1));
        assert_eq!(diag_ts.divergence_path.as_deref(), Some("shared_doc[0]"));
        assert_eq!(diag_ts.divergence_byte_offset, Some(47));
        assert!(diag_ts.wasted_usd_est > 0.0);

        // Explicit drift flags classify ToolOrderDrift, JsonKeyDrift, AllGatherReorder accurately.
        let d_tool = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &ts_mutated,
            20_000,
            &pricing,
            true,
            false,
            0,
            false,
        );
        assert_eq!(d_tool.reason, MissReason::ToolOrderDrift);

        let d_json = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &ts_mutated,
            20_000,
            &pricing,
            false,
            true,
            0,
            false,
        );
        assert_eq!(d_json.reason, MissReason::JsonKeyDrift);

        let d_ag = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &ts_mutated,
            20_000,
            &pricing,
            false,
            false,
            0,
            true,
        );
        assert_eq!(d_ag.reason, MissReason::AllGatherReorder);

        // Pure semantic mutation (no timestamp/UUID markers) -> PrefixMutation.
        let plain_mut = make_chain(&[
            ("system[0]", "system", b"Stable system prompt header.", 800, true),
            (
                "shared_doc[0]",
                "shared_doc",
                b"{\"repo\":\"other_project\",\"body\":\"completely different\"}",
                1200,
                true,
            ),
            ("messages[0]", "message", b"User question.", 200, true),
        ]);
        let d_mut = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &plain_mut,
            20_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d_mut.reason, MissReason::PrefixMutation);
        assert_eq!(d_mut.divergence_byte_offset, Some(9));

        // BelowMinTokens check.
        let tiny = make_chain(&[("system[0]", "system", b"hi", 100, true)]);
        let d_tiny = tree.lookup_and_diagnose(
            7,
            "claude-opus-5-5",
            &tiny,
            20_000,
            &pricing,
            false,
            false,
            0,
            false,
        );
        assert_eq!(d_tiny.reason, MissReason::BelowMinTokens);
    }

    #[test]
    fn test_bounded_memory_oldest_timestamp_eviction_and_reuse_profile() {
        let mut tree = PrefixRadixTree::new(4);
        for i in 0..10u64 {
            let payload = format!("unique-prefix-{i}");
            let blks = make_chain(&[("system[0]", "system", payload.as_bytes(), 1000, true)]);
            let bps = vec![PlannedBreakpoint {
                block_index: 0,
                path: "system[0]".to_string(),
                cumulative_tokens: 1000,
                ttl: CacheTtl::FiveMin,
            }];
            tree.record_request(1, "claude-sonnet-5-5", &blks, &bps, (i + 1) * 1_000);
        }

        let summary = tree.stats_summary();
        assert_eq!(summary.total_nodes, 4);
        assert_eq!(summary.max_nodes, 4);
        assert_eq!(summary.total_evictions, 6);
        assert_eq!(summary.total_requests_recorded, 10);

        // Oldest 6 prefixes (0..6) must be evicted; newest 4 (6..10) must remain!
        let oldest = make_chain(&[("system[0]", "system", b"unique-prefix-0", 1000, true)]);
        assert!(tree
            .get_node(1, "claude-sonnet-5-5", oldest[0].prefix_hash)
            .is_none());

        let newest = make_chain(&[("system[0]", "system", b"unique-prefix-9", 1000, true)]);
        assert!(tree
            .get_node(1, "claude-sonnet-5-5", newest[0].prefix_hash)
            .is_some());

        // Record repeated hits at 10-minute intervals -> reuse profile should show
        // low 5m hits and strong 1h hits, guiding DP to choose 1h TTL!
        let mut tree2 = PrefixRadixTree::new(128);
        let shared = make_chain(&[
            ("system[0]", "system", b"Long-lived agent instructions.", 1000, true),
            ("messages[0]", "message", b"Turn 1 payload", 200, true),
        ]);
        let bp0 = vec![PlannedBreakpoint {
            block_index: 0,
            path: "system[0]".to_string(),
            cumulative_tokens: 1000,
            ttl: CacheTtl::OneHour,
        }];
        for k in 0..4u64 {
            tree2.record_request(1, "claude-sonnet-5-5", &shared, &bp0, (k + 1) * 600_000);
        }
        let profile = tree2.estimate_reuse_profile(1, "claude-sonnet-5-5", &shared, 4 * 600_000 + 1_000);
        assert_eq!(profile.len(), 2);
        assert!(profile[0].already_warm);
        assert_eq!(profile[0].currently_warm_ttl, Some(CacheTtl::OneHour));
        assert!(profile[0].ewma_gap_secs > 500.0);
        assert!(profile[0].expected_hits_1h > profile[0].expected_hits_5m * 5.0);
        // Physical prefix-tree monotonicity: child <= parent.
        assert!(profile[1].expected_hits_5m <= profile[0].expected_hits_5m);
        assert!(profile[1].expected_hits_1h <= profile[0].expected_hits_1h);
    }
}
