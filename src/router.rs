//! Replication-Aware Shadow-KV Adaptive Radix Router for `cachepin`.
//!
//! Solves the eviction-regime P99 latency cliff of single-home prefix routers
//! by combining:
//! 1. Per-backend bounded `O(1)` Shadow KV-Cache LRU (leaf-first prefix eviction
//!    strictly respecting `kv_capacity_tokens`),
//! 2. Per-backend live in-flight prefill/read queues and EWMA observed TTFT,
//! 3. Exponentially-decayed per-prefix hotness tracking and active-replica masks,
//! 4. Exact millisecond-denominated TTFT + collateral eviction pressure scoring
//!    that selectively replicates ONLY saturated hot prefixes onto low-pressure
//!    backends while keeping cool prefixes single-homed to maximize fleet KV capacity.
//!
//! All five [`RoutingPolicy`] strategies run on identical shadow-KV state for
//! apples-to-apples benchmarking.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::types::PrefixBlock;

/// Maximum tracked hotness entries before oldest pruning keeps RAM bounded.
const MAX_HOTNESS_ENTRIES: usize = 16_384;

/// Exponential decay window (seconds) for per-prefix near-term arrival burst estimation.
const HOTNESS_WINDOW_SECS: f64 = 10.0;

/// Recency decay time constant (ms) for evaluating how "hot" an LRU eviction victim is.
const VICTIM_HEAT_TAU_MS: f64 = 15_000.0;

/// Fast SplitMix64 `u64` hasher (zero external crates, ~1ns per lookup).
#[derive(Default)]
struct FastU64Hasher(u64);

impl Hasher for FastU64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut acc = self.0;
        for &b in bytes {
            acc = acc.wrapping_mul(0x100000001b3) ^ (b as u64);
        }
        self.0 = acc;
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

type FastU64Map<V> = HashMap<u64, V, BuildHasherDefault<FastU64Hasher>>;

/// Hardware & serving profile for a single LLM inference backend (e.g. vLLM / SGLang replica).
#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub id: usize,
    pub url: String,
    pub kv_capacity_tokens: u64,
    pub prefill_tps: f64,
    pub cache_read_tps: f64,
    pub max_inflight: usize,
}

/// Supported routing policies evaluated on identical shadow KV-cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    RoundRobin,
    LeastLoaded,
    ConsistentHash,
    SingleHomePrefixDivert,
    CachePinAdaptiveReplication,
}

/// Output of a single routing decision.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub backend_id: usize,
    pub est_matched_tokens: u32,
    pub est_ttft_ms: f64,
    pub triggered_replication: bool,
    pub reason: &'static str,
}

/// Per-backend live shadow-KV and queue telemetry snapshot.
#[derive(Debug, Clone)]
pub struct BackendStats {
    pub backend_id: usize,
    pub used_kv_tokens: u64,
    pub kv_capacity_tokens: u64,
    pub resident_blocks: usize,
    pub inflight_requests: usize,
    pub inflight_prefill_tokens: u64,
    pub ewma_observed_ttft_ms: f64,
    pub total_routed: u64,
    pub total_completed: u64,
    pub total_decode_tokens: u64,
    pub total_evictions: u64,
}

/// Arena slot in the `O(1)` doubly-linked shadow KV-cache LRU.
#[derive(Debug, Clone, Copy)]
struct LruSlot {
    key: u64,
    tokens: u32,
    prev: Option<usize>,
    next: Option<usize>,
    lru_tick: u64,
    last_seen_ms: u64,
}

/// Bounded `O(1)` Shadow KV-Cache LRU tracking resident prefix block hashes
/// and evicting strictly from the LRU tail whenever `used_kv_tokens > kv_capacity_tokens`.
#[derive(Debug)]
struct ShadowKvLru {
    kv_capacity_tokens: u64,
    used_kv_tokens: u64,
    tick: u64,
    total_evictions: u64,
    head: Option<usize>, // MRU
    tail: Option<usize>, // LRU
    slots: Vec<LruSlot>,
    free_list: Vec<usize>,
    index: FastU64Map<usize>,
}

impl ShadowKvLru {
    fn new(kv_capacity_tokens: u64) -> Self {
        Self {
            kv_capacity_tokens,
            used_kv_tokens: 0,
            tick: 0,
            total_evictions: 0,
            head: None,
            tail: None,
            slots: Vec::new(),
            free_list: Vec::new(),
            index: FastU64Map::default(),
        }
    }

    #[inline]
    fn contains(&self, key: u64) -> bool {
        self.index.contains_key(&key)
    }

    #[inline]
    fn len(&self) -> usize {
        self.index.len()
    }

    /// `(last_seen_ms, lru_tick)` of the oldest (LRU tail) resident block, if any.
    #[inline]
    fn tail_recency(&self) -> Option<(u64, u64)> {
        self.tail
            .map(|idx| (self.slots[idx].last_seen_ms, self.slots[idx].lru_tick))
    }

    /// Touch an existing resident block or insert a new one, evicting LRU tail blocks
    /// until `used_kv_tokens <= kv_capacity_tokens`. Returns `(evicted_blocks, evicted_tokens)`.
    fn touch_or_insert(&mut self, key: u64, tokens: u32, now_ms: u64) -> (u64, u64) {
        if self.kv_capacity_tokens == 0 || (tokens as u64) > self.kv_capacity_tokens {
            return (0, 0);
        }

        self.tick = self.tick.wrapping_add(1);
        let tick = self.tick;

        if let Some(&slot_idx) = self.index.get(&key) {
            self.slots[slot_idx].lru_tick = tick;
            self.slots[slot_idx].last_seen_ms = now_ms;
            if self.head != Some(slot_idx) {
                self.unlink(slot_idx);
                self.push_front(slot_idx);
            }
            return (0, 0);
        }

        let mut evicted_blocks = 0u64;
        let mut evicted_tokens = 0u64;
        let add_tokens = tokens as u64;

        // Evict LRU tail entries until the new block fits within `kv_capacity_tokens`.
        while self.used_kv_tokens.saturating_add(add_tokens) > self.kv_capacity_tokens {
            let Some(tail_idx) = self.tail else {
                break;
            };
            self.unlink(tail_idx);
            let removed = self.slots[tail_idx];
            self.index.remove(&removed.key);
            self.free_list.push(tail_idx);

            let rem_tok = removed.tokens as u64;
            self.used_kv_tokens = self.used_kv_tokens.saturating_sub(rem_tok);
            evicted_blocks += 1;
            evicted_tokens = evicted_tokens.saturating_add(rem_tok);
            self.total_evictions = self.total_evictions.saturating_add(1);
        }

        let slot = LruSlot {
            key,
            tokens,
            prev: None,
            next: None,
            lru_tick: tick,
            last_seen_ms: now_ms,
        };

        let slot_idx = if let Some(free_idx) = self.free_list.pop() {
            self.slots[free_idx] = slot;
            free_idx
        } else {
            let idx = self.slots.len();
            self.slots.push(slot);
            idx
        };

        self.push_front(slot_idx);
        self.index.insert(key, slot_idx);
        self.used_kv_tokens = self.used_kv_tokens.saturating_add(add_tokens);

        (evicted_blocks, evicted_tokens)
    }

    #[inline]
    fn unlink(&mut self, idx: usize) {
        let prev = self.slots[idx].prev;
        let next = self.slots[idx].next;
        match prev {
            Some(p) => self.slots[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.slots[n].prev = prev,
            None => self.tail = prev,
        }
        self.slots[idx].prev = None;
        self.slots[idx].next = None;
    }

    #[inline]
    fn push_front(&mut self, idx: usize) {
        self.slots[idx].prev = None;
        self.slots[idx].next = self.head;
        match self.head {
            Some(old_head) => self.slots[old_head].prev = Some(idx),
            None => self.tail = Some(idx),
        }
        self.head = Some(idx);
    }
}

/// Per-backend live runtime state inside [`ShadowKvRouter`].
#[derive(Debug)]
struct BackendState {
    config: BackendConfig,
    lru: ShadowKvLru,
    inflight_requests: usize,
    inflight_prefill_tokens: u64,
    inflight_read_tokens: u64,
    ewma_observed_ttft_ms: f64,
    eviction_pressure: f64,
    total_routed: u64,
    total_completed: u64,
    total_decode_tokens: u64,
}

impl BackendState {
    fn new(config: BackendConfig) -> Self {
        let cap = config.kv_capacity_tokens;
        Self {
            config,
            lru: ShadowKvLru::new(cap),
            inflight_requests: 0,
            inflight_prefill_tokens: 0,
            inflight_read_tokens: 0,
            ewma_observed_ttft_ms: 0.0,
            eviction_pressure: 0.0,
            total_routed: 0,
            total_completed: 0,
            total_decode_tokens: 0,
        }
    }

    #[inline]
    fn prefill_tps(&self) -> f64 {
        if self.config.prefill_tps.is_finite() && self.config.prefill_tps > 1.0 {
            self.config.prefill_tps
        } else {
            1_000.0
        }
    }

    #[inline]
    fn cache_read_tps(&self) -> f64 {
        let prefill = self.prefill_tps();
        if self.config.cache_read_tps.is_finite() && self.config.cache_read_tps > prefill {
            self.config.cache_read_tps
        } else {
            prefill
        }
    }

    /// Count contiguous prefix-matched tokens currently resident in this backend's shadow KV LRU.
    #[inline]
    fn matched_tokens(&self, tenant_key: u64, blocks: &[PrefixBlock], total_tokens: u32) -> u32 {
        let mut matched = 0u32;
        for blk in blocks {
            let k = mix_tenant_prefix(tenant_key, blk.prefix_hash);
            if self.lru.contains(k) {
                matched = matched.saturating_add(blk.est_tokens);
            } else {
                break;
            }
        }
        matched.min(total_tokens)
    }

    /// Predict live queue wait time (ms) on this backend.
    #[inline]
    fn queue_wait_ms(&self) -> f64 {
        let prefill_tps = self.prefill_tps();
        let read_tps = self.cache_read_tps();
        let work_queue_ms = (self.inflight_prefill_tokens as f64 / prefill_tps
            + self.inflight_read_tokens as f64 / read_tps)
            * 1000.0
            + (self.inflight_requests as f64) * 4.0;

        let max_inf = self.config.max_inflight.max(1);
        let concurrency_wait_ms = if self.inflight_requests >= max_inf {
            let overflow = (self.inflight_requests - max_inf + 1) as f64;
            let slot_wait = self.ewma_observed_ttft_ms.max(40.0);
            overflow * slot_wait
        } else {
            let util = self.inflight_requests as f64 / max_inf as f64;
            util * util * 25.0
        };

        work_queue_ms + concurrency_wait_ms
    }

    /// Estimate total TTFT (ms) on this backend given `matched_tokens` and `miss_tokens`.
    #[inline]
    fn predict_ttft_ms(&self, matched_tokens: u32, miss_tokens: u32) -> f64 {
        let queue = self.queue_wait_ms();
        let miss_ms = (miss_tokens as f64 / self.prefill_tps()) * 1000.0;
        let hit_ms = (matched_tokens as f64 / self.cache_read_tps()) * 1000.0;
        queue + miss_ms + hit_ms
    }

    /// Estimate collateral eviction cost (in ms of future GPU re-prefill work)
    /// caused by admitting `miss_tokens` onto this backend.
    #[inline]
    fn collateral_eviction_ms(&self, miss_tokens: u32, now_ms: u64) -> f64 {
        let cap = self.config.kv_capacity_tokens;
        if cap == 0 || miss_tokens == 0 {
            return 0.0;
        }
        let projected = self.lru.used_kv_tokens.saturating_add(miss_tokens as u64);
        if projected <= cap {
            return 0.0;
        }
        let overflow = (projected - cap) as f64;
        let victim_heat = match self.lru.tail_recency() {
            Some((tail_ms, tail_tick)) => {
                let age_ms = now_ms.saturating_sub(tail_ms) as f64;
                let time_heat = (-age_ms / VICTIM_HEAT_TAU_MS).exp();
                let tick_age = self.lru.tick.wrapping_sub(tail_tick) as f64;
                let tick_scale = (self.lru.len().max(1) as f64) * 2.0;
                let tick_heat = (-tick_age / tick_scale).exp();
                time_heat.min(tick_heat).max(time_heat * 0.5)
            }
            None => 0.0,
        };
        let re_prefill_ms = (overflow / self.prefill_tps()) * 1000.0;
        victim_heat * re_prefill_ms * (1.0 + self.eviction_pressure)
    }
}

/// Per-prefix exponentially decayed hotness and active-replica bitmap.
#[derive(Debug, Clone, Copy)]
struct PrefixHotness {
    /// Exponentially decayed arrival count in the recent [`HOTNESS_WINDOW_SECS`] burst window.
    ewma_burst_hits: f64,
    last_seen_ms: u64,
    total_hits: u64,
    /// Bitmask of backends (first 128 backends) currently holding this prefix anchor.
    replica_mask: u128,
}

/// Per-backend candidate evaluation computed in a single tight pass inside [`ShadowKvRouter::route`].
#[derive(Debug, Clone, Copy)]
struct CandidateEval {
    idx: usize,
    matched_tokens: u32,
    miss_tokens: u32,
    queue_wait_ms: f64,
    est_ttft_ms: f64,
    collateral_evict_ms: f64,
}

/// Replication-aware Shadow-KV Adaptive Radix Router.
#[derive(Debug)]
pub struct ShadowKvRouter {
    backends: Vec<BackendState>,
    policy: RoutingPolicy,
    rr_cursor: usize,
    hotness: FastU64Map<PrefixHotness>,
}

impl ShadowKvRouter {
    /// Construct a new router over `backends` using `policy`.
    pub fn new(backends: Vec<BackendConfig>, policy: RoutingPolicy) -> Self {
        let states = backends.into_iter().map(BackendState::new).collect();
        Self {
            backends: states,
            policy,
            rr_cursor: 0,
            hotness: FastU64Map::default(),
        }
    }

    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    pub fn set_policy(&mut self, policy: RoutingPolicy) {
        self.policy = policy;
    }

    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Snapshot live shadow-KV and queue metrics for `backend_id`.
    pub fn backend_stats(&self, backend_id: usize) -> Option<BackendStats> {
        let idx = self.find_backend_index(backend_id)?;
        let b = &self.backends[idx];
        Some(BackendStats {
            backend_id: b.config.id,
            used_kv_tokens: b.lru.used_kv_tokens,
            kv_capacity_tokens: b.config.kv_capacity_tokens,
            resident_blocks: b.lru.len(),
            inflight_requests: b.inflight_requests,
            inflight_prefill_tokens: b.inflight_prefill_tokens,
            ewma_observed_ttft_ms: b.ewma_observed_ttft_ms,
            total_routed: b.total_routed,
            total_completed: b.total_completed,
            total_decode_tokens: b.total_decode_tokens,
            total_evictions: b.lru.total_evictions,
        })
    }

    /// Return the active-replica bitmask (backends 0..128) for a request's prefix anchor.
    pub fn prefix_replica_mask(
        &self,
        tenant_key: u64,
        blocks: &[PrefixBlock],
        total_tokens: u32,
    ) -> u128 {
        let anchor = compute_prefix_anchor(tenant_key, blocks, total_tokens);
        self.hotness
            .get(&anchor)
            .map(|h| h.replica_mask)
            .unwrap_or(0)
    }

    /// Route an incoming request using the configured [`RoutingPolicy`],
    /// updating shadow KV-cache LRU residency, live in-flight queues, and hotness state.
    pub fn route(
        &mut self,
        tenant_key: u64,
        blocks: &[PrefixBlock],
        total_tokens: u32,
        now_ms: u64,
    ) -> RouteDecision {
        let n = self.backends.len();
        if n == 0 {
            return RouteDecision {
                backend_id: 0,
                est_matched_tokens: 0,
                est_ttft_ms: 0.0,
                triggered_replication: false,
                reason: "no_backends",
            };
        }

        let anchor_key = compute_prefix_anchor(tenant_key, blocks, total_tokens);
        let ch_home = jump_consistent_hash(anchor_key, n);

        // Update per-prefix burst hotness and refresh active-replica bitmap in O(N).
        let (horizon_hits, active_replicas) = self.update_hotness_and_replicas(anchor_key, now_ms);

        // Evaluate all backends in a single stack/small-vec pass (`O(N * M)`).
        let mut evals = Vec::with_capacity(n);
        for (idx, b) in self.backends.iter().enumerate() {
            let matched = b.matched_tokens(tenant_key, blocks, total_tokens);
            let miss = total_tokens.saturating_sub(matched);
            let queue_wait = b.queue_wait_ms();
            let est_ttft = b.predict_ttft_ms(matched, miss);
            let collateral_evict = b.collateral_eviction_ms(miss, now_ms);
            evals.push(CandidateEval {
                idx,
                matched_tokens: matched,
                miss_tokens: miss,
                queue_wait_ms: queue_wait,
                est_ttft_ms: est_ttft,
                collateral_evict_ms: collateral_evict,
            });
        }

        let any_warm_holder = evals.iter().any(|e| e.matched_tokens > 0);

        let (chosen_idx, reason, triggered_replication) = match self.policy {
            RoutingPolicy::RoundRobin => {
                let idx = self.rr_cursor % n;
                self.rr_cursor = self.rr_cursor.wrapping_add(1);
                let rep = any_warm_holder && evals[idx].matched_tokens == 0;
                (idx, "round_robin", rep)
            }
            RoutingPolicy::LeastLoaded => {
                let idx = self.pick_least_loaded(&evals);
                let rep = any_warm_holder && evals[idx].matched_tokens == 0;
                (idx, "least_loaded", rep)
            }
            RoutingPolicy::ConsistentHash => {
                let rep = any_warm_holder && evals[ch_home].matched_tokens == 0;
                (ch_home, "consistent_hash", rep)
            }
            RoutingPolicy::SingleHomePrefixDivert => {
                self.select_single_home_divert(&evals, ch_home)
            }
            RoutingPolicy::CachePinAdaptiveReplication => self.select_adaptive_replication(
                &evals,
                ch_home,
                horizon_hits,
                active_replicas,
            ),
        };

        let chosen = evals[chosen_idx];
        self.admit_request(chosen_idx, tenant_key, anchor_key, blocks, &chosen, now_ms);

        RouteDecision {
            backend_id: self.backends[chosen_idx].config.id,
            est_matched_tokens: chosen.matched_tokens,
            est_ttft_ms: chosen.est_ttft_ms,
            triggered_replication,
            reason,
        }
    }

    /// Record request completion on `backend_id`, draining in-flight prefill/read
    /// queues and updating EWMA observed TTFT.
    pub fn complete(
        &mut self,
        backend_id: usize,
        prefill_tokens: u32,
        decode_tokens: u32,
        observed_ttft_ms: f64,
    ) {
        let Some(idx) = self.find_backend_index(backend_id) else {
            return;
        };
        let b = &mut self.backends[idx];
        let prev_inflight = b.inflight_requests;
        b.inflight_requests = b.inflight_requests.saturating_sub(1);
        b.inflight_prefill_tokens = b
            .inflight_prefill_tokens
            .saturating_sub(prefill_tokens as u64);

        if b.inflight_requests == 0 {
            b.inflight_prefill_tokens = 0;
            b.inflight_read_tokens = 0;
        } else if prev_inflight > 0 {
            b.inflight_read_tokens =
                (b.inflight_read_tokens * (b.inflight_requests as u64)) / (prev_inflight as u64);
        }

        if observed_ttft_ms.is_finite() && observed_ttft_ms >= 0.0 {
            if b.total_completed == 0 {
                b.ewma_observed_ttft_ms = observed_ttft_ms;
            } else {
                b.ewma_observed_ttft_ms =
                    0.25 * observed_ttft_ms + 0.75 * b.ewma_observed_ttft_ms;
            }
        }

        b.total_completed = b.total_completed.saturating_add(1);
        b.total_decode_tokens = b.total_decode_tokens.saturating_add(decode_tokens as u64);
    }

    fn find_backend_index(&self, backend_id: usize) -> Option<usize> {
        if backend_id < self.backends.len() && self.backends[backend_id].config.id == backend_id {
            return Some(backend_id);
        }
        self.backends
            .iter()
            .position(|b| b.config.id == backend_id)
    }

    fn pick_least_loaded(&self, evals: &[CandidateEval]) -> usize {
        evals
            .iter()
            .min_by(|a, b| {
                let ba = &self.backends[a.idx];
                let bb = &self.backends[b.idx];
                ba.inflight_requests
                    .cmp(&bb.inflight_requests)
                    .then_with(|| ba.inflight_prefill_tokens.cmp(&bb.inflight_prefill_tokens))
                    .then_with(|| {
                        a.queue_wait_ms
                            .partial_cmp(&b.queue_wait_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| a.idx.cmp(&b.idx))
            })
            .map(|e| e.idx)
            .unwrap_or(0)
    }

    /// Ranvier-style single-home prefix routing with 2x load-imbalance diversion.
    ///
    /// Pins each prefix to its single home node (`ch_home`) and diverts to the
    /// least-loaded backend only when `home_inflight >= max_inflight` or
    /// `home_inflight >= 2 * min_inflight` (without multi-replica tracking or
    /// Shadow-KV eviction/TTFT prediction).
    fn select_single_home_divert(
        &self,
        evals: &[CandidateEval],
        ch_home: usize,
    ) -> (usize, &'static str, bool) {
        let home_idx = ch_home;
        let home_state = &self.backends[home_idx];
        let max_inf = home_state.config.max_inflight.max(1);
        let least_idx = self.pick_least_loaded(evals);
        let least_state = &self.backends[least_idx];

        let two_x_divert = least_idx != home_idx
            && home_state.inflight_requests >= 2
            && home_state.inflight_requests >= least_state.inflight_requests.saturating_mul(2).max(2);

        let overloaded = home_state.inflight_requests >= max_inf || two_x_divert;

        if overloaded && least_idx != home_idx {
            let rep = evals[home_idx].matched_tokens > 0 && evals[least_idx].matched_tokens == 0;
            (least_idx, "single_home_divert", rep)
        } else if evals[home_idx].matched_tokens > 0 {
            (home_idx, "single_home_hit", false)
        } else {
            (home_idx, "single_home_cold", false)
        }
    }

    /// CachePin replication-aware shadow-KV adaptive radix policy.
    fn select_adaptive_replication(
        &self,
        evals: &[CandidateEval],
        ch_home: usize,
        horizon_hits: f64,
        active_replicas: usize,
    ) -> (usize, &'static str, bool) {
        let max_matched = evals.iter().map(|e| e.matched_tokens).max().unwrap_or(0);

        // Case 1: Cold prefix (no warm holder anywhere in the fleet).
        // Place on consistent-hash home when healthy, or pivot to lowest
        // TTFT + collateral-eviction backend when home is thrashing or congested.
        if max_matched == 0 {
            let best_cold = evals
                .iter()
                .min_by(|a, b| {
                    let bias_a = if a.idx == ch_home { 0.0 } else { 5.0 };
                    let bias_b = if b.idx == ch_home { 0.0 } else { 5.0 };
                    let score_a = a.est_ttft_ms + a.collateral_evict_ms + bias_a;
                    let score_b = b.est_ttft_ms + b.collateral_evict_ms + bias_b;
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.idx.cmp(&b.idx))
                })
                .map(|e| e.idx)
                .unwrap_or(ch_home);
            return (best_cold, "adaptive_cold_place", false);
        }

        // Case 2: At least one warm holder exists.
        // Pick the fastest warm holder (automatically load-balancing across ALL active replicas!).
        let warm_threshold = (max_matched / 2).max(1);
        let best_warm = evals
            .iter()
            .filter(|e| e.matched_tokens >= warm_threshold)
            .min_by(|a, b| {
                let sa = a.est_ttft_ms + 0.25 * a.collateral_evict_ms;
                let sb = b.est_ttft_ms + 0.25 * b.collateral_evict_ms;
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.matched_tokens.cmp(&a.matched_tokens))
                    .then_with(|| a.idx.cmp(&b.idx))
            })
            .copied()
            .unwrap_or(evals[0]);

        // Identify the lowest-pressure non-replica candidate for potential selective replication.
        let best_cold_opt = evals
            .iter()
            .filter(|e| e.matched_tokens < warm_threshold)
            .min_by(|a, b| {
                let sa = a.est_ttft_ms + a.collateral_evict_ms;
                let sb = b.est_ttft_ms + b.collateral_evict_ms;
                sa.partial_cmp(&sb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.idx.cmp(&b.idx))
            })
            .copied();

        let Some(best_cold) = best_cold_opt else {
            // Prefix is already replicated across every backend in the cluster!
            return (best_warm.idx, "adaptive_warm_hit", false);
        };

        let warm_state = &self.backends[best_warm.idx];
        let warm_saturated = warm_state.inflight_requests >= warm_state.config.max_inflight.max(1);
        let replica_count = active_replicas.max(1) as f64;

        // Extra GPU prefill + collateral eviction cost to establish an (R + 1)-th replica.
        let extra_miss_tokens = best_cold.miss_tokens.saturating_sub(best_warm.miss_tokens);
        let cold_prefill_tps = self.backends[best_cold.idx].prefill_tps();
        let rep_prefill_ms = (extra_miss_tokens as f64 / cold_prefill_tps) * 1000.0;
        let total_rep_cost_ms = rep_prefill_ms + best_cold.collateral_evict_ms;

        // Near-term hits served by the new replica scale as `horizon_hits / (R + 1)`.
        let eff_hits = horizon_hits.max(1.0);
        let amortized_rep_ms = total_rep_cost_ms * (replica_count + 1.0) / eff_hits;

        // Trigger 1: Hot-prefix saturation (proactive selective replication).
        // Only fires for genuinely hot prefixes (`horizon_hits >= 2.5`) whose warmest
        // holder is saturated or building queue wait exceeding both 25% of cold prefill
        // and the amortized (R + 1)-replica establishment cost.
        let min_queue_gate = (0.25 * rep_prefill_ms).max(40.0);
        let hot_saturation = eff_hits >= 2.5
            && (warm_saturated || best_warm.queue_wait_ms > min_queue_gate)
            && best_warm.queue_wait_ms > best_cold.queue_wait_ms + amortized_rep_ms
            && (best_cold.est_ttft_ms + 0.5 * best_cold.collateral_evict_ms)
                < best_warm.est_ttft_ms * 1.4 + 80.0;

        // Trigger 2: Immediate TTFT victory even including collateral eviction cost
        // (hard overload protection).
        let immediate_win =
            best_cold.est_ttft_ms + 0.5 * best_cold.collateral_evict_ms < best_warm.est_ttft_ms;

        if hot_saturation {
            (best_cold.idx, "adaptive_hot_replicate", true)
        } else if immediate_win {
            (best_cold.idx, "adaptive_overload_spill", true)
        } else {
            (best_warm.idx, "adaptive_warm_hit", false)
        }
    }

    fn admit_request(
        &mut self,
        chosen_idx: usize,
        tenant_key: u64,
        anchor_key: u64,
        blocks: &[PrefixBlock],
        eval: &CandidateEval,
        now_ms: u64,
    ) {
        let b = &mut self.backends[chosen_idx];
        b.inflight_requests = b.inflight_requests.saturating_add(1);
        b.inflight_prefill_tokens = b
            .inflight_prefill_tokens
            .saturating_add(eval.miss_tokens as u64);
        b.inflight_read_tokens = b
            .inflight_read_tokens
            .saturating_add(eval.matched_tokens as u64);
        b.total_routed = b.total_routed.saturating_add(1);

        // Insert/touch prefix blocks in REVERSE order so root blocks (`blocks[0]`)
        // sit at the MRU head and leaf blocks evict first under KV pressure!
        let mut evicted_tokens_total = 0u64;
        for blk in blocks.iter().rev() {
            let k = mix_tenant_prefix(tenant_key, blk.prefix_hash);
            let (_, ev_tok) = b.lru.touch_or_insert(k, blk.est_tokens, now_ms);
            evicted_tokens_total = evicted_tokens_total.saturating_add(ev_tok);
        }

        // Update EWMA eviction pressure cleanly in [0, +inf).
        let cap = b.config.kv_capacity_tokens.max(1) as f64;
        let instant_pressure = evicted_tokens_total as f64 / cap;
        b.eviction_pressure = 0.25 * instant_pressure + 0.75 * b.eviction_pressure;

        // Mark chosen backend in the prefix's active-replica bitmap.
        if chosen_idx < 128 {
            if let Some(h) = self.hotness.get_mut(&anchor_key) {
                h.replica_mask |= 1u128 << chosen_idx;
            }
        }
    }

    fn update_hotness_and_replicas(&mut self, anchor_key: u64, now_ms: u64) -> (f64, usize) {
        // Compute live replica mask from shadow KV LRUs so evicted replicas clear immediately.
        let mut live_mask = 0u128;
        let mut live_count = 0usize;
        for (idx, b) in self.backends.iter().enumerate() {
            if b.lru.contains(anchor_key) {
                live_count += 1;
                if idx < 128 {
                    live_mask |= 1u128 << idx;
                }
            }
        }

        // Bound hotness table memory cleanly.
        if self.hotness.len() >= MAX_HOTNESS_ENTRIES && !self.hotness.contains_key(&anchor_key) {
            let cutoff_ms = now_ms.saturating_sub(60_000);
            self.hotness.retain(|_, v| v.last_seen_ms >= cutoff_ms);
            if self.hotness.len() >= MAX_HOTNESS_ENTRIES {
                // Evict oldest single entry if all entries are fresh.
                if let Some(oldest_k) = self
                    .hotness
                    .iter()
                    .min_by_key(|(_, v)| v.last_seen_ms)
                    .map(|(&k, _)| k)
                {
                    self.hotness.remove(&oldest_k);
                }
            }
        }

        let entry = self.hotness.entry(anchor_key).or_insert(PrefixHotness {
            ewma_burst_hits: 0.0,
            last_seen_ms: now_ms,
            total_hits: 0,
            replica_mask: 0,
        });

        let dt_secs = now_ms.saturating_sub(entry.last_seen_ms) as f64 / 1000.0;
        let decay = (-dt_secs / HOTNESS_WINDOW_SECS).exp();
        entry.ewma_burst_hits = entry.ewma_burst_hits * decay + 1.0;
        entry.last_seen_ms = now_ms;
        entry.total_hits = entry.total_hits.saturating_add(1);
        entry.replica_mask = live_mask;

        // Require at least 2 cumulative observations before trusting burst extrapolation.
        let effective_hits = if entry.total_hits >= 2 {
            entry.ewma_burst_hits
        } else {
            entry.ewma_burst_hits.min(1.0)
        };
        let active = live_count.max(entry.replica_mask.count_ones() as usize);

        (effective_hits, active)
    }
}

/// Mix `tenant_key` and `prefix_hash` so distinct tenants can never collide in shadow KV LRUs.
#[inline]
fn mix_tenant_prefix(tenant_key: u64, prefix_hash: u64) -> u64 {
    let mut z = prefix_hash ^ tenant_key.wrapping_mul(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Select the representative prefix anchor hash for hotness tracking & consistent hashing:
/// the first block reaching >= 50% of `total_tokens` (capturing shared system/context prefixes
/// without hashing the unique trailing user query).
#[inline]
fn compute_prefix_anchor(tenant_key: u64, blocks: &[PrefixBlock], total_tokens: u32) -> u64 {
    if blocks.is_empty() {
        return mix_tenant_prefix(tenant_key, 0);
    }
    let half = (total_tokens / 2).max(1);
    let mut cum = 0u32;
    for blk in blocks {
        cum = cum.saturating_add(blk.est_tokens);
        if cum >= half {
            return mix_tenant_prefix(tenant_key, blk.prefix_hash);
        }
    }
    mix_tenant_prefix(tenant_key, blocks[0].prefix_hash)
}

/// Google Jump Consistent Hash: fast, uniform `O(ln N)` mapping into `0..num_buckets`.
#[inline]
fn jump_consistent_hash(mut key: u64, num_buckets: usize) -> usize {
    if num_buckets <= 1 {
        return 0;
    }
    let mut b: i64 = -1;
    let mut j: i64 = 0;
    while j < num_buckets as i64 {
        b = j;
        key = key.wrapping_mul(2_862_933_555_777_941_757).wrapping_add(1);
        let denom = ((key >> 33) + 1) as f64;
        j = (((b + 1) as f64) * (((1u64 << 31) as f64) / denom)) as i64;
    }
    (b.max(0) as usize).min(num_buckets - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    fn make_prefix_blocks(prefix_id: u64, shared_tokens: u32, turn_id: u64, tail_tokens: u32) -> Vec<PrefixBlock> {
        let h0 = mix_tenant_prefix(0xA11CE, prefix_id.wrapping_add(1));
        let h1 = mix_tenant_prefix(h0, turn_id.wrapping_add(10_000));
        vec![
            PrefixBlock {
                path: "system[0]".to_string(),
                kind: "system".to_string(),
                canonical_bytes: Vec::new(),
                block_hash: h0,
                prefix_hash: h0,
                est_tokens: shared_tokens,
                breakpoint_eligible: true,
            },
            PrefixBlock {
                path: "messages[0]".to_string(),
                kind: "message".to_string(),
                canonical_bytes: Vec::new(),
                block_hash: h1,
                prefix_hash: h1,
                est_tokens: tail_tokens,
                breakpoint_eligible: true,
            },
        ]
    }

    fn make_cluster(num_backends: usize, kv_cap: u64, max_inflight: usize) -> Vec<BackendConfig> {
        (0..num_backends)
            .map(|id| BackendConfig {
                id,
                url: format!("http://127.0.0.1:{}", 8000 + id),
                kv_capacity_tokens: kv_cap,
                prefill_tps: 10_000.0,
                cache_read_tps: 200_000.0,
                max_inflight,
            })
            .collect()
    }

    #[test]
    fn test_shadow_kv_lru_strict_capacity_and_leaf_first_eviction() {
        let mut lru = ShadowKvLru::new(1000);
        assert_eq!(lru.len(), 0);

        // Insert 3 blocks of 400 tokens each -> 3rd must evict the 1st so used <= 1000!
        lru.touch_or_insert(1, 400, 100);
        lru.touch_or_insert(2, 400, 200);
        assert_eq!(lru.used_kv_tokens, 800);

        // Touch block 1 so block 2 becomes LRU tail.
        lru.touch_or_insert(1, 400, 300);
        let (ev_cnt, ev_tok) = lru.touch_or_insert(3, 400, 400);
        assert_eq!(ev_cnt, 1);
        assert_eq!(ev_tok, 400);
        assert!(lru.used_kv_tokens <= 1000);
        assert!(lru.contains(1));
        assert!(!lru.contains(2));
        assert!(lru.contains(3));

        // Oversized block (> capacity) must not panic or break invariant.
        lru.touch_or_insert(99, 5000, 500);
        assert!(lru.used_kv_tokens <= 1000);
    }

    #[test]
    fn test_empty_and_single_backend_panic_free() {
        let mut empty = ShadowKvRouter::new(Vec::new(), RoutingPolicy::CachePinAdaptiveReplication);
        let blks = make_prefix_blocks(1, 2000, 1, 200);
        let dec = empty.route(1, &blks, 2200, 100);
        assert_eq!(dec.reason, "no_backends");
        empty.complete(99, 100, 10, 5.0); // Out-of-bounds id must never panic.

        let mut single =
            ShadowKvRouter::new(make_cluster(1, 16_000, 4), RoutingPolicy::CachePinAdaptiveReplication);
        let d1 = single.route(1, &blks, 2200, 100);
        assert_eq!(d1.backend_id, 0);
        assert_eq!(d1.est_matched_tokens, 0);
        single.complete(0, 2200, 64, d1.est_ttft_ms);

        let blks2 = make_prefix_blocks(1, 2000, 2, 200);
        let d2 = single.route(1, &blks2, 2200, 200);
        assert_eq!(d2.backend_id, 0);
        assert_eq!(d2.est_matched_tokens, 2000);
    }

    #[derive(Debug, Clone)]
    struct SimResult {
        mean_ttft_ms: f64,
        p99_ttft_ms: f64,
        hit_rate: f64,
        total_evictions: u64,
    }

    /// Deterministic discrete-event cluster simulation evaluating all 5 policies
    /// under tight KV memory (eviction regime) with skewed hot-swarm + cool-agent traffic.
    fn simulate_eviction_regime(policy: RoutingPolicy) -> SimResult {
        let num_backends = 4usize;
        // 4 GPUs x 24K KV = 96K total cluster KV capacity.
        // 19 prefixes x 4K = 76K working set: fits comfortably when cool prefixes are
        // single-homed and ONLY the hot swarm prefix selectively replicates!
        let backends = make_cluster(num_backends, 24_000, 3);
        let mut router = ShadowKvRouter::new(backends.clone(), policy);

        // Per-backend physical GPU prefill timeline (in microseconds for exact integer ordering).
        let mut gpu_busy_until_us = vec![0u64; num_backends];
        // Completion event min-heap: (complete_time_ms, seq, backend_id, miss_tokens, decode_tokens, ttft_ms_x100)
        let mut completions: BinaryHeap<Reverse<(u64, u64, usize, u32, u32, u64)>> =
            BinaryHeap::new();

        // Build deterministic mixed workload:
        // - Prefix 0 (HOT swarm): 55% of requests arriving in tight bursts!
        // - Prefixes 1..19 (18 cool/warm tenant agents): 45% of requests.
        let total_requests = 720u64;
        let mut rng: u64 = 0x5EED_2026_CAFE_BABE;
        let mut next_u64 = || {
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };

        let mut now_ms: u64 = 1_000;
        let mut ttfts: Vec<f64> = Vec::with_capacity(total_requests as usize);
        let mut total_matched_tok: u64 = 0;
        let mut total_req_tok: u64 = 0;

        for req_i in 0..total_requests {
            // Burst pattern: 6-request tight bursts (15ms apart) separated by 60ms lulls.
            let inter_ms = if req_i % 6 == 0 { 60 } else { 15 };
            now_ms += inter_ms;

            // Drain completed requests up to `now_ms`.
            while let Some(&Reverse((done_ms, _, b_id, miss_t, dec_t, ttft_x100))) =
                completions.peek()
            {
                if done_ms <= now_ms {
                    completions.pop();
                    router.complete(b_id, miss_t, dec_t, ttft_x100 as f64 / 100.0);
                } else {
                    break;
                }
            }

            let roll = next_u64() % 100;
            let prefix_id = if roll < 55 {
                0u64 // Hot swarm prefix!
            } else {
                1 + (next_u64() % 18) // 18 cool/warm tenant prefixes
            };

            let shared_tokens = 4_000u32;
            let tail_tokens = 128u32;
            let total_tokens = shared_tokens + tail_tokens;
            let blocks = make_prefix_blocks(prefix_id, shared_tokens, req_i, tail_tokens);

            let dec = router.route(1, &blocks, total_tokens, now_ms);
            let b_id = dec.backend_id;
            let matched = dec.est_matched_tokens;
            let miss = total_tokens.saturating_sub(matched);

            total_matched_tok += matched as u64;
            total_req_tok += total_tokens as u64;

            let now_us = now_ms * 1_000;
            let start_us = gpu_busy_until_us[b_id].max(now_us);
            let queue_wait_ms = (start_us - now_us) as f64 / 1_000.0;
            let service_ms = (miss as f64 / backends[b_id].prefill_tps
                + matched as f64 / backends[b_id].cache_read_tps)
                * 1000.0;
            let actual_ttft_ms = queue_wait_ms + service_ms;
            gpu_busy_until_us[b_id] = start_us + (service_ms * 1_000.0) as u64;

            // Skip initial cold warmup (first 40 requests) when measuring steady-state TTFT.
            if req_i >= 40 {
                ttfts.push(actual_ttft_ms);
            }

            // Request holds KV/concurrency slot through prefill + 120ms decode phase.
            let done_ms = now_ms + actual_ttft_ms.ceil() as u64 + 120;
            completions.push(Reverse((
                done_ms,
                req_i,
                b_id,
                miss,
                64,
                (actual_ttft_ms * 100.0) as u64,
            )));
        }

        ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_ttft_ms = ttfts.iter().sum::<f64>() / (ttfts.len().max(1) as f64);
        let p99_idx = ((ttfts.len() as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(ttfts.len().saturating_sub(1));
        let p99_ttft_ms = ttfts[p99_idx];
        let hit_rate = total_matched_tok as f64 / (total_req_tok.max(1) as f64);
        let total_evictions = (0..num_backends)
            .filter_map(|id| router.backend_stats(id))
            .map(|s| s.total_evictions)
            .sum();

        SimResult {
            mean_ttft_ms,
            p99_ttft_ms,
            hit_rate,
            total_evictions,
        }
    }

    #[test]
    fn test_eviction_regime_adaptive_replication_beats_all_baselines() {
        let rr = simulate_eviction_regime(RoutingPolicy::RoundRobin);
        let ll = simulate_eviction_regime(RoutingPolicy::LeastLoaded);
        let sh = simulate_eviction_regime(RoutingPolicy::SingleHomePrefixDivert);
        let cp = simulate_eviction_regime(RoutingPolicy::CachePinAdaptiveReplication);

        // CachePinAdaptiveReplication must decisively beat RoundRobin, LeastLoaded,
        // and SingleHomePrefixDivert on BOTH mean TTFT and P99 TTFT in eviction regime!
        assert!(
            cp.mean_ttft_ms < rr.mean_ttft_ms,
            "Adaptive mean TTFT ({:.1}ms) should beat RoundRobin ({:.1}ms)",
            cp.mean_ttft_ms,
            rr.mean_ttft_ms
        );
        assert!(
            cp.mean_ttft_ms < ll.mean_ttft_ms,
            "Adaptive mean TTFT ({:.1}ms) should beat LeastLoaded ({:.1}ms)",
            cp.mean_ttft_ms,
            ll.mean_ttft_ms
        );
        assert!(
            cp.mean_ttft_ms < sh.mean_ttft_ms,
            "Adaptive mean TTFT ({:.1}ms) should beat SingleHomePrefixDivert ({:.1}ms)",
            cp.mean_ttft_ms,
            sh.mean_ttft_ms
        );
        assert!(
            cp.p99_ttft_ms < rr.p99_ttft_ms,
            "Adaptive P99 TTFT ({:.1}ms) should beat RoundRobin ({:.1}ms)",
            cp.p99_ttft_ms,
            rr.p99_ttft_ms
        );
        assert!(
            cp.p99_ttft_ms < ll.p99_ttft_ms,
            "Adaptive P99 TTFT ({:.1}ms) should beat LeastLoaded ({:.1}ms)",
            cp.p99_ttft_ms,
            ll.p99_ttft_ms
        );
        assert!(
            cp.p99_ttft_ms < sh.p99_ttft_ms,
            "Adaptive P99 TTFT ({:.1}ms) should beat SingleHomePrefixDivert ({:.1}ms)",
            cp.p99_ttft_ms,
            sh.p99_ttft_ms
        );
        assert!(
            cp.hit_rate > rr.hit_rate && cp.hit_rate > ll.hit_rate,
            "Adaptive hit rate ({:.3}) should beat RR ({:.3}) and LL ({:.3})",
            cp.hit_rate,
            rr.hit_rate,
            ll.hit_rate
        );
        assert!(
            cp.total_evictions < rr.total_evictions && cp.total_evictions < ll.total_evictions,
            "Adaptive evictions ({}) should be far below RR ({}) and LL ({})",
            cp.total_evictions,
            rr.total_evictions,
            ll.total_evictions
        );
    }

    #[test]
    fn test_hot_prefix_triggers_selective_replication_while_cool_stays_pinned() {
        let mut router = ShadowKvRouter::new(
            make_cluster(4, 24_000, 2),
            RoutingPolicy::CachePinAdaptiveReplication,
        );

        // Warm up hot prefix 0 on its home backend.
        let mut replicated = false;
        let mut used_backends = std::collections::HashSet::new();
        for i in 0..8u64 {
            let blks = make_prefix_blocks(0, 4_000, i, 128);
            let dec = router.route(1, &blks, 4_128, 1_000 + i * 10);
            used_backends.insert(dec.backend_id);
            if dec.triggered_replication {
                replicated = true;
            }
        }
        assert!(
            replicated && used_backends.len() >= 2,
            "Saturated hot prefix must selectively replicate onto >= 2 backends"
        );

        // Drain all inflight requests.
        for b in 0..4 {
            for _ in 0..8 {
                router.complete(b, 4_128, 64, 50.0);
            }
        }

        // Cool prefix 99 arrives spaced out -> must stay pinned to a single home without replication.
        let b1 = make_prefix_blocks(99, 4_000, 1, 128);
        let d1 = router.route(1, &b1, 4_128, 10_000);
        router.complete(d1.backend_id, 4_128, 64, 420.0);

        let b2 = make_prefix_blocks(99, 4_000, 2, 128);
        let d2 = router.route(1, &b2, 4_128, 40_000);
        assert_eq!(d2.backend_id, d1.backend_id);
        assert_eq!(d2.est_matched_tokens, 4_000);
        assert!(!d2.triggered_replication);
    }
}
