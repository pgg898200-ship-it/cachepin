//! Exact `O(M^2 * K)` constrained mixed-TTL Dynamic Programming breakpoint & TTL optimizer.
//!
//! Solves the exact optimal placement of up to `K = pricing.max_breakpoints`
//! prompt-cache breakpoints with mixed `1h` / `5m` TTL tiers under all provider
//! constraints:
//! 1. At most `pricing.max_breakpoints` breakpoints.
//! 2. First breakpoint cumulative prefix tokens >= `pricing.min_cache_tokens`.
//! 3. Only `block.breakpoint_eligible == true` blocks may carry breakpoints.
//! 4. Strict mixed-TTL monotonicity (Anthropic A/B/C rule): every `OneHour`
//!    breakpoint MUST precede all `FiveMin` breakpoints; `1h` is forbidden when
//!    `!pricing.supports_1h_ttl`.
//! 5. Zero wasteful live TTL churn: if a prefix up to block `i` is already warm
//!    at TTL `tau`, incremental turn-0 write cost is `0.0` (only read cost applies).
//!
//! Also evaluates the Octomind / Anthropic keepalive break-even formula
//! comparing `1h` write premium against `max_tokens: 0` 5-minute keepalive pings
//! (`interval = 270s`).

use crate::types::{BreakpointPlan, CacheTtl, ModelPricing, PlannedBreakpoint, PrefixBlock};

/// Safety-margined keepalive interval in seconds for 5-minute (300s) TTL caches.
pub const KEEPALIVE_INTERVAL_SECS: u32 = 270;

const COST_TIE_EPS: f64 = 1e-12;
const MAX_HITS_CLAMP: f64 = 1e9;

/// Per-block reuse and cache-warmth estimate fed into the exact DP solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockReuseEstimate {
    /// Expected subsequent requests matching prefix `0..=i` within a 5m TTL window.
    pub expected_hits_5m: f64,
    /// Expected subsequent requests matching prefix `0..=i` within a 1h TTL window.
    pub expected_hits_1h: f64,
    /// EWMA inter-arrival gap in seconds for prefix `0..=i` (`0.0` if unknown).
    pub ewma_gap_secs: f64,
    /// Active TTL tier if this prefix is currently warm in provider cache.
    pub currently_warm_ttl: Option<CacheTtl>,
    /// Whether prefix `0..=i` is currently warm in provider cache.
    pub already_warm: bool,
}

impl BlockReuseEstimate {
    /// Cold, zero-reuse block estimate (e.g., volatile query tail).
    pub const fn cold() -> Self {
        Self {
            expected_hits_5m: 0.0,
            expected_hits_1h: 0.0,
            ewma_gap_secs: 0.0,
            currently_warm_ttl: None,
            already_warm: false,
        }
    }
}

impl Default for BlockReuseEstimate {
    fn default() -> Self {
        Self::cold()
    }
}

#[inline]
fn sanitize_hits(v: f64) -> f64 {
    if v.is_finite() && v > 0.0 {
        v.min(MAX_HITS_CLAMP)
    } else {
        0.0
    }
}

#[inline]
fn tier_to_ttl(tier: usize) -> CacheTtl {
    if tier == 0 {
        CacheTtl::OneHour
    } else {
        CacheTtl::FiveMin
    }
}

/// Precomputed per-block cost tables for fast `O(1)` segment updates.
struct PrecomputedBlocks {
    m: usize,
    cum_tokens: Vec<u64>,
    h5: Vec<f64>,
    h1: Vec<f64>,
    in_cost: Vec<f64>,
    read_cost: Vec<f64>,
    write_5m_cost: Vec<f64>,
    write_1h_cost: Vec<f64>,
    tail_uncached: Vec<f64>,
    warm_5m: Vec<bool>,
    warm_1h: Vec<bool>,
    ewma_gap: Vec<f64>,
}

impl PrecomputedBlocks {
    fn new(blocks: &[PrefixBlock], reuse: &[BlockReuseEstimate], pricing: &ModelPricing) -> Self {
        let m = blocks.len();
        let mut cum_tokens = Vec::with_capacity(m);
        let mut h5 = Vec::with_capacity(m);
        let mut h1 = Vec::with_capacity(m);
        let mut in_cost = Vec::with_capacity(m);
        let mut read_cost = Vec::with_capacity(m);
        let mut write_5m_cost = Vec::with_capacity(m);
        let mut write_1h_cost = Vec::with_capacity(m);
        let mut uncached_block = Vec::with_capacity(m);
        let mut warm_5m = Vec::with_capacity(m);
        let mut warm_1h = Vec::with_capacity(m);
        let mut ewma_gap = Vec::with_capacity(m);

        let mut running_tokens: u64 = 0;
        for (i, blk) in blocks.iter().enumerate() {
            let tok = blk.est_tokens as u64;
            running_tokens = running_tokens.saturating_add(tok);
            cum_tokens.push(running_tokens);

            let est = reuse.get(i).copied().unwrap_or_else(BlockReuseEstimate::cold);
            let e5 = sanitize_hits(est.expected_hits_5m);
            let e1 = sanitize_hits(est.expected_hits_1h).max(e5);
            h5.push(e5);
            h1.push(e1);

            let ci = pricing.cost_input(tok);
            let cr = pricing.cost_read(tok);
            let cw5 = pricing.cost_write(tok, CacheTtl::FiveMin);
            let cw1 = pricing.cost_write(tok, CacheTtl::OneHour);

            in_cost.push(ci);
            read_cost.push(cr);
            write_5m_cost.push(cw5);
            write_1h_cost.push(cw1);
            uncached_block.push((1.0 + e1) * ci);

            warm_5m.push(est.already_warm && est.currently_warm_ttl == Some(CacheTtl::FiveMin));
            warm_1h.push(est.already_warm && est.currently_warm_ttl == Some(CacheTtl::OneHour));
            ewma_gap.push(if est.ewma_gap_secs.is_finite() && est.ewma_gap_secs > 0.0 {
                est.ewma_gap_secs
            } else {
                0.0
            });
        }

        let mut tail_uncached = vec![0.0; m + 1];
        for i in (0..m).rev() {
            tail_uncached[i] = tail_uncached[i + 1] + uncached_block[i];
        }

        Self {
            m,
            cum_tokens,
            h5,
            h1,
            in_cost,
            read_cost,
            write_5m_cost,
            write_1h_cost,
            tail_uncached,
            warm_5m,
            warm_1h,
            ewma_gap,
        }
    }

    /// Exact expected horizon cost attributed to block `j` when covered by a
    /// breakpoint at `bp_idx` (`bp_idx >= j`) with TTL tier `tier` (`0 = 1h, 1 = 5m`).
    #[inline]
    fn block_cost_in_segment(&self, j: usize, bp_idx: usize, tier: usize) -> f64 {
        let (bp_hits, is_warm_same_ttl, turn0_write) = if tier == 0 {
            (self.h1[bp_idx], self.warm_1h[bp_idx], self.write_1h_cost[j])
        } else {
            (self.h5[bp_idx], self.warm_5m[bp_idx], self.write_5m_cost[j])
        };
        let turn0 = if is_warm_same_ttl {
            self.read_cost[j]
        } else {
            turn0_write
        };
        let total_future = self.h1[j];
        let hits = bp_hits.min(total_future);
        let misses = total_future - hits;
        turn0 + hits * self.read_cost[j] + misses * self.in_cost[j]
    }

    /// Exact cost of a segment `(prev_bp, curr_bp]` with TTL `tier`.
    fn segment_cost(&self, start_idx: usize, bp_idx: usize, tier: usize) -> f64 {
        let mut total = 0.0;
        for j in start_idx..=bp_idx {
            total += self.block_cost_in_segment(j, bp_idx, tier);
        }
        total
    }
}

/// Evaluate the Octomind / Anthropic keepalive break-even condition.
///
/// Compares `max_tokens: 0` 5-minute keepalive pings every `270s` against both:
/// 1. Allowing a 5m cache to expire and paying the cold 5m rewrite delta, and
/// 2. Paying the 1h cache write premium (when `pricing.supports_1h_ttl` and `gap <= 3600s`).
///
/// Returns `(keepalive_recommended, keepalive_interval_secs)`.
fn evaluate_keepalive(
    breakpoints: &[PlannedBreakpoint],
    pre: &PrecomputedBlocks,
    pricing: &ModelPricing,
) -> (bool, Option<u32>) {
    let interval = KEEPALIVE_INTERVAL_SECS as f64;
    for bp in breakpoints {
        let idx = bp.block_index;
        if idx >= pre.m {
            continue;
        }
        let gap = pre.ewma_gap[idx];
        let tokens = pre.cum_tokens[idx];

        // Natural traffic refreshes 5m TTL when gap <= 270s; beyond 2h keepalive is impractical.
        if !(gap > interval && gap <= 7200.0) {
            continue;
        }
        if tokens < pricing.min_cache_tokens as u64 || pre.h1[idx] <= 0.0 {
            continue;
        }
        // Already live at 1h covering this gap: no 5m keepalive pings needed.
        if pre.warm_1h[idx] && gap <= CacheTtl::OneHour.seconds() {
            continue;
        }

        let pings = ((gap - interval) / interval).ceil().max(1.0);
        let keepalive_cost = pings * pricing.cost_read(tokens);
        let cold_rewrite_extra =
            (pricing.cost_write(tokens, CacheTtl::FiveMin) - pricing.cost_read(tokens)).max(0.0);

        if keepalive_cost >= cold_rewrite_extra - COST_TIE_EPS {
            continue;
        }

        if pricing.supports_1h_ttl && gap <= CacheTtl::OneHour.seconds() {
            let one_hour_premium = (pricing.cost_write(tokens, CacheTtl::OneHour)
                - pricing.cost_write(tokens, CacheTtl::FiveMin))
            .max(0.0);
            if keepalive_cost < one_hour_premium - COST_TIE_EPS {
                return (true, Some(KEEPALIVE_INTERVAL_SECS));
            }
        } else {
            return (true, Some(KEEPALIVE_INTERVAL_SECS));
        }
    }
    (false, None)
}

/// Solve the exact constrained mixed-TTL breakpoint placement problem in `O(M^2 * K)` time.
pub fn solve_optimal_breakpoints(
    blocks: &[PrefixBlock],
    reuse: &[BlockReuseEstimate],
    pricing: &ModelPricing,
) -> BreakpointPlan {
    let m = blocks.len();
    if m == 0 {
        return BreakpointPlan {
            breakpoints: Vec::new(),
            expected_cost_usd: 0.0,
            uncached_cost_usd: 0.0,
            naive_single_bp_cost_usd: 0.0,
            expected_savings_vs_uncached_pct: 0.0,
            keepalive_recommended: false,
            keepalive_interval_secs: None,
        };
    }

    let pre = PrecomputedBlocks::new(blocks, reuse, pricing);
    let uncached_cost_usd = pre.tail_uncached[0];
    let min_cache_tokens = pricing.min_cache_tokens as u64;
    let max_k = pricing.max_breakpoints.min(m);

    // Counterfactual baseline: single 5m breakpoint on the LAST eligible block
    // satisfying `min_cache_tokens` (standard auto-cache / naive harness behavior).
    let naive_single_bp_cost_usd = if max_k >= 1 {
        let last_eligible = (0..m)
            .rev()
            .find(|&i| blocks[i].breakpoint_eligible && pre.cum_tokens[i] >= min_cache_tokens);
        match last_eligible {
            Some(bp_idx) => pre.segment_cost(0, bp_idx, 1) + pre.tail_uncached[bp_idx + 1],
            None => uncached_cost_usd,
        }
    } else {
        uncached_cost_usd
    };

    if max_k == 0 {
        return BreakpointPlan {
            breakpoints: Vec::new(),
            expected_cost_usd: uncached_cost_usd,
            uncached_cost_usd,
            naive_single_bp_cost_usd,
            expected_savings_vs_uncached_pct: 0.0,
            keepalive_recommended: false,
            keepalive_interval_secs: None,
        };
    }

    // Allowed TTL tiers: tier 0 = OneHour, tier 1 = FiveMin.
    // Mixed-TTL monotonicity forbids transitioning from tier 1 -> tier 0.
    let allowed_tiers: &[usize] = if pricing.supports_1h_ttl { &[0, 1] } else { &[1] };

    // Flattened DP tables: index = ((k * m) + i) * 2 + tier, for k in 1..=max_k.
    let stride_k = m * 2;
    let total_states = (max_k + 1) * stride_k;
    let mut dp = vec![f64::INFINITY; total_states];
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; total_states];

    let dp_idx = |k: usize, i: usize, tier: usize| -> usize { k * stride_k + i * 2 + tier };

    // Fill DP table in O(M^2 * K) time with O(1) incremental segment-cost updates.
    for curr in 0..m {
        if !blocks[curr].breakpoint_eligible {
            continue;
        }
        let first_bp_legal = pre.cum_tokens[curr] >= min_cache_tokens;

        for &tier in allowed_tiers {
            let mut seg_cost = 0.0;

            // Sweep segment start `seg_start` backwards from `curr` down to `0`.
            // Segment covers `[seg_start..=curr]` with breakpoint at `curr` (TTL `tier`).
            for seg_start in (0..=curr).rev() {
                seg_cost += pre.block_cost_in_segment(seg_start, curr, tier);

                if seg_start == 0 {
                    // Base case: k = 1 breakpoint at `curr`.
                    if first_bp_legal {
                        let dst = dp_idx(1, curr, tier);
                        if seg_cost < dp[dst] - COST_TIE_EPS {
                            dp[dst] = seg_cost;
                            parent[dst] = None;
                        }
                    }
                } else {
                    let prev = seg_start - 1;
                    if !blocks[prev].breakpoint_eligible {
                        continue;
                    }
                    // Transition (k, prev, prev_tier) -> (k + 1, curr, tier)
                    // Strict mixed-TTL monotonicity: prev_tier <= tier.
                    for k in 1..max_k {
                        for &prev_tier in allowed_tiers {
                            if prev_tier > tier {
                                continue;
                            }
                            let prev_cost = dp[dp_idx(k, prev, prev_tier)];
                            if !prev_cost.is_finite() {
                                continue;
                            }
                            let cand = prev_cost + seg_cost;
                            let dst = dp_idx(k + 1, curr, tier);
                            if cand < dp[dst] - COST_TIE_EPS {
                                dp[dst] = cand;
                                parent[dst] = Some((prev, prev_tier));
                            }
                        }
                    }
                }
            }
        }
    }

    // Select global optimum across k = 0..=max_k (breaking exact ties toward
    // fewer breakpoints, then FiveMin over OneHour, then earlier block index).
    let mut best_cost = uncached_cost_usd;
    let mut best_choice: Option<(usize, usize, usize)> = None; // (k, last_idx, last_tier)

    for k in 1..=max_k {
        for curr in 0..m {
            if !blocks[curr].breakpoint_eligible {
                continue;
            }
            let tail = pre.tail_uncached[curr + 1];
            // Check FiveMin (tier 1) before OneHour (tier 0) so exact ties favor FiveMin.
            for &tier in allowed_tiers.iter().rev() {
                let prefix_cost = dp[dp_idx(k, curr, tier)];
                if !prefix_cost.is_finite() {
                    continue;
                }
                let total = prefix_cost + tail;
                let strictly_better = total < best_cost - COST_TIE_EPS;
                if strictly_better {
                    best_cost = total;
                    best_choice = Some((k, curr, tier));
                }
            }
        }
    }

    // Reconstruct optimal breakpoint sequence.
    let mut breakpoints = Vec::new();
    if let Some((mut k, mut idx, mut tier)) = best_choice {
        breakpoints.reserve(k);
        loop {
            breakpoints.push(PlannedBreakpoint {
                block_index: idx,
                path: blocks[idx].path.clone(),
                cumulative_tokens: pre.cum_tokens[idx].min(u32::MAX as u64) as u32,
                ttl: tier_to_ttl(tier),
            });
            if k <= 1 {
                break;
            }
            if let Some((prev_idx, prev_tier)) = parent[dp_idx(k, idx, tier)] {
                k -= 1;
                idx = prev_idx;
                tier = prev_tier;
            } else {
                break;
            }
        }
        breakpoints.reverse();
    }

    let expected_cost_usd = best_cost.max(0.0);
    let expected_savings_vs_uncached_pct = if uncached_cost_usd > 1e-15 {
        ((uncached_cost_usd - expected_cost_usd) / uncached_cost_usd * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    };

    let (keepalive_recommended, keepalive_interval_secs) =
        evaluate_keepalive(&breakpoints, &pre, pricing);

    BreakpointPlan {
        breakpoints,
        expected_cost_usd,
        uncached_cost_usd,
        naive_single_bp_cost_usd,
        expected_savings_vs_uncached_pct,
        keepalive_recommended,
        keepalive_interval_secs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pricing::lookup_pricing;
    use crate::types::ProviderFormat;

    fn make_block(idx: usize, tokens: u32, eligible: bool) -> PrefixBlock {
        PrefixBlock {
            path: format!("block[{idx}]"),
            kind: "message".to_string(),
            canonical_bytes: vec![b'a'; 16],
            block_hash: (idx as u64).wrapping_add(1),
            prefix_hash: (idx as u64).wrapping_add(100),
            est_tokens: tokens,
            breakpoint_eligible: eligible,
        }
    }

    /// Exhaustive brute-force solver over ALL valid breakpoint subsets & TTLs.
    /// Used strictly to prove exact numerical optimality (`< 1e-12`) of our `O(M^2 K)` DP.
    fn brute_force_optimal(
        blocks: &[PrefixBlock],
        reuse: &[BlockReuseEstimate],
        pricing: &ModelPricing,
    ) -> (f64, Vec<(usize, CacheTtl)>) {
        let pre = PrecomputedBlocks::new(blocks, reuse, pricing);
        let m = blocks.len();
        let max_k = pricing.max_breakpoints.min(m);
        let min_tok = pricing.min_cache_tokens as u64;
        let allowed_tiers: &[usize] = if pricing.supports_1h_ttl { &[0, 1] } else { &[1] };

        let mut best_cost = pre.tail_uncached[0];
        let mut best_seq: Vec<(usize, CacheTtl)> = Vec::new();
        let mut current: Vec<(usize, usize)> = Vec::new();

        #[allow(clippy::too_many_arguments)]
        fn dfs(
            start: usize,
            min_tier: usize,
            cost_so_far: f64,
            max_k: usize,
            min_tok: u64,
            allowed_tiers: &[usize],
            blocks: &[PrefixBlock],
            pre: &PrecomputedBlocks,
            current: &mut Vec<(usize, usize)>,
            best_cost: &mut f64,
            best_seq: &mut Vec<(usize, CacheTtl)>,
        ) {
            // Evaluate closing the plan right here with uncached tail.
            if !current.is_empty() {
                let last_idx = current[current.len() - 1].0;
                let total = cost_so_far + pre.tail_uncached[last_idx + 1];
                if total < *best_cost - COST_TIE_EPS {
                    *best_cost = total;
                    *best_seq = current
                        .iter()
                        .map(|&(i, t)| (i, tier_to_ttl(t)))
                        .collect();
                }
            }
            if current.len() == max_k {
                return;
            }

            let seg_start = if current.is_empty() {
                0
            } else {
                current[current.len() - 1].0 + 1
            };

            for idx in start..blocks.len() {
                if !blocks[idx].breakpoint_eligible {
                    continue;
                }
                if current.is_empty() && pre.cum_tokens[idx] < min_tok {
                    continue;
                }
                for &tier in allowed_tiers {
                    if tier < min_tier {
                        continue;
                    }
                    let seg_cost = pre.segment_cost(seg_start, idx, tier);
                    current.push((idx, tier));
                    dfs(
                        idx + 1,
                        tier,
                        cost_so_far + seg_cost,
                        max_k,
                        min_tok,
                        allowed_tiers,
                        blocks,
                        pre,
                        current,
                        best_cost,
                        best_seq,
                    );
                    current.pop();
                }
            }
        }

        dfs(
            0,
            0,
            0.0,
            max_k,
            min_tok,
            allowed_tiers,
            blocks,
            &pre,
            &mut current,
            &mut best_cost,
            &mut best_seq,
        );
        (best_cost, best_seq)
    }

    /// Deterministic SplitMix64 PRNG (zero external crates).
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }
        fn gen_range(&mut self, upper: u64) -> u64 {
            self.next_u64() % upper
        }
        fn gen_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
        }
    }

    fn verify_plan_invariants(
        blocks: &[PrefixBlock],
        pricing: &ModelPricing,
        plan: &BreakpointPlan,
    ) {
        assert!(plan.breakpoints.len() <= pricing.max_breakpoints);
        assert!(plan.expected_cost_usd <= plan.uncached_cost_usd + 1e-12);
        assert!(plan.expected_cost_usd <= plan.naive_single_bp_cost_usd + 1e-12);
        assert!(plan.expected_savings_vs_uncached_pct >= 0.0);
        assert!(plan.expected_savings_vs_uncached_pct <= 100.0);

        let mut prev_idx: Option<usize> = None;
        let mut saw_five_min = false;
        for (k_i, bp) in plan.breakpoints.iter().enumerate() {
            assert!(blocks[bp.block_index].breakpoint_eligible);
            if k_i == 0 {
                assert!(bp.cumulative_tokens >= pricing.min_cache_tokens);
            }
            if let Some(p) = prev_idx {
                assert!(bp.block_index > p);
            }
            prev_idx = Some(bp.block_index);

            if !pricing.supports_1h_ttl {
                assert_eq!(bp.ttl, CacheTtl::FiveMin);
            }
            // Anthropic A/B/C mixed-TTL monotonicity: OneHour MUST precede FiveMin!
            match bp.ttl {
                CacheTtl::OneHour => {
                    assert!(
                        !saw_five_min,
                        "Mixed-TTL monotonicity violated: OneHour appeared after FiveMin!"
                    );
                }
                CacheTtl::FiveMin => {
                    saw_five_min = true;
                }
            }
        }
    }

    #[test]
    fn test_dp_exact_equality_vs_brute_force_randomized_trees() {
        let pricing_profiles = [
            lookup_pricing("claude-opus-5-5", ProviderFormat::AnthropicMessages),
            lookup_pricing("claude-sonnet-5-5", ProviderFormat::AnthropicMessages),
            lookup_pricing("claude-fable-5-1", ProviderFormat::AnthropicMessages),
            lookup_pricing("claude-sonnet-4", ProviderFormat::AnthropicMessages),
            lookup_pricing("gpt-5.6", ProviderFormat::OpenAiChat),
            lookup_pricing("gpt-5", ProviderFormat::OpenAiChat),
            lookup_pricing("gemini-4", ProviderFormat::Generic),
            lookup_pricing("vllm", ProviderFormat::VllmOpenAi),
        ];

        let mut rng = Rng(0xCACE_B10C_2026_0001);

        for trial in 0..240 {
            let pricing = &pricing_profiles[trial % pricing_profiles.len()];
            let m = 1 + (rng.gen_range(10) as usize); // 1..=10 blocks (fast brute force)

            let mut blocks = Vec::with_capacity(m);
            let mut reuse = Vec::with_capacity(m);

            // Generate realistic branching multi-agent monotone trees on even trials,
            // and arbitrary adversarial non-monotone reuse profiles on odd trials!
            let mut cur_h1 = rng.gen_f64() * 25.0;
            let mut cur_h5 = cur_h1 * rng.gen_f64();

            for i in 0..m {
                let tokens = 128 + (rng.gen_range(2400) as u32);
                let eligible = rng.gen_range(10) > 1; // 80% eligible
                blocks.push(make_block(i, tokens, eligible));

                let (h5, h1) = if trial % 2 == 0 {
                    cur_h1 *= 0.4 + 0.6 * rng.gen_f64();
                    cur_h5 = (cur_h5 * (0.3 + 0.7 * rng.gen_f64())).min(cur_h1);
                    (cur_h5, cur_h1)
                } else {
                    let a = rng.gen_f64() * 15.0;
                    let b = a + rng.gen_f64() * 15.0;
                    (a, b)
                };

                let warm_roll = rng.gen_range(6);
                let (already_warm, currently_warm_ttl) = match warm_roll {
                    0 => (true, Some(CacheTtl::OneHour)),
                    1 => (true, Some(CacheTtl::FiveMin)),
                    _ => (false, None),
                };

                reuse.push(BlockReuseEstimate {
                    expected_hits_5m: h5,
                    expected_hits_1h: h1,
                    ewma_gap_secs: rng.gen_f64() * 1800.0,
                    currently_warm_ttl,
                    already_warm,
                });
            }

            let plan = solve_optimal_breakpoints(&blocks, &reuse, pricing);
            let (brute_cost, _brute_seq) = brute_force_optimal(&blocks, &reuse, pricing);

            verify_plan_invariants(&blocks, pricing, &plan);
            assert!(
                (plan.expected_cost_usd - brute_cost).abs() < 1e-12,
                "Trial {trial} ({}) mismatch: dp={:.15} vs brute={:.15}",
                pricing.model_prefix,
                plan.expected_cost_usd,
                brute_cost
            );
        }
    }

    #[test]
    fn test_anthropic_mixed_ttl_monotonicity_invariant() {
        let pricing = lookup_pricing("claude-sonnet-5-5", ProviderFormat::AnthropicMessages);
        // Adversarial reuse profile trying to tempt a 5m breakpoint BEFORE a 1h breakpoint:
        // Block 0: lots of 5m hits, 0 extra 1h hits -> locally prefers 5m.
        // Block 1: 0 5m hits, 10 extra 1h hits -> locally wants 1h!
        // Under Anthropic's A/B/C rule, (5m at 0, 1h at 1) is ILLEGAL.
        let blocks = vec![
            make_block(0, 1000, true),
            make_block(1, 2000, true),
            make_block(2, 500, true),
        ];
        let reuse = vec![
            BlockReuseEstimate {
                expected_hits_5m: 20.0,
                expected_hits_1h: 20.0,
                ewma_gap_secs: 15.0,
                currently_warm_ttl: None,
                already_warm: false,
            },
            BlockReuseEstimate {
                expected_hits_5m: 0.0,
                expected_hits_1h: 12.0,
                ewma_gap_secs: 450.0,
                currently_warm_ttl: None,
                already_warm: false,
            },
            BlockReuseEstimate::cold(),
        ];

        let plan = solve_optimal_breakpoints(&blocks, &reuse, &pricing);
        let (brute_cost, _) = brute_force_optimal(&blocks, &reuse, &pricing);

        verify_plan_invariants(&blocks, &pricing, &plan);
        assert!((plan.expected_cost_usd - brute_cost).abs() < 1e-12);
        // Must have promoted block 0 to OneHour (or used OneHour at block 1) — never 5m before 1h!
        for w in plan.breakpoints.windows(2) {
            assert!(w[0].ttl <= w[1].ttl);
        }
    }

    #[test]
    fn test_multi_agent_branching_beats_naive_single_bp() {
        let pricing = lookup_pricing("claude-opus-5-5", ProviderFormat::AnthropicMessages);
        // Shared swarm system prompt -> team repo context -> single agent history -> volatile query
        let blocks = vec![
            make_block(0, 2000, true), // 20 agents share within 1h
            make_block(1, 4000, true), // 5 sub-agents share within 5m
            make_block(2, 1000, true), // 1 next turn within 5m
            make_block(3, 800, true),  // volatile query (0 reuse)
        ];
        let reuse = vec![
            BlockReuseEstimate {
                expected_hits_5m: 5.0,
                expected_hits_1h: 20.0,
                ewma_gap_secs: 180.0,
                currently_warm_ttl: None,
                already_warm: false,
            },
            BlockReuseEstimate {
                expected_hits_5m: 5.0,
                expected_hits_1h: 5.0,
                ewma_gap_secs: 40.0,
                currently_warm_ttl: None,
                already_warm: false,
            },
            BlockReuseEstimate {
                expected_hits_5m: 1.0,
                expected_hits_1h: 1.0,
                ewma_gap_secs: 30.0,
                currently_warm_ttl: None,
                already_warm: false,
            },
            BlockReuseEstimate::cold(),
        ];

        let plan = solve_optimal_breakpoints(&blocks, &reuse, &pricing);
        verify_plan_invariants(&blocks, &pricing, &plan);

        // Expect 3 breakpoints: block 0 (1h), block 1 (5m), block 2 (5m); block 3 uncached!
        assert_eq!(plan.breakpoints.len(), 3);
        assert_eq!(plan.breakpoints[0].block_index, 0);
        assert_eq!(plan.breakpoints[0].ttl, CacheTtl::OneHour);
        assert_eq!(plan.breakpoints[1].block_index, 1);
        assert_eq!(plan.breakpoints[1].ttl, CacheTtl::FiveMin);
        assert_eq!(plan.breakpoints[2].block_index, 2);
        assert_eq!(plan.breakpoints[2].ttl, CacheTtl::FiveMin);

        // Must drastically outperform naive single 5m breakpoint on last block!
        assert!(plan.expected_cost_usd < plan.naive_single_bp_cost_usd * 0.5);
        assert!(plan.expected_savings_vs_uncached_pct > 70.0);
    }

    #[test]
    fn test_no_wasteful_live_ttl_churn() {
        let pricing = lookup_pricing("claude-sonnet-5-5", ProviderFormat::AnthropicMessages);
        let blocks = vec![make_block(0, 4000, true)];

        // Already warm at 5m with 2 hits in 5m and 1 extra hit in 1h:
        // Upgrading live 5m -> 1h would burn a full 1h write ($4.00/Mtok) and lose
        // the current-turn read ($0.10/Mtok) just to save 1 miss ($1.90/Mtok).
        let reuse = vec![BlockReuseEstimate {
            expected_hits_5m: 2.0,
            expected_hits_1h: 3.0,
            ewma_gap_secs: 100.0,
            currently_warm_ttl: Some(CacheTtl::FiveMin),
            already_warm: true,
        }];

        let plan = solve_optimal_breakpoints(&blocks, &reuse, &pricing);
        assert_eq!(plan.breakpoints.len(), 1);
        assert_eq!(plan.breakpoints[0].ttl, CacheTtl::FiveMin);

        // Already warm at 1h: keeping 1h pays read cost only (zero write cost!).
        let reuse_1h = vec![BlockReuseEstimate {
            expected_hits_5m: 2.0,
            expected_hits_1h: 2.0,
            ewma_gap_secs: 60.0,
            currently_warm_ttl: Some(CacheTtl::OneHour),
            already_warm: true,
        }];
        let plan_1h = solve_optimal_breakpoints(&blocks, &reuse_1h, &pricing);
        assert_eq!(plan_1h.breakpoints.len(), 1);
        assert_eq!(plan_1h.breakpoints[0].ttl, CacheTtl::OneHour);
    }

    #[test]
    fn test_keepalive_breakeven_formula() {
        let blocks = vec![make_block(0, 4000, true)];

        // Claude Sonnet 5.5 (0.05x read, 0.75x 1h write premium) with 10-min gap:
        // 2 pings * $0.10 = $0.20/Mtok < $1.50/Mtok 1h premium -> keepalive recommended!
        let sonnet = lookup_pricing("claude-sonnet-5-5", ProviderFormat::AnthropicMessages);
        let slow_10m = vec![BlockReuseEstimate {
            expected_hits_5m: 0.1,
            expected_hits_1h: 6.0,
            ewma_gap_secs: 600.0,
            currently_warm_ttl: None,
            already_warm: false,
        }];
        let plan_sonnet = solve_optimal_breakpoints(&blocks, &slow_10m, &sonnet);
        assert!(plan_sonnet.keepalive_recommended);
        assert_eq!(plan_sonnet.keepalive_interval_secs, Some(270));

        // Fast 60s traffic naturally refreshes 5m TTL -> keepalive NOT recommended.
        let fast_60s = vec![BlockReuseEstimate {
            expected_hits_5m: 5.0,
            expected_hits_1h: 20.0,
            ewma_gap_secs: 60.0,
            currently_warm_ttl: None,
            already_warm: false,
        }];
        let plan_fast = solve_optimal_breakpoints(&blocks, &fast_60s, &sonnet);
        assert!(!plan_fast.keepalive_recommended);
        assert_eq!(plan_fast.keepalive_interval_secs, None);

        // Gemini 4 (1h premium = $0.25/Mtok, read = $0.125/Mtok) with 20-min gap:
        // 4 pings * $0.125 = $0.50/Mtok > $0.25/Mtok 1h premium -> 1h TTL beats keepalive!
        let gemini = lookup_pricing("gemini-4", ProviderFormat::Generic);
        let slow_20m = vec![BlockReuseEstimate {
            expected_hits_5m: 0.0,
            expected_hits_1h: 4.0,
            ewma_gap_secs: 1200.0,
            currently_warm_ttl: None,
            already_warm: false,
        }];
        let plan_gemini = solve_optimal_breakpoints(&blocks, &slow_20m, &gemini);
        assert!(!plan_gemini.keepalive_recommended);
        assert_eq!(plan_gemini.keepalive_interval_secs, None);
    }

    #[test]
    fn test_edge_cases_empty_below_min_and_nan() {
        let pricing = lookup_pricing("claude-opus-5-5", ProviderFormat::AnthropicMessages);

        // Empty input.
        let empty = solve_optimal_breakpoints(&[], &[], &pricing);
        assert!(empty.breakpoints.is_empty());
        assert_eq!(empty.expected_cost_usd, 0.0);

        // Below min_cache_tokens (512).
        let tiny = vec![make_block(0, 200, true), make_block(1, 200, true)];
        let reuse_tiny = vec![
            BlockReuseEstimate {
                expected_hits_5m: 100.0,
                expected_hits_1h: 100.0,
                ewma_gap_secs: 10.0,
                currently_warm_ttl: None,
                already_warm: false,
            };
            2
        ];
        let plan_tiny = solve_optimal_breakpoints(&tiny, &reuse_tiny, &pricing);
        assert!(plan_tiny.breakpoints.is_empty());
        assert!((plan_tiny.expected_cost_usd - plan_tiny.uncached_cost_usd).abs() < 1e-12);

        // Adversarial NaN / Inf / mismatched reuse slice length (must never panic!).
        let blocks = vec![make_block(0, 1000, true), make_block(1, 1000, true)];
        let bad_reuse = vec![BlockReuseEstimate {
            expected_hits_5m: f64::NAN,
            expected_hits_1h: f64::INFINITY,
            ewma_gap_secs: f64::NEG_INFINITY,
            currently_warm_ttl: None,
            already_warm: false,
        }];
        let plan_bad = solve_optimal_breakpoints(&blocks, &bad_reuse, &pricing);
        assert!(plan_bad.expected_cost_usd.is_finite());
        assert!(plan_bad.breakpoints.is_empty());
    }
}
