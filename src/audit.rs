//! High-throughput streaming trace auditor (`cachepin audit`).
//!
//! Reads JSONL request traces (Anthropic Messages, OpenAI Chat/Responses, or
//! Claude Code session transcripts) and computes:
//!   - Baseline (raw un-canonicalized) prefix cache hit rate & USD spend
//!   - Counterfactual `cachepin` canonicalized + exact-DP cache hit rate & USD spend
//!   - Root-cause miss breakdown (`json_key_drift`, `tool_order_drift`,
//!     `volatile_timestamp_or_id`, `all_gather_reorder`, `ttl_expired`, `cold_start`)
//!   - Exact divergence byte offsets & JSON paths for top dollar-bleeding misses.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use crate::canonical::{CompilerConfig, PrefixCompiler};
use crate::dp_planner::solve_optimal_breakpoints;
use crate::json::parse_json;
use crate::pricing::lookup_pricing;
use crate::radix::PrefixRadixTree;
use crate::simd::derive_tenant_key;
use crate::types::{CanonicalMode, MissReason, ProviderFormat};

#[derive(Debug, Clone)]
pub struct TopMissRecord {
    pub line_number: usize,
    pub model: String,
    pub reason: MissReason,
    pub divergence_path: Option<String>,
    pub divergence_byte_offset: Option<usize>,
    pub wasted_usd: f64,
    pub advice: String,
}

#[derive(Debug, Clone)]
pub struct AuditReport {
    pub requests_analyzed: usize,
    pub lines_skipped: usize,
    pub bytes_processed: u64,
    pub elapsed_ms: f64,
    pub throughput_mb_s: f64,
    pub total_input_tokens: u64,
    pub raw_cached_tokens: u64,
    pub raw_hit_rate_pct: f64,
    pub raw_est_cost_usd: f64,
    pub opt_cached_tokens: u64,
    pub opt_hit_rate_pct: f64,
    pub opt_est_cost_usd: f64,
    pub saved_usd: f64,
    pub cost_reduction_pct: f64,
    pub cost_multiplier_saved: f64,
    pub tools_reordered_count: usize,
    pub json_keys_reordered_count: usize,
    pub volatile_spans_hoisted: usize,
    pub all_gather_reordered_count: usize,
    pub miss_counts: HashMap<String, usize>,
    pub miss_wasted_usd: HashMap<String, f64>,
    pub top_misses: Vec<TopMissRecord>,
}

impl AuditReport {
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(2048);
        out.push('{');
        out.push_str(&format!(
            "\"requests_analyzed\":{},\"lines_skipped\":{},\"bytes_processed\":{},\"elapsed_ms\":{:.3},\"throughput_mb_s\":{:.2},",
            self.requests_analyzed, self.lines_skipped, self.bytes_processed, self.elapsed_ms, self.throughput_mb_s
        ));
        out.push_str(&format!(
            "\"total_input_tokens\":{},\"raw_cached_tokens\":{},\"raw_hit_rate_pct\":{:.2},\"raw_est_cost_usd\":{:.6},",
            self.total_input_tokens, self.raw_cached_tokens, self.raw_hit_rate_pct, self.raw_est_cost_usd
        ));
        out.push_str(&format!(
            "\"opt_cached_tokens\":{},\"opt_hit_rate_pct\":{:.2},\"opt_est_cost_usd\":{:.6},\"saved_usd\":{:.6},\"cost_reduction_pct\":{:.2},\"cost_multiplier_saved\":{:.2},",
            self.opt_cached_tokens, self.opt_hit_rate_pct, self.opt_est_cost_usd, self.saved_usd, self.cost_reduction_pct, self.cost_multiplier_saved
        ));
        out.push_str(&format!(
            "\"repairs\":{{\"tools_reordered\":{},\"json_keys_reordered\":{},\"volatile_spans_hoisted\":{},\"all_gather_reordered\":{}}},",
            self.tools_reordered_count, self.json_keys_reordered_count, self.volatile_spans_hoisted, self.all_gather_reordered_count
        ));
        out.push_str("\"miss_reasons\":{");
        let mut reasons: Vec<(&String, &usize)> = self.miss_counts.iter().collect();
        reasons.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (i, (k, cnt)) in reasons.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let wasted = self.miss_wasted_usd.get(*k).copied().unwrap_or(0.0);
            out.push_str(&format!(
                "\"{}\":{{\"count\":{},\"wasted_usd\":{:.6}}}",
                escape_json_str(k),
                cnt,
                wasted
            ));
        }
        out.push_str("},\"top_misses\":[");
        for (i, m) in self.top_misses.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"line\":{},\"model\":\"{}\",\"reason\":\"{}\",\"path\":{},\"byte_offset\":{},\"wasted_usd\":{:.6},\"advice\":\"{}\"}}",
                m.line_number,
                escape_json_str(&m.model),
                m.reason.as_str(),
                match &m.divergence_path {
                    Some(p) => format!("\"{}\"", escape_json_str(p)),
                    None => "null".to_string(),
                },
                match m.divergence_byte_offset {
                    Some(b) => b.to_string(),
                    None => "null".to_string(),
                },
                m.wasted_usd,
                escape_json_str(&m.advice)
            ));
        }
        out.push_str("]}");
        out
    }

    pub fn format_human(&self) -> String {
        let mut s = String::new();
        s.push_str("========================================================================\n");
        s.push_str("  cachepin audit — Multi-Agent Prompt Cache Forensics Report\n");
        s.push_str("========================================================================\n");
        s.push_str(&format!(
            "  Requests analyzed : {} (skipped: {}, {:.2} MB in {:.2} ms @ {:.1} MB/s)\n",
            self.requests_analyzed,
            self.lines_skipped,
            self.bytes_processed as f64 / (1024.0 * 1024.0),
            self.elapsed_ms,
            self.throughput_mb_s
        ));
        s.push_str(&format!(
            "  Total input tokens: {}\n",
            self.total_input_tokens
        ));
        s.push_str("------------------------------------------------------------------------\n");
        s.push_str(&format!(
            "  RAW baseline      : {:6.2}% cache hit  |  est. cost ${:.4}\n",
            self.raw_hit_rate_pct, self.raw_est_cost_usd
        ));
        s.push_str(&format!(
            "  AFTER cachepin    : {:6.2}% cache hit  |  est. cost ${:.4}\n",
            self.opt_hit_rate_pct, self.opt_est_cost_usd
        ));
        s.push_str(&format!(
            "  COUNTERFACTUAL    : -${:.4} saved ({:.1}% cheaper, {:.2}x cost reduction)\n",
            self.saved_usd, self.cost_reduction_pct, self.cost_multiplier_saved
        ));
        s.push_str("------------------------------------------------------------------------\n");
        s.push_str(&format!(
            "  Repairs applied   : {} tool sorts, {} JSON key sorts, {} volatile hoists, {} All-Gather alignments\n",
            self.tools_reordered_count,
            self.json_keys_reordered_count,
            self.volatile_spans_hoisted,
            self.all_gather_reordered_count
        ));
        if !self.miss_counts.is_empty() {
            s.push_str("\n  Miss Root Causes (Raw Trace):\n");
            let mut reasons: Vec<(&String, &usize)> = self.miss_counts.iter().collect();
            reasons.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            for (k, cnt) in reasons {
                let wasted = self.miss_wasted_usd.get(k).copied().unwrap_or(0.0);
                s.push_str(&format!("    - {:<26} {:>6} reqs   ${:.4} wasted\n", k, cnt, wasted));
            }
        }
        if !self.top_misses.is_empty() {
            s.push_str("\n  Top Cost-Bleeding Misses:\n");
            for m in self.top_misses.iter().take(5) {
                s.push_str(&format!(
                    "    [line {:>4}] {:<24} path={} byte={}  wasted=${:.4}\n             -> {}\n",
                    m.line_number,
                    m.reason.as_str(),
                    m.divergence_path.as_deref().unwrap_or("-"),
                    m.divergence_byte_offset
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "-".to_string()),
                    m.wasted_usd,
                    m.advice
                ));
            }
        }
        s.push_str("========================================================================\n");
        s
    }
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Audit a JSONL trace file OR recursively scan a directory of `.jsonl` files
/// (including native Claude Code `~/.claude/projects/` session transcripts).
pub fn audit_jsonl_file(
    path: &Path,
    mode: CanonicalMode,
    provider_hint: Option<ProviderFormat>,
) -> Result<AuditReport, String> {
    if path.is_dir() {
        let mut files = Vec::new();
        collect_jsonl_files(path, &mut files, 0)?;
        if files.is_empty() {
            return Err(format!("no .jsonl files found under {}", path.display()));
        }
        files.sort();
        return audit_jsonl_files(&files, mode, provider_hint);
    }
    let file = File::open(path).map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let reader = BufReader::with_capacity(256 * 1024, file);
    audit_jsonl_reader(reader, mode, provider_hint)
}

fn collect_jsonl_files(
    dir: &Path,
    out: &mut Vec<std::path::PathBuf>,
    depth: usize,
) -> Result<(), String> {
    if depth > 8 || out.len() >= 4096 {
        return Ok(());
    }
    let rd = std::fs::read_dir(dir)
        .map_err(|e| format!("failed to read directory {}: {}", dir.display(), e))?;
    for entry_res in rd {
        let Ok(entry) = entry_res else { continue };
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_symlink() {
            continue;
        }
        let p = entry.path();
        if ft.is_dir() {
            collect_jsonl_files(&p, out, depth + 1)?;
        } else if ft.is_file() {
            if p.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(p);
                if out.len() >= 4096 {
                    break;
                }
            }
        }
    }
    Ok(())
}

fn audit_jsonl_files(
    files: &[std::path::PathBuf],
    mode: CanonicalMode,
    provider_hint: Option<ProviderFormat>,
) -> Result<AuditReport, String> {
    let mut combined = Vec::new();
    for p in files {
        if let Ok(bytes) = std::fs::read(p) {
            combined.extend_from_slice(&bytes);
            if !combined.ends_with(b"\n") {
                combined.push(b'\n');
            }
        }
    }
    audit_jsonl_reader(BufReader::new(combined.as_slice()), mode, provider_hint)
}

/// Audit any streaming `BufRead` containing newline-delimited LLM request payloads,
/// trace wrapper objects `{"timestamp_ms":..., "tenant":..., "request": {...}}`,
/// OR native Claude Code session transcripts (`~/.claude/projects/*/*.jsonl`).
pub fn audit_jsonl_reader<R: BufRead>(
    reader: R,
    mode: CanonicalMode,
    provider_hint: Option<ProviderFormat>,
) -> Result<AuditReport, String> {
    let t0 = Instant::now();
    let mut raw_cfg = CompilerConfig::default();
    raw_cfg.mode = CanonicalMode::ObserveOnly;
    let raw_compiler = PrefixCompiler::new(raw_cfg);

    let mut opt_cfg = CompilerConfig::default();
    opt_cfg.mode = mode;
    let opt_compiler = PrefixCompiler::new(opt_cfg);

    let mut raw_radix = PrefixRadixTree::new(65_536);
    let mut opt_radix = PrefixRadixTree::new(65_536);

    let mut requests_analyzed = 0usize;
    let mut lines_skipped = 0usize;
    let mut bytes_processed = 0u64;
    let mut total_input_tokens = 0u64;
    let mut raw_cached_tokens = 0u64;
    let mut opt_cached_tokens = 0u64;
    let mut raw_est_cost_usd = 0.0f64;
    let mut opt_est_cost_usd = 0.0f64;

    let mut tools_reordered_count = 0usize;
    let mut json_keys_reordered_count = 0usize;
    let mut volatile_spans_hoisted = 0usize;
    let mut all_gather_reordered_count = 0usize;

    let mut miss_counts: HashMap<String, usize> = HashMap::new();
    let mut miss_wasted_usd: HashMap<String, f64> = HashMap::new();
    let mut top_misses: Vec<TopMissRecord> = Vec::new();

    // Per-session state for native Claude Code transcript billing forensics:
    // session_id -> (last_ts_ms, prev_cached_prefix_tokens, turn_count)
    let mut cc_sessions: HashMap<String, (u64, u64, usize)> = HashMap::new();

    let mut synthetic_clock_ms: u64 = 1_700_000_000_000;
    let tenant_secret = CompilerConfig::default().tenant_secret;

    for (idx, line_res) in reader.lines().enumerate() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => {
                lines_skipped += 1;
                continue;
            }
        };
        bytes_processed += line.len() as u64 + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') {
            lines_skipped += 1;
            continue;
        }

        // Auto-detect raw API request, wrapper envelope, or native Claude Code session transcript.
        let record = match decode_audit_record(trimmed.as_bytes(), synthetic_clock_ms) {
            Some(r) => r,
            None => {
                lines_skipped += 1;
                continue;
            }
        };

        let (tenant_id, ts_ms, payload_bytes) = match record {
            AuditRecord::Request {
                tenant,
                ts_ms,
                payload,
            } => (tenant, ts_ms, payload),
            AuditRecord::ClaudeCodeTurn {
                session_id,
                model,
                ts_ms,
                input_tokens,
                cache_creation_tokens,
                cache_read_tokens,
            } => {
                synthetic_clock_ms = ts_ms.max(synthetic_clock_ms) + 2_000;
                let turn_total = input_tokens
                    .saturating_add(cache_creation_tokens)
                    .saturating_add(cache_read_tokens);
                if turn_total == 0 {
                    lines_skipped += 1;
                    continue;
                }
                requests_analyzed += 1;
                let pricing = lookup_pricing(&model, ProviderFormat::AnthropicMessages);
                let raw_cost = pricing.cost_input(input_tokens)
                    + pricing.cost_write(cache_creation_tokens, crate::types::CacheTtl::FiveMin)
                    + pricing.cost_read(cache_read_tokens);

                let (prev_ts, prev_prefix, prev_turns) = cc_sessions
                    .get(&session_id)
                    .copied()
                    .unwrap_or((0, 0, 0));
                let gap_secs = if prev_turns > 0 && ts_ms >= prev_ts {
                    (ts_ms - prev_ts) / 1000
                } else {
                    0
                };

                // Classify billing event:
                // - ColdStart: first turn of session
                // - TtlExpired: inter-turn gap > 300s with cache rewrite spike (>= 512 tokens)
                // - PrefixMutation: inter-turn gap <= 300s yet cache_creation spiked and cache_read dropped
                // - CacheHit: steady incremental growth / cache reuse
                let (reason, recoverable_tokens, advice) = if prev_turns == 0 {
                    (
                        MissReason::ColdStart,
                        0u64,
                        "Initial session turn warmed the prompt cache.".to_string(),
                    )
                } else if gap_secs > 300 && cache_creation_tokens >= pricing.min_cache_tokens as u64 {
                    let rec = cache_creation_tokens.min(prev_prefix.max(cache_creation_tokens));
                    (
                        MissReason::TtlExpired,
                        rec,
                        format!(
                            "5m cache TTL expired after {}s gap ({} tokens rewritten); enable 1h DP tier or 270s keepalive.",
                            gap_secs, cache_creation_tokens
                        ),
                    )
                } else if cache_creation_tokens >= pricing.min_cache_tokens as u64
                    && prev_prefix >= pricing.min_cache_tokens as u64
                    && cache_read_tokens * 10 < prev_prefix * 7
                {
                    let rec = cache_creation_tokens.min(prev_prefix);
                    (
                        MissReason::PrefixMutation,
                        rec,
                        format!(
                            "Mid-session cache break ({}s after prior turn: cache_read dropped {} -> {}, {} rewritten); proxy through `cachepin proxy` to freeze tool/key/volatile drift.",
                            gap_secs, prev_prefix, cache_read_tokens, cache_creation_tokens
                        ),
                    )
                } else {
                    (
                        MissReason::CacheHit,
                        0u64,
                        "Warm prefix hit maintained.".to_string(),
                    )
                };

                let wasted_usd = if recoverable_tokens > 0 {
                    (pricing.cost_write(recoverable_tokens, crate::types::CacheTtl::FiveMin)
                        - pricing.cost_read(recoverable_tokens))
                    .max(0.0)
                } else {
                    0.0
                };
                let opt_hit = cache_read_tokens.saturating_add(recoverable_tokens).min(turn_total);
                let opt_cost = (raw_cost - wasted_usd).max(0.0);

                total_input_tokens += turn_total;
                raw_cached_tokens += cache_read_tokens;
                raw_est_cost_usd += raw_cost;
                opt_cached_tokens += opt_hit;
                opt_est_cost_usd += opt_cost;

                let reason_key = reason.as_str().to_string();
                *miss_counts.entry(reason_key.clone()).or_insert(0) += 1;
                if reason != MissReason::CacheHit {
                    *miss_wasted_usd.entry(reason_key).or_insert(0.0) += wasted_usd;
                    if wasted_usd > 0.0 {
                        top_misses.push(TopMissRecord {
                            line_number: idx + 1,
                            model: model.clone(),
                            reason,
                            divergence_path: Some(format!("session:{}", session_id)),
                            divergence_byte_offset: None,
                            wasted_usd,
                            advice,
                        });
                    }
                }

                let active_prefix = cache_read_tokens.saturating_add(cache_creation_tokens);
                cc_sessions.insert(session_id, (ts_ms, active_prefix, prev_turns + 1));
                continue;
            }
        };
        synthetic_clock_ms = ts_ms.max(synthetic_clock_ms) + 2_000;

        let raw_compiled = match raw_compiler.compile_request(&tenant_id, &payload_bytes, provider_hint) {
            Ok(c) => c,
            Err(_) => {
                lines_skipped += 1;
                continue;
            }
        };
        let opt_compiled = match opt_compiler.compile_request(&tenant_id, &payload_bytes, provider_hint) {
            Ok(c) => c,
            Err(_) => {
                lines_skipped += 1;
                continue;
            }
        };

        requests_analyzed += 1;
        if opt_compiled.tools_reordered {
            tools_reordered_count += 1;
        }
        if opt_compiled.json_keys_reordered {
            json_keys_reordered_count += 1;
        }
        volatile_spans_hoisted += opt_compiled.volatile_spans_hoisted;
        if opt_compiled.all_gather_reordered {
            all_gather_reordered_count += 1;
        }

        let pricing = lookup_pricing(&opt_compiled.model, opt_compiled.provider);
        let tk_bytes = derive_tenant_key(&tenant_secret, &tenant_id);
        let tenant_u64 = u64::from_le_bytes([
            tk_bytes[0], tk_bytes[1], tk_bytes[2], tk_bytes[3],
            tk_bytes[4], tk_bytes[5], tk_bytes[6], tk_bytes[7],
        ]);

        // 1. Evaluate RAW baseline (observe-only prefix, default 5m write on miss)
        let raw_diag = raw_radix.lookup_and_diagnose(
            tenant_u64,
            &raw_compiled.model,
            &raw_compiled.blocks,
            ts_ms,
            &pricing,
            opt_compiled.tools_reordered,
            opt_compiled.json_keys_reordered,
            opt_compiled.volatile_spans_hoisted,
            opt_compiled.all_gather_reordered,
        );
        let raw_total = raw_compiled.total_est_tokens.max(1) as u64;
        let raw_hit = (raw_diag.matched_tokens as u64).min(raw_total);
        let raw_miss = raw_total - raw_hit;
        let raw_cost = pricing.cost_read(raw_hit)
            + if raw_total >= pricing.min_cache_tokens as u64 {
                pricing.cost_write(raw_miss, crate::types::CacheTtl::FiveMin)
            } else {
                pricing.cost_input(raw_miss)
            };
        raw_cached_tokens += raw_hit;
        raw_est_cost_usd += raw_cost;
        total_input_tokens += raw_total;

        let raw_reuse = raw_radix.estimate_reuse_profile(tenant_u64, &raw_compiled.model, &raw_compiled.blocks, ts_ms);
        let raw_plan = solve_optimal_breakpoints(&raw_compiled.blocks, &raw_reuse, &pricing);
        raw_radix.record_request(tenant_u64, &raw_compiled.model, &raw_compiled.blocks, &raw_plan.breakpoints, ts_ms);

        // Record miss taxonomy
        let reason_key = raw_diag.reason.as_str().to_string();
        *miss_counts.entry(reason_key.clone()).or_insert(0) += 1;
        if raw_diag.reason != MissReason::CacheHit {
            *miss_wasted_usd.entry(reason_key).or_insert(0.0) += raw_diag.wasted_usd_est;
            if raw_diag.wasted_usd_est > 0.0 {
                top_misses.push(TopMissRecord {
                    line_number: idx + 1,
                    model: raw_compiled.model.clone(),
                    reason: raw_diag.reason,
                    divergence_path: raw_diag.divergence_path.clone(),
                    divergence_byte_offset: raw_diag.divergence_byte_offset,
                    wasted_usd: raw_diag.wasted_usd_est,
                    advice: raw_diag.advice.clone(),
                });
            }
        }

        // 2. Evaluate OPTIMIZED (`cachepin` canonicalized + exact DP breakpoints)
        let opt_diag = opt_radix.lookup_and_diagnose(
            tenant_u64,
            &opt_compiled.model,
            &opt_compiled.blocks,
            ts_ms,
            &pricing,
            false,
            false,
            0,
            false,
        );
        let opt_reuse = opt_radix.estimate_reuse_profile(tenant_u64, &opt_compiled.model, &opt_compiled.blocks, ts_ms);
        let opt_plan = solve_optimal_breakpoints(&opt_compiled.blocks, &opt_reuse, &pricing);
        let opt_total = opt_compiled.total_est_tokens.max(1) as u64;
        let opt_hit = (opt_diag.matched_tokens as u64).min(opt_total);

        // Compute exact step billing under Anthropic A/B/C mixed-TTL rule
        let mut cursor = opt_hit;
        let mut opt_cost = pricing.cost_read(opt_hit);
        for bp in &opt_plan.breakpoints {
            let bp_end = (bp.cumulative_tokens as u64).min(opt_total);
            if bp_end > cursor {
                opt_cost += pricing.cost_write(bp_end - cursor, bp.ttl);
                cursor = bp_end;
            }
        }
        if opt_total > cursor {
            opt_cost += pricing.cost_input(opt_total - cursor);
        }

        opt_cached_tokens += opt_hit;
        opt_est_cost_usd += opt_cost;
        opt_radix.record_request(tenant_u64, &opt_compiled.model, &opt_compiled.blocks, &opt_plan.breakpoints, ts_ms);
    }

    top_misses.sort_by(|a, b| b.wasted_usd.total_cmp(&a.wasted_usd));
    top_misses.truncate(15);

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let throughput_mb_s = if elapsed_ms > 0.0 {
        (bytes_processed as f64 / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0)
    } else {
        0.0
    };
    let denom = total_input_tokens.max(1) as f64;
    let raw_hit_rate_pct = (raw_cached_tokens as f64 / denom) * 100.0;
    let opt_hit_rate_pct = (opt_cached_tokens as f64 / denom) * 100.0;
    let saved_usd = (raw_est_cost_usd - opt_est_cost_usd).max(0.0);
    let cost_reduction_pct = if raw_est_cost_usd > 0.0 {
        (saved_usd / raw_est_cost_usd) * 100.0
    } else {
        0.0
    };
    let cost_multiplier_saved = if opt_est_cost_usd > 1e-12 {
        raw_est_cost_usd / opt_est_cost_usd
    } else {
        1.0
    };

    Ok(AuditReport {
        requests_analyzed,
        lines_skipped,
        bytes_processed,
        elapsed_ms,
        throughput_mb_s,
        total_input_tokens,
        raw_cached_tokens,
        raw_hit_rate_pct,
        raw_est_cost_usd,
        opt_cached_tokens,
        opt_hit_rate_pct,
        opt_est_cost_usd,
        saved_usd,
        cost_reduction_pct,
        cost_multiplier_saved,
        tools_reordered_count,
        json_keys_reordered_count,
        volatile_spans_hoisted,
        all_gather_reordered_count,
        miss_counts,
        miss_wasted_usd,
        top_misses,
    })
}

enum AuditRecord {
    Request {
        tenant: String,
        ts_ms: u64,
        payload: Vec<u8>,
    },
    ClaudeCodeTurn {
        session_id: String,
        model: String,
        ts_ms: u64,
        input_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
    },
}

fn decode_audit_record(line: &[u8], default_ts: u64) -> Option<AuditRecord> {
    let val = parse_json(line).ok()?;
    // 1. Wrapper format `{"timestamp_ms":..., "tenant":..., "request": {...}}`
    if let Some(req) = val.get("request") {
        let tenant = val
            .get("tenant")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        let ts_ms = val
            .get("timestamp_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(default_ts);
        let mut buf = Vec::new();
        crate::json::serialize_json(req, &mut buf);
        return Some(AuditRecord::Request {
            tenant,
            ts_ms,
            payload: buf,
        });
    }
    // 2. Native Claude Code local session transcript (`~/.claude/projects/*/*.jsonl`)
    //    Format: {"type":"assistant","timestamp":"2026-...Z","sessionId":"...","message":{"model":"...","usage":{...}}}
    if let Some(msg) = val.get("message") {
        if let Some(usage) = msg.get("usage") {
            let input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
            let cache_creation_tokens = usage
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let cache_read_tokens = usage
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            if input_tokens + cache_creation_tokens + cache_read_tokens > 0 {
                let model = msg
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or("claude-sonnet-5-5")
                    .to_string();
                let session_id = val
                    .get("sessionId")
                    .or_else(|| val.get("session_id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("default")
                    .to_string();
                let ts_ms = val
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .and_then(parse_iso8601_epoch_ms)
                    .or_else(|| val.get("timestamp_ms").and_then(|v| v.as_u64()))
                    .unwrap_or(default_ts);
                return Some(AuditRecord::ClaudeCodeTurn {
                    session_id,
                    model,
                    ts_ms,
                    input_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                });
            }
        }
    }
    // 3. Direct Anthropic/OpenAI request JSON
    if val.get("messages").is_some() || val.get("input").is_some() || val.get("system").is_some() {
        let tenant = val
            .get("user")
            .or_else(|| val.get("tenant"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();
        return Some(AuditRecord::Request {
            tenant,
            ts_ms: default_ts,
            payload: line.to_vec(),
        });
    }
    None
}

/// Pure-Rust ISO-8601 UTC timestamp parser (`YYYY-MM-DDTHH:MM:SS[.sss]Z`) -> epoch ms.
fn parse_iso8601_epoch_ms(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let y: i64 = s[0..4].parse().ok()?;
    let m: i64 = s[5..7].parse().ok()?;
    let d: i64 = s[8..10].parse().ok()?;
    let hh: i64 = s[11..13].parse().ok()?;
    let mm: i64 = s[14..16].parse().ok()?;
    let ss: i64 = s[17..19].parse().ok()?;
    if !(1970..=2200).contains(&y) || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    // Howard Hinnant's civil-from-days inverse (exact Gregorian days since 1970-01-01)
    let y_adj = y - if m <= 2 { 1 } else { 0 };
    let era = if y_adj >= 0 { y_adj } else { y_adj - 399 } / 400;
    let yoe = y_adj - era * 400;
    let mp = m + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + hh * 3_600 + mm * 60 + ss;
    u64::try_from(secs).ok().map(|v| v * 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claude_code_transcript_ttl_expiry_and_mid_session_break() {
        let trace = concat!(
            "{\"type\":\"user\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-11-22T10:00:00Z\",\"message\":{\"role\":\"user\",\"content\":\"start\"}}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-11-22T10:00:02Z\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":50,\"cache_creation_input_tokens\":40000,\"cache_read_input_tokens\":0}}}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-11-22T10:00:30Z\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":80,\"cache_creation_input_tokens\":200,\"cache_read_input_tokens\":40000}}}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-11-22T10:01:00Z\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":60,\"cache_creation_input_tokens\":40500,\"cache_read_input_tokens\":0}}}\n",
            "{\"type\":\"assistant\",\"sessionId\":\"sess-1\",\"timestamp\":\"2026-11-22T10:08:30Z\",\"message\":{\"model\":\"claude-opus-5-5\",\"usage\":{\"input_tokens\":70,\"cache_creation_input_tokens\":41000,\"cache_read_input_tokens\":0}}}\n"
        );
        let rep = audit_jsonl_reader(
            BufReader::new(trace.as_bytes()),
            CanonicalMode::SwarmAlign,
            None,
        )
        .expect("audit claude code transcript");

        assert_eq!(rep.requests_analyzed, 4);
        assert_eq!(rep.miss_counts.get("cold_start").copied(), Some(1));
        assert_eq!(rep.miss_counts.get("cache_hit").copied(), Some(1));
        assert_eq!(rep.miss_counts.get("prefix_mutation").copied(), Some(1));
        assert_eq!(rep.miss_counts.get("ttl_expired").copied(), Some(1));
        assert!(rep.opt_hit_rate_pct > rep.raw_hit_rate_pct + 40.0);
        assert!(rep.saved_usd > 0.30);
    }

    #[test]
    fn test_iso8601_parser_gap_accuracy() {
        let t1 = parse_iso8601_epoch_ms("2026-11-22T10:00:00Z").expect("t1");
        let t2 = parse_iso8601_epoch_ms("2026-11-22T10:06:00Z").expect("t2");
        assert_eq!((t2 - t1) / 1000, 360);
    }
}
