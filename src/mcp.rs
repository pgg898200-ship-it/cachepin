//! Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 stdio server for `cachepin`.
//!
//! Designed for coding and multi-agent harnesses (Claude Code, Codex, Cursor,
//! Gemini CLI, OpenCode). Strictly line-delimited JSON-RPC 2.0 over stdin/stdout;
//! diagnostic output is never written to stdout.
//!
//! Supported JSON-RPC methods:
//! - `initialize`
//! - `notifications/initialized` (and any `notifications/*`)
//! - `ping`
//! - `tools/list`
//! - `tools/call`
//!
//! Exposed tools:
//! 1. `optimize_prompt`
//! 2. `diagnose_diff`
//! 3. `plan_breakpoints`
//! 4. `audit_trace`
//! 5. `route_simulate`

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::audit::audit_jsonl_file;
use crate::canonical::{apply_breakpoints_to_json, CompilerConfig, PrefixCompiler};
use crate::dp_planner::{solve_optimal_breakpoints, BlockReuseEstimate, KEEPALIVE_INTERVAL_SECS};
use crate::json::{json_to_string, parse_json, serialize_json, JsonValue};
use crate::pricing::lookup_pricing;
use crate::radix::PrefixRadixTree;
use crate::router::{BackendConfig, RoutingPolicy, ShadowKvRouter};
use crate::simd::{derive_tenant_key, keyed_hash64, longest_common_prefix_len};
use crate::types::{CacheTtl, CanonicalMode, MissReason, PrefixBlock, ProviderFormat};
use crate::VERSION;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[inline]
fn j_str(s: impl Into<String>) -> JsonValue {
    JsonValue::String(s.into())
}

#[inline]
fn j_bool(b: bool) -> JsonValue {
    JsonValue::Bool(b)
}

#[inline]
fn j_u64(v: u64) -> JsonValue {
    JsonValue::Number(v.to_string())
}

#[inline]
fn j_i64(v: i64) -> JsonValue {
    JsonValue::Number(v.to_string())
}

#[inline]
fn j_f64(v: f64) -> JsonValue {
    if v.is_finite() {
        JsonValue::Number(format!("{:.6}", v))
    } else {
        JsonValue::Number("0".to_string())
    }
}

#[inline]
fn j_obj(fields: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

/// Stateful MCP JSON-RPC 2.0 server instance.
pub struct McpServer {
    default_mode: CanonicalMode,
    tenant_secret: [u8; 16],
    compiler: PrefixCompiler,
    radix: PrefixRadixTree,
}

impl McpServer {
    /// Creates a new MCP server backed by a prefix compiler and radix tree.
    pub fn new(config: CompilerConfig) -> Self {
        let default_mode = config.mode;
        let tenant_secret = config.tenant_secret;
        Self {
            default_mode,
            tenant_secret,
            compiler: PrefixCompiler::new(config),
            radix: PrefixRadixTree::new(32_768),
        }
    }

    fn make_compiler(&self, mode: CanonicalMode) -> PrefixCompiler {
        let mut cfg = CompilerConfig::default();
        cfg.mode = mode;
        cfg.tenant_secret = self.tenant_secret;
        PrefixCompiler::new(cfg)
    }

    /// Processes a single JSON-RPC 2.0 input line.
    ///
    /// Returns `None` for empty lines and JSON-RPC notifications (which MUST NOT
    /// elicit a response), or `Some(serialized_json_response)` for requests.
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return None;
        }

        let req = match parse_json(trimmed.as_bytes()) {
            Ok(v) => v,
            Err(e) => {
                let err = jsonrpc_error(JsonValue::Null, -32700, &format!("parse error: {}", e));
                return Some(json_to_string(&err));
            }
        };

        let id_opt = req.get("id").cloned();
        let method = match req.get("method").and_then(|v| v.as_str()) {
            Some(m) => m,
            None => {
                // Ignore response-like objects without method unless an id was sent.
                return id_opt.map(|id| {
                    json_to_string(&jsonrpc_error(
                        id,
                        -32600,
                        "invalid JSON-RPC request: missing string 'method'",
                    ))
                });
            }
        };

        // Notifications must never receive a response per JSON-RPC 2.0 & MCP spec.
        if method == "notifications/initialized"
            || method.starts_with("notifications/")
            || id_opt.is_none()
        {
            return None;
        }

        let id = id_opt.unwrap_or(JsonValue::Null);
        let params = req.get("params");

        let response = match method {
            "initialize" => self.handle_initialize(id),
            "ping" => jsonrpc_ok(id, j_obj(vec![])),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, params),
            other => jsonrpc_error(id, -32601, &format!("method not found: {}", other)),
        };

        Some(json_to_string(&response))
    }

    fn handle_initialize(&self, id: JsonValue) -> JsonValue {
        let result = j_obj(vec![
            ("protocolVersion", j_str(MCP_PROTOCOL_VERSION)),
            (
                "capabilities",
                j_obj(vec![(
                    "tools",
                    j_obj(vec![("listChanged", j_bool(false))]),
                )]),
            ),
            (
                "serverInfo",
                j_obj(vec![("name", j_str("cachepin")), ("version", j_str(VERSION))]),
            ),
        ]);
        jsonrpc_ok(id, result)
    }

    fn handle_tools_list(&self, id: JsonValue) -> JsonValue {
        let tools = vec![
            tool_def(
                "optimize_prompt",
                "Canonicalize an Anthropic/OpenAI LLM request payload (deterministic JSON keys, sorted tools, volatile timestamp/UUID hoisting, All-Gather alignment) and compute exact O(M^2 K) mixed-TTL cache breakpoints.",
                j_obj(vec![
                    ("type", j_str("object")),
                    (
                        "properties",
                        j_obj(vec![
                            (
                                "payload",
                                j_obj(vec![(
                                    "description",
                                    j_str("Anthropic/OpenAI request JSON (as string or object)."),
                                )]),
                            ),
                            (
                                "mode",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    (
                                        "enum",
                                        JsonValue::Array(vec![
                                            j_str("observe"),
                                            j_str("json"),
                                            j_str("hoist"),
                                            j_str("swarm"),
                                        ]),
                                    ),
                                    ("description", j_str("Canonicalization mode (default: swarm).")),
                                ]),
                            ),
                            (
                                "model",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional model override for pricing/DP.")),
                                ]),
                            ),
                            (
                                "provider",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional provider format hint.")),
                                ]),
                            ),
                            (
                                "tenant_id",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Tenant isolation identifier.")),
                                ]),
                            ),
                            (
                                "inject_breakpoints",
                                j_obj(vec![
                                    ("type", j_str("boolean")),
                                    ("description", j_str("Whether to inject cache_control breakpoints into canonical_json (default: true).")),
                                ]),
                            ),
                        ]),
                    ),
                    ("required", JsonValue::Array(vec![j_str("payload")])),
                ]),
            ),
            tool_def(
                "diagnose_diff",
                "Compare two LLM prompt payloads using AVX2 SIMD longest-common-prefix divergence detection; pinpoint exact byte offset, JSON path, root cause (json_key_drift, tool_order_drift, volatile_timestamp_or_id, all_gather_reorder, prefix_mutation), and wasted USD.",
                j_obj(vec![
                    ("type", j_str("object")),
                    (
                        "properties",
                        j_obj(vec![
                            (
                                "prev_json",
                                j_obj(vec![(
                                    "description",
                                    j_str("Previous request JSON payload (string or object)."),
                                )]),
                            ),
                            (
                                "curr_json",
                                j_obj(vec![(
                                    "description",
                                    j_str("Current request JSON payload (string or object)."),
                                )]),
                            ),
                            (
                                "provider",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional provider format hint.")),
                                ]),
                            ),
                            (
                                "model",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional model override for USD waste estimate.")),
                                ]),
                            ),
                        ]),
                    ),
                    (
                        "required",
                        JsonValue::Array(vec![j_str("prev_json"), j_str("curr_json")]),
                    ),
                ]),
            ),
            tool_def(
                "plan_breakpoints",
                "Run the exact O(M^2 K) constrained mixed-TTL Dynamic Programming breakpoint optimizer for given block token sizes, reuse counts, model, and inter-arrival gap; returns optimal 1h/5m breakpoints, keepalive break-even advice, and cost vs naive/uncached.",
                j_obj(vec![
                    ("type", j_str("object")),
                    (
                        "properties",
                        j_obj(vec![
                            (
                                "block_tokens",
                                j_obj(vec![
                                    ("type", j_str("array")),
                                    ("items", j_obj(vec![("type", j_str("integer"))])),
                                    ("description", j_str("Token sizes for each sequential prefix block.")),
                                ]),
                            ),
                            (
                                "reuse_counts",
                                j_obj(vec![
                                    ("type", j_str("array")),
                                    ("items", j_obj(vec![("type", j_str("number"))])),
                                    ("description", j_str("Expected future reuse counts for each block.")),
                                ]),
                            ),
                            (
                                "blocks",
                                j_obj(vec![
                                    ("type", j_str("array")),
                                    ("description", j_str("Optional detailed block specs [{tokens, expected_hits_5m, expected_hits_1h, reuse_count, path, eligible, already_warm}].")),
                                ]),
                            ),
                            (
                                "model",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Target model name (default: claude-opus-5-5).")),
                                ]),
                            ),
                            (
                                "provider",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional provider hint.")),
                                ]),
                            ),
                            (
                                "inter_arrival_gap_secs",
                                j_obj(vec![
                                    ("type", j_str("number")),
                                    ("description", j_str("Expected inter-arrival gap in seconds (default: 60).")),
                                ]),
                            ),
                        ]),
                    ),
                ]),
            ),
            tool_def(
                "audit_trace",
                "Scan a local JSONL request trace (Anthropic, OpenAI, or Claude Code transcript) and return full cache-hit forensics, root-cause histogram, top cost-bleeding misses, and counterfactual dollar savings.",
                j_obj(vec![
                    ("type", j_str("object")),
                    (
                        "properties",
                        j_obj(vec![
                            (
                                "path",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Path to local JSONL trace file.")),
                                ]),
                            ),
                            (
                                "mode",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Canonicalization mode (default: swarm).")),
                                ]),
                            ),
                            (
                                "provider",
                                j_obj(vec![
                                    ("type", j_str("string")),
                                    ("description", j_str("Optional provider hint.")),
                                ]),
                            ),
                        ]),
                    ),
                    ("required", JsonValue::Array(vec![j_str("path")])),
                ]),
            ),
            tool_def(
                "route_simulate",
                "Evaluate multi-backend Shadow-KV prefix routing under CachePinAdaptiveReplication versus RoundRobin, LeastLoaded, ConsistentHash, and SingleHomePrefixDivert baselines.",
                j_obj(vec![
                    ("type", j_str("object")),
                    (
                        "properties",
                        j_obj(vec![
                            (
                                "num_backends",
                                j_obj(vec![
                                    ("type", j_str("integer")),
                                    ("description", j_str("Number of KV-cache replicas (default: 4, max: 64).")),
                                ]),
                            ),
                            (
                                "kv_capacity_tokens",
                                j_obj(vec![
                                    ("type", j_str("integer")),
                                    ("description", j_str("KV token capacity per replica (default: 32768).")),
                                ]),
                            ),
                            (
                                "prefix_tokens",
                                j_obj(vec![
                                    ("type", j_str("integer")),
                                    ("description", j_str("Shared prefix length in tokens (default: 4096).")),
                                ]),
                            ),
                            (
                                "num_requests",
                                j_obj(vec![
                                    ("type", j_str("integer")),
                                    ("description", j_str("Simulated burst request count (default: 32, max: 512).")),
                                ]),
                            ),
                            (
                                "num_hot_prefixes",
                                j_obj(vec![
                                    ("type", j_str("integer")),
                                    ("description", j_str("Distinct prefixes in the workload (default: 4, max: 64).")),
                                ]),
                            ),
                            (
                                "prefill_tps",
                                j_obj(vec![
                                    ("type", j_str("number")),
                                    ("description", j_str("Per-replica prefill throughput tok/s (default: 18000).")),
                                ]),
                            ),
                            (
                                "cache_read_tps",
                                j_obj(vec![
                                    ("type", j_str("number")),
                                    ("description", j_str("Per-replica KV-cache read throughput tok/s (default: 360000).")),
                                ]),
                            ),
                        ]),
                    ),
                ]),
            ),
        ];

        jsonrpc_ok(id, j_obj(vec![("tools", JsonValue::Array(tools))]))
    }

    fn handle_tools_call(&mut self, id: JsonValue, params: Option<&JsonValue>) -> JsonValue {
        let params_val = match params {
            Some(p) => p,
            None => return jsonrpc_error(id, -32602, "missing 'params' object"),
        };

        let tool_name = match params_val.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => return jsonrpc_error(id, -32602, "missing string 'params.name'"),
        };

        let empty_args = JsonValue::Object(Vec::new());
        let args = params_val.get("arguments").unwrap_or(&empty_args);

        let tool_res: Result<String, String> = match tool_name {
            "optimize_prompt" => self.tool_optimize_prompt(args),
            "diagnose_diff" => self.tool_diagnose_diff(args),
            "plan_breakpoints" => self.tool_plan_breakpoints(args),
            "audit_trace" => self.tool_audit_trace(args),
            "route_simulate" => self.tool_route_simulate(args),
            unknown => {
                return jsonrpc_error(id, -32602, &format!("unknown tool: {}", unknown));
            }
        };

        match tool_res {
            Ok(text_json) => jsonrpc_ok(
                id,
                j_obj(vec![
                    (
                        "content",
                        JsonValue::Array(vec![j_obj(vec![
                            ("type", j_str("text")),
                            ("text", j_str(text_json)),
                        ])]),
                    ),
                    ("isError", j_bool(false)),
                ]),
            ),
            Err(err_msg) => jsonrpc_ok(
                id,
                j_obj(vec![
                    (
                        "content",
                        JsonValue::Array(vec![j_obj(vec![
                            ("type", j_str("text")),
                            ("text", j_str(err_msg)),
                        ])]),
                    ),
                    ("isError", j_bool(true)),
                ]),
            ),
        }
    }

    /// Tool 1: `optimize_prompt`
    fn tool_optimize_prompt(&mut self, args: &JsonValue) -> Result<String, String> {
        let payload_bytes = extract_json_payload(args, "payload")?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .map(CanonicalMode::from_str_loose)
            .unwrap_or(self.default_mode);
        let provider_hint = args
            .get("provider")
            .and_then(|v| v.as_str())
            .map(ProviderFormat::from_str_loose);
        let tenant_id = args
            .get("tenant_id")
            .and_then(|v| v.as_str())
            .unwrap_or("mcp-default");
        let inject_bps = args
            .get("inject_breakpoints")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let model_override = args.get("model").and_then(|v| v.as_str());

        let compiled = if mode == self.default_mode {
            self.compiler
                .compile_request(tenant_id, &payload_bytes, provider_hint)?
        } else {
            let temp_compiler = self.make_compiler(mode);
            temp_compiler.compile_request(tenant_id, &payload_bytes, provider_hint)?
        };

        let effective_model = model_override
            .filter(|s| !s.is_empty())
            .unwrap_or(&compiled.model);
        let pricing = lookup_pricing(effective_model, compiled.provider);
        let tk = derive_tenant_key(&self.tenant_secret, tenant_id);
        let tenant_u64 = u64::from_le_bytes([
            tk[0], tk[1], tk[2], tk[3], tk[4], tk[5], tk[6], tk[7],
        ]);
        let ts_now = now_ms();

        let diag = self.radix.lookup_and_diagnose(
            tenant_u64,
            effective_model,
            &compiled.blocks,
            ts_now,
            &pricing,
            compiled.tools_reordered,
            compiled.json_keys_reordered,
            compiled.volatile_spans_hoisted,
            compiled.all_gather_reordered,
        );
        let reuse = self
            .radix
            .estimate_reuse_profile(tenant_u64, effective_model, &compiled.blocks, ts_now);
        let plan = solve_optimal_breakpoints(&compiled.blocks, &reuse, &pricing);
        self.radix.record_request(
            tenant_u64,
            effective_model,
            &compiled.blocks,
            &plan.breakpoints,
            ts_now,
        );

        let optimized_bytes = if mode == CanonicalMode::ObserveOnly {
            payload_bytes
        } else if inject_bps && !plan.breakpoints.is_empty() {
            let cache_key = format!("cp-{:016x}", compiled.shared_prefix_hash);
            apply_breakpoints_to_json(
                &compiled.canonical_json,
                compiled.provider,
                &plan,
                Some(&cache_key),
            )
            .unwrap_or_else(|_| compiled.canonical_json.clone())
        } else {
            compiled.canonical_json.clone()
        };

        let bp_json: Vec<JsonValue> = plan
            .breakpoints
            .iter()
            .map(|bp| {
                j_obj(vec![
                    ("block_index", j_u64(bp.block_index as u64)),
                    ("path", j_str(&bp.path)),
                    ("cumulative_tokens", j_u64(bp.cumulative_tokens as u64)),
                    ("ttl", j_str(bp.ttl.as_anthropic_str())),
                ])
            })
            .collect();

        let total_tok = compiled.total_est_tokens.max(1) as f64;
        let cached_prefix_tok = plan
            .breakpoints
            .last()
            .map(|bp| bp.cumulative_tokens)
            .unwrap_or(0);
        let projected_hit_rate_pct = ((cached_prefix_tok as f64) / total_tok * 100.0).min(100.0);

        let out = j_obj(vec![
            (
                "canonical_json",
                j_str(String::from_utf8_lossy(&optimized_bytes).into_owned()),
            ),
            ("provider", j_str(compiled.provider.as_str())),
            ("model", j_str(effective_model)),
            ("compile_ns", j_u64(compiled.compile_ns)),
            ("compile_us", j_f64(compiled.compile_ns as f64 / 1000.0)),
            (
                "hoisted_bindings_count",
                j_u64(compiled.bindings.len() as u64),
            ),
            (
                "repairs",
                j_obj(vec![
                    ("tools_reordered", j_bool(compiled.tools_reordered)),
                    ("json_keys_reordered", j_bool(compiled.json_keys_reordered)),
                    (
                        "all_gather_reordered",
                        j_bool(compiled.all_gather_reordered),
                    ),
                    (
                        "volatile_spans_hoisted",
                        j_u64(compiled.volatile_spans_hoisted as u64),
                    ),
                ]),
            ),
            ("breakpoints", JsonValue::Array(bp_json)),
            (
                "cache_hit_projection",
                j_obj(vec![
                    ("diagnostic_reason", j_str(diag.reason.as_str())),
                    ("observed_matched_tokens", j_u64(diag.matched_tokens as u64)),
                    ("cacheable_prefix_tokens", j_u64(cached_prefix_tok as u64)),
                    ("total_est_tokens", j_u64(compiled.total_est_tokens as u64)),
                    ("projected_hit_rate_pct", j_f64(projected_hit_rate_pct)),
                    ("expected_cost_usd", j_f64(plan.expected_cost_usd)),
                    ("uncached_cost_usd", j_f64(plan.uncached_cost_usd)),
                    (
                        "naive_single_bp_cost_usd",
                        j_f64(plan.naive_single_bp_cost_usd),
                    ),
                    (
                        "expected_savings_vs_uncached_pct",
                        j_f64(plan.expected_savings_vs_uncached_pct),
                    ),
                    (
                        "keepalive_recommended",
                        j_bool(plan.keepalive_recommended),
                    ),
                    (
                        "keepalive_interval_secs",
                        match plan.keepalive_interval_secs {
                            Some(sec) => j_u64(sec as u64),
                            None => JsonValue::Null,
                        },
                    ),
                ]),
            ),
        ]);

        Ok(json_to_string(&out))
    }

    /// Tool 2: `diagnose_diff`
    fn tool_diagnose_diff(&self, args: &JsonValue) -> Result<String, String> {
        let prev_bytes = extract_json_payload(args, "prev_json")?;
        let curr_bytes = extract_json_payload(args, "curr_json")?;
        let provider_hint = args
            .get("provider")
            .and_then(|v| v.as_str())
            .map(ProviderFormat::from_str_loose);
        let model_override = args.get("model").and_then(|v| v.as_str());

        let raw_compiler = self.make_compiler(CanonicalMode::ObserveOnly);
        let opt_compiler = self.make_compiler(CanonicalMode::SwarmAlign);

        let prev_raw = raw_compiler.compile_request("diff-tenant", &prev_bytes, provider_hint)?;
        let curr_raw = raw_compiler.compile_request("diff-tenant", &curr_bytes, provider_hint)?;
        let prev_opt = opt_compiler.compile_request("diff-tenant", &prev_bytes, provider_hint)?;
        let curr_opt = opt_compiler.compile_request("diff-tenant", &curr_bytes, provider_hint)?;

        let effective_model = model_override
            .filter(|s| !s.is_empty())
            .unwrap_or(&curr_raw.model);
        let pricing = lookup_pricing(effective_model, curr_raw.provider);

        // AVX2 SIMD LCP on the exact raw JSON byte payloads.
        let raw_lcp_offset = longest_common_prefix_len(&prev_bytes, &curr_bytes);

        // Block-by-block comparison on raw (un-canonicalized) blocks.
        let min_blocks = prev_raw.blocks.len().min(curr_raw.blocks.len());
        let mut matched_blocks = 0usize;
        let mut matched_tokens: u32 = 0;
        let mut div_block: Option<usize> = None;
        let mut div_path: Option<String> = None;
        let mut block_lcp_offset: Option<usize> = None;

        for i in 0..min_blocks {
            if prev_raw.blocks[i].block_hash == curr_raw.blocks[i].block_hash {
                matched_blocks += 1;
                matched_tokens = matched_tokens.saturating_add(curr_raw.blocks[i].est_tokens);
            } else {
                div_block = Some(i);
                div_path = Some(curr_raw.blocks[i].path.clone());
                let blcp = longest_common_prefix_len(
                    &prev_raw.blocks[i].canonical_bytes,
                    &curr_raw.blocks[i].canonical_bytes,
                );
                block_lcp_offset = Some(blcp);
                break;
            }
        }
        if div_block.is_none() && prev_raw.blocks.len() != curr_raw.blocks.len() {
            div_block = Some(min_blocks);
            div_path = curr_raw
                .blocks
                .get(min_blocks)
                .or_else(|| prev_raw.blocks.get(min_blocks))
                .map(|b| b.path.clone());
            block_lcp_offset = Some(0);
        }

        // Counterfactual comparison AFTER cachepin canonicalization.
        let opt_min = prev_opt.blocks.len().min(curr_opt.blocks.len());
        let mut opt_matched_tokens: u32 = 0;
        let mut opt_matched_blocks = 0usize;
        for i in 0..opt_min {
            if prev_opt.blocks[i].block_hash == curr_opt.blocks[i].block_hash {
                opt_matched_blocks += 1;
                opt_matched_tokens =
                    opt_matched_tokens.saturating_add(curr_opt.blocks[i].est_tokens);
            } else {
                break;
            }
        }

        let canonical_fixed_divergence =
            opt_matched_blocks > matched_blocks || opt_matched_tokens > matched_tokens;

        // Classify root cause deterministically.
        let reason = if prev_raw.model != curr_raw.model
            || prev_raw.config_fingerprint != curr_raw.config_fingerprint
        {
            MissReason::ModelOrConfigFlip
        } else if curr_raw.total_est_tokens < pricing.min_cache_tokens {
            MissReason::BelowMinTokens
        } else if div_block.is_none() && raw_lcp_offset == prev_bytes.len().min(curr_bytes.len())
            && prev_bytes.len() == curr_bytes.len()
        {
            MissReason::CacheHit
        } else if div_block.is_none() {
            // Identical logical blocks, only raw whitespace/key order differed.
            if curr_opt.json_keys_reordered || prev_opt.json_keys_reordered {
                MissReason::JsonKeyDrift
            } else {
                MissReason::CacheHit
            }
        } else if canonical_fixed_divergence {
            let path_ref = div_path.as_deref().unwrap_or("");
            if (curr_opt.tools_reordered || prev_opt.tools_reordered)
                && path_ref.starts_with("tools")
            {
                MissReason::ToolOrderDrift
            } else if (curr_opt.volatile_spans_hoisted > 0 || prev_opt.volatile_spans_hoisted > 0)
                && (curr_opt
                    .bindings
                    .iter()
                    .any(|b| path_ref.is_empty() || b.source_path.starts_with(path_ref))
                    || prev_opt
                        .bindings
                        .iter()
                        .any(|b| path_ref.is_empty() || b.source_path.starts_with(path_ref)))
            {
                MissReason::VolatileTimestampOrId
            } else if curr_opt.all_gather_reordered || prev_opt.all_gather_reordered {
                MissReason::AllGatherReorder
            } else if curr_opt.json_keys_reordered || prev_opt.json_keys_reordered {
                MissReason::JsonKeyDrift
            } else if curr_opt.tools_reordered || prev_opt.tools_reordered {
                MissReason::ToolOrderDrift
            } else if curr_opt.volatile_spans_hoisted > 0 || prev_opt.volatile_spans_hoisted > 0 {
                MissReason::VolatileTimestampOrId
            } else {
                MissReason::PrefixMutation
            }
        } else {
            MissReason::PrefixMutation
        };

        let missed_tokens = curr_raw.total_est_tokens.saturating_sub(matched_tokens);
        let wasted_usd = if reason == MissReason::CacheHit {
            0.0
        } else {
            (pricing.cost_write(missed_tokens as u64, CacheTtl::FiveMin)
                - pricing.cost_read(missed_tokens as u64))
            .max(0.0)
        };

        let advice = match reason {
            MissReason::CacheHit => {
                "Exact prefix match — no cache invalidation detected.".to_string()
            }
            MissReason::JsonKeyDrift => {
                "Non-deterministic JSON object key ordering invalidated the prefix. Enable cachepin JSON canonicalization (--mode json or swarm).".to_string()
            }
            MissReason::ToolOrderDrift => {
                "Tool definition ordering drifted between calls. Enable deterministic tool sorting via cachepin.".to_string()
            }
            MissReason::VolatileTimestampOrId => {
                "Volatile timestamp, UUID, or trace ID appeared in the prefix. Enable cachepin bijective hoisting (--mode hoist or swarm) to move volatile spans to the suffix.".to_string()
            }
            MissReason::AllGatherReorder => {
                "Shared context documents were permuted across agent calls. Enable cachepin SwarmAlign (--mode swarm) to order shared blocks canonically.".to_string()
            }
            MissReason::ModelOrConfigFlip => {
                "Model or provider configuration changed between requests, partitioning the cache namespace.".to_string()
            }
            MissReason::BelowMinTokens => format!(
                "Prompt has {} estimated tokens, below the model's {} token minimum cacheable prefix.",
                curr_raw.total_est_tokens, pricing.min_cache_tokens
            ),
            MissReason::PrefixMutation | MissReason::ColdStart | MissReason::TtlExpired => {
                "Substantive content changed in the shared prefix. Move mutable per-turn content after the stable system/tools prefix.".to_string()
            }
        };

        let prev_snippet = safe_snippet(&prev_bytes, raw_lcp_offset, 48);
        let curr_snippet = safe_snippet(&curr_bytes, raw_lcp_offset, 48);

        let out = j_obj(vec![
            ("reason", j_str(reason.as_str())),
            ("model", j_str(effective_model)),
            ("provider", j_str(curr_raw.provider.as_str())),
            ("divergence_byte_offset", j_u64(raw_lcp_offset as u64)),
            (
                "block_divergence_byte_offset",
                match block_lcp_offset {
                    Some(v) => j_u64(v as u64),
                    None => JsonValue::Null,
                },
            ),
            (
                "divergence_block",
                match div_block {
                    Some(v) => j_u64(v as u64),
                    None => JsonValue::Null,
                },
            ),
            (
                "divergence_path",
                match div_path {
                    Some(p) => j_str(p),
                    None => JsonValue::Null,
                },
            ),
            ("matched_blocks", j_u64(matched_blocks as u64)),
            ("total_blocks", j_u64(curr_raw.blocks.len() as u64)),
            ("matched_tokens", j_u64(matched_tokens as u64)),
            ("total_tokens", j_u64(curr_raw.total_est_tokens as u64)),
            (
                "counterfactual_matched_tokens",
                j_u64(opt_matched_tokens as u64),
            ),
            (
                "recoverable_with_cachepin",
                j_bool(canonical_fixed_divergence),
            ),
            ("wasted_usd", j_f64(wasted_usd)),
            ("prev_snippet", j_str(prev_snippet)),
            ("curr_snippet", j_str(curr_snippet)),
            ("advice", j_str(advice)),
        ]);

        Ok(json_to_string(&out))
    }

    /// Tool 3: `plan_breakpoints`
    fn tool_plan_breakpoints(&self, args: &JsonValue) -> Result<String, String> {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("claude-opus-5-5");
        let provider = args
            .get("provider")
            .and_then(|v| v.as_str())
            .map(ProviderFormat::from_str_loose)
            .unwrap_or_else(|| infer_provider_from_model(model));
        let default_gap = args
            .get("inter_arrival_gap_secs")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(60.0);

        let pricing = lookup_pricing(model, provider);
        let (blocks, reuse) = parse_dp_inputs(args, default_gap, &self.tenant_secret)?;
        let plan = solve_optimal_breakpoints(&blocks, &reuse, &pricing);

        let savings_vs_naive_pct = if plan.naive_single_bp_cost_usd > 1e-12 {
            ((plan.naive_single_bp_cost_usd - plan.expected_cost_usd)
                / plan.naive_single_bp_cost_usd)
                * 100.0
        } else {
            0.0
        };

        let keepalive_advice = if plan.keepalive_recommended {
            format!(
                "Send lightweight 5m keepalive pings every {}s instead of paying the 1h write surcharge.",
                plan.keepalive_interval_secs
                    .unwrap_or(KEEPALIVE_INTERVAL_SECS)
            )
        } else if plan.breakpoints.iter().any(|b| b.ttl == CacheTtl::OneHour) {
            "1h TTL tier is cost-optimal for long inter-arrival gaps; no manual keepalive ping needed.".to_string()
        } else {
            "Standard 5m TTL tier is optimal without keepalive pings.".to_string()
        };

        let bp_json: Vec<JsonValue> = plan
            .breakpoints
            .iter()
            .map(|bp| {
                j_obj(vec![
                    ("block_index", j_u64(bp.block_index as u64)),
                    ("path", j_str(&bp.path)),
                    ("cumulative_tokens", j_u64(bp.cumulative_tokens as u64)),
                    ("ttl", j_str(bp.ttl.as_anthropic_str())),
                ])
            })
            .collect();

        let out = j_obj(vec![
            ("model", j_str(model)),
            ("provider", j_str(pricing.provider.as_str())),
            ("min_cache_tokens", j_u64(pricing.min_cache_tokens as u64)),
            ("max_breakpoints", j_u64(pricing.max_breakpoints as u64)),
            ("supports_1h_ttl", j_bool(pricing.supports_1h_ttl)),
            ("breakpoints", JsonValue::Array(bp_json)),
            ("expected_cost_usd", j_f64(plan.expected_cost_usd)),
            ("uncached_cost_usd", j_f64(plan.uncached_cost_usd)),
            (
                "naive_single_bp_cost_usd",
                j_f64(plan.naive_single_bp_cost_usd),
            ),
            (
                "expected_savings_vs_uncached_pct",
                j_f64(plan.expected_savings_vs_uncached_pct),
            ),
            ("savings_vs_naive_pct", j_f64(savings_vs_naive_pct)),
            (
                "keepalive_recommended",
                j_bool(plan.keepalive_recommended),
            ),
            (
                "keepalive_interval_secs",
                match plan.keepalive_interval_secs {
                    Some(sec) => j_u64(sec as u64),
                    None => JsonValue::Null,
                },
            ),
            ("keepalive_advice", j_str(keepalive_advice)),
        ]);

        Ok(json_to_string(&out))
    }

    /// Tool 4: `audit_trace`
    fn tool_audit_trace(&self, args: &JsonValue) -> Result<String, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "missing required string argument 'path'".to_string())?;
        let mode = args
            .get("mode")
            .and_then(|v| v.as_str())
            .map(CanonicalMode::from_str_loose)
            .unwrap_or(self.default_mode);
        let provider_hint = args
            .get("provider")
            .and_then(|v| v.as_str())
            .map(ProviderFormat::from_str_loose);

        let report = audit_jsonl_file(Path::new(path_str), mode, provider_hint)?;
        Ok(report.to_json_string())
    }

    /// Tool 5: `route_simulate`
    fn tool_route_simulate(&self, args: &JsonValue) -> Result<String, String> {
        let num_backends = args
            .get("num_backends")
            .and_then(|v| v.as_u64())
            .unwrap_or(4)
            .clamp(1, 64) as usize;
        let kv_cap = args
            .get("kv_capacity_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(32_768)
            .max(1024);
        let prefix_tokens = args
            .get("prefix_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(4_096)
            .clamp(256, 262_144) as u32;
        let num_requests = args
            .get("num_requests")
            .and_then(|v| v.as_u64())
            .unwrap_or(32)
            .clamp(1, 512) as usize;
        let num_hot_prefixes = args
            .get("num_hot_prefixes")
            .and_then(|v| v.as_u64())
            .unwrap_or(4)
            .clamp(1, 64) as usize;
        let prefill_tps = args
            .get("prefill_tps")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(18_000.0);
        let cache_read_tps = args
            .get("cache_read_tps")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(360_000.0);

        // Build deterministic prefix catalog broken into 512-token blocks.
        let blocks_per_pfx = ((prefix_tokens as usize) / 512).max(1);
        let tok_per_blk = (prefix_tokens / blocks_per_pfx as u32).max(1);
        let mut catalog: Vec<Vec<PrefixBlock>> = Vec::with_capacity(num_hot_prefixes);
        for p in 0..num_hot_prefixes {
            let mut blks = Vec::with_capacity(blocks_per_pfx);
            let mut acc = 0u64;
            for b in 0..blocks_per_pfx {
                let label = format!("sim-pfx-{:03}-blk-{:02}", p, b);
                let bh = keyed_hash64(&self.tenant_secret, label.as_bytes());
                acc = acc.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ bh;
                blks.push(PrefixBlock {
                    path: label,
                    kind: "system".to_string(),
                    canonical_bytes: Vec::new(),
                    block_hash: bh,
                    prefix_hash: acc,
                    est_tokens: tok_per_blk,
                    breakpoint_eligible: true,
                });
            }
            catalog.push(blks);
        }

        // Skewed deterministic arrival sequence (first prefix receives heavy burst).
        let arrivals: Vec<(u64, usize)> = (0..num_requests)
            .map(|i| {
                let pfx_idx = if i % 3 == 0 {
                    0
                } else {
                    (i * 7 + 1) % num_hot_prefixes
                };
                ((i as u64) * 25, pfx_idx)
            })
            .collect();

        let policies = [
            ("cachepin_adaptive_replication", RoutingPolicy::CachePinAdaptiveReplication),
            ("single_home_prefix_divert", RoutingPolicy::SingleHomePrefixDivert),
            ("consistent_hash", RoutingPolicy::ConsistentHash),
            ("least_loaded", RoutingPolicy::LeastLoaded),
            ("round_robin", RoutingPolicy::RoundRobin),
        ];

        let mut policy_results = Vec::with_capacity(policies.len());
        for (policy_name, policy) in policies {
            let backends: Vec<BackendConfig> = (0..num_backends)
                .map(|id| BackendConfig {
                    id,
                    url: format!("http://replica-{}:8000", id),
                    kv_capacity_tokens: kv_cap,
                    prefill_tps,
                    cache_read_tps,
                    max_inflight: 64,
                })
                .collect();
            let mut router = ShadowKvRouter::new(backends, policy);

            let mut total_toks: u64 = 0;
            let mut matched_toks: u64 = 0;
            let mut total_ttft_ms: f64 = 0.0;
            let mut replications: u64 = 0;
            let mut last_backend: usize = 0;
            let mut last_reason = String::new();
            let mut last_matched: u32 = 0;
            let mut next_free_ms = vec![0.0f64; num_backends];
            let mut inflight: Vec<(f64, usize, u32)> = Vec::new();

            for &(arr_ms, pfx_idx) in &arrivals {
                let now_f = arr_ms as f64;
                let mut still = Vec::with_capacity(inflight.len());
                for (fin, bid, miss_tok) in inflight.drain(..) {
                    if fin <= now_f {
                        router.complete(bid, miss_tok, 64, 0.0);
                    } else {
                        still.push((fin, bid, miss_tok));
                    }
                }
                inflight = still;

                let blocks = &catalog[pfx_idx];
                let req_toks: u32 = blocks.iter().map(|b| b.est_tokens).sum();
                let dec = router.route(1, blocks, req_toks, arr_ms);
                let hit = dec.est_matched_tokens.min(req_toks);
                let miss = req_toks.saturating_sub(hit);

                let service_ms =
                    (miss as f64 / prefill_tps + hit as f64 / cache_read_tps) * 1000.0;
                let bid = dec.backend_id % num_backends;
                let start_ms = now_f.max(next_free_ms[bid]);
                let finish_ms = start_ms + service_ms;
                next_free_ms[bid] = finish_ms;
                let ttft_ms = finish_ms - now_f;
                inflight.push((finish_ms, bid, miss));

                total_toks += req_toks as u64;
                matched_toks += hit as u64;
                total_ttft_ms += ttft_ms;
                if dec.triggered_replication {
                    replications += 1;
                }
                last_backend = dec.backend_id;
                last_reason = dec.reason.to_string();
                last_matched = hit;
            }

            let kv_hit_rate_pct = if total_toks > 0 {
                (matched_toks as f64 / total_toks as f64) * 100.0
            } else {
                0.0
            };
            let mean_ttft_ms = total_ttft_ms / (arrivals.len().max(1) as f64);

            policy_results.push(j_obj(vec![
                ("policy", j_str(policy_name)),
                ("kv_hit_rate_pct", j_f64(kv_hit_rate_pct)),
                ("est_mean_ttft_ms", j_f64(mean_ttft_ms)),
                ("replications_triggered", j_u64(replications)),
                ("last_selected_backend", j_u64(last_backend as u64)),
                ("last_est_matched_tokens", j_u64(last_matched as u64)),
                ("last_route_reason", j_str(last_reason)),
            ]));
        }

        let out = j_obj(vec![
            ("num_backends", j_u64(num_backends as u64)),
            ("kv_capacity_tokens", j_u64(kv_cap)),
            ("prefix_tokens", j_u64(prefix_tokens as u64)),
            ("num_requests", j_u64(num_requests as u64)),
            ("num_hot_prefixes", j_u64(num_hot_prefixes as u64)),
            ("policies", JsonValue::Array(policy_results)),
        ]);

        Ok(json_to_string(&out))
    }
}

fn tool_def(name: &str, description: &str, input_schema: JsonValue) -> JsonValue {
    j_obj(vec![
        ("name", j_str(name)),
        ("description", j_str(description)),
        ("inputSchema", input_schema),
    ])
}

fn jsonrpc_ok(id: JsonValue, result: JsonValue) -> JsonValue {
    j_obj(vec![
        ("jsonrpc", j_str("2.0")),
        ("id", id),
        ("result", result),
    ])
}

fn jsonrpc_error(id: JsonValue, code: i64, message: &str) -> JsonValue {
    j_obj(vec![
        ("jsonrpc", j_str("2.0")),
        ("id", id),
        (
            "error",
            j_obj(vec![("code", j_i64(code)), ("message", j_str(message))]),
        ),
    ])
}

/// Extracts a JSON request payload from an argument field that may be either a
/// JSON string or an inline JSON object.
fn extract_json_payload(args: &JsonValue, field: &str) -> Result<Vec<u8>, String> {
    let val = args
        .get(field)
        .ok_or_else(|| format!("missing required argument '{}'", field))?;
    match val {
        JsonValue::String(s) => {
            if s.trim().is_empty() {
                return Err(format!("argument '{}' cannot be empty", field));
            }
            Ok(s.as_bytes().to_vec())
        }
        JsonValue::Object(_) => {
            let mut buf = Vec::new();
            serialize_json(val, &mut buf);
            Ok(buf)
        }
        _ => Err(format!(
            "argument '{}' must be a JSON string or JSON object",
            field
        )),
    }
}

fn infer_provider_from_model(model: &str) -> ProviderFormat {
    let lower = model.to_ascii_lowercase();
    if lower.starts_with("claude") {
        ProviderFormat::AnthropicMessages
    } else if lower.starts_with("gpt") || lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4") {
        ProviderFormat::OpenAiChat
    } else {
        ProviderFormat::Generic
    }
}

/// Parses either detailed `blocks: [...]` or shorthand `block_tokens` + `reuse_counts`
/// into `(Vec<PrefixBlock>, Vec<BlockReuseEstimate>)`.
fn parse_dp_inputs(
    args: &JsonValue,
    default_gap_secs: f64,
    secret: &[u8; 16],
) -> Result<(Vec<PrefixBlock>, Vec<BlockReuseEstimate>), String> {
    let mut blocks = Vec::new();
    let mut reuse = Vec::new();
    let mut acc = 0u64;

    if let Some(arr) = args.get("blocks").and_then(|v| v.as_array()) {
        if arr.is_empty() {
            return Err("'blocks' array must not be empty".to_string());
        }
        for (i, item) in arr.iter().enumerate() {
            let tokens = item
                .get("tokens")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("blocks[{}] missing integer 'tokens'", i))?
                .min(u32::MAX as u64) as u32;
            let path = item
                .get("path")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("block[{}]", i));
            let eligible = item
                .get("eligible")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let gap = item
                .get("inter_arrival_gap_secs")
                .or_else(|| item.get("ewma_gap_secs"))
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or(default_gap_secs);

            let reuse_count = item
                .get("reuse_count")
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v >= 0.0);

            let hits_5m = item
                .get("expected_hits_5m")
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or_else(|| {
                    let rc = reuse_count.unwrap_or(0.0);
                    if gap > 300.0 {
                        0.0
                    } else {
                        rc
                    }
                });
            let hits_1h = item
                .get("expected_hits_1h")
                .and_then(|v| v.as_f64())
                .filter(|v| v.is_finite() && *v >= 0.0)
                .unwrap_or_else(|| reuse_count.unwrap_or(hits_5m));

            let already_warm = item
                .get("already_warm")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let currently_warm_ttl = match item.get("warm_ttl").and_then(|v| v.as_str()) {
                Some("1h") => Some(CacheTtl::OneHour),
                Some("5m") => Some(CacheTtl::FiveMin),
                _ if already_warm => Some(CacheTtl::FiveMin),
                _ => None,
            };

            let bh = keyed_hash64(secret, path.as_bytes());
            acc = acc.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ bh;
            blocks.push(PrefixBlock {
                path,
                kind: "block".to_string(),
                canonical_bytes: Vec::new(),
                block_hash: bh,
                prefix_hash: acc,
                est_tokens: tokens,
                breakpoint_eligible: eligible,
            });
            reuse.push(BlockReuseEstimate {
                expected_hits_5m: hits_5m,
                expected_hits_1h: hits_1h,
                ewma_gap_secs: gap,
                currently_warm_ttl,
                already_warm,
            });
        }
        return Ok((blocks, reuse));
    }

    let tok_arr = args
        .get("block_tokens")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "provide either 'blocks' or 'block_tokens' array".to_string())?;
    if tok_arr.is_empty() {
        return Err("'block_tokens' array must not be empty".to_string());
    }
    let reuse_arr = args.get("reuse_counts").and_then(|v| v.as_array());

    for (i, tv) in tok_arr.iter().enumerate() {
        let tokens = tv
            .as_u64()
            .ok_or_else(|| format!("block_tokens[{}] must be a non-negative integer", i))?
            .min(u32::MAX as u64) as u32;
        let rc = reuse_arr
            .and_then(|a| a.get(i))
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.0);
        let hits_5m = if default_gap_secs > 300.0 { 0.0 } else { rc };
        let hits_1h = rc;

        let path = format!("block[{}]", i);
        let bh = keyed_hash64(secret, path.as_bytes());
        acc = acc.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ bh;
        blocks.push(PrefixBlock {
            path,
            kind: "block".to_string(),
            canonical_bytes: Vec::new(),
            block_hash: bh,
            prefix_hash: acc,
            est_tokens: tokens,
            breakpoint_eligible: true,
        });
        reuse.push(BlockReuseEstimate {
            expected_hits_5m: hits_5m,
            expected_hits_1h: hits_1h,
            ewma_gap_secs: default_gap_secs,
            currently_warm_ttl: None,
            already_warm: false,
        });
    }

    Ok((blocks, reuse))
}

/// Extracts a panic-free UTF-8 lossy context window around `offset` inside `bytes`.
fn safe_snippet(bytes: &[u8], offset: usize, radius: usize) -> String {
    if bytes.is_empty() {
        return String::new();
    }
    let clamped = offset.min(bytes.len());
    let start = clamped.saturating_sub(radius / 2);
    let end = clamped.saturating_add(radius).min(bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

/// Runs the MCP JSON-RPC 2.0 server loop over arbitrary buffered reader/writer streams.
pub fn run_mcp_io<R: BufRead, W: Write>(
    config: CompilerConfig,
    reader: R,
    mut writer: W,
) -> Result<(), String> {
    let mut server = McpServer::new(config);
    for line_res in reader.lines() {
        let line = line_res.map_err(|e| format!("mcp read error: {}", e))?;
        if let Some(resp) = server.handle_line(&line) {
            writer
                .write_all(resp.as_bytes())
                .map_err(|e| format!("mcp write error: {}", e))?;
            writer
                .write_all(b"\n")
                .map_err(|e| format!("mcp write error: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("mcp flush error: {}", e))?;
        }
    }
    Ok(())
}

/// Runs the native stdio MCP JSON-RPC 2.0 server (`cachepin mcp`).
pub fn run_mcp_stdio(config: CompilerConfig) -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    run_mcp_io(config, stdin.lock(), stdout.lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn call_line(server: &mut McpServer, json: &str) -> JsonValue {
        let resp = server
            .handle_line(json)
            .expect("expected JSON-RPC response");
        parse_json(resp.as_bytes()).expect("response must be valid JSON")
    }

    #[test]
    fn test_mcp_initialize_ping_and_tools_list_roundtrip() {
        let mut server = McpServer::new(CompilerConfig::default());

        // 1. initialize
        let init = call_line(
            &mut server,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#,
        );
        assert_eq!(init.get("jsonrpc").and_then(|v| v.as_str()), Some("2.0"));
        assert_eq!(init.get("id").and_then(|v| v.as_u64()), Some(1));
        let res = init.get("result").expect("missing result");
        assert_eq!(
            res.get("protocolVersion").and_then(|v| v.as_str()),
            Some(MCP_PROTOCOL_VERSION)
        );
        assert_eq!(
            res.get("serverInfo")
                .and_then(|s| s.get("name"))
                .and_then(|v| v.as_str()),
            Some("cachepin")
        );

        // 2. notifications/initialized MUST emit NO response
        let notif = server.handle_line(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        );
        assert!(notif.is_none());

        // 3. ping
        let pong = call_line(&mut server, r#"{"jsonrpc":"2.0","id":"p1","method":"ping"}"#);
        assert_eq!(pong.get("id").and_then(|v| v.as_str()), Some("p1"));
        assert!(pong.get("result").is_some());

        // 4. tools/list must expose all 5 tools
        let list = call_line(&mut server, r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let tools = list
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .expect("expected tools array");
        assert_eq!(tools.len(), 5);
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
            .collect();
        assert_eq!(
            names,
            vec![
                "optimize_prompt",
                "diagnose_diff",
                "plan_breakpoints",
                "audit_trace",
                "route_simulate"
            ]
        );
    }

    #[test]
    fn test_mcp_tools_call_all_five_tools() {
        let mut server = McpServer::new(CompilerConfig::default());

        // Tool 1: optimize_prompt
        let sys_body = "Deterministic multi-agent architecture invariant rule set. ".repeat(60);
        let payload = format!(
            "{{\"model\":\"claude-opus-5-5\",\"max_tokens\":512,\"tools\":[{{\"name\":\"z_tool\",\"description\":\"last\",\"input_schema\":{{\"type\":\"object\",\"properties\":{{}}}}}},{{\"name\":\"a_tool\",\"description\":\"first\",\"input_schema\":{{\"type\":\"object\",\"properties\":{{}}}}}}],\"system\":[{{\"type\":\"text\",\"text\":\"Now: 2026-11-22T15:00:00Z {}\"}}],\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}",
            sys_body
        );
        let opt_req = j_obj(vec![
            ("jsonrpc", j_str("2.0")),
            ("id", j_u64(10)),
            ("method", j_str("tools/call")),
            (
                "params",
                j_obj(vec![
                    ("name", j_str("optimize_prompt")),
                    (
                        "arguments",
                        j_obj(vec![
                            ("payload", j_str(&payload)),
                            ("mode", j_str("swarm")),
                        ]),
                    ),
                ]),
            ),
        ]);
        let opt_resp = call_line(&mut server, &json_to_string(&opt_req));
        let opt_text = opt_resp
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .expect("missing optimize_prompt text");
        let opt_parsed = parse_json(opt_text.as_bytes()).expect("tool output must be JSON");
        assert!(opt_parsed.get("canonical_json").is_some());
        assert!(opt_parsed.get("breakpoints").is_some());
        assert!(opt_parsed.get("compile_us").is_some());

        // Tool 2: diagnose_diff (detects volatile timestamp drift)
        let payload_b = payload.replace("2026-11-22T15:00:00Z", "2026-11-22T15:02:15Z");
        let diff_req = j_obj(vec![
            ("jsonrpc", j_str("2.0")),
            ("id", j_u64(11)),
            ("method", j_str("tools/call")),
            (
                "params",
                j_obj(vec![
                    ("name", j_str("diagnose_diff")),
                    (
                        "arguments",
                        j_obj(vec![
                            ("prev_json", j_str(&payload)),
                            ("curr_json", j_str(&payload_b)),
                        ]),
                    ),
                ]),
            ),
        ]);
        let diff_resp = call_line(&mut server, &json_to_string(&diff_req));
        let diff_text = diff_resp
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .expect("missing diagnose_diff text");
        let diff_parsed = parse_json(diff_text.as_bytes()).expect("diff output must be JSON");
        assert!(diff_parsed.get("divergence_byte_offset").is_some());
        assert!(diff_parsed.get("reason").is_some());
        assert!(diff_parsed.get("wasted_usd").is_some());

        // Tool 3: plan_breakpoints
        let plan_req = r#"{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"plan_breakpoints","arguments":{"block_tokens":[4000,4000,2000,1000],"reuse_counts":[40,12,4,0],"model":"claude-opus-5-5","inter_arrival_gap_secs":90}}}"#;
        let plan_resp = call_line(&mut server, plan_req);
        let plan_text = plan_resp
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .expect("missing plan_breakpoints text");
        let plan_parsed = parse_json(plan_text.as_bytes()).expect("plan output must be JSON");
        let bps = plan_parsed
            .get("breakpoints")
            .and_then(|b| b.as_array())
            .expect("expected breakpoints array");
        assert!(!bps.is_empty());

        // Tool 4: audit_trace on temporary JSONL trace
        let tmp_path = std::env::temp_dir().join(format!(
            "cachepin_mcp_audit_test_{}.jsonl",
            std::process::id()
        ));
        let trace_content = format!("{}\n{}\n", payload, payload_b);
        std::fs::write(&tmp_path, trace_content).expect("write temp trace");
        let audit_req = j_obj(vec![
            ("jsonrpc", j_str("2.0")),
            ("id", j_u64(13)),
            ("method", j_str("tools/call")),
            (
                "params",
                j_obj(vec![
                    ("name", j_str("audit_trace")),
                    (
                        "arguments",
                        j_obj(vec![("path", j_str(tmp_path.to_string_lossy()))]),
                    ),
                ]),
            ),
        ]);
        let audit_resp = call_line(&mut server, &json_to_string(&audit_req));
        let _ = std::fs::remove_file(&tmp_path);
        let audit_text = audit_resp
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .expect("missing audit_trace text");
        let audit_parsed = parse_json(audit_text.as_bytes()).expect("audit output must be JSON");
        assert_eq!(
            audit_parsed
                .get("requests_analyzed")
                .and_then(|v| v.as_u64()),
            Some(2)
        );

        // Tool 5: route_simulate
        let route_req = r#"{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"route_simulate","arguments":{"num_backends":4,"kv_capacity_tokens":16384,"prefix_tokens":2048,"num_requests":16,"num_hot_prefixes":4}}}"#;
        let route_resp = call_line(&mut server, route_req);
        let route_text = route_resp
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|item| item.get("text"))
            .and_then(|t| t.as_str())
            .expect("missing route_simulate text");
        let route_parsed = parse_json(route_text.as_bytes()).expect("route output must be JSON");
        let policies = route_parsed
            .get("policies")
            .and_then(|p| p.as_array())
            .expect("expected policies array");
        assert_eq!(policies.len(), 5);
    }

    #[test]
    fn test_mcp_io_stream_and_error_codes() {
        let input = concat!(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n",
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n",
            "{malformed json\n",
            "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"unknown/method\"}\n"
        );
        let mut out = Vec::new();
        run_mcp_io(CompilerConfig::default(), Cursor::new(input.as_bytes()), &mut out)
            .expect("run_mcp_io failed");

        let out_str = String::from_utf8(out).expect("stdout must be UTF-8");
        let lines: Vec<&str> = out_str.lines().collect();
        // 3 responses (notification produces no line)
        assert_eq!(lines.len(), 3);

        let r0 = parse_json(lines[0].as_bytes()).expect("valid json 0");
        assert!(r0.get("result").is_some());

        let r1 = parse_json(lines[1].as_bytes()).expect("valid json 1");
        assert_eq!(
            r1.get("error")
                .and_then(|e| e.get("code"))
                .and_then(|c| c.as_f64()),
            Some(-32700.0)
        );

        let r2 = parse_json(lines[2].as_bytes()).expect("valid json 2");
        assert_eq!(
            r2.get("error")
                .and_then(|e| e.get("code"))
                .and_then(|c| c.as_f64()),
            Some(-32601.0)
        );
    }
}
