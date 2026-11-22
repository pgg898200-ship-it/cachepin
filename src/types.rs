//! Shared core types, invariants, and provider pricing contracts for `cachepin`.
//! Zero external dependencies; designed for deterministic, low-latency operation.

use std::fmt;

/// Provider wire-format family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderFormat {
    AnthropicMessages,
    OpenAiChat,
    OpenAiResponses,
    VllmOpenAi,
    Generic,
}

impl ProviderFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderFormat::AnthropicMessages => "anthropic",
            ProviderFormat::OpenAiChat => "openai-chat",
            ProviderFormat::OpenAiResponses => "openai-responses",
            ProviderFormat::VllmOpenAi => "vllm",
            ProviderFormat::Generic => "generic",
        }
    }

    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "anthropic" | "claude" | "messages" => ProviderFormat::AnthropicMessages,
            "openai" | "openai-chat" | "chat" => ProviderFormat::OpenAiChat,
            "openai-responses" | "responses" => ProviderFormat::OpenAiResponses,
            "vllm" | "sglang" | "ollama" | "self-hosted" => ProviderFormat::VllmOpenAi,
            _ => ProviderFormat::Generic,
        }
    }
}

/// Cache TTL tier supported by modern LLM prompt-caching APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CacheTtl {
    /// Ephemeral 1-hour cache (must precede 5m breakpoints under Anthropic's mixed-TTL rule).
    OneHour,
    /// Ephemeral 5-minute cache (default).
    FiveMin,
}

impl CacheTtl {
    pub fn seconds(&self) -> f64 {
        match self {
            CacheTtl::FiveMin => 300.0,
            CacheTtl::OneHour => 3600.0,
        }
    }

    pub fn as_anthropic_str(&self) -> &'static str {
        match self {
            CacheTtl::FiveMin => "5m",
            CacheTtl::OneHour => "1h",
        }
    }
}

/// Canonicalization aggressiveness mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalMode {
    /// No payload mutation; observe, diagnose, and route only.
    ObserveOnly,
    /// Deterministic JSON key sorting + tool ordering (strict byte/schema canonicalization).
    JsonCanonical,
    /// JsonCanonical + bijective volatile-token hoisting (timestamps, UUIDs, request/trace IDs).
    HoistVolatile,
    /// HoistVolatile + All-Gather shared-block frequency-ranked prefix alignment across swarms.
    SwarmAlign,
}

impl CanonicalMode {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "observe" | "observe-only" | "off" => CanonicalMode::ObserveOnly,
            "json" | "json-canonical" | "strict" => CanonicalMode::JsonCanonical,
            "hoist" | "hoist-volatile" => CanonicalMode::HoistVolatile,
            _ => CanonicalMode::SwarmAlign,
        }
    }
}

/// A single logical prefix segment (system block, tool block, message, or context doc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixBlock {
    /// Logical path in the request JSON, e.g. "tools[0]", "system[0]", "messages[2]".
    pub path: String,
    /// Block role/category: "tools", "system", "shared_doc", "message", "volatile_tail".
    pub kind: String,
    /// Canonical UTF-8 serialized bytes of this block.
    pub canonical_bytes: Vec<u8>,
    /// Tenant-isolated keyed 64-bit hash of `canonical_bytes`.
    pub block_hash: u64,
    /// Cumulative keyed 64-bit hash from block 0 through this block.
    pub prefix_hash: u64,
    /// Estimated token count (fast byte/4 + structural boundary estimator).
    pub est_tokens: u32,
    /// Whether this block is eligible to receive an explicit `cache_control` breakpoint.
    pub breakpoint_eligible: bool,
}

/// A single bijectively hoisted volatile span (for exact lossless recovery).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoistedBinding {
    pub anchor: String,
    pub original_value: String,
    pub kind: &'static str,
    pub source_path: String,
}

/// Root-cause taxonomy for prompt-cache misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MissReason {
    CacheHit,
    ColdStart,
    JsonKeyDrift,
    ToolOrderDrift,
    VolatileTimestampOrId,
    AllGatherReorder,
    PrefixMutation,
    TtlExpired,
    ModelOrConfigFlip,
    BelowMinTokens,
}

impl MissReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            MissReason::CacheHit => "cache_hit",
            MissReason::ColdStart => "cold_start",
            MissReason::JsonKeyDrift => "json_key_drift",
            MissReason::ToolOrderDrift => "tool_order_drift",
            MissReason::VolatileTimestampOrId => "volatile_timestamp_or_id",
            MissReason::AllGatherReorder => "all_gather_reorder",
            MissReason::PrefixMutation => "prefix_mutation",
            MissReason::TtlExpired => "ttl_expired",
            MissReason::ModelOrConfigFlip => "model_or_config_flip",
            MissReason::BelowMinTokens => "below_min_tokens",
        }
    }
}

impl fmt::Display for MissReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Diagnostic report pinpointing where and why a cache prefix diverged.
#[derive(Debug, Clone)]
pub struct CacheDiagnostic {
    pub reason: MissReason,
    pub matched_blocks: usize,
    pub total_blocks: usize,
    pub matched_tokens: u32,
    pub total_tokens: u32,
    pub divergence_block: Option<usize>,
    pub divergence_path: Option<String>,
    pub divergence_byte_offset: Option<usize>,
    pub wasted_usd_est: f64,
    pub advice: String,
}

/// A planned cache breakpoint chosen by the exact DP solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedBreakpoint {
    /// Index into the request's `PrefixBlock` slice (inclusive end of cached prefix).
    pub block_index: usize,
    pub path: String,
    pub cumulative_tokens: u32,
    pub ttl: CacheTtl,
}

/// Output of the exact DP breakpoint & TTL optimizer.
#[derive(Debug, Clone)]
pub struct BreakpointPlan {
    pub breakpoints: Vec<PlannedBreakpoint>,
    pub expected_cost_usd: f64,
    pub uncached_cost_usd: f64,
    pub naive_single_bp_cost_usd: f64,
    pub expected_savings_vs_uncached_pct: f64,
    pub keepalive_recommended: bool,
    pub keepalive_interval_secs: Option<u32>,
}

/// Result of compiling/canonicalizing an LLM API request payload.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub provider: ProviderFormat,
    pub model: String,
    pub canonical_json: Vec<u8>,
    pub blocks: Vec<PrefixBlock>,
    pub bindings: Vec<HoistedBinding>,
    pub tools_reordered: bool,
    pub json_keys_reordered: bool,
    pub all_gather_reordered: bool,
    pub volatile_spans_hoisted: usize,
    pub total_est_tokens: u32,
    pub shared_prefix_hash: u64,
    pub full_prefix_hash: u64,
    pub config_fingerprint: u64,
    pub config_warnings: Vec<String>,
    pub compile_ns: u64,
}

/// Pricing & cache-geometry profile for a model/provider.
#[derive(Debug, Clone)]
pub struct ModelPricing {
    pub model_prefix: String,
    pub provider: ProviderFormat,
    /// USD per 1M base input tokens.
    pub input_per_mtok: f64,
    /// USD per 1M 5-minute cache write tokens (e.g. 1.25x base).
    pub write_5m_per_mtok: f64,
    /// USD per 1M 1-hour cache write tokens (e.g. 2.0x base).
    pub write_1h_per_mtok: f64,
    /// USD per 1M cache read tokens (e.g. 0.05x or 0.025x or 0.10x base).
    pub read_per_mtok: f64,
    /// Minimum cacheable prefix length in tokens.
    pub min_cache_tokens: u32,
    /// Maximum explicit cache breakpoints per request.
    pub max_breakpoints: usize,
    /// Whether mixed 1h + 5m TTL tiers are supported.
    pub supports_1h_ttl: bool,
    /// KV page/block alignment in tokens (16 for vLLM, 128 for OpenAI).
    pub block_align_tokens: u32,
}

impl ModelPricing {
    #[inline]
    pub fn cost_input(&self, tokens: u64) -> f64 {
        (tokens as f64) * self.input_per_mtok / 1_000_000.0
    }
    #[inline]
    pub fn cost_write(&self, tokens: u64, ttl: CacheTtl) -> f64 {
        let rate = match ttl {
            CacheTtl::FiveMin => self.write_5m_per_mtok,
            CacheTtl::OneHour => self.write_1h_per_mtok,
        };
        (tokens as f64) * rate / 1_000_000.0
    }
    #[inline]
    pub fn cost_read(&self, tokens: u64) -> f64 {
        (tokens as f64) * self.read_per_mtok / 1_000_000.0
    }
}
