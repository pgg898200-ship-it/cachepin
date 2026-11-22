//! Provider pricing and prompt-cache geometry registry for `cachepin`.
//!
//! Contains late-2026 model pricing profiles (USD per 1M tokens) and cache
//! mechanics (minimum cacheable prefix length, maximum explicit breakpoints,
//! mixed 1h/5m TTL support, and KV page/block token alignment).

use crate::types::{ModelPricing, ProviderFormat};

/// Static pricing table entry (ordered from most-specific prefix to least-specific).
struct PricingSpec {
    canonical_name: &'static str,
    aliases: &'static [&'static str],
    provider: ProviderFormat,
    input_per_mtok: f64,
    write_5m_per_mtok: f64,
    write_1h_per_mtok: f64,
    read_per_mtok: f64,
    min_cache_tokens: u32,
    max_breakpoints: usize,
    supports_1h_ttl: bool,
    block_align_tokens: u32,
}

impl PricingSpec {
    fn to_model_pricing(&self, provider_override: Option<ProviderFormat>) -> ModelPricing {
        ModelPricing {
            model_prefix: self.canonical_name.to_string(),
            provider: provider_override.unwrap_or(self.provider),
            input_per_mtok: self.input_per_mtok,
            write_5m_per_mtok: self.write_5m_per_mtok,
            write_1h_per_mtok: self.write_1h_per_mtok,
            read_per_mtok: self.read_per_mtok,
            min_cache_tokens: self.min_cache_tokens,
            max_breakpoints: self.max_breakpoints,
            supports_1h_ttl: self.supports_1h_ttl,
            block_align_tokens: self.block_align_tokens,
        }
    }
}

/// Late-2026 provider pricing and cache geometry catalog.
///
/// Order matters: more specific version prefixes (e.g., `gpt-5.6`) MUST appear
/// before broader prefixes (e.g., `gpt-5`).
const PRICING_CATALOG: &[PricingSpec] = &[
    PricingSpec {
        canonical_name: "claude-opus-5-5",
        aliases: &["claude-opus-5-5", "claude-opus-5.5"],
        provider: ProviderFormat::AnthropicMessages,
        input_per_mtok: 4.00,
        write_5m_per_mtok: 5.00,
        write_1h_per_mtok: 8.00,
        read_per_mtok: 0.20,
        min_cache_tokens: 512,
        max_breakpoints: 4,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "claude-sonnet-5-5",
        aliases: &["claude-sonnet-5-5", "claude-sonnet-5.5"],
        provider: ProviderFormat::AnthropicMessages,
        input_per_mtok: 2.00,
        write_5m_per_mtok: 2.50,
        write_1h_per_mtok: 4.00,
        read_per_mtok: 0.10,
        min_cache_tokens: 512,
        max_breakpoints: 4,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "claude-fable-5-1",
        aliases: &[
            "claude-fable-5-1",
            "claude-fable-5.1",
            "claude-fable-5-2",
            "claude-fable-5.2",
        ],
        provider: ProviderFormat::AnthropicMessages,
        input_per_mtok: 10.00,
        write_5m_per_mtok: 12.50,
        write_1h_per_mtok: 20.00,
        read_per_mtok: 0.25,
        min_cache_tokens: 512,
        max_breakpoints: 4,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "claude-haiku-5-5",
        aliases: &["claude-haiku-5-5", "claude-haiku-5.5"],
        provider: ProviderFormat::AnthropicMessages,
        input_per_mtok: 0.80,
        write_5m_per_mtok: 1.00,
        write_1h_per_mtok: 1.60,
        read_per_mtok: 0.04,
        min_cache_tokens: 512,
        max_breakpoints: 4,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "claude-opus-4",
        aliases: &["claude-opus-4", "claude-sonnet-4"],
        provider: ProviderFormat::AnthropicMessages,
        input_per_mtok: 3.00,
        write_5m_per_mtok: 3.75,
        write_1h_per_mtok: 6.00,
        read_per_mtok: 0.30,
        min_cache_tokens: 1024,
        max_breakpoints: 4,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "gpt-5.6",
        aliases: &["gpt-5.6", "gpt-5-6", "gpt-6"],
        provider: ProviderFormat::OpenAiChat,
        input_per_mtok: 2.50,
        write_5m_per_mtok: 3.125,
        write_1h_per_mtok: 3.125,
        read_per_mtok: 0.125,
        min_cache_tokens: 1024,
        max_breakpoints: 4,
        supports_1h_ttl: false,
        block_align_tokens: 128,
    },
    PricingSpec {
        canonical_name: "gpt-5",
        aliases: &["gpt-5", "gpt-4o"],
        provider: ProviderFormat::OpenAiChat,
        input_per_mtok: 2.50,
        write_5m_per_mtok: 2.50,
        write_1h_per_mtok: 2.50,
        read_per_mtok: 0.25,
        min_cache_tokens: 1024,
        max_breakpoints: 1,
        supports_1h_ttl: false,
        block_align_tokens: 128,
    },
    PricingSpec {
        canonical_name: "gemini-4",
        aliases: &["gemini-4", "gemini-3"],
        provider: ProviderFormat::Generic,
        input_per_mtok: 1.25,
        write_5m_per_mtok: 1.25,
        write_1h_per_mtok: 1.50,
        read_per_mtok: 0.125,
        min_cache_tokens: 1024,
        max_breakpoints: 1,
        supports_1h_ttl: true,
        block_align_tokens: 1,
    },
    PricingSpec {
        canonical_name: "vllm",
        aliases: &["vllm", "sglang"],
        provider: ProviderFormat::VllmOpenAi,
        input_per_mtok: 1.00,
        write_5m_per_mtok: 1.00,
        write_1h_per_mtok: 1.00,
        read_per_mtok: 0.05,
        min_cache_tokens: 64,
        max_breakpoints: 4,
        supports_1h_ttl: false,
        block_align_tokens: 16,
    },
];

/// Normalized compute-unit profile for self-hosted vLLM / SGLang / local runtimes.
fn self_hosted_fallback(provider: ProviderFormat) -> ModelPricing {
    let resolved_provider = match provider {
        ProviderFormat::Generic => ProviderFormat::VllmOpenAi,
        other => other,
    };
    ModelPricing {
        model_prefix: "vllm".to_string(),
        provider: resolved_provider,
        input_per_mtok: 1.00,
        write_5m_per_mtok: 1.00,
        write_1h_per_mtok: 1.00,
        read_per_mtok: 0.05,
        min_cache_tokens: 64,
        max_breakpoints: 4,
        supports_1h_ttl: false,
        block_align_tokens: 16,
    }
}

/// Normalize a model identifier by trimming whitespace, lowercasing, and
/// stripping any leading provider namespace prefix (e.g. `anthropic/`, `openai/`).
fn normalize_model_name(model: &str) -> String {
    let trimmed = model.trim().to_ascii_lowercase();
    match trimmed.rsplit_once('/') {
        Some((_, suffix)) if !suffix.is_empty() => suffix.to_string(),
        _ => trimmed,
    }
}

/// Checks whether `normalized_model` matches `alias` as a clean prefix boundary
/// (exact match, or followed by `-`, `.`, `_`, `:`, or `@`).
fn matches_prefix_boundary(normalized_model: &str, alias: &str) -> bool {
    if let Some(rest) = normalized_model.strip_prefix(alias) {
        if rest.is_empty() {
            return true;
        }
        matches!(rest.as_bytes()[0], b'-' | b'.' | b'_' | b':' | b'@')
    } else {
        false
    }
}

/// Resolve provider override: preserve `OpenAiResponses` for OpenAI-family models,
/// preserve `VllmOpenAi` for self-hosted models, and otherwise use the model's
/// native provider unless an explicit non-generic wire format matches.
fn resolve_provider(spec_provider: ProviderFormat, requested: ProviderFormat) -> Option<ProviderFormat> {
    match (spec_provider, requested) {
        (ProviderFormat::OpenAiChat, ProviderFormat::OpenAiResponses) => {
            Some(ProviderFormat::OpenAiResponses)
        }
        (ProviderFormat::Generic, other) if other != ProviderFormat::Generic => Some(other),
        (_, ProviderFormat::VllmOpenAi) if spec_provider == ProviderFormat::VllmOpenAi => {
            Some(ProviderFormat::VllmOpenAi)
        }
        _ => None,
    }
}

/// Look up pricing and cache geometry for a model identifier and provider format.
///
/// Handles versioned suffixes (e.g. `claude-opus-5-5-20261001`, `gpt-5.6-turbo`),
/// dot/hyphen variants (`claude-sonnet-5.5` vs `claude-sonnet-5-5`), and
/// provider-namespaced names (`anthropic/claude-fable-5-2`).
pub fn lookup_pricing(model: &str, provider: ProviderFormat) -> ModelPricing {
    let normalized = normalize_model_name(model);

    // 1. Exact/boundary-prefix match against catalog entries (most-specific first).
    if !normalized.is_empty() {
        for spec in PRICING_CATALOG {
            for &alias in spec.aliases {
                if matches_prefix_boundary(&normalized, alias) {
                    let override_provider = resolve_provider(spec.provider, provider);
                    return spec.to_model_pricing(override_provider);
                }
            }
        }
        // Loose starts_with fallback for unusual naming variants (e.g. "gpt-6o", "vllm-llama3").
        for spec in PRICING_CATALOG {
            for &alias in spec.aliases {
                if normalized.starts_with(alias) {
                    let override_provider = resolve_provider(spec.provider, provider);
                    return spec.to_model_pricing(override_provider);
                }
            }
        }
    }

    // 2. Family-level fallback if model name carries a recognizable vendor prefix.
    if normalized.starts_with("claude") {
        // Default modern Claude profile (Sonnet 5.5).
        return PRICING_CATALOG[1].to_model_pricing(None);
    }
    if normalized.starts_with("gpt") || normalized.starts_with("o3") || normalized.starts_with("o4") {
        let override_provider = resolve_provider(ProviderFormat::OpenAiChat, provider);
        return PRICING_CATALOG[6].to_model_pricing(override_provider);
    }
    if normalized.starts_with("gemini") {
        let override_provider = resolve_provider(ProviderFormat::Generic, provider);
        return PRICING_CATALOG[7].to_model_pricing(override_provider);
    }

    // 3. Provider-based fallback when model name is custom/unknown.
    match provider {
        ProviderFormat::AnthropicMessages => PRICING_CATALOG[1].to_model_pricing(None),
        ProviderFormat::OpenAiChat | ProviderFormat::OpenAiResponses => {
            PRICING_CATALOG[6].to_model_pricing(Some(provider))
        }
        ProviderFormat::VllmOpenAi | ProviderFormat::Generic => self_hosted_fallback(provider),
    }
}

/// Return the full catalog of supported canonical model pricing profiles.
pub fn list_supported_models() -> Vec<ModelPricing> {
    PRICING_CATALOG
        .iter()
        .map(|spec| spec.to_model_pricing(None))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CacheTtl;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn test_claude_5_5_and_fable_exact_rates() {
        let opus = lookup_pricing("claude-opus-5-5", ProviderFormat::AnthropicMessages);
        assert!(approx_eq(opus.input_per_mtok, 4.00));
        assert!(approx_eq(opus.write_5m_per_mtok, 5.00));
        assert!(approx_eq(opus.write_1h_per_mtok, 8.00));
        assert!(approx_eq(opus.read_per_mtok, 0.20));
        assert_eq!(opus.min_cache_tokens, 512);
        assert_eq!(opus.max_breakpoints, 4);
        assert!(opus.supports_1h_ttl);

        let opus_dot = lookup_pricing("claude-opus-5.5-20261101", ProviderFormat::Generic);
        assert!(approx_eq(opus_dot.input_per_mtok, 4.00));
        assert_eq!(opus_dot.provider, ProviderFormat::AnthropicMessages);

        let sonnet = lookup_pricing("claude-sonnet-5.5", ProviderFormat::AnthropicMessages);
        assert!(approx_eq(sonnet.input_per_mtok, 2.00));
        assert!(approx_eq(sonnet.write_5m_per_mtok, 2.50));
        assert!(approx_eq(sonnet.write_1h_per_mtok, 4.00));
        assert!(approx_eq(sonnet.read_per_mtok, 0.10));
        assert_eq!(sonnet.min_cache_tokens, 512);

        for fable_name in ["claude-fable-5-1", "claude-fable-5.1", "claude-fable-5-2"] {
            let fable = lookup_pricing(fable_name, ProviderFormat::AnthropicMessages);
            assert!(approx_eq(fable.input_per_mtok, 10.00));
            assert!(approx_eq(fable.write_5m_per_mtok, 12.50));
            assert!(approx_eq(fable.write_1h_per_mtok, 20.00));
            assert!(approx_eq(fable.read_per_mtok, 0.25)); // 0.025x read!
            assert_eq!(fable.min_cache_tokens, 512);
            assert_eq!(fable.max_breakpoints, 4);
            assert!(fable.supports_1h_ttl);
        }

        let haiku = lookup_pricing("anthropic/claude-haiku-5-5", ProviderFormat::Generic);
        assert!(approx_eq(haiku.input_per_mtok, 0.80));
        assert!(approx_eq(haiku.write_5m_per_mtok, 1.00));
        assert!(approx_eq(haiku.write_1h_per_mtok, 1.60));
        assert!(approx_eq(haiku.read_per_mtok, 0.04));
        assert_eq!(haiku.min_cache_tokens, 512);
    }

    #[test]
    fn test_claude_4_legacy_and_openai_gemini_vllm() {
        for legacy in ["claude-opus-4", "claude-sonnet-4-20250514"] {
            let p = lookup_pricing(legacy, ProviderFormat::AnthropicMessages);
            assert!(approx_eq(p.input_per_mtok, 3.00));
            assert!(approx_eq(p.write_5m_per_mtok, 3.75));
            assert!(approx_eq(p.write_1h_per_mtok, 6.00));
            assert!(approx_eq(p.read_per_mtok, 0.30));
            assert_eq!(p.min_cache_tokens, 1024);
            assert_eq!(p.max_breakpoints, 4);
            assert!(p.supports_1h_ttl);
        }

        // gpt-5.6 / gpt-6 MUST match before gpt-5!
        for gpt_new in ["gpt-5.6", "gpt-5.6-preview", "gpt-6", "openai/gpt-6-mini"] {
            let p = lookup_pricing(gpt_new, ProviderFormat::OpenAiResponses);
            assert!(approx_eq(p.input_per_mtok, 2.50));
            assert!(approx_eq(p.write_5m_per_mtok, 3.125));
            assert!(approx_eq(p.write_1h_per_mtok, 3.125));
            assert!(approx_eq(p.read_per_mtok, 0.125));
            assert_eq!(p.min_cache_tokens, 1024);
            assert_eq!(p.max_breakpoints, 4);
            assert!(!p.supports_1h_ttl);
            assert_eq!(p.block_align_tokens, 128);
            assert_eq!(p.provider, ProviderFormat::OpenAiResponses);
        }

        for gpt_old in ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-2024-11-20"] {
            let p = lookup_pricing(gpt_old, ProviderFormat::OpenAiChat);
            assert!(approx_eq(p.input_per_mtok, 2.50));
            assert!(approx_eq(p.write_5m_per_mtok, 2.50));
            assert!(approx_eq(p.read_per_mtok, 0.25));
            assert_eq!(p.min_cache_tokens, 1024);
            assert_eq!(p.max_breakpoints, 1);
            assert!(!p.supports_1h_ttl);
            assert_eq!(p.block_align_tokens, 128);
        }

        for gem in ["gemini-4-pro", "gemini-3-flash"] {
            let p = lookup_pricing(gem, ProviderFormat::Generic);
            assert!(approx_eq(p.input_per_mtok, 1.25));
            assert!(approx_eq(p.write_5m_per_mtok, 1.25));
            assert!(approx_eq(p.write_1h_per_mtok, 1.50));
            assert!(approx_eq(p.read_per_mtok, 0.125));
            assert_eq!(p.min_cache_tokens, 1024);
            assert_eq!(p.max_breakpoints, 1);
            assert!(p.supports_1h_ttl);
        }

        for local in ["vllm", "sglang/qwen3-72b", "meta-llama/llama-4-70b"] {
            let p = lookup_pricing(local, ProviderFormat::VllmOpenAi);
            assert!(approx_eq(p.input_per_mtok, 1.00));
            assert!(approx_eq(p.write_5m_per_mtok, 1.00));
            assert!(approx_eq(p.read_per_mtok, 0.05));
            assert_eq!(p.min_cache_tokens, 64);
            assert_eq!(p.max_breakpoints, 4);
            assert_eq!(p.block_align_tokens, 16);
        }
    }

    #[test]
    fn test_list_supported_models_and_cost_helpers() {
        let models = list_supported_models();
        assert_eq!(models.len(), PRICING_CATALOG.len());
        for m in &models {
            assert!(m.input_per_mtok > 0.0);
            assert!(m.read_per_mtok < m.input_per_mtok);
            assert!(m.write_5m_per_mtok >= m.input_per_mtok);
            assert!(m.write_1h_per_mtok >= m.write_5m_per_mtok);
            assert!(m.block_align_tokens >= 1);
            assert!(m.max_breakpoints >= 1);
            let w5 = m.cost_write(1_000_000, CacheTtl::FiveMin);
            assert!(approx_eq(w5, m.write_5m_per_mtok));
        }
    }

    #[test]
    fn test_empty_and_adversarial_model_names() {
        let p_empty = lookup_pricing("", ProviderFormat::Generic);
        assert_eq!(p_empty.min_cache_tokens, 64);
        assert_eq!(p_empty.block_align_tokens, 16);

        let p_slash = lookup_pricing("///", ProviderFormat::AnthropicMessages);
        assert_eq!(p_slash.provider, ProviderFormat::AnthropicMessages);
        assert!(approx_eq(p_slash.input_per_mtok, 2.00));

        let p_ws = lookup_pricing("  CLAUDE-OPUS-5.5  ", ProviderFormat::Generic);
        assert!(approx_eq(p_ws.input_per_mtok, 4.00));
    }
}
