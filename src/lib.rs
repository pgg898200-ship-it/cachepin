//! `cachepin` — Sub-microsecond multi-agent LLM prompt-cache compiler,
//! exact DP breakpoint optimizer, TTFT-triggered single-flight thundering-herd
//! coalescer, and replication-aware Shadow-KV radix router.
//!
//! Zero external crates. Portable Rust 2021 with AVX2 SIMD acceleration and C-ABI FFI.

pub mod audit;
pub mod bench;
pub mod canonical;
pub mod cli;
pub mod coalescer;
pub mod dashboard;
pub mod dp_planner;
pub mod ffi;
pub mod json;
pub mod mcp;
pub mod pricing;
pub mod proxy;
pub mod radix;
pub mod router;
pub mod simd;
pub mod types;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use crate::canonical::{apply_breakpoints_to_json, CompilerConfig, PrefixCompiler};
use crate::dp_planner::solve_optimal_breakpoints;
use crate::pricing::lookup_pricing;
use crate::radix::PrefixRadixTree;
use crate::simd::derive_tenant_key;
use crate::types::{BreakpointPlan, CacheDiagnostic, CanonicalMode, CompileResult, ProviderFormat};

/// High-level embeddable engine combining canonicalization, radix miss forensics,
/// and exact O(M^2 K) mixed-TTL breakpoint planning.
pub struct CachePinEngine {
    compiler: PrefixCompiler,
    radix: PrefixRadixTree,
    tenant_secret: [u8; 16],
    mode: CanonicalMode,
}

#[derive(Debug, Clone)]
pub struct EngineOutput {
    pub compiled: CompileResult,
    pub diagnostic: CacheDiagnostic,
    pub plan: BreakpointPlan,
    pub optimized_json: Vec<u8>,
}

impl CachePinEngine {
    pub fn new(mode: CanonicalMode, max_radix_nodes: usize) -> Self {
        let mut cfg = CompilerConfig::default();
        cfg.mode = mode;
        let tenant_secret = cfg.tenant_secret;
        Self {
            compiler: PrefixCompiler::new(cfg),
            radix: PrefixRadixTree::new(max_radix_nodes),
            tenant_secret,
            mode,
        }
    }

    pub fn process_request(
        &mut self,
        tenant_id: &str,
        raw_json: &[u8],
        provider_hint: Option<ProviderFormat>,
        now_ms: u64,
        inject_breakpoints: bool,
    ) -> Result<EngineOutput, String> {
        let compiled = self
            .compiler
            .compile_request(tenant_id, raw_json, provider_hint)?;
        let pricing = lookup_pricing(&compiled.model, compiled.provider);
        let tk = derive_tenant_key(&self.tenant_secret, tenant_id);
        let tenant_u64 = u64::from_le_bytes([
            tk[0], tk[1], tk[2], tk[3], tk[4], tk[5], tk[6], tk[7],
        ]);

        let diagnostic = self.radix.lookup_and_diagnose(
            tenant_u64,
            &compiled.model,
            &compiled.blocks,
            now_ms,
            &pricing,
            compiled.tools_reordered,
            compiled.json_keys_reordered,
            compiled.volatile_spans_hoisted,
            compiled.all_gather_reordered,
        );
        let reuse =
            self.radix
                .estimate_reuse_profile(tenant_u64, &compiled.model, &compiled.blocks, now_ms);
        let plan = solve_optimal_breakpoints(&compiled.blocks, &reuse, &pricing);
        self.radix.record_request(
            tenant_u64,
            &compiled.model,
            &compiled.blocks,
            &plan.breakpoints,
            now_ms,
        );

        let optimized_json = if self.mode == CanonicalMode::ObserveOnly {
            raw_json.to_vec()
        } else if inject_breakpoints && !plan.breakpoints.is_empty() {
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

        Ok(EngineOutput {
            compiled,
            diagnostic,
            plan,
            optimized_json,
        })
    }
}
