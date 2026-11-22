//! Zero-dependency multi-threaded HTTP/1.1 reverse proxy with:
//!   - Tenant-isolated algebraic prefix canonicalization + zero-reparse exact DP `cache_control` injection
//!   - TTFT-triggered single-flight thundering-herd coalescing (leader wakes followers on first upstream header!)
//!   - Per-backend HTTP/1.1 keep-alive upstream connection pool + persistent client keep-alive
//!   - Replication-aware Shadow-KV multi-backend routing for self-hosted vLLM/SGLang fleets
//!   - Zero-buffering SSE / chunked pass-through streaming + non-blocking upstream provider `usage` reconciliation
//!   - Live `/health`, `/stats`, `/metrics` (Prometheus), and zero-CDN `/dashboard`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::canonical::{apply_breakpoints_to_value, CompilerConfig, PrefixCompiler};
use crate::coalescer::{FlightRole, SingleFlightCoalescer};
use crate::dashboard::{render_dashboard_html, render_prometheus_metrics_with_provider};
use crate::dp_planner::solve_optimal_breakpoints;
use crate::json::{estimate_serialized_len, serialize_json};
use crate::pricing::lookup_pricing;
use crate::radix::PrefixRadixTree;
use crate::router::{BackendConfig, RoutingPolicy, ShadowKvRouter};
use crate::simd::{derive_tenant_key, keyed_hash64};
use crate::types::{CanonicalMode, MissReason, ProviderFormat};

const MAX_POOLED_PER_BACKEND: usize = 32;
const MAX_POOLED_IDLE_SECS: u64 = 25;
const USAGE_TAIL_LIMIT: usize = 8192;

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub listen_addr: String,
    pub upstreams: Vec<String>,
    pub mode: CanonicalMode,
    pub provider_hint: Option<ProviderFormat>,
    pub inject_breakpoints: bool,
    pub enable_coalescing: bool,
    pub coalesce_max_wait_ms: u64,
    pub coalesce_warm_ttl_ms: u64,
    pub routing_policy: RoutingPolicy,
    pub kv_capacity_tokens: u64,
    pub max_body_bytes: usize,
    pub upstream_timeout_ms: u64,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            listen_addr: "127.0.0.1:47820".to_string(),
            upstreams: vec!["http://127.0.0.1:8000".to_string()],
            mode: CanonicalMode::SwarmAlign,
            provider_hint: None,
            inject_breakpoints: true,
            enable_coalescing: true,
            coalesce_max_wait_ms: 1500,
            coalesce_warm_ttl_ms: 280_000,
            routing_policy: RoutingPolicy::CachePinAdaptiveReplication,
            kv_capacity_tokens: 131_072,
            max_body_bytes: 32 * 1024 * 1024,
            upstream_timeout_ms: 120_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecentRequestLog {
    pub ts_ms: u64,
    pub model: String,
    pub reason: String,
    pub matched_tokens: u32,
    pub total_tokens: u32,
    pub compile_us: f64,
    pub waited_ms: u64,
    pub backend_id: usize,
    pub path: String,
    pub saved_usd: f64,
}

pub struct ProxyTelemetry {
    pub total_requests: AtomicU64,
    pub cache_hit_tokens: AtomicU64,
    pub total_input_tokens: AtomicU64,
    pub provider_confirmed_hit_tokens: AtomicU64,
    pub provider_confirmed_write_tokens: AtomicU64,
    pub upstream_pool_reuses: AtomicU64,
    pub total_saved_micro_usd: AtomicU64,
    pub total_wasted_micro_usd: AtomicU64,
    pub tools_reordered: AtomicU64,
    pub json_keys_reordered: AtomicU64,
    pub volatile_hoisted: AtomicU64,
    pub all_gather_reordered: AtomicU64,
    pub upstream_errors: AtomicU64,
    pub compile_latencies_us: Mutex<Vec<f64>>,
    pub miss_reasons: Mutex<HashMap<String, u64>>,
    pub recent: Mutex<Vec<RecentRequestLog>>,
}

impl ProxyTelemetry {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            cache_hit_tokens: AtomicU64::new(0),
            total_input_tokens: AtomicU64::new(0),
            provider_confirmed_hit_tokens: AtomicU64::new(0),
            provider_confirmed_write_tokens: AtomicU64::new(0),
            upstream_pool_reuses: AtomicU64::new(0),
            total_saved_micro_usd: AtomicU64::new(0),
            total_wasted_micro_usd: AtomicU64::new(0),
            tools_reordered: AtomicU64::new(0),
            json_keys_reordered: AtomicU64::new(0),
            volatile_hoisted: AtomicU64::new(0),
            all_gather_reordered: AtomicU64::new(0),
            upstream_errors: AtomicU64::new(0),
            compile_latencies_us: Mutex::new(Vec::with_capacity(4096)),
            miss_reasons: Mutex::new(HashMap::new()),
            recent: Mutex::new(Vec::with_capacity(64)),
        }
    }

    pub fn record_latency(&self, us: f64) {
        if let Ok(mut v) = self.compile_latencies_us.lock() {
            if v.len() >= 8192 {
                v.drain(0..4096);
            }
            v.push(us);
        }
    }

    pub fn latency_percentiles(&self) -> (f64, f64) {
        let mut v = match self.compile_latencies_us.lock() {
            Ok(g) => g.clone(),
            Err(_) => return (0.0, 0.0),
        };
        if v.is_empty() {
            return (0.0, 0.0);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let p50 = v[v.len() / 2];
        let p99 = v[((v.len() as f64 * 0.99) as usize).min(v.len() - 1)];
        (p50, p99)
    }
}

pub struct ProxyState {
    pub config: ProxyConfig,
    pub compiler: PrefixCompiler,
    pub radix: Mutex<PrefixRadixTree>,
    pub coalescer: SingleFlightCoalescer,
    pub router: Mutex<ShadowKvRouter>,
    pub telemetry: ProxyTelemetry,
    pub upstreams_parsed: Vec<ParsedUpstream>,
    pub upstream_pools: Vec<Mutex<Vec<(TcpStream, Instant)>>>,
    pub tenant_secret: [u8; 16],
}

impl ProxyState {
    fn pop_pooled_upstream(&self, backend_idx: usize) -> Option<TcpStream> {
        let pool = self.upstream_pools.get(backend_idx)?;
        let mut guard = pool.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        while let Some((stream, saved_at)) = guard.pop() {
            if now.saturating_duration_since(saved_at).as_secs() < MAX_POOLED_IDLE_SECS {
                return Some(stream);
            }
        }
        None
    }

    fn return_pooled_upstream(&self, backend_idx: usize, stream: TcpStream) {
        if let Some(pool) = self.upstream_pools.get(backend_idx) {
            let mut guard = pool.lock().unwrap_or_else(|e| e.into_inner());
            if guard.len() < MAX_POOLED_PER_BACKEND {
                guard.push((stream, Instant::now()));
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedUpstream {
    pub raw: String,
    pub host: String,
    pub port: u16,
    pub host_header: String,
    pub is_tls: bool,
}

impl ParsedUpstream {
    pub fn parse(url: &str) -> Result<Self, String> {
        let trimmed = url.trim().trim_end_matches('/');
        let (is_tls, rest) = if let Some(r) = trimmed.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = trimmed.strip_prefix("http://") {
            (false, r)
        } else {
            (false, trimmed)
        };
        let authority = rest.split('/').next().unwrap_or(rest);
        let (host, port) = if let Some((h, p)) = authority.rsplit_once(':') {
            let port_num = p
                .parse::<u16>()
                .map_err(|_| format!("invalid port in upstream '{}'", url))?;
            (h.to_string(), port_num)
        } else {
            (authority.to_string(), if is_tls { 443 } else { 80 })
        };
        let host_header = if (is_tls && port == 443) || (!is_tls && port == 80) {
            host.clone()
        } else {
            format!("{}:{}", host, port)
        };
        Ok(Self {
            raw: trimmed.to_string(),
            host,
            port,
            host_header,
            is_tls,
        })
    }
}

pub struct ProxyHandle {
    pub bound_addr: String,
    pub state: Arc<ProxyState>,
    shutdown: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl ProxyHandle {
    pub fn stop(mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(&self.bound_addr);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Start the reverse proxy in a background thread (used by both CLI and live tests/benchmarks).
pub fn start_proxy_background(config: ProxyConfig) -> Result<ProxyHandle, String> {
    if config.upstreams.is_empty() {
        return Err("at least one upstream URL is required".to_string());
    }
    let mut parsed = Vec::with_capacity(config.upstreams.len());
    let mut backend_cfgs = Vec::with_capacity(config.upstreams.len());
    let mut pools = Vec::with_capacity(config.upstreams.len());
    for (idx, u) in config.upstreams.iter().enumerate() {
        let pu = ParsedUpstream::parse(u)?;
        backend_cfgs.push(BackendConfig {
            id: idx,
            url: pu.raw.clone(),
            kv_capacity_tokens: config.kv_capacity_tokens,
            prefill_tps: 40_000.0,
            cache_read_tps: 400_000.0,
            max_inflight: 64,
        });
        parsed.push(pu);
        pools.push(Mutex::new(Vec::with_capacity(MAX_POOLED_PER_BACKEND)));
    }

    let listener = TcpListener::bind(&config.listen_addr)
        .map_err(|e| format!("failed to bind {}: {}", config.listen_addr, e))?;
    let bound_addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| config.listen_addr.clone());

    let mut comp_cfg = CompilerConfig::default();
    comp_cfg.mode = config.mode;
    let tenant_secret = comp_cfg.tenant_secret;

    let state = Arc::new(ProxyState {
        compiler: PrefixCompiler::new(comp_cfg),
        radix: Mutex::new(PrefixRadixTree::new(65_536)),
        coalescer: SingleFlightCoalescer::new(
            config.coalesce_max_wait_ms,
            config.coalesce_warm_ttl_ms,
        ),
        router: Mutex::new(ShadowKvRouter::new(backend_cfgs, config.routing_policy)),
        telemetry: ProxyTelemetry::new(),
        upstreams_parsed: parsed,
        upstream_pools: pools,
        tenant_secret,
        config,
    });

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_loop = Arc::clone(&shutdown);
    let state_loop = Arc::clone(&state);

    let join = thread::spawn(move || {
        for stream_res in listener.incoming() {
            if shutdown_loop.load(Ordering::SeqCst) {
                break;
            }
            let stream = match stream_res {
                Ok(s) => s,
                Err(_) => continue,
            };
            let st = Arc::clone(&state_loop);
            thread::spawn(move || {
                let _ = handle_client(stream, st);
            });
        }
    });

    Ok(ProxyHandle {
        bound_addr,
        state,
        shutdown,
        join: Some(join),
    })
}

/// Blocking foreground proxy runner for CLI (`cachepin proxy`).
pub fn run_proxy_blocking(config: ProxyConfig) -> Result<(), String> {
    let handle = start_proxy_background(config)?;
    eprintln!(
        "[cachepin] listening on http://{}  (dashboard: http://{}/dashboard  |  metrics: http://{}/metrics)",
        handle.bound_addr, handle.bound_addr, handle.bound_addr
    );
    if let Some(j) = handle.join {
        let _ = j.join();
    }
    Ok(())
}

struct ParsedHttpRequest {
    method: String,
    target: String,
    version: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    client_wants_keep_alive: bool,
}

fn read_http_request(
    reader: &mut BufReader<TcpStream>,
    max_body: usize,
) -> Result<ParsedHttpRequest, String> {
    let mut req_line = String::new();
    let n = reader.read_line(&mut req_line).map_err(|e| e.to_string())?;
    if n == 0 {
        return Err("eof".to_string());
    }
    let parts: Vec<&str> = req_line.trim_end().split_whitespace().collect();
    if parts.len() < 3 {
        return Err("malformed request line".to_string());
    }
    let method = parts[0].to_string();
    let target = parts[1].to_string();
    let version = parts[2].to_string();

    let mut headers = Vec::with_capacity(12);
    let mut content_length: usize = 0;
    let mut chunked = false;
    let mut conn_close = false;
    let mut conn_keep_alive = false;
    loop {
        let mut line = String::new();
        let m = reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if m == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_string();
            let val = v.trim().to_string();
            if key.eq_ignore_ascii_case("content-length") {
                content_length = val.parse::<usize>().unwrap_or(0);
            } else if key.eq_ignore_ascii_case("transfer-encoding")
                && val.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            } else if key.eq_ignore_ascii_case("connection") {
                let low = val.to_ascii_lowercase();
                if low.contains("close") {
                    conn_close = true;
                } else if low.contains("keep-alive") {
                    conn_keep_alive = true;
                }
            }
            headers.push((key, val));
        }
    }

    let client_wants_keep_alive = if version.eq_ignore_ascii_case("HTTP/1.1") {
        !conn_close
    } else {
        conn_keep_alive && !conn_close
    };

    let mut body = Vec::new();
    if chunked {
        loop {
            let mut sz_line = String::new();
            if reader.read_line(&mut sz_line).map_err(|e| e.to_string())? == 0 {
                break;
            }
            let sz_hex = sz_line.trim().split(';').next().unwrap_or("0");
            let chunk_sz = usize::from_str_radix(sz_hex, 16).unwrap_or(0);
            if chunk_sz == 0 {
                let mut trailer = String::new();
                let _ = reader.read_line(&mut trailer);
                break;
            }
            if body.len() + chunk_sz > max_body {
                return Err("request body exceeds limit".to_string());
            }
            let start = body.len();
            body.resize(start + chunk_sz, 0);
            reader
                .read_exact(&mut body[start..])
                .map_err(|e| e.to_string())?;
            let mut crlf = [0u8; 2];
            let _ = reader.read_exact(&mut crlf);
        }
    } else if content_length > 0 {
        if content_length > max_body {
            return Err("request body exceeds limit".to_string());
        }
        body.resize(content_length, 0);
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }

    Ok(ParsedHttpRequest {
        method,
        target,
        version,
        headers,
        body,
        client_wants_keep_alive,
    })
}

fn write_simple_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    keep_alive: bool,
) -> Result<(), String> {
    let conn = if keep_alive { "keep-alive" } else { "close" };
    let hdr = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: {}\r\nCache-Control: no-store\r\n\r\n",
        status,
        content_type,
        body.len(),
        conn
    );
    stream.write_all(hdr.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn build_stats_json(state: &ProxyState) -> String {
    let total_reqs = state.telemetry.total_requests.load(Ordering::Relaxed);
    let hit_toks = state.telemetry.cache_hit_tokens.load(Ordering::Relaxed);
    let total_toks = state.telemetry.total_input_tokens.load(Ordering::Relaxed);
    let prov_hit = state
        .telemetry
        .provider_confirmed_hit_tokens
        .load(Ordering::Relaxed);
    let prov_write = state
        .telemetry
        .provider_confirmed_write_tokens
        .load(Ordering::Relaxed);
    let pool_reuses = state.telemetry.upstream_pool_reuses.load(Ordering::Relaxed);
    let hit_rate = if total_toks > 0 {
        (hit_toks as f64 / total_toks as f64) * 100.0
    } else {
        0.0
    };
    let saved_usd = state.telemetry.total_saved_micro_usd.load(Ordering::Relaxed) as f64 / 1e6;
    let wasted_usd = state.telemetry.total_wasted_micro_usd.load(Ordering::Relaxed) as f64 / 1e6;
    let (p50_us, p99_us) = state.telemetry.latency_percentiles();
    let coal = state.coalescer.stats();

    let mut out = String::with_capacity(2048);
    out.push('{');
    out.push_str(&format!(
        "\"total_requests\":{},\"cache_hit_tokens\":{},\"total_input_tokens\":{},\"cache_hit_rate_pct\":{:.2},\"saved_usd\":{:.6},\"wasted_usd\":{:.6},",
        total_reqs, hit_toks, total_toks, hit_rate, saved_usd, wasted_usd
    ));
    out.push_str(&format!(
        "\"provider_confirmed\":{{\"hit_tokens\":{},\"write_tokens\":{},\"upstream_pool_reuses\":{}}},",
        prov_hit, prov_write, pool_reuses
    ));
    out.push_str(&format!(
        "\"compile_p50_us\":{:.2},\"compile_p99_us\":{:.2},\"upstream_errors\":{},",
        p50_us,
        p99_us,
        state.telemetry.upstream_errors.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "\"repairs\":{{\"tools_reordered\":{},\"json_keys_reordered\":{},\"volatile_hoisted\":{},\"all_gather_reordered\":{}}},",
        state.telemetry.tools_reordered.load(Ordering::Relaxed),
        state.telemetry.json_keys_reordered.load(Ordering::Relaxed),
        state.telemetry.volatile_hoisted.load(Ordering::Relaxed),
        state.telemetry.all_gather_reordered.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "\"coalescer\":{{\"leaders\":{},\"followers_coalesced\":{},\"followers_warmed\":{},\"followers_timed_out\":{},\"warm_bypasses\":{},\"total_wait_ms\":{}}},",
        coal.leaders, coal.followers_coalesced, coal.followers_warmed, coal.followers_timed_out, coal.warm_bypasses, coal.total_wait_ms
    ));
    out.push_str("\"miss_reasons\":{");
    if let Ok(mr) = state.telemetry.miss_reasons.lock() {
        let mut items: Vec<(&String, &u64)> = mr.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1));
        for (i, (k, v)) in items.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("\"{}\":{}", k, v));
        }
    }
    out.push_str("},\"recent\":[");
    if let Ok(rec) = state.telemetry.recent.lock() {
        for (i, r) in rec.iter().rev().take(25).enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"ts_ms\":{},\"model\":\"{}\",\"reason\":\"{}\",\"matched_tokens\":{},\"total_tokens\":{},\"compile_us\":{:.2},\"waited_ms\":{},\"backend_id\":{},\"path\":\"{}\",\"saved_usd\":{:.6}}}",
                r.ts_ms, r.model, r.reason, r.matched_tokens, r.total_tokens, r.compile_us, r.waited_ms, r.backend_id, r.path, r.saved_usd
            ));
        }
    }
    out.push_str("]}");
    out
}

/// Append streamed response bytes to a bounded rolling tail for non-blocking
/// post-stream provider `usage` extraction (zero extra latency on client path).
#[inline]
fn capture_usage_tail(tail: &mut Vec<u8>, chunk: &[u8]) {
    if chunk.is_empty() {
        return;
    }
    if chunk.len() >= USAGE_TAIL_LIMIT {
        tail.clear();
        tail.extend_from_slice(&chunk[chunk.len() - USAGE_TAIL_LIMIT..]);
        return;
    }
    if tail.len() + chunk.len() > USAGE_TAIL_LIMIT {
        let drop_n = (tail.len() + chunk.len()) - USAGE_TAIL_LIMIT;
        tail.drain(0..drop_n);
    }
    tail.extend_from_slice(chunk);
}

/// Fast zero-panic byte extractor for real upstream provider-confirmed cache read/write tokens
/// from JSON or SSE response tails (`cache_read_input_tokens`, `cache_creation_input_tokens`, `cached_tokens`).
pub(crate) fn extract_provider_usage_tokens(buf: &[u8]) -> (u64, u64) {
    fn find_last_u64_field(hay: &[u8], key: &[u8]) -> Option<u64> {
        if hay.len() < key.len() + 2 {
            return None;
        }
        let mut best: Option<u64> = None;
        let mut i = 0usize;
        while i + key.len() <= hay.len() {
            if &hay[i..i + key.len()] == key {
                let mut j = i + key.len();
                while j < hay.len() && matches!(hay[j], b' ' | b'\t' | b'\r' | b'\n') {
                    j += 1;
                }
                if j < hay.len() && hay[j] == b':' {
                    j += 1;
                    while j < hay.len() && matches!(hay[j], b' ' | b'\t' | b'\r' | b'\n') {
                        j += 1;
                    }
                    let start = j;
                    let mut val: u64 = 0;
                    while j < hay.len() && hay[j].is_ascii_digit() {
                        val = val.saturating_mul(10).saturating_add((hay[j] - b'0') as u64);
                        j += 1;
                    }
                    if j > start {
                        best = Some(val);
                    }
                }
                i += key.len();
            } else {
                i += 1;
            }
        }
        best
    }

    let anth_hit = find_last_u64_field(buf, b"\"cache_read_input_tokens\"");
    let anth_write = find_last_u64_field(buf, b"\"cache_creation_input_tokens\"").unwrap_or(0);
    let hit = anth_hit
        .or_else(|| find_last_u64_field(buf, b"\"cached_tokens\""))
        .unwrap_or(0);
    (hit, anth_write)
}

fn handle_client(client: TcpStream, state: Arc<ProxyState>) -> Result<(), String> {
    let _ = client.set_nodelay(true);
    let _ = client.set_read_timeout(Some(Duration::from_secs(30)));
    let mut writer = client.try_clone().map_err(|e| e.to_string())?;
    let mut reader = BufReader::with_capacity(16 * 1024, client);
    let mut req_num = 0usize;

    loop {
        if req_num > 0 {
            let _ = reader
                .get_ref()
                .set_read_timeout(Some(Duration::from_secs(5)));
        }
        let req = match read_http_request(&mut reader, state.config.max_body_bytes) {
            Ok(r) => r,
            Err(_) => break,
        };
        req_num += 1;
        let keep_alive_client = req.client_wants_keep_alive;
        let keep_open = handle_single_request(req, &mut writer, &state, keep_alive_client)?;
        if !keep_open || !keep_alive_client {
            break;
        }
    }

    let _ = writer.shutdown(Shutdown::Both);
    Ok(())
}

fn handle_single_request(
    req: ParsedHttpRequest,
    client: &mut TcpStream,
    state: &ProxyState,
    keep_alive_client: bool,
) -> Result<bool, String> {
    // Built-in observability endpoints
    if req.method == "GET" {
        match req.target.split('?').next().unwrap_or("") {
            "/health" => {
                write_simple_response(
                    client,
                    "200 OK",
                    "application/json",
                    b"{\"status\":\"ok\",\"engine\":\"cachepin\"}",
                    keep_alive_client,
                )?;
                return Ok(keep_alive_client);
            }
            "/dashboard" | "/" => {
                write_simple_response(
                    client,
                    "200 OK",
                    "text/html; charset=utf-8",
                    render_dashboard_html().as_bytes(),
                    keep_alive_client,
                )?;
                return Ok(keep_alive_client);
            }
            "/stats" => {
                let body = build_stats_json(state);
                write_simple_response(
                    client,
                    "200 OK",
                    "application/json",
                    body.as_bytes(),
                    keep_alive_client,
                )?;
                return Ok(keep_alive_client);
            }
            "/metrics" => {
                let (p50, p99) = state.telemetry.latency_percentiles();
                let body = render_prometheus_metrics_with_provider(
                    state.telemetry.total_requests.load(Ordering::Relaxed),
                    state.telemetry.cache_hit_tokens.load(Ordering::Relaxed),
                    state.telemetry.total_input_tokens.load(Ordering::Relaxed),
                    state.telemetry.total_saved_micro_usd.load(Ordering::Relaxed) as f64 / 1e6,
                    p50,
                    p99,
                    &state.coalescer.stats(),
                    state
                        .telemetry
                        .provider_confirmed_hit_tokens
                        .load(Ordering::Relaxed),
                    state
                        .telemetry
                        .provider_confirmed_write_tokens
                        .load(Ordering::Relaxed),
                    state.telemetry.upstream_pool_reuses.load(Ordering::Relaxed),
                );
                write_simple_response(
                    client,
                    "200 OK",
                    "text/plain; version=0.0.4",
                    body.as_bytes(),
                    keep_alive_client,
                )?;
                return Ok(keep_alive_client);
            }
            _ => {}
        }
    }

    // Determine tenant scope strictly from auth/API-key header (never share cache across tenants!)
    let tenant_id = req
        .headers
        .iter()
        .find(|(k, _)| {
            k.eq_ignore_ascii_case("x-api-key")
                || k.eq_ignore_ascii_case("authorization")
                || k.eq_ignore_ascii_case("x-cachepin-tenant")
        })
        .map(|(_, v)| v.as_str())
        .unwrap_or("default");

    let is_llm_post = req.method == "POST"
        && !req.body.is_empty()
        && (req.target.contains("/messages")
            || req.target.contains("/chat/completions")
            || req.target.contains("/responses")
            || req.target.contains("/completions"));

    let mut forward_body_opt: Option<Vec<u8>> = None;
    let mut selected_backend: usize = 0;
    let mut leader_guard = None;
    let mut pending_leader_commit: Option<(
        u64,
        String,
        Vec<crate::types::PrefixBlock>,
        Vec<crate::types::PlannedBreakpoint>,
    )> = None;
    let mut waited_ms: u64 = 0;
    let mut est_prefill_tokens: u32 = 0;
    let mut extra_headers: Vec<(String, String)> = Vec::with_capacity(8);
    let ts_now = now_ms();

    if is_llm_post {
        let provider_hint = state.config.provider_hint.or_else(|| {
            if req.target.contains("/messages") {
                Some(ProviderFormat::AnthropicMessages)
            } else if req.target.contains("/responses") {
                Some(ProviderFormat::OpenAiResponses)
            } else if req.target.contains("/chat/completions") {
                Some(ProviderFormat::OpenAiChat)
            } else {
                None
            }
        });

        let emit_canon = state.config.mode == CanonicalMode::ObserveOnly;
        if let Ok((compiled, mut ast)) =
            state
                .compiler
                .compile_request_with_ast(tenant_id, &req.body, provider_hint, emit_canon)
        {
            let compile_us = compiled.compile_ns as f64 / 1000.0;
            state.telemetry.record_latency(compile_us);
            state.telemetry.total_requests.fetch_add(1, Ordering::Relaxed);
            state
                .telemetry
                .total_input_tokens
                .fetch_add(compiled.total_est_tokens as u64, Ordering::Relaxed);
            if compiled.tools_reordered {
                state.telemetry.tools_reordered.fetch_add(1, Ordering::Relaxed);
            }
            if compiled.json_keys_reordered {
                state
                    .telemetry
                    .json_keys_reordered
                    .fetch_add(1, Ordering::Relaxed);
            }
            state
                .telemetry
                .volatile_hoisted
                .fetch_add(compiled.volatile_spans_hoisted as u64, Ordering::Relaxed);
            if compiled.all_gather_reordered {
                state
                    .telemetry
                    .all_gather_reordered
                    .fetch_add(1, Ordering::Relaxed);
            }

            let pricing = lookup_pricing(&compiled.model, compiled.provider);
            let tk_bytes = derive_tenant_key(&state.tenant_secret, tenant_id);
            let tenant_u64 = u64::from_le_bytes([
                tk_bytes[0], tk_bytes[1], tk_bytes[2], tk_bytes[3],
                tk_bytes[4], tk_bytes[5], tk_bytes[6], tk_bytes[7],
            ]);

            // 1. TTFT-triggered single-flight thundering-herd coalescing (BEFORE radix/DP!).
            let model_hash = keyed_hash64(&tk_bytes, compiled.model.as_bytes());
            let flight_key = SingleFlightCoalescer::compute_flight_key(
                tenant_u64,
                model_hash,
                compiled.shared_prefix_hash,
            );
            let mut follower_warmed = false;
            if state.config.enable_coalescing
                && compiled.total_est_tokens >= pricing.min_cache_tokens
            {
                match state.coalescer.acquire(flight_key, now_ms()) {
                    FlightRole::Leader(guard) => {
                        leader_guard = Some(guard);
                    }
                    FlightRole::FollowerWaited {
                        waited_ms: w,
                        leader_warmed,
                    } => {
                        waited_ms = w;
                        follower_warmed = leader_warmed;
                    }
                    FlightRole::BypassWarm => {}
                }
            }

            // 2. Radix lookup + exact DP breakpoint plan (linearized after leader TTFT!).
            let lookup_now = now_ms();
            let (diag, plan) = {
                let mut rx = state.radix.lock().unwrap_or_else(|e| e.into_inner());
                let d = rx.lookup_and_diagnose(
                    tenant_u64,
                    &compiled.model,
                    &compiled.blocks,
                    lookup_now,
                    &pricing,
                    compiled.tools_reordered,
                    compiled.json_keys_reordered,
                    compiled.volatile_spans_hoisted,
                    compiled.all_gather_reordered,
                );
                let reuse = rx.estimate_reuse_profile(
                    tenant_u64,
                    &compiled.model,
                    &compiled.blocks,
                    lookup_now,
                );
                let p = solve_optimal_breakpoints(&compiled.blocks, &reuse, &pricing);
                if leader_guard.is_some() {
                    pending_leader_commit = Some((
                        tenant_u64,
                        compiled.model.clone(),
                        compiled.blocks.clone(),
                        p.breakpoints.clone(),
                    ));
                } else {
                    rx.record_request(
                        tenant_u64,
                        &compiled.model,
                        &compiled.blocks,
                        &p.breakpoints,
                        lookup_now,
                    );
                }
                (d, p)
            };

            let coalesced_hit_bonus: u32 = if follower_warmed {
                compiled
                    .total_est_tokens
                    .saturating_sub(diag.matched_tokens)
            } else {
                0
            };

            // 3. Zero-reparse breakpoint injection directly onto the compiled AST!
            if state.config.mode != CanonicalMode::ObserveOnly {
                let cache_key = format!("cp-{:016x}", compiled.shared_prefix_hash);
                if state.config.inject_breakpoints && !plan.breakpoints.is_empty() {
                    forward_body_opt = Some(apply_breakpoints_to_value(
                        &mut ast,
                        compiled.provider,
                        &plan,
                        Some(&cache_key),
                    ));
                } else {
                    let mut buf = Vec::with_capacity(estimate_serialized_len(&ast) + 32);
                    serialize_json(&ast, &mut buf);
                    forward_body_opt = Some(buf);
                }
            }

            // 4. Multi-backend Replication-Aware Shadow-KV Routing
            if state.upstreams_parsed.len() > 1 {
                if let Ok(mut rt) = state.router.lock() {
                    let dec = rt.route(
                        tenant_u64,
                        &compiled.blocks,
                        compiled.total_est_tokens,
                        now_ms(),
                    );
                    selected_backend = dec.backend_id;
                    est_prefill_tokens = compiled
                        .total_est_tokens
                        .saturating_sub(dec.est_matched_tokens);
                    extra_headers.push((
                        "X-CachePin-Backend".to_string(),
                        dec.backend_id.to_string(),
                    ));
                    extra_headers.push((
                        "X-CachePin-Route-Reason".to_string(),
                        dec.reason.to_string(),
                    ));
                }
            }

            // Record telemetry
            let effective_matched =
                (diag.matched_tokens + coalesced_hit_bonus).min(compiled.total_est_tokens);
            let saved_usd = (pricing.cost_input(effective_matched as u64)
                - pricing.cost_read(effective_matched as u64))
            .max(0.0);
            state
                .telemetry
                .cache_hit_tokens
                .fetch_add(effective_matched as u64, Ordering::Relaxed);
            state
                .telemetry
                .total_saved_micro_usd
                .fetch_add((saved_usd * 1e6) as u64, Ordering::Relaxed);
            state
                .telemetry
                .total_wasted_micro_usd
                .fetch_add((diag.wasted_usd_est * 1e6) as u64, Ordering::Relaxed);

            let effective_reason = if coalesced_hit_bonus > 0 {
                MissReason::CacheHit.as_str().to_string()
            } else {
                diag.reason.as_str().to_string()
            };
            if let Ok(mut mr) = state.telemetry.miss_reasons.lock() {
                *mr.entry(effective_reason.clone()).or_insert(0) += 1;
            }
            if let Ok(mut rec) = state.telemetry.recent.lock() {
                if rec.len() >= 100 {
                    rec.drain(0..50);
                }
                rec.push(RecentRequestLog {
                    ts_ms: ts_now,
                    model: compiled.model.clone(),
                    reason: effective_reason.clone(),
                    matched_tokens: effective_matched,
                    total_tokens: compiled.total_est_tokens,
                    compile_us,
                    waited_ms,
                    backend_id: selected_backend,
                    path: diag
                        .divergence_path
                        .unwrap_or_else(|| "prefix_ok".to_string()),
                    saved_usd,
                });
            }

            extra_headers.push(("X-CachePin-Status".to_string(), effective_reason));
            extra_headers.push((
                "X-CachePin-Matched-Tokens".to_string(),
                effective_matched.to_string(),
            ));
            extra_headers.push((
                "X-CachePin-Compile-Us".to_string(),
                format!("{:.2}", compile_us),
            ));
            extra_headers.push((
                "X-CachePin-Breakpoints".to_string(),
                plan.breakpoints.len().to_string(),
            ));
            if waited_ms > 0 {
                extra_headers.push((
                    "X-CachePin-Coalesced-Ms".to_string(),
                    waited_ms.to_string(),
                ));
            }
        }
    }

    let forward_body: &[u8] = forward_body_opt.as_deref().unwrap_or(&req.body);
    let backend_idx = selected_backend % state.upstreams_parsed.len();
    let upstream = &state.upstreams_parsed[backend_idx];

    let mut out_hdr = String::with_capacity(512);
    out_hdr.push_str(&req.method);
    out_hdr.push(' ');
    out_hdr.push_str(&req.target);
    out_hdr.push(' ');
    out_hdr.push_str(&req.version);
    out_hdr.push_str("\r\n");
    for (k, v) in &req.headers {
        if k.eq_ignore_ascii_case("host")
            || k.eq_ignore_ascii_case("content-length")
            || k.eq_ignore_ascii_case("transfer-encoding")
            || k.eq_ignore_ascii_case("connection")
        {
            continue;
        }
        out_hdr.push_str(k);
        out_hdr.push_str(": ");
        out_hdr.push_str(v);
        out_hdr.push_str("\r\n");
    }
    out_hdr.push_str("Host: ");
    out_hdr.push_str(&upstream.host_header);
    out_hdr.push_str("\r\nContent-Length: ");
    out_hdr.push_str(&forward_body.len().to_string());
    if upstream.is_tls {
        out_hdr.push_str("\r\nConnection: close\r\n\r\n");
    } else {
        out_hdr.push_str("\r\nConnection: keep-alive\r\n\r\n");
    }

    let t_up = Instant::now();
    if upstream.is_tls {
        let valid_host = !upstream.host.is_empty()
            && !upstream.host.starts_with('-')
            && upstream
                .host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b':');
        if !valid_host {
            state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
            drop(leader_guard);
            write_simple_response(
                client,
                "400 Bad Request",
                "application/json",
                b"{\"error\":\"invalid upstream hostname\"}",
                false,
            )?;
            return Ok(false);
        }
        let connect_arg = format!("{}:{}", upstream.host, upstream.port);
        let mut child = match std::process::Command::new("openssl")
            .args([
                "s_client",
                "-quiet",
                "-no_ign_eof",
                "-verify_return_error",
                "-verify_hostname",
                &upstream.host,
                "-servername",
                &upstream.host,
                "-connect",
                &connect_arg,
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
                drop(leader_guard);
                let msg = format!("{{\"error\":\"openssl TLS bridge unavailable: {}\"}}", e);
                write_simple_response(
                    client,
                    "502 Bad Gateway",
                    "application/json",
                    msg.as_bytes(),
                    false,
                )?;
                return Ok(false);
            }
        };
        if let Some(mut stdin) = child.stdin.take() {
            if stdin.write_all(out_hdr.as_bytes()).is_err()
                || stdin.write_all(forward_body).is_err()
                || stdin.flush().is_err()
            {
                state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
                drop(leader_guard);
                let _ = child.kill();
                let _ = child.wait();
                write_simple_response(
                    client,
                    "502 Bad Gateway",
                    "application/json",
                    b"{\"error\":\"TLS upstream write failed\"}",
                    false,
                )?;
                return Ok(false);
            }
        }
        let stdout = match child.stdout.take() {
            Some(s) => s,
            None => {
                state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
                drop(leader_guard);
                let _ = child.kill();
                let _ = child.wait();
                write_simple_response(
                    client,
                    "502 Bad Gateway",
                    "application/json",
                    b"{\"error\":\"TLS upstream stdout missing\"}",
                    false,
                )?;
                return Ok(false);
            }
        };
        let mut up_reader = BufReader::new(stdout);
        let mut status_line = String::new();
        if up_reader.read_line(&mut status_line).is_err() || status_line.is_empty() {
            state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
            drop(leader_guard);
            let _ = child.kill();
            let _ = child.wait();
            write_simple_response(
                client,
                "502 Bad Gateway",
                "application/json",
                b"{\"error\":\"empty TLS upstream response\"}",
                false,
            )?;
            return Ok(false);
        }
        let ttft_ms = t_up.elapsed().as_secs_f64() * 1000.0;
        let is_2xx = status_line.contains(" 200 ") || status_line.contains(" 201 ");
        if let Some(guard) = leader_guard.take() {
            if is_2xx {
                let commit_now = now_ms();
                if let Some((t_u64, mdl, blks, bps)) = pending_leader_commit.take() {
                    let mut rx = state.radix.lock().unwrap_or_else(|e| e.into_inner());
                    rx.record_request(t_u64, &mdl, &blks, &bps, commit_now);
                }
                guard.mark_warm_and_release(commit_now);
            } else {
                drop(guard);
            }
        }
        let mut resp_hdr = status_line;
        loop {
            let mut line = String::new();
            let n = up_reader.read_line(&mut line).map_err(|e| e.to_string())?;
            if n == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            resp_hdr.push_str(&line);
        }
        for (k, v) in &extra_headers {
            resp_hdr.push_str(&format!("{}: {}\r\n", k, v));
        }
        resp_hdr.push_str("\r\n");
        client
            .write_all(resp_hdr.as_bytes())
            .map_err(|e| e.to_string())?;
        client.flush().map_err(|e| e.to_string())?;
        let mut buf = [0u8; 8192];
        let mut usage_tail = Vec::new();
        loop {
            match up_reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if is_llm_post && is_2xx {
                        capture_usage_tail(&mut usage_tail, &buf[..n]);
                    }
                    if client.write_all(&buf[..n]).is_err() {
                        break;
                    }
                    let _ = client.flush();
                }
                Err(_) => break,
            }
        }
        let _ = child.wait();
        if is_llm_post && is_2xx && !usage_tail.is_empty() {
            let (hit, wr) = extract_provider_usage_tokens(&usage_tail);
            if hit > 0 {
                state
                    .telemetry
                    .provider_confirmed_hit_tokens
                    .fetch_add(hit, Ordering::Relaxed);
            }
            if wr > 0 {
                state
                    .telemetry
                    .provider_confirmed_write_tokens
                    .fetch_add(wr, Ordering::Relaxed);
            }
        }
        if state.upstreams_parsed.len() > 1 {
            if let Ok(mut rt) = state.router.lock() {
                rt.complete(selected_backend, est_prefill_tokens, 64, ttft_ms);
            }
        }
        return Ok(false);
    }

    // Plain HTTP upstream with warm keep-alive connection pool + fresh-connect retry fallback.
    let addr = format!("{}:{}", upstream.host, upstream.port);
    let timeout = Duration::from_millis(state.config.upstream_timeout_ms);
    let mut connected_reader: Option<(BufReader<TcpStream>, String)> = None;
    let mut last_err = "empty upstream response".to_string();

    // Fast path: reuse a warm pooled keep-alive socket (zero TCP handshake!)
    if let Some(mut pooled) = state.pop_pooled_upstream(backend_idx) {
        let _ = pooled.set_read_timeout(Some(timeout));
        let _ = pooled.set_write_timeout(Some(timeout));
        if pooled.write_all(out_hdr.as_bytes()).is_ok()
            && pooled.write_all(forward_body).is_ok()
            && pooled.flush().is_ok()
        {
            let mut reader = BufReader::with_capacity(16 * 1024, pooled);
            let mut first_line = String::new();
            if matches!(reader.read_line(&mut first_line), Ok(n) if n > 0) {
                state
                    .telemetry
                    .upstream_pool_reuses
                    .fetch_add(1, Ordering::Relaxed);
                connected_reader = Some((reader, first_line));
            }
        }
    }

    // Fallback: fresh TCP connect with 3-attempt transient pre-header retry.
    if connected_reader.is_none() {
        for attempt in 0..3u32 {
            if attempt > 0 {
                thread::sleep(Duration::from_millis(5 * (attempt as u64)));
            }
            let mut up_stream = match TcpStream::connect(&addr) {
                Ok(s) => s,
                Err(e) => {
                    last_err = format!("upstream connect failed: {}", e);
                    continue;
                }
            };
            let _ = up_stream.set_nodelay(true);
            let _ = up_stream.set_read_timeout(Some(timeout));
            let _ = up_stream.set_write_timeout(Some(timeout));

            if up_stream.write_all(out_hdr.as_bytes()).is_err()
                || up_stream.write_all(forward_body).is_err()
                || up_stream.flush().is_err()
            {
                last_err = "upstream write failed".to_string();
                continue;
            }

            let mut reader = BufReader::with_capacity(16 * 1024, up_stream);
            let mut first_line = String::new();
            match reader.read_line(&mut first_line) {
                Ok(n) if n > 0 => {
                    connected_reader = Some((reader, first_line));
                    break;
                }
                _ => {
                    last_err = "empty upstream response".to_string();
                }
            }
        }
    }

    let (mut up_reader, status_line) = match connected_reader {
        Some(pair) => pair,
        None => {
            state.telemetry.upstream_errors.fetch_add(1, Ordering::Relaxed);
            drop(leader_guard);
            let msg = format!("{{\"error\":\"{}\"}}", last_err.replace('"', "\\\""));
            write_simple_response(
                client,
                "502 Bad Gateway",
                "application/json",
                msg.as_bytes(),
                false,
            )?;
            return Ok(false);
        }
    };

    let ttft_ms = t_up.elapsed().as_secs_f64() * 1000.0;
    let is_2xx = status_line.contains(" 200 ") || status_line.contains(" 201 ");
    if let Some(guard) = leader_guard.take() {
        if is_2xx {
            let commit_now = now_ms();
            if let Some((t_u64, mdl, blks, bps)) = pending_leader_commit.take() {
                let mut rx = state.radix.lock().unwrap_or_else(|e| e.into_inner());
                rx.record_request(t_u64, &mdl, &blks, &bps, commit_now);
            }
            guard.mark_warm_and_release(commit_now);
        } else {
            drop(guard);
        }
    }

    let mut resp_hdr = String::with_capacity(512);
    let up_is_http10 = status_line.starts_with("HTTP/1.0");
    resp_hdr.push_str(&status_line);
    let mut up_content_length: Option<usize> = None;
    let mut up_chunked = false;
    let mut up_close = up_is_http10;
    loop {
        let mut line = String::new();
        let n = up_reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let kt = k.trim();
            let vt = v.trim();
            if kt.eq_ignore_ascii_case("content-length") {
                if let Ok(len) = vt.parse::<usize>() {
                    up_content_length = Some(len);
                }
            } else if kt.eq_ignore_ascii_case("transfer-encoding")
                && vt.to_ascii_lowercase().contains("chunked")
            {
                up_chunked = true;
            } else if kt.eq_ignore_ascii_case("connection") {
                let low = vt.to_ascii_lowercase();
                if low.contains("close") {
                    up_close = true;
                } else if low.contains("keep-alive") {
                    up_close = false;
                }
                continue;
            }
        }
        resp_hdr.push_str(&line);
    }

    let known_framing = up_content_length.is_some() || up_chunked;
    let client_keep_alive_out = keep_alive_client && known_framing;
    for (k, v) in &extra_headers {
        resp_hdr.push_str(k);
        resp_hdr.push_str(": ");
        resp_hdr.push_str(v);
        resp_hdr.push_str("\r\n");
    }
    resp_hdr.push_str(if client_keep_alive_out {
        "Connection: keep-alive\r\n\r\n"
    } else {
        "Connection: close\r\n\r\n"
    });
    client
        .write_all(resp_hdr.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut buf = [0u8; 8192];
    let mut usage_tail = Vec::new();
    let mut clean_upstream_eof = false;
    let mut client_ok = true;

    if let Some(mut remaining) = up_content_length {
        while remaining > 0 {
            let to_read = remaining.min(buf.len());
            match up_reader.read(&mut buf[..to_read]) {
                Ok(0) => break,
                Ok(n) => {
                    remaining -= n;
                    if is_llm_post && is_2xx {
                        capture_usage_tail(&mut usage_tail, &buf[..n]);
                    }
                    if client.write_all(&buf[..n]).is_err() {
                        client_ok = false;
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = client.flush();
        if remaining == 0 {
            clean_upstream_eof = true;
        }
    } else if up_chunked {
        let mut done_chunks = false;
        loop {
            let mut sz_line = String::new();
            match up_reader.read_line(&mut sz_line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            if client.write_all(sz_line.as_bytes()).is_err() {
                client_ok = false;
                break;
            }
            let sz_hex = sz_line.trim().split(';').next().unwrap_or("0");
            let chunk_sz = match usize::from_str_radix(sz_hex, 16) {
                Ok(v) => v,
                Err(_) => break,
            };
            if chunk_sz == 0 {
                // Forward trailing headers up to terminating empty CRLF line
                loop {
                    let mut tr = String::new();
                    match up_reader.read_line(&mut tr) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {
                            let _ = client.write_all(tr.as_bytes());
                            if tr == "\r\n" || tr == "\n" {
                                done_chunks = true;
                                break;
                            }
                        }
                    }
                }
                let _ = client.flush();
                break;
            }
            let mut rem = chunk_sz;
            let mut broken = false;
            while rem > 0 {
                let to_read = rem.min(buf.len());
                match up_reader.read(&mut buf[..to_read]) {
                    Ok(0) | Err(_) => {
                        broken = true;
                        break;
                    }
                    Ok(n) => {
                        rem -= n;
                        if is_llm_post && is_2xx {
                            capture_usage_tail(&mut usage_tail, &buf[..n]);
                        }
                        if client.write_all(&buf[..n]).is_err() {
                            client_ok = false;
                            broken = true;
                            break;
                        }
                    }
                }
            }
            if broken {
                break;
            }
            let mut crlf = [0u8; 2];
            if up_reader.read_exact(&mut crlf).is_err() {
                break;
            }
            if client.write_all(&crlf).is_err() {
                client_ok = false;
                break;
            }
            let _ = client.flush();
        }
        if done_chunks {
            clean_upstream_eof = true;
        }
    } else {
        // Unknown framing: zero-buffering pass-through until upstream closes.
        let _ = client.flush();
        loop {
            match up_reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if is_llm_post && is_2xx {
                        capture_usage_tail(&mut usage_tail, &buf[..n]);
                    }
                    if client.write_all(&buf[..n]).is_err() {
                        client_ok = false;
                        break;
                    }
                    let _ = client.flush();
                }
                Err(_) => break,
            }
        }
    }

    // Return clean keep-alive upstream socket to pool for zero-handshake reuse!
    if clean_upstream_eof && !up_close && up_reader.buffer().is_empty() {
        let warm_stream = up_reader.into_inner();
        state.return_pooled_upstream(backend_idx, warm_stream);
    }

    // Reconcile real upstream provider-confirmed cache read/write usage (after client flush!).
    if is_llm_post && is_2xx && !usage_tail.is_empty() {
        let (hit, wr) = extract_provider_usage_tokens(&usage_tail);
        if hit > 0 {
            state
                .telemetry
                .provider_confirmed_hit_tokens
                .fetch_add(hit, Ordering::Relaxed);
        }
        if wr > 0 {
            state
                .telemetry
                .provider_confirmed_write_tokens
                .fetch_add(wr, Ordering::Relaxed);
        }
    }

    if state.upstreams_parsed.len() > 1 {
        if let Ok(mut rt) = state.router.lock() {
            rt.complete(selected_backend, est_prefill_tokens, 64, ttft_ms);
        }
    }
    Ok(client_ok && client_keep_alive_out && clean_upstream_eof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_provider_usage_tokens_anthropic_and_openai() {
        let anth = br#"event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":50,"cache_creation_input_tokens":1200,"cache_read_input_tokens":38400}}}"#;
        assert_eq!(extract_provider_usage_tokens(anth), (38400, 1200));

        let oai = br#"{"id":"chatcmpl-1","usage":{"prompt_tokens":32000,"prompt_tokens_details":{"cached_tokens":31500}}}"#;
        assert_eq!(extract_provider_usage_tokens(oai), (31500, 0));
    }
}
