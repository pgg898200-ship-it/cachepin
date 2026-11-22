//! Zero-CDN, self-contained live telemetry dashboard and Prometheus exporter for `cachepin`.
//!
//! Serves an embedded dark-mode HTML/CSS/JS console at `/dashboard` that polls
//! `/stats` every 1.5 seconds, alongside `/metrics` in Prometheus text exposition
//! format (0.0.4).

use std::fmt::Write as _;

use crate::coalescer::CoalescerStats;

/// Sanitizes non-finite floating-point values to `0.0` for safe Prometheus output.
#[inline]
fn finite_f64(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Renders Prometheus text exposition metrics (version 0.0.4) for `/metrics`.
pub fn render_prometheus_metrics(
    total_requests: u64,
    cache_hit_tokens: u64,
    total_input_tokens: u64,
    saved_usd: f64,
    compile_p50_us: f64,
    compile_p99_us: f64,
    coalescer: &CoalescerStats,
) -> String {
    let mut out = String::with_capacity(1536);

    let hit_ratio = if total_input_tokens > 0 {
        (cache_hit_tokens as f64 / total_input_tokens as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let _ = writeln!(out, "# HELP cachepin_requests_total Total LLM requests processed.");
    let _ = writeln!(out, "# TYPE cachepin_requests_total counter");
    let _ = writeln!(out, "cachepin_requests_total {}", total_requests);

    let _ = writeln!(
        out,
        "# HELP cachepin_cache_hit_tokens_total Total estimated prompt tokens served from cache."
    );
    let _ = writeln!(out, "# TYPE cachepin_cache_hit_tokens_total counter");
    let _ = writeln!(out, "cachepin_cache_hit_tokens_total {}", cache_hit_tokens);

    let _ = writeln!(
        out,
        "# HELP cachepin_input_tokens_total Total estimated input tokens processed."
    );
    let _ = writeln!(out, "# TYPE cachepin_input_tokens_total counter");
    let _ = writeln!(out, "cachepin_input_tokens_total {}", total_input_tokens);

    let _ = writeln!(
        out,
        "# HELP cachepin_cache_hit_ratio Ratio of cached input tokens to total input tokens [0, 1]."
    );
    let _ = writeln!(out, "# TYPE cachepin_cache_hit_ratio gauge");
    let _ = writeln!(out, "cachepin_cache_hit_ratio {:.6}", hit_ratio);

    let _ = writeln!(
        out,
        "# HELP cachepin_estimated_usd_saved_total Estimated cumulative USD saved via prompt caching."
    );
    let _ = writeln!(out, "# TYPE cachepin_estimated_usd_saved_total counter");
    let _ = writeln!(
        out,
        "cachepin_estimated_usd_saved_total {:.6}",
        finite_f64(saved_usd)
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_compile_latency_microseconds Prefix compiler latency summary in microseconds."
    );
    let _ = writeln!(out, "# TYPE cachepin_compile_latency_microseconds gauge");
    let _ = writeln!(
        out,
        "cachepin_compile_latency_microseconds{{quantile=\"0.5\"}} {:.3}",
        finite_f64(compile_p50_us)
    );
    let _ = writeln!(
        out,
        "cachepin_compile_latency_microseconds{{quantile=\"0.99\"}} {:.3}",
        finite_f64(compile_p99_us)
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_leaders_total Single-flight leaders elected for cold prefixes."
    );
    let _ = writeln!(out, "# TYPE cachepin_coalescer_leaders_total counter");
    let _ = writeln!(out, "cachepin_coalescer_leaders_total {}", coalescer.leaders);

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_followers_coalesced_total Thundering-herd followers coalesced."
    );
    let _ = writeln!(
        out,
        "# TYPE cachepin_coalescer_followers_coalesced_total counter"
    );
    let _ = writeln!(
        out,
        "cachepin_coalescer_followers_coalesced_total {}",
        coalescer.followers_coalesced
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_followers_warmed_total Followers released after leader TTFT warm."
    );
    let _ = writeln!(
        out,
        "# TYPE cachepin_coalescer_followers_warmed_total counter"
    );
    let _ = writeln!(
        out,
        "cachepin_coalescer_followers_warmed_total {}",
        coalescer.followers_warmed
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_followers_timed_out_total Followers that failed open on timeout."
    );
    let _ = writeln!(
        out,
        "# TYPE cachepin_coalescer_followers_timed_out_total counter"
    );
    let _ = writeln!(
        out,
        "cachepin_coalescer_followers_timed_out_total {}",
        coalescer.followers_timed_out
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_warm_bypasses_total Fast-path warm bypasses."
    );
    let _ = writeln!(out, "# TYPE cachepin_coalescer_warm_bypasses_total counter");
    let _ = writeln!(
        out,
        "cachepin_coalescer_warm_bypasses_total {}",
        coalescer.warm_bypasses
    );

    let _ = writeln!(
        out,
        "# HELP cachepin_coalescer_wait_milliseconds_total Cumulative follower wait time in ms."
    );
    let _ = writeln!(
        out,
        "# TYPE cachepin_coalescer_wait_milliseconds_total counter"
    );
    let _ = writeln!(
        out,
        "cachepin_coalescer_wait_milliseconds_total {}",
        coalescer.total_wait_ms
    );

    out
}

/// Returns the self-contained, zero-CDN HTML/CSS/JS live telemetry dashboard
/// served by `cachepin proxy` at `/dashboard`.
pub fn render_dashboard_html() -> &'static str {
    DASHBOARD_HTML
}

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cachepin · Live Prefix-Cache Telemetry</title>
<style>
:root {
  --bg: #080c14;
  --surface: #0f1624;
  --surface-2: #151e30;
  --border: #1e2a40;
  --border-hi: #2e3f5e;
  --text: #e6edf7;
  --muted: #8b9bb4;
  --dim: #5e708c;
  --emerald: #10b981;
  --emerald-bg: rgba(16, 185, 129, 0.12);
  --cyan: #06b6d4;
  --cyan-bg: rgba(6, 182, 212, 0.12);
  --amber: #f59e0b;
  --amber-bg: rgba(245, 158, 11, 0.14);
  --rose: #f43f5e;
  --rose-bg: rgba(244, 63, 94, 0.14);
  --violet: #8b5cf6;
  --violet-bg: rgba(139, 92, 246, 0.14);
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
.shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 20px 24px 40px;
}
header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.logo {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
  display: grid;
  place-items: center;
  font-family: var(--mono);
  font-weight: 800;
  font-size: 16px;
  color: #06131a;
  box-shadow: 0 0 24px rgba(16, 185, 129, 0.25);
}
.brand h1 {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.02em;
}
.brand .sub {
  font-size: 12px;
  color: var(--muted);
}
.meta-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface);
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
}
.pill strong { color: var(--text); font-weight: 600; }
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--emerald);
  box-shadow: 0 0 8px var(--emerald);
}
.dot.offline { background: var(--rose); box-shadow: 0 0 8px var(--rose); }
.dot.paused { background: var(--amber); box-shadow: 0 0 8px var(--amber); }
.btn {
  cursor: pointer;
  background: var(--surface-2);
  color: var(--text);
  border: 1px solid var(--border-hi);
  border-radius: 7px;
  padding: 5px 11px;
  font-size: 12px;
  font-family: var(--sans);
  font-weight: 500;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.btn:hover { background: #1c2840; border-color: #415982; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  position: relative;
  overflow: hidden;
}
.kpi-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  margin-bottom: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.kpi-value {
  font-family: var(--mono);
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
}
.kpi-value.emerald { color: var(--emerald); }
.kpi-value.cyan { color: var(--cyan); }
.kpi-value.amber { color: var(--amber); }
.kpi-value.violet { color: var(--violet); }
.kpi-sub {
  margin-top: 6px;
  font-size: 12px;
  color: var(--muted);
  font-family: var(--mono);
}
.meter {
  margin-top: 10px;
  height: 5px;
  border-radius: 999px;
  background: #1a2438;
  overflow: hidden;
}
.meter > span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--emerald), var(--cyan));
  transition: width 0.35s ease;
}
.two-col {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
  gap: 16px;
  margin-bottom: 18px;
}
@media (max-width: 980px) {
  .two-col { grid-template-columns: 1fr; }
}
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 14px;
}
.section-head h2 {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text);
}
.section-head .hint {
  font-size: 11px;
  color: var(--dim);
  font-family: var(--mono);
}
.bars {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.bar-row {
  display: grid;
  grid-template-columns: 195px minmax(0, 1fr) 84px;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}
@media (max-width: 600px) {
  .bar-row { grid-template-columns: 135px minmax(0, 1fr) 68px; }
}
.bar-name {
  font-family: var(--mono);
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar-track {
  height: 10px;
  border-radius: 999px;
  background: #172133;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.35s ease;
}
.bar-val {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  text-align: right;
}
.spark-wrap {
  width: 100%;
  height: 120px;
  background: #0b111d;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px;
  margin-bottom: 14px;
}
.spark-wrap svg { width: 100%; height: 100%; display: block; }
.mini-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.mini-box {
  background: #0b111d;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.mini-box .lbl {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 3px;
}
.mini-box .val {
  font-family: var(--mono);
  font-size: 16px;
  font-weight: 700;
}
.table-wrap {
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid var(--border);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
thead th {
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  background: #0b111d;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
tbody td {
  padding: 9px 12px;
  border-bottom: 1px solid rgba(30, 42, 64, 0.55);
  font-family: var(--mono);
  vertical-align: middle;
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--mono);
}
.b-hit { background: var(--emerald-bg); color: var(--emerald); }
.b-drift { background: var(--rose-bg); color: var(--rose); }
.b-warn { background: var(--amber-bg); color: var(--amber); }
.b-info { background: var(--cyan-bg); color: var(--cyan); }
.b-muted { background: rgba(139, 155, 180, 0.14); color: var(--muted); }
.path-code {
  color: #cbd5e1;
  max-width: 280px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: inline-block;
  vertical-align: bottom;
}
.kv-bar {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 150px;
}
.kv-track {
  flex: 1;
  height: 7px;
  border-radius: 999px;
  background: #1a2438;
  overflow: hidden;
}
.kv-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.3s ease;
}
.empty {
  text-align: center;
  padding: 24px;
  color: var(--dim);
  font-family: var(--mono);
  font-size: 12px;
}
.stack { display: flex; flex-direction: column; gap: 18px; }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div class="brand">
      <div class="logo">cp</div>
      <div>
        <h1>cachepin</h1>
        <div class="sub">Prefix-Cache Compiler · Exact DP Planner · Single-Flight Coalescer · Shadow-KV Router</div>
      </div>
    </div>
    <div class="meta-bar">
      <span class="pill"><span id="liveDot" class="dot"></span><strong id="liveText">LIVE</strong></span>
      <span class="pill">POLL <strong>1.5s</strong></span>
      <span class="pill">ERRORS <strong id="errVal">0</strong></span>
      <span class="pill">UPDATED <strong id="updatedVal">--</strong></span>
      <button id="toggleBtn" class="btn" type="button">Pause</button>
      <button id="refreshBtn" class="btn" type="button">Refresh</button>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="card">
      <div class="kpi-label"><span>Cache Hit Rate</span><span id="hitsRatio">token-weighted</span></div>
      <div id="hitRateVal" class="kpi-value emerald">0.0%</div>
      <div class="meter"><span id="hitRateBar" style="width:0%"></span></div>
      <div id="tokenRatioSub" class="kpi-sub">0 / 0 tokens reused</div>
    </div>

    <div class="card">
      <div class="kpi-label"><span>Estimated USD Saved</span><span>cumulative</span></div>
      <div id="usdSavedVal" class="kpi-value emerald">$0.0000</div>
      <div id="usdWastedSub" class="kpi-sub">wasted: $0.0000</div>
    </div>

    <div class="card">
      <div class="kpi-label"><span>Total Requests</span><span id="errSub">0 errors</span></div>
      <div id="totalReqVal" class="kpi-value">0</div>
      <div id="hitsMissesSub" class="kpi-sub">0 cached · 0 misses</div>
    </div>

    <div class="card">
      <div class="kpi-label"><span>Compile Latency</span><span>P50 / P99</span></div>
      <div id="latencyVal" class="kpi-value cyan">0.0 / 0.0 µs</div>
      <div class="kpi-sub">canonicalize + radix + exact DP</div>
    </div>

    <div class="card">
      <div class="kpi-label"><span>Herds Coalesced</span><span>writes avoided</span></div>
      <div id="coalescedVal" class="kpi-value violet">0</div>
      <div id="coalescedSub" class="kpi-sub">0 warmed · 0 bypass · 0 timeout</div>
    </div>

    <div class="card">
      <div class="kpi-label"><span>Canonical Fixes</span><span>auto-repaired</span></div>
      <div id="fixesVal" class="kpi-value amber">0</div>
      <div id="fixesSub" class="kpi-sub">0 hoist · 0 json · 0 tools · 0 swarm</div>
    </div>
  </section>

  <section class="two-col">
    <div class="card">
      <div class="section-head">
        <h2>Miss-Reason Forensics Breakdown</h2>
        <span class="hint">deterministic root-cause taxonomy</span>
      </div>
      <div id="missBars" class="bars">
        <div class="empty">Waiting for requests on /stats ...</div>
      </div>
    </div>

    <div class="card">
      <div class="section-head">
        <h2>Hit Rate Trend &amp; Coalescer Telemetry</h2>
        <span class="hint">1.5s polling · 40-sample window</span>
      </div>
      <div class="spark-wrap">
        <svg id="sparkSvg" viewBox="0 0 400 108" preserveAspectRatio="none" aria-label="Cache hit rate trend"></svg>
      </div>
      <div class="mini-grid">
        <div class="mini-box">
          <div class="lbl">Leaders Elected</div>
          <div id="cLeaders" class="val">0</div>
        </div>
        <div class="mini-box">
          <div class="lbl">Followers Warmed</div>
          <div id="cWarmed" class="val" style="color:var(--emerald)">0</div>
        </div>
        <div class="mini-box">
          <div class="lbl">Warm Bypasses</div>
          <div id="cBypass" class="val" style="color:var(--cyan)">0</div>
        </div>
        <div class="mini-box">
          <div class="lbl">Timed Out</div>
          <div id="cTimeout" class="val" style="color:var(--amber)">0</div>
        </div>
        <div class="mini-box">
          <div class="lbl">Avg Wait</div>
          <div id="cAvgWait" class="val">0.0 ms</div>
        </div>
        <div class="mini-box">
          <div class="lbl">Total Wait</div>
          <div id="cTotWait" class="val">0 ms</div>
        </div>
      </div>
    </div>
  </section>

  <div class="stack">
    <section id="backendsSection" class="card">
      <div class="section-head">
        <h2>Backend Replica KV Utilization &amp; Routing Table</h2>
        <span id="backendsCount" class="hint">single-upstream mode</span>
      </div>
      <div id="backendsWrap" class="table-wrap">
        <div class="empty">Multi-backend mode inactive — launch with --backends URL1,URL2 to view replica KV utilization.</div>
      </div>
    </section>

    <section class="card">
      <div class="section-head">
        <h2>Recent Requests Forensics</h2>
        <span id="recentCount" class="hint">0 captured</span>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Model</th>
              <th>Outcome</th>
              <th>Tokens (Hit / Total)</th>
              <th>Compile</th>
              <th>Wait</th>
              <th>Divergence Path</th>
              <th>Backend</th>
              <th>USD Saved</th>
            </tr>
          </thead>
          <tbody id="recentBody">
            <tr><td colspan="9" class="empty">No requests recorded yet.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</div>

<script>
(function() {
  "use strict";
  const POLL_MS = 1500;
  const MAX_HISTORY = 40;
  let paused = false;
  let timer = null;
  const history = [];

  const REASON_COLORS = {
    cache_hit: "#10b981",
    json_key_drift: "#f43f5e",
    tool_order_drift: "#fb7185",
    volatile_timestamp_or_id: "#f59e0b",
    all_gather_reorder: "#8b5cf6",
    ttl_expired: "#06b6d4",
    prefix_mutation: "#ec4899",
    cold_start: "#64748b",
    model_or_config_flip: "#eab308",
    below_min_tokens: "#475569"
  };

  function esc(s) {
    if (s === null || s === undefined) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function num(v, fallback) {
    const n = Number(v);
    return Number.isFinite(n) ? n : (fallback || 0);
  }

  function fmtInt(n) {
    return Math.round(num(n, 0)).toLocaleString("en-US");
  }

  function fmtUsd(v) {
    const x = num(v, 0);
    if (Math.abs(x) >= 1) return "$" + x.toFixed(2);
    return "$" + x.toFixed(4);
  }

  function fmtTime(ms) {
    const t = num(ms, 0);
    if (t <= 0) return "--";
    return new Date(t).toTimeString().slice(0, 8);
  }

  function badgeClass(reason) {
    if (reason === "cache_hit") return "b-hit";
    if (reason === "cold_start" || reason === "below_min_tokens") return "b-muted";
    if (reason === "ttl_expired" || reason === "volatile_timestamp_or_id") return "b-warn";
    return "b-drift";
  }

  function setStatus(state) {
    const dot = document.getElementById("liveDot");
    const txt = document.getElementById("liveText");
    dot.className = "dot" + (state === "live" ? "" : (" " + state));
    txt.textContent = state === "live" ? "LIVE" : (state === "paused" ? "PAUSED" : "RECONNECTING");
  }

  function renderSparkline() {
    const svg = document.getElementById("sparkSvg");
    if (history.length === 0) {
      svg.innerHTML = '<text x="200" y="58" text-anchor="middle" fill="#5e708c" font-size="11" font-family="monospace">Collecting samples...</text>';
      return;
    }
    const w = 400, h = 108, padX = 6, padY = 10;
    const innerW = w - padX * 2;
    const innerH = h - padY * 2;
    const pts = history.map(function(v, i) {
      const x = history.length === 1 ? padX + innerW : padX + (i / (history.length - 1)) * innerW;
      const y = padY + innerH - (Math.max(0, Math.min(100, v)) / 100) * innerH;
      return [x.toFixed(1), y.toFixed(1)];
    });
    const lineD = "M " + pts.map(function(p) { return p[0] + " " + p[1]; }).join(" L ");
    const areaD = lineD + " L " + pts[pts.length - 1][0] + " " + (h - padY) + " L " + pts[0][0] + " " + (h - padY) + " Z";
    svg.innerHTML =
      '<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="#10b981" stop-opacity="0.38"/>' +
      '<stop offset="100%" stop-color="#10b981" stop-opacity="0.02"/>' +
      '</linearGradient></defs>' +
      '<line x1="6" y1="34" x2="394" y2="34" stroke="#1e2a40" stroke-dasharray="3 3" stroke-width="1"/>' +
      '<line x1="6" y1="66" x2="394" y2="66" stroke="#1e2a40" stroke-dasharray="3 3" stroke-width="1"/>' +
      '<path d="' + areaD + '" fill="url(#sg)"/>' +
      '<path d="' + lineD + '" fill="none" stroke="#10b981" stroke-width="2" stroke-linejoin="round"/>' +
      '<circle cx="' + pts[pts.length - 1][0] + '" cy="' + pts[pts.length - 1][1] + '" r="3.5" fill="#06b6d4"/>';
  }

  function renderMissBars(missReasons) {
    const el = document.getElementById("missBars");
    const entries = Object.entries(missReasons || {})
      .map(function(kv) { return [kv[0], num(kv[1], 0)]; })
      .filter(function(kv) { return kv[1] > 0; })
      .sort(function(a, b) { return b[1] - a[1]; });
    if (entries.length === 0) {
      el.innerHTML = '<div class="empty">No diagnostic events recorded yet.</div>';
      return;
    }
    const total = entries.reduce(function(acc, kv) { return acc + kv[1]; }, 0) || 1;
    el.innerHTML = entries.map(function(kv) {
      const name = kv[0];
      const count = kv[1];
      const pct = (count / total) * 100;
      const color = REASON_COLORS[name] || "#06b6d4";
      return '<div class="bar-row">' +
        '<div class="bar-name" title="' + esc(name) + '">' + esc(name) + '</div>' +
        '<div class="bar-track"><div class="bar-fill" style="width:' + pct.toFixed(1) + '%;background:' + color + '"></div></div>' +
        '<div class="bar-val">' + fmtInt(count) + ' (' + pct.toFixed(0) + '%)</div>' +
        '</div>';
    }).join("");
  }

  function deriveBackendsFromRecent(recent) {
    const map = {};
    (recent || []).forEach(function(r) {
      if (r.backend_id === undefined || r.backend_id === null) return;
      const key = "backend-" + r.backend_id;
      if (!map[key]) {
        map[key] = { id: key, routed_requests: 0, prefix_hits: 0, matched_tokens: 0, total_tokens: 0 };
      }
      map[key].routed_requests += 1;
      if (r.reason === "cache_hit") map[key].prefix_hits += 1;
      map[key].matched_tokens += num(r.matched_tokens, 0);
      map[key].total_tokens += num(r.total_tokens, 0);
    });
    return Object.values(map);
  }

  function renderBackends(backends, recent) {
    const wrap = document.getElementById("backendsWrap");
    const countEl = document.getElementById("backendsCount");
    const explicit = Array.isArray(backends) ? backends : [];
    if (explicit.length > 0) {
      countEl.textContent = explicit.length + " replica" + (explicit.length === 1 ? "" : "s") + " active";
      const rows = explicit.map(function(b) {
        const used = num(b.kv_used_tokens, 0);
        const cap = Math.max(1, num(b.kv_capacity_tokens, 1));
        const util = Math.min(100, (used / cap) * 100);
        const barColor = util > 85 ? "#f43f5e" : (util > 65 ? "#f59e0b" : "#10b981");
        const routed = num(b.routed_requests, 0);
        const hits = num(b.prefix_hits, 0);
        const hitPct = routed > 0 ? ((hits / routed) * 100).toFixed(1) + "%" : "--";
        return "<tr>" +
          "<td><strong>" + esc(b.id) + "</strong></td>" +
          '<td style="color:var(--muted)">' + esc(b.url || "--") + "</td>" +
          '<td><div class="kv-bar"><div class="kv-track"><div class="kv-fill" style="width:' + util.toFixed(1) + '%;background:' + barColor + '"></div></div><span>' + util.toFixed(1) + "%</span></div></td>" +
          "<td>" + fmtInt(used) + " / " + fmtInt(cap) + "</td>" +
          "<td>" + fmtInt(b.inflight_requests) + "</td>" +
          "<td>" + fmtInt(routed) + "</td>" +
          '<td style="color:var(--emerald)">' + hitPct + "</td>" +
          "</tr>";
      }).join("");
      wrap.innerHTML =
        "<table><thead><tr>" +
        "<th>Replica</th><th>Endpoint</th><th>KV Utilization</th><th>KV Tokens</th><th>In-Flight</th><th>Routed</th><th>Prefix Hit %</th>" +
        "</tr></thead><tbody>" + rows + "</tbody></table>";
      return;
    }

    const derived = deriveBackendsFromRecent(recent);
    if (derived.length <= 1) {
      countEl.textContent = "single-upstream mode";
      wrap.innerHTML = '<div class="empty">Multi-backend mode inactive — launch with --backends URL1,URL2 to view replica KV utilization.</div>';
      return;
    }
    countEl.textContent = derived.length + " replicas observed in recent traffic";
    const rows = derived.map(function(b) {
      const tokPct = b.total_tokens > 0 ? (b.matched_tokens / b.total_tokens) * 100 : 0;
      return "<tr>" +
        "<td><strong>" + esc(b.id) + "</strong></td>" +
        "<td>" + fmtInt(b.routed_requests) + "</td>" +
        "<td>" + fmtInt(b.prefix_hits) + "</td>" +
        '<td><div class="kv-bar"><div class="kv-track"><div class="kv-fill" style="width:' + tokPct.toFixed(1) + '%;background:#10b981"></div></div><span>' + tokPct.toFixed(1) + "%</span></div></td>" +
        "<td>" + fmtInt(b.matched_tokens) + " / " + fmtInt(b.total_tokens) + "</td>" +
        "</tr>";
    }).join("");
    wrap.innerHTML =
      "<table><thead><tr><th>Replica</th><th>Recent Routed</th><th>Cache Hits</th><th>Prefix Reuse %</th><th>Matched / Total Tokens</th></tr></thead><tbody>" +
      rows + "</tbody></table>";
  }

  function renderRecent(recent) {
    const body = document.getElementById("recentBody");
    const countEl = document.getElementById("recentCount");
    const list = Array.isArray(recent) ? recent : [];
    countEl.textContent = list.length + " captured";
    if (list.length === 0) {
      body.innerHTML = '<tr><td colspan="9" class="empty">No requests recorded yet.</td></tr>';
      return;
    }
    body.innerHTML = list.slice(0, 50).map(function(r) {
      const saved = num(r.saved_usd !== undefined ? r.saved_usd : r.usd_saved, 0);
      const ts = r.ts_ms !== undefined ? r.ts_ms : r.timestamp_ms;
      const pathStr = r.path !== undefined ? r.path : (r.divergence_path || "prefix_ok");
      const waited = num(r.waited_ms, 0);
      const waitCell = waited > 0
        ? '<span class="badge b-info">' + fmtInt(waited) + " ms</span>"
        : '<span style="color:var(--dim)">0 ms</span>';
      const backend = r.backend_id !== undefined && r.backend_id !== null
        ? "#" + esc(r.backend_id)
        : (r.routed_backend ? esc(r.routed_backend) : "default");
      const savedCell = saved > 0
        ? '<span style="color:var(--emerald)">+' + fmtUsd(saved) + "</span>"
        : '<span style="color:var(--dim)">$0.0000</span>';
      return "<tr>" +
        "<td>" + fmtTime(ts) + "</td>" +
        "<td>" + esc(r.model || "--") + "</td>" +
        '<td><span class="badge ' + badgeClass(r.reason) + '">' + esc(r.reason || "unknown") + "</span></td>" +
        "<td>" + fmtInt(r.matched_tokens) + " / " + fmtInt(r.total_tokens) + "</td>" +
        "<td>" + num(r.compile_us, 0).toFixed(1) + " µs</td>" +
        "<td>" + waitCell + "</td>" +
        '<td><span class="path-code" title="' + esc(pathStr) + '">' + esc(pathStr) + "</span></td>" +
        "<td>" + backend + "</td>" +
        "<td>" + savedCell + "</td>" +
        "</tr>";
    }).join("");
  }

  function applyStats(d) {
    const total = num(d.total_requests, 0);
    const hitToks = num(d.cache_hit_tokens !== undefined ? d.cache_hit_tokens : d.cached_tokens_hit, 0);
    const totalToks = num(d.total_input_tokens !== undefined ? d.total_input_tokens : d.total_tokens_processed, 0);
    const hitRate = d.cache_hit_rate_pct !== undefined
      ? num(d.cache_hit_rate_pct, 0)
      : (d.hit_rate_pct !== undefined ? num(d.hit_rate_pct, 0) : (totalToks > 0 ? (hitToks / totalToks) * 100 : 0));

    const savedUsd = num(d.saved_usd !== undefined ? d.saved_usd : d.estimated_usd_saved, 0);
    const wastedUsd = num(d.wasted_usd !== undefined ? d.wasted_usd : d.estimated_usd_wasted, 0);
    const p50 = num(d.compile_p50_us !== undefined ? d.compile_p50_us : d.p50_compile_us, 0);
    const p99 = num(d.compile_p99_us !== undefined ? d.compile_p99_us : d.p99_compile_us, 0);
    const errs = num(d.upstream_errors, 0);

    document.getElementById("errVal").textContent = fmtInt(errs);
    document.getElementById("errSub").textContent = fmtInt(errs) + " upstream err";
    document.getElementById("updatedVal").textContent = new Date().toTimeString().slice(0, 8);

    document.getElementById("hitRateVal").textContent = hitRate.toFixed(1) + "%";
    document.getElementById("hitRateBar").style.width = Math.max(0, Math.min(100, hitRate)).toFixed(1) + "%";
    document.getElementById("tokenRatioSub").textContent =
      fmtInt(hitToks) + " / " + fmtInt(totalToks) + " tokens reused";

    document.getElementById("usdSavedVal").textContent = fmtUsd(savedUsd);
    document.getElementById("usdWastedSub").textContent = "wasted: " + fmtUsd(wastedUsd);

    const mr = d.miss_reasons || {};
    const reqHits = num(mr.cache_hit, 0);
    const reqMisses = Math.max(0, total - reqHits);
    document.getElementById("totalReqVal").textContent = fmtInt(total);
    document.getElementById("hitsMissesSub").textContent = fmtInt(reqHits) + " hits · " + fmtInt(reqMisses) + " misses";

    document.getElementById("latencyVal").textContent = p50.toFixed(1) + " / " + p99.toFixed(1) + " µs";

    const c = d.coalescer || {};
    const followers = num(c.followers_coalesced, 0);
    const warmed = num(c.followers_warmed, 0);
    const bypass = num(c.warm_bypasses, 0);
    const timedOut = num(c.followers_timed_out, 0);
    const totalWait = num(c.total_wait_ms, 0);
    const avgWait = followers > 0 ? (totalWait / followers) : 0;

    document.getElementById("coalescedVal").textContent = fmtInt(followers);
    document.getElementById("coalescedSub").textContent =
      fmtInt(warmed) + " warmed · " + fmtInt(bypass) + " bypass · " + fmtInt(timedOut) + " timeout";
    document.getElementById("cLeaders").textContent = fmtInt(c.leaders);
    document.getElementById("cWarmed").textContent = fmtInt(warmed);
    document.getElementById("cBypass").textContent = fmtInt(bypass);
    document.getElementById("cTimeout").textContent = fmtInt(timedOut);
    document.getElementById("cAvgWait").textContent = avgWait.toFixed(1) + " ms";
    document.getElementById("cTotWait").textContent = fmtInt(totalWait) + " ms";

    const rep = d.repairs || {};
    const hoist = num(rep.volatile_hoisted !== undefined ? rep.volatile_hoisted : d.volatile_spans_hoisted, 0);
    const jsonFix = num(rep.json_keys_reordered !== undefined ? rep.json_keys_reordered : d.json_keys_reordered, 0);
    const toolFix = num(rep.tools_reordered !== undefined ? rep.tools_reordered : d.tools_reordered, 0);
    const swarmFix = num(rep.all_gather_reordered !== undefined ? rep.all_gather_reordered : d.all_gather_reordered, 0);
    document.getElementById("fixesVal").textContent = fmtInt(hoist + jsonFix + toolFix + swarmFix);
    document.getElementById("fixesSub").textContent =
      fmtInt(hoist) + " hoist · " + fmtInt(jsonFix) + " json · " + fmtInt(toolFix) + " tools · " + fmtInt(swarmFix) + " swarm";

    history.push(hitRate);
    if (history.length > MAX_HISTORY) history.shift();
    renderSparkline();
    renderMissBars(mr);
    const recentList = d.recent || d.recent_requests || [];
    renderBackends(d.backends, recentList);
    renderRecent(recentList);
  }

  async function poll() {
    if (paused) return;
    try {
      const res = await fetch("/stats", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      setStatus("live");
      applyStats(data);
    } catch (_err) {
      setStatus("offline");
    }
  }

  function schedule() {
    if (timer) clearInterval(timer);
    timer = setInterval(poll, POLL_MS);
  }

  document.getElementById("toggleBtn").addEventListener("click", function() {
    paused = !paused;
    this.textContent = paused ? "Resume" : "Pause";
    setStatus(paused ? "paused" : "live");
    if (!paused) poll();
  });

  document.getElementById("refreshBtn").addEventListener("click", function() {
    const wasPaused = paused;
    paused = false;
    poll().finally(function() { paused = wasPaused; });
  });

  poll();
  schedule();
})();
</script>
</body>
</html>
"##;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_html_is_zero_cdn_and_complete() {
        let html = render_dashboard_html();
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(
            !html.contains("http://"),
            "dashboard must not reference external http URLs"
        );
        assert!(
            !html.contains("https://"),
            "dashboard must not reference external https CDN URLs"
        );
        assert!(html.contains("/stats"));
        assert!(html.contains("1500"));
        assert!(html.contains("json_key_drift"));
        assert!(html.contains("tool_order_drift"));
        assert!(html.contains("volatile_timestamp_or_id"));
        assert!(html.contains("all_gather_reorder"));
        assert!(html.contains("ttl_expired"));
    }

    #[test]
    fn test_prometheus_metrics_exposition_and_sanitization() {
        let coal = CoalescerStats {
            leaders: 12,
            followers_coalesced: 47,
            followers_warmed: 46,
            followers_timed_out: 1,
            warm_bypasses: 41,
            total_wait_ms: 920,
        };

        let prom = render_prometheus_metrics(100, 410_000, 500_000, 1.2345, 14.2, 48.7, &coal);
        assert!(prom.contains("# TYPE cachepin_requests_total counter"));
        assert!(prom.contains("cachepin_requests_total 100"));
        assert!(prom.contains("cachepin_cache_hit_tokens_total 410000"));
        assert!(prom.contains("cachepin_input_tokens_total 500000"));
        assert!(prom.contains("cachepin_cache_hit_ratio 0.820000"));
        assert!(prom.contains("cachepin_estimated_usd_saved_total 1.234500"));
        assert!(prom.contains("cachepin_compile_latency_microseconds{quantile=\"0.5\"} 14.200"));
        assert!(prom.contains("cachepin_compile_latency_microseconds{quantile=\"0.99\"} 48.700"));
        assert!(prom.contains("cachepin_coalescer_followers_coalesced_total 47"));
        assert!(prom.contains("cachepin_coalescer_warm_bypasses_total 41"));

        // Non-finite floats must be sanitized to 0.0.
        let prom_nan = render_prometheus_metrics(0, 0, 0, f64::INFINITY, f64::NAN, f64::NEG_INFINITY, &coal);
        assert!(prom_nan.contains("cachepin_estimated_usd_saved_total 0.000000"));
        assert!(prom_nan.contains("cachepin_compile_latency_microseconds{quantile=\"0.5\"} 0.000"));
        assert!(prom_nan.contains("cachepin_compile_latency_microseconds{quantile=\"0.99\"} 0.000"));
    }
}
