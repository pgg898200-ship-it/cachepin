// Matched Node.js 22 V8 control for `cachepin` canonicalization + LCP diffing.
// Performs deterministic key sorting, tool ordering, volatile timestamp/UUID/ID
// extraction & hoisting, and byte-level LCP diff on identical multi-agent payloads.

import fs from "node:fs";
import { performance } from "node:perf_hooks";

const ISO_RE = /\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b/g;
const UUID_RE = /\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b/g;
const ID_RE = /\b(?:req|trace|span|run|session|call|agent_run)_[0-9a-zA-Z]{8,}\b/g;

function canonicalizeObj(val) {
  if (Array.isArray(val)) {
    return val.map(canonicalizeObj);
  }
  if (val !== null && typeof val === "object") {
    const out = {};
    const keys = Object.keys(val).sort();
    for (const k of keys) {
      out[k] = canonicalizeObj(val[k]);
    }
    return out;
  }
  return val;
}

function hoistVolatile(text, bindings) {
  let out = text.replaceAll("⟦cp:", "⟦cp:esc:");
  const sub = (kind) => (m) => {
    const idx = bindings.length + 1;
    const anchor = `⟦cp:v${idx}⟧`;
    bindings.push({ anchor, value: m, kind });
    return anchor;
  };
  out = out.replace(ISO_RE, sub("iso_timestamp"));
  out = out.replace(UUID_RE, sub("uuid"));
  out = out.replace(ID_RE, sub("opaque_id"));
  return out;
}

function compileRequestJs(rawStr) {
  const obj = JSON.parse(rawStr);
  if (Array.isArray(obj.tools)) {
    obj.tools.sort((a, b) => {
      const na = a?.name ?? a?.function?.name ?? "";
      const nb = b?.name ?? b?.function?.name ?? "";
      return na < nb ? -1 : na > nb ? 1 : 0;
    });
  }
  const bindings = [];
  if (typeof obj.system === "string") {
    obj.system = hoistVolatile(obj.system, bindings);
  } else if (Array.isArray(obj.system)) {
    for (const blk of obj.system) {
      if (blk && typeof blk.text === "string") {
        blk.text = hoistVolatile(blk.text, bindings);
      }
    }
  }
  const canon = canonicalizeObj(obj);
  return JSON.stringify(canon);
}

function lcpLen(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

function percentile(arr, p) {
  const sorted = [...arr].sort((x, y) => x - y);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))];
}

const payloadPath = process.argv[2];
if (!payloadPath) {
  console.error("Usage: node node_control.mjs <payloads.jsonl>");
  process.exit(1);
}

const lines = fs
  .readFileSync(payloadPath, "utf8")
  .split("\n")
  .filter((l) => l.trim().length > 0);

// Warmup JIT
for (let i = 0; i < 32; i++) {
  compileRequestJs(lines[i % lines.length]);
}

const compileUs = [];
const iters = 256;
for (let i = 0; i < iters; i++) {
  const t0 = performance.now();
  compileRequestJs(lines[i % lines.length]);
  compileUs.push((performance.now() - t0) * 1000.0);
}

// LCP throughput on 256 KiB buffers
const bufSize = 256 * 1024;
const bufA = Buffer.alloc(bufSize, 65);
const bufB = Buffer.alloc(bufSize, 65);
bufB[bufSize - 17] = 66;
let acc = 0;
for (let i = 0; i < 20; i++) acc ^= lcpLen(bufA, bufB);
const lcpIters = 400;
const tLcp = performance.now();
for (let i = 0; i < lcpIters; i++) acc ^= lcpLen(bufA, bufB);
const lcpSecs = Math.max(1e-9, (performance.now() - tLcp) / 1000.0);
const lcpGbS = (lcpIters * 2 * (bufSize - 17)) / lcpSecs / 1e9;

console.log(
  JSON.stringify({
    runtime: `node-${process.version}`,
    compile_p50_us: Number(percentile(compileUs, 0.5).toFixed(2)),
    compile_p99_us: Number(percentile(compileUs, 0.99).toFixed(2)),
    lcp_throughput_gb_s: Number(lcpGbS.toFixed(3)),
    check: acc,
  })
);
