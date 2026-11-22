//! Memory-safe C-ABI FFI surface for `libcachepin.so` / `libcachepin.dylib`.
//!
//! Enables Python (`ctypes`), Go (`cgo`), Node (`ffi-napi`), and C/C++ callers
//! to invoke the `cachepin` prefix compiler, radix miss diagnostics, exact DP
//! breakpoint optimizer, and AVX2 longest-common-prefix kernel natively.
//!
//! Safety guarantees:
//! - Every `extern "C"` entrypoint is guarded with `std::panic::catch_unwind`
//!   so Rust panics never unwind across the C ABI boundary.
//! - Every pointer parameter is checked for null and bounds before dereferencing.
//! - Returned byte buffers use exact-size `Box<[u8]>` layout matching
//!   [`cachepin_string_free`] with zero allocator layout mismatch.

use std::fmt::Write as _;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::simd::longest_common_prefix_len;
use crate::types::{CanonicalMode, ProviderFormat};
use crate::{CachePinEngine, EngineOutput};

/// Static NUL-terminated crate version string for [`cachepin_version`].
static VERSION_CSTR: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Opaque C-ABI handle owning a thread-safe [`CachePinEngine`].
pub struct CachePinHandle {
    engine: Mutex<CachePinEngine>,
}

fn mode_from_u32(mode: u32) -> CanonicalMode {
    match mode {
        0 => CanonicalMode::ObserveOnly,
        1 => CanonicalMode::JsonCanonical,
        2 => CanonicalMode::HoistVolatile,
        _ => CanonicalMode::SwarmAlign,
    }
}

fn provider_from_u32(provider: u32) -> Option<ProviderFormat> {
    match provider {
        1 => Some(ProviderFormat::AnthropicMessages),
        2 => Some(ProviderFormat::OpenAiChat),
        3 => Some(ProviderFormat::OpenAiResponses),
        4 => Some(ProviderFormat::VllmOpenAi),
        5 => Some(ProviderFormat::Generic),
        _ => None,
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[inline]
fn finite_f64(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Safely constructs a byte slice from a raw FFI pointer and length.
/// Returns `None` if `ptr` is null when `len > 0` or if `len > isize::MAX`.
unsafe fn slice_from_ffi<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() || len > (isize::MAX as usize) {
        return None;
    }
    Some(slice::from_raw_parts(ptr, len))
}

/// Transfers ownership of a `Vec<u8>` across FFI using exact `Box<[u8]>` layout.
fn alloc_ffi_bytes(bytes: Vec<u8>, out_len: *mut usize) -> *mut u8 {
    let boxed: Box<[u8]> = bytes.into_boxed_slice();
    let len = boxed.len();
    if !out_len.is_null() {
        // SAFETY: Caller provided a non-null output pointer.
        unsafe { *out_len = len };
    }
    if len == 0 {
        return ptr::null_mut();
    }
    Box::into_raw(boxed) as *mut u8
}

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn serialize_engine_envelope(out: &EngineOutput) -> Vec<u8> {
    let mut s = String::with_capacity(out.optimized_json.len() + 768);
    s.push_str("{\"canonical_json\":");
    let canonical_str = String::from_utf8_lossy(&out.optimized_json);
    write_json_str(&mut s, &canonical_str);

    s.push_str(",\"provider\":");
    write_json_str(&mut s, out.compiled.provider.as_str());

    s.push_str(",\"model\":");
    write_json_str(&mut s, &out.compiled.model);

    let _ = write!(
        s,
        ",\"compile_ns\":{},\"total_est_tokens\":{},\"shared_prefix_hash\":\"{:016x}\",\"full_prefix_hash\":\"{:016x}\",",
        out.compiled.compile_ns,
        out.compiled.total_est_tokens,
        out.compiled.shared_prefix_hash,
        out.compiled.full_prefix_hash,
    );

    let _ = write!(
        s,
        "\"repairs\":{{\"tools_reordered\":{},\"json_keys_reordered\":{},\"all_gather_reordered\":{},\"volatile_spans_hoisted\":{}}},",
        out.compiled.tools_reordered,
        out.compiled.json_keys_reordered,
        out.compiled.all_gather_reordered,
        out.compiled.volatile_spans_hoisted,
    );

    // Diagnostic block
    s.push_str("\"diagnostic\":{\"reason\":");
    write_json_str(&mut s, out.diagnostic.reason.as_str());
    let _ = write!(
        s,
        ",\"matched_blocks\":{},\"total_blocks\":{},\"matched_tokens\":{},\"total_tokens\":{},\"divergence_block\":",
        out.diagnostic.matched_blocks,
        out.diagnostic.total_blocks,
        out.diagnostic.matched_tokens,
        out.diagnostic.total_tokens,
    );
    match out.diagnostic.divergence_block {
        Some(idx) => {
            let _ = write!(s, "{}", idx);
        }
        None => s.push_str("null"),
    }
    s.push_str(",\"divergence_path\":");
    match &out.diagnostic.divergence_path {
        Some(p) => write_json_str(&mut s, p),
        None => s.push_str("null"),
    }
    s.push_str(",\"divergence_byte_offset\":");
    match out.diagnostic.divergence_byte_offset {
        Some(off) => {
            let _ = write!(s, "{}", off);
        }
        None => s.push_str("null"),
    }
    let _ = write!(
        s,
        ",\"wasted_usd_est\":{:.6},\"advice\":",
        finite_f64(out.diagnostic.wasted_usd_est),
    );
    write_json_str(&mut s, &out.diagnostic.advice);
    s.push_str("},");

    // Breakpoint plan block
    s.push_str("\"plan\":{\"breakpoints\":[");
    for (idx, bp) in out.plan.breakpoints.iter().enumerate() {
        if idx > 0 {
            s.push(',');
        }
        let _ = write!(s, "{{\"block_index\":{},\"path\":", bp.block_index);
        write_json_str(&mut s, &bp.path);
        let _ = write!(
            s,
            ",\"cumulative_tokens\":{},\"ttl\":",
            bp.cumulative_tokens
        );
        write_json_str(&mut s, bp.ttl.as_anthropic_str());
        s.push('}');
    }
    let _ = write!(
        s,
        "],\"expected_cost_usd\":{:.6},\"uncached_cost_usd\":{:.6},\"naive_single_bp_cost_usd\":{:.6},\"expected_savings_vs_uncached_pct\":{:.3},\"keepalive_recommended\":{},\"keepalive_interval_secs\":",
        finite_f64(out.plan.expected_cost_usd),
        finite_f64(out.plan.uncached_cost_usd),
        finite_f64(out.plan.naive_single_bp_cost_usd),
        finite_f64(out.plan.expected_savings_vs_uncached_pct),
        out.plan.keepalive_recommended,
    );
    match out.plan.keepalive_interval_secs {
        Some(sec) => {
            let _ = write!(s, "{}", sec);
        }
        None => s.push_str("null"),
    }
    s.push_str("}}");

    s.into_bytes()
}

/// Allocates a new `CachePinHandle` configured with `mode` and radix capacity `max_nodes`.
///
/// `mode` values:
/// - `0`: `ObserveOnly`
/// - `1`: `JsonCanonical`
/// - `2`: `HoistVolatile`
/// - `3` (or any other value): `SwarmAlign`
///
/// If `max_nodes == 0`, defaults to `65_536`.
/// Free with [`cachepin_engine_free`].
#[no_mangle]
pub extern "C" fn cachepin_engine_new(mode: u32, max_nodes: usize) -> *mut CachePinHandle {
    match catch_unwind(AssertUnwindSafe(|| {
        let cap = if max_nodes == 0 {
            65_536
        } else {
            max_nodes.max(64)
        };
        let engine = CachePinEngine::new(mode_from_u32(mode), cap);
        Box::into_raw(Box::new(CachePinHandle {
            engine: Mutex::new(engine),
        }))
    })) {
        Ok(ptr) => ptr,
        Err(_) => ptr::null_mut(),
    }
}

/// Frees a [`CachePinHandle`] previously returned by [`cachepin_engine_new`].
///
/// Passing `NULL` is a safe no-op.
#[no_mangle]
pub extern "C" fn cachepin_engine_free(handle: *mut CachePinHandle) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: `handle` is non-null and was allocated via `Box::into_raw` in `cachepin_engine_new`.
        let _ = unsafe { Box::from_raw(handle) };
    }));
}

/// Canonicalizes an LLM JSON request payload, diagnoses prefix misses against the
/// handle's tenant-isolated radix tree, and solves exact O(M^2 K) mixed-TTL breakpoints.
///
/// `provider` values:
/// - `0`: auto-detect from JSON structure
/// - `1`: `AnthropicMessages`
/// - `2`: `OpenAiChat`
/// - `3`: `OpenAiResponses`
/// - `4`: `VllmOpenAi`
/// - `5`: `Generic`
///
/// Returns a heap-allocated UTF-8 JSON envelope:
/// `{"canonical_json":..., "diagnostic":..., "plan":..., "compile_ns":...}`
/// and writes its byte length into `*out_len`. Free the returned buffer using
/// [`cachepin_string_free`]. Returns `NULL` on null/invalid input or parse failure.
#[no_mangle]
pub extern "C" fn cachepin_compile_and_plan(
    handle: *mut CachePinHandle,
    tenant_ptr: *const u8,
    tenant_len: usize,
    json_ptr: *const u8,
    json_len: usize,
    provider: u32,
    out_len: *mut usize,
) -> *mut u8 {
    match catch_unwind(AssertUnwindSafe(|| {
        if !out_len.is_null() {
            // SAFETY: Non-null output pointer initialized defensively before any early return.
            unsafe { *out_len = 0 };
        }
        if handle.is_null() || json_ptr.is_null() || json_len == 0 {
            return ptr::null_mut();
        }

        // SAFETY: `json_ptr` and `tenant_ptr` are checked inside `slice_from_ffi`.
        let json_bytes = match unsafe { slice_from_ffi(json_ptr, json_len) } {
            Some(s) if !s.is_empty() => s,
            _ => return ptr::null_mut(),
        };

        let tenant_str = if tenant_ptr.is_null() || tenant_len == 0 {
            "default".to_string()
        } else {
            match unsafe { slice_from_ffi(tenant_ptr, tenant_len) } {
                Some(tb) => String::from_utf8_lossy(tb).into_owned(),
                None => return ptr::null_mut(),
            }
        };

        // SAFETY: `handle` is non-null and points to a live `CachePinHandle`.
        let handle_ref = unsafe { &*handle };
        let mut guard = handle_ref
            .engine
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let provider_hint = provider_from_u32(provider);
        let result = match guard.process_request(
            &tenant_str,
            json_bytes,
            provider_hint,
            now_ms(),
            true,
        ) {
            Ok(out) => out,
            Err(_) => return ptr::null_mut(),
        };

        let envelope = serialize_engine_envelope(&result);
        alloc_ffi_bytes(envelope, out_len)
    })) {
        Ok(p) => p,
        Err(_) => {
            if !out_len.is_null() {
                unsafe { *out_len = 0 };
            }
            ptr::null_mut()
        }
    }
}

/// Computes the byte length of the longest common prefix of two byte slices
/// using runtime-dispatched AVX2 SIMD (or 64-bit word-wise fallback).
///
/// Passing null pointers or zero lengths safely returns `0`.
#[no_mangle]
pub extern "C" fn cachepin_lcp_len(
    a_ptr: *const u8,
    a_len: usize,
    b_ptr: *const u8,
    b_len: usize,
) -> usize {
    match catch_unwind(AssertUnwindSafe(|| {
        if a_ptr.is_null() || b_ptr.is_null() || a_len == 0 || b_len == 0 {
            return 0;
        }
        // SAFETY: Checked non-null and length bounded inside `slice_from_ffi`.
        let (a, b) = match (
            unsafe { slice_from_ffi(a_ptr, a_len) },
            unsafe { slice_from_ffi(b_ptr, b_len) },
        ) {
            (Some(a), Some(b)) => (a, b),
            _ => return 0,
        };
        longest_common_prefix_len(a, b)
    })) {
        Ok(n) => n,
        Err(_) => 0,
    }
}

/// Frees a byte buffer previously returned by [`cachepin_compile_and_plan`].
///
/// Passing `NULL` or `len == 0` is a safe no-op.
#[no_mangle]
pub extern "C" fn cachepin_string_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Buffer was allocated as `Box<[u8]>` of exact length `len` in `alloc_ffi_bytes`.
        let slice_ptr = ptr::slice_from_raw_parts_mut(ptr, len);
        let _ = unsafe { Box::from_raw(slice_ptr) };
    }));
}

/// Returns a static NUL-terminated UTF-8 string containing the `cachepin` version.
#[no_mangle]
pub extern "C" fn cachepin_version() -> *const c_char {
    VERSION_CSTR.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_version_cstr_matches_crate_version() {
        let ptr = cachepin_version();
        assert!(!ptr.is_null());
        // SAFETY: `VERSION_CSTR` is a static NUL-terminated string.
        let s = unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .expect("version must be valid UTF-8");
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_lcp_len_and_null_safety() {
        let a = b"prefix_shared_1234567890_abcdef_TAIL_A";
        let b = b"prefix_shared_1234567890_abcdef_TAIL_B";
        assert_eq!(
            cachepin_lcp_len(a.as_ptr(), a.len(), b.as_ptr(), b.len()),
            a.len() - 1
        );
        // Null and zero-length edge cases must never crash.
        assert_eq!(cachepin_lcp_len(ptr::null(), 10, b.as_ptr(), b.len()), 0);
        assert_eq!(cachepin_lcp_len(a.as_ptr(), a.len(), ptr::null(), 10), 0);
        assert_eq!(cachepin_lcp_len(ptr::null(), 0, ptr::null(), 0), 0);
        assert_eq!(cachepin_lcp_len(a.as_ptr(), 0, b.as_ptr(), b.len()), 0);
    }

    #[test]
    fn test_null_pointer_guards_on_lifecycle_and_compile() {
        cachepin_engine_free(ptr::null_mut());
        cachepin_string_free(ptr::null_mut(), 0);
        cachepin_string_free(ptr::null_mut(), 42);

        let mut out_len: usize = 999;
        let res = cachepin_compile_and_plan(
            ptr::null_mut(),
            ptr::null(),
            0,
            ptr::null(),
            0,
            0,
            &mut out_len,
        );
        assert!(res.is_null());
        assert_eq!(out_len, 0);

        let handle = cachepin_engine_new(3, 1024);
        assert!(!handle.is_null());

        // Malformed JSON must fail cleanly with NULL and out_len == 0.
        let bad = b"{not valid json";
        out_len = 123;
        let res_bad = cachepin_compile_and_plan(
            handle,
            b"tenant-a".as_ptr(),
            8,
            bad.as_ptr(),
            bad.len(),
            1,
            &mut out_len,
        );
        assert!(res_bad.is_null());
        assert_eq!(out_len, 0);

        // Null json_ptr must return NULL.
        let res_null_json = cachepin_compile_and_plan(
            handle,
            b"tenant-a".as_ptr(),
            8,
            ptr::null(),
            10,
            1,
            &mut out_len,
        );
        assert!(res_null_json.is_null());
        assert_eq!(out_len, 0);

        cachepin_engine_free(handle);
    }

    #[test]
    fn test_compile_and_plan_roundtrip_lifecycle() {
        let handle = cachepin_engine_new(3, 4096);
        assert!(!handle.is_null());

        let system_pad = "Architecture invariant specification block. ".repeat(80);
        let payload = format!(
            "{{\"model\":\"claude-opus-5-5\",\"max_tokens\":512,\"system\":[{{\"type\":\"text\",\"text\":\"Timestamp 2026-11-22T15:00:00Z {}\"}}],\"messages\":[{{\"role\":\"user\",\"content\":\"Check invariants\"}}]}}",
            system_pad
        );
        let tenant = b"org-alpha";

        // First call (cold start)
        let mut out_len1: usize = 0;
        let ptr1 = cachepin_compile_and_plan(
            handle,
            tenant.as_ptr(),
            tenant.len(),
            payload.as_bytes().as_ptr(),
            payload.len(),
            1,
            &mut out_len1,
        );
        assert!(!ptr1.is_null());
        assert!(out_len1 > 0);
        let env1 = unsafe { slice::from_raw_parts(ptr1, out_len1) };
        let env1_str = std::str::from_utf8(env1).expect("envelope must be valid UTF-8");
        assert!(env1_str.contains("\"canonical_json\":"));
        assert!(env1_str.contains("\"diagnostic\":"));
        assert!(env1_str.contains("\"plan\":"));
        assert!(env1_str.contains("\"compile_ns\":"));
        cachepin_string_free(ptr1, out_len1);

        // Second call with different volatile timestamp should hit the cached prefix!
        let payload2 = format!(
            "{{\"model\":\"claude-opus-5-5\",\"max_tokens\":512,\"system\":[{{\"type\":\"text\",\"text\":\"Timestamp 2026-11-22T15:01:30Z {}\"}}],\"messages\":[{{\"role\":\"user\",\"content\":\"Check invariants\"}}]}}",
            system_pad
        );
        let mut out_len2: usize = 0;
        let ptr2 = cachepin_compile_and_plan(
            handle,
            tenant.as_ptr(),
            tenant.len(),
            payload2.as_bytes().as_ptr(),
            payload2.len(),
            1,
            &mut out_len2,
        );
        assert!(!ptr2.is_null());
        assert!(out_len2 > 0);
        let env2 = unsafe { slice::from_raw_parts(ptr2, out_len2) };
        let env2_str = std::str::from_utf8(env2).expect("envelope must be valid UTF-8");
        assert!(env2_str.contains("\"diagnostic\":"));
        assert!(env2_str.contains("\"plan\":"));
        cachepin_string_free(ptr2, out_len2);

        cachepin_engine_free(handle);
    }
}
