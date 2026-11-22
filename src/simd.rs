//! Zero-dependency SIMD acceleration, fast token estimation, and SipHash
//! tenant-isolated cryptographic hashing for `cachepin`.
//!
//! Uses runtime AVX2 dispatch on x86/x86_64, native NEON on little-endian
//! ARM64 (`aarch64`), and portable 64-bit word-wise fallbacks on all other
//! architectures.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Returns the active SIMD backend identifier (`"avx2"`, `"neon"`, or `"scalar-u64"`).
#[inline]
pub fn simd_backend_name() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        return "neon";
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    "scalar-u64"
}

/// Computes the byte length of the longest common prefix of `a` and `b`.
///
/// Dispatches to a 32-byte/iteration AVX2 kernel on x86/x86_64, a 16-byte/iteration
/// NEON kernel on little-endian ARM64, or an 8-byte `u64` word-wise kernel otherwise.
#[inline]
pub fn longest_common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature detection verified AVX2 is available.
            return unsafe { longest_common_prefix_avx2(a, b) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        // SAFETY: NEON is guaranteed by the AArch64 ABI.
        return unsafe { longest_common_prefix_neon(a, b) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    longest_common_prefix_scalar(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn longest_common_prefix_avx2(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: `i + 32 <= len <= a.len()` and `i + 32 <= len <= b.len()`,
        // and `_mm256_loadu_si256` handles arbitrary alignment.
        let va = _mm256_loadu_si256(pa.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(pb.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(va, vb);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != u32::MAX {
            return i + (!mask).trailing_zeros() as usize;
        }
        i += 32;
    }

    i + longest_common_prefix_scalar(&a[i..len], &b[i..len])
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn longest_common_prefix_neon(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;

    while i + 16 <= len {
        // SAFETY: `i + 16 <= len <= a.len()` and `i + 16 <= len <= b.len()`.
        let va = vld1q_u8(pa.add(i));
        let vb = vld1q_u8(pb.add(i));
        let eq = vceqq_u8(va, vb);
        if vminvq_u8(eq) != 0xFF {
            let neq = vmvnq_u8(eq);
            let u64s = vreinterpretq_u64_u8(neq);
            let lo = vgetq_lane_u64::<0>(u64s);
            if lo != 0 {
                return i + (lo.trailing_zeros() as usize / 8);
            }
            let hi = vgetq_lane_u64::<1>(u64s);
            return i + 8 + (hi.trailing_zeros() as usize / 8);
        }
        i += 16;
    }

    i + longest_common_prefix_scalar(&a[i..len], &b[i..len])
}

/// Portable 8-byte word-wise longest common prefix kernel.
#[inline]
fn longest_common_prefix_scalar(a: &[u8], b: &[u8]) -> usize {
    let len = a.len().min(b.len());
    let mut i = 0usize;

    while i + 8 <= len {
        let wa = u64::from_le_bytes([
            a[i],
            a[i + 1],
            a[i + 2],
            a[i + 3],
            a[i + 4],
            a[i + 5],
            a[i + 6],
            a[i + 7],
        ]);
        let wb = u64::from_le_bytes([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ]);
        let diff = wa ^ wb;
        if diff != 0 {
            return i + (diff.trailing_zeros() as usize / 8);
        }
        i += 8;
    }

    while i < len && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Counts occurrences of `needle` inside `buf`.
///
/// Dispatches to AVX2 on x86/x86_64, NEON on little-endian ARM64, or a
/// carry-free SWAR `u64` byte counter otherwise.
#[inline]
pub fn count_byte_simd(buf: &[u8], needle: u8) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            // SAFETY: Runtime feature detection verified AVX2 and POPCNT are available.
            return unsafe { count_byte_avx2(buf, needle) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        // SAFETY: NEON is guaranteed by the AArch64 ABI.
        return unsafe { count_byte_neon(buf, needle) };
    }
    #[cfg(not(all(target_arch = "aarch64", target_endian = "little")))]
    count_byte_scalar(buf, needle)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn count_byte_avx2(buf: &[u8], needle: u8) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let vneedle = _mm256_set1_epi8(needle as i8);
    let mut count = 0usize;
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: `i + 32 <= len` and `_mm256_loadu_si256` permits unaligned loads.
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk, vneedle);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        count += mask.count_ones() as usize;
        i += 32;
    }

    count + count_byte_scalar(&buf[i..], needle)
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn count_byte_neon(buf: &[u8], needle: u8) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let vneedle = vdupq_n_u8(needle);
    let mut count = 0usize;
    let mut i = 0usize;

    while i + 16 <= len {
        let mut acc = vdupq_n_u8(0);
        let mut iters = 0usize;
        while iters < 255 && i + 16 <= len {
            let chunk = vld1q_u8(ptr.add(i));
            let cmp = vceqq_u8(chunk, vneedle);
            // Matching lanes hold 0xFF (-1i8), so subtracting adds +1 per lane without overflow.
            acc = vsubq_u8(acc, cmp);
            i += 16;
            iters += 1;
        }
        count += vaddlvq_u8(acc) as usize;
    }

    count + count_byte_scalar(&buf[i..], needle)
}

/// Exact carry-free SWAR 8-byte counter for `needle` in `buf`.
#[inline]
fn count_byte_scalar(buf: &[u8], needle: u8) -> usize {
    const LO_7F: u64 = 0x7F7F_7F7F_7F7F_7F7F;
    let broadcast = u64::from_ne_bytes([needle; 8]);
    let len = buf.len();
    let mut count = 0usize;
    let mut i = 0usize;

    while i + 8 <= len {
        let word = u64::from_ne_bytes([
            buf[i],
            buf[i + 1],
            buf[i + 2],
            buf[i + 3],
            buf[i + 4],
            buf[i + 5],
            buf[i + 6],
            buf[i + 7],
        ]);
        let x = word ^ broadcast;
        // For every byte lane independently (zero inter-byte carry because
        // `(b & 0x7F) + 0x7F <= 0xFE`): bit 7 of `m` is 0 iff `b == 0`.
        let m = ((x & LO_7F) + LO_7F) | x | LO_7F;
        count += (!m).count_ones() as usize;
        i += 8;
    }

    while i < len {
        if buf[i] == needle {
            count += 1;
        }
        i += 1;
    }
    count
}

/// Returns the byte offset of the first JSON string-special byte
/// (`b < 0x20 || b == b'"' || b == b'\\'`), or `buf.len()` if none exists.
///
/// Dispatches to a 32-byte/iteration AVX2 kernel on x86/x86_64 or a
/// 16-byte/iteration NEON kernel on little-endian ARM64.
#[inline]
pub fn find_json_special_byte(buf: &[u8]) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if buf.len() >= 32 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support verified at runtime above.
            return unsafe { find_json_special_byte_avx2(buf) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if buf.len() >= 16 {
            // SAFETY: NEON is guaranteed by the AArch64 ABI.
            return unsafe { find_json_special_byte_neon(buf) };
        }
    }
    find_json_special_byte_scalar(buf)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn find_json_special_byte_avx2(buf: &[u8]) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v_ctrl_max = _mm256_set1_epi8(0x1F);
    let v_quote = _mm256_set1_epi8(b'"' as i8);
    let v_bs = _mm256_set1_epi8(b'\\' as i8);
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: `i + 32 <= len`; `_mm256_loadu_si256` permits unaligned loads.
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        // Unsigned min(chunk, 0x1F) == chunk iff byte is in 0x00..=0x1F (safe for UTF-8 >= 0x80).
        let is_ctrl = _mm256_cmpeq_epi8(_mm256_min_epu8(chunk, v_ctrl_max), chunk);
        let is_q = _mm256_cmpeq_epi8(chunk, v_quote);
        let is_bs = _mm256_cmpeq_epi8(chunk, v_bs);
        let any = _mm256_or_si256(is_ctrl, _mm256_or_si256(is_q, is_bs));
        let mask = _mm256_movemask_epi8(any) as u32;
        if mask != 0 {
            return i + (mask.trailing_zeros() as usize);
        }
        i += 32;
    }

    i + find_json_special_byte_scalar(&buf[i..])
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn find_json_special_byte_neon(buf: &[u8]) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v_ctrl_max = vdupq_n_u8(0x1F);
    let v_quote = vdupq_n_u8(b'"');
    let v_bs = vdupq_n_u8(b'\\');
    let mut i = 0usize;

    while i + 16 <= len {
        let chunk = vld1q_u8(ptr.add(i));
        // Unsigned `<= 0x1F` (safe for UTF-8 bytes >= 0x80).
        let is_ctrl = vcleq_u8(chunk, v_ctrl_max);
        let is_q = vceqq_u8(chunk, v_quote);
        let is_bs = vceqq_u8(chunk, v_bs);
        let any = vorrq_u8(is_ctrl, vorrq_u8(is_q, is_bs));
        if vmaxvq_u8(any) != 0 {
            let u64s = vreinterpretq_u64_u8(any);
            let lo = vgetq_lane_u64::<0>(u64s);
            if lo != 0 {
                return i + (lo.trailing_zeros() as usize / 8);
            }
            let hi = vgetq_lane_u64::<1>(u64s);
            return i + 8 + (hi.trailing_zeros() as usize / 8);
        }
        i += 16;
    }

    i + find_json_special_byte_scalar(&buf[i..])
}

#[inline(always)]
fn find_json_special_byte_scalar(buf: &[u8]) -> usize {
    let len = buf.len();
    let mut i = 0usize;
    while i < len {
        let b = buf[i];
        if b < 0x20 || b == b'"' || b == b'\\' {
            return i;
        }
        i += 1;
    }
    len
}

/// Returns the byte offset of the first occurrence of `b1`, `b2`, or `b3` in `buf`,
/// or `buf.len()` if none exists.
///
/// Dispatches to a 32-byte/iteration AVX2 kernel on x86/x86_64, a 16-byte/iteration
/// NEON kernel on little-endian ARM64, or a tight scalar loop otherwise.
#[inline]
pub fn find_any3_byte(buf: &[u8], b1: u8, b2: u8, b3: u8) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if buf.len() >= 32 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 support verified at runtime above.
            return unsafe { find_any3_byte_avx2(buf, b1, b2, b3) };
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if buf.len() >= 16 {
            // SAFETY: NEON is guaranteed by the AArch64 ABI.
            return unsafe { find_any3_byte_neon(buf, b1, b2, b3) };
        }
    }
    find_any3_byte_scalar(buf, b1, b2, b3)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn find_any3_byte_avx2(buf: &[u8], b1: u8, b2: u8, b3: u8) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v1 = _mm256_set1_epi8(b1 as i8);
    let v2 = _mm256_set1_epi8(b2 as i8);
    let v3 = _mm256_set1_epi8(b3 as i8);
    let mut i = 0usize;

    while i + 32 <= len {
        // SAFETY: `i + 32 <= len`; `_mm256_loadu_si256` permits unaligned loads.
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let m1 = _mm256_cmpeq_epi8(chunk, v1);
        let m2 = _mm256_cmpeq_epi8(chunk, v2);
        let m3 = _mm256_cmpeq_epi8(chunk, v3);
        let any = _mm256_or_si256(m1, _mm256_or_si256(m2, m3));
        let mask = _mm256_movemask_epi8(any) as u32;
        if mask != 0 {
            return i + (mask.trailing_zeros() as usize);
        }
        i += 32;
    }

    i + find_any3_byte_scalar(&buf[i..], b1, b2, b3)
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn find_any3_byte_neon(buf: &[u8], b1: u8, b2: u8, b3: u8) -> usize {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v1 = vdupq_n_u8(b1);
    let v2 = vdupq_n_u8(b2);
    let v3 = vdupq_n_u8(b3);
    let mut i = 0usize;

    while i + 16 <= len {
        let chunk = vld1q_u8(ptr.add(i));
        let m1 = vceqq_u8(chunk, v1);
        let m2 = vceqq_u8(chunk, v2);
        let m3 = vceqq_u8(chunk, v3);
        let any = vorrq_u8(m1, vorrq_u8(m2, m3));
        if vmaxvq_u8(any) != 0 {
            let u64s = vreinterpretq_u64_u8(any);
            let lo = vgetq_lane_u64::<0>(u64s);
            if lo != 0 {
                return i + (lo.trailing_zeros() as usize / 8);
            }
            let hi = vgetq_lane_u64::<1>(u64s);
            return i + 8 + (hi.trailing_zeros() as usize / 8);
        }
        i += 16;
    }

    i + find_any3_byte_scalar(&buf[i..], b1, b2, b3)
}

#[inline(always)]
fn find_any3_byte_scalar(buf: &[u8], b1: u8, b2: u8, b3: u8) -> usize {
    let len = buf.len();
    let mut i = 0usize;
    while i < len {
        let b = buf[i];
        if b == b1 || b == b2 || b == b3 {
            return i;
        }
        i += 1;
    }
    len
}

/// Visitor invoked by [`scan_any3_byte`] for every match of `b1`, `b2`, or `b3`.
///
/// Using a trait method allows implementors to mark `on_hit` as `#[inline(always)]`
/// so fast-reject checks inline directly inside the AVX2/NEON vector loop with
/// zero per-hit SIMD transitions.
pub(crate) trait Any3Visitor {
    fn on_hit(&mut self, rel: usize) -> usize;
}

/// Single-pass SIMD scanner invoking `visitor.on_hit(byte_index)` for every
/// occurrence of `b1`, `b2`, or `b3` in `buf`, loading each vector chunk once
/// with a single SIMD transition per buffer.
///
/// `on_hit(hit)` returns the next byte index to continue scanning from
/// (returning `hit + 1` to continue normally, `end` to jump past a matched span,
/// or `usize::MAX` to terminate scanning early).
#[inline]
pub(crate) fn scan_any3_byte<V: Any3Visitor>(
    buf: &[u8],
    b1: u8,
    b2: u8,
    b3: u8,
    visitor: &mut V,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if buf.len() >= 32 && std::is_x86_feature_detected!("avx2") {
            // SAFETY: Runtime feature detection verified AVX2 is available.
            unsafe { scan_any3_byte_avx2(buf, b1, b2, b3, visitor) };
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    {
        if buf.len() >= 16 {
            // SAFETY: NEON is guaranteed by the AArch64 ABI.
            unsafe { scan_any3_byte_neon(buf, b1, b2, b3, visitor) };
            return;
        }
    }
    scan_any3_byte_scalar_from(buf, 0, b1, b2, b3, visitor);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scan_any3_byte_avx2<V: Any3Visitor>(
    buf: &[u8],
    b1: u8,
    b2: u8,
    b3: u8,
    visitor: &mut V,
) {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v1 = _mm256_set1_epi8(b1 as i8);
    let v2 = _mm256_set1_epi8(b2 as i8);
    let v3 = _mm256_set1_epi8(b3 as i8);
    let mut i = 0usize;

    'chunks: while i + 32 <= len {
        // SAFETY: `i + 32 <= len`; `_mm256_loadu_si256` permits unaligned loads.
        let chunk = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let m1 = _mm256_cmpeq_epi8(chunk, v1);
        let m2 = _mm256_cmpeq_epi8(chunk, v2);
        let m3 = _mm256_cmpeq_epi8(chunk, v3);
        let any = _mm256_or_si256(m1, _mm256_or_si256(m2, m3));
        let mut mask = _mm256_movemask_epi8(any) as u32;
        while mask != 0 {
            let bit = mask.trailing_zeros() as usize;
            let hit = i + bit;
            let next = visitor.on_hit(hit);
            if next == usize::MAX {
                return;
            }
            if next <= hit + 1 {
                mask &= mask - 1;
            } else if next < i + 32 {
                let shift = next - i;
                mask &= !((1u32 << shift) - 1);
            } else {
                i = next;
                continue 'chunks;
            }
        }
        i += 32;
    }

    if i < len {
        scan_any3_byte_scalar_from(buf, i, b1, b2, b3, visitor);
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
unsafe fn scan_any3_byte_neon<V: Any3Visitor>(
    buf: &[u8],
    b1: u8,
    b2: u8,
    b3: u8,
    visitor: &mut V,
) {
    let len = buf.len();
    let ptr = buf.as_ptr();
    let v1 = vdupq_n_u8(b1);
    let v2 = vdupq_n_u8(b2);
    let v3 = vdupq_n_u8(b3);
    let mut i = 0usize;

    'chunks: while i + 16 <= len {
        let chunk = vld1q_u8(ptr.add(i));
        let m1 = vceqq_u8(chunk, v1);
        let m2 = vceqq_u8(chunk, v2);
        let m3 = vceqq_u8(chunk, v3);
        let any = vorrq_u8(m1, vorrq_u8(m2, m3));
        if vmaxvq_u8(any) != 0 {
            let u64s = vreinterpretq_u64_u8(any);
            let mut lo = vgetq_lane_u64::<0>(u64s);
            let mut hi = vgetq_lane_u64::<1>(u64s);
            while lo != 0 {
                let lane = (lo.trailing_zeros() as usize) / 8;
                let hit = i + lane;
                let next = visitor.on_hit(hit);
                if next == usize::MAX {
                    return;
                }
                if next >= i + 16 {
                    i = next;
                    continue 'chunks;
                }
                let advance = next.max(hit + 1);
                if advance >= i + 8 {
                    hi &= !0u64 << ((advance - (i + 8)) * 8);
                    break;
                }
                lo &= !0u64 << ((advance - i) * 8);
            }
            while hi != 0 {
                let lane = (hi.trailing_zeros() as usize) / 8;
                let hit = i + 8 + lane;
                let next = visitor.on_hit(hit);
                if next == usize::MAX {
                    return;
                }
                if next >= i + 16 {
                    i = next;
                    continue 'chunks;
                }
                let advance = next.max(hit + 1);
                if advance >= i + 16 {
                    break;
                }
                hi &= !0u64 << ((advance - (i + 8)) * 8);
            }
        }
        i += 16;
    }

    if i < len {
        scan_any3_byte_scalar_from(buf, i, b1, b2, b3, visitor);
    }
}

#[inline(always)]
fn scan_any3_byte_scalar_from<V: Any3Visitor>(
    buf: &[u8],
    mut i: usize,
    b1: u8,
    b2: u8,
    b3: u8,
    visitor: &mut V,
) {
    let len = buf.len();
    while i < len {
        let b = buf[i];
        if b == b1 || b == b2 || b == b3 {
            let next = visitor.on_hit(i);
            if next == usize::MAX {
                return;
            }
            i = next.max(i + 1);
        } else {
            i += 1;
        }
    }
}

const CLS_NONE: u8 = 0;
const CLS_ALPHA: u8 = 1;
const CLS_DIGIT: u8 = 2;
const CLS_PUNCT: u8 = 3;
const CLS_SPACE: u8 = 4;
const CLS_NEWLINE: u8 = 5;
const CLS_UTF8_CONT: u8 = 6;
const CLS_UTF8_2: u8 = 7;
const CLS_UTF8_3: u8 = 8;
const CLS_UTF8_4: u8 = 9;

const fn build_token_byte_class() -> [u8; 256] {
    let mut t = [CLS_PUNCT; 256];
    let mut i = 0usize;
    while i < 256 {
        let b = i as u8;
        t[i] = if b == b'\n' {
            CLS_NEWLINE
        } else if (b >= b'a' && b <= b'z') || (b >= b'A' && b <= b'Z') {
            CLS_ALPHA
        } else if b >= b'0' && b <= b'9' {
            CLS_DIGIT
        } else if b == b' ' || b == b'\t' || b == b'\r' {
            CLS_SPACE
        } else if b < 0x80 {
            CLS_PUNCT
        } else if b <= 0xBF {
            CLS_UTF8_CONT
        } else if b <= 0xDF {
            CLS_UTF8_2
        } else if b <= 0xEF {
            CLS_UTF8_3
        } else if b <= 0xF7 {
            CLS_UTF8_4
        } else {
            CLS_UTF8_2
        };
        i += 1;
    }
    t
}

static TOKEN_BYTE_CLASS: [u8; 256] = build_token_byte_class();

#[inline(always)]
fn flush_run(class: u8, run_len: usize) -> u64 {
    if run_len == 0 {
        return 0;
    }
    // Fixed-point 1/8-token units calibrated against cl100k/o200k/Claude BPE.
    match class {
        CLS_ALPHA => {
            if run_len == 1 {
                6
            } else if run_len <= 6 {
                8
            } else {
                8 + (((run_len - 6) as u64) * 3) / 2
            }
        }
        CLS_DIGIT => (((run_len + 2) / 3) as u64) * 8,
        CLS_PUNCT => 4 + (run_len as u64) * 2,
        CLS_SPACE => (((run_len - 1) / 4) as u64) * 8,
        _ => 0,
    }
}

#[inline]
fn estimate_tokens_units_scalar(buf: &[u8]) -> u64 {
    let mut units: u64 = 0;
    let mut class = CLS_NONE;
    let mut run_len: usize = 0;

    for &b in buf {
        let next = TOKEN_BYTE_CLASS[b as usize];
        if next == class {
            run_len += 1;
            continue;
        }
        if next <= CLS_SPACE {
            units += flush_run(class, run_len);
            class = next;
            run_len = 1;
        } else {
            units += flush_run(class, run_len);
            class = CLS_NONE;
            run_len = 0;
            units += match next {
                CLS_NEWLINE => 6,
                CLS_UTF8_CONT => 0,
                CLS_UTF8_2 => 8,
                CLS_UTF8_3 => 10,
                CLS_UTF8_4 => 16,
                _ => 8,
            };
        }
    }

    units + flush_run(class, run_len)
}

#[inline]
fn estimate_tokens_scalar(buf: &[u8]) -> u32 {
    if buf.is_empty() {
        return 0;
    }
    let units = estimate_tokens_units_scalar(buf);
    let est = (units + 4) / 8;
    let clamped = est.max(1);
    u32::try_from(clamped).unwrap_or(u32::MAX)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn estimate_tokens_avx2(buf: &[u8]) -> u32 {
    let len = buf.len();
    let ptr = buf.as_ptr();

    // Signed-range bias constants for pure ASCII (0x00..=0x7F):
    // - Alpha: `lower = b | 0x20`; `lower.wrapping_add(0x1F) as i8 < -102` iff `b'a'..=b'z'`
    // - Digit: `b.wrapping_add(0x50) as i8 < -118` iff `b'0'..=b'9'`
    // - Space: `b == 0x20 (' ') || b == 0x09 ('\t') || b == 0x0D ('\r')`
    // - Newline: `b == 0x0A ('\n')`
    let v_0x20 = _mm256_set1_epi8(0x20);
    let v_alpha_bias = _mm256_set1_epi8(0x1F);
    let v_alpha_lim = _mm256_set1_epi8(-102);
    let v_digit_bias = _mm256_set1_epi8(0x50);
    let v_digit_lim = _mm256_set1_epi8(-118);
    let v_tab = _mm256_set1_epi8(b'\t' as i8);
    let v_cr = _mm256_set1_epi8(b'\r' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);

    // Accumulate in 1/16-token fixed-point units (`2 * scalar_units`) so every
    // weight (including alpha excess `3/2` -> `3`) folds into a single register.
    let mut units2: u64 = 0;

    // Cross-chunk carry state across 64-byte iterations (minimal register footprint).
    let mut prev_alpha6: u64 = 0;
    let mut prev_space4: u64 = 0;
    let mut prev_punct: u64 = 0;
    let mut prev_digit3: u64 = 0;
    let mut prev_d4: u64 = 0;
    let mut pending_single_alpha: u64 = 0;

    macro_rules! classify_32 {
        ($c:expr) => {{
            let lower = _mm256_or_si256($c, v_0x20);
            let is_alpha = _mm256_cmpgt_epi8(v_alpha_lim, _mm256_add_epi8(lower, v_alpha_bias));
            let is_digit = _mm256_cmpgt_epi8(v_digit_lim, _mm256_add_epi8($c, v_digit_bias));
            let is_space = _mm256_or_si256(
                _mm256_cmpeq_epi8($c, v_0x20),
                _mm256_or_si256(_mm256_cmpeq_epi8($c, v_tab), _mm256_cmpeq_epi8($c, v_cr)),
            );
            let is_nl = _mm256_cmpeq_epi8($c, v_nl);
            (
                _mm256_movemask_epi8(is_alpha) as u32 as u64,
                _mm256_movemask_epi8(is_digit) as u32 as u64,
                _mm256_movemask_epi8(is_space) as u32 as u64,
                _mm256_movemask_epi8(is_nl) as u32 as u64,
            )
        }};
    }

    macro_rules! accum_64 {
        ($c0:expr, $c1:expr, $valid:expr $(, $update:ident)?) => {{
            let (a0, d0, s0, n0) = classify_32!($c0);
            let (a1, d1, s1, n1) = classify_32!($c1);

            let valid: u64 = $valid;
            let m_alpha = (a0 | (a1 << 32)) & valid;
            let m_digit = (d0 | (d1 << 32)) & valid;
            let m_space = (s0 | (s1 << 32)) & valid;
            let m_nl = (n0 | (n1 << 32)) & valid;
            let m_punct = !(m_alpha | m_digit | m_space | m_nl) & valid;

            // Newlines: exact 12 units (6 * 2) per '\n'; skipped completely on compact JSON.
            if m_nl != 0 {
                units2 += (m_nl.count_ones() as u64) * 12;
            }

            // Punctuation: exact `8 * runs + 4 * bytes` in 1/16-token units.
            let punct_starts = (m_punct & !((m_punct << 1) | prev_punct)).count_ones() as u64;
            units2 += punct_starts * 8 + (m_punct.count_ones() as u64) * 4;

            // Alpha runs (100% branchless, exact across 64-byte boundaries):
            // len == 1 -> 12, len 2..=6 -> 16, len > 6 -> 16 + (len - 6) * 3.
            let prev_alpha1 = prev_alpha6 >> 5;
            let alpha_starts = m_alpha & !((m_alpha << 1) | prev_alpha1);
            let a_starts_cnt = alpha_starts.count_ones() as u64;
            let single_cnt = (pending_single_alpha & !m_alpha & 1)
                + ((alpha_starts & !(m_alpha >> 1)) & (u64::MAX >> 1)).count_ones() as u64;

            let pa2 = prev_alpha6 & (prev_alpha6 << 1);
            let pa4 = pa2 & (pa2 << 2);
            let a2 = m_alpha & ((m_alpha << 1) | prev_alpha1);
            let a4 = a2 & ((a2 << 2) | ((pa2 >> 4) & 3));
            let a7 = a4 & ((a4 << 3) | ((pa4 >> 3) & 7));

            units2 = units2 + a_starts_cnt * 16 + (a7.count_ones() as u64) * 3 - single_cnt * 4;

            // Digit runs: exact 16 for len 1..=3; rare >=4-digit runs add 8/start + 5/excess byte.
            let prev_digit1 = prev_digit3 >> 2;
            let d_starts = (m_digit & !((m_digit << 1) | prev_digit1)).count_ones() as u64;
            units2 += d_starts * 16;
            let pd2 = prev_digit3 & (prev_digit3 << 1);
            let d2 = m_digit & ((m_digit << 1) | prev_digit1);
            let d4 = d2 & ((d2 << 2) | ((pd2 >> 1) & 3));
            if d4 != 0 {
                let d4_starts = (d4 & !((d4 << 1) | prev_d4)).count_ones() as u64;
                units2 += d4_starts * 8 + (d4.count_ones() as u64) * 5;
            }

            // Space runs: exact 0 for len <= 4, +4 units (1/16) per excess space beyond 4.
            let ps2 = prev_space4 & (prev_space4 << 1);
            let s2 = m_space & ((m_space << 1) | (prev_space4 >> 3));
            let s4 = s2 & ((s2 << 2) | ((ps2 >> 2) & 3));
            let s5 = s4 & ((m_space << 4) | prev_space4);
            if s5 != 0 {
                units2 += (s5.count_ones() as u64) * 4;
            }

            $(
                let _ = stringify!($update);
                pending_single_alpha = alpha_starts >> 63;
                prev_punct = m_punct >> 63;
                prev_alpha6 = m_alpha >> 58;
                prev_digit3 = m_digit >> 61;
                prev_d4 = d4 >> 63;
                prev_space4 = m_space >> 60;
            )?
        }};
    }

    let mut i = 0usize;
    while i + 64 <= len {
        // SAFETY: `i + 64 <= len`; `_mm256_loadu_si256` permits unaligned loads.
        let c0 = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let c1 = _mm256_loadu_si256(ptr.add(i + 32) as *const __m256i);
        if _mm256_movemask_epi8(_mm256_or_si256(c0, c1)) != 0 {
            if pending_single_alpha != 0 {
                if TOKEN_BYTE_CLASS[buf[i] as usize] != CLS_ALPHA {
                    units2 -= 4;
                }
                pending_single_alpha = 0;
            }
            prev_alpha6 = 0;
            prev_space4 = 0;
            prev_punct = 0;
            prev_digit3 = 0;
            prev_d4 = 0;
            units2 += estimate_tokens_units_scalar(&buf[i..i + 64]) * 2;
            i += 64;
            continue;
        }
        accum_64!(c0, c1, u64::MAX, save_carry);
        i += 64;
    }

    let rem = len - i;
    if rem > 0 {
        let (c0, c1) = if rem >= 32 {
            let mut tail = [0u8; 32];
            let rem2 = rem - 32;
            if rem2 > 0 {
                // SAFETY: `i + 32 + rem2 == len` and `rem2 < 32`.
                std::ptr::copy_nonoverlapping(ptr.add(i + 32), tail.as_mut_ptr(), rem2);
            }
            (
                _mm256_loadu_si256(ptr.add(i) as *const __m256i),
                _mm256_loadu_si256(tail.as_ptr() as *const __m256i),
            )
        } else {
            let mut tail = [0u8; 32];
            // SAFETY: `i + rem == len` and `rem < 32`.
            std::ptr::copy_nonoverlapping(ptr.add(i), tail.as_mut_ptr(), rem);
            (
                _mm256_loadu_si256(tail.as_ptr() as *const __m256i),
                _mm256_setzero_si256(),
            )
        };
        if _mm256_movemask_epi8(_mm256_or_si256(c0, c1)) != 0 {
            if pending_single_alpha != 0 {
                if TOKEN_BYTE_CLASS[buf[i] as usize] != CLS_ALPHA {
                    units2 -= 4;
                }
                pending_single_alpha = 0;
            }
            units2 += estimate_tokens_units_scalar(&buf[i..len]) * 2;
        } else {
            accum_64!(c0, c1, (1u64 << rem) - 1);
            pending_single_alpha = 0;
        }
    }

    if pending_single_alpha != 0 {
        units2 -= 4;
    }

    let est = (units2 + 8) / 16;
    let clamped = est.max(1);
    u32::try_from(clamped).unwrap_or(u32::MAX)
}

/// Fast single-pass token estimator calibrated within ~10% of modern BPE
/// tokenizers across JSON, source code, English prose, and UTF-8 multibyte text.
///
/// Uses a 32-byte/iteration AVX2+POPCNT bitmask fast path on x86/x86_64 ASCII
/// spans with seamless scalar fallback on non-ASCII chunks.
///
/// Always returns `0` on empty input and at least `1` on any non-empty input.
#[inline]
pub fn estimate_tokens_fast(buf: &[u8]) -> u32 {
    if buf.is_empty() {
        return 0;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if buf.len() >= 32
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("popcnt")
        {
            // SAFETY: Runtime feature detection verified AVX2 and POPCNT are available.
            return unsafe { estimate_tokens_avx2(buf) };
        }
    }
    estimate_tokens_scalar(buf)
}

// ============================================================================
// Pure-Rust SipHash-c-d (reference Aumasson & Bernstein 2012 specification)
// ============================================================================

#[inline(always)]
fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = v1.rotate_left(13);
    *v1 ^= *v0;
    *v0 = v0.rotate_left(32);

    *v2 = v2.wrapping_add(*v3);
    *v3 = v3.rotate_left(16);
    *v3 ^= *v2;

    *v0 = v0.wrapping_add(*v3);
    *v3 = v3.rotate_left(21);
    *v3 ^= *v0;

    *v2 = v2.wrapping_add(*v1);
    *v1 = v1.rotate_left(17);
    *v1 ^= *v2;
    *v2 = v2.rotate_left(32);
}

struct SipHasher<const C: usize, const D: usize> {
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
    tail: u64,
    ntail: usize,
    length: usize,
}

impl<const C: usize, const D: usize> SipHasher<C, D> {
    #[inline]
    fn new(key: &[u8; 16]) -> Self {
        let k0 = u64::from_le_bytes([
            key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        ]);
        let k1 = u64::from_le_bytes([
            key[8], key[9], key[10], key[11], key[12], key[13], key[14], key[15],
        ]);
        Self {
            v0: k0 ^ 0x736f_6d65_7073_6575,
            v1: k1 ^ 0x646f_7261_6e64_6f6d,
            v2: k0 ^ 0x6c79_6765_6e65_7261,
            v3: k1 ^ 0x7465_6462_7974_6573,
            tail: 0,
            ntail: 0,
            length: 0,
        }
    }

    #[inline(always)]
    fn compress(&mut self, m: u64) {
        self.v3 ^= m;
        for _ in 0..C {
            sipround(&mut self.v0, &mut self.v1, &mut self.v2, &mut self.v3);
        }
        self.v0 ^= m;
    }

    #[inline]
    fn write(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len());

        if self.ntail > 0 {
            let need = 8 - self.ntail;
            if data.len() < need {
                for &b in data {
                    self.tail |= (b as u64) << (self.ntail * 8);
                    self.ntail += 1;
                }
                return;
            }
            for &b in &data[..need] {
                self.tail |= (b as u64) << (self.ntail * 8);
                self.ntail += 1;
            }
            let block = self.tail;
            self.tail = 0;
            self.ntail = 0;
            self.compress(block);
            data = &data[need..];
        }

        let (mut v0, mut v1, mut v2, mut v3) = (self.v0, self.v1, self.v2, self.v3);
        let chunks = data.chunks_exact(8);
        let rem = chunks.remainder();
        for chunk in chunks {
            let m = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            v3 ^= m;
            for _ in 0..C {
                sipround(&mut v0, &mut v1, &mut v2, &mut v3);
            }
            v0 ^= m;
        }
        self.v0 = v0;
        self.v1 = v1;
        self.v2 = v2;
        self.v3 = v3;

        for &b in rem {
            self.tail |= (b as u64) << (self.ntail * 8);
            self.ntail += 1;
        }
    }

    #[inline]
    fn finish(mut self) -> u64 {
        let b = self.tail | (((self.length as u64) & 0xff) << 56);
        self.compress(b);
        self.v2 ^= 0xff;
        for _ in 0..D {
            sipround(&mut self.v0, &mut self.v1, &mut self.v2, &mut self.v3);
        }
        self.v0 ^ self.v1 ^ self.v2 ^ self.v3
    }
}

/// Streaming SipHash-1-3 hasher for zero-allocation normalized block hashing.
pub(crate) struct KeyedHasher(SipHasher<1, 3>);

impl KeyedHasher {
    #[inline]
    pub(crate) fn new(key: &[u8; 16]) -> Self {
        Self(SipHasher::<1, 3>::new(key))
    }

    #[inline]
    pub(crate) fn write(&mut self, data: &[u8]) {
        self.0.write(data);
    }

    #[inline]
    pub(crate) fn finish(self) -> u64 {
        self.0.finish()
    }
}

#[inline]
fn siphash_oneshot<const C: usize, const D: usize>(key: &[u8; 16], data: &[u8]) -> u64 {
    let mut h = SipHasher::<C, D>::new(key);
    h.write(data);
    h.finish()
}

/// Pure-Rust SipHash-2-4 KDF deriving a 128-bit per-tenant key from `secret`
/// and `tenant_id`, isolating block hashes across tenants against cross-tenant
/// dictionary probing.
pub fn derive_tenant_key(secret: &[u8], tenant_id: &str) -> [u8; 16] {
    const KDF_EXTRACT_LO: [u8; 16] = *b"cachepin-kdf-ex0";
    const KDF_EXTRACT_HI: [u8; 16] = *b"cachepin-kdf-ex1";

    let prk_lo = siphash_oneshot::<2, 4>(&KDF_EXTRACT_LO, secret);
    let prk_hi = siphash_oneshot::<2, 4>(&KDF_EXTRACT_HI, secret);
    let mut prk = [0u8; 16];
    prk[0..8].copy_from_slice(&prk_lo.to_le_bytes());
    prk[8..16].copy_from_slice(&prk_hi.to_le_bytes());

    let tenant_bytes = tenant_id.as_bytes();
    let tenant_len = (tenant_bytes.len() as u64).to_le_bytes();
    let secret_len = (secret.len() as u64).to_le_bytes();

    let mut h_lo = SipHasher::<2, 4>::new(&prk);
    h_lo.write(b"cachepin:tenant:lo:v1\0");
    h_lo.write(&tenant_len);
    h_lo.write(tenant_bytes);
    h_lo.write(&secret_len);
    h_lo.write(secret);
    let out_lo = h_lo.finish();

    let mut h_hi = SipHasher::<2, 4>::new(&prk);
    h_hi.write(b"cachepin:tenant:hi:v1\0");
    h_hi.write(&tenant_len);
    h_hi.write(tenant_bytes);
    h_hi.write(&secret_len);
    h_hi.write(secret);
    let out_hi = h_hi.finish();

    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&out_lo.to_le_bytes());
    out[8..16].copy_from_slice(&out_hi.to_le_bytes());
    out
}

/// Cryptographically keyed 64-bit SipHash-1-3 digest.
#[inline]
pub fn keyed_hash64(key: &[u8; 16], data: &[u8]) -> u64 {
    siphash_oneshot::<1, 3>(key, data)
}

/// Keyed, domain-separated, non-commutative chained 64-bit hash combining
/// the cumulative prefix hash `prev` with the next `block_hash`.
#[inline]
pub fn chain_hash64(key: &[u8; 16], prev: u64, block_hash: u64) -> u64 {
    let mut buf = [0u8; 17];
    buf[0] = 0xC1;
    buf[1..9].copy_from_slice(&prev.to_le_bytes());
    buf[9..17].copy_from_slice(&block_hash.to_le_bytes());
    keyed_hash64(key, &buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_lcp(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    fn naive_count(buf: &[u8], needle: u8) -> usize {
        buf.iter().filter(|&&b| b == needle).count()
    }

    #[test]
    fn test_lcp_avx2_vs_scalar_exhaustive_lengths_and_alignments() {
        let mut buf_a = vec![0u8; 1088];
        for (i, slot) in buf_a.iter_mut().enumerate() {
            *slot = ((i.wrapping_mul(37)).wrapping_add(13) & 0xFF) as u8;
        }
        let buf_b = buf_a.clone();

        // Exhaustive lengths 0..=1024 with full match and single-byte mismatches.
        for len in 0..=1024 {
            let a = &buf_a[..len];
            let b = &buf_b[..len];
            let expected = naive_lcp(a, b);
            assert_eq!(longest_common_prefix_scalar(a, b), expected);
            assert_eq!(longest_common_prefix_len(a, b), expected);
        }

        // Test every alignment offset 0..32 and mismatch positions around boundaries.
        for offset in 0..32 {
            let max_len = 256;
            let base_a = &buf_a[offset..offset + max_len];
            let mut local_b = base_a.to_vec();

            for pos in [
                0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255,
            ] {
                local_b[pos] ^= 0x01;
                let expected = naive_lcp(base_a, &local_b);
                assert_eq!(expected, pos);
                assert_eq!(longest_common_prefix_scalar(base_a, &local_b), expected);
                assert_eq!(longest_common_prefix_len(base_a, &local_b), expected);
                local_b[pos] ^= 0x01;
            }
        }

        // Asymmetric slice lengths.
        assert_eq!(longest_common_prefix_len(&buf_a[..50], &buf_b[..200]), 50);
        assert_eq!(longest_common_prefix_len(&buf_a[..200], &buf_b[..50]), 50);
        assert_eq!(longest_common_prefix_len(b"", b"abc"), 0);
        assert_eq!(longest_common_prefix_len(b"abc", b""), 0);
        assert_eq!(longest_common_prefix_len(b"", b""), 0);

        // Ensure backend string is one of the documented names.
        let name = simd_backend_name();
        assert!(name == "avx2" || name == "neon" || name == "scalar-u64");
    }

    #[test]
    fn test_count_byte_avx2_vs_scalar_and_swar_carry_edges() {
        // Adversarial carry pattern that breaks naive `(x - 0x0101..) & !x & 0x8080..`.
        let carry_buf = [0x00u8, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01];
        for needle in [0x00u8, 0x01, 0x02, 0x7F, 0x80, 0xFF] {
            let expected = naive_count(&carry_buf, needle);
            assert_eq!(count_byte_scalar(&carry_buf, needle), expected);
            assert_eq!(count_byte_simd(&carry_buf, needle), expected);
        }

        let mut buf = vec![0u8; 1088];
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = match i % 7 {
                0 => 0x00,
                1 => 0x01,
                2 => b'\n',
                3 => 0x7F,
                4 => 0x80,
                5 => 0xFF,
                _ => (i as u8).wrapping_mul(19),
            };
        }

        for len in 0..=1024 {
            for needle in [0x00u8, 0x01, b'\n', 0x7F, 0x80, 0xFF] {
                let s = &buf[..len];
                let expected = naive_count(s, needle);
                assert_eq!(count_byte_scalar(s, needle), expected);
                assert_eq!(count_byte_simd(s, needle), expected);
            }
        }

        for offset in 0..32 {
            let s = &buf[offset..offset + 257];
            for needle in [0x00u8, 0x01, b'\n', 0xFF] {
                assert_eq!(count_byte_simd(s, needle), naive_count(s, needle));
                assert_eq!(count_byte_scalar(s, needle), naive_count(s, needle));
            }
        }
    }

    #[test]
    fn test_estimate_tokens_fast_accuracy_and_bounds() {
        assert_eq!(estimate_tokens_fast(b""), 0);
        assert_eq!(estimate_tokens_fast(b"a"), 1);
        assert_eq!(estimate_tokens_fast(b" "), 1);
        assert_eq!(estimate_tokens_fast(b"\n"), 1);

        // English prose (~10 BPE tokens).
        let english = b"The quick brown fox jumps over the lazy dog.";
        let eng_tok = estimate_tokens_fast(english);
        assert!(
            (9..=11).contains(&eng_tok),
            "english tokens out of 10% band: {eng_tok}"
        );

        // Compact JSON (~18 BPE tokens).
        let json = br#"{"role":"user","content":"Hello world","model":"claude-sonnet-4"}"#;
        let json_tok = estimate_tokens_fast(json);
        assert!(
            (16..=20).contains(&json_tok),
            "json tokens out of 10% band: {json_tok}"
        );

        // Rust source code (~21 BPE tokens).
        let code = b"pub fn longest_common_prefix_len(a: &[u8], b: &[u8]) -> usize {";
        let code_tok = estimate_tokens_fast(code);
        assert!(
            (18..=24).contains(&code_tok),
            "code tokens out of ~10% band: {code_tok}"
        );

        // UTF-8 CJK (7 chars -> ~7..11 tokens, never 0, doesn't triple-count continuation bytes).
        let cjk = "こんにちは世界".as_bytes();
        let cjk_tok = estimate_tokens_fast(cjk);
        assert!(
            (7..=11).contains(&cjk_tok),
            "cjk tokens out of expected range: {cjk_tok}"
        );

        // Emoji (4-byte UTF-8).
        let emoji = "😀🚀🔥".as_bytes();
        let emoji_tok = estimate_tokens_fast(emoji);
        assert!((4..=8).contains(&emoji_tok), "emoji tokens: {emoji_tok}");
    }

    #[test]
    fn test_siphash24_official_reference_vectors() {
        // Official Aumasson & Bernstein (2012) Appendix A test vectors:
        // key = 00 01 02 ... 0f
        let key: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        // Empty message -> 0x726fdb47dd0e0e31
        assert_eq!(siphash_oneshot::<2, 4>(&key, &[]), 0x726f_db47_dd0e_0e31);

        // 15-byte message 00 01 .. 0e -> 0xa129ca6149be45e5
        let msg15: [u8; 15] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e,
        ];
        assert_eq!(
            siphash_oneshot::<2, 4>(&key, &msg15),
            0xa129_ca61_49be_45e5
        );

        // Verify streaming SipHasher matches one-shot across every split point.
        for split in 0..=msg15.len() {
            let mut h = SipHasher::<2, 4>::new(&key);
            h.write(&msg15[..split]);
            h.write(&msg15[split..]);
            assert_eq!(h.finish(), 0xa129_ca61_49be_45e5);
        }

        // Verify streaming SipHash-1-3 matches one-shot across byte-by-byte feeding.
        let sample = b"cachepin multi-agent prefix compiler test payload 1234567890";
        let expected = keyed_hash64(&key, sample);
        let mut streaming = SipHasher::<1, 3>::new(&key);
        for chunk in sample.chunks(3) {
            streaming.write(chunk);
        }
        assert_eq!(streaming.finish(), expected);
    }

    #[test]
    fn test_tenant_kdf_and_chain_hash_invariants() {
        let secret1 = b"super-secret-master-key-01";
        let secret2 = b"super-secret-master-key-02";

        let k_a1 = derive_tenant_key(secret1, "tenant-alpha");
        let k_a2 = derive_tenant_key(secret1, "tenant-alpha");
        let k_b = derive_tenant_key(secret1, "tenant-beta");
        let k_a_other_secret = derive_tenant_key(secret2, "tenant-alpha");

        assert_eq!(k_a1, k_a2);
        assert_ne!(k_a1, k_b);
        assert_ne!(k_a1, k_a_other_secret);

        // Same block under different tenants produces distinct hashes.
        let block = b"{\"role\":\"system\",\"content\":\"shared\"}";
        let h_a = keyed_hash64(&k_a1, block);
        let h_b = keyed_hash64(&k_b, block);
        assert_ne!(h_a, h_b);

        // Chain hash is deterministic and non-commutative.
        let c1 = chain_hash64(&k_a1, 111, 222);
        let c2 = chain_hash64(&k_a1, 111, 222);
        let c_rev = chain_hash64(&k_a1, 222, 111);
        let c_other_key = chain_hash64(&k_b, 111, 222);
        assert_eq!(c1, c2);
        assert_ne!(c1, c_rev);
        assert_ne!(c1, c_other_key);
    }

    #[test]
    fn test_find_json_special_byte_avx2_vs_scalar() {
        // Test across lengths 0..=260, alignments 0..=32, all special bytes, and UTF-8 >= 0x80.
        let mut buf = vec![b'a'; 300];
        for i in (0..buf.len()).step_by(5) {
            buf[i] = 0xE2; // High UTF-8 lead byte must NOT trigger `< 0x20`!
        }
        for len in 0..=260 {
            for align in 0..=32 {
                if align + len > buf.len() {
                    continue;
                }
                let slice = &buf[align..align + len];
                assert_eq!(
                    find_json_special_byte(slice),
                    find_json_special_byte_scalar(slice)
                );
            }
        }
        for &special in &[0x00u8, 0x0A, 0x1F, b'"', b'\\'] {
            for pos in [0usize, 1, 15, 16, 31, 32, 33, 63, 64, 95, 127, 128, 199] {
                let mut v = vec![b'x'; 200];
                v[pos] = special;
                assert_eq!(find_json_special_byte(&v), pos);
                assert_eq!(find_json_special_byte_scalar(&v), pos);
            }
        }
    }

    #[test]
    fn test_find_any3_byte_avx2_vs_scalar() {
        let mut buf = vec![b'a'; 300];
        for i in (0..buf.len()).step_by(7) {
            buf[i] = 0xE2;
        }
        // Exhaustive no-match across lengths and alignments.
        for len in 0..=260 {
            for align in 0..=32 {
                if align + len > buf.len() {
                    continue;
                }
                let slice = &buf[align..align + len];
                assert_eq!(
                    find_any3_byte(slice, b'-', b'_', b':'),
                    find_any3_byte_scalar(slice, b'-', b'_', b':')
                );
                assert_eq!(find_any3_byte(slice, b'-', b'_', b':'), len);
            }
        }
        // Single hit at every boundary and alignment for each of the 3 target bytes.
        for &target in &[b'-', b'_', b':', 0x00u8, 0x7Fu8, 0x80u8, 0xFFu8] {
            for pos in [
                0usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 199,
            ] {
                let mut v = vec![b'x'; 200];
                v[pos] = target;
                assert_eq!(find_any3_byte(&v, b'-', b'_', target), pos);
                assert_eq!(find_any3_byte_scalar(&v, b'-', b'_', target), pos);
                assert_eq!(find_any3_byte(&v, target, b'_', b':'), pos);
                assert_eq!(find_any3_byte(&v, b'-', target, b':'), pos);
            }
        }
        // Multiple hits returns strictly the earliest offset.
        let mut multi = vec![b'z'; 128];
        multi[33] = b':';
        multi[40] = b'-';
        multi[64] = b'_';
        assert_eq!(find_any3_byte(&multi, b'-', b'_', b':'), 33);
        assert_eq!(find_any3_byte(&multi[34..], b'-', b'_', b':'), 6);
        assert_eq!(find_any3_byte(&multi[41..], b'-', b'_', b':'), 23);
        assert_eq!(find_any3_byte(&multi[65..], b'-', b'_', b':'), 63);
    }

    #[test]
    fn test_estimate_tokens_fast_simd_vs_scalar_parity() {
        let samples: &[&[u8]] = &[
            b"The quick brown fox jumps over the lazy dog.",
            br#"{"role":"user","content":"Hello world","model":"claude-sonnet-4"}"#,
            b"pub fn longest_common_prefix_len(a: &[u8], b: &[u8]) -> usize {",
            b"a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J",
            b"supercalifragilisticexpialidocious internationalization synchronization determinism",
            b"0 12 123 1234 12345 123456 1234567 1234567890 2026-11-22T15:48:57Z 1732290537",
            b"    four        eight            twelve                sixteen\n\n\nline4\r\n",
            "Current time: ⟦cp:v1⟧ | Trace: ⟦cp:v2⟧ | Session: ⟦cp:v3⟧\nYou are an autonomous systems engineer working on a shared repository with こんにちは世界 and 🚀🔥.".as_bytes(),
        ];

        // Test every sample directly and also repeated across many 32-byte boundaries & offsets.
        for &sample in samples {
            let simd = estimate_tokens_fast(sample);
            let scalar = estimate_tokens_scalar(sample);
            let diff = simd.abs_diff(scalar);
            let max_allowed = ((scalar as f64) * 0.05).ceil().max(1.0) as u32;
            assert!(
                diff <= max_allowed,
                "SIMD ({simd}) vs scalar ({scalar}) diverged by {diff} (> {max_allowed}) on {:?}",
                String::from_utf8_lossy(sample)
            );

            // Repeat to create multi-kilobyte payloads crossing dozens of 32-byte boundaries.
            let mut big = Vec::with_capacity(sample.len() * 40);
            for _ in 0..40 {
                big.extend_from_slice(sample);
                big.push(b' ');
            }
            for trim in [0usize, 1, 7, 15, 16, 31, 32, 33, 62, 63, 64, 65] {
                if trim < big.len() {
                    let s = &big[trim..];
                    let simd_b = estimate_tokens_fast(s);
                    let scalar_b = estimate_tokens_scalar(s);
                    let diff_b = simd_b.abs_diff(scalar_b);
                    let max_b = ((scalar_b as f64) * 0.05).ceil().max(2.0) as u32;
                    assert!(
                        diff_b <= max_b,
                        "Big trim={trim}: SIMD ({simd_b}) vs scalar ({scalar_b}) diverged by {diff_b} (> {max_b})"
                    );
                }
            }
        }

        // Targeted 64-byte chunk boundary carry tests (single-alpha, long-alpha, digits, spaces).
        let boundary_cases = [
            format!("{}a ", " ".repeat(63)),
            format!("{}ab ", " ".repeat(63)),
            format!("{}abcdefghij", " ".repeat(60)),
            format!("{}12345678 ", " ".repeat(61)),
            format!("{}        word", "x".repeat(61)),
            format!("{}x", " ".repeat(63)),
        ];
        for case in &boundary_cases {
            let b = case.as_bytes();
            let simd = estimate_tokens_fast(b);
            let scalar = estimate_tokens_scalar(b);
            assert!(
                simd.abs_diff(scalar) <= 1,
                "64-byte boundary mismatch on len {}: simd={simd} scalar={scalar}",
                b.len()
            );
        }

        // Verify multi-outer-loop count_byte (> 255 * 16 = 4080 bytes for NEON accumulator flush).
        let big_count = vec![b'a'; 5000];
        assert_eq!(count_byte_simd(&big_count, b'a'), 5000);
        assert_eq!(count_byte_simd(&big_count, b'b'), 0);
    }
}
