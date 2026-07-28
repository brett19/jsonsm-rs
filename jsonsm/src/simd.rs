//! Explicit SIMD implementations of the tokenizer's [`Scan`] primitives.
//!
//! This module contains **all** of the crate's `unsafe`. Everything else is compiled under
//! `deny(unsafe_code)` (and under a literal `forbid(unsafe_code)` when the `simd` feature is
//! off), so the audit surface is exactly this file.
//!
//! # What is accelerated
//!
//! The four bulk run-crossing primitives — the closing of a string literal, a run of
//! whitespace, a run of digits, a run free of container brackets — plus `skip_container`,
//! which crosses an entire irrelevant value. Two x86-64 backends implement them (see
//! [`Backend`]). The
//! JSON state machine itself is untouched and shared with
//! the scalar tokenizer ([`GenericTokenizer`]), so grammar, error kinds, and error offsets
//! are identical by construction; a run-crossing kernel can only affect *how fast* an
//! uninteresting run is crossed. Each kernel's tail delegates to
//! [`ScalarScan`](crate::tokenizer::ScalarScan), which is
//! therefore both the fallback and the behavioural reference.
//!
//! `skip_container` is the exception and deserves the extra scrutiny it gets: the vector
//! version shares no code with the portable one, tracking string interiors and bracket
//! depth through bitmask arithmetic rather than by walking bytes. It is held to the
//! portable walk by `skip_matches_bytewise_on_every_backend`, whose corpus is built so the
//! interesting bytes land inside full 64-byte windows at every alignment.
//!
//! # Dispatch
//!
//! The backend is a **type**, not a stored value: each implementation of [`Scan`] is a
//! separate monomorphisation, so a kernel call is a direct call that inlines, and there is no
//! per-scan test of which backend is in force. [`Backend::detect`] is consulted once, when a
//! matcher is constructed, and the choice is turned into a type parameter for the duration of
//! one document.
//!
//! Dispatching on a stored enum instead puts a comparison at the top of every bulk scan —
//! once per string, number or whitespace run — and blocks the inlining that makes the small
//! kernels worth having.
//!
//! # Safety invariants
//!
//! Every kernel here upholds the same three:
//!
//! 1. **Feature gating.** A kernel is only ever reached through a `self.backend` match arm,
//!    and a `Backend` variant is only ever produced by `is_x86_feature_detected!` returning
//!    true for that feature on this CPU (see [`Backend::detect`] / [`Backend::available`]).
//!    There is no other path to a kernel. SSE2 is additionally guaranteed by the x86-64
//!    baseline, which is why those kernels need no `#[target_feature]` — and, in turn, why
//!    they inline into the state machine and outperform the wider AVX2 ones.
//! 2. **In-bounds loads.** A `W`-byte load at offset `i` runs only while `i + W <= len`,
//!    where `len` is the length of the slice the pointer came from. There is no over-read
//!    past the end of the buffer, masked or otherwise; the remainder is handled scalar.
//! 3. **No aliasing or mutation.** The kernels take `&[u8]` and only read through
//!    `as_ptr()`, using unaligned loads (`loadu`), so no alignment precondition exists.
#![allow(unsafe_code)]

use crate::tokenizer::{
    skip_container_bytewise, GenericTokenizer, Scan, SkipError, MAX_SKIP_DEPTH,
};

/// The scan implementation chosen for this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Portable byte loops — what non-x86-64 targets run, and what the kernels' sub-vector
    /// tails delegate to. Behaviourally identical to
    /// [`JsonTokenizer`](crate::tokenizer::JsonTokenizer), and about 4% slower than it,
    /// since it still pays the backend branch on every bulk scan.
    Scalar,
    /// 16-byte SSE2 kernels. SSE2 is part of the x86-64 baseline, so these need no
    /// `#[target_feature]` and inline into the state machine — but what makes them the
    /// better choice there is the narrower vector, not the inlining. See [`HybridScan`].
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// 32-byte AVX2 kernels. Wider, and worth it only on runs long enough to amortise the
    /// wider load — which among these kernels means [`Scan::skip_container`] and nothing
    /// else. Also reached through a non-inlinable call, though that costs far less than the
    /// width does; [`HybridScan`] has the measurements separating the two.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// SSE2 kernels for the state machine, AVX2 only for [`Scan::skip_container`].
    ///
    /// The two kernel groups are called at completely different rates, so one backend for
    /// both is a compromise neither wants. See [`HybridScan`].
    #[cfg(target_arch = "x86_64")]
    Hybrid,
}

impl Backend {
    /// Probe the running CPU. Cheap and idempotent (`is_x86_feature_detected!` caches its
    /// answer in a static), but still called only once per [`SimdScan`].
    ///
    /// Note this prefers neither of the pure vector backends: [`Avx2`](Backend::Avx2) loses
    /// to [`Sse2`](Backend::Sse2) on the state machine's short runs, and `Sse2` loses to it
    /// on the long ones a skip crosses, so [`Hybrid`](Backend::Hybrid) takes each where it
    /// wins. See [`HybridScan`] for why the two kernel groups want different widths.
    ///
    /// `#[inline(never)]` is deliberate and load-bearing. Inlined, this function's result
    /// constant-folds into every [`FastMatcher::new`](crate::matcher::FastMatcher::new), and
    /// which backend that constant names then cascades into inlining decisions across the
    /// whole matcher — including code the choice does not otherwise touch. The attribute
    /// costs one call per matcher construction and makes the build insensitive to which
    /// backend is preferred.
    #[inline(never)]
    pub fn detect() -> Backend {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return Backend::Hybrid;
            }
            // sse2 is guaranteed by the x86-64 ABI; the check documents the requirement.
            if std::arch::is_x86_feature_detected!("sse2") {
                return Backend::Sse2;
            }
        }
        Backend::Scalar
    }

    /// Every backend this CPU can actually run, not just the one [`Self::detect`] prefers.
    ///
    /// Tests iterate this so a backend is never silently left unexercised just because it
    /// is not the default — which is exactly what happened when `Sse2` was introduced and
    /// took over from `Avx2` as the detected choice.
    pub fn available() -> Vec<Backend> {
        let mut v = vec![Backend::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse2") {
                v.push(Backend::Sse2);
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                v.push(Backend::Avx2);
                v.push(Backend::Hybrid);
            }
        }
        v
    }
}

/// 16-byte SSE2 kernels.
///
/// One type per backend, so the choice is a compile-time generic rather than a runtime enum
/// test — such a test would sit on every bulk scan. SSE2 is in the x86-64 base ABI, so this
/// needs neither detection nor a
/// `#[target_feature]` context — the kernels inline into the state machine directly.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sse2Scan;

#[cfg(target_arch = "x86_64")]
impl Scan for Sse2Scan {
    #[inline(always)]
    fn new() -> Self {
        Sse2Scan
    }
    #[inline(always)]
    fn string_event(&self, data: &[u8], from: usize) -> usize {
        sse2::string_event(data, from)
    }
    #[inline(always)]
    fn skip_ws(&self, data: &[u8], from: usize) -> usize {
        sse2::skip_ws(data, from)
    }
    #[inline(always)]
    fn skip_digits(&self, data: &[u8], from: usize) -> usize {
        sse2::skip_digits(data, from)
    }
    #[inline(always)]
    fn structural_event(&self, data: &[u8], from: usize) -> usize {
        sse2::structural_event(data, from)
    }
    #[inline(always)]
    fn skip_container(&self, data: &[u8], from: usize, outer: usize) -> Result<usize, SkipError> {
        skip_container_blocked(self, sse2::classify64, data, from, outer)
    }
    // No `enter` override: SSE2 is in the x86-64 baseline, so these already inline.
}

/// 32-byte AVX2 kernels, entered through a `#[target_feature]` context.
///
/// [`Scan::enter`] wraps the *whole* scan — state machine included — in that context, which
/// is what lets the kernels inline. Reached per bulk scan instead, each one becomes a
/// non-inlinable call and the width stops paying for itself; wrapped this way AVX2 retires
/// fewer instructions than SSE2 and is the better choice where runs are long.
///
/// # Safety
/// This type's mere existence asserts that the CPU supports AVX2: [`Scan::new`] panics
/// otherwise, and there is no other constructor. That is what makes the kernels below sound
/// despite not carrying `#[target_feature]` themselves.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Avx2Scan(());

#[cfg(target_arch = "x86_64")]
impl Scan for Avx2Scan {
    #[inline]
    fn new() -> Self {
        assert!(
            std::arch::is_x86_feature_detected!("avx2"),
            "Avx2Scan constructed on a CPU without AVX2"
        );
        Avx2Scan(())
    }
    #[inline(always)]
    fn enter<R>(f: impl FnOnce() -> R) -> R {
        // SAFETY: an `Avx2Scan` cannot exist unless detection confirmed AVX2.
        unsafe { avx2::context(f) }
    }
    #[inline(always)]
    fn string_event(&self, data: &[u8], from: usize) -> usize {
        // SAFETY: `self` proves AVX2 is present; `enter` has opened an AVX2 context.
        unsafe { avx2::string_event(data, from) }
    }
    #[inline(always)]
    fn skip_ws(&self, data: &[u8], from: usize) -> usize {
        // SAFETY: as above.
        unsafe { avx2::skip_ws(data, from) }
    }
    #[inline(always)]
    fn skip_digits(&self, data: &[u8], from: usize) -> usize {
        // SAFETY: as above.
        unsafe { avx2::skip_digits(data, from) }
    }
    #[inline(always)]
    fn structural_event(&self, data: &[u8], from: usize) -> usize {
        // SAFETY: as above.
        unsafe { avx2::structural_event(data, from) }
    }
    #[inline(always)]
    fn skip_container(&self, data: &[u8], from: usize, outer: usize) -> Result<usize, SkipError> {
        // SAFETY: as above — `classify64` is only reached from inside `enter`.
        skip_container_blocked(
            self,
            |d, at| unsafe { avx2::classify64(d, at) },
            data,
            from,
            outer,
        )
    }
}

/// SSE2 kernels for the state machine, AVX2 for [`Scan::skip_container`] alone.
///
/// The two kernel groups are not called at comparable rates, and a single backend has to
/// serve both. The state machine's kernels run per token, on runs a few bytes long;
/// `skip_container` runs once per skipped value and may cross kilobytes. Whole-document AVX2
/// is a large win on the second and a loss on the first.
///
/// The cause is vector width meeting run length, not the `#[target_feature]` context AVX2
/// kernels are reached through. That context does cost something — such code cannot be
/// inlined into a caller lacking the feature, so under [`Avx2Scan`] every [`Scan::enter`] on
/// the hot path is a real call out of the state machine — but a build with the feature
/// enabled globally, where the context is free, still leaves AVX2 behind on short runs. A
/// 32-byte load, compare and movemask simply needs a longer run to amortise than a JSON
/// token gives it.
///
/// So: width where the runs are long, and none of its costs where they are short. This
/// backend leaves [`Scan::enter`] as the identity, so the state machine keeps baseline SSE2
/// kernels that inline into it completely, and opens an AVX2 context inside
/// `skip_container` alone — once per skipped container rather than once per bulk scan.
///
/// # Safety
/// As [`Avx2Scan`]: this type's existence asserts AVX2, since [`Scan::new`] is the only
/// constructor and it panics otherwise.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridScan(());

#[cfg(target_arch = "x86_64")]
impl Scan for HybridScan {
    #[inline]
    fn new() -> Self {
        assert!(
            std::arch::is_x86_feature_detected!("avx2"),
            "HybridScan constructed on a CPU without AVX2"
        );
        HybridScan(())
    }
    // No `enter` override, and that is the whole point: the state machine stays in the
    // baseline target, where these SSE2 kernels inline into it.
    #[inline(always)]
    fn string_event(&self, data: &[u8], from: usize) -> usize {
        sse2::string_event(data, from)
    }
    #[inline(always)]
    fn skip_ws(&self, data: &[u8], from: usize) -> usize {
        sse2::skip_ws(data, from)
    }
    #[inline(always)]
    fn skip_digits(&self, data: &[u8], from: usize) -> usize {
        sse2::skip_digits(data, from)
    }
    #[inline(always)]
    fn structural_event(&self, data: &[u8], from: usize) -> usize {
        sse2::structural_event(data, from)
    }
    /// The one place this backend differs from [`Sse2Scan`], and the one place the AVX2
    /// context is worth entering: a single call amortised over a whole skipped container.
    ///
    /// The window loop's `classify64` is AVX2; the sub-window tail falls back to `self`'s
    /// SSE2 kernels, which are legal and inline here too — SSE2 is a subset of AVX2, so the
    /// context widens what is available without withdrawing anything.
    #[inline(always)]
    fn skip_container(&self, data: &[u8], from: usize, outer: usize) -> Result<usize, SkipError> {
        // SAFETY: `self` proves AVX2 is present, and `classify64` is only reached from
        // inside the context opened here.
        unsafe {
            avx2::context(|| {
                skip_container_blocked(self, |d, at| avx2::classify64(d, at), data, from, outer)
            })
        }
    }
}

/// The scanner this target uses when no backend is named: whichever vector scanner the
/// architecture *guarantees*, so selecting it costs nothing at runtime.
///
/// On x86-64 that is [`Sse2Scan`] (SSE2 is in the base ABI). Backends that need CPU
/// detection — AVX2 today, AVX-512/SVE later — are not reachable through this alias; they
/// are selected per backend by [`FastMatcher`](crate::matcher::FastMatcher), which is
/// monomorphised over the scanner and so pays no per-scan branch either.
#[cfg(target_arch = "x86_64")]
pub type SimdScan = Sse2Scan;
/// The scanner this target uses when no backend is named. No vector scanner is guaranteed
/// here, so this is the portable one.
#[cfg(not(target_arch = "x86_64"))]
pub type SimdScan = crate::tokenizer::ScalarScan;

/// The SIMD-accelerated JSON tokenizer. Same tokens, same errors, same offsets as
/// [`JsonTokenizer`](crate::tokenizer::JsonTokenizer) — see the module docs.
pub type SimdTokenizer<'a> = GenericTokenizer<'a, SimdScan>;

/// AVX2 kernels deliberately *without* `#[target_feature]` on each kernel, so that they can
/// be inlined into a caller that already has it — see [`Avx2Scan::enter`], which supplies
/// that caller. Marking each kernel instead would make every bulk scan a non-inlinable call,
/// which costs more than the extra width buys.
///
/// Sound only because [`Avx2Scan`] cannot be constructed without detection having confirmed
/// AVX2, so possessing one is proof the instructions are available.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::Masks64;
    use crate::tokenizer::{ScalarScan, Scan};
    use std::arch::x86_64::*;

    const LANES: usize = 32;

    /// The context every kernel here must run inside.
    ///
    /// # Safety
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn context<R>(f: impl FnOnce() -> R) -> R {
        f()
    }

    /// # Safety
    /// Must be called from within [`context`].
    #[inline(always)]
    pub unsafe fn string_event(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let quote = _mm256_set1_epi8(b'"' as i8);
        let backslash = _mm256_set1_epi8(b'\\' as i8);
        let ctrl_hi = _mm256_set1_epi8(0x1f);
        let mut i = from;
        while i + LANES <= len {
            let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let hits = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(v, quote), _mm256_cmpeq_epi8(v, backslash)),
                _mm256_cmpeq_epi8(_mm256_max_epu8(v, ctrl_hi), ctrl_hi),
            );
            let mask = _mm256_movemask_epi8(hits) as u32;
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += LANES;
        }
        ScalarScan.string_event(data, i)
    }

    /// # Safety
    /// Must be called from within [`context`].
    #[inline(always)]
    pub unsafe fn skip_ws(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let space = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let cr = _mm256_set1_epi8(b'\r' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        let mut i = from;
        while i + LANES <= len {
            let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let ws = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(v, space), _mm256_cmpeq_epi8(v, tab)),
                _mm256_or_si256(_mm256_cmpeq_epi8(v, cr), _mm256_cmpeq_epi8(v, lf)),
            );
            let mask = !(_mm256_movemask_epi8(ws) as u32);
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += LANES;
        }
        ScalarScan.skip_ws(data, i)
    }

    /// # Safety
    /// Must be called from within [`context`].
    #[inline(always)]
    pub unsafe fn skip_digits(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let zero = _mm256_set1_epi8(b'0' as i8);
        let nine = _mm256_set1_epi8(9);
        let mut i = from;
        while i + LANES <= len {
            let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let shifted = _mm256_sub_epi8(v, zero);
            let is_digit = _mm256_cmpeq_epi8(_mm256_max_epu8(shifted, nine), nine);
            let mask = !(_mm256_movemask_epi8(is_digit) as u32);
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += LANES;
        }
        ScalarScan.skip_digits(data, i)
    }

    /// # Safety
    /// Must be called from within [`context`].
    #[inline(always)]
    pub unsafe fn structural_event(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let obj_open = _mm256_set1_epi8(b'{' as i8);
        let obj_close = _mm256_set1_epi8(b'}' as i8);
        let arr_open = _mm256_set1_epi8(b'[' as i8);
        let arr_close = _mm256_set1_epi8(b']' as i8);
        let quote = _mm256_set1_epi8(b'"' as i8);
        let mut i = from;
        while i + LANES <= len {
            let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let hits = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(v, obj_open),
                    _mm256_cmpeq_epi8(v, obj_close),
                ),
                _mm256_or_si256(
                    _mm256_or_si256(
                        _mm256_cmpeq_epi8(v, arr_open),
                        _mm256_cmpeq_epi8(v, arr_close),
                    ),
                    _mm256_cmpeq_epi8(v, quote),
                ),
            );
            let mask = _mm256_movemask_epi8(hits) as u32;
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += LANES;
        }
        ScalarScan.structural_event(data, i)
    }

    /// Classify 64 bytes at `at` into [`Masks64`], two 32-byte loads at a time.
    ///
    /// # Safety
    /// Must be called from within [`context`], and `at + 64 <= data.len()`.
    #[inline(always)]
    pub unsafe fn classify64(data: &[u8], at: usize) -> Masks64 {
        debug_assert!(at + 64 <= data.len());
        let ptr = data.as_ptr();
        let mut m = Masks64::default();
        let quote = _mm256_set1_epi8(b'"' as i8);
        let backslash = _mm256_set1_epi8(b'\\' as i8);
        let obj_open = _mm256_set1_epi8(b'{' as i8);
        let obj_close = _mm256_set1_epi8(b'}' as i8);
        let arr_open = _mm256_set1_epi8(b'[' as i8);
        let arr_close = _mm256_set1_epi8(b']' as i8);
        for chunk in 0..2 {
            let v = _mm256_loadu_si256(ptr.add(at + chunk * LANES) as *const __m256i);
            let bit = |hits| (_mm256_movemask_epi8(hits) as u32 as u64) << (chunk * LANES);
            m.quote |= bit(_mm256_cmpeq_epi8(v, quote));
            m.backslash |= bit(_mm256_cmpeq_epi8(v, backslash));
            m.open |= bit(_mm256_or_si256(
                _mm256_cmpeq_epi8(v, obj_open),
                _mm256_cmpeq_epi8(v, arr_open),
            ));
            m.close |= bit(_mm256_or_si256(
                _mm256_cmpeq_epi8(v, obj_close),
                _mm256_cmpeq_epi8(v, arr_close),
            ));
        }
        m
    }
}

#[cfg(target_arch = "x86_64")]
mod sse2 {
    //! 16-bytes-at-a-time kernels, same shape as [`super::avx2`] but half the width.
    //!
    //! The point of these is *not* the vector width — it is that SSE2 is part of the
    //! x86-64 baseline, so these functions need no `#[target_feature]` and are ordinary
    //! safe `#[inline]` functions. LLVM inlines them straight into the tokenizer's state
    //! machine, which the AVX2 kernels can never be (a caller without `avx2` cannot inline
    //! a callee that requires it). For runs of a handful of bytes — JSON keys, small
    //! numbers — avoiding the call is worth far more than the extra 16 lanes.
    //!
    //! # Safety
    //! The intrinsics used here are all SSE2, which the `x86_64` target guarantees, so the
    //! only precondition is the in-bounds one: a 16-byte load happens only while
    //! `i + 16 <= len`.

    use super::Masks64;
    use crate::tokenizer::{ScalarScan, Scan};
    use std::arch::x86_64::*;

    const LANES: usize = 16;

    /// First byte at or after `from` that is `"`, `\`, or a control character (`< 0x20`).
    #[inline]
    pub fn string_event(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut i = from;
        // SAFETY: every load below is guarded by `i + LANES <= len`, and SSE2 is
        // unconditionally available on x86-64.
        unsafe {
            let quote = _mm_set1_epi8(b'"' as i8);
            let backslash = _mm_set1_epi8(b'\\' as i8);
            // `max_epu8(v, 0x1f) == 0x1f` iff `v <= 0x1f` unsigned.
            let ctrl_hi = _mm_set1_epi8(0x1f);
            while i + LANES <= len {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let hits = _mm_or_si128(
                    _mm_or_si128(_mm_cmpeq_epi8(v, quote), _mm_cmpeq_epi8(v, backslash)),
                    _mm_cmpeq_epi8(_mm_max_epu8(v, ctrl_hi), ctrl_hi),
                );
                let mask = _mm_movemask_epi8(hits) as u32;
                if mask != 0 {
                    return i + mask.trailing_zeros() as usize;
                }
                i += LANES;
            }
        }
        ScalarScan.string_event(data, i)
    }

    /// First byte at or after `from` that is not JSON whitespace.
    #[inline]
    pub fn skip_ws(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut i = from;
        // SAFETY: as above.
        unsafe {
            let space = _mm_set1_epi8(b' ' as i8);
            let tab = _mm_set1_epi8(b'\t' as i8);
            let cr = _mm_set1_epi8(b'\r' as i8);
            let lf = _mm_set1_epi8(b'\n' as i8);
            while i + LANES <= len {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let ws = _mm_or_si128(
                    _mm_or_si128(_mm_cmpeq_epi8(v, space), _mm_cmpeq_epi8(v, tab)),
                    _mm_or_si128(_mm_cmpeq_epi8(v, cr), _mm_cmpeq_epi8(v, lf)),
                );
                // Only the low 16 bits are lanes; invert just those.
                let mask = !(_mm_movemask_epi8(ws) as u32) & 0xFFFF;
                if mask != 0 {
                    return i + mask.trailing_zeros() as usize;
                }
                i += LANES;
            }
        }
        ScalarScan.skip_ws(data, i)
    }

    /// First byte at or after `from` that is not an ASCII digit.
    #[inline]
    pub fn skip_digits(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut i = from;
        // SAFETY: as above.
        unsafe {
            let zero = _mm_set1_epi8(b'0' as i8);
            let nine = _mm_set1_epi8(9);
            while i + LANES <= len {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                // Wrapping `v - '0'`, then unsigned `<= 9`; bytes below '0' wrap high and
                // so correctly fail the test.
                let shifted = _mm_sub_epi8(v, zero);
                let is_digit = _mm_cmpeq_epi8(_mm_max_epu8(shifted, nine), nine);
                let mask = !(_mm_movemask_epi8(is_digit) as u32) & 0xFFFF;
                if mask != 0 {
                    return i + mask.trailing_zeros() as usize;
                }
                i += LANES;
            }
        }
        ScalarScan.skip_digits(data, i)
    }

    /// First byte at or after `from` that is `{`, `}`, `[`, `]`, or `"`.
    #[inline]
    pub fn structural_event(data: &[u8], from: usize) -> usize {
        let len = data.len();
        let ptr = data.as_ptr();
        let mut i = from;
        // SAFETY: as above.
        unsafe {
            let obj_open = _mm_set1_epi8(b'{' as i8);
            let obj_close = _mm_set1_epi8(b'}' as i8);
            let arr_open = _mm_set1_epi8(b'[' as i8);
            let arr_close = _mm_set1_epi8(b']' as i8);
            let quote = _mm_set1_epi8(b'"' as i8);
            while i + LANES <= len {
                let v = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let hits = _mm_or_si128(
                    _mm_or_si128(_mm_cmpeq_epi8(v, obj_open), _mm_cmpeq_epi8(v, obj_close)),
                    _mm_or_si128(
                        _mm_or_si128(_mm_cmpeq_epi8(v, arr_open), _mm_cmpeq_epi8(v, arr_close)),
                        _mm_cmpeq_epi8(v, quote),
                    ),
                );
                let mask = _mm_movemask_epi8(hits) as u32;
                if mask != 0 {
                    return i + mask.trailing_zeros() as usize;
                }
                i += LANES;
            }
        }
        ScalarScan.structural_event(data, i)
    }

    /// Classify 64 bytes at `at` into [`Masks64`], four 16-byte loads at a time.
    ///
    /// # Panics
    /// Debug-asserts `at + 64 <= data.len()`; the caller guarantees it.
    #[inline(always)]
    pub fn classify64(data: &[u8], at: usize) -> Masks64 {
        debug_assert!(at + 64 <= data.len());
        let ptr = data.as_ptr();
        let mut m = Masks64::default();
        // SAFETY: `at + 64 <= len`, so each of the four 16-byte loads is in bounds; SSE2 is
        // unconditionally available on x86-64.
        unsafe {
            let quote = _mm_set1_epi8(b'"' as i8);
            let backslash = _mm_set1_epi8(b'\\' as i8);
            let obj_open = _mm_set1_epi8(b'{' as i8);
            let obj_close = _mm_set1_epi8(b'}' as i8);
            let arr_open = _mm_set1_epi8(b'[' as i8);
            let arr_close = _mm_set1_epi8(b']' as i8);
            for chunk in 0..4 {
                let v = _mm_loadu_si128(ptr.add(at + chunk * LANES) as *const __m128i);
                let bit = |hits| (_mm_movemask_epi8(hits) as u32 as u64) << (chunk * LANES);
                m.quote |= bit(_mm_cmpeq_epi8(v, quote));
                m.backslash |= bit(_mm_cmpeq_epi8(v, backslash));
                m.open |= bit(_mm_or_si128(
                    _mm_cmpeq_epi8(v, obj_open),
                    _mm_cmpeq_epi8(v, arr_open),
                ));
                m.close |= bit(_mm_or_si128(
                    _mm_cmpeq_epi8(v, obj_close),
                    _mm_cmpeq_epi8(v, arr_close),
                ));
            }
        }
        m
    }
}

/// One 64-byte window of input, classified into bitmasks: bit `i` describes byte `at + i`.
///
/// 64 bytes regardless of vector width — SSE2 fills it from four loads, AVX2 from two — so
/// that every bit-level algorithm below is written once, against `u64`, and is shared and
/// tested independently of which kernel produced the masks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Masks64 {
    pub(crate) quote: u64,
    pub(crate) backslash: u64,
    /// `{` or `[`.
    pub(crate) open: u64,
    /// `}` or `]`.
    pub(crate) close: u64,
}

/// Carried state between consecutive [`Masks64`] windows.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SkipCarry {
    /// Whether byte 0 of the next window is escaped by a backslash run ending in this one.
    escaped: u64,
    /// All-ones while the next window starts inside a string literal, else zero.
    in_string: u64,
}

/// Bits holding a character that a preceding backslash escapes.
///
/// The subtlety is runs: in `\\\\"` the quote is *not* escaped, because the backslashes pair
/// off. So a backslash only escapes the next byte if it is not itself escaped, which makes
/// the property depend on the parity of the run it sits in — and runs can straddle the
/// 64-byte window, which is what `carry` tracks.
///
/// Parity comes out of an addition rather than a loop: adding the run-start bits to the
/// backslash bits makes each run carry into the byte just past its end, and whether that
/// landing bit is odd or even reveals the run's length parity.
#[inline(always)]
fn find_escaped(backslash: u64, carry: &mut u64) -> u64 {
    // A backslash that is itself escaped starts nothing.
    let backslash = backslash & !*carry;
    let follows_escape = (backslash << 1) | *carry;
    const EVEN_BITS: u64 = 0x5555_5555_5555_5555;

    let odd_sequence_starts = backslash & !EVEN_BITS & !follows_escape;
    let (sequences_starting_on_even_bits, overflow) =
        odd_sequence_starts.overflowing_add(backslash);
    *carry = u64::from(overflow);
    let invert_mask = sequences_starting_on_even_bits << 1;

    (EVEN_BITS ^ invert_mask) & follows_escape
}

/// XOR-scan: bit `i` of the result is the XOR of bits `0..=i` of `x`.
///
/// Applied to the real (unescaped) quote bits this yields the string interior: the bit is
/// set from an opening quote up to, but not including, its closing quote. Six shift/XOR
/// pairs, so no `pclmulqdq` and no feature beyond the baseline.
#[inline(always)]
fn prefix_xor(mut x: u64) -> u64 {
    x ^= x << 1;
    x ^= x << 2;
    x ^= x << 4;
    x ^= x << 8;
    x ^= x << 16;
    x ^= x << 32;
    x
}

/// Fold one window into the running depth, returning `Some(bit)` if the container closed
/// inside it — `bit` being the index of the closing bracket.
///
/// `depth` counts containers still open, including the one being skipped, so it reaching
/// zero is the terminating condition.
///
/// The fast path is the point of the whole exercise. If `depth` exceeds the number of
/// closers in the window, no prefix of it can drive the depth to zero — even were every
/// closer to come first — so the exact positions never need to be examined and the window
/// collapses to two `popcnt`s. On a large skipped value almost every window takes it.
#[inline(always)]
fn fold_window(
    m: Masks64,
    carry: &mut SkipCarry,
    depth: &mut usize,
    outer: usize,
) -> Result<Option<u32>, SkipError> {
    let escaped = find_escaped(m.backslash, &mut carry.escaped);
    let quotes = m.quote & !escaped;
    let in_string = prefix_xor(quotes) ^ carry.in_string;
    // Sign-extend the top bit: all-ones iff the window ends inside a string.
    carry.in_string = ((in_string as i64) >> 63) as u64;

    let open = m.open & !in_string;
    let close = m.close & !in_string;
    let opens = open.count_ones() as usize;
    let closes = close.count_ones() as usize;

    if *depth > closes && outer + *depth + opens < MAX_SKIP_DEPTH {
        *depth = *depth + opens - closes;
        return Ok(None);
    }

    // The window can close the container (or breach the depth ceiling), so walk its
    // brackets in order.
    let mut bits = open | close;
    while bits != 0 {
        let i = bits.trailing_zeros();
        if (close >> i) & 1 == 1 {
            *depth -= 1;
            if *depth == 0 {
                return Ok(Some(i));
            }
        } else {
            *depth += 1;
            if outer + *depth >= MAX_SKIP_DEPTH {
                return Err(SkipError::TooDeep);
            }
        }
        bits &= bits - 1;
    }
    Ok(None)
}

/// [`Scan::skip_container`] for a backend that can classify a 64-byte window.
///
/// Whole windows are folded by [`fold_window`]; the sub-window tail is handed to the
/// portable byte walk, which is also the reference this is tested against. The tail can only
/// be entered outside a string — a window ending mid-string leaves `carry.in_string` set, and
/// the string is crossed byte-wise first — so the two halves agree on where they are.
#[inline(always)]
pub(crate) fn skip_container_blocked<S: Scan>(
    scan: &S,
    classify: impl Fn(&[u8], usize) -> Masks64,
    data: &[u8],
    from: usize,
    outer: usize,
) -> Result<usize, SkipError> {
    const W: usize = 64;
    let len = data.len();
    let mut pos = from;
    let mut depth = 1usize;
    let mut carry = SkipCarry::default();

    while pos + W <= len {
        if let Some(bit) = fold_window(classify(data, pos), &mut carry, &mut depth, outer)? {
            return Ok(pos + bit as usize + 1);
        }
        pos += W;
    }

    // A string straddling the end of the last full window would leave the byte-wise tail
    // believing it starts outside one, so finish it here. `carry.escaped` matters: if the
    // window ended on an unpaired backslash, the byte the tail starts on is escaped, and a
    // `"` there closes nothing.
    if carry.in_string != 0 {
        pos = cross_string_tail(scan, data, pos, carry.escaped != 0)?;
    }

    // `skip_container_bytewise` closes exactly one container per call, so run it once per
    // container still open. Sequential, not nested: each call returns just past the
    // innermost closer, leaving the cursor inside the next one out.
    for _ in 0..depth {
        pos = skip_container_bytewise(scan, data, pos, outer)?;
    }
    Ok(pos)
}

/// Advance past the remainder of a string literal already in progress at `from`.
///
/// `starts_escaped` says the byte at `from` is the target of a backslash that ended the
/// previous window — it is string content whatever it is, so a `"` there does not close.
#[inline(always)]
fn cross_string_tail<S: Scan>(
    scan: &S,
    data: &[u8],
    from: usize,
    starts_escaped: bool,
) -> Result<usize, SkipError> {
    let len = data.len();
    let mut pos = from + usize::from(starts_escaped);
    if pos > len {
        return Err(SkipError::Unterminated);
    }
    loop {
        pos = scan.string_event(data, pos);
        if pos >= len {
            return Err(SkipError::Unterminated);
        }
        match data[pos] {
            b'"' => return Ok(pos + 1),
            b'\\' => pos += 2,
            _ => pos += 1,
        }
        if pos > len {
            return Err(SkipError::Unterminated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::ScalarScan;

    /// The vector width the widest kernel works in; buffers are sized around it so both the
    /// 16- and 32-byte kernels see full blocks, partial tails, and empty tails.
    const W: usize = 32;

    /// Check one scanner against [`ScalarScan`], which is the behavioural reference.
    ///
    /// Exhaustive rather than sampled: every start offset and every length up to three
    /// vectors, which is where boundary bugs live (a hit in the last lane of a block, a hit
    /// in the scalar tail, an empty tail, a buffer shorter than one vector).
    fn check_every_offset_and_length<S: Scan>(name: &str, scan: S) {
        // Byte soup containing every interesting class, arranged so that as the window
        // slides, hits land at every position modulo both vector widths.
        let alphabet: &[u8] = b"a\" \\\t0\n9\r\x00\x1f{,}z:1\"\\ 7[]}";
        let buf: Vec<u8> = (0..W * 3 + 5)
            .map(|i| alphabet[i % alphabet.len()])
            .collect();

        for len in 0..=buf.len() {
            let data = &buf[..len];
            for from in 0..=len {
                assert_eq!(
                    scan.string_event(data, from),
                    ScalarScan.string_event(data, from),
                    "{name}: string_event len={len} from={from}"
                );
                assert_eq!(
                    scan.skip_ws(data, from),
                    ScalarScan.skip_ws(data, from),
                    "{name}: skip_ws len={len} from={from}"
                );
                assert_eq!(
                    scan.skip_digits(data, from),
                    ScalarScan.skip_digits(data, from),
                    "{name}: skip_digits len={len} from={from}"
                );
                assert_eq!(
                    scan.structural_event(data, from),
                    ScalarScan.structural_event(data, from),
                    "{name}: structural_event len={len} from={from}"
                );
            }
        }
    }

    /// Plant the single byte the predicate stops on at every position in turn, so every lane
    /// of every block is checked as *the* hit — a stride or mask bug that a dense soup hides
    /// (because it always hits inside the first block) fails here.
    fn check_hit_in_every_lane<S: Scan>(name: &str, scan: S) {
        let len = W * 2 + 7;
        for at in 0..len {
            for (filler, needle) in [(b'x', b'"'), (b'x', 0x1f), (b'5', b'x')] {
                let mut buf = vec![filler; len];
                buf[at] = needle;
                let got = if filler == b'5' {
                    scan.skip_digits(&buf, 0)
                } else {
                    scan.string_event(&buf, 0)
                };
                assert_eq!(got, at, "{name}: needle {needle:#04x} at {at}");
            }
            // Each of the five structural bytes must be found in every lane. `x` filler is
            // not structural, so any hit before `at` is a false positive.
            for needle in [b'{', b'}', b'[', b']', b'"'] {
                let mut buf = vec![b'x'; len];
                buf[at] = needle;
                assert_eq!(
                    scan.structural_event(&buf, 0),
                    at,
                    "{name}: structural {needle:#04x} at {at}"
                );
            }
            // Whitespace is the inverted predicate, so it gets the same treatment.
            let mut buf = vec![b' '; len];
            buf[at] = b'x';
            assert_eq!(scan.skip_ws(&buf, 0), at, "{name}: ws stop at {at}");
        }
        // A run that never terminates must report the end of input, not a false hit.
        assert_eq!(
            scan.skip_ws(&vec![b'\t'; len], 0),
            len,
            "{name}: all whitespace"
        );
        assert_eq!(
            scan.string_event(&vec![b'x'; len], 0),
            len,
            "{name}: no event"
        );
        assert_eq!(
            scan.structural_event(&vec![b'x'; len], 0),
            len,
            "{name}: no structural event"
        );
    }

    /// The kernels use range tricks (`<= 0x1f`, `'0'..='9'`), so every one of the 256 byte
    /// values is classified against the reference — in the vector path, not the tail.
    fn check_every_byte_value<S: Scan>(name: &str, scan: S) {
        for b in 0u8..=255 {
            let mut buf = vec![b'|'; W];
            buf[0] = b;
            buf.extend_from_slice(&[b'|'; W]);
            assert_eq!(
                scan.string_event(&buf, 0),
                ScalarScan.string_event(&buf, 0),
                "{name}: string_event byte {b:#04x}"
            );

            let mut buf = vec![b' '; W * 2];
            buf[0] = b;
            assert_eq!(
                scan.skip_ws(&buf, 0),
                ScalarScan.skip_ws(&buf, 0),
                "{name}: skip_ws byte {b:#04x}"
            );

            let mut buf = vec![b'0'; W * 2];
            buf[0] = b;
            assert_eq!(
                scan.skip_digits(&buf, 0),
                ScalarScan.skip_digits(&buf, 0),
                "{name}: skip_digits byte {b:#04x}"
            );
        }
    }

    /// Run the whole battery against a scanner.
    fn check_all<S: Scan>(name: &str, scan: S) {
        check_every_offset_and_length(name, scan);
        check_hit_in_every_lane(name, scan);
        check_every_byte_value(name, scan);
    }

    #[test]
    fn scalar_reference_is_self_consistent() {
        check_all("scalar", ScalarScan);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sse2_matches_scalar() {
        check_all("sse2", Sse2Scan);
    }

    /// Skipped rather than failed on a CPU without AVX2 — but [`avx2_backend_is_available`]
    /// asserts it *is* available on this machine, so a silent skip cannot hide a regression
    /// in CI on capable hardware.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar() {
        if !Backend::available().contains(&Backend::Avx2) {
            return;
        }
        check_all("avx2", <Avx2Scan as Scan>::new());
    }

    #[test]
    fn detection_picks_a_guaranteed_backend() {
        let picked = Backend::detect();
        assert!(
            Backend::available().contains(&picked),
            "detect() picked {picked:?}, which this CPU cannot run"
        );
        #[cfg(target_arch = "x86_64")]
        {
            // SSE2 is in the x86-64 base ABI, so a vector backend needs no detection and
            // this must never fall back to the portable scanner.
            assert_ne!(picked, Backend::Scalar);
            // Which vector backend depends only on AVX2, and `Hybrid` needs it for
            // `skip_container` exactly as `Avx2` does.
            let expected = if std::arch::is_x86_feature_detected!("avx2") {
                Backend::Hybrid
            } else {
                Backend::Sse2
            };
            assert_eq!(picked, expected);
        }
    }

    /// Guards the `return` in [`avx2_matches_scalar`]: on this development machine AVX2 is
    /// present, so if this ever fails the AVX2 tests have quietly stopped running.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_backend_is_available() {
        assert!(
            std::arch::is_x86_feature_detected!("avx2"),
            "expected AVX2 on this machine; the AVX2 tests would be silently skipped"
        );
    }

    /// Every skip the vector backends perform must land exactly where the portable byte walk
    /// lands — that walk is the reference, and the only reason the block counter's bit math
    /// (escape-run parity, string masking, the popcount fast path) is trustworthy.
    ///
    /// Layout matters more than it looks. The filler goes *inside* the container, before and
    /// after the interesting bytes: the leading run slides those bytes across the 64-byte
    /// window at every alignment, and the trailing run guarantees they are covered by *full*
    /// windows rather than falling into the byte-wise tail. An earlier version of this test
    /// padded before the opening bracket, which left every document too short to enter the
    /// block path at all — it passed against four deliberately broken kernels.
    fn check_skip_matches_bytewise<S: Scan>(name: &str, scan: S) {
        // Balanced container *contents* — no trailing `}`, which the harness appends.
        let bodies: &[&str] = &[
            r#""}]}]":1"#,
            r#""a":1,"b":[1,2,{"c":3}],"d":"}""#,
            r#""esc":"\"}]","x":[[[]]],"y":"\\""#,
            // Backslash runs of every parity: an off-by-one in escape parity flips a quote
            // between content and terminator, which moves where the container appears to end.
            r#""a":"\\","b":"\\\\","c":"\\\\\\","d":"\"","e":"\\\"""#,
            // A string of nothing but brackets — the popcount fast path must not see them.
            r#""s":"{{{{{{{{{{[[[[[[[[[[}}}}}}}}}}]]]]]]]]]]""#,
            r#"[[[[[[[[[[]]]]]]]]]],"z":0"#,
            // A lone backslash-escaped quote, the case the window carry exists for.
            r#""k":"\"""#,
            // Closers bunched together, so a window can begin with exactly as many closers
            // as there are open containers — the boundary the fast-path guard turns on.
            r#"[[[[[["a"]]]]]]"#,
            r#"{"a":{"b":{"c":{"d":1}}}}"#,
        ];

        for body in bodies {
            for pad in 0..=(64 * 2 + 5) {
                let mut doc = vec![b'{'];
                doc.extend(std::iter::repeat_n(b' ', pad));
                doc.extend(body.bytes());
                // Enough filler that the body always sits inside full windows.
                doc.extend(std::iter::repeat_n(b' ', 160));
                doc.push(b'}');
                // And enough *after* the close that the close itself does too. Without this
                // the container always ended in the byte-wise tail, and the popcount fast
                // path's guard condition was never actually exercised.
                doc.extend(std::iter::repeat_n(b' ', 160));
                let from = 1;

                assert_eq!(
                    scan.skip_container(&doc, from, 0),
                    skip_container_bytewise(&ScalarScan, &doc, from, 0),
                    "{name}: pad={pad} body={body:?}"
                );

                // Truncation is where an unterminated string or a half-consumed escape at the
                // buffer end shows up. Strided, since the product of every pad and every cut
                // is large and neighbouring cuts test the same thing.
                for cut in (from..doc.len()).step_by(13) {
                    let part = &doc[..cut];
                    assert_eq!(
                        scan.skip_container(part, from, 0),
                        skip_container_bytewise(&ScalarScan, part, from, 0),
                        "{name}: truncated to {cut}, pad={pad}, body={body:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn skip_matches_bytewise_on_every_backend() {
        // Not just the detected backend: each is separately compiled, so running one leaves
        // the others genuinely untested.
        for backend in Backend::available() {
            match backend {
                Backend::Scalar => {}
                #[cfg(target_arch = "x86_64")]
                Backend::Sse2 => check_skip_matches_bytewise("sse2", Sse2Scan),
                #[cfg(target_arch = "x86_64")]
                Backend::Avx2 => <Avx2Scan as Scan>::enter(|| {
                    check_skip_matches_bytewise("avx2", <Avx2Scan as Scan>::new())
                }),
                // No `enter`: `HybridScan::skip_container` opens its own AVX2 context, and
                // calling it from outside one is exactly how the matcher calls it.
                #[cfg(target_arch = "x86_64")]
                Backend::Hybrid => {
                    check_skip_matches_bytewise("hybrid", <HybridScan as Scan>::new())
                }
            }
        }
    }

    #[test]
    fn escape_and_string_bit_math_matches_a_scalar_model() {
        // prefix_xor: bit i is the parity of bits 0..=i.
        for x in [0u64, 1, 0b1010, u64::MAX, 1 << 63, (1 << 63) | 1] {
            let mut want = 0u64;
            let mut acc = false;
            for i in 0..64 {
                acc ^= (x >> i) & 1 == 1;
                want |= u64::from(acc) << i;
            }
            assert_eq!(prefix_xor(x), want, "prefix_xor({x:#x})");
        }

        // find_escaped over every backslash pattern in the low 12 bits, one window.
        for bits in 0u64..(1 << 12) {
            let mut carry = 0u64;
            let got = find_escaped(bits, &mut carry);
            // Model: walk bytes; a backslash escapes the next byte unless itself escaped.
            let mut want = 0u64;
            let mut escaped = false;
            for i in 0..12 {
                if escaped {
                    want |= 1 << i;
                }
                escaped = !escaped && (bits >> i) & 1 == 1;
            }
            assert_eq!(got & 0xFFF, want, "find_escaped({bits:#x})");
        }

        // A run straddling two windows: `\` as the last byte must escape byte 0 of the next.
        let mut carry = 0u64;
        find_escaped(1 << 63, &mut carry);
        assert_eq!(carry, 1, "a trailing backslash must carry");
        let next = find_escaped(0, &mut carry);
        assert_eq!(next & 1, 1, "byte 0 of the next window is escaped");
        assert_eq!(carry, 0, "and the carry is then spent");
    }
}
