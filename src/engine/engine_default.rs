use crate::engine::{EngineType, NoSimd, Engine, GfElement, ShardsRefMut, GF_ORDER};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::engine::{Avx2, Ssse3};

#[cfg(target_arch = "aarch64")]
use crate::engine::Neon;

#[cfg(target_arch = "wasm32")]
use crate::engine::Wasm;

use core::sync::atomic::{AtomicU8, Ordering};
static DETECTED_ENGINE: AtomicU8 = AtomicU8::new(EngineType::Uninitialized as u8);

#[inline(always)]
/// Warning: This could actually cause issues if running on CPUs with
/// heterogeneous core types. For example, on x86 where bigger cores support
/// AVX2 but smaller cores don't. Or on ARM when some cores support Neon but
/// others don't.
pub fn get_detected_engine() -> EngineType {
    // 1. FAST PATH: Single raw CPU read instruction.
    let current = DETECTED_ENGINE.load(Ordering::Relaxed);
    
    if current != EngineType::Uninitialized as u8 {
        return unsafe { core::mem::transmute(current) };
    }

    // 2. SLOW PATH: Run your detection logic (only happens at startup)
    let detected = detect_optimal_engine();

    DETECTED_ENGINE.store(detected as u8, Ordering::Relaxed);
    
    detected
}

/// On x86(-64) the engine is chosen in the following order of preference:
/// 1. [`Avx2`]
/// 2. [`Ssse3`]
/// 3. [`NoSimd`]
///
/// On `AArch64` the engine is chosen in the following order of preference:
/// 1. [`Neon`]
/// 2. [`NoSimd`]
///
/// On `wasm32` the engine is chosen in the following order of preference:
/// 1. [`Wasm`] (if WebAssembly SIMD128 is supported at runtime)
/// 2. [`NoSimd`]
pub(crate) fn detect_optimal_engine() -> EngineType {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        cpufeatures::new!(has_avx2, "avx2");
        if has_avx2::get() {
            return EngineType::Avx2;
        }

        cpufeatures::new!(has_ssse3, "ssse3");
        if has_ssse3::get() {
            return EngineType::Ssse3;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        cpufeatures::new!(has_neon, "neon");
        if has_neon::get() {
            return EngineType::Neon;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        if Wasm::wasm_simd128_supported() {
            return EngineType::Wasm;
        }
    }

    EngineType::NoSimd
}

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] that at runtime selects the best Engine.
#[derive(Default)]
pub struct DefaultEngine;

// ======================================================================
// DefaultEngine - IMPL Engine

impl Engine for DefaultEngine {
    #[inline(always)]
    fn fft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::fft(data, pos, size, truncated_size, skew_delta),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::fft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::fft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::fft(data, pos, size, truncated_size, skew_delta),
            EngineType::NoSimd => NoSimd::fft(data, pos, size, truncated_size, skew_delta),
        };
    }

    #[inline(always)]
    fn ifft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::ifft(data, pos, size, truncated_size, skew_delta),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::ifft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::ifft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::ifft(data, pos, size, truncated_size, skew_delta),
            EngineType::NoSimd => NoSimd::ifft(data, pos, size, truncated_size, skew_delta),
        };
    }

    #[inline(always)]
    fn mul(x: &mut [[u8; 64]], log_m: GfElement) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::mul(x, log_m),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::mul(x, log_m),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::mul(x, log_m),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::mul(x, log_m),
            EngineType::NoSimd => NoSimd::mul(x, log_m),
        };
    }

    #[inline(always)]
    fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::xor(xs, ys),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::xor(xs, ys),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::xor(xs, ys),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::xor(xs, ys),
            EngineType::NoSimd => NoSimd::xor(xs, ys),
        };
    }

    #[inline(always)]
    fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::xor_within(data, x, y, count),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::xor_within(data, x, y, count),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::xor_within(data, x, y, count),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::xor_within(data, x, y, count),
            EngineType::NoSimd => NoSimd::xor_within(data, x, y, count),
        };
    }

    #[inline(always)]
    fn formal_derivative(data: &mut ShardsRefMut) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::formal_derivative(data),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::formal_derivative(data),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::formal_derivative(data),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::formal_derivative(data),
            EngineType::NoSimd => NoSimd::formal_derivative(data),
        };
    }

    #[inline(always)]
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        match get_detected_engine() {
            EngineType::Uninitialized => unreachable!("An engine was not selected"),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Avx2 => Avx2::eval_poly(erasures, truncated_size),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            EngineType::Ssse3 => Ssse3::eval_poly(erasures, truncated_size),
            #[cfg(target_arch = "aarch64")]
            EngineType::Neon => Neon::eval_poly(erasures, truncated_size),
            #[cfg(target_arch = "wasm32")]
            EngineType::Wasm => Wasm::eval_poly(erasures, truncated_size),
            EngineType::NoSimd => NoSimd::eval_poly(erasures, truncated_size),
        };
    }
}
