//! Low-level building blocks for Reed-Solomon encoding/decoding.
//!
//! **This is an advanced module which is not needed for [simple usage] or [basic usage].**
//!
//! This module is relevant if you want to
//! - use [`rate`] module and need an [`Engine`] to use with it.
//! - create your own [`Engine`].
//! - understand/benchmark/test at low level.
//!
//! # Engines
//!
//! An [`Engine`] is an implementation of basic low-level algorithms
//! needed for Reed-Solomon encoding/decoding.
//!
//! - [`Naive`]
//!     - Simple reference implementation.
//! - [`NoSimd`]
//!     - Basic optimized engine without SIMD so that it works on all CPUs.
//! - [`Avx2`]
//!     - Optimized engine that takes advantage of the x86(-64) AVX2 SIMD instructions.
//! - [`Ssse3`]
//!     - Optimized engine that takes advantage of the x86(-64) SSSE3 SIMD instructions.
//! - [`Neon`]
//!     - Optimized engine that takes advantage of the `AArch64` Neon SIMD instructions.
//! - [`Wasm`]
//!     - Optimized engine that takes advantage of the WebAssembly SIMD128 instructions.
//! - [`DefaultEngine`]
//!     - Default engine which is used when no specific engine is given.
//!     - Automatically selects best engine at runtime.
//!
//! [simple usage]: crate#simple-usage
//! [basic usage]: crate#basic-usage
//! [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
//! [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
//! [`rate`]: crate::rate

pub(crate) use self::shards::Shards;
pub(crate) use utils::{fft_skew_end, ifft_skew_end};

pub use self::{
    engine_default::DefaultEngine, engine_naive::Naive, engine_nosimd::NoSimd, shards::ShardsRefMut,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::{engine_avx2::Avx2, engine_ssse3::Ssse3};

#[cfg(target_arch = "aarch64")]
pub use self::engine_neon::Neon;

#[cfg(target_arch = "wasm32")]
pub use self::engine_wasm::Wasm;

mod engine_default;
mod engine_naive;
mod engine_nosimd;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_ssse3;

#[cfg(target_arch = "aarch64")]
mod engine_neon;

#[cfg(target_arch = "wasm32")]
mod engine_wasm;

mod shards;

pub mod tables;
pub mod utils;

use crate::constants::{GfElement, GF_ORDER};

// ======================================================================
// Engine - PUBLIC

/// Trait for compute-intensive low-level algorithms needed
/// for Reed-Solomon encoding/decoding.
///
/// This is the trait you would implement to provide SIMD support
/// for a CPU architecture not already provided.
///
/// [`Naive`] engine is provided for those who want to
/// study the source code to understand [`Engine`].
pub trait Engine {
    // ============================================================
    // REQUIRED

    /// In-place decimation-in-time FFT (fast Fourier transform).
    ///
    /// - FFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid FFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid FFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    fn fft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) where
        Self: Sized;

    /// In-place decimation-in-time IFFT (inverse fast Fourier transform).
    ///
    /// - IFFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid IFFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid IFFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    fn ifft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) where
        Self: Sized;

    /// `x[] *= log_m`
    fn mul(x: &mut [[u8; 64]], log_m: GfElement)
    where
        Self: Sized;

    // ============================================================
    // PROVIDED

    /// Evaluate polynomial.
    #[inline(always)]
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize)
    where
        Self: Sized,
    {
        utils::eval_poly(erasures, truncated_size);
    }

    /// `x[] ^= y[]`
    #[inline(always)]
    fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]])
    where
        Self: Sized,
    {
        utils::xor(xs, ys);
    }

    /// `data[x .. x + count] ^= data[y .. y + count]`
    ///
    /// Ranges must not overlap.
    #[inline(always)]
    fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize)
    where
        Self: Sized,
    {
        let (xs, ys) = data.flat2_mut(x, y, count);
        Self::xor(xs, ys);
    }

    /// Formal derivative.
    #[inline(always)]
    fn formal_derivative(data: &mut ShardsRefMut)
    where
        Self: Sized,
    {
        for i in 1..data.len() {
            let width: usize = 1 << i.trailing_zeros();
            Self::xor_within(data, i - width, i, width);
        }
    }
}



#[derive(Default, Clone, Copy)]
#[repr(u8)]
pub(crate) enum EngineType {
    #[default]
    Uninitialized = 0,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2 = 1,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Ssse3 = 2,
    #[cfg(target_arch = "aarch64")]
    Neon = 3,
    #[cfg(target_arch = "wasm32")]
    Wasm = 4,
    NoSimd = 255,
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
