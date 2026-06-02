//! A collection of utility functions and helpers to facilitate the implementation of the [`Engine`] trait.
//!
//! [`Engine`]: crate::engine::Engine

use crate::engine::tables::{EXP, LOG};
use crate::engine::{Engine, ShardsRefMut, tables};
use crate::constants::{GF_BITS, GF_ORDER, GfElement};
use crate::generation::fwht;

use core::iter::zip;

// ======================================================================
// FUNCTIONS - PUBLIC

/// Evaluate Polynomial using Fast Walsh-Hadamard Transform (FWHT).
///
/// This function is designed to be inlined and be compiled with SIMD
/// features enabled within an Engine's implementation of `eval_poly`.
///
/// See [`Avx2`] for an example on how to do this.
///
/// [`Avx2`]: crate::engine::Avx2
#[inline(always)]
pub fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    let log_walsh = tables::LOG_WALSH;

    fwht::fwht(erasures, truncated_size);

    for (e, factor) in zip(erasures.iter_mut(), log_walsh.iter()) {
        let product = u32::from(*e) * u32::from(*factor);
        *e = add_mod(product as GfElement, (product >> GF_BITS) as GfElement);
    }

    fwht::fwht(erasures, GF_ORDER);
}

/// `x[] ^= y[]`
#[inline(always)]
pub fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
    debug_assert_eq!(xs.len(), ys.len());

    for (x_chunk, y_chunk) in zip(xs.iter_mut(), ys.iter()) {
        for (x, y) in zip(x_chunk.iter_mut(), y_chunk.iter()) {
            *x ^= y;
        }
    }
}

// ======================================================================
// FUNCTIONS - CRATE - Galois field operations

/// Some kind of addition.
#[inline(always)]
const fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = x as u32 + y as u32;
    (sum + (sum >> GF_BITS)) as GfElement
}
/// Calculates `x * log_m` using [`Exp`] and [`Log`] tables.
#[inline(always)]
pub fn mul(x: GfElement, log_m: GfElement) -> GfElement {
    if x == 0 {
        0
    } else {
        EXP[add_mod(LOG[x as usize], log_m) as usize]
    }
}

// ======================================================================
// FUNCTIONS - CRATE

/// FFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn fft_skew_end<E: Engine>(
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    E::fft(data, pos, size, truncated_size, pos + size);
}

/// IFFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn ifft_skew_end<E: Engine>(
    data: &mut ShardsRefMut,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    E::ifft(data, pos, size, truncated_size, pos + size);
}

