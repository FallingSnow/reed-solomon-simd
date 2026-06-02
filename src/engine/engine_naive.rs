use crate::constants::{GfElement, GF_MODULUS};
use crate::engine::{tables, utils, Engine, ShardsRefMut};

// ======================================================================
// Naive - PUBLIC

/// Simple reference implementation of [`Engine`].
///
/// - [`Naive`] is meant for those who want to study
///   the source code to understand [`Engine`].
/// - [`Naive`] also includes some debug assertions
///   which are not present in other implementations.
#[derive(Clone, Copy)]
pub struct Naive;

impl Engine for Naive {
    fn fft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        debug_assert!(size.is_power_of_two());
        debug_assert!(truncated_size <= size);

        let mut dist = size / 2;
        while dist > 0 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = tables::SKEW[r + dist + skew_delta - 1];
                for i in r..r + dist {
                    let (a, b) = data.dist2_mut(pos + i, dist);

                    // FFT BUTTERFLY

                    if log_m != GF_MODULUS {
                        Self::mul_add(a, b, log_m);
                    }
                    Self::xor(b, a);
                }
                r += dist * 2;
            }
            dist /= 2;
        }
    }

    fn ifft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        debug_assert!(size.is_power_of_two());
        debug_assert!(truncated_size <= size);

        let mut dist = 1;
        while dist < size {
            let mut r = 0;
            while r < truncated_size {
                let log_m = tables::SKEW[r + dist + skew_delta - 1];
                for i in r..r + dist {
                    let (a, b) = data.dist2_mut(pos + i, dist);

                    // IFFT BUTTERFLY

                    Self::xor(b, a);
                    if log_m != GF_MODULUS {
                        Self::mul_add(a, b, log_m);
                    }
                }
                r += dist * 2;
            }
            dist *= 2;
        }
    }

    fn mul(x: &mut [[u8; 64]], log_m: GfElement) {
        for chunk in x.iter_mut() {
            for i in 0..32 {
                let lo = GfElement::from(chunk[i]);
                let hi = GfElement::from(chunk[i + 32]);
                let prod = utils::mul(lo | (hi << 8), log_m);
                chunk[i] = prod as u8;
                chunk[i + 32] = (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// Naive - PRIVATE

impl Naive {
    /// `x[] ^= y[] * log_m`
    #[inline(always)]
    fn mul_add(x: &mut [[u8; 64]], y: &[[u8; 64]], log_m: GfElement) {
        debug_assert_eq!(x.len(), y.len());

        for (x_chunk, y_chunk) in core::iter::zip(x.iter_mut(), y.iter()) {
            for i in 0..32 {
                let lo = GfElement::from(y_chunk[i]);
                let hi = GfElement::from(y_chunk[i + 32]);
                let prod = utils::mul(lo | (hi << 8), log_m);
                x_chunk[i] ^= prod as u8;
                x_chunk[i + 32] ^= (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
