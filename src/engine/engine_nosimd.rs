use core::iter::zip;

use crate::constants::{GfElement, GF_MODULUS};
use crate::engine::{tables, Engine, ShardsRefMut};

// ======================================================================
// NoSimd - PUBLIC

/// Optimized [`Engine`] without SIMD.
///
/// [`NoSimd`] is a basic optimized engine which works on all CPUs.
#[derive(Clone, Copy, Default)]
pub struct NoSimd;

impl Engine for NoSimd {
    fn fft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        Self::fft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn ifft(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        Self::ifft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn mul(x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = tables::MUL_16[log_m as usize];

        for x_chunk in x.iter_mut() {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);

            for i in 0..32 {
                let lo = x_lo[i];
                let hi = x_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] = prod as u8;
                x_hi[i] = (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE

impl NoSimd {
    /// `x[] ^= y[] * log_m`
    #[inline(always)]
    fn mul_add(x: &mut [[u8; 64]], y: &[[u8; 64]], log_m: GfElement) {
        let lut = tables::MUL_16[log_m as usize];

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter()) {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);
            let (y_lo, y_hi) = y_chunk.split_at(32);

            for i in 0..32 {
                let lo = y_lo[i];
                let hi = y_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] ^= prod as u8;
                x_hi[i] ^= (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE - FFT (fast Fourier transform)

impl NoSimd {
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        Self::mul_add(x, y, log_m);
        Self::xor(y, x);
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            Self::fft_butterfly_partial(s0, s2, log_m02);
            Self::fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            Self::fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            Self::fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    #[inline(always)]
    fn fft_private(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = tables::SKEW[base];
                let log_m02 = tables::SKEW[base + dist];
                let log_m23 = tables::SKEW[base + dist * 2];

                for i in r..r + dist {
                    Self::fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = tables::SKEW[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    Self::xor(y, x);
                } else {
                    Self::fft_butterfly_partial(x, y, log_m);
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE - IFFT (inverse fast Fourier transform)

impl NoSimd {
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn ifft_butterfly_partial(x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        Self::xor(y, x);
        Self::mul_add(x, y, log_m);
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            Self::ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            Self::ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            Self::ifft_butterfly_partial(s0, s2, log_m02);
            Self::ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    #[inline(always)]
    fn ifft_private(
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = tables::SKEW[base];
                let log_m02 = tables::SKEW[base + dist];
                let log_m23 = tables::SKEW[base + dist * 2];

                for i in r..r + dist {
                    Self::ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = tables::SKEW[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                Self::xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    Self::ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.

#[cfg(test)]
mod tests {
    use crate::engine::{Engine, Naive, NoSimd};

    #[cfg(not(feature = "std"))]
    use alloc::vec;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn mul() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);

        for shard_chunks in 0..6 {
            let mut data_nosimd = vec![[0; 64]; shard_chunks];
            rng.fill(data_nosimd.as_flattened_mut());
            let mut data_naive = data_nosimd.clone();

            let log_m = rng.random();

            NoSimd::mul(&mut data_nosimd, log_m);
            Naive::mul(&mut data_naive, log_m);

            assert_eq!(data_nosimd, data_naive);
        }
    }
}
