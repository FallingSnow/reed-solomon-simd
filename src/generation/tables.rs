// The generation folder is used by both build.rs and the crate itself. While
// some code is shared, other code is unique to build.rs or the crate. This
// causes unused code warnings.
#![allow(dead_code)]

use crate::generation::fwht;

use super::constants::{GfElement, GF_BITS, GF_MODULUS, GF_ORDER};
use super::utils::add_mod;

/// Galois field polynomial.
const GF_POLYNOMIAL: usize = 0x1002D;

/// TODO
const CANTOR_BASIS: [GfElement; GF_BITS] = [
    0x0001, 0xACCA, 0x3C0E, 0x163E, 0xC582, 0xED2E, 0x914C, 0x4012, 0x6C98, 0x10D8, 0x6A72, 0xB900,
    0xFDB8, 0xFB34, 0xFF38, 0x991E,
];

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
type Exp = [GfElement; GF_ORDER];

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
type Log = [GfElement; GF_ORDER];

/// Used by [`Avx2`] and [`Ssse3`] engines for multiplications.
///
/// [`Avx2`]: crate::engine::Avx2
/// [`Ssse3`]: crate::engine::Ssse3
type Mul128 = [Multiply128lutT; GF_ORDER];

/// Elements of the Mul128 table
#[derive(Clone, Debug)]
struct Multiply128lutT {
    /// Lower half of `GfElements`
    pub lo: [u128; 4],
    /// Upper half of `GfElements`
    pub hi: [u128; 4],
}

/// Used by all [`Engine`]:s in [`Engine::eval_poly`].
///
/// [`Engine`]: crate::engine
/// [`Engine::eval_poly`]: crate::engine::Engine::eval_poly
type LogWalsh = [GfElement; GF_ORDER];

/// Used by [`NoSimd`] engine for multiplications.
///
/// [`NoSimd`]: crate::engine::NoSimd
type Mul16 = [[[GfElement; 16]; 4]; GF_ORDER];

/// Used by all [`Engine`]:s for FFT and IFFT.
///
/// [`Engine`]: crate::engine
type Skew = [GfElement; GF_MODULUS as usize];

/// Struct holding the [`Exp`] and [`Log`] lookup tables.
struct ExpLog {
    /// Exponentiation table.
    pub exp: Box<Exp>,
    /// Logarithm table.
    pub log: Box<Log>,
}

/// Calculates `x * log_m` using [`Exp`] and [`Log`] tables.
#[inline(always)]
fn mul(x: GfElement, log_m: GfElement, exp: &Exp, log: &Log) -> GfElement {
    if x == 0 {
        0
    } else {
        exp[add_mod(log[x as usize], log_m) as usize]
    }
}

// ======================================================================
// FUNCTIONS - PRIVATE - initialize tables

#[allow(clippy::needless_range_loop)]
fn initialize_exp_log() -> ExpLog {
    let mut exp = Box::new([0; GF_ORDER]);
    let mut log = Box::new([0; GF_ORDER]);

    // GENERATE LFSR TABLE

    let mut state = 1;
    for i in 0..GF_MODULUS {
        exp[state] = i;
        state <<= 1;
        if state >= GF_ORDER {
            state ^= GF_POLYNOMIAL;
        }
    }
    exp[0] = GF_MODULUS;

    // CONVERT TO CANTOR BASIS

    log[0] = 0;
    for i in 0..GF_BITS {
        let width = 1usize << i;
        for j in 0..width {
            log[j + width] = log[j] ^ CANTOR_BASIS[i];
        }
    }

    for i in 0..GF_ORDER {
        log[i] = exp[log[i] as usize];
    }

    for i in 0..GF_ORDER {
        exp[log[i] as usize] = i as GfElement;
    }

    exp[GF_MODULUS as usize] = exp[0];

    ExpLog { exp, log }
}

fn initialize_log_walsh() -> Box<LogWalsh> {
    let exp_log = initialize_exp_log();
    let log = exp_log.log.as_slice();

    let mut log_walsh: Box<LogWalsh> = Box::new([0; GF_ORDER]);

    log_walsh.copy_from_slice(log);
    log_walsh[0] = 0;
    fwht::fwht(log_walsh.as_mut(), GF_ORDER);

    log_walsh
}

fn initialize_mul16() -> Box<Mul16> {
    let exp = &initialize_exp_log().exp;
    let log = &initialize_exp_log().log;
    let mut mul16 = vec![[[0; 16]; 4]; GF_ORDER];

    for log_m in 0..=GF_MODULUS {
        let lut = &mut mul16[log_m as usize];
        for i in 0..16 {
            lut[0][i] = mul(i as GfElement, log_m, exp, log);
            lut[1][i] = mul((i << 4) as GfElement, log_m, exp, log);
            lut[2][i] = mul((i << 8) as GfElement, log_m, exp, log);
            lut[3][i] = mul((i << 12) as GfElement, log_m, exp, log);
        }
    }

    mul16.into_boxed_slice().try_into().unwrap()
}

fn initialize_mul128() -> Box<Mul128> {
    // Based on:
    // https://github.com/catid/leopard/blob/22ddc7804998d31c8f1a2617ee720e063b1fa6cd/LeopardFF16.cpp#L375
    let exp = &initialize_exp_log().exp;
    let log = &initialize_exp_log().log;

    let mut mul128 = vec![
        Multiply128lutT {
            lo: [0; 4],
            hi: [0; 4],
        };
        GF_ORDER
    ];

    for log_m in 0..=GF_MODULUS {
        for i in 0..=3 {
            let mut prod_lo = [0u8; 16];
            let mut prod_hi = [0u8; 16];
            for x in 0..16 {
                let prod = mul((x << (i * 4)) as GfElement, log_m, exp, log);
                prod_lo[x] = prod as u8;
                prod_hi[x] = (prod >> 8) as u8;
            }
            mul128[log_m as usize].lo[i] = u128::from_le_bytes(prod_lo);
            mul128[log_m as usize].hi[i] = u128::from_le_bytes(prod_hi);
        }
    }

    mul128.into_boxed_slice().try_into().unwrap()
}

#[allow(clippy::needless_range_loop)]
fn initialize_skew() -> Box<Skew> {
    let exp = &initialize_exp_log().exp;
    let log = &initialize_exp_log().log;

    let mut skew = Box::new([0; GF_MODULUS as usize]);

    let mut temp = [0; GF_BITS - 1];

    for i in 1..GF_BITS {
        temp[i - 1] = 1 << i;
    }

    for m in 0..GF_BITS - 1 {
        let step: usize = 1 << (m + 1);

        skew[(1 << m) - 1] = 0;

        for i in m..GF_BITS - 1 {
            let s: usize = 1 << (i + 1);
            let mut j = (1 << m) - 1;
            while j < s {
                skew[j + s] = skew[j] ^ temp[i];
                j += step;
            }
        }

        temp[m] = GF_MODULUS - log[mul(temp[m], log[(temp[m] ^ 1) as usize], exp, log) as usize];

        for i in m + 1..GF_BITS - 1 {
            let sum = add_mod(log[(temp[i] ^ 1) as usize], temp[m]);
            temp[i] = mul(temp[i], sum, exp, log);
        }
    }

    for i in 0..GF_MODULUS as usize {
        skew[i] = log[skew[i] as usize];
    }
    skew
}

// ======================================================================
// FUNCTIONS - PUBLIC - code generation

/// Generate Rust source for the EXP static table.
pub fn generate_exp_source() -> String {
    let exp_log = initialize_exp_log();
    let values: Vec<String> = exp_log.exp.iter().map(|v| format!("{v}u16")).collect();
    format!(
        "/// Exponentiation lookup table.\npub static EXP: Exp = [{values}];\n\n",
        values = values.join(", ")
    )
}

/// Generate Rust source for the LOG static table.
pub fn generate_log_source() -> String {
    let exp_log = initialize_exp_log();
    let values: Vec<String> = exp_log.log.iter().map(|v| format!("{v}u16")).collect();
    format!(
        "/// Logarithm lookup table.\npub static LOG: Log = [{values}];\n\n",
        values = values.join(", ")
    )
}

/// Generate Rust source for the LOG_WALSH static table.
pub fn generate_log_walsh_source() -> String {
    let log_walsh = initialize_log_walsh();
    let values: Vec<String> = log_walsh.iter().map(|v| format!("{v}u16")).collect();
    format!("/// Walsh-transformed logarithm lookup table.\npub static LOG_WALSH: LogWalsh = [{values}];\n\n", values = values.join(", "))
}

/// Generate Rust source for the SKEW static table.
pub fn generate_skew_source() -> String {
    let skew = initialize_skew();
    let values: Vec<String> = skew.iter().map(|v| format!("{v}u16")).collect();
    format!(
        "/// Skew lookup table for FFT/IFFT.\npub static SKEW: Skew = [{values}];\n\n",
        values = values.join(", ")
    )
}

/// Generate Rust source for the MUL_16 static table.
pub fn generate_mul16_source() -> String {
    let mul16 = initialize_mul16();
    let entries: Vec<String> = mul16
        .iter()
        .map(|layer| {
            let rows: Vec<String> = layer
                .iter()
                .map(|row| {
                    let vals: Vec<String> = row.iter().map(|v| format!("{v}u16")).collect();
                    format!("[{}]", vals.join(", "))
                })
                .collect();
            format!("[{}]", rows.join(", "))
        })
        .collect();
    format!("/// Multiplication lookup table for NoSimd engine.\npub static MUL_16: Mul16 = [{entries}];\n\n", entries = entries.join(", "))
}

/// Generate Rust source for the MUL_128 static table.
pub fn generate_mul128_source() -> String {
    let mul128 = initialize_mul128();
    let entries: Vec<String> = mul128
        .iter()
        .map(|entry| {
            let lo_vals: Vec<String> = entry.lo.iter().map(|v| format!("{v}u128")).collect();
            let hi_vals: Vec<String> = entry.hi.iter().map(|v| format!("{v}u128")).collect();
            format!(
                "Multiply128lutT {{ lo: [{}], hi: [{}] }}",
                lo_vals.join(", "),
                hi_vals.join(", ")
            )
        })
        .collect();
    format!("/// Multiplication lookup table for SIMD engines.\npub static MUL_128: Mul128 = [{entries}];\n\n", entries = entries.join(", "))
}
