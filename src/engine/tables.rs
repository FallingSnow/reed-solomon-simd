//! Lookup-tables used by [`Engine`]:s.
//!
//! All tables are global and each is initialized at most once.
//!
//! # Tables
//!
//! | Table        | Size    | Used in encoding | Used in decoding | By engines         |
//! | ------------ | ------- | ---------------- | ---------------- | ------------------ |
//! | [`Exp`]      | 128 kiB | yes              | yes              | all                |
//! | [`Log`]      | 128 kiB | yes              | yes              | all                |
//! | [`LogWalsh`] | 128 kiB | -                | yes              | all                |
//! | [`Mul16`]    | 8 MiB   | yes              | yes              | [`NoSimd`]         |
//! | [`Mul128`]   | 8 MiB   | yes              | yes              | [`Avx2`] [`Ssse3`] [`Neon`] [`Wasm`] |
//! | [`Skew`]     | 128 kiB | yes              | yes              | all                |
//!
//! [`NoSimd`]: crate::engine::NoSimd
//! [`Avx2`]: crate::engine::Avx2
//! [`Ssse3`]: crate::engine::Ssse3
//! [`Neon`]: crate::engine::Neon
//! [`Wasm`]: crate::engine::Wasm
//! [`Engine`]: crate::engine
//!
use crate::constants::{GfElement, GF_ORDER, GF_MODULUS};

// ======================================================================
// TYPE ALIASES - PUBLIC

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
pub type Exp = [GfElement; GF_ORDER];

/// Used by [`Naive`] engine for multiplications
/// and by all [`Engine`]:s to initialize other tables.
///
/// [`Naive`]: crate::engine::Naive
/// [`Engine`]: crate::engine
pub type Log = [GfElement; GF_ORDER];

/// Used by [`Avx2`] and [`Ssse3`] engines for multiplications.
///
/// [`Avx2`]: crate::engine::Avx2
/// [`Ssse3`]: crate::engine::Ssse3
pub type Mul128 = [Multiply128lutT; GF_ORDER];

/// Elements of the Mul128 table
#[derive(Clone, Copy, Debug)]
pub struct Multiply128lutT {
    /// Lower half of `GfElements`
    pub lo: [u128; 4],
    /// Upper half of `GfElements`
    pub hi: [u128; 4],
}

/// Used by all [`Engine`]:s in [`Engine::eval_poly`].
///
/// [`Engine`]: crate::engine
/// [`Engine::eval_poly`]: crate::engine::Engine::eval_poly
pub type LogWalsh = [GfElement; GF_ORDER];

/// Used by [`NoSimd`] engine for multiplications.
///
/// [`NoSimd`]: crate::engine::NoSimd
pub type Mul16 = [[[GfElement; 16]; 4]; GF_ORDER];

/// Used by all [`Engine`]:s for FFT and IFFT.
///
/// [`Engine`]: crate::engine
pub type Skew = [GfElement; GF_MODULUS as usize];

// ======================================================================
// GENERATED TABLES - include at compile time

include!(concat!(env!("OUT_DIR"), "/generated_tables.rs"));

