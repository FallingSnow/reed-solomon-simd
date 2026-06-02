// ======================================================================
// TYPE ALIASES - PUBLIC

/// Galois field element.
pub type GfElement = u16;

// ======================================================================
// CONST - PUBLIC

/// Size of Galois field element [`GfElement`] in bits.
pub const GF_BITS: usize = 16;

/// Galois field order, i.e. number of elements.
pub const GF_ORDER: usize = 65536;

/// `GF_ORDER - 1`
pub const GF_MODULUS: GfElement = 65535;