use super::constants::{GfElement, GF_BITS};

/// Some kind of addition.
#[inline(always)]
pub(crate) const fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = x as u32 + y as u32;
    (sum + (sum >> GF_BITS)) as GfElement
}

/// Some kind of subtraction.
#[inline(always)]
pub(crate) const fn sub_mod(x: GfElement, y: GfElement) -> GfElement {
    let dif = (x as u32).wrapping_sub(y as u32);
    dif.wrapping_add(dif >> GF_BITS) as GfElement
}
