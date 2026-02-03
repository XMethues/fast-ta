//! SIMD types and platform-specific constants

/// Lane count for each SIMD level.
pub struct Lanes;

impl Lanes {
    /// Scalar lane count
    pub const SCALAR: usize = 1;
    /// AVX2 lane count
    pub const AVX2: usize = if cfg!(feature = "f32") { 8 } else { 4 };
    /// AVX-512 lane count
    pub const AVX512: usize = if cfg!(feature = "f32") { 16 } else { 8 };
    /// Neon lane count
    pub const NEON: usize = if cfg!(feature = "f32") { 4 } else { 2 };
    /// 128-bit SIMD lane count
    pub const SIMD128: usize = if cfg!(feature = "f32") { 4 } else { 2 };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanes_consts() {
        assert_eq!(Lanes::SCALAR, 1);
        const _: () = {
            assert!(Lanes::AVX2 >= 4);
        };
        const _: () = {
            assert!(Lanes::AVX512 >= 8);
        };
        const _: () = {
            assert!(Lanes::NEON >= 2);
        };
        const _: () = {
            assert!(Lanes::SIMD128 >= 2);
        };
    }
}
