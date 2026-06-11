//! Minimal IEEE 754 binary16 conversion (no external crate needed).
//!
//! Used only for the model artifact: weights are rounded to the nearest f16
//! (ties to even, saturating at ±65504) when the artifact is built, stored as
//! 2 bytes each, and expanded back to f32 exactly on load. Because the
//! in-memory model holds the already-rounded values, save/load round-trips
//! are bit-exact.

/// Largest finite f16 value (0x7BFF; the saturation target, used in tests).
#[cfg(test)]
const F16_MAX: f32 = 65504.0;

/// Convert f32 -> f16 bits, round-to-nearest-even, saturating (no infinities
/// are produced for finite inputs; NaN maps to a quiet NaN).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;

    if exp == 255 {
        // NaN stays NaN; +/-inf saturates to the largest finite value.
        if man != 0 {
            return sign | 0x7E00;
        }
        return sign | 0x7BFF;
    }

    let unbiased = exp - 127;
    if unbiased >= 16 {
        // Too large for f16: saturate (clamp semantics suit model weights).
        return sign | 0x7BFF;
    }
    if unbiased >= -14 {
        // Normal f16 range. Drop 13 mantissa bits with round-to-nearest-even.
        let mut half_exp = (unbiased + 15) as u32;
        let mut half_man = man >> 13;
        let round_bits = man & 0x1FFF;
        if round_bits > 0x1000 || (round_bits == 0x1000 && (half_man & 1) == 1) {
            half_man += 1;
            if half_man == 0x400 {
                half_man = 0;
                half_exp += 1;
                if half_exp >= 31 {
                    return sign | 0x7BFF; // rounded past the top: saturate
                }
            }
        }
        return sign | ((half_exp as u16) << 10) | half_man as u16;
    }
    if unbiased >= -25 {
        // Subnormal f16. Shift the (implicit-1) mantissa into place; a
        // rounding carry flows naturally into the exponent bits.
        let full_man = man | 0x80_0000;
        let shift = (13 - 14 - unbiased) as u32; // 14..=24
        let half_man = full_man >> shift;
        let halfway = 1u32 << (shift - 1);
        let round_bits = full_man & ((1u32 << shift) - 1);
        let mut hm = half_man;
        if round_bits > halfway || (round_bits == halfway && (hm & 1) == 1) {
            hm += 1;
        }
        return sign | hm as u16;
    }
    // Underflow to (signed) zero.
    sign
}

/// Convert f16 bits -> f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,                                  // +/- 0
        (0, _) => {
            // Subnormal: normalize into f32.
            let mut e = -14i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
        (31, 0) => sign | 0x7F80_0000,                   // inf
        (31, _) => sign | 0x7FC0_0000,                   // NaN
        _ => sign | ((exp + 112) << 23) | (man << 13),   // normal
    };
    f32::from_bits(bits)
}

/// Round an f32 to the nearest f16-representable value (saturating).
#[inline]
pub fn round_to_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_round_trip_is_identity() {
        // Every finite f16 bit pattern must survive f16 -> f32 -> f16 exactly.
        for h in 0u16..=0xFFFF {
            let exp = (h >> 10) & 0x1F;
            let man = h & 0x3FF;
            if exp == 31 {
                continue; // inf/NaN: inf saturates by design, NaN != NaN
            }
            let x = f16_bits_to_f32(h);
            assert_eq!(f32_to_f16_bits(x), h, "bits 0x{h:04x} -> {x} round-trips");
        }
    }

    #[test]
    fn rounding_is_nearest() {
        // The rounded value must never be farther than the gap to the next
        // representable f16 on either side.
        let mut state = 0x12345678u64;
        for _ in 0..100_000 {
            // Cheap LCG over interesting magnitudes.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mag = ((state >> 33) as f32 / u32::MAX as f32) * 8.0 - 4.0; // ~[-4, 4]
            let x = (10.0f32).powf(mag) * if state & 1 == 0 { 1.0 } else { -1.0 };
            let r = round_to_f16(x);
            assert!(r.abs() <= F16_MAX);
            // Error bound: half the spacing at this magnitude (or the
            // subnormal spacing 2^-24 near zero), with saturation allowed.
            if x.abs() <= F16_MAX {
                let spacing = (x.abs().max(6.1e-5) * (1.0 / 1024.0)).max(5.97e-8);
                assert!(
                    (r - x).abs() <= spacing * 0.5 + f32::EPSILON,
                    "x={x} r={r}"
                );
            }
        }
    }

    #[test]
    fn specials() {
        assert_eq!(round_to_f16(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(round_to_f16(-0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(round_to_f16(1.0), 1.0);
        assert_eq!(round_to_f16(f32::INFINITY), F16_MAX);
        assert_eq!(round_to_f16(-f32::INFINITY), -F16_MAX);
        assert_eq!(round_to_f16(1e9), F16_MAX);
        assert!(round_to_f16(f32::NAN).is_nan());
        assert_eq!(round_to_f16(1e-10), 0.0);
    }
}
