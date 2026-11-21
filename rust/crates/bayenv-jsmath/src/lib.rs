//! # bayenv-jsmath
//!
//! Bit-exact Rust ports of JavaScript `Math.*` as evaluated by **Node v22.22.0
//! (V8 12.4.254.21-node.33) on macOS arm64** (Homebrew `node@22`, shared
//! `libnode.127.dylib`). Every function returns the *same IEEE-754 bits* as the
//! JS expression for every `f64` input. NaN results are NaN, but the payload
//! is not reproduced; JS can only observe a payload through typed-array
//! aliasing. Validated against node with zero mismatches (see "Validation").
//! Dependency-free.
//!
//! ## API / provenance
//!
//! | Rust | JS | ported from |
//! |------|----|-------------|
//! | [`sin`] [`cos`] [`tan`] | `Math.sin/cos/tan` | V8 12.4 `src/base/ieee754.cc` (fdlibm k_sin/k_cos/k_tan, e_rem_pio2, k_rem_pio2) |
//! | [`asin`] [`acos`] [`atan`] [`atan2`] | `Math.asin/acos/atan/atan2` | V8 12.4 `src/base/ieee754.cc` (fdlibm) |
//! | [`exp`] [`expm1`] [`log`] [`log2`] [`tanh`] | `Math.exp/expm1/log/log2/tanh` | V8 12.4 `src/base/ieee754.cc` (fdlibm; log2 from FreeBSD msun e_log2.c via V8) |
//! | [`pow`] | `Math.pow(x, y)` and `x ** y` | V8 12.4 `src/base/ieee754.cc` `pow` (fdlibm e_pow.c; `(±1) ** ±Infinity` is NaN, as in JS) |
//! | [`hypot`] [`hypot3`] [`hypot_slice`] | `Math.hypot(a, b[, c, ...])` | V8 12.4 `src/builtins/math.tq` `MathHypot`: max-abs scaling, Kahan sum of squares, `sqrt(sum) * max` |
//! | [`round`] | `Math.round` | V8 CSA `Float64Round`: `r = ceil(x); if r - 0.5 > x { r - 1 } else { r }` |
//! | [`max`] [`min`] [`max_slice`] [`min_slice`] | `Math.max/min` | ES spec / V8 `Float64Max/Min`: NaN propagates, `-0 < +0` |
//! | [`sign`] [`fround`] [`imul`] | `Math.sign/fround/imul` | ES spec |
//! | [`to_int32`] [`to_uint32`] | `x \| 0`, `x >>> 0` | ES spec ToInt32 / ToUint32 (NaN, ±inf -> 0; modulo 2^32) |
//! | [`floor`] [`ceil`] [`trunc`] [`sqrt`] [`abs`] | `Math.floor/ceil/trunc/sqrt/abs` | exact IEEE ops: thin `#[inline]` wrappers over `std` |
//!
//! `Math.max(a, b, c)` == `max(max(max(-inf, a), b), c)` == [`max_slice`]`(&[a, b, c])`.
//! `Math.min` folds from `+inf`. `Math.max()` is `-inf`, `Math.min()` is `+inf`
//! and `Math.hypot()` is `+0`.
//!
//! **Do not substitute `std`.** On the validation vectors, `f64::sin/cos/exp/ln/atan2/powf/hypot`
//! (macOS libm) disagree with node on 0.2-10% of inputs (`tan` 32%, `hypot` 17%).
//!
//! ## Which sin/cos? (fdlibm, not glibc)
//!
//! V8 12.x can be built with `V8_USE_LIBM_TRIG_FUNCTIONS`, which uses glibc
//! `s_sin.c` sin/cos behind `--use-libm-trig-functions`. **This node build
//! does not.** Three pieces of evidence show it:
//! `libnode.127.dylib` exports `v8::base::ieee754::sin/cos` and contains no
//! `fdlibm_sin`, `libm_sin` or `glibc_*` symbols. `node --v8-options` has no
//! `--use-libm-trig-functions` flag. The fdlibm port here matches node bit for
//! bit, while the near-correctly-rounded macOS libm `sin`, which is what glibc
//! would also produce, mismatches on about 3% of inputs. No glibc (LGPL) code
//! is used.
//!
//! ## FMA contraction (important)
//!
//! Apple clang compiled the arm64 `libnode` with its default
//! `-ffp-contract=on`. That fuses every *syntactic* `a*b ± c` inside one C
//! expression of `ieee754.cc` into a single fused multiply-add, including
//! `x += a*b` and `c - a*b`. Fusion never crosses statements, and when both
//! operands are products the left one is fused. The disassembly has 269
//! `fmadd`/`fmsub`/`fnmsub` instructions in those functions. A plain
//! (uncontracted) fdlibm port is therefore **not** bit-identical: it
//! mismatched on 0.02-1% of inputs per function. `fdlibm.rs` writes each
//! contracted site as `mul_add`, which gives zero mismatches.
//!
//! The cargo feature `no-fma` reproduces an *uncontracted* build instead, for
//! example x86-64 node, whose baseline ISA has no FMA. That variant is
//! **unvalidated**, because only arm64 node was available. This means JS
//! `Math.sin` and friends give different results on x86-64 and arm64 node, so
//! JS reference traces are platform-specific. On targets without hardware FMA,
//! `mul_add` falls back to a slow but correctly rounded software fma, so the
//! default build still reproduces arm64 node bits on every platform.
//!
//! [`hypot`] is a Torque builtin compiled by V8's own backend, which never
//! contracts, so it uses separately rounded multiply and add (validated).
//!
//! ## Licence
//!
//! * `fdlibm.rs` translates V8's `src/base/ieee754.cc`. That file is fdlibm
//!   (Sun Microsystems: "Permission to use, copy, modify, and distribute this
//!   software is freely granted, provided that this notice is preserved.")
//!   modified by the V8 authors under BSD-3-Clause. The Sun notice and the V8
//!   copyright line are kept at the top of `fdlibm.rs`. Both licences are
//!   compatible with MIT.
//! * [`hypot`] follows V8's `math.tq` (BSD-3-Clause). Everything else is
//!   derived from the ES spec.
//! * No glibc/LGPL code.
//!
//! ## Validation
//!
//! ```text
//! cd rust/bayenv-jsmath
//! ./validate/run.sh [outdir]          # N=10_000_000 random records per function
//! N=1000000 ./validate/run.sh         # quicker
//! JITCHECK=0 KEEP=1 ./validate/run.sh # skip interpreter cross-check, keep .bin files
//! ```
//! `validate/gen.mjs` generates the inputs and evaluates them in node inside a
//! hot loop, so the code is JIT-optimised. It then writes raw little-endian f64
//! records `[inputs..., output]` to `<outdir>/<fn>.bin`. The inputs cover:
//! - physical ranges, such as trig in [-2pi, 2pi] and [-1e4, 1e4], exp in
//!   [-50, 50] and log in (0, 1e7]
//! - values within a few ulps of k*pi/2
//! - tiny, huge and subnormal magnitudes
//! - random bit patterns
//! - about 400 special values (±0, ±inf, NaN, subnormals, powers of two,
//!   thresholds), tested in full cartesian products for the 2-argument functions
//!
//! `examples/validate.rs` recomputes every record in Rust and counts bit
//! mismatches, alongside the count for the `std` equivalent. `run.sh` then
//! re-evaluates the stored inputs with `node --jitless` (interpreter plus CSA
//! builtins) and `node --no-turbofan` (Maglev), and checks that every tier
//! produces the same bits.
//!
//! Benchmarks: `cargo run --release -p bayenv-jsmath --example bench`.

mod fdlibm;

pub use fdlibm::{acos, asin, atan, atan2, cos, exp, expm1, log, log2, pow, sin, tan, tanh};

/// `Math.floor` (exact; identical to `f64::floor`).
#[inline(always)]
pub fn floor(x: f64) -> f64 {
    x.floor()
}
/// `Math.ceil` (exact; identical to `f64::ceil`).
#[inline(always)]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}
/// `Math.trunc` (exact; identical to `f64::trunc`).
#[inline(always)]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}
/// `Math.sqrt` (correctly rounded IEEE sqrt; identical to `f64::sqrt`).
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
/// `Math.abs` (identical to `f64::abs`).
#[inline(always)]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// `Math.round(x)`: round half toward +inf, preserving `-0` and mapping
/// `[-0.5, -0]` to `-0`. Exactly V8's CSA `Float64Round`:
/// `r = ceil(x); if r - 0.5 > x { r - 1 } else { r }` (NaN/inf/-0 fall through).
#[inline]
pub fn round(x: f64) -> f64 {
    let r = x.ceil();
    if r - 0.5 > x {
        r - 1.0
    } else {
        r
    }
}

/// `Math.max(a, b)`: NaN if either is NaN; `max(-0, +0) = +0`.
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    // Branch-free select chain (compiles to fcmp/csel); equal => only ±0 can
    // differ in bits, and AND of the bits gives +0.
    let r = if a > b { a } else { b };
    let r = if a == b { f64::from_bits(a.to_bits() & b.to_bits()) } else { r };
    if a.is_nan() | b.is_nan() {
        f64::NAN
    } else {
        r
    }
}

/// `Math.min(a, b)`: NaN if either is NaN; `min(+0, -0) = -0`.
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    let r = if a < b { a } else { b };
    let r = if a == b { f64::from_bits(a.to_bits() | b.to_bits()) } else { r };
    if a.is_nan() | b.is_nan() {
        f64::NAN
    } else {
        r
    }
}

/// `Math.max(...xs)` (fold from `-Infinity`; `Math.max()` = `-Infinity`).
#[inline]
pub fn max_slice(xs: &[f64]) -> f64 {
    xs.iter().fold(f64::NEG_INFINITY, |acc, &v| max(acc, v))
}

/// `Math.min(...xs)` (fold from `+Infinity`; `Math.min()` = `+Infinity`).
#[inline]
pub fn min_slice(xs: &[f64]) -> f64 {
    xs.iter().fold(f64::INFINITY, |acc, &v| min(acc, v))
}

/// `Math.sign(x)`: ±1, or x itself for ±0 and NaN.
#[inline]
pub fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// `Math.fround(x)`: round to nearest f32 (ties-to-even), widen back.
#[inline(always)]
pub fn fround(x: f64) -> f64 {
    x as f32 as f64
}

/// ECMAScript ToUint32 (`x >>> 0`) for any f64: NaN/±inf -> 0, otherwise
/// truncate toward zero and reduce modulo 2^32.
#[inline]
pub fn to_uint32(x: f64) -> u32 {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp < 1023 || exp == 0x7ff {
        return 0; // |x| < 1, NaN, inf
    }
    let mant = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
    let shift = exp - 1075; // |x| = mant * 2^shift
    let v: u32 = if shift >= 0 {
        if shift >= 32 {
            0
        } else {
            (mant << shift) as u32
        }
    } else {
        (mant >> (-shift)) as u32 // -shift in 1..=52
    };
    if (bits >> 63) != 0 {
        v.wrapping_neg()
    } else {
        v
    }
}

/// ECMAScript ToInt32 (`x | 0`) for any f64.
#[inline(always)]
pub fn to_int32(x: f64) -> i32 {
    to_uint32(x) as i32
}

/// `Math.imul(a, b)`: wrapping i32 product of ToInt32(a) and ToInt32(b).
#[inline(always)]
pub fn imul(a: f64, b: f64) -> i32 {
    to_int32(a).wrapping_mul(to_int32(b))
}

/// Shared tail of V8's `MathHypot` once the max abs value is known (max > 0, finite).
#[inline(always)]
fn hypot_kahan(xs: &[f64], max: f64) -> f64 {
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &v in xs {
        let n = v.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// `Math.hypot(...xs)` for any number of arguments (`Math.hypot()` = 0),
/// exactly as V8's Torque `MathHypot`: if any |x| is inf -> inf; else if any
/// NaN -> NaN; else if all zero -> +0; else max-abs scaling + Kahan-compensated
/// sum of squares, `sqrt(sum) * max`. NOT the same as `f64::hypot` (libm).
#[inline]
pub fn hypot_slice(xs: &[f64]) -> f64 {
    let mut one_nan = false;
    let mut max = 0.0f64;
    for &v in xs {
        if v.is_nan() {
            one_nan = true;
        } else {
            let a = v.abs();
            if a > max {
                max = a;
            }
        }
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    } else if one_nan {
        return f64::NAN;
    } else if max == 0.0 {
        return 0.0;
    }
    hypot_kahan(xs, max)
}

/// `Math.hypot(a, b)`.
#[inline]
pub fn hypot(a: f64, b: f64) -> f64 {
    hypot_slice(&[a, b])
}

/// `Math.hypot(a, b, c)`.
#[inline]
pub fn hypot3(a: f64, b: f64, c: f64) -> f64 {
    hypot_slice(&[a, b, c])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    #[test]
    fn spec_spot_values() {
        assert!(same(round(-0.4), -0.0));
        assert!(same(round(0.49999999999999994), 0.0));
        assert!(same(round(-0.5), -0.0));
        assert!(same(round(-0.0), -0.0));
        assert!(same(round(2.5), 3.0));
        assert!(same(round(-2.5), -2.0));
        assert!(same(round(4503599627370497.0), 4503599627370497.0));
        assert!(same(max(-0.0, 0.0), 0.0));
        assert!(same(max(0.0, -0.0), 0.0));
        assert!(same(min(0.0, -0.0), -0.0));
        assert!(same(max(1.0, f64::NAN), f64::NAN));
        assert!(same(max_slice(&[]), f64::NEG_INFINITY));
        assert_eq!(to_int32(4294967296.5), 0);
        assert_eq!(to_int32(2147483648.0), -2147483648);
        assert_eq!(to_int32(-1.9), -1);
        assert_eq!(to_uint32(-1.0), 4294967295);
        assert_eq!(to_int32(1e300), 0);
        assert_eq!(to_int32(f64::NAN), 0);
        assert_eq!(imul(0xffffffffu32 as f64, 5.0), -5);
        assert!(same(hypot(f64::NAN, f64::INFINITY), f64::INFINITY));
        assert!(same(hypot(-0.0, -0.0), 0.0));
        assert!(same(hypot(3.0, 4.0), 5.0));
        assert!(same(hypot_slice(&[]), 0.0));
        assert!(same(pow(1.0, f64::INFINITY), f64::NAN));
        assert!(same(pow(f64::NAN, 0.0), 1.0));
        assert!(same(exp(1.0), std::f64::consts::E));
        assert!(same(sign(-0.0), -0.0));
    }
}
