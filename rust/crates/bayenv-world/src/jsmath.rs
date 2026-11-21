//! Bit-exact mirrors of the JavaScript (V8) `Math` functions the reference implementation uses.
//!
//! The JS sources are the oracle, and V8's `Math.hypot` is NOT `sqrt(x*x + y*y)` nor libm `hypot`:
//! it scales by the max and Kahan-sums. Trig / exp / log in V8 are fdlibm ports (`ieee754::*`); the
//! `libm` crate is a musl (fdlibm-derived) port and gives the same results. Use these everywhere a
//! port must reproduce JS output bit for bit.

/// V8 `Math.hypot(a, b)` (src/builtins/math.tq): max-scaled, Kahan-compensated sum, `sqrt(sum) * max`.
#[inline]
pub fn hypot(a: f64, b: f64) -> f64 {
    let (x, y) = (a.abs(), b.abs());
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let max = if x > y { x } else { y };
    if max == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut comp = 0.0f64;
    for v in [x, y] {
        let n = v / max;
        let summand = n * n - comp;
        let pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    sum.sqrt() * max
}

/// V8 `Math.hypot` with three arguments.
#[inline]
pub fn hypot3(a: f64, b: f64, c: f64) -> f64 {
    let v = [a.abs(), b.abs(), c.abs()];
    if v.iter().any(|x| x.is_infinite()) {
        return f64::INFINITY;
    }
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let max = v.iter().cloned().fold(0.0f64, |m, x| if x > m { x } else { m });
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut comp) = (0.0f64, 0.0f64);
    for x in v {
        let n = x / max;
        let summand = n * n - comp;
        let pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    sum.sqrt() * max
}

#[inline] pub fn atan2(y: f64, x: f64) -> f64 { libm::atan2(y, x) }
#[inline] pub fn atan(x: f64) -> f64 { libm::atan(x) }
#[inline] pub fn sin(x: f64) -> f64 { libm::sin(x) }
#[inline] pub fn cos(x: f64) -> f64 { libm::cos(x) }
#[inline] pub fn tan(x: f64) -> f64 { libm::tan(x) }
#[inline] pub fn acos(x: f64) -> f64 { libm::acos(x) }
#[inline] pub fn asin(x: f64) -> f64 { libm::asin(x) }
#[inline] pub fn exp(x: f64) -> f64 { libm::exp(x) }
#[inline] pub fn log(x: f64) -> f64 { libm::log(x) }
#[inline] pub fn sqrt(x: f64) -> f64 { x.sqrt() }

/// JS `Math.round`: round half up (towards +inf), -0 preserved for (-0.5, -0].
#[inline]
pub fn round(x: f64) -> f64 {
    if !x.is_finite() { return x; }
    let f = x.floor();
    if x - f >= 0.5 { f + 1.0 } else { f }
}

/// JS `Math.min(a, b)` (NaN-propagating, -0 < +0).
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { return f64::NAN; }
    if a == b { return if a.is_sign_negative() { a } else { b }; }
    if a < b { a } else { b }
}
/// JS `Math.max(a, b)` (NaN-propagating, +0 > -0).
#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { return f64::NAN; }
    if a == b { return if a.is_sign_positive() { a } else { b }; }
    if a > b { a } else { b }
}

/// `Math.fround` (JS f32 store) as a round trip.
#[inline] pub fn fround(x: f64) -> f64 { x as f32 as f64 }

/// JS `x >>> 0` for a double holding an integer value (ToUint32).
#[inline]
pub fn to_u32(x: f64) -> u32 {
    if !x.is_finite() { return 0; }
    let t = x.trunc();
    (t.rem_euclid(4294967296.0)) as u32
}
/// JS `x | 0` (ToInt32).
#[inline] pub fn to_i32(x: f64) -> i32 { to_u32(x) as i32 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hypot_basic() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, 0.0), 0.0);
        assert_eq!(round(-0.5), 0.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(to_i32(-1.0), -1);
        assert_eq!(to_u32(-1.0), u32::MAX);
    }
}
