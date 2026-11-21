//! JS `Math.*` semantics used by the port. Transcendentals come from `bayenv-jsmath` (bit-exact with
//! Node 22 / V8 12.4); this module adds the JS-specific helpers (max/min with NaN and signed-zero rules,
//! round half up, `|0` / `>>>0`, typed-array stores).
pub use std::f64::consts::PI;

#[inline(always)] pub fn sin(x: f64) -> f64 { x.sin() }
#[inline(always)] pub fn cos(x: f64) -> f64 { x.cos() }
#[inline(always)] pub fn tan(x: f64) -> f64 { x.tan() }
#[inline(always)] pub fn atan(x: f64) -> f64 { x.atan() }
#[inline(always)] pub fn atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
#[inline(always)] pub fn acos(x: f64) -> f64 { x.acos() }
#[inline(always)] pub fn exp(x: f64) -> f64 { x.exp() }
#[inline(always)] pub fn log(x: f64) -> f64 { x.ln() }
#[inline(always)] pub fn tanh(x: f64) -> f64 { x.tanh() }
#[inline(always)] pub fn hypot(a: f64, b: f64) -> f64 { a.hypot(b) }

/// Math.max(a, b): NaN if either is NaN; max(+0, -0) = +0.
#[inline(always)]
pub fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else if b > a { b } else if a == b { if a == 0.0 && a.is_sign_negative() { b } else { a } } else { f64::NAN }
}
/// Math.min(a, b): NaN if either is NaN; min(+0, -0) = -0.
#[inline(always)]
pub fn min(a: f64, b: f64) -> f64 {
    if a < b { a } else if b < a { b } else if a == b { if a == 0.0 && b.is_sign_negative() { b } else { a } } else { f64::NAN }
}
/// Math.round: nearest integer, ties toward +Infinity (V8 Float64Round: ceil, minus 1 if that overshoots by > 0.5).
#[inline(always)]
pub fn round(x: f64) -> f64 { let r = x.ceil(); if r - 0.5 > x { r - 1.0 } else { r } }
#[inline(always)] pub fn floor(x: f64) -> f64 { x.floor() }
#[inline(always)] pub fn ceil(x: f64) -> f64 { x.ceil() }
#[inline(always)] pub fn abs(x: f64) -> f64 { x.abs() }
#[inline(always)] pub fn sqrt(x: f64) -> f64 { x.sqrt() }
/// Math.sign
#[inline(always)] pub fn sign(x: f64) -> f64 { if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { x } }
/// Float32Array store: round to nearest f32.
#[inline(always)] pub fn f32r(x: f64) -> f32 { x as f32 }
/// ECMAScript ToInt32 (`x | 0`).
#[inline(always)]
pub fn to_i32(x: f64) -> i32 {
    if !x.is_finite() { return 0; }
    let t = x.trunc();
    if t.abs() < 2147483648.0 { return t as i32; }
    let m = t.rem_euclid(4294967296.0);
    (m as u64 as u32) as i32
}
/// ECMAScript ToUint32 (`x >>> 0`).
#[inline(always)] pub fn to_u32(x: f64) -> u32 { to_i32(x) as u32 }
/// JS `a % b` for numbers (fmod semantics: sign of the dividend).
#[inline(always)] pub fn fmod(a: f64, b: f64) -> f64 { a % b }
/// `x || d` for a number: d when x is 0, -0 or NaN.
#[inline(always)] pub fn or(x: f64, d: f64) -> f64 { if x == 0.0 || x.is_nan() { d } else { x } }
/// Compare like a JS sort comparator result (negative / positive / 0 or NaN).
#[inline(always)]
pub fn cmp_of(x: f64) -> std::cmp::Ordering { if x < 0.0 { std::cmp::Ordering::Less } else if x > 0.0 { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal } }

/// V8's Array.prototype.sort for arrays shorter than 64 (TimSort = CountAndMakeRun + BinaryInsertionSort).
/// Needed where a comparator is not a consistent order (src/geom/hops.js), so the result depends on the
/// algorithm. `cmp(a, b)` returns the JS comparator value.
pub fn v8_sort_small<T: Copy>(a: &mut [T], mut cmp: impl FnMut(T, T) -> f64) {
    let n = a.len();
    if n < 2 { return; }
    assert!(n < 64, "v8_sort_small: n = {n}");
    // CountAndMakeRun(0, n)
    let mut run = 2usize;
    let desc = cmp(a[1], a[0]) < 0.0;
    let mut prev = a[1];
    for idx in 2..n {
        let cur = a[idx];
        let o = cmp(cur, prev);
        if desc { if !(o < 0.0) { break; } } else if o < 0.0 { break; }
        prev = cur; run += 1;
    }
    if desc { a[..run].reverse(); }
    // BinaryInsertionSort(0, run, n)
    let mut start = if run == 0 { 1 } else { run };
    while start < n {
        let pivot = a[start];
        let (mut left, mut right) = (0usize, start);
        while left < right {
            let mid = left + ((right - left) >> 1);
            if cmp(pivot, a[mid]) < 0.0 { right = mid; } else { left = mid + 1; }
        }
        let mut p = start;
        while p > left { a[p] = a[p - 1]; p -= 1; }
        a[left] = pivot;
        start += 1;
    }
}
