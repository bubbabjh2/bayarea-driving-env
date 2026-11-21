//! Two-lane versions of `jsmath::tanh` and `jsmath::atan`, bit-identical to calling the scalar function on each lane.
//!
//! Each lane's range classification, argument reduction and final combination run as the scalar code does (per lane,
//! with its branches); only the branch-free polynomial core in between is evaluated for both lanes at once with
//! 2 x f64 NEON operations. Every NEON op is the same IEEE operation the scalar code performs on that lane (`vfmaq` is
//! the fused multiply-add the scalar `fma` emits on aarch64, `vdivq` / `vmulq` / `vaddq` / `vsubq` round like the
//! scalar ops), so each lane's bits are the scalar result's. Lanes that return before the polynomial (special values,
//! tiny arguments, saturation) get their scalar result; the polynomial is still evaluated on a harmless placeholder.
//! Off aarch64 these just call the scalar functions.
//!
//! Not used by the sim: in isolation atan_x2 runs 2.1x the scalar throughput (tanh_x2 1.1x), but inside
//! physics::step_dynamics4 (4 cars interleaved, already throughput-bound) neither gains (bayenv-sim BENCHMARKS.md).
use crate::jsmath;

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    #[inline(always)]
    pub fn ld(a: [f64; 2]) -> float64x2_t { unsafe { vld1q_f64(a.as_ptr()) } }
    #[inline(always)]
    pub fn st(v: float64x2_t) -> [f64; 2] { let mut o = [0.0; 2]; unsafe { vst1q_f64(o.as_mut_ptr(), v) }; o }
    #[inline(always)]
    pub fn sp(x: f64) -> float64x2_t { unsafe { vdupq_n_f64(x) } }
    /// fma(a, b, c) = a * b + c, fused
    #[inline(always)]
    pub fn fma(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t { unsafe { vfmaq_f64(c, a, b) } }
    #[inline(always)]
    pub fn mul(a: float64x2_t, b: float64x2_t) -> float64x2_t { unsafe { vmulq_f64(a, b) } }
    #[inline(always)]
    pub fn add(a: float64x2_t, b: float64x2_t) -> float64x2_t { unsafe { vaddq_f64(a, b) } }
    #[inline(always)]
    pub fn sub(a: float64x2_t, b: float64x2_t) -> float64x2_t { unsafe { vsubq_f64(a, b) } }
    #[inline(always)]
    pub fn div(a: float64x2_t, b: float64x2_t) -> float64x2_t { unsafe { vdivq_f64(a, b) } }
    #[inline(always)]
    pub fn neg(a: float64x2_t) -> float64x2_t { unsafe { vnegq_f64(a) } }
}

#[inline(always)]
fn hi(x: f64) -> i32 { (x.to_bits() >> 32) as u32 as i32 }
#[inline(always)]
fn lo(x: f64) -> u32 { x.to_bits() as u32 }
#[inline(always)]
fn words(h: u32, l: u32) -> f64 { f64::from_bits(((h as u64) << 32) | l as u64) }
#[inline(always)]
fn with_hi(x: f64, h: u32) -> f64 { f64::from_bits(((h as u64) << 32) | (x.to_bits() & 0xFFFF_FFFF)) }
#[inline(always)]
fn fma1(a: f64, b: f64, c: f64) -> f64 { a.mul_add(b, c) }

const HUGE: f64 = 1.0e300;
const TINY: f64 = 1.0e-300;
const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;

/// expm1 state of one lane after the scalar prelude: Ok((x, c, k)) to continue, Err(result) for an early return
#[inline(always)]
fn expm1_pre(mut x: f64) -> Result<(f64, f64, i32), f64> {
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    let mut hx = hi(x) as u32;
    let xsb = hx & 0x80000000;
    hx &= 0x7FFFFFFF;
    let k: i32;
    let mut c = 0.0;
    if hx >= 0x4043687A {
        if hx >= 0x40862E42 {
            if hx >= 0x7FF00000 {
                if ((hx & 0xFFFFF) | lo(x)) != 0 { return Err(x + x); }
                return Err(if xsb == 0 { x } else { -1.0 });
            }
            if x > O_THRESHOLD { return Err(HUGE * HUGE); }
        }
        if xsb != 0 && x + TINY < 0.0 { return Err(TINY - 1.0); }
    }
    if hx > 0x3FD62E42 {
        let (hi_, lo_);
        if hx < 0x3FF0A2B2 {
            if xsb == 0 { hi_ = x - LN2_HI; lo_ = LN2_LO; k = 1; } else { hi_ = x + LN2_HI; lo_ = -LN2_LO; k = -1; }
        } else {
            k = fma1(INVLN2, x, if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi_ = fma1(-t, LN2_HI, x);
            lo_ = t * LN2_LO;
        }
        x = hi_ - lo_;
        c = (hi_ - x) - lo_;
    } else if hx < 0x3C900000 {
        let t = HUGE + x;
        return Err(x - (t - (HUGE + x)));
    } else {
        k = 0;
    }
    Ok((x, c, k))
}

/// expm1 tail of one lane (after the polynomial: hxs, e)
#[inline(always)]
fn expm1_post(x: f64, c: f64, k: i32, hxs: f64, mut e: f64) -> f64 {
    if k == 0 { return x - fma1(x, e, -hxs); }
    let twopk = words(0x3FF00000i32.wrapping_add(((k as u32) << 20) as i32) as u32, 0);
    e = fma1(x, e - c, -c);
    e -= hxs;
    if k == -1 { return fma1(0.5, x - e, -0.5); }
    if k == 1 {
        if x < -0.25 { return -2.0 * (e - (x + 0.5)); }
        return fma1(2.0, x - e, 1.0);
    }
    if k <= -2 || k > 56 {
        let mut y = 1.0 - (e - x);
        if k == 1024 { y = y * 2.0 * 8.98846567431158e+307; } else { y *= twopk; }
        return y - 1.0;
    }
    if k < 20 {
        let t = with_hi(1.0, (0x3FF00000 - (0x200000 >> k)) as u32);
        (t - (e - x)) * twopk
    } else {
        let t = with_hi(1.0, ((0x3FF - k) << 20) as u32);
        let mut y = x - (e + t);
        y += 1.0;
        y * twopk
    }
}

/// `[expm1(a[0]), expm1(a[1])]`
#[inline]
pub fn expm1_x2(a: [f64; 2]) -> [f64; 2] {
    #[cfg(target_arch = "aarch64")]
    {
        use neon::*;
        const Q1: f64 = -3.33333333333331316428e-02;
        const Q2: f64 = 1.58730158725481460165e-03;
        const Q3: f64 = -7.93650757867487942473e-05;
        const Q4: f64 = 4.00821782732936239552e-06;
        const Q5: f64 = -2.01099218183624371326e-07;
        let (p0, p1) = (expm1_pre(a[0]), expm1_pre(a[1]));
        let (x0, c0, k0) = *p0.as_ref().unwrap_or(&(0.0, 0.0, 0));
        let (x1, c1, k1) = *p1.as_ref().unwrap_or(&(0.0, 0.0, 0));
        let x = ld([x0, x1]);
        let hfx = mul(sp(0.5), x);
        let hxs = mul(x, hfx);
        let r1 = fma(hxs, fma(hxs, fma(hxs, fma(hxs, fma(hxs, sp(Q5), sp(Q4)), sp(Q3)), sp(Q2)), sp(Q1)), sp(1.0));
        let t = fma(neg(r1), hfx, sp(3.0));
        let e = mul(hxs, div(sub(r1, t), fma(neg(x), t, sp(6.0))));
        let (hv, ev) = (st(hxs), st(e));
        [match p0 { Ok(_) => expm1_post(x0, c0, k0, hv[0], ev[0]), Err(r) => r }, match p1 { Ok(_) => expm1_post(x1, c1, k1, hv[1], ev[1]), Err(r) => r }]
    }
    #[cfg(not(target_arch = "aarch64"))]
    { [jsmath::expm1(a[0]), jsmath::expm1(a[1])] }
}

/// `[tanh(x[0]), tanh(x[1])]`
#[inline]
pub fn tanh_x2(x: [f64; 2]) -> [f64; 2] {
    // per lane: Err(result) = done, Ok((expm1 argument, big)) = needs expm1
    let pre = |x: f64| -> Result<(f64, bool), f64> {
        let jx = hi(x);
        let ix = jx & 0x7FFFFFFF;
        if ix >= 0x7FF00000 { return Err(if jx >= 0 { 1.0 / x + 1.0 } else { 1.0 / x - 1.0 }); }
        if ix < 0x40360000 {
            if ix < 0x3E300000 && HUGE + x > 1.0 { return Err(x); }
            if ix >= 0x3FF00000 { Ok((2.0 * x.abs(), true)) } else { Ok((-2.0 * x.abs(), false)) }
        } else {
            let z = 1.0 - TINY;
            Err(if jx >= 0 { z } else { -z })
        }
    };
    let post = |x: f64, t: f64, big: bool| -> f64 {
        let z = if big { 1.0 - 2.0 / (t + 2.0) } else { -t / (t + 2.0) };
        if hi(x) >= 0 { z } else { -z }
    };
    match (pre(x[0]), pre(x[1])) {
        (Ok((a0, b0)), Ok((a1, b1))) => { let t = expm1_x2([a0, a1]); [post(x[0], t[0], b0), post(x[1], t[1], b1)] }
        (Ok((a0, b0)), Err(r1)) => [post(x[0], jsmath::expm1(a0), b0), r1],
        (Err(r0), Ok((a1, b1))) => [r0, post(x[1], jsmath::expm1(a1), b1)],
        (Err(r0), Err(r1)) => [r0, r1],
    }
}

/// `[atan(x[0]), atan(x[1])]`
#[inline]
pub fn atan_x2(xs: [f64; 2]) -> [f64; 2] {
    #[cfg(target_arch = "aarch64")]
    {
        use neon::*;
        const ATANHI: [f64; 4] = [4.63647609000806093515e-01, 7.85398163397448278999e-01, 9.82793723247329054082e-01, 1.57079632679489655800e+00];
        const ATANLO: [f64; 4] = [2.26987774529616870924e-17, 3.06161699786838301793e-17, 1.39033110312309984516e-17, 6.12323399573676603587e-17];
        const AT: [f64; 11] = [3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01, -1.11111104054623557880e-01,
            9.09088713343650656196e-02, -7.69187620504482999495e-02, 6.66107313738753120669e-02, -5.83357013379057348645e-02,
            4.97687799461593236017e-02, -3.65315727442169155270e-02, 1.62858201153657823623e-02];
        // fast path: both lanes in the no-reduction range (2^-27 <= |x| < 0.4375; atan's id = -1 branch)
        let (i0, i1) = (hi(xs[0]) & 0x7FFFFFFF, hi(xs[1]) & 0x7FFFFFFF);
        if (0x3E400000..0x3FDC0000).contains(&i0) && (0x3E400000..0x3FDC0000).contains(&i1) {
            let x = ld(xs);
            let z = mul(x, x);
            let w = mul(z, z);
            let s1 = mul(z, fma(w, fma(w, fma(w, fma(w, fma(w, sp(AT[10]), sp(AT[8])), sp(AT[6])), sp(AT[4])), sp(AT[2])), sp(AT[0])));
            let s2 = mul(w, fma(w, fma(w, fma(w, fma(w, sp(AT[9]), sp(AT[7])), sp(AT[5])), sp(AT[3])), sp(AT[1])));
            return st(fma(neg(x), add(s1, s2), x));
        }
        // per lane: Err(result) or Ok((reduced x, id))
        let pre = |mut x: f64| -> Result<(f64, i32), f64> {
            let hx = hi(x);
            let ix = hx & 0x7FFFFFFF;
            if ix >= 0x44100000 {
                if ix > 0x7FF00000 || (ix == 0x7FF00000 && lo(x) != 0) { return Err(x + x); }
                return Err(if hx > 0 { ATANHI[3] + ATANLO[3] } else { -ATANHI[3] - ATANLO[3] });
            }
            let id: i32;
            if ix < 0x3FDC0000 {
                if ix < 0x3E400000 && HUGE + x > 1.0 { return Err(x); }
                id = -1;
            } else {
                x = x.abs();
                if ix < 0x3FF30000 {
                    if ix < 0x3FE60000 { id = 0; x = fma1(2.0, x, -1.0) / (2.0 + x); } else { id = 1; x = (x - 1.0) / (x + 1.0); }
                } else if ix < 0x40038000 { id = 2; x = (x - 1.5) / fma1(1.5, x, 1.0); } else { id = 3; x = -1.0 / x; }
            }
            Ok((x, id))
        };
        let (p0, p1) = (pre(xs[0]), pre(xs[1]));
        if let (Err(r0), Err(r1)) = (p0, p1) { return [r0, r1]; }
        let (x0, id0) = p0.unwrap_or((0.0, -1));
        let (x1, id1) = p1.unwrap_or((0.0, -1));
        let x = ld([x0, x1]);
        let z = mul(x, x);
        let w = mul(z, z);
        let s1 = mul(z, fma(w, fma(w, fma(w, fma(w, fma(w, sp(AT[10]), sp(AT[8])), sp(AT[6])), sp(AT[4])), sp(AT[2])), sp(AT[0])));
        let s2 = mul(w, fma(w, fma(w, fma(w, fma(w, sp(AT[9]), sp(AT[7])), sp(AT[5])), sp(AT[3])), sp(AT[1])));
        let ss = st(add(s1, s2));
        let post = |x: f64, id: i32, ss: f64, orig: f64| -> f64 {
            if id < 0 { return fma1(-x, ss, x); }
            let id = id as usize;
            let z = ATANHI[id] - (fma1(x, ss, -ATANLO[id]) - x);
            if hi(orig) < 0 { -z } else { z }
        };
        [match p0 { Ok(_) => post(x0, id0, ss[0], xs[0]), Err(r) => r }, match p1 { Ok(_) => post(x1, id1, ss[1], xs[1]), Err(r) => r }]
    }
    #[cfg(not(target_arch = "aarch64"))]
    { [jsmath::atan(xs[0]), jsmath::atan(xs[1])] }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsmath;
    fn inputs() -> Vec<f64> {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut v = vec![0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 22.0, -22.0, 1e-300, -1e-300, 5e-324, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 709.78, -709.78, 710.0, 0.34657359, 1.0397, 56.0 * std::f64::consts::LN_2];
        for i in 0..3_000_000u64 {
            let r = rnd();
            v.push(match i % 4 {
                0 => f64::from_bits(r),
                1 => ((r % 2_000_000) as f64 - 1_000_000.0) * 1e-5,
                2 => ((r % 2_000_000) as f64 - 1_000_000.0) * 1e-2,
                _ => (if r & 1 == 0 { 1.0 } else { -1.0 }) * 10f64.powf((r >> 8) as f64 / (1u64 << 56) as f64 * 40.0 - 30.0),
            });
        }
        v
    }
    fn same(a: f64, b: f64) -> bool { a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) }
    #[test]
    fn x2_equal_scalar() {
        let v = inputs();
        let mut bad = (0, 0, 0);
        for w in v.windows(2) {
            let (a, b) = (w[0], w[1]);
            let t = tanh_x2([a, b]); if !same(t[0], jsmath::tanh(a)) || !same(t[1], jsmath::tanh(b)) { bad.0 += 1; }
            let t = atan_x2([a, b]); if !same(t[0], jsmath::atan(a)) || !same(t[1], jsmath::atan(b)) { bad.1 += 1; }
            let t = expm1_x2([a, b]); if !same(t[0], jsmath::expm1(a)) || !same(t[1], jsmath::expm1(b)) { bad.2 += 1; }
        }
        assert_eq!(bad, (0, 0, 0), "mismatching pairs (tanh, atan, expm1)");
    }
}
