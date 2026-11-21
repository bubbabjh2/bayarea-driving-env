//! Bit-exact mirrors of the JavaScript (V8) `Math` functions the reference implementation uses.
//!
//! The JS sources are the oracle. V8's `Math.hypot` is NOT `sqrt(x*x + y*y)` nor libm `hypot`: it scales by the
//! max and Kahan-sums. The transcendental functions are a statement-by-statement port of V8's fdlibm
//! (`src/base/ieee754.cc`) including the fused multiply-adds clang creates on arm64 (see the section comment
//! below): neither the `libm` crate nor the platform libm reproduces node, both are off by an ulp on ~0.1-10 % of
//! inputs. Use these everywhere a port must reproduce JS output bit for bit. Checked against node by
//! `tests/jsmath_oracle.rs` (dump: `node rust/tools/oracle/gen_math.mjs`).

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

// ---------------------------------------------------------------------------------------------------------------
// fdlibm, as compiled into V8 (src/base/ieee754.cc, V8 12.4.254.21 = node 22.22.0)
// ---------------------------------------------------------------------------------------------------------------
//
// Node builds V8 with gyp, which never defines `V8_USE_LIBM_TRIG_FUNCTIONS`, so `Math.sin/cos` are the fdlibm ones
// too (the glibc-derived third_party/glibc path is a Chromium GN option). `Math.pow` is `base::ieee754::pow`
// (math.tq -> Float64Pow -> ieee754_pow_function), i.e. fdlibm's e_pow.c with V8's (+-1)**(+-inf) = NaN.
//
// The C file is compiled by clang with its default `-ffp-contract=on`: inside ONE C expression, `a*b + c`,
// `a*b - c` and `c - a*b` become a fused multiply-add (clang `tryEmitFMulAdd`: the left operand's multiply is
// preferred when both sides are products; `x += a*b` contracts too; nothing contracts across statements, function
// arguments or casts). On arm64 every fmuladd is emitted as an FMA instruction. `fma()` below marks exactly those
// spots; the port otherwise mirrors the C statement by statement (same temporaries, same association).
//
// Verified bit-exact on aarch64 (node 22.22.0, darwin arm64) by `tests/jsmath_oracle.rs`. On other targets `fma`
// is a separate multiply and add, which is what clang emits for x86-64 without `-mfma` (V8's default x64 build);
// that path is UNVERIFIED against a real x86-64 node.

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

#[inline(always)]
fn hi(x: f64) -> i32 {
    (x.to_bits() >> 32) as u32 as i32
}
#[inline(always)]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}
#[inline(always)]
fn words(h: u32, l: u32) -> f64 {
    f64::from_bits(((h as u64) << 32) | l as u64)
}
#[inline(always)]
fn with_hi(x: f64, h: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xFFFF_FFFF) | ((h as u64) << 32))
}
#[inline(always)]
fn with_lo(x: f64, l: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xFFFF_FFFF_0000_0000) | l as u64)
}
#[inline(always)]
fn scalbn(x: f64, n: i32) -> f64 {
    libm::scalbn(x, n)
}

const HUGE: f64 = 1.0e300;
const TINY: f64 = 1.0e-300;

static TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E,
    0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
];
static NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB, 0x402C463A,
    0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
    0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC,
    0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

/// `__ieee754_rem_pio2`: x - n*pi/2 as y[0] + y[1], returns n.
fn rem_pio2(x: f64, y: &mut [f64; 2]) -> i32 {
    const HALF: f64 = 5.00000000000000000000e-01;
    const TWO24: f64 = 1.67772160000000000000e+07;
    const INVPIO2: f64 = 6.36619772367581382433e-01;
    const PIO2_1: f64 = 1.57079632673412561417e+00;
    const PIO2_1T: f64 = 6.07710050650619224932e-11;
    const PIO2_2: f64 = 6.07710050630396597660e-11;
    const PIO2_2T: f64 = 2.02226624879595063154e-21;
    const PIO2_3: f64 = 2.02226624871116645580e-21;
    const PIO2_3T: f64 = 8.47842766036889956997e-32;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        y[0] = x;
        y[1] = 0.0;
        return 0;
    }
    if ix < 0x4002D97C {
        if hx > 0 {
            let mut z = x - PIO2_1;
            if ix != 0x3FF921FB {
                y[0] = z - PIO2_1T;
                y[1] = (z - y[0]) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y[0] = z - PIO2_2T;
                y[1] = (z - y[0]) - PIO2_2T;
            }
            return 1;
        } else {
            let mut z = x + PIO2_1;
            if ix != 0x3FF921FB {
                y[0] = z + PIO2_1T;
                y[1] = (z - y[0]) + PIO2_1T;
            } else {
                z += PIO2_2;
                y[0] = z + PIO2_2T;
                y[1] = (z - y[0]) + PIO2_2T;
            }
            return -1;
        }
    }
    if ix <= 0x413921FB {
        let t = x.abs();
        let n = fma(t, INVPIO2, HALF) as i32;
        let fnn = n as f64;
        let mut r = fma(-fnn, PIO2_1, t);
        let mut w = fnn * PIO2_1T;
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y[0] = r - w;
        } else {
            let j = ix >> 20;
            y[0] = r - w;
            let high = hi(y[0]) as u32;
            let i = j - ((high >> 20) & 0x7FF) as i32;
            if i > 16 {
                let t = r;
                w = fnn * PIO2_2;
                r = t - w;
                w = fma(fnn, PIO2_2T, -((t - r) - w));
                y[0] = r - w;
                let high = hi(y[0]) as u32;
                let i = j - ((high >> 20) & 0x7FF) as i32;
                if i > 49 {
                    let t = r;
                    w = fnn * PIO2_3;
                    r = t - w;
                    w = fma(fnn, PIO2_3T, -((t - r) - w));
                    y[0] = r - w;
                }
            }
        }
        y[1] = (r - y[0]) - w;
        if hx < 0 {
            y[0] = -y[0];
            y[1] = -y[1];
            return -n;
        }
        return n;
    }
    if ix >= 0x7FF00000 {
        let v = x - x;
        y[0] = v;
        y[1] = v;
        return 0;
    }
    let mut z = with_lo(0.0, lo(x));
    let e0 = (ix >> 20) - 1046;
    z = with_hi(z, ix.wrapping_sub(((e0 as u32) << 20) as i32) as u32);
    let mut tx = [0.0f64; 3];
    for t in tx.iter_mut().take(2) {
        *t = (z as i32) as f64;
        z = (z - *t) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3;
    while tx[nx - 1] == 0.0 {
        nx -= 1;
    }
    let n = kernel_rem_pio2(&tx, y, e0, nx);
    if hx < 0 {
        y[0] = -y[0];
        y[1] = -y[1];
        return -n;
    }
    n
}

/// `__kernel_rem_pio2` with prec = 2 (the only use).
fn kernel_rem_pio2(x: &[f64; 3], y: &mut [f64; 2], e0: i32, nx: usize) -> i32 {
    const PIO2: [f64; 8] = [
        1.57079625129699707031e+00,
        7.54978941586159635335e-08,
        5.39030252995776476554e-15,
        3.28200341580791294123e-22,
        1.27065575308067607349e-29,
        1.22933308981111328932e-36,
        2.73370053816464559624e-44,
        2.16741683877804819444e-51,
    ];
    const TWO24: f64 = 1.67772160000000000000e+07;
    const TWON24: f64 = 5.96046447753906250000e-08;
    let ipio2 = &TWO_OVER_PI;
    let jk: i32 = 4;
    let jp = jk;
    let jx = nx as i32 - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);
    let mut f = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut q = [0.0f64; 20];
    let mut iq = [0i32; 20];
    let u = |i: i32| i as usize;

    let mut j = jv - jx;
    let m = jx + jk;
    for i in 0..=m {
        f[u(i)] = if j < 0 { 0.0 } else { ipio2[u(j)] as f64 };
        j += 1;
    }
    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw = fma(x[u(j)], f[u(jx + i - j)], fw);
        }
        q[u(i)] = fw;
    }
    let mut jz = jk;
    let (mut z, mut n, mut ih);
    loop {
        // recompute:
        let mut i = 0;
        let mut j = jz;
        z = q[u(jz)];
        while j > 0 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[u(i)] = fma(-TWO24, fw, z) as i32;
            z = q[u(j - 1)] + fw;
            i += 1;
            j -= 1;
        }
        z = scalbn(z, q0);
        z = fma(-8.0, (z * 0.125).floor(), z);
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            let i = iq[u(jz - 1)] >> (24 - q0);
            n += i;
            iq[u(jz - 1)] -= i << (24 - q0);
            ih = iq[u(jz - 1)] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[u(jz - 1)] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }
        if ih > 0 {
            n += 1;
            let mut carry = 0;
            for v in iq.iter_mut().take(u(jz)) {
                let j = *v;
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        *v = 0x1000000 - j;
                    }
                } else {
                    *v = 0xFFFFFF - j;
                }
            }
            if q0 == 1 {
                iq[u(jz - 1)] &= 0x7FFFFF;
            } else if q0 == 2 {
                iq[u(jz - 1)] &= 0x3FFFFF;
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= scalbn(1.0, q0);
                }
            }
        }
        if z == 0.0 {
            let mut j = 0;
            let mut i = jz - 1;
            while i >= jk {
                j |= iq[u(i)];
                i -= 1;
            }
            if j == 0 {
                let mut k = 1;
                while jk >= k && iq[u(jk - k)] == 0 {
                    k += 1;
                }
                for i in jz + 1..=jz + k {
                    f[u(jx + i)] = ipio2[u(jv + i)] as f64;
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw = fma(x[u(j)], f[u(jx + i - j)], fw);
                    }
                    q[u(i)] = fw;
                }
                jz += k;
                continue;
            }
        }
        break;
    }
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[u(jz)] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = scalbn(z, -q0);
        if z >= TWO24 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[u(jz)] = fma(-TWO24, fw, z) as i32;
            jz += 1;
            q0 += 24;
            iq[u(jz)] = fw as i32;
        } else {
            iq[u(jz)] = z as i32;
        }
    }
    let mut fw = scalbn(1.0, q0);
    let mut i = jz;
    while i >= 0 {
        q[u(i)] = fw * iq[u(i)] as f64;
        fw *= TWON24;
        i -= 1;
    }
    let mut i = jz;
    while i >= 0 {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw = fma(PIO2[u(k)], q[u(i + k)], fw);
            k += 1;
        }
        fq[u(jz - i)] = fw;
        i -= 1;
    }
    let mut fw = 0.0;
    let mut i = jz;
    while i >= 0 {
        fw += fq[u(i)];
        i -= 1;
    }
    y[0] = if ih == 0 { fw } else { -fw };
    let mut fw = fq[0] - fw;
    for i in 1..=jz {
        fw += fq[u(i)];
    }
    y[1] = if ih == 0 { fw } else { -fw };
    n & 7
}

#[inline(always)]
fn k_cos(x: f64, y: f64) -> f64 {
    const C1: f64 = 4.16666666666666019037e-02;
    const C2: f64 = -1.38888888888741095749e-03;
    const C3: f64 = 2.48015872894767294178e-05;
    const C4: f64 = -2.75573143513906633035e-07;
    const C5: f64 = 2.08757232129817482790e-09;
    const C6: f64 = -1.13596475577881948265e-11;
    let ix = hi(x) & 0x7FFFFFFF;
    if ix < 0x3E400000 && (x as i32) == 0 {
        return 1.0;
    }
    let z = x * x;
    let r = z * fma(z, fma(z, fma(z, fma(z, fma(z, C6, C5), C4), C3), C2), C1);
    if ix < 0x3FD33333 {
        return 1.0 - fma(0.5, z, -fma(z, r, -(x * y)));
    }
    let qx = if ix > 0x3FE90000 { 0.28125 } else { words((ix - 0x00200000) as u32, 0) };
    let iz = fma(0.5, z, -qx);
    let a = 1.0 - qx;
    a - (iz - fma(z, r, -(x * y)))
}

#[inline(always)]
fn k_sin(x: f64, y: f64, iy: i32) -> f64 {
    const HALF: f64 = 5.00000000000000000000e-01;
    const S1: f64 = -1.66666666666666324348e-01;
    const S2: f64 = 8.33333333332248946124e-03;
    const S3: f64 = -1.98412698298579493134e-04;
    const S4: f64 = 2.75573137070700676789e-06;
    const S5: f64 = -2.50507602534068634195e-08;
    const S6: f64 = 1.58969099521155010221e-10;
    let ix = hi(x) & 0x7FFFFFFF;
    if ix < 0x3E400000 && (x as i32) == 0 {
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = fma(z, fma(z, fma(z, fma(z, S6, S5), S4), S3), S2);
    if iy == 0 {
        fma(v, fma(z, r, S1), x)
    } else {
        x - fma(-v, S1, fma(z, fma(HALF, y, -(v * r)), -y))
    }
}

fn k_tan(mut x: f64, mut y: f64, iy: i32) -> f64 {
    const T: [f64; 13] = [
        3.33333333333334091986e-01,
        1.33333333333201242699e-01,
        5.39682539762260521377e-02,
        2.18694882948595424599e-02,
        8.86323982359930005737e-03,
        3.59207910759131235356e-03,
        1.45620945432529025516e-03,
        5.88041240820264096874e-04,
        2.46463134818469906812e-04,
        7.81794442939557092300e-05,
        7.14072491382608190305e-05,
        -1.85586374855275456654e-05,
        2.59073051863633712884e-05,
    ];
    const PIO4: f64 = 7.85398163397448278999e-01;
    const PIO4LO: f64 = 3.06161699786838301793e-17;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix < 0x3E300000 && (x as i32) == 0 {
        let low = lo(x);
        if ((ix as u32 | low) | (iy + 1) as u32) == 0 {
            return 1.0 / x.abs();
        } else if iy == 1 {
            return x;
        } else {
            let w = x + y;
            let z = with_lo(w, 0);
            let v = y - (z - x);
            let a = -1.0 / w;
            let t = with_lo(a, 0);
            let s = fma(t, z, 1.0);
            return fma(a, fma(t, v, s), t);
        }
    }
    if ix >= 0x3FE59428 {
        if hx < 0 {
            x = -x;
            y = -y;
        }
        let z = PIO4 - x;
        let w = PIO4LO - y;
        x = z + w;
        y = 0.0;
    }
    let z = x * x;
    let w = z * z;
    let mut r = fma(w, fma(w, fma(w, fma(w, fma(w, T[11], T[9]), T[7]), T[5]), T[3]), T[1]);
    let v = z * fma(w, fma(w, fma(w, fma(w, fma(w, T[12], T[10]), T[8]), T[6]), T[4]), T[2]);
    let s = z * x;
    r = fma(z, fma(s, r + v, y), y);
    r = fma(T[0], s, r);
    let w = x + r;
    if ix >= 0x3FE59428 {
        let v = iy as f64;
        return ((1 - ((hx >> 30) & 2)) as f64) * fma(-2.0, x - (w * w / (w + v) - r), v);
    }
    if iy == 1 {
        return w;
    }
    let z = with_lo(w, 0);
    let v = r - (z - x);
    let a = -1.0 / w;
    let t = with_lo(a, 0);
    let s = fma(t, z, 1.0);
    fma(a, fma(t, v, s), t)
}

/// V8 `Math.sin` (fdlibm `sin`).
pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return k_sin(x, 0.0, 0);
    }
    if ix >= 0x7FF00000 {
        return x - x;
    }
    let mut y = [0.0; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => k_sin(y[0], y[1], 1),
        1 => k_cos(y[0], y[1]),
        2 => -k_sin(y[0], y[1], 1),
        _ => -k_cos(y[0], y[1]),
    }
}

/// V8 `Math.cos` (fdlibm `cos`).
pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return k_cos(x, 0.0);
    }
    if ix >= 0x7FF00000 {
        return x - x;
    }
    let mut y = [0.0; 2];
    let n = rem_pio2(x, &mut y);
    match n & 3 {
        0 => k_cos(y[0], y[1]),
        1 => -k_sin(y[0], y[1], 1),
        2 => -k_cos(y[0], y[1]),
        _ => k_sin(y[0], y[1], 1),
    }
}

/// V8 `Math.tan` (fdlibm `tan`).
pub fn tan(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        return k_tan(x, 0.0, 1);
    }
    if ix >= 0x7FF00000 {
        return x - x;
    }
    let mut y = [0.0; 2];
    let n = rem_pio2(x, &mut y);
    k_tan(y[0], y[1], 1 - ((n & 1) << 1))
}

const PS0: f64 = 1.66666666666666657415e-01;
const PS1: f64 = -3.25565818622400915405e-01;
const PS2: f64 = 2.01212532134862925881e-01;
const PS3: f64 = -4.00555345006794114027e-02;
const PS4: f64 = 7.91534994289814532176e-04;
const PS5: f64 = 3.47933107596021167570e-05;
const QS1: f64 = -2.40339491173441421878e+00;
const QS2: f64 = 2.02094576023350569471e+00;
const QS3: f64 = -6.88283971605453293030e-01;
const QS4: f64 = 7.70381505559019352791e-02;
const PIO2_HI: f64 = 1.57079632679489655800e+00;
const PIO2_LO: f64 = 6.12323399573676603587e-17;
#[inline(always)]
fn asin_pq(z: f64) -> (f64, f64) {
    let p = z * fma(z, fma(z, fma(z, fma(z, fma(z, PS5, PS4), PS3), PS2), PS1), PS0);
    let q = fma(z, fma(z, fma(z, fma(z, QS4, QS3), QS2), QS1), 1.0);
    (p, q)
}

/// V8 `Math.acos` (fdlibm `acos`).
pub fn acos(x: f64) -> f64 {
    const PI: f64 = 3.14159265358979311600e+00;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix >= 0x3FF00000 {
        if ((ix - 0x3FF00000) as u32 | lo(x)) == 0 {
            return if hx > 0 { 0.0 } else { fma(2.0, PIO2_LO, PI) };
        }
        return f64::NAN;
    }
    if ix < 0x3FE00000 {
        if ix <= 0x3C600000 {
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let (p, q) = asin_pq(z);
        let r = p / q;
        PIO2_HI - (x - fma(-x, r, PIO2_LO))
    } else if hx < 0 {
        let z = (1.0 + x) * 0.5;
        let (p, q) = asin_pq(z);
        let s = z.sqrt();
        let r = p / q;
        let w = fma(r, s, -PIO2_LO);
        fma(-2.0, s + w, PI)
    } else {
        let z = (1.0 - x) * 0.5;
        let s = z.sqrt();
        let df = with_lo(s, 0);
        let c = fma(-df, df, z) / (s + df);
        let (p, q) = asin_pq(z);
        let r = p / q;
        let w = fma(r, s, c);
        2.0 * (df + w)
    }
}

/// V8 `Math.asin` (fdlibm `asin`).
pub fn asin(x: f64) -> f64 {
    const PIO4_HI: f64 = 7.85398163397448278999e-01;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix >= 0x3FF00000 {
        if ((ix - 0x3FF00000) as u32 | lo(x)) == 0 {
            return fma(x, PIO2_HI, x * PIO2_LO);
        }
        return f64::NAN;
    } else if ix < 0x3FE00000 {
        let mut t = 0.0;
        if ix < 0x3E400000 {
            if HUGE + x > 1.0 {
                return x;
            }
        } else {
            t = x * x;
        }
        let (p, q) = asin_pq(t);
        let w = p / q;
        return fma(x, w, x);
    }
    let w = 1.0 - x.abs();
    let t = w * 0.5;
    let (p, q) = asin_pq(t);
    let s = t.sqrt();
    let t = if ix >= 0x3FEF3333 {
        let w = p / q;
        PIO2_HI - fma(2.0, fma(s, w, s), -PIO2_LO)
    } else {
        let w = with_lo(s, 0);
        let c = fma(-w, w, t) / (s + w);
        let r = p / q;
        let p = fma(2.0 * s, r, -fma(-2.0, c, PIO2_LO));
        let q = fma(-2.0, w, PIO4_HI);
        PIO4_HI - (p - q)
    };
    if hx > 0 {
        t
    } else {
        -t
    }
}

/// V8 `Math.atan` (fdlibm `atan`).
pub fn atan(mut x: f64) -> f64 {
    const ATANHI: [f64; 4] = [
        4.63647609000806093515e-01,
        7.85398163397448278999e-01,
        9.82793723247329054082e-01,
        1.57079632679489655800e+00,
    ];
    const ATANLO: [f64; 4] = [
        2.26987774529616870924e-17,
        3.06161699786838301793e-17,
        1.39033110312309984516e-17,
        6.12323399573676603587e-17,
    ];
    const AT: [f64; 11] = [
        3.33333333333329318027e-01,
        -1.99999999998764832476e-01,
        1.42857142725034663711e-01,
        -1.11111104054623557880e-01,
        9.09088713343650656196e-02,
        -7.69187620504482999495e-02,
        6.66107313738753120669e-02,
        -5.83357013379057348645e-02,
        4.97687799461593236017e-02,
        -3.65315727442169155270e-02,
        1.62858201153657823623e-02,
    ];
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix >= 0x44100000 {
        if ix > 0x7FF00000 || (ix == 0x7FF00000 && lo(x) != 0) {
            return x + x;
        }
        return if hx > 0 { ATANHI[3] + ATANLO[3] } else { -ATANHI[3] - ATANLO[3] };
    }
    let id: i32;
    if ix < 0x3FDC0000 {
        if ix < 0x3E400000 && HUGE + x > 1.0 {
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3FF30000 {
            if ix < 0x3FE60000 {
                id = 0;
                x = fma(2.0, x, -1.0) / (2.0 + x);
            } else {
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x40038000 {
            id = 2;
            x = (x - 1.5) / fma(1.5, x, 1.0);
        } else {
            id = 3;
            x = -1.0 / x;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * fma(w, fma(w, fma(w, fma(w, fma(w, AT[10], AT[8]), AT[6]), AT[4]), AT[2]), AT[0]);
    let s2 = w * fma(w, fma(w, fma(w, fma(w, AT[9], AT[7]), AT[5]), AT[3]), AT[1]);
    if id < 0 {
        return fma(-x, s1 + s2, x);
    }
    let id = id as usize;
    let z = ATANHI[id] - (fma(x, s1 + s2, -ATANLO[id]) - x);
    if hx < 0 {
        -z
    } else {
        z
    }
}

/// V8 `Math.atan2(y, x)` (fdlibm `atan2`).
pub fn atan2(y: f64, x: f64) -> f64 {
    const PI_O_4: f64 = 7.8539816339744827900E-01;
    const PI_O_2: f64 = 1.5707963267948965580E+00;
    const PI: f64 = 3.1415926535897931160E+00;
    const PI_LO: f64 = 1.2246467991473531772E-16;
    let (hx, lx) = (hi(x), lo(x));
    let ix = hx & 0x7FFFFFFF;
    let (hy, ly) = (hi(y), lo(y));
    let iy = hy & 0x7FFFFFFF;
    if (ix as u32 | ((lx | lx.wrapping_neg()) >> 31)) > 0x7FF00000
        || (iy as u32 | ((ly | ly.wrapping_neg()) >> 31)) > 0x7FF00000
    {
        return x + y;
    }
    if (hx.wrapping_sub(0x3FF00000) as u32 | lx) == 0 {
        return atan(y);
    }
    let mut m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
    if (iy as u32 | ly) == 0 {
        return match m {
            0 | 1 => y,
            2 => PI + TINY,
            _ => -PI - TINY,
        };
    }
    if (ix as u32 | lx) == 0 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    if ix == 0x7FF00000 {
        if iy == 0x7FF00000 {
            return match m {
                0 => PI_O_4 + TINY,
                1 => -PI_O_4 - TINY,
                2 => fma(3.0, PI_O_4, TINY),
                _ => fma(-3.0, PI_O_4, -TINY),
            };
        } else {
            return match m {
                0 => 0.0,
                1 => -0.0,
                2 => PI + TINY,
                _ => -PI - TINY,
            };
        }
    }
    if iy == 0x7FF00000 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    let k = (iy - ix) >> 20;
    let z = if k > 60 {
        m &= 1;
        fma(0.5, PI_LO, PI_O_2)
    } else if hx < 0 && k < -60 {
        0.0
    } else {
        atan((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

/// V8 `Math.exp` (fdlibm `exp`).
pub fn exp(mut x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const U_THRESHOLD: f64 = -7.45133219101941108420e+02;
    const LN2HI: [f64; 2] = [6.93147180369123816490e-01, -6.93147180369123816490e-01];
    const LN2LO: [f64; 2] = [1.90821492927058770002e-10, -1.90821492927058770002e-10];
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const P1: f64 = 1.66666666666666019037e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;
    const E: f64 = 2.718281828459045;
    const TWOM1000: f64 = 9.33263618503218878990e-302;
    const TWO1023: f64 = 8.988465674311579539e307;
    let mut hx = hi(x) as u32;
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7FFFFFFF;
    let (mut hi_, mut lo_) = (0.0, 0.0);
    let mut k: i32 = 0;
    if hx >= 0x40862E42 {
        if hx >= 0x7FF00000 {
            if ((hx & 0xFFFFF) | lo(x)) != 0 {
                return x + x;
            }
            return if xsb == 0 { x } else { 0.0 };
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE;
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000;
        }
    }
    if hx > 0x3FD62E42 {
        if hx < 0x3FF0A2B2 {
            if x == 1.0 {
                return E;
            }
            hi_ = x - LN2HI[xsb];
            lo_ = LN2LO[xsb];
            k = 1 - xsb as i32 - xsb as i32;
        } else {
            k = fma(INVLN2, x, HALF[xsb]) as i32;
            let t = k as f64;
            hi_ = fma(-t, LN2HI[0], x);
            lo_ = t * LN2LO[0];
        }
        x = hi_ - lo_;
    } else if hx < 0x3E300000 {
        if HUGE + x > 1.0 {
            return 1.0 + x;
        }
    }
    let t = x * x;
    let twopk = if k >= -1021 {
        words(0x3FF00000i32.wrapping_add(((k as u32) << 20) as i32) as u32, 0)
    } else {
        words(0x3FF00000u32.wrapping_add(((k + 1000) as u32) << 20), 0)
    };
    let c = fma(-t, fma(t, fma(t, fma(t, fma(t, P5, P4), P3), P2), P1), x);
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_ - (x * c) / (2.0 - c)) - hi_);
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * TWO1023;
        }
        return y * twopk;
    }
    y * twopk * TWOM1000
}

/// V8 `Math.atanh` (fdlibm `atanh`).
pub fn atanh(x: f64) -> f64 {
    let (hx, lx) = (hi(x), lo(x));
    let ix = hx & 0x7FFFFFFF;
    if (ix as u32 | ((lx | lx.wrapping_neg()) >> 31)) > 0x3FF00000 {
        return f64::NAN;
    }
    if ix == 0x3FF00000 {
        return if x > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if ix < 0x3E300000 && (HUGE + x) > 0.0 {
        return x;
    }
    let x = with_hi(x, ix as u32);
    let t = if ix < 0x3FE00000 {
        let t = x + x;
        0.5 * log1p(t + t * x / (1.0 - x))
    } else {
        0.5 * log1p((x + x) / (1.0 - x))
    };
    if hx >= 0 {
        t
    } else {
        -t
    }
}

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const TWO54: f64 = 1.80143985094819840000e+16;
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

/// V8 `Math.log` (fdlibm `log`).
pub fn log(mut x: f64) -> f64 {
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k = 0i32;
    if hx < 0x00100000 {
        if ((hx & 0x7FFFFFFF) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000FFFFF;
    let i = (hx + 0x95F64) & 0x100000;
    x = with_hi(x, (hx | (i ^ 0x3FF00000)) as u32);
    k += i >> 20;
    let f = x - 1.0;
    if (0x000FFFFF & (2 + hx)) < 3 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return fma(dk, LN2_HI, dk * LN2_LO);
        }
        let r = f * f * fma(-0.33333333333333333, f, 0.5);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return fma(dk, LN2_HI, -(fma(-dk, LN2_LO, r) - f));
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147A;
    let w = z * z;
    let j = 0x6B851 - hx;
    let t1 = w * fma(w, fma(w, LG6, LG4), LG2);
    let t2 = z * fma(w, fma(w, fma(w, LG7, LG5), LG3), LG1);
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - fma(-s, hfsq + r, hfsq)
        } else {
            fma(dk, LN2_HI, -((hfsq - fma(s, hfsq + r, dk * LN2_LO)) - f))
        }
    } else if k == 0 {
        fma(-s, f - r, f)
    } else {
        fma(dk, LN2_HI, -(fma(s, f - r, -(dk * LN2_LO)) - f))
    }
}

/// V8 `Math.log1p` (fdlibm `log1p`).
pub fn log1p(x: f64) -> f64 {
    const LP: [f64; 8] = [0.0, LG1, LG2, LG3, LG4, LG5, LG6, LG7];
    let hx = hi(x);
    let ax = hx & 0x7FFFFFFF;
    let mut k = 1i32;
    let mut f = 0.0;
    let mut hu = 0i32;
    let mut c = 0.0;
    if hx < 0x3FDA827A {
        if ax >= 0x3FF00000 {
            return if x == -1.0 { f64::NEG_INFINITY } else { f64::NAN };
        }
        if ax < 0x3E200000 {
            if TWO54 + x > 0.0 && ax < 0x3C900000 {
                return x;
            }
            return fma(-(x * x), 0.5, x);
        }
        if hx > 0 || hx <= 0xBFD2BEC4u32 as i32 {
            k = 0;
            f = x;
            hu = 1;
        }
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    if k != 0 {
        let mut u;
        if hx < 0x43400000 {
            u = 1.0 + x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = if k > 0 { 1.0 - (u - x) } else { x - (u - 1.0) };
            c /= u;
        } else {
            u = x;
            hu = hi(u);
            k = (hu >> 20) - 1023;
            c = 0.0;
        }
        hu &= 0x000FFFFF;
        if hu < 0x6A09E {
            u = with_hi(u, (hu | 0x3FF00000) as u32);
        } else {
            k += 1;
            u = with_hi(u, (hu | 0x3FE00000) as u32);
            hu = (0x00100000 - hu) >> 2;
        }
        f = u - 1.0;
    }
    let hfsq = 0.5 * f * f;
    let dk = k as f64;
    if hu == 0 {
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            c = fma(dk, LN2_LO, c);
            return fma(dk, LN2_HI, c);
        }
        let r = hfsq * fma(-0.66666666666666666, f, 1.0);
        if k == 0 {
            return f - r;
        }
        return fma(dk, LN2_HI, -((r - fma(dk, LN2_LO, c)) - f));
    }
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * fma(z, fma(z, fma(z, fma(z, fma(z, fma(z, LP[7], LP[6]), LP[5]), LP[4]), LP[3]), LP[2]), LP[1]);
    if k == 0 {
        f - fma(-s, hfsq + r, hfsq)
    } else {
        fma(dk, LN2_HI, -((hfsq - fma(s, hfsq + r, fma(dk, LN2_LO, c))) - f))
    }
}

#[inline(always)]
fn k_log1p(f: f64) -> f64 {
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * fma(w, fma(w, LG6, LG4), LG2);
    let t2 = z * fma(w, fma(w, fma(w, LG7, LG5), LG3), LG1);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    s * (hfsq + r)
}

/// V8 `Math.log2` (fdlibm-style `log2` from FreeBSD e_log2.c).
pub fn log2(mut x: f64) -> f64 {
    const IVLN2HI: f64 = 1.44269504072144627571e+00;
    const IVLN2LO: f64 = 1.67517131648865118353e-10;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k = 0i32;
    if hx < 0x00100000 {
        if ((hx & 0x7FFFFFFF) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    if hx == 0x3FF00000 && lx == 0 {
        return 0.0;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000FFFFF;
    let i = (hx + 0x95F64) & 0x100000;
    x = with_hi(x, (hx | (i ^ 0x3FF00000)) as u32);
    k += i >> 20;
    let y = k as f64;
    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let r = k_log1p(f);
    let hi_ = with_lo(f - hfsq, 0);
    let lo_ = (f - hi_) - hfsq + r;
    let mut val_hi = hi_ * IVLN2HI;
    let mut val_lo = fma(lo_ + hi_, IVLN2LO, lo_ * IVLN2HI);
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;
    val_lo + val_hi
}

/// V8 `Math.log10` (fdlibm `log10`).
pub fn log10(mut x: f64) -> f64 {
    const IVLN10: f64 = 4.34294481903251816668e-01;
    const LOG10_2HI: f64 = 3.01029995663611771306e-01;
    const LOG10_2LO: f64 = 3.69423907715893078616e-13;
    let mut hx = hi(x);
    let mut lx = lo(x);
    let mut k = 0i32;
    if hx < 0x00100000 {
        if ((hx & 0x7FFFFFFF) as u32 | lx) == 0 {
            return f64::NEG_INFINITY;
        }
        if hx < 0 {
            return f64::NAN;
        }
        k -= 54;
        x *= TWO54;
        hx = hi(x);
        lx = lo(x);
    }
    if hx >= 0x7FF00000 {
        return x + x;
    }
    if hx == 0x3FF00000 && lx == 0 {
        return 0.0;
    }
    k += (hx >> 20) - 1023;
    let i = ((k as u32 & 0x80000000) >> 31) as i32;
    hx = (hx & 0x000FFFFF) | ((0x3FF - i) << 20);
    let y = (k + i) as f64;
    x = with_lo(with_hi(x, hx as u32), lx);
    let z = fma(y, LOG10_2LO, IVLN10 * log(x));
    fma(y, LOG10_2HI, z)
}

/// V8 `Math.expm1` (fdlibm `expm1`).
pub fn expm1(mut x: f64) -> f64 {
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const Q1: f64 = -3.33333333333331316428e-02;
    const Q2: f64 = 1.58730158725481460165e-03;
    const Q3: f64 = -7.93650757867487942473e-05;
    const Q4: f64 = 4.00821782732936239552e-06;
    const Q5: f64 = -2.01099218183624371326e-07;
    let mut hx = hi(x) as u32;
    let xsb = hx & 0x80000000;
    hx &= 0x7FFFFFFF;
    let k: i32;
    let mut c = 0.0;
    if hx >= 0x4043687A {
        if hx >= 0x40862E42 {
            if hx >= 0x7FF00000 {
                if ((hx & 0xFFFFF) | lo(x)) != 0 {
                    return x + x;
                }
                return if xsb == 0 { x } else { -1.0 };
            }
            if x > O_THRESHOLD {
                return HUGE * HUGE;
            }
        }
        if xsb != 0 && x + TINY < 0.0 {
            return TINY - 1.0;
        }
    }
    if hx > 0x3FD62E42 {
        let (hi_, lo_);
        if hx < 0x3FF0A2B2 {
            if xsb == 0 {
                hi_ = x - LN2_HI;
                lo_ = LN2_LO;
                k = 1;
            } else {
                hi_ = x + LN2_HI;
                lo_ = -LN2_LO;
                k = -1;
            }
        } else {
            k = fma(INVLN2, x, if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi_ = fma(-t, LN2_HI, x);
            lo_ = t * LN2_LO;
        }
        x = hi_ - lo_;
        c = (hi_ - x) - lo_;
    } else if hx < 0x3C900000 {
        let t = HUGE + x;
        return x - (t - (HUGE + x));
    } else {
        k = 0;
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = fma(hxs, fma(hxs, fma(hxs, fma(hxs, fma(hxs, Q5, Q4), Q3), Q2), Q1), 1.0);
    let t = fma(-r1, hfx, 3.0);
    let mut e = hxs * ((r1 - t) / fma(-x, t, 6.0));
    if k == 0 {
        return x - fma(x, e, -hxs);
    }
    let twopk = words(0x3FF00000i32.wrapping_add(((k as u32) << 20) as i32) as u32, 0);
    e = fma(x, e - c, -c);
    e -= hxs;
    if k == -1 {
        return fma(0.5, x - e, -0.5);
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        }
        return fma(2.0, x - e, 1.0);
    }
    if k <= -2 || k > 56 {
        let mut y = 1.0 - (e - x);
        if k == 1024 {
            y = y * 2.0 * 8.98846567431158e+307;
        } else {
            y *= twopk;
        }
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

/// V8 `Math.cbrt` (FreeBSD s_cbrt.c as in V8).
pub fn cbrt(x: f64) -> f64 {
    const B1: u32 = 715094163;
    const B2: u32 = 696219795;
    const P0: f64 = 1.87595182427177009643;
    const P1: f64 = -1.88497979543377169875;
    const P2: f64 = 1.621429720105354466140;
    const P3: f64 = -0.758397934778766047437;
    const P4: f64 = 0.145996192886612446982;
    let mut hx = hi(x);
    let low = lo(x);
    let sign = hx as u32 & 0x80000000;
    hx ^= sign as i32;
    if hx >= 0x7FF00000 {
        return x + x;
    }
    let mut t;
    if hx < 0x00100000 {
        if (hx as u32 | low) == 0 {
            return x;
        }
        t = with_hi(0.0, 0x43500000);
        t *= x;
        let high = hi(t) as u32;
        t = words(sign | ((high & 0x7FFFFFFF) / 3 + B2), 0);
    } else {
        t = words(sign | ((hx as u32) / 3 + B1), 0);
    }
    let r = (t * t) * (t / x);
    t *= fma((r * r) * r, fma(r, P4, P3), fma(r, fma(r, P2, P1), P0));
    t = f64::from_bits(t.to_bits().wrapping_add(0x80000000) & 0xFFFFFFFFC0000000);
    let s = t * t;
    let r = x / s;
    let w = t + t;
    let r = (r - t) / (w + r);
    fma(t, r, t)
}

/// V8 `Math.cosh`.
pub fn cosh(x: f64) -> f64 {
    const KCOSH_OVERFLOW: f64 = 710.4758600739439;
    let ix = hi(x) & 0x7FFFFFFF;
    if ix < 0x3FD62E43 {
        let t = expm1(x.abs());
        let w = 1.0 + t;
        if ix < 0x3C800000 {
            return w;
        }
        return 1.0 + (t * t) / (w + w);
    }
    if ix < 0x40360000 {
        let t = exp(x.abs());
        return fma(0.5, t, 0.5 / t);
    }
    if ix < 0x40862E42 {
        return 0.5 * exp(x.abs());
    }
    if x.abs() <= KCOSH_OVERFLOW {
        let w = exp(0.5 * x.abs());
        let t = 0.5 * w;
        return t * w;
    }
    if ix >= 0x7FF00000 {
        return x * x;
    }
    HUGE * HUGE
}

/// V8 `Math.sinh`.
pub fn sinh(x: f64) -> f64 {
    const KSINH_OVERFLOW: f64 = 710.4758600739439;
    const TWO_M28: f64 = 3.725290298461914e-9;
    const LOG_MAXD: f64 = 709.7822265625;
    const SHUGE: f64 = 1.0e307;
    let h = if x < 0.0 { -0.5 } else { 0.5 };
    let ax = x.abs();
    if ax < 22.0 {
        if ax < TWO_M28 {
            return x;
        }
        let t = expm1(ax);
        if ax < 1.0 {
            return h * fma(2.0, t, -(t * t / (t + 1.0)));
        }
        return h * (t + t / (t + 1.0));
    }
    if ax < LOG_MAXD {
        return h * exp(ax);
    }
    if ax <= KSINH_OVERFLOW {
        let w = exp(0.5 * ax);
        let t = h * w;
        return t * w;
    }
    x * SHUGE
}

/// V8 `Math.tanh`.
pub fn tanh(x: f64) -> f64 {
    let jx = hi(x);
    let ix = jx & 0x7FFFFFFF;
    if ix >= 0x7FF00000 {
        return if jx >= 0 { 1.0 / x + 1.0 } else { 1.0 / x - 1.0 };
    }
    let z;
    if ix < 0x40360000 {
        if ix < 0x3E300000 && HUGE + x > 1.0 {
            return x;
        }
        if ix >= 0x3FF00000 {
            let t = expm1(2.0 * x.abs());
            z = 1.0 - 2.0 / (t + 2.0);
        } else {
            let t = expm1(-2.0 * x.abs());
            z = -t / (t + 2.0);
        }
    } else {
        z = 1.0 - TINY;
    }
    if jx >= 0 {
        z
    } else {
        -z
    }
}

/// V8 `Math.pow(x, y)` / `x ** y` (fdlibm `pow`; (+-1) ** (+-Infinity) is NaN as ECMAScript requires).
pub fn pow(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01];
    const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08];
    const TWO53: f64 = 9007199254740992.0;
    const L1: f64 = 5.99999999999994648725e-01;
    const L2: f64 = 4.28571428578550184252e-01;
    const L3: f64 = 3.33333329818377432918e-01;
    const L4: f64 = 2.72728123808534006489e-01;
    const L5: f64 = 2.30660745775561754067e-01;
    const L6: f64 = 2.06975017800338417784e-01;
    const P1: f64 = 1.66666666666666019037e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;
    const LG2: f64 = 6.93147180559945286227e-01;
    const LG2_H: f64 = 6.93147182464599609375e-01;
    const LG2_L: f64 = -1.90465429995776804525e-09;
    const OVT: f64 = 8.0085662595372944372e-0017;
    const CP: f64 = 9.61796693925975554329e-01;
    const CP_H: f64 = 9.61796700954437255859e-01;
    const CP_L: f64 = -7.02846165095275826516e-09;
    const IVLN2: f64 = 1.44269504088896338700e+00;
    const IVLN2_H: f64 = 1.44269502162933349609e+00;
    const IVLN2_L: f64 = 1.92596299112661746887e-08;

    let (hx, lx) = (hi(x), lo(x));
    let (hy, ly) = (hi(y), lo(y));
    let mut ix = hx & 0x7fffffff;
    let iy = hy & 0x7fffffff;
    if (iy as u32 | ly) == 0 {
        return 1.0;
    }
    if ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) || iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0) {
        return x + y;
    }
    let mut yisint = 0i32;
    if hx < 0 {
        if iy >= 0x43400000 {
            yisint = 2;
        } else if iy >= 0x3ff00000 {
            let k = (iy >> 20) - 0x3ff;
            if k > 20 {
                let j = ly >> (52 - k);
                if (j << (52 - k)) == ly {
                    yisint = 2 - (j & 1) as i32;
                }
            } else if ly == 0 {
                let j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }
    if ly == 0 {
        if iy == 0x7ff00000 {
            if ((ix.wrapping_sub(0x3ff00000)) as u32 | lx) == 0 {
                return y - y;
            } else if ix >= 0x3ff00000 {
                return if hy >= 0 { y } else { 0.0 };
            } else {
                return if hy < 0 { -y } else { 0.0 };
            }
        }
        if iy == 0x3ff00000 {
            return if hy < 0 { 1.0 / x } else { x };
        }
        if hy == 0x40000000 {
            return x * x;
        }
        if hy == 0x3fe00000 && hx >= 0 {
            return x.sqrt();
        }
    }
    let mut ax = x.abs();
    if lx == 0 && (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
        let mut z = ax;
        if hy < 0 {
            z = 1.0 / z;
        }
        if hx < 0 {
            if ((ix - 0x3ff00000) | yisint) == 0 {
                z = f64::NAN;
            } else if yisint == 1 {
                z = -z;
            }
        }
        return z;
    }
    let mut n = (hx >> 31) + 1;
    if (n | yisint) == 0 {
        return f64::NAN;
    }
    let mut s = 1.0;
    if (n | (yisint - 1)) == 0 {
        s = -1.0;
    }
    let (t1, t2);
    if iy > 0x41e00000 {
        if iy > 0x43f00000 {
            if ix <= 0x3fefffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff00000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        if ix < 0x3fefffff {
            return if hy < 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        if ix > 0x3ff00000 {
            return if hy > 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        let t = ax - 1.0;
        let w = (t * t) * fma(-t, fma(-t, 0.25, 0.3333333333333333333333), 0.5);
        let u = IVLN2_H * t;
        let v = fma(t, IVLN2_L, -(w * IVLN2));
        t1 = with_lo(u + v, 0);
        t2 = v - (t1 - u);
    } else {
        n = 0;
        if ix < 0x00100000 {
            ax *= TWO53;
            n -= 53;
            ix = hi(ax);
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000fffff;
        ix = j | 0x3ff00000;
        let k: usize;
        if j <= 0x3988E {
            k = 0;
        } else if j < 0xBB67A {
            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = with_hi(ax, ix as u32);
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = with_lo(ss, 0);
        let t_h = with_hi(0.0, (((ix >> 1) | 0x20000000) + 0x00080000 + ((k as i32) << 18)) as u32);
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * fma(-s_h, t_l, fma(-s_h, t_h, u));
        let mut s2 = ss * ss;
        let mut r = s2 * s2 * fma(s2, fma(s2, fma(s2, fma(s2, fma(s2, L6, L5), L4), L3), L2), L1);
        r = fma(s_l, s_h + ss, r);
        s2 = s_h * s_h;
        let t_h = with_lo(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        let u = s_h * t_h;
        let v = fma(s_l, t_h, t_l * ss);
        let p_h = with_lo(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h;
        let z_l = fma(CP_L, p_h, p_l * CP) + DP_L[k];
        let t = n as f64;
        t1 = with_lo(((z_h + z_l) + DP_H[k]) + t, 0);
        t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
    }
    let y1 = with_lo(y, 0);
    let p_l = fma(y - y1, t1, y * t2);
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let j = hi(z);
    let i = lo(z) as i32;
    if j >= 0x40900000 {
        if ((j - 0x40900000) | i) != 0 || p_l + OVT > z - p_h {
            return s * HUGE * HUGE;
        }
    } else if (j & 0x7fffffff) >= 0x4090cc00
        && ((j.wrapping_sub(0xc090cc00u32 as i32) | i) != 0 || p_l <= z - p_h)
    {
        return s * TINY * TINY;
    }
    let i = j & 0x7fffffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n = 0i32;
    if i > 0x3fe00000 {
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        let t = with_hi(0.0, (n & !(0x000fffff >> k)) as u32);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_lo(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = fma(p_l - (t - p_h), LG2, t * LG2_L);
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = fma(-t, fma(t, fma(t, fma(t, fma(t, P5, P4), P3), P2), P1), z);
    let r = (z * t1) / ((t1 - 2.0) - fma(z, w, w));
    z = 1.0 - (r - z);
    let j = hi(z).wrapping_add(((n as u32) << 20) as i32);
    if (j >> 20) <= 0 {
        z = scalbn(z, n);
    } else {
        z = with_hi(z, hi(z).wrapping_add(((n as u32) << 20) as i32) as u32);
    }
    s * z
}

/// `Math.sqrt` (IEEE, correctly rounded everywhere).
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
