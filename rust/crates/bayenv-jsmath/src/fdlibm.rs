//! Line-by-line port of V8 12.4's `src/base/ieee754.cc` (fdlibm-derived).
//!
//! Source: <https://raw.githubusercontent.com/v8/v8/12.4-lkgr/src/base/ieee754.cc>
//!
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//!
//! The original source code covered by the above license above has been
//! modified significantly by Google Inc.
//! Copyright 2016 the V8 project authors. All rights reserved.
//! (V8 is BSD-3-Clause licensed.)
//!
//! ## FMA contraction
//!
//! Node binaries for arm64 are built by clang with its default
//! `-ffp-contract=on`, which fuses every *syntactic* `a*b + c` / `a*b - c` /
//! `c - a*b` inside a single C expression (including `x += a*b`) into one
//! fused multiply-add. Contraction never crosses statements. When both
//! operands of the `+`/`-` are products, the left one is fused.
//! Every such site below is written with [`mla`] (fused by default, or plain
//! `a*b + c` with the `no-fma` cargo feature, which reproduces a build without
//! contraction such as x86-64 node). Everything else is written as plain
//! separately-rounded operations, exactly as in the C source.

#![allow(clippy::excessive_precision, clippy::many_single_char_names)]
#![allow(clippy::approx_constant, clippy::unreadable_literal)]

/// `a*b + c` exactly as clang -ffp-contract=on compiles it on arm64 (one rounding).
#[inline(always)]
fn mla(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(not(feature = "no-fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(feature = "no-fma")]
    {
        a * b + c
    }
}

// ---- word access helpers (EXTRACT_WORDS etc.) ----
#[inline(always)]
fn hi(d: f64) -> i32 {
    (d.to_bits() >> 32) as u32 as i32
}
#[inline(always)]
fn lo(d: f64) -> u32 {
    d.to_bits() as u32
}
#[inline(always)]
fn from_words(h: u32, l: u32) -> f64 {
    f64::from_bits(((h as u64) << 32) | l as u64)
}
#[inline(always)]
fn with_hi(d: f64, h: u32) -> f64 {
    f64::from_bits((d.to_bits() & 0x0000_0000_FFFF_FFFF) | ((h as u64) << 32))
}
#[inline(always)]
fn with_lo(d: f64, l: u32) -> f64 {
    f64::from_bits((d.to_bits() & 0xFFFF_FFFF_0000_0000) | l as u64)
}

/// Correctly rounded scalbn (musl algorithm; same results as any conforming libm).
fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000);
    let x1p53 = f64::from_bits(0x4340_0000_0000_0000);
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000);
    let mut y = x;
    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // make sure final n < -53 to avoid double rounding in the subnormal range
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

// ---------------------------------------------------------------------------
// Argument reduction
// ---------------------------------------------------------------------------

static TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

static NPIO2_HW: [i32; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB,
    0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB,
    0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C,
    0x4041475C, 0x4042106C, 0x4042D97C, 0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB,
    0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

/// __ieee754_rem_pio2: returns n and y[0]+y[1] = x - n*pi/2.
#[inline]
fn rem_pio2(x: f64) -> (i32, f64, f64) {
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
        return (0, x, 0.0);
    }
    if ix < 0x4002D97C {
        // |x| < 3pi/4, special case with n=+-1
        if hx > 0 {
            let mut z = x - PIO2_1;
            let (y0, y1);
            if ix != 0x3FF921FB {
                y0 = z - PIO2_1T;
                y1 = (z - y0) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y0 = z - PIO2_2T;
                y1 = (z - y0) - PIO2_2T;
            }
            return (1, y0, y1);
        } else {
            let mut z = x + PIO2_1;
            let (y0, y1);
            if ix != 0x3FF921FB {
                y0 = z + PIO2_1T;
                y1 = (z - y0) + PIO2_1T;
            } else {
                z += PIO2_2;
                y0 = z + PIO2_2T;
                y1 = (z - y0) + PIO2_2T;
            }
            return (-1, y0, y1);
        }
    }
    if ix <= 0x413921FB {
        // |x| ~<= 2^19*(pi/2), medium size
        let mut t = x.abs();
        let n = mla(t, INVPIO2, HALF) as i32;
        let fnn = n as f64;
        let mut r = mla(-fnn, PIO2_1, t);
        let mut w = fnn * PIO2_1T;
        let mut y0;
        if n < 32 && ix != NPIO2_HW[(n - 1) as usize] {
            y0 = r - w;
        } else {
            let j = ix >> 20;
            y0 = r - w;
            let high = hi(y0) as u32;
            let mut i = j - ((high >> 20) & 0x7FF) as i32;
            if i > 16 {
                t = r;
                w = fnn * PIO2_2;
                r = t - w;
                w = mla(fnn, PIO2_2T, -((t - r) - w));
                y0 = r - w;
                let high = hi(y0) as u32;
                i = j - ((high >> 20) & 0x7FF) as i32;
                if i > 49 {
                    t = r;
                    w = fnn * PIO2_3;
                    r = t - w;
                    w = mla(fnn, PIO2_3T, -((t - r) - w));
                    y0 = r - w;
                }
            }
        }
        let y1 = (r - y0) - w;
        if hx < 0 {
            return (-n, -y0, -y1);
        } else {
            return (n, y0, y1);
        }
    }
    // all other (large) arguments
    if ix >= 0x7FF00000 {
        let v = x - x;
        return (0, v, v);
    }
    // set z = scalbn(|x|,ilogb(x)-23)
    let e0 = (ix >> 20) - 1046;
    let mut z = from_words((ix - ((e0 as u32) << 20) as i32) as u32, lo(x));
    let mut tx = [0.0f64; 3];
    for t in tx.iter_mut().take(2) {
        *t = (z as i32) as f64;
        z = (z - *t) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3usize;
    while tx[nx - 1] == 0.0 {
        nx -= 1;
    }
    let mut y = [0.0f64; 2];
    let n = kernel_rem_pio2(&tx[..nx], &mut y, e0, 2);
    if hx < 0 {
        (-n, -y[0], -y[1])
    } else {
        (n, y[0], y[1])
    }
}

/// __kernel_rem_pio2 with ipio2 = two_over_pi. `x.len()` = nx; prec 0..=2 supported.
#[cold]
#[inline(never)]
fn kernel_rem_pio2(x: &[f64], y: &mut [f64; 2], e0: i32, prec: usize) -> i32 {
    const INIT_JK: [i32; 4] = [2, 3, 4, 6];
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

    let nx = x.len() as i32;
    let mut iq = [0i32; 20];
    let mut f = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut q = [0.0f64; 20];

    let jk = INIT_JK[prec];
    let jp = jk;

    let jx = nx - 1;
    let mut jv = (e0 - 3) / 24;
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);

    let mut j = jv - jx;
    let m = jx + jk;
    for i in 0..=m {
        f[i as usize] = if j < 0 { 0.0 } else { ipio2[j as usize] as f64 };
        j += 1;
    }

    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw = mla(x[j as usize], f[(jx + i - j) as usize], fw);
        }
        q[i as usize] = fw;
    }

    let mut jz = jk;
    let mut z;
    let mut n;
    let mut ih;
    loop {
        // recompute:
        let mut i = 0i32;
        let mut j = jz;
        z = q[jz as usize];
        while j > 0 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[i as usize] = mla(-TWO24, fw, z) as i32;
            z = q[(j - 1) as usize] + fw;
            i += 1;
            j -= 1;
        }

        z = scalbn(z, q0);
        z = mla(-8.0, (z * 0.125).floor(), z);
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            let i = iq[(jz - 1) as usize] >> (24 - q0);
            n += i;
            iq[(jz - 1) as usize] -= i << (24 - q0);
            ih = iq[(jz - 1) as usize] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[(jz - 1) as usize] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            n += 1;
            let mut carry = 0;
            for i in 0..jz {
                let j = iq[i as usize];
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        iq[i as usize] = 0x1000000 - j;
                    }
                } else {
                    iq[i as usize] = 0xFFFFFF - j;
                }
            }
            if q0 > 0 {
                match q0 {
                    1 => iq[(jz - 1) as usize] &= 0x7FFFFF,
                    2 => iq[(jz - 1) as usize] &= 0x3FFFFF,
                    _ => {}
                }
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
                j |= iq[i as usize];
                i -= 1;
            }
            if j == 0 {
                let mut k = 1;
                while jk >= k && iq[(jk - k) as usize] == 0 {
                    k += 1;
                }
                for i in (jz + 1)..=(jz + k) {
                    f[(jx + i) as usize] = ipio2[(jv + i) as usize] as f64;
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw = mla(x[j as usize], f[(jx + i - j) as usize], fw);
                    }
                    q[i as usize] = fw;
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
        while iq[jz as usize] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = scalbn(z, -q0);
        if z >= TWO24 {
            let fw = ((TWON24 * z) as i32) as f64;
            iq[jz as usize] = mla(-TWO24, fw, z) as i32;
            jz += 1;
            q0 += 24;
            iq[jz as usize] = fw as i32;
        } else {
            iq[jz as usize] = z as i32;
        }
    }

    let mut fw = scalbn(1.0, q0);
    let mut i = jz;
    while i >= 0 {
        q[i as usize] = fw * iq[i as usize] as f64;
        fw *= TWON24;
        i -= 1;
    }

    let mut i = jz;
    while i >= 0 {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw = mla(PIO2[k as usize], q[(i + k) as usize], fw);
            k += 1;
        }
        fq[(jz - i) as usize] = fw;
        i -= 1;
    }

    match prec {
        0 => {
            let mut fw = 0.0;
            let mut i = jz;
            while i >= 0 {
                fw += fq[i as usize];
                i -= 1;
            }
            y[0] = if ih == 0 { fw } else { -fw };
        }
        _ => {
            let mut fw = 0.0;
            let mut i = jz;
            while i >= 0 {
                fw += fq[i as usize];
                i -= 1;
            }
            y[0] = if ih == 0 { fw } else { -fw };
            fw = fq[0] - fw;
            for i in 1..=jz {
                fw += fq[i as usize];
            }
            y[1] = if ih == 0 { fw } else { -fw };
        }
    }
    n & 7
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

#[inline(always)]
fn kernel_cos(x: f64, y: f64) -> f64 {
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
    let r = z * mla(z, mla(z, mla(z, mla(z, mla(z, C6, C5), C4), C3), C2), C1);
    if ix < 0x3FD33333 {
        // one - (0.5 * z - (z * r - x * y))
        1.0 - mla(0.5, z, -mla(z, r, -(x * y)))
    } else {
        let qx = if ix > 0x3FE90000 {
            0.28125
        } else {
            from_words((ix - 0x00200000) as u32, 0)
        };
        let iz = mla(0.5, z, -qx);
        let a = 1.0 - qx;
        a - (iz - mla(z, r, -(x * y)))
    }
}

#[inline(always)]
fn kernel_sin(x: f64, y: f64, iy: i32) -> f64 {
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
    let r = mla(z, mla(z, mla(z, mla(z, S6, S5), S4), S3), S2);
    if iy == 0 {
        mla(v, mla(z, r, S1), x)
    } else {
        // x - ((z * (half * y - v * r) - y) - v * S1)
        let a = mla(HALF, y, -(v * r));
        let b = mla(z, a, -y);
        let c = mla(-v, S1, b);
        x - c
    }
}

fn kernel_tan(mut x: f64, mut y: f64, iy: i32) -> f64 {
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
    const ONE: f64 = 1.0;
    const PIO4: f64 = 7.85398163397448278999e-01;
    const PIO4LO: f64 = 3.06161699786838301793e-17;

    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix < 0x3E300000 && (x as i32) == 0 {
        let low = lo(x);
        if ((ix as u32 | low) | ((iy + 1) as u32)) == 0 {
            return ONE / x.abs();
        } else if iy == 1 {
            return x;
        } else {
            let w = x + y;
            let z = with_lo(w, 0);
            let v = y - (z - x);
            let a = -ONE / w;
            let t = with_lo(a, 0);
            let s = mla(t, z, ONE);
            return mla(a, mla(t, v, s), t);
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
    let mut r = mla(w, mla(w, mla(w, mla(w, mla(w, T[11], T[9]), T[7]), T[5]), T[3]), T[1]);
    let v = z * mla(w, mla(w, mla(w, mla(w, mla(w, T[12], T[10]), T[8]), T[6]), T[4]), T[2]);
    let s = z * x;
    r = mla(z, mla(s, r + v, y), y);
    r = mla(T[0], s, r);
    let w = x + r;
    if ix >= 0x3FE59428 {
        let v = iy as f64;
        let sgn = (1 - ((hx >> 30) & 2)) as f64;
        return sgn * mla(-2.0, x - (w * w / (w + v) - r), v);
    }
    if iy == 1 {
        w
    } else {
        let z = with_lo(w, 0);
        let v = r - (z - x);
        let a = -1.0 / w;
        let t = with_lo(a, 0);
        let s = mla(t, z, 1.0);
        mla(a, mla(t, v, s), t)
    }
}

// ---------------------------------------------------------------------------
// Public (crate) functions
// ---------------------------------------------------------------------------

pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        kernel_sin(x, 0.0, 0)
    } else if ix >= 0x7FF00000 {
        x - x
    } else {
        let (n, y0, y1) = rem_pio2(x);
        match n & 3 {
            0 => kernel_sin(y0, y1, 1),
            1 => kernel_cos(y0, y1),
            2 => -kernel_sin(y0, y1, 1),
            _ => -kernel_cos(y0, y1),
        }
    }
}

pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        kernel_cos(x, 0.0)
    } else if ix >= 0x7FF00000 {
        x - x
    } else {
        let (n, y0, y1) = rem_pio2(x);
        match n & 3 {
            0 => kernel_cos(y0, y1),
            1 => -kernel_sin(y0, y1, 1),
            2 => -kernel_cos(y0, y1),
            _ => kernel_sin(y0, y1, 1),
        }
    }
}

pub fn tan(x: f64) -> f64 {
    let ix = hi(x) & 0x7FFFFFFF;
    if ix <= 0x3FE921FB {
        kernel_tan(x, 0.0, 1)
    } else if ix >= 0x7FF00000 {
        x - x
    } else {
        let (n, y0, y1) = rem_pio2(x);
        kernel_tan(y0, y1, 1 - ((n & 1) << 1))
    }
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

#[inline(always)]
fn asin_p(z: f64) -> f64 {
    z * mla(z, mla(z, mla(z, mla(z, mla(z, PS5, PS4), PS3), PS2), PS1), PS0)
}
#[inline(always)]
fn asin_q(z: f64) -> f64 {
    mla(z, mla(z, mla(z, mla(z, QS4, QS3), QS2), QS1), 1.0)
}

pub fn acos(x: f64) -> f64 {
    const PI: f64 = 3.14159265358979311600e+00;
    const PIO2_HI: f64 = 1.57079632679489655800e+00;
    const PIO2_LO: f64 = 6.12323399573676603587e-17;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix >= 0x3FF00000 {
        let lx = lo(x);
        if ((ix - 0x3FF00000) as u32 | lx) == 0 {
            if hx > 0 {
                return 0.0;
            } else {
                return mla(2.0, PIO2_LO, PI);
            }
        }
        return f64::NAN;
    }
    if ix < 0x3FE00000 {
        if ix <= 0x3C600000 {
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let p = asin_p(z);
        let q = asin_q(z);
        let r = p / q;
        PIO2_HI - (x - mla(-x, r, PIO2_LO))
    } else if hx < 0 {
        let z = (1.0 + x) * 0.5;
        let p = asin_p(z);
        let q = asin_q(z);
        let s = z.sqrt();
        let r = p / q;
        let w = mla(r, s, -PIO2_LO);
        mla(-2.0, s + w, PI)
    } else {
        let z = (1.0 - x) * 0.5;
        let s = z.sqrt();
        let df = with_lo(s, 0);
        let c = mla(-df, df, z) / (s + df);
        let p = asin_p(z);
        let q = asin_q(z);
        let r = p / q;
        let w = mla(r, s, c);
        2.0 * (df + w)
    }
}

pub fn asin(x: f64) -> f64 {
    const HUGE: f64 = 1.000e+300;
    const PIO2_HI: f64 = 1.57079632679489655800e+00;
    const PIO2_LO: f64 = 6.12323399573676603587e-17;
    const PIO4_HI: f64 = 7.85398163397448278999e-01;
    let mut t = 0.0;
    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    if ix >= 0x3FF00000 {
        let lx = lo(x);
        if ((ix - 0x3FF00000) as u32 | lx) == 0 {
            return mla(x, PIO2_HI, x * PIO2_LO);
        }
        return f64::NAN;
    } else if ix < 0x3FE00000 {
        if ix < 0x3E400000 {
            if HUGE + x > 1.0 {
                return x;
            }
        } else {
            t = x * x;
        }
        let p = asin_p(t);
        let q = asin_q(t);
        let w = p / q;
        return mla(x, w, x);
    }
    let w = 1.0 - x.abs();
    let t = w * 0.5;
    let p = asin_p(t);
    let q = asin_q(t);
    let s = t.sqrt();
    let t = if ix >= 0x3FEF3333 {
        let w = p / q;
        PIO2_HI - mla(2.0, mla(s, w, s), -PIO2_LO)
    } else {
        let w = with_lo(s, 0);
        let c = mla(-w, w, t) / (s + w);
        let r = p / q;
        let p = mla(2.0 * s, r, -mla(-2.0, c, PIO2_LO));
        let q = mla(-2.0, w, PIO4_HI);
        PIO4_HI - (p - q)
    };
    if hx > 0 {
        t
    } else {
        -t
    }
}

pub fn atan(x: f64) -> f64 {
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
    const ONE: f64 = 1.0;
    const HUGE: f64 = 1.0e300;

    let hx = hi(x);
    let ix = hx & 0x7FFFFFFF;
    let mut x = x;
    if ix >= 0x44100000 {
        let low = lo(x);
        if ix > 0x7FF00000 || (ix == 0x7FF00000 && low != 0) {
            return x + x;
        }
        if hx > 0 {
            return ATANHI[3] + ATANLO[3];
        } else {
            return -ATANHI[3] - ATANLO[3];
        }
    }
    let id: i32;
    if ix < 0x3FDC0000 {
        if ix < 0x3E400000 && HUGE + x > ONE {
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3FF30000 {
            if ix < 0x3FE60000 {
                id = 0;
                x = mla(2.0, x, -ONE) / (2.0 + x);
            } else {
                id = 1;
                x = (x - ONE) / (x + ONE);
            }
        } else if ix < 0x40038000 {
            id = 2;
            x = (x - 1.5) / mla(1.5, x, ONE);
        } else {
            id = 3;
            x = -1.0 / x;
        }
    }
    let z = x * x;
    let w = z * z;
    let s1 = z * mla(w, mla(w, mla(w, mla(w, mla(w, AT[10], AT[8]), AT[6]), AT[4]), AT[2]), AT[0]);
    let s2 = w * mla(w, mla(w, mla(w, mla(w, AT[9], AT[7]), AT[5]), AT[3]), AT[1]);
    if id < 0 {
        mla(-x, s1 + s2, x)
    } else {
        let id = id as usize;
        let z = ATANHI[id] - (mla(x, s1 + s2, -ATANLO[id]) - x);
        if hx < 0 {
            -z
        } else {
            z
        }
    }
}

pub fn atan2(y: f64, x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    const ZERO: f64 = 0.0;
    const PI_O_4: f64 = 7.8539816339744827900E-01;
    const PI_O_2: f64 = 1.5707963267948965580E+00;
    const PI: f64 = 3.1415926535897931160E+00;
    const PI_LO: f64 = 1.2246467991473531772E-16;

    let hx = hi(x);
    let lx = lo(x);
    let ix = hx & 0x7FFFFFFF;
    let hy = hi(y);
    let ly = lo(y);
    let iy = hy & 0x7FFFFFFF;
    if ((ix as u32 | ((lx | (lx as i32).wrapping_neg() as u32) >> 31)) > 0x7FF00000)
        || ((iy as u32 | ((ly | (ly as i32).wrapping_neg() as u32) >> 31)) > 0x7FF00000)
    {
        return x + y;
    }
    if (hx.wrapping_sub(0x3FF00000) as u32 | lx) == 0 {
        return atan(y);
    }
    let mut m = ((hy >> 31) & 1) | ((hx >> 30) & 2);

    if (iy as u32 | ly) == 0 {
        match m {
            0 | 1 => return y,
            2 => return PI + TINY,
            _ => return -PI - TINY,
        }
    }
    if (ix as u32 | lx) == 0 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    if ix == 0x7FF00000 {
        if iy == 0x7FF00000 {
            match m {
                0 => return PI_O_4 + TINY,
                1 => return -PI_O_4 - TINY,
                2 => return mla(3.0, PI_O_4, TINY),
                _ => return mla(-3.0, PI_O_4, -TINY),
            }
        } else {
            match m {
                0 => return ZERO,
                1 => return -ZERO,
                2 => return PI + TINY,
                _ => return -PI - TINY,
            }
        }
    }
    if iy == 0x7FF00000 {
        return if hy < 0 { -PI_O_2 - TINY } else { PI_O_2 + TINY };
    }
    let k = (iy - ix) >> 20;
    let z;
    if k > 60 {
        z = mla(0.5, PI_LO, PI_O_2);
        m &= 1;
    } else if hx < 0 && k < -60 {
        z = 0.0;
    } else {
        z = atan((y / x).abs());
    }
    match m {
        0 => z,
        1 => -z,
        2 => PI - (z - PI_LO),
        _ => (z - PI_LO) - PI,
    }
}

pub fn exp(x: f64) -> f64 {
    const ONE: f64 = 1.0;
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
    const HUGE: f64 = 1.0e+300;
    const TWOM1000: f64 = 9.33263618503218878990e-302;
    const TWO1023: f64 = 8.988465674311579539e307;

    let mut x = x;
    let mut hi_ = 0.0;
    let mut lo_ = 0.0;
    let mut k: i32 = 0;
    let mut hx = hi(x) as u32;
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7FFFFFFF;

    if hx >= 0x40862E42 {
        if hx >= 0x7FF00000 {
            let lx = lo(x);
            if ((hx & 0xFFFFF) | lx) != 0 {
                return x + x;
            } else {
                return if xsb == 0 { x } else { 0.0 };
            }
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
            k = mla(INVLN2, x, HALF[xsb]) as i32;
            let t = k as f64;
            hi_ = mla(-t, LN2HI[0], x);
            lo_ = t * LN2LO[0];
        }
        x = hi_ - lo_;
    } else if hx < 0x3E300000 {
        if HUGE + x > ONE {
            return ONE + x;
        }
    } else {
        k = 0;
    }

    let t = x * x;
    let twopk = if k >= -1021 {
        from_words((0x3FF00000i32.wrapping_add(((k as u32) << 20) as i32)) as u32, 0)
    } else {
        from_words(0x3FF00000u32.wrapping_add(((k + 1000) as u32) << 20), 0)
    };
    let c = mla(-t, mla(t, mla(t, mla(t, mla(t, P5, P4), P3), P2), P1), x);
    let y;
    if k == 0 {
        return ONE - ((x * c) / (c - 2.0) - x);
    } else {
        y = ONE - ((lo_ - (x * c) / (2.0 - c)) - hi_);
    }
    if k >= -1021 {
        if k == 1024 {
            return y * 2.0 * TWO1023;
        }
        y * twopk
    } else {
        y * twopk * TWOM1000
    }
}

pub fn expm1(x: f64) -> f64 {
    const ONE: f64 = 1.0;
    const TINY: f64 = 1.0e-300;
    const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
    const LN2_HI: f64 = 6.93147180369123816490e-01;
    const LN2_LO: f64 = 1.90821492927058770002e-10;
    const INVLN2: f64 = 1.44269504088896338700e+00;
    const Q1: f64 = -3.33333333333331316428e-02;
    const Q2: f64 = 1.58730158725481460165e-03;
    const Q3: f64 = -7.93650757867487942473e-05;
    const Q4: f64 = 4.00821782732936239552e-06;
    const Q5: f64 = -2.01099218183624371326e-07;
    const HUGE: f64 = 1.0e+300;

    let mut x = x;
    let mut hx = hi(x) as u32;
    let xsb = hx & 0x80000000;
    hx &= 0x7FFFFFFF;

    if hx >= 0x4043687A {
        if hx >= 0x40862E42 {
            if hx >= 0x7FF00000 {
                let low = lo(x);
                if ((hx & 0xFFFFF) | low) != 0 {
                    return x + x;
                } else {
                    return if xsb == 0 { x } else { -1.0 };
                }
            }
            if x > O_THRESHOLD {
                return HUGE * HUGE;
            }
        }
        if xsb != 0 && x + TINY < 0.0 {
            return TINY - ONE;
        }
    }

    let k: i32;
    let mut c = 0.0;
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
            k = mla(INVLN2, x, if xsb == 0 { 0.5 } else { -0.5 }) as i32;
            let t = k as f64;
            hi_ = mla(-t, LN2_HI, x);
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
    let r1 = mla(hxs, mla(hxs, mla(hxs, mla(hxs, mla(hxs, Q5, Q4), Q3), Q2), Q1), ONE);
    let t = mla(-r1, hfx, 3.0);
    let mut e = hxs * ((r1 - t) / mla(-x, t, 6.0));
    if k == 0 {
        return x - mla(x, e, -hxs);
    }
    let twopk = from_words((0x3FF00000i32.wrapping_add(((k as u32) << 20) as i32)) as u32, 0);
    e = mla(x, e - c, -c);
    e -= hxs;
    if k == -1 {
        return mla(0.5, x - e, -0.5);
    }
    if k == 1 {
        if x < -0.25 {
            return -2.0 * (e - (x + 0.5));
        } else {
            return mla(2.0, x - e, ONE);
        }
    }
    if k <= -2 || k > 56 {
        let mut y = ONE - (e - x);
        if k == 1024 {
            y = y * 2.0 * 8.98846567431158e+307;
        } else {
            y *= twopk;
        }
        return y - ONE;
    }
    let mut y;
    if k < 20 {
        let t = with_hi(ONE, (0x3FF00000 - (0x200000 >> k)) as u32);
        y = t - (e - x);
        y *= twopk;
    } else {
        let t = with_hi(ONE, ((0x3FF - k) << 20) as u32);
        y = x - (e + t);
        y += ONE;
        y *= twopk;
    }
    y
}

pub fn tanh(x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const HUGE: f64 = 1.0e300;
    let jx = hi(x);
    let ix = jx & 0x7FFFFFFF;
    if ix >= 0x7FF00000 {
        if jx >= 0 {
            return ONE / x + ONE;
        } else {
            return ONE / x - ONE;
        }
    }
    let z;
    if ix < 0x40360000 {
        if ix < 0x3E300000 && HUGE + x > ONE {
            return x;
        }
        if ix >= 0x3FF00000 {
            let t = expm1(TWO * x.abs());
            z = ONE - TWO / (t + TWO);
        } else {
            let t = expm1(-TWO * x.abs());
            z = -t / (t + TWO);
        }
    } else {
        z = ONE - TINY;
    }
    if jx >= 0 {
        z
    } else {
        -z
    }
}

pub fn log(x: f64) -> f64 {
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

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
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
            } else {
                let dk = k as f64;
                return mla(dk, LN2_HI, dk * LN2_LO);
            }
        }
        let r = (f * f) * mla(-0.33333333333333333, f, 0.5);
        if k == 0 {
            return f - r;
        } else {
            let dk = k as f64;
            return mla(dk, LN2_HI, -(mla(-dk, LN2_LO, r) - f));
        }
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147A;
    let w = z * z;
    let j = 0x6B851 - hx;
    let t1 = w * mla(w, mla(w, LG6, LG4), LG2);
    let t2 = z * mla(w, mla(w, mla(w, LG7, LG5), LG3), LG1);
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            f - mla(-s, hfsq + r, hfsq)
        } else {
            mla(dk, LN2_HI, -((hfsq - mla(s, hfsq + r, dk * LN2_LO)) - f))
        }
    } else if k == 0 {
        mla(-s, f - r, f)
    } else {
        mla(dk, LN2_HI, -(mla(s, f - r, -(dk * LN2_LO)) - f))
    }
}

pub fn log2(x: f64) -> f64 {
    const TWO54: f64 = 1.80143985094819840000e+16;
    const IVLN2HI: f64 = 1.44269504072144627571e+00;
    const IVLN2LO: f64 = 1.67517131648865118353e-10;
    const LG1: f64 = 6.666666666666735130e-01;
    const LG2: f64 = 3.999999999940941908e-01;
    const LG3: f64 = 2.857142874366239149e-01;
    const LG4: f64 = 2.222219843214978396e-01;
    const LG5: f64 = 1.818357216161805012e-01;
    const LG6: f64 = 1.531383769920937332e-01;
    const LG7: f64 = 1.479819860511658591e-01;

    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k: i32 = 0;
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
    // k_log1p(f)
    let r = {
        let s = f / (2.0 + f);
        let z = s * s;
        let w = z * z;
        let t1 = w * mla(w, mla(w, LG6, LG4), LG2);
        let t2 = z * mla(w, mla(w, mla(w, LG7, LG5), LG3), LG1);
        let rr = t2 + t1;
        let hfsq = 0.5 * f * f;
        s * (hfsq + rr)
    };
    let hi_ = with_lo(f - hfsq, 0);
    let lo_ = (f - hi_) - hfsq + r;
    let mut val_hi = hi_ * IVLN2HI;
    let mut val_lo = mla(lo_ + hi_, IVLN2LO, lo_ * IVLN2HI);
    let w = y + val_hi;
    val_lo += (y - w) + val_hi;
    val_hi = w;
    val_lo + val_hi
}

pub fn pow(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01];
    const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08];
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const TWO53: f64 = 9007199254740992.0;
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;
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

    let hx = hi(x);
    let lx = lo(x);
    let hy = hi(y);
    let ly = lo(y);
    let mut ix = hx & 0x7fffffff;
    let iy = hy & 0x7fffffff;

    if (iy as u32 | ly) == 0 {
        return ONE;
    }
    if ix > 0x7ff00000 || (ix == 0x7ff00000 && lx != 0) || iy > 0x7ff00000 || (iy == 0x7ff00000 && ly != 0) {
        return x + y;
    }

    let mut yisint = 0;
    if hx < 0 {
        if iy >= 0x43400000 {
            yisint = 2;
        } else if iy >= 0x3ff00000 {
            let k = (iy >> 20) - 0x3ff;
            if k > 20 {
                let j = (ly >> (52 - k)) as i32;
                if (j.wrapping_shl((52 - k) as u32)) == ly as i32 {
                    yisint = 2 - (j & 1);
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
            if ((ix - 0x3ff00000) as u32 | lx) == 0 {
                return y - y;
            } else if ix >= 0x3ff00000 {
                return if hy >= 0 { y } else { ZERO };
            } else {
                return if hy < 0 { -y } else { ZERO };
            }
        }
        if iy == 0x3ff00000 {
            return if hy < 0 { ONE / x } else { x };
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
            z = ONE / z;
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

    let mut n: i32 = (hx >> 31) + 1;
    if (n | yisint) == 0 {
        return f64::NAN;
    }
    let mut s = ONE;
    if (n | (yisint - 1)) == 0 {
        s = -ONE;
    }

    let t1;
    let t2;
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
        let t = ax - ONE;
        let w = (t * t) * mla(-t, mla(-t, 0.25, 0.3333333333333333333333), 0.5);
        let u = IVLN2_H * t;
        let v = mla(t, IVLN2_L, -(w * IVLN2));
        let tt = with_lo(u + v, 0);
        t1 = tt;
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
        let v = ONE / (ax + BP[k]);
        let ss = u * v;
        let s_h = with_lo(ss, 0);
        let t_h = from_words((((ix >> 1) | 0x20000000) + 0x00080000 + ((k as i32) << 18)) as u32, 0);
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * mla(-s_h, t_l, mla(-s_h, t_h, u));
        let mut s2 = ss * ss;
        let mut r = (s2 * s2) * mla(s2, mla(s2, mla(s2, mla(s2, mla(s2, L6, L5), L4), L3), L2), L1);
        r = mla(s_l, s_h + ss, r);
        s2 = s_h * s_h;
        let t_h = with_lo(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        let u = s_h * t_h;
        let v = mla(s_l, t_h, t_l * ss);
        let p_h = with_lo(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h;
        let z_l = mla(CP_L, p_h, p_l * CP) + DP_L[k];
        let t = n as f64;
        t1 = with_lo(((z_h + z_l) + DP_H[k]) + t, 0);
        t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
    }

    let y1 = with_lo(y, 0);
    let p_l = mla(y - y1, t1, y * t2);
    let mut p_h = y1 * t1;
    let mut z = p_l + p_h;
    let j = hi(z);
    let i = lo(z) as i32;
    if j >= 0x40900000 {
        if ((j - 0x40900000) | i) != 0 {
            return s * HUGE * HUGE;
        } else if p_l + OVT > z - p_h {
            return s * HUGE * HUGE;
        }
    } else if (j & 0x7fffffff) >= 0x4090cc00 {
        if ((j.wrapping_sub(0xc090cc00u32 as i32)) | i) != 0 {
            return s * TINY * TINY;
        } else if p_l <= z - p_h {
            return s * TINY * TINY;
        }
    }

    let i = j & 0x7fffffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;
    if i > 0x3fe00000 {
        n = j.wrapping_add(0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        let t = from_words((n & !(0x000fffff >> k)) as u32, 0);
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_lo(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = mla(p_l - (t - p_h), LG2, t * LG2_L);
    z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = mla(-t, mla(t, mla(t, mla(t, mla(t, P5, P4), P3), P2), P1), z);
    let r = (z * t1) / ((t1 - TWO) - mla(z, w, w));
    z = ONE - (r - z);
    let mut j = hi(z);
    j = j.wrapping_add(((n as u32) << 20) as i32);
    if (j >> 20) <= 0 {
        z = scalbn(z, n);
    } else {
        let tmp = hi(z);
        z = with_hi(z, tmp.wrapping_add(((n as u32) << 20) as i32) as u32);
    }
    s * z
}
