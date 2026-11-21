//! Micro-benchmark: bayenv-jsmath vs std f64 methods (ns per call, throughput loop).
//!   cargo run --release -p bayenv-jsmath --example bench
use bayenv_jsmath as m;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1 << 16;
const REPS: usize = 200;

fn inputs(lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..N)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}

fn time1(xs: &[f64], f: impl Fn(f64) -> f64) -> f64 {
    let mut acc = 0.0;
    let t = Instant::now();
    for _ in 0..REPS {
        for &x in xs {
            acc += f(black_box(x));
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (REPS * xs.len()) as f64
}

fn time2(xs: &[f64], ys: &[f64], f: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = 0.0;
    let t = Instant::now();
    for _ in 0..REPS {
        for (&x, &y) in xs.iter().zip(ys) {
            acc += f(black_box(x), black_box(y));
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (REPS * xs.len()) as f64
}

fn row(name: &str, ours: f64, std: f64) {
    println!("{name:<22} {ours:>8.2} ns {std:>8.2} ns {:>7.2}x", ours / std);
}

fn main() {
    println!("{:<22} {:>11} {:>11} {:>8}", "function (range)", "jsmath", "std", "ratio");
    let a = inputs(-6.3, 6.3, 1);
    let big = inputs(-1e4, 1e4, 2);
    let u = inputs(-1.0, 1.0, 3);
    let e = inputs(-50.0, 50.0, 4);
    let p = inputs(1e-3, 1e7, 5);
    let b = inputs(-1e3, 1e3, 6);
    let c = inputs(-1e3, 1e3, 7);
    let y = inputs(-4.0, 4.0, 8);
    row("sin [-2pi,2pi]", time1(&a, m::sin), time1(&a, f64::sin));
    row("sin [-1e4,1e4]", time1(&big, m::sin), time1(&big, f64::sin));
    row("cos [-2pi,2pi]", time1(&a, m::cos), time1(&a, f64::cos));
    row("cos [-1e4,1e4]", time1(&big, m::cos), time1(&big, f64::cos));
    row("tan [-2pi,2pi]", time1(&a, m::tan), time1(&a, f64::tan));
    row("atan [-1e3,1e3]", time1(&b, m::atan), time1(&b, f64::atan));
    row("atan2", time2(&b, &c, m::atan2), time2(&b, &c, f64::atan2));
    row("acos [-1,1]", time1(&u, m::acos), time1(&u, f64::acos));
    row("exp [-50,50]", time1(&e, m::exp), time1(&e, f64::exp));
    row("log (0,1e7]", time1(&p, m::log), time1(&p, f64::ln));
    row("log2 (0,1e7]", time1(&p, m::log2), time1(&p, f64::log2));
    row("tanh [-50,50]", time1(&e, m::tanh), time1(&e, f64::tanh));
    row("pow (0,1e7]^[-4,4]", time2(&p, &y, m::pow), time2(&p, &y, f64::powf));
    row("hypot(a,b)", time2(&b, &c, m::hypot), time2(&b, &c, f64::hypot));
    row("hypot3 vs sqrt(aa+bb+cc)", time2(&b, &c, |x, y| m::hypot3(x, y, x)), time2(&b, &c, |x, y| (x * x + y * y + x * x).sqrt()));
    row("round vs f64::round", time1(&b, m::round), time1(&b, f64::round));
    row("max vs f64::max", time2(&b, &c, m::max), time2(&b, &c, f64::max));
    row("to_int32 vs as i32", time1(&big, |x| m::to_int32(x) as f64), time1(&big, |x| (x as i32) as f64));
}
