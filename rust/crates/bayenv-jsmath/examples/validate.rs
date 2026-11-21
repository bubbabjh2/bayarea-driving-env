//! Compare bayenv-jsmath against node reference vectors written by validate/gen.mjs.
//!
//!   cargo run --release -p bayenv-jsmath --example validate -- <dir> [fn,fn,...]
//!
//! Exit code 1 if any mismatch. NaN results compare equal regardless of payload.
use bayenv_jsmath as m;
use std::io::Read;

type F1 = fn(f64) -> f64;
type F2 = fn(f64, f64) -> f64;
type F3 = fn(f64, f64, f64) -> f64;

enum K {
    U(F1),
    B(F2),
    T(F3),
}

fn table() -> Vec<(&'static str, K)> {
    vec![
        ("sin", K::U(m::sin)),
        ("cos", K::U(m::cos)),
        ("tan", K::U(m::tan)),
        ("atan", K::U(m::atan)),
        ("asin", K::U(m::asin)),
        ("acos", K::U(m::acos)),
        ("exp", K::U(m::exp)),
        ("expm1", K::U(m::expm1)),
        ("log", K::U(m::log)),
        ("log2", K::U(m::log2)),
        ("tanh", K::U(m::tanh)),
        ("pow", K::B(m::pow)),
        ("powop", K::B(m::pow)),
        ("atan2", K::B(m::atan2)),
        ("hypot2", K::B(m::hypot)),
        ("hypot3", K::T(m::hypot3)),
        ("round", K::U(m::round)),
        ("sign", K::U(m::sign)),
        ("fround", K::U(m::fround)),
        ("sqrt", K::U(m::sqrt)),
        ("floor", K::U(m::floor)),
        ("ceil", K::U(m::ceil)),
        ("toint32", K::U(|x| m::to_int32(x) as f64)),
        ("touint32", K::U(|x| m::to_uint32(x) as f64)),
        ("imul", K::B(|a, b| m::imul(a, b) as f64)),
        ("max2", K::B(m::max)),
        ("min2", K::B(m::min)),
        ("max3", K::T(|a, b, c| m::max_slice(&[a, b, c]))),
        ("min3", K::T(|a, b, c| m::min_slice(&[a, b, c]))),
    ]
}

/// Plain std/libm equivalents, to show how many results a naive port would get wrong.
fn std_equiv(name: &str) -> Option<K> {
    Some(match name {
        "sin" => K::U(f64::sin),
        "cos" => K::U(f64::cos),
        "tan" => K::U(f64::tan),
        "atan" => K::U(f64::atan),
        "asin" => K::U(f64::asin),
        "acos" => K::U(f64::acos),
        "exp" => K::U(f64::exp),
        "expm1" => K::U(f64::exp_m1),
        "log" => K::U(f64::ln),
        "log2" => K::U(f64::log2),
        "tanh" => K::U(f64::tanh),
        "pow" | "powop" => K::B(f64::powf),
        "atan2" => K::B(f64::atan2),
        "hypot2" => K::B(f64::hypot),
        _ => return None,
    })
}

fn same(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

fn run(k: &K, d: &[f64], verbose: bool, name: &str) -> (usize, usize) {
    let ar = match k {
        K::U(_) => 1,
        K::B(_) => 2,
        K::T(_) => 3,
    };
    let s = ar + 1;
    let n = d.len() / s;
    let mut bad = 0;
    for i in 0..n {
        let r = &d[i * s..i * s + s];
        let got = match k {
            K::U(f) => f(r[0]),
            K::B(f) => f(r[0], r[1]),
            K::T(f) => f(r[0], r[1], r[2]),
        };
        if !same(got, r[ar]) {
            if verbose && bad < 8 {
                eprintln!(
                    "  {name} MISMATCH in={:?} ({:?}) node={:e} ({:#018x}) rust={:e} ({:#018x})",
                    &r[..ar],
                    r[..ar].iter().map(|x| format!("{:#018x}", x.to_bits())).collect::<Vec<_>>(),
                    r[ar],
                    r[ar].to_bits(),
                    got,
                    got.to_bits()
                );
            }
            bad += 1;
        }
    }
    (n, bad)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).expect("usage: validate <dir> [fns]");
    let only: Option<Vec<String>> = args.get(2).map(|s| s.split(',').map(String::from).collect());
    let mut total_bad = 0;
    println!("{:<10} {:>12} {:>10} {:>14}", "function", "records", "mismatch", "std-mismatch");
    for (name, k) in table() {
        if let Some(o) = &only {
            if !o.iter().any(|x| x == name) {
                continue;
            }
        }
        let path = format!("{dir}/{name}.bin");
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes).unwrap();
        let d: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (n, bad) = run(&k, &d, true, name);
        let stdbad = std_equiv(name)
            .map(|k| format!("{}", run(&k, &d, false, name).1))
            .unwrap_or_else(|| "-".into());
        println!("{name:<10} {n:>12} {bad:>10} {stdbad:>14}");
        total_bad += bad;
    }
    if total_bad > 0 {
        std::process::exit(1);
    }
}
