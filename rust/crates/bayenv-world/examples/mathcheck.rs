//! Check bayenv_world::jsmath against a `mathcheck.mjs` dump dir (pairs [x, f(x)], pow/atan2 triples).
use bayenv_world::jsmath as m;
fn main() {
    let d = std::env::args().nth(1).unwrap();
    let rd = |f: &str| -> Vec<f64> { let b = std::fs::read(format!("{d}/math-{f}.bin")).unwrap(); bytemuck::cast_slice(&b).to_vec() };
    let one: [(&str, fn(f64) -> f64); 7] = [("exp", m::exp), ("log", m::log), ("cos", m::cos), ("sin", m::sin), ("tanh", m::tanh), ("cbrt", m::cbrt), ("atan", m::atan)];
    for (n, f) in one { let v = rd(n); let bad = v.chunks(2).filter(|c| f(c[0]).to_bits() != c[1].to_bits()).count(); println!("{n:6} {} inputs, {bad} mismatches", v.len() / 2); }
    for (n, f) in [("pow", m::pow as fn(f64, f64) -> f64), ("atan2", m::atan2)] { let v = rd(n); let bad = v.chunks(3).filter(|c| f(c[0], c[1]).to_bits() != c[2].to_bits()).count(); println!("{n:6} {} inputs, {bad} mismatches", v.len() / 3); }
}
