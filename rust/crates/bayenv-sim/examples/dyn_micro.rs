//! step_dynamics cost per car-step vs the transcendental calls it makes (atan x2, tanh x2, cos / sin of hr per substep)
use bayenv_sim::physics::{step_dynamics, step_dynamics4, DynState, VEHICLE_PARAMS};
use bayenv_world::jsmath as jm;
use std::time::Instant;
fn st(k: usize) -> DynState { DynState { u: 18.0 + k as f64, v: 0.05 * k as f64, r: 0.01 * (k as f64 - 1.5), steer: 0.002 * k as f64, sl: 0.0, ax: 0.1, hr: 0.01, lat: 1.5, s: 10.0, fuel_l: 0.0, power: 0.0 } }
fn main() {
    let n = 400_000usize;
    let p = &VEHICLE_PARAMS[0];
    let mut acc = 0.0;
    let t = Instant::now();
    for i in 0..n { let mut s = st(i % 4); step_dynamics(&mut s, 0.01, 0.3, 0.0, p, 0.1, 0.0, false, true, None); acc += s.u; }
    let t1 = t.elapsed().as_secs_f64() / n as f64;
    let t = Instant::now();
    for _ in 0..n / 4 { let (mut a, mut b, mut c, mut d) = (st(0), st(1), st(2), st(3));
        step_dynamics4([(&mut a, 0.01, 0.3, 0.0, p, 0.0, false, None), (&mut b, 0.01, 0.3, 0.0, p, 0.0, false, None), (&mut c, 0.01, 0.3, 0.0, p, 0.0, false, None), (&mut d, 0.01, 0.3, 0.0, p, 0.0, false, None)], 0.1, true); acc += a.u + d.u; }
    let t4 = t.elapsed().as_secs_f64() / n as f64;
    // the transcendental calls of one car-step: 10 substeps x (2 atan + 2 tanh + cos + sin)
    let m = n * 10;
    let t = Instant::now();
    let mut x = 0.1f64;
    for i in 0..m { let y = 0.001 * (i % 97) as f64; acc += jm::atan(y + x) + jm::atan(y - x) + jm::tanh(3.0 * y + x) + jm::tanh(2.0 * y - x) + jm::cos(0.01 + y) + jm::sin(0.01 + y); x = acc * 1e-30 + 0.1; }
    let tt = t.elapsed().as_secs_f64() / n as f64;
    println!("per car-step: step_dynamics {:.0} ns, step_dynamics4 {:.0} ns; transcendentals alone (dependent chain) {:.0} ns  [{acc:.3e}]", t1 * 1e9, t4 * 1e9, tt * 1e9);
}
