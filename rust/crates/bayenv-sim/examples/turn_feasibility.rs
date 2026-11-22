//! Can the steering model make a tight junction turn? Per vehicle type at 4.5 m/s (held by the pedals): the steady
//! radius at full lock, the time / distance to reach full lock from straight (steer-rate and lag limits), and the path
//! length until the car has turned 90 degrees when it starts steering at the arc's start, for a full-lock command and
//! for a command sized to a 6 m (cars) / 12 m (trucks, buses) kinematic radius.
//!   cargo run --release -p bayenv-sim --example turn_feasibility
use bayenv_sim::physics::{step_dynamics, DynState, VEHICLE_PARAMS};
const NAMES: [&str; 4] = ["sedan", "suv", "truck", "bus"];
fn run(ty: usize, u0: f64, cmd: f64) -> (f64, f64, f64, f64) {
    let p = &VEHICLE_PARAMS[ty];
    let mut s = DynState { u: u0, v: 0.0, r: 0.0, steer: 0.0, sl: 0.0, ax: 0.0, hr: 0.0, lat: 0.0, s: 0.0, fuel_l: 0.0, power: 0.0 };
    let dt = 0.05;
    let (mut t, mut dist, mut t_full, mut d90) = (0.0, 0.0, f64::NAN, f64::NAN);
    let (mut x, mut z) = (0.0f64, 0.0f64);
    let mut pts = Vec::new();
    while t < 30.0 {
        let gas = ((u0 - s.u) * 0.8).clamp(0.0, 1.0); let brake = ((s.u - u0) * 0.5).clamp(0.0, 1.0);
        let (s0, l0) = (s.s, s.lat);
        step_dynamics(&mut s, cmd, gas, brake, p, dt, 0.0, false, true, None);
        t += dt;
        let (dx, dz) = (s.s - s0, s.lat - l0); dist += (dx * dx + dz * dz).sqrt(); x += dx; z += dz; pts.push((x, z));
        if t_full.is_nan() && (s.steer.abs() >= 0.98 * (cmd.abs() * p.max_steer)) { t_full = t; }
        if d90.is_nan() && s.hr.abs() >= std::f64::consts::FRAC_PI_2 { d90 = dist; break; }
    }
    let rad = if s.r.abs() > 1e-6 { s.u / s.r.abs() } else { f64::INFINITY };
    (rad, t_full, t_full * u0, d90)
}
fn main() {
    let u0 = 4.5;
    println!("at {u0} m/s (dt 0.05, position steer)");
    for ty in 0..4 {
        let p = &VEHICLE_PARAMS[ty];
        let kin = p.l / p.max_steer.tan();
        let (r, tf, df, d90) = run(ty, u0, 1.0);
        let target: f64 = if ty < 2 { 6.0 } else { 12.0 };
        let cmd = ((p.l / target).atan() / p.max_steer).min(1.0);
        let (r2, _, _, d2) = run(ty, u0, cmd);
        println!("{:6} L {:.2} max_steer {:.2} rad rate {:.2} rad/s | kinematic min R {:.1} m | full lock: steady R {:.1} m, lock reached in {:.2} s ({:.1} m), 90 deg after {:.1} m of path | {target} m command ({:.0}% lock): steady R {:.1} m, 90 deg after {:.1} m",
            NAMES[ty], p.l, p.max_steer, p.max_steer_rate, kin, r, tf, df, d90, 100.0 * cmd, r2, d2);
    }
}
