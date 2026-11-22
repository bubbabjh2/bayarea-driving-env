//! GPU mirror of bayenv-sim tests/physics_fixes.rs `static_hold_and_idle_creep_pull_away` and
//! `steer_delay_is_short_pedals_keep_reaction_time` (kernels/model_integrate.wgsl: MF_STATIC_HOLD, MF_IDLE_CREEP,
//! MF_STEER_DELAY). Cars scattered at rest on the real world (collisions off, every car on the same external action);
//! medians over the cars, so a few on steep grades do not matter. Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, sim::GpuOptions};
use std::{path::Path, sync::Arc};

const N: usize = 300;
// veh words (layout.rs VEH_FIELDS): speed 2, steerAngle 6, state 7, odometer 13
fn world() -> Option<Arc<bayenv_world::World>> {
    match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Some(Arc::new(w)), Err(e) => { bayenv_world::oracle::skip("pedals", &format!("world: {e}")); None } }
}
fn sim(w: &Arc<bayenv_world::World>, js: bool) -> Option<bayenv_gpu::sim::GpuSim> {
    let mut o = GpuOptions { capacity: 1024, obs: ObsMode::None, ..Default::default() };
    if js { o = o.js_model(); }
    o.features.collisions = false; o.features.accidents = false; o.features.fuel = false;
    let mut s = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("pedals: no GPU ({e}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut s, N, 7);
    Some(s)
}
fn median(mut v: Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); if v.is_empty() { f64::NAN } else { v[v.len() / 2] } }
fn act(s: &mut bayenv_gpu::sim::GpuSim, a: [f32; 3]) { let v: Vec<f32> = (0..s.cap).flat_map(|_| a).collect(); s.set_actions(&v, 0); }

/// 20 s from rest with the action: (median distance m, median final speed m/s) over the cars still driving
fn pull_away(w: &Arc<bayenv_world::World>, js: bool, gas: f32, brake: f32) -> Option<(f64, f64)> {
    let mut s = sim(w, js)?;
    act(&mut s, [0.0, 0.0, 1.0]); s.run_steps(20);   // settle at rest
    let st0 = s.read_state();
    act(&mut s, [0.0, gas, brake]); s.run_steps(400);
    let st = s.read_state();
    let (mut d, mut u) = (Vec::new(), Vec::new());
    for i in 0..N { if st.u(i, 7) == 1 && st0.u(i, 7) == 1 { d.push((st.f(i, 13) - st0.f(i, 13)) as f64); u.push(st.f(i, 2) as f64); } }
    Some((median(d), median(u)))
}

#[test]
fn gpu_static_hold_and_idle_creep_pull_away() {
    let Some(w) = world() else { return };
    let Some((d_gas, _)) = pull_away(&w, false, 0.02, 0.0) else { return };
    let (_, u_creep) = pull_away(&w, false, 0.0, 0.0).unwrap();
    let (d_brk, _) = pull_away(&w, false, 0.0, 0.1).unwrap();
    let (d_js, _) = pull_away(&w, true, 0.0, 0.0).unwrap();
    eprintln!("GPU, 20 s from rest (medians): gas 2 % {d_gas:.1} m; no pedals: creep {u_creep:.2} m/s (js {d_js:.2} m); brake 10 %: {d_brk:.3} m");
    assert!(d_gas > 10.0, "gas 2 % pulls away: {d_gas}");
    assert!((1.2..=1.8).contains(&u_creep), "idle creep about 1.5 m/s: {u_creep}");
    assert!(d_brk < 0.01, "a light brake holds against the creep: {d_brk}");
    assert!(d_js < 0.01, "the JS: no creep: {d_js}");
}

/// first step (s) after the command change at which the median car's steer angle moved / speed dropped by 0.5 m/s
fn response(w: &Arc<bayenv_world::World>, js: bool) -> Option<(f64, f64)> {
    let mut s = sim(w, js)?;
    act(&mut s, [0.0, 0.25, 0.0]); s.run_steps(200);   // rolling (> 0.5 m/s), wheel straight
    let st0 = s.read_state();
    let cars: Vec<usize> = (0..N).filter(|&i| st0.u(i, 7) == 1 && st0.f(i, 2) > 3.0).collect();
    let (mut ts, mut tb) = (vec![f64::NAN; N], vec![f64::NAN; N]);
    act(&mut s, [0.3, 0.0, 1.0]);
    for k in 0..60 {
        s.run_steps(1);
        let st = s.read_state(); let t = (k + 1) as f64 * s.dt;
        for &i in &cars {
            if ts[i].is_nan() && (st.f(i, 6) - st0.f(i, 6)).abs() > 1e-4 { ts[i] = t; }
            if tb[i].is_nan() && st.f(i, 2) < st0.f(i, 2) - 0.5 { tb[i] = t; }
        }
    }
    let m = |v: &[f64]| median(cars.iter().map(|&i| if v[i].is_nan() { 99.0 } else { v[i] }).collect());
    Some((m(&ts), m(&tb)))
}

#[test]
fn gpu_steer_delay_is_short_pedals_keep_reaction_time() {
    let Some(w) = world() else { return };
    let Some((s, b)) = response(&w, false) else { return };
    let (js, jb) = response(&w, true).unwrap();
    eprintln!("GPU first response (median, s): steer {s:.2}, brake {b:.2}; js steer {js:.2}, brake {jb:.2}");
    assert!(s > 0.0 && s <= 0.25, "steering after about 0.15 s: {s}");
    assert!(b >= 0.5, "pedals keep the reaction time: {b}");
    assert!(js >= 0.5, "the JS delays steering by the reaction time: {js}");
}

/// obs v2 delay compensation with steer_delay (kernels/model_sense.wgsl): input 5 (heading error, delay-compensated)
/// replays only the steering commands not applied yet, so after a steer step it predicts the heading error the car
/// actually has once those are applied: sds steps later with steer_delay (the JS window replayed d steps, counting the
/// d - sds already applied ones twice). Returns the mean |prediction - actual| at t + sds and at t + d.
fn heading_prediction_error(w: &Arc<bayenv_world::World>, steer_delay: f64) -> Option<(f64, f64)> {
    let mut o = GpuOptions { capacity: 1024, obs: ObsMode::V2, steer_delay, ..Default::default() };
    o.features.collisions = false; o.features.accidents = false; o.features.fuel = false;
    let mut s = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("pedals: no GPU ({e}): not run"); return None } };
    bayenv_gpu::env::scatter(&mut s, N, 9);
    act(&mut s, [0.0, 0.25, 0.0]); s.run_steps(120);
    let cap = s.cap; let mut obs = vec![0f32; cap * 70];
    let (mut pred, mut actual): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
    let mut alive = vec![true; N];
    for k in 0..60 {
        // steer pulses: 0.3 for 10 steps, -0.3 for 10, ...
        act(&mut s, [if (k / 10) % 2 == 0 { 0.3 } else { -0.3 }, 0.25, 0.0]);
        s.run_steps(1); s.read_obs_into(&mut obs);
        let st = s.read_state();
        // sensed in this step before integrate: the prediction for after the pending commands; actual = after integrate
        pred.push((0..N).map(|i| obs[i * 70 + 5] * std::f32::consts::FRAC_PI_4).collect());
        actual.push((0..N).map(|i| -st.f(i, 5)).collect());
        for i in 0..N { if st.u(i, 7) != 1 || st.f(i, 2) < 3.0 || st.f(i, 30) >= 1.0 || (st.f(i, 5)).abs() > 0.7 { alive[i] = false; } }
    }
    let st = s.read_state();
    let sds = (steer_delay / s.dt).round() as i64;
    let (mut e_s, mut e_d, mut n) = (0.0f64, 0.0f64, 0usize);
    for i in 0..N {
        if !alive[i] { continue; }
        let d = st.par[i * bayenv_gpu::layout::PAR_WORDS + 3] as usize;
        let win = if steer_delay >= 0.0 && (sds as usize) < d { sds as usize } else { d };
        // with steer_delay the prediction made at step t covers the pending commands and the one act writes in step t
        // (model_sense.wgsl): it predicts the state after integrate t + window; the JS window (d) stops at t - 1 + d
        let (hs, hd) = if win < d { (win, d) } else { (d - 1, d - 1) };
        for t in 1..(60 - d) {
            e_s += (pred[t][i] - actual[t + hs][i]).abs() as f64;
            e_d += (pred[t][i] - actual[t + hd][i]).abs() as f64;
            n += 1;
        }
    }
    Some((e_s / n.max(1) as f64, e_d / n.max(1) as f64))
}

#[test]
fn gpu_obs_delay_compensation_follows_steer_delay() {
    let Some(w) = world() else { return };
    let Some((s15, d15)) = heading_prediction_error(&w, 0.15) else { return };
    let (sj, dj) = heading_prediction_error(&w, -1.0).unwrap();
    eprintln!("GPU obs 5 prediction error (rad, mean): steer_delay 0.15: vs actual after sds {s15:.4}, after d {d15:.4}; JS: after d {dj:.4} ({sj:.4})");
    assert!(s15 < d15, "with steer_delay the compensation predicts the heading after sds steps");
    assert!(s15 < 1.5 * dj, "the compensation is as accurate as the JS one: {s15} vs {dj}");
}
