//! Closed-loop CPU vs GPU through the observation (regression test for the steer_delay delay compensation,
//! kernels/model_sense.wgsl): the same population on both backends, driven by the same policy-like controller on obs v2
//! (CPU: bayenv-sim V2Encoder, as bayenv-py obs="v2"; GPU: sense.wgsl). Steering is a lane keeper on the delay-
//! compensated lane offset (4) and heading error (5) plus a per-car swing, so a compensation that predicts the wrong
//! horizon makes the cars weave into the kerbs; pedals keep speed, follow the leader and stop (light brake) at red
//! lights. Barrier rows and crashes must agree between the backends for steer_delay 0, 0.15 and 0.3 s. Measured:
//! crashed cpu 27 / gpu 29, barrier rows 52 / 47 (0 s); 55 / 56, 110 / 95 (0.15 s); 23 / 24, 164 / 163 (0.3 s), with
//! obs_after_step (the GPU observes the state after the step, as the CPU). With the JS GPU order (one step older obs)
//! and the in-flight-command compensation: crashed 26 / 44, 53 / 50, 22 / 32.
//! Before the fix (GPU replaying the full reaction window, then the window without the in-flight command), 150 s:
//! crashed cpu 27 / gpu 156 (0 s), 55 / 172 (0.15 s), 23 / 124 (0.3 s). Skips without the world data or a GPU.
use bayenv_gpu::{defines::ObsMode, sim::{GpuOptions, GpuPopulation}};
use std::{path::Path, sync::Arc};

const N: usize = 1000;
const STEPS: usize = 3000;
const START: f64 = 7.0 * 3600.0;
const W: usize = 70;

fn policy(o: &[f32], slot: usize, t: usize) -> [f32; 3] {
    let v = o[0] * 40.0; let vl = (o[1] * 40.0).max(5.0); let gap = o[8] * 120.0;
    let mut a = 0.8 * (0.9 * vl - v);
    if gap < 2.0 * v + 6.0 { a = a.min(-1.5 - (2.0 * v + 6.0 - gap) * 0.5); }
    if gap < 4.0 { a = -4.0; }
    let stopc = o[53] > 0.5 || o[52] > 0.5 || (o[47] > 0.5 && o[61] < 0.5);
    let d = o[56] * 250.0;
    let need = stopc && d < v * v / 3.0 + 8.0;
    if need { a = a.min(-(v * v) / (2.0 * (d - 1.5).max(0.5)) - 0.3); }
    let noise = 0.08 * (slot as f32 * 0.37 + t as f32 * 0.013).sin();
    let steer = (-2.5 * o[4] + 2.0 * o[5] + noise).clamp(-1.0, 1.0);
    let (mut gas, mut brake) = ((a / 3.0).clamp(0.0, 1.0), (-a / 6.5).clamp(0.0, 1.0));
    if v < 0.5 && (need || gap < 6.0) { gas = 0.0; brake = 0.03; }
    [steer, gas, brake]
}

/// (crashed, barrier rows, contact rows) on the CPU and on the GPU
fn run(w: &Arc<bayenv_world::World>, steer_delay: f64) -> Option<([usize; 3], [usize; 3])> {
    let data = Path::new("../../../data");
    let cap = (N + 1023) / 1024 * 1024;
    let mut o = GpuOptions { capacity: cap, start_time: START, seed: 777, obs: ObsMode::V2, refuel_budget_ms: f64::INFINITY, ..Default::default() };
    o.steer_delay = steer_delay;
    let mut gpu = match bayenv_gpu::env::create(w.clone(), o) { Ok(s) => s, Err(e) => { eprintln!("closed_loop: no GPU ({e}): not run"); return None } };
    gpu.set_population(Arc::new(GpuPopulation::generate(w, N, 777.0, true).unwrap()), true);
    let sw = Arc::new(bayenv_sim::world::World::from_bayenv(w));
    let pop = Arc::new(bayenv_sim::population::Population::generate_like_env(&sw, "real", N, 777.0, Some(data)).unwrap());
    let mut opts = bayenv_sim::sim::SimOpts::default(); opts.dt = 0.05; opts.seed = 777.0; opts.steer_delay = steer_delay;
    let cfg = bayenv_sim::env::EnvConfig { opts, start_time: Some(START), static_obstacles: Vec::new(), scenario_obstacles: Vec::new(), obstacles: None };
    let mut cpu = bayenv_sim::env::Env::new(sw.clone(), pop, cfg, Box::new(|w| bayenv_sim::harness::router_for(&w)));
    let mut enc = bayenv_sim::obs_v2::V2Encoder::new(&sw);
    let (mut c, mut g) = ([0usize; 3], [0usize; 3]);
    let count = |log: &bayenv_world::events::EventLog, x: &mut [usize; 3]| for (k, o) in log.kind.iter().zip(&log.other) { if *k == 2 { x[0] += 1; } if *k == 3 { x[2] += 1; if *o == -1 { x[1] += 1; } } };
    let mut cobs = Vec::<f32>::new(); let mut gobs = vec![0f32; cap * W]; let mut acts = vec![0f32; cap * 3];
    for t in 0..STEPS {
        // CPU: actions from the obs encoded after the last step, then record + step + encode (bayenv-py CpuBackend)
        let n = cpu.sim.vs.high;
        let rows: Vec<f32> = (0..n).flat_map(|i| if cpu.sim.vs.state[i] == bayenv_sim::vehicle::STATE_DRIVING && cobs.len() >= (i + 1) * W { policy(&cobs[i * W..(i + 1) * W], i, t) } else { [0.0, 0.0, 0.3] }).collect();
        cpu.act_rows(&rows, 0);
        enc.record_actions(&cpu.sim);
        cpu.sim.step();
        count(&cpu.take_records(), &mut c);
        let n = cpu.sim.vs.high; cobs.clear(); cobs.resize(n * W, 0.0);
        for i in 0..n { if cpu.sim.vs.state[i] == bayenv_sim::vehicle::STATE_DRIVING { enc.encode_v2(&mut cpu.sim, i, &mut cobs[i * W..(i + 1) * W]); } }
        for a in cpu.sim.actions.iter_mut() { *a = None; }
        // GPU: actions from the obs sensed in the last step
        let st = gpu.read_state();
        for i in 0..cap { acts[3 * i..3 * i + 3].copy_from_slice(&if st.u(i, 7) == 1 && t > 0 { policy(&gobs[i * W..(i + 1) * W], i, t) } else { [0.0, 0.0, 0.3] }); }
        gpu.set_actions(&acts, 0); gpu.run_steps(1);
        count(&gpu.take_records(), &mut g);
        gpu.read_obs_into(&mut gobs);
    }
    eprintln!("steer_delay {steer_delay}: crashed cpu {} gpu {} | barrier rows cpu {} gpu {} | contact rows cpu {} gpu {}", c[0], g[0], c[1], g[1], c[2], g[2]);
    Some((c, g))
}

#[test]
fn closed_loop_through_the_obs_agrees() {
    let w = match bayenv_world::World::load(Path::new("../../../data")) { Ok(w) => Arc::new(w), Err(e) => { bayenv_world::oracle::skip("closed_loop", &format!("world: {e}")); return } };
    let mut fails = Vec::new();
    for sd in [0.0, 0.15, 0.3] {
        let Some((c, g)) = run(&w, sd) else { return };
        let near = |a: usize, b: usize, abs: f64, rel: f64| (a as f64 - b as f64).abs() <= abs.max(rel * a as f64);
        if !near(c[1], g[1], 15.0, 0.25) { fails.push(format!("steer_delay {sd}: barrier rows cpu {} gpu {}", c[1], g[1])); }
        if !near(c[0], g[0], 8.0, 0.25) { fails.push(format!("steer_delay {sd}: crashed cpu {} gpu {}", c[0], g[0])); }
    }
    assert!(fails.is_empty(), "{fails:?}");
}
